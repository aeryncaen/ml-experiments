"""
Unified Sequence Block (USB) Implementation

Fuses SSM-style scans, attention, and MLP into a single expand-process-contract block.
"""

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .scan import forward_scan, backward_scan
from .rope import apply_data_dependent_rope, apply_rope
from heuristic_secrets.models.scatter_attention import SIRENDownsampleND


@dataclass
class USBConfig:
    """Configuration for USB block."""
    d_model: int
    headdim: int = 64
    expansion_factor: int = 2
    layer_idx: int = 0  # For future use (e.g., layer-dependent params)
    
    # Scan state modes per group (G1, G2, G3): 'elementwise' (k*v) or 'outer' (k⊗v)
    # - elementwise: state is (nheads, headdim), good for state-tracking (parity)
    # - outer: state is (nheads, headdim, headdim), good for retrieval (induction)
    # Default: G1/G2 outer (directional retrieval), G3 elementwise (global state)
    scan_state_modes: tuple = ('outer', 'outer', 'elementwise')
    
    # Derived dimensions
    @property
    def d_expanded(self) -> int:
        return self.d_model * self.expansion_factor
    
    @property
    def d_group(self) -> int:
        """Width per channel group (G1-G4)."""
        return self.d_expanded // 4
    
    @property
    def nheads_per_group(self) -> int:
        return self.d_group // self.headdim
    
    @property
    def nheads_total(self) -> int:
        return self.nheads_per_group * 4


class GatedRMSNorm(nn.Module):
    """RMSNorm with optional gating."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor, gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        # RMSNorm
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms * self.weight
        
        # Optional gating
        if gate is not None:
            x = x * F.silu(gate)
        
        return x


class PerHeadProjections(nn.Module):
    """
    Fused projections for per-head content-dependent parameters.
    
    Each scan head needs: α, β, γ (trapezoidal coeffs), gate, rope_freq
    We fuse these into a single projection per group for efficiency.
    """
    
    def __init__(self, d_input: int, nheads: int, headdim: int):
        super().__init__()
        self.nheads = nheads
        self.headdim = headdim
        
        # Fused projection for all per-head parameters:
        # - alpha: 1 scalar per head (decay)
        # - delta: 1 scalar per head (step size)
        # - lambda: 1 scalar per head (trapezoidal mixing)
        # - gate: headdim values per head (per-dimension gating)
        # - rope_freq: headdim // 2 values per head (rotation frequencies for pairs)
        self.n_scalar_params = 3  # alpha, delta, lambda
        self.n_gate_params = headdim
        self.n_rope_params = headdim // 2
        
        total_per_head = self.n_scalar_params + self.n_gate_params + self.n_rope_params
        self.proj = nn.Linear(d_input, nheads * total_per_head, bias=False)
        
        # Initialize decay projection to produce reasonable α values
        # α = exp(-softplus(x)), so x ≈ 0 gives α ≈ exp(-0.69) ≈ 0.5
        nn.init.zeros_(self.proj.weight)
    
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: (batch, seq_len, d_input)
        
        Returns:
            dict with:
                alpha: (batch, seq_len, nheads) - decay rates in (0, 1)
                beta, gamma: (batch, seq_len, nheads) - trapezoidal coefficients
                gate: (batch, seq_len, nheads, headdim) - per-dim injection gates in (0, 1)
                rope_freq: (batch, seq_len, nheads, headdim // 2) - rotation frequencies
        """
        batch, seq_len, _ = x.shape
        
        # Fused projection
        out = self.proj(x)  # (batch, seq_len, nheads * total_per_head)
        out = rearrange(out, 'b t (h d) -> b t h d', h=self.nheads)
        
        # Split into components
        idx = 0
        
        # Scalars: alpha, delta, lambda
        alpha_raw = out[..., idx]
        idx += 1
        delta_raw = out[..., idx]
        idx += 1
        lambda_raw = out[..., idx]
        idx += 1
        
        # Gate: per-dimension
        gate_raw = out[..., idx:idx + self.n_gate_params]
        idx += self.n_gate_params
        
        # RoPE frequencies
        rope_freq = out[..., idx:idx + self.n_rope_params]
        
        # Apply activation functions
        alpha = torch.exp(-F.softplus(alpha_raw))  # (0, 1)
        delta = F.softplus(delta_raw)
        lam = torch.sigmoid(lambda_raw)
        beta = (1.0 - lam) * delta * alpha
        gamma = lam * delta
        gate = torch.sigmoid(gate_raw)  # (0, 1) per dimension
        
        return {
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'gate': gate,
            'rope_freq': rope_freq,
        }


class USBBlock(nn.Module):
    """
    Unified Sequence Block.
    
    Fuses expansion, QKV projection, directional scans, attention, and contraction
    into a single block.
    """
    
    def __init__(self, config: USBConfig):
        super().__init__()
        self.config = config

        # Debug instrumentation (set externally)
        self.debug = False
        self.debug_active = False
        self.last_debug = None
        
        d_model = config.d_model
        d_expanded = config.d_expanded
        d_group = config.d_group
        nheads_per_group = config.nheads_per_group
        nheads_total = config.nheads_total
        headdim = config.headdim
        
        # Step 1: Expansion
        self.expand = nn.Linear(d_model, d_expanded, bias=False)
        
        # Step 2: QKV projections (full MHA - all heads independent)
        self.q_proj = nn.Linear(d_expanded, d_expanded, bias=False)
        self.k_proj = nn.Linear(d_expanded, d_expanded, bias=False)
        self.v_proj = nn.Linear(d_expanded, d_expanded, bias=False)
        
        # Gated RMSNorm for Q, K, V
        self.q_norm = GatedRMSNorm(d_expanded)
        self.k_norm = GatedRMSNorm(d_expanded)
        self.v_norm = GatedRMSNorm(d_expanded)
        
        # QK bias (per-head, initialized to 1.0)
        self.q_bias = nn.Parameter(torch.ones(nheads_total, headdim))
        self.k_bias = nn.Parameter(torch.ones(nheads_total, headdim))
        
        # Per-head projections for scan groups (G1, G2, G3)
        # G4 is passthrough, no scan parameters needed
        self.scan_proj_g1 = PerHeadProjections(d_expanded, nheads_per_group, headdim)
        self.scan_proj_g2 = PerHeadProjections(d_expanded, nheads_per_group, headdim)
        self.scan_proj_g3 = PerHeadProjections(d_expanded, nheads_per_group, headdim)

        # G3 depthwise conv (k=3) replacing centered scan
        self.conv_g3 = nn.Conv1d(
            d_group, d_group, kernel_size=3, padding=1, groups=d_group, bias=False
        )

        # Low-rank attention downsamplers (SIREN)
        self.attn_downsample = SIRENDownsampleND(d_expanded, ndim=1)
        self.attn_rope_downsample = SIRENDownsampleND(d_group // 2, ndim=1)

        # Router head for multifocal attention centers
        self.router_norm = GatedRMSNorm(d_expanded)
        self.router = nn.Linear(d_expanded, 1, bias=False)
        self.router_temp = nn.Parameter(torch.tensor(1.0))
        self.local_gate = nn.Parameter(torch.tensor(-2.0))
        nn.init.zeros_(self.router.weight)
        
        # Learnable initial states for scan heads (G1, G2, G3)
        # Shape depends on per-group scan_state_modes:
        # - elementwise: (nheads_per_group, headdim)
        # - outer: (nheads_per_group, headdim, headdim)
        def init_shape(mode):
            if mode == 'outer':
                return (nheads_per_group, headdim, headdim)
            return (nheads_per_group, headdim)
        
        modes = config.scan_state_modes
        self.init_state_g1 = nn.Parameter(torch.zeros(*init_shape(modes[0])))
        self.init_state_g2 = nn.Parameter(torch.zeros(*init_shape(modes[1])))
        self.init_state_g3 = nn.Parameter(torch.zeros(*init_shape(modes[2])))
        
        # Mark initial states as no weight decay
        self.init_state_g1._no_weight_decay = True  # type: ignore[attr-defined]
        self.init_state_g2._no_weight_decay = True  # type: ignore[attr-defined]
        self.init_state_g3._no_weight_decay = True  # type: ignore[attr-defined]
        
        # Step 7: Down-projection
        self.down_proj = nn.Linear(d_expanded, d_model, bias=False)
        
        # Pre-norm for the block
        self.norm = GatedRMSNorm(d_model)
    
    def forward(
        self, 
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            attention_mask: Optional mask for attention
        
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        config = self.config
        residual = x

        do_debug = self.debug and self.debug_active
        if do_debug:
            self.last_debug = None
        
        # Pre-norm
        x = self.norm(x)
        
        # Step 1: Expansion with SiLU
        x_exp = F.silu(self.expand(x))  # (batch, seq_len, d_expanded)
        
        # Step 2: QKV projection (full MHA)
        q = self.q_proj(x_exp)  # (batch, seq, nheads_total * headdim)
        k = self.k_proj(x_exp)
        v = self.v_proj(x_exp)
        
        # Apply RMSNorm (no activation on Q/K/V)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        
        # Reshape to heads
        q = rearrange(q, 'b t (h d) -> b t h d', h=config.nheads_total)
        k = rearrange(k, 'b t (h d) -> b t h d', h=config.nheads_total)
        v = rearrange(v, 'b t (h d) -> b t h d', h=config.nheads_total)
        
        # Apply QK bias (after norm)
        q = q + self.q_bias
        k = k + self.k_bias
        
        # Step 3: Channel split into 4 groups
        # Each group gets nheads_per_group heads
        nph = config.nheads_per_group
        q_g1, q_g2, q_g3, q_g4 = q[..., :nph, :], q[..., nph:2*nph, :], q[..., 2*nph:3*nph, :], q[..., 3*nph:, :]
        k_g1, k_g2, k_g3, k_g4 = k[..., :nph, :], k[..., nph:2*nph, :], k[..., 2*nph:3*nph, :], k[..., 3*nph:, :]
        v_g1, v_g2, v_g3, v_g4 = v[..., :nph, :], v[..., nph:2*nph, :], v[..., 2*nph:3*nph, :], v[..., 3*nph:, :]
        
        # Step 4: Directional scans for G1, G2; G3 uses a 3-wide conv
        
        # Get per-head parameters for each scan group
        params_g1 = self.scan_proj_g1(x_exp)
        params_g2 = self.scan_proj_g2(x_exp)
        params_g3 = self.scan_proj_g3(x_exp)
        
        # Apply data-dependent RoPE to K before scanning
        k_g1_attn = k_g1
        k_g2_attn = k_g2
        k_g3_attn = k_g3
        k_g4_attn = k_g4

        k_g1 = apply_data_dependent_rope(k_g1, params_g1['rope_freq'])
        k_g2 = apply_data_dependent_rope(k_g2, params_g2['rope_freq'])
        k_g3 = apply_data_dependent_rope(k_g3, params_g3['rope_freq'])
        
        # Compute K·V for each group (mode-dependent)
        # Shape depends on scan_state_modes[i]:
        # - elementwise: (batch, seq_len, nheads, headdim)
        # - outer: (batch, seq_len, nheads, headdim, headdim) = k ⊗ v
        modes = config.scan_state_modes
        
        # G1: forward scan
        if modes[0] == 'outer':
            kv_g1 = k_g1.unsqueeze(-1) * v_g1.unsqueeze(-2)
            init_g1 = repeat(self.init_state_g1, 'h d1 d2 -> b h d1 d2', b=batch)
        else:
            kv_g1 = k_g1 * v_g1
            init_g1 = repeat(self.init_state_g1, 'h d -> b h d', b=batch)
        
        # G2: backward scan
        if modes[1] == 'outer':
            kv_g2 = k_g2.unsqueeze(-1) * v_g2.unsqueeze(-2)
            init_g2 = repeat(self.init_state_g2, 'h d1 d2 -> b h d1 d2', b=batch)
        else:
            kv_g2 = k_g2 * v_g2
            init_g2 = repeat(self.init_state_g2, 'h d -> b h d', b=batch)
        
        
        # Run scans
        # Forward scan (G1): starts at t=0, moves forward
        state_g1 = forward_scan(
            kv=kv_g1,
            alpha=params_g1['alpha'],
            beta=params_g1['beta'],
            gamma=params_g1['gamma'],
            init_state=init_g1,
        )
        
        # Backward scan (G2): starts at t=-1, moves backward
        state_g2 = backward_scan(
            kv=kv_g2,
            alpha=params_g2['alpha'],
            beta=params_g2['beta'],
            gamma=params_g2['gamma'],
            init_state=init_g2,
        )
        
        
        # Gate state injection back into hiddens (mode-dependent readout)
        # h_t = h_t + gate_t * state_read_t
        # For outer mode: query with k to retrieve v (k-based readout)
        # For elementwise mode: state is already the right shape
        scale = config.headdim ** -0.5

        state_g1_read = None
        state_g2_read = None
        
        # G1 readout
        if modes[0] == 'outer':
            state_g1_read = torch.einsum('bthjk,bthj->bthk', state_g1, k_g1) * scale
            out_g1 = v_g1 + params_g1['gate'] * state_g1_read
        else:
            out_g1 = v_g1 + params_g1['gate'] * state_g1
        
        # G2 readout
        if modes[1] == 'outer':
            state_g2_read = torch.einsum('bthjk,bthj->bthk', state_g2, k_g2) * scale
            out_g2 = v_g2 + params_g2['gate'] * state_g2_read
        else:
            out_g2 = v_g2 + params_g2['gate'] * state_g2
        
        # G3 readout: depthwise conv over sequence
        def conv_g3(x_heads: torch.Tensor) -> torch.Tensor:
            b_, t_, h_, d_ = x_heads.shape
            x_flat = rearrange(x_heads, 'b t h d -> b (h d) t')
            y = self.conv_g3(x_flat)
            return rearrange(y, 'b (h d) t -> b t h d', h=h_)

        conv_g3_out = conv_g3(v_g3)
        out_g3 = v_g3 + params_g3['gate'] * conv_g3_out
        
        # Step 5: Passthrough for G4 (standard RoPE only)
        out_g4 = v_g4  # No scan, just passthrough

        # Step 6: Low-rank attention (full Q against downsampled KV)
        v_all = torch.cat([out_g1, out_g2, out_g3, out_g4], dim=-2)

        target_len = max(1, int(math.sqrt(seq_len * 1.5)))

        k_flat = rearrange(torch.cat([k_g1_attn, k_g2_attn, k_g3_attn, k_g4_attn], dim=-2), 'b t h d -> b t (h d)')
        v_flat = rearrange(v_all, 'b t h d -> b t (h d)')
        k_down_flat = self.attn_downsample(k_flat, (target_len,))
        v_down_flat = self.attn_downsample(v_flat, (target_len,))
        k_down = rearrange(k_down_flat, 'b t (h d) -> b t h d', h=config.nheads_total)
        v_down = rearrange(v_down_flat, 'b t (h d) -> b t h d', h=config.nheads_total)

        # Apply data-dependent RoPE to Q
        q_g1 = apply_data_dependent_rope(q_g1, params_g1['rope_freq'])
        q_g2 = apply_data_dependent_rope(q_g2, params_g2['rope_freq'])
        q_g3 = apply_data_dependent_rope(q_g3, params_g3['rope_freq'])

        # Standard RoPE for G3/G4 on Q (after DD-RoPE for G3)
        q_g3 = apply_rope(q_g3, seq_len)
        q_g4 = apply_rope(q_g4, seq_len)

        # Downsample rope_freq for K (apply RoPE after downsampling)
        if target_len == seq_len:
            rope_g1 = params_g1['rope_freq']
            rope_g2 = params_g2['rope_freq']
            rope_g3 = params_g3['rope_freq']
        else:
            rope_g1_flat = rearrange(params_g1['rope_freq'], 'b t h d -> b t (h d)')
            rope_g2_flat = rearrange(params_g2['rope_freq'], 'b t h d -> b t (h d)')
            rope_g3_flat = rearrange(params_g3['rope_freq'], 'b t h d -> b t (h d)')
            rope_g1_flat = self.attn_rope_downsample(rope_g1_flat, (target_len,))
            rope_g2_flat = self.attn_rope_downsample(rope_g2_flat, (target_len,))
            rope_g3_flat = self.attn_rope_downsample(rope_g3_flat, (target_len,))
            rope_g1 = rearrange(rope_g1_flat, 'b t (h d) -> b t h d', h=nph)
            rope_g2 = rearrange(rope_g2_flat, 'b t (h d) -> b t h d', h=nph)
            rope_g3 = rearrange(rope_g3_flat, 'b t (h d) -> b t h d', h=nph)

        k_g1_d, k_g2_d, k_g3_d, k_g4_d = (
            k_down[..., :nph, :],
            k_down[..., nph:2*nph, :],
            k_down[..., 2*nph:3*nph, :],
            k_down[..., 3*nph:, :],
        )

        k_g1_d = apply_data_dependent_rope(k_g1_d, rope_g1)
        k_g2_d = apply_data_dependent_rope(k_g2_d, rope_g2)
        k_g3_d = apply_data_dependent_rope(k_g3_d, rope_g3)

        # Standard RoPE for G3/G4 on K (after DD-RoPE for G3)
        k_g3_d = apply_rope(k_g3_d, target_len)
        k_g4_d = apply_rope(k_g4_d, target_len)

        # Concatenate for SDPA
        q_all = torch.cat([q_g1, q_g2, q_g3, q_g4], dim=-2)
        k_all = torch.cat([k_g1_d, k_g2_d, k_g3_d, k_g4_d], dim=-2)

        # Transpose for SDPA: (B, T, H, D) -> (B, H, T, D)
        q_sdpa = q_all.transpose(1, 2)
        k_sdpa = k_all.transpose(1, 2)
        v_sdpa = v_down.transpose(1, 2)

        # Acausal attention (mask ignored for low-rank)
        attn_out = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=None,
            is_causal=False,
        )

        # Step 7: Multifocal attention over routed windows
        k_full_g1 = apply_data_dependent_rope(k_g1_attn, params_g1['rope_freq'])
        k_full_g2 = apply_data_dependent_rope(k_g2_attn, params_g2['rope_freq'])
        k_full_g3 = apply_data_dependent_rope(k_g3_attn, params_g3['rope_freq'])
        k_full_g3 = apply_rope(k_full_g3, seq_len)
        k_full_g4 = apply_rope(k_g4_attn, seq_len)

        q_full = q_all
        k_full = torch.cat([k_full_g1, k_full_g2, k_full_g3, k_full_g4], dim=-2)

        q_full = q_full.transpose(1, 2)
        k_full = k_full.transpose(1, 2)
        v_full = v_all.transpose(1, 2)

        attn_out = attn_out.transpose(1, 2)
        attn_out = rearrange(attn_out, 'b t h d -> b t (h d)')

        router_in = self.router_norm(attn_out)
        router_scores = self.router(router_in).squeeze(-1)
        router_temp = F.softplus(self.router_temp) + 1e-4
        router_scores = router_scores / router_temp
        num_centers = max(1, int(round(seq_len ** (1.0 / 3.0))))
        window = max(1, int(round(math.sqrt(seq_len))))
        half = window // 2

        topk = torch.topk(router_scores, k=num_centers, dim=1)
        topk_idx = topk.indices
        topk_w = torch.softmax(topk.values, dim=1)

        offsets = torch.arange(-half, half + 1, device=attn_out.device)
        idx = topk_idx.unsqueeze(-1) + offsets
        idx = idx.clamp(0, seq_len - 1)

        idx_flat = idx.reshape(batch, -1)
        idx_exp = idx_flat[:, None, :, None].expand(
            batch, config.nheads_total, idx_flat.shape[1], config.headdim
        )

        q_flat = torch.gather(q_full, dim=2, index=idx_exp)
        k_flat = torch.gather(k_full, dim=2, index=idx_exp)
        v_flat = torch.gather(v_full, dim=2, index=idx_exp)

        q_win = q_flat.view(batch, config.nheads_total, num_centers, offsets.numel(), config.headdim)
        k_win = k_flat.view(batch, config.nheads_total, num_centers, offsets.numel(), config.headdim)
        v_win = v_flat.view(batch, config.nheads_total, num_centers, offsets.numel(), config.headdim)

        q_win = q_win.permute(0, 2, 1, 3, 4).contiguous()
        k_win = k_win.permute(0, 2, 1, 3, 4).contiguous()
        v_win = v_win.permute(0, 2, 1, 3, 4).contiguous()

        q_win = q_win.view(batch * num_centers * config.nheads_total, offsets.numel(), config.headdim)
        k_win = k_win.view(batch * num_centers * config.nheads_total, offsets.numel(), config.headdim)
        v_win = v_win.view(batch * num_centers * config.nheads_total, offsets.numel(), config.headdim)

        win_out = F.scaled_dot_product_attention(
            q_win, k_win, v_win,
            attn_mask=None,
            is_causal=False,
        )

        win_out = win_out.view(batch, num_centers, config.nheads_total, offsets.numel(), config.headdim)
        win_out = win_out.permute(0, 1, 3, 2, 4).contiguous()
        win_out = win_out.view(batch, num_centers * offsets.numel(), config.d_expanded)

        local_sum = torch.zeros(batch, seq_len, config.d_expanded, device=attn_out.device, dtype=attn_out.dtype)
        local_weight = torch.zeros(batch, seq_len, device=attn_out.device, dtype=attn_out.dtype)

        idx_out = idx.reshape(batch, -1)
        idx_out_exp = idx_out.unsqueeze(-1).expand(batch, idx_out.shape[1], config.d_expanded)
        win_weight = topk_w.unsqueeze(-1).expand(batch, num_centers, offsets.numel()).reshape(batch, -1)
        win_weight_exp = win_weight.unsqueeze(-1).expand(batch, win_weight.shape[1], config.d_expanded)

        local_sum.scatter_add_(1, idx_out_exp, win_out * win_weight_exp)
        local_weight.scatter_add_(1, idx_out, win_weight)

        local_out = local_sum / local_weight.clamp_min(1e-6).unsqueeze(-1)

        local_gate = torch.sigmoid(self.local_gate)
        attn_out = attn_out + local_gate * local_out
        # attn_out already (B, T, H*D)

        if do_debug:
            eps = 1e-6

            def _rms(t: torch.Tensor) -> float:
                t = t.detach()
                if t.is_sparse:
                    t = t.coalesce().values()
                return t.float().pow(2).mean().sqrt().item()

            def _stats(t: torch.Tensor) -> dict:
                t = t.detach()
                if t.is_sparse:
                    t = t.coalesce().values()
                return {
                    'mean': t.float().mean().item(),
                    'min': t.min().item(),
                    'max': t.max().item(),
                }

            def _gate_stats(g: torch.Tensor) -> dict:
                g = g.detach()
                return {
                    'mean': g.float().mean().item(),
                    'low': (g < 0.01).float().mean().item(),
                    'high': (g > 0.99).float().mean().item(),
                }

            def _seq_rms(t: torch.Tensor, outer: bool) -> dict:
                if outer:
                    dims = (0, 2, 3, 4)
                else:
                    dims = (0, 2, 3)
                seq_rms = t.detach().float().pow(2).mean(dim=dims).sqrt()
                mid = seq_rms.shape[0] // 2
                return {
                    't0': seq_rms[0].item(),
                    'tmid': seq_rms[mid].item(),
                    'tend': seq_rms[-1].item(),
                    'min': seq_rms.min().item(),
                    'max': seq_rms.max().item(),
                }

            def _param_block(params: dict) -> dict:
                alpha = params['alpha']
                beta = params['beta']
                gamma = params['gamma']
                delta = gamma + beta / (alpha + eps)
                lam = gamma / (delta + eps)
                return {
                    'alpha': _stats(alpha),
                    'beta': _stats(beta),
                    'gamma': _stats(gamma),
                    'delta': _stats(delta),
                    'lambda': _stats(lam),
                    'gate': _gate_stats(params['gate']),
                }

            debug = {
                'g1': _param_block(params_g1),
                'g2': _param_block(params_g2),
                'g3': _param_block(params_g3),
                'rms': {
                    'k_g1': _rms(k_g1),
                    'v_g1': _rms(v_g1),
                    'state_g1': _rms(state_g1),
                    'state_read_g1': _rms(state_g1_read) if state_g1_read is not None else 0.0,
                    'out_g1': _rms(out_g1),
                    'k_g2': _rms(k_g2),
                    'v_g2': _rms(v_g2),
                    'state_g2': _rms(state_g2),
                    'state_read_g2': _rms(state_g2_read) if state_g2_read is not None else 0.0,
                    'out_g2': _rms(out_g2),
                    'v_g3': _rms(v_g3),
                    'conv_g3': _rms(conv_g3_out),
                    'out_g3': _rms(out_g3),
                    'attn_out': _rms(attn_out),
                },
                'seq_rms': {
                    'g1': _seq_rms(state_g1, modes[0] == 'outer'),
                    'g2': _seq_rms(state_g2, modes[1] == 'outer'),
                },
            }

            self.last_debug = debug
        
        # Step 7: Down-projection
        out = self.down_proj(attn_out)
        
        # Residual connection
        out = out + residual
        
        return out
