"""
Unified Sequence Block (USB) Implementation

    Fuses SSM-style scans and conv into a single expand-process-contract block.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .scan import forward_scan
from .rope import apply_data_dependent_rope


@dataclass
class USBConfig:
    """Configuration for USB block."""
    d_model: int
    headdim: int = 64
    expansion_factor: int = 2
    layer_idx: int = 0  # For future use (e.g., layer-dependent params)

    # G4 attention mode: "polarity_sparse" or "none"
    g4_attention: str = "polarity_sparse"
    g4_sparse_keys: int = 64
    g4_num_hash: int = 8
    g4_use_lsh: bool = True
    g4_use_key_selection: bool = True
    
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
    
    Each scan head needs: α, δ/ε/ζ (Simpson coeffs), gate, rope_freq
    We fuse these into a single projection per group for efficiency.
    """
    
    def __init__(self, d_input: int, nheads: int, headdim: int):
        super().__init__()
        self.nheads = nheads
        self.headdim = headdim
        
        # Fused projection for all per-head parameters:
        # - alpha: 1 scalar per head (decay)
        # - dt: 1 scalar per head (step size)
        # - simpson_logits: 3 scalars per head (δ/ε/ζ weights)
        # - gate: headdim values per head (per-dimension gating)
        # - rope_freq: headdim // 2 values per head (rotation frequencies for pairs)
        self.n_scalar_params = 5  # alpha, dt, simpson_logits(3)
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
                delta, epsilon, zeta: (batch, seq_len, nheads) - Simpson coefficients
                gate: (batch, seq_len, nheads, headdim) - per-dim injection gates in (0, 1)
                rope_freq: (batch, seq_len, nheads, headdim // 2) - rotation frequencies
        """
        batch, seq_len, _ = x.shape
        
        # Fused projection
        out = self.proj(x)  # (batch, seq_len, nheads * total_per_head)
        out = rearrange(out, 'b t (h d) -> b t h d', h=self.nheads)
        
        # Split into components
        idx = 0
        
        # Scalars: alpha, dt, simpson logits
        alpha_raw = out[..., idx]
        idx += 1
        dt_raw = out[..., idx]
        idx += 1
        simpson_logits = out[..., idx:idx + 3]
        idx += 3
        
        # Gate: per-dimension
        gate_raw = out[..., idx:idx + self.n_gate_params]
        idx += self.n_gate_params
        
        # RoPE frequencies
        rope_freq = out[..., idx:idx + self.n_rope_params]
        
        # Apply activation functions
        alpha = torch.exp(-F.softplus(alpha_raw))  # (0, 1)
        dt = F.softplus(dt_raw)
        simpson_bias = torch.tensor([0.0, math.log(4.0), 0.0], device=x.device, dtype=x.dtype)
        simpson_weights = F.softmax(simpson_logits + simpson_bias, dim=-1)
        delta = simpson_weights[..., 0] * dt * alpha.pow(2)
        epsilon = simpson_weights[..., 1] * dt * alpha
        zeta = simpson_weights[..., 2] * dt
        gate = torch.sigmoid(gate_raw)  # (0, 1) per dimension
        
        return {
            'alpha': alpha,
            'delta': delta,
            'epsilon': epsilon,
            'zeta': zeta,
            'gate': gate,
            'rope_freq': rope_freq,
        }


class PolaritySparseAttention(nn.Module):
    """Polarity-aware sparse attention (HAX-style masks)."""

    def __init__(
        self,
        nheads: int,
        headdim: int,
        sparse_keys: int = 64,
        num_hash: int = 8,
        use_lsh: bool = True,
        use_key_selection: bool = True,
    ):
        super().__init__()
        self.nheads = nheads
        self.headdim = headdim
        self.feat_dim = headdim * 2
        self.sparse_keys = sparse_keys
        self.num_hash = num_hash
        self.use_lsh = use_lsh
        self.use_key_selection = use_key_selection

        self.s1 = nn.Parameter(torch.zeros(1, 1, nheads, self.feat_dim))
        self.s2 = nn.Parameter(torch.zeros(1, 1, nheads, self.feat_dim))

        if self.use_key_selection:
            self.dx_proj_1 = nn.Linear(self.feat_dim * 2, self.feat_dim * 2)
            self.dx_proj_2 = nn.Linear(self.feat_dim * 2, self.feat_dim * 2)
            self.dx_proj_3 = nn.Linear(self.feat_dim * 2, 1)

        self.attn_gate = nn.Parameter(torch.zeros(nheads, 1))

    @staticmethod
    def _cumtopk(x: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cumulative top-k per position along last dim."""
        *batch_shape, size = x.shape
        x_flat = x.reshape(-1, size)
        mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=x.device))
        x_cumulative = x_flat.unsqueeze(1).expand(-1, size, -1)
        x_cumulative = x_cumulative.masked_fill(~mask.unsqueeze(0), float("-inf"))
        topk_values, topk_indices = x_cumulative.topk(k, dim=2)
        out_shape = batch_shape + [size, k]
        values_out = topk_values.reshape(*out_shape)
        indices_out = topk_indices.reshape(*out_shape)
        return values_out, indices_out

    @staticmethod
    def _lsh_sliding_window(mask: torch.Tensor, budget: int) -> torch.Tensor:
        """Keep up to budget matches per query from the right."""
        reversed_mask = mask.flip(dims=[-1])
        cumsum = torch.cumsum(reversed_mask.int(), dim=-1)
        keep = cumsum <= budget
        return (reversed_mask & keep).flip(dims=[-1])

    def _polarity_map(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        q_pos = F.relu(q)
        q_neg = F.relu(-q)
        k_pos = F.relu(k)
        k_neg = F.relu(-k)

        s1 = torch.sigmoid(self.s1)
        s2 = torch.sigmoid(self.s2)

        q_sim = torch.expm1(torch.cat([q_pos, q_neg], dim=-1) * s1)
        q_opp = torch.expm1(torch.cat([q_neg, q_pos], dim=-1) * s2)
        q_map = 0.5 * (q_sim + q_opp)

        k_map = torch.expm1(torch.cat([k_pos, k_neg], dim=-1))
        return q_map, k_map

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            q, k, v: (batch, seq_len, nheads, headdim)
            attention_mask: (batch, seq_len) or None
        Returns:
            (batch, seq_len, nheads, headdim)
        """
        bsz, seq_len, nheads, _ = q.shape
        device = q.device

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q_map, k_map = self._polarity_map(q, k)

        if attention_mask is not None:
            valid = attention_mask.to(torch.bool)
            q_map = q_map * valid[:, None, :, None]
            k_map = k_map * valid[:, None, :, None]
        else:
            valid = None

        causal = torch.arange(seq_len, device=device)
        causal_mask = causal[None, None, :] <= causal[None, :, None]
        causal_mask = causal_mask.unsqueeze(0)

        diag = torch.eye(seq_len, device=device, dtype=torch.bool).view(1, 1, seq_len, seq_len)

        budget = max(1, min(self.sparse_keys, seq_len))

        lsh_mask = torch.zeros(bsz, nheads, seq_len, seq_len, device=device, dtype=torch.bool)
        if self.use_lsh:
            assert self.num_hash < 64
            nq = q_map - q_map.mean(dim=2, keepdim=True)
            nk = k_map - k_map.mean(dim=2, keepdim=True)
            nq = F.normalize(nq, dim=-1)
            nk = F.normalize(nk, dim=-1)
            proj = torch.randn((self.feat_dim, self.num_hash), device=device, dtype=q.dtype)
            hq = torch.matmul(nq, proj)
            hk = torch.matmul(nk, proj)
            hq_bits = (hq > 0).to(torch.long)
            hk_bits = (hk > 0).to(torch.long)
            weights = (1 << torch.arange(self.num_hash, device=device, dtype=torch.long)).view(1, 1, 1, self.num_hash)
            hq_idx = (hq_bits * weights).sum(-1)
            hk_idx = (hk_bits * weights).sum(-1)
            lsh_mask = hq_idx.unsqueeze(-1) == hk_idx.unsqueeze(-2)
            lsh_mask = self._lsh_sliding_window(lsh_mask & causal_mask, budget=budget)

        key_selection_mask = torch.zeros(bsz, nheads, seq_len, seq_len, device=device, dtype=torch.bool)
        if self.use_key_selection:
            tq = q_map.detach().cumsum(dim=2)
            tq = F.normalize(tq, dim=-1)
            tk = k_map.detach()
            t2 = torch.cat([tk, tq], dim=-1)
            t2 = F.relu(self.dx_proj_1(t2))
            t2 = F.relu(self.dx_proj_2(t2))
            key_score = self.dx_proj_3(t2).squeeze(-1)
            if valid is not None:
                key_score = key_score.masked_fill(~valid[:, None, :], float("-inf"))
            _, key_idx = self._cumtopk(key_score, k=budget)

            bh = bsz * nheads
            key_idx = key_idx.reshape(bh, seq_len, budget)
            mask = torch.zeros(bh, seq_len, seq_len, device=device, dtype=torch.bool)
            bid = torch.arange(bh, device=device).view(-1, 1, 1).expand(bh, seq_len, budget)
            sid = torch.arange(seq_len, device=device).view(1, -1, 1).expand(bh, seq_len, budget)
            mask[bid, sid, key_idx] = True
            key_selection_mask = mask.view(bsz, nheads, seq_len, seq_len)

        sparse_mask = (lsh_mask | key_selection_mask | diag) & causal_mask
        if valid is not None:
            sparse_mask = sparse_mask & valid[:, None, None, :]

        scale = (self.feat_dim ** -0.5)
        attn_score = torch.matmul(q_map, k_map.transpose(-1, -2)) * scale
        attn_score = attn_score + (1.0 - sparse_mask.to(attn_score.dtype)) * torch.finfo(attn_score.dtype).min
        attn = F.softmax(attn_score, dim=-1)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2)

        if valid is not None:
            attn_out = attn_out * valid[:, :, None, None]

        gate = torch.sigmoid(self.attn_gate).view(1, 1, nheads, 1)
        return v.transpose(1, 2) + gate * attn_out


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
        
        # Step 2: KV projections
        self.k_proj = nn.Linear(d_expanded, d_expanded, bias=False)
        self.v_proj = nn.Linear(d_expanded, d_expanded, bias=False)

        # Gated RMSNorm for K, V
        self.k_norm = GatedRMSNorm(d_expanded)
        self.v_norm = GatedRMSNorm(d_expanded)

        # KV bias (per-head, initialized to 1.0)
        self.k_bias = nn.Parameter(torch.ones(nheads_total, headdim))
        self.v_bias = nn.Parameter(torch.ones(nheads_total, headdim))
        
        # Per-head projections for scan groups (G1, G2, G3)
        # G4 is passthrough, no scan parameters needed
        self.scan_proj_g1 = PerHeadProjections(d_expanded, nheads_per_group, headdim)
        self.scan_proj_g2 = PerHeadProjections(d_expanded, nheads_per_group, headdim)
        self.scan_proj_g3 = PerHeadProjections(d_expanded, nheads_per_group, headdim)

        # G3 depthwise conv (k=3) replacing centered scan (causal)
        self.conv_g3 = nn.Conv1d(
            d_group, d_group, kernel_size=3, padding=0, groups=d_group, bias=False
        )

        # G4 polarity-aware sparse attention
        if config.g4_attention == "polarity_sparse":
            self.g4_attn = PolaritySparseAttention(
                nheads=nheads_per_group,
                headdim=headdim,
                sparse_keys=config.g4_sparse_keys,
                num_hash=config.g4_num_hash,
                use_lsh=config.g4_use_lsh,
                use_key_selection=config.g4_use_key_selection,
            )
        else:
            self.g4_attn = None
        
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
        
        # Step 2: KV projection
        k = self.k_proj(x_exp)
        v = self.v_proj(x_exp)

        # Apply RMSNorm (no activation on K/V)
        k = self.k_norm(k)
        v = self.v_norm(v)

        # Reshape to heads
        k = rearrange(k, 'b t (h d) -> b t h d', h=config.nheads_total)
        v = rearrange(v, 'b t (h d) -> b t h d', h=config.nheads_total)

        # Apply KV bias (after norm)
        k = k + self.k_bias
        v = v + self.v_bias
        
        # Step 3: Channel split into 4 groups
        # Each group gets nheads_per_group heads
        nph = config.nheads_per_group
        k_g1, k_g2, k_g3, k_g4 = k[..., :nph, :], k[..., nph:2*nph, :], k[..., 2*nph:3*nph, :], k[..., 3*nph:, :]
        v_g1, v_g2, v_g3, v_g4 = v[..., :nph, :], v[..., nph:2*nph, :], v[..., 2*nph:3*nph, :], v[..., 3*nph:, :]
        
        # Step 4: Scan G1+G2 together; G3 uses a 3-wide conv
        
        # Get per-head parameters for each scan group
        params_g1 = self.scan_proj_g1(x_exp)
        params_g2 = self.scan_proj_g2(x_exp)
        params_g3 = self.scan_proj_g3(x_exp)
        
        # Apply data-dependent RoPE to K before scanning
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
        
        # Combine G1+G2 for a single scan
        k_g12 = torch.cat([k_g1, k_g2], dim=-2)
        v_g12 = torch.cat([v_g1, v_g2], dim=-2)
        gate_g12 = torch.cat([params_g1['gate'], params_g2['gate']], dim=-2)

        params_g12 = {
            'alpha': torch.cat([params_g1['alpha'], params_g2['alpha']], dim=-1),
            'delta': torch.cat([params_g1['delta'], params_g2['delta']], dim=-1),
            'epsilon': torch.cat([params_g1['epsilon'], params_g2['epsilon']], dim=-1),
            'zeta': torch.cat([params_g1['zeta'], params_g2['zeta']], dim=-1),
        }

        if modes[0] == 'outer':
            kv_g12 = k_g12.unsqueeze(-1) * v_g12.unsqueeze(-2)
            init_g12 = torch.cat([self.init_state_g1, self.init_state_g2], dim=0)
            init_g12 = repeat(init_g12, 'h d1 d2 -> b h d1 d2', b=batch)
        else:
            kv_g12 = k_g12 * v_g12
            init_g12 = torch.cat([self.init_state_g1, self.init_state_g2], dim=0)
            init_g12 = repeat(init_g12, 'h d -> b h d', b=batch)
        
        
        # Run scans
        # Forward scan (G1): starts at t=0, moves forward
        state_g1 = forward_scan(
            kv=kv_g1,
            alpha=params_g1['alpha'],
            delta=params_g1['delta'],
            epsilon=params_g1['epsilon'],
            zeta=params_g1['zeta'],
            init_state=init_g1,
        )
        
        # Forward scan (G1+G2): starts at t=0, moves forward
        state_g12 = forward_scan(
            kv=kv_g12,
            alpha=params_g12['alpha'],
            delta=params_g12['delta'],
            epsilon=params_g12['epsilon'],
            zeta=params_g12['zeta'],
            init_state=init_g12,
        )
        
        
        # Gate state injection back into hiddens (mode-dependent readout)
        # h_t = h_t + gate_t * state_read_t
        # For outer mode: query with k to retrieve v (k-based readout)
        # For elementwise mode: state is already the right shape
        scale = config.headdim ** -0.5

        state_g1_read = None
        state_g2_read = None
        
        # G1+G2 readout
        if modes[0] == 'outer':
            state_g12_read = torch.einsum('bthjk,bthj->bthk', state_g12, k_g12) * scale
            out_g12 = v_g12 + gate_g12 * state_g12_read
        else:
            state_g12_read = None
            out_g12 = v_g12 + gate_g12 * state_g12

        out_g1, out_g2 = out_g12[..., :nph, :], out_g12[..., nph:, :]
        state_g1, state_g2 = state_g12[..., :nph, ...], state_g12[..., nph:, ...]
        if state_g12_read is not None:
            state_g1_read, state_g2_read = (
                state_g12_read[..., :nph, :],
                state_g12_read[..., nph:, :],
            )
        else:
            state_g1_read = None
            state_g2_read = None
        
        # G3 readout: depthwise conv over sequence (causal)
        def conv_g3(x_heads: torch.Tensor) -> torch.Tensor:
            b_, t_, h_, d_ = x_heads.shape
            x_flat = rearrange(x_heads, 'b t h d -> b (h d) t')
            x_flat = F.pad(x_flat, (2, 0))
            y = self.conv_g3(x_flat)
            return rearrange(y, 'b (h d) t -> b t h d', h=h_)

        conv_g3_out = conv_g3(v_g3)
        out_g3 = v_g3 + params_g3['gate'] * conv_g3_out
        
        # Step 5: Passthrough for G4
        if self.g4_attn is not None:
            out_g4 = self.g4_attn(k_g4, k_g4, v_g4, attention_mask=attention_mask)
        else:
            out_g4 = v_g4  # No scan, just passthrough

        # Step 6: No attention - concatenate scan outputs
        attn_out = torch.cat([out_g1, out_g2, out_g3, out_g4], dim=-2)
        attn_out = rearrange(attn_out, 'b t h d -> b t (h d)')

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
                delta = params['delta']
                epsilon = params['epsilon']
                zeta = params['zeta']
                return {
                    'alpha': _stats(alpha),
                    'delta': _stats(delta),
                    'epsilon': _stats(epsilon),
                    'zeta': _stats(zeta),
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
                    'out_g4': _rms(out_g4),
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
