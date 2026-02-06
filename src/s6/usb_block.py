"""
Unified Sequence Block (USB) Implementation

Fuses SSM-style scans, attention, and MLP into a single expand-process-contract block.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .scan import forward_scan, backward_scan
from .rope import apply_data_dependent_rope


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
        
        # Step 2: KV projections
        self.k_proj = nn.Linear(d_expanded, d_expanded, bias=False)
        self.v_proj = nn.Linear(d_expanded, d_expanded, bias=False)

        # Gated RMSNorm for K, V
        self.k_norm = GatedRMSNorm(d_expanded)
        self.v_norm = GatedRMSNorm(d_expanded)

        # K bias (per-head, initialized to 1.0)
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

        # Apply K bias (after norm)
        k = k + self.k_bias
        
        # Step 3: Channel split into 4 groups
        # Each group gets nheads_per_group heads
        nph = config.nheads_per_group
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
