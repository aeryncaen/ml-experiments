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

from .scan import forward_scan, backward_scan, centered_scan
from .rope import apply_data_dependent_rope, apply_rope, apply_paired_rope


@dataclass
class USBConfig:
    """Configuration for USB block."""
    d_model: int
    headdim: int = 64
    expansion_factor: int = 2
    qkv_rank: int = 16  # Rank for per-group low-rank QKV deltas (0 = no sharing, use fused)
    paired_heads: bool = False  # Pair adjacent heads for cross-head mixing
    
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


class LowRankQKVProjection(nn.Module):
    """
    QKV projection with shared base + per-group low-rank deltas.
    
    output_i = W_base @ x + (W_down_i @ W_up_i) @ x
    
    - W_base: shared across all groups (bulk of params/compute)
    - W_down_i, W_up_i: per-group low-rank delta (allows specialization)
    
    Key difference from dimension-sharing: each group gets DIFFERENT outputs
    while sharing the bulk of the weights. The delta starts at zero so
    initially all groups behave the same, then specialize during training.
    """
    
    def __init__(self, d_input: int, d_group: int, n_groups: int = 4, rank: int = 16):
        super().__init__()
        self.d_input = d_input
        self.d_group = d_group
        self.n_groups = n_groups
        self.rank = rank
        
        # Shared base projection: d_input -> 3 * d_group (Q, K, V for one group)
        # This is reused for all groups
        self.qkv_base = nn.Linear(d_input, 3 * d_group, bias=False)
        
        # Per-group low-rank deltas: W_delta = W_down @ W_up
        # W_down: (d_input, rank), W_up: (rank, 3 * d_group)
        # Initialize W_down small, W_up to zero -> delta starts at zero
        self.qkv_down = nn.ParameterList([
            nn.Parameter(torch.randn(d_input, rank) * 0.01)
            for _ in range(n_groups)
        ])
        self.qkv_up = nn.ParameterList([
            nn.Parameter(torch.zeros(rank, 3 * d_group))
            for _ in range(n_groups)
        ])
        
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, d_input)
        
        Returns:
            Q, K, V: each (batch, seq_len, n_groups * d_group)
        """
        # Shared base (computed once, reused)
        base = self.qkv_base(x)  # (batch, seq, 3 * d_group)
        
        q_groups, k_groups, v_groups = [], [], []
        for i in range(self.n_groups):
            # Low-rank delta for this group: x @ W_down @ W_up
            delta = x @ self.qkv_down[i] @ self.qkv_up[i]  # (batch, seq, 3 * d_group)
            
            # Combine: base + delta (different output per group!)
            qkv_i = base + delta
            q_i, k_i, v_i = qkv_i.chunk(3, dim=-1)
            
            q_groups.append(q_i)
            k_groups.append(k_i)
            v_groups.append(v_i)
        
        Q = torch.cat(q_groups, dim=-1)  # (batch, seq_len, n_groups * d_group)
        K = torch.cat(k_groups, dim=-1)
        V = torch.cat(v_groups, dim=-1)
        
        return Q, K, V


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
    
    Each scan head needs: α[t], c0[t], c1[t], c2[t], gate[t], rope_freq[t]
    We fuse these into a single projection per group for efficiency.
    """
    
    def __init__(self, d_input: int, nheads: int, headdim: int):
        super().__init__()
        self.nheads = nheads
        self.headdim = headdim
        
        # Fused projection for all per-head parameters:
        # - α: 1 scalar per head
        # - c0, c1, c2: 3 scalars per head  
        # - gate: headdim values per head (per-dimension gating)
        # - rope_freq: headdim // 2 values per head (rotation frequencies for pairs)
        self.n_scalar_params = 4  # α, c0, c1, c2
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
                c0, c1, c2: (batch, seq_len, nheads) - AB-2 coefficients
                gate: (batch, seq_len, nheads, headdim) - per-dim injection gates in (0, 1)
                rope_freq: (batch, seq_len, nheads, headdim // 2) - rotation frequencies
        """
        batch, seq_len, _ = x.shape
        
        # Fused projection
        out = self.proj(x)  # (batch, seq_len, nheads * total_per_head)
        out = rearrange(out, 'b t (h d) -> b t h d', h=self.nheads)
        
        # Split into components
        idx = 0
        
        # Scalars: α, c0, c1, c2
        alpha_raw = out[..., idx]
        idx += 1
        c0 = out[..., idx]
        idx += 1
        c1 = out[..., idx]
        idx += 1
        c2 = out[..., idx]
        idx += 1
        
        # Gate: per-dimension
        gate_raw = out[..., idx:idx + self.n_gate_params]
        idx += self.n_gate_params
        
        # RoPE frequencies
        rope_freq = out[..., idx:idx + self.n_rope_params]
        
        # Apply activation functions
        alpha = torch.exp(-F.softplus(alpha_raw))  # (0, 1)
        gate = torch.sigmoid(gate_raw)  # (0, 1) per dimension
        
        return {
            'alpha': alpha,
            'c0': c0,
            'c1': c1,
            'c2': c2,
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
        
        d_model = config.d_model
        d_expanded = config.d_expanded
        d_group = config.d_group
        nheads_per_group = config.nheads_per_group
        headdim = config.headdim
        
        # Step 1: Expansion
        self.expand = nn.Linear(d_model, d_expanded, bias=False)
        
        # Step 2: QKV projection
        if config.qkv_rank > 0:
            # Low-rank sharing: shared base + per-group low-rank deltas
            self.qkv_proj = LowRankQKVProjection(
                d_input=d_expanded,
                d_group=d_group,
                n_groups=4,
                rank=config.qkv_rank,
            )
        else:
            # No sharing: simple fused QKV projection
            self.qkv_proj = nn.Linear(d_expanded, 3 * d_expanded, bias=False)
        self._use_fused_qkv = config.qkv_rank == 0
        
        # Gated RMSNorm for Q, K, V
        self.q_norm = GatedRMSNorm(d_expanded)
        self.k_norm = GatedRMSNorm(d_expanded)
        self.v_norm = GatedRMSNorm(d_expanded)
        
        # QK bias (per-head, initialized to 1.0)
        # 4 groups, each with nheads_per_group heads
        self.q_bias = nn.Parameter(torch.ones(config.nheads_total, headdim))
        self.k_bias = nn.Parameter(torch.ones(config.nheads_total, headdim))
        
        # Per-head projections for scan groups (G1, G2, G3)
        # G4 is passthrough, no scan parameters needed
        self.scan_proj_g1 = PerHeadProjections(d_expanded, nheads_per_group, headdim)
        self.scan_proj_g2 = PerHeadProjections(d_expanded, nheads_per_group, headdim)
        self.scan_proj_g3 = PerHeadProjections(d_expanded, nheads_per_group, headdim)
        
        # Learnable initial states for scan heads (G1, G2, G3)
        # Shape: (nheads_per_group, headdim) per group
        self.init_state_g1 = nn.Parameter(torch.zeros(nheads_per_group, headdim))
        self.init_state_g2 = nn.Parameter(torch.zeros(nheads_per_group, headdim))
        self.init_state_g3 = nn.Parameter(torch.zeros(nheads_per_group, headdim))
        
        # Mark initial states as no weight decay
        self.init_state_g1._no_weight_decay = True
        self.init_state_g2._no_weight_decay = True
        self.init_state_g3._no_weight_decay = True
        
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
        
        # Pre-norm
        x = self.norm(x)
        
        # Step 1: Expansion with SiLU
        x_exp = F.silu(self.expand(x))  # (batch, seq_len, d_expanded)
        
        # Step 2: QKV projection
        if self._use_fused_qkv:
            # Simple fused: split output into Q, K, V
            qkv = self.qkv_proj(x_exp)  # (batch, seq_len, 3 * d_expanded)
            qkv = rearrange(qkv, 'b t (three d) -> three b t d', three=3)
            q_raw, k_raw, v_raw = qkv[0], qkv[1], qkv[2]
        else:
            # Low-rank: returns tuple directly
            q_raw, k_raw, v_raw = self.qkv_proj(x_exp)  # each (batch, seq_len, d_expanded)
        
        # Apply SiLU and gated RMSNorm
        q = self.q_norm(F.silu(q_raw))
        k = self.k_norm(F.silu(k_raw))
        v = self.v_norm(F.silu(v_raw))
        
        # Reshape to heads for bias application
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
        
        # Step 4: Directional scans for G1, G2, G3
        
        # Get per-head parameters for each scan group
        params_g1 = self.scan_proj_g1(x_exp)
        params_g2 = self.scan_proj_g2(x_exp)
        params_g3 = self.scan_proj_g3(x_exp)
        
        # Apply data-dependent RoPE to K before scanning
        k_g1 = apply_data_dependent_rope(k_g1, params_g1['rope_freq'])
        k_g2 = apply_data_dependent_rope(k_g2, params_g2['rope_freq'])
        k_g3 = apply_data_dependent_rope(k_g3, params_g3['rope_freq'])
        
        # Compute K·V for each group
        # For efficiency, we can think of this as the "input" to the scan
        # Shape: (batch, seq_len, nheads, headdim)
        kv_g1 = k_g1 * v_g1
        kv_g2 = k_g2 * v_g2
        kv_g3 = k_g3 * v_g3
        
        # Get initial states (expand to batch dimension)
        init_g1 = repeat(self.init_state_g1, 'h d -> b h d', b=batch)
        init_g2 = repeat(self.init_state_g2, 'h d -> b h d', b=batch)
        init_g3 = repeat(self.init_state_g3, 'h d -> b h d', b=batch)
        
        # Run scans
        # Forward scan (G1): starts at t=0, moves forward
        state_g1 = forward_scan(
            kv=kv_g1,
            alpha=params_g1['alpha'],
            c0=params_g1['c0'],
            c1=params_g1['c1'],
            c2=params_g1['c2'],
            init_state=init_g1,
        )
        
        # Backward scan (G2): starts at t=-1, moves backward
        state_g2 = backward_scan(
            kv=kv_g2,
            alpha=params_g2['alpha'],
            c0=params_g2['c0'],
            c1=params_g2['c1'],
            c2=params_g2['c2'],
            init_state=init_g2,
        )
        
        # Centered scan (G3): starts at midpoint, expands outward
        state_g3 = centered_scan(
            kv=kv_g3,
            alpha=params_g3['alpha'],
            c0=params_g3['c0'],
            c1=params_g3['c1'],
            c2=params_g3['c2'],
            init_state=init_g3,
        )
        
        # Gate state injection back into hiddens
        # h_t = h_t + gate_t * state_t
        # Here "h_t" is the value representation
        out_g1 = v_g1 + params_g1['gate'] * state_g1
        out_g2 = v_g2 + params_g2['gate'] * state_g2
        out_g3 = v_g3 + params_g3['gate'] * state_g3
        
        # Apply data-dependent RoPE to Q for attention
        q_g1 = apply_data_dependent_rope(q_g1, params_g1['rope_freq'])
        q_g2 = apply_data_dependent_rope(q_g2, params_g2['rope_freq'])
        q_g3 = apply_data_dependent_rope(q_g3, params_g3['rope_freq'])
        
        # Step 5: Passthrough for G4 (standard RoPE only)
        q_g4 = apply_rope(q_g4, seq_len)
        k_g4 = apply_rope(k_g4, seq_len)
        out_g4 = v_g4  # No scan, just passthrough
        
        # G3 also gets standard RoPE at attention level (in addition to DD-RoPE at scan level)
        q_g3 = apply_rope(q_g3, seq_len)
        # Note: k_g3 already has DD-RoPE, we apply standard RoPE on top
        k_g3_for_attn = apply_rope(k_g3, seq_len)
        
        # Step 6: Attention (standard or paired heads)
        # Concatenate all groups back together
        q_all = torch.cat([q_g1, q_g2, q_g3, q_g4], dim=-2)  # (batch, seq_len, nheads_total, headdim)
        k_all = torch.cat([k_g1, k_g2, k_g3_for_attn, k_g4], dim=-2)
        v_all = torch.cat([out_g1, out_g2, out_g3, out_g4], dim=-2)
        
        if config.paired_heads:
            # Paired head attention: adjacent heads' queries attend to each other's keys
            # This enables cross-head information mixing through position interleaving
            nheads = config.nheads_total
            headdim = config.headdim
            
            # Step 6a: Merge adjacent head pairs - (batch, seq, nheads, headdim) -> (batch, seq, nheads//2, headdim*2)
            q_paired = rearrange(q_all, 'b t (h2 two) d -> b t h2 (two d)', two=2)
            k_paired = rearrange(k_all, 'b t (h2 two) d -> b t h2 (two d)', two=2)
            
            # Step 6b: Apply paired RoPE (even positions for first half, odd for second half)
            q_paired = apply_paired_rope(q_paired, seq_len)
            k_paired = apply_paired_rope(k_paired, seq_len)
            
            # Step 6c: Reshape to double sequence length, restore head_dim
            # This interleaves adjacent heads along the sequence dimension
            # (batch, seq, nheads//2, headdim*2) -> (batch, seq*2, nheads//2, headdim)
            q_interleaved = rearrange(q_paired, 'b t h (two d) -> b (t two) h d', two=2)
            k_interleaved = rearrange(k_paired, 'b t h (two d) -> b (t two) h d', two=2)
            # V doesn't get RoPE, just interleave
            v_interleaved = rearrange(v_all, 'b t (h2 two) d -> b (t two) h2 d', two=2)
            
            # Step 6d: Scaled dot-product attention on doubled sequence
            scale = headdim ** -0.5
            attn_weights = torch.einsum('bthd,bshd->bhts', q_interleaved, k_interleaved) * scale
            
            # Causal mask for interleaved sequence (position i can attend to positions <= i)
            # Note: attention_mask from caller doesn't apply cleanly to interleaved positions
            seq_doubled = seq_len * 2
            causal_mask = torch.tril(torch.ones(seq_doubled, seq_doubled, device=x.device, dtype=torch.bool))
            attn_weights = attn_weights.masked_fill(~causal_mask, float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_out_interleaved = torch.einsum('bhts,bshd->bthd', attn_weights, v_interleaved)
            
            # Step 6e: Reshape back to original sequence length and head count
            # (batch, seq*2, nheads//2, headdim) -> (batch, seq, nheads, headdim)
            attn_out = rearrange(attn_out_interleaved, 'b (t two) h d -> b t (h two) d', two=2)
            attn_out = rearrange(attn_out, 'b t h d -> b t (h d)')
        else:
            # Standard full attention
            scale = config.headdim ** -0.5
            attn_weights = torch.einsum('bthd,bshd->bhts', q_all, k_all) * scale
            
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(~attention_mask, float('-inf'))
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_out = torch.einsum('bhts,bshd->bthd', attn_weights, v_all)
            attn_out = rearrange(attn_out, 'b t h d -> b t (h d)')
        
        # Step 7: Down-projection
        out = self.down_proj(attn_out)
        
        # Residual connection
        out = out + residual
        
        return out
