"""ULBBlock — Universal Learning Block.

Generalized from FusedGateBlock: a fused attention+MLP block that achieves
SSM-equivalent capabilities (state tracking, pattern matching) purely through
attention preprocessing, with zero sequential scan.

Architecture (forward pass):
    h_up = up_act(up_proj(x))                           # thin linear + Swish

    q, k, v = q_proj(h_up), k_proj(h_up), v_proj(h_up)
    q = q_norm(q) * q_bias       # per-head RMSNorm + post-norm bias
    k = k_norm(k) * k_bias       # (Mamba-3 style BC bias, init ones)

    k = causal_lerp(k, x)        # last 1/4 of head_dim blended with t-1
    q = acausal_lerp(q, x)       # last 1/2 of head_dim blended with t-1 and t+1

    dd_angles = rope.compute_dd_angles(x)
    q = rope(q, dd_angles)       # hybrid RoPE: fixed + data-dependent
    k = rope(k, dd_angles)

    y = attend(q, k, v)          # softmax, silu2, or blend

    y = attn_norm(y) * h_up      # skip-MULTIPLY (not skip-add)
    y = down_proj(down_act(y))                          # thin linear + Swish
    return y                      # caller does x = x + block(x)

Key design decisions (preserved from FusedGateBlock):
    - Skip-multiply, NOT skip-add
    - No internal residual — block returns delta only
    - V is left alone — no V blending
    - KV have bias, Q has no bias
    - Optional inner_dim expansion for wider QKV / attention
    - No conv — hurts retrieval
    - Learnable Swish (per-channel beta), not SiLU
    - Up/down are thin linears (d → d) with Swish activation
"""

from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import LearnableSwish
from .attention import BlendAttention, silu2_attention
from .lerp import AcausalLerp, CausalLerp, QTemporalConv
from .rope import HybridRoPE, PairedRoPE


@dataclass
class ULBConfig:
    """Configuration for a ULBBlock.

    Args:
        d_model:    Model dimension.
        n_heads:    Number of attention heads. d_model must be divisible by n_heads.
        paired:     If True, use paired-head attention (interleave adjacent heads
                    at 2x sequence length). Requires n_heads to be even.
        attn_mode:  Attention mechanism: 'softmax', 'silu2', or 'blend'.
        rope_base:  Base for fixed RoPE inverse frequencies.
        k_lerp_bias:     Initial bias for K causal lerp gate (default -2.0).
        q_lerp_bias:     Initial bias for Q acausal lerp gates (default -2.0).
        blend_gate_bias: Initial bias for blend attention gate (default -1.1, ~25% silu2).
        inner_dim:  Inner dimension for QKV and attention. If None, defaults to
                    d_model (no expansion). Must be divisible by n_heads, and
                    inner_dim // n_heads must be divisible by 4 (for hybrid RoPE).
    """
    d_model: int = 128
    n_heads: int = 4
    paired: bool = True
    attn_mode: Literal['softmax', 'silu2', 'blend'] = 'blend'
    rope_base: float = 10000.0
    k_lerp_bias: float = -2.0
    q_lerp_bias: float = -2.0
    blend_gate_bias: float = -1.1
    q_mix: Literal['none', 'lerp', 'conv2', 'conv3'] = 'lerp'
    k_lerp: bool = True
    swish_mode: Literal['learnable', 'silu'] = 'learnable'
    inner_dim: int = 96

    def __post_init__(self):
        if self.inner_dim == 0:
            self.inner_dim = self.d_model
        assert self.inner_dim % self.n_heads == 0, (
            f"inner_dim ({self.inner_dim}) must be divisible by n_heads ({self.n_heads})")
        if self.paired:
            assert self.n_heads % 2 == 0, (
                f"n_heads ({self.n_heads}) must be even for paired mode")
        head_dim = self.inner_dim // self.n_heads
        assert head_dim % 4 == 0, (
            f"head_dim ({head_dim}) must be divisible by 4 for hybrid RoPE")

    @property
    def head_dim(self) -> int:
        return self.inner_dim // self.n_heads


class ULBBlock(nn.Module):
    """Universal Learning Block.

    A fused attention+MLP block with SSM-equivalent preprocessing.
    No internal residual — caller adds residual. No internal pre-norm — caller pre-norms.

    Args:
        config: ULBConfig instance, or will be constructed from kwargs.
    """

    def __init__(self, config: ULBConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = ULBConfig(**kwargs)
        self.config = config

        d = config.d_model
        inner = config.inner_dim
        n_heads = config.n_heads
        head_dim = config.head_dim

        # --- Up/down projections (thin linear + activation) ---
        self.up_proj = nn.Linear(d, inner, bias=False)
        self.down_proj = nn.Linear(inner, d, bias=False)

        if config.swish_mode == 'learnable':
            self.up_act = LearnableSwish(inner)
            self.down_act = LearnableSwish(inner)
        elif config.swish_mode == 'silu':
            self.up_act = nn.SiLU()
            self.down_act = nn.SiLU()
        else:
            raise ValueError(f"Unknown swish_mode: {config.swish_mode}")

        self.aux_loss = 0.0

        # QKV: KV have bias, Q does not
        self.q_proj = nn.Linear(inner, inner, bias=False)
        self.k_proj = nn.Linear(inner, inner, bias=True)
        self.v_proj = nn.Linear(inner, inner, bias=True)

        # --- Post-attention norm (before skip-multiply) ---
        self.attn_norm = nn.RMSNorm(inner)

        # --- QK norm + post-norm bias (Mamba-3 style BC bias, init ones) ---
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.q_bias = nn.Parameter(torch.ones(head_dim))
        self.k_bias = nn.Parameter(torch.ones(head_dim))

        # --- Temporal lerps ---
        quarter_dim = head_dim // 4
        half_dim = head_dim // 2
        self.k_lerp: CausalLerp | None = None
        if config.k_lerp:
            self.k_lerp = CausalLerp(d, n_heads, quarter_dim, init_bias=config.k_lerp_bias)
        self.q_lerp: AcausalLerp | None = None
        self.q_conv: QTemporalConv | None = None
        if config.q_mix == 'lerp':
            self.q_lerp = AcausalLerp(d, n_heads, half_dim, init_bias=config.q_lerp_bias)
        elif config.q_mix == 'none':
            pass
        elif config.q_mix == 'conv2':
            self.q_conv = QTemporalConv(n_heads, half_dim, kernel_size=2)
        elif config.q_mix == 'conv3':
            self.q_conv = QTemporalConv(n_heads, half_dim, kernel_size=3)
        else:
            raise ValueError(f"Unknown q_mix mode: {config.q_mix}")

        # --- Hybrid RoPE ---
        self.rope = HybridRoPE(d, n_heads, head_dim, rope_base=config.rope_base)
        if config.paired:
            self.paired_rope = PairedRoPE(n_heads, head_dim, rope_base=config.rope_base)

        # --- Attention ---
        self.blend_attn: BlendAttention | None = None
        if config.attn_mode == 'blend':
            self.blend_attn = BlendAttention(
                d, n_heads, head_dim, init_bias=config.blend_gate_bias)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                blend_gate: torch.Tensor | None = None) -> torch.Tensor:
        """Dispatch to the configured attention mode.

        Args:
            q, k, v:    (B, H, T, D) — standard SDPA layout.
            blend_gate: (B, H, T, 1) — only used for 'blend' mode.
        """
        if self.config.attn_mode == 'silu2':
            return silu2_attention(q, k, v)
        elif self.config.attn_mode == 'blend':
            assert self.blend_attn is not None
            assert blend_gate is not None
            return self.blend_attn(q, k, v, blend_gate)
        else:
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, D) — pre-normed input from the stacking model.

        Returns:
            (B, T, D) — the delta to add to the residual stream.
            Side effect: sets self.aux_loss (scalar tensor or 0.0).
        """
        cfg = self.config
        b, t, d = x.shape
        n_heads = cfg.n_heads
        head_dim = cfg.head_dim

        # --- Up projection ---
        h_up = self.up_act(self.up_proj(x))

        # --- QKV projections ---
        q = self.q_proj(h_up).view(b, t, n_heads, head_dim)
        k = self.k_proj(h_up).view(b, t, n_heads, head_dim)
        v = self.v_proj(h_up).view(b, t, n_heads, head_dim)

        # --- QK norm + post-norm bias ---
        q = self.q_norm(q) * self.q_bias
        k = self.k_norm(k) * self.k_bias

        # --- Temporal lerps (before RoPE) ---
        if self.k_lerp is not None:
            k = self.k_lerp(k, x)
        if self.q_lerp is not None:
            q = self.q_lerp(q, x)
        elif self.q_conv is not None:
            q = self.q_conv(q)

        # --- Data-dependent rotation angles (shared for Q and K) ---
        dd_angles = self.rope.compute_dd_angles(x)

        # --- Blend gate (only for blend mode) ---
        blend_gate = None
        if cfg.attn_mode == 'blend':
            assert self.blend_attn is not None
            blend_gate = self.blend_attn.compute_gate(x)  # (B, H, T, 1)

        # --- RoPE + Attention ---
        if cfg.paired:
            n2 = n_heads // 2
            # Pair adjacent heads: (B, T, H, hd) -> (B, T, H/2, 2*hd)
            q = q.view(b, t, n2, head_dim * 2)
            k = k.view(b, t, n2, head_dim * 2)
            q = self.paired_rope(q, dd_angles)
            k = self.paired_rope(k, dd_angles)
            # Interleave to 2T sequence: (B, T, H/2, 2*hd) -> (B, 2T, H/2, hd)
            q = q.view(b, t * 2, n2, head_dim)
            k = k.view(b, t * 2, n2, head_dim)
            v = v.reshape(b, t * 2, n2, head_dim)
            # Transpose to SDPA layout: (B, H, T, D)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if blend_gate is not None:
                # Paired doubles T: reshape gate (B,H,T,1) -> (B,n2,2T,1)
                bg = blend_gate.view(b, n2, 2, t, 1)
                blend_gate = bg.permute(0, 1, 3, 2, 4).reshape(b, n2, t * 2, 1)
            y = self._attend(q, k, v, blend_gate)
            # Undo paired layout: (B, H/2, 2T, hd) -> (B, T, H, hd)
            y = y.transpose(1, 2).contiguous().view(b, t, n_heads, head_dim)
        else:
            q = self.rope(q, dd_angles)
            k = self.rope(k, dd_angles)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = self._attend(q, k, v, blend_gate)
            y = y.transpose(1, 2)

        y = y.contiguous().view(b, t, cfg.inner_dim)

        # --- Skip-multiply (NOT skip-add) ---
        y = self.attn_norm(y) * h_up

        # --- Down projection ---
        y = self.down_proj(self.down_act(y))

        return y
