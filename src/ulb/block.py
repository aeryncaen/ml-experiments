"""ULBBlock — Universal Learning Block.

Generalized from FusedGateBlock: a fused attention+MLP block that achieves
SSM-equivalent capabilities (state tracking, pattern matching) purely through
attention preprocessing, with zero sequential scan.

Architecture (forward pass):
    h_up = up_act(up_proj(x))                           # thin d→d linear + Swish

    q, k, v = q_proj(h_up), k_proj(h_up), v_proj(h_up) # d→inner_dim (upscale)
    q = q_norm(q) * q_bias       # per-head RMSNorm + post-norm bias
    k = k_norm(k) * k_bias       # (Mamba-3 style BC bias, init ones)

    k = k_mix(k, x)              # last 1/4 of head_dim: temporal mixing

    dd_angles = rope.compute_dd_angles(x)
    q = rope(q, dd_angles)       # hybrid RoPE: fixed + data-dependent
    k = rope(k, dd_angles)

    y = attend(q, k, v)          # softmax, silu2, or blend
    y = o_proj(y)                 # inner_dim→d (downscale)

    y = attn_norm(y) * h_up      # skip-MULTIPLY (not skip-add)
    y = down_proj(down_act(y))                          # thin d→d linear + Swish
    return y                      # caller does x = x + block(x)

Key design decisions (preserved from FusedGateBlock):
    - Skip-multiply, NOT skip-add
    - No internal residual — block returns delta only
    - V is left alone — no V blending
    - KV have bias, Q has no bias
    - QKV projections handle inner_dim expansion (d→inner), o_proj compresses back
    - Up/down are thin d→d linears with Swish activation
    - No conv — hurts retrieval
    - Learnable Swish (per-channel beta), not SiLU
"""

import math
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .activations import LearnableSwish
from .attention import BlendAttention, silu2_attention
from .lerp import (CausalAdd, CausalLerp,
                    KAcausalAdd, KAcausalLerp, KTemporalConv)
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
        k_lerp_bias:     Initial bias for K temporal mixing gate (default -2.0).
        blend_gate_bias: Initial bias for blend attention gate (default -1.1, ~25% silu2).
        inner_ratio: Ratio of inner_dim to d_model (default 1.0). Inner dim is
                     snapped to nearest multiple of n_heads*4 for RoPE compatibility.
                     Values > 1.0 give wider QKV / attention at higher param cost.
    """
    d_model: int = 128
    n_heads: int = 4
    paired: bool = True
    attn_mode: Literal['softmax', 'silu2', 'blend'] = 'softmax'
    rope_base: float = 10000.0
    k_lerp_bias: float = -2.0
    blend_gate_bias: float = -1.1
    k_mix: Literal['none', 'lerp', 'add', 'acausal_lerp', 'acausal_add', 'conv2', 'conv3'] = 'lerp'
    is_causal: bool = True
    k_lerp: bool = True  # legacy compat — overridden by k_mix if set
    swish_mode: Literal['learnable', 'silu'] = 'learnable'
    inner_ratio: float = 1.75  # inner_dim = round(d_model * inner_ratio), snapped to n_heads*4

    # 2D feature-attention (replaces MLP with attention over features)
    feat_attn: bool = False
    feat_c_h: int | None = None  # channel height; default sqrt(d_model)
    feat_c_w: int | None = None  # channel width; default d_model // feat_c_h

    def __post_init__(self):
        if self.paired:
            assert self.n_heads % 2 == 0, (
                f"n_heads ({self.n_heads}) must be even for paired mode")
        # Snap inner_dim to nearest multiple of n_heads*4 (RoPE needs head_dim % 4 == 0)
        snap = self.n_heads * 4
        self._inner_dim = round(self.d_model * self.inner_ratio / snap) * snap
        assert self._inner_dim > 0, (
            f"inner_dim resolved to 0 (d_model={self.d_model}, inner_ratio={self.inner_ratio})")
        # Resolve 2D feature-attention channel dims
        if self.feat_attn:
            if self.feat_c_h is None:
                # Find the largest factor of d_model that is <= sqrt(d_model)
                s = int(self.d_model ** 0.5)
                while s > 1 and self.d_model % s != 0:
                    s -= 1
                self.feat_c_h = s
            if self.feat_c_w is None:
                self.feat_c_w = self.d_model // self.feat_c_h
            assert self.feat_c_h * self.feat_c_w == self.d_model, (
                f"feat_c_h * feat_c_w ({self.feat_c_h} * {self.feat_c_w}) "
                f"!= d_model ({self.d_model})")

    @property
    def inner_dim(self) -> int:
        return self._inner_dim

    @property
    def head_dim(self) -> int:
        return self._inner_dim // self.n_heads


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

        # --- Up/down projections (thin linear + activation, both at d_model) ---
        self.up_proj = nn.Linear(d, d, bias=False)
        self.down_proj = nn.Linear(d, d, bias=False)

        if config.swish_mode == 'learnable':
            self.up_act = LearnableSwish(d)
            self.down_act = LearnableSwish(d)
        elif config.swish_mode == 'silu':
            self.up_act = nn.SiLU()
            self.down_act = nn.SiLU()
        else:
            raise ValueError(f"Unknown swish_mode: {config.swish_mode}")

        self.aux_loss = 0.0

        # QKV: project from d_model up to inner_dim. KV have bias, Q does not.
        self.q_proj = nn.Linear(d, inner, bias=False)
        self.k_proj = nn.Linear(d, inner, bias=True)
        self.v_proj = nn.Linear(d, inner, bias=True)

        # Output projection: inner_dim back to d_model
        self.o_proj = nn.Linear(inner, d, bias=False)

        # --- Post-attention norm (before skip-multiply, at d_model) ---
        self.attn_norm = nn.RMSNorm(d)

        # --- QK norm + post-norm bias (Mamba-3 style BC bias, init ones) ---
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.q_bias = nn.Parameter(torch.ones(head_dim))
        self.k_bias = nn.Parameter(torch.ones(head_dim))

        # --- Temporal mixing ---
        quarter_dim = head_dim // 4

        # K mixing
        self.k_lerp: CausalLerp | CausalAdd | KAcausalLerp | KAcausalAdd | None = None
        self.k_conv: KTemporalConv | None = None
        k_mix = config.k_mix
        if k_mix == 'lerp':
            self.k_lerp = CausalLerp(d, n_heads, quarter_dim, init_bias=config.k_lerp_bias)
        elif k_mix == 'add':
            self.k_lerp = CausalAdd(d, n_heads, quarter_dim, init_bias=config.k_lerp_bias)
        elif k_mix == 'acausal_lerp':
            self.k_lerp = KAcausalLerp(d, n_heads, quarter_dim, init_bias=config.k_lerp_bias)
        elif k_mix == 'acausal_add':
            self.k_lerp = KAcausalAdd(d, n_heads, quarter_dim, init_bias=config.k_lerp_bias)
        elif k_mix == 'conv2':
            self.k_conv = KTemporalConv(n_heads, quarter_dim, kernel_size=2)
        elif k_mix == 'conv3':
            self.k_conv = KTemporalConv(n_heads, quarter_dim, kernel_size=3)
        elif k_mix == 'none':
            pass
        else:
            raise ValueError(f"Unknown k_mix mode: {k_mix}")

        # --- Hybrid RoPE ---
        self.rope = HybridRoPE(d, n_heads, head_dim, rope_base=config.rope_base)
        if config.paired:
            self.paired_rope = PairedRoPE(n_heads, head_dim, rope_base=config.rope_base)

        # --- Attention ---
        self.blend_attn: BlendAttention | None = None
        if config.attn_mode == 'blend':
            self.blend_attn = BlendAttention(
                d, n_heads, head_dim, init_bias=config.blend_gate_bias)

        # --- Sigmoid gate for attention delta (replaces skip-multiply) ---
        self.attn_gate_proj = nn.Linear(d, d, bias=True)
        nn.init.zeros_(self.attn_gate_proj.weight)
        nn.init.zeros_(self.attn_gate_proj.bias)

        # --- 2D Feature-attention (optional) ---
        self.use_feat_attn = config.feat_attn
        if config.feat_attn:
            C_h, C_w = config.feat_c_h, config.feat_c_w
            self.feat_c_h = C_h
            self.feat_c_w = C_w
            init_scale = (C_h * C_w) ** -0.5
            self.feat_wq = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
            self.feat_wk = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
            self.feat_wv = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
            self.feat_wo = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
            self.feat_norm = nn.RMSNorm(d)
            self.feat_out_norm = nn.RMSNorm(d)
            self.feat_gate_proj = nn.Linear(d, d, bias=True)
            nn.init.zeros_(self.feat_gate_proj.weight)
            nn.init.zeros_(self.feat_gate_proj.bias)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                blend_gate: torch.Tensor | None = None) -> torch.Tensor:
        """Dispatch to the configured attention mode.

        Args:
            q, k, v:    (B, H, T, D) — standard SDPA layout.
            blend_gate: (B, H, T, 1) — only used for 'blend' mode.
        """
        is_causal = self.config.is_causal
        if self.config.attn_mode == 'silu2':
            return silu2_attention(q, k, v, is_causal=is_causal)
        elif self.config.attn_mode == 'blend':
            assert self.blend_attn is not None
            assert blend_gate is not None
            return self.blend_attn(q, k, v, blend_gate, is_causal=is_causal)
        else:
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    def preprocess_qk(self, k: torch.Tensor, h_up: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply temporal lerps, compute dd_angles and blend gate.

        Args:
            k: (B, T, H, head_dim) — after QK norm + bias.
            h_up: (B, T, D) — up-projected hidden state (content signal).

        Returns:
            k: Preprocessed K (B, T, H, head_dim).
            dd_angles: Data-dependent rotation angles for RoPE.
            blend_gate: (B, H, T, 1) or None.
        """
        if self.k_lerp is not None:
            k = self.k_lerp(k, h_up)
        elif self.k_conv is not None:
            k = self.k_conv(k)

        dd_angles = self.rope.compute_dd_angles(h_up)

        blend_gate = None
        if self.config.attn_mode == 'blend':
            assert self.blend_attn is not None
            blend_gate = self.blend_attn.compute_gate(h_up)

        return k, dd_angles, blend_gate

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               dd_angles: torch.Tensor, blend_gate: torch.Tensor | None
               ) -> torch.Tensor:
        """Apply RoPE, paired layout if needed, and attention.

        Args:
            q: (B, T, H, head_dim) — preprocessed Q.
            k: (B, T, H, head_dim) — preprocessed K.
            v: (B, T, H, head_dim) — V (untouched).
            dd_angles: Data-dependent rotation angles.
            blend_gate: (B, H, T, 1) or None.

        Returns:
            y: (B, T, H, head_dim) — attention output.
        """
        cfg = self.config
        b, t = q.shape[:2]
        n_heads = cfg.n_heads
        head_dim = cfg.head_dim

        if cfg.paired:
            n2 = n_heads // 2
            q = q.view(b, t, n2, head_dim * 2)
            k = k.view(b, t, n2, head_dim * 2)
            q = self.paired_rope(q, dd_angles)
            k = self.paired_rope(k, dd_angles)
            q = q.view(b, t * 2, n2, head_dim)
            k = k.view(b, t * 2, n2, head_dim)
            v = v.reshape(b, t * 2, n2, head_dim)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if blend_gate is not None:
                bg = blend_gate.view(b, n2, 2, t, 1)
                blend_gate = bg.permute(0, 1, 3, 2, 4).reshape(b, n2, t * 2, 1)
            y = self._attend(q, k, v, blend_gate)
            y = y.transpose(1, 2).contiguous().view(b, t, n_heads, head_dim)
        else:
            q = self.rope(q, dd_angles)
            k = self.rope(k, dd_angles)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = self._attend(q, k, v, blend_gate)
            y = y.transpose(1, 2)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (B, T, D) — pre-normed input from the stacking model.

        Returns:
            (B, T, D) block output for this layer.
            Callers typically combine it with a residual path externally
            (e.g. ``x = x + block(norm(x))``).
            Side effect: sets self.aux_loss (scalar tensor or 0.0).
        """
        cfg = self.config
        b, t, d = x.shape
        n_heads = cfg.n_heads
        head_dim = cfg.head_dim

        # --- Up projection ---
        h_up = self.up_act(self.up_proj(x))

        # --- QKV projections + norm + bias ---
        q = self.q_proj(h_up).view(b, t, n_heads, head_dim)
        k = self.k_proj(h_up).view(b, t, n_heads, head_dim)
        v = self.v_proj(h_up).view(b, t, n_heads, head_dim)
        q = self.q_norm(q) * self.q_bias
        k = self.k_norm(k) * self.k_bias

        # --- Preprocessing (lerps, RoPE angles, blend gate) ---
        k, dd_angles, blend_gate = self.preprocess_qk(k, h_up)

        # --- Attention ---
        y = self.attend(q, k, v, dd_angles, blend_gate)
        y = y.contiguous().view(b, t, cfg.inner_dim)

        # --- Output projection (inner_dim → d_model) ---
        y = self.o_proj(y)

        # --- Gated attention delta accumulation (replaces skip-multiply) ---
        h_up = h_up + self.attn_norm(y) * torch.sigmoid(self.attn_gate_proj(h_up))

        # --- 2D Feature-attention (optional) ---
        if self.use_feat_attn:
            C_h, C_w = self.feat_c_h, self.feat_c_w
            feat_in = self.feat_norm(h_up).view(b, t, C_h, C_w)
            fQ = torch.einsum('...cd,cdef->...ef', feat_in, self.feat_wq)
            fK = torch.einsum('...cd,cdef->...ef', feat_in, self.feat_wk)
            fV = torch.einsum('...cd,cdef->...ef', feat_in, self.feat_wv)
            # Non-causal attention over C_h channels, each C_w dim
            fQ = fQ.reshape(b * t, 1, C_h, C_w)
            fK = fK.reshape(b * t, 1, C_h, C_w)
            fV = fV.reshape(b * t, 1, C_h, C_w)
            feat_out = F.scaled_dot_product_attention(fQ, fK, fV, is_causal=False)
            feat_out = feat_out.view(b, t, C_h, C_w)
            feat_delta = torch.einsum('...cd,cdef->...ef', feat_out, self.feat_wo)
            feat_delta = feat_delta.reshape(b, t, d)
            h_up = h_up + self.feat_out_norm(feat_delta) * torch.sigmoid(
                self.feat_gate_proj(h_up))

        # --- Down projection ---
        y = self.down_proj(self.down_act(h_up))

        return y


@dataclass
class ULB2DConfig:
    """Configuration for ULB2DBlock — true end-to-end 2D token processing.

    Tokens live as (B, T, C_h, C_w) throughout. All projections are
    (C_h, C_w, C_h, C_w) tensors applied via einsum('...cd,cdef->...ef', x, w).

    Seq-attention uses C_h channels as heads, C_w as head_dim.
    Feat-attention uses non-causal attention over C_h channels at each position.

    Args:
        c_h:     Channel height (number of seq-attn heads).
        c_w:     Channel width (head_dim for seq-attn). Must be divisible by 4 (RoPE).
        is_causal: Whether seq-attention is causal.
        rope_base: Base for fixed RoPE inverse frequencies.
        use_blend: Use blend attention (softmax + silu2 lerp) instead of pure softmax.
        blend_gate_bias: Initial blend gate bias.
        use_feat_attn: Whether to include the feat-attn sublayer.
        k_lerp_bias: Initial K temporal mixing gate bias.
    """
    c_h: int = 8
    c_w: int = 8
    is_causal: bool = True
    rope_base: float = 10000.0
    use_blend: bool = True
    blend_gate_bias: float = -1.1
    use_feat_attn: bool = True
    k_lerp_bias: float = -2.0

    def __post_init__(self):
        assert self.c_w % 4 == 0, (
            f"c_w ({self.c_w}) must be divisible by 4 for RoPE")

    @property
    def d_model(self) -> int:
        return self.c_h * self.c_w


class ULB2DBlock(nn.Module):
    """True end-to-end 2D Universal Learning Block.

    All tokens are (B, T, C_h, C_w). All projections are (C_h, C_w, C_h, C_w)
    tensor params applied via einsum. No nn.Linear anywhere.

    Architecture:
        h = up_act(proj2d(x, w_up))              # 2D up-projection + Swish
        q, k, v = proj2d(h, wq/wk/wv)           # 2D QKV
        k = k_lerp(k, h)                         # temporal mixing on last C_w//4
        q, k = hybrid_rope(q, k, dd_angles(h))   # fixed + data-dependent RoPE
        y = seq_attn(q, k, v)                    # C_h heads, C_w head_dim, causal
        y = proj2d(y, w_o)                       # 2D output projection
        h = h + norm(y) * sigmoid(proj2d(h, w_gate))  # gated delta
        [optional feat-attn sublayer]
        out = down_act(proj2d(h, w_down))         # 2D down-projection

    Caller does: x = x + block(norm(x))  where x is (B, T, C_h, C_w).
    """

    def __init__(self, config: ULB2DConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = ULB2DConfig(**kwargs)
        self.config = config
        C_h = config.c_h
        C_w = config.c_w
        D = config.d_model
        init_scale = (C_h * C_w) ** -0.5

        # --- 2D Up/Down projections ---
        self.w_up = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
        self.w_down = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
        self.up_act = LearnableSwish(D)
        self.down_act = LearnableSwish(D)

        # --- 2D QKV + output projections ---
        self.w_q = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
        self.w_k = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
        self.w_v = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
        self.w_o = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)

        # KV bias (as in ULBBlock — K and V have bias, Q does not)
        self.k_bias_param = nn.Parameter(torch.zeros(C_h, C_w))
        self.v_bias_param = nn.Parameter(torch.zeros(C_h, C_w))

        # --- QK norm + post-norm bias ---
        self.q_norm = nn.RMSNorm(C_w)
        self.k_norm = nn.RMSNorm(C_w)
        self.q_post_bias = nn.Parameter(torch.ones(C_w))
        self.k_post_bias = nn.Parameter(torch.ones(C_w))

        # --- K temporal lerp (last quarter of C_w) ---
        quarter = C_w // 4
        self.quarter = quarter
        # Gate projection: 2D proj from (C_h, C_w) → (C_h, quarter)
        self.w_k_gate = nn.Parameter(torch.zeros(C_h, C_w, C_h, quarter))
        self.k_gate_bias = nn.Parameter(torch.full((C_h, quarter), config.k_lerp_bias))

        # --- Hybrid RoPE ---
        self.fixed_pairs = C_w // 4
        self.dd_pairs = C_w // 4
        inv_freq = 1.0 / (config.rope_base ** (
            torch.arange(0, self.fixed_pairs * 2, 2).float() / C_w))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        # DD angle projection: 2D proj from (C_h, C_w) → (C_h, dd_pairs)
        self.w_dd = nn.Parameter(torch.zeros(C_h, C_w, C_h, self.dd_pairs))
        self.dd_bias = nn.Parameter(torch.zeros(C_h, self.dd_pairs))

        # --- Blend attention (optional) ---
        self.use_blend = config.use_blend
        if config.use_blend:
            # Gate: 2D proj from (C_h, C_w) → (C_h, 1)
            self.w_blend = nn.Parameter(torch.zeros(C_h, C_w, C_h, 1))
            self.blend_bias = nn.Parameter(torch.full((C_h, 1), config.blend_gate_bias))
            self.silu2_norm = nn.RMSNorm(C_w)

        # --- Post-attention norm + sigmoid gate ---
        self.attn_norm = nn.RMSNorm(D)
        self.w_attn_gate = nn.Parameter(torch.zeros(C_h, C_w, C_h, C_w))
        self.attn_gate_bias = nn.Parameter(torch.zeros(C_h, C_w))

        # --- Feat-attn sublayer (optional) ---
        self.use_feat_attn = config.use_feat_attn
        if config.use_feat_attn:
            self.feat_w_q = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
            self.feat_w_k = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
            self.feat_w_v = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
            self.feat_w_o = nn.Parameter(torch.randn(C_h, C_w, C_h, C_w) * init_scale)
            self.feat_norm = nn.RMSNorm(D)
            self.feat_out_norm = nn.RMSNorm(D)
            self.w_feat_gate = nn.Parameter(torch.zeros(C_h, C_w, C_h, C_w))
            self.feat_gate_bias = nn.Parameter(torch.zeros(C_h, C_w))

        self.aux_loss = 0.0

    def _proj2d(self, x: torch.Tensor, w: torch.Tensor,
                bias: torch.Tensor | None = None) -> torch.Tensor:
        """2D projection: x (..., C_h, C_w) @ w (C_h, C_w, out_h, out_w) -> (..., out_h, out_w)"""
        y = torch.einsum('...cd,cdef->...ef', x, w)
        if bias is not None:
            y = y + bias
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, T, C_h, C_w), returns (B, T, C_h, C_w)."""
        cfg = self.config
        C_h, C_w = cfg.c_h, cfg.c_w
        D = cfg.d_model
        b, t = x.shape[:2]

        # --- Up projection ---
        h = self._proj2d(x, self.w_up)                          # (B, T, C_h, C_w)
        h = self.up_act(h.reshape(b, t, D)).view(b, t, C_h, C_w)

        # --- QKV ---
        q = self._proj2d(h, self.w_q)                           # (B, T, C_h, C_w)
        k = self._proj2d(h, self.w_k, self.k_bias_param)
        v = self._proj2d(h, self.w_v, self.v_bias_param)

        # QK norm + post-norm bias (per-head RMSNorm over C_w)
        q = self.q_norm(q) * self.q_post_bias                  # (B, T, C_h, C_w)
        k = self.k_norm(k) * self.k_post_bias

        # --- K temporal lerp (last quarter of C_w) ---
        if t >= 2:
            qd = self.quarter
            gate = torch.sigmoid(
                self._proj2d(h, self.w_k_gate, self.k_gate_bias))  # (B, T, C_h, quarter)
            k_static = k[..., :-qd]
            k_cur = k[..., -qd:]
            k_prev = F.pad(k_cur[:, :-1], (0, 0, 0, 0, 1, 0))
            k = torch.cat([k_static, (1 - gate) * k_cur + gate * k_prev], dim=-1)

        # --- Hybrid RoPE (data-dependent + fixed) ---
        dd_deltas = self._proj2d(h, self.w_dd, self.dd_bias)    # (B, T, C_h, dd_pairs)
        dd_angles = dd_deltas.cumsum(dim=1)

        fp = self.fixed_pairs
        dp = self.dd_pairs
        pos = torch.arange(t, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)                 # (T, fp)
        cos_f = freqs.cos()[None, :, None, :]                   # (1, T, 1, fp)
        sin_f = freqs.sin()[None, :, None, :]

        for qk in (q, k):
            # Can't modify in-place during autograd, so we'll do it below
            pass

        # Apply RoPE to Q and K
        def apply_hybrid_rope(qk: torch.Tensor) -> torch.Tensor:
            from .rope import apply_rotary
            qk_fixed = torch.cat([qk[..., :fp], qk[..., fp:2*fp]], dim=-1)
            qk_dd = torch.cat([qk[..., 2*fp:2*fp+dp], qk[..., 2*fp+dp:]], dim=-1)
            qk_fixed = apply_rotary(qk_fixed, cos_f, sin_f)
            qk_dd = apply_rotary(qk_dd, dd_angles.cos(), dd_angles.sin())
            return torch.cat([qk_fixed[..., :fp], qk_fixed[..., fp:],
                              qk_dd[..., :dp], qk_dd[..., dp:]], dim=-1)

        q = apply_hybrid_rope(q)
        k = apply_hybrid_rope(k)

        # --- Seq-attention (C_h heads, C_w head_dim) ---
        # Permute to (B, C_h, T, C_w) for SDPA
        q_s = q.permute(0, 2, 1, 3)                            # (B, C_h, T, C_w)
        k_s = k.permute(0, 2, 1, 3)
        v_s = v.permute(0, 2, 1, 3)

        if self.use_blend:
            blend_gate = torch.sigmoid(
                self._proj2d(h, self.w_blend, self.blend_bias))  # (B, T, C_h, 1)
            blend_gate = blend_gate.permute(0, 2, 1, 3)         # (B, C_h, T, 1)
            y_soft = F.scaled_dot_product_attention(
                q_s, k_s, v_s, is_causal=cfg.is_causal)
            y_silu2 = self.silu2_norm(
                silu2_attention(q_s, k_s, v_s, is_causal=cfg.is_causal))
            y = (1 - blend_gate) * y_soft + blend_gate * y_silu2
        else:
            y = F.scaled_dot_product_attention(
                q_s, k_s, v_s, is_causal=cfg.is_causal)         # (B, C_h, T, C_w)

        y = y.permute(0, 2, 1, 3).contiguous()                  # (B, T, C_h, C_w)

        # --- Output projection ---
        y = self._proj2d(y, self.w_o)

        # --- Gated delta accumulation ---
        attn_gate = torch.sigmoid(
            self._proj2d(h, self.w_attn_gate, self.attn_gate_bias))
        y_normed = self.attn_norm(y.reshape(b, t, D)).view(b, t, C_h, C_w)
        h = h + y_normed * attn_gate

        # --- Feat-attn sublayer (optional) ---
        if self.use_feat_attn:
            feat_in = self.feat_norm(h.reshape(b, t, D)).view(b, t, C_h, C_w)
            fQ = self._proj2d(feat_in, self.feat_w_q)
            fK = self._proj2d(feat_in, self.feat_w_k)
            fV = self._proj2d(feat_in, self.feat_w_v)
            # Non-causal attention over C_h channels, each C_w dim
            fQ = fQ.reshape(b * t, 1, C_h, C_w)
            fK = fK.reshape(b * t, 1, C_h, C_w)
            fV = fV.reshape(b * t, 1, C_h, C_w)
            feat_out = F.scaled_dot_product_attention(fQ, fK, fV, is_causal=False)
            feat_out = feat_out.view(b, t, C_h, C_w)
            feat_out = self._proj2d(feat_out, self.feat_w_o)
            feat_delta = self.feat_out_norm(
                feat_out.reshape(b, t, D)).view(b, t, C_h, C_w)
            feat_gate = torch.sigmoid(
                self._proj2d(h, self.w_feat_gate, self.feat_gate_bias))
            h = h + feat_delta * feat_gate

        # --- Down projection ---
        y = self._proj2d(h, self.w_down)
        y = self.down_act(y.reshape(b, t, D)).view(b, t, C_h, C_w)

        return y


class UniversalSequenceBlock(nn.Module):
    """Pure attention compute block for TriplePoolGraphLearner.

    Operates entirely at inner_dim. The token pool banks handle
    up-projection (d_model → inner_dim) and down-projection
    (inner_dim → d_model). This block just does QKV → attention →
    skip-multiply against its input (the up-projected hidden state).

    Architecture:
        q, k, v = q_proj(x), k_proj(x), v_proj(x)   # inner→inner
        q = q_norm(q) * q_bias
        k = k_norm(k) * k_bias
        k = k_mix(k, x)
        q, k = rope(q, k, dd_angles)
        y = attend(q, k, v)
        y = attn_norm(y) * x                          # skip-multiply at inner_dim
        return y

    Args:
        config: ULBConfig instance, or will be constructed from kwargs.
    """

    def __init__(self, config: ULBConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = ULBConfig(**kwargs)
        self.config = config

        inner = config.inner_dim
        n_heads = config.n_heads
        head_dim = config.head_dim

        self.aux_loss = 0.0

        # QKV: all at inner_dim. KV have bias, Q does not.
        self.q_proj = nn.Linear(inner, inner, bias=False)
        self.k_proj = nn.Linear(inner, inner, bias=True)
        self.v_proj = nn.Linear(inner, inner, bias=True)

        # --- Post-attention norm (before skip-multiply, at inner_dim) ---
        self.attn_norm = nn.RMSNorm(inner)

        # --- QK norm + post-norm bias (Mamba-3 style BC bias, init ones) ---
        self.q_norm = nn.RMSNorm(head_dim)
        self.k_norm = nn.RMSNorm(head_dim)
        self.q_bias = nn.Parameter(torch.ones(head_dim))
        self.k_bias = nn.Parameter(torch.ones(head_dim))

        # --- Temporal mixing ---
        quarter_dim = head_dim // 4

        # K mixing (all at inner_dim)
        self.k_lerp: CausalLerp | CausalAdd | KAcausalLerp | KAcausalAdd | None = None
        self.k_conv: KTemporalConv | None = None
        k_mix = config.k_mix
        if k_mix == 'lerp':
            self.k_lerp = CausalLerp(inner, n_heads, quarter_dim, init_bias=config.k_lerp_bias)
        elif k_mix == 'add':
            self.k_lerp = CausalAdd(inner, n_heads, quarter_dim, init_bias=config.k_lerp_bias)
        elif k_mix == 'acausal_lerp':
            self.k_lerp = KAcausalLerp(inner, n_heads, quarter_dim, init_bias=config.k_lerp_bias)
        elif k_mix == 'acausal_add':
            self.k_lerp = KAcausalAdd(inner, n_heads, quarter_dim, init_bias=config.k_lerp_bias)
        elif k_mix == 'conv2':
            self.k_conv = KTemporalConv(n_heads, quarter_dim, kernel_size=2)
        elif k_mix == 'conv3':
            self.k_conv = KTemporalConv(n_heads, quarter_dim, kernel_size=3)
        elif k_mix == 'none':
            pass
        else:
            raise ValueError(f"Unknown k_mix mode: {k_mix}")

        # --- Hybrid RoPE (DD angles from inner_dim input) ---
        self.rope = HybridRoPE(inner, n_heads, head_dim, rope_base=config.rope_base)
        if config.paired:
            self.paired_rope = PairedRoPE(n_heads, head_dim, rope_base=config.rope_base)

        # --- Attention ---
        self.blend_attn: BlendAttention | None = None
        if config.attn_mode == 'blend':
            self.blend_attn = BlendAttention(
                inner, n_heads, head_dim, init_bias=config.blend_gate_bias)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                blend_gate: torch.Tensor | None = None) -> torch.Tensor:
        is_causal = self.config.is_causal
        if self.config.attn_mode == 'silu2':
            return silu2_attention(q, k, v, is_causal=is_causal)
        elif self.config.attn_mode == 'blend':
            assert self.blend_attn is not None
            assert blend_gate is not None
            return self.blend_attn(q, k, v, blend_gate, is_causal=is_causal)
        else:
            return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    def preprocess_qk(self, q: torch.Tensor, k: torch.Tensor, x: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.k_lerp is not None:
            k = self.k_lerp(k, x)
        elif self.k_conv is not None:
            k = self.k_conv(k)
        dd_angles = self.rope.compute_dd_angles(x)
        blend_gate = None
        if self.config.attn_mode == 'blend':
            assert self.blend_attn is not None
            blend_gate = self.blend_attn.compute_gate(x)
        return q, k, dd_angles, blend_gate

    def attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
               dd_angles: torch.Tensor, blend_gate: torch.Tensor | None
               ) -> torch.Tensor:
        cfg = self.config
        b, t = q.shape[:2]
        n_heads = cfg.n_heads
        head_dim = cfg.head_dim

        if cfg.paired:
            n2 = n_heads // 2
            q = q.view(b, t, n2, head_dim * 2)
            k = k.view(b, t, n2, head_dim * 2)
            q = self.paired_rope(q, dd_angles)
            k = self.paired_rope(k, dd_angles)
            q = q.view(b, t * 2, n2, head_dim)
            k = k.view(b, t * 2, n2, head_dim)
            v = v.reshape(b, t * 2, n2, head_dim)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if blend_gate is not None:
                bg = blend_gate.view(b, n2, 2, t, 1)
                blend_gate = bg.permute(0, 1, 3, 2, 4).reshape(b, n2, t * 2, 1)
            y = self._attend(q, k, v, blend_gate)
            y = y.transpose(1, 2).contiguous().view(b, t, n_heads, head_dim)
        else:
            q = self.rope(q, dd_angles)
            k = self.rope(k, dd_angles)
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            y = self._attend(q, k, v, blend_gate)
            y = y.transpose(1, 2)

        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input and output are at inner_dim.

        Args:
            x: (B, T, inner_dim) — up-projected input from pre-token bank.

        Returns:
            (B, T, inner_dim) — attention output with skip-multiply.
        """
        cfg = self.config
        b, t, _ = x.shape
        n_heads = cfg.n_heads
        head_dim = cfg.head_dim

        # --- QKV projections + norm + bias (inner→inner) ---
        q = self.q_proj(x).view(b, t, n_heads, head_dim)
        k = self.k_proj(x).view(b, t, n_heads, head_dim)
        v = self.v_proj(x).view(b, t, n_heads, head_dim)
        q = self.q_norm(q) * self.q_bias
        k = self.k_norm(k) * self.k_bias

        # --- Preprocessing (lerps, RoPE angles, blend gate) ---
        q, k, dd_angles, blend_gate = self.preprocess_qk(q, k, x)

        # --- Attention ---
        y = self.attend(q, k, v, dd_angles, blend_gate)
        y = y.contiguous().view(b, t, cfg.inner_dim)

        # --- Skip-multiply against input (NOT skip-add) ---
        y = self.attn_norm(y) * x

        return y


class UniversalTokenBlock(nn.Module):
    """Activated linear databank unit — d→d with activation.

    Used as token-pool experts in TriplePoolGraphLearner.
    These recall and compute over trained data, enriching inputs to
    or refining outputs from the sequence attention experts.

    Architecture:
        y = act(linear(x))

    Args:
        dim: Model dimension.
        swish_mode: 'learnable' or 'silu'.
    """

    def __init__(self, dim: int, swish_mode: str = 'learnable'):
        super().__init__()
        self.linear = nn.Linear(dim, dim, bias=False)
        if swish_mode == 'learnable':
            self.act = LearnableSwish(dim)
        elif swish_mode == 'silu':
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown swish_mode: {swish_mode}")
        self.aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.linear(x))


class TokenParamBank(nn.Module):
    """Banked token-level experts with batched dispatch — no Python loops.

    Stacks all expert weights into parameter banks. Each expert is
    act(linear(x)): a (in_dim, out_dim) weight + learnable swish beta.

    Supports asymmetric dims for up-projection (d_model → inner_dim) and
    down-projection (inner_dim → d_model).

    Weight sharing: when shared_fraction > 0, out_dim is split into
    shared + private slices. The shared slice is stored once (in_dim, shared_out),
    the private slice is per-expert (pool_size, in_dim, private_out).
    Same for beta. This reduces total params.

    Args:
        pool_size: Number of experts in the bank.
        in_dim: Input dimension.
        out_dim: Output dimension.
        top_k: Top-k experts per token.
        n_options: Router output size (pool_size + exit slots).
        swish_mode: 'learnable' or 'silu'.
        shared_fraction: Fraction of out_dim to share across experts (0.0 = none).
    """

    def __init__(self, pool_size: int, in_dim: int, out_dim: int, top_k: int = 2,
                 n_options: int | None = None, swish_mode: str = 'learnable',
                 shared_fraction: float = 0.0):
        super().__init__()
        self.pool_size = pool_size
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.top_k = top_k
        self.n_options = n_options if n_options is not None else pool_size + 1
        self.swish_mode = swish_mode

        # Weight sharing split
        self.shared_out = round(out_dim * shared_fraction) if shared_fraction > 0 else 0
        self.private_out = out_dim - self.shared_out

        if self.shared_out > 0:
            self.shared_weight = nn.Parameter(
                torch.randn(in_dim, self.shared_out) * (in_dim ** -0.5))
        else:
            self.shared_weight = None

        # Private weight bank: (pool_size, in_dim, private_out)
        self.weight_bank = nn.Parameter(
            torch.randn(pool_size, in_dim, self.private_out) * (in_dim ** -0.5))

        # Activation
        if swish_mode == 'learnable':
            if self.shared_out > 0:
                self.shared_beta = nn.Parameter(torch.ones(self.shared_out))
            else:
                self.shared_beta = None
            self.beta_bank = nn.Parameter(torch.ones(pool_size, self.private_out))
        else:
            self.shared_beta = None
            self.beta_bank = None

    def _get_weights(self, safe_idx: torch.Tensor, N: int
                     ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Gather per-token weights and betas.

        Returns:
            w: (N, top_k, in_dim, out_dim)
            beta: (N, top_k, out_dim) or None
        """
        # Private: (N*top_k, in_dim, private_out) → (N, top_k, in_dim, private_out)
        priv_w = self.weight_bank[safe_idx.view(-1)].view(
            N, self.top_k, self.in_dim, self.private_out)

        if self.shared_weight is not None:
            # Expand shared to (N, top_k, in_dim, shared_out)
            shared_w = self.shared_weight.unsqueeze(0).unsqueeze(0).expand(
                N, self.top_k, -1, -1)
            w = torch.cat([shared_w, priv_w], dim=-1)  # (N, top_k, in_dim, out_dim)
        else:
            w = priv_w

        beta = None
        if self.beta_bank is not None:
            priv_beta = self.beta_bank[safe_idx.view(-1)].view(
                N, self.top_k, self.private_out)
            if self.shared_beta is not None:
                shared_beta = self.shared_beta.unsqueeze(0).unsqueeze(0).expand(
                    N, self.top_k, -1)
                beta = torch.cat([shared_beta, priv_beta], dim=-1)
            else:
                beta = priv_beta

        return w, beta

    def forward(self, x_flat: torch.Tensor, topk_idx: torch.Tensor,
                topk_weights: torch.Tensor, all_exit: torch.Tensor) -> torch.Tensor:
        """Batched expert dispatch.

        Args:
            x_flat: (N, in_dim) flattened input tokens.
            topk_idx: (N, top_k) expert indices (may include exit slots >= pool_size).
            topk_weights: (N, top_k) normalized weights (exit slots already zeroed).
            all_exit: (N,) bool — tokens where all top-k are exit slots.

        Returns:
            (N, out_dim) output tokens. All-exit tokens get zeros.
        """
        N = x_flat.shape[0]
        safe_idx = topk_idx.clamp(max=self.pool_size - 1)

        w, beta = self._get_weights(safe_idx, N)

        # Batched matmul: (N, in_dim) @ (N, top_k, in_dim, out_dim) → (N, top_k, out_dim)
        expert_out = torch.einsum('ni,nkij->nkj', x_flat, w)

        # Activation
        if beta is not None:
            expert_out = expert_out * torch.sigmoid(beta * expert_out)
        else:
            expert_out = F.silu(expert_out)

        # Weighted sum over top-k
        out = (expert_out * topk_weights.unsqueeze(-1)).sum(dim=1)
        out = out.masked_fill(all_exit.unsqueeze(-1), 0.0)
        return out


class RouterParamBank(nn.Module):
    """Banked routers with batched dispatch and optional weight sharing.

    Each router is a Linear(in_dim, n_options) with no bias and no activation.
    Stacks all router weights into a bank for batched matmul.

    Weight sharing: when shared_fraction > 0, n_options is split into
    shared + private slices (same scheme as TokenParamBank).

    Args:
        pool_size: Number of routers in the bank.
        in_dim: Input dimension (what the router reads from).
        n_options: Output size per router (pool_size + exit slots for target pool).
        shared_fraction: Fraction of n_options to share across routers.
    """

    def __init__(self, pool_size: int, in_dim: int, n_options: int,
                 shared_fraction: float = 0.0):
        super().__init__()
        self.pool_size = pool_size
        self.in_dim = in_dim
        self.n_options = n_options

        self.shared_out = round(n_options * shared_fraction) if shared_fraction > 0 else 0
        self.private_out = n_options - self.shared_out

        if self.shared_out > 0:
            self.shared_weight = nn.Parameter(
                torch.randn(in_dim, self.shared_out) * (in_dim ** -0.5))
        else:
            self.shared_weight = None

        # Private weight bank: (pool_size, in_dim, private_out)
        self.weight_bank = nn.Parameter(
            torch.randn(pool_size, in_dim, self.private_out) * (in_dim ** -0.5))

    def forward(self, x: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Route using a specific expert's router.

        Args:
            x: (N, in_dim) input.
            expert_idx: Which router to use.

        Returns:
            (N, n_options) logits.
        """
        priv_out = torch.einsum('ni,ij->nj', x, self.weight_bank[expert_idx])
        if self.shared_weight is not None:
            shared_out = torch.einsum('ni,ij->nj', x, self.shared_weight)
            return torch.cat([shared_out, priv_out], dim=-1)
        return priv_out


def ulb_megatron_init_(model: nn.Module, n_layers: int, std: float = 0.02,
                       cutoff_factor: float = 2.0):
    """Full Megatron init for ULB-based models.

    - Input projections (up_proj, q_proj, k_proj, v_proj, w_gate, w_up, qkv_proj):
      trunc_normal(std)
    - Output projections (down_proj, o_proj, w_down): trunc_normal(std / sqrt(2 * n_layers))
    - Embeddings (token_embed): trunc_normal(std)
    - All biases: zero
    """
    out_std = std / math.sqrt(2.0 * n_layers)
    cutoff = cutoff_factor * std
    out_cutoff = cutoff_factor * out_std

    # Gate projections use zero-init from __init__; skip them in Megatron init
    _skip_linear = ('.attn_gate_proj', '.feat_gate_proj')

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(s) for s in _skip_linear):
                pass  # keep zero-init
            elif name.endswith('.down_proj') or name.endswith('.o_proj') or name.endswith('.w_down'):
                nn.init.trunc_normal_(module.weight, std=out_std,
                                      a=-out_cutoff, b=out_cutoff)
            elif name.endswith('.head'):
                pass  # weight-tied to embedding
            else:
                nn.init.trunc_normal_(module.weight, std=std,
                                      a=-cutoff, b=cutoff)
            if module.bias is not None:
                if not any(name.endswith(s) for s in _skip_linear):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=std,
                                  a=-cutoff, b=cutoff)

    # 4D feature-attention tensor params
    for name, param in model.named_parameters():
        if name.endswith('.feat_wo'):
            nn.init.trunc_normal_(param, std=out_std,
                                  a=-out_cutoff, b=out_cutoff)
        elif name.endswith(('.feat_wq', '.feat_wk', '.feat_wv')):
            nn.init.trunc_normal_(param, std=std,
                                  a=-cutoff, b=cutoff)
