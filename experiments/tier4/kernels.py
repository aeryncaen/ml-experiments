"""Custom op wrappers for CuTE-DSL flash attention and fused feature attention.

Registers CuTE-DSL flash_attn as a torch custom op so torch.compile treats it
as opaque (won't try to trace into CuTE-DSL compilation code).

Also provides a fused feature attention kernel for small shapes (N_f=48, D_f=16)
that CuTE-DSL flash_attn doesn't support (min hdim=64).
"""

import torch
import torch.nn.functional as F
from typing import Tuple

# ── CuTE-DSL flash attention custom op ──────────────────────────────────────

try:
    from flash_attn.cute.interface import _flash_attn_fwd, _flash_attn_bwd
    HAS_CUTE_FLASH = True
except ImportError:
    HAS_CUTE_FLASH = False


if HAS_CUTE_FLASH:

    class _CuteFlashAttn(torch.autograd.Function):
        """Thin wrapper around CuTE-DSL flash attention, opaque to torch.compile."""

        @staticmethod
        def forward(ctx, q, k, v, causal):
            out, lse = _flash_attn_fwd(q, k, v, causal=causal)
            ctx.save_for_backward(q, k, v, out, lse)
            ctx.causal = causal
            return out

        @staticmethod
        def backward(ctx, dout):
            q, k, v, out, lse = ctx.saved_tensors
            dq, dk, dv = _flash_attn_bwd(
                q, k, v, out, dout, lse,
                None,  # softmax_scale (auto)
                ctx.causal,
                0.0,   # softcap
            )
            return dq, dk, dv, None

    # Mark as opaque to torch.compile
    torch.compiler.allow_in_graph(_CuteFlashAttn)

    def flash_attn_func(q, k, v, causal=False):
        """Drop-in replacement for flash_attn_func, compatible with torch.compile."""
        return _CuteFlashAttn.apply(q, k, v, causal)

else:
    def flash_attn_func(q, k, v, causal=False):
        raise RuntimeError("flash_attn.cute not available")


# ── Fused feature attention (small shapes, batch-parallel) ──────────────────

class _FusedFeatureAttention(torch.autograd.Function):
    """Fused bmm-softmax-bmm for feature attention.

    Shapes: Q,K=(B, N_f, D_f), V=(B, N_f, D_f), out=(B, N_f, D_f)
    where B=batch*seq_len, N_f=48, D_f=16.

    Forward: out = softmax(Q @ K^T / sqrt(D_f)) @ V
    Uses F.scaled_dot_product_attention with memory-efficient backend
    via the (B, 1, N_f, D_f) reshape trick — single head, N_f "sequence".

    This avoids materializing the (B, N_f, N_f) score matrix in global memory
    while being compatible with torch.compile (unlike CuTE-DSL flash).
    """

    @staticmethod
    def forward(ctx, q, k, v):
        # (B, N_f, D_f) -> (B, 1, N_f, D_f)
        q4 = q.unsqueeze(1)
        k4 = k.unsqueeze(1)
        v4 = v.unsqueeze(1)
        # SDPA with memory-efficient backend — does NOT materialize N_f x N_f
        # score matrix for backward (stores only O(N_f) logsumexp per row)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION):
            out4 = F.scaled_dot_product_attention(q4, k4, v4, is_causal=False)
        out = out4.squeeze(1)
        ctx.save_for_backward(q, k, v, out)
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, out = ctx.saved_tensors
        # Recompute attention weights for backward (memory-efficient)
        scale = q.shape[-1] ** -0.5
        # For small N_f (48), materializing scores in backward is fine —
        # it's only (B, 48, 48) and we need it for the gradient computation
        scores = torch.bmm(q, k.transpose(-2, -1)) * scale
        weights = torch.softmax(scores, dim=-1)

        # dV = weights^T @ dout
        dv = torch.bmm(weights.transpose(-2, -1), dout)

        # dweights = dout @ V^T
        dweights = torch.bmm(dout, v.transpose(-2, -1))

        # dsoftmax: d_scores = weights * (dweights - (dweights * weights).sum(-1, keepdim=True))
        d_scores = weights * (dweights - (dweights * weights).sum(dim=-1, keepdim=True))
        d_scores = d_scores * scale

        # dQ = d_scores @ K, dK = d_scores^T @ Q
        dq = torch.bmm(d_scores, k)
        dk = torch.bmm(d_scores.transpose(-2, -1), q)

        return dq, dk, dv


torch.compiler.allow_in_graph(_FusedFeatureAttention)


def feature_attention(q, k, v, activation='softmax'):
    """Feature attention dispatcher.

    For softmax: uses fused SDPA-based implementation (memory-efficient).
    For silu/silu2: explicit bmm (no fused kernel available).

    Args:
        q, k, v: (B*T, N_f, D_f) tensors
        activation: 'softmax' | 'silu' | 'silu2'

    Returns:
        (B*T, N_f, D_f) attended output
    """
    # Ensure matching dtypes
    k = k.to(q.dtype)
    v = v.to(q.dtype)

    if activation == 'softmax':
        return _FusedFeatureAttention.apply(q, k, v)
    else:
        scale = k.shape[-1] ** -0.5
        scores = torch.bmm(q, k.transpose(-2, -1)) * scale
        if activation == 'silu':
            weights = F.silu(scores)
        elif activation == 'silu2':
            weights = F.silu(scores).square()
        else:
            raise ValueError(f"Unknown feature attention activation: {activation}")
        return torch.bmm(weights, v)
