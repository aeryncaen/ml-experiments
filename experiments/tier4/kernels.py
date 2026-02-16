"""Custom op wrappers for CuTE-DSL flash attention and fused feature attention.

Registers CuTE-DSL flash_attn as a torch custom op so torch.compile treats it
as opaque (won't try to trace into CuTE-DSL compilation code).

Fused triton kernel for feature attention: bmm-softmax-bmm in one kernel,
no intermediate score matrix in global memory. One program per batch element.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

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
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
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


# ── Fused feature attention triton kernels ──────────────────────────────────
#
# Shape: Q, K, V are (B, N_f, D_f) where B=batch*seq_len (~1M), N_f=48, D_f=16.
# One triton program per batch element. Everything fits in SRAM:
#   Q: 48*16 = 768 elements, K: 768, V: 768, scores: 48*48 = 2304
# Total ~4608 bf16 elements = ~9KB. Trivial.
#
# Forward:  O = softmax(Q @ K^T / sqrt(D_f)) @ V
# Backward: recompute scores+weights from saved Q,K,V, then compute dQ,dK,dV.

@triton.jit
def _fused_feat_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    scale,
    N_F: tl.constexpr,
    D_F: tl.constexpr,
):
    """One program per batch element. Computes fused QK^T -> softmax -> @V."""
    bid = tl.program_id(0)

    # Offsets for this batch element
    q_base = Q_ptr + bid * stride_qb
    k_base = K_ptr + bid * stride_kb
    v_base = V_ptr + bid * stride_vb
    o_base = O_ptr + bid * stride_ob

    # Load Q (N_F, D_F) and K (N_F, D_F)
    offs_n = tl.arange(0, N_F)
    offs_d = tl.arange(0, D_F)

    q = tl.load(q_base + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd)  # (N_F, D_F)
    k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)  # (N_F, D_F)

    # Scores = Q @ K^T * scale -> (N_F, N_F)
    scores = tl.dot(q, tl.trans(k)) * scale

    # Online softmax (row-wise)
    row_max = tl.max(scores, axis=1)  # (N_F,)
    scores = scores - row_max[:, None]
    weights = tl.exp(scores)
    row_sum = tl.sum(weights, axis=1)  # (N_F,)
    weights = weights / row_sum[:, None]

    # Load V (N_F, D_F)
    v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)  # (N_F, D_F)

    # Output = weights @ V -> (N_F, D_F)
    out = tl.dot(weights.to(v.dtype), v)

    # Store O (N_F, D_F)
    tl.store(o_base + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od, out)


@triton.jit
def _fused_feat_attn_bwd_kernel(
    Q_ptr, K_ptr, V_ptr, dO_ptr,
    dQ_ptr, dK_ptr, dV_ptr,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_dob, stride_don, stride_dod,
    stride_dqb, stride_dqn, stride_dqd,
    stride_dkb, stride_dkn, stride_dkd,
    stride_dvb, stride_dvn, stride_dvd,
    scale,
    N_F: tl.constexpr,
    D_F: tl.constexpr,
):
    """Backward: recompute weights from Q,K, then compute dQ,dK,dV."""
    bid = tl.program_id(0)

    q_base = Q_ptr + bid * stride_qb
    k_base = K_ptr + bid * stride_kb
    v_base = V_ptr + bid * stride_vb
    do_base = dO_ptr + bid * stride_dob
    dq_base = dQ_ptr + bid * stride_dqb
    dk_base = dK_ptr + bid * stride_dkb
    dv_base = dV_ptr + bid * stride_dvb

    offs_n = tl.arange(0, N_F)
    offs_d = tl.arange(0, D_F)

    # Reload Q, K, V, dO
    q = tl.load(q_base + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd).to(tl.float32)
    k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd).to(tl.float32)
    v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd).to(tl.float32)
    do = tl.load(do_base + offs_n[:, None] * stride_don + offs_d[None, :] * stride_dod).to(tl.float32)

    # Recompute softmax weights
    scores = tl.dot(q, tl.trans(k)) * scale
    row_max = tl.max(scores, axis=1)
    scores = scores - row_max[:, None]
    weights = tl.exp(scores)
    row_sum = tl.sum(weights, axis=1)
    weights = weights / row_sum[:, None]  # (N_F, N_F)

    # dV = weights^T @ dO  -> (N_F, D_F)
    dv = tl.dot(tl.trans(weights), do)

    # dweights = dO @ V^T  -> (N_F, N_F)
    dweights = tl.dot(do, tl.trans(v))

    # dsoftmax: d_scores = weights * (dweights - rowsum(dweights * weights))
    row_dot = tl.sum(dweights * weights, axis=1)  # (N_F,)
    d_scores = weights * (dweights - row_dot[:, None]) * scale

    # dQ = d_scores @ K  -> (N_F, D_F)
    dq = tl.dot(d_scores, k)

    # dK = d_scores^T @ Q  -> (N_F, D_F)
    dk = tl.dot(tl.trans(d_scores), q)

    # Store gradients (cast back to input dtype)
    in_dtype = Q_ptr.dtype.element_ty
    tl.store(dq_base + offs_n[:, None] * stride_dqn + offs_d[None, :] * stride_dqd, dq.to(in_dtype))
    tl.store(dk_base + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd, dk.to(in_dtype))
    tl.store(dv_base + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd, dv.to(in_dtype))


class _FusedFeatureAttention(torch.autograd.Function):
    """Fused bmm-softmax-bmm via triton. One program per batch element.

    No intermediate score matrix in global memory. Everything in SRAM.
    Q=K supported (pass same tensor for both).
    """

    @staticmethod
    def forward(ctx, q, k, v):
        B, N_F, D_F = q.shape
        assert k.shape == (B, N_F, D_F) and v.shape == (B, N_F, D_F)
        scale = D_F ** -0.5
        out = torch.empty_like(v)

        grid = (B,)
        _fused_feat_attn_fwd_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            scale,
            N_F=N_F,
            D_F=D_F,
            num_warps=1,
        )

        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v = ctx.saved_tensors
        B, N_F, D_F = q.shape
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        grid = (B,)
        _fused_feat_attn_bwd_kernel[grid](
            q, k, v, dout,
            dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            dout.stride(0), dout.stride(1), dout.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            ctx.scale,
            N_F=N_F,
            D_F=D_F,
            num_warps=1,
        )
        return dq, dk, dv


torch.compiler.allow_in_graph(_FusedFeatureAttention)


def feature_attention(q, k, v, activation='softmax'):
    """Feature attention dispatcher.

    For softmax: fused triton kernel (no global memory for scores).
    For silu/silu2: explicit bmm (no fused kernel).

    Args:
        q, k, v: (B*T, N_f, D_f) tensors
        activation: 'softmax' | 'silu' | 'silu2'

    Returns:
        (B*T, N_f, D_f) attended output
    """
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
