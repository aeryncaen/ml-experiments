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
    actual_N: tl.constexpr,  # actual feature count (e.g. 48)
    BLOCK_N: tl.constexpr,   # padded to next power of 2 (e.g. 64)
    D_F: tl.constexpr,
):
    """One program per batch element. Computes fused QK^T -> softmax -> @V."""
    bid = tl.program_id(0)

    q_base = Q_ptr + bid * stride_qb
    k_base = K_ptr + bid * stride_kb
    v_base = V_ptr + bid * stride_vb
    o_base = O_ptr + bid * stride_ob

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_F)
    mask_n = offs_n < actual_N

    # Load Q, K with masking for padded rows (BLOCK_N, D_F)
    q = tl.load(q_base + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                mask=mask_n[:, None], other=0.0)
    k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None], other=0.0)

    # Scores = Q @ K^T * scale -> (BLOCK_N, BLOCK_N)
    scores = tl.dot(q, tl.trans(k)) * scale

    # Mask out padded positions (rows and cols) with -inf for softmax
    mask_2d = mask_n[:, None] & mask_n[None, :]
    scores = tl.where(mask_2d, scores, float('-inf'))

    # Softmax (row-wise)
    row_max = tl.max(scores, axis=1)
    scores = scores - row_max[:, None]
    weights = tl.exp(scores)
    # Zero out padded rows so they don't contribute
    weights = tl.where(mask_2d, weights, 0.0)
    row_sum = tl.sum(weights, axis=1)
    weights = weights / tl.where(row_sum > 0, row_sum, 1.0)[:, None]

    # Load V (BLOCK_N, D_F)
    v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None], other=0.0)

    # Output = weights @ V -> (BLOCK_N, D_F)
    out = tl.dot(weights.to(v.dtype), v)

    # Store only valid rows
    tl.store(o_base + offs_n[:, None] * stride_on + offs_d[None, :] * stride_od,
             out, mask=mask_n[:, None])


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
    actual_N: tl.constexpr,
    BLOCK_N: tl.constexpr,
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

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_F)
    mask_n = offs_n < actual_N

    # Reload Q, K, V, dO with masking
    q = tl.load(q_base + offs_n[:, None] * stride_qn + offs_d[None, :] * stride_qd,
                mask=mask_n[:, None], other=0.0).to(tl.float32)
    k = tl.load(k_base + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd,
                mask=mask_n[:, None], other=0.0).to(tl.float32)
    v = tl.load(v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd,
                mask=mask_n[:, None], other=0.0).to(tl.float32)
    do = tl.load(do_base + offs_n[:, None] * stride_don + offs_d[None, :] * stride_dod,
                 mask=mask_n[:, None], other=0.0).to(tl.float32)

    # Recompute softmax weights
    mask_2d = mask_n[:, None] & mask_n[None, :]
    scores = tl.dot(q, tl.trans(k)) * scale
    scores = tl.where(mask_2d, scores, float('-inf'))
    row_max = tl.max(scores, axis=1)
    scores = scores - row_max[:, None]
    weights = tl.exp(scores)
    weights = tl.where(mask_2d, weights, 0.0)
    row_sum = tl.sum(weights, axis=1)
    weights = weights / tl.where(row_sum > 0, row_sum, 1.0)[:, None]

    # dV = weights^T @ dO  -> (BLOCK_N, D_F)
    dv = tl.dot(tl.trans(weights), do)

    # dweights = dO @ V^T  -> (BLOCK_N, BLOCK_N)
    dweights = tl.dot(do, tl.trans(v))

    # dsoftmax: d_scores = weights * (dweights - rowsum(dweights * weights))
    row_dot = tl.sum(dweights * weights, axis=1)
    d_scores = weights * (dweights - row_dot[:, None]) * scale

    # dQ = d_scores @ K  -> (BLOCK_N, D_F)
    dq = tl.dot(d_scores, k)

    # dK = d_scores^T @ Q  -> (BLOCK_N, D_F)
    dk = tl.dot(tl.trans(d_scores), q)

    # Store gradients, masked to valid rows
    in_dtype = Q_ptr.dtype.element_ty
    tl.store(dq_base + offs_n[:, None] * stride_dqn + offs_d[None, :] * stride_dqd,
             dq.to(in_dtype), mask=mask_n[:, None])
    tl.store(dk_base + offs_n[:, None] * stride_dkn + offs_d[None, :] * stride_dkd,
             dk.to(in_dtype), mask=mask_n[:, None])
    tl.store(dv_base + offs_n[:, None] * stride_dvn + offs_d[None, :] * stride_dvd,
             dv.to(in_dtype), mask=mask_n[:, None])


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
        # Pad N_F to next power of 2 for triton
        BLOCK_N = triton.next_power_of_2(N_F)
        out = torch.empty_like(v)

        grid = (B,)
        _fused_feat_attn_fwd_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            scale,
            actual_N=N_F,
            BLOCK_N=BLOCK_N,
            D_F=D_F,
            num_warps=1,
        )

        ctx.save_for_backward(q, k, v)
        ctx.scale = scale
        ctx.BLOCK_N = BLOCK_N
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
            actual_N=N_F,
            BLOCK_N=ctx.BLOCK_N,
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
