"""Triton-fused silu2 attention: flash-style tiled forward and backward.

silu2 attention: O = (silu(QK^T * scale)^2 * causal_mask) @ V

Unlike softmax attention, silu2 is unnormalized — no log-sum-exp tracking
needed. Tiles accumulate directly into the output.

Forward: tile over K/V blocks, accumulate silu(S)^2 @ V.
Backward: recompute S from saved Q, K per tile (no T^2 storage).

Layout: Q, K, V are (B, H, T, D) — standard SDPA layout, contiguous on D.
"""

import math

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Forward kernel
# ---------------------------------------------------------------------------

@triton.heuristics({
    "EVEN_M": lambda args: args["seqlen"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["seqlen"] % args["BLOCK_N"] == 0,
    "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
})
@triton.jit
def _silu2_attn_fwd_kernel(
    Q, K, V, Out,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    nheads, seqlen, headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Program IDs
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    # Offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Pointers
    q_ptrs = Q + off_b * stride_qb + off_h * stride_qh + (offs_m[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + off_b * stride_kb + off_h * stride_kh + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + off_b * stride_vb + off_h * stride_vh + (offs_n[:, None] * stride_vn + offs_d[None, :])

    # Load Q block — stays in SRAM
    if EVEN_M:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen) & (offs_d[None, :] < headdim), other=0.0)

    # Accumulator
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32)

    # Causal: only iterate up to min((start_m + 1) * BLOCK_M, seqlen)
    end_n = tl.minimum((start_m + 1) * BLOCK_M, seqlen)

    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)

        # Load K block
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen, other=0.0)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn,
                            mask=((start_n + offs_n)[:, None] < seqlen) & (offs_d[None, :] < headdim), other=0.0)

        # QK^T
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k), input_precision="ieee")
        qk = qk * softmax_scale

        # Causal mask
        qk = tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], qk, 0.0)

        # silu(qk)^2
        # silu(x) = x * sigmoid(x)
        sig = tl.sigmoid(qk)
        silu_qk = qk * sig
        weights = silu_qk * silu_qk

        # Load V block
        if EVEN_N & EVEN_M:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=(start_n + offs_n)[:, None] < seqlen, other=0.0)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn,
                            mask=((start_n + offs_n)[:, None] < seqlen) & (offs_d[None, :] < headdim), other=0.0)

        # Accumulate
        acc_o += tl.dot(weights.to(v.dtype), v, input_precision="ieee")

    # Store output
    out_ptrs = Out + off_b * stride_ob + off_h * stride_oh + (offs_m[:, None] * stride_om + offs_d[None, :])
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen)
        else:
            tl.store(out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen) & (offs_d[None, :] < headdim))


# ---------------------------------------------------------------------------
# Backward kernel
# ---------------------------------------------------------------------------

# Backward iterates over K/V column blocks (like flash-attn bwd).
# For each K/V block, we loop over Q/dO row blocks to accumulate dK, dV,
# and scatter-add into dQ.
#
# Recomputes S = QK^T * scale per tile (no T^2 storage).
#
# Derivatives:
#   A = silu(S)^2 * causal_mask       (attention weights)
#   dV = A^T @ dO
#   dA = dO @ V^T
#   dS = dA * 2 * silu(S) * dsilu(S) * causal_mask
#       where dsilu(x) = sigmoid(x) * (1 + x - x*sigmoid(x))
#                      = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
#   dQ = dS @ K * scale
#   dK = dS^T @ Q * scale

@triton.heuristics({
    "EVEN_M": lambda args: args["seqlen"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["seqlen"] % args["BLOCK_N"] == 0,
    "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
})
@triton.jit
def _silu2_attn_bwd_kernel(
    Q, K, V, DO,
    DQ, DK, DV,
    softmax_scale,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_dob, stride_doh, stride_dom,
    stride_dqb, stride_dqh, stride_dqm,
    stride_dkb, stride_dkh, stride_dkn,
    stride_dvb, stride_dvh, stride_dvn,
    nheads, seqlen, headdim,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # This kernel is launched with grid = (cdiv(seqlen, BLOCK_N), B*H)
    # Each program handles one K/V column block.
    start_n = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)

    # Base pointers for this batch/head
    q_base = Q + off_b * stride_qb + off_h * stride_qh
    k_base = K + off_b * stride_kb + off_h * stride_kh
    v_base = V + off_b * stride_vb + off_h * stride_vh
    do_base = DO + off_b * stride_dob + off_h * stride_doh
    dq_base = DQ + off_b * stride_dqb + off_h * stride_dqh
    dk_base = DK + off_b * stride_dkb + off_h * stride_dkh
    dv_base = DV + off_b * stride_dvb + off_h * stride_dvh

    # Load K, V for this column block — stays in SRAM
    k_ptrs = k_base + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = v_base + (offs_n[:, None] * stride_vn + offs_d[None, :])

    if EVEN_N:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen, other=0.0)
        else:
            k = tl.load(k_ptrs, mask=(offs_n[:, None] < seqlen) & (offs_d[None, :] < headdim), other=0.0)
            v = tl.load(v_ptrs, mask=(offs_n[:, None] < seqlen) & (offs_d[None, :] < headdim), other=0.0)

    # Accumulators for dK, dV
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)

    # Causal: only Q rows >= start_n can attend to this K block
    # Start from the first BLOCK_M-aligned row >= start_n * BLOCK_N
    begin_m = (start_n * BLOCK_N // BLOCK_M) * BLOCK_M
    num_block_m = tl.cdiv(seqlen, BLOCK_M)

    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m

        # Load Q block
        q_ptrs = q_base + (offs_m_curr[:, None] * stride_qm + offs_d[None, :])
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen, other=0.0)
            else:
                q = tl.load(q_ptrs, mask=(offs_m_curr[:, None] < seqlen) & (offs_d[None, :] < headdim), other=0.0)

        # Recompute S = QK^T * scale
        qk = tl.dot(q, tl.trans(k), input_precision="ieee") * softmax_scale

        # Causal mask
        causal = offs_m_curr[:, None] >= offs_n[None, :]
        qk = tl.where(causal, qk, 0.0)

        # Recompute silu(S), sigmoid(S)
        sig = tl.sigmoid(qk)
        silu_qk = qk * sig  # silu(S)

        # A = silu(S)^2 (already causally masked since qk is zeroed)
        A = silu_qk * silu_qk

        # Load dO block
        do_ptrs = do_base + (offs_m_curr[:, None] * stride_dom + offs_d[None, :])
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            if EVEN_HEADDIM:
                do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen, other=0.0)
            else:
                do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen) & (offs_d[None, :] < headdim), other=0.0)

        # dV += A^T @ dO
        dv += tl.dot(tl.trans(A.to(do.dtype)), do, input_precision="ieee")

        # dA = dO @ V^T   (BLOCK_M x BLOCK_N)
        dA = tl.dot(do, tl.trans(v), input_precision="ieee")

        # dS = dA * 2 * silu(S) * dsilu(S) * causal_mask
        # dsilu(x) = sig(x) + x * sig(x) * (1 - sig(x))
        #          = sig * (1 + x * (1 - sig))
        #          = sig * (1 + x - x * sig)
        dsilu = sig * (1.0 + qk - qk * sig)
        dS = dA * 2.0 * silu_qk * dsilu
        dS = tl.where(causal, dS, 0.0)
        dS = dS * softmax_scale

        # dK += dS^T @ Q
        dk += tl.dot(tl.trans(dS.to(q.dtype)), q, input_precision="ieee")

        # dQ += dS @ K  (scatter-add into dQ)
        dq_contrib = tl.dot(dS.to(k.dtype), k, input_precision="ieee")
        dq_ptrs = dq_base + (offs_m_curr[:, None] * stride_dqm + offs_d[None, :])
        if EVEN_M & EVEN_HEADDIM:
            tl.atomic_add(dq_ptrs, dq_contrib)
        else:
            if EVEN_HEADDIM:
                tl.atomic_add(dq_ptrs, dq_contrib, mask=offs_m_curr[:, None] < seqlen)
            else:
                tl.atomic_add(dq_ptrs, dq_contrib,
                              mask=(offs_m_curr[:, None] < seqlen) & (offs_d[None, :] < headdim))

    # Store dK, dV
    dk_ptrs = dk_base + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    dv_ptrs = dv_base + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    if EVEN_N:
        if EVEN_HEADDIM:
            tl.store(dk_ptrs, dk)
            tl.store(dv_ptrs, dv)
        else:
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen)
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen)
        else:
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen) & (offs_d[None, :] < headdim))
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen) & (offs_d[None, :] < headdim))


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _silu2_attn_forward(q, k, v, softmax_scale=None):
    """Triton silu2 attention forward.

    Args:
        q, k, v: (B, H, T, D) — contiguous on last dim.

    Returns:
        o: (B, H, T, D) — attention output.
    """
    B, H, T, D = q.shape
    assert k.shape == (B, H, T, D) and v.shape == (B, H, T, D)
    assert q.is_cuda and k.is_cuda and v.is_cuda
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, \
        "Last dim must be contiguous"

    softmax_scale = softmax_scale or (1.0 / math.sqrt(D))
    o = torch.empty_like(q)

    BLOCK_HEADDIM = max(triton.next_power_of_2(D), 16)
    BLOCK = 64
    num_warps = 4 if D <= 64 else 8
    grid = (triton.cdiv(T, BLOCK), B * H)

    _silu2_attn_fwd_kernel[grid](
        q, k, v, o,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        H, T, D,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return o


def _silu2_attn_backward(do, q, k, v, softmax_scale=None):
    """Triton silu2 attention backward.

    Args:
        do: (B, H, T, D) — gradient of output.
        q, k, v: (B, H, T, D) — saved from forward.

    Returns:
        dq, dk, dv: (B, H, T, D) — gradients.
    """
    B, H, T, D = q.shape
    assert do.shape == (B, H, T, D)

    softmax_scale = softmax_scale or (1.0 / math.sqrt(D))

    # Ensure contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()

    # dQ needs atomic adds, so init to zero
    dq = torch.zeros_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    BLOCK_HEADDIM = max(triton.next_power_of_2(D), 16)
    BLOCK = 64
    num_warps = 4 if D <= 64 else 8
    grid = (triton.cdiv(T, BLOCK), B * H)

    _silu2_attn_bwd_kernel[grid](
        q, k, v, do,
        dq, dk, dv,
        softmax_scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        do.stride(0), do.stride(1), do.stride(2),
        dq.stride(0), dq.stride(1), dq.stride(2),
        dk.stride(0), dk.stride(1), dk.stride(2),
        dv.stride(0), dv.stride(1), dv.stride(2),
        H, T, D,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
    return dq, dk, dv


# ---------------------------------------------------------------------------
# Custom op registration for torch.compile compatibility
# ---------------------------------------------------------------------------

# Both forward and backward are registered as custom ops with fake impls,
# so torch.compile/inductor never tries to trace into the Triton kernel
# launches (which access .data_ptr() and would crash on FakeTensors).

@torch.library.custom_op("ulb::silu2_attn_fwd", mutates_args=())
def _silu2_attn_fwd_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
    D = q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    return _silu2_attn_forward(q, k, v, scale)


@_silu2_attn_fwd_op.register_fake
def _silu2_attn_fwd_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(q)


@torch.library.custom_op("ulb::silu2_attn_bwd", mutates_args=())
def _silu2_attn_bwd_op(
    grad_output: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    D = q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    return _silu2_attn_backward(grad_output, q, k, v, scale)


@_silu2_attn_bwd_op.register_fake
def _silu2_attn_bwd_fake(
    grad_output: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)


def _silu2_attn_setup_context(ctx, inputs, output):
    q, k, v = inputs
    ctx.save_for_backward(q, k, v)


def _silu2_attn_backward_impl(ctx, grad_output):
    q, k, v = ctx.saved_tensors
    return _silu2_attn_bwd_op(grad_output, q, k, v)


_silu2_attn_fwd_op.register_autograd(
    _silu2_attn_backward_impl,
    setup_context=_silu2_attn_setup_context,
)


def silu2_attention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for silu2_attention using Triton kernels.

    Registered as torch custom ops so torch.compile doesn't graph-break.

    Args:
        q, k, v: (B, H, T, D) — standard SDPA layout, CUDA tensors.

    Returns:
        (B, H, T, D) attention output.
    """
    return _silu2_attn_fwd_op(q, k, v)
