"""Triton kernels for S6 — fused projections, activations, scan, and readout."""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    # ========== Reusable kernels (from triton_adaptive_conv.py) ==========

    @triton.jit
    def linear_fwd_kernel(
        x_ptr, w_ptr, b_ptr, out_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k in range(0, K, BLOCK_K):
            k_mask = (k + offs_k) < K
            x_tile = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
            w_tile = tl.load(w_ptrs, mask=(offs_n[:, None] < N) & k_mask[None, :], other=0.0)
            acc = tl.dot(x_tile, tl.trans(w_tile), acc)
            x_ptrs += BLOCK_K * stride_xk
            w_ptrs += BLOCK_K * stride_wk
        if HAS_BIAS:
            bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
            acc = acc + bias[None, :]
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc, mask=out_mask)

    @triton.jit
    def linear_bwd_dx_kernel(
        d_out_ptr, w_ptr, d_x_ptr,
        M, N, K,
        stride_dom, stride_don,
        stride_wn, stride_wk,
        stride_dxm, stride_dxk,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = tl.arange(0, BLOCK_N)
        do_ptrs = d_out_ptr + offs_m[:, None] * stride_dom + offs_n[None, :] * stride_don
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
        for n in range(0, N, BLOCK_N):
            n_mask = (n + offs_n) < N
            do_tile = tl.load(do_ptrs, mask=(offs_m[:, None] < M) & n_mask[None, :], other=0.0)
            w_tile = tl.load(w_ptrs, mask=n_mask[:, None] & (offs_k[None, :] < K), other=0.0)
            acc = tl.dot(do_tile, w_tile, acc)
            do_ptrs += BLOCK_N * stride_don
            w_ptrs += BLOCK_N * stride_wn
        dx_ptrs = d_x_ptr + offs_m[:, None] * stride_dxm + offs_k[None, :] * stride_dxk
        dx_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        tl.store(dx_ptrs, acc, mask=dx_mask)

    @triton.jit
    def linear_bwd_dw_kernel(
        d_out_ptr, x_ptr, d_w_ptr,
        M, N, K,
        stride_dom, stride_don,
        stride_xm, stride_xk,
        stride_dwn, stride_dwk,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_k = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_m = tl.arange(0, BLOCK_M)
        do_ptrs = d_out_ptr + offs_m[:, None] * stride_dom + offs_n[None, :] * stride_don
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
        for m in range(0, M, BLOCK_M):
            m_mask = (m + offs_m) < M
            do_tile = tl.load(do_ptrs, mask=m_mask[:, None] & (offs_n[None, :] < N), other=0.0)
            x_tile = tl.load(x_ptrs, mask=m_mask[:, None] & (offs_k[None, :] < K), other=0.0)
            acc = tl.dot(tl.trans(do_tile), x_tile, acc)
            do_ptrs += BLOCK_M * stride_dom
            x_ptrs += BLOCK_M * stride_xm
        dw_ptrs = d_w_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk
        dw_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        tl.store(dw_ptrs, acc, mask=dw_mask)

    @triton.jit
    def rmsnorm_fwd_kernel(
        x_ptr, gamma_ptr, out_ptr,
        M, N,
        stride_xm, stride_xn,
        stride_om, stride_on,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        x_ptrs = x_ptr + pid_m * stride_xm + offs_n * stride_xn
        x = tl.load(x_ptrs, mask=n_mask, other=0.0)
        variance = tl.sum(x * x) / N
        x_normed = x * tl.rsqrt(variance + eps)
        gamma = tl.load(gamma_ptr + offs_n, mask=n_mask, other=1.0)
        out = gamma * x_normed
        out_ptrs = out_ptr + pid_m * stride_om + offs_n * stride_on
        tl.store(out_ptrs, out, mask=n_mask)

    @triton.jit
    def silu_fwd_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        sig = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + offs, x * sig, mask=mask)

    @triton.jit
    def silu_bwd_kernel(d_out_ptr, x_ptr, d_x_ptr, N, BLOCK: tl.constexpr):
        """d_x = d_out * silu'(x) = d_out * sigmoid(x) * (1 + x*(1-sigmoid(x)))"""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        d_out = tl.load(d_out_ptr + offs, mask=mask)
        x = tl.load(x_ptr + offs, mask=mask)
        sig = 1.0 / (1.0 + tl.exp(-x))
        d_x = d_out * sig * (1.0 + x * (1.0 - sig))
        tl.store(d_x_ptr + offs, d_x, mask=mask)

    @triton.jit
    def rmsnorm_bwd_kernel(
        d_out_ptr, x_ptr, gamma_ptr,
        d_x_ptr, d_gamma_ptr,
        M, N,
        stride_dom, stride_don,
        stride_xm, stride_xn,
        stride_dxm, stride_dxn,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        """Backward for RMSNorm. One program per row (M rows).
        d_gamma is accumulated via atomic_add since each row contributes.
        """
        pid_m = tl.program_id(0)
        offs_n = tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        x = tl.load(x_ptr + pid_m * stride_xm + offs_n * stride_xn, mask=n_mask, other=0.0)
        d_out = tl.load(d_out_ptr + pid_m * stride_dom + offs_n * stride_don, mask=n_mask, other=0.0)
        gamma = tl.load(gamma_ptr + offs_n, mask=n_mask, other=1.0)

        # Forward recompute
        variance = tl.sum(x * x) / N
        rrms = tl.rsqrt(variance + eps)
        x_hat = x * rrms  # normalized

        # d_gamma (per-element, atomically accumulated across rows)
        d_gamma_local = d_out * x_hat
        tl.atomic_add(d_gamma_ptr + offs_n, d_gamma_local, mask=n_mask)

        # d_x_hat = d_out * gamma
        d_x_hat = d_out * gamma

        # d_x = rrms * (d_x_hat - x_hat * mean(d_x_hat * x_hat))
        inner = tl.sum(d_x_hat * x_hat) / N
        d_x = rrms * (d_x_hat - x_hat * inner)

        tl.store(d_x_ptr + pid_m * stride_dxm + offs_n * stride_dxn, d_x, mask=n_mask)

    # ========== S6-specific kernels ==========

    @triton.jit
    def fused_dt_lam_kernel(
        dt_raw_ptr, lam_raw_ptr, log_dt_bias_ptr,
        dt_out_ptr, lam_out_ptr,
        N,
        BLOCK: tl.constexpr,
    ):
        """Fused: dt = softplus(silu(dt_raw) + log_dt_bias), lam = sigmoid(lam_raw)"""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        # dt: softplus(silu(x) + bias), bias pre-broadcast by caller to (B*L*P)
        dt_raw = tl.load(dt_raw_ptr + offs, mask=mask)
        sig_dt = 1.0 / (1.0 + tl.exp(-dt_raw))
        silu_dt = dt_raw * sig_dt
        pre = silu_dt + tl.load(log_dt_bias_ptr + offs, mask=mask)
        # softplus: log(1 + exp(x)), numerically stable
        dt = tl.where(pre > 20.0, pre, tl.log(1.0 + tl.exp(pre)))
        tl.store(dt_out_ptr + offs, dt, mask=mask)

        # lam: sigmoid
        lam_raw = tl.load(lam_raw_ptr + offs, mask=mask)
        lam = 1.0 / (1.0 + tl.exp(-lam_raw))
        tl.store(lam_out_ptr + offs, lam, mask=mask)

    @triton.jit
    def fused_dt_lam_bwd_kernel(
        d_dt_ptr, d_lam_ptr,
        dt_raw_ptr, lam_raw_ptr, log_dt_bias_ptr,
        d_dt_raw_ptr, d_lam_raw_ptr, d_log_dt_bias_ptr,
        N,
        BLOCK: tl.constexpr,
    ):
        """Backward for fused dt/lam activations."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        # Forward recompute for dt
        dt_raw = tl.load(dt_raw_ptr + offs, mask=mask)
        sig_dt = 1.0 / (1.0 + tl.exp(-dt_raw))
        silu_dt = dt_raw * sig_dt
        bias = tl.load(log_dt_bias_ptr + offs, mask=mask)
        pre = silu_dt + bias

        # softplus backward: d_pre = d_dt * sigmoid(pre)
        d_dt = tl.load(d_dt_ptr + offs, mask=mask)
        sig_pre = 1.0 / (1.0 + tl.exp(-pre))
        d_pre = d_dt * sig_pre

        # d_log_dt_bias = d_pre (additive)
        tl.store(d_log_dt_bias_ptr + offs, d_pre, mask=mask)

        # silu backward: d_dt_raw = d_pre * (sig_dt + dt_raw * sig_dt * (1 - sig_dt))
        d_silu = d_pre * (sig_dt + dt_raw * sig_dt * (1.0 - sig_dt))
        tl.store(d_dt_raw_ptr + offs, d_silu, mask=mask)

        # lam backward: d_lam_raw = d_lam * lam * (1 - lam)
        lam_raw = tl.load(lam_raw_ptr + offs, mask=mask)
        lam = 1.0 / (1.0 + tl.exp(-lam_raw))
        d_lam = tl.load(d_lam_ptr + offs, mask=mask)
        d_lam_raw = d_lam * lam * (1.0 - lam)
        tl.store(d_lam_raw_ptr + offs, d_lam_raw, mask=mask)

    @triton.jit
    def rope_fwd_kernel(
        x_ptr, angles_ptr, out_ptr,
        N, P,  # N = total elements in batch dims, P = last dim (even)
        BLOCK: tl.constexpr,
    ):
        """RoPE: rotate pairs of elements by angle. x: (..., P), angles: (..., P//2)."""
        pid = tl.program_id(0)
        # Each program handles one row of P elements
        row = pid
        if row >= N:
            return
        half_P = P // 2
        offs = tl.arange(0, BLOCK)
        mask = offs < half_P

        x_base = row * P
        a_base = row * half_P

        x1 = tl.load(x_ptr + x_base + offs * 2, mask=mask)
        x2 = tl.load(x_ptr + x_base + offs * 2 + 1, mask=mask)
        angle = tl.load(angles_ptr + a_base + offs, mask=mask)

        cos_a = tl.cos(angle)
        sin_a = tl.sin(angle)

        out1 = x1 * cos_a - x2 * sin_a
        out2 = x1 * sin_a + x2 * cos_a

        tl.store(out_ptr + x_base + offs * 2, out1, mask=mask)
        tl.store(out_ptr + x_base + offs * 2 + 1, out2, mask=mask)

    @triton.jit
    def rope_bwd_kernel(
        d_out_ptr, angles_ptr, d_x_ptr, d_angles_ptr,
        x_ptr,  # need original x for d_angles
        N, P,
        BLOCK: tl.constexpr,
    ):
        """Backward for RoPE."""
        pid = tl.program_id(0)
        row = pid
        if row >= N:
            return
        half_P = P // 2
        offs = tl.arange(0, BLOCK)
        mask = offs < half_P

        x_base = row * P
        a_base = row * half_P

        d_o1 = tl.load(d_out_ptr + x_base + offs * 2, mask=mask)
        d_o2 = tl.load(d_out_ptr + x_base + offs * 2 + 1, mask=mask)
        angle = tl.load(angles_ptr + a_base + offs, mask=mask)
        x1 = tl.load(x_ptr + x_base + offs * 2, mask=mask)
        x2 = tl.load(x_ptr + x_base + offs * 2 + 1, mask=mask)

        cos_a = tl.cos(angle)
        sin_a = tl.sin(angle)

        # d_x1 = d_o1 * cos + d_o2 * sin
        # d_x2 = -d_o1 * sin + d_o2 * cos
        d_x1 = d_o1 * cos_a + d_o2 * sin_a
        d_x2 = -d_o1 * sin_a + d_o2 * cos_a
        tl.store(d_x_ptr + x_base + offs * 2, d_x1, mask=mask)
        tl.store(d_x_ptr + x_base + offs * 2 + 1, d_x2, mask=mask)

        # d_angle = d_o1 * (-x1*sin - x2*cos) + d_o2 * (x1*cos - x2*sin)
        d_a = d_o1 * (-x1 * sin_a - x2 * cos_a) + d_o2 * (x1 * cos_a - x2 * sin_a)
        tl.store(d_angles_ptr + a_base + offs, d_a, mask=mask)

    @triton.jit
    def complex_discretize_fwd_kernel(
        dt_ptr, lam_ptr, Bu_ptr, Bu_prev_ptr,
        A_real_ptr, A_imag_ptr,
        alpha_re_ptr, alpha_im_ptr, inject_re_ptr, inject_im_ptr,
        N, P,
        BLOCK: tl.constexpr,
    ):
        """Compute complex alpha = exp(dt * A) and trapezoidal inject.
        All inputs real except A which is complex (stored as separate real/imag).
        Output alpha and inject are complex (stored as separate real/imag).
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        dt = tl.load(dt_ptr + offs, mask=mask)
        lam = tl.load(lam_ptr + offs, mask=mask)
        Bu = tl.load(Bu_ptr + offs, mask=mask)
        Bu_prev = tl.load(Bu_prev_ptr + offs, mask=mask)

        # A is (P,), index with offs % P
        p_idx = offs % P
        a_re = tl.load(A_real_ptr + p_idx, mask=mask)
        a_im = tl.load(A_imag_ptr + p_idx, mask=mask)

        # alpha = exp(dt * A) where A = a_re + j*a_im
        # exp(dt*(a_re + j*a_im)) = exp(dt*a_re) * (cos(dt*a_im) + j*sin(dt*a_im))
        exp_re = tl.exp(dt * a_re)
        angle = dt * a_im
        alpha_re = exp_re * tl.cos(angle)
        alpha_im = exp_re * tl.sin(angle)

        tl.store(alpha_re_ptr + offs, alpha_re, mask=mask)
        tl.store(alpha_im_ptr + offs, alpha_im, mask=mask)

        # inject = lam * dt * Bu + (1-lam) * dt * alpha * Bu_prev
        # alpha is complex, Bu/Bu_prev are real, so:
        # inject_re = lam*dt*Bu + (1-lam)*dt*(alpha_re*Bu_prev)
        # inject_im = (1-lam)*dt*(alpha_im*Bu_prev)
        fwd_term = lam * dt * Bu
        bwd_re = (1.0 - lam) * dt * alpha_re * Bu_prev
        bwd_im = (1.0 - lam) * dt * alpha_im * Bu_prev

        tl.store(inject_re_ptr + offs, fwd_term + bwd_re, mask=mask)
        tl.store(inject_im_ptr + offs, bwd_im, mask=mask)

    @triton.jit
    def complex_discretize_bwd_kernel(
        d_inject_re_ptr, d_inject_im_ptr, d_alpha_re_ptr, d_alpha_im_ptr,
        dt_ptr, lam_ptr, Bu_ptr, Bu_prev_ptr,
        A_real_ptr, A_imag_ptr,
        alpha_re_ptr, alpha_im_ptr,
        d_dt_ptr, d_lam_ptr, d_Bu_ptr, d_Bu_prev_ptr,
        N, P,
        BLOCK: tl.constexpr,
    ):
        """Backward for complex discretization."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N

        d_inj_re = tl.load(d_inject_re_ptr + offs, mask=mask)
        d_inj_im = tl.load(d_inject_im_ptr + offs, mask=mask)
        d_alph_re = tl.load(d_alpha_re_ptr + offs, mask=mask)
        d_alph_im = tl.load(d_alpha_im_ptr + offs, mask=mask)

        dt = tl.load(dt_ptr + offs, mask=mask)
        lam = tl.load(lam_ptr + offs, mask=mask)
        Bu = tl.load(Bu_ptr + offs, mask=mask)
        Bu_prev = tl.load(Bu_prev_ptr + offs, mask=mask)

        p_idx = offs % P
        a_re = tl.load(A_real_ptr + p_idx, mask=mask)
        a_im = tl.load(A_imag_ptr + p_idx, mask=mask)
        alpha_re = tl.load(alpha_re_ptr + offs, mask=mask)
        alpha_im = tl.load(alpha_im_ptr + offs, mask=mask)

        one_minus_lam = 1.0 - lam

        # d_Bu = d_inj_re * lam * dt
        d_Bu = d_inj_re * lam * dt
        tl.store(d_Bu_ptr + offs, d_Bu, mask=mask)

        # d_Bu_prev = d_inj_re * (1-lam)*dt*alpha_re + d_inj_im * (1-lam)*dt*alpha_im
        d_Bu_prev = (d_inj_re * alpha_re + d_inj_im * alpha_im) * one_minus_lam * dt
        tl.store(d_Bu_prev_ptr + offs, d_Bu_prev, mask=mask)

        # d_lam: d_inj_re * (dt*Bu - dt*alpha_re*Bu_prev) + d_inj_im * (-dt*alpha_im*Bu_prev)
        d_lam = d_inj_re * dt * (Bu - alpha_re * Bu_prev) + d_inj_im * (-dt * alpha_im * Bu_prev)
        tl.store(d_lam_ptr + offs, d_lam, mask=mask)

        # d_alpha from inject: d_inj_re * (1-lam)*dt*Bu_prev (re part), d_inj_im * (1-lam)*dt*Bu_prev (im part)
        d_alph_re = d_alph_re + d_inj_re * one_minus_lam * dt * Bu_prev
        d_alph_im = d_alph_im + d_inj_im * one_minus_lam * dt * Bu_prev

        # d_dt from inject: d_inj_re * (lam*Bu + (1-lam)*alpha_re*Bu_prev) + d_inj_im * (1-lam)*alpha_im*Bu_prev
        d_dt_from_inject = (d_inj_re * (lam * Bu + one_minus_lam * alpha_re * Bu_prev)
                           + d_inj_im * one_minus_lam * alpha_im * Bu_prev)

        # d_dt from alpha: alpha = exp(dt*A), d_alpha/d_dt = A * alpha
        # d_dt += d_alph_re * (a_re*alpha_re - a_im*alpha_im) + d_alph_im * (a_re*alpha_im + a_im*alpha_re)
        d_dt_from_alpha = (d_alph_re * (a_re * alpha_re - a_im * alpha_im)
                          + d_alph_im * (a_re * alpha_im + a_im * alpha_re))

        tl.store(d_dt_ptr + offs, d_dt_from_inject + d_dt_from_alpha, mask=mask)

    @triton.jit
    def chunked_scan_fwd_kernel(
        alpha_re_ptr, alpha_im_ptr, inject_re_ptr, inject_im_ptr,
        state_re_ptr, state_im_ptr,
        out_re_ptr, out_im_ptr,
        new_state_re_ptr, new_state_im_ptr,
        B_batch, K, P,
        stride_al, stride_ap,
        BLOCK_P: tl.constexpr,
    ):
        """Intra-chunk scan via decay matrix matmul for one (batch, p_block).
        K = chunk size, P = state dim.
        Computes h[i] = (prod_{t=j+1}^{i} alpha[t]) * inject[j] summed over j<=i, plus carried state.
        """
        pid_b = tl.program_id(0)
        pid_p = tl.program_id(1)

        p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        p_mask = p_offs < P

        # Load incoming state for this batch element
        s_re = tl.load(state_re_ptr + pid_b * P + p_offs, mask=p_mask, other=0.0)
        s_im = tl.load(state_im_ptr + pid_b * P + p_offs, mask=p_mask, other=0.0)

        # Sequential scan within chunk (K is small, typically 32-64)
        for t in range(K):
            a_off = pid_b * K * P + t * P + p_offs
            a_re = tl.load(alpha_re_ptr + a_off, mask=p_mask)
            a_im = tl.load(alpha_im_ptr + a_off, mask=p_mask)
            inj_re = tl.load(inject_re_ptr + a_off, mask=p_mask)
            inj_im = tl.load(inject_im_ptr + a_off, mask=p_mask)

            # complex multiply: s = alpha * s + inject
            new_re = a_re * s_re - a_im * s_im + inj_re
            new_im = a_re * s_im + a_im * s_re + inj_im
            s_re = new_re
            s_im = new_im

            tl.store(out_re_ptr + a_off, s_re, mask=p_mask)
            tl.store(out_im_ptr + a_off, s_im, mask=p_mask)

        # Store final state
        tl.store(new_state_re_ptr + pid_b * P + p_offs, s_re, mask=p_mask)
        tl.store(new_state_im_ptr + pid_b * P + p_offs, s_im, mask=p_mask)

    @triton.jit
    def chunked_scan_bwd_kernel(
        alpha_re_ptr, alpha_im_ptr,
        d_out_re_ptr, d_out_im_ptr,
        d_alpha_re_ptr, d_alpha_im_ptr,
        d_inject_re_ptr, d_inject_im_ptr,
        d_state_re_ptr, d_state_im_ptr,
        h_re_ptr, h_im_ptr,  # saved forward states (B, K, P) — h[t-1] needed
        state_re_ptr, state_im_ptr,  # incoming state for this chunk
        B_batch, K, P,
        BLOCK_P: tl.constexpr,
    ):
        """Backward scan: reverse sequential through chunk.
        d_h[t] comes from d_out[t] + carried adjoint from d_h[t+1].
        d_alpha[t] = d_h[t] * conj(h[t-1])  (complex product rule)
        d_inject[t] = d_h[t]
        """
        pid_b = tl.program_id(0)
        pid_p = tl.program_id(1)

        p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        p_mask = p_offs < P

        # Adjoint state (carried backward)
        adj_re = tl.zeros((BLOCK_P,), dtype=tl.float32)
        adj_im = tl.zeros((BLOCK_P,), dtype=tl.float32)

        for t_rev in range(K):
            t = K - 1 - t_rev
            a_off = pid_b * K * P + t * P + p_offs

            # d_h[t] = d_out[t] + alpha[t+1]^* . adj  (but we process in reverse)
            # Actually: h[t] = alpha[t]*h[t-1] + inject[t]
            # d_h[t-1] += alpha[t]^* . d_h[t]  (adjoint through alpha*h[t-1])
            # d_alpha[t] = d_h[t] . h[t-1]^*
            # d_inject[t] = d_h[t]

            d_out_re = tl.load(d_out_re_ptr + a_off, mask=p_mask, other=0.0)
            d_out_im = tl.load(d_out_im_ptr + a_off, mask=p_mask, other=0.0)

            # Total gradient at this position
            d_h_re = d_out_re + adj_re
            d_h_im = d_out_im + adj_im

            # d_inject = d_h (direct)
            tl.store(d_inject_re_ptr + a_off, d_h_re, mask=p_mask)
            tl.store(d_inject_im_ptr + a_off, d_h_im, mask=p_mask)

            # Load alpha[t]
            a_re = tl.load(alpha_re_ptr + a_off, mask=p_mask)
            a_im = tl.load(alpha_im_ptr + a_off, mask=p_mask)

            # Load h[t-1]
            if t > 0:
                prev_off = pid_b * K * P + (t - 1) * P + p_offs
                h_prev_re = tl.load(h_re_ptr + prev_off, mask=p_mask)
                h_prev_im = tl.load(h_im_ptr + prev_off, mask=p_mask)
            else:
                h_prev_re = tl.load(state_re_ptr + pid_b * P + p_offs, mask=p_mask, other=0.0)
                h_prev_im = tl.load(state_im_ptr + pid_b * P + p_offs, mask=p_mask, other=0.0)

            # d_alpha[t] = d_h[t] * conj(h[t-1])
            # (d_re + j*d_im) * (h_re - j*h_im) = (d_re*h_re + d_im*h_im) + j*(d_im*h_re - d_re*h_im)
            d_a_re = d_h_re * h_prev_re + d_h_im * h_prev_im
            d_a_im = d_h_im * h_prev_re - d_h_re * h_prev_im
            tl.store(d_alpha_re_ptr + a_off, d_a_re, mask=p_mask)
            tl.store(d_alpha_im_ptr + a_off, d_a_im, mask=p_mask)

            # Carry adjoint: adj = conj(alpha[t]) * d_h[t]
            # (a_re - j*a_im) * (d_re + j*d_im) = (a_re*d_re + a_im*d_im) + j*(a_re*d_im - a_im*d_re)
            adj_re = a_re * d_h_re + a_im * d_h_im
            adj_im = a_re * d_h_im - a_im * d_h_re

        # Store adjoint as d_state (gradient w.r.t. incoming state)
        tl.store(d_state_re_ptr + pid_b * P + p_offs, adj_re, mask=p_mask)
        tl.store(d_state_im_ptr + pid_b * P + p_offs, adj_im, mask=p_mask)

    @triton.jit
    def complex_gate_readout_fwd_kernel(
        h_re_ptr, h_im_ptr, gate_ptr, out_re_ptr, out_im_ptr,
        N,
        BLOCK: tl.constexpr,
    ):
        """h_gated = h * gate where h is complex, gate is real."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        h_re = tl.load(h_re_ptr + offs, mask=mask)
        h_im = tl.load(h_im_ptr + offs, mask=mask)
        g = tl.load(gate_ptr + offs, mask=mask)
        tl.store(out_re_ptr + offs, h_re * g, mask=mask)
        tl.store(out_im_ptr + offs, h_im * g, mask=mask)

    @triton.jit
    def complex_gate_readout_bwd_kernel(
        d_out_re_ptr, d_out_im_ptr,
        h_re_ptr, h_im_ptr, gate_ptr,
        d_h_re_ptr, d_h_im_ptr, d_gate_ptr,
        N,
        BLOCK: tl.constexpr,
    ):
        """Backward for h_gated = h * gate."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        d_re = tl.load(d_out_re_ptr + offs, mask=mask)
        d_im = tl.load(d_out_im_ptr + offs, mask=mask)
        h_re = tl.load(h_re_ptr + offs, mask=mask)
        h_im = tl.load(h_im_ptr + offs, mask=mask)
        g = tl.load(gate_ptr + offs, mask=mask)
        tl.store(d_h_re_ptr + offs, d_re * g, mask=mask)
        tl.store(d_h_im_ptr + offs, d_im * g, mask=mask)
        tl.store(d_gate_ptr + offs, d_re * h_re + d_im * h_im, mask=mask)

    @triton.jit
    def skip_silu_fwd_kernel(
        y_ptr, u_ptr, D_ptr, out_ptr,
        N, H,
        BLOCK: tl.constexpr,
    ):
        """out = silu(y + u * D) where D is (H,) broadcast."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        y = tl.load(y_ptr + offs, mask=mask)
        u = tl.load(u_ptr + offs, mask=mask)
        d = tl.load(D_ptr + offs % H, mask=mask)
        x = y + u * d
        sig = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + offs, x * sig, mask=mask)

    @triton.jit
    def skip_silu_bwd_kernel(
        d_out_ptr, y_ptr, u_ptr, D_ptr,
        d_y_ptr, d_u_ptr, d_D_ptr,
        N, H,
        BLOCK: tl.constexpr,
    ):
        """Backward for out = silu(y + u*D)."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        d_out = tl.load(d_out_ptr + offs, mask=mask)
        y = tl.load(y_ptr + offs, mask=mask)
        u = tl.load(u_ptr + offs, mask=mask)
        d = tl.load(D_ptr + offs % H, mask=mask)
        x = y + u * d
        sig = 1.0 / (1.0 + tl.exp(-x))
        # silu'(x) = sig(x) + x*sig(x)*(1-sig(x)) = sig(x)*(1 + x*(1-sig(x)))
        d_x = d_out * sig * (1.0 + x * (1.0 - sig))
        tl.store(d_y_ptr + offs, d_x, mask=mask)
        tl.store(d_u_ptr + offs, d_x * d, mask=mask)
        # d_D: need atomic add since D is (H,) broadcast over B*L
        tl.atomic_add(d_D_ptr + offs % H, d_x * u, mask=mask)


# ========== Python wrappers ==========

def triton_linear(x, w, b=None):
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    linear_fwd_kernel[grid](
        x, w, b if b is not None else x, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        b is not None,
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return out


def triton_rmsnorm(x, gamma, eps=1e-6):
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    rmsnorm_fwd_kernel[(M,)](x, gamma, out, M, N, x.stride(0), x.stride(1), out.stride(0), out.stride(1), eps, BLOCK_N)
    return out


def triton_silu(x):
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    silu_fwd_kernel[(triton.cdiv(N, BLOCK),)](x.view(-1), out.view(-1), N, BLOCK)
    return out


def triton_silu_bwd(d_out, x):
    d_x = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    silu_bwd_kernel[(triton.cdiv(N, BLOCK),)](d_out.view(-1), x.view(-1), d_x.view(-1), N, BLOCK)
    return d_x


def triton_rmsnorm_bwd(d_out, x, gamma, eps=1e-6):
    M, N = x.shape
    d_x = torch.empty_like(x)
    d_gamma = torch.zeros_like(gamma)
    BLOCK_N = triton.next_power_of_2(N)
    rmsnorm_bwd_kernel[(M,)](
        d_out, x, gamma, d_x, d_gamma,
        M, N,
        d_out.stride(0), d_out.stride(1),
        x.stride(0), x.stride(1),
        d_x.stride(0), d_x.stride(1),
        eps, BLOCK_N)
    return d_x, d_gamma


def triton_fused_dt_lam(dt_raw, lam_raw, log_dt_bias):
    """dt = softplus(silu(dt_raw) + log_dt_bias), lam = sigmoid(lam_raw)."""
    dt_out = torch.empty_like(dt_raw)
    lam_out = torch.empty_like(lam_raw)
    N = dt_raw.numel()
    BLOCK = 1024
    # Pre-broadcast log_dt_bias to match dt_raw shape
    bias_bc = log_dt_bias.expand_as(dt_raw).contiguous().view(-1)
    fused_dt_lam_kernel[(triton.cdiv(N, BLOCK),)](
        dt_raw.view(-1), lam_raw.view(-1), bias_bc,
        dt_out.view(-1), lam_out.view(-1),
        N, BLOCK,
    )
    return dt_out, lam_out


def triton_rope(x, angles):
    """x: (..., P), angles: (..., P//2). Returns rotated x."""
    shape = x.shape
    P = shape[-1]
    N = x.numel() // P  # number of rows
    out = torch.empty_like(x)
    BLOCK = triton.next_power_of_2(P // 2)
    rope_fwd_kernel[(N,)](x.contiguous().view(-1), angles.contiguous().view(-1), out.view(-1), N, P, BLOCK)
    return out.view(shape)


def triton_complex_discretize(dt, lam, Bu, Bu_prev, A_real, A_imag):
    """Returns (alpha_re, alpha_im, inject_re, inject_im)."""
    N = dt.numel()
    P = A_real.shape[0]
    alpha_re = torch.empty_like(dt)
    alpha_im = torch.empty_like(dt)
    inject_re = torch.empty_like(dt)
    inject_im = torch.empty_like(dt)
    BLOCK = 1024
    complex_discretize_fwd_kernel[(triton.cdiv(N, BLOCK),)](
        dt.view(-1), lam.view(-1), Bu.view(-1), Bu_prev.view(-1),
        A_real, A_imag,
        alpha_re.view(-1), alpha_im.view(-1), inject_re.view(-1), inject_im.view(-1),
        N, P, BLOCK,
    )
    return alpha_re, alpha_im, inject_re, inject_im


def triton_chunked_scan(alpha_re, alpha_im, inject_re, inject_im, chunk_size=32):
    """Chunked complex scan. Returns (h_re, h_im) of shape (B, L, P)."""
    B, L, P = alpha_re.shape
    device = alpha_re.device
    dtype = alpha_re.dtype

    h_re = torch.empty(B, L, P, device=device, dtype=dtype)
    h_im = torch.empty(B, L, P, device=device, dtype=dtype)
    state_re = torch.zeros(B, P, device=device, dtype=dtype)
    state_im = torch.zeros(B, P, device=device, dtype=dtype)
    new_state_re = torch.empty_like(state_re)
    new_state_im = torch.empty_like(state_im)

    BLOCK_P = triton.next_power_of_2(P)

    for chunk_start in range(0, L, chunk_size):
        chunk_end = min(chunk_start + chunk_size, L)
        K = chunk_end - chunk_start

        chunked_scan_fwd_kernel[(B, triton.cdiv(P, BLOCK_P))](
            alpha_re[:, chunk_start:chunk_end].contiguous(),
            alpha_im[:, chunk_start:chunk_end].contiguous(),
            inject_re[:, chunk_start:chunk_end].contiguous(),
            inject_im[:, chunk_start:chunk_end].contiguous(),
            state_re, state_im,
            h_re[:, chunk_start:chunk_end],
            h_im[:, chunk_start:chunk_end],
            new_state_re, new_state_im,
            B, K, P,
            K * P, P,  # stride_al, stride_ap (not used but kept for interface)
            BLOCK_P,
        )
        state_re, new_state_re = new_state_re, state_re
        state_im, new_state_im = new_state_im, state_im

    return h_re, h_im


# ========== autograd.Function ==========

class _TritonS6(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                u,  # (B, L, H)
                # Kernel params (x_proj fused)
                x_proj_w, x_proj_b,  # Linear(H, P+P+P+P//2)
                b_norm_gamma, b_bias, log_dt_bias,
                log_A_real, A_imag,
                # C-side params
                c_proj_w, c_proj_b,
                c_norm_gamma, c_bias,
                C_re, C_im,  # (H, P) each
                D,  # (H,)
                # Config
                P, chunk_size):
        B, L, H = u.shape
        M = B * L
        split_sizes = [P, P, P, P // 2]

        u_flat = u.reshape(M, H)

        # 1. Fused input projection
        x_proj_out = triton_linear(u_flat, x_proj_w, x_proj_b)  # (M, 3.5P)
        Bu_raw, dt_raw, lam_raw, theta = x_proj_out.split(split_sizes, dim=-1)

        # 2. dt and lam activations
        dt, lam = triton_fused_dt_lam(dt_raw, lam_raw, log_dt_bias.expand(M, P).contiguous())

        # 3. B: silu → rmsnorm → + bias
        Bu_silu = triton_silu(Bu_raw)  # (M, P)
        Bu_normed = triton_rmsnorm(Bu_silu, b_norm_gamma)  # (M, P)
        # Add bias (fused would be better, but keep simple)
        Bu = Bu_normed + b_bias.unsqueeze(0)  # (M, P)

        # Reshape for sequence ops
        Bu_3d = Bu.view(B, L, P)
        Bu_prev = torch.zeros_like(Bu_3d)
        Bu_prev[:, 1:] = Bu_3d[:, :-1]

        # 4. RoPE
        dt_3d = dt.view(B, L, P)
        theta_3d = theta.view(B, L, P // 2)
        dt_half = dt_3d.view(B, L, P // 2, 2).mean(-1)  # (B, L, P//2)
        cum_theta = torch.cumsum(dt_half * theta_3d, dim=1)  # (B, L, P//2)

        Bu_rotated = triton_rope(Bu_3d, cum_theta)
        Bu_prev_rotated = triton_rope(Bu_prev, cum_theta)

        # 5. Complex discretization
        A_real_neg = -torch.exp(log_A_real)  # (P,)
        alpha_re, alpha_im, inject_re, inject_im = triton_complex_discretize(
            dt_3d, lam.view(B, L, P), Bu_rotated, Bu_prev_rotated, A_real_neg, A_imag)

        # 6. Chunked scan
        h_re, h_im = triton_chunked_scan(alpha_re, alpha_im, inject_re, inject_im, chunk_size)

        # 7. C-side: projection + silu + rmsnorm + bias + RoPE
        c_proj_out = triton_linear(u_flat, c_proj_w, c_proj_b)  # (M, P)
        c_silu = triton_silu(c_proj_out)
        c_normed = triton_rmsnorm(c_silu, c_norm_gamma)
        c_gate = (c_normed + c_bias.unsqueeze(0)).view(B, L, P)
        c_gate_rotated = triton_rope(c_gate, cum_theta)

        # 8. Complex gating: h_gated = h * c_gate (complex * real)
        N_elem = B * L * P
        BLOCK = 1024
        hg_re = torch.empty_like(h_re)
        hg_im = torch.empty_like(h_im)
        complex_gate_readout_fwd_kernel[(triton.cdiv(N_elem, BLOCK),)](
            h_re.view(-1), h_im.view(-1), c_gate_rotated.contiguous().view(-1),
            hg_re.view(-1), hg_im.view(-1), N_elem, BLOCK)

        # 9. MIMO readout: y = C_re @ hg_re - C_im @ hg_im  (real part of complex matmul)
        hg_re_flat = hg_re.reshape(M, P)
        hg_im_flat = hg_im.reshape(M, P)
        # triton_linear does x @ w^T, so w=(H,P), x=(M,P) → out (M, H)
        y_re = triton_linear(hg_re_flat, C_re)  # (M, H)
        y_im = triton_linear(hg_im_flat, C_im)  # (M, H)
        y = (y_re - y_im).view(B, L, H)  # real part of complex product

        # 10. Skip + SiLU: out = silu(y + u * D)
        N_out = B * L * H
        out = torch.empty(B, L, H, device=u.device, dtype=u.dtype)
        skip_silu_fwd_kernel[(triton.cdiv(N_out, BLOCK),)](
            y.contiguous().view(-1), u.contiguous().view(-1), D,
            out.view(-1), N_out, H, BLOCK)

        # Save for backward
        ctx.save_for_backward(
            u, Bu_raw, Bu_silu, Bu_normed, Bu_3d, Bu_prev,
            dt_raw, dt, lam_raw, lam, theta, dt_half, cum_theta,
            Bu_rotated, Bu_prev_rotated,
            alpha_re, alpha_im, inject_re, inject_im,
            h_re, h_im,
            c_proj_out, c_silu, c_normed, c_gate, c_gate_rotated,
            hg_re, hg_im, y,
            x_proj_w, x_proj_b, b_norm_gamma, b_bias, log_dt_bias, log_A_real, A_imag,
            c_proj_w, c_proj_b, c_norm_gamma, c_bias, C_re, C_im, D,
        )
        ctx.P = P
        ctx.chunk_size = chunk_size
        ctx.split_sizes = split_sizes

        return out

    @staticmethod
    def backward(ctx, d_out):
        (u, Bu_raw, Bu_silu, Bu_normed, Bu_3d, Bu_prev,
         dt_raw, dt, lam_raw, lam, theta, dt_half, cum_theta,
         Bu_rotated, Bu_prev_rotated,
         alpha_re, alpha_im, inject_re, inject_im,
         h_re, h_im,
         c_proj_out, c_silu, c_normed, c_gate, c_gate_rotated,
         hg_re, hg_im, y,
         x_proj_w, x_proj_b, b_norm_gamma, b_bias, log_dt_bias, log_A_real, A_imag,
         c_proj_w, c_proj_b, c_norm_gamma, c_bias, C_re, C_im, D,
        ) = ctx.saved_tensors

        P = ctx.P
        chunk_size = ctx.chunk_size
        split_sizes = ctx.split_sizes
        B, L, H = u.shape
        M = B * L
        N_elem = B * L * P
        N_out = B * L * H
        BLOCK = 1024
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32

        u_flat = u.reshape(M, H)

        # ---- 10. Backward through skip_silu: out = silu(y + u*D) ----
        d_y = torch.empty(B, L, H, device=u.device, dtype=u.dtype)
        d_u_skip = torch.empty_like(d_y)
        d_D = torch.zeros_like(D)
        skip_silu_bwd_kernel[(triton.cdiv(N_out, BLOCK),)](
            d_out.contiguous().view(-1), y.contiguous().view(-1),
            u.contiguous().view(-1), D,
            d_y.view(-1), d_u_skip.view(-1), d_D,
            N_out, H, BLOCK)

        # ---- 9. Backward through C readout: y = C_re @ hg_re - C_im @ hg_im ----
        d_y_flat = d_y.reshape(M, H)
        hg_re_flat = hg_re.reshape(M, P)
        hg_im_flat = hg_im.reshape(M, P)

        # d_hg_re = d_y @ C_re^T ... wait, y_re = hg_re @ C_re^T (triton_linear does x @ w^T)
        # so d_hg_re = d_y @ C_re (since d/d_x of x@W^T is d_out @ W)
        d_hg_re = torch.empty(M, P, device=u.device, dtype=u.dtype)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(P, BLOCK_K))
        linear_bwd_dx_kernel[grid](
            d_y_flat, C_re, d_hg_re,
            M, H, P,
            d_y_flat.stride(0), d_y_flat.stride(1),
            C_re.stride(0), C_re.stride(1),
            d_hg_re.stride(0), d_hg_re.stride(1),
            BLOCK_M, BLOCK_K, BLOCK_N)

        # d_hg_im = -d_y @ C_im (negative from real part of complex product)
        d_hg_im = torch.empty(M, P, device=u.device, dtype=u.dtype)
        linear_bwd_dx_kernel[grid](
            d_y_flat, C_im, d_hg_im,
            M, H, P,
            d_y_flat.stride(0), d_y_flat.stride(1),
            C_im.stride(0), C_im.stride(1),
            d_hg_im.stride(0), d_hg_im.stride(1),
            BLOCK_M, BLOCK_K, BLOCK_N)
        d_hg_im = -d_hg_im

        # d_C_re = d_y^T @ hg_re
        d_C_re = torch.zeros_like(C_re)
        grid = (triton.cdiv(H, BLOCK_N), triton.cdiv(P, BLOCK_K))
        linear_bwd_dw_kernel[grid](
            d_y_flat, hg_re_flat, d_C_re,
            M, H, P,
            d_y_flat.stride(0), d_y_flat.stride(1),
            hg_re_flat.stride(0), hg_re_flat.stride(1),
            d_C_re.stride(0), d_C_re.stride(1),
            BLOCK_N, BLOCK_K, BLOCK_M)

        # d_C_im = -d_y^T @ hg_im
        d_C_im = torch.zeros_like(C_im)
        linear_bwd_dw_kernel[grid](
            d_y_flat, hg_im_flat, d_C_im,
            M, H, P,
            d_y_flat.stride(0), d_y_flat.stride(1),
            hg_im_flat.stride(0), hg_im_flat.stride(1),
            d_C_im.stride(0), d_C_im.stride(1),
            BLOCK_N, BLOCK_K, BLOCK_M)
        d_C_im = -d_C_im

        # ---- 8. Backward through complex gating ----
        d_h_re = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        d_h_im = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        d_c_gate_rotated = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        complex_gate_readout_bwd_kernel[(triton.cdiv(N_elem, BLOCK),)](
            d_hg_re.view(B, L, P).contiguous().view(-1),
            d_hg_im.view(B, L, P).contiguous().view(-1),
            h_re.contiguous().view(-1), h_im.contiguous().view(-1),
            c_gate_rotated.contiguous().view(-1),
            d_h_re.view(-1), d_h_im.view(-1), d_c_gate_rotated.view(-1),
            N_elem, BLOCK)

        # ---- 7. Backward through C-side RoPE + rmsnorm + silu + linear ----
        # d_c_gate from d_c_gate_rotated through RoPE backward
        d_c_gate = torch.empty_like(d_c_gate_rotated)
        d_cum_theta_c = torch.empty(B, L, P // 2, device=u.device, dtype=u.dtype)
        BLOCK_ROPE = triton.next_power_of_2(P // 2)
        N_rows = B * L
        rope_bwd_kernel[(N_rows,)](
            d_c_gate_rotated.contiguous().view(-1), cum_theta.contiguous().view(-1),
            d_c_gate.view(-1), d_cum_theta_c.view(-1),
            c_gate.contiguous().view(-1),
            N_rows, P, BLOCK_ROPE)

        # d_c_bias = sum(d_c_gate)
        d_c_bias = d_c_gate.view(M, P).sum(0)

        # d_c_normed = d_c_gate (bias is additive)
        d_c_normed = d_c_gate.view(M, P)

        # Backward through rmsnorm (Triton)
        d_c_silu, d_c_norm_gamma = triton_rmsnorm_bwd(d_c_normed, c_silu, c_norm_gamma)

        # Backward through silu (Triton)
        d_c_proj_out = triton_silu_bwd(d_c_silu, c_proj_out)

        # Backward through c_proj linear
        d_c_proj_w = torch.zeros_like(c_proj_w)
        grid = (triton.cdiv(P, BLOCK_N), triton.cdiv(H, BLOCK_K))
        linear_bwd_dw_kernel[grid](
            d_c_proj_out, u_flat, d_c_proj_w,
            M, P, H,
            d_c_proj_out.stride(0), d_c_proj_out.stride(1),
            u_flat.stride(0), u_flat.stride(1),
            d_c_proj_w.stride(0), d_c_proj_w.stride(1),
            BLOCK_N, BLOCK_K, BLOCK_M)
        d_c_proj_b = d_c_proj_out.sum(0)

        d_u_c = torch.zeros(M, H, device=u.device, dtype=u.dtype)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_K))
        linear_bwd_dx_kernel[grid](
            d_c_proj_out, c_proj_w, d_u_c,
            M, P, H,
            d_c_proj_out.stride(0), d_c_proj_out.stride(1),
            c_proj_w.stride(0), c_proj_w.stride(1),
            d_u_c.stride(0), d_u_c.stride(1),
            BLOCK_M, BLOCK_K, BLOCK_N)

        # ---- 6. Backward through chunked scan ----
        # For now use PyTorch sequential backward (scan backward kernel exists but
        # needs careful chunk-boundary state management)
        # TODO: use chunked_scan_bwd_kernel with proper inter-chunk adjoint passing
        d_alpha_re = torch.empty_like(alpha_re)
        d_alpha_im = torch.empty_like(alpha_im)
        d_inject_re = torch.empty_like(inject_re)
        d_inject_im = torch.empty_like(inject_im)

        BLOCK_P = triton.next_power_of_2(P)
        # Process chunks in reverse order
        # We need the incoming state for each chunk from the forward pass
        # For simplicity, recompute states
        states_re = [torch.zeros(B, P, device=u.device, dtype=u.dtype)]
        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            states_re.append(h_re[:, chunk_end - 1].clone())
        states_im = [torch.zeros(B, P, device=u.device, dtype=u.dtype)]
        for chunk_start in range(0, L, chunk_size):
            chunk_end = min(chunk_start + chunk_size, L)
            states_im.append(h_im[:, chunk_end - 1].clone())

        # Adjoint state carried between chunks (backward)
        adj_state_re = torch.zeros(B, P, device=u.device, dtype=u.dtype)
        adj_state_im = torch.zeros(B, P, device=u.device, dtype=u.dtype)

        chunk_starts = list(range(0, L, chunk_size))
        for ci in reversed(range(len(chunk_starts))):
            cs = chunk_starts[ci]
            ce = min(cs + chunk_size, L)
            K = ce - cs

            # Add adjoint from next chunk to d_h at last position of this chunk
            d_h_re_chunk = d_h_re[:, cs:ce].contiguous()
            d_h_im_chunk = d_h_im[:, cs:ce].contiguous()
            d_h_re_chunk[:, -1] += adj_state_re
            d_h_im_chunk[:, -1] += adj_state_im

            d_alpha_re_chunk = torch.empty(B, K, P, device=u.device, dtype=u.dtype)
            d_alpha_im_chunk = torch.empty(B, K, P, device=u.device, dtype=u.dtype)
            d_inject_re_chunk = torch.empty(B, K, P, device=u.device, dtype=u.dtype)
            d_inject_im_chunk = torch.empty(B, K, P, device=u.device, dtype=u.dtype)
            d_state_re_chunk = torch.empty(B, P, device=u.device, dtype=u.dtype)
            d_state_im_chunk = torch.empty(B, P, device=u.device, dtype=u.dtype)

            chunked_scan_bwd_kernel[(B, triton.cdiv(P, BLOCK_P))](
                alpha_re[:, cs:ce].contiguous(), alpha_im[:, cs:ce].contiguous(),
                d_h_re_chunk, d_h_im_chunk,
                d_alpha_re_chunk, d_alpha_im_chunk,
                d_inject_re_chunk, d_inject_im_chunk,
                d_state_re_chunk, d_state_im_chunk,
                h_re[:, cs:ce].contiguous(), h_im[:, cs:ce].contiguous(),
                states_re[ci], states_im[ci],
                B, K, P, BLOCK_P)

            d_alpha_re[:, cs:ce] = d_alpha_re_chunk
            d_alpha_im[:, cs:ce] = d_alpha_im_chunk
            d_inject_re[:, cs:ce] = d_inject_re_chunk
            d_inject_im[:, cs:ce] = d_inject_im_chunk
            adj_state_re = d_state_re_chunk
            adj_state_im = d_state_im_chunk

        # ---- 5. Backward through complex discretization ----
        d_dt_disc = torch.empty_like(dt).view(B, L, P)
        d_lam_disc = torch.empty_like(lam).view(B, L, P)
        d_Bu_rot = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        d_Bu_prev_rot = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        A_real_neg = -torch.exp(log_A_real)

        complex_discretize_bwd_kernel[(triton.cdiv(N_elem, BLOCK),)](
            d_inject_re.contiguous().view(-1), d_inject_im.contiguous().view(-1),
            d_alpha_re.contiguous().view(-1), d_alpha_im.contiguous().view(-1),
            dt.view(B, L, P).contiguous().view(-1), lam.view(B, L, P).contiguous().view(-1),
            Bu_rotated.contiguous().view(-1), Bu_prev_rotated.contiguous().view(-1),
            A_real_neg, A_imag,
            alpha_re.contiguous().view(-1), alpha_im.contiguous().view(-1),
            d_dt_disc.view(-1), d_lam_disc.view(-1),
            d_Bu_rot.view(-1), d_Bu_prev_rot.view(-1),
            N_elem, P, BLOCK)

        # ---- 4. Backward through RoPE on Bu and Bu_prev ----
        d_Bu_3d = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        d_cum_theta_b1 = torch.empty(B, L, P // 2, device=u.device, dtype=u.dtype)
        rope_bwd_kernel[(N_rows,)](
            d_Bu_rot.contiguous().view(-1), cum_theta.contiguous().view(-1),
            d_Bu_3d.view(-1), d_cum_theta_b1.view(-1),
            Bu_3d.contiguous().view(-1),
            N_rows, P, BLOCK_ROPE)

        d_Bu_prev_3d = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        d_cum_theta_b2 = torch.empty(B, L, P // 2, device=u.device, dtype=u.dtype)
        rope_bwd_kernel[(N_rows,)](
            d_Bu_prev_rot.contiguous().view(-1), cum_theta.contiguous().view(-1),
            d_Bu_prev_3d.view(-1), d_cum_theta_b2.view(-1),
            Bu_prev.contiguous().view(-1),
            N_rows, P, BLOCK_ROPE)

        # Shift d_Bu_prev back: d_Bu[:, :-1] += d_Bu_prev[:, 1:]
        d_Bu_3d[:, :-1] += d_Bu_prev_3d[:, 1:]

        # ---- cum_theta grad: reverse cumsum ----
        d_cum_theta = d_cum_theta_c + d_cum_theta_b1 + d_cum_theta_b2
        d_dt_half_theta = d_cum_theta.flip(1).cumsum(1).flip(1)  # reverse cumsum
        d_dt_half = d_dt_half_theta * theta.view(B, L, P // 2)
        d_theta = d_dt_half_theta * dt_half
        # d_dt from dt_half: dt_half = dt.view(B,L,P//2,2).mean(-1)
        d_dt_from_rope = d_dt_half.unsqueeze(-1).expand(B, L, P // 2, 2).reshape(B, L, P) / 2

        # ---- 3. Backward through B norm + bias ----
        d_Bu_flat = d_Bu_3d.reshape(M, P)
        # d_b_bias = sum(d_Bu)
        d_b_bias = d_Bu_flat.sum(0)
        # d_Bu_normed = d_Bu (additive bias)
        d_Bu_normed = d_Bu_flat
        # rmsnorm backward (Triton)
        d_Bu_silu, d_b_norm_gamma = triton_rmsnorm_bwd(d_Bu_normed, Bu_silu, b_norm_gamma)
        # silu backward (Triton)
        d_Bu_raw = triton_silu_bwd(d_Bu_silu, Bu_raw)

        # ---- 2. Backward through dt/lam activations ----
        d_dt_total = d_dt_disc.reshape(M, P) + d_dt_from_rope.reshape(M, P)
        d_dt_raw = torch.empty_like(dt_raw)
        d_lam_raw = torch.empty_like(lam_raw)
        d_log_dt_bias_flat = torch.empty(M, P, device=u.device, dtype=u.dtype)
        bias_bc = log_dt_bias.expand(M, P).contiguous()

        fused_dt_lam_bwd_kernel[(triton.cdiv(M * P, BLOCK),)](
            d_dt_total.contiguous().view(-1), d_lam_disc.reshape(M, P).contiguous().view(-1),
            dt_raw.contiguous().view(-1), lam_raw.contiguous().view(-1),
            bias_bc.view(-1),
            d_dt_raw.view(-1), d_lam_raw.view(-1), d_log_dt_bias_flat.view(-1),
            M * P, BLOCK)

        d_log_dt_bias = d_log_dt_bias_flat.sum(0)

        # ---- 1. Backward through x_proj linear ----
        # Reassemble d_x_proj from splits
        d_x_proj = torch.cat([d_Bu_raw, d_dt_raw, d_lam_raw, d_theta.reshape(M, P // 2)], dim=-1)

        d_x_proj_w = torch.zeros_like(x_proj_w)
        out_dim = x_proj_w.shape[0]
        grid = (triton.cdiv(out_dim, BLOCK_N), triton.cdiv(H, BLOCK_K))
        linear_bwd_dw_kernel[grid](
            d_x_proj, u_flat, d_x_proj_w,
            M, out_dim, H,
            d_x_proj.stride(0), d_x_proj.stride(1),
            u_flat.stride(0), u_flat.stride(1),
            d_x_proj_w.stride(0), d_x_proj_w.stride(1),
            BLOCK_N, BLOCK_K, BLOCK_M)
        d_x_proj_b = d_x_proj.sum(0)

        d_u_proj = torch.zeros(M, H, device=u.device, dtype=u.dtype)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_K))
        linear_bwd_dx_kernel[grid](
            d_x_proj, x_proj_w, d_u_proj,
            M, out_dim, H,
            d_x_proj.stride(0), d_x_proj.stride(1),
            x_proj_w.stride(0), x_proj_w.stride(1),
            d_u_proj.stride(0), d_u_proj.stride(1),
            BLOCK_M, BLOCK_K, BLOCK_N)

        # Total d_u
        d_u = (d_u_proj + d_u_c + d_u_skip.view(M, H)).view(B, L, H)

        # d_log_A_real: from d_alpha through A_real_neg = -exp(log_A_real)
        # d_A_real_neg comes from discretize backward (accumulated over B,L)
        # We need to sum d_alpha contributions... this was handled inside complex_discretize_bwd
        # which gives d_dt. For d_A we need: d_alpha * d(exp(dt*A))/dA = d_alpha * dt * alpha
        # This requires another pass or we fold it into the discretize kernel.
        # For now: compute from chain rule
        # alpha = exp(dt * A), A = A_real_neg + j*A_imag
        # d_A_real_neg = sum over B,L of: d_alpha_re * dt * alpha_re_from_real - ...
        # Actually this is complex: d_A = sum(d_alpha * dt * alpha) (conjugate)
        # Let's just do it in PyTorch for now since it's a (P,) reduction
        dt_3d_cont = dt.view(B, L, P)
        d_A_real = (d_alpha_re * dt_3d_cont * alpha_re + d_alpha_im * dt_3d_cont * alpha_im).sum(dim=(0, 1))
        d_A_imag = (-d_alpha_re * dt_3d_cont * alpha_im + d_alpha_im * dt_3d_cont * alpha_re).sum(dim=(0, 1))
        # Chain through -exp(log_A_real): d_log_A_real = d_A_real * A_real_neg
        d_log_A_real = d_A_real * A_real_neg

        return (d_u,
                d_x_proj_w, d_x_proj_b,
                d_b_norm_gamma, d_b_bias, d_log_dt_bias,
                d_log_A_real, d_A_imag,
                d_c_proj_w, d_c_proj_b,
                d_c_norm_gamma, d_c_bias,
                d_C_re, d_C_im,
                d_D,
                None, None)  # P, chunk_size


# ========== nn.Module wrapper ==========

class TritonS6(nn.Module):
    """S6 using Triton kernels. Falls back to PyTorch S6 on non-CUDA devices."""

    def __init__(self, d_model, d_state=64, chunk_size=32, layer_idx=None, **kwargs):
        super().__init__()
        from .s6 import S6, S6Kernel, make_DPLR_HiPPO, RMSNorm
        import numpy as np

        H = d_model
        P = d_state
        self.H = H
        self.P = P
        self.chunk_size = chunk_size

        # Build a PyTorch S6 for init and fallback
        self._pytorch_s6 = S6(d_model, d_state, layer_idx=layer_idx, **kwargs)

        # Extract references to the actual parameters (shared with _pytorch_s6)
        # This way the Module's parameters() includes everything

    def forward(self, u, **kwargs):
        if not u.is_cuda or not HAS_TRITON:
            return self._pytorch_s6(u, **kwargs)

        s6 = self._pytorch_s6
        kern = s6.kernel

        # Split complex C into real/imag for Triton
        C = torch.view_as_complex(s6.C)
        C_re = C.real.contiguous()
        C_im = C.imag.contiguous()

        # Materialize A
        A_real_neg = -torch.exp(kern.log_A_real)

        return _TritonS6.apply(
            u,
            kern.x_proj.weight, kern.x_proj.bias,
            kern.b_norm.weight, kern.b_bias, kern.log_dt_bias,
            kern.log_A_real, kern.A_imag,
            s6.c_proj.weight, s6.c_proj.bias,
            s6.c_norm.weight, s6.c_bias,
            C_re, C_im,
            s6.D,
            self.P, self.chunk_size,
        )
