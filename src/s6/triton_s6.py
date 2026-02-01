"""Triton kernels for S6 — fused projections, activations, scan, and readout.

Rewrite: 3 fused custom kernels + existing linear kernels.
Forward: x_proj linear → phi_B (PyTorch) → fused_prescan → cumsum → fused_scan → c_proj linear → fused_postscan
"""

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

LOG2E = math.log2(math.e)

if HAS_TRITON:

    # ========== Linear kernels (unchanged from triton_adaptive_conv.py) ==========

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
            acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=False)
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
            acc = tl.dot(do_tile, w_tile, acc, allow_tf32=False)
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
            acc = tl.dot(tl.trans(do_tile), x_tile, acc, allow_tf32=False)
            do_ptrs += BLOCK_M * stride_dom
            x_ptrs += BLOCK_M * stride_xm
        dw_ptrs = d_w_ptr + offs_n[:, None] * stride_dwn + offs_k[None, :] * stride_dwk
        dw_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        tl.store(dw_ptrs, acc, mask=dw_mask)

    # ========== Fused kernel 1: prescan ==========
    # Per row (B*L rows, P cols): dt/lam activations, Bu RMSNorm+bias, dt_half*theta
    # Grid: (M,) where M = B*L, one program per row

    @triton.jit
    def fused_prescan_kernel(
        # Input pointers
        dt_raw_ptr, lam_raw_ptr, theta_ptr,  # from x_proj split: (M, P), (M, P), (M, P//2)
        Bu_raw_ptr,  # (M, P) from phi_B
        # Param pointers
        log_dt_bias_ptr,  # (P,)
        b_norm_gamma_ptr,  # (P,)
        b_bias_ptr,  # (P,)
        # Output pointers
        dt_ptr, lam_ptr, Bu_ptr,  # (M, P) each
        dt_half_theta_ptr,  # (M, P//2) — for cumsum outside
        # Dims
        M, P,
        eps,
        BLOCK_P: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= M:
            return
        offs_p = tl.arange(0, BLOCK_P)
        p_mask = offs_p < P

        # --- dt activation: softplus(silu(dt_raw) + log_dt_bias) ---
        dt_raw = tl.load(dt_raw_ptr + row * P + offs_p, mask=p_mask, other=0.0)
        sig_dt = 1.0 / (1.0 + tl.exp(-dt_raw))
        silu_dt = dt_raw * sig_dt
        bias = tl.load(log_dt_bias_ptr + offs_p, mask=p_mask, other=0.0)
        pre = silu_dt + bias
        dt = tl.where(pre > 20.0, pre, tl.log(1.0 + tl.exp(pre)))
        tl.store(dt_ptr + row * P + offs_p, dt, mask=p_mask)

        # --- lam activation: sigmoid(lam_raw) ---
        lam_raw = tl.load(lam_raw_ptr + row * P + offs_p, mask=p_mask, other=0.0)
        lam = 1.0 / (1.0 + tl.exp(-lam_raw))
        tl.store(lam_ptr + row * P + offs_p, lam, mask=p_mask)

        # --- Bu: RMSNorm(Bu_raw) + b_bias ---
        Bu_raw = tl.load(Bu_raw_ptr + row * P + offs_p, mask=p_mask, other=0.0)
        variance = tl.sum(Bu_raw * Bu_raw) / P
        Bu_normed = Bu_raw * tl.rsqrt(variance + eps)
        gamma = tl.load(b_norm_gamma_ptr + offs_p, mask=p_mask, other=1.0)
        b_bias = tl.load(b_bias_ptr + offs_p, mask=p_mask, other=0.0)
        Bu = gamma * Bu_normed + b_bias
        tl.store(Bu_ptr + row * P + offs_p, Bu, mask=p_mask)

        # --- dt_half * theta: for cumsum ---
        half_P = P // 2
        offs_h = tl.arange(0, BLOCK_P // 2)
        h_mask = offs_h < half_P
        # dt_half = dt.view(P//2, 2).mean(-1) = (dt[2i] + dt[2i+1]) / 2
        dt_even = tl.load(dt_ptr + row * P + offs_h * 2, mask=h_mask, other=0.0)
        dt_odd = tl.load(dt_ptr + row * P + offs_h * 2 + 1, mask=h_mask, other=0.0)
        dt_half = (dt_even + dt_odd) * 0.5
        theta = tl.load(theta_ptr + row * half_P + offs_h, mask=h_mask, other=0.0)
        tl.store(dt_half_theta_ptr + row * half_P + offs_h, dt_half * theta, mask=h_mask)

    # ========== Fused kernel 2: scan ==========
    # Grid: (B, cdiv(P, BLOCK_P_SCAN))
    # Each program loops over ALL L positions sequentially for its batch element
    # and state-dim block. Fuses: RoPE on Bu, shift, complex discretize, recurrence.

    @triton.jit
    def fused_scan_kernel(
        # Inputs (all contiguous, row-major)
        dt_ptr,  # (B, L, P)
        lam_ptr,  # (B, L, P)
        Bu_ptr,  # (B, L, P)
        cum_theta_ptr,  # (B, L, P//2)
        # SSM params
        A_real_ptr,  # (P,) negative real parts (already negated: -exp(log_A_real))
        A_imag_ptr,  # (P,) imaginary parts
        # Outputs
        h_re_ptr, h_im_ptr,  # (B, L, P)
        alpha_re_ptr, alpha_im_ptr,  # (B, L, P) — saved for backward
        inject_re_ptr, inject_im_ptr,  # (B, L, P) — saved for backward
        Bu_rot_ptr,  # (B, L, P) — saved for backward (Bu after RoPE)
        # Dims
        B_batch, L, P,
        BLOCK_P: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_p = tl.program_id(1)
        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        p_mask = offs_p < P

        # Load A params (static)
        a_re = tl.load(A_real_ptr + offs_p, mask=p_mask, other=0.0)  # negative
        a_im = tl.load(A_imag_ptr + offs_p, mask=p_mask, other=0.0)

        # Running state (complex)
        s_re = tl.zeros((BLOCK_P,), dtype=tl.float32)
        s_im = tl.zeros((BLOCK_P,), dtype=tl.float32)

        # Previous Bu_rotated for trapezoidal (initialized to zero)
        Bu_prev_re = tl.zeros((BLOCK_P,), dtype=tl.float32)
        Bu_prev_im = tl.zeros((BLOCK_P,), dtype=tl.float32)

        # Base offsets
        base = pid_b * L * P
        half_P = P // 2
        base_theta = pid_b * L * half_P

        # Pair indices for RoPE: offs_p maps to pair index offs_p // 2
        # and within-pair index offs_p % 2
        pair_idx = offs_p // 2  # (BLOCK_P,)
        is_odd = (offs_p % 2).to(tl.float32)  # 0.0 or 1.0
        is_even = 1.0 - is_odd

        for t in range(L):
            off = base + t * P + offs_p
            off_theta = base_theta + t * half_P + pair_idx

            # Load dt, lam, Bu for this position
            dt = tl.load(dt_ptr + off, mask=p_mask, other=0.0)
            lam = tl.load(lam_ptr + off, mask=p_mask, other=0.0)
            Bu = tl.load(Bu_ptr + off, mask=p_mask, other=0.0)

            # Load cum_theta for this position (P//2 values, indexed by pair)
            angle = tl.load(cum_theta_ptr + off_theta, mask=p_mask, other=0.0)

            # RoPE on Bu: rotate pairs
            # For even index: Bu_rot = Bu_even * cos - Bu_odd * sin
            # For odd index:  Bu_rot = Bu_even * sin + Bu_odd * cos
            # We need the paired value. Use shuffle trick:
            # Even elements need odd partner, odd elements need even partner.
            cos_a = tl.cos(angle)
            sin_a = tl.sin(angle)

            # Load the partner value
            partner_offs = tl.where(is_odd > 0.5, offs_p - 1, offs_p + 1)
            partner_off = base + t * P + partner_offs
            partner_mask = partner_offs < P
            Bu_partner = tl.load(Bu_ptr + partner_off, mask=p_mask & partner_mask, other=0.0)

            # Apply rotation
            # even: out = Bu * cos - partner * sin
            # odd:  out = partner * sin + Bu * cos  =>  partner_even * sin + Bu_odd * cos
            # Wait — standard RoPE:
            #   x1' = x1*cos - x2*sin  (even)
            #   x2' = x1*sin + x2*cos  (odd)
            # So even needs itself*cos - odd_partner*sin
            #    odd needs even_partner*sin + itself*cos
            Bu_rot = tl.where(
                is_odd > 0.5,
                Bu_partner * sin_a + Bu * cos_a,  # odd: even_partner*sin + self*cos
                Bu * cos_a - Bu_partner * sin_a,  # even: self*cos - odd_partner*sin
            )

            # Store Bu_rot for backward
            tl.store(Bu_rot_ptr + off, Bu_rot, mask=p_mask)

            # Complex discretization: alpha = exp(dt * A)
            dt_a_re = dt * a_re  # dt * A_real (A_real is negative)
            dt_a_im = dt * a_im

            exp_re = tl.exp(dt_a_re)
            alpha_re_t = exp_re * tl.cos(dt_a_im)
            alpha_im_t = exp_re * tl.sin(dt_a_im)

            # Store alpha for backward
            tl.store(alpha_re_ptr + off, alpha_re_t, mask=p_mask)
            tl.store(alpha_im_ptr + off, alpha_im_t, mask=p_mask)

            # Trapezoidal inject (Bu_rot is real, alpha is complex)
            # inject = lam*dt*Bu_rot + (1-lam)*dt*alpha*Bu_prev_rot
            # Bu_prev_rot is complex (from previous position's RoPE)
            # Actually — Bu_rot is REAL (RoPE applied to real Bu gives real result)
            # Bu_prev is the PREVIOUS position's Bu_rot, also real
            # So inject_re = lam*dt*Bu_rot + (1-lam)*dt*(alpha_re*Bu_prev_re - alpha_im*Bu_prev_im)
            # inject_im = (1-lam)*dt*(alpha_re*Bu_prev_im + alpha_im*Bu_prev_re)
            # But Bu_prev_im = 0 since Bu_prev is real! So:
            # inject_re = lam*dt*Bu_rot + (1-lam)*dt*alpha_re*Bu_prev_re
            # inject_im = (1-lam)*dt*alpha_im*Bu_prev_re
            one_minus_lam = 1.0 - lam
            inj_re = lam * dt * Bu_rot + one_minus_lam * dt * alpha_re_t * Bu_prev_re
            inj_im = one_minus_lam * dt * alpha_im_t * Bu_prev_re

            # Store inject for backward
            tl.store(inject_re_ptr + off, inj_re, mask=p_mask)
            tl.store(inject_im_ptr + off, inj_im, mask=p_mask)

            # Recurrence: h[t] = alpha[t] * h[t-1] + inject[t]
            new_re = alpha_re_t * s_re - alpha_im_t * s_im + inj_re
            new_im = alpha_re_t * s_im + alpha_im_t * s_re + inj_im
            s_re = new_re
            s_im = new_im

            # Store h
            tl.store(h_re_ptr + off, s_re, mask=p_mask)
            tl.store(h_im_ptr + off, s_im, mask=p_mask)

            # Bu_prev for next iteration (real, from current Bu_rot)
            Bu_prev_re = Bu_rot

    # ========== Fused kernel 3: postscan ==========
    # Computes c_gate (silu + rmsnorm + bias + RoPE) and stores it.
    # Grid: (M,) where M = B*L
    # Separate readout kernel handles the MIMO matmul + skip + silu.

    @triton.jit
    def fused_cgate_kernel(
        # Inputs
        c_proj_out_ptr,  # (M, P) — output of c_proj linear
        cum_theta_ptr,  # (B, L, P//2) — same as used in scan
        # Params
        c_norm_gamma_ptr,  # (P,)
        c_bias_ptr,  # (P,)
        # Output
        c_gate_ptr,  # (M, P)
        # Dims
        M, P, B_batch, L,
        eps,
        BLOCK_P: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= M:
            return
        offs_p = tl.arange(0, BLOCK_P)
        p_mask = offs_p < P
        half_P = P // 2

        # silu(c_proj_out)
        x = tl.load(c_proj_out_ptr + row * P + offs_p, mask=p_mask, other=0.0)
        sig = 1.0 / (1.0 + tl.exp(-x))
        x_silu = x * sig

        # RMSNorm
        variance = tl.sum(x_silu * x_silu) / P
        x_normed = x_silu * tl.rsqrt(variance + eps)
        gamma = tl.load(c_norm_gamma_ptr + offs_p, mask=p_mask, other=1.0)
        c_gate = gamma * x_normed

        # + bias
        c_b = tl.load(c_bias_ptr + offs_p, mask=p_mask, other=0.0)
        c_gate = c_gate + c_b

        # RoPE using cum_theta
        # cum_theta is (B, L, P//2), row index maps to (b, l)
        # b = row // L, l = row % L — but cum_theta is stored as (B*L, P//2) contiguous
        pair_idx = offs_p // 2
        is_odd = (offs_p % 2).to(tl.float32)

        angle = tl.load(cum_theta_ptr + row * half_P + pair_idx, mask=p_mask, other=0.0)
        cos_a = tl.cos(angle)
        sin_a = tl.sin(angle)

        partner_offs = tl.where(is_odd > 0.5, offs_p - 1, offs_p + 1)
        partner_mask = partner_offs < P
        c_gate_partner = tl.load(c_proj_out_ptr + row * P + partner_offs, mask=p_mask & partner_mask, other=0.0)
        # Need to recompute partner's silu + rmsnorm + bias for correct rotation...
        # Actually we already computed c_gate for ALL elements in this row.
        # The RoPE needs to rotate c_gate pairs, not c_proj_out pairs.
        # But we just computed c_gate element-by-element and now need partner's c_gate value.
        # Problem: we can't easily get the partner's c_gate since it's computed from the same
        # RMSNorm normalization factor. But c_gate IS fully computed above (before RoPE).
        # We need c_gate[partner] to do RoPE. Since all elements share the same norm factor,
        # and we have all elements in registers, we can use the partner index directly.
        #
        # Actually c_gate is in registers as a vector — we need to access c_gate[partner_offs].
        # In Triton, we can't index into a register vector by another vector.
        # Solution: store c_gate to global memory, then reload partner. Two stores+loads
        # but on same cache line so should be fast.

        # Store pre-RoPE c_gate temporarily
        tl.store(c_gate_ptr + row * P + offs_p, c_gate, mask=p_mask)

        # Reload partner value
        c_gate_partner = tl.load(c_gate_ptr + row * P + partner_offs, mask=p_mask & partner_mask, other=0.0)

        c_gate_rot = tl.where(
            is_odd > 0.5,
            c_gate_partner * sin_a + c_gate * cos_a,
            c_gate * cos_a - c_gate_partner * sin_a,
        )

        tl.store(c_gate_ptr + row * P + offs_p, c_gate_rot, mask=p_mask)

    @triton.jit
    def fused_readout_kernel(
        # Inputs
        h_re_ptr, h_im_ptr,  # (M, P) — scan output
        c_gate_ptr,  # (M, P) — real gate
        C_re_ptr, C_im_ptr,  # (H, P) — static MIMO matrix
        u_ptr,  # (M, H) — original input
        D_ptr,  # (H,) — skip param
        # Output
        out_ptr,  # (M, H)
        # Dims
        M, H, P,
        BLOCK_H: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        """Per (row, h_block): gate h, MIMO readout, skip+silu."""
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        if pid_m >= M:
            return
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = offs_h < H
        offs_p = tl.arange(0, BLOCK_P)
        p_mask = offs_p < P

        # Load h and c_gate for this row (full P in one shot — P<=128 typical)
        h_re = tl.load(h_re_ptr + pid_m * P + offs_p, mask=p_mask, other=0.0)
        h_im = tl.load(h_im_ptr + pid_m * P + offs_p, mask=p_mask, other=0.0)
        gate = tl.load(c_gate_ptr + pid_m * P + offs_p, mask=p_mask, other=0.0)

        # Complex gating: h_gated = h * gate (gate is real)
        hg_re = h_re * gate  # (BLOCK_P,)
        hg_im = h_im * gate  # (BLOCK_P,)

        # MIMO readout: y[h] = sum_p (C_re[h,p]*hg_re[p] - C_im[h,p]*hg_im[p])
        # Load C as (BLOCK_H, BLOCK_P) tile, reduce with elementwise multiply + sum
        C_re_tile = tl.load(
            C_re_ptr + offs_h[:, None] * P + offs_p[None, :],
            mask=h_mask[:, None] & p_mask[None, :], other=0.0
        )
        C_im_tile = tl.load(
            C_im_ptr + offs_h[:, None] * P + offs_p[None, :],
            mask=h_mask[:, None] & p_mask[None, :], other=0.0
        )
        y_vals = tl.sum(C_re_tile * hg_re[None, :] - C_im_tile * hg_im[None, :], axis=1)  # (BLOCK_H,)

        # Skip + silu: out = silu(y + u * D)
        u_vals = tl.load(u_ptr + pid_m * H + offs_h, mask=h_mask, other=0.0)
        d_vals = tl.load(D_ptr + offs_h, mask=h_mask, other=0.0)
        x = y_vals + u_vals * d_vals
        sig = 1.0 / (1.0 + tl.exp(-x))
        out = x * sig

        tl.store(out_ptr + pid_m * H + offs_h, out, mask=h_mask)

    # ========== Fused scan backward kernel ==========
    # Single launch: grid (B, cdiv(P, BLOCK_P)), loops over L in reverse

    @triton.jit
    def fused_scan_bwd_kernel(
        # Forward saved tensors
        alpha_re_ptr, alpha_im_ptr,  # (B, L, P)
        h_re_ptr, h_im_ptr,  # (B, L, P) — forward hidden states
        # Gradient inputs
        d_h_re_ptr, d_h_im_ptr,  # (B, L, P) — grad from postscan
        # Gradient outputs
        d_alpha_re_ptr, d_alpha_im_ptr,  # (B, L, P)
        d_inject_re_ptr, d_inject_im_ptr,  # (B, L, P)
        # Dims
        B_batch, L, P,
        BLOCK_P: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_p = tl.program_id(1)
        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        p_mask = offs_p < P
        base = pid_b * L * P

        # Adjoint state (carried backward through time)
        adj_re = tl.zeros((BLOCK_P,), dtype=tl.float32)
        adj_im = tl.zeros((BLOCK_P,), dtype=tl.float32)

        for t_rev in range(L):
            t = L - 1 - t_rev
            off = base + t * P + offs_p

            # Total gradient at position t
            d_out_re = tl.load(d_h_re_ptr + off, mask=p_mask, other=0.0)
            d_out_im = tl.load(d_h_im_ptr + off, mask=p_mask, other=0.0)
            d_h_re = d_out_re + adj_re
            d_h_im = d_out_im + adj_im

            # d_inject = d_h (direct, since h = alpha*h_prev + inject)
            tl.store(d_inject_re_ptr + off, d_h_re, mask=p_mask)
            tl.store(d_inject_im_ptr + off, d_h_im, mask=p_mask)

            # Load alpha[t]
            a_re = tl.load(alpha_re_ptr + off, mask=p_mask, other=0.0)
            a_im = tl.load(alpha_im_ptr + off, mask=p_mask, other=0.0)

            # Load h[t-1]
            if t > 0:
                prev_off = base + (t - 1) * P + offs_p
                h_prev_re = tl.load(h_re_ptr + prev_off, mask=p_mask, other=0.0)
                h_prev_im = tl.load(h_im_ptr + prev_off, mask=p_mask, other=0.0)
            else:
                h_prev_re = tl.zeros((BLOCK_P,), dtype=tl.float32)
                h_prev_im = tl.zeros((BLOCK_P,), dtype=tl.float32)

            # d_alpha[t] = d_h[t] * conj(h[t-1])
            d_a_re = d_h_re * h_prev_re + d_h_im * h_prev_im
            d_a_im = d_h_im * h_prev_re - d_h_re * h_prev_im
            tl.store(d_alpha_re_ptr + off, d_a_re, mask=p_mask)
            tl.store(d_alpha_im_ptr + off, d_a_im, mask=p_mask)

            # Carry adjoint backward: adj = conj(alpha[t]) * d_h[t]
            adj_re = a_re * d_h_re + a_im * d_h_im
            adj_im = a_re * d_h_im - a_im * d_h_re

    # ========== Fused prescan backward kernel ==========

    @triton.jit
    def fused_prescan_bwd_kernel(
        # Gradient inputs
        d_dt_ptr,  # (M, P) — total gradient on dt
        d_lam_ptr,  # (M, P) — gradient on lam
        d_Bu_ptr,  # (M, P) — gradient on Bu (after norm+bias)
        # Forward saved values
        dt_raw_ptr, lam_raw_ptr,  # (M, P)
        log_dt_bias_ptr,  # (P,)
        Bu_raw_ptr,  # (M, P)
        b_norm_gamma_ptr,  # (P,)
        # Gradient outputs
        d_dt_raw_ptr, d_lam_raw_ptr,  # (M, P)
        d_log_dt_bias_ptr,  # (M, P) — per-row, caller sums to (P,)
        d_Bu_raw_ptr,  # (M, P)
        d_b_norm_gamma_ptr,  # (M, P) — per-row, caller sums to (P,)
        d_b_bias_ptr,  # (M, P) — per-row, caller sums to (P,)
        # Dims
        M, P,
        eps,
        BLOCK_P: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= M:
            return
        offs_p = tl.arange(0, BLOCK_P)
        p_mask = offs_p < P

        # --- dt backward: d_dt -> d_dt_raw, d_log_dt_bias ---
        d_dt = tl.load(d_dt_ptr + row * P + offs_p, mask=p_mask, other=0.0)
        dt_raw = tl.load(dt_raw_ptr + row * P + offs_p, mask=p_mask, other=0.0)
        bias = tl.load(log_dt_bias_ptr + offs_p, mask=p_mask, other=0.0)

        sig_dt = 1.0 / (1.0 + tl.exp(-dt_raw))
        silu_dt = dt_raw * sig_dt
        pre = silu_dt + bias
        # softplus backward: d_pre = d_dt * sigmoid(pre)
        sig_pre = 1.0 / (1.0 + tl.exp(-pre))
        d_pre = d_dt * sig_pre
        # d_log_dt_bias = d_pre
        tl.store(d_log_dt_bias_ptr + row * P + offs_p, d_pre, mask=p_mask)
        # silu backward: d_dt_raw = d_pre * silu'(dt_raw)
        d_dt_raw = d_pre * (sig_dt + dt_raw * sig_dt * (1.0 - sig_dt))
        tl.store(d_dt_raw_ptr + row * P + offs_p, d_dt_raw, mask=p_mask)

        # --- lam backward ---
        d_lam = tl.load(d_lam_ptr + row * P + offs_p, mask=p_mask, other=0.0)
        lam_raw = tl.load(lam_raw_ptr + row * P + offs_p, mask=p_mask, other=0.0)
        lam = 1.0 / (1.0 + tl.exp(-lam_raw))
        d_lam_raw = d_lam * lam * (1.0 - lam)
        tl.store(d_lam_raw_ptr + row * P + offs_p, d_lam_raw, mask=p_mask)

        # --- Bu backward: through bias -> rmsnorm ---
        d_Bu = tl.load(d_Bu_ptr + row * P + offs_p, mask=p_mask, other=0.0)
        # d_b_bias = d_Bu (additive)
        tl.store(d_b_bias_ptr + row * P + offs_p, d_Bu, mask=p_mask)
        # RMSNorm forward: out = gamma * x / sqrt(var + eps)
        Bu_raw = tl.load(Bu_raw_ptr + row * P + offs_p, mask=p_mask, other=0.0)
        gamma = tl.load(b_norm_gamma_ptr + offs_p, mask=p_mask, other=1.0)
        variance = tl.sum(Bu_raw * Bu_raw) / P
        rrms = tl.rsqrt(variance + eps)
        x_hat = Bu_raw * rrms
        tl.store(d_b_norm_gamma_ptr + row * P + offs_p, d_Bu * x_hat, mask=p_mask)
        d_x_hat = d_Bu * gamma
        inner = tl.sum(d_x_hat * x_hat) / P
        d_Bu_raw = rrms * (d_x_hat - x_hat * inner)
        tl.store(d_Bu_raw_ptr + row * P + offs_p, d_Bu_raw, mask=p_mask)

    # ========== Fused backward: readout + cgate (steps 7+6) ==========
    # Grid: (M, cdiv(H, BLOCK_H)) — one program per (row, h_block)
    # Handles: silu bwd on readout, MIMO matmul bwd, c_gate RoPE bwd, rmsnorm bwd, silu bwd
    # Accumulates d_C_re/im per row — caller reduces across M.

    @triton.jit
    def fused_bwd_readout_cgate_kernel(
        # Forward saved
        h_re_ptr, h_im_ptr,  # (M, P)
        c_gate_ptr,  # (M, P)
        c_proj_out_ptr,  # (M, P)
        cum_theta_ptr,  # (M, P//2)
        C_re_ptr, C_im_ptr,  # (H, P)
        u_ptr,  # (M, H)
        D_ptr,  # (H,)
        c_norm_gamma_ptr,  # (P,)
        c_bias_ptr,  # (P,)
        # Gradient input
        d_out_ptr,  # (M, H)
        # Gradient outputs
        d_h_re_ptr, d_h_im_ptr,  # (M, P) — only written by h_block 0
        d_c_proj_out_ptr,  # (M, P) — only written by h_block 0
        d_cum_theta_c_ptr,  # (M, P//2) — only written by h_block 0
        d_u_ptr,  # (M, H) — skip contribution, per h_block
        # Per-row accumulators for reduction (M, H, P) would be too big
        # Instead: d_C_re/im accumulated via atomics (H, P)
        d_C_re_ptr, d_C_im_ptr,  # (H, P)
        d_D_ptr,  # (H,) — atomically accumulated
        d_c_bias_ptr,  # (P,) — atomically accumulated
        d_c_norm_gamma_ptr,  # (P,) — atomically accumulated
        # Dims
        M, H, P,
        # Debug
        dbg_ptr,  # (M, 4) for debug: [max_abs_y, max_abs_xskip, max_abs_dxskip, max_abs_dhgre]
        BLOCK_H: tl.constexpr,
        BLOCK_P: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        if pid_m >= M:
            return
        offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
        h_mask = offs_h < H
        offs_p = tl.arange(0, BLOCK_P)
        p_mask = offs_p < P
        half_P = P // 2

        # Load h, c_gate for this row
        h_re = tl.load(h_re_ptr + pid_m * P + offs_p, mask=p_mask, other=0.0)
        h_im = tl.load(h_im_ptr + pid_m * P + offs_p, mask=p_mask, other=0.0)
        gate = tl.load(c_gate_ptr + pid_m * P + offs_p, mask=p_mask, other=0.0)

        hg_re = h_re * gate
        hg_im = h_im * gate

        # Load C tile (BLOCK_H, P)
        C_re_tile = tl.load(
            C_re_ptr + offs_h[:, None] * P + offs_p[None, :],
            mask=h_mask[:, None] & p_mask[None, :], other=0.0
        )
        C_im_tile = tl.load(
            C_im_ptr + offs_h[:, None] * P + offs_p[None, :],
            mask=h_mask[:, None] & p_mask[None, :], other=0.0
        )

        # Recompute y for silu backward
        y_vals = tl.sum(C_re_tile * hg_re[None, :] - C_im_tile * hg_im[None, :], axis=1)

        # silu backward on (y + u*D)
        u_vals = tl.load(u_ptr + pid_m * H + offs_h, mask=h_mask, other=0.0)
        d_vals = tl.load(D_ptr + offs_h, mask=h_mask, other=0.0)
        x_skip = y_vals + u_vals * d_vals
        sig_skip = 1.0 / (1.0 + tl.exp(-x_skip))
        d_out = tl.load(d_out_ptr + pid_m * H + offs_h, mask=h_mask, other=0.0)
        d_x_skip = d_out * (sig_skip + x_skip * sig_skip * (1.0 - sig_skip))

        # d_u_skip = d_x_skip * D
        d_u_skip = d_x_skip * d_vals
        tl.store(d_u_ptr + pid_m * H + offs_h, d_u_skip, mask=h_mask)

        # d_D: atomic add (d_x_skip * u)
        tl.atomic_add(d_D_ptr + offs_h, d_x_skip * u_vals, mask=h_mask)

        # d_y = d_x_skip (BLOCK_H,)
        d_y = d_x_skip

        # MIMO backward: d_hg_re = C_re^T @ d_y, d_hg_im = -C_im^T @ d_y
        # d_hg_re[p] = sum_h C_re[h,p] * d_y[h]
        d_hg_re = tl.sum(C_re_tile * d_y[:, None], axis=0)  # (BLOCK_P,)
        d_hg_im = -tl.sum(C_im_tile * d_y[:, None], axis=0)  # (BLOCK_P,)

        # DEBUG: write intermediates
        silu_deriv = sig_skip + x_skip * sig_skip * (1.0 - sig_skip)
        if pid_h == 0:
            tl.store(dbg_ptr + pid_m * 8 + 0, tl.max(tl.abs(y_vals), axis=0))
            tl.store(dbg_ptr + pid_m * 8 + 1, tl.max(tl.abs(d_x_skip), axis=0))
            tl.store(dbg_ptr + pid_m * 8 + 2, tl.max(tl.abs(d_hg_re), axis=0))
            tl.store(dbg_ptr + pid_m * 8 + 3, tl.max(tl.abs(d_hg_re * gate), axis=0))
            tl.store(dbg_ptr + pid_m * 8 + 4, tl.max(tl.abs(d_out), axis=0))
            tl.store(dbg_ptr + pid_m * 8 + 5, tl.max(tl.abs(silu_deriv), axis=0))
            tl.store(dbg_ptr + pid_m * 8 + 6, tl.max(tl.abs(x_skip), axis=0))
            tl.store(dbg_ptr + pid_m * 8 + 7, tl.max(tl.abs(sig_skip), axis=0))

        # d_C: atomic add (H, P)
        # d_C_re[h,p] += d_y[h] * hg_re[p]
        tl.atomic_add(
            d_C_re_ptr + offs_h[:, None] * P + offs_p[None, :],
            d_y[:, None] * hg_re[None, :],
            mask=h_mask[:, None] & p_mask[None, :],
        )
        tl.atomic_add(
            d_C_im_ptr + offs_h[:, None] * P + offs_p[None, :],
            -d_y[:, None] * hg_im[None, :],
            mask=h_mask[:, None] & p_mask[None, :],
        )

        # d_h and d_c_gate: only first h_block writes (all h_blocks computed the same d_hg)
        # Actually d_hg depends on C tile — each h_block sees different C rows.
        # We need to SUM d_hg across h_blocks. Use atomics on d_h_re/im.
        # d_h_re = d_hg_re * gate, d_h_im = d_hg_im * gate
        tl.atomic_add(d_h_re_ptr + pid_m * P + offs_p, d_hg_re * gate, mask=p_mask)
        tl.atomic_add(d_h_im_ptr + pid_m * P + offs_p, d_hg_im * gate, mask=p_mask)

        # d_c_gate = d_hg_re * h_re + d_hg_im * h_im (also needs sum across h_blocks)
        d_c_gate_contrib = d_hg_re * h_re + d_hg_im * h_im

        # Only first h_block does the c_gate backward chain (RoPE, rmsnorm, silu)
        # Others just contribute to d_c_gate via atomics on a temp buffer.
        # Actually simpler: accumulate d_c_gate via atomics, then do cgate bwd in separate pass.
        # BUT that adds a launch. Instead: each h_block atomics its d_c_gate contribution,
        # and the LAST h_block (or a separate tiny kernel) does the chain rule.
        #
        # Simplest correct approach: store per-(m, h_block) d_c_gate contributions,
        # then reduce. But that's memory. Let's use atomics on a (M, P) buffer.

        # We'll use a separate small kernel for the c_gate chain. Write d_c_gate atomically.
        # Actually — d_c_proj_out_ptr is (M, P), we can repurpose it as d_c_gate accumulator
        # and then run cgate_bwd separately. This is still only 2 kernels vs 60 PyTorch ops.
        tl.atomic_add(d_c_proj_out_ptr + pid_m * P + offs_p, d_c_gate_contrib, mask=p_mask)

    # ========== Fused backward: c_gate chain rule (RoPE, rmsnorm, silu) ==========
    # Grid: (M,) — one program per row
    # Input: d_c_gate (accumulated from readout bwd) in d_c_proj_out_ptr
    # Output: d_c_proj_out (overwritten), d_cum_theta_c, d_c_bias, d_c_norm_gamma

    @triton.jit
    def fused_bwd_cgate_chain_kernel(
        # d_c_gate is stored in d_c_gate_ptr (M, P) — INPUT
        d_c_gate_ptr,  # (M, P) — accumulated d_c_gate from readout bwd
        # Forward saved
        c_proj_out_ptr,  # (M, P)
        cum_theta_ptr,  # (M, P//2)
        c_norm_gamma_ptr,  # (P,)
        c_bias_ptr,  # (P,)
        # Outputs
        d_c_proj_out_ptr,  # (M, P) — overwrite with actual d_c_proj_out
        d_cum_theta_c_ptr,  # (M, P//2)
        d_c_bias_ptr,  # (P,) — atomic
        d_c_norm_gamma_ptr,  # (P,) — atomic
        # Dims
        M, P,
        eps,
        BLOCK_P: tl.constexpr,
    ):
        row = tl.program_id(0)
        if row >= M:
            return
        offs_p = tl.arange(0, BLOCK_P)
        p_mask = offs_p < P
        half_P = P // 2

        # Load accumulated d_c_gate
        d_cg = tl.load(d_c_gate_ptr + row * P + offs_p, mask=p_mask, other=0.0)

        # Load cum_theta for RoPE backward
        pair_idx = offs_p // 2
        is_odd = (offs_p % 2).to(tl.float32)
        angle = tl.load(cum_theta_ptr + row * half_P + pair_idx, mask=p_mask, other=0.0)
        cos_a = tl.cos(angle)
        sin_a = tl.sin(angle)

        # Recompute c_gate_pre_rope = rmsnorm(silu(c_proj_out)) + c_bias
        c_proj = tl.load(c_proj_out_ptr + row * P + offs_p, mask=p_mask, other=0.0)
        sig_c = 1.0 / (1.0 + tl.exp(-c_proj))
        c_silu = c_proj * sig_c
        variance_c = tl.sum(c_silu * c_silu) / P
        rrms_c = tl.rsqrt(variance_c + eps)
        gamma = tl.load(c_norm_gamma_ptr + offs_p, mask=p_mask, other=1.0)
        c_normed = c_silu * rrms_c * gamma
        c_b = tl.load(c_bias_ptr + offs_p, mask=p_mask, other=0.0)
        cg_pre = c_normed + c_b

        # RoPE backward: d_cg is gradient on rotated output
        # Need partner values of d_cg and cg_pre
        partner_offs = tl.where(is_odd > 0.5, offs_p - 1, offs_p + 1)
        partner_mask = partner_offs < P

        # Store d_cg and cg_pre temporarily for partner access
        tl.store(d_c_proj_out_ptr + row * P + offs_p, d_cg, mask=p_mask)
        d_cg_partner = tl.load(d_c_proj_out_ptr + row * P + partner_offs, mask=p_mask & partner_mask, other=0.0)

        # even: d_x1 = d_o1*cos + d_o2*sin
        # odd:  d_x2 = -d_o1*sin + d_o2*cos
        d_cgpre = tl.where(
            is_odd > 0.5,
            -d_cg_partner * sin_a + d_cg * cos_a,
            d_cg * cos_a + d_cg_partner * sin_a,
        )

        # d_cum_theta from c_gate RoPE
        # Store cg_pre for partner access
        tl.store(d_c_proj_out_ptr + row * P + offs_p, cg_pre, mask=p_mask)
        cg_pre_partner = tl.load(d_c_proj_out_ptr + row * P + partner_offs, mask=p_mask & partner_mask, other=0.0)

        # even elements: cg1 = cg_pre[even], cg2 = cg_pre[odd] = partner
        # odd elements: cg1 = cg_pre[even] = partner, cg2 = cg_pre[odd] = self
        cg1 = tl.where(is_odd > 0.5, cg_pre_partner, cg_pre)
        cg2 = tl.where(is_odd > 0.5, cg_pre, cg_pre_partner)
        d_o1 = tl.where(is_odd > 0.5, d_cg_partner, d_cg)
        d_o2 = tl.where(is_odd > 0.5, d_cg, d_cg_partner)

        # d_theta = d_o1 * (-cg1*sin - cg2*cos) + d_o2 * (cg1*cos - cg2*sin)
        # Only store for even indices (pair_idx based)
        d_ct = d_o1 * (-cg1 * sin_a - cg2 * cos_a) + d_o2 * (cg1 * cos_a - cg2 * sin_a)
        # Only even elements write (to avoid double-writing)
        even_mask = (offs_p % 2 == 0) & p_mask & (pair_idx < half_P)
        tl.store(d_cum_theta_c_ptr + row * half_P + pair_idx, d_ct, mask=even_mask)

        # d_c_bias = d_cgpre (atomic)
        tl.atomic_add(d_c_bias_ptr + offs_p, d_cgpre, mask=p_mask)

        # rmsnorm backward
        c_hat = c_silu * rrms_c
        d_c_normed = d_cgpre
        # d_c_norm_gamma = d_c_normed * c_hat (atomic)
        tl.atomic_add(d_c_norm_gamma_ptr + offs_p, d_c_normed * c_hat, mask=p_mask)
        # d_c_hat = d_c_normed * gamma
        d_c_hat = d_c_normed * gamma
        inner_c = tl.sum(d_c_hat * c_hat) / P
        d_c_silu = rrms_c * (d_c_hat - c_hat * inner_c)

        # silu backward
        d_c_proj_final = d_c_silu * (sig_c + c_proj * sig_c * (1.0 - sig_c))
        tl.store(d_c_proj_out_ptr + row * P + offs_p, d_c_proj_final, mask=p_mask)

    # ========== Fused backward: scan + discretization + RoPE (steps 5+4+RoPE+cumsum) ==========
    # Grid: (B, cdiv(P, BLOCK_P))
    # Loops over L in reverse. RECOMPUTES alpha, inject, Bu_rot from saved dt/lam/Bu/cum_theta/A.
    # This eliminates 5 x (B,L,P) saved tensors.

    @triton.jit
    def fused_bwd_scan_disc_kernel(
        # Forward saved (read-only)
        dt_ptr,  # (B, L, P)
        lam_ptr,  # (B, L, P)
        Bu_ptr,  # (B, L, P) — Bu after norm+bias
        cum_theta_ptr,  # (B, L, P//2)
        A_real_ptr,  # (P,) — negative real parts (-exp(log_A_real))
        A_imag_ptr,  # (P,)
        h_re_ptr, h_im_ptr,  # (B, L, P) — forward hidden states
        theta_ptr,  # (B*L, P//2) — for d_dt_half_theta -> d_dt from rope
        # Gradient input
        d_h_re_ptr, d_h_im_ptr,  # (B, L, P)
        d_cum_theta_c_ptr,  # (B*L, P//2) — from c_gate RoPE backward
        # Gradient outputs
        d_dt_ptr,  # (B, L, P) — total d_dt (from inject + alpha + rope)
        d_lam_ptr,  # (B, L, P)
        d_Bu_ptr,  # (B, L, P) — gradient on Bu (to feed into prescan_bwd)
        d_log_A_real_ptr,  # (P,) — atomic
        d_A_imag_ptr,  # (P,) — atomic
        d_dt_half_theta_ptr,  # (B, L, P//2) — for prescan bwd to handle
        # Dims
        B_batch, L, P,
        BLOCK_P: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_p = tl.program_id(1)
        offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        p_mask = offs_p < P
        base = pid_b * L * P
        half_P = P // 2
        base_theta = pid_b * L * half_P

        # Load A params
        a_re = tl.load(A_real_ptr + offs_p, mask=p_mask, other=0.0)  # negative
        a_im = tl.load(A_imag_ptr + offs_p, mask=p_mask, other=0.0)

        # RoPE helpers
        pair_idx = offs_p // 2
        is_odd = (offs_p % 2).to(tl.float32)

        # Accumulators for A grads
        d_A_real_acc = tl.zeros((BLOCK_P,), dtype=tl.float32)
        d_A_imag_acc = tl.zeros((BLOCK_P,), dtype=tl.float32)

        # Adjoint state (carried backward through time)
        adj_re = tl.zeros((BLOCK_P,), dtype=tl.float32)
        adj_im = tl.zeros((BLOCK_P,), dtype=tl.float32)

        # Running reverse cumsum for d_cum_theta -> d_dt_half_theta
        d_ct_accum = tl.zeros((BLOCK_P,), dtype=tl.float32)  # P-wide but only half used

        # Carry for Bu_prev contribution (from t+1 to t during reverse scan)
        d_Bu_prev_carry = tl.zeros((BLOCK_P,), dtype=tl.float32)

        # ---- FORWARD PASS to build Bu_rot array (needed for bwd) ----
        # We need Bu_rot[t] and Bu_rot[t-1] during backward.
        # Option: recompute forward RoPE inline during backward by caching in SRAM.
        # With L up to 512 and BLOCK_P up to 128, storing all L Bu_rots is too much.
        # Instead: do TWO passes. First forward pass builds Bu_rot into a temp buffer.
        # But that needs memory we're trying to save...
        #
        # Better: recompute Bu_rot[t] and Bu_rot[t-1] on the fly during backward.
        # At each reverse step t, compute Bu_rot[t] from Bu[t] + cum_theta[t],
        # and Bu_prev = Bu_rot[t-1] from Bu[t-1] + cum_theta[t-1].
        # This is 2 RoPE computations per step instead of storing. Worth it for memory.

        for t_rev in range(L):
            t = L - 1 - t_rev
            off = base + t * P + offs_p
            off_theta = base_theta + t * half_P + pair_idx

            # --- Recompute alpha[t], Bu_rot[t], Bu_prev[t], inject[t] ---
            dt_t = tl.load(dt_ptr + off, mask=p_mask, other=0.0)
            lam_t = tl.load(lam_ptr + off, mask=p_mask, other=0.0)
            Bu_t = tl.load(Bu_ptr + off, mask=p_mask, other=0.0)
            angle_t = tl.load(cum_theta_ptr + off_theta, mask=p_mask, other=0.0)

            # RoPE on Bu[t]
            cos_t = tl.cos(angle_t)
            sin_t = tl.sin(angle_t)
            partner_p = tl.where(is_odd > 0.5, offs_p - 1, offs_p + 1)
            partner_off_t = base + t * P + partner_p
            partner_mask_p = partner_p < P
            Bu_partner_t = tl.load(Bu_ptr + partner_off_t, mask=p_mask & partner_mask_p, other=0.0)
            Bu_rot_t = tl.where(
                is_odd > 0.5,
                Bu_partner_t * sin_t + Bu_t * cos_t,
                Bu_t * cos_t - Bu_partner_t * sin_t,
            )

            # Bu_prev = Bu_rot[t-1] (zero if t==0)
            Bu_prev_re = tl.zeros((BLOCK_P,), dtype=tl.float32)
            if t > 0:
                off_prev = base + (t - 1) * P + offs_p
                off_theta_prev = base_theta + (t - 1) * half_P + pair_idx
                Bu_prev_raw = tl.load(Bu_ptr + off_prev, mask=p_mask, other=0.0)
                angle_prev = tl.load(cum_theta_ptr + off_theta_prev, mask=p_mask, other=0.0)
                cos_prev = tl.cos(angle_prev)
                sin_prev = tl.sin(angle_prev)
                partner_off_prev = base + (t - 1) * P + partner_p
                Bu_partner_prev = tl.load(Bu_ptr + partner_off_prev, mask=p_mask & partner_mask_p, other=0.0)
                Bu_prev_re = tl.where(
                    is_odd > 0.5,
                    Bu_partner_prev * sin_prev + Bu_prev_raw * cos_prev,
                    Bu_prev_raw * cos_prev - Bu_partner_prev * sin_prev,
                )

            # alpha = exp(dt * A)
            dt_a_re = dt_t * a_re
            dt_a_im = dt_t * a_im
            exp_re = tl.exp(dt_a_re)
            alpha_re_t = exp_re * tl.cos(dt_a_im)
            alpha_im_t = exp_re * tl.sin(dt_a_im)

            # --- Scan backward: adjoint propagation ---
            d_out_re = tl.load(d_h_re_ptr + off, mask=p_mask, other=0.0)
            d_out_im = tl.load(d_h_im_ptr + off, mask=p_mask, other=0.0)
            d_h_re_t = d_out_re + adj_re
            d_h_im_t = d_out_im + adj_im

            # d_inject = d_h
            d_inj_re = d_h_re_t
            d_inj_im = d_h_im_t

            # d_alpha from recurrence: h = alpha*h_prev + inject
            if t > 0:
                prev_off = base + (t - 1) * P + offs_p
                h_prev_re = tl.load(h_re_ptr + prev_off, mask=p_mask, other=0.0)
                h_prev_im = tl.load(h_im_ptr + prev_off, mask=p_mask, other=0.0)
            else:
                h_prev_re = tl.zeros((BLOCK_P,), dtype=tl.float32)
                h_prev_im = tl.zeros((BLOCK_P,), dtype=tl.float32)

            d_a_re = d_h_re_t * h_prev_re + d_h_im_t * h_prev_im
            d_a_im = d_h_im_t * h_prev_re - d_h_re_t * h_prev_im

            # Carry adjoint: adj = conj(alpha) * d_h
            adj_re = alpha_re_t * d_h_re_t + alpha_im_t * d_h_im_t
            adj_im = alpha_re_t * d_h_im_t - alpha_im_t * d_h_re_t

            # --- Discretization backward ---
            one_minus_lam = 1.0 - lam_t

            # d_lam from inject
            d_lam_t = (d_inj_re * dt_t * (Bu_rot_t - alpha_re_t * Bu_prev_re)
                       + d_inj_im * (-dt_t * alpha_im_t * Bu_prev_re))
            tl.store(d_lam_ptr + off, d_lam_t, mask=p_mask)

            # d_Bu_rot from inject (current position)
            d_Bu_rot_t = d_inj_re * lam_t * dt_t

            # d_Bu_prev from inject
            d_Bu_prev_t = ((d_inj_re * alpha_re_t + d_inj_im * alpha_im_t)
                           * one_minus_lam * dt_t)

            # d_dt from inject
            d_dt_from_inject = (d_inj_re * (lam_t * Bu_rot_t + one_minus_lam * alpha_re_t * Bu_prev_re)
                                + d_inj_im * one_minus_lam * alpha_im_t * Bu_prev_re)

            # d_alpha from inject (add to scan's d_alpha)
            d_a_re_total = d_a_re + d_inj_re * one_minus_lam * dt_t * Bu_prev_re
            d_a_im_total = d_a_im + d_inj_im * one_minus_lam * dt_t * Bu_prev_re

            # d_dt from alpha: alpha = exp(dt*A), d_dt = d_alpha * alpha * A
            d_dt_from_alpha = (d_a_re_total * (a_re * alpha_re_t - a_im * alpha_im_t)
                               + d_a_im_total * (a_re * alpha_im_t + a_im * alpha_re_t))

            d_dt_disc = d_dt_from_inject + d_dt_from_alpha

            # Accumulate A grads
            d_A_real_acc += d_a_re_total * dt_t * alpha_re_t + d_a_im_total * dt_t * alpha_im_t
            d_A_imag_acc += -d_a_re_total * dt_t * alpha_im_t + d_a_im_total * dt_t * alpha_re_t

            # --- Bu RoPE backward ---
            # d_Bu_rot_t needs to include contribution from NEXT position's d_Bu_prev
            # We handle this by loading d_Bu_prev from next step.
            # But we compute d_Bu_prev[t] above which applies to Bu_rot[t-1].
            # So at step t, we have d_Bu_rot[t] from current inject, and we need to
            # ADD d_Bu_prev[t+1] which was computed at step t+1 (the previous iteration).
            # Store d_Bu_rot[t] now; at t-1 we'll add d_Bu_prev[t] to d_Bu_rot[t-1].
            #
            # Actually simpler: store d_Bu_prev[t] into d_Bu output for position t-1.
            # We do this at the NEXT reverse iteration when t_rev processes t-1.
            # For now, store d_Bu_rot[t] into d_Bu buffer for position t.
            # After the loop, shift d_Bu_prev contributions.
            #
            # Cleanest approach: write d_Bu_rot[t] and d_Bu_prev_for_tminus1[t] separately,
            # then combine. But that doubles memory.
            #
            # Instead: at each step, we also apply Bu_prev contribution from previous iteration.
            # Keep d_Bu_prev_carry in registers.

            # Actually let me just do it right:
            # At reverse step processing time t:
            # - We compute d_Bu_rot[t] from inject at t
            # - We have d_Bu_prev_carry from the previous reverse step (which was t+1)
            #   This is the d_Bu_prev[t+1] = contribution to Bu_rot[t]
            # - Total d_Bu_rot[t] = d_Bu_rot_t + d_Bu_prev_carry

            d_Bu_rot_total = d_Bu_rot_t + d_Bu_prev_carry

            # Update carry for next iteration (this position's d_Bu_prev goes to t-1)
            d_Bu_prev_carry = d_Bu_prev_t

            # RoPE backward on Bu: d_Bu from d_Bu_rot_total
            # forward: Bu_rot = RoPE(Bu, angle)
            # bwd: d_Bu_even = d_rot_even*cos + d_rot_odd*sin
            #      d_Bu_odd = -d_rot_even*sin + d_rot_odd*cos
            # Need partner of d_Bu_rot_total
            # Store temporarily for partner access
            tl.store(d_Bu_ptr + off, d_Bu_rot_total, mask=p_mask)
            d_br_partner = tl.load(d_Bu_ptr + (base + t * P + partner_p),
                                   mask=p_mask & partner_mask_p, other=0.0)

            d_Bu_final = tl.where(
                is_odd > 0.5,
                -d_br_partner * sin_t + d_Bu_rot_total * cos_t,
                d_Bu_rot_total * cos_t + d_br_partner * sin_t,
            )
            tl.store(d_Bu_ptr + off, d_Bu_final, mask=p_mask)

            # d_cum_theta from Bu RoPE
            bu1 = tl.where(is_odd > 0.5, Bu_partner_t, Bu_t)
            bu2 = tl.where(is_odd > 0.5, Bu_t, Bu_partner_t)
            d_rot1 = tl.where(is_odd > 0.5, d_br_partner, d_Bu_rot_total)
            d_rot2 = tl.where(is_odd > 0.5, d_Bu_rot_total, d_br_partner)
            d_ct_b = d_rot1 * (-bu1 * sin_t - bu2 * cos_t) + d_rot2 * (bu1 * cos_t - bu2 * sin_t)

            # d_dt from rope: need reverse cumsum of d_cum_theta * theta -> d_dt_half
            # Accumulate reverse cumsum inline (includes c_gate and Bu RoPE contributions)
            even_mask_p = (offs_p % 2 == 0) & p_mask & (pair_idx < half_P)
            # Load d_cum_theta_c for this position
            row_idx_ct = pid_b * L + t
            d_ct_c = tl.load(d_cum_theta_c_ptr + row_idx_ct * half_P + pair_idx, mask=even_mask_p, other=0.0)
            # Only even elements carry the pair's d_ct (avoid double)
            d_ct_accum += tl.where(offs_p % 2 == 0, d_ct_b + d_ct_c, 0.0)

            # d_dt_half_theta[t] = d_ct_accum (this IS the reverse cumsum)
            tl.store(d_dt_half_theta_ptr + off_theta, d_ct_accum, mask=even_mask_p)

            # d_dt from rope: load the stored value back (even wrote, both read via pair_idx)
            row_idx = pid_b * L + t
            theta_t = tl.load(theta_ptr + row_idx * half_P + pair_idx, mask=p_mask & (pair_idx < half_P), other=0.0)
            d_ct_val = tl.load(d_dt_half_theta_ptr + off_theta, mask=p_mask & (pair_idx < half_P), other=0.0)
            d_dt_rope_val = d_ct_val * theta_t * 0.5  # both even/odd get same value
            d_dt_total_t = d_dt_disc + d_dt_rope_val
            tl.store(d_dt_ptr + off, d_dt_total_t, mask=p_mask)

        # Store A grads (atomic across batch and p-blocks)
        # d_log_A_real = d_A_real * A_real_neg = d_A_real_acc * a_re (since a_re = -exp(log_A_real) = A_real_neg)
        tl.atomic_add(d_log_A_real_ptr + offs_p, d_A_real_acc * a_re, mask=p_mask)
        tl.atomic_add(d_A_imag_ptr + offs_p, d_A_imag_acc, mask=p_mask)


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


# ========== autograd.Function ==========

class _TritonS6(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                u,  # (B, L, H)
                Bu_raw,  # (B, L, P) — from feature bank
                x_proj_w, x_proj_b,
                b_norm_gamma, b_bias, log_dt_bias,
                log_A_real, A_imag,
                c_proj_w, c_proj_b,
                c_norm_gamma, c_bias,
                C_re, C_im,
                D,
                P, chunk_size):
        B, L, H = u.shape
        M = B * L
        split_sizes = [P, P, P // 2]
        u_flat = u.reshape(M, H)

        # 1. x_proj linear: (M, H) -> (M, P+P+P//2)
        x_proj_out = triton_linear(u_flat, x_proj_w, x_proj_b)
        dt_raw, lam_raw, theta = x_proj_out.split(split_sizes, dim=-1)
        dt_raw = dt_raw.contiguous()
        lam_raw = lam_raw.contiguous()
        theta = theta.contiguous()

        # 2. Fused prescan: dt/lam activations, Bu norm+bias, dt_half*theta
        Bu_raw_flat = Bu_raw.reshape(M, P).contiguous()
        dt = torch.empty(M, P, device=u.device, dtype=u.dtype)
        lam = torch.empty(M, P, device=u.device, dtype=u.dtype)
        Bu = torch.empty(M, P, device=u.device, dtype=u.dtype)
        dt_half_theta = torch.empty(M, P // 2, device=u.device, dtype=u.dtype)

        BLOCK_P = triton.next_power_of_2(P)
        fused_prescan_kernel[(M,)](
            dt_raw, lam_raw, theta,
            Bu_raw_flat,
            log_dt_bias, b_norm_gamma, b_bias,
            dt, lam, Bu, dt_half_theta,
            M, P, 1e-6,
            BLOCK_P,
        )

        # 3. Cumulative theta (PyTorch — one launch)
        cum_theta = torch.cumsum(dt_half_theta.view(B, L, P // 2), dim=1).contiguous()

        # 4. Fused scan: RoPE + discretize + recurrence
        h_re = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        h_im = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        # alpha/inject/Bu_rot no longer saved — recomputed in backward
        alpha_re = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        alpha_im = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        inject_re = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        inject_im = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        Bu_rot = torch.empty(B, L, P, device=u.device, dtype=u.dtype)

        A_real_neg = -torch.exp(log_A_real)  # (P,)
        BLOCK_P_SCAN = triton.next_power_of_2(P)

        fused_scan_kernel[(B, triton.cdiv(P, BLOCK_P_SCAN))](
            dt.view(B, L, P).contiguous(), lam.view(B, L, P).contiguous(),
            Bu.view(B, L, P).contiguous(), cum_theta,
            A_real_neg, A_imag,
            h_re, h_im, alpha_re, alpha_im, inject_re, inject_im, Bu_rot,
            B, L, P,
            BLOCK_P_SCAN,
        )

        # 5. c_proj linear: (M, H) -> (M, P)
        c_proj_out = triton_linear(u_flat, c_proj_w, c_proj_b)

        # 6. Fused c_gate: silu + rmsnorm + bias + RoPE
        c_gate = torch.empty(M, P, device=u.device, dtype=u.dtype)
        fused_cgate_kernel[(M,)](
            c_proj_out, cum_theta.view(M, P // 2),
            c_norm_gamma, c_bias,
            c_gate,
            M, P, B, L, 1e-6,
            BLOCK_P,
        )

        # 7. Fused readout: gate h, MIMO, skip+silu
        BLOCK_H = min(64, triton.next_power_of_2(H))
        out = torch.empty(B, L, H, device=u.device, dtype=u.dtype)

        fused_readout_kernel[(M, triton.cdiv(H, BLOCK_H))](
            h_re.view(M, P), h_im.view(M, P),
            c_gate,
            C_re, C_im,
            u_flat, D,
            out.view(M, H),
            M, H, P,
            BLOCK_H, BLOCK_P,
        )

        # Save for backward — no alpha/inject/Bu_rot (recomputed in bwd)
        ctx.save_for_backward(
            u, Bu_raw_flat,
            dt_raw, dt, lam_raw, lam, theta, dt_half_theta, cum_theta,
            Bu,
            h_re, h_im,
            c_proj_out, c_gate,
            x_proj_w, c_proj_w, b_norm_gamma, b_bias, log_dt_bias, log_A_real, A_imag,
            c_norm_gamma, c_bias, C_re, C_im, D,
        )
        ctx.P = P
        ctx.chunk_size = chunk_size
        ctx.split_sizes = split_sizes

        return out

    @staticmethod
    def backward(ctx, d_out):
        (u, Bu_raw_flat,
         dt_raw, dt, lam_raw, lam, theta, dt_half_theta, cum_theta,
         Bu,
         h_re, h_im,
         c_proj_out, c_gate,
         x_proj_w, c_proj_w, b_norm_gamma, b_bias, log_dt_bias, log_A_real, A_imag,
         c_norm_gamma, c_bias, C_re, C_im, D,
        ) = ctx.saved_tensors

        P = ctx.P
        split_sizes = ctx.split_sizes
        B, L, H = u.shape
        M = B * L
        BLOCK_P = triton.next_power_of_2(P)

        u_flat = u.reshape(M, H)
        A_real_neg = -torch.exp(log_A_real)

        # ---- Steps 7+6: Fused readout + c_gate backward ----
        BLOCK_H = min(64, triton.next_power_of_2(H))
        d_h_re = torch.zeros(M, P, device=u.device, dtype=u.dtype)
        d_h_im = torch.zeros(M, P, device=u.device, dtype=u.dtype)
        d_c_gate_buf = torch.zeros(M, P, device=u.device, dtype=u.dtype)
        d_u_skip = torch.empty(M, H, device=u.device, dtype=u.dtype)
        d_C_re = torch.zeros(H, P, device=u.device, dtype=u.dtype)
        d_C_im = torch.zeros(H, P, device=u.device, dtype=u.dtype)
        d_D = torch.zeros(H, device=u.device, dtype=u.dtype)
        _dbg = torch.zeros(M, 8, device=u.device, dtype=u.dtype)

        fused_bwd_readout_cgate_kernel[(M, triton.cdiv(H, BLOCK_H))](
            h_re.view(M, P), h_im.view(M, P),
            c_gate, c_proj_out,
            cum_theta.view(M, P // 2),
            C_re, C_im, u_flat, D,
            c_norm_gamma, c_bias,
            d_out.view(M, H),
            d_h_re, d_h_im,
            d_c_gate_buf,
            torch.empty(M, P // 2, device=u.device, dtype=u.dtype),
            d_u_skip,
            d_C_re, d_C_im, d_D,
            torch.empty(P, device=u.device, dtype=u.dtype),
            torch.empty(P, device=u.device, dtype=u.dtype),
            M, H, P,
            _dbg,
            BLOCK_H, BLOCK_P,
        )

        # DEBUG: print debug info for NaN rows
        if d_h_re.isnan().any():
            nan_rows = d_h_re.isnan().any(dim=1).nonzero(as_tuple=False).squeeze()
            for r in nan_rows[:3]:
                r = r.item()
                print(f"Row {r}: max_y={_dbg[r,0]:.4f} max_dxskip={_dbg[r,1]:.4f} max_dhgre={_dbg[r,2]:.4f} max_dhre={_dbg[r,3]:.4f}")
                print(f"  d_h_re[{r}]={d_h_re[r]}")
                print(f"  h_re[{r}] range=[{h_re.view(M,P)[r].min():.4f}, {h_re.view(M,P)[r].max():.4f}]")
                print(f"  gate[{r}] range=[{c_gate[r].min():.4f}, {c_gate[r].max():.4f}]")

        # c_gate chain rule
        d_c_proj_out = torch.empty(M, P, device=u.device, dtype=u.dtype)
        d_cum_theta_c = torch.empty(M, P // 2, device=u.device, dtype=u.dtype)
        d_c_bias = torch.zeros(P, device=u.device, dtype=u.dtype)
        d_c_norm_gamma = torch.zeros(P, device=u.device, dtype=u.dtype)

        fused_bwd_cgate_chain_kernel[(M,)](
            d_c_gate_buf,
            c_proj_out, cum_theta.view(M, P // 2),
            c_norm_gamma, c_bias,
            d_c_proj_out, d_cum_theta_c,
            d_c_bias, d_c_norm_gamma,
            M, P, 1e-6,
            BLOCK_P,
        )

        # c_proj linear backward
        d_c_proj_w = d_c_proj_out.t() @ u_flat
        d_c_proj_b = d_c_proj_out.sum(0)
        d_u_c = d_c_proj_out @ c_proj_w

        # ---- Steps 5+4+RoPE+cumsum: Fused scan + disc + RoPE backward ----
        BLOCK_P_SCAN = triton.next_power_of_2(P)
        d_dt_total = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        d_lam_disc = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        d_Bu_3d = torch.empty(B, L, P, device=u.device, dtype=u.dtype)
        d_log_A_real = torch.zeros(P, device=u.device, dtype=u.dtype)
        d_A_imag = torch.zeros(P, device=u.device, dtype=u.dtype)
        d_dt_half_theta = torch.empty(B, L, P // 2, device=u.device, dtype=u.dtype)

        fused_bwd_scan_disc_kernel[(B, triton.cdiv(P, BLOCK_P_SCAN))](
            dt.view(B, L, P), lam.view(B, L, P),
            Bu.view(B, L, P), cum_theta,
            A_real_neg, A_imag,
            h_re, h_im,
            theta.view(M, P // 2),
            d_h_re.view(B, L, P), d_h_im.view(B, L, P),
            d_cum_theta_c,
            d_dt_total, d_lam_disc, d_Bu_3d,
            d_log_A_real, d_A_imag,
            d_dt_half_theta,
            B, L, P,
            BLOCK_P_SCAN,
        )

        # DEBUG: check for NaN after scan_disc kernel
        assert not d_dt_total.isnan().any(), "NaN in d_dt_total after scan_disc"
        assert not d_lam_disc.isnan().any(), "NaN in d_lam_disc after scan_disc"
        assert not d_Bu_3d.isnan().any(), "NaN in d_Bu_3d after scan_disc"
        assert not d_log_A_real.isnan().any(), "NaN in d_log_A_real after scan_disc"
        assert not d_A_imag.isnan().any(), "NaN in d_A_imag after scan_disc"
        assert not d_dt_half_theta.isnan().any(), "NaN in d_dt_half_theta after scan_disc"

        # d_theta from d_dt_half_theta: d_theta[t] = d_dt_half_theta[t] * dt_half[t]
        dt_half = dt.view(B, L, P // 2, 2).mean(-1)
        d_theta = (d_dt_half_theta * dt_half).view(M, P // 2)

        # ---- 2. Prescan backward (dt/lam activations, Bu rmsnorm) ----
        d_dt_raw = torch.empty(M, P, device=u.device, dtype=u.dtype)
        d_lam_raw = torch.empty(M, P, device=u.device, dtype=u.dtype)
        d_log_dt_bias_rows = torch.empty(M, P, device=u.device, dtype=u.dtype)
        d_Bu_raw = torch.empty(M, P, device=u.device, dtype=u.dtype)
        d_b_norm_gamma_rows = torch.empty(M, P, device=u.device, dtype=u.dtype)
        d_b_bias_rows = torch.empty(M, P, device=u.device, dtype=u.dtype)

        fused_prescan_bwd_kernel[(M,)](
            d_dt_total.reshape(M, P), d_lam_disc.reshape(M, P).contiguous(),
            d_Bu_3d.reshape(M, P).contiguous(),
            dt_raw, lam_raw, log_dt_bias,
            Bu_raw_flat, b_norm_gamma,
            d_dt_raw, d_lam_raw, d_log_dt_bias_rows,
            d_Bu_raw, d_b_norm_gamma_rows, d_b_bias_rows,
            M, P, 1e-6,
            BLOCK_P,
        )

        d_log_dt_bias = d_log_dt_bias_rows.sum(0)
        d_b_norm_gamma = d_b_norm_gamma_rows.sum(0)
        d_b_bias = d_b_bias_rows.sum(0)

        # ---- 1. x_proj linear backward ----
        d_x_proj = torch.cat([d_dt_raw, d_lam_raw, d_theta], dim=-1)
        d_x_proj_w = d_x_proj.t() @ u_flat
        d_x_proj_b = d_x_proj.sum(0)
        d_u_proj = d_x_proj @ x_proj_w

        # Total d_u
        d_u = (d_u_proj + d_u_c + d_u_skip).view(B, L, H)

        d_Bu_raw_out = d_Bu_raw.view(B, L, P)

        return (d_u,
                d_Bu_raw_out,
                d_x_proj_w, d_x_proj_b,
                d_b_norm_gamma, d_b_bias, d_log_dt_bias,
                d_log_A_real, d_A_imag,
                d_c_proj_w, d_c_proj_b,
                d_c_norm_gamma, d_c_bias,
                d_C_re, d_C_im,
                d_D,
                None, None)


# ========== nn.Module wrapper ==========

class TritonS6(nn.Module):
    """S6 using Triton kernels. Falls back to PyTorch S6 on non-CUDA devices."""

    def __init__(self, d_model, d_state=64, M=4, chunk_size=32, layer_idx=None, **kwargs):
        super().__init__()
        from .s6 import S6

        H = d_model
        P = d_state
        self.H = H
        self.P = P
        self.chunk_size = chunk_size

        self._pytorch_s6 = S6(d_model, d_state, M=M, layer_idx=layer_idx, **kwargs)

    def forward(self, u, **kwargs):
        if not u.is_cuda or not HAS_TRITON:
            return self._pytorch_s6(u, **kwargs)

        s6 = self._pytorch_s6
        kern = s6.kernel

        # Feature bank in PyTorch (small MLP)
        Bu_raw = kern.phi_B(u.unsqueeze(1)).squeeze(1)  # (B, L, P)

        # Split complex C
        C = torch.view_as_complex(s6.C)
        C_re = C.real.contiguous()
        C_im = C.imag.contiguous()

        return _TritonS6.apply(
            u,
            Bu_raw,
            kern.x_proj.weight, kern.x_proj.bias,
            kern.b_norm.weight, kern.b_bias, kern.log_dt_bias,
            kern.log_A_real, kern.A_imag,
            s6.c_proj.weight, s6.c_proj.bias,
            s6.c_norm.weight, s6.c_bias,
            C_re, C_im,
            s6.D,
            self.P, self.chunk_size,
        )
