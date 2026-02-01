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
        LOG2E_VAL: tl.constexpr,
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
            # Using exp2 trick: exp(x) = exp2(x * log2(e))
            dt_a_re = dt * a_re  # dt * A_real (A_real is negative)
            dt_a_im = dt * a_im

            exp_re = tl.exp2(dt_a_re * LOG2E_VAL)
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
        # d_Bu_normed = d_Bu (bias is additive, gamma is in rmsnorm)
        # RMSNorm forward: out = gamma * x / sqrt(var + eps)
        Bu_raw = tl.load(Bu_raw_ptr + row * P + offs_p, mask=p_mask, other=0.0)
        gamma = tl.load(b_norm_gamma_ptr + offs_p, mask=p_mask, other=1.0)
        variance = tl.sum(Bu_raw * Bu_raw) / P
        rrms = tl.rsqrt(variance + eps)
        x_hat = Bu_raw * rrms
        # d_gamma = d_Bu * x_hat (per-element)
        tl.store(d_b_norm_gamma_ptr + row * P + offs_p, d_Bu * x_hat, mask=p_mask)
        # d_x_hat = d_Bu * gamma
        d_x_hat = d_Bu * gamma
        # d_Bu_raw = rrms * (d_x_hat - x_hat * mean(d_x_hat * x_hat))
        inner = tl.sum(d_x_hat * x_hat) / P
        d_Bu_raw = rrms * (d_x_hat - x_hat * inner)
        tl.store(d_Bu_raw_ptr + row * P + offs_p, d_Bu_raw, mask=p_mask)


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
            LOG2E,
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

        # Save for backward
        ctx.save_for_backward(
            u, Bu_raw_flat,
            dt_raw, dt, lam_raw, lam, theta, dt_half_theta, cum_theta,
            Bu, Bu_rot,
            alpha_re, alpha_im, inject_re, inject_im,
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
         Bu, Bu_rot,
         alpha_re, alpha_im, inject_re, inject_im,
         h_re, h_im,
         c_proj_out, c_gate,
         x_proj_w, c_proj_w, b_norm_gamma, b_bias, log_dt_bias, log_A_real, A_imag,
         c_norm_gamma, c_bias, C_re, C_im, D,
        ) = ctx.saved_tensors

        P = ctx.P
        split_sizes = ctx.split_sizes
        B, L, H = u.shape
        M = B * L
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        BLOCK_P = triton.next_power_of_2(P)

        u_flat = u.reshape(M, H)

        # ---- 7. Backward through readout: out = silu(y + u*D) where y = Re(C @ (h*gate)) ----
        # Use PyTorch for this since it's a mix of matmul + elementwise
        h_re_flat = h_re.view(M, P)
        h_im_flat = h_im.view(M, P)
        c_gate_flat = c_gate

        hg_re = h_re_flat * c_gate_flat
        hg_im = h_im_flat * c_gate_flat

        # Recompute y for backward
        y_re = hg_re @ C_re.t()  # (M, H)
        y_im = hg_im @ C_im.t()  # (M, H)
        y = y_re - y_im

        # silu backward: out = silu(y + u*D)
        x_skip = y + u_flat * D
        sig_skip = torch.sigmoid(x_skip)
        d_x_skip = d_out.view(M, H) * (sig_skip + x_skip * sig_skip * (1 - sig_skip))

        d_y = d_x_skip
        d_u_skip = d_x_skip * D
        d_D = (d_x_skip * u_flat).sum(0)

        # y = C_re @ hg_re - C_im @ hg_im
        d_hg_re = d_y @ C_re  # (M, P)
        d_hg_im = -(d_y @ C_im)  # (M, P)
        d_C_re = d_y.t() @ hg_re  # (H, P)
        d_C_im = -(d_y.t() @ hg_im)  # (H, P)

        # hg = h * gate
        d_h_re = d_hg_re * c_gate_flat
        d_h_im = d_hg_im * c_gate_flat
        d_c_gate = d_hg_re * h_re_flat + d_hg_im * h_im_flat

        # ---- 6. Backward through c_gate: RoPE + rmsnorm + silu + bias ----
        # c_gate = RoPE(rmsnorm(silu(c_proj_out)) + c_bias, cum_theta)
        # Backward through RoPE
        cum_theta_flat = cum_theta.view(M, P // 2)
        from .s6 import apply_rotary_emb  # reuse PyTorch RoPE for backward

        # Recompute pre-RoPE c_gate for backward
        c_silu = F.silu(c_proj_out)
        c_var = c_silu.pow(2).mean(-1, keepdim=True)
        c_normed = c_silu / torch.sqrt(c_var + 1e-6) * c_norm_gamma
        c_gate_pre_rope = c_normed + c_bias

        # RoPE backward (PyTorch)
        # d_c_gate is gradient on rotated output
        # forward: c_gate_rot = apply_rotary_emb(c_gate_pre_rope, cum_theta)
        # backward: d_c_gate_pre_rope = apply_rotary_emb(d_c_gate, -cum_theta)  (rotate back)
        # d_cum_theta: need chain rule through angles
        cos_ct = torch.cos(cum_theta_flat)
        sin_ct = torch.sin(cum_theta_flat)

        d_cg = d_c_gate.view(M, P)
        d_cg1, d_cg2 = d_cg[..., 0::2], d_cg[..., 1::2]
        cg1, cg2 = c_gate_pre_rope[..., 0::2], c_gate_pre_rope[..., 1::2]

        # RoPE bwd: d_x1 = d_o1*cos + d_o2*sin, d_x2 = -d_o1*sin + d_o2*cos
        d_cgpre = torch.empty_like(c_gate_pre_rope)
        d_cgpre[..., 0::2] = d_cg1 * cos_ct + d_cg2 * sin_ct
        d_cgpre[..., 1::2] = -d_cg1 * sin_ct + d_cg2 * cos_ct

        # d_cum_theta from c_gate RoPE
        d_cum_theta_c = (d_cg1 * (-cg1 * sin_ct - cg2 * cos_ct) +
                         d_cg2 * (cg1 * cos_ct - cg2 * sin_ct))

        d_c_bias = d_cgpre.sum(0)

        # rmsnorm backward
        d_c_normed = d_cgpre
        rrms_c = torch.rsqrt(c_var + 1e-6)
        c_hat = c_silu * rrms_c
        d_c_norm_gamma = (d_c_normed * c_hat).sum(0)
        d_c_hat = d_c_normed * c_norm_gamma
        inner_c = (d_c_hat * c_hat).sum(-1, keepdim=True) / P
        d_c_silu = rrms_c * (d_c_hat - c_hat * inner_c)

        # silu backward
        sig_c = torch.sigmoid(c_proj_out)
        d_c_proj_out = d_c_silu * (sig_c + c_proj_out * sig_c * (1 - sig_c))

        # c_proj linear backward (PyTorch for simplicity)
        d_c_proj_w = d_c_proj_out.t() @ u_flat
        d_c_proj_b = d_c_proj_out.sum(0)
        d_u_c = d_c_proj_out @ c_proj_w

        # ---- 5. Backward through scan (fused kernel) ----
        d_h_re_3d = d_h_re.view(B, L, P).contiguous()
        d_h_im_3d = d_h_im.view(B, L, P).contiguous()

        d_alpha_re = torch.empty_like(alpha_re)
        d_alpha_im = torch.empty_like(alpha_im)
        d_inject_re = torch.empty_like(inject_re)
        d_inject_im = torch.empty_like(inject_im)

        BLOCK_P_SCAN = triton.next_power_of_2(P)
        fused_scan_bwd_kernel[(B, triton.cdiv(P, BLOCK_P_SCAN))](
            alpha_re, alpha_im, h_re, h_im,
            d_h_re_3d, d_h_im_3d,
            d_alpha_re, d_alpha_im, d_inject_re, d_inject_im,
            B, L, P,
            BLOCK_P_SCAN,
        )

        # ---- 4. Backward through discretization ----
        # inject_re = lam*dt*Bu_rot + (1-lam)*dt*alpha_re*Bu_prev_re
        # inject_im = (1-lam)*dt*alpha_im*Bu_prev_re
        # where Bu_prev_re[:, t] = Bu_rot[:, t-1] (shifted)
        dt_3d = dt.view(B, L, P)
        lam_3d = lam.view(B, L, P)
        Bu_rot_3d = Bu_rot  # (B, L, P)
        Bu_prev = torch.zeros_like(Bu_rot_3d)
        Bu_prev[:, 1:] = Bu_rot_3d[:, :-1]

        A_real_neg = -torch.exp(log_A_real)

        # d_lam
        d_lam_disc = (d_inject_re * dt_3d * (Bu_rot_3d - alpha_re * Bu_prev)
                      + d_inject_im * (-dt_3d * alpha_im * Bu_prev))

        # d_Bu_rot
        d_Bu_rot = d_inject_re * lam_3d * dt_3d

        # d_Bu_prev
        d_Bu_prev = ((d_inject_re * alpha_re + d_inject_im * alpha_im)
                     * (1 - lam_3d) * dt_3d)

        # Reverse shift: d_Bu_rot[:, :-1] += d_Bu_prev[:, 1:]
        d_Bu_rot[:, :-1] += d_Bu_prev[:, 1:]

        # d_dt from inject
        one_minus_lam = 1 - lam_3d
        d_dt_from_inject = (d_inject_re * (lam_3d * Bu_rot_3d + one_minus_lam * alpha_re * Bu_prev)
                            + d_inject_im * one_minus_lam * alpha_im * Bu_prev)

        # d_alpha from inject
        d_alpha_re_total = d_alpha_re + d_inject_re * one_minus_lam * dt_3d * Bu_prev
        d_alpha_im_total = d_alpha_im + d_inject_im * one_minus_lam * dt_3d * Bu_prev

        # d_dt from alpha: alpha = exp(dt * A)
        d_dt_from_alpha = (d_alpha_re_total * (A_real_neg * alpha_re - A_imag * alpha_im)
                           + d_alpha_im_total * (A_real_neg * alpha_im + A_imag * alpha_re))

        d_dt_disc = d_dt_from_inject + d_dt_from_alpha

        # d_A grads
        d_A_real = (d_alpha_re_total * dt_3d * alpha_re + d_alpha_im_total * dt_3d * alpha_im).sum(dim=(0, 1))
        d_A_imag = (-d_alpha_re_total * dt_3d * alpha_im + d_alpha_im_total * dt_3d * alpha_re).sum(dim=(0, 1))
        d_log_A_real = d_A_real * A_real_neg

        # ---- Backward through Bu RoPE ----
        # Bu_rot = RoPE(Bu, cum_theta)
        cos_ct_b = torch.cos(cum_theta)
        sin_ct_b = torch.sin(cum_theta)
        Bu_3d = Bu.view(B, L, P)

        d_br = d_Bu_rot
        d_br1, d_br2 = d_br[..., 0::2], d_br[..., 1::2]
        bu1, bu2 = Bu_3d[..., 0::2], Bu_3d[..., 1::2]

        d_Bu_3d = torch.empty_like(Bu_3d)
        d_Bu_3d[..., 0::2] = d_br1 * cos_ct_b + d_br2 * sin_ct_b
        d_Bu_3d[..., 1::2] = -d_br1 * sin_ct_b + d_br2 * cos_ct_b

        d_cum_theta_b = (d_br1 * (-bu1 * sin_ct_b - bu2 * cos_ct_b) +
                         d_br2 * (bu1 * cos_ct_b - bu2 * sin_ct_b))

        # ---- cum_theta grad: reverse cumsum ----
        d_cum_theta = d_cum_theta_c.view(B, L, P // 2) + d_cum_theta_b
        d_dt_half_theta = d_cum_theta.flip(1).cumsum(1).flip(1)

        dt_half = dt_3d.view(B, L, P // 2, 2).mean(-1)
        d_dt_half = d_dt_half_theta * theta.view(B, L, P // 2)
        d_theta = d_dt_half_theta * dt_half
        d_dt_from_rope = d_dt_half.unsqueeze(-1).expand(B, L, P // 2, 2).reshape(B, L, P) / 2

        # ---- 2. Backward through prescan (fused kernel) ----
        d_dt_total = d_dt_disc.reshape(M, P) + d_dt_from_rope.reshape(M, P)
        d_dt_raw = torch.empty(M, P, device=u.device, dtype=u.dtype)
        d_lam_raw = torch.empty(M, P, device=u.device, dtype=u.dtype)
        d_log_dt_bias_rows = torch.empty(M, P, device=u.device, dtype=u.dtype)
        d_Bu_raw = torch.empty(M, P, device=u.device, dtype=u.dtype)
        d_b_norm_gamma_rows = torch.empty(M, P, device=u.device, dtype=u.dtype)
        d_b_bias_rows = torch.empty(M, P, device=u.device, dtype=u.dtype)

        fused_prescan_bwd_kernel[(M,)](
            d_dt_total, d_lam_disc.reshape(M, P).contiguous(),
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

        # ---- 1. Backward through x_proj linear ----
        d_x_proj = torch.cat([d_dt_raw, d_lam_raw, d_theta.reshape(M, P // 2)], dim=-1)
        d_x_proj_w = d_x_proj.t() @ u_flat
        d_x_proj_b = d_x_proj.sum(0)
        d_u_proj = d_x_proj @ x_proj_w

        # Total d_u
        d_u = (d_u_proj + d_u_c + d_u_skip).view(B, L, H)

        # d_Bu_raw
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
