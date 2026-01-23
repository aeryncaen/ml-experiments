"""Triton kernels for AdaptiveLocalConv - fused projections and gather-conv."""

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
    def adaptive_conv_fwd_kernel(
        v_ptr, out_ptr,
        half_win_ptr, center_off_ptr, kernel_ptr,
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
        half_window_max: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        hw_off = pid_b * L * H + pid_l * H + pid_h
        half_win = tl.load(half_win_ptr + hw_off)
        half_win = tl.maximum(half_win, 0.5)
        center_off = tl.load(center_off_ptr + hw_off)
        
        k_offs = tl.arange(0, K)
        kernel_base = pid_b * L * H * K + pid_l * H * K + pid_h * K
        kernel_vals = tl.load(kernel_ptr + kernel_base + k_offs, mask=k_offs < K, other=0.0)
        
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        out_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        weight_sum = 0.0
        
        for s_idx in range(2 * half_window_max + 1):
            s_off = s_idx - half_window_max
            neighbor_pos_f = pid_l + center_off + s_off
            
            valid = (neighbor_pos_f >= 0.0) & (neighbor_pos_f < L)
            valid_f = valid.to(tl.float32)
            
            rel_dist = tl.abs(s_off * 1.0) / half_win
            window_w = tl.exp(-rel_dist * rel_dist)
            
            norm_pos = tl.minimum(rel_dist, 1.0) * (K - 1)
            idx_floor = norm_pos.to(tl.int32)
            idx_floor = tl.minimum(idx_floor, K - 2)
            idx_ceil = idx_floor + 1
            w_ceil = norm_pos - idx_floor.to(tl.float32)
            w_floor = 1.0 - w_ceil
            
            k_floor = tl.sum(tl.where(k_offs == idx_floor, kernel_vals, 0.0))
            k_ceil = tl.sum(tl.where(k_offs == idx_ceil, kernel_vals, 0.0))
            kernel_w = k_floor * w_floor + k_ceil * w_ceil
            kernel_w = tl.maximum(kernel_w, 0.0)
            
            weight = (kernel_w + 1.0) * window_w * valid_f
            weight_sum += weight
            
            pos_clamped = tl.maximum(tl.minimum(neighbor_pos_f, L - 1.001), 0.0)
            pos_floor = tl.floor(pos_clamped).to(tl.int32)
            pos_floor = tl.maximum(tl.minimum(pos_floor, L - 1), 0)
            pos_ceil = tl.minimum(pos_floor + 1, L - 1)
            pos_frac = pos_clamped - pos_floor.to(tl.float32)
            
            v_base_floor = pid_b * L * C + pos_floor * C + pid_h * D
            v_base_ceil = pid_b * L * C + pos_ceil * C + pid_h * D
            
            val_floor = tl.load(v_ptr + v_base_floor + d_offs, mask=d_mask, other=0.0)
            val_ceil = tl.load(v_ptr + v_base_ceil + d_offs, mask=d_mask, other=0.0)
            val = val_floor * (1.0 - pos_frac) + val_ceil * pos_frac
            
            out_acc += val * weight
        
        weight_sum = tl.maximum(weight_sum, 1.0)
        out_acc = out_acc / weight_sum
        
        out_base = pid_b * L * C + pid_l * C + pid_h * D
        tl.store(out_ptr + out_base + d_offs, out_acc, mask=d_mask)

    @triton.jit
    def adaptive_conv_bwd_kernel(
        v_ptr, d_out_ptr, d_v_ptr,
        half_win_ptr, center_off_ptr, kernel_ptr,
        d_half_win_ptr, d_center_off_ptr, d_kernel_ptr,
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
        half_window_max: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        hw_off = pid_b * L * H + pid_l * H + pid_h
        half_win = tl.load(half_win_ptr + hw_off)
        half_win = tl.maximum(half_win, 0.5)
        center_off = tl.load(center_off_ptr + hw_off)
        
        k_offs = tl.arange(0, K)
        kernel_base = pid_b * L * H * K + pid_l * H * K + pid_h * K
        kernel_vals = tl.load(kernel_ptr + kernel_base + k_offs, mask=k_offs < K, other=0.0)
        
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        d_out_base = pid_b * L * C + pid_l * C + pid_h * D
        d_out_h = tl.load(d_out_ptr + d_out_base + d_offs, mask=d_mask, other=0.0)
        
        weight_sum = 0.0
        out_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        
        for s_idx in range(2 * half_window_max + 1):
            s_off = s_idx - half_window_max
            neighbor_pos_f = pid_l + center_off + s_off
            
            valid = (neighbor_pos_f >= 0.0) & (neighbor_pos_f < L)
            valid_f = valid.to(tl.float32)
            
            rel_dist = tl.abs(s_off * 1.0) / half_win
            window_w = tl.exp(-rel_dist * rel_dist)
            
            norm_pos = tl.minimum(rel_dist, 1.0) * (K - 1)
            idx_floor = norm_pos.to(tl.int32)
            idx_floor = tl.minimum(idx_floor, K - 2)
            idx_ceil = idx_floor + 1
            w_ceil = norm_pos - idx_floor.to(tl.float32)
            w_floor = 1.0 - w_ceil
            
            k_floor = tl.sum(tl.where(k_offs == idx_floor, kernel_vals, 0.0))
            k_ceil = tl.sum(tl.where(k_offs == idx_ceil, kernel_vals, 0.0))
            kernel_w = k_floor * w_floor + k_ceil * w_ceil
            kernel_w = tl.maximum(kernel_w, 0.0)
            
            weight = (kernel_w + 1.0) * window_w * valid_f
            weight_sum += weight
            
            pos_clamped = tl.maximum(tl.minimum(neighbor_pos_f, L - 1.001), 0.0)
            pos_floor = tl.floor(pos_clamped).to(tl.int32)
            pos_floor = tl.maximum(tl.minimum(pos_floor, L - 1), 0)
            pos_ceil = tl.minimum(pos_floor + 1, L - 1)
            pos_frac = pos_clamped - pos_floor.to(tl.float32)
            
            v_base_floor = pid_b * L * C + pos_floor * C + pid_h * D
            v_base_ceil = pid_b * L * C + pos_ceil * C + pid_h * D
            
            val_floor = tl.load(v_ptr + v_base_floor + d_offs, mask=d_mask, other=0.0)
            val_ceil = tl.load(v_ptr + v_base_ceil + d_offs, mask=d_mask, other=0.0)
            val = val_floor * (1.0 - pos_frac) + val_ceil * pos_frac
            
            out_acc += val * weight
        
        weight_sum = tl.maximum(weight_sum, 1.0)
        
        d_kernel_acc = tl.zeros((K,), dtype=tl.float32)
        d_half_win_acc = 0.0
        d_center_off_acc = 0.0
        
        for s_idx in range(2 * half_window_max + 1):
            s_off = s_idx - half_window_max
            neighbor_pos_f = pid_l + center_off + s_off
            
            valid = (neighbor_pos_f >= 0.0) & (neighbor_pos_f < L)
            valid_f = valid.to(tl.float32)
            
            rel_dist = tl.abs(s_off * 1.0) / half_win
            window_w = tl.exp(-rel_dist * rel_dist)
            
            norm_pos = tl.minimum(rel_dist, 1.0) * (K - 1)
            idx_floor = norm_pos.to(tl.int32)
            idx_floor = tl.minimum(idx_floor, K - 2)
            idx_ceil = idx_floor + 1
            w_ceil = norm_pos - idx_floor.to(tl.float32)
            w_floor = 1.0 - w_ceil
            
            k_floor = tl.sum(tl.where(k_offs == idx_floor, kernel_vals, 0.0))
            k_ceil = tl.sum(tl.where(k_offs == idx_ceil, kernel_vals, 0.0))
            kernel_interp = k_floor * w_floor + k_ceil * w_ceil
            kernel_w = tl.maximum(kernel_interp, 0.0)
            
            weight = (kernel_w + 1.0) * window_w * valid_f
            
            pos_clamped = tl.maximum(tl.minimum(neighbor_pos_f, L - 1.001), 0.0)
            pos_floor = tl.floor(pos_clamped).to(tl.int32)
            pos_floor = tl.maximum(tl.minimum(pos_floor, L - 1), 0)
            pos_ceil = tl.minimum(pos_floor + 1, L - 1)
            pos_frac = pos_clamped - pos_floor.to(tl.float32)
            
            v_base_floor = pid_b * L * C + pos_floor * C + pid_h * D
            v_base_ceil = pid_b * L * C + pos_ceil * C + pid_h * D
            
            val_floor = tl.load(v_ptr + v_base_floor + d_offs, mask=d_mask, other=0.0)
            val_ceil = tl.load(v_ptr + v_base_ceil + d_offs, mask=d_mask, other=0.0)
            val = val_floor * (1.0 - pos_frac) + val_ceil * pos_frac
            
            d_weighted_val = d_out_h / weight_sum
            d_val = d_weighted_val * weight
            d_weight = tl.sum(d_weighted_val * val)
            
            d_val_floor = d_val * (1.0 - pos_frac)
            d_val_ceil = d_val * pos_frac
            tl.atomic_add(d_v_ptr + v_base_floor + d_offs, d_val_floor, mask=d_mask)
            tl.atomic_add(d_v_ptr + v_base_ceil + d_offs, d_val_ceil, mask=d_mask)
            
            d_kernel_w = d_weight * window_w * valid_f
            is_positive = (kernel_interp >= 0.0).to(tl.float32)
            d_kernel_interp = d_kernel_w * is_positive
            
            d_kernel_acc += tl.where(k_offs == idx_floor, d_kernel_interp * w_floor, 0.0)
            d_kernel_acc += tl.where(k_offs == idx_ceil, d_kernel_interp * w_ceil, 0.0)
            
            d_window_w = d_weight * (kernel_w + 1.0) * valid_f
            d_rel_dist = d_window_w * window_w * (-2.0 * rel_dist)
            d_half_win_acc += d_rel_dist * (-tl.abs(s_off * 1.0) / (half_win * half_win))
            
            d_pos_frac = tl.sum(d_val * (val_ceil - val_floor))
            d_center_off_acc += d_pos_frac * valid_f
        
        tl.store(d_kernel_ptr + kernel_base + k_offs, d_kernel_acc, mask=k_offs < K)
        tl.atomic_add(d_half_win_ptr + hw_off, d_half_win_acc)
        tl.atomic_add(d_center_off_ptr + hw_off, d_center_off_acc)

    @triton.jit
    def silu_fwd_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        sig = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + offs, x * sig, mask=mask)

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


def triton_linear(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
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


def triton_rmsnorm(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    M, N = x.shape
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)
    rmsnorm_fwd_kernel[grid](x, gamma, out, M, N, x.stride(0), x.stride(1), out.stride(0), out.stride(1), eps, BLOCK_N)
    return out


def triton_silu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    silu_fwd_kernel[grid](x.view(-1), out.view(-1), N, BLOCK)
    return out


def triton_adaptive_conv(v, half_win, center_off, kernel, half_window_max):
    B, L, C = v.shape
    H = half_win.shape[-1]
    K = kernel.shape[-1]
    D = C // H
    BLOCK_D = triton.next_power_of_2(D)
    
    out = torch.empty_like(v)
    grid = (B, L, H)
    adaptive_conv_fwd_kernel[grid](
        v, out, half_win, center_off, kernel,
        B, L, C, H, D, K, half_window_max, BLOCK_D,
    )
    return out


class _TritonAdaptiveConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, window_w, window_b, window_gamma, offset_w, offset_b, offset_gamma,
                kernel_w, kernel_b, kernel_gamma, v_w, v_b, out_w,
                H, K, min_window, chunk_size):
        B, L, C = x.shape
        D = C // H
        
        max_window = min(int(math.sqrt(L)), K)
        half_window_max = max_window // 2
        max_offset = int(math.sqrt(L))
        
        M = B * L
        x_flat = x.reshape(M, C)
        
        window_linear = triton_linear(x_flat, window_w, window_b)
        window_normed = triton_rmsnorm(window_linear, window_gamma)
        window_raw = torch.sigmoid(window_normed)
        window_sizes = min_window + window_raw * (max_window - min_window)
        half_windows = (window_sizes / 2).view(B, L, H)
        
        offset_linear = triton_linear(x_flat, offset_w, offset_b)
        offset_normed = triton_rmsnorm(offset_linear, offset_gamma)
        center_offsets = (torch.tanh(offset_normed) * max_offset).view(B, L, H)
        
        kernel_linear = triton_linear(x_flat, kernel_w, kernel_b)
        kernel_normed = triton_rmsnorm(kernel_linear, kernel_gamma)
        kernel_weights = triton_silu(kernel_normed).view(B, L, H, K)
        
        v = triton_linear(x_flat, v_w, v_b).view(B, L, C)
        
        hidden = triton_adaptive_conv(v, half_windows, center_offsets, kernel_weights, half_window_max)
        
        out_flat = triton_linear(hidden.view(M, C), out_w)
        out = triton_silu(out_flat).view(B, L, C)
        
        ctx.save_for_backward(x, v, half_windows, center_offsets, kernel_weights, hidden,
                              window_w, window_b, window_gamma, window_linear, window_normed, window_raw,
                              offset_w, offset_b, offset_gamma, offset_linear, offset_normed,
                              kernel_w, kernel_b, kernel_gamma, kernel_linear, kernel_normed,
                              v_w, v_b, out_w, out_flat)
        ctx.H = H
        ctx.K = K
        ctx.half_window_max = half_window_max
        ctx.max_window = max_window
        ctx.max_offset = max_offset
        ctx.min_window = min_window
        
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        (x, v, half_windows, center_offsets, kernel_weights, hidden,
         window_w, window_b, window_gamma, window_linear, window_normed, window_raw,
         offset_w, offset_b, offset_gamma, offset_linear, offset_normed,
         kernel_w, kernel_b, kernel_gamma, kernel_linear, kernel_normed,
         v_w, v_b, out_w, out_flat) = ctx.saved_tensors
        
        H = ctx.H
        K = ctx.K
        half_window_max = ctx.half_window_max
        max_window = ctx.max_window
        max_offset = ctx.max_offset
        min_window = ctx.min_window
        
        B, L, C = x.shape
        D = C // H
        M = B * L
        BLOCK_D = triton.next_power_of_2(D)
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        
        d_out_flat = d_out.view(M, C)
        
        sig_out = torch.sigmoid(out_flat)
        d_out_pre = d_out_flat * sig_out * (1.0 + out_flat * (1.0 - sig_out))
        
        d_out_w = torch.zeros_like(out_w)
        grid = (triton.cdiv(C, BLOCK_N), triton.cdiv(C, BLOCK_K))
        linear_bwd_dw_kernel[grid](
            d_out_pre, hidden.view(M, C), d_out_w,
            M, C, C,
            d_out_pre.stride(0), d_out_pre.stride(1),
            hidden.view(M, C).stride(0), hidden.view(M, C).stride(1),
            d_out_w.stride(0), d_out_w.stride(1),
            BLOCK_N, BLOCK_K, BLOCK_M,
        )
        
        d_hidden = torch.zeros(M, C, device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C, BLOCK_K))
        linear_bwd_dx_kernel[grid](
            d_out_pre, out_w, d_hidden,
            M, C, C,
            d_out_pre.stride(0), d_out_pre.stride(1),
            out_w.stride(0), out_w.stride(1),
            d_hidden.stride(0), d_hidden.stride(1),
            BLOCK_M, BLOCK_K, BLOCK_N,
        )
        d_hidden = d_hidden.view(B, L, C)
        
        d_v = torch.zeros_like(v)
        d_half_windows = torch.zeros_like(half_windows)
        d_center_offsets = torch.zeros_like(center_offsets)
        d_kernel_weights = torch.zeros_like(kernel_weights)
        
        grid = (B, L, H)
        adaptive_conv_bwd_kernel[grid](
            v, d_hidden, d_v,
            half_windows, center_offsets, kernel_weights,
            d_half_windows, d_center_offsets, d_kernel_weights,
            B, L, C, H, D, K, half_window_max, BLOCK_D,
        )
        
        d_v_flat = d_v.view(M, C)
        
        d_v_w = torch.zeros_like(v_w)
        grid = (triton.cdiv(C, BLOCK_N), triton.cdiv(C, BLOCK_K))
        linear_bwd_dw_kernel[grid](
            d_v_flat, x.view(M, C), d_v_w,
            M, C, C,
            d_v_flat.stride(0), d_v_flat.stride(1),
            x.view(M, C).stride(0), x.view(M, C).stride(1),
            d_v_w.stride(0), d_v_w.stride(1),
            BLOCK_N, BLOCK_K, BLOCK_M,
        )
        d_v_b = d_v_flat.sum(0)
        
        d_x_from_v = torch.zeros(M, C, device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C, BLOCK_K))
        linear_bwd_dx_kernel[grid](
            d_v_flat, v_w, d_x_from_v,
            M, C, C,
            d_v_flat.stride(0), d_v_flat.stride(1),
            v_w.stride(0), v_w.stride(1),
            d_x_from_v.stride(0), d_x_from_v.stride(1),
            BLOCK_M, BLOCK_K, BLOCK_N,
        )
        
        d_kernel_flat = d_kernel_weights.view(M, H * K)
        sig_kernel = torch.sigmoid(kernel_normed)
        d_kernel_normed = d_kernel_flat * sig_kernel * (1.0 + kernel_normed * (1.0 - sig_kernel))
        
        d_kernel_gamma = (d_kernel_normed * kernel_linear / (kernel_linear.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)).sum(0)
        d_kernel_linear = d_kernel_normed * kernel_gamma / (kernel_linear.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
        
        d_kernel_w = torch.zeros_like(kernel_w)
        grid = (triton.cdiv(H * K, BLOCK_N), triton.cdiv(C, BLOCK_K))
        linear_bwd_dw_kernel[grid](
            d_kernel_linear, x.view(M, C), d_kernel_w,
            M, H * K, C,
            d_kernel_linear.stride(0), d_kernel_linear.stride(1),
            x.view(M, C).stride(0), x.view(M, C).stride(1),
            d_kernel_w.stride(0), d_kernel_w.stride(1),
            BLOCK_N, BLOCK_K, BLOCK_M,
        )
        d_kernel_b = d_kernel_linear.sum(0)
        
        d_x_from_kernel = torch.zeros(M, C, device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C, BLOCK_K))
        linear_bwd_dx_kernel[grid](
            d_kernel_linear, kernel_w, d_x_from_kernel,
            M, H * K, C,
            d_kernel_linear.stride(0), d_kernel_linear.stride(1),
            kernel_w.stride(0), kernel_w.stride(1),
            d_x_from_kernel.stride(0), d_x_from_kernel.stride(1),
            BLOCK_M, BLOCK_K, BLOCK_N,
        )
        
        d_window_sizes = d_half_windows / 2
        d_window_raw = d_window_sizes.view(M, H) * (max_window - min_window)
        d_window_normed = d_window_raw * window_raw * (1.0 - window_raw)
        
        d_window_gamma = (d_window_normed * window_linear / (window_linear.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)).sum(0)
        d_window_linear = d_window_normed * window_gamma / (window_linear.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
        
        d_window_w = torch.zeros_like(window_w)
        grid = (triton.cdiv(H, BLOCK_N), triton.cdiv(C, BLOCK_K))
        linear_bwd_dw_kernel[grid](
            d_window_linear, x.view(M, C), d_window_w,
            M, H, C,
            d_window_linear.stride(0), d_window_linear.stride(1),
            x.view(M, C).stride(0), x.view(M, C).stride(1),
            d_window_w.stride(0), d_window_w.stride(1),
            BLOCK_N, BLOCK_K, BLOCK_M,
        )
        d_window_b = d_window_linear.sum(0)
        
        d_x_from_window = torch.zeros(M, C, device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C, BLOCK_K))
        linear_bwd_dx_kernel[grid](
            d_window_linear, window_w, d_x_from_window,
            M, H, C,
            d_window_linear.stride(0), d_window_linear.stride(1),
            window_w.stride(0), window_w.stride(1),
            d_x_from_window.stride(0), d_x_from_window.stride(1),
            BLOCK_M, BLOCK_K, BLOCK_N,
        )
        
        d_center_off_flat = d_center_offsets.view(M, H)
        tanh_off = torch.tanh(offset_normed)
        d_offset_normed = d_center_off_flat * max_offset * (1.0 - tanh_off * tanh_off)
        
        d_offset_gamma = (d_offset_normed * offset_linear / (offset_linear.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)).sum(0)
        d_offset_linear = d_offset_normed * offset_gamma / (offset_linear.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)
        
        d_offset_w = torch.zeros_like(offset_w)
        grid = (triton.cdiv(H, BLOCK_N), triton.cdiv(C, BLOCK_K))
        linear_bwd_dw_kernel[grid](
            d_offset_linear, x.view(M, C), d_offset_w,
            M, H, C,
            d_offset_linear.stride(0), d_offset_linear.stride(1),
            x.view(M, C).stride(0), x.view(M, C).stride(1),
            d_offset_w.stride(0), d_offset_w.stride(1),
            BLOCK_N, BLOCK_K, BLOCK_M,
        )
        d_offset_b = d_offset_linear.sum(0)
        
        d_x_from_offset = torch.zeros(M, C, device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C, BLOCK_K))
        linear_bwd_dx_kernel[grid](
            d_offset_linear, offset_w, d_x_from_offset,
            M, H, C,
            d_offset_linear.stride(0), d_offset_linear.stride(1),
            offset_w.stride(0), offset_w.stride(1),
            d_x_from_offset.stride(0), d_x_from_offset.stride(1),
            BLOCK_M, BLOCK_K, BLOCK_N,
        )
        
        d_x = (d_x_from_v + d_x_from_kernel + d_x_from_window + d_x_from_offset).view(B, L, C)
        
        return (d_x, d_window_w, d_window_b, d_window_gamma, d_offset_w, d_offset_b, d_offset_gamma,
                d_kernel_w, d_kernel_b, d_kernel_gamma, d_v_w, d_v_b, d_out_w,
                None, None, None, None)


class TritonAdaptiveLocalConv(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        max_kernel_size: int = 64,
        min_window: float = 1.0,
        chunk_size: int = 1024,
    ):
        super().__init__()
        if not HAS_TRITON:
            raise RuntimeError("Triton not available")
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.max_kernel_size = max_kernel_size
        self.min_window = min_window
        self.chunk_size = chunk_size
        
        H = num_heads
        K = max_kernel_size
        
        self.window_proj = nn.Linear(channels, H)
        self.window_gamma = nn.Parameter(torch.ones(H))
        self.offset_proj = nn.Linear(channels, H)
        self.offset_gamma = nn.Parameter(torch.ones(H))
        self.kernel_proj = nn.Linear(channels, H * K)
        self.kernel_gamma = nn.Parameter(torch.ones(H * K))
        self.v_proj = nn.Linear(channels, channels)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.zeros_(self.window_proj.weight)
        nn.init.constant_(self.window_proj.bias, 0.0)
        nn.init.zeros_(self.offset_proj.weight)
        nn.init.zeros_(self.offset_proj.bias)
        nn.init.xavier_uniform_(self.kernel_proj.weight, gain=0.01)
        nn.init.zeros_(self.kernel_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        out = _TritonAdaptiveConv.apply(
            x,
            self.window_proj.weight, self.window_proj.bias, self.window_gamma,
            self.offset_proj.weight, self.offset_proj.bias, self.offset_gamma,
            self.kernel_proj.weight, self.kernel_proj.bias, self.kernel_gamma,
            self.v_proj.weight, self.v_proj.bias, self.out_proj.weight,
            self.num_heads, self.max_kernel_size, self.min_window, self.chunk_size,
        )
        return out, {}


if __name__ == "__main__":
    if not HAS_TRITON:
        print("Triton not available")
        exit()
    
    import time
    from .adaptive_local_conv import AdaptiveLocalConv
    
    device = "cuda"
    B, L, C = 4, 1024, 256
    H = 8
    K = 64
    
    print(f"Config: B={B}, L={L}, C={C}, H={H}, K={K}")
    print()
    
    pytorch_model = AdaptiveLocalConv(channels=C, num_heads=H, max_kernel_size=K).to(device)
    triton_model = TritonAdaptiveLocalConv(channels=C, num_heads=H, max_kernel_size=K).to(device)
    
    with torch.no_grad():
        triton_model.window_proj.weight.copy_(pytorch_model.window_proj.weight)
        triton_model.window_proj.bias.copy_(pytorch_model.window_proj.bias)
        triton_model.window_gamma.copy_(pytorch_model.window_gamma)
        triton_model.offset_proj.weight.copy_(pytorch_model.offset_proj.weight)
        triton_model.offset_proj.bias.copy_(pytorch_model.offset_proj.bias)
        triton_model.offset_gamma.copy_(pytorch_model.offset_gamma)
        triton_model.kernel_proj.weight.copy_(pytorch_model.kernel_proj.weight)
        triton_model.kernel_proj.bias.copy_(pytorch_model.kernel_proj.bias)
        triton_model.kernel_gamma.copy_(pytorch_model.kernel_gamma)
        triton_model.v_proj.weight.copy_(pytorch_model.v_proj.weight)
        triton_model.v_proj.bias.copy_(pytorch_model.v_proj.bias)
        triton_model.out_proj.weight.copy_(pytorch_model.out_proj.weight)
    
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    
    torch.manual_seed(42)
    x = torch.randn(B, L, C, device=device)
    
    out_pt, _ = pytorch_model(x)
    out_tr, _ = triton_model(x)
    
    fwd_diff = (out_pt - out_tr).abs().max().item()
    print(f"Forward match: {'PASS' if fwd_diff < 1e-2 else 'FAIL'} (max diff: {fwd_diff:.2e})")
    
    x_pt = x.clone().requires_grad_(True)
    x_tr = x.clone().requires_grad_(True)
    
    out_pt, _ = pytorch_model(x_pt)
    out_pt.sum().backward()
    
    out_tr, _ = triton_model(x_tr)
    out_tr.sum().backward()
    
    dx_diff = (x_pt.grad - x_tr.grad).abs().max().item()
    print(f"d_x match: {'PASS' if dx_diff < 1e-1 else 'FAIL'} (max diff: {dx_diff:.2e})")
    print()
    
    print("=" * 60)
    print("BENCHMARKS")
    print("=" * 60)
    
    warmup, iters = 5, 20
    
    def bench(model, x):
        x_grad = x.clone().requires_grad_(True)
        
        for _ in range(warmup):
            out, _ = model(x_grad)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        for _ in range(iters):
            out, _ = model(x_grad)
            torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - start) / iters * 1000
        fwd_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        for _ in range(warmup):
            out, _ = model(x_grad)
            out.sum().backward()
            model.zero_grad()
            x_grad.grad = None
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        for _ in range(iters):
            out, _ = model(x_grad)
            out.sum().backward()
            model.zero_grad()
            x_grad.grad = None
            torch.cuda.synchronize()
        total_ms = (time.perf_counter() - start) / iters * 1000
        bwd_ms = total_ms - fwd_ms
        bwd_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        return fwd_ms, bwd_ms, fwd_mem, bwd_mem
    
    print(f"{'L':>6} | {'------- PyTorch -------':^24} | {'------- Triton --------':^24} | {'Speedup':^8}")
    print(f"{'':>6} | {'Fwd':>6} {'Bwd':>6} {'FwdMB':>5} {'BwdMB':>5} | {'Fwd':>6} {'Bwd':>6} {'FwdMB':>5} {'BwdMB':>5} | {'':>8}")
    print("-" * 90)
    
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    
    for L in seq_lengths:
        torch.cuda.empty_cache()
        
        try:
            x = torch.randn(B, L, C, device=device)
            pt_fwd, pt_bwd, pt_fwd_mem, pt_bwd_mem = bench(pytorch_model, x)
            tr_fwd, tr_bwd, tr_fwd_mem, tr_bwd_mem = bench(triton_model, x)
            speedup = (pt_fwd + pt_bwd) / (tr_fwd + tr_bwd)
            print(f"{L:>6} | {pt_fwd:>6.1f} {pt_bwd:>6.1f} {pt_fwd_mem:>5.0f} {pt_bwd_mem:>5.0f} | {tr_fwd:>6.1f} {tr_bwd:>6.1f} {tr_fwd_mem:>5.0f} {tr_bwd_mem:>5.0f} | {speedup:>6.2f}x")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{L:>6} | OOM")
                torch.cuda.empty_cache()
            else:
                raise
    
    print("=" * 90)
