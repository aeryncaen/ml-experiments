"""Triton kernels for GatherConv - minimal memory, proper tiling."""

from __future__ import annotations

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
    # ========== Linear + SiLU kernel (batched) ==========
    # Computes: out = silu(x @ W.T + b) for x: (M, K), W: (N, K), b: (N,) -> out: (M, N)
    # M = B*L (flattened batch), K = input dim, N = output dim
    
    @triton.jit
    def linear_silu_fwd_kernel(
        x_ptr, w_ptr, b_ptr, out_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
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
        
        # Add bias
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
        
        # SiLU activation: x * sigmoid(x)
        sigmoid_acc = 1.0 / (1.0 + tl.exp(-acc))
        out = acc * sigmoid_acc
        
        # Store
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, out, mask=out_mask)

    @triton.jit
    def linear_fwd_kernel(
        x_ptr, w_ptr, out_ptr,
        M, N, K,
        stride_xm, stride_xk,
        stride_wn, stride_wk,
        stride_om, stride_on,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Linear without bias or activation"""
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
        
        out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
        out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(out_ptrs, acc, mask=out_mask)

    @triton.jit
    def silu_fwd_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        sig = 1.0 / (1.0 + tl.exp(-x))
        tl.store(out_ptr + offs, x * sig, mask=mask)

    @triton.jit
    def rmsnorm_silu_fwd_kernel(
        x_ptr, gamma_ptr, out_ptr,
        M, N,
        stride_xm, stride_xn,
        stride_om, stride_on,
        eps,
        BLOCK_N: tl.constexpr,
    ):
        """Fused RMSNorm -> SiLU. Each program handles one row."""
        pid_m = tl.program_id(0)
        
        offs_n = tl.arange(0, BLOCK_N)
        n_mask = offs_n < N
        
        # Load input row
        x_ptrs = x_ptr + pid_m * stride_xm + offs_n * stride_xn
        x = tl.load(x_ptrs, mask=n_mask, other=0.0)
        
        # RMSNorm: y = x / sqrt(mean(x^2) + eps) * gamma
        x_sq = x * x
        mean_sq = tl.sum(x_sq) / N
        rrms = 1.0 / tl.sqrt(mean_sq + eps)
        
        gamma = tl.load(gamma_ptr + offs_n, mask=n_mask, other=1.0)
        normed = x * rrms * gamma
        
        # SiLU: x * sigmoid(x)
        sigmoid_val = 1.0 / (1.0 + tl.exp(-normed))
        out = normed * sigmoid_val
        
        # Store
        out_ptrs = out_ptr + pid_m * stride_om + offs_n * stride_on
        tl.store(out_ptrs, out, mask=n_mask)

    # ========== Gather-Conv kernel ==========
    @triton.jit
    def gather_conv_fwd_kernel(
        x_ptr, out_ptr,
        freq_ptr, phase_ptr, kernel_ptr,
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
        half_s: tl.constexpr, max_receptive,
        chunk_start, chunk_len: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l_local = tl.program_id(1)  # Local position within chunk
        pid_h = tl.program_id(2)
        
        pid_l = chunk_start + pid_l_local  # Absolute position in sequence
        
        # freq/phase/kernel are stored per-chunk, use local index
        freq_off = pid_b * chunk_len * H + pid_l_local * H + pid_h
        freq_h = tl.load(freq_ptr + freq_off)
        phase_h = tl.load(phase_ptr + freq_off)
        
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        k_offs = tl.arange(0, K)
        # kernel stored per-chunk
        kernel_base = pid_b * chunk_len * H * K + pid_l_local * H * K + pid_h * K
        kernel_vals = tl.load(kernel_ptr + kernel_base + k_offs, mask=k_offs < K, other=0.0)
        
        out_h = tl.zeros((BLOCK_D,), dtype=tl.float32)
        
        for s_idx in range(S):
            s_off = s_idx - half_s
            sample_pos_f = pid_l + s_off * freq_h + phase_h
            
            # Soft indexing: interpolate between floor and ceil positions
            pos_clamped = tl.maximum(tl.minimum(sample_pos_f, L - 1.001), 0.0)
            pos_floor = tl.floor(pos_clamped).to(tl.int32)
            pos_ceil = tl.minimum(pos_floor + 1, L - 1)
            pos_frac = pos_clamped - pos_floor.to(tl.float32)
            
            valid = (sample_pos_f >= 0.0) & (sample_pos_f < L)
            valid_f = valid.to(tl.float32)
            
            rel_pos = tl.abs(s_off * freq_h)
            norm_pos = rel_pos / max_receptive
            norm_pos = tl.minimum(norm_pos, 1.0)
            
            idx_float = norm_pos * (K - 1)
            idx_floor = idx_float.to(tl.int32)
            idx_floor = tl.minimum(idx_floor, K - 2)
            idx_ceil = idx_floor + 1
            w_ceil = idx_float - idx_floor.to(tl.float32)
            w_floor = 1.0 - w_ceil
            
            k_floor = tl.sum(tl.where(k_offs == idx_floor, kernel_vals, 0.0))
            k_ceil = tl.sum(tl.where(k_offs == idx_ceil, kernel_vals, 0.0))
            kernel_w = k_floor * w_floor + k_ceil * w_ceil
            kernel_w = kernel_w * valid_f
            
            # Soft gather: interpolate between floor and ceil values
            val_base_floor = pid_b * L * C + pos_floor * C + pid_h * D
            val_base_ceil = pid_b * L * C + pos_ceil * C + pid_h * D
            vals_floor = tl.load(x_ptr + val_base_floor + d_offs, mask=d_mask, other=0.0)
            vals_ceil = tl.load(x_ptr + val_base_ceil + d_offs, mask=d_mask, other=0.0)
            vals = vals_floor * (1.0 - pos_frac) + vals_ceil * pos_frac
            out_h += vals * kernel_w
        
        # Write to chunk output using local position
        out_base = pid_b * chunk_len * C + pid_l_local * C + pid_h * D
        tl.store(out_ptr + out_base + d_offs, out_h, mask=d_mask)

    @triton.jit
    def gather_conv_bwd_kernel(
        x_ptr, d_out_ptr, d_x_ptr,
        freq_ptr, phase_ptr, kernel_ptr,
        d_freq_ptr, d_phase_ptr, d_kernel_ptr,
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
        half_s: tl.constexpr, max_receptive,
        chunk_start, chunk_len: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l_local = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        pid_l = chunk_start + pid_l_local
        
        freq_off = pid_b * chunk_len * H + pid_l_local * H + pid_h
        freq_h = tl.load(freq_ptr + freq_off)
        phase_h = tl.load(phase_ptr + freq_off)
        
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        k_offs = tl.arange(0, K)
        kernel_base = pid_b * chunk_len * H * K + pid_l_local * H * K + pid_h * K
        kernel_vals = tl.load(kernel_ptr + kernel_base + k_offs, mask=k_offs < K, other=0.0)
        
        d_out_base = pid_b * chunk_len * C + pid_l_local * C + pid_h * D
        d_out_h = tl.load(d_out_ptr + d_out_base + d_offs, mask=d_mask, other=0.0)
        
        d_freq_acc = 0.0
        d_phase_acc = 0.0
        d_kernel_acc = tl.zeros((K,), dtype=tl.float32)
        
        for s_idx in range(S):
            s_off = s_idx - half_s
            sample_pos_f = pid_l + s_off * freq_h + phase_h
            
            # Soft indexing with floor/ceil interpolation
            pos_clamped = tl.maximum(tl.minimum(sample_pos_f, L - 1.001), 0.0)
            pos_floor = tl.floor(pos_clamped).to(tl.int32)
            pos_ceil = tl.minimum(pos_floor + 1, L - 1)
            pos_frac = pos_clamped - pos_floor.to(tl.float32)
            
            valid = (sample_pos_f >= 0.0) & (sample_pos_f < L)
            valid_f = valid.to(tl.float32)
            
            rel_pos = tl.abs(s_off * freq_h)
            norm_pos = rel_pos / max_receptive
            not_clamped = (norm_pos < 1.0).to(tl.float32)
            norm_pos = tl.minimum(norm_pos, 1.0)
            
            idx_float = norm_pos * (K - 1)
            idx_floor = idx_float.to(tl.int32)
            idx_floor = tl.minimum(idx_floor, K - 2)
            idx_ceil = idx_floor + 1
            w_ceil = idx_float - idx_floor.to(tl.float32)
            w_floor = 1.0 - w_ceil
            
            k_floor = tl.sum(tl.where(k_offs == idx_floor, kernel_vals, 0.0))
            k_ceil = tl.sum(tl.where(k_offs == idx_ceil, kernel_vals, 0.0))
            kernel_w = k_floor * w_floor + k_ceil * w_ceil
            kernel_w = kernel_w * valid_f
            
            # Load values from floor and ceil positions
            val_base_floor = pid_b * L * C + pos_floor * C + pid_h * D
            val_base_ceil = pid_b * L * C + pos_ceil * C + pid_h * D
            vals_floor = tl.load(x_ptr + val_base_floor + d_offs, mask=d_mask, other=0.0)
            vals_ceil = tl.load(x_ptr + val_base_ceil + d_offs, mask=d_mask, other=0.0)
            vals = vals_floor * (1.0 - pos_frac) + vals_ceil * pos_frac
            
            # d_x: distribute to both floor and ceil positions
            d_vals = d_out_h * kernel_w
            d_vals_floor = d_vals * (1.0 - pos_frac)
            d_vals_ceil = d_vals * pos_frac
            tl.atomic_add(d_x_ptr + val_base_floor + d_offs, d_vals_floor, mask=d_mask)
            tl.atomic_add(d_x_ptr + val_base_ceil + d_offs, d_vals_ceil, mask=d_mask)
            
            # d_pos_frac from soft interpolation: d_frac = d_out * kernel_w * (vals_ceil - vals_floor)
            d_pos_frac = tl.sum(d_out_h * kernel_w * (vals_ceil - vals_floor))
            
            # d_kernel
            d_kernel_w = tl.sum(d_out_h * vals) * valid_f
            d_kernel_acc += tl.where(k_offs == idx_floor, d_kernel_w * w_floor, 0.0)
            d_kernel_acc += tl.where(k_offs == idx_ceil, d_kernel_w * w_ceil, 0.0)
            
            # d_freq from kernel interpolation
            d_w_ceil = d_kernel_w * (k_ceil - k_floor)
            d_norm_pos = d_w_ceil * (K - 1)
            d_rel_pos = d_norm_pos / max_receptive * not_clamped
            s_off_f = s_off * 1.0
            d_freq_kernel = d_rel_pos * tl.abs(s_off_f)
            
            # d_freq and d_phase from position interpolation
            # sample_pos = center + s_off * freq + phase
            # d_freq += d_pos_frac * s_off, d_phase += d_pos_frac
            d_freq_pos = d_pos_frac * s_off_f * valid_f
            d_phase_pos = d_pos_frac * valid_f
            
            d_freq_acc += d_freq_kernel + d_freq_pos
            d_phase_acc += d_phase_pos
        
        tl.store(d_kernel_ptr + kernel_base + k_offs, d_kernel_acc, mask=k_offs < K)
        tl.store(d_freq_ptr + freq_off, d_freq_acc)
        tl.store(d_phase_ptr + freq_off, d_phase_acc)

    # ========== Backward kernels for linear layers ==========
    @triton.jit
    def linear_bwd_dx_kernel(
        d_out_ptr, w_ptr, d_x_ptr,
        M, N, K,  # d_out: (M, N), w: (N, K), d_x: (M, K)
        stride_dom, stride_don,
        stride_wn, stride_wk,
        stride_dxm, stride_dxk,
        BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        """d_x = d_out @ W"""
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
        M, N, K,  # d_out: (M, N), x: (M, K), d_w: (N, K)
        stride_dom, stride_don,
        stride_xm, stride_xk,
        stride_dwn, stride_dwk,
        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr,
    ):
        """d_W = d_out.T @ x"""
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


# ========== Helper functions ==========
def triton_linear_silu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """x: (M, K), w: (N, K), b: (N,) -> (M, N)"""
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    linear_silu_fwd_kernel[grid](
        x, w, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return out


def triton_linear(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x: (M, K), w: (N, K) -> (M, N)"""
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    linear_fwd_kernel[grid](
        x, w, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
    )
    return out


def triton_silu(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    silu_fwd_kernel[grid](x.view(-1), out.view(-1), N, BLOCK)
    return out


def triton_rmsnorm_silu(x: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """x: (M, N), gamma: (N,) -> (M, N). Fused RMSNorm + SiLU."""
    M, N = x.shape
    out = torch.empty_like(x)
    
    BLOCK_N = triton.next_power_of_2(N)
    grid = (M,)
    
    rmsnorm_silu_fwd_kernel[grid](
        x, gamma, out,
        M, N,
        x.stride(0), x.stride(1),
        out.stride(0), out.stride(1),
        eps,
        BLOCK_N,
    )
    return out


def triton_linear_rmsnorm_silu(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, gamma: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """x: (M, K), w: (N, K), b: (N,), gamma: (N,) -> (M, N). Linear + RMSNorm + SiLU."""
    linear_out = triton_linear(x, w) + b
    return triton_rmsnorm_silu(linear_out, gamma, eps)


def triton_gather_conv(x, freq, phase, kernel, half_s, max_receptive, chunk_start=0):
    """freq/phase/kernel are for a chunk starting at chunk_start, output is chunk-sized"""
    B, L, C = x.shape
    chunk_len = freq.shape[1]  # freq is (B, chunk_len, H)
    H = freq.shape[-1]
    K = kernel.shape[-1]
    D = C // H
    S = 2 * half_s + 1
    BLOCK_D = triton.next_power_of_2(D)
    
    out = torch.empty(B, chunk_len, C, device=x.device, dtype=x.dtype)
    grid = (B, chunk_len, H)
    gather_conv_fwd_kernel[grid](
        x, out, freq, phase, kernel,
        B, L, C, H, D, S, K, half_s, max_receptive,
        chunk_start, chunk_len, BLOCK_D,
    )
    return out


# ========== Main module ==========
class TritonGatherConv(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        max_samples: int = 32,
        max_freq: float = 16.0,
        min_freq: float = 1.0,
        max_kernel_size: int = 64,
        chunk_size: int = 1024,
    ):
        super().__init__()
        if not HAS_TRITON:
            raise RuntimeError("Triton not available")
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.max_kernel_size = max_kernel_size
        self.chunk_size = chunk_size
        
        self.half_s = max_samples // 2
        self.max_receptive = self.half_s * max_freq
        
        self.wave_w = nn.Parameter(torch.empty(2 * num_heads, channels))
        self.wave_b = nn.Parameter(torch.zeros(2 * num_heads))
        self.wave_gamma = nn.Parameter(torch.ones(2 * num_heads))
        self.kernel_w = nn.Parameter(torch.empty(num_heads * max_kernel_size, channels))
        self.kernel_b = nn.Parameter(torch.zeros(num_heads * max_kernel_size))
        self.kernel_gamma = nn.Parameter(torch.ones(num_heads * max_kernel_size))
        self.out_w = nn.Parameter(torch.empty(channels, channels))
        
        nn.init.zeros_(self.wave_w)
        nn.init.xavier_uniform_(self.kernel_w, gain=0.1)
        nn.init.xavier_uniform_(self.out_w, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return _GatherConvTriton.apply(
            x, self.wave_w, self.wave_b, self.wave_gamma,
            self.kernel_w, self.kernel_b, self.kernel_gamma, self.out_w,
            self.num_heads, self.half_s, self.max_receptive,
            self.max_freq, self.min_freq, self.max_kernel_size, self.chunk_size,
        ), {}


class _GatherConvTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wave_w, wave_b, wave_gamma, kernel_w, kernel_b, kernel_gamma, out_w,
                H, half_s, max_receptive, max_freq, min_freq, K, chunk_size):
        B, L, C = x.shape
        D = C // H
        S = 2 * half_s + 1
        BLOCK_D = triton.next_power_of_2(D)
        chunk_size = min(chunk_size, L)
        
        out = torch.empty_like(x)
        
        # Process in chunks to minimize intermediate memory
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            chunk_len = end - start
            M_chunk = B * chunk_len
            
            x_chunk = x[:, start:end, :].reshape(M_chunk, C)
            
            # wave_proj for chunk: linear -> RMSNorm -> SiLU
            wave_out = triton_linear_rmsnorm_silu(x_chunk, wave_w, wave_b, wave_gamma)  # (M_chunk, 2H)
            wave_out = wave_out.view(B, chunk_len, 2, H)
            
            freq_pre = wave_out[:, :, 0, :]
            phase_pre = wave_out[:, :, 1, :]
            freq_chunk = torch.sigmoid(freq_pre) * (max_freq - min_freq) + min_freq
            phase_chunk = torch.tanh(phase_pre) * max_freq
            
            # kernel_proj for chunk: linear -> RMSNorm -> SiLU
            kernel_out = triton_linear_rmsnorm_silu(x_chunk, kernel_w, kernel_b, kernel_gamma)  # (M_chunk, H*K)
            kernel_chunk = kernel_out.view(B, chunk_len, H, K)
            
            # gather-conv for chunk (reads from full x, writes to chunk)
            hidden_chunk = torch.empty(B, chunk_len, C, device=x.device, dtype=x.dtype)
            grid = (B, chunk_len, H)
            gather_conv_fwd_kernel[grid](
                x, hidden_chunk,
                freq_chunk.contiguous(), phase_chunk.contiguous(), kernel_chunk.contiguous(),
                B, L, C, H, D, S, K, half_s, max_receptive,
                start, chunk_len, BLOCK_D,
            )
            
            # out_proj for chunk
            hidden_flat = hidden_chunk.view(M_chunk, C)
            out_pre = triton_linear(hidden_flat, out_w)
            out[:, start:end, :] = triton_silu(out_pre).view(B, chunk_len, C)
        
        # Save x and weights for backward (recompute intermediates)
        ctx.save_for_backward(x, wave_w, wave_b, wave_gamma, kernel_w, kernel_b, kernel_gamma, out_w)
        ctx.H = H
        ctx.half_s = half_s
        ctx.max_receptive = max_receptive
        ctx.max_freq = max_freq
        ctx.min_freq = min_freq
        ctx.K = K
        ctx.chunk_size = chunk_size
        ctx.shape = (B, L, C)
        
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        H = ctx.H
        half_s = ctx.half_s
        max_receptive = ctx.max_receptive
        max_freq = ctx.max_freq
        min_freq = ctx.min_freq
        K = ctx.K
        B, L, C = ctx.shape
        D = C // H
        S = 2 * half_s + 1
        BLOCK_D = triton.next_power_of_2(D)
        chunk_size = min(ctx.chunk_size, L)
        
        x, wave_w, wave_b, wave_gamma, kernel_w, kernel_b, kernel_gamma, out_w = ctx.saved_tensors
        
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        eps = 1e-5  # RMSNorm epsilon
        
        # Initialize gradient accumulators
        d_x = torch.zeros_like(x)
        d_wave_w = torch.zeros_like(wave_w)
        d_wave_b = torch.zeros_like(wave_b)
        d_wave_gamma = torch.zeros_like(wave_gamma)
        d_kernel_w = torch.zeros_like(kernel_w)
        d_kernel_b = torch.zeros_like(kernel_b)
        d_kernel_gamma = torch.zeros_like(kernel_gamma)
        d_out_w = torch.zeros_like(out_w)
        
        # Process backward in chunks
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            chunk_len = end - start
            M_chunk = B * chunk_len
            
            x_chunk = x[:, start:end, :].reshape(M_chunk, C)
            d_out_chunk = d_out[:, start:end, :].reshape(M_chunk, C)
            
            # === Recompute forward for this chunk ===
            # wave_proj: linear -> RMSNorm -> SiLU
            wave_linear = triton_linear(x_chunk, wave_w) + wave_b
            wave_rrms = 1.0 / torch.sqrt((wave_linear ** 2).mean(dim=-1, keepdim=True) + eps)
            wave_normed = wave_linear * wave_rrms * wave_gamma
            wave_out = (wave_normed * torch.sigmoid(wave_normed)).view(B, chunk_len, 2, H)
            
            freq_pre = wave_out[:, :, 0, :]
            phase_pre = wave_out[:, :, 1, :]
            sig_freq = torch.sigmoid(freq_pre)
            freq_chunk = sig_freq * (max_freq - min_freq) + min_freq
            tanh_phase = torch.tanh(phase_pre)
            phase_chunk = tanh_phase * max_freq
            
            # kernel_proj: linear -> RMSNorm -> SiLU
            kernel_linear = triton_linear(x_chunk, kernel_w) + kernel_b
            kernel_rrms = 1.0 / torch.sqrt((kernel_linear ** 2).mean(dim=-1, keepdim=True) + eps)
            kernel_normed = kernel_linear * kernel_rrms * kernel_gamma
            kernel_out = kernel_normed * torch.sigmoid(kernel_normed)
            kernel_chunk = kernel_out.view(B, chunk_len, H, K)
            
            # Recompute hidden for chunk
            hidden_chunk = triton_gather_conv(x, freq_chunk.contiguous(), phase_chunk.contiguous(),
                                              kernel_chunk.contiguous(), half_s, max_receptive, start)
            hidden_flat = hidden_chunk.view(M_chunk, C)
            
            out_pre = triton_linear(hidden_flat, out_w)
            
            # === Backward through out_proj + silu ===
            sig_out = torch.sigmoid(out_pre)
            d_out_pre = d_out_chunk * sig_out * (1.0 + out_pre * (1.0 - sig_out))
            
            # Accumulate d_out_w
            d_out_w_chunk = torch.zeros_like(out_w)
            grid = (triton.cdiv(C, BLOCK_N), triton.cdiv(C, BLOCK_K))
            linear_bwd_dw_kernel[grid](
                d_out_pre, hidden_flat, d_out_w_chunk,
                M_chunk, C, C,
                d_out_pre.stride(0), d_out_pre.stride(1),
                hidden_flat.stride(0), hidden_flat.stride(1),
                d_out_w_chunk.stride(0), d_out_w_chunk.stride(1),
                BLOCK_N, BLOCK_K, BLOCK_M,
            )
            d_out_w += d_out_w_chunk
            
            # d_hidden for chunk
            d_hidden_flat = torch.zeros(M_chunk, C, device=x.device, dtype=x.dtype)
            grid = (triton.cdiv(M_chunk, BLOCK_M), triton.cdiv(C, BLOCK_K))
            linear_bwd_dx_kernel[grid](
                d_out_pre, out_w, d_hidden_flat,
                M_chunk, C, C,
                d_out_pre.stride(0), d_out_pre.stride(1),
                out_w.stride(0), out_w.stride(1),
                d_hidden_flat.stride(0), d_hidden_flat.stride(1),
                BLOCK_M, BLOCK_K, BLOCK_N,
            )
            d_hidden_chunk = d_hidden_flat.view(B, chunk_len, C)
            
            # === Backward through gather-conv ===
            d_freq_chunk = torch.zeros(B, chunk_len, H, device=x.device, dtype=x.dtype)
            d_phase_chunk = torch.zeros(B, chunk_len, H, device=x.device, dtype=x.dtype)
            d_kernel_chunk = torch.zeros(B, chunk_len, H, K, device=x.device, dtype=x.dtype)
            
            grid = (B, chunk_len, H)
            gather_conv_bwd_kernel[grid](
                x, d_hidden_chunk.contiguous(), d_x,
                freq_chunk.contiguous(), phase_chunk.contiguous(), kernel_chunk.contiguous(),
                d_freq_chunk, d_phase_chunk, d_kernel_chunk,
                B, L, C, H, D, S, K, half_s, max_receptive,
                start, chunk_len, BLOCK_D,
            )
            
            # === Backward through kernel_proj (SiLU -> RMSNorm -> Linear) ===
            d_kernel_flat = d_kernel_chunk.view(M_chunk, H * K)
            
            # Backward through SiLU: d_normed = d_out * sigmoid(x) * (1 + x * (1 - sigmoid(x)))
            sig_kernel = torch.sigmoid(kernel_normed)
            d_kernel_normed = d_kernel_flat * sig_kernel * (1.0 + kernel_normed * (1.0 - sig_kernel))
            
            # Backward through RMSNorm
            # y = x * rrms * gamma  =>  d_gamma = sum(dy * x * rrms), d_x = rrms * (dy * gamma - y_norm * mean(dy * gamma * y_norm))
            kernel_y_norm = kernel_linear * kernel_rrms
            d_kernel_gamma += (d_kernel_normed * kernel_y_norm).sum(dim=0)
            inner_k = (d_kernel_normed * kernel_gamma * kernel_y_norm).mean(dim=-1, keepdim=True)
            d_kernel_linear = kernel_rrms * (d_kernel_normed * kernel_gamma - kernel_y_norm * inner_k)
            
            d_kernel_pre = d_kernel_linear  # For compatibility with existing code below
            
            d_kernel_w_chunk = torch.zeros_like(kernel_w)
            grid = (triton.cdiv(H * K, BLOCK_N), triton.cdiv(C, BLOCK_K))
            linear_bwd_dw_kernel[grid](
                d_kernel_pre, x_chunk, d_kernel_w_chunk,
                M_chunk, H * K, C,
                d_kernel_pre.stride(0), d_kernel_pre.stride(1),
                x_chunk.stride(0), x_chunk.stride(1),
                d_kernel_w_chunk.stride(0), d_kernel_w_chunk.stride(1),
                BLOCK_N, BLOCK_K, BLOCK_M,
            )
            d_kernel_w += d_kernel_w_chunk
            d_kernel_b += d_kernel_pre.sum(0)
            
            d_x_kernel_chunk = torch.zeros(M_chunk, C, device=x.device, dtype=x.dtype)
            grid = (triton.cdiv(M_chunk, BLOCK_M), triton.cdiv(C, BLOCK_K))
            linear_bwd_dx_kernel[grid](
                d_kernel_pre, kernel_w, d_x_kernel_chunk,
                M_chunk, H * K, C,
                d_kernel_pre.stride(0), d_kernel_pre.stride(1),
                kernel_w.stride(0), kernel_w.stride(1),
                d_x_kernel_chunk.stride(0), d_x_kernel_chunk.stride(1),
                BLOCK_M, BLOCK_K, BLOCK_N,
            )
            d_x[:, start:end, :] += d_x_kernel_chunk.view(B, chunk_len, C)
            
            # === Backward through wave_proj (sigmoid/tanh -> SiLU -> RMSNorm -> Linear) ===
            # freq = sigmoid(freq_pre) * (max_freq - min_freq) + min_freq
            # d_freq_pre = d_freq * (max_freq - min_freq) * sigmoid'(freq_pre)
            d_freq_pre = d_freq_chunk * (max_freq - min_freq) * sig_freq * (1.0 - sig_freq)
            # phase = tanh(phase_pre) * max_freq
            # d_phase_pre = d_phase * max_freq * tanh'(phase_pre) = d_phase * max_freq * (1 - tanh^2)
            d_phase_pre = d_phase_chunk * max_freq * (1.0 - tanh_phase * tanh_phase)
            d_wave_out = torch.stack([d_freq_pre, d_phase_pre], dim=2).view(M_chunk, 2 * H)
            
            # Backward through SiLU
            sig_wave = torch.sigmoid(wave_normed)
            d_wave_normed = d_wave_out * sig_wave * (1.0 + wave_normed * (1.0 - sig_wave))
            
            # Backward through RMSNorm
            wave_y_norm = wave_linear * wave_rrms
            d_wave_gamma += (d_wave_normed * wave_y_norm).sum(dim=0)
            inner_w = (d_wave_normed * wave_gamma * wave_y_norm).mean(dim=-1, keepdim=True)
            d_wave_linear = wave_rrms * (d_wave_normed * wave_gamma - wave_y_norm * inner_w)
            
            d_wave_pre = d_wave_linear  # For compatibility with existing code below
            
            d_wave_w_chunk = torch.zeros_like(wave_w)
            grid = (triton.cdiv(2 * H, BLOCK_N), triton.cdiv(C, BLOCK_K))
            linear_bwd_dw_kernel[grid](
                d_wave_pre, x_chunk, d_wave_w_chunk,
                M_chunk, 2 * H, C,
                d_wave_pre.stride(0), d_wave_pre.stride(1),
                x_chunk.stride(0), x_chunk.stride(1),
                d_wave_w_chunk.stride(0), d_wave_w_chunk.stride(1),
                BLOCK_N, BLOCK_K, BLOCK_M,
            )
            d_wave_w += d_wave_w_chunk
            d_wave_b += d_wave_pre.sum(0)
            
            d_x_wave_chunk = torch.zeros(M_chunk, C, device=x.device, dtype=x.dtype)
            grid = (triton.cdiv(M_chunk, BLOCK_M), triton.cdiv(C, BLOCK_K))
            linear_bwd_dx_kernel[grid](
                d_wave_pre, wave_w, d_x_wave_chunk,
                M_chunk, 2 * H, C,
                d_wave_pre.stride(0), d_wave_pre.stride(1),
                wave_w.stride(0), wave_w.stride(1),
                d_x_wave_chunk.stride(0), d_x_wave_chunk.stride(1),
                BLOCK_M, BLOCK_K, BLOCK_N,
            )
            d_x[:, start:end, :] += d_x_wave_chunk.view(B, chunk_len, C)
        
        return d_x, d_wave_w, d_wave_b, d_wave_gamma, d_kernel_w, d_kernel_b, d_kernel_gamma, d_out_w, \
               None, None, None, None, None, None, None  # 7 Nones for H, half_s, max_receptive, max_freq, min_freq, K, chunk_size


if __name__ == "__main__":
    if not HAS_TRITON:
        print("Triton not available")
        exit()
    
    import time
    
    device = "cuda"
    B, L, C = 4, 8192, 256
    H = 8
    warmup_iters = 3
    bench_iters = 10
    
    print(f"Config: B={B}, L={L}, C={C}, H={H}")
    print()
    
    def bench_model(model, x):
        # Warmup
        for _ in range(warmup_iters):
            out, _ = model(x)
            out.sum().backward()
            model.zero_grad()
            x.grad.zero_()
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward
        start = time.perf_counter()
        for _ in range(bench_iters):
            with torch.no_grad():
                out, _ = model(x)
        torch.cuda.synchronize()
        fwd_time = (time.perf_counter() - start) / bench_iters * 1000
        fwd_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        torch.cuda.reset_peak_memory_stats()
        
        # Forward + backward
        start = time.perf_counter()
        for _ in range(bench_iters):
            out, _ = model(x)
            out.sum().backward()
            model.zero_grad()
            x.grad.zero_()
        torch.cuda.synchronize()
        total_time = (time.perf_counter() - start) / bench_iters * 1000
        bwd_time = total_time - fwd_time
        bwd_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        return fwd_time, bwd_time, fwd_mem, bwd_mem
    
    # Default benchmark
    model = TritonGatherConv(channels=C, num_heads=H).to(device)
    x = torch.randn(B, L, C, device=device, requires_grad=True)
    fwd_time, bwd_time, fwd_mem, bwd_mem = bench_model(model, x)
    print(f"Default (chunk=1024): Fwd: {fwd_time:5.1f}ms ({fwd_mem:5.0f}MB)  Bwd: {bwd_time:5.1f}ms ({bwd_mem:5.0f}MB)")
    
    # Chunk size sweep
    print()
    print("=" * 70)
    print("CHUNK SIZE SWEEP")
    print("=" * 70)
    print(f"{'ChunkSize':>10} | {'Fwd(ms)':>8} {'Bwd(ms)':>8} {'FwdMem':>8} {'BwdMem':>8}")
    print("-" * 70)
    
    chunk_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    
    for cs in chunk_sizes:
        model_cs = TritonGatherConv(channels=C, num_heads=H, chunk_size=cs).to(device)
        model_cs.load_state_dict(model.state_dict())
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        
        fwd, bwd, fwd_m, bwd_m = bench_model(model_cs, x)
        print(f"{cs:>10} | {fwd:>8.1f} {bwd:>8.1f} {fwd_m:>8.0f} {bwd_m:>8.0f}")
        
        del model_cs, x
        torch.cuda.empty_cache()
    
    print("=" * 70)
