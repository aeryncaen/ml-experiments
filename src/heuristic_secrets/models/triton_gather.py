"""Triton kernels for GatherConv - fused gather + interpolated kernel convolution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def gather_conv_bwd_kernel(
        x_ptr, d_out_ptr, d_x_ptr,
        freq_ptr, phase_ptr,
        kernel_ptr,
        d_freq_ptr, d_kernel_ptr,
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
        half_s: tl.constexpr, max_receptive: tl.constexpr,
        chunk_start,
        chunk_len: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l_local = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        pid_l = chunk_start + pid_l_local
        
        freq_h = tl.load(freq_ptr + pid_b * chunk_len * H + pid_l_local * H + pid_h)
        phase_h = tl.load(phase_ptr + pid_b * chunk_len * H + pid_l_local * H + pid_h)
        
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        k_offs = tl.arange(0, K)
        kernel_base = pid_b * chunk_len * H * K + pid_l_local * H * K + pid_h * K
        kernel_vals = tl.load(kernel_ptr + kernel_base + k_offs, mask=k_offs < K, other=0.0)
        
        d_out_base = pid_b * chunk_len * C + pid_l_local * C + pid_h * D
        d_out_h = tl.load(d_out_ptr + d_out_base + d_offs, mask=d_mask, other=0.0)
        
        d_freq_acc = 0.0
        d_kernel_acc = tl.zeros((K,), dtype=tl.float32)
        
        for s_idx in range(S):
            s_off = s_idx - half_s
            sample_pos_f = pid_l + s_off * freq_h + phase_h
            
            valid = (sample_pos_f >= 0.0) & (sample_pos_f < L)
            valid_f = valid.to(tl.float32)
            sample_idx = tl.maximum(tl.minimum(sample_pos_f, L - 1.0), 0.0).to(tl.int32)
            
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
            
            val_base = pid_b * L * C + sample_idx * C + pid_h * D
            vals = tl.load(x_ptr + val_base + d_offs, mask=d_mask, other=0.0)
            
            d_x_contrib = d_out_h * kernel_w
            d_x_base = pid_b * L * C + sample_idx * C + pid_h * D
            tl.atomic_add(d_x_ptr + d_x_base + d_offs, d_x_contrib, mask=d_mask)
            
            d_kernel_w = tl.sum(d_out_h * vals) * valid_f
            
            d_kernel_acc += tl.where(k_offs == idx_floor, d_kernel_w * w_floor, 0.0)
            d_kernel_acc += tl.where(k_offs == idx_ceil, d_kernel_w * w_ceil, 0.0)
            
            d_w_ceil = d_kernel_w * (k_ceil - k_floor)
            d_idx_float = d_w_ceil
            d_norm_pos = d_idx_float * (K - 1)
            d_rel_pos = d_norm_pos / max_receptive * not_clamped
            
            s_off_f = s_off * 1.0
            d_freq_acc += d_rel_pos * tl.abs(s_off_f)
        
        tl.store(d_kernel_ptr + kernel_base + k_offs, d_kernel_acc, mask=k_offs < K)
        tl.store(d_freq_ptr + pid_b * chunk_len * H + pid_l_local * H + pid_h, d_freq_acc)
    
    @triton.jit
    def gather_conv_fwd_kernel_chunked(
        x_ptr, out_ptr,
        freq_ptr, phase_ptr,
        kernel_ptr,
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
        half_s: tl.constexpr, max_receptive: tl.constexpr,
        chunk_start,
        chunk_len: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l_local = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        pid_l = chunk_start + pid_l_local
        
        freq_h = tl.load(freq_ptr + pid_b * chunk_len * H + pid_l_local * H + pid_h)
        phase_h = tl.load(phase_ptr + pid_b * chunk_len * H + pid_l_local * H + pid_h)
        
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        k_offs = tl.arange(0, K)
        kernel_base = pid_b * chunk_len * H * K + pid_l_local * H * K + pid_h * K
        kernel_vals = tl.load(kernel_ptr + kernel_base + k_offs, mask=k_offs < K, other=0.0)
        
        out_h = tl.zeros((BLOCK_D,), dtype=tl.float32)
        
        for s_idx in range(S):
            s_off = s_idx - half_s
            sample_pos_f = pid_l + s_off * freq_h + phase_h
            
            valid = (sample_pos_f >= 0.0) & (sample_pos_f < L)
            sample_idx = tl.maximum(tl.minimum(sample_pos_f, L - 1.0), 0.0).to(tl.int32)
            
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
            kernel_w = kernel_w * valid.to(tl.float32)
            
            val_base = pid_b * L * C + sample_idx * C + pid_h * D
            vals = tl.load(x_ptr + val_base + d_offs, mask=d_mask, other=0.0)
            out_h += vals * kernel_w
        
        out_base = pid_b * chunk_len * C + pid_l_local * C + pid_h * D
        tl.store(out_ptr + out_base + d_offs, out_h, mask=d_mask)


HAS_FP8 = hasattr(torch, 'float8_e4m3fn')


def quantize_fp8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = t.abs().max() / 448.0  # E4M3 max is ~448
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    t_scaled = t / scale
    t_fp8 = t_scaled.to(torch.float8_e4m3fn)
    return t_fp8, scale


def dequantize_fp8(t_fp8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t_fp8.to(dtype) * scale


def quantize_int4(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t_min = t.min()
    t_max = t.max()
    scale = (t_max - t_min) / 15.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    
    t_norm = ((t - t_min) / scale).round().clamp(0, 15).to(torch.uint8)
    
    t_flat_q = t_norm.flatten()
    if t_flat_q.numel() % 2 == 1:
        t_flat_q = F.pad(t_flat_q, (0, 1))
    packed = (t_flat_q[0::2] << 4) | t_flat_q[1::2]
    
    return packed, scale, t_min


def dequantize_int4(packed: torch.Tensor, scale: torch.Tensor, t_min: torch.Tensor, shape: tuple, dtype: torch.dtype) -> torch.Tensor:
    hi = (packed >> 4).to(dtype)
    lo = (packed & 0x0F).to(dtype)
    t_flat = torch.stack([hi, lo], dim=-1).flatten()
    
    numel = 1
    for s in shape:
        numel *= s
    t_flat = t_flat[:numel]
    
    return (t_flat * scale + t_min).view(shape)


class GatherConvFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freq, phase, kernel, chunk_start, half_s, max_receptive, quantize_bwd):
        B, L, C = x.shape
        _, chunk_len, H = freq.shape
        K = kernel.shape[-1]
        D = C // H
        S = 2 * half_s + 1
        
        BLOCK_D = triton.next_power_of_2(D)
        out = torch.empty(B, chunk_len, C, device=x.device, dtype=x.dtype)
        
        grid = (B, chunk_len, H)
        gather_conv_fwd_kernel_chunked[grid](
            x, out,
            freq, phase, kernel,
            B, L, C, H, D, S, K,
            half_s, max_receptive,
            chunk_start, chunk_len, BLOCK_D,
        )
        
        ctx.quantize_bwd = quantize_bwd
        ctx.x_shape = x.shape
        ctx.kernel_shape = kernel.shape
        ctx.orig_dtype = x.dtype
        
        if quantize_bwd:
            if HAS_FP8:
                x_q, x_scale = quantize_fp8(x)
                kernel_q, kernel_scale = quantize_fp8(kernel)
                ctx.save_for_backward(x_q, x_scale, freq, phase, kernel_q, kernel_scale)
                ctx.use_fp8 = True
            else:
                x_q, x_scale, x_min = quantize_int4(x)
                kernel_q, kernel_scale, kernel_min = quantize_int4(kernel)
                ctx.save_for_backward(x_q, x_scale, x_min, freq, phase, kernel_q, kernel_scale, kernel_min)
                ctx.use_fp8 = False
        else:
            ctx.save_for_backward(x, freq, phase, kernel)
        
        ctx.chunk_start = chunk_start
        ctx.half_s = half_s
        ctx.max_receptive = max_receptive
        
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        chunk_start = ctx.chunk_start
        half_s = ctx.half_s
        max_receptive = ctx.max_receptive
        dtype = ctx.orig_dtype
        
        if ctx.quantize_bwd:
            if ctx.use_fp8:
                x_q, x_scale, freq, phase, kernel_q, kernel_scale = ctx.saved_tensors
                x = dequantize_fp8(x_q, x_scale, dtype)
                kernel = dequantize_fp8(kernel_q, kernel_scale, dtype)
            else:
                x_q, x_scale, x_min, freq, phase, kernel_q, kernel_scale, kernel_min = ctx.saved_tensors
                x = dequantize_int4(x_q, x_scale, x_min, ctx.x_shape, dtype)
                kernel = dequantize_int4(kernel_q, kernel_scale, kernel_min, ctx.kernel_shape, dtype)
        else:
            x, freq, phase, kernel = ctx.saved_tensors
        
        B, L, C = x.shape
        _, chunk_len, H = freq.shape
        K = kernel.shape[-1]
        D = C // H
        S = 2 * half_s + 1
        
        BLOCK_D = triton.next_power_of_2(D)
        d_out = d_out.contiguous()
        
        d_x = torch.zeros_like(x)
        d_freq = torch.zeros_like(freq)
        d_kernel = torch.zeros_like(kernel)
        
        grid = (B, chunk_len, H)
        
        gather_conv_bwd_kernel[grid](
            x, d_out, d_x,
            freq, phase, kernel,
            d_freq, d_kernel,
            B, L, C, H, D, S, K,
            half_s, max_receptive,
            chunk_start, chunk_len, BLOCK_D,
        )
        
        return d_x, d_freq, None, d_kernel, None, None, None, None


gather_conv_fn = GatherConvFunction.apply


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
        self.num_samples = 2 * self.half_s + 1
        self.max_receptive = self.half_s * max_freq
        
        self.wave_proj = nn.Linear(channels, 2 * num_heads)
        self.kernel_proj = nn.Linear(channels, num_heads * max_kernel_size)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        
        nn.init.zeros_(self.wave_proj.weight)
        nn.init.zeros_(self.wave_proj.bias)
        nn.init.xavier_uniform_(self.kernel_proj.weight, gain=0.1)
        nn.init.zeros_(self.kernel_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, L, C = x.shape
        H = self.num_heads
        K = self.max_kernel_size
        chunk_size = min(self.chunk_size, L)
        
        out = torch.empty_like(x)
        
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            chunk_len = end - start
            x_chunk = x[:, start:end, :]
            
            wave_out = F.silu(self.wave_proj(x_chunk))
            wave_out = wave_out.view(B, chunk_len, 2, H)
            freq = torch.sigmoid(wave_out[:, :, 0, :]) * (self.max_freq - self.min_freq) + self.min_freq
            phase = torch.tanh(wave_out[:, :, 1, :]) * self.max_freq
            
            kernel = F.silu(self.kernel_proj(x_chunk)).view(B, chunk_len, H, K).contiguous()
            
            hidden_chunk = gather_conv_fn(
                x, freq.contiguous(), phase.contiguous(), kernel,
                start, self.half_s, self.max_receptive,
            )
            
            out[:, start:end, :] = F.silu(F.linear(hidden_chunk, self.out_proj.weight))
        
        return out, {}


if __name__ == "__main__":
    if not HAS_TRITON:
        print("Triton not available")
        exit()
    
    import time
    import sys
    
    device = "cuda"
    B, L, C = 4, 8192, 256
    H = 8
    
    if "--trace" in sys.argv:
        torch.cuda.reset_peak_memory_stats()
        
        def mem():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1024**2
        
        model = TritonGatherConv(channels=C, num_heads=H).to(device)
        print(f"After model init: {mem():.1f} MB")
        
        x = torch.randn(B, L, C, device=device)
        print(f"After x alloc: {mem():.1f} MB")
        
        torch.cuda.reset_peak_memory_stats()
        out = torch.empty_like(x)
        print(f"After out alloc: {mem():.1f} MB")
        
        torch.cuda.reset_peak_memory_stats()
        chunk_size = 1024
        x_chunk = x[:, :chunk_size, :]
        wave_out = F.silu(model.wave_proj(x_chunk))
        print(f"After wave_proj: {mem():.1f} MB")
        
        torch.cuda.reset_peak_memory_stats()
        kernel = F.silu(model.kernel_proj(x_chunk))
        print(f"After kernel_proj: {mem():.1f} MB")
        
        torch.cuda.reset_peak_memory_stats()
        out_final = F.silu(model.out_proj(x))
        print(f"After out_proj (full L): {mem():.1f} MB")
        
        del wave_out, kernel, out_final, out, x_chunk
        torch.cuda.empty_cache()
        
        torch.cuda.reset_peak_memory_stats()
        out, _ = model(x)
        print(f"Full forward: {mem():.1f} MB")
    else:
        model = TritonGatherConv(channels=C, num_heads=H).to(device)
        x = torch.randn(B, L, C, device=device)
        
        for _ in range(10):
            out, _ = model(x)
            torch.cuda.synchronize()
        
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(20):
            out, _ = model(x)
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / 20 * 1000
        mem = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"Triton GatherConv: {elapsed:.2f} ms, {mem:.1f} MB")
        print(f"Output shape: {out.shape}")
