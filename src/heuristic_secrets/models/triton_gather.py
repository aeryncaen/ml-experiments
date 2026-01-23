"""Triton kernels for GatherConv - fused gather + interpolated kernel convolution."""

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
    @triton.jit
    def gather_conv_fwd_kernel(
        x_ptr, out_ptr,
        freq_ptr, phase_ptr, kernel_ptr,
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
        half_s: tl.constexpr, max_receptive: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        freq_off = pid_b * L * H + pid_l * H + pid_h
        freq_h = tl.load(freq_ptr + freq_off)
        phase_h = tl.load(phase_ptr + freq_off)
        
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        k_offs = tl.arange(0, K)
        kernel_base = pid_b * L * H * K + pid_l * H * K + pid_h * K
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
        
        out_base = pid_b * L * C + pid_l * C + pid_h * D
        tl.store(out_ptr + out_base + d_offs, out_h, mask=d_mask)

    @triton.jit
    def gather_conv_bwd_kernel(
        x_ptr, d_out_ptr, d_x_ptr,
        freq_ptr, phase_ptr, kernel_ptr,
        d_freq_ptr, d_kernel_ptr,
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
        half_s: tl.constexpr, max_receptive: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        freq_off = pid_b * L * H + pid_l * H + pid_h
        freq_h = tl.load(freq_ptr + freq_off)
        phase_h = tl.load(phase_ptr + freq_off)
        
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        k_offs = tl.arange(0, K)
        kernel_base = pid_b * L * H * K + pid_l * H * K + pid_h * K
        kernel_vals = tl.load(kernel_ptr + kernel_base + k_offs, mask=k_offs < K, other=0.0)
        
        d_out_base = pid_b * L * C + pid_l * C + pid_h * D
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
            
            # d_x via atomic add (scatter)
            d_x_contrib = d_out_h * kernel_w
            tl.atomic_add(d_x_ptr + val_base + d_offs, d_x_contrib, mask=d_mask)
            
            # d_kernel
            d_kernel_w = tl.sum(d_out_h * vals) * valid_f
            d_kernel_acc += tl.where(k_offs == idx_floor, d_kernel_w * w_floor, 0.0)
            d_kernel_acc += tl.where(k_offs == idx_ceil, d_kernel_w * w_ceil, 0.0)
            
            # d_freq (through kernel interpolation)
            d_w_ceil = d_kernel_w * (k_ceil - k_floor)
            d_idx_float = d_w_ceil
            d_norm_pos = d_idx_float * (K - 1)
            d_rel_pos = d_norm_pos / max_receptive * not_clamped
            s_off_f = s_off * 1.0
            d_freq_acc += d_rel_pos * tl.abs(s_off_f)
        
        tl.store(d_kernel_ptr + kernel_base + k_offs, d_kernel_acc, mask=k_offs < K)
        tl.store(d_freq_ptr + freq_off, d_freq_acc)


HAS_FP8 = hasattr(torch, 'float8_e4m3fn')


def quantize_fp8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = t.abs().max() / 448.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return t.div(scale).to(torch.float8_e4m3fn), scale


def dequantize_fp8(t_fp8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t_fp8.to(dtype) * scale


class GatherConvOp(torch.autograd.Function):
    """Single autograd boundary for gather-conv operation.
    
    Takes pre-computed freq, phase, kernel. Returns gathered output.
    Saves tensors ONCE for backward.
    """
    
    @staticmethod
    def forward(ctx, x, freq, phase, kernel, half_s, max_receptive, quantize_bwd):
        B, L, C = x.shape
        H = freq.shape[-1]
        K = kernel.shape[-1]
        D = C // H
        S = 2 * half_s + 1
        BLOCK_D = triton.next_power_of_2(D)
        
        out = torch.empty_like(x)
        grid = (B, L, H)
        
        gather_conv_fwd_kernel[grid](
            x, out,
            freq, phase, kernel,
            B, L, C, H, D, S, K,
            half_s, max_receptive, BLOCK_D,
        )
        
        # Save for backward - ONCE
        ctx.half_s = half_s
        ctx.max_receptive = max_receptive
        ctx.quantize_bwd = quantize_bwd
        
        if quantize_bwd and HAS_FP8:
            x_q, x_scale = quantize_fp8(x)
            kernel_q, kernel_scale = quantize_fp8(kernel)
            ctx.save_for_backward(x_q, x_scale, freq, phase, kernel_q, kernel_scale)
            ctx.use_quant = True
        else:
            ctx.save_for_backward(x, freq, phase, kernel)
            ctx.use_quant = False
        
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        half_s = ctx.half_s
        max_receptive = ctx.max_receptive
        
        if ctx.use_quant:
            x_q, x_scale, freq, phase, kernel_q, kernel_scale = ctx.saved_tensors
            x = dequantize_fp8(x_q, x_scale, d_out.dtype)
            kernel = dequantize_fp8(kernel_q, kernel_scale, d_out.dtype)
        else:
            x, freq, phase, kernel = ctx.saved_tensors
        
        B, L, C = x.shape
        H = freq.shape[-1]
        K = kernel.shape[-1]
        D = C // H
        S = 2 * half_s + 1
        BLOCK_D = triton.next_power_of_2(D)
        
        d_out = d_out.contiguous()
        d_x = torch.zeros_like(x)
        d_freq = torch.zeros_like(freq)
        d_kernel = torch.zeros_like(kernel)
        
        grid = (B, L, H)
        gather_conv_bwd_kernel[grid](
            x, d_out, d_x,
            freq, phase, kernel,
            d_freq, d_kernel,
            B, L, C, H, D, S, K,
            half_s, max_receptive, BLOCK_D,
        )
        
        return d_x, d_freq, None, d_kernel, None, None, None


class TritonGatherConv(nn.Module):
    """GatherConv using Triton kernels."""
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        max_samples: int = 32,
        max_freq: float = 16.0,
        min_freq: float = 1.0,
        max_kernel_size: int = 64,
        quantize_bwd: bool = False,
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
        self.quantize_bwd = quantize_bwd
        
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
        
        # Compute freq/phase/kernel (autograd tracks these normally)
        wave_out = F.silu(self.wave_proj(x))  # (B, L, 2*H)
        wave_out = wave_out.view(B, L, 2, H)
        freq = torch.sigmoid(wave_out[:, :, 0, :]) * (self.max_freq - self.min_freq) + self.min_freq
        phase = torch.tanh(wave_out[:, :, 1, :]) * self.max_freq
        kernel = F.silu(self.kernel_proj(x)).view(B, L, H, K)
        
        # Single autograd boundary for gather-conv
        hidden = GatherConvOp.apply(
            x, freq.contiguous(), phase.contiguous(), kernel.contiguous(),
            self.half_s, self.max_receptive, self.quantize_bwd,
        )
        
        out = F.silu(self.out_proj(hidden))
        return out, {}


if __name__ == "__main__":
    if not HAS_TRITON:
        print("Triton not available")
        exit()
    
    import time
    
    device = "cuda"
    B, L, C = 4, 8192, 256
    H = 8
    
    print(f"Config: B={B}, L={L}, C={C}, H={H}")
    print(f"FP8 available: {HAS_FP8}")
    print()
    
    for quant in [False, True]:
        label = "Triton+Q" if quant else "Triton"
        model = TritonGatherConv(channels=C, num_heads=H, quantize_bwd=quant).to(device)
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        
        # Warmup
        for _ in range(5):
            out, _ = model(x)
            out.sum().backward()
            model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward only
        start = time.perf_counter()
        for _ in range(20):
            with torch.no_grad():
                out, _ = model(x)
        torch.cuda.synchronize()
        fwd_time = (time.perf_counter() - start) / 20 * 1000
        fwd_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        torch.cuda.reset_peak_memory_stats()
        
        # Forward + backward
        start = time.perf_counter()
        for _ in range(20):
            out, _ = model(x)
            out.sum().backward()
            model.zero_grad()
            x.grad.zero_()
        torch.cuda.synchronize()
        total_time = (time.perf_counter() - start) / 20 * 1000
        bwd_time = total_time - fwd_time
        bwd_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"{label:10s}: Fwd {fwd_time:5.1f}ms ({fwd_mem:5.0f}MB)  Bwd {bwd_time:5.1f}ms ({bwd_mem:5.0f}MB)")
