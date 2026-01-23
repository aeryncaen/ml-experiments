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
    def gather_conv_fwd_kernel(
        x_ptr, out_ptr,
        freq_ptr, phase_ptr,
        kernel_ptr,
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
        half_s: tl.constexpr, max_receptive: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        freq_avg = tl.load(freq_ptr + pid_b * L + pid_l)
        phase_avg = tl.load(phase_ptr + pid_b * L + pid_l)
        
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        k_offs = tl.arange(0, K)
        kernel_base = pid_b * L * H * K + pid_l * H * K + pid_h * K
        kernel_vals = tl.load(kernel_ptr + kernel_base + k_offs, mask=k_offs < K, other=0.0)
        
        out_h = tl.zeros((BLOCK_D,), dtype=tl.float32)
        kernel_sum = 0.0
        
        for s_idx in range(S):
            s_off = s_idx - half_s
            sample_pos_f = pid_l + s_off * freq_avg + phase_avg
            
            valid = (sample_pos_f >= 0.0) & (sample_pos_f < L)
            sample_idx = tl.maximum(tl.minimum(sample_pos_f, L - 1.0), 0.0).to(tl.int32)
            
            rel_pos = tl.abs(s_off * freq_avg)
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
            kernel_sum += kernel_w
            
            val_base = pid_b * L * C + sample_idx * C + pid_h * D
            vals = tl.load(x_ptr + val_base + d_offs, mask=d_mask, other=0.0)
            out_h += vals * kernel_w
        
        kernel_sum = tl.maximum(kernel_sum, 1e-8)
        out_h = out_h / kernel_sum
        
        out_base = pid_b * L * C + pid_l * C + pid_h * D
        tl.store(out_ptr + out_base + d_offs, out_h, mask=d_mask)


class TritonGatherConv(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        max_samples: int = 32,
        max_freq: float = 16.0,
        min_freq: float = 1.0,
        max_kernel_size: int = 64,
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
        D = self.head_dim
        S = self.num_samples
        K = self.max_kernel_size
        
        wave_out = F.silu(self.wave_proj(x)).view(B, L, 2, H)
        freq = torch.sigmoid(wave_out[:, :, 0, :]) * (self.max_freq - self.min_freq) + self.min_freq
        phase = torch.tanh(wave_out[:, :, 1, :]) * self.max_freq
        freq_avg = freq.mean(dim=-1).contiguous()
        phase_avg = phase.mean(dim=-1).contiguous()
        
        kernel = F.silu(self.kernel_proj(x)).view(B, L, H, K).contiguous()
        
        out = torch.zeros_like(x)
        
        BLOCK_D = triton.next_power_of_2(D)
        
        grid = (B, L, H)
        gather_conv_fwd_kernel[grid](
            x, out,
            freq_avg, phase_avg,
            kernel,
            B, L, C, H, D, S, K,
            self.half_s, self.max_receptive,
            BLOCK_D,
        )
        
        out = F.silu(self.out_proj(out))
        return out, {}


if __name__ == "__main__":
    if not HAS_TRITON:
        print("Triton not available")
        exit()
    
    import time
    
    device = "cuda"
    B, L, C = 4, 8192, 256
    H = 8
    
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
