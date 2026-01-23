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
except Exception as e:
    import sys
    print(f"triton_gather.py: Triton import failed: {type(e).__name__}: {e}", file=sys.stderr)
    HAS_TRITON = False


if HAS_TRITON:
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
        D = self.head_dim
        S = self.num_samples
        K = self.max_kernel_size
        chunk_size = min(self.chunk_size, L)
        
        out = torch.empty_like(x)
        BLOCK_D = triton.next_power_of_2(D)
        
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            chunk_len = end - start
            x_chunk = x[:, start:end, :]
            
            wave_out = F.silu(self.wave_proj(x_chunk))
            wave_out = wave_out.view(B, chunk_len, 2, H)
            freq = torch.sigmoid(wave_out[:, :, 0, :]) * (self.max_freq - self.min_freq) + self.min_freq
            phase = torch.tanh(wave_out[:, :, 1, :]) * self.max_freq
            
            kernel = F.silu(self.kernel_proj(x_chunk)).view(B, chunk_len, H, K).contiguous()
            
            hidden_chunk = torch.empty(B, chunk_len, C, device=x.device, dtype=x.dtype)
            
            grid = (B, chunk_len, H)
            gather_conv_fwd_kernel_chunked[grid](
                x, hidden_chunk,
                freq.contiguous(), phase.contiguous(),
                kernel,
                B, L, C, H, D, S, K,
                self.half_s, self.max_receptive,
                start,
                chunk_len,
                BLOCK_D,
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
