"""
GatherConv: Learned position sampling with interpolated kernel convolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul


class GatherConvND(nn.Module):
    """
    Gather samples at learned positions, convolve with interpolated kernel.
    
    For each position: predict freq/phase → compute sample positions →
    gather values → project kernel and interpolate to positions → apply.
    """
    
    stride_grid: torch.Tensor
    
    def __init__(
        self,
        channels: int,
        ndim: int = 1,
        max_samples: int = 32,
        num_heads: int = 1,
        max_freq: float = 16.0,
        min_freq: float = 1.0,
        max_kernel_size: int = 64,
        chunk_size: int = 2048,
    ):
        super().__init__()
        self.channels = channels
        self.ndim = ndim
        self.max_samples = max_samples
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.chunk_size = chunk_size
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.max_kernel_size = max_kernel_size
        
        H = num_heads
        
        self.wave_proj = nn.Linear(channels, 2 * H)
        self.kernel_proj = nn.Linear(channels, H * max_kernel_size)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        
        samples_per_dim = max(3, int(max_samples ** (1.0 / ndim)))
        half_s = samples_per_dim // 2
        offsets_1d = torch.arange(-half_s, half_s + 1).float()
        
        if ndim == 1:
            stride_grid = offsets_1d.unsqueeze(-1)
        else:
            grids = torch.meshgrid(*[offsets_1d] * ndim, indexing='ij')
            stride_grid = torch.stack([g.flatten() for g in grids], dim=-1)
        
        self.register_buffer('stride_grid', stride_grid)
        self.num_samples = stride_grid.shape[0]
        
        self.max_offset = half_s
        self.max_receptive = half_s * max_freq
        
        nn.init.zeros_(self.wave_proj.weight)
        nn.init.zeros_(self.wave_proj.bias)
        nn.init.xavier_uniform_(self.kernel_proj.weight, gain=0.1)
        nn.init.zeros_(self.kernel_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
    
    def _forward_chunk(
        self,
        x_flat: torch.Tensor,
        x_chunk: torch.Tensor,
        chunk_start: int,
        chunk_end: int,
        spatial: tuple[int, ...],
        L: int,
    ) -> torch.Tensor:
        B = x_flat.shape[0]
        C = x_flat.shape[-1]
        H = self.num_heads
        D = self.head_dim
        S = self.num_samples
        K = self.max_kernel_size
        chunk_len = chunk_end - chunk_start
        
        wave_params = F.silu(self.wave_proj(x_chunk)).view(B, chunk_len, 2, H)
        
        freq_raw = wave_params[:, :, 0, :]
        phase_raw = wave_params[:, :, 1, :]
        
        freq = torch.sigmoid(freq_raw) * (self.max_freq - self.min_freq) + self.min_freq
        phase = torch.tanh(phase_raw) * self.max_freq
        
        freq_avg = freq.mean(dim=-1)
        phase_avg = phase.mean(dim=-1)
        
        if self.ndim == 1:
            centers = torch.arange(chunk_start, chunk_end, device=x_flat.device, dtype=x_flat.dtype)
            sample_pos = (
                centers.view(1, chunk_len, 1) + 
                self.stride_grid[:, 0].view(1, 1, S) * freq_avg.unsqueeze(-1) + 
                phase_avg.unsqueeze(-1)
            )
            valid_mask = (sample_pos >= 0) & (sample_pos < L)
            sample_idx = sample_pos.long().clamp(0, L - 1)
            
            rel_pos = self.stride_grid[:, 0].view(1, 1, S) * freq_avg.unsqueeze(-1) + phase_avg.unsqueeze(-1)
        else:
            coords = [torch.arange(s, device=x_flat.device, dtype=x_flat.dtype) for s in spatial]
            mesh = torch.meshgrid(*coords, indexing='ij')
            centers_all = torch.stack([m.flatten() for m in mesh], dim=-1)
            centers = centers_all[chunk_start:chunk_end]
            
            sample_pos = (
                centers.view(1, chunk_len, 1, self.ndim) + 
                self.stride_grid.view(1, 1, S, self.ndim) * freq_avg.view(B, chunk_len, 1, 1) + 
                phase_avg.view(B, chunk_len, 1, 1)
            )
            
            valid_mask = torch.ones(B, chunk_len, S, dtype=torch.bool, device=x_flat.device)
            for dim in range(self.ndim):
                valid_mask = valid_mask & (sample_pos[..., dim] >= 0) & (sample_pos[..., dim] < spatial[dim])
            
            sample_coords = sample_pos.long()
            for dim in range(self.ndim):
                sample_coords[..., dim] = sample_coords[..., dim].clamp(0, spatial[dim] - 1)
            
            strides = [1]
            for dim in range(self.ndim - 1, 0, -1):
                strides.insert(0, strides[0] * spatial[dim])
            sample_idx = sum(sample_coords[..., dim] * strides[dim] for dim in range(self.ndim))
            
            rel_pos_nd = self.stride_grid.view(1, 1, S, self.ndim) * freq_avg.view(B, chunk_len, 1, 1)
            rel_pos = rel_pos_nd.norm(dim=-1)
        
        batch_idx = torch.arange(B, device=x_flat.device).view(B, 1, 1).expand(B, chunk_len, S)
        values = x_flat[batch_idx, sample_idx].view(B, chunk_len, S, H, D).permute(0, 1, 3, 2, 4)
        
        kernel_max = F.silu(self.kernel_proj(x_chunk)).view(B, chunk_len, H, K)
        
        norm_pos = (rel_pos + self.max_receptive) / (2 * self.max_receptive)
        norm_pos = norm_pos.clamp(0, 1)
        
        idx_float = norm_pos * (K - 1)
        idx_floor = idx_float.long().clamp(0, K - 2)
        idx_ceil = idx_floor + 1
        w_ceil = (idx_float - idx_floor.float()).unsqueeze(2)
        w_floor = 1.0 - w_ceil
        
        idx_floor = idx_floor.unsqueeze(2).expand(B, chunk_len, H, S)
        idx_ceil = idx_ceil.unsqueeze(2).expand(B, chunk_len, H, S)
        
        k_floor = kernel_max.gather(-1, idx_floor)
        k_ceil = kernel_max.gather(-1, idx_ceil)
        
        kernel = k_floor * w_floor + k_ceil * w_ceil
        
        valid_mask = valid_mask.unsqueeze(2).expand(B, chunk_len, H, S)
        kernel = kernel * valid_mask.float()
        kernel = kernel / (kernel.sum(dim=-1, keepdim=True) + 1e-8)
        
        output = torch.einsum('blhsd,blhs->blhd', values, kernel)
        output = output.reshape(B, chunk_len, C)
        
        return output
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B = x.shape[0]
        spatial = x.shape[1:-1]
        C = x.shape[-1]
        L = reduce(mul, spatial, 1)
        
        x_flat = x.reshape(B, L, C)
        
        if L <= self.chunk_size:
            output = self._forward_chunk(x_flat, x_flat, 0, L, spatial, L)
        else:
            outputs = []
            for start in range(0, L, self.chunk_size):
                end = min(start + self.chunk_size, L)
                chunk_out = self._forward_chunk(x_flat, x_flat[:, start:end], start, end, spatial, L)
                outputs.append(chunk_out)
            output = torch.cat(outputs, dim=1)
        
        output = F.silu(self.out_proj(output)).reshape(B, *spatial, C)
        
        return output, {}


GatherConv1d = GatherConvND
GatherConv2d = lambda *args, **kwargs: GatherConvND(*args, ndim=2, **kwargs)
GatherConv3d = lambda *args, **kwargs: GatherConvND(*args, ndim=3, **kwargs)


if __name__ == "__main__":
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, L, C = 4, 8192, 256
    num_heads = 8
    max_samples = 32
    warmup_iters = 10
    bench_iters = 50
    
    print(f"Device: {device}")
    print(f"Shape: ({B}, {L}, {C}), heads={num_heads}, max_samples={max_samples}")
    print("-" * 60)
    
    conv = GatherConvND(
        channels=C,
        ndim=1,
        max_samples=max_samples,
        num_heads=num_heads,
    ).to(device)
    
    x = torch.randn(B, L, C, device=device)
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    for _ in range(warmup_iters):
        out, _ = conv(x)
        if device == "cuda":
            torch.cuda.synchronize()
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(bench_iters):
        out, _ = conv(x)
        if device == "cuda":
            torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - start) / bench_iters * 1000
    
    if device == "cuda":
        fwd_mem = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
    else:
        fwd_mem = 0
    
    x_grad = torch.randn(B, L, C, device=device, requires_grad=True)
    
    for _ in range(warmup_iters):
        out, _ = conv(x_grad)
        loss = out.sum()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
    
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(bench_iters):
        out, _ = conv(x_grad)
        loss = out.sum()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
    fwd_bwd_time = (time.perf_counter() - start) / bench_iters * 1000
    
    if device == "cuda":
        fwd_bwd_mem = torch.cuda.max_memory_allocated() / 1024**2
    else:
        fwd_bwd_mem = 0
    
    params = sum(p.numel() for p in conv.parameters())
    
    print(f"Parameters: {params:,}")
    print(f"Forward:     {fwd_time:.2f} ms | {fwd_mem:.1f} MB peak")
    print(f"Fwd+Bwd:     {fwd_bwd_time:.2f} ms | {fwd_bwd_mem:.1f} MB peak")
