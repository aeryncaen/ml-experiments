"""
AdaptiveLocalConv: Per-position learned window size, offset, and kernel weights.
Each position learns: receptive field width, center offset, and kernel weights.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def llama_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)


class AdaptiveLocalConv(nn.Module):
    """
    Per-position adaptive local convolution with:
    - Learned window size in [1, sqrt(L)]
    - Learned center offset in [-sqrt(L), +sqrt(L)] (deformable)
    - Projected kernel weights interpolated by relative position
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        max_kernel_size: int = 64,
        min_window: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.max_kernel_size = max_kernel_size
        self.min_window = min_window
        
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
        nn.init.xavier_uniform_(self.kernel_proj.weight, gain=0.1)
        nn.init.zeros_(self.kernel_proj.bias)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, L, C = x.shape
        H = self.num_heads
        D = self.head_dim
        K = self.max_kernel_size
        
        max_window = min(int(math.sqrt(L)), K)
        half_window_max = max_window // 2
        max_offset = int(math.sqrt(L))
        
        # Predict window sizes: (B, L, H) in [min_window, max_window]
        window_pre = self.window_proj(x)
        window_normed = llama_rmsnorm(window_pre, self.window_gamma)
        window_raw = torch.sigmoid(window_normed)
        window_sizes = self.min_window + window_raw * (max_window - self.min_window)
        
        # Predict center offsets: (B, L, H) in [-max_offset, +max_offset]
        offset_pre = self.offset_proj(x)
        offset_normed = llama_rmsnorm(offset_pre, self.offset_gamma)
        center_offsets = torch.tanh(offset_normed) * max_offset
        
        # Project kernel weights: (B, L, H, K)
        kernel_pre = self.kernel_proj(x)
        kernel_normed = llama_rmsnorm(kernel_pre, self.kernel_gamma)
        kernel_weights = F.silu(kernel_normed).view(B, L, H, K)
        
        # Project values: (B, L, H, D)
        v = self.v_proj(x).view(B, L, H, D)
        
        # Build neighbor offsets (relative to window center)
        local_offsets = torch.arange(-half_window_max, half_window_max + 1, device=x.device, dtype=x.dtype)
        num_offsets = local_offsets.shape[0]
        
        # Neighbor positions = position + center_offset + local_offset
        # positions: (L,), center_offsets: (B, L, H), local_offsets: (num_offsets,)
        positions = torch.arange(L, device=x.device, dtype=x.dtype).view(1, L, 1, 1)
        center_offsets_expanded = center_offsets.unsqueeze(-1)  # (B, L, H, 1)
        local_offsets_expanded = local_offsets.view(1, 1, 1, num_offsets)
        
        neighbor_positions = positions + center_offsets_expanded + local_offsets_expanded  # (B, L, H, num_offsets)
        
        # Valid mask and clamped positions for gathering
        valid_mask = (neighbor_positions >= 0) & (neighbor_positions < L)
        neighbor_positions_clamped = neighbor_positions.clamp(0, L - 1)
        
        # Soft window mask based on distance from window center (not from self)
        # rel_dist: 0 at window center, 1 at window edge
        abs_local_offsets = local_offsets.abs().view(1, 1, 1, num_offsets)
        half_windows = (window_sizes / 2).unsqueeze(-1)
        rel_dist = abs_local_offsets / (half_windows + 1e-6)
        
        sharpness = 5.0
        window_mask = torch.sigmoid(sharpness * (1.0 - rel_dist)) * valid_mask.float()
        
        # Interpolate kernel weights by relative position in window
        kernel_idx_float = rel_dist.clamp(0, 1) * (K - 1)
        idx_floor = kernel_idx_float.long().clamp(0, K - 2)
        idx_ceil = idx_floor + 1
        w_ceil = kernel_idx_float - idx_floor.float()
        w_floor = 1.0 - w_ceil
        
        k_floor = kernel_weights.gather(-1, idx_floor)
        k_ceil = kernel_weights.gather(-1, idx_ceil)
        kernel_interp = k_floor * w_floor + k_ceil * w_ceil
        
        # Final attention: kernel * window_mask, normalized
        attn_weights = kernel_interp * window_mask
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-6)
        
        # Bilinear interpolation for gathering (since positions are continuous)
        pos_floor = neighbor_positions_clamped.floor().long().clamp(0, L - 1)
        pos_ceil = (pos_floor + 1).clamp(0, L - 1)
        pos_frac = neighbor_positions_clamped - pos_floor.float()
        
        # Gather values at floor and ceil positions
        v_flat = v.transpose(1, 2).reshape(B * H, L, D)
        
        floor_idx = pos_floor.permute(0, 2, 1, 3).reshape(B * H, L, num_offsets)
        ceil_idx = pos_ceil.permute(0, 2, 1, 3).reshape(B * H, L, num_offsets)
        
        v_floor = v_flat.gather(1, floor_idx.unsqueeze(-1).expand(-1, -1, -1, D))
        v_ceil = v_flat.gather(1, ceil_idx.unsqueeze(-1).expand(-1, -1, -1, D))
        
        # Interpolate
        pos_frac_flat = pos_frac.permute(0, 2, 1, 3).reshape(B * H, L, num_offsets, 1)
        v_neighbors = v_floor * (1 - pos_frac_flat) + v_ceil * pos_frac_flat
        v_neighbors = v_neighbors.view(B, H, L, num_offsets, D).permute(0, 2, 1, 3, 4)
        
        # Apply attention: (B, L, H, D)
        output = (attn_weights.unsqueeze(-1) * v_neighbors).sum(dim=3)
        output = output.reshape(B, L, C)
        output = F.silu(self.out_proj(output))
        
        return output, {
            'window_sizes': window_sizes.detach(),
            'center_offsets': center_offsets.detach(),
            'max_window': torch.tensor(max_window),
            'max_offset': torch.tensor(max_offset),
        }


class AdaptiveLocalConvBlock(nn.Module):
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        max_kernel_size: int = 64,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.conv = AdaptiveLocalConv(channels, num_heads, max_kernel_size)
        self.norm2 = nn.LayerNorm(channels)
        
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        h, info = self.conv(self.norm1(x))
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x, info


if __name__ == "__main__":
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()
    
    B, L, C = 4, 1024, 256
    H = 8
    K = 64
    
    print(f"Config: B={B}, L={L}, C={C}, H={H}, K={K}")
    print(f"Max window (sqrt(L)): {int(math.sqrt(L))}")
    print()
    
    model = AdaptiveLocalConv(channels=C, num_heads=H, max_kernel_size=K).to(device)
    
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    
    torch.manual_seed(42)
    x = torch.randn(B, L, C, device=device)
    
    out, info = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Window sizes range: [{info['window_sizes'].min().item():.2f}, {info['window_sizes'].max().item():.2f}]")
    print(f"Center offsets range: [{info['center_offsets'].min().item():.2f}, {info['center_offsets'].max().item():.2f}]")
    print(f"Max window: {info['max_window'].item()}, Max offset: {info['max_offset'].item()}")
    print(f"Output has NaN: {torch.isnan(out).any().item()}")
    print(f"Output has Inf: {torch.isinf(out).any().item()}")
    
    x_grad = x.clone().requires_grad_(True)
    out, _ = model(x_grad)
    out.sum().backward()
    
    assert x_grad.grad is not None
    print(f"Input grad has NaN: {torch.isnan(x_grad.grad).any().item()}")
    print(f"window_proj grad: {model.window_proj.weight.grad.abs().sum().item():.2e}")  # type: ignore
    print(f"offset_proj grad: {model.offset_proj.weight.grad.abs().sum().item():.2e}")  # type: ignore
    print(f"kernel_proj grad: {model.kernel_proj.weight.grad.abs().sum().item():.2e}")  # type: ignore
    print(f"v_proj grad: {model.v_proj.weight.grad.abs().sum().item():.2e}")  # type: ignore
    print()
    
    print("=" * 60)
    print("BENCHMARKS")
    print("=" * 60)
    
    warmup, iters = 10, 50
    
    def bench_fwd(model, x):
        for _ in range(warmup):
            out, _ = model(x)
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        for _ in range(iters):
            out, _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - start) / iters * 1000
        fwd_mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        return fwd_ms, fwd_mem
    
    def bench_bwd(model, x):
        x_grad = x.clone().requires_grad_(True)
        for _ in range(warmup):
            out, _ = model(x_grad)
            out.sum().backward()
            model.zero_grad()
            x_grad.grad = None
        
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        for _ in range(iters):
            out, _ = model(x_grad)
            out.sum().backward()
            model.zero_grad()
            x_grad.grad = None
            if device == "cuda":
                torch.cuda.synchronize()
        total_ms = (time.perf_counter() - start) / iters * 1000
        bwd_mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        return total_ms, bwd_mem
    
    print(f"{'L':>8} {'sqrt(L)':>8} | {'Fwd(ms)':>8} {'Fwd+Bwd':>8} {'Mem(MB)':>8}")
    print("-" * 60)
    
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    
    for L in seq_lengths:
        if device == "cuda":
            torch.cuda.empty_cache()
        
        try:
            x = torch.randn(B, L, C, device=device)
            fwd_ms, fwd_mem = bench_fwd(model, x)
            total_ms, bwd_mem = bench_bwd(model, x)
            print(f"{L:>8} {int(math.sqrt(L)):>8} | {fwd_ms:>8.2f} {total_ms:>8.2f} {bwd_mem:>8.0f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{L:>8} {int(math.sqrt(L)):>8} | OOM")
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise
    
    print("=" * 60)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
