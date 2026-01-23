"""
AdaptiveLocalConv: Per-position learned window size, offset, and kernel weights.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=1)
        scale = torch.sigmoid(self.fc2(F.silu(self.fc1(scale))))
        return x * scale.unsqueeze(1)


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
        scale_power: float = 0.30,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.max_kernel_size = max_kernel_size
        self.min_window = min_window
        self.scale_power = scale_power
        
        H = num_heads
        K = max_kernel_size
        
        self.window_proj = nn.Linear(channels, H)
        self.window_gamma = nn.Parameter(torch.ones(H))
        self.offset_proj = nn.Linear(channels, H)
        self.offset_gamma = nn.Parameter(torch.ones(H))
        self.kernel_proj = nn.Linear(channels, H * K)
        self.kernel_gamma = nn.Parameter(torch.ones(H * K))
        self.v_proj = nn.Linear(channels, channels)
        self.se = SqueezeExcite1D(channels)
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
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B, L, C = x.shape
        H = self.num_heads
        D = self.head_dim
        K = self.max_kernel_size
        
        max_window = min(int(L ** self.scale_power), K)
        half_window_max = max_window // 2
        max_offset = int(L ** self.scale_power)
        
        window_pre = self.window_proj(x)
        window_normed = llama_rmsnorm(window_pre, self.window_gamma)
        window_raw = torch.sigmoid(window_normed)
        window_sizes = self.min_window + window_raw * (max_window - self.min_window)
        half_windows = window_sizes / 2
        
        offset_pre = self.offset_proj(x)
        offset_normed = llama_rmsnorm(offset_pre, self.offset_gamma)
        center_offsets = torch.tanh(offset_normed) * max_offset
        
        kernel_pre = self.kernel_proj(x)
        kernel_normed = llama_rmsnorm(kernel_pre, self.kernel_gamma)
        kernel_weights = F.silu(kernel_normed).view(B, L, H, K)
        
        v = self.v_proj(x)
        
        positions = torch.arange(L, device=x.device, dtype=x.dtype)
        batch_idx = torch.arange(B, device=x.device).view(B, 1).expand(B, L)
        local_offsets = torch.arange(-half_window_max, half_window_max + 1, device=x.device, dtype=x.dtype)
        
        output = torch.zeros(B, L, C, device=x.device, dtype=x.dtype)
        
        for h in range(H):
            v_head = v[..., h * D : (h + 1) * D]
            out_head = output[..., h * D : (h + 1) * D]
            
            center_off_h = center_offsets[:, :, h]
            half_win_h = half_windows[:, :, h].clamp(min=0.5)
            kernel_h = kernel_weights[:, :, h, :]
            
            weight_sum = torch.zeros(B, L, device=x.device, dtype=x.dtype)
            
            for s_off in local_offsets.tolist():
                neighbor_pos = positions.view(1, L) + center_off_h + s_off
                
                valid = (neighbor_pos >= 0) & (neighbor_pos < L)
                valid_f = valid.float()
                
                rel_dist = abs(s_off) / half_win_h
                window_w = torch.exp(-rel_dist * rel_dist)
                
                norm_pos = rel_dist.clamp(0, 1) * (K - 1)
                idx_floor = norm_pos.long().clamp(0, K - 2)
                idx_ceil = idx_floor + 1
                w_ceil = norm_pos - idx_floor.float()
                w_floor = 1.0 - w_ceil
                
                k_floor = kernel_h.gather(-1, idx_floor.unsqueeze(-1)).squeeze(-1)
                k_ceil = kernel_h.gather(-1, idx_ceil.unsqueeze(-1)).squeeze(-1)
                kernel_w = (k_floor * w_floor + k_ceil * w_ceil).clamp(min=0)
                
                weight = (kernel_w + 1.0) * window_w * valid_f
                weight_sum += weight
                
                pos_clamped = neighbor_pos.clamp(0, L - 1.001)
                pos_floor = pos_clamped.floor().long().clamp(0, L - 1)
                pos_ceil = (pos_floor + 1).clamp(0, L - 1)
                pos_frac = pos_clamped - pos_floor.float()
                
                val_floor = v_head[batch_idx, pos_floor]
                val_ceil = v_head[batch_idx, pos_ceil]
                val = val_floor * (1.0 - pos_frac.unsqueeze(-1)) + val_ceil * pos_frac.unsqueeze(-1)
                
                out_head.addcmul_(weight.unsqueeze(-1), val)
            
            out_head.div_(weight_sum.unsqueeze(-1).clamp(min=1.0))
        
        output = self.se(output)
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
    
    warmup, iters = 5, 20
    
    def bench(model, x):
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
        mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        return total_ms, mem
    
    print(f"{'L':>8} {'sqrt(L)':>8} | {'Fwd+Bwd':>10} {'Mem(MB)':>10}")
    print("-" * 50)
    
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    
    for L in seq_lengths:
        if device == "cuda":
            torch.cuda.empty_cache()
        
        try:
            x = torch.randn(B, L, C, device=device)
            total_ms, mem = bench(model, x)
            print(f"{L:>8} {int(math.sqrt(L)):>8} | {total_ms:>10.2f} {mem:>10.0f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{L:>8} {int(math.sqrt(L)):>8} | OOM")
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise
    
    print("=" * 60)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
