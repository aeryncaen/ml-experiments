#!/usr/bin/env python3
"""Find crossover where hierarchical local attention beats full O(L²) attention."""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from heuristic_secrets.models.scatter_attention import (
    LocalAttentionND,
    HierarchicalLocalAttentionND,
    RMSNorm,
    apply_rope,
)


class FullAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        L = x.numel() // (B * x.shape[-1])
        C = x.shape[-1]

        x_flat = x.reshape(B, L, C)
        qkv = self.qkv(x_flat).reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self.q_norm(q.transpose(1, 2)).transpose(1, 2)
        k = self.k_norm(k.transpose(1, 2)).transpose(1, 2)
        q = apply_rope(q.transpose(1, 2)).transpose(1, 2)
        k = apply_rope(k.transpose(1, 2)).transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(*x.shape[:-1], C)
        return self.out(out)


def benchmark_forward(model: nn.Module, x: torch.Tensor, warmup: int = 5, runs: int = 20) -> tuple[float, float]:
    """Benchmark forward pass, return (median time in ms, peak memory in MB)."""
    device = x.device
    
    for _ in range(warmup):
        _ = model(x)
    
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    times = []
    for _ in range(runs):
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        _ = model(x)
        
        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()
        
        times.append((time.perf_counter() - start) * 1000)
    
    times.sort()
    median_idx = len(times) // 2
    
    peak_mem_mb = 0.0
    if device.type == 'cuda':
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    
    return times[median_idx], peak_mem_mb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda', 'auto'], default='auto')
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--embed-dim', type=int, default=64)
    parser.add_argument('--window', type=int, default=17)
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--ndim', type=int, default=1, choices=[1, 2])
    parser.add_argument('--min-len', type=int, default=64)
    parser.add_argument('--max-len', type=int, default=16384)
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Device: {device}")
    print(f"Batch: {args.batch}, Embed: {args.embed_dim}, Window: {args.window}, Channels: {args.channels}")
    print(f"ndim: {args.ndim}")
    print()

    full_attn = FullAttention(
        embed_dim=args.embed_dim,
        num_heads=args.channels,
    ).to(device).eval()

    local_attn = LocalAttentionND(
        embed_dim=args.embed_dim,
        kernel_size=args.window,
        ndim=args.ndim,
        num_channels=args.channels,
    ).to(device).eval()
    
    hier_attn = HierarchicalLocalAttentionND(
        embed_dim=args.embed_dim,
        window_size=args.window,
        ndim=args.ndim,
        num_channels=args.channels,
    ).to(device).eval()
    
    lengths = []
    L = args.min_len
    while L <= args.max_len:
        lengths.append(L)
        L *= 2
    
    if args.ndim == 2:
        print(f"{'Size':>10} {'L':>8} {'Full':>12} {'Local':>12} {'Hier':>12} {'Lvl':>4} {'H/F':>6} {'FMem':>8} {'LMem':>8} {'HMem':>8}")
    else:
        print(f"{'L':>8} {'Full':>12} {'Local':>12} {'Hier':>12} {'Lvl':>4} {'H/F':>6} {'FMem':>8} {'LMem':>8} {'HMem':>8}")
    print("-" * 110)
    
    crossover_found = None
    prev_winner = None
    
    with torch.no_grad():
        for L in lengths:
            try:
                if args.ndim == 1:
                    x = torch.randn(args.batch, L, args.embed_dim, device=device)
                    shape_str = ""
                else:
                    side = int(L ** 0.5)
                    x = torch.randn(args.batch, side, side, args.embed_dim, device=device)
                    actual_L = side * side
                    shape_str = f"{side}x{side}"
                    L = actual_L
                
                spatial_shape = x.shape[1:-1]
                n_levels = hier_attn._compute_n_levels(spatial_shape)

                try:
                    full_time, full_mem = benchmark_forward(full_attn, x)
                except RuntimeError:
                    full_time, full_mem = float('inf'), 0.0

                local_time, local_mem = benchmark_forward(local_attn, x)
                hier_time, hier_mem = benchmark_forward(hier_attn, x)

                hier_vs_full = full_time / hier_time if hier_time > 0 else 0

                full_str = f"{full_time:>8.2f}ms" if full_time < 1e6 else "OOM"
                if args.ndim == 2:
                    print(f"{shape_str:>10} {L:>8} {full_str:>12} {local_time:>8.2f}ms {hier_time:>8.2f}ms {n_levels:>4} {hier_vs_full:>5.1f}x {full_mem:>7.1f}M {local_mem:>7.1f}M {hier_mem:>7.1f}M")
                else:
                    print(f"{L:>8} {full_str:>12} {local_time:>8.2f}ms {hier_time:>8.2f}ms {n_levels:>4} {hier_vs_full:>5.1f}x {full_mem:>7.1f}M {local_mem:>7.1f}M {hier_mem:>7.1f}M")

                if prev_winner != "HIER" and hier_time < full_time and crossover_found is None:
                    crossover_found = L
                prev_winner = "HIER" if hier_time < full_time else "full"
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{L:>8} OOM")
                    break
                raise
    
    print()
    if crossover_found:
        print(f"Crossover point: L = {crossover_found} (hier becomes faster than full attention)")
    else:
        print("No crossover found in tested range")


if __name__ == '__main__':
    main()
