#!/usr/bin/env python3
"""
Test script for Triton kernels. Run on GPU with triton installed.

Usage:
    python scripts/test_triton_kernels.py
    python scripts/test_triton_kernels.py --benchmark
"""

import argparse
import time
import torch
import torch.nn.functional as F

from heuristic_secrets.models.triton_scatter import (
    TritonScatterConv,
    TritonLocalWindowAttn,
    TritonSSMStep,
    HAS_TRITON,
)


def test_scatter_conv(device, B=32, L=512, C=64):
    print(f"\n=== TritonScatterConv (B={B}, L={L}, C={C}) ===")
    
    scatter = TritonScatterConv(channels=C, max_samples=17, num_channels=4).to(device)
    scatter.out_proj.weight.data = torch.eye(C, device=device) * 0.1
    
    x = torch.randn(B, L, C, device=device, requires_grad=True)
    
    out, _ = scatter(x)
    loss = out.sum()
    loss.backward()
    
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Out mean: {out.mean().item():.6f}, std: {out.std().item():.6f}")
    print(f"  Grad norm: {x.grad.norm().item():.4f}")
    print("  PASSED")
    
    return scatter, x.shape


def test_local_window_attn(device, B=32, L=512, C=64):
    print(f"\n=== TritonLocalWindowAttn (B={B}, L={L}, C={C}) ===")
    
    attn = TritonLocalWindowAttn(embed_dim=C, kernel_size=17, num_channels=4).to(device)
    
    x = torch.randn(B, L, C, device=device, requires_grad=True)
    
    out = attn(x)
    loss = out.sum()
    loss.backward()
    
    print(f"  Input:  {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Out mean: {out.mean().item():.6f}, std: {out.std().item():.6f}")
    print(f"  Grad norm: {x.grad.norm().item():.4f}")
    print("  PASSED")
    
    return attn, x.shape


def test_ssm_step(device, B=32, L=512, C=64):
    print(f"\n=== TritonSSMStep (B={B}, L={L}, C={C}) ===")
    
    ssm = TritonSSMStep(dim=C, state_dim=128, mimo_rank=8).to(device)
    
    x = torch.randn(B, L, C, device=device, requires_grad=True)
    H = ssm.init_state(x)
    
    H_new, out = ssm(x, H, layer_idx=0)
    loss = out.sum()
    loss.backward()
    
    print(f"  Input:  {x.shape}")
    print(f"  State:  {H.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Out mean: {out.mean().item():.6f}, std: {out.std().item():.6f}")
    print(f"  Grad norm: {x.grad.norm().item():.4f}")
    print("  PASSED")
    
    return ssm, x.shape


def benchmark_module(module, input_shape, device, n_warmup=10, n_iter=100, name="Module"):
    print(f"\n=== Benchmark: {name} ===")
    
    x = torch.randn(*input_shape, device=device, requires_grad=True)
    
    is_ssm = isinstance(module, TritonSSMStep)
    if is_ssm:
        H = module.init_state(x)
    
    for _ in range(n_warmup):
        if is_ssm:
            _, out = module(x, H, 0)
        elif isinstance(module, TritonScatterConv):
            out, _ = module(x)
        else:
            out = module(x)
        out.sum().backward()
        x.grad = None
    
    torch.cuda.synchronize()
    
    fwd_times = []
    bwd_times = []
    
    for _ in range(n_iter):
        x.grad = None
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        
        if is_ssm:
            _, out = module(x, H, 0)
        elif isinstance(module, TritonScatterConv):
            out, _ = module(x)
        else:
            out = module(x)
        
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        
        loss = out.sum()
        loss.backward()
        
        torch.cuda.synchronize()
        t2 = time.perf_counter()
        
        fwd_times.append(t1 - t0)
        bwd_times.append(t2 - t1)
    
    fwd_mean = sum(fwd_times) / len(fwd_times) * 1000
    bwd_mean = sum(bwd_times) / len(bwd_times) * 1000
    
    print(f"  Forward:  {fwd_mean:.3f} ms")
    print(f"  Backward: {bwd_mean:.3f} ms")
    print(f"  Total:    {fwd_mean + bwd_mean:.3f} ms")
    
    return fwd_mean, bwd_mean


def check_triton_vs_fallback(device):
    print("\n=== Numerical Comparison: Triton vs PyTorch Fallback ===")
    
    torch.manual_seed(42)
    
    scatter = TritonScatterConv(channels=64, max_samples=9, num_channels=4).to(device)
    scatter.out_proj.weight.data = torch.eye(64, device=device) * 0.1
    
    x_cpu = torch.randn(2, 64, 64)
    x_gpu = x_cpu.to(device)
    
    out_cpu, _ = scatter.cpu()(x_cpu)
    scatter.to(device)
    out_gpu, _ = scatter(x_gpu)
    
    diff = (out_cpu - out_gpu.cpu()).abs().max().item()
    print(f"  Scatter max diff: {diff:.6e}")
    
    if diff < 1e-4:
        print("  PASSED (diff < 1e-4)")
    else:
        print("  WARNING: Large numerical difference")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--seq", type=int, default=512, help="Sequence length")
    parser.add_argument("--dim", type=int, default=64, help="Embedding dimension")
    args = parser.parse_args()
    
    print(f"Triton available: {HAS_TRITON}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\nNo CUDA device found. Running on CPU (fallback only).")
        device = "cpu"
    else:
        device = "cuda"
        print(f"Device: {torch.cuda.get_device_name()}")
    
    scatter, scatter_shape = test_scatter_conv(device, args.batch, args.seq, args.dim)
    attn, attn_shape = test_local_window_attn(device, args.batch, args.seq, args.dim)
    ssm, ssm_shape = test_ssm_step(device, args.batch, args.seq, args.dim)
    
    if device == "cuda" and HAS_TRITON:
        check_triton_vs_fallback(device)
    
    if args.benchmark and device == "cuda":
        benchmark_module(scatter, scatter_shape, device, name="TritonScatterConv")
        benchmark_module(attn, attn_shape, device, name="TritonLocalWindowAttn")
        benchmark_module(ssm, ssm_shape, device, name="TritonSSMStep")
    
    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    main()
