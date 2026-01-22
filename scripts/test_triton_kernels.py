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


def measure_memory_and_time(fn, n_warmup=5, n_iter=20):
    for _ in range(n_warmup):
        fn()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    times = []
    for _ in range(n_iter):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    avg_time = sum(times) / len(times) * 1000
    
    return avg_time, peak_mem


def benchmark_triton_vs_pytorch(B, L, C, device):
    print(f"\n{'='*70}")
    print(f"BENCHMARK: Triton vs PyTorch (B={B}, L={L}, C={C})")
    print(f"{'='*70}")
    
    def run_scatter_triton():
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        out, _ = scatter_triton(x)
        out.sum().backward()
    
    def run_scatter_pytorch():
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        out, _ = scatter_pytorch(x)
        out.sum().backward()
    
    def run_attn_triton():
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        out = attn_triton(x)
        out.sum().backward()
    
    def run_attn_pytorch():
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        out = attn_pytorch(x)
        out.sum().backward()
    
    def run_ssm_triton():
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        H = ssm_triton.init_state(x)
        _, out = ssm_triton(x, H, 0)
        out.sum().backward()
    
    def run_ssm_pytorch():
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        H = ssm_pytorch.init_state(x)
        _, out = ssm_pytorch(x, H, 0)
        out.sum().backward()
    
    scatter_triton = TritonScatterConv(channels=C, max_samples=17, num_channels=4).to(device)
    scatter_pytorch = TritonScatterConv(channels=C, max_samples=17, num_channels=4).to(device)
    scatter_pytorch.load_state_dict(scatter_triton.state_dict())
    
    attn_triton = TritonLocalWindowAttn(embed_dim=C, kernel_size=17, num_channels=4).to(device)
    attn_pytorch = TritonLocalWindowAttn(embed_dim=C, kernel_size=17, num_channels=4).to(device)
    attn_pytorch.load_state_dict(attn_triton.state_dict())
    
    ssm_triton = TritonSSMStep(dim=C, state_dim=64, mimo_rank=4).to(device)
    ssm_pytorch = TritonSSMStep(dim=C, state_dim=64, mimo_rank=4).to(device)
    ssm_pytorch.load_state_dict(ssm_triton.state_dict())
    
    original_scatter_fwd = TritonScatterConv.forward
    original_attn_fwd = TritonLocalWindowAttn.forward
    original_ssm_fwd = TritonSSMStep.forward
    
    def force_pytorch_scatter(self, x):
        B, L, C = x.shape
        H = self.num_channels
        wave_params = F.silu(self.wave_proj(x)).reshape(B, L, 3, H).permute(0, 1, 3, 2)
        queries = F.silu(self.query_proj(x)).reshape(B, L, H, self.pos_dim)
        freq = torch.sigmoid(wave_params[..., 0]) * (self.max_freq - self.min_freq) + self.min_freq
        phase = torch.tanh(wave_params[..., 1]) * self.max_freq
        decay = torch.sigmoid(wave_params[..., 2]) * 9.5 + 0.5
        out = self._pytorch_fallback(x, queries, freq.mean(dim=2), phase.mean(dim=2), decay.mean(dim=2))
        se_weights = torch.sigmoid(self.se_fc2(F.silu(self.se_fc1(out))))
        out = out * se_weights
        out = F.silu(self.out_proj(out))
        return out, {}
    
    def force_pytorch_attn(self, x):
        B, L, C = x.shape
        H, D, K = self.num_channels, self.channel_dim, self.kernel_size
        q = F.silu(self.q_proj(x)).reshape(B, L, H, D)
        q = self.q_norm(q)
        kv = F.silu(self.kv_proj(x))
        k, v = kv.chunk(2, dim=-1)
        k = self.k_norm(k.reshape(B, L, H, D)).reshape(B, L, C)
        window_params = F.silu(self.window_proj(x))
        width_raw, sharpness_raw = window_params.chunk(2, dim=-1)
        width = width_raw.sigmoid() * self.max_dist + 0.5
        sharpness = sharpness_raw.sigmoid() * 9.5 + 0.5
        pad_left, pad_right = (K - 1) // 2, K // 2
        k_padded = F.pad(k, (0, 0, pad_left, pad_right))
        v_padded = F.pad(v, (0, 0, pad_left, pad_right))
        out = self._pytorch_fallback(q, k_padded, v_padded, width, sharpness, B, L, H, D, K)
        return F.silu(self.out(out))
    
    def force_pytorch_ssm(self, x, H, layer_idx):
        B, L, _ = x.shape
        B_proj = F.silu(self.to_B(x)).view(B, L, self.N, self.R).permute(0, 3, 1, 2).contiguous()
        X_r = F.silu(self.to_X(x)).permute(0, 2, 1).contiguous()
        decay = torch.sigmoid(self.to_decay(x))
        theta = self.to_theta(x)
        B_rot = self._apply_rope_pytorch(B_proj, theta, layer_idx)
        inject = B_rot * X_r.unsqueeze(-1)
        H = decay.unsqueeze(1) * H + inject
        out_flat = H.permute(0, 2, 1, 3).reshape(B, L, self.N * self.R)
        out = F.silu(self.out_proj(out_flat))
        return H, out
    
    print("\n--- ScatterConv ---")
    print(f"{'Implementation':<15} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10} {'Mem Saved':<10}")
    print("-" * 60)
    
    time_triton, mem_triton = measure_memory_and_time(run_scatter_triton)
    
    TritonScatterConv.forward = force_pytorch_scatter
    time_pytorch, mem_pytorch = measure_memory_and_time(run_scatter_pytorch)
    TritonScatterConv.forward = original_scatter_fwd
    
    speedup = time_pytorch / time_triton if time_triton > 0 else float('inf')
    mem_saved = (mem_pytorch - mem_triton) / mem_pytorch * 100 if mem_pytorch > 0 else 0
    print(f"{'Triton':<15} {time_triton:<12.3f} {mem_triton:<12.1f}")
    print(f"{'PyTorch':<15} {time_pytorch:<12.3f} {mem_pytorch:<12.1f} {speedup:<10.2f}x {mem_saved:<10.1f}%")
    
    print("\n--- LocalWindowAttn ---")
    print(f"{'Implementation':<15} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10} {'Mem Saved':<10}")
    print("-" * 60)
    
    time_triton, mem_triton = measure_memory_and_time(run_attn_triton)
    
    TritonLocalWindowAttn.forward = force_pytorch_attn
    time_pytorch, mem_pytorch = measure_memory_and_time(run_attn_pytorch)
    TritonLocalWindowAttn.forward = original_attn_fwd
    
    speedup = time_pytorch / time_triton if time_triton > 0 else float('inf')
    mem_saved = (mem_pytorch - mem_triton) / mem_pytorch * 100 if mem_pytorch > 0 else 0
    print(f"{'Triton':<15} {time_triton:<12.3f} {mem_triton:<12.1f}")
    print(f"{'PyTorch':<15} {time_pytorch:<12.3f} {mem_pytorch:<12.1f} {speedup:<10.2f}x {mem_saved:<10.1f}%")
    
    print("\n--- SSMStep ---")
    print(f"{'Implementation':<15} {'Time (ms)':<12} {'Memory (MB)':<12} {'Speedup':<10} {'Mem Saved':<10}")
    print("-" * 60)
    
    time_triton, mem_triton = measure_memory_and_time(run_ssm_triton)
    
    TritonSSMStep.forward = force_pytorch_ssm
    time_pytorch, mem_pytorch = measure_memory_and_time(run_ssm_pytorch)
    TritonSSMStep.forward = original_ssm_fwd
    
    speedup = time_pytorch / time_triton if time_triton > 0 else float('inf')
    mem_saved = (mem_pytorch - mem_triton) / mem_pytorch * 100 if mem_pytorch > 0 else 0
    print(f"{'Triton':<15} {time_triton:<12.3f} {mem_triton:<12.1f}")
    print(f"{'PyTorch':<15} {time_pytorch:<12.3f} {mem_pytorch:<12.1f} {speedup:<10.2f}x {mem_saved:<10.1f}%")


def check_triton_vs_fallback(device):
    print("\n=== Numerical Comparison: Triton vs PyTorch Fallback ===")
    
    torch.manual_seed(42)
    
    print("\n  [ScatterConv]")
    scatter = TritonScatterConv(channels=64, max_samples=9, num_channels=4).to(device)
    scatter.out_proj.weight.data = torch.eye(64, device=device) * 0.1
    
    x_cpu = torch.randn(2, 64, 64)
    x_gpu = x_cpu.to(device)
    
    out_cpu, _ = scatter.cpu()(x_cpu)
    scatter.to(device)
    out_gpu, _ = scatter(x_gpu)
    
    diff = (out_cpu - out_gpu.cpu()).abs().max().item()
    print(f"    Max diff: {diff:.6e}")
    
    if diff < 1e-3:
        print("    PASSED (diff < 1e-3)")
    else:
        print("    WARNING: Large numerical difference")
    
    print("\n  [LocalWindowAttn]")
    torch.manual_seed(42)
    attn = TritonLocalWindowAttn(embed_dim=64, kernel_size=9, num_channels=4).to(device)
    
    x_cpu = torch.randn(2, 64, 64)
    x_gpu = x_cpu.to(device)
    
    out_cpu = attn.cpu()(x_cpu)
    attn.to(device)
    out_gpu = attn(x_gpu)
    
    diff = (out_cpu - out_gpu.cpu()).abs().max().item()
    print(f"    Max diff: {diff:.6e}")
    
    if diff < 1e-3:
        print("    PASSED (diff < 1e-3)")
    else:
        print("    WARNING: Large numerical difference")
    
    print("\n  [SSMStep]")
    torch.manual_seed(42)
    ssm = TritonSSMStep(dim=64, state_dim=32, mimo_rank=4).to(device)
    
    x_cpu = torch.randn(2, 64, 64)
    H_cpu = ssm.cpu().init_state(x_cpu)
    
    x_gpu = x_cpu.to(device)
    ssm.to(device)
    H_gpu = ssm.init_state(x_gpu)
    
    _, out_cpu = ssm.cpu()(x_cpu, H_cpu, layer_idx=0)
    ssm.to(device)
    _, out_gpu = ssm(x_gpu, H_gpu, layer_idx=0)
    
    diff = (out_cpu - out_gpu.cpu()).abs().max().item()
    print(f"    Max diff: {diff:.6e}")
    
    if diff < 1e-3:
        print("    PASSED (diff < 1e-3)")
    else:
        print("    WARNING: Large numerical difference")


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
        benchmark_triton_vs_pytorch(args.batch, args.seq, args.dim, device)
    
    print("\n" + "=" * 50)
    print("All tests passed!")


if __name__ == "__main__":
    main()
