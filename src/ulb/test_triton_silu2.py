"""Test Triton silu2 attention against reference implementation.

Run on CUDA:
    python src/ulb/test_triton_silu2.py
"""

import math
import sys
import torch
import torch.nn.functional as F


def silu2_attention_ref(q, k, v):
    """Reference: materializes full T^2 attention matrix."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = (q @ k.transpose(-2, -1)) * scale
    T = logits.shape[-1]
    causal_mask = torch.tril(torch.ones(T, T, device=logits.device, dtype=logits.dtype))
    weights = F.silu(logits) ** 2 * causal_mask
    return weights @ v


def test_forward(B=2, H=4, T=128, D=32, dtype=torch.float32):
    """Compare Triton forward to reference."""
    from triton_silu2 import silu2_attention_triton

    torch.manual_seed(42)
    q = torch.randn(B, H, T, D, device='cuda', dtype=dtype)
    k = torch.randn(B, H, T, D, device='cuda', dtype=dtype)
    v = torch.randn(B, H, T, D, device='cuda', dtype=dtype)

    ref = silu2_attention_ref(q, k, v)
    tri = silu2_attention_triton(q, k, v)

    max_diff = (ref - tri).abs().max().item()
    mean_diff = (ref - tri).abs().mean().item()
    ref_max = ref.abs().max().item() + 1e-8
    rel_err = max_diff / ref_max
    mean_rel = mean_diff / (ref.abs().mean().item() + 1e-8)
    print(f"Forward  B={B} H={H} T={T} D={D} dtype={dtype}")
    print(f"  max_diff={max_diff:.2e}  rel_err={rel_err:.2e}  mean_rel={mean_rel:.2e}")

    ok = rel_err < 1e-4 if dtype == torch.float32 else rel_err < 5e-2
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


def test_backward(B=2, H=4, T=128, D=32, dtype=torch.float32):
    """Compare Triton backward to reference via autograd."""
    from triton_silu2 import silu2_attention_triton

    torch.manual_seed(42)
    q_ref = torch.randn(B, H, T, D, device='cuda', dtype=dtype, requires_grad=True)
    k_ref = torch.randn(B, H, T, D, device='cuda', dtype=dtype, requires_grad=True)
    v_ref = torch.randn(B, H, T, D, device='cuda', dtype=dtype, requires_grad=True)

    q_tri = q_ref.detach().clone().requires_grad_(True)
    k_tri = k_ref.detach().clone().requires_grad_(True)
    v_tri = v_ref.detach().clone().requires_grad_(True)

    # Forward
    out_ref = silu2_attention_ref(q_ref, k_ref, v_ref)
    out_tri = silu2_attention_triton(q_tri, k_tri, v_tri)

    # Backward
    grad_out = torch.randn_like(out_ref)
    out_ref.backward(grad_out)
    out_tri.backward(grad_out)

    results = {}
    for name, g_ref, g_tri in [
        ('dQ', q_ref.grad, q_tri.grad),
        ('dK', k_ref.grad, k_tri.grad),
        ('dV', v_ref.grad, v_tri.grad),
    ]:
        max_diff = (g_ref - g_tri).abs().max().item()
        ref_max = g_ref.abs().max().item() + 1e-8
        rel_err = max_diff / ref_max
        ok = rel_err < 1e-3 if dtype == torch.float32 else rel_err < 5e-2
        results[name] = (max_diff, rel_err, ok)
        print(f"  {name}: max_diff={max_diff:.2e}  rel_err={rel_err:.2e}  {'PASS' if ok else 'FAIL'}")

    print(f"Backward B={B} H={H} T={T} D={D} dtype={dtype}")
    return all(ok for _, _, ok in results.values())


def test_non_power_of_2_seqlen():
    """Test with seqlen that's not a multiple of block size."""
    return test_forward(B=1, H=2, T=137, D=32) and test_backward(B=1, H=2, T=137, D=32)


def test_various_headdims():
    """Test multiple head dimensions."""
    all_ok = True
    for D in [16, 32, 48, 64, 96, 128]:
        ok = test_forward(B=1, H=2, T=64, D=D)
        all_ok = all_ok and ok
    return all_ok


def _bench_fn(fn, q, k, v, grad, warmup, iters):
    """Time forward, forward+backward, and peak memory for a given attention fn."""
    import time

    # Forward warmup + timing
    for _ in range(warmup):
        o = fn(q, k, v)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        o = fn(q, k, v)
    torch.cuda.synchronize()
    fwd_ms = (time.perf_counter() - t0) / iters * 1000

    # Backward warmup + timing
    for _ in range(warmup):
        o = fn(q, k, v)
        o.backward(grad)
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        o = fn(q, k, v)
        o.backward(grad)
        q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    bwd_ms = (time.perf_counter() - t0) / iters * 1000

    # Peak memory
    torch.cuda.reset_peak_memory_stats()
    o = fn(q, k, v)
    o.backward(grad)
    q.grad = k.grad = v.grad = None
    torch.cuda.synchronize()
    mem_mb = torch.cuda.max_memory_allocated() / 1e6

    return fwd_ms, bwd_ms, mem_mb


def bench(B=4, H=8, D=32, seqlens=(64, 128, 256, 512, 1024, 2048, 4096),
          warmup=10, iters=50):
    """Benchmark Triton vs PyTorch reference vs compiled reference."""
    from triton_silu2 import silu2_attention_triton

    # Compile the reference
    compiled_ref = torch.compile(silu2_attention_ref)

    print(f"\nBenchmark: B={B} H={H} D={D}, warmup={warmup}, iters={iters}")
    print(f"{'T':>6}  {'ref_fwd':>9} {'comp_fwd':>9} {'tri_fwd':>9}  "
          f"{'ref_bwd':>9} {'comp_bwd':>9} {'tri_bwd':>9}  "
          f"{'ref_mem':>9} {'comp_mem':>9} {'tri_mem':>9}  "
          f"{'tri/comp':>8}")
    print("-" * 140)

    for T in seqlens:
        elem = B * H * T * D
        mem_est_gb = elem * 4 * 6 / 1e9
        if mem_est_gb > 40:
            print(f"{T:>6}  SKIPPED (estimated {mem_est_gb:.1f}GB)")
            continue

        torch.manual_seed(0)
        q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
        grad = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32)

        # Reference (raw PyTorch)
        ref_fwd, ref_bwd, ref_mem = _bench_fn(silu2_attention_ref, q, k, v, grad, warmup, iters)

        # Compiled reference
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)
        comp_fwd, comp_bwd, comp_mem = _bench_fn(compiled_ref, q2, k2, v2, grad, warmup, iters)

        # Triton
        q3 = q.detach().clone().requires_grad_(True)
        k3 = k.detach().clone().requires_grad_(True)
        v3 = v.detach().clone().requires_grad_(True)
        tri_fwd, tri_bwd, tri_mem = _bench_fn(silu2_attention_triton, q3, k3, v3, grad, warmup, iters)

        # Triton vs compiled speedup (the comparison that matters)
        tri_vs_comp_fwd = comp_fwd / tri_fwd
        tri_vs_comp_bwd = comp_bwd / tri_bwd

        print(f"{T:>6}  {ref_fwd:>7.2f}ms {comp_fwd:>7.2f}ms {tri_fwd:>7.2f}ms  "
              f"{ref_bwd:>7.2f}ms {comp_bwd:>7.2f}ms {tri_bwd:>7.2f}ms  "
              f"{ref_mem:>7.1f}MB {comp_mem:>7.1f}MB {tri_mem:>7.1f}MB  "
              f"{tri_vs_comp_bwd:>6.2f}x")

        del q, k, v, q2, k2, v2, q3, k3, v3, grad, o
        torch.cuda.empty_cache()


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available, skipping Triton tests.")
        sys.exit(0)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--bench', action='store_true', help='Run benchmark only')
    parser.add_argument('--B', type=int, default=4, help='Batch size for bench')
    parser.add_argument('--H', type=int, default=8, help='Heads for bench')
    parser.add_argument('--D', type=int, default=32, help='Head dim for bench')
    args = parser.parse_args()

    if args.bench:
        bench(B=args.B, H=args.H, D=args.D)
        sys.exit(0)

    print("=" * 60)
    print("Triton silu2 attention tests")
    print("=" * 60)

    all_pass = True

    print("\n--- Forward (standard) ---")
    all_pass &= test_forward()

    print("\n--- Forward (non-power-of-2 seqlen) ---")
    all_pass &= test_forward(B=1, H=2, T=137, D=32)

    print("\n--- Forward (various headdims) ---")
    all_pass &= test_various_headdims()

    print("\n--- Backward (standard) ---")
    all_pass &= test_backward()

    print("\n--- Backward (non-power-of-2 seqlen) ---")
    all_pass &= test_backward(B=1, H=2, T=137, D=32)

    print("\n--- Benchmark ---")
    bench()

    print("\n" + "=" * 60)
    if all_pass:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)
