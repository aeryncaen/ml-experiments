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


def bench(B=4, H=8, D=32, seqlens=(64, 128, 256, 512, 1024, 2048, 4096),
          warmup=10, iters=50):
    """Benchmark Triton vs PyTorch reference, forward + backward."""
    from triton_silu2 import silu2_attention_triton
    import time

    print(f"\nBenchmark: B={B} H={H} D={D}, warmup={warmup}, iters={iters}")
    print(f"{'T':>6}  {'ref_fwd':>10}  {'tri_fwd':>10}  {'speedup':>8}  "
          f"{'ref_bwd':>10}  {'tri_bwd':>10}  {'speedup':>8}  "
          f"{'ref_mem':>10}  {'tri_mem':>10}")
    print("-" * 110)

    for T in seqlens:
        # Check if we can fit this
        elem = B * H * T * D
        mem_est_gb = elem * 4 * 6 / 1e9  # rough: q,k,v,grad,out * float32
        if mem_est_gb > 40:
            print(f"{T:>6}  SKIPPED (estimated {mem_est_gb:.1f}GB)")
            continue

        torch.manual_seed(0)
        q = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
        k = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
        v = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32, requires_grad=True)
        grad = torch.randn(B, H, T, D, device='cuda', dtype=torch.float32)

        # --- Reference forward ---
        for _ in range(warmup):
            o = silu2_attention_ref(q, k, v)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            o = silu2_attention_ref(q, k, v)
        torch.cuda.synchronize()
        ref_fwd_ms = (time.perf_counter() - t0) / iters * 1000

        # --- Reference backward ---
        for _ in range(warmup):
            o = silu2_attention_ref(q, k, v)
            o.backward(grad)
            q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            o = silu2_attention_ref(q, k, v)
            o.backward(grad)
            q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()
        ref_bwd_ms = (time.perf_counter() - t0) / iters * 1000

        # Reference peak memory
        torch.cuda.reset_peak_memory_stats()
        o = silu2_attention_ref(q, k, v)
        o.backward(grad)
        q.grad = k.grad = v.grad = None
        torch.cuda.synchronize()
        ref_mem_mb = torch.cuda.max_memory_allocated() / 1e6

        # --- Triton forward ---
        q2 = q.detach().clone().requires_grad_(True)
        k2 = k.detach().clone().requires_grad_(True)
        v2 = v.detach().clone().requires_grad_(True)

        for _ in range(warmup):
            o = silu2_attention_triton(q2, k2, v2)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            o = silu2_attention_triton(q2, k2, v2)
        torch.cuda.synchronize()
        tri_fwd_ms = (time.perf_counter() - t0) / iters * 1000

        # --- Triton backward ---
        for _ in range(warmup):
            o = silu2_attention_triton(q2, k2, v2)
            o.backward(grad)
            q2.grad = k2.grad = v2.grad = None
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            o = silu2_attention_triton(q2, k2, v2)
            o.backward(grad)
            q2.grad = k2.grad = v2.grad = None
        torch.cuda.synchronize()
        tri_bwd_ms = (time.perf_counter() - t0) / iters * 1000

        # Triton peak memory
        torch.cuda.reset_peak_memory_stats()
        o = silu2_attention_triton(q2, k2, v2)
        o.backward(grad)
        q2.grad = k2.grad = v2.grad = None
        torch.cuda.synchronize()
        tri_mem_mb = torch.cuda.max_memory_allocated() / 1e6

        fwd_speedup = ref_fwd_ms / tri_fwd_ms
        bwd_speedup = ref_bwd_ms / tri_bwd_ms

        print(f"{T:>6}  {ref_fwd_ms:>8.2f}ms  {tri_fwd_ms:>8.2f}ms  {fwd_speedup:>7.2f}x  "
              f"{ref_bwd_ms:>8.2f}ms  {tri_bwd_ms:>8.2f}ms  {bwd_speedup:>7.2f}x  "
              f"{ref_mem_mb:>8.1f}MB  {tri_mem_mb:>8.1f}MB")

        del q, k, v, q2, k2, v2, grad, o
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
