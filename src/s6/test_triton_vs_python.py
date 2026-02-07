"""
Numerical validation: custom backward (ChunkedScanFn) vs Python autograd path.

Compares forward outputs and all backward gradients between:
  - _forward_scan_chunked_autograd (Python, autograd backward)
  - ChunkedScanFn (custom backward using _chunked_scan_bwd)

Run on GPU: python -m s6.test_triton_vs_python
Run on CPU: python -m s6.test_triton_vs_python --cpu
"""

import argparse
import torch

import time

from .scan import (
    _forward_scan_chunked_autograd,
    forward_scan_chunked,
    forward_scan_elementwise_streaming,
    ChunkedScanFn,
)


def _make_inputs(B, T, H, D, device, dtype=torch.float32):
    """Create matched inputs for both paths."""
    torch.manual_seed(42)
    kv = torch.randn(B, T, H, D, device=device, dtype=dtype)
    alpha = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
    delta = torch.randn(B, T, H, device=device, dtype=dtype) * 0.1
    epsilon = torch.randn(B, T, H, device=device, dtype=dtype) * 0.1
    zeta = torch.randn(B, T, H, device=device, dtype=dtype) * 0.1
    init_state = torch.randn(B, H, D, device=device, dtype=dtype) * 0.1
    return kv, alpha, delta, epsilon, zeta, init_state


def _clone_inputs_with_grad(*inputs):
    return tuple(x.detach().clone().requires_grad_(True) for x in inputs)


def _compare(name, a, b, atol=1e-4, rtol=1e-3):
    """Compare two tensors, return (passed, info_str)."""
    if a is None and b is None:
        return True, f"  {name}: both None"
    if a is None or b is None:
        return False, f"  {name}: FAILED (one is None)"
    a_f, b_f = a.float(), b.float()
    abs_diff = (a_f - b_f).abs().max().item()
    denom = max(a_f.abs().max().item(), b_f.abs().max().item(), 1e-8)
    rel_diff = abs_diff / denom
    passed = rel_diff < rtol and abs_diff < atol + rtol * denom
    status = "PASSED" if passed else "FAILED"
    info = f"  {name}: abs={abs_diff:.2e} rel={rel_diff:.2e} {status}"
    return passed, info


def test_forward_match(device, chunk_sizes=(32, 64), sizes=((2, 128, 4, 64), (1, 200, 2, 32))):
    """Compare forward output of autograd path vs ChunkedScanFn."""
    print("\nForward match (autograd vs ChunkedScanFn):")
    all_passed = True

    for B, T, H, D in sizes:
        for C in chunk_sizes:
            if C > T:
                continue
            kv, alpha, delta, epsilon, zeta, init_state = _make_inputs(B, T, H, D, device)

            out_auto = _forward_scan_chunked_autograd(kv, alpha, delta, epsilon, zeta, init_state, C)
            out_custom = ChunkedScanFn.apply(kv, alpha, delta, epsilon, zeta, init_state, C)

            passed, info = _compare(f"T={T:4d} C={C:3d}", out_auto, out_custom)
            print(info)
            all_passed = all_passed and passed

    assert all_passed, "Forward match FAILED"
    print("  All forward match: PASSED")


def test_backward_match(device, chunk_sizes=(32, 64), sizes=((2, 128, 4, 64), (1, 200, 2, 32))):
    """Compare backward gradients of autograd path vs ChunkedScanFn."""
    print("\nBackward match (autograd vs ChunkedScanFn):")
    all_passed = True

    input_names = ("kv", "alpha", "delta", "epsilon", "zeta", "init_state")

    for B, T, H, D in sizes:
        for C in chunk_sizes:
            if C > T:
                continue
            kv, alpha, delta, epsilon, zeta, init_state = _make_inputs(B, T, H, D, device)

            # Autograd path
            inputs_auto = _clone_inputs_with_grad(kv, alpha, delta, epsilon, zeta, init_state)
            out_auto = _forward_scan_chunked_autograd(*inputs_auto, C)
            out_auto.sum().backward()

            # ChunkedScanFn path
            inputs_custom = _clone_inputs_with_grad(kv, alpha, delta, epsilon, zeta, init_state)
            out_custom = ChunkedScanFn.apply(*inputs_custom, C)
            out_custom.sum().backward()

            print(f"  T={T:4d} C={C:3d}:")
            for name, t_auto, t_custom in zip(input_names, inputs_auto, inputs_custom):
                passed, info = _compare(f"    d_{name}", t_auto.grad, t_custom.grad)
                print(info)
                all_passed = all_passed and passed

    assert all_passed, "Backward match FAILED"
    print("  All backward match: PASSED")


def test_backward_match_random_dout(device):
    """Same as above but with random dout instead of ones (more thorough)."""
    print("\nBackward match (random dout):")
    all_passed = True

    input_names = ("kv", "alpha", "delta", "epsilon", "zeta", "init_state")
    B, T, H, D, C = 2, 128, 4, 64, 32

    for trial in range(3):
        torch.manual_seed(trial * 1000)
        kv, alpha, delta, epsilon, zeta, init_state = _make_inputs(B, T, H, D, device)
        dout = torch.randn(B, T, H, D, device=device)

        inputs_auto = _clone_inputs_with_grad(kv, alpha, delta, epsilon, zeta, init_state)
        out_auto = _forward_scan_chunked_autograd(*inputs_auto, C)
        out_auto.backward(dout)

        inputs_custom = _clone_inputs_with_grad(kv, alpha, delta, epsilon, zeta, init_state)
        out_custom = ChunkedScanFn.apply(*inputs_custom, C)
        out_custom.backward(dout)

        print(f"  trial {trial}:")
        for name, t_auto, t_custom in zip(input_names, inputs_auto, inputs_custom):
            passed, info = _compare(f"    d_{name}", t_auto.grad, t_custom.grad)
            print(info)
            all_passed = all_passed and passed

    assert all_passed, "Backward match (random dout) FAILED"
    print("  All backward match (random dout): PASSED")


# ---------------------------------------------------------------------------
# Speed + memory benchmarks
# ---------------------------------------------------------------------------

def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _bench_fn(fn, warmup=5, repeats=20, device="cpu"):
    """Time a callable, returning median ms."""
    for _ in range(warmup):
        fn()
    _sync(device)

    times = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def _measure_memory(fn, device):
    """Run fn once and return peak GPU memory delta in MB. Returns 0.0 on CPU."""
    if device != "cuda":
        fn()
        return 0.0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    mem_before = torch.cuda.memory_allocated()
    fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - mem_before) / (1024 * 1024)


def _fmt_mem(mb):
    """Format memory in MB or (n/a) for CPU."""
    if mb == 0.0:
        return ""
    return f"{mb:8.1f}"


def bench_speed(device, seq_lengths=(256, 1024, 4096), chunk_sizes=(64,),
                B=4, H=8, D=64):
    """Benchmark forward and forward+backward for chunked vs streaming scan."""
    is_cuda = device == "cuda"
    print("\n" + "=" * 60)
    print("Speed benchmarks")
    print("=" * 60)
    print(f"  Device={device}  B={B}  H={H}  D={D}  warmup=5  repeats=20")
    print(f"  Chunk sizes: {chunk_sizes}")
    print()

    # Header
    mem_hdr = "  Fwd (MB)  Fwd+Bwd (MB)" if is_cuda else ""
    mem_sep = "  --------  ------------" if is_cuda else ""
    print(f"  {'T':>5}  {'C':>3}  {'Method':<12}  {'Fwd (ms)':>10}  {'Fwd+Bwd (ms)':>13}{mem_hdr}")
    print(f"  {'-----':>5}  {'---':>3}  {'------------':<12}  {'----------':>10}  {'-------------':>13}{mem_sep}")

    for T in seq_lengths:
        for C in chunk_sizes:
            if C > T:
                continue

            kv_base, alpha_base, delta_base, epsilon_base, zeta_base, init_base = \
                _make_inputs(B, T, H, D, device)

            def _make_grad_inputs():
                return (kv_base.detach().requires_grad_(True),
                        alpha_base.detach().requires_grad_(True),
                        delta_base.detach().requires_grad_(True),
                        epsilon_base.detach().requires_grad_(True),
                        zeta_base.detach().requires_grad_(True),
                        init_base.detach().requires_grad_(True))

            # --- Chunked scan ---
            def _chunked_fwd():
                kv, alpha, delta, epsilon, zeta, init_s = _make_grad_inputs()
                return forward_scan_chunked(kv, alpha, delta, epsilon, zeta, init_s, C)

            def _chunked_fwd_bwd():
                kv, alpha, delta, epsilon, zeta, init_s = _make_grad_inputs()
                out = forward_scan_chunked(kv, alpha, delta, epsilon, zeta, init_s, C)
                out.sum().backward()

            fwd_chunked = _bench_fn(_chunked_fwd, device=device)
            fwd_bwd_chunked = _bench_fn(_chunked_fwd_bwd, device=device)
            mem_fwd_chunked = _measure_memory(_chunked_fwd, device)
            mem_bwd_chunked = _measure_memory(_chunked_fwd_bwd, device)

            mem_str = ""
            if is_cuda:
                mem_str = f"  {_fmt_mem(mem_fwd_chunked):>8}  {_fmt_mem(mem_bwd_chunked):>12}"
            print(f"  {T:5d}  {C:3d}  {'chunked':<12}  {fwd_chunked:10.2f}  {fwd_bwd_chunked:13.2f}{mem_str}")

            # --- Streaming scan (skip for long sequences — too slow) ---
            if T <= 1024:
                def _streaming_fwd():
                    kv, alpha, delta, epsilon, zeta, init_s = _make_grad_inputs()
                    return forward_scan_elementwise_streaming(kv, alpha, delta, epsilon, zeta, init_s)

                def _streaming_fwd_bwd():
                    kv, alpha, delta, epsilon, zeta, init_s = _make_grad_inputs()
                    out = forward_scan_elementwise_streaming(kv, alpha, delta, epsilon, zeta, init_s)
                    out.sum().backward()

                fwd_stream = _bench_fn(_streaming_fwd, device=device)
                fwd_bwd_stream = _bench_fn(_streaming_fwd_bwd, device=device)
                mem_fwd_stream = _measure_memory(_streaming_fwd, device)
                mem_bwd_stream = _measure_memory(_streaming_fwd_bwd, device)

                mem_str = ""
                if is_cuda:
                    mem_str = f"  {_fmt_mem(mem_fwd_stream):>8}  {_fmt_mem(mem_bwd_stream):>12}"
                print(f"  {T:5d}  {'':>3}  {'streaming':<12}  {fwd_stream:10.2f}  {fwd_bwd_stream:13.2f}{mem_str}")
            else:
                skip = "(skipped)"
                print(f"  {T:5d}  {'':>3}  {'streaming':<12}  {skip:>10}  {skip:>13}")

            # Speedup
            if T <= 1024:
                fwd_speedup = fwd_stream / fwd_chunked
                bwd_speedup = fwd_bwd_stream / fwd_bwd_chunked
                mem_ratio = ""
                if is_cuda and mem_bwd_stream > 0:
                    mem_ratio = f"  {'':>8}  {mem_bwd_stream / max(mem_bwd_chunked, 0.1):11.1f}x"
                print(f"  {T:5d}  {'':>3}  {'speedup':<12}  {fwd_speedup:9.1f}x  {bwd_speedup:12.1f}x{mem_ratio}")
            print()


# ---------------------------------------------------------------------------
# Granular Triton vs Python kernel comparison (for GPU debugging)
# ---------------------------------------------------------------------------

def test_triton_kernels(device):
    """
    Step-by-step comparison of Triton kernels vs Python reference.
    Exercises each kernel independently so failures are easy to localize.
    Only runs when Triton is available (CUDA).
    """
    from .scan import (
        _tridiag_conv,
        _chunked_scan_bwd,
        HAS_TRITON as _HAS_TRITON,
    )
    if not _HAS_TRITON or device != "cuda":
        print("\nTriton kernel tests: SKIPPED (no Triton / not CUDA)")
        return

    from .scan import (
        _chunked_scan_fwd_triton,
        _state_passing_fwd_triton,
        _chunked_scan_bwd_triton,
        _state_passing_bwd_triton,
    )

    print("\nTriton kernel tests (step-by-step):")

    B, T, H, D, C = 2, 128, 4, 64, 32
    torch.manual_seed(42)
    kv, alpha, delta, epsilon, zeta, init_state = _make_inputs(B, T, H, D, device)
    all_passed = True

    # --- Setup: tridiag conv + chunking (shared between Python and Triton) ---
    s = _tridiag_conv(kv, zeta, epsilon, delta)
    s_f = s.float()
    alpha_f = alpha.clamp(min=1e-6).float()
    log_alpha = alpha_f.log()

    nchunks = (T + C - 1) // C
    T_padded = nchunks * C
    if T_padded > T:
        s_f = torch.nn.functional.pad(s_f, (0, 0, 0, 0, 0, T_padded - T))
        log_alpha = torch.nn.functional.pad(log_alpha, (0, 0, 0, T_padded - T))

    s_chunks = s_f.view(B, nchunks, C, H, D)
    la_chunks = log_alpha.view(B, nchunks, C, H)
    cumA = la_chunks.cumsum(dim=2)
    chunk_total_decay = cumA[:, :, -1, :]

    BNH = B * nchunks * H
    cumA_flat = cumA.permute(0, 1, 3, 2).reshape(BNH, C).contiguous()
    s_flat = s_chunks.permute(0, 1, 3, 2, 4).reshape(BNH, C, D).contiguous()

    # Flatten log_alpha for Triton (new API takes log_alpha, not cumA)
    la_flat = la_chunks.permute(0, 1, 3, 2).reshape(BNH, C).contiguous()

    # --- 0. Diagnostic: verify Triton cumsum matches Python cumsum ---
    zero_prev = torch.zeros(BNH, D, device=device, dtype=torch.float32)
    _, cumA_triton = _chunked_scan_fwd_triton(s_flat, la_flat, zero_prev, C, D, BNH, store_cumA=True)
    passed_cumA, info_cumA = _compare("cumA (triton vs python)", cumA_flat, cumA_triton)
    print(info_cumA)
    if not passed_cumA:
        print(f"    Python cumA[0,:5]:  {cumA_flat[0,:5].tolist()}")
        print(f"    Triton cumA[0,:5]:  {cumA_triton[0,:5].tolist()}")
        print(f"    la_flat[0,:5]:      {la_flat[0,:5].tolist()}")
    all_passed = all_passed and passed_cumA

    # --- 1. Intra-chunk forward (zero prev_states) ---
    # Python reference
    causal = torch.tril(torch.ones(C, C, device=device, dtype=torch.bool))
    decay_diff = cumA_flat[:, :, None] - cumA_flat[:, None, :]
    decay_mask = torch.where(causal, decay_diff.exp(), torch.zeros_like(decay_diff))
    intra_flat_py = torch.bmm(decay_mask, s_flat)

    # Triton (now takes log_alpha, computes cumsum internally)
    intra_flat_tr = _chunked_scan_fwd_triton(s_flat, la_flat, zero_prev, C, D, BNH)

    passed, info = _compare("intra-chunk fwd (zero prev)", intra_flat_py, intra_flat_tr)
    print(info)
    if not passed:
        # Check first element in detail
        print(f"    intra_py[0,0,:4]:  {intra_flat_py[0,0,:4].tolist()}")
        print(f"    intra_tr[0,0,:4]:  {intra_flat_tr[0,0,:4].tolist()}")
        print(f"    intra_py[0,-1,:4]: {intra_flat_py[0,-1,:4].tolist()}")
        print(f"    intra_tr[0,-1,:4]: {intra_flat_tr[0,-1,:4].tolist()}")
    all_passed = all_passed and passed

    # --- 2. State passing forward ---
    intra_py = intra_flat_py.view(B, nchunks, H, C, D).permute(0, 1, 3, 2, 4)
    chunk_new_state = intra_py[:, :, -1, :, :].contiguous()

    # Python reference
    prev_states_py = torch.zeros(B, nchunks, H, D, device=device, dtype=torch.float32)
    state = init_state.float()
    for c in range(nchunks):
        prev_states_py[:, c] = state
        state = chunk_total_decay[:, c].exp()[..., None] * state + chunk_new_state[:, c]

    # Triton
    prev_states_tr = _state_passing_fwd_triton(
        chunk_new_state, chunk_total_decay, init_state.float(), B, nchunks, H, D)

    passed, info = _compare("state passing fwd", prev_states_py, prev_states_tr)
    print(info)
    all_passed = all_passed and passed

    # --- 3. Full forward with real prev_states ---
    prev_states_flat = prev_states_py.reshape(BNH, D).contiguous()
    out_flat_py = torch.bmm(decay_mask, s_flat) + \
        (cumA_flat[:, :, None].exp() * prev_states_flat[:, None, :])

    # Triton (takes log_alpha, computes cumsum internally)
    out_flat_tr = _chunked_scan_fwd_triton(s_flat, la_flat, prev_states_flat, C, D, BNH)

    passed, info = _compare("full fwd (with prev)", out_flat_py, out_flat_tr)
    print(info)
    all_passed = all_passed and passed

    # --- 4. Backward: intra-chunk kernel ---
    # Make a fake dout
    torch.manual_seed(99)
    dout_flat = torch.randn(BNH, C, D, device=device, dtype=torch.float32)

    # Python reference
    ds_py = torch.bmm(decay_mask.transpose(1, 2), dout_flat)
    d_decay_mask_py = torch.bmm(dout_flat, s_flat.transpose(1, 2))
    d_dm_times_dm_py = d_decay_mask_py * decay_mask
    d_cumA_intra_py = d_dm_times_dm_py.sum(dim=2) - d_dm_times_dm_py.sum(dim=1)

    # Triton (takes log_alpha, computes cumsum + decay_mask internally)
    ds_tr, d_cumA_intra_tr, _ = _chunked_scan_bwd_triton(
        dout_flat, la_flat, s_flat, prev_states_flat,
        C, D, BNH)

    passed, info = _compare("bwd ds_flat", ds_py, ds_tr)
    print(info)
    all_passed = all_passed and passed

    passed, info = _compare("bwd d_cumA_from_intra", d_cumA_intra_py, d_cumA_intra_tr)
    print(info)
    all_passed = all_passed and passed

    # --- 5. Backward: state passing ---
    # Build d_prev_states as the Python backward would
    dout_f = dout_flat.view(B, nchunks, H, C, D).permute(0, 1, 3, 2, 4)  # (B, nc, C, H, D)
    exp_cumA = cumA[..., None].exp()
    d_prev_states_inter = (dout_f * exp_cumA).sum(dim=2)  # (B, nc, H, D)

    # Python reference
    d_cns_py_list = []
    d_td_py_list = []
    d_state = torch.zeros(B, H, D, device=device, dtype=torch.float32)
    for c in range(nchunks - 1, -1, -1):
        decay_c = chunk_total_decay[:, c].exp()
        d_cns_py_list.append(d_state.clone())
        d_td_c = (d_state * (decay_c[..., None] * prev_states_py[:, c])).sum(dim=-1)
        d_td_py_list.append(d_td_c)
        d_state = decay_c[..., None] * d_state + d_prev_states_inter[:, c]
    d_init_state_py = d_state
    d_cns_py = torch.stack(d_cns_py_list[::-1], dim=1)
    d_td_py = torch.stack(d_td_py_list[::-1], dim=1)

    # Triton
    d_cns_tr, d_td_tr, d_init_state_tr = _state_passing_bwd_triton(
        d_prev_states_inter, prev_states_py, chunk_total_decay, B, nchunks, H, D)

    passed, info = _compare("bwd d_chunk_new_state", d_cns_py, d_cns_tr)
    print(info)
    all_passed = all_passed and passed

    passed, info = _compare("bwd d_chunk_total_decay", d_td_py, d_td_tr)
    print(info)
    all_passed = all_passed and passed

    passed, info = _compare("bwd d_init_state", d_init_state_py, d_init_state_tr)
    print(info)
    all_passed = all_passed and passed

    if all_passed:
        print("  All Triton kernel tests: PASSED")
    else:
        print("  SOME Triton kernel tests FAILED")
    assert all_passed, "Triton kernel tests FAILED"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of CUDA")
    parser.add_argument("--bench-only", action="store_true", help="Run only speed benchmarks")
    parser.add_argument("--no-bench", action="store_true", help="Skip speed benchmarks")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")
    print("=" * 60)

    if not args.bench_only:
        test_forward_match(device)
        test_backward_match(device)
        test_backward_match_random_dout(device)
        test_triton_kernels(device)

        print("\n" + "=" * 60)
        print("All validation tests PASSED!")
        print("=" * 60)

    if not args.no_bench:
        bench_speed(device)
