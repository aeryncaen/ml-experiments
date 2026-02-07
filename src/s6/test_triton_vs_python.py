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

from .scan import (
    _forward_scan_chunked_autograd,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Run on CPU instead of CUDA")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")
    print("=" * 60)

    test_forward_match(device)
    test_backward_match(device)
    test_backward_match_random_dout(device)

    print("\n" + "=" * 60)
    print("All validation tests PASSED!")
    print("=" * 60)
