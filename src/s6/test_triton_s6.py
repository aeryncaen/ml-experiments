"""Numerical validation: TritonS6 vs PyTorch S6.

Run on CUDA box:
    python -m src.s6.test_triton_s6
"""

import torch
import torch.nn as nn

SEED = 42


def test_forward_match():
    """Check that Triton forward matches PyTorch forward."""
    from .s6 import S6
    from .triton_s6 import TritonS6

    torch.manual_seed(SEED)
    H, P, L, B = 32, 64, 128, 2

    model = TritonS6(d_model=H, d_state=P, chunk_size=32).cuda()
    pytorch_s6 = model._pytorch_s6

    torch.manual_seed(SEED + 1)
    x = torch.randn(B, L, H, device='cuda')

    # PyTorch reference
    with torch.no_grad():
        y_pt = pytorch_s6(x)

    # Triton
    with torch.no_grad():
        y_tr = model(x)

    max_diff = (y_pt - y_tr).abs().max().item()
    mean_diff = (y_pt - y_tr).abs().mean().item()
    rel_diff = ((y_pt - y_tr).abs() / (y_pt.abs() + 1e-8)).mean().item()

    print(f"Forward match:")
    print(f"  max abs diff:  {max_diff:.2e}")
    print(f"  mean abs diff: {mean_diff:.2e}")
    print(f"  mean rel diff: {rel_diff:.2e}")
    print(f"  PASS: {max_diff < 1e-2}")
    return max_diff < 1e-2


def test_backward_match():
    """Check that Triton backward matches PyTorch backward on all params."""
    from .s6 import S6
    from .triton_s6 import TritonS6

    torch.manual_seed(SEED)
    H, P, L, B = 32, 64, 64, 2

    # Build Triton model
    model_tr = TritonS6(d_model=H, d_state=P, chunk_size=32).cuda()

    # Build separate PyTorch model with identical weights
    model_pt = model_tr._pytorch_s6

    # We need two separate forward/backward passes, so clone params into a fresh model
    # Actually — model_tr.forward with CUDA calls Triton, model_pt.forward calls PyTorch
    # They share params, so we need to be careful. Let's just do them one at a time.

    torch.manual_seed(SEED + 1)
    x = torch.randn(B, L, H, device='cuda')

    # PyTorch backward
    x_pt = x.clone().requires_grad_(True)
    y_pt = model_pt(x_pt)
    y_pt.sum().backward()

    grads_pt = {}
    for name, p in model_pt.named_parameters():
        if p.grad is not None:
            grads_pt[name] = p.grad.clone()
    dx_pt = x_pt.grad.clone()

    # Zero grads
    model_pt.zero_grad()

    # Triton backward (through model_tr which calls _TritonS6.apply)
    x_tr = x.clone().requires_grad_(True)
    y_tr = model_tr(x_tr)
    y_tr.sum().backward()

    # Collect grads — params are shared, so same names
    grads_tr = {}
    for name, p in model_pt.named_parameters():
        if p.grad is not None:
            grads_tr[name] = p.grad.clone()
    dx_tr = x_tr.grad.clone()

    print(f"\nBackward match:")
    all_pass = True

    # dx
    dx_diff = (dx_pt - dx_tr).abs().max().item()
    dx_rel = ((dx_pt - dx_tr).abs() / (dx_pt.abs() + 1e-8)).mean().item()
    ok = dx_diff < 1e-1
    all_pass &= ok
    print(f"  d_x:          max={dx_diff:.2e}  rel={dx_rel:.2e}  {'PASS' if ok else 'FAIL'}")

    for name in sorted(grads_pt.keys()):
        if name not in grads_tr:
            print(f"  {name:30s}: MISSING in Triton")
            all_pass = False
            continue
        g_pt = grads_pt[name]
        g_tr = grads_tr[name]
        max_d = (g_pt - g_tr).abs().max().item()
        rel_d = ((g_pt - g_tr).abs() / (g_pt.abs() + 1e-8)).mean().item()
        # Looser tolerance for accumulated grads
        tol = 1e-1 if g_pt.numel() > 100 else 5e-1
        ok = max_d < tol
        all_pass &= ok
        print(f"  {name:30s}: max={max_d:.2e}  rel={rel_d:.2e}  {'PASS' if ok else 'FAIL'}")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_gradient_flow():
    """Verify all params get gradients through Triton path."""
    from .triton_s6 import TritonS6

    torch.manual_seed(SEED)
    H, P, L, B = 32, 64, 32, 2

    model = TritonS6(d_model=H, d_state=P, chunk_size=16).cuda()
    x = torch.randn(B, L, H, device='cuda', requires_grad=True)
    y = model(x)
    y.sum().backward()

    print(f"\nGradient flow:")
    all_have_grad = True
    for name, p in model.named_parameters():
        has = p.grad is not None and p.grad.abs().max().item() > 0
        if not has:
            print(f"  {name:30s}: NO GRAD")
            all_have_grad = False
        else:
            print(f"  {name:30s}: grad norm {p.grad.norm().item():.4f}")
    print(f"  x.grad norm: {x.grad.norm().item():.4f}")
    print(f"  Overall: {'PASS' if all_have_grad else 'FAIL'}")
    return all_have_grad


def bench(H=64, P=64, L=512, B=4, warmup=5, iters=20):
    """Quick speed comparison."""
    import time
    from .s6 import S6
    from .triton_s6 import TritonS6

    torch.manual_seed(SEED)

    model_pt = S6(d_model=H, d_state=P).cuda()
    model_tr = TritonS6(d_model=H, d_state=P, chunk_size=32).cuda()

    # Copy weights
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(model_pt.named_parameters(), model_tr._pytorch_s6.named_parameters()):
            p2.copy_(p1)

    x = torch.randn(B, L, H, device='cuda')

    def time_fwd_bwd(model, x):
        x = x.clone().requires_grad_(True)
        for _ in range(warmup):
            y = model(x)
            y.sum().backward()
            model.zero_grad()
            if x.grad is not None:
                x.grad = None
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iters):
            y = model(x)
            y.sum().backward()
            model.zero_grad()
            if x.grad is not None:
                x.grad = None
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) / iters * 1000
        return elapsed

    pt_ms = time_fwd_bwd(model_pt, x)
    tr_ms = time_fwd_bwd(model_tr, x)

    print(f"\nBenchmark (H={H}, P={P}, L={L}, B={B}):")
    print(f"  PyTorch: {pt_ms:.1f} ms")
    print(f"  Triton:  {tr_ms:.1f} ms")
    print(f"  Speedup: {pt_ms / tr_ms:.2f}x")


if __name__ == '__main__':
    print("=" * 60)
    print("S6 Triton vs PyTorch Validation")
    print("=" * 60)

    ok1 = test_forward_match()
    ok2 = test_backward_match()
    ok3 = test_gradient_flow()

    print("\n" + "=" * 60)
    if ok1 and ok2 and ok3:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)

    bench()
