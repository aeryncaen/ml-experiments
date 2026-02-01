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
    H, P, L, B, M = 32, 16, 128, 2, 4

    model = TritonS6(d_model=H, d_state=P, M=M, chunk_size=32).cuda()
    pytorch_s6 = model._pytorch_s6

    torch.manual_seed(SEED + 1)
    x = torch.randn(B, L, H, device='cuda')

    with torch.no_grad():
        y_pt = pytorch_s6(x)
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
    """Check that Triton backward matches PyTorch backward on all params and dx."""
    from .s6 import S6
    from .triton_s6 import TritonS6

    torch.manual_seed(SEED)
    H, P, L, B, M = 32, 16, 64, 2, 4

    model_tr = TritonS6(d_model=H, d_state=P, M=M, chunk_size=32).cuda()
    model_pt = model_tr._pytorch_s6

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

    model_pt.zero_grad()

    # Triton backward
    x_tr = x.clone().requires_grad_(True)
    y_tr = model_tr(x_tr)
    y_tr.sum().backward()

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
    print(f"  {'d_x':35s}: max={dx_diff:.2e}  rel={dx_rel:.2e}  {'PASS' if ok else 'FAIL'}")

    # All named params
    all_names = sorted(set(list(grads_pt.keys()) + list(grads_tr.keys())))
    for name in all_names:
        if name not in grads_pt:
            print(f"  {name:35s}: MISSING in PyTorch")
            all_pass = False
            continue
        if name not in grads_tr:
            print(f"  {name:35s}: MISSING in Triton")
            all_pass = False
            continue
        g_pt = grads_pt[name]
        g_tr = grads_tr[name]
        max_d = (g_pt - g_tr).abs().max().item()
        rel_d = ((g_pt - g_tr).abs() / (g_pt.abs() + 1e-8)).mean().item()
        tol = 1e-1 if g_pt.numel() > 100 else 5e-1
        ok = max_d < tol
        all_pass &= ok
        print(f"  {name:35s}: max={max_d:.2e}  rel={rel_d:.2e}  {'PASS' if ok else 'FAIL'}")

    # Check no params were missed
    pt_names = set(n for n, p in model_pt.named_parameters())
    for name in pt_names:
        if name not in grads_pt and name not in grads_tr:
            print(f"  {name:35s}: NO GRAD in either (possible bug)")

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


def test_gradient_flow():
    """Verify all params get nonzero gradients through Triton path."""
    from .triton_s6 import TritonS6

    torch.manual_seed(SEED)
    H, P, L, B, M = 32, 16, 32, 2, 4

    model = TritonS6(d_model=H, d_state=P, M=M, chunk_size=16).cuda()
    x = torch.randn(B, L, H, device='cuda', requires_grad=True)
    y = model(x)
    y.sum().backward()

    print(f"\nGradient flow:")
    all_have_grad = True
    for name, p in model.named_parameters():
        has = p.grad is not None and p.grad.abs().max().item() > 0
        if not has:
            print(f"  {name:35s}: NO GRAD")
            all_have_grad = False
        else:
            print(f"  {name:35s}: grad norm {p.grad.norm().item():.4f}")
    print(f"  {'x.grad':35s}: norm {x.grad.norm().item():.4f}")
    print(f"  Overall: {'PASS' if all_have_grad else 'FAIL'}")
    return all_have_grad


def bench(H=64, P=64, L=512, B=4, M=4, warmup=5, iters=20):
    """Quick speed comparison."""
    import time
    from .s6 import S6
    from .triton_s6 import TritonS6

    torch.manual_seed(SEED)

    model_pt = S6(d_model=H, d_state=P, M=M).cuda()
    model_tr = TritonS6(d_model=H, d_state=P, M=M, chunk_size=32).cuda()

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

    print(f"\nBenchmark (H={H}, P={P}, L={L}, B={B}, M={M}):")
    print(f"  PyTorch: {pt_ms:.1f} ms")
    print(f"  Triton:  {tr_ms:.1f} ms")
    print(f"  Speedup: {pt_ms / tr_ms:.2f}x")


def debug_forward_stepwise():
    """Step-by-step comparison to find where Triton diverges from PyTorch."""
    from .s6 import S6, apply_rotary_emb
    from .triton_s6 import (TritonS6, triton_linear, triton_rmsnorm, triton_silu,
                             triton_fused_dt_lam, triton_rope, triton_complex_discretize,
                             triton_chunked_scan, complex_gate_readout_fwd_kernel)
    import triton

    torch.manual_seed(SEED)
    H, P, L, B, M = 32, 16, 32, 2, 4

    model = TritonS6(d_model=H, d_state=P, M=M, chunk_size=32).cuda()
    s6 = model._pytorch_s6
    kern = s6.kernel

    torch.manual_seed(SEED + 1)
    x = torch.randn(B, L, H, device='cuda')
    ML = B * L

    def cmp(name, pt, tr):
        d = (pt.float() - tr.float()).abs()
        print(f"  {name:30s}: max={d.max().item():.2e}  mean={d.mean().item():.2e}  "
              f"pt_range=[{pt.float().min().item():.3f},{pt.float().max().item():.3f}]")

    print("\nStep-by-step forward comparison:")

    # Step 0: phi_B (same code path)
    with torch.no_grad():
        Bu_raw_pt = kern.phi_B(x.unsqueeze(1)).squeeze(1)  # (B, L, P)
        Bu_raw_tr = kern.phi_B(x.unsqueeze(1)).squeeze(1)
    cmp("phi_B output", Bu_raw_pt, Bu_raw_tr)

    # Step 1: x_proj linear
    with torch.no_grad():
        xp_pt = kern.x_proj(x)  # (B, L, P+P+P//2)
        xp_tr = triton_linear(x.reshape(ML, H), kern.x_proj.weight, kern.x_proj.bias)
    cmp("x_proj", xp_pt.reshape(ML, -1), xp_tr)

    # Step 2: split and dt/lam
    with torch.no_grad():
        dt_raw_pt, lam_raw_pt, theta_pt = xp_pt.split(kern._split_sizes, dim=-1)
        dt_pt = torch.nn.functional.softplus(torch.nn.functional.silu(dt_raw_pt) + kern.log_dt_bias)
        lam_pt = torch.sigmoid(lam_raw_pt)

        dt_raw_tr, lam_raw_tr, theta_tr = xp_tr.split(kern._split_sizes, dim=-1)
        dt_raw_tr = dt_raw_tr.contiguous()
        lam_raw_tr = lam_raw_tr.contiguous()
        theta_tr = theta_tr.contiguous()
        dt_tr, lam_tr = triton_fused_dt_lam(dt_raw_tr, lam_raw_tr,
                                             kern.log_dt_bias.expand(ML, P).contiguous())
    cmp("dt", dt_pt.reshape(ML, P), dt_tr)
    cmp("lam", lam_pt.reshape(ML, P), lam_tr)

    # Step 3: B rmsnorm + bias
    with torch.no_grad():
        Bu_pt = kern.b_norm(Bu_raw_pt) + kern.b_bias
        Bu_normed_tr = triton_rmsnorm(Bu_raw_pt.reshape(ML, P), kern.b_norm.weight)
        Bu_tr = Bu_normed_tr + kern.b_bias.unsqueeze(0)
    cmp("Bu (norm+bias)", Bu_pt.reshape(ML, P), Bu_tr)

    # Step 4: RoPE + shift
    with torch.no_grad():
        dt_3d = dt_pt.view(B, L, P)
        dt_half = dt_3d.view(B, L, P//2, 2).mean(-1)
        cum_theta = torch.cumsum(dt_half * theta_pt, dim=1)

        Bu_rot_pt = apply_rotary_emb(Bu_pt, cum_theta)
        Bu_prev_pt = torch.nn.functional.pad(Bu_rot_pt[:, :-1], (0, 0, 1, 0))

        Bu_rot_tr = triton_rope(Bu_pt, cum_theta)  # use same Bu_pt to isolate rope
        Bu_prev_tr = torch.zeros_like(Bu_rot_tr)
        Bu_prev_tr[:, 1:] = Bu_rot_tr[:, :-1]
    cmp("Bu rotated", Bu_rot_pt, Bu_rot_tr)
    cmp("Bu_prev", Bu_prev_pt, Bu_prev_tr)

    # Step 5: discretization
    with torch.no_grad():
        A = -torch.exp(kern.log_A_real) + 1j * kern.A_imag
        alpha_pt = torch.exp(dt_3d.to(torch.cfloat) * A)
        Bu_c = Bu_rot_pt.to(torch.cfloat)
        Bu_prev_c = Bu_prev_pt.to(torch.cfloat)
        dt_c = dt_3d.to(torch.cfloat)
        lam_3d = lam_pt.view(B, L, P)
        inject_pt = lam_3d * dt_c * Bu_c + (1 - lam_3d) * dt_c * alpha_pt * Bu_prev_c

        A_real_neg = -torch.exp(kern.log_A_real)
        a_re_tr, a_im_tr, i_re_tr, i_im_tr = triton_complex_discretize(
            dt_3d, lam_3d, Bu_rot_pt, Bu_prev_pt, A_real_neg, kern.A_imag)
    cmp("alpha real", alpha_pt.real, a_re_tr)
    cmp("alpha imag", alpha_pt.imag, a_im_tr)
    cmp("inject real", inject_pt.real, i_re_tr)
    cmp("inject imag", inject_pt.imag, i_im_tr)

    # Step 6: scan
    with torch.no_grad():
        from .scan import sequential_scan
        h_pt = sequential_scan(alpha_pt, inject_pt)

        h_re_tr, h_im_tr = triton_chunked_scan(a_re_tr, a_im_tr, i_re_tr, i_im_tr, 32)
    cmp("h real", h_pt.real, h_re_tr)
    cmp("h imag", h_pt.imag, h_im_tr)

    # Step 7: C gate
    with torch.no_grad():
        import torch.nn.functional as F
        c_gate_pt = s6.c_norm(F.silu(s6.c_proj(x))) + s6.c_bias
        c_gate_pt = apply_rotary_emb(c_gate_pt, cum_theta)

        c_proj_out = triton_linear(x.reshape(ML, H), s6.c_proj.weight, s6.c_proj.bias)
        from .triton_s6 import triton_silu
        c_silu = triton_silu(c_proj_out)
        c_normed = triton_rmsnorm(c_silu, s6.c_norm.weight)
        c_gate_tr = (c_normed + s6.c_bias.unsqueeze(0)).view(B, L, P)
        c_gate_tr = triton_rope(c_gate_tr, cum_theta)
    cmp("c_gate", c_gate_pt, c_gate_tr)

    # Step 8: gated readout
    with torch.no_grad():
        h_gated_pt = h_pt * c_gate_pt
        C = torch.view_as_complex(s6.C)
        y_pt = torch.einsum('hp,blp->blh', C, h_gated_pt.to(C.dtype)).real

        # Triton gating
        N_elem = B * L * P
        BLOCK = 1024
        hg_re_tr = torch.empty(B, L, P, device='cuda')
        hg_im_tr = torch.empty(B, L, P, device='cuda')
        complex_gate_readout_fwd_kernel[(triton.cdiv(N_elem, BLOCK),)](
            h_re_tr.contiguous().view(-1), h_im_tr.contiguous().view(-1),
            c_gate_tr.contiguous().view(-1),
            hg_re_tr.view(-1), hg_im_tr.view(-1), N_elem, BLOCK)
        C_re = C.real.contiguous()
        C_im = C.imag.contiguous()
        y_re_tr = triton_linear(hg_re_tr.reshape(ML, P), C_re)
        y_im_tr = triton_linear(hg_im_tr.reshape(ML, P), C_im)
        y_tr = (y_re_tr - y_im_tr).view(B, L, H)
    cmp("y (readout)", y_pt, y_tr)

    # Step 9: skip + silu
    with torch.no_grad():
        out_pt = F.silu(y_pt + x * s6.D)

        from .triton_s6 import skip_silu_fwd_kernel
        N_out = B * L * H
        out_tr = torch.empty(B, L, H, device='cuda')
        skip_silu_fwd_kernel[(triton.cdiv(N_out, BLOCK),)](
            y_tr.contiguous().view(-1), x.contiguous().view(-1), s6.D,
            out_tr.view(-1), N_out, H, BLOCK)
    cmp("output (skip+silu)", out_pt, out_tr)


if __name__ == '__main__':
    print("=" * 60)
    print("S6 Triton vs PyTorch Validation")
    print("=" * 60)

    debug_forward_stepwise()

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
