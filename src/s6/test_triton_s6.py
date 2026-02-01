"""Numerical validation: TritonS6 vs PyTorch S6.

Run on CUDA box:
    python -m src.s6.test_triton_s6
"""

import torch
import torch.nn as nn

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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

    # CUDA graph path
    model_cg = TritonS6(d_model=H, d_state=P, M=M, chunk_size=32).cuda()
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(model_pt.named_parameters(), model_cg._pytorch_s6.named_parameters()):
            p2.copy_(p1)
    sample = torch.randn(B, L, H, device='cuda')
    model_cg.enable_cuda_graph(sample)
    # Warmup graph replay
    for _ in range(warmup):
        model_cg(sample)
        model_cg.zero_grad()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        model_cg(sample)
        model_cg.zero_grad()
        torch.cuda.synchronize()
    cg_ms = (time.perf_counter() - start) / iters * 1000

    print(f"\nBenchmark (H={H}, P={P}, L={L}, B={B}, M={M}):")
    print(f"  PyTorch:    {pt_ms:.1f} ms")
    print(f"  Triton:     {tr_ms:.1f} ms")
    print(f"  Triton+CG:  {cg_ms:.1f} ms")
    print(f"  Speedup (Triton):    {pt_ms / tr_ms:.2f}x")
    print(f"  Speedup (Triton+CG): {pt_ms / cg_ms:.2f}x")


def bench_profile(H=64, P=64, L=512, B=4, M=4, warmup=5, iters=20):
    """Profile where time is spent in the Triton path."""
    import time
    from .triton_s6 import TritonS6, _TritonPhiB, _TritonS6

    torch.manual_seed(SEED)
    model = TritonS6(d_model=H, d_state=P, M=M, chunk_size=32).cuda()
    x = torch.randn(B, L, H, device='cuda')

    s6 = model._pytorch_s6
    kern = s6.kernel
    phi = kern.phi_B
    mlp = phi.channel_mlp

    def sync():
        torch.cuda.synchronize()

    def timed(fn, label):
        # warmup
        for _ in range(warmup):
            fn()
        sync()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
            sync()
        ms = (time.perf_counter() - start) / iters * 1000
        print(f"  {label:30s}: {ms:.2f} ms")
        return ms

    print(f"\nProfile (H={H}, P={P}, L={L}, B={B}, M={M}):")
    print("  --- Forward ---")

    N_rows = B * L

    # phi_B (Triton)
    def f_phi_b():
        raw = _TritonPhiB.apply(
            x.reshape(N_rows, H), phi.W, phi.b,
            mlp.fc1.weight, mlp.fc1.bias, mlp.fc2.weight, mlp.fc2.bias,
            phi.M, phi.L, mlp.fc1.out_features,
        )
        if phi.ch_rms:
            raw_3d = raw.view(N_rows, phi.M, phi.L)
            rms = torch.sqrt(raw_3d.pow(2).mean(dim=(0, 1)) + 1e-6)
            s = (phi.ch_rms_target / (rms + 1e-6)).clamp(max=1.0)
            raw = (raw_3d * s.view(1, 1, phi.L)).view(N_rows, -1)
        return (raw * phi.scale).view(B, L, -1)
    timed(f_phi_b, "phi_B (Triton)")

    Bu_raw = f_phi_b()

    # SSM core (Triton)
    C = torch.view_as_complex(s6.C)
    C_re = C.real.contiguous()
    C_im = C.imag.contiguous()

    def f_ssm():
        return _TritonS6.apply(
            x, Bu_raw,
            kern.x_proj.weight, kern.x_proj.bias,
            kern.b_norm.weight, kern.b_bias, kern.log_dt_bias,
            kern.log_A_real, kern.A_imag,
            s6.c_proj.weight, s6.c_proj.bias,
            s6.c_norm.weight, s6.c_bias,
            C_re, C_im, s6.D,
            P, 32,
        )
    timed(f_ssm, "SSM core (Triton)")

    y_ssm = f_ssm()

    # readout norm + residual
    def f_norm():
        return s6.readout_norm(y_ssm) + x
    timed(f_norm, "readout_norm + residual")

    y_normed = f_norm()

    # msconv (after SSM)
    from .triton_s6 import _TritonMSConv
    msconv = s6.msconv
    def f_msconv():
        return _TritonMSConv.apply(
            y_normed,
            msconv.convs[0].weight, msconv.convs[0].bias,
            msconv.convs[1].weight, msconv.convs[1].bias,
            msconv.convs[2].weight, msconv.convs[2].bias,
            msconv.se.fc1.weight, msconv.se.fc2.weight,
            msconv.group_size,
        )
    timed(f_msconv, "msconv (Triton)")

    y_conv = f_msconv()
    c_proj_out = s6.c_proj(x)

    # c_proj recompute
    def f_cproj():
        return s6.c_proj(x)
    timed(f_cproj, "c_proj recompute")

    # attention
    def f_attn():
        return s6.attn(y_conv, y_conv)
    timed(f_attn, "attention")

    # --- Backward profiling ---
    # We need to run forward, then time each backward piece.
    # Strategy: run full fwd, then backward with hooks on intermediate tensors.
    print("  --- Backward (via torch.autograd.profiler) ---")

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        xx = x.clone().requires_grad_(True)
        y = model(xx)
        y.sum().backward()
        model.zero_grad()

    # Print top CUDA ops by self time
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25)
    print(table)

    # Full fwd+bwd
    print("  --- Full fwd+bwd ---")
    def f_full():
        xx = x.clone().requires_grad_(True)
        y = model(xx)
        y.sum().backward()
        model.zero_grad()
    timed(f_full, "TOTAL fwd+bwd")


def debug_forward_stepwise():
    """Step-by-step comparison to find where Triton diverges from PyTorch.

    Tests the new fused kernels: fused_prescan, fused_scan, fused_cgate, fused_readout.
    """
    from .s6 import S6, apply_rotary_emb
    from .triton_s6 import (TritonS6, triton_linear,
                             fused_prescan_kernel, fused_scan_kernel,
                             fused_cgate_kernel, fused_readout_kernel)
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

    # Step 0: phi_B (Triton vs PyTorch)
    from .triton_s6 import _TritonPhiB
    with torch.no_grad():
        Bu_raw_pt = kern.phi_B(x.unsqueeze(1)).squeeze(1)  # (B, L, P)
        phi = kern.phi_B
        mlp = phi.channel_mlp
        raw_tr = _TritonPhiB.apply(
            x.reshape(ML, H), phi.W, phi.b,
            mlp.fc1.weight, mlp.fc1.bias, mlp.fc2.weight, mlp.fc2.bias,
            phi.M, phi.L, mlp.fc1.out_features,
        )
        # Apply ch_rms + scale same as TritonS6.forward
        if phi.ch_rms:
            raw_3d = raw_tr.view(ML, phi.M, phi.L)
            rms = torch.sqrt(raw_3d.pow(2).mean(dim=(0, 1)) + 1e-6)
            s = (phi.ch_rms_target / (rms + 1e-6)).clamp(max=1.0)
            raw_tr = (raw_3d * s.view(1, 1, phi.L)).view(ML, -1)
        Bu_raw_tr = (raw_tr * phi.scale).view(B, L, -1)
    cmp("phi_B output", Bu_raw_pt, Bu_raw_tr)

    # Step 1: x_proj linear
    with torch.no_grad():
        xp_pt = kern.x_proj(x)  # (B, L, P+P+P//2)
        xp_tr = triton_linear(x.reshape(ML, H), kern.x_proj.weight, kern.x_proj.bias)
    cmp("x_proj", xp_pt.reshape(ML, -1), xp_tr)

    # Step 2: fused_prescan (dt/lam activations + Bu rmsnorm + bias + dt_half*theta)
    with torch.no_grad():
        import torch.nn.functional as F

        # PyTorch reference
        dt_raw_pt, lam_raw_pt, theta_pt = xp_pt.split(kern._split_sizes, dim=-1)
        dt_pt = F.softplus(F.silu(dt_raw_pt) + kern.log_dt_bias)
        lam_pt = torch.sigmoid(lam_raw_pt)
        Bu_pt = kern.b_norm(Bu_raw_pt) + kern.b_bias
        dt_half_pt = dt_pt.view(B, L, P // 2, 2).mean(-1)
        dt_half_theta_pt = dt_half_pt * theta_pt

        # Triton fused prescan
        dt_raw_tr, lam_raw_tr, theta_tr = xp_tr.split(kern._split_sizes, dim=-1)
        dt_raw_tr = dt_raw_tr.contiguous()
        lam_raw_tr = lam_raw_tr.contiguous()
        theta_tr = theta_tr.contiguous()
        Bu_raw_flat = Bu_raw_pt.reshape(ML, P).contiguous()

        dt_tr = torch.empty(ML, P, device='cuda')
        lam_tr = torch.empty(ML, P, device='cuda')
        Bu_tr = torch.empty(ML, P, device='cuda')
        dt_half_theta_tr = torch.empty(ML, P // 2, device='cuda')

        BLOCK_P = triton.next_power_of_2(P)
        fused_prescan_kernel[(ML,)](
            dt_raw_tr, lam_raw_tr, theta_tr,
            Bu_raw_flat,
            kern.log_dt_bias, kern.b_norm.weight, kern.b_bias,
            dt_tr, lam_tr, Bu_tr, dt_half_theta_tr,
            ML, P, 1e-6,
            BLOCK_P,
        )

    cmp("dt", dt_pt.reshape(ML, P), dt_tr)
    cmp("lam", lam_pt.reshape(ML, P), lam_tr)
    cmp("Bu (norm+bias)", Bu_pt.reshape(ML, P), Bu_tr)
    cmp("dt_half*theta", dt_half_theta_pt.reshape(ML, P // 2), dt_half_theta_tr)

    # Step 3: cumsum (same op)
    with torch.no_grad():
        cum_theta_pt = torch.cumsum(dt_half_theta_pt, dim=1)
        cum_theta_tr = torch.cumsum(dt_half_theta_tr.view(B, L, P // 2), dim=1).contiguous()
    cmp("cum_theta", cum_theta_pt, cum_theta_tr)

    # Step 4: fused_scan (RoPE + discretize + recurrence)
    with torch.no_grad():
        # PyTorch reference
        Bu_rot_pt = apply_rotary_emb(Bu_pt, cum_theta_pt)
        Bu_prev_pt = F.pad(Bu_rot_pt[:, :-1], (0, 0, 1, 0))
        dt_3d = dt_pt.view(B, L, P)
        A = -torch.exp(kern.log_A_real) + 1j * kern.A_imag
        alpha_pt = torch.exp(dt_3d.to(torch.cfloat) * A)
        Bu_c = Bu_rot_pt.to(torch.cfloat)
        Bu_prev_c = Bu_prev_pt.to(torch.cfloat)
        dt_c = dt_3d.to(torch.cfloat)
        lam_3d = lam_pt.view(B, L, P)
        inject_pt = lam_3d * dt_c * Bu_c + (1 - lam_3d) * dt_c * alpha_pt * Bu_prev_c
        from .scan import sequential_scan
        h_pt = sequential_scan(alpha_pt, inject_pt)

        # Triton fused scan
        A_real_neg = -torch.exp(kern.log_A_real)
        h_re_tr = torch.empty(B, L, P, device='cuda')
        h_im_tr = torch.empty(B, L, P, device='cuda')
        alpha_re_tr = torch.empty(B, L, P, device='cuda')
        alpha_im_tr = torch.empty(B, L, P, device='cuda')
        inject_re_tr = torch.empty(B, L, P, device='cuda')
        inject_im_tr = torch.empty(B, L, P, device='cuda')
        Bu_rot_tr = torch.empty(B, L, P, device='cuda')

        BLOCK_P_SCAN = triton.next_power_of_2(P)
        fused_scan_kernel[(B, triton.cdiv(P, BLOCK_P_SCAN))](
            dt_tr.view(B, L, P).contiguous(), lam_tr.view(B, L, P).contiguous(),
            Bu_tr.view(B, L, P).contiguous(), cum_theta_tr,
            A_real_neg, kern.A_imag,
            h_re_tr, h_im_tr, alpha_re_tr, alpha_im_tr,
            inject_re_tr, inject_im_tr, Bu_rot_tr,
            B, L, P,
            BLOCK_P_SCAN,
        )

    cmp("Bu rotated", Bu_rot_pt, Bu_rot_tr)
    cmp("alpha real", alpha_pt.real, alpha_re_tr)
    cmp("alpha imag", alpha_pt.imag, alpha_im_tr)
    cmp("inject real", inject_pt.real, inject_re_tr)
    cmp("inject imag", inject_pt.imag, inject_im_tr)
    cmp("h real", h_pt.real, h_re_tr)
    cmp("h imag", h_pt.imag, h_im_tr)

    # Step 5: c_proj linear
    with torch.no_grad():
        c_proj_pt = s6.c_proj(x)  # (B, L, P)
        c_proj_tr = triton_linear(x.reshape(ML, H), s6.c_proj.weight, s6.c_proj.bias)
    cmp("c_proj", c_proj_pt.reshape(ML, P), c_proj_tr)

    # Step 6: fused_cgate (silu + rmsnorm + bias + RoPE)
    with torch.no_grad():
        c_gate_pt = s6.c_norm(F.silu(c_proj_pt)) + s6.c_bias
        c_gate_pt = apply_rotary_emb(c_gate_pt, cum_theta_pt)

        c_gate_tr = torch.empty(ML, P, device='cuda')
        fused_cgate_kernel[(ML,)](
            c_proj_tr, cum_theta_tr.view(ML, P // 2),
            s6.c_norm.weight, s6.c_bias,
            c_gate_tr,
            ML, P, B, L, 1e-6,
            BLOCK_P,
        )
    cmp("c_gate", c_gate_pt.reshape(ML, P), c_gate_tr)

    # Step 7: fused_readout (gate + MIMO + skip + silu)
    with torch.no_grad():
        h_gated_pt = h_pt * c_gate_pt
        C = torch.view_as_complex(s6.C)
        y_pt = torch.einsum('hp,blp->blh', C, h_gated_pt.to(C.dtype)).real
        out_pt = F.silu(y_pt + x * s6.D)

        C_re = C.real.contiguous()
        C_im = C.imag.contiguous()
        BLOCK_H = min(64, triton.next_power_of_2(H))
        out_tr = torch.empty(B, L, H, device='cuda')
        fused_readout_kernel[(ML, triton.cdiv(H, BLOCK_H))](
            h_re_tr.view(ML, P), h_im_tr.view(ML, P),
            c_gate_tr,
            C_re, C_im,
            x.reshape(ML, H), s6.D,
            out_tr.view(ML, H),
            ML, H, P,
            BLOCK_H, BLOCK_P,
        )
    cmp("output", out_pt, out_tr.view(B, L, H))

    # Step 8: full forward comparison (end-to-end)
    with torch.no_grad():
        full_pt = s6(x)
        full_tr = model(x)
    cmp("FULL FORWARD", full_pt, full_tr)


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
    bench_profile()
