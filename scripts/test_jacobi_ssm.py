"""
Sanity tests for MIMOJacobiSSM.

Run: python scripts/test_jacobi_ssm.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
import torch.nn as nn

from heuristic_secrets.models.scatter_attention import MIMOJacobiSSM, MIMOJacobiSSM_ND


def test_output_shape():
    dim, seq_len, batch = 64, 32, 2
    ssm = MIMOJacobiSSM(dim, n_iters=4)
    x = torch.randn(batch, seq_len, dim)
    out = ssm(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print(f"[PASS] output shape: {out.shape}")


def test_convergence():
    """Output delta should shrink as we add more iterations."""
    dim, seq_len, batch = 64, 32, 2
    torch.manual_seed(42)
    x = torch.randn(batch, seq_len, dim)

    deltas = []
    prev_out = None
    for n_iters in [1, 2, 4, 8, 16, 32, 64, 128]:
        torch.manual_seed(0)
        ssm = MIMOJacobiSSM(dim, n_iters=n_iters)
        ssm.eval()
        with torch.no_grad():
            out = ssm(x)
        if prev_out is not None:
            delta = (out - prev_out).abs().mean().item()
            deltas.append((n_iters, delta))
        prev_out = out.clone()

    print(f"[INFO] convergence (iter → Δ from prev):")
    for n, d in deltas:
        print(f"       iters={n:2d}  Δ={d:.6f}")

    # last delta should be smaller than first
    if deltas[-1][1] < deltas[0][1]:
        print(f"[PASS] convergence: Δ shrinks ({deltas[0][1]:.6f} → {deltas[-1][1]:.6f})")
    else:
        print(f"[WARN] no convergence: Δ grew ({deltas[0][1]:.6f} → {deltas[-1][1]:.6f})")


def test_gradient_flow():
    """Gradients should reach all parameters through n_iters iterations."""
    dim, seq_len, batch = 64, 32, 2
    ssm = MIMOJacobiSSM(dim, n_iters=4)
    x = torch.randn(batch, seq_len, dim, requires_grad=True)
    out = ssm(x)
    loss = out.sum()
    loss.backward()

    dead_params = []
    for name, p in ssm.named_parameters():
        if p.grad is None or p.grad.abs().max().item() == 0:
            dead_params.append(name)

    if dead_params:
        print(f"[FAIL] dead gradients: {dead_params}")
    else:
        print(f"[PASS] all {sum(1 for _ in ssm.parameters())} params have nonzero gradients")

    assert x.grad is not None and x.grad.abs().max().item() > 0, "No gradient to input"
    print(f"[PASS] input gradient flows (max={x.grad.abs().max().item():.6f})")


def test_positional_sensitivity():
    """Output at position t should depend on positions < t (via diffuse conv)."""
    dim, seq_len, batch = 64, 32, 1
    torch.manual_seed(42)
    ssm = MIMOJacobiSSM(dim, n_iters=4)
    ssm.eval()

    x = torch.randn(batch, seq_len, dim)
    with torch.no_grad():
        out_orig = ssm(x).clone()

    # perturb early positions
    x_perturbed = x.clone()
    x_perturbed[:, :4, :] += 5.0
    with torch.no_grad():
        out_pert = ssm(x_perturbed)

    # check that later positions are affected
    early_delta = (out_pert[:, :4] - out_orig[:, :4]).abs().mean().item()
    late_delta = (out_pert[:, 16:] - out_orig[:, 16:]).abs().mean().item()

    print(f"[INFO] positional sensitivity:")
    print(f"       early positions (0-3) Δ={early_delta:.6f}")
    print(f"       late positions (16+)  Δ={late_delta:.6f}")

    if late_delta > 1e-6:
        print(f"[PASS] perturbation at t<4 affects t>16 (Δ={late_delta:.6f})")
    else:
        print(f"[FAIL] no information propagation to later positions")


def test_decay_gate_behavior():
    """With decay forced to 0, state should not carry across iterations."""
    dim, seq_len, batch = 64, 16, 1
    torch.manual_seed(42)
    ssm = MIMOJacobiSSM(dim, n_iters=4)
    ssm.eval()

    x = torch.randn(batch, seq_len, dim)

    # force decay bias very negative so sigmoid → 0
    with torch.no_grad():
        ssm.to_decay.bias.fill_(-20.0)

    with torch.no_grad():
        out_no_memory = ssm(x)

    # force decay bias very positive so sigmoid → 1
    with torch.no_grad():
        ssm.to_decay.bias.fill_(20.0)
        out_full_memory = ssm(x)

    delta = (out_no_memory - out_full_memory).abs().mean().item()
    print(f"[INFO] decay gate test: Δ between decay=0 and decay=1: {delta:.6f}")
    if delta > 0.01:
        print(f"[PASS] decay gate controls state retention")
    else:
        print(f"[WARN] decay gate has little effect (Δ={delta:.6f})")


def test_determinism():
    """Same input + same weights → same output."""
    dim, seq_len, batch = 64, 32, 2
    torch.manual_seed(0)
    ssm = MIMOJacobiSSM(dim, n_iters=4)
    ssm.eval()
    x = torch.randn(batch, seq_len, dim)

    with torch.no_grad():
        out1 = ssm(x).clone()
        out2 = ssm(x).clone()

    diff = (out1 - out2).abs().max().item()
    assert diff == 0.0, f"Non-deterministic: max diff={diff}"
    print(f"[PASS] deterministic (diff={diff})")


def test_iter_count_matters():
    """More iterations should produce different (not identical) output."""
    dim, seq_len, batch = 64, 32, 2
    torch.manual_seed(0)
    ssm1 = MIMOJacobiSSM(dim, n_iters=1)
    torch.manual_seed(0)
    ssm4 = MIMOJacobiSSM(dim, n_iters=4)
    ssm1.eval()
    ssm4.eval()

    x = torch.randn(batch, seq_len, dim)
    with torch.no_grad():
        out1 = ssm1(x)
        out4 = ssm4(x)

    diff = (out1 - out4).abs().mean().item()
    print(f"[INFO] 1-iter vs 4-iter mean Δ={diff:.6f}")
    if diff > 1e-4:
        print(f"[PASS] additional iterations change output")
    else:
        print(f"[FAIL] iterations have no effect")


if __name__ == "__main__":
    print("=" * 60)
    print("MIMOJacobiSSM sanity tests")
    print("=" * 60)

    test_output_shape()
    print()
    test_determinism()
    print()
    test_gradient_flow()
    print()
    test_iter_count_matters()
    print()
    test_convergence()
    print()
    test_positional_sensitivity()
    print()
    test_decay_gate_behavior()

    print()
    print("=" * 60)
    print("All tests complete.")
