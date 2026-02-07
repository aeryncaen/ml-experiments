"""
Tests for USB block and scan implementations.
"""

import torch
import torch.nn as nn

from .usb_block import USBBlock, USBConfig


def test_chunked_scan_vs_streaming():
    """Test that chunked scan matches the streaming reference exactly."""
    from .scan import forward_scan_chunked, forward_scan_elementwise_streaming

    print("\nChunked scan vs streaming reference:")

    for T in [32, 64, 128, 200, 256]:
        for chunk_size in [32, 64, 128]:
            if chunk_size > T:
                continue
            B, H, D = 2, 4, 64
            kv = torch.randn(B, T, H, D)
            alpha = torch.sigmoid(torch.randn(B, T, H))
            delta = torch.randn(B, T, H) * 0.1
            epsilon = torch.randn(B, T, H) * 0.1
            zeta = torch.randn(B, T, H) * 0.1
            init_state = torch.randn(B, H, D) * 0.1

            ref = forward_scan_elementwise_streaming(kv, alpha, delta, epsilon, zeta, init_state)
            out = forward_scan_chunked(kv, alpha, delta, epsilon, zeta, init_state, chunk_size=chunk_size)

            diff = (ref.float() - out.float()).abs().max().item()
            rel_diff = diff / (ref.float().abs().max().item() + 1e-8)
            status = "PASSED" if rel_diff < 1e-4 else f"FAILED (rel={rel_diff:.2e})"
            print(f"  T={T:4d} C={chunk_size:3d}: max_abs={diff:.2e} rel={rel_diff:.2e} {status}")
            assert rel_diff < 1e-4, f"Chunked scan mismatch: T={T}, C={chunk_size}, rel_diff={rel_diff}"

    print("  All chunked scan tests: PASSED")


def test_chunked_scan_backward():
    """Test that gradients flow through the chunked scan."""
    from .scan import forward_scan_chunked

    print("\nChunked scan backward (flow):")

    B, T, H, D = 2, 128, 4, 64
    kv = torch.randn(B, T, H, D, requires_grad=True)
    alpha_raw = torch.randn(B, T, H, requires_grad=True)
    delta_raw = torch.randn(B, T, H, requires_grad=True)
    epsilon_raw = torch.randn(B, T, H, requires_grad=True)
    zeta_raw = torch.randn(B, T, H, requires_grad=True)
    init_state = torch.randn(B, H, D, requires_grad=True)

    alpha = torch.sigmoid(alpha_raw)
    delta = delta_raw * 0.1
    epsilon = epsilon_raw * 0.1
    zeta = zeta_raw * 0.1

    out = forward_scan_chunked(kv, alpha, delta, epsilon, zeta, init_state, chunk_size=64)
    loss = out.sum()
    loss.backward()

    for name, t in [("kv", kv), ("alpha_raw", alpha_raw), ("delta_raw", delta_raw),
                    ("epsilon_raw", epsilon_raw), ("zeta_raw", zeta_raw), ("init_state", init_state)]:
        assert t.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(t.grad).any(), f"Gradient for {name} contains NaN"
        print(f"  {name} grad: OK (rms={t.grad.float().pow(2).mean().sqrt().item():.4e})")

    print("  Backward flow: PASSED")


def test_chunked_scan_gradcheck():
    """Test gradient correctness via finite differences (torch.autograd.gradcheck)."""
    from .scan import forward_scan_chunked

    print("\nChunked scan gradcheck:")

    # Small sizes for finite diff
    B, T, H, D = 1, 16, 2, 8
    torch.manual_seed(42)
    kv = torch.randn(B, T, H, D, dtype=torch.float64, requires_grad=True)
    alpha = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float64, requires_grad=True))
    delta = torch.randn(B, T, H, dtype=torch.float64, requires_grad=True) * 0.1
    epsilon = torch.randn(B, T, H, dtype=torch.float64, requires_grad=True) * 0.1
    zeta = torch.randn(B, T, H, dtype=torch.float64, requires_grad=True) * 0.1
    init_state = torch.randn(B, H, D, dtype=torch.float64, requires_grad=True) * 0.1

    def fn(kv, alpha, delta, epsilon, zeta, init_state):
        return forward_scan_chunked(kv, alpha, delta, epsilon, zeta, init_state, chunk_size=8)

    passed = torch.autograd.gradcheck(fn, (kv, alpha, delta, epsilon, zeta, init_state),
                                       eps=1e-6, atol=1e-4, rtol=1e-3)
    assert passed, "gradcheck FAILED"
    print("  gradcheck: PASSED")


def test_usb_forward():
    """Test basic forward pass."""
    config = USBConfig(
        d_model=256,
        headdim=64,
        expansion_factor=2,
        scan_state_modes=('elementwise', 'elementwise', 'elementwise'),
    )

    print(f"\nUSB forward pass:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_expanded: {config.d_expanded}")
    print(f"  d_group: {config.d_group}")
    print(f"  nheads_per_group: {config.nheads_per_group}")
    print(f"  nheads_total: {config.nheads_total}")

    model = USBBlock(config)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  n_params: {n_params:,}")

    batch_size = 2
    seq_len = 128

    x = torch.randn(batch_size, seq_len, config.d_model)

    with torch.no_grad():
        out = model(x)

    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")

    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("  Shape check: PASSED")

    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    assert out.abs().sum() > 0, "Output is all zeros"
    print("  Numerical check: PASSED")


def test_usb_backward():
    """Test backward pass (gradient flow)."""
    config = USBConfig(
        d_model=256,
        headdim=64,
        expansion_factor=2,
        scan_state_modes=('elementwise', 'elementwise', 'elementwise'),
    )

    model = USBBlock(config)

    batch_size = 2
    seq_len = 64

    x = torch.randn(batch_size, seq_len, config.d_model, requires_grad=True)

    out = model(x)
    loss = out.sum()
    loss.backward()

    print(f"\nUSB backward pass:")
    print(f"  Input grad shape: {x.grad.shape}")

    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
    print("  Gradient check: PASSED")

    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"
    print("  Parameter gradient check: PASSED")


def test_scan_functions():
    """Test legacy scan functions."""
    from .scan import forward_scan, backward_scan, centered_scan

    batch = 2
    seq_len = 32
    nheads = 4
    headdim = 64

    kv = torch.randn(batch, seq_len, nheads, headdim)
    alpha = torch.sigmoid(torch.randn(batch, seq_len, nheads))
    delta = torch.randn(batch, seq_len, nheads)
    epsilon = torch.randn(batch, seq_len, nheads)
    zeta = torch.randn(batch, seq_len, nheads)
    init_state = torch.zeros(batch, nheads, headdim)

    print(f"\nLegacy scan functions:")

    state_fwd = forward_scan(kv, alpha, delta, epsilon, zeta, init_state)
    assert state_fwd.shape == kv.shape
    assert not torch.isnan(state_fwd).any()
    print("  Forward scan: PASSED")

    state_bwd = backward_scan(kv, alpha, delta, epsilon, zeta, init_state)
    assert state_bwd.shape == kv.shape
    assert not torch.isnan(state_bwd).any()
    print("  Backward scan: PASSED")

    state_ctr = centered_scan(kv, alpha, delta, epsilon, zeta, init_state)
    assert state_ctr.shape == kv.shape
    assert not torch.isnan(state_ctr).any()
    print("  Centered scan: PASSED")


def test_rope_functions():
    """Test RoPE functions."""
    from .rope import apply_rope, apply_data_dependent_rope

    batch = 2
    seq_len = 32
    nheads = 4
    headdim = 64

    x = torch.randn(batch, seq_len, nheads, headdim)

    print(f"\nRoPE functions:")

    x_rope = apply_rope(x, seq_len)
    assert x_rope.shape == x.shape
    assert not torch.isnan(x_rope).any()
    print("  Standard RoPE: PASSED")

    freqs = torch.randn(batch, seq_len, nheads, headdim // 2)
    x_dd_rope = apply_data_dependent_rope(x, freqs)
    assert x_dd_rope.shape == x.shape
    assert not torch.isnan(x_dd_rope).any()
    print("  Data-dependent RoPE: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("USB Block Tests")
    print("=" * 60)

    test_chunked_scan_vs_streaming()
    test_chunked_scan_backward()
    test_chunked_scan_gradcheck()
    test_scan_functions()
    test_rope_functions()
    test_usb_forward()
    test_usb_backward()

    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
