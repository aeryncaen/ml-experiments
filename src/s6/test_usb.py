"""
Basic tests for USB block.
"""

import torch
import torch.nn as nn

from .usb_block import USBBlock, USBConfig


def test_usb_forward():
    """Test basic forward pass."""
    config = USBConfig(
        d_model=256,
        headdim=64,
        expansion_factor=2,
    )
    
    print(f"Config:")
    print(f"  d_model: {config.d_model}")
    print(f"  d_expanded: {config.d_expanded}")
    print(f"  d_group: {config.d_group}")
    print(f"  nheads_per_group: {config.nheads_per_group}")
    print(f"  nheads_total: {config.nheads_total}")
    
    model = USBBlock(config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  n_params: {n_params:,}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 128
    
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    with torch.no_grad():
        out = model(x)
    
    print(f"\nForward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {out.shape}")
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    print("  Shape check: PASSED")
    
    # Check that output is not all zeros or NaN
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
    )
    
    model = USBBlock(config)
    
    batch_size = 2
    seq_len = 64
    
    x = torch.randn(batch_size, seq_len, config.d_model, requires_grad=True)
    
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    print(f"\nBackward pass:")
    print(f"  Input grad shape: {x.grad.shape}")
    
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
    print("  Gradient check: PASSED")
    
    # Check that all parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN"
    print("  Parameter gradient check: PASSED")


def test_scan_functions():
    """Test individual scan functions."""
    from .scan import forward_scan, backward_scan, centered_scan
    
    batch = 2
    seq_len = 32
    nheads = 4
    headdim = 64
    
    kv = torch.randn(batch, seq_len, nheads, headdim)
    alpha = torch.sigmoid(torch.randn(batch, seq_len, nheads))  # (0, 1)
    c0 = torch.randn(batch, seq_len, nheads)
    c1 = torch.randn(batch, seq_len, nheads)
    c2 = torch.randn(batch, seq_len, nheads)
    init_state = torch.zeros(batch, nheads, headdim)
    
    print(f"\nScan functions:")
    
    # Forward scan
    state_fwd = forward_scan(kv, alpha, c0, c1, c2, init_state)
    assert state_fwd.shape == kv.shape, f"Forward scan shape mismatch"
    assert not torch.isnan(state_fwd).any(), "Forward scan contains NaN"
    print("  Forward scan: PASSED")
    
    # Backward scan
    state_bwd = backward_scan(kv, alpha, c0, c1, c2, init_state)
    assert state_bwd.shape == kv.shape, f"Backward scan shape mismatch"
    assert not torch.isnan(state_bwd).any(), "Backward scan contains NaN"
    print("  Backward scan: PASSED")
    
    # Centered scan
    state_ctr = centered_scan(kv, alpha, c0, c1, c2, init_state)
    assert state_ctr.shape == kv.shape, f"Centered scan shape mismatch"
    assert not torch.isnan(state_ctr).any(), "Centered scan contains NaN"
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
    
    # Standard RoPE
    x_rope = apply_rope(x, seq_len)
    assert x_rope.shape == x.shape, "Standard RoPE shape mismatch"
    assert not torch.isnan(x_rope).any(), "Standard RoPE contains NaN"
    print("  Standard RoPE: PASSED")
    
    # Data-dependent RoPE
    freqs = torch.randn(batch, seq_len, nheads, headdim // 2)
    x_dd_rope = apply_data_dependent_rope(x, freqs)
    assert x_dd_rope.shape == x.shape, "DD-RoPE shape mismatch"
    assert not torch.isnan(x_dd_rope).any(), "DD-RoPE contains NaN"
    print("  Data-dependent RoPE: PASSED")


def test_usb_lowrank():
    """Test USB with low-rank attention."""
    config = USBConfig(
        d_model=256,
        headdim=64,
        expansion_factor=2,
        attention_type="lowrank",
        lowrank_factor=1.5,
    )
    
    print(f"\nLow-rank attention:")
    print(f"  attention_type: {config.attention_type}")
    print(f"  lowrank_factor: {config.lowrank_factor}")
    
    model = USBBlock(config)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  n_params: {n_params:,}")
    
    # Test with longer sequence to see compression
    batch_size = 2
    seq_len = 256
    r = int((seq_len * config.lowrank_factor) ** 0.5)
    print(f"  seq_len: {seq_len}, compressed to r={r}")
    
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    with torch.no_grad():
        out = model(x)
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    print("  Forward pass: PASSED")
    
    # Test backward
    x.requires_grad = True
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
    print("  Backward pass: PASSED")


def test_usb_linear():
    """Test USB with linear attention."""
    config = USBConfig(
        d_model=256,
        headdim=64,
        expansion_factor=2,
        attention_type="linear",
        linear_feature_map="elu",
    )
    
    print(f"\nLinear attention:")
    print(f"  attention_type: {config.attention_type}")
    print(f"  linear_feature_map: {config.linear_feature_map}")
    
    model = USBBlock(config)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  n_params: {n_params:,}")
    
    batch_size = 2
    seq_len = 256
    print(f"  seq_len: {seq_len} (O(L·d²) complexity)")
    
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    with torch.no_grad():
        out = model(x)
    
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} vs {x.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert not torch.isinf(out).any(), "Output contains Inf"
    print("  Forward pass: PASSED")
    
    # Test backward
    x.requires_grad = True
    out = model(x)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "No gradient for input"
    assert not torch.isnan(x.grad).any(), "Input gradient contains NaN"
    print("  Backward pass: PASSED")


if __name__ == "__main__":
    print("=" * 60)
    print("USB Block Tests")
    print("=" * 60)
    
    test_scan_functions()
    test_rope_functions()
    test_usb_forward()
    test_usb_backward()
    test_usb_lowrank()
    test_usb_linear()
    
    print("\n" + "=" * 60)
    print("All tests PASSED!")
    print("=" * 60)
