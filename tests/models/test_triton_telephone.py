"""
Comprehensive tests for TelephoneAttention: Triton vs PyTorch sanity checks and benchmarks.

Tests are designed to:
1. Run on CPU (PyTorch implementation validation)
2. Run on CUDA (Triton vs PyTorch comparison when CUDA available)
3. Verify forward and backward correctness
4. Benchmark performance

Run with:
    pytest tests/models/test_triton_telephone.py -v
    
Run benchmarks (CUDA only):
    python tests/models/test_triton_telephone.py --benchmark
"""

import pytest
import torch
import torch.nn as nn
import time
from functools import reduce
from operator import mul

from heuristic_secrets.models.telephone_attention import (
    TelephoneAttentionND, 
    llama_rmsnorm,
    HAS_TRITON,
)

CUDA_AVAILABLE = torch.cuda.is_available()
TRITON_AVAILABLE = HAS_TRITON and CUDA_AVAILABLE


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def small_config():
    """Small config for quick tests."""
    return dict(
        channels=64,
        ndim=1,
        max_samples=8,
        num_heads=4,
        max_freq=8.0,
        min_freq=1.0,
        max_kernel_size=16,
        chunk_size=256,
        checkpoint=False,
    )


@pytest.fixture
def medium_config():
    """Medium config for more thorough tests."""
    return dict(
        channels=128,
        ndim=1,
        max_samples=16,
        num_heads=8,
        max_freq=16.0,
        min_freq=1.0,
        max_kernel_size=32,
        chunk_size=512,
        checkpoint=False,
    )


@pytest.fixture
def default_config():
    """Default config matching typical usage."""
    return dict(
        channels=256,
        ndim=1,
        max_samples=32,
        num_heads=8,
        max_freq=16.0,
        min_freq=1.0,
        max_kernel_size=64,
        chunk_size=1024,
        checkpoint=False,
    )


# ============================================================================
# PyTorch Implementation Tests (CPU)
# ============================================================================

class TestPyTorchImplementation:
    """Tests for PyTorch implementation that run on CPU."""
    
    def test_forward_shape(self, small_config):
        """Test forward pass produces correct output shape."""
        model = TelephoneAttentionND(**small_config, use_triton=False)
        B, L, C = 2, 128, small_config['channels']
        x = torch.randn(B, L, C)
        
        out, _ = model(x)
        
        assert out.shape == x.shape
    
    def test_forward_no_nan(self, small_config):
        """Test forward pass produces no NaN values."""
        model = TelephoneAttentionND(**small_config, use_triton=False)
        B, L, C = 2, 128, small_config['channels']
        x = torch.randn(B, L, C)
        
        out, _ = model(x)
        
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()
    
    def test_backward_gradients_exist(self, small_config):
        """Test backward pass produces gradients for all parameters."""
        model = TelephoneAttentionND(**small_config, use_triton=False)
        B, L, C = 2, 128, small_config['channels']
        x = torch.randn(B, L, C, requires_grad=True)
        
        out, _ = model(x)
        out.sum().backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_gradients_nonzero(self, small_config):
        """Test that key parameter gradients are non-zero."""
        model = TelephoneAttentionND(**small_config, use_triton=False)
        B, L, C = 2, 128, small_config['channels']
        x = torch.randn(B, L, C, requires_grad=True)
        
        out, _ = model(x)
        out.sum().backward()
        
        H = small_config['num_heads']
        
        freq_grad = model.wave_proj.weight.grad[:H].abs().sum().item()
        phase_grad = model.wave_proj.weight.grad[H:].abs().sum().item()
        assert freq_grad > 0, "freq gradients are zero"
        assert phase_grad > 0, "phase gradients are zero"
        
        kernel_grad = model.kernel_proj.weight.grad.abs().sum().item()
        assert kernel_grad > 0, "kernel_proj gradients are zero"
        
        exponent_grad = model.exponent_proj.weight.grad.abs().sum().item()
        assert exponent_grad > 0, "exponent_proj gradients are zero"
        
        out_grad = model.out_proj.weight.grad.abs().sum().item()
        assert out_grad > 0, "out_proj gradients are zero"
    
    def test_chunked_vs_full(self, small_config):
        """Test chunked processing matches full sequence processing."""
        config = small_config.copy()
        config['chunk_size'] = 32
        model_chunked = TelephoneAttentionND(**config, use_triton=False)
        
        config_full = small_config.copy()
        config_full['chunk_size'] = 1024
        model_full = TelephoneAttentionND(**config_full, use_triton=False)
        model_full.load_state_dict(model_chunked.state_dict())
        
        B, L, C = 2, 128, small_config['channels']
        torch.manual_seed(42)
        x = torch.randn(B, L, C)
        
        out_chunked, _ = model_chunked(x)
        out_full, _ = model_full(x)
        
        max_diff = (out_chunked - out_full).abs().max().item()
        assert max_diff < 1e-5, f"Chunked vs full mismatch: {max_diff}"
    
    def test_different_seq_lengths(self, small_config):
        """Test model works with various sequence lengths."""
        model = TelephoneAttentionND(**small_config, use_triton=False)
        B, C = 2, small_config['channels']
        
        for L in [16, 64, 128, 256, 512]:
            x = torch.randn(B, L, C)
            out, _ = model(x)
            assert out.shape == x.shape, f"Shape mismatch at L={L}"
            assert not torch.isnan(out).any(), f"NaN at L={L}"
    
    def test_deterministic(self, small_config):
        """Test forward pass is deterministic."""
        model = TelephoneAttentionND(**small_config, use_triton=False)
        model.eval()
        
        B, L, C = 2, 128, small_config['channels']
        x = torch.randn(B, L, C)
        
        with torch.no_grad():
            out1, _ = model(x)
            out2, _ = model(x)
        
        assert torch.allclose(out1, out2), "Forward pass not deterministic"


# ============================================================================
# Triton vs PyTorch Comparison Tests (CUDA only)
# ============================================================================

@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton/CUDA not available")
class TestTritonVsPyTorch:
    """Tests comparing Triton and PyTorch implementations."""
    
    def test_forward_match(self, small_config):
        """Test Triton forward matches PyTorch forward."""
        device = "cuda"
        
        model_triton = TelephoneAttentionND(**small_config, use_triton=True).to(device)
        model_pytorch = TelephoneAttentionND(**small_config, use_triton=False).to(device)
        model_pytorch.load_state_dict(model_triton.state_dict())
        
        B, L, C = 2, 256, small_config['channels']
        torch.manual_seed(42)
        x = torch.randn(B, L, C, device=device)
        
        with torch.no_grad():
            out_triton, _ = model_triton(x)
            out_pytorch, _ = model_pytorch(x)
        
        max_diff = (out_triton - out_pytorch).abs().max().item()
        assert max_diff < 1e-3, f"Forward mismatch: max diff = {max_diff}"
    
    def test_backward_dx_match(self, small_config):
        """Test Triton d_x gradient matches PyTorch."""
        device = "cuda"
        
        model_triton = TelephoneAttentionND(**small_config, use_triton=True).to(device)
        model_pytorch = TelephoneAttentionND(**small_config, use_triton=False).to(device)
        model_pytorch.load_state_dict(model_triton.state_dict())
        
        B, L, C = 2, 256, small_config['channels']
        torch.manual_seed(42)
        x_triton = torch.randn(B, L, C, device=device, requires_grad=True)
        x_pytorch = x_triton.detach().clone().requires_grad_(True)
        
        out_triton, _ = model_triton(x_triton)
        out_triton.sum().backward()
        
        out_pytorch, _ = model_pytorch(x_pytorch)
        out_pytorch.sum().backward()
        
        max_diff = (x_triton.grad - x_pytorch.grad).abs().max().item()
        assert max_diff < 5e-3, f"d_x mismatch: max diff = {max_diff}"
    
    def test_backward_wave_proj_match(self, small_config):
        """Test Triton wave_proj gradients match PyTorch."""
        device = "cuda"
        
        model_triton = TelephoneAttentionND(**small_config, use_triton=True).to(device)
        model_pytorch = TelephoneAttentionND(**small_config, use_triton=False).to(device)
        model_pytorch.load_state_dict(model_triton.state_dict())
        
        B, L, C = 2, 256, small_config['channels']
        torch.manual_seed(42)
        x = torch.randn(B, L, C, device=device)
        
        x_triton = x.clone().requires_grad_(True)
        x_pytorch = x.clone().requires_grad_(True)
        
        out_triton, _ = model_triton(x_triton)
        out_triton.sum().backward()
        
        out_pytorch, _ = model_pytorch(x_pytorch)
        out_pytorch.sum().backward()
        
        w_grad_tr = model_triton.wave_proj.weight.grad
        w_grad_pt = model_pytorch.wave_proj.weight.grad
        rel_diff = (w_grad_tr - w_grad_pt).abs() / (w_grad_pt.abs() + 1e-8)
        assert rel_diff.mean().item() < 0.2, f"wave_proj.weight grad mismatch: mean rel diff = {rel_diff.mean().item()}"
        
        b_grad_tr = model_triton.wave_proj.bias.grad
        b_grad_pt = model_pytorch.wave_proj.bias.grad
        rel_diff = (b_grad_tr - b_grad_pt).abs() / (b_grad_pt.abs() + 1e-8)
        assert rel_diff.mean().item() < 0.2, f"wave_proj.bias grad mismatch: mean rel diff = {rel_diff.mean().item()}"
    
    def test_backward_kernel_proj_match(self, small_config):
        """Test Triton kernel_proj gradients match PyTorch."""
        device = "cuda"
        
        model_triton = TelephoneAttentionND(**small_config, use_triton=True).to(device)
        model_pytorch = TelephoneAttentionND(**small_config, use_triton=False).to(device)
        model_pytorch.load_state_dict(model_triton.state_dict())
        
        B, L, C = 2, 256, small_config['channels']
        torch.manual_seed(42)
        x = torch.randn(B, L, C, device=device)
        
        x_triton = x.clone().requires_grad_(True)
        x_pytorch = x.clone().requires_grad_(True)
        
        out_triton, _ = model_triton(x_triton)
        out_triton.sum().backward()
        
        out_pytorch, _ = model_pytorch(x_pytorch)
        out_pytorch.sum().backward()
        
        w_grad_tr = model_triton.kernel_proj.weight.grad
        w_grad_pt = model_pytorch.kernel_proj.weight.grad
        rel_diff = (w_grad_tr - w_grad_pt).abs() / (w_grad_pt.abs() + 1e-8)
        assert rel_diff.mean().item() < 0.1, f"kernel_proj.weight grad mismatch: mean rel diff = {rel_diff.mean().item()}"
    
    def test_backward_exponent_proj_match(self, small_config):
        """Test Triton exponent_proj gradients match PyTorch."""
        device = "cuda"
        
        model_triton = TelephoneAttentionND(**small_config, use_triton=True).to(device)
        model_pytorch = TelephoneAttentionND(**small_config, use_triton=False).to(device)
        model_pytorch.load_state_dict(model_triton.state_dict())
        
        B, L, C = 2, 256, small_config['channels']
        torch.manual_seed(42)
        x = torch.randn(B, L, C, device=device)
        
        x_triton = x.clone().requires_grad_(True)
        x_pytorch = x.clone().requires_grad_(True)
        
        out_triton, _ = model_triton(x_triton)
        out_triton.sum().backward()
        
        out_pytorch, _ = model_pytorch(x_pytorch)
        out_pytorch.sum().backward()
        
        w_grad_tr = model_triton.exponent_proj.weight.grad
        w_grad_pt = model_pytorch.exponent_proj.weight.grad
        rel_diff = (w_grad_tr - w_grad_pt).abs() / (w_grad_pt.abs() + 1e-8)
        assert rel_diff.mean().item() < 0.2, f"exponent_proj.weight grad mismatch: mean rel diff = {rel_diff.mean().item()}"
    
    def test_backward_out_proj_match(self, small_config):
        """Test Triton out_proj gradients match PyTorch."""
        device = "cuda"
        
        model_triton = TelephoneAttentionND(**small_config, use_triton=True).to(device)
        model_pytorch = TelephoneAttentionND(**small_config, use_triton=False).to(device)
        model_pytorch.load_state_dict(model_triton.state_dict())
        
        B, L, C = 2, 256, small_config['channels']
        torch.manual_seed(42)
        x = torch.randn(B, L, C, device=device)
        
        x_triton = x.clone().requires_grad_(True)
        x_pytorch = x.clone().requires_grad_(True)
        
        out_triton, _ = model_triton(x_triton)
        out_triton.sum().backward()
        
        out_pytorch, _ = model_pytorch(x_pytorch)
        out_pytorch.sum().backward()
        
        w_grad_tr = model_triton.out_proj.weight.grad
        w_grad_pt = model_pytorch.out_proj.weight.grad
        rel_diff = (w_grad_tr - w_grad_pt).abs() / (w_grad_pt.abs() + 1e-8)
        assert rel_diff.mean().item() < 0.1, f"out_proj.weight grad mismatch: mean rel diff = {rel_diff.mean().item()}"
    
    def test_backward_gamma_match(self, small_config):
        """Test Triton RMSNorm gamma gradients match PyTorch."""
        device = "cuda"
        
        model_triton = TelephoneAttentionND(**small_config, use_triton=True).to(device)
        model_pytorch = TelephoneAttentionND(**small_config, use_triton=False).to(device)
        model_pytorch.load_state_dict(model_triton.state_dict())
        
        B, L, C = 2, 256, small_config['channels']
        torch.manual_seed(42)
        x = torch.randn(B, L, C, device=device)
        
        x_triton = x.clone().requires_grad_(True)
        x_pytorch = x.clone().requires_grad_(True)
        
        out_triton, _ = model_triton(x_triton)
        out_triton.sum().backward()
        
        out_pytorch, _ = model_pytorch(x_pytorch)
        out_pytorch.sum().backward()
        
        g_tr = model_triton.wave_gamma.grad.abs().sum().item()
        g_pt = model_pytorch.wave_gamma.grad.abs().sum().item()
        assert abs(g_tr - g_pt) / (g_pt + 1e-8) < 0.3, f"wave_gamma grad mismatch: tr={g_tr}, pt={g_pt}"
        
        g_tr = model_triton.kernel_gamma.grad.abs().sum().item()
        g_pt = model_pytorch.kernel_gamma.grad.abs().sum().item()
        assert abs(g_tr - g_pt) / (g_pt + 1e-8) < 0.2, f"kernel_gamma grad mismatch: tr={g_tr}, pt={g_pt}"
        
        g_tr = model_triton.exponent_gamma.grad.abs().sum().item()
        g_pt = model_pytorch.exponent_gamma.grad.abs().sum().item()
        assert abs(g_tr - g_pt) / (g_pt + 1e-8) < 0.3, f"exponent_gamma grad mismatch: tr={g_tr}, pt={g_pt}"
    
    def test_various_seq_lengths(self, small_config):
        """Test Triton vs PyTorch match at various sequence lengths."""
        device = "cuda"
        
        model_triton = TelephoneAttentionND(**small_config, use_triton=True).to(device)
        model_pytorch = TelephoneAttentionND(**small_config, use_triton=False).to(device)
        model_pytorch.load_state_dict(model_triton.state_dict())
        
        B, C = 2, small_config['channels']
        
        for L in [64, 128, 256, 512, 1024]:
            torch.manual_seed(42 + L)
            x = torch.randn(B, L, C, device=device)
            
            with torch.no_grad():
                out_triton, _ = model_triton(x)
                out_pytorch, _ = model_pytorch(x)
            
            max_diff = (out_triton - out_pytorch).abs().max().item()
            assert max_diff < 1e-3, f"Forward mismatch at L={L}: max diff = {max_diff}"
    
    def test_various_chunk_sizes(self, small_config):
        """Test Triton consistency across different chunk sizes."""
        device = "cuda"
        
        B, L, C = 2, 512, small_config['channels']
        torch.manual_seed(42)
        x = torch.randn(B, L, C, device=device)
        
        chunk_sizes = [64, 128, 256, 512]
        outputs = []
        
        for cs in chunk_sizes:
            config = small_config.copy()
            config['chunk_size'] = cs
            model = TelephoneAttentionND(**config, use_triton=True).to(device)
            
            if outputs:
                model.load_state_dict(outputs[0][1])
            
            with torch.no_grad():
                out, _ = model(x)
            outputs.append((out, model.state_dict()))
        
        base_out = outputs[0][0]
        for i, (out, _) in enumerate(outputs[1:], 1):
            max_diff = (out - base_out).abs().max().item()
            assert max_diff < 5e-3, f"Chunk size {chunk_sizes[i]} vs {chunk_sizes[0]}: max diff = {max_diff}"


# ============================================================================
# Standalone Triton Module Tests (CUDA only)
# ============================================================================

@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton/CUDA not available")  
class TestTritonTelephoneAttention:
    """Tests for standalone TritonTelephoneAttention class."""
    
    def test_forward_shape(self, small_config):
        """Test standalone Triton module forward shape."""
        from heuristic_secrets.models.triton_telephone import TritonTelephoneAttention
        
        device = "cuda"
        model = TritonTelephoneAttention(
            channels=small_config['channels'],
            num_heads=small_config['num_heads'],
            max_samples=small_config['max_samples'],
            max_freq=small_config['max_freq'],
            min_freq=small_config['min_freq'],
            max_kernel_size=small_config['max_kernel_size'],
            chunk_size=small_config['chunk_size'],
        ).to(device)
        
        B, L, C = 2, 256, small_config['channels']
        x = torch.randn(B, L, C, device=device)
        
        out, _ = model(x)
        
        assert out.shape == x.shape
    
    def test_backward_works(self, small_config):
        """Test standalone Triton module backward pass."""
        from heuristic_secrets.models.triton_telephone import TritonTelephoneAttention
        
        device = "cuda"
        model = TritonTelephoneAttention(
            channels=small_config['channels'],
            num_heads=small_config['num_heads'],
            max_samples=small_config['max_samples'],
            max_freq=small_config['max_freq'],
            min_freq=small_config['min_freq'],
            max_kernel_size=small_config['max_kernel_size'],
            chunk_size=small_config['chunk_size'],
        ).to(device)
        
        B, L, C = 2, 256, small_config['channels']
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        
        out, _ = model(x)
        out.sum().backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        assert model.wave_w.grad is not None
        assert model.kernel_w.grad is not None
        assert model.exponent_w.grad is not None
        assert model.out_w.grad is not None


# ============================================================================
# Benchmarks
# ============================================================================

def run_benchmark(model, x, warmup_iters=3, bench_iters=10):
    """Run forward and forward+backward benchmark."""
    device = x.device
    
    for _ in range(warmup_iters):
        out = model(x)[0] if hasattr(model, '__call__') else model(x)
        if x.requires_grad:
            out.sum().backward()
            model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    
    start = time.perf_counter()
    for _ in range(bench_iters):
        with torch.no_grad():
            out = model(x)[0] if hasattr(model, '__call__') else model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - start) / bench_iters * 1000
    fwd_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    x_grad = x.detach().clone().requires_grad_(True)
    start = time.perf_counter()
    for _ in range(bench_iters):
        out = model(x_grad)[0] if hasattr(model, '__call__') else model(x_grad)
        out.sum().backward()
        model.zero_grad()
        x_grad.grad.zero_()
        if device.type == "cuda":
            torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) / bench_iters * 1000
    bwd_mem = torch.cuda.max_memory_allocated() / 1024**2 if device.type == "cuda" else 0
    
    return {
        'fwd_ms': fwd_time,
        'bwd_ms': total_time - fwd_time,
        'fwd_mem_mb': fwd_mem,
        'bwd_mem_mb': bwd_mem,
    }


def benchmark_suite():
    """Run full benchmark suite (CUDA only)."""
    if not TRITON_AVAILABLE:
        print("Triton/CUDA not available, skipping benchmarks")
        return
    
    device = "cuda"
    B, C = 4, 256
    num_heads = 8
    max_samples = 32
    
    print("=" * 80)
    print("TelephoneAttention Benchmark Suite")
    print("=" * 80)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: B={B}, C={C}, H={num_heads}, max_samples={max_samples}")
    print()
    
    model_triton = TelephoneAttentionND(
        channels=C, ndim=1, max_samples=max_samples, num_heads=num_heads,
        checkpoint=False, use_triton=True
    ).to(device)
    model_triton.train()
    
    model_pytorch = TelephoneAttentionND(
        channels=C, ndim=1, max_samples=max_samples, num_heads=num_heads,
        checkpoint=False, use_triton=False
    ).to(device)
    model_pytorch.load_state_dict(model_triton.state_dict())
    model_pytorch.train()
    
    print("=" * 80)
    print("SEQUENCE LENGTH SWEEP")
    print("=" * 80)
    print(f"{'SeqLen':>8} | {'Triton Fwd':>10} {'Triton Bwd':>10} {'Triton Mem':>10} | {'PyTorch Fwd':>11} {'PyTorch Bwd':>11} {'PyTorch Mem':>11} | {'Speedup':>7}")
    print("-" * 100)
    
    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    
    for L in seq_lengths:
        torch.cuda.empty_cache()
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        
        try:
            tr = run_benchmark(model_triton, x)
            pt = run_benchmark(model_pytorch, x)
            
            speedup = (pt['fwd_ms'] + pt['bwd_ms']) / (tr['fwd_ms'] + tr['bwd_ms'])
            
            print(f"{L:>8} | {tr['fwd_ms']:>10.1f} {tr['bwd_ms']:>10.1f} {tr['bwd_mem_mb']:>10.0f} | {pt['fwd_ms']:>11.1f} {pt['bwd_ms']:>11.1f} {pt['bwd_mem_mb']:>11.0f} | {speedup:>7.2f}x")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{L:>8} | {'OOM':>10} {'OOM':>10} {'OOM':>10} | {'OOM':>11} {'OOM':>11} {'OOM':>11} | {'N/A':>7}")
                torch.cuda.empty_cache()
            else:
                raise
    
    print()
    print("=" * 80)
    print("CHUNK SIZE SWEEP (L=4096)")
    print("=" * 80)
    print(f"{'ChunkSize':>10} | {'Fwd (ms)':>10} {'Bwd (ms)':>10} {'Fwd Mem':>10} {'Bwd Mem':>10}")
    print("-" * 60)
    
    L = 4096
    chunk_sizes = [128, 256, 512, 1024, 2048, 4096]
    
    for cs in chunk_sizes:
        torch.cuda.empty_cache()
        model_cs = TelephoneAttentionND(
            channels=C, ndim=1, max_samples=max_samples, num_heads=num_heads,
            chunk_size=cs, checkpoint=False, use_triton=True
        ).to(device)
        model_cs.load_state_dict(model_triton.state_dict())
        model_cs.train()
        
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        
        try:
            result = run_benchmark(model_cs, x)
            print(f"{cs:>10} | {result['fwd_ms']:>10.1f} {result['bwd_ms']:>10.1f} {result['fwd_mem_mb']:>10.0f} {result['bwd_mem_mb']:>10.0f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{cs:>10} | {'OOM':>10} {'OOM':>10} {'OOM':>10} {'OOM':>10}")
                torch.cuda.empty_cache()
            else:
                raise
        
        del model_cs
    
    print("=" * 80)


# ============================================================================
# Main entry point for running benchmarks directly
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if "--benchmark" in sys.argv:
        benchmark_suite()
    else:
        print("Run with --benchmark for benchmarks, or use pytest for tests")
        print()
        print("Examples:")
        print("  pytest tests/models/test_triton_telephone.py -v")
        print("  python tests/models/test_triton_telephone.py --benchmark")
