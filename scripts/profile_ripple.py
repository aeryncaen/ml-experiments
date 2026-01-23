#!/usr/bin/env python3
"""
Profiling script for RippleAttention with fine-grained component instrumentation.

Usage:
    python scripts/profile_ripple.py [--seq-len 1024] [--batch 4] [--width 256] [--device cuda]
    
Outputs:
    - Console table sorted by total time
    - trace.json for chrome://tracing visualization
"""

from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity
from functools import wraps

import sys
sys.path.insert(0, 'src')

from heuristic_secrets.models import ripple_attention
from heuristic_secrets.models import telephone_attention
from heuristic_secrets.models import adaptive_local_conv
from heuristic_secrets.models import scatter_attention

try:
    from heuristic_secrets.models import triton_adaptive_conv
    HAS_TRITON = triton_adaptive_conv.HAS_TRITON
except ImportError:
    HAS_TRITON = False
    triton_adaptive_conv = None


def wrap_linear(module: nn.Linear, name: str) -> None:
    """Wrap a Linear layer's forward with profiling."""
    original_forward = module.forward
    
    @wraps(original_forward)
    def profiled_forward(x):
        with record_function(name):
            return original_forward(x)
    
    module.forward = profiled_forward


def wrap_method(obj, method_name: str, label: str):
    """Wrap a method with profiling."""
    original = getattr(obj, method_name)
    
    @wraps(original)
    def wrapped(*args, **kwargs):
        with record_function(label):
            return original(*args, **kwargs)
    
    setattr(obj, method_name, wrapped)


def instrument_telephone(tele: telephone_attention.TelephoneAttentionND, prefix: str = "Telephone"):
    """Instrument a TelephoneAttentionND instance."""
    wrap_linear(tele.wave_proj, f"{prefix}::wave_proj")
    wrap_linear(tele.kernel_proj, f"{prefix}::kernel_proj")
    wrap_linear(tele.exponent_proj, f"{prefix}::exponent_proj")
    wrap_linear(tele.out_proj, f"{prefix}::out_proj")
    
    # Wrap the chunk method
    original_chunk = tele._forward_chunk
    
    @wraps(original_chunk)
    def profiled_chunk(x_flat, x_chunk, chunk_start, chunk_end, spatial, L):
        with record_function(f"{prefix}::_forward_chunk"):
            return original_chunk(x_flat, x_chunk, chunk_start, chunk_end, spatial, L)
    
    tele._forward_chunk = profiled_chunk
    
    # Wrap the main forward
    original_forward = tele.forward
    
    @wraps(original_forward)
    def profiled_forward(x):
        with record_function(f"{prefix}::forward"):
            return original_forward(x)
    
    tele.forward = profiled_forward


def instrument_adaptive_conv(conv: adaptive_local_conv.AdaptiveLocalConv, prefix: str = "AdaptiveConv"):
    """Instrument an AdaptiveLocalConv instance."""
    wrap_linear(conv.window_proj, f"{prefix}::window_proj")
    wrap_linear(conv.offset_proj, f"{prefix}::offset_proj")
    wrap_linear(conv.kernel_proj, f"{prefix}::kernel_proj")
    wrap_linear(conv.v_proj, f"{prefix}::v_proj")
    wrap_linear(conv.out_proj, f"{prefix}::out_proj")
    
    # Wrap SE block
    original_se = conv.se.forward
    
    @wraps(original_se)
    def profiled_se(x):
        with record_function(f"{prefix}::squeeze_excite"):
            return original_se(x)
    
    conv.se.forward = profiled_se
    
    # Wrap main forward with internal profiling
    original_forward = conv.forward
    
    @wraps(original_forward)
    def profiled_forward(x):
        with record_function(f"{prefix}::forward"):
            return original_forward(x)
    
    conv.forward = profiled_forward


def instrument_triton_conv(conv, prefix: str = "TritonConv"):
    """Instrument a TritonAdaptiveLocalConv instance."""
    if conv is None:
        return
    
    wrap_linear(conv.window_proj, f"{prefix}::window_proj")
    wrap_linear(conv.offset_proj, f"{prefix}::offset_proj")
    wrap_linear(conv.kernel_proj, f"{prefix}::kernel_proj")
    wrap_linear(conv.v_proj, f"{prefix}::v_proj")
    wrap_linear(conv.out_proj, f"{prefix}::out_proj")
    
    # Wrap SE block
    original_se = conv.se.forward
    
    @wraps(original_se)
    def profiled_se(x):
        with record_function(f"{prefix}::squeeze_excite"):
            return original_se(x)
    
    conv.se.forward = profiled_se
    
    original_forward = conv.forward
    
    @wraps(original_forward)
    def profiled_forward(x):
        with record_function(f"{prefix}::forward"):
            return original_forward(x)
    
    conv.forward = profiled_forward


def instrument_lowrank(lowrank: scatter_attention.LowRankAttention, prefix: str = "LowRank"):
    """Instrument a LowRankAttention instance."""
    wrap_linear(lowrank.q_proj, f"{prefix}::q_proj")
    wrap_linear(lowrank.k_proj, f"{prefix}::k_proj")
    wrap_linear(lowrank.v_proj, f"{prefix}::v_proj")
    wrap_linear(lowrank.out_proj, f"{prefix}::out_proj")
    
    # Wrap downsample
    original_downsample = lowrank.downsample.forward
    
    @wraps(original_downsample)
    def profiled_downsample(x, target_shape):
        with record_function(f"{prefix}::downsample"):
            return original_downsample(x, target_shape)
    
    lowrank.downsample.forward = profiled_downsample
    
    # Wrap Q/K norms
    original_q_norm = lowrank.q_norm.forward
    original_k_norm = lowrank.k_norm.forward
    
    @wraps(original_q_norm)
    def profiled_q_norm(x):
        with record_function(f"{prefix}::q_norm"):
            return original_q_norm(x)
    
    @wraps(original_k_norm)
    def profiled_k_norm(x):
        with record_function(f"{prefix}::k_norm"):
            return original_k_norm(x)
    
    lowrank.q_norm.forward = profiled_q_norm
    lowrank.k_norm.forward = profiled_k_norm
    
    # Wrap head norm
    original_head_norm = lowrank.head_norm.forward
    
    @wraps(original_head_norm)
    def profiled_head_norm(x):
        with record_function(f"{prefix}::head_norm"):
            return original_head_norm(x)
    
    lowrank.head_norm.forward = profiled_head_norm
    
    # Wrap main forward with detailed internal profiling
    original_forward = lowrank.forward
    
    def profiled_forward(x):
        B, L, C = x.shape
        H = lowrank.num_heads
        d = lowrank.head_dim
        r = max(1, int(L ** lowrank.reduction_power))
        
        with record_function(f"{prefix}::forward"):
            # Downsample (already wrapped)
            x_down = lowrank.downsample(x, (r,))
            seq_len = x_down.shape[1]
            
            # QKV projections (already wrapped)
            with record_function(f"{prefix}::qkv_reshape"):
                q = lowrank.q_proj(x_down).reshape(B, seq_len, H, 2, d)
                k = lowrank.k_proj(x_down).reshape(B, seq_len, H, 2, d)
                v = lowrank.v_proj(x_down).reshape(B, seq_len, H, 2 * d)
            
            with record_function(f"{prefix}::split_heads"):
                q1, q2 = q[..., 0, :], q[..., 1, :]
                k1, k2 = k[..., 0, :], k[..., 1, :]
            
            with record_function(f"{prefix}::norm_rope"):
                q1 = scatter_attention.apply_rope(lowrank.q_norm(q1))
                q2 = scatter_attention.apply_rope(lowrank.q_norm(q2))
                k1 = scatter_attention.apply_rope(lowrank.k_norm(k1))
                k2 = scatter_attention.apply_rope(lowrank.k_norm(k2))
            
            with record_function(f"{prefix}::transpose"):
                q1 = q1.transpose(1, 2)
                q2 = q2.transpose(1, 2)
                k1 = k1.transpose(1, 2)
                k2 = k2.transpose(1, 2)
                v = v.transpose(1, 2)
            
            with record_function(f"{prefix}::lambda_compute"):
                lambda_val = (
                    torch.exp(torch.sum(lowrank.lambda_q1 * lowrank.lambda_k1))
                    - torch.exp(torch.sum(lowrank.lambda_q2 * lowrank.lambda_k2))
                    + lowrank.lambda_init
                )
            
            with record_function(f"{prefix}::attention_scores"):
                attn1 = F.softmax(q1 @ k1.transpose(-2, -1) * lowrank.scale, dim=-1)
                attn2 = F.softmax(q2 @ k2.transpose(-2, -1) * lowrank.scale, dim=-1)
            
            with record_function(f"{prefix}::diff_attn"):
                diff_attn = (attn1 - lambda_val * attn2) @ v
            
            with record_function(f"{prefix}::head_norm_scale"):
                diff_attn = diff_attn.transpose(1, 2)
                diff_attn = lowrank.head_norm(diff_attn) * (1 - lowrank.lambda_init)
            
            with record_function(f"{prefix}::reshape_silu"):
                lowrank_out = diff_attn.reshape(B, seq_len, C)
                lowrank_out = F.silu(lowrank.out_proj(lowrank_out))
            
            with record_function(f"{prefix}::interpolate"):
                full_out = scatter_attention.interpolate_nd(lowrank_out, (L,))
            
            return full_out, lowrank_out
    
    lowrank.forward = profiled_forward


def instrument_ripple(model: ripple_attention.RippleAttention):
    """Instrument a RippleAttention model and all its subcomponents."""
    # Instrument subcomponents
    instrument_telephone(model.telephone, "Telephone")
    
    # Check if using Triton or vanilla adaptive conv
    if hasattr(model.conv, 'chunk_size') and HAS_TRITON:
        # TritonAdaptiveLocalConv
        instrument_triton_conv(model.conv, "TritonConv")
    else:
        # AdaptiveLocalConv
        instrument_adaptive_conv(model.conv, "AdaptiveConv")
    
    instrument_lowrank(model.lowrank, "LowRank")
    
    # Wrap norms
    for name, norm in model.norms.items():
        original_norm = norm.forward
        
        @wraps(original_norm)
        def make_profiled_norm(orig, n):
            def profiled_norm(x):
                with record_function(f"Ripple::norm_{n}"):
                    return orig(x)
            return profiled_norm
        
        norm.forward = make_profiled_norm(original_norm, name)
    
    # Wrap main forward
    original_forward = model.forward
    
    @wraps(original_forward)
    def profiled_forward(x):
        with record_function("RippleAttention::total"):
            h = x
            info = {}
            
            for name in model.order:
                if name == 'tele':
                    with record_function("Ripple::tele_block"):
                        out, _ = model.telephone(h)
                        h = h + model.norms[name](out)
                elif name == 'conv':
                    with record_function("Ripple::conv_block"):
                        out, info = model.conv(h)
                        h = h + model.norms[name](out)
                elif name == 'lowrank':
                    with record_function("Ripple::lowrank_block"):
                        out, _ = model.lowrank(h)
                        h = h + model.norms[name](out)
                else:
                    raise ValueError(f"Unknown layer: {name}")
            
            return h, info
    
    model.forward = profiled_forward


def run_profile(
    seq_len: int = 1024,
    batch_size: int = 4,
    width: int = 256,
    num_heads: int = 8,
    device: str = "cuda",
    warmup: int = 3,
    iterations: int = 5,
    use_triton: bool = True,
    trace_file: str | None = "trace.json",
):
    """Run profiling on RippleAttention."""
    
    # Build model
    model = ripple_attention.RippleAttention(
        channels=width,
        num_heads=num_heads,
        use_triton=use_triton and HAS_TRITON,
        max_seq_len=seq_len * 2,
    ).to(device)
    model.train()
    
    # Apply instrumentation AFTER building
    instrument_ripple(model)
    
    print(f"Model: RippleAttention")
    print(f"  Channels: {width}")
    print(f"  Num heads: {num_heads}")
    print(f"  Using Triton: {use_triton and HAS_TRITON}")
    print(f"  Conv type: {type(model.conv).__name__}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print()
    print(f"Input: B={batch_size}, L={seq_len}, C={width}")
    print(f"Device: {device}")
    print()
    
    # Create input
    x = torch.randn(batch_size, seq_len, width, device=device, requires_grad=True)
    
    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    for _ in range(warmup):
        out, _ = model(x)
        out.sum().backward()
        model.zero_grad()
        if x.grad is not None:
            x.grad = None
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Profile
    activities = [ProfilerActivity.CPU]
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    print(f"Profiling ({iterations} iterations)...")
    print()
    
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(iterations):
            with record_function(f"iteration_{i}"):
                with record_function("forward"):
                    out, _ = model(x)
                with record_function("backward"):
                    out.sum().backward()
                model.zero_grad()
                if x.grad is not None:
                    x.grad = None
                if device == "cuda":
                    torch.cuda.synchronize()
    
    # Print results
    sort_key = "cuda_time_total" if device == "cuda" else "cpu_time_total"
    
    print("=" * 100)
    print("PROFILE RESULTS (sorted by total time)")
    print("=" * 100)
    print()
    print(prof.key_averages().table(sort_by=sort_key, row_limit=50))
    
    # Export trace
    if trace_file:
        prof.export_chrome_trace(trace_file)
        print(f"\nChrome trace exported to: {trace_file}")
        print("Open chrome://tracing in Chrome and load the trace file to visualize.")
    
    # Component summary
    print()
    print("=" * 100)
    print("COMPONENT SUMMARY (per iteration)")
    print("=" * 100)
    print()
    
    def get_time(event, use_cuda: bool) -> float:
        """Get time from event, handling CUDA/CPU differences."""
        if use_cuda:
            # Try cuda_time_total first, fall back to self_cuda_time_total
            if hasattr(event, 'cuda_time_total'):
                return event.cuda_time_total
            elif hasattr(event, 'self_cuda_time_total'):
                return event.self_cuda_time_total
            else:
                return event.cpu_time_total
        return event.cpu_time_total
    
    use_cuda = device == "cuda"
    component_times: dict[str, float] = {}
    for event in prof.key_averages():
        name = event.key
        time_us = get_time(event, use_cuda)
        
        # Group by component prefix
        if "::" in name:
            prefix = name.split("::")[0]
            if prefix not in component_times:
                component_times[prefix] = 0
            component_times[prefix] += time_us
    
    # Also add top-level forward/backward
    for event in prof.key_averages():
        name = event.key
        if name in ("forward", "backward"):
            time_us = get_time(event, use_cuda)
            component_times[name] = time_us
    
    # Sort by time
    sorted_components = sorted(component_times.items(), key=lambda x: -x[1])
    total_time = sum(v for k, v in component_times.items() if k not in ("forward", "backward"))
    
    print(f"{'Component':<30} {'Time (ms)':>12} {'% of Model':>12}")
    print("-" * 56)
    for name, time_us in sorted_components:
        time_ms = time_us / 1000 / iterations
        if name in ("forward", "backward"):
            pct_str = "-"
        else:
            pct = 100 * time_us / total_time if total_time > 0 else 0
            pct_str = f"{pct:.1f}%"
        print(f"{name:<30} {time_ms:>12.3f} {pct_str:>12}")
    
    print()
    print("=" * 100)
    print("DETAILED BREAKDOWN")
    print("=" * 100)
    print()
    
    for component in ["Telephone", "AdaptiveConv", "TritonConv", "LowRank", "Ripple"]:
        ops = []
        for event in prof.key_averages():
            if event.key.startswith(f"{component}::"):
                time_us = get_time(event, use_cuda)
                ops.append((event.key, time_us))
        
        if ops:
            print(f"\n{component}:")
            ops.sort(key=lambda x: -x[1])
            for name, time_us in ops:
                time_ms = time_us / 1000 / iterations
                print(f"  {name.split('::')[1]:<25} {time_ms:>10.3f} ms")
    
    return prof


def main():
    parser = argparse.ArgumentParser(description="Profile RippleAttention components")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--width", type=int, default=256, help="Model width (channels)")
    parser.add_argument("--heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=5, help="Profile iterations")
    parser.add_argument("--no-triton", action="store_true", help="Disable Triton kernels")
    parser.add_argument("--trace", type=str, default="trace.json", help="Chrome trace output file (empty to disable)")
    
    args = parser.parse_args()
    
    run_profile(
        seq_len=args.seq_len,
        batch_size=args.batch,
        width=args.width,
        num_heads=args.heads,
        device=args.device,
        warmup=args.warmup,
        iterations=args.iterations,
        use_triton=not args.no_triton,
        trace_file=args.trace if args.trace else None,
    )


if __name__ == "__main__":
    main()
