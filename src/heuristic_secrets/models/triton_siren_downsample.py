"""Triton kernels for SIRENDownsampleND - reuses triton_telephone linear kernels."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from .triton_telephone import (
        triton_linear,
        linear_bwd_dx_kernel,
        linear_bwd_dw_kernel,
    )
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def sin_fwd_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, tl.sin(x), mask=mask)

    @triton.jit
    def sin_bwd_kernel(x_ptr, d_out_ptr, d_x_ptr, N, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + offs, mask=mask)
        d_out = tl.load(d_out_ptr + offs, mask=mask)
        tl.store(d_x_ptr + offs, d_out * tl.cos(x), mask=mask)

    @triton.jit
    def depthwise_conv1d_fwd(
        x_ptr, kernel_ptr, out_ptr,
        B: tl.constexpr, L_in: tl.constexpr, C: tl.constexpr,
        L_out: tl.constexpr, K: tl.constexpr, stride: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_c_block = tl.program_id(2)
        
        c_idx = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        in_start = pid_l * stride
        
        for k in range(K):
            in_pos = in_start + k
            in_valid = in_pos < L_in
            x_offs = pid_b * C * L_in + c_idx * L_in + in_pos
            x_vals = tl.load(x_ptr + x_offs, mask=c_mask & in_valid, other=0.0)
            k_offs = c_idx * K + k
            k_vals = tl.load(kernel_ptr + k_offs, mask=c_mask, other=0.0)
            acc += x_vals * k_vals
        
        out_offs = pid_b * C * L_out + c_idx * L_out + pid_l
        tl.store(out_ptr + out_offs, acc, mask=c_mask)

    @triton.jit
    def depthwise_conv1d_bwd_x(
        d_out_ptr, kernel_ptr, d_x_ptr,
        B: tl.constexpr, L_in: tl.constexpr, C: tl.constexpr,
        L_out: tl.constexpr, K: tl.constexpr, stride: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_c_block = tl.program_id(2)
        
        c_idx = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        
        d_out_offs = pid_b * C * L_out + c_idx * L_out + pid_l
        d_out_vals = tl.load(d_out_ptr + d_out_offs, mask=c_mask, other=0.0)
        
        in_start = pid_l * stride
        for k in range(K):
            in_pos = in_start + k
            if in_pos < L_in:
                k_offs = c_idx * K + k
                k_vals = tl.load(kernel_ptr + k_offs, mask=c_mask, other=0.0)
                d_x_vals = d_out_vals * k_vals
                d_x_offs = pid_b * C * L_in + c_idx * L_in + in_pos
                tl.atomic_add(d_x_ptr + d_x_offs, d_x_vals, mask=c_mask)

    @triton.jit
    def depthwise_conv1d_bwd_kernel(
        x_ptr, d_out_ptr, d_kernel_ptr,
        B: tl.constexpr, L_in: tl.constexpr, C: tl.constexpr,
        L_out: tl.constexpr, K: tl.constexpr, stride: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_c_block = tl.program_id(2)
        
        c_idx = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        
        d_out_offs = pid_b * C * L_out + c_idx * L_out + pid_l
        d_out_vals = tl.load(d_out_ptr + d_out_offs, mask=c_mask, other=0.0)
        
        in_start = pid_l * stride
        for k in range(K):
            in_pos = in_start + k
            if in_pos < L_in:
                x_offs = pid_b * C * L_in + c_idx * L_in + in_pos
                x_vals = tl.load(x_ptr + x_offs, mask=c_mask, other=0.0)
                d_k_vals = d_out_vals * x_vals
                d_k_offs = c_idx * K + k
                tl.atomic_add(d_kernel_ptr + d_k_offs, d_k_vals, mask=c_mask)


def triton_sin(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    sin_fwd_kernel[grid](x, out, N, BLOCK)
    return out


def triton_sin_bwd(x: torch.Tensor, d_out: torch.Tensor) -> torch.Tensor:
    d_x = torch.empty_like(x)
    N = x.numel()
    BLOCK = 1024
    grid = (triton.cdiv(N, BLOCK),)
    sin_bwd_kernel[grid](x, d_out, d_x, N, BLOCK)
    return d_x


def triton_linear_bwd(d_out: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
    M, N = d_out.shape
    K = w.shape[1]
    
    d_x = torch.empty(M, K, device=d_out.device, dtype=d_out.dtype)
    d_w = torch.empty_like(w)
    
    BLOCK_M, BLOCK_K, BLOCK_N = 32, 32, 32
    
    grid_dx = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    linear_bwd_dx_kernel[grid_dx](
        d_out, w, d_x,
        M, N, K,
        d_out.stride(0), d_out.stride(1),
        w.stride(0), w.stride(1),
        d_x.stride(0), d_x.stride(1),
        BLOCK_M, BLOCK_K, BLOCK_N,
    )
    
    grid_dw = (triton.cdiv(N, BLOCK_N), triton.cdiv(K, BLOCK_K))
    linear_bwd_dw_kernel[grid_dw](
        d_out, x, d_w,
        M, N, K,
        d_out.stride(0), d_out.stride(1),
        x.stride(0), x.stride(1),
        d_w.stride(0), d_w.stride(1),
        BLOCK_N, BLOCK_K, BLOCK_M,
    )
    
    d_b = d_out.sum(0)
    
    return d_x, d_w, d_b


def triton_depthwise_conv1d(x: torch.Tensor, kernel: torch.Tensor, stride: int) -> torch.Tensor:
    B, C, L_in = x.shape
    K = kernel.shape[1]
    L_out = (L_in - K) // stride + 1
    
    out = torch.empty(B, C, L_out, device=x.device, dtype=x.dtype)
    BLOCK_C = min(64, triton.next_power_of_2(C))
    grid = (B, L_out, triton.cdiv(C, BLOCK_C))
    
    depthwise_conv1d_fwd[grid](x, kernel, out, B, L_in, C, L_out, K, stride, BLOCK_C)
    return out


def triton_depthwise_conv1d_bwd(x: torch.Tensor, kernel: torch.Tensor, d_out: torch.Tensor, stride: int):
    B, C, L_in = x.shape
    K = kernel.shape[1]
    L_out = d_out.shape[2]
    
    d_x = torch.zeros_like(x)
    d_kernel = torch.zeros_like(kernel)
    BLOCK_C = min(64, triton.next_power_of_2(C))
    
    grid = (B, L_out, triton.cdiv(C, BLOCK_C))
    depthwise_conv1d_bwd_x[grid](d_out, kernel, d_x, B, L_in, C, L_out, K, stride, BLOCK_C)
    depthwise_conv1d_bwd_kernel[grid](x, d_out, d_kernel, B, L_in, C, L_out, K, stride, BLOCK_C)
    
    return d_x, d_kernel


class _TritonSIRENDownsample1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, b0, w1, b1, w2, b2, w3, b3, omega_0, target_len):
        B, L_in, C = x.shape
        L_out = target_len
        
        if L_in == L_out:
            ctx.is_identity = True
            return x.clone()
        
        stride = max(1, L_in // L_out)
        K = stride
        
        pos = torch.linspace(-1, 1, K, device=x.device, dtype=x.dtype).unsqueeze(-1) * omega_0
        
        h0_pre = triton_linear(pos, w0) + b0
        h0 = triton_sin(h0_pre)
        h1_pre = triton_linear(h0, w1) + b1
        h1 = triton_sin(h1_pre)
        h2_pre = triton_linear(h1, w2) + b2
        h2 = triton_sin(h2_pre)
        kernel_flat = triton_linear(h2, w3) + b3
        
        kernel = kernel_flat.T.contiguous()
        x_t = x.movedim(-1, 1).contiguous()
        out_t = triton_depthwise_conv1d(x_t, kernel, stride)
        
        L_actual = out_t.shape[2]
        if L_actual != L_out:
            out_t = F.interpolate(out_t, size=L_out, mode='linear', align_corners=False)
        
        out = out_t.movedim(1, -1)
        
        ctx.save_for_backward(x_t, pos, kernel, h0_pre, h0, h1_pre, h1, h2_pre, h2, w0, w1, w2, w3)
        ctx.omega_0 = omega_0
        ctx.stride = stride
        ctx.K = K
        ctx.L_out = L_out
        ctx.L_actual = L_actual
        ctx.is_identity = False
        
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        if ctx.is_identity:
            return d_out, None, None, None, None, None, None, None, None, None, None
        
        x_t, pos, kernel, h0_pre, h0, h1_pre, h1, h2_pre, h2, w0, w1, w2, w3 = ctx.saved_tensors
        stride = ctx.stride
        L_out = ctx.L_out
        L_actual = ctx.L_actual
        
        d_out_t = d_out.movedim(-1, 1).contiguous()
        if L_actual != L_out:
            d_out_t = F.interpolate(d_out_t, size=L_actual, mode='linear', align_corners=False)
        
        d_x_t, d_kernel = triton_depthwise_conv1d_bwd(x_t, kernel, d_out_t, stride)
        d_kernel_flat = d_kernel.T.contiguous()
        
        d_h2, d_w3, d_b3 = triton_linear_bwd(d_kernel_flat, h2, w3)
        d_h2_pre = triton_sin_bwd(h2_pre, d_h2)
        d_h1, d_w2, d_b2 = triton_linear_bwd(d_h2_pre, h1, w2)
        d_h1_pre = triton_sin_bwd(h1_pre, d_h1)
        d_h0, d_w1, d_b1 = triton_linear_bwd(d_h1_pre, h0, w1)
        d_h0_pre = triton_sin_bwd(h0_pre, d_h0)
        _, d_w0, d_b0 = triton_linear_bwd(d_h0_pre, pos, w0)
        
        d_x = d_x_t.movedim(1, -1)
        
        return d_x, d_w0, d_b0, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3, None, None


class TritonSIRENDownsample1D(nn.Module):
    def __init__(self, channels: int, hidden: int = 32, omega_0: float = 30.0):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.omega_0 = omega_0
        
        self.w0 = nn.Parameter(torch.empty(hidden, 1))
        self.b0 = nn.Parameter(torch.zeros(hidden))
        self.w1 = nn.Parameter(torch.empty(hidden, hidden))
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.w2 = nn.Parameter(torch.empty(hidden, hidden))
        self.b2 = nn.Parameter(torch.zeros(hidden))
        self.w3 = nn.Parameter(torch.empty(channels, hidden))
        self.b3 = nn.Parameter(torch.zeros(channels))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.uniform_(self.w0, -1.0, 1.0)
        nn.init.uniform_(self.w1, -1.0 / self.hidden, 1.0 / self.hidden)
        nn.init.uniform_(self.w2, -1.0 / self.hidden, 1.0 / self.hidden)
        nn.init.uniform_(self.w3, -1.0 / self.hidden, 1.0 / self.hidden)
    
    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        if not HAS_TRITON or not x.is_cuda:
            return self._forward_pytorch(x, target_len)
        return _TritonSIRENDownsample1D.apply(
            x, self.w0, self.b0, self.w1, self.b1, self.w2, self.b2, self.w3, self.b3,
            self.omega_0, target_len
        )
    
    def _forward_pytorch(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        B, L_in, C = x.shape
        if L_in == target_len:
            return x
        
        stride = max(1, L_in // target_len)
        K = stride
        
        pos = torch.linspace(-1, 1, K, device=x.device, dtype=x.dtype).unsqueeze(-1) * self.omega_0
        h = torch.sin(F.linear(pos, self.w0, self.b0))
        h = torch.sin(F.linear(h, self.w1, self.b1))
        h = torch.sin(F.linear(h, self.w2, self.b2))
        kernel_flat = F.linear(h, self.w3, self.b3)
        
        kernel = kernel_flat.T.reshape(C, 1, K)
        x_t = x.movedim(-1, 1)
        out_t = F.conv1d(x_t, kernel, stride=stride, groups=C)
        
        if out_t.shape[2] != target_len:
            out_t = F.interpolate(out_t, size=target_len, mode='linear', align_corners=False)
        
        return out_t.movedim(1, -1)


if __name__ == "__main__":
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"HAS_TRITON: {HAS_TRITON}")
    print()
    
    B, L_in, C = 4, 1024, 256
    L_out = 256
    hidden = 32
    
    print(f"Config: B={B}, L_in={L_in}, L_out={L_out}, C={C}, hidden={hidden}")
    print()
    
    from heuristic_secrets.models.scatter_attention import SIRENDownsampleND
    
    pytorch_model = SIRENDownsampleND(channels=C, ndim=1, hidden=hidden).to(device)
    triton_model = TritonSIRENDownsample1D(channels=C, hidden=hidden).to(device)
    
    with torch.no_grad():
        triton_model.w0.copy_(pytorch_model.kernel_net.net[0].weight)
        triton_model.b0.copy_(pytorch_model.kernel_net.net[0].bias)
        triton_model.w1.copy_(pytorch_model.kernel_net.net[2].weight)
        triton_model.b1.copy_(pytorch_model.kernel_net.net[2].bias)
        triton_model.w2.copy_(pytorch_model.kernel_net.net[4].weight)
        triton_model.b2.copy_(pytorch_model.kernel_net.net[4].bias)
        triton_model.w3.copy_(pytorch_model.kernel_net.net[6].weight)
        triton_model.b3.copy_(pytorch_model.kernel_net.net[6].bias)
    
    print("=" * 100)
    print("SANITY CHECKS")
    print("=" * 100)
    
    torch.manual_seed(42)
    x = torch.randn(B, L_in, C, device=device)
    
    out_pytorch = pytorch_model(x, (L_out,))
    out_triton = triton_model(x, L_out)
    
    fwd_diff = (out_pytorch - out_triton).abs().max().item()
    print(f"Forward match: {'PASS' if fwd_diff < 1e-3 else 'FAIL'} (max diff: {fwd_diff:.2e})")
    
    x_pt = x.clone().requires_grad_(True)
    x_tr = x.clone().requires_grad_(True)
    
    out_pt = pytorch_model(x_pt, (L_out,))
    out_pt.sum().backward()
    
    out_tr = triton_model(x_tr, L_out)
    out_tr.sum().backward()
    
    dx_diff = (x_pt.grad - x_tr.grad).abs().max().item()
    print(f"d_x match: {'PASS' if dx_diff < 5e-3 else 'FAIL'} (max diff: {dx_diff:.2e})")
    
    w0_grad_pt = pytorch_model.kernel_net.net[0].weight.grad.abs().sum().item()
    w0_grad_tr = triton_model.w0.grad.abs().sum().item()
    print(f"w0 grad: pt={w0_grad_pt:.2e}, tr={w0_grad_tr:.2e}, match={'PASS' if abs(w0_grad_pt - w0_grad_tr) / (w0_grad_pt + 1e-8) < 0.2 else 'FAIL'}")
    
    w3_grad_pt = pytorch_model.kernel_net.net[6].weight.grad.abs().sum().item()
    w3_grad_tr = triton_model.w3.grad.abs().sum().item()
    print(f"w3 grad: pt={w3_grad_pt:.2e}, tr={w3_grad_tr:.2e}, match={'PASS' if abs(w3_grad_pt - w3_grad_tr) / (w3_grad_pt + 1e-8) < 0.2 else 'FAIL'}")
    
    print()
    print("=" * 100)
    print("BENCHMARKS")
    print("=" * 100)
    
    warmup, iters = 10, 50
    
    def bench(model, x, target, is_triton):
        for _ in range(warmup):
            out = model(x, target) if is_triton else model(x, (target,))
        
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        for _ in range(iters):
            out = model(x, target) if is_triton else model(x, (target,))
            if device == "cuda":
                torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - start) / iters * 1000
        fwd_mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        
        x_grad = x.detach().clone().requires_grad_(True)
        for _ in range(warmup):
            out = model(x_grad, target) if is_triton else model(x_grad, (target,))
            out.sum().backward()
            model.zero_grad()
            x_grad.grad = None
        
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        for _ in range(iters):
            out = model(x_grad, target) if is_triton else model(x_grad, (target,))
            out.sum().backward()
            model.zero_grad()
            x_grad.grad = None
            if device == "cuda":
                torch.cuda.synchronize()
        total_ms = (time.perf_counter() - start) / iters * 1000
        bwd_ms = total_ms - fwd_ms
        bwd_mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        
        return fwd_ms, bwd_ms, fwd_mem, bwd_mem
    
    print(f"{'L_in':>8} {'L_out':>8} | {'PyT Fwd':>8} {'PyT Bwd':>8} {'PyT Mem':>8} | {'Tri Fwd':>8} {'Tri Bwd':>8} {'Tri Mem':>8} | {'Speedup':>8}")
    print("-" * 100)
    
    configs = [(512, 128), (1024, 256), (2048, 512), (4096, 1024), (8192, 2048)]
    
    for L_in, L_out in configs:
        x = torch.randn(B, L_in, C, device=device)
        
        try:
            pt_fwd, pt_bwd, _, pt_mem = bench(pytorch_model, x, L_out, is_triton=False)
            tr_fwd, tr_bwd, _, tr_mem = bench(triton_model, x, L_out, is_triton=True)
            
            speedup = (pt_fwd + pt_bwd) / (tr_fwd + tr_bwd)
            print(f"{L_in:>8} {L_out:>8} | {pt_fwd:>8.2f} {pt_bwd:>8.2f} {pt_mem:>8.0f} | {tr_fwd:>8.2f} {tr_bwd:>8.2f} {tr_mem:>8.0f} | {speedup:>8.2f}x")
        except Exception as e:
            print(f"{L_in:>8} {L_out:>8} | ERROR: {e}")
    
    print("=" * 100)
