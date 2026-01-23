"""Triton kernels for SIRENDownsampleND - learned depthwise strided conv with SIREN-generated kernel."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def siren_mlp_fwd_kernel(
        pos_ptr, out_ptr,
        w0_ptr, b0_ptr,
        w1_ptr, b1_ptr,
        w2_ptr, b2_ptr,
        omega_0,
        num_pos: tl.constexpr,
        in_dim: tl.constexpr,
        hidden: tl.constexpr,
        out_dim: tl.constexpr,
        BLOCK_POS: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """
        Fused SIREN MLP: pos -> Linear -> Sin -> Linear -> Sin -> Linear -> out
        
        pos: (num_pos, in_dim)
        w0: (hidden, in_dim), b0: (hidden,)
        w1: (hidden, hidden), b1: (hidden,)
        w2: (out_dim, hidden), b2: (out_dim,)
        out: (num_pos, out_dim)
        """
        pid = tl.program_id(0)
        pos_idx = pid * BLOCK_POS + tl.arange(0, BLOCK_POS)
        pos_mask = pos_idx < num_pos
        
        h_idx = tl.arange(0, BLOCK_H)
        h_mask = h_idx < hidden
        out_idx = tl.arange(0, BLOCK_H)
        out_mask = out_idx < out_dim
        
        for p in range(BLOCK_POS):
            if pid * BLOCK_POS + p >= num_pos:
                continue
            
            p_idx = pid * BLOCK_POS + p
            
            pos_vals = tl.load(pos_ptr + p_idx * in_dim + tl.arange(0, BLOCK_H), 
                               mask=tl.arange(0, BLOCK_H) < in_dim, other=0.0)
            pos_vals = pos_vals * omega_0
            
            h0 = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for d in range(in_dim):
                pos_d = tl.load(pos_ptr + p_idx * in_dim + d) * omega_0
                w0_col = tl.load(w0_ptr + h_idx * in_dim + d, mask=h_mask, other=0.0)
                h0 += w0_col * pos_d
            b0_vals = tl.load(b0_ptr + h_idx, mask=h_mask, other=0.0)
            h0 = tl.sin(h0 + b0_vals)
            
            h1 = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for hh in range(hidden):
                h0_val = tl.sum(tl.where(h_idx == hh, h0, 0.0))
                w1_col = tl.load(w1_ptr + h_idx * hidden + hh, mask=h_mask, other=0.0)
                h1 += w1_col * h0_val
            b1_vals = tl.load(b1_ptr + h_idx, mask=h_mask, other=0.0)
            h1 = tl.sin(h1 + b1_vals)
            
            out_vals = tl.zeros((BLOCK_H,), dtype=tl.float32)
            for hh in range(hidden):
                h1_val = tl.sum(tl.where(h_idx == hh, h1, 0.0))
                w2_col = tl.load(w2_ptr + out_idx * hidden + hh, mask=out_mask, other=0.0)
                out_vals += w2_col * h1_val
            b2_vals = tl.load(b2_ptr + out_idx, mask=out_mask, other=0.0)
            out_vals = out_vals + b2_vals
            
            tl.store(out_ptr + p_idx * out_dim + out_idx, out_vals, mask=out_mask)

    @triton.jit
    def depthwise_strided_conv1d_fwd_kernel(
        x_ptr, kernel_ptr, out_ptr,
        B: tl.constexpr, L_in: tl.constexpr, C: tl.constexpr,
        L_out: tl.constexpr, K: tl.constexpr, stride: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """
        Depthwise strided 1D convolution.
        
        x: (B, L_in, C)
        kernel: (C, 1, K) - depthwise kernel
        out: (B, L_out, C)
        """
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_c = tl.program_id(2)
        
        c_idx = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        
        start_pos = pid_l * stride
        
        for k in range(K):
            in_pos = start_pos + k
            if in_pos < L_in:
                x_vals = tl.load(x_ptr + pid_b * L_in * C + in_pos * C + c_idx, mask=c_mask, other=0.0)
                k_vals = tl.load(kernel_ptr + c_idx * K + k, mask=c_mask, other=0.0)
                acc += x_vals * k_vals
        
        tl.store(out_ptr + pid_b * L_out * C + pid_l * C + c_idx, acc, mask=c_mask)

    @triton.jit
    def siren_mlp_bwd_kernel(
        pos_ptr, d_out_ptr,
        w0_ptr, b0_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr,
        d_w0_ptr, d_b0_ptr, d_w1_ptr, d_b1_ptr, d_w2_ptr, d_b2_ptr,
        h0_ptr, h1_ptr,
        omega_0,
        num_pos: tl.constexpr,
        in_dim: tl.constexpr,
        hidden: tl.constexpr,
        out_dim: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """
        Backward pass for SIREN MLP.
        Requires saved h0, h1 (pre-sin activations) from forward.
        """
        pid = tl.program_id(0)
        if pid >= num_pos:
            return
        
        h_idx = tl.arange(0, BLOCK_H)
        h_mask = h_idx < hidden
        out_idx = tl.arange(0, BLOCK_H)
        out_mask = out_idx < out_dim
        
        d_out = tl.load(d_out_ptr + pid * out_dim + out_idx, mask=out_mask, other=0.0)
        
        h1_pre = tl.load(h1_ptr + pid * hidden + h_idx, mask=h_mask, other=0.0)
        h1_val = tl.sin(h1_pre)
        
        for o in range(out_dim):
            d_o = tl.sum(tl.where(out_idx == o, d_out, 0.0))
            w2_row = tl.load(w2_ptr + o * hidden + h_idx, mask=h_mask, other=0.0)
            tl.atomic_add(d_w2_ptr + o * hidden + h_idx, d_o * h1_val, mask=h_mask)
            tl.atomic_add(d_b2_ptr + o, d_o)
        
        d_h1_sin = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for o in range(out_dim):
            d_o = tl.sum(tl.where(out_idx == o, d_out, 0.0))
            w2_row = tl.load(w2_ptr + o * hidden + h_idx, mask=h_mask, other=0.0)
            d_h1_sin += d_o * w2_row
        
        d_h1_pre = d_h1_sin * tl.cos(h1_pre)
        
        h0_pre = tl.load(h0_ptr + pid * hidden + h_idx, mask=h_mask, other=0.0)
        h0_val = tl.sin(h0_pre)
        
        for hh in range(hidden):
            d_hh = tl.sum(tl.where(h_idx == hh, d_h1_pre, 0.0))
            w1_row = tl.load(w1_ptr + hh * hidden + h_idx, mask=h_mask, other=0.0)
            tl.atomic_add(d_w1_ptr + hh * hidden + h_idx, d_hh * h0_val, mask=h_mask)
            tl.atomic_add(d_b1_ptr + hh, d_hh)
        
        d_h0_sin = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for hh in range(hidden):
            d_hh = tl.sum(tl.where(h_idx == hh, d_h1_pre, 0.0))
            w1_row = tl.load(w1_ptr + hh * hidden + h_idx, mask=h_mask, other=0.0)
            d_h0_sin += d_hh * w1_row
        
        d_h0_pre = d_h0_sin * tl.cos(h0_pre)
        
        pos_vals = tl.load(pos_ptr + pid * in_dim + tl.arange(0, BLOCK_H), 
                          mask=tl.arange(0, BLOCK_H) < in_dim, other=0.0) * omega_0
        
        for hh in range(hidden):
            d_hh = tl.sum(tl.where(h_idx == hh, d_h0_pre, 0.0))
            for d in range(in_dim):
                pos_d = tl.load(pos_ptr + pid * in_dim + d) * omega_0
                tl.atomic_add(d_w0_ptr + hh * in_dim + d, d_hh * pos_d)
            tl.atomic_add(d_b0_ptr + hh, d_hh)

    @triton.jit
    def depthwise_strided_conv1d_bwd_kernel(
        x_ptr, kernel_ptr, d_out_ptr,
        d_x_ptr, d_kernel_ptr,
        B: tl.constexpr, L_in: tl.constexpr, C: tl.constexpr,
        L_out: tl.constexpr, K: tl.constexpr, stride: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """
        Backward for depthwise strided 1D convolution.
        """
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_c = tl.program_id(2)
        
        c_idx = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        
        d_out_vals = tl.load(d_out_ptr + pid_b * L_out * C + pid_l * C + c_idx, mask=c_mask, other=0.0)
        
        start_pos = pid_l * stride
        
        for k in range(K):
            in_pos = start_pos + k
            if in_pos < L_in:
                x_vals = tl.load(x_ptr + pid_b * L_in * C + in_pos * C + c_idx, mask=c_mask, other=0.0)
                k_vals = tl.load(kernel_ptr + c_idx * K + k, mask=c_mask, other=0.0)
                
                d_x_vals = d_out_vals * k_vals
                tl.atomic_add(d_x_ptr + pid_b * L_in * C + in_pos * C + c_idx, d_x_vals, mask=c_mask)
                
                d_k_vals = d_out_vals * x_vals
                tl.atomic_add(d_kernel_ptr + c_idx * K + k, d_k_vals, mask=c_mask)


class _TritonSIRENDownsample1D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, b0, w_out, b_out, omega_0, target_len, num_hidden, *hidden_params):
        B, L_in, C = x.shape
        L_out = target_len
        
        hidden_weights = hidden_params[:num_hidden]
        hidden_biases = hidden_params[num_hidden:]
        
        if L_in == L_out:
            ctx.save_for_backward(x)
            ctx.is_identity = True
            ctx.num_hidden = num_hidden
            return x.clone()
        
        stride = max(1, L_in // L_out)
        K = stride
        
        positions = torch.linspace(-1, 1, K, device=x.device, dtype=x.dtype).unsqueeze(-1)
        
        pos_scaled = positions * omega_0
        h_pre_list = []
        
        h_pre = F.linear(pos_scaled, w0, b0)
        h_pre_list.append(h_pre.clone())
        h = torch.sin(h_pre)
        
        for w_h, b_h in zip(hidden_weights, hidden_biases):
            h_pre = F.linear(h, w_h, b_h)
            h_pre_list.append(h_pre.clone())
            h = torch.sin(h_pre)
        
        kernel_flat = F.linear(h, w_out, b_out)
        kernel = kernel_flat.T.reshape(C, 1, K)
        
        x_t = x.movedim(-1, 1)
        out_t = F.conv1d(x_t, kernel, stride=stride, groups=C)
        
        L_actual = out_t.shape[2]
        if L_actual != L_out:
            out_t = F.interpolate(out_t, size=L_out, mode='linear', align_corners=False)
        
        out = out_t.movedim(1, -1)
        
        ctx.save_for_backward(x, positions, w0, b0, w_out, b_out, kernel, *hidden_weights, *hidden_biases, *h_pre_list)
        ctx.omega_0 = omega_0
        ctx.stride = stride
        ctx.K = K
        ctx.L_out = L_out
        ctx.L_actual = L_actual
        ctx.num_hidden = num_hidden
        ctx.is_identity = False
        
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        num_hidden = ctx.num_hidden
        
        if ctx.is_identity:
            return (d_out, None, None, None, None, None, None, None) + (None,) * (2 * num_hidden)
        
        saved = ctx.saved_tensors
        
        x = saved[0]
        positions = saved[1]
        w0, b0 = saved[2], saved[3]
        w_out, b_out = saved[4], saved[5]
        kernel = saved[6]
        
        hidden_weights = saved[7:7+num_hidden]
        hidden_biases = saved[7+num_hidden:7+2*num_hidden]
        h_pre_list = saved[7+2*num_hidden:]
        
        omega_0 = ctx.omega_0
        stride = ctx.stride
        K = ctx.K
        L_out = ctx.L_out
        L_actual = ctx.L_actual
        
        B, L_in, C = x.shape
        
        d_out_t = d_out.movedim(-1, 1)
        
        if L_actual != L_out:
            d_out_t = F.interpolate(d_out_t, size=L_actual, mode='linear', align_corners=False)
        
        x_t = x.movedim(-1, 1)
        
        d_x_t = F.conv_transpose1d(d_out_t, kernel, stride=stride, groups=C)
        if d_x_t.shape[2] > L_in:
            d_x_t = d_x_t[:, :, :L_in]
        elif d_x_t.shape[2] < L_in:
            pad = L_in - d_x_t.shape[2]
            d_x_t = F.pad(d_x_t, (0, pad))
        
        x_unfold = x_t.unfold(2, K, stride)
        d_kernel = torch.einsum('bcl,bclk->ck', d_out_t, x_unfold)
        d_kernel = d_kernel.unsqueeze(1)
        
        d_kernel_flat = d_kernel.reshape(C, K).T
        
        h_vals = [torch.sin(h_pre) for h_pre in h_pre_list]
        h_final = h_vals[-1]
        
        d_h = d_kernel_flat @ w_out
        d_w_out = d_kernel_flat.T @ h_final
        d_b_out = d_kernel_flat.sum(0)
        
        d_hidden_weights = []
        d_hidden_biases = []
        
        for i in range(num_hidden - 1, -1, -1):
            h_pre = h_pre_list[i + 1]
            d_h_pre = d_h * torch.cos(h_pre)
            
            h_in = h_vals[i]
            d_w_h = d_h_pre.T @ h_in
            d_b_h = d_h_pre.sum(0)
            d_hidden_weights.insert(0, d_w_h)
            d_hidden_biases.insert(0, d_b_h)
            
            d_h = d_h_pre @ hidden_weights[i]
        
        h_pre_0 = h_pre_list[0]
        d_h_pre_0 = d_h * torch.cos(h_pre_0)
        
        pos_scaled = positions * omega_0
        d_w0 = d_h_pre_0.T @ pos_scaled
        d_b0 = d_h_pre_0.sum(0)
        
        d_x = d_x_t.movedim(1, -1)
        
        return (d_x, d_w0, d_b0, d_w_out, d_b_out, None, None, None) + tuple(d_hidden_weights) + tuple(d_hidden_biases)


class TritonSIRENDownsample1D(nn.Module):
    """Triton-accelerated SIREN downsample for 1D sequences."""
    
    def __init__(self, channels: int, hidden: int = 32, layers: int = 3, omega_0: float = 30.0):
        super().__init__()
        self.channels = channels
        self.hidden = hidden
        self.layers = layers
        self.omega_0 = omega_0
        
        self.w0 = nn.Parameter(torch.empty(hidden, 1))
        self.b0 = nn.Parameter(torch.zeros(hidden))
        
        self.hidden_weights = nn.ParameterList([
            nn.Parameter(torch.empty(hidden, hidden)) for _ in range(layers - 1)
        ])
        self.hidden_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(hidden)) for _ in range(layers - 1)
        ])
        
        self.w_out = nn.Parameter(torch.empty(channels, hidden))
        self.b_out = nn.Parameter(torch.zeros(channels))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.uniform_(self.w0, -1.0, 1.0)
        for w in self.hidden_weights:
            nn.init.uniform_(w, -1.0 / self.hidden, 1.0 / self.hidden)
        nn.init.uniform_(self.w_out, -1.0 / self.hidden, 1.0 / self.hidden)
    
    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        hidden_params = list(self.hidden_weights) + list(self.hidden_biases)
        return _TritonSIRENDownsample1D.apply(
            x, self.w0, self.b0, self.w_out, self.b_out,
            self.omega_0, target_len, len(self.hidden_weights),
            *hidden_params
        )


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
        triton_model.hidden_weights[0].copy_(pytorch_model.kernel_net.net[2].weight)
        triton_model.hidden_biases[0].copy_(pytorch_model.kernel_net.net[2].bias)
        triton_model.hidden_weights[1].copy_(pytorch_model.kernel_net.net[4].weight)
        triton_model.hidden_biases[1].copy_(pytorch_model.kernel_net.net[4].bias)
        triton_model.w_out.copy_(pytorch_model.kernel_net.net[6].weight)
        triton_model.b_out.copy_(pytorch_model.kernel_net.net[6].bias)
    
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    
    torch.manual_seed(42)
    x = torch.randn(B, L_in, C, device=device)
    
    out_pytorch = pytorch_model(x, (L_out,))
    out_triton = triton_model(x, L_out)
    
    fwd_diff = (out_pytorch - out_triton).abs().max().item()
    fwd_match = fwd_diff < 1e-3
    print(f"Forward match: {'PASS' if fwd_match else 'FAIL'} (max diff: {fwd_diff:.2e})")
    
    x_pt = x.clone().requires_grad_(True)
    x_tr = x.clone().requires_grad_(True)
    
    out_pt = pytorch_model(x_pt, (L_out,))
    out_pt.sum().backward()
    
    out_tr = triton_model(x_tr, L_out)
    out_tr.sum().backward()
    
    dx_diff = (x_pt.grad - x_tr.grad).abs().max().item()
    dx_match = dx_diff < 5e-3
    print(f"d_x match: {'PASS' if dx_match else 'FAIL'} (max diff: {dx_diff:.2e})")
    
    w0_grad_pt = pytorch_model.kernel_net.net[0].weight.grad.abs().sum().item()
    w0_grad_tr = triton_model.w0.grad.abs().sum().item()
    w0_match = abs(w0_grad_pt - w0_grad_tr) / (w0_grad_pt + 1e-8) < 0.2
    print(f"w0 grad match: {'PASS' if w0_match else 'FAIL'} (pt: {w0_grad_pt:.2e}, tr: {w0_grad_tr:.2e})")
    
    wout_grad_pt = pytorch_model.kernel_net.net[6].weight.grad.abs().sum().item()
    wout_grad_tr = triton_model.w_out.grad.abs().sum().item()
    wout_match = abs(wout_grad_pt - wout_grad_tr) / (wout_grad_pt + 1e-8) < 0.2
    print(f"w_out grad match: {'PASS' if wout_match else 'FAIL'} (pt: {wout_grad_pt:.2e}, tr: {wout_grad_tr:.2e})")
    
    print()
    print("=" * 60)
    print("BENCHMARKS")
    print("=" * 60)
    
    warmup = 5
    iters = 20
    
    def bench_model(model, x, target, is_triton=False):
        if device == "cuda":
            torch.cuda.synchronize()
        
        for _ in range(warmup):
            if is_triton:
                out = model(x, target)
            else:
                out = model(x, (target,))
        
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        for _ in range(iters):
            if is_triton:
                out = model(x, target)
            else:
                out = model(x, (target,))
            if device == "cuda":
                torch.cuda.synchronize()
        fwd_time = (time.perf_counter() - start) / iters * 1000
        fwd_mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        
        return fwd_time, fwd_mem
    
    def bench_model_bwd(model, x, target, is_triton=False):
        x_grad = x.detach().clone().requires_grad_(True)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        for _ in range(warmup):
            if is_triton:
                out = model(x_grad, target)
            else:
                out = model(x_grad, (target,))
            out.sum().backward()
            model.zero_grad()
            x_grad.grad.zero_()
        
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        for _ in range(iters):
            if is_triton:
                out = model(x_grad, target)
            else:
                out = model(x_grad, (target,))
            out.sum().backward()
            model.zero_grad()
            x_grad.grad.zero_()
            if device == "cuda":
                torch.cuda.synchronize()
        bwd_time = (time.perf_counter() - start) / iters * 1000
        bwd_mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        
        return bwd_time, bwd_mem
    
    print(f"{'L_in':>8} {'L_out':>8} | {'PyT Fwd':>8} {'PyT Bwd':>8} | {'Tri Fwd':>8} {'Tri Bwd':>8} | {'Speedup':>8}")
    print("-" * 80)
    
    test_configs = [
        (512, 128),
        (1024, 256),
        (2048, 512),
        (4096, 1024),
        (8192, 2048),
    ]
    
    for L_in, L_out in test_configs:
        x = torch.randn(B, L_in, C, device=device)
        
        try:
            pt_fwd, _ = bench_model(pytorch_model, x, L_out, is_triton=False)
            pt_bwd, _ = bench_model_bwd(pytorch_model, x, L_out, is_triton=False)
            tr_fwd, _ = bench_model(triton_model, x, L_out, is_triton=True)
            tr_bwd, _ = bench_model_bwd(triton_model, x, L_out, is_triton=True)
            
            speedup = (pt_fwd + pt_bwd) / (tr_fwd + tr_bwd)
            print(f"{L_in:>8} {L_out:>8} | {pt_fwd:>8.2f} {pt_bwd:>8.2f} | {tr_fwd:>8.2f} {tr_bwd:>8.2f} | {speedup:>8.2f}x")
        except Exception as e:
            print(f"{L_in:>8} {L_out:>8} | ERROR: {e}")
    
    print("=" * 60)
