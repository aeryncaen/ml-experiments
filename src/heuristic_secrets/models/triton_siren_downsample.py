"""Triton kernels for SIRENDownsampleND - all fused kernels, minimal PyTorch ops."""

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
    def siren_kernel_gen_fwd(
        pos_ptr, out_ptr,
        w0_ptr, b0_ptr,
        w1_ptr, b1_ptr,
        w2_ptr, b2_ptr,
        w3_ptr, b3_ptr,
        h0_ptr, h1_ptr, h2_ptr,
        omega_0,
        K: tl.constexpr,
        hidden: tl.constexpr,
        out_dim: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """
        Fused SIREN kernel generation: pos -> Linear -> Sin -> Linear -> Sin -> Linear -> Sin -> Linear
        Each program handles one position. Saves pre-sin activations for backward.
        
        pos: (K, 1) - kernel positions
        w0: (hidden, 1), b0: (hidden,)
        w1, w2: (hidden, hidden), b1, b2: (hidden,)
        w3: (out_dim, hidden), b3: (out_dim,)
        out: (K, out_dim) - kernel weights per channel
        h0, h1, h2: (K, hidden) - pre-sin activations for backward
        """
        pid_k = tl.program_id(0)
        
        if pid_k >= K:
            return
        
        pos = tl.load(pos_ptr + pid_k) * omega_0
        
        h_idx = tl.arange(0, BLOCK_H)
        h_mask = h_idx < hidden
        
        w0_vals = tl.load(w0_ptr + h_idx, mask=h_mask, other=0.0)
        b0_vals = tl.load(b0_ptr + h_idx, mask=h_mask, other=0.0)
        h0_pre = w0_vals * pos + b0_vals
        tl.store(h0_ptr + pid_k * hidden + h_idx, h0_pre, mask=h_mask)
        h0 = tl.sin(h0_pre)
        
        h1_pre = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for hh in range(hidden):
            h0_val = tl.sum(tl.where(h_idx == hh, h0, 0.0))
            w1_col = tl.load(w1_ptr + h_idx * hidden + hh, mask=h_mask, other=0.0)
            h1_pre += w1_col * h0_val
        b1_vals = tl.load(b1_ptr + h_idx, mask=h_mask, other=0.0)
        h1_pre = h1_pre + b1_vals
        tl.store(h1_ptr + pid_k * hidden + h_idx, h1_pre, mask=h_mask)
        h1 = tl.sin(h1_pre)
        
        h2_pre = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for hh in range(hidden):
            h1_val = tl.sum(tl.where(h_idx == hh, h1, 0.0))
            w2_col = tl.load(w2_ptr + h_idx * hidden + hh, mask=h_mask, other=0.0)
            h2_pre += w2_col * h1_val
        b2_vals = tl.load(b2_ptr + h_idx, mask=h_mask, other=0.0)
        h2_pre = h2_pre + b2_vals
        tl.store(h2_ptr + pid_k * hidden + h_idx, h2_pre, mask=h_mask)
        h2 = tl.sin(h2_pre)
        
        out_idx = tl.arange(0, BLOCK_H)
        out_mask = out_idx < out_dim
        
        out_vals = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for hh in range(hidden):
            h2_val = tl.sum(tl.where(h_idx == hh, h2, 0.0))
            w3_col = tl.load(w3_ptr + out_idx * hidden + hh, mask=out_mask, other=0.0)
            out_vals += w3_col * h2_val
        b3_vals = tl.load(b3_ptr + out_idx, mask=out_mask, other=0.0)
        out_vals = out_vals + b3_vals
        
        tl.store(out_ptr + pid_k * out_dim + out_idx, out_vals, mask=out_mask)

    @triton.jit
    def depthwise_conv1d_fwd(
        x_ptr, kernel_ptr, out_ptr,
        B: tl.constexpr, L_in: tl.constexpr, C: tl.constexpr,
        L_out: tl.constexpr, K: tl.constexpr, stride: tl.constexpr,
        BLOCK_C: tl.constexpr,
    ):
        """
        Fused depthwise strided 1D conv.
        x: (B, C, L_in) - channels first
        kernel: (C, K) - kernel per channel
        out: (B, C, L_out)
        """
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
        """
        Backward for x: d_x[b, c, in_pos] += d_out[b, c, l] * kernel[c, k] for all l,k where in_start + k == in_pos
        """
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
        """
        Backward for kernel: d_kernel[c, k] = sum over b,l of d_out[b,c,l] * x[b,c,in_start+k]
        """
        pid_k = tl.program_id(0)
        pid_c_block = tl.program_id(1)
        
        c_idx = pid_c_block * BLOCK_C + tl.arange(0, BLOCK_C)
        c_mask = c_idx < C
        
        acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
        
        for b in range(B):
            for l in range(L_out):
                in_pos = l * stride + pid_k
                if in_pos < L_in:
                    d_out_offs = b * C * L_out + c_idx * L_out + l
                    d_out_vals = tl.load(d_out_ptr + d_out_offs, mask=c_mask, other=0.0)
                    
                    x_offs = b * C * L_in + c_idx * L_in + in_pos
                    x_vals = tl.load(x_ptr + x_offs, mask=c_mask, other=0.0)
                    
                    acc += d_out_vals * x_vals
        
        d_k_offs = c_idx * K + pid_k
        tl.store(d_kernel_ptr + d_k_offs, acc, mask=c_mask)

    @triton.jit
    def siren_kernel_gen_bwd(
        pos_ptr, d_out_ptr,
        w0_ptr, w1_ptr, w2_ptr, w3_ptr,
        h0_ptr, h1_ptr, h2_ptr,
        d_w0_ptr, d_b0_ptr,
        d_w1_ptr, d_b1_ptr,
        d_w2_ptr, d_b2_ptr,
        d_w3_ptr, d_b3_ptr,
        omega_0,
        K: tl.constexpr,
        hidden: tl.constexpr,
        out_dim: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        """
        Backward for SIREN kernel generation. Each program handles one position.
        """
        pid_k = tl.program_id(0)
        
        if pid_k >= K:
            return
        
        pos = tl.load(pos_ptr + pid_k) * omega_0
        
        h_idx = tl.arange(0, BLOCK_H)
        h_mask = h_idx < hidden
        out_idx = tl.arange(0, BLOCK_H)
        out_mask = out_idx < out_dim
        
        h0_pre = tl.load(h0_ptr + pid_k * hidden + h_idx, mask=h_mask, other=0.0)
        h1_pre = tl.load(h1_ptr + pid_k * hidden + h_idx, mask=h_mask, other=0.0)
        h2_pre = tl.load(h2_ptr + pid_k * hidden + h_idx, mask=h_mask, other=0.0)
        
        h0 = tl.sin(h0_pre)
        h1 = tl.sin(h1_pre)
        h2 = tl.sin(h2_pre)
        
        d_out = tl.load(d_out_ptr + pid_k * out_dim + out_idx, mask=out_mask, other=0.0)
        
        for o in range(out_dim):
            d_o = tl.sum(tl.where(out_idx == o, d_out, 0.0))
            tl.atomic_add(d_w3_ptr + o * hidden + h_idx, d_o * h2, mask=h_mask)
            tl.atomic_add(d_b3_ptr + o, d_o)
        
        d_h2 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for o in range(out_dim):
            d_o = tl.sum(tl.where(out_idx == o, d_out, 0.0))
            w3_row = tl.load(w3_ptr + o * hidden + h_idx, mask=h_mask, other=0.0)
            d_h2 += d_o * w3_row
        
        d_h2_pre = d_h2 * tl.cos(h2_pre)
        
        for hh in range(hidden):
            d_hh = tl.sum(tl.where(h_idx == hh, d_h2_pre, 0.0))
            tl.atomic_add(d_w2_ptr + hh * hidden + h_idx, d_hh * h1, mask=h_mask)
            tl.atomic_add(d_b2_ptr + hh, d_hh)
        
        d_h1 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for hh in range(hidden):
            d_hh = tl.sum(tl.where(h_idx == hh, d_h2_pre, 0.0))
            w2_row = tl.load(w2_ptr + hh * hidden + h_idx, mask=h_mask, other=0.0)
            d_h1 += d_hh * w2_row
        
        d_h1_pre = d_h1 * tl.cos(h1_pre)
        
        for hh in range(hidden):
            d_hh = tl.sum(tl.where(h_idx == hh, d_h1_pre, 0.0))
            tl.atomic_add(d_w1_ptr + hh * hidden + h_idx, d_hh * h0, mask=h_mask)
            tl.atomic_add(d_b1_ptr + hh, d_hh)
        
        d_h0 = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for hh in range(hidden):
            d_hh = tl.sum(tl.where(h_idx == hh, d_h1_pre, 0.0))
            w1_row = tl.load(w1_ptr + hh * hidden + h_idx, mask=h_mask, other=0.0)
            d_h0 += d_hh * w1_row
        
        d_h0_pre = d_h0 * tl.cos(h0_pre)
        
        for hh in range(hidden):
            d_hh = tl.sum(tl.where(h_idx == hh, d_h0_pre, 0.0))
            tl.atomic_add(d_w0_ptr + hh, d_hh * pos)
            tl.atomic_add(d_b0_ptr + hh, d_hh)


def triton_siren_kernel_gen(
    positions: torch.Tensor,
    w0: torch.Tensor, b0: torch.Tensor,
    w1: torch.Tensor, b1: torch.Tensor,
    w2: torch.Tensor, b2: torch.Tensor,
    w3: torch.Tensor, b3: torch.Tensor,
    omega_0: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate kernel weights using SIREN MLP. Returns (kernel, h0, h1, h2)."""
    K = positions.shape[0]
    hidden = w0.shape[0]
    out_dim = w3.shape[0]
    
    kernel = torch.empty(K, out_dim, device=positions.device, dtype=positions.dtype)
    h0 = torch.empty(K, hidden, device=positions.device, dtype=positions.dtype)
    h1 = torch.empty(K, hidden, device=positions.device, dtype=positions.dtype)
    h2 = torch.empty(K, hidden, device=positions.device, dtype=positions.dtype)
    
    BLOCK_H = triton.next_power_of_2(max(hidden, out_dim))
    grid = (K,)
    
    siren_kernel_gen_fwd[grid](
        positions, kernel,
        w0, b0, w1, b1, w2, b2, w3, b3,
        h0, h1, h2,
        omega_0,
        K, hidden, out_dim, BLOCK_H,
    )
    
    return kernel, h0, h1, h2


def triton_siren_kernel_gen_bwd(
    positions: torch.Tensor,
    d_kernel: torch.Tensor,
    w0: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor, w3: torch.Tensor,
    h0: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor,
    omega_0: float,
) -> tuple[torch.Tensor, ...]:
    """Backward for SIREN kernel generation."""
    K = positions.shape[0]
    hidden = w0.shape[0]
    out_dim = w3.shape[0]
    
    d_w0 = torch.zeros_like(w0)
    d_b0 = torch.zeros(hidden, device=positions.device, dtype=positions.dtype)
    d_w1 = torch.zeros_like(w1)
    d_b1 = torch.zeros(hidden, device=positions.device, dtype=positions.dtype)
    d_w2 = torch.zeros_like(w2)
    d_b2 = torch.zeros(hidden, device=positions.device, dtype=positions.dtype)
    d_w3 = torch.zeros_like(w3)
    d_b3 = torch.zeros(out_dim, device=positions.device, dtype=positions.dtype)
    
    BLOCK_H = triton.next_power_of_2(max(hidden, out_dim))
    grid = (K,)
    
    siren_kernel_gen_bwd[grid](
        positions, d_kernel,
        w0, w1, w2, w3,
        h0, h1, h2,
        d_w0, d_b0, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3,
        omega_0,
        K, hidden, out_dim, BLOCK_H,
    )
    
    return d_w0, d_b0, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3


def triton_depthwise_conv1d(
    x: torch.Tensor,
    kernel: torch.Tensor,
    stride: int,
) -> torch.Tensor:
    """Depthwise strided conv1d. x: (B, C, L_in), kernel: (C, K) -> (B, C, L_out)"""
    B, C, L_in = x.shape
    K = kernel.shape[1]
    L_out = (L_in - K) // stride + 1
    
    out = torch.empty(B, C, L_out, device=x.device, dtype=x.dtype)
    
    BLOCK_C = min(64, triton.next_power_of_2(C))
    grid = (B, L_out, triton.cdiv(C, BLOCK_C))
    
    depthwise_conv1d_fwd[grid](
        x, kernel, out,
        B, L_in, C, L_out, K, stride,
        BLOCK_C,
    )
    
    return out


def triton_depthwise_conv1d_bwd(
    x: torch.Tensor,
    kernel: torch.Tensor,
    d_out: torch.Tensor,
    stride: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward for depthwise conv1d."""
    B, C, L_in = x.shape
    K = kernel.shape[1]
    L_out = d_out.shape[2]
    
    d_x = torch.zeros_like(x)
    d_kernel = torch.zeros_like(kernel)
    
    BLOCK_C = min(64, triton.next_power_of_2(C))
    
    grid_x = (B, L_out, triton.cdiv(C, BLOCK_C))
    depthwise_conv1d_bwd_x[grid_x](
        d_out, kernel, d_x,
        B, L_in, C, L_out, K, stride,
        BLOCK_C,
    )
    
    grid_k = (K, triton.cdiv(C, BLOCK_C))
    depthwise_conv1d_bwd_kernel[grid_k](
        x, d_out, d_kernel,
        B, L_in, C, L_out, K, stride,
        BLOCK_C,
    )
    
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
        
        positions = torch.linspace(-1, 1, K, device=x.device, dtype=x.dtype)
        
        kernel_flat, h0, h1, h2 = triton_siren_kernel_gen(
            positions, w0, b0, w1, b1, w2, b2, w3, b3, omega_0
        )
        
        kernel = kernel_flat.T.contiguous()
        
        x_t = x.movedim(-1, 1).contiguous()
        out_t = triton_depthwise_conv1d(x_t, kernel, stride)
        
        L_actual = out_t.shape[2]
        if L_actual != L_out:
            out_t = F.interpolate(out_t, size=L_out, mode='linear', align_corners=False)
        
        out = out_t.movedim(1, -1)
        
        ctx.save_for_backward(x_t, positions, kernel, h0, h1, h2, w0, w1, w2, w3, b0, b1, b2, b3)
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
        
        x_t, positions, kernel, h0, h1, h2, w0, w1, w2, w3, b0, b1, b2, b3 = ctx.saved_tensors
        omega_0 = ctx.omega_0
        stride = ctx.stride
        L_out = ctx.L_out
        L_actual = ctx.L_actual
        
        B, C, L_in = x_t.shape
        
        d_out_t = d_out.movedim(-1, 1).contiguous()
        
        if L_actual != L_out:
            d_out_t = F.interpolate(d_out_t, size=L_actual, mode='linear', align_corners=False)
        
        d_x_t, d_kernel = triton_depthwise_conv1d_bwd(x_t, kernel, d_out_t, stride)
        
        d_kernel_flat = d_kernel.T.contiguous()
        
        d_w0, d_b0, d_w1, d_b1, d_w2, d_b2, d_w3, d_b3 = triton_siren_kernel_gen_bwd(
            positions, d_kernel_flat, w0, w1, w2, w3, h0, h1, h2, omega_0
        )
        
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
        
        positions = torch.linspace(-1, 1, K, device=x.device, dtype=x.dtype).unsqueeze(-1)
        pos_scaled = positions * self.omega_0
        
        h = torch.sin(F.linear(pos_scaled, self.w0, self.b0))
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
    
    print("=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    
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
    print("=" * 70)
    print("BENCHMARKS")
    print("=" * 70)
    
    warmup = 10
    iters = 50
    
    def bench(model, x, target, is_triton=False):
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
    
    print("=" * 70)
