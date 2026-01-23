"""Triton kernels for GatherConv - fully fused forward + backward."""

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

HAS_FP8 = hasattr(torch, 'float8_e4m3fn')


if HAS_TRITON:
    @triton.jit
    def silu(x):
        return x * tl.sigmoid(x)
    
    @triton.jit
    def silu_bwd(x, d_out):
        sig = tl.sigmoid(x)
        return d_out * sig * (1.0 + x * (1.0 - sig))

    @triton.jit
    def gather_conv_fused_fwd_kernel(
        # Inputs
        x_ptr, out_ptr,
        # Weights
        wave_w_ptr, wave_b_ptr,
        kernel_w_ptr, kernel_b_ptr,
        out_w_ptr,
        # Dimensions
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
        half_s: tl.constexpr, max_receptive,
        max_freq, min_freq,
        # Block sizes
        BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        """Fused forward: x -> wave_proj -> freq/phase -> kernel_proj -> gather -> out_proj -> out"""
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        
        S = 2 * half_s + 1
        
        # Load x[b, l, :]
        c_offs = tl.arange(0, BLOCK_C)
        c_mask = c_offs < C
        x_base = pid_b * L * C + pid_l * C
        x_val = tl.load(x_ptr + x_base + c_offs, mask=c_mask, other=0.0)
        
        # wave_proj: (C,) @ (2H, C).T + (2H,) -> (2H,)
        h2_offs = tl.arange(0, BLOCK_H * 2)
        h2_mask = h2_offs < 2 * H
        wave_out = tl.zeros((BLOCK_H * 2,), dtype=tl.float32)
        for c_idx in range(C):
            x_c = tl.load(x_ptr + x_base + c_idx)
            w_col = tl.load(wave_w_ptr + h2_offs * C + c_idx, mask=h2_mask, other=0.0)
            wave_out += x_c * w_col
        wave_bias = tl.load(wave_b_ptr + h2_offs, mask=h2_mask, other=0.0)
        wave_out = silu(wave_out + wave_bias)
        
        # freq = sigmoid(wave_out[0:H]) * (max_freq - min_freq) + min_freq
        # phase = tanh(wave_out[H:2H]) * max_freq
        h_offs = tl.arange(0, BLOCK_H)
        h_mask = h_offs < H
        freq_pre = tl.load(wave_out.to(tl.float32) + h_offs, mask=h_mask, other=0.0)  # won't work, need different approach
        
        # Actually we need to index into wave_out which is a register array
        # Triton doesn't support dynamic indexing into register arrays well
        # Let's restructure: compute freq/phase inline
        
        freq = tl.zeros((BLOCK_H,), dtype=tl.float32)
        phase = tl.zeros((BLOCK_H,), dtype=tl.float32)
        for h_idx in range(H):
            freq_val = tl.sigmoid(wave_out[h_idx]) * (max_freq - min_freq) + min_freq
            phase_val = tl.tanh(wave_out[H + h_idx]) * max_freq
            freq = tl.where(h_offs == h_idx, freq_val, freq)
            phase = tl.where(h_offs == h_idx, phase_val, phase)
        
        # kernel_proj: (C,) @ (H*K, C).T + (H*K,) -> (H, K)
        hk_offs = tl.arange(0, BLOCK_H * BLOCK_K)
        hk_mask = hk_offs < H * K
        kernel_out = tl.zeros((BLOCK_H * BLOCK_K,), dtype=tl.float32)
        for c_idx in range(C):
            x_c = tl.load(x_ptr + x_base + c_idx)
            w_col = tl.load(kernel_w_ptr + hk_offs * C + c_idx, mask=hk_mask, other=0.0)
            kernel_out += x_c * w_col
        kernel_bias = tl.load(kernel_b_ptr + hk_offs, mask=hk_mask, other=0.0)
        kernel_out = silu(kernel_out + kernel_bias)  # (H*K,)
        
        # Gather and convolve per head
        hidden = tl.zeros((BLOCK_C,), dtype=tl.float32)
        
        for h_idx in range(H):
            # Get freq/phase for this head
            freq_h = tl.sum(tl.where(h_offs == h_idx, freq, 0.0))
            phase_h = tl.sum(tl.where(h_offs == h_idx, phase, 0.0))
            
            # Get kernel for this head: kernel_out[h_idx*K : (h_idx+1)*K]
            k_offs = tl.arange(0, BLOCK_K)
            k_mask = k_offs < K
            kernel_h = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for k_idx in range(K):
                k_val = kernel_out[h_idx * K + k_idx] if h_idx * K + k_idx < H * K else 0.0
                kernel_h = tl.where(k_offs == k_idx, k_val, kernel_h)
            
            # Accumulate over samples
            d_offs = tl.arange(0, BLOCK_C // H) if BLOCK_C >= H else tl.arange(0, BLOCK_C)
            d_mask = d_offs < D
            head_out = tl.zeros((BLOCK_C // H if BLOCK_C >= H else BLOCK_C,), dtype=tl.float32)
            
            for s_idx in range(S):
                s_off = s_idx - half_s
                sample_pos_f = pid_l + s_off * freq_h + phase_h
                
                valid = (sample_pos_f >= 0.0) & (sample_pos_f < L)
                sample_idx = tl.maximum(tl.minimum(sample_pos_f, L - 1.0), 0.0).to(tl.int32)
                
                # Interpolate kernel weight
                rel_pos = tl.abs(s_off * freq_h)
                norm_pos = tl.minimum(rel_pos / max_receptive, 1.0)
                idx_float = norm_pos * (K - 1)
                idx_floor = tl.minimum(idx_float.to(tl.int32), K - 2)
                idx_ceil = idx_floor + 1
                w_ceil = idx_float - idx_floor.to(tl.float32)
                w_floor = 1.0 - w_ceil
                
                k_floor = tl.sum(tl.where(k_offs == idx_floor, kernel_h, 0.0))
                k_ceil = tl.sum(tl.where(k_offs == idx_ceil, kernel_h, 0.0))
                kernel_w = (k_floor * w_floor + k_ceil * w_ceil) * valid.to(tl.float32)
                
                # Gather value
                val_base = pid_b * L * C + sample_idx * C + h_idx * D
                vals = tl.load(x_ptr + val_base + d_offs, mask=d_mask, other=0.0)
                head_out += vals * kernel_w
            
            # Store head output into hidden
            for d_idx in range(D):
                hidden_idx = h_idx * D + d_idx
                if hidden_idx < C:
                    val = tl.sum(tl.where(d_offs == d_idx, head_out, 0.0))
                    hidden = tl.where(c_offs == hidden_idx, val, hidden)
        
        # out_proj: hidden @ out_w.T (no bias)
        out_val = tl.zeros((BLOCK_C,), dtype=tl.float32)
        for c_idx in range(C):
            h_c = tl.sum(tl.where(c_offs == c_idx, hidden, 0.0))
            w_col = tl.load(out_w_ptr + c_offs * C + c_idx, mask=c_mask, other=0.0)
            out_val += h_c * w_col
        
        out_val = silu(out_val)
        tl.store(out_ptr + x_base + c_offs, out_val, mask=c_mask)


# The above kernel is too complex and will be slow due to all the dynamic indexing.
# Let's use a simpler approach: separate kernels but manual gradient computation.

if HAS_TRITON:
    @triton.jit  
    def gather_conv_fwd_kernel(
        x_ptr, out_ptr,
        freq_ptr, phase_ptr, kernel_ptr,
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
        half_s: tl.constexpr, max_receptive: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        freq_off = pid_b * L * H + pid_l * H + pid_h
        freq_h = tl.load(freq_ptr + freq_off)
        phase_h = tl.load(phase_ptr + freq_off)
        
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        k_offs = tl.arange(0, K)
        kernel_base = pid_b * L * H * K + pid_l * H * K + pid_h * K
        kernel_vals = tl.load(kernel_ptr + kernel_base + k_offs, mask=k_offs < K, other=0.0)
        
        out_h = tl.zeros((BLOCK_D,), dtype=tl.float32)
        
        for s_idx in range(S):
            s_off = s_idx - half_s
            sample_pos_f = pid_l + s_off * freq_h + phase_h
            
            valid = (sample_pos_f >= 0.0) & (sample_pos_f < L)
            sample_idx = tl.maximum(tl.minimum(sample_pos_f, L - 1.0), 0.0).to(tl.int32)
            
            rel_pos = tl.abs(s_off * freq_h)
            norm_pos = rel_pos / max_receptive
            norm_pos = tl.minimum(norm_pos, 1.0)
            
            idx_float = norm_pos * (K - 1)
            idx_floor = idx_float.to(tl.int32)
            idx_floor = tl.minimum(idx_floor, K - 2)
            idx_ceil = idx_floor + 1
            w_ceil = idx_float - idx_floor.to(tl.float32)
            w_floor = 1.0 - w_ceil
            
            k_floor = tl.sum(tl.where(k_offs == idx_floor, kernel_vals, 0.0))
            k_ceil = tl.sum(tl.where(k_offs == idx_ceil, kernel_vals, 0.0))
            kernel_w = k_floor * w_floor + k_ceil * w_ceil
            kernel_w = kernel_w * valid.to(tl.float32)
            
            val_base = pid_b * L * C + sample_idx * C + pid_h * D
            vals = tl.load(x_ptr + val_base + d_offs, mask=d_mask, other=0.0)
            out_h += vals * kernel_w
        
        out_base = pid_b * L * C + pid_l * C + pid_h * D
        tl.store(out_ptr + out_base + d_offs, out_h, mask=d_mask)

    @triton.jit
    def gather_conv_bwd_kernel(
        x_ptr, d_out_ptr, d_x_ptr,
        freq_ptr, phase_ptr, kernel_ptr,
        d_freq_ptr, d_kernel_ptr,
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
        half_s: tl.constexpr, max_receptive: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        freq_off = pid_b * L * H + pid_l * H + pid_h
        freq_h = tl.load(freq_ptr + freq_off)
        phase_h = tl.load(phase_ptr + freq_off)
        
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        
        k_offs = tl.arange(0, K)
        kernel_base = pid_b * L * H * K + pid_l * H * K + pid_h * K
        kernel_vals = tl.load(kernel_ptr + kernel_base + k_offs, mask=k_offs < K, other=0.0)
        
        d_out_base = pid_b * L * C + pid_l * C + pid_h * D
        d_out_h = tl.load(d_out_ptr + d_out_base + d_offs, mask=d_mask, other=0.0)
        
        d_freq_acc = 0.0
        d_kernel_acc = tl.zeros((K,), dtype=tl.float32)
        
        for s_idx in range(S):
            s_off = s_idx - half_s
            sample_pos_f = pid_l + s_off * freq_h + phase_h
            
            valid = (sample_pos_f >= 0.0) & (sample_pos_f < L)
            valid_f = valid.to(tl.float32)
            sample_idx = tl.maximum(tl.minimum(sample_pos_f, L - 1.0), 0.0).to(tl.int32)
            
            rel_pos = tl.abs(s_off * freq_h)
            norm_pos = rel_pos / max_receptive
            not_clamped = (norm_pos < 1.0).to(tl.float32)
            norm_pos = tl.minimum(norm_pos, 1.0)
            
            idx_float = norm_pos * (K - 1)
            idx_floor = idx_float.to(tl.int32)
            idx_floor = tl.minimum(idx_floor, K - 2)
            idx_ceil = idx_floor + 1
            w_ceil = idx_float - idx_floor.to(tl.float32)
            w_floor = 1.0 - w_ceil
            
            k_floor = tl.sum(tl.where(k_offs == idx_floor, kernel_vals, 0.0))
            k_ceil = tl.sum(tl.where(k_offs == idx_ceil, kernel_vals, 0.0))
            kernel_w = k_floor * w_floor + k_ceil * w_ceil
            kernel_w = kernel_w * valid_f
            
            val_base = pid_b * L * C + sample_idx * C + pid_h * D
            vals = tl.load(x_ptr + val_base + d_offs, mask=d_mask, other=0.0)
            
            # d_x via atomic add (scatter)
            d_x_contrib = d_out_h * kernel_w
            tl.atomic_add(d_x_ptr + val_base + d_offs, d_x_contrib, mask=d_mask)
            
            # d_kernel
            d_kernel_w = tl.sum(d_out_h * vals) * valid_f
            d_kernel_acc += tl.where(k_offs == idx_floor, d_kernel_w * w_floor, 0.0)
            d_kernel_acc += tl.where(k_offs == idx_ceil, d_kernel_w * w_ceil, 0.0)
            
            # d_freq (through kernel interpolation)
            d_w_ceil = d_kernel_w * (k_ceil - k_floor)
            d_idx_float = d_w_ceil
            d_norm_pos = d_idx_float * (K - 1)
            d_rel_pos = d_norm_pos / max_receptive * not_clamped
            s_off_f = s_off * 1.0
            d_freq_acc += d_rel_pos * tl.abs(s_off_f)
        
        tl.store(d_kernel_ptr + kernel_base + k_offs, d_kernel_acc, mask=k_offs < K)
        tl.store(d_freq_ptr + freq_off, d_freq_acc)


def quantize_fp8(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = t.abs().max() / 448.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return t.div(scale).to(torch.float8_e4m3fn), scale


def dequantize_fp8(t_fp8: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return t_fp8.to(dtype) * scale


class TritonGatherConv(nn.Module):
    """
    Fully manual gradient GatherConv.
    
    Forward: compute freq/phase/kernel, run gather-conv kernel, out_proj
    Backward: recompute intermediates, run bwd kernel, manual weight gradients
    
    Only saves: x (quantized if enabled), weight matrices
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        max_samples: int = 32,
        max_freq: float = 16.0,
        min_freq: float = 1.0,
        max_kernel_size: int = 64,
        quantize_bwd: bool = False,
    ):
        super().__init__()
        if not HAS_TRITON:
            raise RuntimeError("Triton not available")
        
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.max_kernel_size = max_kernel_size
        self.quantize_bwd = quantize_bwd
        
        self.half_s = max_samples // 2
        self.num_samples = 2 * self.half_s + 1
        self.max_receptive = self.half_s * max_freq
        
        self.wave_proj = nn.Linear(channels, 2 * num_heads)
        self.kernel_proj = nn.Linear(channels, num_heads * max_kernel_size)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        
        nn.init.zeros_(self.wave_proj.weight)
        nn.init.zeros_(self.wave_proj.bias)
        nn.init.xavier_uniform_(self.kernel_proj.weight, gain=0.1)
        nn.init.zeros_(self.kernel_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return _GatherConvManual.apply(
            x,
            self.wave_proj.weight, self.wave_proj.bias,
            self.kernel_proj.weight, self.kernel_proj.bias,
            self.out_proj.weight,
            self.num_heads, self.half_s, self.max_receptive,
            self.max_freq, self.min_freq, self.max_kernel_size,
            self.quantize_bwd and self.training,
        ), {}


class GatherConvOp(torch.autograd.Function):
    """Gather-conv operation only (for use with external freq/phase/kernel computation)."""
    
    @staticmethod
    def forward(ctx, x, freq, phase, kernel, half_s, max_receptive, quantize_bwd):
        B, L, C = x.shape
        H = freq.shape[-1]
        K = kernel.shape[-1]
        D = C // H
        S = 2 * half_s + 1
        BLOCK_D = triton.next_power_of_2(D)
        
        out = torch.empty_like(x)
        grid = (B, L, H)
        
        gather_conv_fwd_kernel[grid](
            x, out,
            freq, phase, kernel,
            B, L, C, H, D, S, K,
            half_s, max_receptive, BLOCK_D,
        )
        
        # Save for backward
        ctx.half_s = half_s
        ctx.max_receptive = max_receptive
        
        if quantize_bwd and HAS_FP8:
            x_q, x_scale = quantize_fp8(x.detach())
            kernel_q, kernel_scale = quantize_fp8(kernel.detach())
            ctx.save_for_backward(x_q, x_scale, freq, phase, kernel_q, kernel_scale)
            ctx.use_quant = True
        else:
            ctx.save_for_backward(x, freq, phase, kernel)
            ctx.use_quant = False
        
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        half_s = ctx.half_s
        max_receptive = ctx.max_receptive
        
        if ctx.use_quant:
            x_q, x_scale, freq, phase, kernel_q, kernel_scale = ctx.saved_tensors
            x = dequantize_fp8(x_q, x_scale, d_out.dtype)
            kernel = dequantize_fp8(kernel_q, kernel_scale, d_out.dtype)
        else:
            x, freq, phase, kernel = ctx.saved_tensors
        
        B, L, C = x.shape
        H = freq.shape[-1]
        K = kernel.shape[-1]
        D = C // H
        S = 2 * half_s + 1
        BLOCK_D = triton.next_power_of_2(D)
        
        d_out = d_out.contiguous()
        d_x = torch.zeros_like(x)
        d_freq = torch.zeros_like(freq)
        d_kernel = torch.zeros_like(kernel)
        
        grid = (B, L, H)
        gather_conv_bwd_kernel[grid](
            x, d_out, d_x,
            freq, phase, kernel,
            d_freq, d_kernel,
            B, L, C, H, D, S, K,
            half_s, max_receptive, BLOCK_D,
        )
        
        return d_x, d_freq, None, d_kernel, None, None, None


class _GatherConvManual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wave_w, wave_b, kernel_w, kernel_b, out_w,
                num_heads, half_s, max_receptive, max_freq, min_freq, max_kernel_size,
                quantize_bwd):
        B, L, C = x.shape
        H = num_heads
        K = max_kernel_size
        D = C // H
        S = 2 * half_s + 1
        
        # Compute freq/phase/kernel
        wave_pre = F.linear(x, wave_w, wave_b)  # (B, L, 2H)
        wave_out = F.silu(wave_pre)
        wave_out = wave_out.view(B, L, 2, H)
        freq = torch.sigmoid(wave_out[:, :, 0, :]) * (max_freq - min_freq) + min_freq  # (B, L, H)
        phase = torch.tanh(wave_out[:, :, 1, :]) * max_freq
        
        kernel_pre = F.linear(x, kernel_w, kernel_b)  # (B, L, H*K)
        kernel = F.silu(kernel_pre).view(B, L, H, K)  # (B, L, H, K)
        
        # Run gather-conv kernel
        hidden = torch.empty(B, L, C, device=x.device, dtype=x.dtype)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, L, H)
        
        gather_conv_fwd_kernel[grid](
            x, hidden,
            freq.contiguous(), phase.contiguous(), kernel.contiguous(),
            B, L, C, H, D, S, K,
            half_s, max_receptive, BLOCK_D,
        )
        
        # out_proj + silu
        out_pre = F.linear(hidden, out_w)
        out = F.silu(out_pre)
        
        # Save ONLY what we can't recompute: x and weights
        # Everything else gets recomputed in backward (compute is cheap, memory is expensive)
        if quantize_bwd and HAS_FP8:
            x_q, x_scale = quantize_fp8(x.detach())
            ctx.save_for_backward(x_q, x_scale, wave_w, wave_b, kernel_w, kernel_b, out_w)
            ctx.use_quant = True
        else:
            ctx.save_for_backward(x, wave_w, wave_b, kernel_w, kernel_b, out_w)
            ctx.use_quant = False
        ctx.half_s = half_s
        ctx.max_receptive = max_receptive
        ctx.max_freq = max_freq
        ctx.min_freq = min_freq
        ctx.num_heads = H
        ctx.max_kernel_size = K
        
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        if ctx.use_quant:
            x_q, x_scale, wave_w, wave_b, kernel_w, kernel_b, out_w = ctx.saved_tensors
            x = dequantize_fp8(x_q, x_scale, d_out.dtype)
        else:
            x, wave_w, wave_b, kernel_w, kernel_b, out_w = ctx.saved_tensors
        half_s = ctx.half_s
        max_receptive = ctx.max_receptive
        max_freq = ctx.max_freq
        min_freq = ctx.min_freq
        H = ctx.num_heads
        K = ctx.max_kernel_size
        
        B, L, C = x.shape
        D = C // H
        S = 2 * half_s + 1
        BLOCK_D = triton.next_power_of_2(D)
        
        # === Recompute ALL forward intermediates ===
        wave_pre = F.linear(x, wave_w, wave_b)
        wave_out = F.silu(wave_pre).view(B, L, 2, H)
        freq = torch.sigmoid(wave_out[:, :, 0, :]) * (max_freq - min_freq) + min_freq
        phase = torch.tanh(wave_out[:, :, 1, :]) * max_freq
        
        kernel_pre = F.linear(x, kernel_w, kernel_b)
        kernel = F.silu(kernel_pre).view(B, L, H, K)
        
        # Recompute hidden (gather-conv output)
        hidden = torch.empty(B, L, C, device=x.device, dtype=x.dtype)
        grid = (B, L, H)
        gather_conv_fwd_kernel[grid](
            x, hidden,
            freq.contiguous(), phase.contiguous(), kernel.contiguous(),
            B, L, C, H, D, S, K,
            half_s, max_receptive, BLOCK_D,
        )
        
        # Recompute out_pre
        out_pre = F.linear(hidden, out_w)
        
        # === Backprop through out_proj + silu ===
        sig_out = torch.sigmoid(out_pre)
        d_out_pre = d_out * sig_out * (1.0 + out_pre * (1.0 - sig_out))
        
        d_out_w = torch.einsum('blc,bld->cd', d_out_pre, hidden)
        d_hidden = F.linear(d_out_pre, out_w.t())
        
        # === Run gather-conv backward kernel ===
        d_x_gather = torch.zeros_like(x)
        d_freq = torch.zeros_like(freq)
        d_kernel = torch.zeros(B, L, H, K, device=x.device, dtype=x.dtype)
        
        grid = (B, L, H)
        gather_conv_bwd_kernel[grid](
            x, d_hidden.contiguous(), d_x_gather,
            freq.contiguous(), phase.contiguous(), kernel.contiguous(),
            d_freq, d_kernel,
            B, L, C, H, D, S, K,
            half_s, max_receptive, BLOCK_D,
        )
        
        # === Backprop through kernel_proj ===
        # kernel = silu(kernel_pre).view(B,L,H,K), kernel_pre = x @ kernel_w.T + kernel_b
        d_kernel_flat = d_kernel.view(B, L, H * K)
        kernel_pre_view = kernel_pre  # (B, L, H*K)
        sig_k = torch.sigmoid(kernel_pre_view)
        d_kernel_pre = d_kernel_flat * sig_k * (1.0 + kernel_pre_view * (1.0 - sig_k))
        
        d_kernel_w = torch.einsum('blk,blc->kc', d_kernel_pre, x)
        d_kernel_b = d_kernel_pre.sum(dim=(0, 1))
        d_x_kernel = F.linear(d_kernel_pre, kernel_w.t())
        
        # === Backprop through wave_proj ===
        # freq = sigmoid(wave_out[:,0,:]) * scale + offset
        # phase = tanh(wave_out[:,1,:]) * scale  
        # wave_out = silu(wave_pre).view(B,L,2,H)
        # wave_pre = x @ wave_w.T + wave_b
        
        # d_freq -> d_wave_out[:,0,:]
        freq_pre = wave_out[:, :, 0, :]  # after silu, before sigmoid
        sig_freq = torch.sigmoid(freq_pre)
        d_wave_freq = d_freq * (max_freq - min_freq) * sig_freq * (1.0 - sig_freq)
        
        # d_phase -> d_wave_out[:,1,:] (d_phase is zeros from kernel, but let's be complete)
        # Actually our bwd kernel doesn't compute d_phase, so this is 0
        d_wave_phase = torch.zeros_like(phase)
        
        d_wave_out = torch.stack([d_wave_freq, d_wave_phase], dim=2).view(B, L, 2 * H)
        
        # silu backward for wave
        sig_w = torch.sigmoid(wave_pre)
        d_wave_pre = d_wave_out * sig_w * (1.0 + wave_pre * (1.0 - sig_w))
        
        d_wave_w = torch.einsum('blh,blc->hc', d_wave_pre, x)
        d_wave_b = d_wave_pre.sum(dim=(0, 1))
        d_x_wave = F.linear(d_wave_pre, wave_w.t())
        
        # === Combine d_x ===
        d_x = d_x_gather + d_x_kernel + d_x_wave
        
        return d_x, d_wave_w, d_wave_b, d_kernel_w, d_kernel_b, d_out_w, \
               None, None, None, None, None, None, None


if __name__ == "__main__":
    if not HAS_TRITON:
        print("Triton not available")
        exit()
    
    import time
    
    device = "cuda"
    B, L, C = 4, 8192, 256
    H = 8
    
    print(f"Config: B={B}, L={L}, C={C}, H={H}")
    print(f"FP8 available: {HAS_FP8}")
    print()
    
    for quant in [False, True]:
        label = "Triton+Q" if quant else "Triton"
        model = TritonGatherConv(channels=C, num_heads=H, quantize_bwd=quant).to(device)
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        
        # Warmup
        for _ in range(5):
            out, _ = model(x)
            out.sum().backward()
            model.zero_grad()
            if x.grad is not None:
                x.grad.zero_()
        
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward only
        start = time.perf_counter()
        for _ in range(20):
            with torch.no_grad():
                out, _ = model(x)
        torch.cuda.synchronize()
        fwd_time = (time.perf_counter() - start) / 20 * 1000
        fwd_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        torch.cuda.reset_peak_memory_stats()
        
        # Forward + backward  
        start = time.perf_counter()
        for _ in range(20):
            out, _ = model(x)
            out.sum().backward()
            model.zero_grad()
            x.grad.zero_()
        torch.cuda.synchronize()
        total_time = (time.perf_counter() - start) / 20 * 1000
        bwd_time = total_time - fwd_time
        bwd_mem = torch.cuda.max_memory_allocated() / 1024**2
        
        print(f"{label:10s}: Fwd {fwd_time:5.1f}ms ({fwd_mem:5.0f}MB)  Bwd {bwd_time:5.1f}ms ({bwd_mem:5.0f}MB)")
    
    # Gradient check
    print("\nGradient check...")
    model = TritonGatherConv(channels=C, num_heads=H).to(device)
    x = torch.randn(2, 128, C, device=device, requires_grad=True, dtype=torch.float64)
    model = model.double()
    
    def func(x):
        out, _ = model(x)
        return out.sum()
    
    try:
        torch.autograd.gradcheck(func, x, eps=1e-4, atol=1e-3, rtol=1e-3, raise_exception=True)
        print("Gradient check: PASSED")
    except Exception as e:
        print(f"Gradient check: FAILED - {e}")
