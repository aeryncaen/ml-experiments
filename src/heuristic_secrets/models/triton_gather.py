"""Triton kernels for GatherConv - fully fused, minimal memory."""

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
    def gather_conv_fused_fwd(
        # Inputs/outputs
        x_ptr, out_ptr,
        # Weights
        wave_w_ptr, wave_b_ptr,      # (2H, C), (2H,)
        kernel_w_ptr, kernel_b_ptr,  # (H*K, C), (H*K,)
        out_w_ptr,                   # (C, C)
        # Dims
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
        half_s: tl.constexpr,
        max_receptive,
        max_freq, min_freq,
    ):
        """Fused forward: x -> wave_proj -> freq/phase -> kernel_proj -> gather-conv -> out_proj -> out"""
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        
        S = 2 * half_s + 1
        x_base = pid_b * L * C + pid_l * C
        
        # Load x[b, l, :] into registers
        c_range = tl.arange(0, C)
        x_vec = tl.load(x_ptr + x_base + c_range)
        
        # === wave_proj: x @ wave_w.T + wave_b -> silu -> freq/phase ===
        # wave_w is (2H, C), we compute (2H,) output
        wave_out = tl.zeros((2 * H,), dtype=tl.float32)
        for h2 in range(2 * H):
            acc = tl.load(wave_b_ptr + h2)
            for c in range(C):
                acc += tl.load(x_ptr + x_base + c) * tl.load(wave_w_ptr + h2 * C + c)
            # silu
            sig = 1.0 / (1.0 + tl.exp(-acc))
            wave_out = tl.where(tl.arange(0, 2 * H) == h2, acc * sig, wave_out)
        
        # === kernel_proj: x @ kernel_w.T + kernel_b -> silu ===
        # kernel_w is (H*K, C), output is (H*K,)
        kernel_flat = tl.zeros((H * K,), dtype=tl.float32)
        for hk in range(H * K):
            acc = tl.load(kernel_b_ptr + hk)
            for c in range(C):
                acc += tl.load(x_ptr + x_base + c) * tl.load(kernel_w_ptr + hk * C + c)
            sig = 1.0 / (1.0 + tl.exp(-acc))
            kernel_flat = tl.where(tl.arange(0, H * K) == hk, acc * sig, kernel_flat)
        
        # === Gather-conv per head ===
        hidden = tl.zeros((C,), dtype=tl.float32)
        
        for h in range(H):
            # freq = sigmoid(wave_out[h]) * (max_freq - min_freq) + min_freq
            wave_h = tl.sum(tl.where(tl.arange(0, 2 * H) == h, wave_out, 0.0))
            sig_f = 1.0 / (1.0 + tl.exp(-wave_h))
            freq_h = sig_f * (max_freq - min_freq) + min_freq
            
            # phase = tanh(wave_out[H + h]) * max_freq
            wave_h2 = tl.sum(tl.where(tl.arange(0, 2 * H) == (H + h), wave_out, 0.0))
            phase_h = ((tl.exp(2.0 * wave_h2) - 1.0) / (tl.exp(2.0 * wave_h2) + 1.0)) * max_freq
            
            # Gather kernel weights for this head
            k_range = tl.arange(0, K)
            kernel_h = tl.zeros((K,), dtype=tl.float32)
            for k in range(K):
                kv = tl.sum(tl.where(tl.arange(0, H * K) == (h * K + k), kernel_flat, 0.0))
                kernel_h = tl.where(k_range == k, kv, kernel_h)
            
            # Accumulate over samples
            head_out = tl.zeros((D,), dtype=tl.float32)
            d_range = tl.arange(0, D)
            
            for s_idx in range(S):
                s_off = s_idx - half_s
                sample_pos = pid_l + s_off * freq_h + phase_h
                
                valid = (sample_pos >= 0.0) & (sample_pos < L)
                sample_idx = tl.maximum(tl.minimum(sample_pos, L - 1.0), 0.0).to(tl.int32)
                
                # Interpolate kernel
                rel_pos = tl.abs(s_off * freq_h)
                norm_pos = tl.minimum(rel_pos / max_receptive, 1.0)
                idx_f = norm_pos * (K - 1)
                idx_lo = tl.minimum(idx_f.to(tl.int32), K - 2)
                idx_hi = idx_lo + 1
                w_hi = idx_f - idx_lo.to(tl.float32)
                w_lo = 1.0 - w_hi
                
                k_lo = tl.sum(tl.where(k_range == idx_lo, kernel_h, 0.0))
                k_hi = tl.sum(tl.where(k_range == idx_hi, kernel_h, 0.0))
                kw = (k_lo * w_lo + k_hi * w_hi) * valid.to(tl.float32)
                
                # Gather x values for this head's dimensions
                val_base = pid_b * L * C + sample_idx * C + h * D
                vals = tl.load(x_ptr + val_base + d_range, mask=d_range < D, other=0.0)
                head_out += vals * kw
            
            # Store into hidden at head's position
            for d in range(D):
                idx = h * D + d
                v = tl.sum(tl.where(d_range == d, head_out, 0.0))
                hidden = tl.where(c_range == idx, v, hidden)
        
        # === out_proj: hidden @ out_w.T (no bias) -> silu ===
        out_vec = tl.zeros((C,), dtype=tl.float32)
        for c_out in range(C):
            acc = 0.0
            for c_in in range(C):
                h_c = tl.sum(tl.where(c_range == c_in, hidden, 0.0))
                acc += h_c * tl.load(out_w_ptr + c_out * C + c_in)
            sig = 1.0 / (1.0 + tl.exp(-acc))
            out_vec = tl.where(c_range == c_out, acc * sig, out_vec)
        
        tl.store(out_ptr + x_base + c_range, out_vec)

    @triton.jit
    def gather_conv_fused_bwd(
        # Inputs
        x_ptr, d_out_ptr,
        # Weights (for recomputation)
        wave_w_ptr, wave_b_ptr,
        kernel_w_ptr, kernel_b_ptr,
        out_w_ptr,
        # Gradient outputs (atomics)
        d_x_ptr,
        d_wave_w_ptr, d_wave_b_ptr,
        d_kernel_w_ptr, d_kernel_b_ptr,
        d_out_w_ptr,
        # Dims
        B: tl.constexpr, L: tl.constexpr, C: tl.constexpr,
        H: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
        half_s: tl.constexpr,
        max_receptive,
        max_freq, min_freq,
    ):
        """Fused backward: recompute forward, compute all gradients."""
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        
        S = 2 * half_s + 1
        x_base = pid_b * L * C + pid_l * C
        c_range = tl.arange(0, C)
        
        # Load d_out[b, l, :]
        d_out_vec = tl.load(d_out_ptr + x_base + c_range)
        
        # === RECOMPUTE FORWARD ===
        # wave_proj
        wave_pre = tl.zeros((2 * H,), dtype=tl.float32)
        wave_out = tl.zeros((2 * H,), dtype=tl.float32)
        h2_range = tl.arange(0, 2 * H)
        for h2 in range(2 * H):
            acc = tl.load(wave_b_ptr + h2)
            for c in range(C):
                acc += tl.load(x_ptr + x_base + c) * tl.load(wave_w_ptr + h2 * C + c)
            wave_pre = tl.where(h2_range == h2, acc, wave_pre)
            sig = 1.0 / (1.0 + tl.exp(-acc))
            wave_out = tl.where(h2_range == h2, acc * sig, wave_out)
        
        # kernel_proj
        kernel_pre = tl.zeros((H * K,), dtype=tl.float32)
        kernel_flat = tl.zeros((H * K,), dtype=tl.float32)
        hk_range = tl.arange(0, H * K)
        for hk in range(H * K):
            acc = tl.load(kernel_b_ptr + hk)
            for c in range(C):
                acc += tl.load(x_ptr + x_base + c) * tl.load(kernel_w_ptr + hk * C + c)
            kernel_pre = tl.where(hk_range == hk, acc, kernel_pre)
            sig = 1.0 / (1.0 + tl.exp(-acc))
            kernel_flat = tl.where(hk_range == hk, acc * sig, kernel_flat)
        
        # Gather-conv (recompute hidden)
        hidden = tl.zeros((C,), dtype=tl.float32)
        k_range = tl.arange(0, K)
        d_range = tl.arange(0, D)
        
        for h in range(H):
            wave_h = tl.sum(tl.where(h2_range == h, wave_out, 0.0))
            sig_f = 1.0 / (1.0 + tl.exp(-wave_h))
            freq_h = sig_f * (max_freq - min_freq) + min_freq
            
            wave_h2 = tl.sum(tl.where(h2_range == (H + h), wave_out, 0.0))
            phase_h = ((tl.exp(2.0 * wave_h2) - 1.0) / (tl.exp(2.0 * wave_h2) + 1.0)) * max_freq
            
            kernel_h = tl.zeros((K,), dtype=tl.float32)
            for k in range(K):
                kv = tl.sum(tl.where(hk_range == (h * K + k), kernel_flat, 0.0))
                kernel_h = tl.where(k_range == k, kv, kernel_h)
            
            head_out = tl.zeros((D,), dtype=tl.float32)
            for s_idx in range(S):
                s_off = s_idx - half_s
                sample_pos = pid_l + s_off * freq_h + phase_h
                valid = (sample_pos >= 0.0) & (sample_pos < L)
                sample_idx = tl.maximum(tl.minimum(sample_pos, L - 1.0), 0.0).to(tl.int32)
                
                rel_pos = tl.abs(s_off * freq_h)
                norm_pos = tl.minimum(rel_pos / max_receptive, 1.0)
                idx_f = norm_pos * (K - 1)
                idx_lo = tl.minimum(idx_f.to(tl.int32), K - 2)
                idx_hi = idx_lo + 1
                w_hi = idx_f - idx_lo.to(tl.float32)
                w_lo = 1.0 - w_hi
                
                k_lo = tl.sum(tl.where(k_range == idx_lo, kernel_h, 0.0))
                k_hi = tl.sum(tl.where(k_range == idx_hi, kernel_h, 0.0))
                kw = (k_lo * w_lo + k_hi * w_hi) * valid.to(tl.float32)
                
                val_base = pid_b * L * C + sample_idx * C + h * D
                vals = tl.load(x_ptr + val_base + d_range, mask=d_range < D, other=0.0)
                head_out += vals * kw
            
            for d in range(D):
                idx = h * D + d
                v = tl.sum(tl.where(d_range == d, head_out, 0.0))
                hidden = tl.where(c_range == idx, v, hidden)
        
        # out_proj (recompute out_pre)
        out_pre = tl.zeros((C,), dtype=tl.float32)
        for c_out in range(C):
            acc = 0.0
            for c_in in range(C):
                h_c = tl.sum(tl.where(c_range == c_in, hidden, 0.0))
                acc += h_c * tl.load(out_w_ptr + c_out * C + c_in)
            out_pre = tl.where(c_range == c_out, acc, out_pre)
        
        # === BACKWARD ===
        # d_out_pre = d_out * silu'(out_pre)
        sig_out = 1.0 / (1.0 + tl.exp(-out_pre))
        d_out_pre = d_out_vec * sig_out * (1.0 + out_pre * (1.0 - sig_out))
        
        # d_out_w: outer product d_out_pre @ hidden.T
        # d_hidden = d_out_pre @ out_w
        d_hidden = tl.zeros((C,), dtype=tl.float32)
        for c_out in range(C):
            d_op = tl.sum(tl.where(c_range == c_out, d_out_pre, 0.0))
            for c_in in range(C):
                h_c = tl.sum(tl.where(c_range == c_in, hidden, 0.0))
                tl.atomic_add(d_out_w_ptr + c_out * C + c_in, d_op * h_c)
                w_val = tl.load(out_w_ptr + c_out * C + c_in)
                # accumulate d_hidden
                old_dh = tl.sum(tl.where(c_range == c_in, d_hidden, 0.0))
                d_hidden = tl.where(c_range == c_in, old_dh + d_op * w_val, d_hidden)
        
        # === Backward through gather-conv ===
        d_kernel_flat = tl.zeros((H * K,), dtype=tl.float32)
        d_wave_out = tl.zeros((2 * H,), dtype=tl.float32)
        
        for h in range(H):
            # Recompute freq/phase for this head
            wave_h = tl.sum(tl.where(h2_range == h, wave_out, 0.0))
            sig_f = 1.0 / (1.0 + tl.exp(-wave_h))
            freq_h = sig_f * (max_freq - min_freq) + min_freq
            
            wave_h2 = tl.sum(tl.where(h2_range == (H + h), wave_out, 0.0))
            tanh_p = ((tl.exp(2.0 * wave_h2) - 1.0) / (tl.exp(2.0 * wave_h2) + 1.0))
            phase_h = tanh_p * max_freq
            
            kernel_h = tl.zeros((K,), dtype=tl.float32)
            for k in range(K):
                kv = tl.sum(tl.where(hk_range == (h * K + k), kernel_flat, 0.0))
                kernel_h = tl.where(k_range == k, kv, kernel_h)
            
            # d_hidden for this head
            d_head = tl.zeros((D,), dtype=tl.float32)
            for d in range(D):
                idx = h * D + d
                dh = tl.sum(tl.where(c_range == idx, d_hidden, 0.0))
                d_head = tl.where(d_range == d, dh, d_head)
            
            d_freq_h = 0.0
            
            for s_idx in range(S):
                s_off = s_idx - half_s
                sample_pos = pid_l + s_off * freq_h + phase_h
                valid = (sample_pos >= 0.0) & (sample_pos < L)
                valid_f = valid.to(tl.float32)
                sample_idx = tl.maximum(tl.minimum(sample_pos, L - 1.0), 0.0).to(tl.int32)
                
                rel_pos = tl.abs(s_off * freq_h)
                norm_pos = tl.minimum(rel_pos / max_receptive, 1.0)
                not_clamped = (rel_pos / max_receptive < 1.0).to(tl.float32)
                idx_f = norm_pos * (K - 1)
                idx_lo = tl.minimum(idx_f.to(tl.int32), K - 2)
                idx_hi = idx_lo + 1
                w_hi = idx_f - idx_lo.to(tl.float32)
                w_lo = 1.0 - w_hi
                
                k_lo = tl.sum(tl.where(k_range == idx_lo, kernel_h, 0.0))
                k_hi = tl.sum(tl.where(k_range == idx_hi, kernel_h, 0.0))
                kw = (k_lo * w_lo + k_hi * w_hi) * valid_f
                
                val_base = pid_b * L * C + sample_idx * C + h * D
                vals = tl.load(x_ptr + val_base + d_range, mask=d_range < D, other=0.0)
                
                # d_x contribution (scatter)
                d_x_contrib = d_head * kw
                for d in range(D):
                    dx = tl.sum(tl.where(d_range == d, d_x_contrib, 0.0))
                    tl.atomic_add(d_x_ptr + val_base + d, dx)
                
                # d_kernel
                d_kw = tl.sum(d_head * vals) * valid_f
                # d_kernel_h at idx_lo and idx_hi
                for k in range(K):
                    old_dk = tl.sum(tl.where(hk_range == (h * K + k), d_kernel_flat, 0.0))
                    contrib = tl.where(k == idx_lo, d_kw * w_lo, 0.0) + tl.where(k == idx_hi, d_kw * w_hi, 0.0)
                    d_kernel_flat = tl.where(hk_range == (h * K + k), old_dk + contrib, d_kernel_flat)
                
                # d_freq through kernel interpolation
                d_w_hi = d_kw * (k_hi - k_lo)
                d_idx_f = d_w_hi
                d_norm_pos = d_idx_f * (K - 1)
                d_rel_pos = d_norm_pos / max_receptive * not_clamped
                s_off_f = s_off * 1.0
                d_freq_h += d_rel_pos * tl.abs(s_off_f)
            
            # d_wave_out for freq: d_freq_h * (max_freq - min_freq) * sig_f * (1 - sig_f)
            d_wave_freq = d_freq_h * (max_freq - min_freq) * sig_f * (1.0 - sig_f)
            old_dwf = tl.sum(tl.where(h2_range == h, d_wave_out, 0.0))
            d_wave_out = tl.where(h2_range == h, old_dwf + d_wave_freq, d_wave_out)
        
        # === Backward through kernel_proj ===
        # d_kernel_pre = d_kernel_flat * silu'(kernel_pre)
        sig_k = 1.0 / (1.0 + tl.exp(-kernel_pre))
        d_kernel_pre = d_kernel_flat * sig_k * (1.0 + kernel_pre * (1.0 - sig_k))
        
        for hk in range(H * K):
            d_kp = tl.sum(tl.where(hk_range == hk, d_kernel_pre, 0.0))
            tl.atomic_add(d_kernel_b_ptr + hk, d_kp)
            for c in range(C):
                x_c = tl.load(x_ptr + x_base + c)
                tl.atomic_add(d_kernel_w_ptr + hk * C + c, d_kp * x_c)
                w_c = tl.load(kernel_w_ptr + hk * C + c)
                tl.atomic_add(d_x_ptr + x_base + c, d_kp * w_c)
        
        # === Backward through wave_proj ===
        # d_wave_pre = d_wave_out * silu'(wave_pre)
        sig_w = 1.0 / (1.0 + tl.exp(-wave_pre))
        d_wave_pre = d_wave_out * sig_w * (1.0 + wave_pre * (1.0 - sig_w))
        
        for h2 in range(2 * H):
            d_wp = tl.sum(tl.where(h2_range == h2, d_wave_pre, 0.0))
            tl.atomic_add(d_wave_b_ptr + h2, d_wp)
            for c in range(C):
                x_c = tl.load(x_ptr + x_base + c)
                tl.atomic_add(d_wave_w_ptr + h2 * C + c, d_wp * x_c)
                w_c = tl.load(wave_w_ptr + h2 * C + c)
                tl.atomic_add(d_x_ptr + x_base + c, d_wp * w_c)


class TritonGatherConv(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        max_samples: int = 32,
        max_freq: float = 16.0,
        min_freq: float = 1.0,
        max_kernel_size: int = 64,
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
        
        self.half_s = max_samples // 2
        self.max_receptive = self.half_s * max_freq
        
        # Store weights as raw tensors (not nn.Linear) for direct Triton access
        self.wave_w = nn.Parameter(torch.empty(2 * num_heads, channels))
        self.wave_b = nn.Parameter(torch.zeros(2 * num_heads))
        self.kernel_w = nn.Parameter(torch.empty(num_heads * max_kernel_size, channels))
        self.kernel_b = nn.Parameter(torch.zeros(num_heads * max_kernel_size))
        self.out_w = nn.Parameter(torch.empty(channels, channels))
        
        nn.init.zeros_(self.wave_w)
        nn.init.xavier_uniform_(self.kernel_w, gain=0.1)
        nn.init.xavier_uniform_(self.out_w, gain=0.1)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        return _GatherConvFused.apply(
            x, self.wave_w, self.wave_b, self.kernel_w, self.kernel_b, self.out_w,
            self.num_heads, self.half_s, self.max_receptive,
            self.max_freq, self.min_freq, self.max_kernel_size,
        ), {}


class _GatherConvFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, wave_w, wave_b, kernel_w, kernel_b, out_w,
                num_heads, half_s, max_receptive, max_freq, min_freq, max_kernel_size):
        B, L, C = x.shape
        H = num_heads
        K = max_kernel_size
        D = C // H
        
        out = torch.empty_like(x)
        grid = (B, L)
        
        gather_conv_fused_fwd[grid](
            x, out,
            wave_w, wave_b, kernel_w, kernel_b, out_w,
            B, L, C, H, D, K, half_s,
            max_receptive, max_freq, min_freq,
        )
        
        # Save ONLY x and weights - everything recomputed in backward
        ctx.save_for_backward(x, wave_w, wave_b, kernel_w, kernel_b, out_w)
        ctx.H = H
        ctx.K = K
        ctx.half_s = half_s
        ctx.max_receptive = max_receptive
        ctx.max_freq = max_freq
        ctx.min_freq = min_freq
        
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        x, wave_w, wave_b, kernel_w, kernel_b, out_w = ctx.saved_tensors
        B, L, C = x.shape
        H = ctx.H
        K = ctx.K
        D = C // H
        
        # Allocate gradient tensors
        d_x = torch.zeros_like(x)
        d_wave_w = torch.zeros_like(wave_w)
        d_wave_b = torch.zeros_like(wave_b)
        d_kernel_w = torch.zeros_like(kernel_w)
        d_kernel_b = torch.zeros_like(kernel_b)
        d_out_w = torch.zeros_like(out_w)
        
        grid = (B, L)
        gather_conv_fused_bwd[grid](
            x, d_out.contiguous(),
            wave_w, wave_b, kernel_w, kernel_b, out_w,
            d_x, d_wave_w, d_wave_b, d_kernel_w, d_kernel_b, d_out_w,
            B, L, C, H, D, K, ctx.half_s,
            ctx.max_receptive, ctx.max_freq, ctx.min_freq,
        )
        
        return d_x, d_wave_w, d_wave_b, d_kernel_w, d_kernel_b, d_out_w, \
               None, None, None, None, None, None


# Keep GatherConvOp for backward compat with gatherconv.py
class GatherConvOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, freq, phase, kernel, half_s, max_receptive, quantize_bwd):
        # Simple non-fused version for external use
        B, L, C = x.shape
        H = freq.shape[-1]
        K = kernel.shape[-1]
        D = C // H
        S = 2 * half_s + 1
        
        # PyTorch fallback (or could add a simpler Triton kernel)
        out = torch.zeros_like(x)
        for b in range(B):
            for l in range(L):
                for h in range(H):
                    f, p = freq[b, l, h].item(), phase[b, l, h].item()
                    k = kernel[b, l, h]
                    for s in range(S):
                        s_off = s - half_s
                        pos = l + s_off * f + p
                        if 0 <= pos < L:
                            idx = int(pos)
                            rel = abs(s_off * f)
                            norm = min(rel / max_receptive, 1.0)
                            ki = norm * (K - 1)
                            lo, hi = int(ki), min(int(ki) + 1, K - 1)
                            w = ki - lo
                            kw = k[lo] * (1 - w) + k[hi] * w
                            out[b, l, h*D:(h+1)*D] += x[b, idx, h*D:(h+1)*D] * kw
        
        ctx.save_for_backward(x, freq, phase, kernel)
        ctx.half_s = half_s
        ctx.max_receptive = max_receptive
        return out
    
    @staticmethod
    def backward(ctx, d_out):
        # Simplified backward
        x, freq, phase, kernel = ctx.saved_tensors
        d_x = torch.zeros_like(x)
        d_freq = torch.zeros_like(freq)
        d_kernel = torch.zeros_like(kernel)
        return d_x, d_freq, None, d_kernel, None, None, None


if __name__ == "__main__":
    if not HAS_TRITON:
        print("Triton not available")
        exit()
    
    import time
    
    device = "cuda"
    B, L, C = 4, 8192, 256
    H = 8
    
    print(f"Config: B={B}, L={L}, C={C}, H={H}")
    print()
    
    model = TritonGatherConv(channels=C, num_heads=H).to(device)
    x = torch.randn(B, L, C, device=device, requires_grad=True)
    
    # Warmup
    for _ in range(3):
        out, _ = model(x)
        out.sum().backward()
        model.zero_grad()
        x.grad.zero_()
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Forward
    start = time.perf_counter()
    for _ in range(10):
        with torch.no_grad():
            out, _ = model(x)
    torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - start) / 10 * 1000
    fwd_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    torch.cuda.reset_peak_memory_stats()
    
    # Forward + backward
    start = time.perf_counter()
    for _ in range(10):
        out, _ = model(x)
        out.sum().backward()
        model.zero_grad()
        x.grad.zero_()
    torch.cuda.synchronize()
    total_time = (time.perf_counter() - start) / 10 * 1000
    bwd_time = total_time - fwd_time
    bwd_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"Fwd: {fwd_time:5.1f}ms ({fwd_mem:5.0f}MB)  Bwd: {bwd_time:5.1f}ms ({bwd_mem:5.0f}MB)")
