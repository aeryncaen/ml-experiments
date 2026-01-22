"""
Triton kernels for fused scatter attention.

Avoids materializing the B×L×S×C values tensor by fusing:
- sample position computation
- gather from x_flat
- attention score computation  
- softmax over samples
- weighted sum to output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _scatter_attn_fwd_kernel(
        x_ptr,              # [B, L, C]
        q_ptr,              # [B, L, H, pos_dim]
        freq_ptr,           # [B, L, H]
        phase_ptr,          # [B, L, H]
        decay_ptr,          # [B, L, H]
        key_weight_ptr,     # [pos_dim]
        stride_grid_ptr,    # [S]
        out_ptr,            # [B, L, H, D]
        B: tl.constexpr,
        L: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        S: tl.constexpr,
        pos_dim: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        if (pid_b >= B) | (pid_l >= L) | (pid_h >= H):
            return
        
        freq_offset = pid_b * L * H + pid_l * H + pid_h
        freq = tl.load(freq_ptr + freq_offset)
        phase = tl.load(phase_ptr + freq_offset)
        decay = tl.load(decay_ptr + freq_offset)
        
        q_base = pid_b * L * H * pos_dim + pid_l * H * pos_dim + pid_h * pos_dim
        q = tl.load(q_ptr + q_base + tl.arange(0, pos_dim))
        
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        
        head_start = pid_h * D
        
        max_score = -1e9
        sum_exp = 1e-8
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        
        for s in range(S):
            offset = tl.load(stride_grid_ptr + s)
            sample_pos = pid_l + offset * freq + phase
            sample_idx = tl.minimum(tl.maximum(sample_pos.to(tl.int32), 0), L - 1)
            valid = (sample_pos >= 0) & (sample_pos < L)
            
            x_base = pid_b * L * C + sample_idx * C + head_start
            v = tl.load(x_ptr + x_base + d_offsets, mask=d_mask, other=0.0)
            
            rel_dist = tl.abs(offset * freq)
            
            key = tl.load(key_weight_ptr + tl.arange(0, pos_dim)) * rel_dist
            score = tl.sum(q * key) * scale
            
            decay_factor = tl.exp(-rel_dist / tl.maximum(decay, 0.1))
            
            score = tl.where(valid, score, -1e9)
            
            new_max = tl.maximum(max_score, score)
            exp_old = tl.exp(max_score - new_max)
            exp_new = tl.exp(score - new_max)
            
            acc = acc * exp_old + v * exp_new * decay_factor
            sum_exp = sum_exp * exp_old + exp_new * decay_factor
            max_score = new_max
        
        out = acc / sum_exp
        
        out_base = pid_b * L * H * D + pid_l * H * D + pid_h * D
        tl.store(out_ptr + out_base + d_offsets, out, mask=d_mask)


    @triton.jit
    def _scatter_attn_bwd_kernel(
        grad_out_ptr,       # [B, L, H, D]
        x_ptr,              # [B, L, C]
        q_ptr,              # [B, L, H, pos_dim]
        freq_ptr,           # [B, L, H]
        phase_ptr,          # [B, L, H]
        decay_ptr,          # [B, L, H]
        key_weight_ptr,     # [pos_dim]
        stride_grid_ptr,    # [S]
        attn_ptr,           # [B, L, H, S] - saved from forward
        grad_x_ptr,         # [B, L, C]
        grad_q_ptr,         # [B, L, H, pos_dim]
        grad_freq_ptr,      # [B, L, H]
        grad_phase_ptr,     # [B, L, H]
        grad_decay_ptr,     # [B, L, H]
        B: tl.constexpr,
        L: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        S: tl.constexpr,
        pos_dim: tl.constexpr,
        scale: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        if (pid_b >= B) | (pid_l >= L) | (pid_h >= H):
            return
        
        freq_offset = pid_b * L * H + pid_l * H + pid_h
        freq = tl.load(freq_ptr + freq_offset)
        phase = tl.load(phase_ptr + freq_offset)
        decay = tl.load(decay_ptr + freq_offset)
        
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        head_start = pid_h * D
        
        grad_out_base = pid_b * L * H * D + pid_l * H * D + pid_h * D
        grad_out = tl.load(grad_out_ptr + grad_out_base + d_offsets, mask=d_mask, other=0.0)
        
        for s in range(S):
            offset = tl.load(stride_grid_ptr + s)
            sample_pos = pid_l + offset * freq + phase
            sample_idx = tl.minimum(tl.maximum(sample_pos.to(tl.int32), 0), L - 1)
            valid = (sample_pos >= 0) & (sample_pos < L)
            
            attn_base = pid_b * L * H * S + pid_l * H * S + pid_h * S + s
            attn_weight = tl.load(attn_ptr + attn_base)
            attn_weight = tl.where(valid, attn_weight, 0.0)
            
            grad_v = grad_out * attn_weight
            
            x_base = pid_b * L * C + sample_idx * C + head_start
            tl.atomic_add(grad_x_ptr + x_base + d_offsets, grad_v, mask=d_mask & valid)


class ScatterAttentionFunc(Function):
    @staticmethod
    def forward(ctx, x, q, freq, phase, decay, key_weight, stride_grid, scale):
        B, L, C = x.shape
        _, _, H, pos_dim = q.shape
        S = stride_grid.shape[0]
        D = C // H
        
        out = torch.empty(B, L, H, D, device=x.device, dtype=x.dtype)
        
        BLOCK_D = triton.next_power_of_2(D)
        
        grid = (B, L, H)
        _scatter_attn_fwd_kernel[grid](
            x, q, freq, phase, decay, key_weight, stride_grid, out,
            B, L, C, H, D, S, pos_dim, scale,
            BLOCK_D=BLOCK_D,
        )
        
        ctx.save_for_backward(x, q, freq, phase, decay, key_weight, stride_grid)
        ctx.scale = scale
        ctx.shapes = (B, L, C, H, D, S, pos_dim)
        
        return out.reshape(B, L, C)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, q, freq, phase, decay, key_weight, stride_grid = ctx.saved_tensors
        B, L, C, H, D, S, pos_dim = ctx.shapes
        scale = ctx.scale
        
        grad_output = grad_output.reshape(B, L, H, D).contiguous()
        
        centers = torch.arange(L, device=x.device, dtype=x.dtype)
        sample_pos = centers.view(1, L, 1, 1) + stride_grid.view(1, 1, S, 1) * freq.view(B, L, 1, H) + phase.view(B, L, 1, H)
        sample_pos = sample_pos[..., 0]
        valid_mask = (sample_pos >= 0) & (sample_pos < L)
        sample_idx = sample_pos.long().clamp(0, L - 1)
        
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, L, S)
        values = x[batch_idx, sample_idx].reshape(B, L, S, H, D).permute(0, 1, 3, 2, 4)
        
        rel_dist = stride_grid.abs().view(1, 1, 1, S) * freq.view(B, L, H, 1)
        keys = key_weight.view(1, 1, 1, 1, pos_dim) * rel_dist.unsqueeze(-1)
        scores = torch.einsum('blhd,blhsd->blhs', q, keys) * scale
        decay_env = torch.exp(-rel_dist / decay.view(B, L, H, 1).clamp(min=0.1))
        
        valid_mask_exp = valid_mask.unsqueeze(2).expand(B, L, H, S)
        scores = scores.masked_fill(~valid_mask_exp, float('-inf'))
        attn = F.softmax(scores, dim=-1) * decay_env * valid_mask_exp.float()
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        grad_values = torch.einsum('blhd,blhs->blhsd', grad_output.reshape(B, L, H, D), attn)
        grad_values = grad_values.permute(0, 1, 3, 2, 4).reshape(B, L, S, C)
        
        grad_x = torch.zeros_like(x)
        for s in range(S):
            idx = sample_idx[:, :, s]
            grad_x.scatter_add_(1, idx.unsqueeze(-1).expand(-1, -1, C), grad_values[:, :, s])
        
        return grad_x, None, None, None, None, None, None, None


def triton_scatter_attention(x, q, freq, phase, decay, key_weight, stride_grid, scale):
    if HAS_TRITON and x.is_cuda:
        return ScatterAttentionFunc.apply(x, q, freq, phase, decay, key_weight, stride_grid, scale)
    else:
        raise RuntimeError("Triton scatter attention requires CUDA and triton")


if HAS_TRITON:
    @triton.jit
    def _local_window_attn_fwd_kernel(
        q_ptr,              # [B, L, H, D]
        k_ptr,              # [B, L + K - 1, C] (padded)
        v_ptr,              # [B, L + K - 1, C] (padded)
        width_ptr,          # [B, L, H]
        sharpness_ptr,      # [B, L, H]
        rel_dist_ptr,       # [K]
        out_ptr,            # [B, L, H, D]
        B: tl.constexpr,
        L: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        K: tl.constexpr,
        scale: tl.constexpr,
        half_k: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        if (pid_b >= B) | (pid_l >= L) | (pid_h >= H):
            return
        
        param_offset = pid_b * L * H + pid_l * H + pid_h
        width = tl.load(width_ptr + param_offset)
        sharpness = tl.load(sharpness_ptr + param_offset)
        
        q_base = pid_b * L * H * D + pid_l * H * D + pid_h * D
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        q = tl.load(q_ptr + q_base + d_offsets, mask=d_mask, other=0.0)
        
        head_start = pid_h * D
        k_row_start = pid_b * (L + K - 1) * C + pid_l * C + head_start
        v_row_start = pid_b * (L + K - 1) * C + pid_l * C + head_start
        
        max_score = -float('inf')
        sum_exp = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        
        for w in range(K):
            rel_d = tl.load(rel_dist_ptr + w)
            
            k_offset = k_row_start + w * C
            k_vec = tl.load(k_ptr + k_offset + d_offsets, mask=d_mask, other=0.0)
            
            score = tl.sum(q * k_vec) * scale
            
            soft_mask = tl.sigmoid((width - rel_d) * sharpness)
            score = score - (1.0 - soft_mask) * 1e4
            
            v_offset = v_row_start + w * C
            v_vec = tl.load(v_ptr + v_offset + d_offsets, mask=d_mask, other=0.0)
            
            new_max = tl.maximum(max_score, score)
            exp_old = tl.exp(max_score - new_max)
            exp_new = tl.exp(score - new_max)
            
            acc = acc * exp_old + v_vec * exp_new
            sum_exp = sum_exp * exp_old + exp_new
            max_score = new_max
        
        out = acc / (sum_exp + 1e-8)
        
        out_base = pid_b * L * H * D + pid_l * H * D + pid_h * D
        tl.store(out_ptr + out_base + d_offsets, out, mask=d_mask)


    @triton.jit
    def _local_window_attn_fwd_kernel_with_attn(
        q_ptr,              # [B, L, H, D]
        k_ptr,              # [B, L + K - 1, C] (padded)
        v_ptr,              # [B, L + K - 1, C] (padded)
        width_ptr,          # [B, L, H]
        sharpness_ptr,      # [B, L, H]
        rel_dist_ptr,       # [K]
        out_ptr,            # [B, L, H, D]
        attn_ptr,           # [B, L, H, K] - save attention weights for backward
        B: tl.constexpr,
        L: tl.constexpr,
        C: tl.constexpr,
        H: tl.constexpr,
        D: tl.constexpr,
        K: tl.constexpr,
        scale: tl.constexpr,
        half_k: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Forward kernel that also saves attention weights for backward pass."""
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_h = tl.program_id(2)
        
        if (pid_b >= B) | (pid_l >= L) | (pid_h >= H):
            return
        
        param_offset = pid_b * L * H + pid_l * H + pid_h
        width = tl.load(width_ptr + param_offset)
        sharpness = tl.load(sharpness_ptr + param_offset)
        
        q_base = pid_b * L * H * D + pid_l * H * D + pid_h * D
        d_offsets = tl.arange(0, BLOCK_D)
        d_mask = d_offsets < D
        q = tl.load(q_ptr + q_base + d_offsets, mask=d_mask, other=0.0)
        
        head_start = pid_h * D
        k_row_start = pid_b * (L + K - 1) * C + pid_l * C + head_start
        v_row_start = pid_b * (L + K - 1) * C + pid_l * C + head_start
        
        # First pass: compute max score for numerical stability
        max_score = -1e9
        for w in range(K):
            rel_d = tl.load(rel_dist_ptr + w)
            k_offset = k_row_start + w * C
            k_vec = tl.load(k_ptr + k_offset + d_offsets, mask=d_mask, other=0.0)
            score = tl.sum(q * k_vec) * scale
            soft_mask = tl.sigmoid((width - rel_d) * sharpness)
            score = score - (1.0 - soft_mask) * 1e4
            max_score = tl.maximum(max_score, score)
        
        # Second pass: compute softmax and weighted sum, save attention
        sum_exp = 0.0
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        attn_base = pid_b * L * H * K + pid_l * H * K + pid_h * K
        
        for w in range(K):
            rel_d = tl.load(rel_dist_ptr + w)
            k_offset = k_row_start + w * C
            k_vec = tl.load(k_ptr + k_offset + d_offsets, mask=d_mask, other=0.0)
            score = tl.sum(q * k_vec) * scale
            soft_mask = tl.sigmoid((width - rel_d) * sharpness)
            score = score - (1.0 - soft_mask) * 1e4
            
            exp_score = tl.exp(score - max_score)
            sum_exp += exp_score
            
            v_offset = v_row_start + w * C
            v_vec = tl.load(v_ptr + v_offset + d_offsets, mask=d_mask, other=0.0)
            acc += v_vec * exp_score
        
        # Normalize and save
        attn_norm = 1.0 / (sum_exp + 1e-8)
        out = acc * attn_norm
        
        out_base = pid_b * L * H * D + pid_l * H * D + pid_h * D
        tl.store(out_ptr + out_base + d_offsets, out, mask=d_mask)
        
        # Save normalized attention weights for backward
        for w in range(K):
            rel_d = tl.load(rel_dist_ptr + w)
            k_offset = k_row_start + w * C
            k_vec = tl.load(k_ptr + k_offset + d_offsets, mask=d_mask, other=0.0)
            score = tl.sum(q * k_vec) * scale
            soft_mask = tl.sigmoid((width - rel_d) * sharpness)
            score = score - (1.0 - soft_mask) * 1e4
            attn_w = tl.exp(score - max_score) * attn_norm
            tl.store(attn_ptr + attn_base + w, attn_w)


    @triton.jit
    def _ssm_fused_step_kernel(
        H_ptr,              # [B, R, L, N] - state (in/out)
        B_rot_ptr,          # [B, R, L, N] - rotated input projection
        X_r_ptr,            # [B, R, L] - input scale per rank
        decay_ptr,          # [B, L, N] - decay factors
        out_ptr,            # [B, L, N * R] - output
        B_dim: tl.constexpr,
        R: tl.constexpr,
        L_dim: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        
        if (pid_b >= B_dim) | (pid_l >= L_dim):
            return
        
        n_offsets = tl.arange(0, BLOCK_N)
        n_mask = n_offsets < N
        
        decay_base = pid_b * L_dim * N + pid_l * N
        decay = tl.load(decay_ptr + decay_base + n_offsets, mask=n_mask, other=0.0)
        
        out_base = pid_b * L_dim * N * R + pid_l * N * R
        
        for r in range(R):
            H_base = pid_b * R * L_dim * N + r * L_dim * N + pid_l * N
            H_val = tl.load(H_ptr + H_base + n_offsets, mask=n_mask, other=0.0)
            
            B_rot_base = pid_b * R * L_dim * N + r * L_dim * N + pid_l * N
            B_rot = tl.load(B_rot_ptr + B_rot_base + n_offsets, mask=n_mask, other=0.0)
            
            X_r_idx = pid_b * R * L_dim + r * L_dim + pid_l
            X_r = tl.load(X_r_ptr + X_r_idx)
            
            inject = B_rot * X_r
            
            H_new = decay * H_val + inject
            
            tl.store(H_ptr + H_base + n_offsets, H_new, mask=n_mask)
            
            out_offset = out_base + r * N
            tl.store(out_ptr + out_offset + n_offsets, H_new, mask=n_mask)


    @triton.jit  
    def _rope_kernel(
        x_ptr,              # [B, R, L, N] - input/output
        theta_ptr,          # [B, L, N//2] - angles
        layer_idx: tl.constexpr,
        B_dim: tl.constexpr,
        R: tl.constexpr,
        L_dim: tl.constexpr,
        N: tl.constexpr,
        BLOCK_N2: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_r = tl.program_id(1)
        pid_l = tl.program_id(2)
        
        if (pid_b >= B_dim) | (pid_r >= R) | (pid_l >= L_dim):
            return
        
        N2 = N // 2
        n2_offsets = tl.arange(0, BLOCK_N2)
        n2_mask = n2_offsets < N2
        
        theta_base = pid_b * L_dim * N2 + pid_l * N2
        theta = tl.load(theta_ptr + theta_base + n2_offsets, mask=n2_mask, other=0.0)
        theta = theta * (layer_idx + 1)
        
        cos_t = tl.cos(theta)
        sin_t = tl.sin(theta)
        
        x_base = pid_b * R * L_dim * N + pid_r * L_dim * N + pid_l * N
        
        x1 = tl.load(x_ptr + x_base + n2_offsets * 2, mask=n2_mask, other=0.0)
        x2 = tl.load(x_ptr + x_base + n2_offsets * 2 + 1, mask=n2_mask, other=0.0)
        
        out1 = x1 * cos_t - x2 * sin_t
        out2 = x1 * sin_t + x2 * cos_t
        
        tl.store(x_ptr + x_base + n2_offsets * 2, out1, mask=n2_mask)
        tl.store(x_ptr + x_base + n2_offsets * 2 + 1, out2, mask=n2_mask)


class LocalWindowAttnFunc(Function):
    """Autograd function for local window attention with Triton forward."""
    
    @staticmethod
    def forward(ctx, q, k_padded, v_padded, width, sharpness, rel_dist, scale, K):
        B, L, H, D = q.shape
        C = k_padded.shape[-1]
        
        out = torch.empty(B, L, H, D, device=q.device, dtype=q.dtype)
        attn = torch.empty(B, L, H, K, device=q.device, dtype=q.dtype)
        
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B, L, H)
        
        _local_window_attn_fwd_kernel_with_attn[grid](
            q, k_padded, v_padded, width, sharpness, rel_dist, out, attn,
            B, L, C, H, D, K, scale, K // 2,
            BLOCK_D=BLOCK_D,
        )
        
        ctx.save_for_backward(q, k_padded, v_padded, attn, rel_dist)
        ctx.scale = scale
        ctx.shapes = (B, L, C, H, D, K)
        
        return out
    
    @staticmethod
    def backward(ctx, grad_output):
        q, k_padded, v_padded, attn, rel_dist = ctx.saved_tensors
        B, L, C, H, D, K = ctx.shapes
        scale = ctx.scale
        
        grad_output = grad_output.contiguous()
        
        # Unfold k and v to get windows: [B, L, K, C] -> [B, L, K, H, D]
        # k_padded is [B, L+K-1, C], we need windows of size K for each position
        k_win = k_padded.unfold(1, K, 1)  # [B, L, C, K]
        v_win = v_padded.unfold(1, K, 1)  # [B, L, C, K]
        
        k_win = k_win.permute(0, 1, 3, 2).reshape(B, L, K, H, D).permute(0, 1, 3, 2, 4)  # [B, L, H, K, D]
        v_win = v_win.permute(0, 1, 3, 2).reshape(B, L, K, H, D).permute(0, 1, 3, 2, 4)  # [B, L, H, K, D]
        
        # grad_v: [B, L, H, K, D] = attn[B, L, H, K] * grad_output[B, L, H, D]
        grad_v_win = torch.einsum('blhk,blhd->blhkd', attn, grad_output)
        
        # grad_attn: [B, L, H, K] = grad_output[B, L, H, D] · v_win[B, L, H, K, D]
        grad_attn = torch.einsum('blhd,blhkd->blhk', grad_output, v_win)
        
        # Softmax backward: grad_scores = attn * (grad_attn - (grad_attn * attn).sum(-1, keepdim=True))
        grad_scores = attn * (grad_attn - (grad_attn * attn).sum(-1, keepdim=True))
        grad_scores = grad_scores * scale
        
        # grad_q: [B, L, H, D] = grad_scores[B, L, H, K] · k_win[B, L, H, K, D]
        grad_q = torch.einsum('blhk,blhkd->blhd', grad_scores, k_win)
        
        # grad_k_win: [B, L, H, K, D] = grad_scores[B, L, H, K] * q[B, L, H, D]
        grad_k_win = torch.einsum('blhk,blhd->blhkd', grad_scores, q)
        
        # Fold gradients back to padded tensors
        # grad_k_win and grad_v_win are [B, L, H, K, D], need to scatter back to [B, L+K-1, C]
        grad_k_win = grad_k_win.permute(0, 1, 3, 2, 4).reshape(B, L, K, C)  # [B, L, K, C]
        grad_v_win = grad_v_win.permute(0, 1, 3, 2, 4).reshape(B, L, K, C)  # [B, L, K, C]
        
        grad_k_padded = torch.zeros_like(k_padded)
        grad_v_padded = torch.zeros_like(v_padded)
        
        # Scatter gradients back - each window position w at query position l contributes to position l+w
        for w in range(K):
            grad_k_padded[:, w:w+L, :] += grad_k_win[:, :, w, :]
            grad_v_padded[:, w:w+L, :] += grad_v_win[:, :, w, :]
        
        # We don't backprop through width/sharpness for now (would need soft_mask backward)
        return grad_q, grad_k_padded, grad_v_padded, None, None, None, None, None


def triton_local_window_attn(q, k_padded, v_padded, width, sharpness, rel_dist, scale, K):
    """Dispatch to Triton or raise error."""
    if HAS_TRITON and q.is_cuda:
        return LocalWindowAttnFunc.apply(q, k_padded, v_padded, width, sharpness, rel_dist, scale, K)
    else:
        raise RuntimeError("Triton local window attention requires CUDA and triton")


class TritonLocalWindowAttn(nn.Module):
    """Fused local window attention using Triton (1D only)."""
    
    rel_dist: torch.Tensor
    
    def __init__(self, embed_dim: int, kernel_size: int = 17, num_channels: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.num_channels = num_channels
        self.channel_dim = embed_dim // num_channels
        self.half_k = kernel_size // 2
        self.scale = self.channel_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.channel_dim)
        self.k_norm = nn.LayerNorm(self.channel_dim)
        self.window_proj = nn.Linear(embed_dim, 2 * num_channels)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        
        rel_dist = torch.abs(torch.arange(kernel_size).float() - kernel_size // 2)
        self.register_buffer('rel_dist', rel_dist)
        self.max_dist = rel_dist.max().item()
        
        nn.init.zeros_(self.window_proj.weight)
        nn.init.zeros_(self.window_proj.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H = self.num_channels
        D = self.channel_dim
        K = self.kernel_size
        
        q = F.silu(self.q_proj(x)).reshape(B, L, H, D)
        q = self.q_norm(q)
        
        kv = F.silu(self.kv_proj(x))
        k, v = kv.chunk(2, dim=-1)
        k = self.k_norm(k.reshape(B, L, H, D)).reshape(B, L, C)
        
        window_params = F.silu(self.window_proj(x))
        width_raw, sharpness_raw = window_params.chunk(2, dim=-1)
        width = width_raw.sigmoid() * self.max_dist + 0.5
        sharpness = sharpness_raw.sigmoid() * 9.5 + 0.5
        
        pad_left = (K - 1) // 2
        pad_right = K // 2
        k_padded = F.pad(k, (0, 0, pad_left, pad_right))
        v_padded = F.pad(v, (0, 0, pad_left, pad_right))
        
        if HAS_TRITON and x.is_cuda:
            # Use autograd-wrapped Triton kernel for training support
            out = LocalWindowAttnFunc.apply(
                q, k_padded, v_padded, width, sharpness, self.rel_dist, self.scale, K
            )
            out = out.reshape(B, L, C)
        else:
            out = self._pytorch_fallback(q, k_padded, v_padded, width, sharpness, B, L, H, D, K)
        
        return F.silu(self.out(out))
    
    def _pytorch_fallback(self, q, k_padded, v_padded, width, sharpness, B, L, H, D, K):
        k_win = k_padded.unfold(1, K, 1)
        v_win = v_padded.unfold(1, K, 1)
        
        k_win = k_win.permute(0, 1, 3, 2).reshape(B, L, K, H, D).permute(0, 1, 3, 4, 2)
        v_win = v_win.permute(0, 1, 3, 2).reshape(B, L, K, H, D).permute(0, 1, 3, 4, 2)
        
        width = width.reshape(B, L, H, 1)
        sharpness = sharpness.reshape(B, L, H, 1)
        
        scores = torch.einsum('blhd,blhdw->blhw', q, k_win) * self.scale
        soft_mask = torch.sigmoid((width - self.rel_dist) * sharpness)
        scores = scores - (1 - soft_mask) * 1e4
        
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('blhw,blhdw->blhd', attn, v_win).reshape(B, L, -1)
        return out


class TritonSSMStep(nn.Module):
    """Fused SSM step operations using Triton."""
    
    def __init__(self, dim: int, state_dim: int = 64, mimo_rank: int = 4):
        super().__init__()
        self.D = dim
        self.N = state_dim
        self.R = mimo_rank
        
        self.to_B = nn.Linear(dim, state_dim * mimo_rank)
        self.to_X = nn.Linear(dim, mimo_rank)
        self.to_decay = nn.Linear(dim, state_dim)
        self.to_theta = nn.Linear(dim, state_dim // 2)
        self.out_proj = nn.Linear(state_dim * mimo_rank, dim)
    
    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return torch.zeros(B, self.R, L, self.N, device=x.device, dtype=x.dtype)
    
    def forward(self, x: torch.Tensor, H: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        
        B_proj = F.silu(self.to_B(x)).view(B, L, self.N, self.R).permute(0, 3, 1, 2).contiguous()
        X_r = F.silu(self.to_X(x)).permute(0, 2, 1).contiguous()
        decay = torch.sigmoid(self.to_decay(x))
        theta = self.to_theta(x)
        
        if HAS_TRITON and x.is_cuda:
            BLOCK_N2 = triton.next_power_of_2(self.N // 2)
            grid_rope = (B, self.R, L)
            _rope_kernel[grid_rope](
                B_proj, theta, layer_idx,
                B, self.R, L, self.N, BLOCK_N2,
            )
            
            out_flat = torch.empty(B, L, self.N * self.R, device=x.device, dtype=x.dtype)
            
            BLOCK_N = triton.next_power_of_2(self.N)
            grid_ssm = (B, L)
            _ssm_fused_step_kernel[grid_ssm](
                H, B_proj, X_r, decay, out_flat,
                B, self.R, L, self.N, BLOCK_N,
            )
        else:
            B_rot = self._apply_rope_pytorch(B_proj, theta, layer_idx)
            inject = B_rot * X_r.unsqueeze(-1)
            H = decay.unsqueeze(1) * H + inject
            out_flat = H.permute(0, 2, 1, 3).reshape(B, L, self.N * self.R)
        
        out = F.silu(self.out_proj(out_flat))
        return H, out
    
    def _apply_rope_pytorch(self, x: torch.Tensor, theta: torch.Tensor, layer_idx: int) -> torch.Tensor:
        theta_k = theta * (layer_idx + 1)
        theta_k = theta_k.unsqueeze(1)
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos = theta_k.cos()
        sin = theta_k.sin()
        out = torch.empty_like(x)
        out[..., ::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class TritonScatterConv(nn.Module):
    """
    Drop-in replacement for AdaptiveConvND (1D only) using fused Triton kernel.
    """
    
    stride_grid: torch.Tensor
    
    def __init__(
        self,
        channels: int,
        max_samples: int = 32,
        num_channels: int = 1,
        max_freq: float = 16.0,
        min_freq: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.num_channels = num_channels
        self.channel_dim = channels // num_channels
        self.max_freq = max_freq
        self.min_freq = min_freq
        
        H = num_channels
        pos_dim = 16
        self.pos_dim = pos_dim
        self.scale = pos_dim ** -0.5
        
        self.wave_proj = nn.Linear(channels, 3 * H)
        self.query_proj = nn.Linear(channels, H * pos_dim)
        self.key_weight = nn.Parameter(torch.randn(pos_dim) * 0.02)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        
        self.se_fc1 = nn.Linear(channels, channels // 4)
        self.se_fc2 = nn.Linear(channels // 4, channels)
        
        half_s = max_samples // 2
        stride_grid = torch.arange(-half_s, half_s + 1).float()
        self.register_buffer('stride_grid', stride_grid)
        self.num_samples = stride_grid.shape[0]
        
        nn.init.zeros_(self.wave_proj.weight)
        nn.init.zeros_(self.wave_proj.bias)
        nn.init.zeros_(self.out_proj.weight)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, L, C = x.shape
        H = self.num_channels
        
        wave_params = F.silu(self.wave_proj(x)).reshape(B, L, 3, H).permute(0, 1, 3, 2)
        queries = F.silu(self.query_proj(x)).reshape(B, L, H, self.pos_dim)
        
        freq = torch.sigmoid(wave_params[..., 0]) * (self.max_freq - self.min_freq) + self.min_freq
        phase = torch.tanh(wave_params[..., 1]) * self.max_freq
        decay = torch.sigmoid(wave_params[..., 2]) * 9.5 + 0.5
        
        freq_avg = freq.mean(dim=2)
        phase_avg = phase.mean(dim=2)
        decay_avg = decay.mean(dim=2)
        
        if HAS_TRITON and x.is_cuda:
            out = triton_scatter_attention(
                x, queries,
                freq_avg.unsqueeze(-1).expand(-1, -1, H),
                phase_avg.unsqueeze(-1).expand(-1, -1, H),
                decay_avg.unsqueeze(-1).expand(-1, -1, H),
                self.key_weight, self.stride_grid, self.scale
            )
        else:
            out = self._pytorch_fallback(x, queries, freq_avg, phase_avg, decay_avg)
        
        se_weights = torch.sigmoid(self.se_fc2(F.silu(self.se_fc1(out))))
        out = out * se_weights
        out = F.silu(self.out_proj(out))
        
        return out, {}
    
    def _pytorch_fallback(self, x, queries, freq_avg, phase_avg, decay_avg):
        B, L, C = x.shape
        H = self.num_channels
        D = self.channel_dim
        S = self.num_samples
        
        centers = torch.arange(L, device=x.device, dtype=x.dtype)
        sample_pos = centers.view(1, L, 1) + self.stride_grid.view(1, 1, S) * freq_avg.unsqueeze(-1) + phase_avg.unsqueeze(-1)
        valid_mask = (sample_pos >= 0) & (sample_pos < L)
        sample_idx = sample_pos.long().clamp(0, L - 1)
        
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, L, S)
        values = x[batch_idx, sample_idx].reshape(B, L, S, H, D).permute(0, 1, 3, 2, 4)
        
        rel_dist = (self.stride_grid.view(1, 1, 1, S).abs() * freq_avg.view(B, L, 1, 1).expand(-1, -1, H, -1))
        keys = self.key_weight.view(1, 1, 1, 1, -1) * rel_dist.unsqueeze(-1)
        
        scores = torch.einsum('blhd,blhsd->blhs', queries, keys) * self.scale
        
        decay_envelope = torch.exp(-rel_dist / decay_avg.view(B, L, 1, 1).expand(-1, -1, H, S).clamp(min=0.1))
        
        valid_mask = valid_mask.unsqueeze(2).expand(B, L, H, S)
        scores = scores.masked_fill(~valid_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1) * decay_envelope * valid_mask.float()
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        
        out = torch.einsum('blhsd,blhs->blhd', values, attn).reshape(B, L, C)
        return out
