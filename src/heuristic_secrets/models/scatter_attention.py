"""
Scatter Attention: learned position warp + power law envelope + bilinear splat.
Dual of gather-attention: scatter weights to positions, average collisions, apply.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from functools import reduce
from operator import mul

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .backbone import AdaptiveDeformConv1d


class KernelNetND(nn.Module):
    def __init__(self, ndim: int, out_channels: int, hidden: int = 32, layers: int = 3, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        net = [nn.Linear(ndim, hidden)]
        for _ in range(layers - 1):
            net.extend([nn.SiLU(), nn.Linear(hidden, hidden)])
        net.extend([nn.SiLU(), nn.Linear(hidden, out_channels)])
        self.net = nn.Sequential(*net)
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.net(positions * self.omega_0)


class AdaptiveConvND(nn.Module):
    """
    Learned stride convolution: samples full sequence at learned frequency intervals.
    
    For each position, samples at: center, center±freq, center±2*freq, ...
    across the entire sequence. Decay weights distant samples, output normalized
    by valid sample count.
    """
    
    stride_grid: torch.Tensor
    
    def __init__(
        self,
        channels: int,
        ndim: int = 1,
        max_samples: int = 32,
        num_channels: int = 1,
        max_freq: float = 16.0,
        min_freq: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.ndim = ndim
        self.max_samples = max_samples
        self.num_channels = num_channels
        self.channel_dim = channels // num_channels
        H = num_channels
        
        self.max_freq = max_freq
        self.min_freq = min_freq
        
        self.wave_proj = nn.Linear(channels, 3 * H)
        
        pos_dim = 16
        self.pos_dim = pos_dim
        self.query_proj = nn.Linear(channels, H * pos_dim)
        self.key_proj = nn.Linear(1, pos_dim, bias=False)
        self.scale = pos_dim ** -0.5
        
        self.out_proj = nn.Linear(channels, channels, bias=False)
        
        self.se_fc1 = nn.Linear(channels, channels // 4)
        self.se_fc2 = nn.Linear(channels // 4, channels)
        
        samples_per_dim = max(3, int(max_samples ** (1.0 / ndim)))
        half_s = samples_per_dim // 2
        offsets_1d = torch.arange(-half_s, half_s + 1).float()
        
        if ndim == 1:
            stride_grid = offsets_1d.unsqueeze(-1)
        else:
            grids = torch.meshgrid(*[offsets_1d] * ndim, indexing='ij')
            stride_grid = torch.stack([g.flatten() for g in grids], dim=-1)
        
        self.register_buffer('stride_grid', stride_grid)
        self.num_samples = stride_grid.shape[0]
        
        nn.init.zeros_(self.wave_proj.weight)
        nn.init.zeros_(self.wave_proj.bias)
        nn.init.zeros_(self.out_proj.weight)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B = x.shape[0]
        spatial = x.shape[1:-1]
        C = x.shape[-1]
        H = self.num_channels
        D = self.channel_dim
        L = reduce(mul, spatial, 1)
        S = self.num_samples
        
        x_flat = x.reshape(B, L, C)
        
        wave_params = F.silu(self.wave_proj(x_flat)).reshape(B, L, 3, H).permute(0, 1, 3, 2)
        freq = torch.sigmoid(wave_params[..., 0]) * (self.max_freq - self.min_freq) + self.min_freq
        phase = torch.tanh(wave_params[..., 1]) * self.max_freq
        decay = torch.sigmoid(wave_params[..., 2]) * 9.5 + 0.5
        
        freq_avg = freq.mean(dim=2)
        phase_avg = phase.mean(dim=2)
        
        if self.ndim == 1:
            centers = torch.arange(L, device=x.device, dtype=x.dtype)
            sample_pos = centers.view(1, L, 1) + self.stride_grid[:, 0].view(1, 1, S) * freq_avg.unsqueeze(-1) + phase_avg.unsqueeze(-1)
            valid_mask = (sample_pos >= 0) & (sample_pos < L)
            sample_idx = sample_pos.long().clamp(0, L - 1)
        else:
            coords = [torch.arange(s, device=x.device, dtype=x.dtype) for s in spatial]
            mesh = torch.meshgrid(*coords, indexing='ij')
            centers = torch.stack([m.flatten() for m in mesh], dim=-1)
            
            sample_pos = centers.view(1, L, 1, self.ndim) + self.stride_grid.view(1, 1, S, self.ndim) * freq_avg.view(B, L, 1, 1) + phase_avg.view(B, L, 1, 1)
            
            valid_mask = torch.ones(B, L, S, dtype=torch.bool, device=x.device)
            for dim in range(self.ndim):
                valid_mask = valid_mask & (sample_pos[..., dim] >= 0) & (sample_pos[..., dim] < spatial[dim])
            
            sample_coords = sample_pos.long()
            for dim in range(self.ndim):
                sample_coords[..., dim] = sample_coords[..., dim].clamp(0, spatial[dim] - 1)
            
            strides = [1]
            for dim in range(self.ndim - 1, 0, -1):
                strides.insert(0, strides[0] * spatial[dim])
            sample_idx = sum(sample_coords[..., dim] * strides[dim] for dim in range(self.ndim))
        
        batch_idx = torch.arange(B, device=x.device).view(B, 1, 1).expand(B, L, S)
        values = x_flat[batch_idx, sample_idx].reshape(B, L, S, H, D).permute(0, 1, 3, 2, 4)
        
        valid_mask = valid_mask.unsqueeze(2).expand(B, L, H, S)
        
        queries = F.silu(self.query_proj(x_flat)).reshape(B, L, H, self.pos_dim)
        
        rel_dist = (self.stride_grid.norm(dim=-1).view(1, 1, 1, S) * freq.unsqueeze(-1)).abs()
        keys = self.key_proj(rel_dist.unsqueeze(-1))
        
        attn_logits = torch.einsum('blhd,blhsd->blhs', queries, keys) * self.scale
        
        decay_envelope = torch.exp(-rel_dist / decay.unsqueeze(-1).clamp(min=0.1))
        
        attn_logits = attn_logits.masked_fill(~valid_mask, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1) * decay_envelope * valid_mask.float()
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        output = torch.einsum('blhsd,blhs->blhd', values, attn_weights).reshape(B, L, C)
        
        se_weights = torch.sigmoid(self.se_fc2(F.silu(self.se_fc1(output))))
        output = output * se_weights
        
        output = self.out_proj(output).reshape(B, *spatial, C)
        
        entropy = -(attn_weights * (attn_weights + 1e-8).log()).sum(dim=-1).mean()
        return output, {"entropy_reg": -entropy}


# Backward compatibility alias
AdaptiveDeformConvND = AdaptiveConvND


class ScatterAttention1d(nn.Module):
    
    k_indices: torch.Tensor
    
    def __init__(
        self,
        num_samples: int = 16,
        max_offset: float = 32.0,
        init_stride: float = 1.0,
        init_beta: float = 1.0,
        init_alpha: float = 1.0,
        init_strength: float = 1.0,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.max_offset = max_offset
        
        self.base_deformation = nn.Parameter(torch.tensor(0.0))
        self.base_stride = nn.Parameter(torch.tensor(init_stride))
        self.base_beta_fwd = nn.Parameter(torch.tensor(init_beta))
        self.base_beta_bwd = nn.Parameter(torch.tensor(init_beta))
        self.base_strength = nn.Parameter(torch.tensor(init_strength))
        self.base_alpha_fwd = nn.Parameter(torch.tensor(init_alpha))
        self.base_alpha_bwd = nn.Parameter(torch.tensor(init_alpha))
        self.sample_bias = nn.Parameter(torch.zeros(num_samples))
        
        k_indices = torch.arange(num_samples, dtype=torch.float32) - num_samples // 2
        self.register_buffer('k_indices', k_indices)
    
    def forward(
        self,
        x: torch.Tensor,
        film_biases: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        B, L = x.shape
        device = x.device
        
        if film_biases is None:
            film_biases = {}
        
        deformation = self.base_deformation + film_biases.get('deformation', 0)
        stride = F.softplus(self.base_stride + film_biases.get('stride', 0))
        beta_fwd = F.softplus(self.base_beta_fwd + film_biases.get('beta_fwd', 0))
        beta_bwd = F.softplus(self.base_beta_bwd + film_biases.get('beta_bwd', 0))
        strength = F.softplus(self.base_strength + film_biases.get('strength', 0))
        alpha_fwd = F.softplus(self.base_alpha_fwd + film_biases.get('alpha_fwd', 0))
        alpha_bwd = F.softplus(self.base_alpha_bwd + film_biases.get('alpha_bwd', 0))
        sample_bias = self.sample_bias + film_biases.get('sample_bias', 0)
        
        deformation = deformation.clamp(-self.max_offset, self.max_offset)
        
        k = self.k_indices
        k_abs = k.abs()
        
        warped_fwd = (k_abs ** beta_fwd) * stride
        warped_bwd = (k_abs ** beta_bwd) * stride
        warped = torch.where(k >= 0, warped_fwd, -warped_bwd)
        
        l_indices = torch.arange(L, device=device, dtype=torch.float32)
        centers = l_indices + deformation
        
        positions = centers.unsqueeze(-1) + warped.unsqueeze(0)
        positions = positions.clamp(0, L - 1)
        positions = positions.unsqueeze(0).expand(B, -1, -1)
        
        envelope_fwd = strength / (1 + k_abs) ** alpha_fwd
        envelope_bwd = strength / (1 + k_abs) ** alpha_bwd
        envelope = torch.where(k >= 0, envelope_fwd, envelope_bwd)
        
        raw_weights = envelope * (1 + sample_bias.tanh())
        raw_weights = raw_weights.unsqueeze(0).unsqueeze(0).expand(B, L, -1)
        
        weight_map, hit_count = self._splat(positions, raw_weights, L)
        
        avg_weights = weight_map / hit_count.clamp(min=1e-6)
        norm_weights = avg_weights / avg_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        
        return x * norm_weights
    
    def _splat(
        self,
        positions: torch.Tensor,
        weights: torch.Tensor,
        L: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = positions.shape[0]
        device = positions.device
        
        pos_floor = positions.floor().long().clamp(0, L - 1)
        pos_ceil = (pos_floor + 1).clamp(0, L - 1)
        
        frac = positions - pos_floor.float()
        w_floor = (1 - frac) * weights
        w_ceil = frac * weights
        h_floor = 1 - frac
        h_ceil = frac
        
        batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand_as(pos_floor)
        
        idx_floor = (batch_idx * L + pos_floor).reshape(-1)
        idx_ceil = (batch_idx * L + pos_ceil).reshape(-1)
        
        weight_map = torch.zeros(B * L, device=device, dtype=weights.dtype)
        hit_count = torch.zeros(B * L, device=device, dtype=weights.dtype)
        
        weight_map.scatter_add_(0, idx_floor, w_floor.reshape(-1))
        weight_map.scatter_add_(0, idx_ceil, w_ceil.reshape(-1))
        hit_count.scatter_add_(0, idx_floor, h_floor.reshape(-1))
        hit_count.scatter_add_(0, idx_ceil, h_ceil.reshape(-1))
        
        return weight_map.view(B, L), hit_count.view(B, L)


class SEBlock1d(nn.Module):
    
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
    
    def forward(
        self,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pooled = x.mean(dim=1)
        s = F.silu(self.fc1(pooled))
        s = self.fc2(s)
        if bias is not None:
            s = s + bias
        s = torch.sigmoid(s)
        return x * s.unsqueeze(1)


class SEBlock2d(nn.Module):
    
    def __init__(self, scatter_channels: int, num_samples: int, reduction: int = 4):
        super().__init__()
        total = scatter_channels * num_samples
        mid = max(1, total // reduction)
        self.scatter_channels = scatter_channels
        self.num_samples = num_samples
        
        self.fc1_global = nn.Linear(total, mid)
        self.fc2_global = nn.Linear(mid, total)
        
        self.fc1_local = nn.Linear(total, mid)
        self.fc2_local = nn.Linear(mid, total)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, SC, K = x.shape
        total = SC * K
        
        pooled = x.mean(dim=1).reshape(B, total)
        s_global = F.silu(self.fc1_global(pooled))
        s_global = torch.sigmoid(self.fc2_global(s_global)).reshape(B, 1, SC, K)
        x = x * s_global
        
        x_flat = x.reshape(B * L, total)
        s_local = F.silu(self.fc1_local(x_flat))
        s_local = torch.sigmoid(self.fc2_local(s_local)).reshape(B, L, SC, K)
        x = x * s_local
        
        return x


class RMSNorm(nn.Module):
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class GatedMerge(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, 2.0)
    
    def forward(self, embed: torch.Tensor, processed: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(F.silu(self.gate_proj(embed)))
        return gate * embed + (1 - gate) * processed


class LearnedMerge(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.scale = dim ** -0.5
    
    def forward(self, embed: torch.Tensor, processed: torch.Tensor) -> torch.Tensor:
        B, L = embed.shape[:2]
        if processed.shape[1] == 1 and L > 1:
            processed = processed.expand(B, L, -1)
        
        q = F.silu(self.q_proj(processed))
        kv = F.silu(self.kv_proj(embed))
        k, v = kv.chunk(2, dim=-1)
        
        scores = torch.einsum('blc,bkc->blk', q, k) * self.scale
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('blk,bkc->blc', attn, v)
        
        return F.silu(self.out_proj(out))


class LowRankAttentionMerge(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, embed: torch.Tensor, processed: torch.Tensor) -> torch.Tensor:
        B, L, C = embed.shape
        if processed.shape[1] == 1 and L > 1:
            processed = processed.expand_as(embed)
        
        r = max(1, int(L ** 0.5))
        embed_down = F.adaptive_avg_pool1d(embed.mT, r).mT
        processed_down = F.adaptive_avg_pool1d(processed.mT, r).mT
        
        q = self.q_proj(processed_down)
        k = self.k_proj(embed_down)
        v = self.v_proj(embed_down)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = F.silu(self.out_proj(out))
        out = F.interpolate(out.mT, size=L, mode='linear', align_corners=False).mT
        
        return processed + out


class LowRankAttention(nn.Module):
    """(r,r) attention where r=sqrt(L). Returns (full_out, lowrank_out) for collapse reuse."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, C = x.shape
        r = max(1, int(L ** 0.5))
        
        x_down = F.adaptive_avg_pool1d(x.mT, r).mT
        
        q = self.q_proj(x_down)
        k = self.k_proj(x_down)
        v = self.v_proj(x_down)
        
        lowrank_out = F.scaled_dot_product_attention(q, k, v)
        lowrank_out = F.silu(self.out_proj(lowrank_out))
        
        full_out = F.interpolate(lowrank_out.mT, size=L, mode='linear', align_corners=False).mT
        
        return full_out, lowrank_out


class LowRankCollapseMerge(nn.Module):
    """Merge low-rank representations during collapse. Interpolates coarser to finer size."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, finer: torch.Tensor, coarser: torch.Tensor) -> torch.Tensor:
        r_fine = finer.shape[1]
        
        coarser_up = F.interpolate(coarser.mT, size=r_fine, mode='linear', align_corners=False).mT
        
        q = self.q_proj(coarser_up)
        k = self.k_proj(finer)
        v = self.v_proj(finer)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = F.silu(self.out_proj(out))
        
        return coarser_up + out


class LowRankAttentionND(nn.Module):
    """Full attention with sqrt(L) rank reduction. Drop-in replacement for LocalAttentionND."""
    
    def __init__(self, embed_dim: int, window_size: int | tuple[int, ...] = 17, ndim: int = 1, num_channels: int = 1):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.num_channels = num_channels
        self.channel_dim = embed_dim // num_channels
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[1:-1]
        B, C = x.shape[0], x.shape[-1]
        L = 1
        for s in spatial_shape:
            L *= s
        
        x_flat = x.reshape(B, L, C)
        r = max(1, int(L ** 0.5))
        
        x_down = F.adaptive_avg_pool1d(x_flat.mT, r).mT
        
        q = self.q_proj(x_down)
        k = self.k_proj(x_down)
        v = self.v_proj(x_down)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = F.silu(self.out_proj(out))
        out = F.interpolate(out.mT, size=L, mode='linear', align_corners=False).mT
        
        return out.reshape(B, *spatial_shape, C)


def apply_rope(x: torch.Tensor, positions: torch.Tensor | None = None, base: float = 10000.0) -> torch.Tensor:
    """Apply rotary position embedding.
    
    Args:
        x: (..., C) tensor
        positions: (...) integer positions. If None, uses [0, 1, ..., L-1] for dim -2.
        base: RoPE base frequency
    """
    C = x.shape[-1]
    half_c = C // 2
    device = x.device
    dtype = x.dtype
    
    if positions is None:
        L = x.shape[-2]
        positions = torch.arange(L, device=device)
        for _ in range(x.ndim - 2):
            positions = positions.unsqueeze(0)
    
    dim_idx = torch.arange(half_c, device=device, dtype=dtype)
    freqs = 1.0 / (base ** (dim_idx / half_c))
    
    angles = positions.unsqueeze(-1).float() * freqs
    cos = angles.cos()
    sin = angles.sin()
    
    x1, x2 = x[..., :half_c], x[..., half_c:half_c * 2]
    
    x_rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    
    if C % 2 == 1:
        x_rotated = torch.cat([x_rotated, x[..., -1:]], dim=-1)
    
    return x_rotated


def sinusoidal_pos_embed(L: int, C: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0) -> torch.Tensor:
    pos = torch.arange(L, device=device, dtype=dtype)
    dim_idx = torch.arange(C // 2, device=device, dtype=dtype)
    freqs = 1.0 / (base ** (2 * dim_idx / C))
    
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    
    embed = torch.zeros(L, C, device=device, dtype=dtype)
    embed[:, 0::2] = angles.sin()
    embed[:, 1::2] = angles.cos()
    
    return embed


def sinusoidal_pos_embed_nd(
    shape: tuple[int, ...],
    embed_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    base: float = 10000.0,
) -> torch.Tensor:
    ndim = len(shape)
    dim_per_axis = embed_dim // ndim
    remainder = embed_dim % ndim
    
    embeds = []
    for axis, size in enumerate(shape):
        axis_dim = dim_per_axis + (1 if axis < remainder else 0)
        half_dim = axis_dim // 2
        
        pos = torch.arange(size, device=device, dtype=dtype)
        dim_idx = torch.arange(half_dim, device=device, dtype=dtype)
        freqs = 1.0 / (base ** (2 * dim_idx / max(axis_dim, 1)))
        
        angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
        
        axis_embed = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if axis_dim % 2 == 1:
            axis_embed = torch.cat([axis_embed, torch.zeros(size, 1, device=device, dtype=dtype)], dim=-1)
        
        view_shape = [1] * axis + [size] + [1] * (ndim - axis - 1) + [axis_dim]
        expand_shape = list(shape) + [axis_dim]
        axis_embed = axis_embed.view(*view_shape).expand(*expand_shape)
        embeds.append(axis_embed)
    
    return torch.cat(embeds, dim=-1)


class LocalAttentionND(nn.Module):
    
    rel_dist: torch.Tensor
    
    def __init__(self, embed_dim: int, kernel_size: int | tuple[int, ...] = 7, ndim: int = 1, num_channels: int = 1, checkpoint: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.num_channels = num_channels
        self.channel_dim = embed_dim // num_channels
        self.checkpoint = checkpoint
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim
        self.kernel_size = kernel_size
        self.half_k = tuple(k // 2 for k in kernel_size)
        self.window_size = 1
        for k in kernel_size:
            self.window_size *= k
        
        self.scale = self.channel_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.q_norm = RMSNorm(self.channel_dim)
        self.k_norm = RMSNorm(self.channel_dim)
        self.window_proj = nn.Linear(embed_dim, 2 * num_channels)
        self.out_norm = RMSNorm(embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_gate = GatedMerge(embed_dim)
        
        nn.init.zeros_(self.window_proj.weight)
        nn.init.zeros_(self.window_proj.bias)
        
        rel_coords = [torch.arange(k).float() - k // 2 for k in kernel_size]
        grids = torch.meshgrid(*rel_coords, indexing='ij')
        rel_dist = torch.sqrt(torch.stack([g**2 for g in grids]).sum(dim=0))
        self.register_buffer('rel_dist', rel_dist.flatten())
        self.max_dist = rel_dist.max().item()
    
    def _unfold_nd(self, x: torch.Tensor) -> torch.Tensor:
        # Asymmetric padding to handle even kernel sizes correctly
        # Total pad = k-1, so after unfold we get original size back
        pad_dims = [0, 0]
        for k in reversed(self.kernel_size):
            pad_left = (k - 1) // 2
            pad_right = k // 2
            pad_dims.extend([pad_left, pad_right])
        x = F.pad(x, pad_dims)
        
        for dim in range(self.ndim):
            x = x.unfold(dim + 1, self.kernel_size[dim], 1)
        return x
    
    def _attention_fn(self, x: torch.Tensor, q_flat: torch.Tensor, k: torch.Tensor, v: torch.Tensor, width: torch.Tensor, sharpness: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[1:-1]
        B = x.shape[0]
        C = x.shape[-1]
        L = q_flat.shape[1]
        H = self.num_channels
        D = self.channel_dim
        
        k_win = self._unfold_nd(k)
        v_win = self._unfold_nd(v)
        
        perm = [0] + list(range(1, self.ndim + 1)) + list(range(self.ndim + 2, self.ndim * 2 + 2)) + [self.ndim + 1]
        k_win = k_win.permute(*perm).reshape(B, L, self.window_size, H, D).permute(0, 1, 3, 4, 2)
        v_win = v_win.permute(*perm).reshape(B, L, self.window_size, H, D).permute(0, 1, 3, 4, 2)
        
        width_flat = width.reshape(B, L, H, 1)
        sharpness_flat = sharpness.reshape(B, L, H, 1)
        
        scores = torch.einsum('blhd,blhdw->blhw', q_flat, k_win) * self.scale
        
        soft_mask = torch.sigmoid((width_flat - self.rel_dist) * sharpness_flat)
        scores = scores - (1 - soft_mask) * 1e4
        
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('blhw,blhdw->blhd', attn, v_win).reshape(B, L, C)
        
        out = self.out_norm(out.reshape(*x.shape[:-1], C))
        merged = self.v_gate(v, out)
        return F.silu(self.out(merged))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[1:-1]
        B = x.shape[0]
        C = x.shape[-1]
        L = 1
        for s in spatial_shape:
            L *= s
        
        H = self.num_channels
        D = self.channel_dim
        
        qkv = F.silu(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=-1)
        
        q_flat = self.q_norm(q.reshape(B, L, H, D))
        k_flat = self.k_norm(k.reshape(B, L, H, D))
        q_flat = apply_rope(q_flat)
        k_flat = apply_rope(k_flat)
        k = k_flat.reshape(*k.shape)
        
        window_params = F.silu(self.window_proj(x))
        width_raw, sharpness_raw = window_params.chunk(2, dim=-1)
        width = width_raw.sigmoid() * self.max_dist + 0.5
        sharpness = sharpness_raw.sigmoid() * 9.5 + 0.5
        
        if self.checkpoint and self.training:
            return grad_checkpoint(  # type: ignore[return-value]
                self._attention_fn, x, q_flat, k, v, width, sharpness, use_reentrant=False
            )
        return self._attention_fn(x, q_flat, k, v, width, sharpness)


class LocalAttention(LocalAttentionND):
    def __init__(self, embed_dim: int, kernel_size: int = 17, num_channels: int = 1):
        super().__init__(embed_dim, kernel_size, ndim=1, num_channels=num_channels)


class LocalBlock(nn.Module):
    
    def __init__(self, embed_dim: int, kernel_size: int = 17, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps)
        self.attn = LocalAttention(embed_dim, kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class HierarchicalLocalAttentionND(nn.Module):
    
    def __init__(self, embed_dim: int, window_size: int | tuple[int, ...] = 17, ndim: int = 1, num_channels: int = 1, poolable_dims: tuple[int, ...] | None = None, min_size: int = 4, conv_position: str = 'both', attn_residual: bool = True, merge_mode: str = 'lowrank', lowrank_hier: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.num_channels = num_channels
        self.channel_dim = embed_dim // num_channels
        self.poolable_dims = poolable_dims if poolable_dims is not None else tuple(range(ndim))
        self.min_size = min_size
        self.conv_position = conv_position
        self.attn_residual = attn_residual
        self.merge_mode = merge_mode
        self.lowrank_hier = lowrank_hier
        
        if isinstance(window_size, int):
            window_size = (window_size,) * ndim
        self.window_size = window_size
        
        conv_cls = [nn.Conv1d, nn.Conv2d, nn.Conv3d][ndim - 1]
        
        merge_cls = {'gate': GatedMerge, 'learned': LearnedMerge, 'lowrank': LowRankAttentionMerge}[merge_mode]
        
        self.scatter_pre = AdaptiveConvND(embed_dim, ndim=ndim, max_samples=window_size[0], num_channels=num_channels)
        self.scatter_pre_norm = RMSNorm(embed_dim)
        self.scatter_pre_gate = merge_cls(embed_dim)
        
        self.scatter_post = AdaptiveConvND(embed_dim, ndim=ndim, max_samples=window_size[0], num_channels=num_channels)
        self.scatter_post_norm = RMSNorm(embed_dim)
        self.scatter_post_gate = merge_cls(embed_dim)
        
        self.reduce_norm = RMSNorm(embed_dim)
        if len(self.poolable_dims) == ndim:
            self.reduce_conv = conv_cls(embed_dim, embed_dim, kernel_size=2, stride=2)
        else:
            reduce_stride = tuple(2 if d in self.poolable_dims else 1 for d in range(ndim))
            reduce_kernel = tuple(2 if d in self.poolable_dims else 1 for d in range(ndim))
            self.reduce_conv = conv_cls(embed_dim, embed_dim, kernel_size=reduce_kernel, stride=reduce_stride)
        
        self.gather = LocalAttentionND(embed_dim, window_size, ndim, num_channels)
        
        self.lowrank_attn = LowRankAttention(embed_dim)
        
        self.collapse_merge = LowRankCollapseMerge(embed_dim)
        
        self.final_merge = merge_cls(embed_dim)
    
    def _compute_n_levels(self, spatial_shape: tuple[int, ...]) -> int:
        if not self.poolable_dims:
            return 1
        min_poolable = min(spatial_shape[d] for d in self.poolable_dims)
        return max(1, (min_poolable // self.min_size).bit_length())
    
    def _reduce(self, h: torch.Tensor) -> torch.Tensor:
        return self.reduce_conv(h)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[1:-1]
        B = x.shape[0]
        C = x.shape[-1]
        L = 1
        for s in spatial_shape:
            L *= s
        
        n_levels = self._compute_n_levels(spatial_shape)
        
        h = x
        lowrank_outputs: list[torch.Tensor] = []
        current_shape = list(spatial_shape)
        
        for i in range(n_levels):
            h_input = h
            
            conv_out, _ = self.scatter_pre(h)
            conv_out = self.scatter_pre_norm(conv_out.reshape(B, -1, C)).reshape(B, *h.shape[1:-1], C)
            h = self.scatter_pre_gate(h.reshape(B, -1, C), conv_out.reshape(B, -1, C)).reshape(B, *h.shape[1:-1], C)
            
            h = self.gather(h)
            if self.attn_residual:
                h = h + h_input
            
            conv_out, _ = self.scatter_post(h)
            conv_out = self.scatter_post_norm(conv_out.reshape(B, -1, C)).reshape(B, *h.shape[1:-1], C)
            h = self.scatter_post_gate(h.reshape(B, -1, C), conv_out.reshape(B, -1, C)).reshape(B, *h.shape[1:-1], C)
            
            h_flat = h.reshape(B, -1, C)
            full_out, lowrank_out = self.lowrank_attn(h_flat)
            h = (h_flat + full_out).reshape(B, *h.shape[1:-1], C)
            lowrank_outputs.append(lowrank_out)
            
            if i < n_levels - 1:
                h_spatial = h.shape[1:-1]
                h_normed = self.reduce_norm(h.reshape(B, -1, C)).reshape(B, *h_spatial, C)
                h_flat = h_normed.reshape(B, -1, C).mT.reshape(B, C, *h_spatial)
                h_reduced = self._reduce(h_flat)
                current_shape = list(h_reduced.shape[2:])
                h = h_reduced.reshape(B, C, -1).mT.reshape(B, *current_shape, C)
        
        lr = lowrank_outputs[-1]
        for i in range(len(lowrank_outputs) - 2, -1, -1):
            lr = self.collapse_merge(lowrank_outputs[i], lr)
        
        final = F.interpolate(lr.mT, size=L, mode='linear', align_corners=False).mT
        
        return final.reshape(B, *spatial_shape, C)


class HierarchicalLocalAttention(HierarchicalLocalAttentionND):
    def __init__(self, embed_dim: int, window_size: int = 17, num_channels: int = 1, poolable_dims: tuple[int, ...] | None = None, conv_position: str = 'both', attn_residual: bool = True, merge_mode: str = 'lowrank', lowrank_hier: bool = True):
        super().__init__(embed_dim, window_size, ndim=1, num_channels=num_channels, poolable_dims=poolable_dims, conv_position=conv_position, attn_residual=attn_residual, merge_mode=merge_mode, lowrank_hier=lowrank_hier)


class HierarchicalLocalBlockND(nn.Module):
    
    def __init__(self, embed_dim: int, window_size: int | tuple[int, ...] = 17, ndim: int = 1, num_channels: int = 1, poolable_dims: tuple[int, ...] | None = None, eps: float = 1e-6, conv_position: str = 'both', attn_residual: bool = True, merge_mode: str = 'lowrank', lowrank_hier: bool = True):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps)
        self.attn = HierarchicalLocalAttentionND(embed_dim, window_size, ndim, num_channels, poolable_dims, conv_position=conv_position, attn_residual=attn_residual, merge_mode=merge_mode, lowrank_hier=lowrank_hier)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class HierarchicalLocalBlock(HierarchicalLocalBlockND):
    def __init__(self, embed_dim: int, window_size: int = 17, num_channels: int = 1, poolable_dims: tuple[int, ...] | None = None, eps: float = 1e-6, conv_position: str = 'both', attn_residual: bool = True, merge_mode: str = 'lowrank', lowrank_hier: bool = True):
        super().__init__(embed_dim, window_size, ndim=1, num_channels=num_channels, poolable_dims=poolable_dims, eps=eps, conv_position=conv_position, attn_residual=attn_residual, merge_mode=merge_mode, lowrank_hier=lowrank_hier)


LocalKernelAttention = LocalAttention
LocalKernelBlock = LocalBlock
