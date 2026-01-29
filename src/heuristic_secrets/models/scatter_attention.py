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

try:
    from .triton_scatter import (
        TritonScatterConv,
        TritonLocalWindowAttn,
        TritonSSMStep,
        HAS_TRITON,
    )
except ImportError:
    HAS_TRITON = False
    TritonScatterConv = None
    TritonLocalWindowAttn = None

try:
    from .triton_adaptive_conv import TritonAdaptiveLocalConv
    HAS_TRITON_ADAPTIVE = True
except ImportError:
    HAS_TRITON_ADAPTIVE = False
    TritonAdaptiveLocalConv = None

try:
    from .adaptive_local_conv import AdaptiveLocalConv
except ImportError:
    AdaptiveLocalConv = None
    TritonSSMStep = None


def avg_pool_nd(x: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Average pool spatial dimensions to target shape. Input: (B, *spatial, C)."""
    spatial = x.shape[1:-1]
    ndim = len(spatial)
    
    if spatial == target_shape:
        return x
    
    x = x.movedim(-1, 1)
    
    kernel_size = tuple(max(1, s // t) for s, t in zip(spatial, target_shape))
    stride = kernel_size
    
    pool_fn = [F.avg_pool1d, F.avg_pool2d, F.avg_pool3d][ndim - 1]
    x = pool_fn(x, kernel_size, stride)
    
    current = x.shape[2:]
    if current != target_shape:
        mode = ['linear', 'bilinear', 'trilinear'][ndim - 1]
        x = F.interpolate(x, size=target_shape, mode=mode, align_corners=False)
    
    return x.movedim(1, -1)


def interpolate_nd(x: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
    """Interpolate spatial dimensions to target shape. Input: (B, *spatial, C)."""
    ndim = len(x.shape) - 2
    x = x.movedim(-1, 1)
    mode = ['linear', 'bilinear', 'trilinear'][ndim - 1]
    x = F.interpolate(x, size=target_shape, mode=mode, align_corners=False)
    return x.movedim(1, -1)


class Sine(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


def relu_squared(x: torch.Tensor) -> torch.Tensor:
    return F.relu(x).square()


class SqueezeExciteND(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, relu2: bool = False):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, channels, bias=False)
        self.act = relu_squared if relu2 else F.silu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_dims = tuple(range(1, x.ndim - 1))
        scale = x.mean(dim=spatial_dims)
        scale = torch.sigmoid(self.fc2(self.act(self.fc1(scale))))
        for _ in spatial_dims:
            scale = scale.unsqueeze(1)
        return x * scale


class SIRENDownsampleND(nn.Module):
    
    def __init__(self, channels: int, ndim: int = 1, hidden: int = 32, omega_0: float = 30.0):
        super().__init__()
        self.channels = channels
        self.ndim = ndim
        self.kernel_net = KernelNetND(ndim, channels, hidden=hidden, omega_0=omega_0)
        self.se = SqueezeExciteND(channels)
    
    def forward(self, x: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
        spatial = x.shape[1:-1]
        B, C = x.shape[0], x.shape[-1]
        
        if spatial == target_shape:
            return x
        
        kernel_size = tuple(max(1, s // t) for s, t in zip(spatial, target_shape))
        
        grids = [torch.linspace(-1, 1, k, device=x.device, dtype=x.dtype) for k in kernel_size]
        if self.ndim == 1:
            positions = grids[0].unsqueeze(-1)
        else:
            mesh = torch.meshgrid(*grids, indexing='ij')
            positions = torch.stack(mesh, dim=-1).reshape(-1, self.ndim)
        
        kernel_flat = self.kernel_net(positions)
        kernel = kernel_flat.T.reshape(C, 1, *kernel_size)
        
        x = x.movedim(-1, 1)
        conv_fn = [F.conv1d, F.conv2d, F.conv3d][self.ndim - 1]
        x = conv_fn(x, kernel, stride=kernel_size, groups=C)
        
        current = x.shape[2:]
        if current != target_shape:
            x = F.interpolate(x, size=target_shape, mode='nearest')
        
        x = x.movedim(1, -1)
        
        return self.se(x)


class SIRENUpsampleND(nn.Module):
    
    def __init__(self, channels: int, ndim: int = 1, hidden: int = 32, omega_0: float = 30.0):
        super().__init__()
        self.channels = channels
        self.ndim = ndim
        self.kernel_net = KernelNetND(ndim, channels, hidden=hidden, omega_0=omega_0)
        self.se = SqueezeExciteND(channels)
    
    def forward(self, x: torch.Tensor, target_shape: tuple[int, ...]) -> torch.Tensor:
        spatial = x.shape[1:-1]
        B, C = x.shape[0], x.shape[-1]
        
        if spatial == target_shape:
            return x
        
        stride = tuple(max(1, t // s) for s, t in zip(spatial, target_shape))
        kernel_size = stride
        
        grids = [torch.linspace(-1, 1, k, device=x.device, dtype=x.dtype) for k in kernel_size]
        if self.ndim == 1:
            positions = grids[0].unsqueeze(-1)
        else:
            mesh = torch.meshgrid(*grids, indexing='ij')
            positions = torch.stack(mesh, dim=-1).reshape(-1, self.ndim)
        
        kernel_flat = self.kernel_net(positions)
        kernel = kernel_flat.T.reshape(C, 1, *kernel_size)
        
        x = x.movedim(-1, 1)
        conv_t_fn = [F.conv_transpose1d, F.conv_transpose2d, F.conv_transpose3d][self.ndim - 1]
        x = conv_t_fn(x, kernel, stride=stride, groups=C)
        
        current = x.shape[2:]
        if current != target_shape:
            x = F.interpolate(x, size=target_shape, mode='nearest')
        
        x = x.movedim(1, -1)
        return self.se(x)


class KernelNetND(nn.Module):
    def __init__(self, ndim: int, out_channels: int, hidden: int = 32, layers: int = 3, omega_0: float = 30.0):
        super().__init__()
        self.omega_0 = omega_0
        net: list[nn.Module] = [nn.Linear(ndim, hidden)]
        for _ in range(layers - 1):
            net.extend([Sine(), nn.Linear(hidden, hidden)])
        net.extend([Sine(), nn.Linear(hidden, out_channels)])
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
        chunk_size: int = 2048,
    ):
        super().__init__()
        self.channels = channels
        self.ndim = ndim
        self.max_samples = max_samples
        self.num_channels = num_channels
        self.channel_dim = channels // num_channels
        self.chunk_size = chunk_size
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
        
        self._triton_mod = None
        if ndim == 1 and HAS_TRITON and TritonScatterConv is not None:
            self._triton_mod = TritonScatterConv(
                channels=channels,
                max_samples=max_samples,
                num_channels=num_channels,
                max_freq=max_freq,
                min_freq=min_freq,
            )
    
    def _forward_chunk(self, x_flat: torch.Tensor, x_chunk: torch.Tensor,
                        chunk_start: int, chunk_end: int,
                        spatial: tuple, L: int) -> torch.Tensor:
        B = x_flat.shape[0]
        C = x_flat.shape[-1]
        H = self.num_channels
        D = self.channel_dim
        S = self.num_samples
        chunk_len = chunk_end - chunk_start
        
        wave_params = F.silu(self.wave_proj(x_chunk)).reshape(B, chunk_len, 3, H).permute(0, 1, 3, 2)
        queries = F.silu(self.query_proj(x_chunk)).reshape(B, chunk_len, H, self.pos_dim)
        
        freq = torch.sigmoid(wave_params[..., 0]) * (self.max_freq - self.min_freq) + self.min_freq
        phase = torch.tanh(wave_params[..., 1]) * self.max_freq
        decay = torch.sigmoid(wave_params[..., 2]) * 9.5 + 0.5
        
        freq_avg = freq.mean(dim=2)
        phase_avg = phase.mean(dim=2)
        
        if self.ndim == 1:
            centers = torch.arange(chunk_start, chunk_end, device=x_flat.device, dtype=x_flat.dtype)
            sample_pos = centers.view(1, chunk_len, 1) + self.stride_grid[:, 0].view(1, 1, S) * freq_avg.unsqueeze(-1) + phase_avg.unsqueeze(-1)
            valid_mask = (sample_pos >= 0) & (sample_pos < L)
            sample_idx = sample_pos.long().clamp(0, L - 1)
        else:
            coords = [torch.arange(s, device=x_flat.device, dtype=x_flat.dtype) for s in spatial]
            mesh = torch.meshgrid(*coords, indexing='ij')
            centers_all = torch.stack([m.flatten() for m in mesh], dim=-1)
            centers = centers_all[chunk_start:chunk_end]
            
            sample_pos = centers.view(1, chunk_len, 1, self.ndim) + self.stride_grid.view(1, 1, S, self.ndim) * freq_avg.view(B, chunk_len, 1, 1) + phase_avg.view(B, chunk_len, 1, 1)
            
            valid_mask = torch.ones(B, chunk_len, S, dtype=torch.bool, device=x_flat.device)
            for dim in range(self.ndim):
                valid_mask = valid_mask & (sample_pos[..., dim] >= 0) & (sample_pos[..., dim] < spatial[dim])
            
            sample_coords = sample_pos.long()
            for dim in range(self.ndim):
                sample_coords[..., dim] = sample_coords[..., dim].clamp(0, spatial[dim] - 1)
            
            strides = [1]
            for dim in range(self.ndim - 1, 0, -1):
                strides.insert(0, strides[0] * spatial[dim])
            sample_idx = sum(sample_coords[..., dim] * strides[dim] for dim in range(self.ndim))
        
        batch_idx = torch.arange(B, device=x_flat.device).view(B, 1, 1).expand(B, chunk_len, S)
        values = x_flat[batch_idx, sample_idx].reshape(B, chunk_len, S, H, D).permute(0, 1, 3, 2, 4)
        
        valid_mask = valid_mask.unsqueeze(2).expand(B, chunk_len, H, S)
        
        rel_dist = (self.stride_grid.norm(dim=-1).view(1, 1, 1, S) * freq.unsqueeze(-1)).abs()
        keys = F.silu(self.key_proj(rel_dist.unsqueeze(-1)))
        
        attn_logits = torch.einsum('blhd,blhsd->blhs', queries, keys) * self.scale
        
        decay_envelope = torch.exp(-rel_dist / decay.unsqueeze(-1).clamp(min=0.1))
        
        attn_logits = attn_logits.masked_fill(~valid_mask, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1) * decay_envelope * valid_mask.float()
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        output = torch.einsum('blhsd,blhs->blhd', values, attn_weights).reshape(B, chunk_len, C)
        
        se_weights = torch.sigmoid(self.se_fc2(F.silu(self.se_fc1(output))))
        output = output * se_weights
        
        return output
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B = x.shape[0]
        spatial = x.shape[1:-1]
        C = x.shape[-1]
        L = reduce(mul, spatial, 1)
        
        if self._triton_mod is not None and x.is_cuda:
            out, info = self._triton_mod(x.reshape(B, L, C))
            return F.silu(self.out_proj(out)).reshape(B, *spatial, C), info
        
        x_flat = x.reshape(B, L, C)
        
        if L <= self.chunk_size:
            output = self._forward_chunk(x_flat, x_flat, 0, L, spatial, L)
        else:
            outputs = []
            for start in range(0, L, self.chunk_size):
                end = min(start + self.chunk_size, L)
                chunk_out = self._forward_chunk(x_flat, x_flat[:, start:end], start, end, spatial, L)
                outputs.append(chunk_out)
            output = torch.cat(outputs, dim=1)
        
        output = F.silu(self.out_proj(output)).reshape(B, *spatial, C)
        
        return output, {}


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
        embed_down = avg_pool_nd(embed, (r,))
        processed_down = avg_pool_nd(processed, (r,))
        
        q = F.silu(self.q_proj(processed_down))
        k = F.silu(self.k_proj(embed_down))
        v = F.silu(self.v_proj(embed_down))
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = F.silu(self.out_proj(out))
        out = interpolate_nd(out, (L,))
        
        return processed + out


class LowRankAttentionMergeND(nn.Module):
    
    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
    
    def forward(self, embed: torch.Tensor, processed: torch.Tensor) -> torch.Tensor:
        spatial_shape = embed.shape[1:-1]
        B, C = embed.shape[0], embed.shape[-1]
        
        target_shape = tuple(max(1, int(s ** 0.5)) for s in spatial_shape)
        L = 1
        for t in target_shape:
            L *= t
        
        embed_down = avg_pool_nd(embed, target_shape).reshape(B, L, C)
        processed_down = avg_pool_nd(processed, target_shape).reshape(B, L, C)
        
        q = F.silu(self.q_proj(processed_down))
        k = F.silu(self.k_proj(embed_down))
        v = F.silu(self.v_proj(embed_down))
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = F.silu(self.out_proj(out))
        out = out.reshape(B, *target_shape, C)
        out = interpolate_nd(out, spatial_shape)
        
        return processed + out
    
    def forward_accumulated(
        self,
        x_full: torch.Tensor,
        accumulated: list[torch.Tensor],
    ) -> torch.Tensor:
        if len(accumulated) < 2:
            return x_full
        
        spatial_shape = x_full.shape[1:-1]
        B, C = x_full.shape[0], x_full.shape[-1]
        
        q_reduced = accumulated[-1]
        reduced_shape = q_reduced.shape[1:-1]
        r = 1
        for s in reduced_shape:
            r *= s
        
        kv_list = accumulated[:-1]
        kv_cat = torch.cat([t.reshape(B, r, C) for t in kv_list], dim=1)
        
        q = F.silu(self.q_proj(q_reduced.reshape(B, r, C)))
        k = F.silu(self.k_proj(kv_cat))
        v = F.silu(self.v_proj(kv_cat))
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = F.silu(self.out_proj(out))
        out = out.reshape(B, *reduced_shape, C)
        out = interpolate_nd(out, spatial_shape)
        
        return x_full + out


class LowRankAttention(nn.Module):
    """Differential Attention (Ye et al. 2025): DiffAttn = (softmax(Q1K1^T) - λ·softmax(Q2K2^T))V"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        reduction_power: float = 0.75,
        lambda_init: float = 0.8,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.reduction_power = reduction_power
        self.lambda_init = lambda_init
        
        assert embed_dim % (2 * num_heads) == 0, \
            f"embed_dim ({embed_dim}) must be divisible by 2*num_heads ({2*num_heads})"
        self.head_dim = embed_dim // (2 * num_heads)
        self.v_head_dim = 2 * self.head_dim
        self.scale = self.head_dim ** -0.5
        
        self.downsample = lambda x, shape: F.interpolate(
            x.transpose(1, 2), size=shape[0], mode='linear', align_corners=False
        ).transpose(1, 2)
        self.upsample = lambda x, shape: F.interpolate(
            x.transpose(1, 2), size=shape[0], mode='linear', align_corners=False
        ).transpose(1, 2)
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        
        # λ = exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim))
        
        self.head_norm = RMSNorm(self.v_head_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, C = x.shape
        H = self.num_heads
        d = self.head_dim
        r = max(1, int(L ** self.reduction_power))
        
        x_down = self.downsample(x, (r,))
        seq_len = x_down.shape[1]
        
        q = self.q_proj(x_down).reshape(B, seq_len, H, 2, d)
        k = self.k_proj(x_down).reshape(B, seq_len, H, 2, d)
        v = self.v_proj(x_down).reshape(B, seq_len, H, 2 * d)
        
        q1, q2 = q[..., 0, :], q[..., 1, :]
        k1, k2 = k[..., 0, :], k[..., 1, :]
        
        q1 = apply_rope(self.q_norm(q1))
        q2 = apply_rope(self.q_norm(q2))
        k1 = apply_rope(self.k_norm(k1))
        k2 = apply_rope(self.k_norm(k2))
        
        q1 = q1.transpose(1, 2)
        q2 = q2.transpose(1, 2)
        k1 = k1.transpose(1, 2)
        k2 = k2.transpose(1, 2)
        v = v.transpose(1, 2)
        
        lambda_val = (
            torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1))
            - torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2))
            + self.lambda_init
        )
        
        attn1 = F.softmax(q1 @ k1.transpose(-2, -1) * self.scale, dim=-1)
        attn2 = F.softmax(q2 @ k2.transpose(-2, -1) * self.scale, dim=-1)
        diff_attn = (attn1 - lambda_val * attn2) @ v
        
        diff_attn = diff_attn.transpose(1, 2)
        diff_attn = self.head_norm(diff_attn) * (1 - self.lambda_init)
        
        lowrank_out = diff_attn.reshape(B, seq_len, C)
        lowrank_out = F.silu(self.out_proj(lowrank_out))
        
        full_out = self.upsample(lowrank_out, (L,))
        
        return full_out, lowrank_out


class LowRankCollapseMerge(nn.Module):
    """Merge low-rank representations during collapse. Upsamples coarser to finer size."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.upsample = SIRENUpsampleND(embed_dim, ndim=1)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, finer: torch.Tensor, coarser: torch.Tensor) -> torch.Tensor:
        target_shape = finer.shape[1:-1]
        
        coarser_up = self.upsample(coarser, target_shape)
        
        q = self.q_proj(coarser_up)
        k = self.k_proj(finer)
        v = self.v_proj(finer)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = F.silu(self.out_proj(out))
        
        return coarser_up + out


class LowRankAttentionND(nn.Module):
    
    def __init__(self, embed_dim: int, window_size: int | tuple[int, ...] = 17, ndim: int = 1, num_channels: int = 1, reduction_power: float = 0.75):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.num_channels = num_channels
        self.channel_dim = embed_dim // num_channels
        self.reduction_power = reduction_power
        
        self.downsample = SIRENDownsampleND(embed_dim, ndim=ndim)
        self.upsample = SIRENUpsampleND(embed_dim, ndim=ndim)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[1:-1]
        B, C = x.shape[0], x.shape[-1]
        
        target_shape = tuple(max(1, int(s ** self.reduction_power)) for s in spatial_shape)
        
        x_down = self.downsample(x, target_shape)
        
        r = 1
        for t in target_shape:
            r *= t
        x_flat = x_down.reshape(B, r, C)
        
        q = F.silu(self.q_proj(x_flat))
        k = F.silu(self.k_proj(x_flat))
        v = F.silu(self.v_proj(x_flat))
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = F.silu(self.out_proj(out))
        
        out = out.reshape(B, *target_shape, C)
        out = self.upsample(out, spatial_shape)
        
        return out


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
    
    def __init__(self, embed_dim: int, kernel_size: int | tuple[int, ...] = 7, ndim: int = 1, num_channels: int = 1, checkpoint: bool = False, chunk_size: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.num_channels = num_channels
        self.channel_dim = embed_dim // num_channels
        self.checkpoint = checkpoint
        self.chunk_size = chunk_size
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim
        self.kernel_size = kernel_size
        self.half_k = tuple(k // 2 for k in kernel_size)
        self.window_size = 1
        for k in kernel_size:
            self.window_size *= k
        
        self.scale = self.channel_dim ** -0.5
        
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
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
        
        self._triton_mod = None
        if ndim == 1 and HAS_TRITON and TritonLocalWindowAttn is not None:
            self._triton_mod = TritonLocalWindowAttn(
                embed_dim=embed_dim,
                kernel_size=kernel_size[0],
                num_channels=num_channels,
            )
    
    def _unfold_nd(self, x: torch.Tensor) -> torch.Tensor:
        pad_dims = [0, 0]
        for k in reversed(self.kernel_size):
            pad_left = (k - 1) // 2
            pad_right = k // 2
            pad_dims.extend([pad_left, pad_right])
        x = F.pad(x, pad_dims)
        
        for dim in range(self.ndim):
            x = x.unfold(dim + 1, self.kernel_size[dim], 1)
        return x
    
    def _forward_chunk_1d(self, x_chunk: torch.Tensor, k_padded: torch.Tensor, v_padded: torch.Tensor,
                          chunk_start: int, L: int) -> torch.Tensor:
        B, chunk_len, C = x_chunk.shape
        H = self.num_channels
        D = self.channel_dim
        K = self.kernel_size[0]
        
        q = F.silu(self.q_proj(x_chunk))
        q_flat = self.q_norm(q.reshape(B, chunk_len, H, D))
        q_flat = apply_rope(q_flat)
        
        window_params = F.silu(self.window_proj(x_chunk))
        width_raw, sharpness_raw = window_params.chunk(2, dim=-1)
        width = width_raw.sigmoid() * self.max_dist + 0.5
        sharpness = sharpness_raw.sigmoid() * 9.5 + 0.5
        
        k_win = k_padded[:, chunk_start:chunk_start + chunk_len + K - 1].unfold(1, K, 1)
        v_win = v_padded[:, chunk_start:chunk_start + chunk_len + K - 1].unfold(1, K, 1)
        
        k_win = k_win.permute(0, 1, 3, 2).reshape(B, chunk_len, K, H, D).permute(0, 1, 3, 4, 2)
        v_win = v_win.permute(0, 1, 3, 2).reshape(B, chunk_len, K, H, D).permute(0, 1, 3, 4, 2)
        
        width_flat = width.reshape(B, chunk_len, H, 1)
        sharpness_flat = sharpness.reshape(B, chunk_len, H, 1)
        
        scores = torch.einsum('blhd,blhdw->blhw', q_flat, k_win) * self.scale
        
        soft_mask = torch.sigmoid((width_flat - self.rel_dist) * sharpness_flat)
        scores = scores - (1 - soft_mask) * 1e4
        
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('blhw,blhdw->blhd', attn, v_win).reshape(B, chunk_len, C)
        return out
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[1:-1]
        B = x.shape[0]
        C = x.shape[-1]
        L = 1
        for s in spatial_shape:
            L *= s
        
        if self._triton_mod is not None and x.is_cuda:
            return self._triton_mod(x.reshape(B, L, C)).reshape(B, *spatial_shape, C)
        
        H = self.num_channels
        D = self.channel_dim
        
        if self.ndim == 1 and L > self.chunk_size:
            x_flat = x.reshape(B, L, C)
            kv = F.silu(self.kv_proj(x_flat))
            k, v = kv.chunk(2, dim=-1)
            
            k_flat = self.k_norm(k.reshape(B, L, H, D))
            k_flat = apply_rope(k_flat)
            k = k_flat.reshape(B, L, C)
            
            K = self.kernel_size[0]
            pad_left = (K - 1) // 2
            pad_right = K // 2
            k_padded = F.pad(k, (0, 0, pad_left, pad_right))
            v_padded = F.pad(v, (0, 0, pad_left, pad_right))
            
            outputs = []
            for start in range(0, L, self.chunk_size):
                end = min(start + self.chunk_size, L)
                chunk_out = self._forward_chunk_1d(x_flat[:, start:end], k_padded, v_padded, start, L)
                outputs.append(chunk_out)
            out = torch.cat(outputs, dim=1)
            
            out = self.out_norm(out)
            merged = self.v_gate(v, out)
            return F.silu(self.out(merged)).reshape(B, *spatial_shape, C)
        else:
            q = F.silu(self.q_proj(x))
            kv = F.silu(self.kv_proj(x))
            k, v = kv.chunk(2, dim=-1)
            
            q = self.q_norm(q.reshape(B, *spatial_shape, H, D))
            k = self.k_norm(k.reshape(B, *spatial_shape, H, D))
            
            if self.ndim == 1:
                q = apply_rope(q)
                k = apply_rope(k)
            
            window_params = F.silu(self.window_proj(x))
            width_raw, sharpness_raw = window_params.chunk(2, dim=-1)
            width = (width_raw.sigmoid() * self.max_dist + 0.5).unsqueeze(-1)
            sharpness = (sharpness_raw.sigmoid() * 9.5 + 0.5).unsqueeze(-1)
            
            k_spatial = k.reshape(B, *spatial_shape, C)
            v_spatial = v.reshape(B, *spatial_shape, C)
            
            if self.ndim == 2:
                Hs, Ws = spatial_shape
                L = Hs * Ws
                k_cf = k_spatial.permute(0, 3, 1, 2)
                v_cf = v_spatial.permute(0, 3, 1, 2)
                k_unfold = F.unfold(k_cf, kernel_size=self.kernel_size, padding=self.half_k, stride=1)
                v_unfold = F.unfold(v_cf, kernel_size=self.kernel_size, padding=self.half_k, stride=1)
                k_win = k_unfold.view(B, H, D, self.window_size, L).permute(0, 4, 3, 1, 2)
                v_win = v_unfold.view(B, H, D, self.window_size, L).permute(0, 4, 3, 1, 2)
                q_flat = q.reshape(B, L, H, D)
                width_flat = width.reshape(B, L, H, 1)
                sharpness_flat = sharpness.reshape(B, L, H, 1)
                scores = torch.einsum('blhd,blkhd->blhk', q_flat, k_win) * self.scale
                soft_mask = torch.sigmoid((width_flat - self.rel_dist) * sharpness_flat)
                scores = scores - (1 - soft_mask) * 1e4
                attn = F.softmax(scores, dim=-1)
                out = torch.einsum('blhk,blkhd->blhd', attn, v_win).reshape(B, Hs, Ws, C)
            else:
                k_win = self._unfold_nd(k_spatial)
                v_win = self._unfold_nd(v_spatial)
                
                perm = [0] + list(range(1, self.ndim + 1)) + list(range(self.ndim + 2, self.ndim * 2 + 2)) + [self.ndim + 1]
                k_win = k_win.permute(*perm).reshape(B, *spatial_shape, self.window_size, H, D)
                v_win = v_win.permute(*perm).reshape(B, *spatial_shape, self.window_size, H, D)
                
                scores = torch.einsum('b...hd,b...whd->b...hw', q, k_win) * self.scale
                
                soft_mask = torch.sigmoid((width - self.rel_dist) * sharpness)
                scores = scores - (1 - soft_mask) * 1e4
                
                attn = F.softmax(scores, dim=-1)
                out = torch.einsum('b...hw,b...whd->b...hd', attn, v_win)
                out = out.reshape(B, *spatial_shape, C)
            
            out = self.out_norm(out)
            merged = self.v_gate(v_spatial, out)
            return F.silu(self.out(merged))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.checkpoint and self.training:
            return grad_checkpoint(self._forward_impl, x, use_reentrant=False)  # type: ignore[return-value]
        return self._forward_impl(x)


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


class SGSBAttentionND(nn.Module):
    
    def __init__(self, embed_dim: int, kernel_size: int | tuple[int, ...] = 17, ndim: int = 1, num_channels: int = 1, chunk_size: int = 2048):
        super().__init__()
        self.embed_dim = embed_dim
        self.ndim = ndim
        self.num_channels = num_channels
        self.chunk_size = chunk_size
        
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * ndim
        self.kernel_size = kernel_size
        
        self.scatter = AdaptiveConvND(embed_dim, ndim=ndim, max_samples=kernel_size[0], num_channels=num_channels, chunk_size=chunk_size)
        self.scatter_norm1 = RMSNorm(embed_dim)
        self.scatter_norm2 = RMSNorm(embed_dim)
        
        self.gather = LocalAttentionND(embed_dim, kernel_size, ndim, num_channels, chunk_size=chunk_size)
        self.gather_norm = RMSNorm(embed_dim)
        
        self.broadcast = LowRankAttention(embed_dim)
        self.broadcast_norm = RMSNorm(embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[1:-1]
        B, C = x.shape[0], x.shape[-1]
        L = 1
        for s in spatial_shape:
            L *= s
        
        h = x
        
        conv_out, _ = self.scatter(h)
        h = h + self.scatter_norm1(conv_out)
        
        h = h + self.gather_norm(self.gather(h))
        
        conv_out, _ = self.scatter(h)
        h = h + self.scatter_norm2(conv_out)
        
        h_flat = h.reshape(B, L, C)
        broadcast_out, _ = self.broadcast(h_flat)
        h = h + self.broadcast_norm(broadcast_out).reshape(B, *spatial_shape, C)
        
        return F.silu(self.out_proj(h))


class SGSBAttention(SGSBAttentionND):
    def __init__(self, embed_dim: int, kernel_size: int = 17, num_channels: int = 1, chunk_size: int = 2048):
        super().__init__(embed_dim, kernel_size, ndim=1, num_channels=num_channels, chunk_size=chunk_size)


class SGSBBlockND(nn.Module):
    
    def __init__(self, embed_dim: int, kernel_size: int | tuple[int, ...] = 17, ndim: int = 1, num_channels: int = 1, eps: float = 1e-6, chunk_size: int = 2048):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps)
        self.attn = SGSBAttentionND(embed_dim, kernel_size, ndim, num_channels, chunk_size=chunk_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class SGSBBlock(SGSBBlockND):
    def __init__(self, embed_dim: int, kernel_size: int = 17, num_channels: int = 1, eps: float = 1e-6):
        super().__init__(embed_dim, kernel_size, ndim=1, num_channels=num_channels, eps=eps)


class SGSBModelND(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        output_dim: int,
        n_layers: int = 4,
        kernel_size: int | tuple[int, ...] = 17,
        ndim: int = 1,
        num_channels: int = 4,
    ):
        super().__init__()
        self.ndim = ndim
        
        self.embed = nn.Linear(input_dim, embed_dim)
        self.embed_norm = RMSNorm(embed_dim)
        
        self.layers = nn.ModuleList([
            SGSBBlockND(embed_dim, kernel_size, ndim, num_channels)
            for _ in range(n_layers)
        ])
        
        self.out_norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_norm(F.silu(self.embed(x)))
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.out_norm(x)
        return self.head(x)


class SGSBModel(SGSBModelND):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        output_dim: int,
        n_layers: int = 4,
        kernel_size: int = 17,
        num_channels: int = 4,
    ):
        super().__init__(input_dim, embed_dim, output_dim, n_layers, kernel_size, ndim=1, num_channels=num_channels)


class SGSBClassifierND(nn.Module):
    
    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        n_layers: int = 4,
        kernel_size: int | tuple[int, ...] = 17,
        ndim: int = 1,
        num_channels: int = 4,
    ):
        super().__init__()
        self.ndim = ndim
        
        self.embed = nn.Linear(1, embed_dim)
        self.embed_norm = RMSNorm(embed_dim)
        
        self.layers = nn.ModuleList([
            SGSBBlockND(embed_dim, kernel_size, ndim, num_channels)
            for _ in range(n_layers)
        ])
        
        self.out_norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        x = self.embed_norm(F.silu(self.embed(x)))
        
        for layer in self.layers:
            x = layer(x)
        
        pool_dims = tuple(range(1, self.ndim + 1))
        x = self.out_norm(x).mean(dim=pool_dims)
        return self.head(x)


class SGSBClassifier(SGSBClassifierND):
    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        n_layers: int = 4,
        kernel_size: int = 17,
        num_channels: int = 4,
    ):
        super().__init__(embed_dim, n_classes, n_layers, kernel_size, ndim=1, num_channels=num_channels)


class MIMOJacobiSSM_ND(nn.Module):
    
    def __init__(
        self,
        dim: int,
        state_dim: int = 64,
        mimo_rank: int = 4,
        ndim: int = 2,
        n_iters: int = 12,
        diffuse_se: bool = False,
        diff_inject: bool = False,
        diff_readout: bool = False,
        bc_norm: bool = False,
        relu2: bool = False,
    ):
        super().__init__()
        self.D = dim
        self.N = state_dim
        self.R = mimo_rank
        self.ndim = ndim
        self.n_iters = n_iters
        self.diff_inject = diff_inject
        self.diff_readout = diff_readout
        self.bc_norm = bc_norm
        self.act = relu_squared if relu2 else F.silu
        
        self.to_B = nn.Linear(dim, state_dim * mimo_rank)
        self.to_C = nn.Linear(dim, state_dim * mimo_rank)
        self.to_X = nn.Linear(dim, mimo_rank)
        self.to_decay = nn.Linear(dim, state_dim)
        self.to_theta = nn.Linear(dim, state_dim // 2)
        self.to_lambda = nn.Linear(dim, 1)
        
        self.B_bias = nn.Parameter(torch.ones(state_dim * mimo_rank))
        self.C_bias = nn.Parameter(torch.ones(state_dim * mimo_rank))
        
        if diff_inject:
            self.inject_lambda = nn.Parameter(torch.tensor(0.5))
        if diff_readout:
            self.readout_lambda = nn.Parameter(torch.tensor(0.5))
        if bc_norm:
            self.b_norm = RMSNorm(state_dim)
            self.c_norm = RMSNorm(state_dim)
        
        if ndim == 1:
            self.diffuse = nn.Conv1d(state_dim, state_dim, kernel_size=3, padding=1, groups=state_dim)
        elif ndim == 2:
            self.diffuse = nn.Conv2d(state_dim, state_dim, kernel_size=3, padding=1, groups=state_dim)
        else:
            self.diffuse = nn.Conv3d(state_dim, state_dim, kernel_size=3, padding=1, groups=state_dim)
        self.diffuse_se = SqueezeExciteND(state_dim, relu2=relu2) if diffuse_se else None
        self._diffuse_ndim = ndim
        
        self.out_proj = nn.Linear(state_dim * mimo_rank, dim)
    
    def _apply_rope(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # x: (B, R, *spatial, N)
        # theta: (B, 1, *spatial, N//2) - broadcasts to R dim
        # x[..., ::2] gets even N indices (N//2), x[..., 1::2] gets odd (N//2)
        x1, x2 = x[..., ::2], x[..., 1::2]
        cos = theta.cos()
        sin = theta.sin()
        # Interleave back: x1*cos - x2*sin for even, x1*sin + x2*cos for odd
        out = torch.empty_like(x)
        out[..., ::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out
    
    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[1:-1]
        B = x.shape[0]
        # H layout: (B, R, *spatial, N) - R second for easy flatten to (B*R, *spatial, N)
        return torch.zeros(B, self.R, *spatial_shape, self.N, device=x.device, dtype=x.dtype)
    
    def _compute_inject(self, x: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute injection, decay, lambda, and C for a given iteration."""
        spatial_shape = x.shape[1:-1]
        B = x.shape[0]
        ndim = len(spatial_shape)

        B_proj = self.act(self.to_B(x) + self.B_bias).view(B, *spatial_shape, self.N, self.R)
        C_proj = self.act(self.to_C(x) + self.C_bias).view(B, *spatial_shape, self.N, self.R)
        perm_to_BR = (0, ndim + 2) + tuple(range(1, ndim + 1)) + (ndim + 1,)
        B_base = B_proj.permute(*perm_to_BR).contiguous()
        C_base = C_proj.permute(*perm_to_BR).contiguous()

        if self.bc_norm:
            B_base = self.b_norm(B_base)
            C_base = self.c_norm(C_base)

        X_r = self.act(self.to_X(x))
        decay = torch.sigmoid(self.to_decay(x))
        theta = self.to_theta(x)
        lam = torch.sigmoid(self.to_lambda(x))

        theta_k = theta * (layer_idx + 1)
        B_rot = self._apply_rope(B_base, theta_k.unsqueeze(1))
        C_rot = self._apply_rope(C_base, theta_k.unsqueeze(1))

        perm_Xr = (0, ndim + 1) + tuple(range(1, ndim + 1))
        X_r_bcast = X_r.permute(*perm_Xr).unsqueeze(-1)

        if self.diff_inject:
            half_N = self.N // 2
            B1, B2 = B_rot[..., :half_N], B_rot[..., half_N:]
            inject1 = B1 * X_r_bcast
            inject2 = B2 * X_r_bcast
            inject = torch.cat([inject1 - self.inject_lambda * inject2,
                                inject1 + self.inject_lambda * inject2], dim=-1)
        else:
            inject = B_rot * X_r_bcast

        return inject, decay, lam, C_rot

    def _diffuse_state(self, H: torch.Tensor, batch_size: int, spatial_shape: tuple) -> torch.Tensor:
        H_flat = H.view(batch_size * self.R, *spatial_shape, self.N)
        if self._diffuse_ndim == 1:
            H_flat = self.diffuse(H_flat.transpose(1, 2)).transpose(1, 2)
        elif self._diffuse_ndim == 2:
            H_flat = self.diffuse(H_flat.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        else:
            H_flat = self.diffuse(H_flat.permute(0, 4, 1, 2, 3)).permute(0, 2, 3, 4, 1)
        if self.diffuse_se is not None:
            H_flat = self.diffuse_se(H_flat)
        return H_flat.view(batch_size, self.R, *spatial_shape, self.N)

    def _readout(self, H: torch.Tensor, C_rot: torch.Tensor, spatial_shape: tuple, batch_size: int) -> torch.Tensor:
        ndim = len(spatial_shape)
        if self.diff_readout:
            half_N = self.N // 2
            C1, C2 = C_rot[..., :half_N], C_rot[..., half_N:]
            H1, H2 = H[..., :half_N], H[..., half_N:]
            gated1 = C1 * H1
            gated2 = C2 * H2
            H_gated = torch.cat([gated1 - self.readout_lambda * gated2,
                                 gated1 + self.readout_lambda * gated2], dim=-1)
        else:
            H_gated = C_rot * H
        perm_to_spatial = (0,) + tuple(range(2, ndim + 2)) + (1, ndim + 2)
        H_out = H_gated.permute(*perm_to_spatial).reshape(batch_size, *spatial_shape, self.N * self.R)
        return self.act(self.out_proj(H_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_shape = x.shape[1:-1]
        B = x.shape[0]
        H = self.init_state(x)
        prev_inject = None

        for i in range(self.n_iters):
            inject, decay, lam, C_rot = self._compute_inject(x, i)

            H = self._diffuse_state(H, B, spatial_shape)

            # Trapezoidal: h = α*H + β*prev_inject + γ*curr_inject
            # β = (1-λ)*α, γ = λ  (α folded into decay)
            alpha = decay.unsqueeze(1)
            gamma = lam.unsqueeze(1)  # (B, 1, *spatial, 1)
            if prev_inject is not None:
                beta = (1.0 - gamma) * alpha
                H = alpha * H + beta * prev_inject + gamma * inject
            else:
                H = alpha * H + inject

            prev_inject = inject

        out = self._readout(H, C_rot, spatial_shape, B)
        return out


class MIMOJacobiSSM(MIMOJacobiSSM_ND):
    
    def __init__(
        self,
        dim: int,
        state_dim: int = 64,
        mimo_rank: int = 4,
        n_iters: int = 12,
        diffuse_se: bool = False,
        diff_inject: bool = False,
        diff_readout: bool = False,
        bc_norm: bool = False,
        relu2: bool = False,
    ):
        super().__init__(dim, state_dim, mimo_rank, ndim=1, n_iters=n_iters, diffuse_se=diffuse_se, diff_inject=diff_inject, diff_readout=diff_readout, bc_norm=bc_norm, relu2=relu2)


class MIMOJacobiBlock_ND(nn.Module):
    
    def __init__(
        self,
        dim: int,
        state_dim: int = 64,
        mimo_rank: int = 4,
        ndim: int = 2,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = RMSNorm(dim, eps)
        self.ssm = MIMOJacobiSSM_ND(dim, state_dim, mimo_rank, ndim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ssm(self.norm(x))


class MIMOJacobiBlock(MIMOJacobiBlock_ND):
    
    def __init__(
        self,
        dim: int,
        state_dim: int = 64,
        mimo_rank: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__(dim, state_dim, mimo_rank, ndim=1, eps=eps)


class RippleLayerND(nn.Module):
    
    def __init__(
        self,
        dim: int,
        kernel_size: int | tuple[int, ...] = 17,
        ndim: int = 1,
        num_channels: int = 4,
        eps: float = 1e-6,
        checkpoint_merges: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        self.checkpoint_merges = checkpoint_merges
        
        self.norm1 = RMSNorm(dim, eps)
        self.norm2 = RMSNorm(dim, eps)
        self.state_norm = RMSNorm(dim, eps)
        
        self.attn = SGSBAttentionND(dim, kernel_size, ndim, num_channels)
        self.merge = LowRankAttentionMergeND(dim)
        
        self.accum_attn = LowRankAttentionMergeND(dim)
        self.reduce_norm = RMSNorm(dim, eps)
        self.downsample = SIRENDownsampleND(dim, ndim=ndim)
        
        self.state_gate = nn.Linear(dim, dim)
        nn.init.zeros_(self.state_gate.weight)
        nn.init.constant_(self.state_gate.bias, -2.0)
    
    def init_state(self, x: torch.Tensor) -> torch.Tensor:
        """Initialize per-layer hidden state (zeros, same shape as x)."""
        return torch.zeros_like(x)
    
    def _merge_fn(self, embed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.merge(embed, x)
    
    def _accum_fn(self, x: torch.Tensor, *accumulated: torch.Tensor) -> torch.Tensor:
        return self.accum_attn.forward_accumulated(x, list(accumulated))
    
    def forward(
        self,
        x: torch.Tensor,
        embed: torch.Tensor,
        ssm_out: torch.Tensor,
        accumulated: list[torch.Tensor],
        layer_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if len(accumulated) > 1:
            if self.checkpoint_merges and self.training:
                x = grad_checkpoint(self._accum_fn, x, *accumulated, use_reentrant=False)  # type: ignore[assignment]
            else:
                x = self.accum_attn.forward_accumulated(x, accumulated)
        
        x = x + self.attn(self.norm1(x))
        x = x + ssm_out
        
        if self.checkpoint_merges and self.training:
            x = grad_checkpoint(self._merge_fn, embed, x, use_reentrant=False)  # type: ignore[assignment]
        else:
            x = self.merge(embed, x)
        
        gate = torch.sigmoid(self.state_gate(x))
        layer_state = self.state_norm(layer_state + x)
        x = gate * layer_state + (1 - gate) * x
        
        spatial_shape = x.shape[1:-1]
        target_shape = tuple(max(1, int(s ** 0.5)) for s in spatial_shape)
        x_reduced = self.downsample(self.reduce_norm(x), target_shape)
        
        return x, x_reduced, layer_state


class RippleLayer(RippleLayerND):
    
    def __init__(
        self,
        dim: int,
        kernel_size: int = 17,
        num_channels: int = 4,
        eps: float = 1e-6,
    ):
        super().__init__(dim, kernel_size, ndim=1, num_channels=num_channels, eps=eps)


class RippleModelND(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        output_dim: int,
        n_layers: int = 4,
        kernel_size: int | tuple[int, ...] = 17,
        ndim: int = 1,
        num_channels: int = 4,
        state_dim: int = 64,
        mimo_rank: int = 4,
        ssm_scale: int = 2,
    ):
        super().__init__()
        self.ndim = ndim
        
        self.embed = nn.Linear(input_dim, embed_dim)
        self.embed_norm = RMSNorm(embed_dim)
        
        self.ssm = MIMOJacobiSSM_ND(embed_dim, state_dim * ssm_scale, mimo_rank * ssm_scale, ndim)
        self.ssm_norm = RMSNorm(embed_dim)
        
        self.layers = nn.ModuleList([
            RippleLayerND(embed_dim, kernel_size, ndim, num_channels)
            for _ in range(n_layers)
        ])
        
        self.out_norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.embed_norm(F.silu(self.embed(x)))
        
        x = embed
        H = self.ssm.init_state(x)
        layer_states = [layer.init_state(x) for layer in self.layers]
        accumulated: list[torch.Tensor] = []
        
        for i, layer in enumerate(self.layers):
            H, ssm_out = self.ssm.step(self.ssm_norm(x), H, i)
            x, x_reduced, layer_states[i] = layer(x, embed, ssm_out, accumulated, layer_states[i])
            accumulated.append(x_reduced)
        
        x = self.out_norm(x)
        return self.head(x)


class RippleModel(RippleModelND):
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        output_dim: int,
        n_layers: int = 4,
        kernel_size: int = 17,
        num_channels: int = 4,
        state_dim: int = 64,
        mimo_rank: int = 4,
        ssm_scale: int = 2,
    ):
        super().__init__(input_dim, embed_dim, output_dim, n_layers, kernel_size,
                         ndim=1, num_channels=num_channels, state_dim=state_dim,
                         mimo_rank=mimo_rank, ssm_scale=ssm_scale)


class RippleClassifierND(nn.Module):
    
    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        n_layers: int = 4,
        kernel_size: int | tuple[int, ...] = 17,
        ndim: int = 1,
        num_channels: int = 4,
        state_dim: int = 64,
        mimo_rank: int = 4,
        ssm_scale: int = 2,
    ):
        super().__init__()
        self.ndim = ndim
        
        self.embed = nn.Linear(1, embed_dim)
        self.embed_norm = RMSNorm(embed_dim)
        
        self.ssm = MIMOJacobiSSM_ND(embed_dim, state_dim * ssm_scale, mimo_rank * ssm_scale, ndim)
        self.ssm_norm = RMSNorm(embed_dim)
        
        self.layers = nn.ModuleList([
            RippleLayerND(embed_dim, kernel_size, ndim, num_channels)
            for _ in range(n_layers)
        ])
        
        self.out_norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        embed = self.embed_norm(F.silu(self.embed(x)))
        
        x = embed
        H = self.ssm.init_state(x)
        layer_states = [layer.init_state(x) for layer in self.layers]
        accumulated: list[torch.Tensor] = []
        
        for i, layer in enumerate(self.layers):
            H, ssm_out = self.ssm.step(self.ssm_norm(x), H, i)
            x, x_reduced, layer_states[i] = layer(x, embed, ssm_out, accumulated, layer_states[i])
            accumulated.append(x_reduced)
        
        pool_dims = tuple(range(1, self.ndim + 1))
        x = self.out_norm(x).mean(dim=pool_dims)
        return self.head(x)


class RippleClassifier(RippleClassifierND):
    
    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        n_layers: int = 4,
        kernel_size: int = 17,
        num_channels: int = 4,
        state_dim: int = 64,
        mimo_rank: int = 4,
        ssm_scale: int = 2,
    ):
        super().__init__(embed_dim, n_classes, n_layers, kernel_size, ndim=1,
                         num_channels=num_channels, state_dim=state_dim,
                         mimo_rank=mimo_rank, ssm_scale=ssm_scale)


class FlatRippleND(nn.Module):
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        output_dim: int,
        iterations: int = 8,
        kernel_size: int | tuple[int, ...] = 17,
        ndim: int = 1,
        num_channels: int = 4,
        state_dim: int = 64,
        mimo_rank: int = 4,
        ssm_scale: int = 2,
    ):
        super().__init__()
        self.ndim = ndim
        self.iterations = iterations
        
        self.embed = nn.Linear(input_dim, embed_dim)
        self.embed_norm = RMSNorm(embed_dim)
        
        self.ssm = MIMOJacobiSSM_ND(embed_dim, state_dim * ssm_scale, mimo_rank * ssm_scale, ndim)
        self.ssm_norm = RMSNorm(embed_dim)
        
        self.attn = SGSBAttentionND(embed_dim, kernel_size, ndim, num_channels)
        self.attn_norm = RMSNorm(embed_dim)
        
        self.merge = LowRankAttentionMergeND(embed_dim)
        
        self.state_norm = RMSNorm(embed_dim)
        self.state_gate = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.state_gate.weight)
        nn.init.constant_(self.state_gate.bias, -2.0)
        
        self.out_norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embed = self.embed_norm(F.silu(self.embed(x)))
        
        x = embed
        H = self.ssm.init_state(x)
        state = torch.zeros_like(x)
        
        for i in range(self.iterations):
            H, ssm_out = self.ssm.step(self.ssm_norm(x), H, i)
            x = x + self.attn(self.attn_norm(x))
            x = x + ssm_out
            x = self.merge(embed, x)
            
            gate = torch.sigmoid(self.state_gate(x))
            state = self.state_norm(state + x)
            x = gate * state + (1 - gate) * x
        
        x = self.out_norm(x)
        return self.head(x)


class FlatRipple(FlatRippleND):
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        output_dim: int,
        iterations: int = 8,
        kernel_size: int = 17,
        num_channels: int = 4,
        state_dim: int = 64,
        mimo_rank: int = 4,
        ssm_scale: int = 2,
    ):
        super().__init__(input_dim, embed_dim, output_dim, iterations, kernel_size,
                         ndim=1, num_channels=num_channels, state_dim=state_dim,
                         mimo_rank=mimo_rank, ssm_scale=ssm_scale)


class FlatRippleClassifierND(nn.Module):
    
    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        iterations: int = 8,
        kernel_size: int | tuple[int, ...] = 17,
        ndim: int = 1,
        num_channels: int = 4,
        state_dim: int = 64,
        mimo_rank: int = 4,
        ssm_scale: int = 2,
    ):
        super().__init__()
        self.ndim = ndim
        self.iterations = iterations
        
        self.embed = nn.Linear(1, embed_dim)
        self.embed_norm = RMSNorm(embed_dim)
        
        self.ssm = MIMOJacobiSSM_ND(embed_dim, state_dim * ssm_scale, mimo_rank * ssm_scale, ndim)
        self.ssm_norm = RMSNorm(embed_dim)
        
        self.attn = SGSBAttentionND(embed_dim, kernel_size, ndim, num_channels)
        self.attn_norm = RMSNorm(embed_dim)
        
        self.merge = LowRankAttentionMergeND(embed_dim)
        
        self.state_norm = RMSNorm(embed_dim)
        self.state_gate = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.state_gate.weight)
        nn.init.constant_(self.state_gate.bias, -2.0)
        
        self.out_norm = RMSNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(-1)
        embed = self.embed_norm(F.silu(self.embed(x)))
        
        x = embed
        H = self.ssm.init_state(x)
        state = torch.zeros_like(x)
        
        for i in range(self.iterations):
            H, ssm_out = self.ssm.step(self.ssm_norm(x), H, i)
            x = x + self.attn(self.attn_norm(x))
            x = x + ssm_out
            x = self.merge(embed, x)
            
            gate = torch.sigmoid(self.state_gate(x))
            state = self.state_norm(state + x)
            x = gate * state + (1 - gate) * x
        
        pool_dims = tuple(range(1, self.ndim + 1))
        x = self.out_norm(x).mean(dim=pool_dims)
        return self.head(x)


class FlatRippleClassifier(FlatRippleClassifierND):
    
    def __init__(
        self,
        embed_dim: int,
        n_classes: int,
        iterations: int = 8,
        kernel_size: int = 17,
        num_channels: int = 4,
        state_dim: int = 64,
        mimo_rank: int = 4,
        ssm_scale: int = 2,
    ):
        super().__init__(embed_dim, n_classes, iterations, kernel_size, ndim=1,
                         num_channels=num_channels, state_dim=state_dim,
                         mimo_rank=mimo_rank, ssm_scale=ssm_scale)


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
