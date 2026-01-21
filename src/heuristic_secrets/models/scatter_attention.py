"""
Scatter Attention: learned position warp + power law envelope + bilinear splat.
Dual of gather-attention: scatter weights to positions, average collisions, apply.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    B, L, C = x.shape
    half_c = C // 2
    device = x.device
    dtype = x.dtype
    
    pos = torch.arange(L, device=device, dtype=dtype)
    dim_idx = torch.arange(half_c, device=device, dtype=dtype)
    freqs = 1.0 / (base ** (dim_idx / half_c))
    
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    cos = angles.cos()
    sin = angles.sin()
    
    x1, x2 = x[..., :half_c], x[..., half_c:]
    
    x_rope = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)
    
    return x_rope


def sinusoidal_pos_embed(L: int, C: int, device: torch.device, dtype: torch.dtype, base: float = 10000.0) -> torch.Tensor:
    pos = torch.arange(L, device=device, dtype=dtype)
    dim_idx = torch.arange(C // 2, device=device, dtype=dtype)
    freqs = 1.0 / (base ** (2 * dim_idx / C))
    
    angles = pos.unsqueeze(1) * freqs.unsqueeze(0)
    
    embed = torch.zeros(L, C, device=device, dtype=dtype)
    embed[:, 0::2] = angles.sin()
    embed[:, 1::2] = angles.cos()
    
    return embed


class LocalAttention(nn.Module):
    
    rel_pos: torch.Tensor
    
    def __init__(self, embed_dim: int, kernel_size: int = 17):
        super().__init__()
        self.embed_dim = embed_dim
        self.kernel_size = kernel_size
        self.half_k = kernel_size // 2
        self.scale = embed_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.width_proj = nn.Linear(embed_dim, 1)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.register_buffer('rel_pos', torch.abs(torch.arange(kernel_size).float() - kernel_size // 2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        K = self.kernel_size
        
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        width = self.width_proj(x).sigmoid() * (K - 1) + 1
        
        k_pad = F.pad(k, (0, 0, self.half_k, self.half_k))
        v_pad = F.pad(v, (0, 0, self.half_k, self.half_k))
        
        k_win = k_pad.unfold(1, K, 1)  # (B, L, C, K)
        v_win = v_pad.unfold(1, K, 1)
        
        scores = torch.einsum('blc,blck->blk', q, k_win) * self.scale
        
        soft_mask = torch.sigmoid((width - self.rel_pos) * 5.0)
        scores = scores - (1 - soft_mask) * 1e4
        
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('blk,blck->blc', attn, v_win)
        
        return self.out(out + v)


class LocalBlock(nn.Module):
    
    def __init__(self, embed_dim: int, kernel_size: int = 17, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps)
        self.attn = LocalAttention(embed_dim, kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


class HierarchicalLocalAttention(nn.Module):
    
    def __init__(self, embed_dim: int, kernel_size: int = 17, n_levels: int = 4):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_levels = n_levels
        
        self.attn = LocalAttention(embed_dim, kernel_size)
        self.query = nn.Parameter(torch.randn(embed_dim * n_levels) * 0.02)
        self.film = nn.Linear(embed_dim * n_levels, embed_dim * 2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        levels = []
        h = x
        
        for i in range(self.n_levels):
            h = self.attn(h)
            levels.append(h)
            if i < self.n_levels - 1:
                h = F.avg_pool1d(h.mT, 2).mT
        
        upsampled = [levels[0]]
        for lvl in levels[1:]:
            upsampled.append(F.interpolate(lvl.mT, size=L, mode='nearest').mT)
        concat = torch.cat(upsampled, dim=-1)
        
        scores = concat @ self.query
        attn = F.softmax(scores, dim=-1)
        global_ctx = torch.einsum('bl,blc->bc', attn, concat)
        
        film_params = self.film(global_ctx)
        scale, bias = film_params.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)
        bias = bias.unsqueeze(1)
        
        return levels[0] * (1 + scale) + bias


class HierarchicalLocalBlock(nn.Module):
    
    def __init__(self, embed_dim: int, kernel_size: int = 17, n_levels: int = 4, eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps)
        self.attn = HierarchicalLocalAttention(embed_dim, kernel_size, n_levels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.attn(self.norm(x))


LocalKernelAttention = LocalAttention
LocalKernelBlock = LocalBlock
