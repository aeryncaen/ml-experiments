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


class InceptionScatterAttention(nn.Module):
    
    k_indices: torch.Tensor
    inception_k: torch.Tensor
    
    def __init__(
        self,
        embed_dim: int,
        scatter_channels: int = 4,
        num_samples: int = 16,
        expand_factor: int = 1,
        max_offset: float = 32.0,
        inception_kernel_size: int = 17,
        se_reduction: int = 4,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.scatter_channels = scatter_channels
        self.expand_factor = expand_factor
        self.inner_dim = embed_dim * expand_factor
        self.channel_dim = self.inner_dim // scatter_channels
        self.num_samples = num_samples
        self.max_offset = max_offset
        self.inception_kernel_size = inception_kernel_size
        self.rope_base = rope_base
        
        assert self.inner_dim % scatter_channels == 0, \
            f"inner_dim ({self.inner_dim}) must be divisible by scatter_channels ({scatter_channels})"
        
        K = num_samples
        SC = scatter_channels
        IK = inception_kernel_size
        
        if expand_factor > 1:
            self.up_proj = nn.Linear(embed_dim, self.inner_dim, bias=False)
            self.down_proj = nn.Linear(self.inner_dim, embed_dim, bias=False)
        else:
            self.up_proj = None
            self.down_proj = None
        
        self.base_deformation = nn.Parameter(torch.randn(SC) * 0.1)
        self.base_stride = nn.Parameter(torch.ones(SC) + torch.randn(SC) * 0.1)
        self.base_beta_fwd = nn.Parameter(torch.ones(SC) + torch.randn(SC) * 0.1)
        self.base_beta_bwd = nn.Parameter(torch.ones(SC) + torch.randn(SC) * 0.1)
        self.base_strength = nn.Parameter(torch.ones(SC) + torch.randn(SC) * 0.1)
        self.base_alpha_fwd = nn.Parameter(torch.ones(SC) + torch.randn(SC) * 0.1)
        self.base_alpha_bwd = nn.Parameter(torch.ones(SC) + torch.randn(SC) * 0.1)
        self.sample_bias = nn.Parameter(torch.randn(SC, num_samples) * 0.1)
        
        k_indices = torch.arange(num_samples, dtype=torch.float32) - num_samples // 2
        self.register_buffer('k_indices', k_indices)
        
        inception_k = torch.arange(IK, dtype=torch.float32) - IK // 2
        self.register_buffer('inception_k', inception_k)
        
        self.num_film_params = 7 + K
        film_hidden = SC * 4
        film_out = nn.Linear(film_hidden, SC * self.num_film_params)
        nn.init.normal_(film_out.weight, std=0.02)
        nn.init.normal_(film_out.bias, std=0.02)
        self.film_encoder = nn.Sequential(
            nn.Linear(self.inner_dim, film_hidden),
            nn.SiLU(),
            nn.Linear(film_hidden, film_hidden),
            nn.SiLU(),
            film_out,
        )
        
        self.value_proj = nn.Conv1d(self.inner_dim, self.inner_dim, 1, groups=SC, bias=False)
        
        self.se_scatter = SEBlock2d(SC, K, se_reduction)
        self.se_output = SEBlock1d(self.inner_dim, se_reduction)
    
    def _build_inception_kernel(self) -> torch.Tensor:
        SC = self.scatter_channels
        CD = self.channel_dim
        ik = self.inception_k
        
        stride = F.softplus(self.base_stride)
        beta_fwd = F.softplus(self.base_beta_fwd)
        beta_bwd = F.softplus(self.base_beta_bwd)
        
        sigma_fwd = (stride * beta_fwd).clamp(min=0.5)
        sigma_bwd = (stride * beta_bwd).clamp(min=0.5)
        
        ik_sq = ik ** 2
        gauss_fwd = torch.exp(-ik_sq / (2 * sigma_fwd.unsqueeze(1) ** 2))
        gauss_bwd = torch.exp(-ik_sq / (2 * sigma_bwd.unsqueeze(1) ** 2))
        
        kernel = torch.where(ik >= 0, gauss_fwd, gauss_bwd)
        kernel = kernel / kernel.sum(dim=1, keepdim=True).clamp(min=1e-6)
        
        kernel = kernel.unsqueeze(1).repeat_interleave(CD, dim=0)
        return kernel
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        SC = self.scatter_channels
        CD = self.channel_dim
        K = self.num_samples
        
        residual = x
        
        if self.up_proj is not None:
            x = self.up_proj(x)
        
        x = x + sinusoidal_pos_embed(L, self.inner_dim, x.device, x.dtype, self.rope_base)
        
        kernel = self._build_inception_kernel()
        x_t = x.transpose(1, 2)
        inception_ctx = F.conv1d(
            x_t, kernel, padding=self.inception_kernel_size // 2, groups=self.inner_dim
        ).transpose(1, 2)
        
        inception_ctx_rope = apply_rope(inception_ctx, self.rope_base)
        film_flat = self.film_encoder(inception_ctx_rope)
        film = film_flat.reshape(B, L, SC, self.num_film_params)
        
        d_deform = film[..., 0]
        d_stride = film[..., 1]
        d_beta_fwd = film[..., 2]
        d_beta_bwd = film[..., 3]
        d_strength = film[..., 4]
        d_alpha_fwd = film[..., 5]
        d_alpha_bwd = film[..., 6]
        d_sample = film[..., 7:]
        
        deformation_biased = (self.base_deformation + d_deform).clamp(-self.max_offset, self.max_offset)
        stride_biased = F.softplus(self.base_stride + d_stride)
        beta_fwd_biased = F.softplus(self.base_beta_fwd + d_beta_fwd)
        beta_bwd_biased = F.softplus(self.base_beta_bwd + d_beta_bwd)
        strength_biased = F.softplus(self.base_strength + d_strength)
        alpha_fwd_biased = F.softplus(self.base_alpha_fwd + d_alpha_fwd)
        alpha_bwd_biased = F.softplus(self.base_alpha_bwd + d_alpha_bwd)
        sample_bias_biased = (self.sample_bias + d_sample).tanh()
        sample_bias_biased = self.se_scatter(sample_bias_biased)
        
        positions, envelope = self._compute_positions_and_envelope(
            L, deformation_biased, stride_biased, beta_fwd_biased, beta_bwd_biased,
            strength_biased, alpha_fwd_biased, alpha_bwd_biased, sample_bias_biased
        )
        
        receiver_ctx = self._gather_at_positions(inception_ctx_rope, positions)
        sender_ctx = inception_ctx_rope.reshape(B, L, 1, 1, self.inner_dim).expand(B, L, SC, K, self.inner_dim)
        value_bias = (receiver_ctx * sender_ctx).sum(dim=-1, keepdim=True)
        
        x_rope = apply_rope(x, self.rope_base)
        values = self.value_proj(x_rope.transpose(1, 2)).transpose(1, 2)
        values = values.reshape(B, L, SC, 1, CD).expand(B, L, SC, K, CD)
        values = values + value_bias
        values = values * envelope.unsqueeze(-1)
        
        out = self._scatter_values_direct(positions, values.contiguous(), envelope, L)
        out = out.reshape(B, L, self.inner_dim)
        
        out = self.se_output(out, None)
        
        if self.down_proj is not None:
            out = self.down_proj(out)
        
        out = out + residual
        
        return out
    
    def _compute_positions_and_envelope(
        self,
        L: int,
        deformation: torch.Tensor,
        stride: torch.Tensor,
        beta_fwd: torch.Tensor,
        beta_bwd: torch.Tensor,
        strength: torch.Tensor,
        alpha_fwd: torch.Tensor,
        alpha_bwd: torch.Tensor,
        sample_bias: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = deformation.shape[0]
        SC = self.scatter_channels
        K = self.num_samples
        device = deformation.device
        dtype = deformation.dtype
        
        k = self.k_indices
        k_abs = k.abs()
        
        warp_fwd = (k_abs.reshape(1, 1, 1, K) ** beta_fwd.unsqueeze(-1)) * stride.unsqueeze(-1)
        warp_bwd = (k_abs.reshape(1, 1, 1, K) ** beta_bwd.unsqueeze(-1)) * stride.unsqueeze(-1)
        warp = torch.where(k >= 0, warp_fwd, -warp_bwd)
        
        l_idx = torch.arange(L, device=device, dtype=dtype).reshape(1, L, 1)
        centers = l_idx + deformation
        
        positions = centers.unsqueeze(-1) + warp
        positions = positions.clamp(0, L - 1)
        
        env_fwd = strength.unsqueeze(-1) / (1 + k_abs.reshape(1, 1, 1, K)) ** alpha_fwd.unsqueeze(-1)
        env_bwd = strength.unsqueeze(-1) / (1 + k_abs.reshape(1, 1, 1, K)) ** alpha_bwd.unsqueeze(-1)
        envelope = torch.where(k >= 0, env_fwd, env_bwd)
        
        envelope = envelope * (1 + sample_bias)
        
        return positions, envelope
    
    def _gather_at_positions(
        self,
        ctx: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        B, L, SC, K = positions.shape
        C = ctx.shape[-1]
        
        grid_x = (positions / (L - 1)) * 2 - 1
        grid = torch.stack([grid_x, torch.zeros_like(grid_x)], dim=-1)
        grid = grid.reshape(B, L * SC * K, 1, 2)
        
        ctx_2d = ctx.transpose(1, 2).unsqueeze(-1)
        
        gathered = F.grid_sample(ctx_2d, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        gathered = gathered.squeeze(-1).transpose(1, 2).reshape(B, L, SC, K, C)
        
        return gathered
    
    def _scatter_values_direct(
        self,
        positions: torch.Tensor,
        values: torch.Tensor,
        envelope: torch.Tensor,
        L: int,
    ) -> torch.Tensor:
        B, _, SC, K, CD = values.shape
        device = positions.device
        dtype = values.dtype
        
        pos_floor = positions.floor().long().clamp(0, L - 1)
        pos_ceil = (pos_floor + 1).clamp(0, L - 1)
        frac = positions - pos_floor.float()
        
        w_floor = (1 - frac) * envelope
        w_ceil = frac * envelope
        
        batch_idx = torch.arange(B, device=device).reshape(B, 1, 1, 1)
        chan_idx = torch.arange(SC, device=device).reshape(1, 1, SC, 1)
        
        idx_floor = (batch_idx * (L * SC) + pos_floor * SC + chan_idx).reshape(-1)
        idx_ceil = (batch_idx * (L * SC) + pos_ceil * SC + chan_idx).reshape(-1)
        
        values_flat = values.reshape(B * L * SC * K, CD)
        
        out = torch.zeros(B * L * SC, CD, device=device, dtype=dtype)
        weight_sum = torch.zeros(B * L * SC, 1, device=device, dtype=dtype)
        
        out.index_add_(0, idx_floor, values_flat * w_floor.reshape(-1, 1))
        out.index_add_(0, idx_ceil, values_flat * w_ceil.reshape(-1, 1))
        weight_sum.index_add_(0, idx_floor, w_floor.reshape(-1, 1))
        weight_sum.index_add_(0, idx_ceil, w_ceil.reshape(-1, 1))
        
        out = out / weight_sum.clamp(min=1e-6)
        return out.reshape(B, L, SC, CD)


class InceptionScatterBlock(nn.Module):
    
    def __init__(
        self,
        embed_dim: int,
        scatter_channels: int = 4,
        num_samples: int = 16,
        expand_factor: int = 1,
        max_offset: float = 32.0,
        inception_kernel_size: int = 17,
        se_reduction: int = 4,
        rope_base: float = 10000.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.norm = RMSNorm(embed_dim, eps)
        self.attn = InceptionScatterAttention(
            embed_dim=embed_dim,
            scatter_channels=scatter_channels,
            num_samples=num_samples,
            expand_factor=expand_factor,
            max_offset=max_offset,
            inception_kernel_size=inception_kernel_size,
            se_reduction=se_reduction,
            rope_base=rope_base,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.norm(x))
