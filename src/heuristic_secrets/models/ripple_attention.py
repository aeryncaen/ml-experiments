"""RippleAttention: configurable order of telephone, conv, lowrank with norm+residual."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .telephone_attention import TelephoneAttentionND
from .adaptive_local_conv import AdaptiveLocalConv
from .scatter_attention import LowRankAttention, RMSNorm, SIRENDownsampleND, SIRENUpsampleND, SqueezeExciteND, interpolate_nd, apply_rope, MIMOJacobiSSM, AdaptiveConvND

try:
    from .triton_adaptive_conv import TritonAdaptiveLocalConv, HAS_TRITON
except ImportError:
    HAS_TRITON = False
    TritonAdaptiveLocalConv = None


class LearnedSinusoidal2DEmbed(nn.Module):
    
    def __init__(
        self,
        height: int,
        width: int,
        embed_dim: int,
        n_freq: int = 32,
        vocab_size: int = 256,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.embed_dim = embed_dim
        self.n_freq = n_freq
        
        self.pixel_embed = nn.Embedding(vocab_size, embed_dim)
        
        self.freq_h = nn.Parameter(torch.randn(n_freq) * 0.1)
        self.freq_w = nn.Parameter(torch.randn(n_freq) * 0.1)
        self.phase_h = nn.Parameter(torch.zeros(n_freq))
        self.phase_w = nn.Parameter(torch.zeros(n_freq))
        
        self.pos_proj = nn.Linear(n_freq * 4, embed_dim)
        self.norm = RMSNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W = x.shape
        
        h_pos = torch.arange(H, device=x.device, dtype=torch.float) / H
        w_pos = torch.arange(W, device=x.device, dtype=torch.float) / W
        
        freq_h = F.softplus(self.freq_h) * 10
        freq_w = F.softplus(self.freq_w) * 10
        
        h_angles = h_pos.unsqueeze(-1) * freq_h + self.phase_h
        w_angles = w_pos.unsqueeze(-1) * freq_w + self.phase_w
        
        h_enc = torch.cat([h_angles.sin(), h_angles.cos()], dim=-1)
        w_enc = torch.cat([w_angles.sin(), w_angles.cos()], dim=-1)
        
        pos_enc = torch.cat([
            h_enc.unsqueeze(1).expand(H, W, -1),
            w_enc.unsqueeze(0).expand(H, W, -1),
        ], dim=-1)
        
        pos_embed = self.pos_proj(pos_enc)
        
        # Convert normalized floats [0,1] to pixel indices [0,255]
        x_int = (x * 255).clamp(0, 255).long()
        x_flat = x_int.reshape(B, H * W)
        x_embed = self.pixel_embed(x_flat)
        
        x_embed = x_embed + pos_embed.reshape(1, H * W, self.embed_dim)
        
        return self.norm(x_embed)


class LayerHistoryAccumulator(nn.Module):
    """Downsamples layer output to L^reduction_power for cross-layer history accumulation.
    
    Placed at the END of each layer (after normalization). Produces a low-rank
    representation that gets appended to the cross-layer history stack.
    """
    
    def __init__(
        self,
        channels: int,
        reduction_power: float = 0.75,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.reduction_power = reduction_power
        self.downsample = SIRENDownsampleND(channels, ndim=1)
        self.norm = RMSNorm(channels, eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample and normalize layer output for history accumulation.
        
        Args:
            x: (B, L, C) layer output after normalization
            
        Returns:
            (B, L^reduction_power, C) low-rank representation
        """
        B, L, C = x.shape
        target_len = max(1, int(L ** self.reduction_power))
        x_down = self.downsample(x, (target_len,))
        return self.norm(x_down)


class CausalSelfAttention(nn.Module):
    """Differential causal self-attention (Ye et al., ICLR 2025), usable as a channel op.

    Computes attention as: (softmax(Q1K1^T) - λ·softmax(Q2K2^T)) · V
    where Q,K are split into two halves and λ is a learnable scalar.
    """

    def __init__(self, channels: int, num_heads: int = 1, dropout: float = 0.1, layer_idx: int = 0, differential: bool = True):
        super().__init__()
        import math as _math
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.half_dim = self.head_dim // 2
        self.dropout = dropout
        self.differential = differential
        self.lambda_init = 0.8 - 0.6 * _math.exp(-0.3 * layer_idx)

        self.qkv = nn.Linear(channels, 3 * channels, bias=True)
        if differential:
            self.q_norm = RMSNorm(self.half_dim)
            self.k_norm = RMSNorm(self.half_dim)
            self.head_norm = RMSNorm(self.head_dim)

            # λ reparameterization: exp(λ_q1·λ_k1) - exp(λ_q2·λ_k2) + λ_init
            self.lambda_q1 = nn.Parameter(torch.randn(self.half_dim) * 0.1)
            self.lambda_k1 = nn.Parameter(torch.randn(self.half_dim) * 0.1)
            self.lambda_q2 = nn.Parameter(torch.randn(self.half_dim) * 0.1)
            self.lambda_k2 = nn.Parameter(torch.randn(self.half_dim) * 0.1)
        else:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.out_proj = nn.Linear(channels, channels, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        B, L, C = x.shape
        H, D, D2 = self.num_heads, self.head_dim, self.half_dim

        qkv = F.silu(self.qkv(x))
        q, k, v = qkv.reshape(B, L, 3, H, D).unbind(2)
        drop_p = self.dropout if self.training else 0.0

        if self.differential:
            q1, q2 = q[..., :D2], q[..., D2:]
            k1, k2 = k[..., :D2], k[..., D2:]

            q1 = self.q_norm(q1).transpose(1, 2)
            q2 = self.q_norm(q2).transpose(1, 2)
            k1 = self.k_norm(k1).transpose(1, 2)
            k2 = self.k_norm(k2).transpose(1, 2)
            v = v.transpose(1, 2)

            lam = (torch.exp(torch.dot(self.lambda_q1, self.lambda_k1))
                   - torch.exp(torch.dot(self.lambda_q2, self.lambda_k2))
                   + self.lambda_init)

            out1 = F.scaled_dot_product_attention(q1, k1, v, is_causal=True, dropout_p=drop_p)
            out2 = F.scaled_dot_product_attention(q2, k2, v, is_causal=True, dropout_p=drop_p)

            diff = out1 - lam * out2
            diff = self.head_norm(diff) * (1 - self.lambda_init)
            out = diff.transpose(1, 2).reshape(B, L, C)
        else:
            q = self.q_norm(q).transpose(1, 2)
            k = self.k_norm(k).transpose(1, 2)
            v = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)
            out = out.transpose(1, 2).reshape(B, L, C)

        return F.silu(self.out_proj(out)), {}


class CrossLayerAttention(nn.Module):
    """Attends across accumulated layer history at the START of each layer.
    
    If history has 2+ entries:
    - Q = most recent entry (top of stack)
    - K/V = all prior entries
    - Full attention with RoPE encoding layer depth
    - Interpolate result back to full sequence length
    - Add as residual to layer input
    """
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps)
        self.k_norm = RMSNorm(self.head_dim, eps)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.out_norm = RMSNorm(channels, eps)
        self.upsample = SIRENUpsampleND(channels, ndim=1)
    
    def _apply_depth_rope(self, x: torch.Tensor, depth: int, base: float = 10000.0) -> torch.Tensor:
        """Apply same rotation to all positions based on a scalar depth."""
        D = x.shape[-1]
        half_d = D // 2
        device = x.device
        dtype = x.dtype
        
        dim_idx = torch.arange(half_d, device=device, dtype=dtype)
        freqs = 1.0 / (base ** (dim_idx / half_d))
        angles = depth * freqs
        cos = angles.cos()
        sin = angles.sin()
        
        x1, x2 = x[..., :half_d], x[..., half_d:half_d * 2]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        history: list[torch.Tensor],
    ) -> torch.Tensor:
        """Apply cross-layer attention if history has 2+ entries."""
        if len(history) < 2:
            return x
        
        B, L, C = x.shape
        H = self.num_heads
        D = self.head_dim
        
        q_src = history[-1]
        L_q = q_src.shape[1]
        q_depth = len(history) - 1
        
        kv_list = history[:-1]
        
        q = self.q_proj(q_src).reshape(B, L_q, H, D)
        q = self._apply_depth_rope(self.q_norm(q), q_depth)
        
        k_parts = []
        v_parts = []
        for depth_idx, kv in enumerate(kv_list):
            L_kv = kv.shape[1]
            k_i = self.k_proj(kv).reshape(B, L_kv, H, D)
            v_i = self.v_proj(kv).reshape(B, L_kv, H, D)
            k_parts.append(self._apply_depth_rope(self.k_norm(k_i), depth_idx))
            v_parts.append(v_i)
        
        k = torch.cat(k_parts, dim=1)
        v = torch.cat(v_parts, dim=1)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L_q, C)
        out = self.out_proj(out)
        
        out = self.upsample(out, (L,))
        
        return x + self.out_norm(out)


class RippleAttention(nn.Module):
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        max_kernel_size: int = None,
        max_freq: float = 16.0,
        min_freq: float = 1.0,
        chunk_size: int = 1024,
        use_triton: bool = True,
        eps: float = 1e-6,
        order: str = "tele,conv,lowrank",
        lowrank_power: float = 0.75,
        telephone_power: float = 0.5625,
        conv_power: float = 0.421875,
        max_seq_len: int = 8192,
        jacobi_iters: int = 12,
        siren_conv: bool = False,
        differential: bool = True,
        embed_residual: bool = True,
        plain_conv_size: int = 0,
        diffuse_se: bool = False,
    ):
        super().__init__()
        import math
        if max_kernel_size is None:
            max_kernel_size = 16
        self.channels = channels
        self.num_heads = num_heads
        self.embed_residual = embed_residual
        self.order = [s.strip() for s in order.split(",")]
        
        unique_ops = list(dict.fromkeys(self.order))
        
        if 'tele' in unique_ops:
            self.telephone = TelephoneAttentionND(
                channels=channels,
                ndim=1,
                num_heads=num_heads,
                max_freq=max_freq,
                min_freq=min_freq,
                max_kernel_size=max_kernel_size,
                chunk_size=chunk_size,
                use_triton=use_triton,
                scale_power=telephone_power,
                max_seq_len=max_seq_len,
            )
        
        if 'conv' in unique_ops:
            if plain_conv_size > 0:
                self.conv = nn.Conv1d(channels, channels, plain_conv_size, padding=plain_conv_size // 2, groups=channels)
                self._plain_conv = True
            elif siren_conv:
                import math
                siren_samples = int(math.sqrt(max_seq_len))
                self.conv = AdaptiveConvND(channels, ndim=1, max_samples=siren_samples, num_channels=num_heads)
                self._plain_conv = False
            elif use_triton and HAS_TRITON and TritonAdaptiveLocalConv is not None:
                self.conv = TritonAdaptiveLocalConv(
                    channels=channels,
                    num_heads=num_heads,
                    max_kernel_size=max_kernel_size,
                    chunk_size=chunk_size,
                    scale_power=conv_power,
                )
                self._plain_conv = False
            else:
                self.conv = AdaptiveLocalConv(
                    channels=channels,
                    num_heads=num_heads,
                    max_kernel_size=max_kernel_size,
                    scale_power=conv_power,
                )
                self._plain_conv = False
        
        if 'lowrank' in unique_ops:
            self.lowrank = LowRankAttention(
                channels,
                num_heads=num_heads,
                reduction_power=lowrank_power,
            )
        
        if 'attn' in unique_ops:
            self.attn_op = CausalSelfAttention(channels, num_heads=num_heads, differential=differential)

        if 'jacobi' in unique_ops:
            self.jacobi = MIMOJacobiSSM(channels, n_iters=jacobi_iters, diffuse_se=diffuse_se)
        
        self.norms = nn.ModuleDict({
            name: RMSNorm(channels, eps)
            for name in unique_ops
        })
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        h = x
        info = {}
        
        for name in self.order:
            if name == 'tele':
                out, _ = self.telephone(h)
            elif name == 'conv':
                if getattr(self, '_plain_conv', False):
                    out = self.conv(h.transpose(1, 2)).transpose(1, 2)
                else:
                    out, info = self.conv(h)
            elif name == 'lowrank':
                out, _ = self.lowrank(h)
            elif name == 'attn':
                out, _ = self.attn_op(h)
            elif name == 'jacobi':
                out = self.jacobi(h)
            else:
                raise ValueError(f"Unknown layer: {name}")
            h = h + self.norms[name](out)
        
        return (x + h, info) if self.embed_residual else (h, info)


class RippleBlock(nn.Module):
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        max_kernel_size: int = 64,
        mlp_ratio: float = 4.0,
        use_triton: bool = True,
        eps: float = 1e-6,
        order: str = "tele,conv,lowrank",
        lowrank_power: float = 0.75,
        telephone_power: float = 0.5625,
        conv_power: float = 0.421875,
        max_seq_len: int = 8192,
        cross_layer: bool = False,
        jacobi_iters: int = 12,
        siren_conv: bool = False,
    ):
        super().__init__()
        self.cross_layer = cross_layer
        self.attn = RippleAttention(
            channels=channels,
            num_heads=num_heads,
            max_kernel_size=max_kernel_size,
            use_triton=use_triton,
            eps=eps,
            order=order,
            lowrank_power=lowrank_power,
            telephone_power=telephone_power,
            conv_power=conv_power,
            max_seq_len=max_seq_len,
            jacobi_iters=jacobi_iters,
            siren_conv=siren_conv,
        )
        
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
        )
        
        if cross_layer:
            self.cross_layer_attn = CrossLayerAttention(channels, num_heads, eps)
            self.history_accum = LayerHistoryAccumulator(channels, lowrank_power, eps)
            self.decay_head = nn.Sequential(
                nn.Linear(channels, channels // 4),
                nn.SiLU(),
                nn.Linear(channels // 4, 1),
            )
    
    def forward(
        self,
        x: torch.Tensor,
        history: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict] | tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor]:
        if self.cross_layer and history is not None:
            x = self.cross_layer_attn(x, history)
        
        attn_out, info = self.attn(x)
        x = x + attn_out
        x = x + self.mlp(x)
        
        if self.cross_layer:
            x_lowrank = self.history_accum(x)
            decay = 0.5 + 0.5 * torch.sigmoid(self.decay_head(x.mean(dim=1))).unsqueeze(-1)
            return x, info, x_lowrank, decay
        
        return x, info


class RippleClassifier(nn.Module):
    """Sequence classifier with RippleBlocks and optional cross-layer attention."""
    
    def __init__(
        self,
        width: int,
        n_layers: int,
        n_classes: int,
        seq_len: int,
        num_heads: int = 8,
        max_kernel_size: int = 64,
        mlp_ratio: float = 4.0,
        use_triton: bool = True,
        eps: float = 1e-6,
        order: str = "tele,conv,lowrank",
        lowrank_power: float = 0.75,
        telephone_power: float = 0.5625,
        conv_power: float = 0.421875,
        max_seq_len: int = 8192,
        cross_layer: bool = False,
        embed_2d: tuple[int, int] | None = None,
        vocab_size: int | None = None,
        jacobi_iters: int = 12,
        siren_conv: bool = False,
    ):
        super().__init__()
        self.cross_layer = cross_layer
        self.lowrank_power = lowrank_power
        self.embed_2d = embed_2d
        self.vocab_size = vocab_size
        
        if embed_2d is not None:
            height, width_2d = embed_2d
            self.embed = LearnedSinusoidal2DEmbed(height, width_2d, width)
        elif vocab_size is not None:
            self.embed = nn.Embedding(vocab_size, width)
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, width) * 0.02)
            self.embed_norm = RMSNorm(width, eps)
        else:
            self.embed = nn.Linear(1, width)
            self.embed_norm = RMSNorm(width, eps)
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, width) * 0.02)
            self.pos_norm = RMSNorm(width, eps)
        
        self.layers = nn.ModuleList([
            RippleBlock(
                channels=width,
                num_heads=num_heads,
                max_kernel_size=max_kernel_size,
                mlp_ratio=mlp_ratio,
                use_triton=use_triton,
                eps=eps,
                order=order,
                lowrank_power=lowrank_power,
                telephone_power=telephone_power,
                conv_power=conv_power,
                max_seq_len=max_seq_len,
                cross_layer=cross_layer,
                jacobi_iters=jacobi_iters,
                siren_conv=siren_conv,
            )
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(width, eps)
        self.head = nn.Linear(width, n_classes)
        
        if cross_layer:
            self.embed_accum = LayerHistoryAccumulator(width, lowrank_power, eps)
            self.head_cross_attn = CrossLayerAttention(width, num_heads, eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embed_2d is not None:
            x = self.embed(x)
        elif self.vocab_size is not None:
            x_long = x.long()
            assert x_long.min() >= 0 and x_long.max() < self.vocab_size, \
                f"Token indices out of bounds: min={x_long.min().item()}, max={x_long.max().item()}, vocab_size={self.vocab_size}"
            x = self.embed_norm(self.embed(x_long) + self.pos_embed)
        else:
            x = self.embed_norm(F.silu(self.embed(x.unsqueeze(-1)))) + self.pos_norm(F.silu(self.pos_embed))
        
        if self.cross_layer:
            history: list[torch.Tensor] = [self.embed_accum(x)]
            
            for layer in self.layers:
                x, _, x_lowrank, decay = layer(x, history)
                history = [h * decay for h in history]
                history.append(x_lowrank)
            
            x = self.head_cross_attn(x, history)
        else:
            for layer in self.layers:
                x, _ = layer(x)
        
        x = self.norm(x)
        if self.vocab_size is not None:
            return self.head(x)
        return self.head(x.mean(dim=1))


def _make_channel_op(
    name: str,
    channels: int,
    num_heads: int,
    max_kernel_size: int,
    use_triton: bool,
    lowrank_power: float,
    telephone_power: float,
    conv_power: float,
    max_seq_len: int,
    chunk_size: int = 1024,
    max_freq: float = 16.0,
    min_freq: float = 1.0,
    siren_conv: bool = False,
    jacobi_iters: int = 12,
) -> nn.Module:
    if name == 'tele':
        return TelephoneAttentionND(
            channels=channels, ndim=1, num_heads=num_heads,
            max_freq=max_freq, min_freq=min_freq,
            max_kernel_size=max_kernel_size, chunk_size=chunk_size,
            use_triton=use_triton, scale_power=telephone_power,
            max_seq_len=max_seq_len,
        )
    elif name == 'conv':
        if siren_conv:
            from math import sqrt
            return AdaptiveConvND(channels, ndim=1, max_samples=int(sqrt(max_seq_len)), num_channels=num_heads)
        if use_triton and HAS_TRITON and TritonAdaptiveLocalConv is not None:
            return TritonAdaptiveLocalConv(
                channels=channels, num_heads=num_heads,
                max_kernel_size=max_kernel_size, chunk_size=chunk_size,
                scale_power=conv_power,
            )
        return AdaptiveLocalConv(
            channels=channels, num_heads=num_heads,
            max_kernel_size=max_kernel_size, scale_power=conv_power,
        )
    elif name == 'lowrank':
        return LowRankAttention(
            channels, num_heads=num_heads, reduction_power=lowrank_power,
        )
    elif name == 'attn':
        return CausalSelfAttention(channels, num_heads=num_heads)
    elif name == 'jacobi':
        return MIMOJacobiSSM(channels, n_iters=jacobi_iters)
    else:
        raise ValueError(f"Unknown op: {name}")


class ChannelSegment(nn.Module):
    
    def __init__(
        self,
        op_names: list[str],
        n_channels: int,
        channel_width: int,
        num_heads: int = 8,
        max_kernel_size: int = 64,
        use_triton: bool = True,
        eps: float = 1e-6,
        lowrank_power: float = 0.75,
        telephone_power: float = 0.5625,
        conv_power: float = 0.421875,
        max_seq_len: int = 8192,
        siren_conv: bool = False,
        jacobi_iters: int = 12,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.channel_width = channel_width
        self.op_names = op_names
        
        self.channels = nn.ModuleList()
        for _ in range(n_channels):
            ops = nn.ModuleList()
            norms = nn.ModuleList()
            for name in op_names:
                ops.append(_make_channel_op(
                    name, channel_width, num_heads, max_kernel_size,
                    use_triton, lowrank_power, telephone_power, conv_power,
                    max_seq_len, siren_conv=siren_conv, jacobi_iters=jacobi_iters,
                ))
                norms.append(RMSNorm(channel_width, eps))
            self.channels.append(nn.ModuleDict({'ops': ops, 'norms': norms}))
    
    def forward(self, x: torch.Tensor, routing_weights: torch.Tensor | None = None) -> torch.Tensor:
        B, L, C = x.shape
        
        if routing_weights is not None:
            active_mask = routing_weights > 0
            active_indices = [i for i in range(self.n_channels) if active_mask[0, i].item()]
            n_active = len(active_indices)
            slices = x.split(self.channel_width, dim=-1)
            
            outputs = []
            for slot, ch_idx in enumerate(active_indices):
                ch = self.channels[ch_idx]
                h = slices[slot]
                w = routing_weights[:, ch_idx].unsqueeze(1).unsqueeze(2)
                for op, norm in zip(ch['ops'], ch['norms']):
                    out, _ = op(h)
                    h = h + norm(out)
                outputs.append(h * w)
            
            return torch.cat(outputs, dim=-1)
        else:
            slices = x.split(self.channel_width, dim=-1)
            outputs = []
            for i, ch in enumerate(self.channels):
                h = slices[i]
                for op, norm in zip(ch['ops'], ch['norms']):
                    out, _ = op(h)
                    h = h + norm(out)
                outputs.append(h)
            return torch.cat(outputs, dim=-1)


class ChannelMixer(nn.Module):
    
    def __init__(self, width: int, n_channels: int = 0, mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.se = SqueezeExciteND(width)
        self.norm = RMSNorm(width, eps)
        hidden = int(width * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, width),
        )
        self.mlp_norm = RMSNorm(width, eps)
        self.router = nn.Linear(width, n_channels) if n_channels > 0 else None
    
    def forward(self, x: torch.Tensor, top_k: int = 0) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = x + self.norm(self.se(x) - x)
        x = x + self.mlp_norm(self.mlp(x))
        if self.router is not None and top_k > 0:
            pooled = x.mean(dim=1)
            logits = self.router(pooled)
            routing_weights = self._top_k_route(logits, top_k)
            return x, routing_weights
        return x
    
    def _top_k_route(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        topk_vals, topk_idx = logits.topk(k, dim=-1)
        weights = torch.zeros_like(logits)
        weights.scatter_(1, topk_idx, F.softmax(topk_vals, dim=-1))
        return weights


class RippleChannelClassifier(nn.Module):
    
    def __init__(
        self,
        width: int,
        n_channels: int,
        n_classes: int,
        seq_len: int,
        topology: str = "tele,conv,lowrank,se,tele,conv,lowrank,se",
        num_heads: int = 8,
        max_kernel_size: int = 64,
        mlp_ratio: float = 4.0,
        use_triton: bool = True,
        eps: float = 1e-6,
        lowrank_power: float = 0.75,
        telephone_power: float = 0.5625,
        conv_power: float = 0.421875,
        max_seq_len: int = 8192,
        embed_2d: tuple[int, int] | None = None,
        vocab_size: int | None = None,
        router_top_k: int = 0,
        siren_conv: bool = False,
        jacobi_iters: int = 12,
    ):
        super().__init__()
        self.embed_2d = embed_2d
        self.vocab_size = vocab_size
        self.router_top_k = router_top_k
        self.n_channels = n_channels
        
        if router_top_k > 0:
            assert width % router_top_k == 0, f"width ({width}) must be divisible by router_top_k ({router_top_k})"
            channel_width = width // router_top_k
        else:
            assert width % n_channels == 0, f"width ({width}) must be divisible by n_channels ({n_channels})"
            channel_width = width // n_channels
        
        if embed_2d is not None:
            height, width_2d = embed_2d
            self.embed = LearnedSinusoidal2DEmbed(height, width_2d, width)
        elif vocab_size is not None:
            self.embed = nn.Embedding(vocab_size, width)
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, width) * 0.02)
            self.embed_norm = RMSNorm(width, eps)
        else:
            self.embed = nn.Linear(1, width)
            self.embed_norm = RMSNorm(width, eps)
            self.pos_embed = nn.Parameter(torch.randn(1, seq_len, width) * 0.02)
            self.pos_norm = RMSNorm(width, eps)
        
        tokens = [t.strip() for t in topology.split(",")]
        
        self.stages = nn.ModuleList()
        current_ops: list[str] = []
        for token in tokens:
            if token == 'se':
                if current_ops:
                    self.stages.append(ChannelSegment(
                        current_ops, n_channels, channel_width, num_heads,
                        max_kernel_size, use_triton, eps, lowrank_power,
                        telephone_power, conv_power, max_seq_len,
                        siren_conv=siren_conv, jacobi_iters=jacobi_iters,
                    ))
                    current_ops = []
                self.stages.append(ChannelMixer(width, n_channels=n_channels if router_top_k > 0 else 0, mlp_ratio=mlp_ratio, eps=eps))
            else:
                current_ops.append(token)
        if current_ops:
            self.stages.append(ChannelSegment(
                current_ops, n_channels, channel_width, num_heads,
                max_kernel_size, use_triton, eps, lowrank_power,
                telephone_power, conv_power, max_seq_len,
                siren_conv=siren_conv, jacobi_iters=jacobi_iters,
            ))
        
        self.norm = RMSNorm(width, eps)
        self.head = nn.Linear(width, n_classes)
        
        if router_top_k > 0:
            self.embed_router = nn.Linear(width, n_channels)
        else:
            self.embed_router = None
    
    def _top_k_route(self, logits: torch.Tensor) -> torch.Tensor:
        topk_vals, topk_idx = logits.topk(self.router_top_k, dim=-1)
        weights = torch.zeros_like(logits)
        weights.scatter_(1, topk_idx, F.softmax(topk_vals, dim=-1))
        return weights
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embed_2d is not None:
            x = self.embed(x)
        elif self.vocab_size is not None:
            x_long = x.long()
            x = self.embed_norm(self.embed(x_long) + self.pos_embed)
        else:
            x = self.embed_norm(F.silu(self.embed(x.unsqueeze(-1)))) + self.pos_norm(F.silu(self.pos_embed))
        
        routing = None
        if self.embed_router is not None:
            routing = self._top_k_route(self.embed_router(x.mean(dim=1)))
        
        for stage in self.stages:
            if isinstance(stage, ChannelSegment):
                x = stage(x, routing_weights=routing)
                routing = None
            elif isinstance(stage, ChannelMixer):
                result = stage(x, top_k=self.router_top_k)
                if isinstance(result, tuple):
                    x, routing = result
                else:
                    x = result
            else:
                x = stage(x)
        
        x = self.norm(x)
        if self.vocab_size is not None:
            return self.head(x)
        return self.head(x.mean(dim=1))


if __name__ == "__main__":
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"HAS_TRITON: {HAS_TRITON}")
    print()
    
    B, L, C = 4, 1024, 256
    H = 8
    
    print(f"Config: B={B}, L={L}, C={C}, H={H}")
    print()
    
    model = RippleAttention(channels=C, num_heads=H, use_triton=True).to(device)
    model.train()
    
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    
    torch.manual_seed(42)
    x = torch.randn(B, L, C, device=device)
    
    out, info = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output has NaN: {torch.isnan(out).any().item()}")
    print(f"Output has Inf: {torch.isinf(out).any().item()}")
    
    x_grad = x.clone().requires_grad_(True)
    out, _ = model(x_grad)
    out.sum().backward()
    
    assert x_grad.grad is not None
    print(f"Input grad has NaN: {torch.isnan(x_grad.grad).any().item()}")
    print(f"Input grad norm: {x_grad.grad.norm().item():.2e}")
    print()
    
    print("=" * 60)
    print("BENCHMARKS")
    print("=" * 60)
    
    warmup, iters = 5, 20
    
    def bench(model, x):
        x_grad = x.clone().requires_grad_(True)
        for _ in range(warmup):
            out, _ = model(x_grad)
            out.sum().backward()
            model.zero_grad()
            x_grad.grad = None
        
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        
        start = time.perf_counter()
        for _ in range(iters):
            out, _ = model(x_grad)
            out.sum().backward()
            model.zero_grad()
            x_grad.grad = None
            if device == "cuda":
                torch.cuda.synchronize()
        total_ms = (time.perf_counter() - start) / iters * 1000
        mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        return total_ms, mem
    
    print(f"{'L':>8} | {'Fwd+Bwd(ms)':>12} {'Mem(MB)':>10}")
    print("-" * 40)
    
    seq_lengths = [256, 512, 1024, 2048, 4096]
    
    for L in seq_lengths:
        if device == "cuda":
            torch.cuda.empty_cache()
        
        try:
            x = torch.randn(B, L, C, device=device)
            total_ms, mem = bench(model, x)
            print(f"{L:>8} | {total_ms:>12.2f} {mem:>10.0f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{L:>8} | OOM")
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                raise
    
    print("=" * 60)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
