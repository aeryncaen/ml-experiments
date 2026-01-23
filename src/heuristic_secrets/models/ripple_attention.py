"""RippleAttention: configurable order of telephone, conv, lowrank with norm+residual."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .telephone_attention import TelephoneAttentionND
from .adaptive_local_conv import AdaptiveLocalConv
from .scatter_attention import LowRankAttention, RMSNorm, SIRENDownsampleND, interpolate_nd, apply_rope

try:
    from .triton_adaptive_conv import TritonAdaptiveLocalConv, HAS_TRITON
except ImportError:
    HAS_TRITON = False
    TritonAdaptiveLocalConv = None


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
        
        out = interpolate_nd(out, (L,))
        
        return x + self.out_norm(out)


class RippleAttention(nn.Module):
    
    def __init__(
        self,
        channels: int,
        num_heads: int = 8,
        max_kernel_size: int = 64,
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
    ):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.order = [s.strip() for s in order.split(",")]
        
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
        
        if use_triton and HAS_TRITON and TritonAdaptiveLocalConv is not None:
            self.conv = TritonAdaptiveLocalConv(
                channels=channels,
                num_heads=num_heads,
                max_kernel_size=max_kernel_size,
                chunk_size=chunk_size,
                scale_power=conv_power,
            )
        else:
            self.conv = AdaptiveLocalConv(
                channels=channels,
                num_heads=num_heads,
                max_kernel_size=max_kernel_size,
                scale_power=conv_power,
            )
        
        self.lowrank = LowRankAttention(
            channels,
            num_heads=num_heads,
            reduction_power=lowrank_power,
        )
        
        self.norms = nn.ModuleDict({
            'tele': RMSNorm(channels, eps),
            'conv': RMSNorm(channels, eps),
            'lowrank': RMSNorm(channels, eps),
        })
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        h = x
        info = {}
        
        for name in self.order:
            if name == 'tele':
                out, _ = self.telephone(h)
            elif name == 'conv':
                out, info = self.conv(h)
            elif name == 'lowrank':
                out, _ = self.lowrank(h)
            else:
                raise ValueError(f"Unknown layer: {name}")
            h = h + self.norms[name](out)
        
        return h, info


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
    
    def forward(
        self,
        x: torch.Tensor,
        history: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict] | tuple[torch.Tensor, dict, torch.Tensor]:
        if self.cross_layer and history is not None:
            x = self.cross_layer_attn(x, history)
        
        attn_out, info = self.attn(x)
        x = x + attn_out
        x = x + self.mlp(x)
        
        if self.cross_layer:
            x_lowrank = self.history_accum(x)
            return x, info, x_lowrank
        
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
    ):
        super().__init__()
        self.cross_layer = cross_layer
        self.lowrank_power = lowrank_power
        
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
            )
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(width, eps)
        self.head = nn.Linear(width, n_classes)
        
        if cross_layer:
            self.embed_accum = LayerHistoryAccumulator(width, lowrank_power, eps)
            self.head_cross_attn = CrossLayerAttention(width, num_heads, eps)
            self.history_decay = nn.Parameter(torch.full((n_layers,), 2.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed_norm(F.silu(self.embed(x.unsqueeze(-1)))) + self.pos_norm(F.silu(self.pos_embed))
        
        if self.cross_layer:
            history: list[torch.Tensor] = [self.embed_accum(x)]
            
            for i, layer in enumerate(self.layers):
                x, _, x_lowrank = layer(x, history)
                decay = 0.5 + 0.5 * torch.sigmoid(self.history_decay[i])
                history = [h * decay for h in history]
                history.append(x_lowrank)
            
            x = self.head_cross_attn(x, history)
        else:
            for layer in self.layers:
                x, _ = layer(x)
        
        x = self.norm(x)
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
