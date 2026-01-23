"""RippleAttention: configurable order of telephone, conv, lowrank with norm+residual."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .telephone_attention import TelephoneAttentionND
from .adaptive_local_conv import AdaptiveLocalConv
from .scatter_attention import LowRankAttention, RMSNorm

try:
    from .triton_adaptive_conv import TritonAdaptiveLocalConv, HAS_TRITON
except ImportError:
    HAS_TRITON = False
    TritonAdaptiveLocalConv = None


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
        
        self.lowrank = LowRankAttention(channels, reduction_power=lowrank_power)
        
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
    ):
        super().__init__()
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
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        attn_out, info = self.attn(x)
        x = x + attn_out
        x = x + self.mlp(x)
        return x, info


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
