import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from heuristic_secrets.models.scatter_attention import (
    InceptionScatterAttention,
    RMSNorm,
)


class SimpleAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = RMSNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3, bias=False)
        self.out = nn.Linear(channels, channels, bias=False)
    
    def forward(self, x):
        B, L, C = x.shape
        x_norm = self.norm(x)
        
        qkv = self.qkv(x_norm).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.out(out) + x


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def sync_device(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


def benchmark(model, x, n_warmup=5, n_iters=20):
    device = x.device
    
    for _ in range(n_warmup):
        _ = model(x)
    sync_device(device)
    
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = model(x)
    sync_device(device)
    fwd_time = (time.perf_counter() - t0) / n_iters * 1000
    
    x_grad = x.detach().requires_grad_(True)
    sync_device(device)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = model(x_grad)
        out.sum().backward()
        x_grad.grad = None
    sync_device(device)
    fwd_bwd_time = (time.perf_counter() - t0) / n_iters * 1000
    
    return fwd_time, fwd_bwd_time


device = torch.device('cpu')#'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Device: {device}')
print('=' * 80)
print('INCEPTION SCATTER vs STANDARD ATTENTION')
print('=' * 80)

B = 16
C = 32

inception = InceptionScatterAttention(embed_dim=C, scatter_channels=4, num_samples=16).to(device)
attn = SimpleAttention(channels=C, num_heads=4).to(device)

print(f'Inception (SC=4, K=16):   {count_params(inception):>8,} params')
print(f'Attention (C={C}, H=4):   {count_params(attn):>8,} params')
print()

print('Sequence Length Scaling:')
print('-' * 60)
print(f'{"L":>6} | {"Inception":>12} | {"Attention":>12} | {"Speedup":>10}')
print('-' * 60)

seq_lengths = [128, 256, 512, 1024]

for L in tqdm(seq_lengths, desc="Benchmarking"):
    x = torch.randn(B, L, C, device=device)
    
    try:
        i_fwd, _ = benchmark(inception, x, n_warmup=3, n_iters=10)
        a_fwd, _ = benchmark(attn, x, n_warmup=3, n_iters=10)
        
        speedup = a_fwd / i_fwd
        tqdm.write(f'{L:>6} | {i_fwd:>10.2f}ms | {a_fwd:>10.2f}ms | {speedup:>9.2f}x')
    except Exception as e:
        tqdm.write(f'{L:>6} | Error: {e}')
