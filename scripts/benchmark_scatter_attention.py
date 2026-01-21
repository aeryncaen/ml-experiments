import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from heuristic_secrets.models.scatter_attention import LocalAttention, RMSNorm


class FullAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
    
    def forward(self, x):
        B, L, C = x.shape
        
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.out(out)


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
        _ = model(x)
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


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('=' * 80)
    print('LOCAL ATTENTION vs FULL ATTENTION')
    print('=' * 80)

    B = 16
    C = 64
    K = 17

    local = LocalAttention(embed_dim=C, kernel_size=K).to(device)
    full = FullAttention(embed_dim=C, num_heads=4).to(device)

    print(f'Local (K={K}):      {count_params(local):>8,} params')
    print(f'Full (H=4):         {count_params(full):>8,} params')
    print()

    print('Sequence Length Scaling:')
    print('-' * 70)
    print(f'{"L":>6} | {"Local":>12} | {"Full":>12} | {"Speedup":>10} | {"Complexity"}')
    print('-' * 70)

    seq_lengths = [128, 256, 512, 1024, 2048, 4096]

    for L in tqdm(seq_lengths, desc="Benchmarking"):
        x = torch.randn(B, L, C, device=device)
        
        try:
            local_fwd, _ = benchmark(local, x, n_warmup=3, n_iters=10)
            full_fwd, _ = benchmark(full, x, n_warmup=3, n_iters=10)
            
            speedup = full_fwd / local_fwd
            complexity = f'O({L}*{K}) vs O({L}^2)'
            tqdm.write(f'{L:>6} | {local_fwd:>10.2f}ms | {full_fwd:>10.2f}ms | {speedup:>9.2f}x | {complexity}')
        except Exception as e:
            tqdm.write(f'{L:>6} | Error: {e}')


if __name__ == '__main__':
    main()
