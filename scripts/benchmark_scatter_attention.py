import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
from heuristic_secrets.models.scatter_attention import LocalAttention, LocalAttentionND, LowRankAttentionND, RMSNorm


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


def get_peak_memory_mb(device):
    if device.type == 'cuda':
        return torch.cuda.max_memory_allocated(device) / 1024 / 1024
    elif device.type == 'mps':
        return torch.mps.current_allocated_memory() / 1024 / 1024
    return 0.0


def reset_memory_stats(device):
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def benchmark(model, x, n_warmup=5, n_iters=20, backward=False):
    device = x.device
    
    for _ in range(n_warmup):
        out = model(x)
        if backward:
            out.sum().backward()
    sync_device(device)
    
    reset_memory_stats(device)
    
    if backward:
        x = x.detach().requires_grad_(True)
    
    t0 = time.perf_counter()
    for _ in range(n_iters):
        out = model(x)
        if backward:
            out.sum().backward()
            if x.grad is not None:
                x.grad = None
    sync_device(device)
    time_ms = (time.perf_counter() - t0) / n_iters * 1000
    peak_mem = get_peak_memory_mb(device)
    
    return time_ms, peak_mem


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', action='store_true', help='Enable gradient checkpointing for LocalAttentionND')
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('=' * 100)
    print(f'LOCAL ATTENTION vs LOWRANK vs FULL ATTENTION (checkpoint={args.checkpoint})')
    print('=' * 100)

    B = 2
    C = 64
    K = 17
    H = 4

    local = LocalAttentionND(embed_dim=C, kernel_size=K, ndim=1, num_channels=H, checkpoint=args.checkpoint).to(device)
    lowrank = LowRankAttentionND(embed_dim=C, window_size=K, ndim=1, num_channels=H).to(device)
    full = FullAttention(embed_dim=C, num_heads=H).to(device)

    print(f'Local (K={K}):         {count_params(local):>8,} params')
    print(f'LowRank (sqrt(L)):     {count_params(lowrank):>8,} params')
    print(f'Full (H={H}):           {count_params(full):>8,} params')
    print()

    seq_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]

    for mode, backward in [('Forward', False), ('Fwd+Bwd', True)]:
        print(f'\n{mode} pass (time, peak memory):')
        print('-' * 100)
        print(f'{"L":>6} | {"Local":>20} | {"LowRank":>20} | {"Full":>20}')
        print('-' * 100)

        for L in tqdm(seq_lengths, desc=mode):
            x = torch.randn(B, L, C, device=device)
            results = {}
            
            for name, model in [('local', local), ('lowrank', lowrank), ('full', full)]:
                try:
                    model.train()
                    reset_memory_stats(device)
                    time_ms, peak_mem = benchmark(model, x, n_warmup=3, n_iters=10, backward=backward)
                    results[name] = (time_ms, peak_mem)
                except Exception as e:
                    results[name] = (float('inf'), float('inf'))
                    tqdm.write(f'{name} failed at L={L}: {e}')
            
            local_t, local_m = results['local']
            lr_t, lr_m = results['lowrank']
            full_t, full_m = results['full']
            
            tqdm.write(f'{L:>6} | {local_t:>7.2f}ms {local_m:>7.1f}MB | {lr_t:>7.2f}ms {lr_m:>7.1f}MB | {full_t:>7.2f}ms {full_m:>7.1f}MB')

        print('-' * 100)


if __name__ == '__main__':
    main()
