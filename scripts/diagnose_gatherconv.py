#!/usr/bin/env python3
import torch
import sys
sys.path.insert(0, '.')

from src.heuristic_secrets.models.gatherconv import GatherConvND

device = 'cuda'
B, C = 4, 256
H = 8
seq_lengths = [512, 1024, 2048, 4096, 8192]

def get_peak():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

print(f'B={B}, C={C}, H={H}')
print(f'{"SeqLen":>8} | {"Fwd Peak":>10} | {"Bwd Peak":>10}')
print('-' * 36)

conv = GatherConvND(channels=C, ndim=1, max_samples=32, num_heads=H, checkpoint=False).to(device)
conv.train()

for L in seq_lengths:
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        
        out, _ = conv(x)
        fwd_peak = get_peak()
        
        out.sum().backward()
        bwd_peak = get_peak()
        
        print(f'{L:>8} | {fwd_peak:>8.1f} MB | {bwd_peak:>8.1f} MB')
        
        del x, out
        conv.zero_grad()
        
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'{L:>8} | OOM')
            torch.cuda.empty_cache()
        else:
            raise
