#!/usr/bin/env python3
"""Diagnose memory usage of different gather strategies."""

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device != 'cuda':
    print('No CUDA - cannot measure memory accurately')
    exit()

B, C = 4, 256
H, D = 8, 32
S = 33
seq_lengths = [512, 1024, 2048, 4096, 8192, 16384]

def get_peak():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

print(f'B={B}, C={C}, H={H}, D={D}, S={S}')
print()
print(f'{"SeqLen":>8} | {"Strat1 (all)":>14} | {"Strat3 (H*S)":>14} | {"Strat5 (idx)":>14}')
print(f'{"":>8} | {"fwd / bwd":>14} | {"fwd / bwd":>14} | {"fwd / bwd":>14}')
print('-' * 62)

for L in seq_lengths:
    chunk_len = min(768, L)
    
    try:
        x = torch.randn(B, L, C, device=device, requires_grad=True)
        batch_idx = torch.arange(B, device=device).view(B,1,1).expand(B, chunk_len, S)
        sample_idx = torch.randint(0, L, (B, chunk_len, S), device=device)
        
        results = []
        
        for strat in [1, 3, 5]:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            fwd_peak = bwd_peak = 0.0
            if strat == 1:
                g = x[batch_idx, sample_idx]
                fwd_peak = get_peak()
                g.sum().backward()
                bwd_peak = get_peak()
                del g
                
            elif strat == 3:
                out = torch.zeros(B, chunk_len, C, device=device)
                for h in range(H):
                    x_h = x[..., h*D:(h+1)*D]
                    for s in range(S):
                        v = x_h[batch_idx[:,:,s], sample_idx[:,:,s]]
                        out[..., h*D:(h+1)*D] += v
                fwd_peak = get_peak()
                out.sum().backward()
                bwd_peak = get_peak()
                del out
                
            elif strat == 5:
                out = torch.zeros(B, chunk_len, C, device=device)
                for s in range(S):
                    idx_s = sample_idx[:, :, s]
                    for b in range(B):
                        out[b] += x[b].index_select(0, idx_s[b])
                fwd_peak = get_peak()
                out.sum().backward()
                bwd_peak = get_peak()
                del out
            
            x.grad = None
            results.append(f'{fwd_peak:>5.0f}/{bwd_peak:>5.0f}')
        
        print(f'{L:>8} | {results[0]:>14} | {results[1]:>14} | {results[2]:>14}')
        del x, batch_idx, sample_idx
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f'{L:>8} | OOM')
            torch.cuda.empty_cache()
        else:
            raise
