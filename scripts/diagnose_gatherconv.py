#!/usr/bin/env python3
import torch
import sys
sys.path.insert(0, '.')

from src.heuristic_secrets.models.gatherconv import GatherConvND

device = 'cuda'
B, L, C = 4, 8192, 256
H = 8

def mem():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

print(f'B={B}, L={L}, C={C}, H={H}')
print()

for ckpt in [False, True]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    conv = GatherConvND(channels=C, ndim=1, max_samples=32, num_heads=H, checkpoint=ckpt).to(device)
    conv.train()
    
    x = torch.randn(B, L, C, device=device, requires_grad=True)
    
    out, _ = conv(x)
    fwd_peak = mem()
    
    torch.cuda.reset_peak_memory_stats()
    out.sum().backward()
    bwd_peak = mem()
    
    print(f'checkpoint={ckpt}: fwd={fwd_peak:.1f} MB, bwd={bwd_peak:.1f} MB')
    
    del conv, x, out
    torch.cuda.empty_cache()
