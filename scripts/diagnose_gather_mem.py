#!/usr/bin/env python3
"""Diagnose memory usage of different gather strategies."""

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
if device != 'cuda':
    print('No CUDA - cannot measure memory accurately')
    exit()

B, L, C = 4, 8192, 256
H, D = 8, 32
S = 33
chunk_len = 768

def show(label):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()/1024**2
    peak = torch.cuda.max_memory_allocated()/1024**2
    print(f'{label}: {alloc:.1f} MB alloc, {peak:.1f} MB peak')

print(f'B={B}, L={L}, C={C}, H={H}, D={D}, S={S}, chunk_len={chunk_len}')
print()

torch.cuda.reset_peak_memory_stats()

x = torch.randn(B, L, C, device=device, requires_grad=True)
show('x allocated')

batch_idx = torch.arange(B, device=device).view(B,1,1).expand(B, chunk_len, S)
sample_idx = torch.randint(0, L, (B, chunk_len, S), device=device)
show('indices allocated')

print('\n=== STRATEGY 1: Gather all channels at once ===')
torch.cuda.reset_peak_memory_stats()
g_all = x[batch_idx, sample_idx]
show(f'after gather {list(g_all.shape)}')
loss = g_all.sum()
loss.backward()
show('after backward')
del g_all, loss
x.grad = None
torch.cuda.empty_cache()

print('\n=== STRATEGY 2: Gather per head (H iterations) ===')
torch.cuda.reset_peak_memory_stats()
out = []
for h in range(H):
    x_h = x[..., h*D:(h+1)*D]
    g_h = x_h[batch_idx, sample_idx]
    out.append(g_h)
out = torch.cat(out, dim=-1)
show(f'after gather loop {list(out.shape)}')
loss = out.sum()
loss.backward()
show('after backward')
del out, loss
x.grad = None
torch.cuda.empty_cache()

print('\n=== STRATEGY 3: Gather per sample, accumulate (H*S iterations) ===')
torch.cuda.reset_peak_memory_stats()
out = torch.zeros(B, chunk_len, C, device=device)
for h in range(H):
    x_h = x[..., h*D:(h+1)*D]
    for s in range(S):
        v = x_h[batch_idx[:,:,s], sample_idx[:,:,s]]
        out[..., h*D:(h+1)*D] += v
show(f'after accumulate loop')
loss = out.sum()
loss.backward()
show('after backward')
del out, loss
x.grad = None
torch.cuda.empty_cache()

print('\n=== STRATEGY 4: torch.gather (might be different) ===')
torch.cuda.reset_peak_memory_stats()
x_expanded = x.unsqueeze(2).expand(B, L, S, C)
idx_expanded = sample_idx.unsqueeze(-1).expand(B, chunk_len, S, C)
gathered = torch.zeros(B, chunk_len, S, C, device=device)
for b in range(B):
    gathered[b] = x_expanded[b].gather(0, idx_expanded[b])
show(f'after torch.gather workaround')
loss = gathered.sum()
loss.backward()
show('after backward')
del gathered, x_expanded, idx_expanded, loss
x.grad = None
torch.cuda.empty_cache()

print('\n=== STRATEGY 5: index_select per sample position ===')
torch.cuda.reset_peak_memory_stats()
out = torch.zeros(B, chunk_len, C, device=device)
for s in range(S):
    idx_s = sample_idx[:, :, s]
    for b in range(B):
        out[b] += x[b].index_select(0, idx_s[b])
show('after index_select loop')
loss = out.sum()
loss.backward()
show('after backward')
