#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
from operator import mul

device = 'cuda'
B, L, C = 4, 8192, 256
H, D = 8, 32
S = 33
K = 64
chunk_size = 768

def mem():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

def show(label):
    print(f'{label}: {mem():.1f} MB peak')

print(f'B={B}, L={L}, C={C}, H={H}, chunk_size={chunk_size}')
print()

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

wave_proj = nn.Linear(C, 2 * H).to(device)
kernel_proj = nn.Linear(C, H * K).to(device)
out_proj = nn.Linear(C, C, bias=False).to(device)

x = torch.randn(B, L, C, device=device, requires_grad=True)
show('after x alloc')

x_flat = x.reshape(B, L, C)
outputs = []

for chunk_idx, start in enumerate(range(0, L, chunk_size)):
    end = min(start + chunk_size, L)
    chunk_len = end - start
    x_chunk = x_flat[:, start:end, :]
    
    wave_params = F.silu(wave_proj(x_chunk)).view(B, chunk_len, 2, H)
    if chunk_idx == 0:
        show(f'  chunk0: after wave_proj')
    
    freq_avg = torch.sigmoid(wave_params[:,:,0,:]).mean(dim=-1) * 15 + 1
    phase_avg = torch.tanh(wave_params[:,:,1,:]).mean(dim=-1) * 16
    
    stride_grid = torch.arange(-16, 17, device=device).float()
    centers = torch.arange(start, end, device=device).float()
    sample_pos = centers.view(1, chunk_len, 1) + stride_grid.view(1, 1, S) * freq_avg.unsqueeze(-1) + phase_avg.unsqueeze(-1)
    valid_mask = (sample_pos >= 0) & (sample_pos < L)
    sample_idx = sample_pos.long().clamp(0, L - 1)
    
    batch_idx = torch.arange(B, device=device).view(B, 1, 1).expand(B, chunk_len, S)
    
    kernel_max = F.silu(kernel_proj(x_chunk)).view(B, chunk_len, H, K)
    if chunk_idx == 0:
        show(f'  chunk0: after kernel_proj')
    
    norm_pos = (sample_pos - sample_pos.min()) / (sample_pos.max() - sample_pos.min() + 1e-8)
    idx_float = norm_pos * (K - 1)
    idx_floor = idx_float.long().clamp(0, K - 2)
    idx_ceil = idx_floor + 1
    w_ceil = idx_float - idx_floor.float()
    w_floor = 1.0 - w_ceil
    
    valid_mask_f = valid_mask.float()
    output = torch.zeros(B, chunk_len, C, device=device)
    
    for h in range(H):
        km_h = kernel_max[:, :, h, :]
        k_floor_h = km_h.gather(-1, idx_floor)
        k_ceil_h = km_h.gather(-1, idx_ceil)
        kernel_h = k_floor_h * w_floor + k_ceil_h * w_ceil
        kernel_h = kernel_h * valid_mask_f
        kernel_h = kernel_h / (kernel_h.sum(dim=-1, keepdim=True) + 1e-8)
        
        x_head = x_flat[..., h * D : (h + 1) * D]
        out_h = output[:, :, h * D : (h + 1) * D]
        
        for s in range(S):
            val_s = x_head[batch_idx[:, :, s], sample_idx[:, :, s]]
            out_h.addcmul_(kernel_h[:, :, s : s + 1], val_s)
    
    if chunk_idx == 0:
        show(f'  chunk0: after gather loops')
    
    outputs.append(output)

show('after all chunks')

output = torch.cat(outputs, dim=1)
show('after concat')

output = F.silu(out_proj(output))
show('after out_proj')

loss = output.sum()
loss.backward()
show('after backward')
