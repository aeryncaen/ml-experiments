"""
GatherConv: Learned position sampling with interpolated kernel convolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from functools import reduce
from operator import mul

try:
    from .triton_gather import TritonGatherConv, HAS_TRITON
except ImportError:
    HAS_TRITON = False
    TritonGatherConv = None


class GatherConvND(nn.Module):
    """
    Gather samples at learned positions, convolve with interpolated kernel.
    
    For each position: predict freq/phase → compute sample positions →
    gather values → project kernel and interpolate to positions → apply.
    """
    
    stride_grid: torch.Tensor
    
    def __init__(
        self,
        channels: int,
        ndim: int = 1,
        max_samples: int = 32,
        num_heads: int = 1,
        max_freq: float = 16.0,
        min_freq: float = 1.0,
        max_kernel_size: int = 64,
        chunk_size: int = 768,
        checkpoint: bool = True,
        use_triton: bool = True,
    ):
        super().__init__()
        self.checkpoint = checkpoint
        self.use_triton = use_triton and HAS_TRITON and ndim == 1
        self.channels = channels
        self.ndim = ndim
        self.max_samples = max_samples
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.chunk_size = chunk_size
        self.max_freq = max_freq
        self.min_freq = min_freq
        self.max_kernel_size = max_kernel_size
        
        H = num_heads
        
        self.wave_proj = nn.Linear(channels, 2 * H)
        self.kernel_proj = nn.Linear(channels, H * max_kernel_size)
        self.out_proj = nn.Linear(channels, channels, bias=False)
        
        samples_per_dim = max(3, int(max_samples ** (1.0 / ndim)))
        half_s = samples_per_dim // 2
        offsets_1d = torch.arange(-half_s, half_s + 1).float()
        
        if ndim == 1:
            stride_grid = offsets_1d.unsqueeze(-1)
        else:
            grids = torch.meshgrid(*[offsets_1d] * ndim, indexing='ij')
            stride_grid = torch.stack([g.flatten() for g in grids], dim=-1)
        
        self.register_buffer('stride_grid', stride_grid)
        self.num_samples = stride_grid.shape[0]
        
        self.max_offset = half_s
        self.max_receptive = half_s * max_freq
        
        nn.init.zeros_(self.wave_proj.weight)
        nn.init.zeros_(self.wave_proj.bias)
        nn.init.xavier_uniform_(self.kernel_proj.weight, gain=0.1)
        nn.init.zeros_(self.kernel_proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)
    
    def _forward_chunk(
        self,
        x_flat: torch.Tensor,
        x_chunk: torch.Tensor,
        chunk_start: int,
        chunk_end: int,
        spatial: tuple[int, ...],
        L: int,
    ) -> torch.Tensor:
        B = x_flat.shape[0]
        C = x_flat.shape[-1]
        H = self.num_heads
        D = self.head_dim
        S = self.num_samples
        K = self.max_kernel_size
        chunk_len = chunk_end - chunk_start
        
        wave_params = F.silu(self.wave_proj(x_chunk)).view(B, chunk_len, 2, H)
        
        freq_raw = wave_params[:, :, 0, :]
        phase_raw = wave_params[:, :, 1, :]
        
        freq = torch.sigmoid(freq_raw) * (self.max_freq - self.min_freq) + self.min_freq
        phase = torch.tanh(phase_raw) * self.max_freq
        
        freq_avg = freq.mean(dim=-1)
        phase_avg = phase.mean(dim=-1)
        
        if self.ndim == 1:
            centers = torch.arange(chunk_start, chunk_end, device=x_flat.device, dtype=x_flat.dtype)
            sample_pos = (
                centers.view(1, chunk_len, 1) + 
                self.stride_grid[:, 0].view(1, 1, S) * freq_avg.unsqueeze(-1) + 
                phase_avg.unsqueeze(-1)
            )
            valid_mask = (sample_pos >= 0) & (sample_pos < L)
            sample_idx = sample_pos.long().clamp(0, L - 1)
            
            rel_pos = self.stride_grid[:, 0].view(1, 1, S) * freq_avg.unsqueeze(-1) + phase_avg.unsqueeze(-1)
        else:
            coords = [torch.arange(s, device=x_flat.device, dtype=x_flat.dtype) for s in spatial]
            mesh = torch.meshgrid(*coords, indexing='ij')
            centers_all = torch.stack([m.flatten() for m in mesh], dim=-1)
            centers = centers_all[chunk_start:chunk_end]
            
            sample_pos = (
                centers.view(1, chunk_len, 1, self.ndim) + 
                self.stride_grid.view(1, 1, S, self.ndim) * freq_avg.view(B, chunk_len, 1, 1) + 
                phase_avg.view(B, chunk_len, 1, 1)
            )
            
            valid_mask = torch.ones(B, chunk_len, S, dtype=torch.bool, device=x_flat.device)
            for dim in range(self.ndim):
                valid_mask = valid_mask & (sample_pos[..., dim] >= 0) & (sample_pos[..., dim] < spatial[dim])
            
            sample_coords = sample_pos.long()
            for dim in range(self.ndim):
                sample_coords[..., dim] = sample_coords[..., dim].clamp(0, spatial[dim] - 1)
            
            strides = [1]
            for dim in range(self.ndim - 1, 0, -1):
                strides.insert(0, strides[0] * spatial[dim])
            sample_idx = sum(sample_coords[..., dim] * strides[dim] for dim in range(self.ndim))
            
            rel_pos_nd = self.stride_grid.view(1, 1, S, self.ndim) * freq_avg.view(B, chunk_len, 1, 1)
            rel_pos = rel_pos_nd.norm(dim=-1)
        
        batch_idx = torch.arange(B, device=x_flat.device).view(B, 1, 1).expand(B, chunk_len, S)
        
        kernel_max = F.silu(self.kernel_proj(x_chunk)).view(B, chunk_len, H, K)
        
        norm_pos = (rel_pos + self.max_receptive) / (2 * self.max_receptive)
        norm_pos = norm_pos.clamp(0, 1)
        
        idx_float = norm_pos * (K - 1)
        idx_floor = idx_float.long().clamp(0, K - 2)
        idx_ceil = idx_floor + 1
        w_ceil = idx_float - idx_floor.float()
        w_floor = 1.0 - w_ceil
        
        valid_mask_f = valid_mask.float()
        
        output = torch.zeros(B, chunk_len, C, device=x_flat.device, dtype=x_flat.dtype)
        
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
        
        return output
    
    def _forward_triton(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        H = self.num_heads
        D = self.head_dim
        S = self.num_samples
        K = self.max_kernel_size
        chunk_size = min(self.chunk_size, L)
        
        out = torch.empty_like(x)
        BLOCK_D = 1
        while BLOCK_D < D:
            BLOCK_D *= 2
        
        from .triton_gather import gather_conv_fwd_kernel_chunked
        
        for start in range(0, L, chunk_size):
            end = min(start + chunk_size, L)
            chunk_len = end - start
            x_chunk = x[:, start:end, :]
            
            wave_out = F.silu(self.wave_proj(x_chunk))
            wave_out = wave_out.view(B, chunk_len, 2, H)
            freq = torch.sigmoid(wave_out[:, :, 0, :]) * (self.max_freq - self.min_freq) + self.min_freq
            phase = torch.tanh(wave_out[:, :, 1, :]) * self.max_freq
            freq_avg = freq.mean(dim=-1).contiguous()
            phase_avg = phase.mean(dim=-1).contiguous()
            
            kernel = F.silu(self.kernel_proj(x_chunk)).view(B, chunk_len, H, K).contiguous()
            
            hidden_chunk = torch.empty(B, chunk_len, C, device=x.device, dtype=x.dtype)
            
            grid = (B, chunk_len, H)
            gather_conv_fwd_kernel_chunked[grid](
                x, hidden_chunk,
                freq_avg, phase_avg,
                kernel,
                B, L, C, H, D, S, K,
                self.max_offset, self.max_receptive,
                start,
                chunk_len,
                BLOCK_D,
            )
            
            out[:, start:end, :] = F.silu(F.linear(hidden_chunk, self.out_proj.weight))
        
        return out
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B = x.shape[0]
        spatial = x.shape[1:-1]
        C = x.shape[-1]
        L = reduce(mul, spatial, 1)
        
        x_flat = x.reshape(B, L, C)
        
        if self.use_triton and self.ndim == 1:
            output = self._forward_triton(x_flat)
            return output.reshape(B, *spatial, C), {}
        
        output = torch.empty_like(x_flat)
        
        if L <= self.chunk_size:
            if self.checkpoint and self.training:
                hidden = grad_checkpoint(
                    self._forward_chunk, x_flat, x_flat, 0, L, spatial, L,
                    use_reentrant=False,
                )
            else:
                hidden = self._forward_chunk(x_flat, x_flat, 0, L, spatial, L)
            output = F.silu(self.out_proj(hidden))
        else:
            for start in range(0, L, self.chunk_size):
                end = min(start + self.chunk_size, L)
                if self.checkpoint and self.training:
                    hidden = grad_checkpoint(
                        self._forward_chunk, x_flat, x_flat[:, start:end], start, end, spatial, L,
                        use_reentrant=False,
                    )
                else:
                    hidden = self._forward_chunk(x_flat, x_flat[:, start:end], start, end, spatial, L)
                output[:, start:end, :] = F.silu(self.out_proj(hidden))
        
        return output.reshape(B, *spatial, C), {}


GatherConv1d = GatherConvND
GatherConv2d = lambda *args, **kwargs: GatherConvND(*args, ndim=2, **kwargs)
GatherConv3d = lambda *args, **kwargs: GatherConvND(*args, ndim=3, **kwargs)


class SDPAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Linear(channels, 3 * channels)
        self.out = nn.Linear(channels, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.out(out)


if __name__ == "__main__":
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, C = 4, 256
    num_heads = 8
    max_samples = 32
    warmup_iters = 5
    bench_iters = 20
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    print(f"Device: {device}")
    print(f"HAS_TRITON: {HAS_TRITON}")
    print(f"Batch={B}, Channels={C}, Heads={num_heads}")
    print()
    
    gather_triton = GatherConvND(
        channels=C, ndim=1, max_samples=max_samples, num_heads=num_heads, 
        checkpoint=True, use_triton=True
    ).to(device)
    gather_triton.train()
    
    gather_pytorch = GatherConvND(
        channels=C, ndim=1, max_samples=max_samples, num_heads=num_heads,
        checkpoint=True, use_triton=False
    ).to(device)
    gather_pytorch.train()
    
    sdp_attn = SDPAttention(channels=C, num_heads=num_heads).to(device)
    sdp_attn.train()
    
    def bench(model, x, is_gather=False):
        if device == "cuda":
            torch.cuda.synchronize()
        for _ in range(warmup_iters):
            out = model(x)[0] if is_gather else model(x)
            if device == "cuda":
                torch.cuda.synchronize()
        
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(bench_iters):
            out = model(x)[0] if is_gather else model(x)
            if device == "cuda":
                torch.cuda.synchronize()
        fwd_time = (time.perf_counter() - start) / bench_iters * 1000
        fwd_mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        return fwd_time, fwd_mem
    
    def bench_bwd(model, x, is_gather=False):
        if device == "cuda":
            torch.cuda.synchronize()
        for _ in range(warmup_iters):
            out = model(x)[0] if is_gather else model(x)
            out.sum().backward()
            model.zero_grad()
            if device == "cuda":
                torch.cuda.synchronize()
        
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(bench_iters):
            out = model(x)[0] if is_gather else model(x)
            out.sum().backward()
            model.zero_grad()
            if device == "cuda":
                torch.cuda.synchronize()
        fwd_bwd_time = (time.perf_counter() - start) / bench_iters * 1000
        fwd_bwd_mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        return fwd_bwd_time, fwd_bwd_mem
    
    def run_bench(model, name, L, is_gather=False):
        fwd = bwd = fwd_mem = bwd_mem = "OOM"
        if device == "cuda":
            torch.cuda.empty_cache()
        try:
            x = torch.randn(B, L, C, device=device)
            fwd, fwd_mem = bench(model, x, is_gather=is_gather)
            del x
            
            x = torch.randn(B, L, C, device=device, requires_grad=True)
            bwd, bwd_mem = bench_bwd(model, x, is_gather=is_gather)
            del x
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if device == "cuda":
                torch.cuda.empty_cache()
        return fwd, bwd, fwd_mem, bwd_mem
    
    triton_label = "GatherConv (Triton)" if gather_triton.use_triton else "GatherConv (PyTorch*)"
    
    print("=" * 130)
    print(f"{'':>8} | {triton_label:^28} | {'GatherConv (PyTorch)':^28} | {'SDPA Full Attention':^28}")
    print(f"{'SeqLen':>8} | {'Fwd':>6} {'Bwd':>6} {'FMB':>6} {'BMB':>6} | {'Fwd':>6} {'Bwd':>6} {'FMB':>6} {'BMB':>6} | {'Fwd':>6} {'Bwd':>6} {'FMB':>6} {'BMB':>6}")
    print("=" * 130)
    
    for L in seq_lengths:
        tr_fwd, tr_bwd, tr_fwd_mem, tr_bwd_mem = run_bench(gather_triton, "triton", L, is_gather=True)
        pt_fwd, pt_bwd, pt_fwd_mem, pt_bwd_mem = run_bench(gather_pytorch, "pytorch", L, is_gather=True)
        sd_fwd, sd_bwd, sd_fwd_mem, sd_bwd_mem = run_bench(sdp_attn, "sdpa", L, is_gather=False)
        
        def fmt(v):
            return f"{v:>6.1f}" if isinstance(v, float) else f"{v:>6}"
        def fmt_mem(v):
            return f"{v:>6.0f}" if isinstance(v, float) else f"{v:>6}"
        
        print(f"{L:>8} | {fmt(tr_fwd)} {fmt(tr_bwd)} {fmt_mem(tr_fwd_mem)} {fmt_mem(tr_bwd_mem)} | {fmt(pt_fwd)} {fmt(pt_bwd)} {fmt_mem(pt_fwd_mem)} {fmt_mem(pt_bwd_mem)} | {fmt(sd_fwd)} {fmt(sd_bwd)} {fmt_mem(sd_fwd_mem)} {fmt_mem(sd_bwd_mem)}")
    
    print("=" * 130)
    print(f"GatherConv params: {sum(p.numel() for p in gather_triton.parameters()):,}")
    print(f"SDPA params: {sum(p.numel() for p in sdp_attn.parameters()):,}")
    if not gather_triton.use_triton:
        print("* Triton not available, both columns use PyTorch implementation")
