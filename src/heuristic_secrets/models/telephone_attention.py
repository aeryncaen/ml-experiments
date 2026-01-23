"""
TelephoneAttention: Learned position sampling with power-law decay.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from functools import reduce
from operator import mul

try:
    from .triton_telephone import _TelephoneAttentionTriton, HAS_TRITON
except ImportError:
    try:
        from triton_telephone import _TelephoneAttentionTriton, HAS_TRITON
    except ImportError:
        HAS_TRITON = False
        _TelephoneAttentionTriton = None


def llama_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """LlamaRMSNorm as a function - equivalent to T5LayerNorm."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(input_dtype)


class TelephoneAttentionND(nn.Module):
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
        chunk_size: int = 1024,
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
        self.wave_gamma = nn.Parameter(torch.ones(2 * H))
        self.kernel_proj = nn.Linear(channels, H * max_kernel_size)
        self.kernel_gamma = nn.Parameter(torch.ones(H * max_kernel_size))
        self.exponent_proj = nn.Linear(channels, 1)
        self.exponent_gamma = nn.Parameter(torch.ones(1))
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
        nn.init.zeros_(self.exponent_proj.weight)
        nn.init.constant_(self.exponent_proj.bias, 0.0)  # sigmoid(0)*3.5+0.5 = 2.25
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
        
        # wave_proj: linear -> RMSNorm -> SiLU
        wave_pre = self.wave_proj(x_chunk)
        wave_normed = llama_rmsnorm(wave_pre, self.wave_gamma)
        wave_params = F.silu(wave_normed).view(B, chunk_len, 2, H)
        
        freq_raw = wave_params[:, :, 0, :]
        phase_raw = wave_params[:, :, 1, :]
        
        freq = torch.sigmoid(freq_raw) * (self.max_freq - self.min_freq) + self.min_freq
        phase = torch.tanh(phase_raw) * self.max_freq
        
        if self.ndim == 1:
            centers = torch.arange(chunk_start, chunk_end, device=x_flat.device, dtype=x_flat.dtype)
            sample_pos = (
                centers.view(1, chunk_len, 1, 1) + 
                self.stride_grid[:, 0].view(1, 1, 1, S) * freq.unsqueeze(-1) + 
                phase.unsqueeze(-1)
            )
            valid_mask = (sample_pos >= 0) & (sample_pos < L)
            # Soft indexing: floor, ceil, and fractional weight
            pos_clamped = sample_pos.clamp(0, L - 1.001)
            sample_floor = pos_clamped.floor().long().clamp(0, L - 1)
            sample_ceil = (sample_floor + 1).clamp(0, L - 1)
            sample_frac = pos_clamped - sample_floor.float()
            
            rel_pos = self.stride_grid[:, 0].view(1, 1, 1, S) * freq.unsqueeze(-1)
        else:
            coords = [torch.arange(s, device=x_flat.device, dtype=x_flat.dtype) for s in spatial]
            mesh = torch.meshgrid(*coords, indexing='ij')
            centers_all = torch.stack([m.flatten() for m in mesh], dim=-1)
            centers = centers_all[chunk_start:chunk_end]
            
            sample_pos = (
                centers.view(1, chunk_len, 1, 1, self.ndim) + 
                self.stride_grid.view(1, 1, 1, S, self.ndim) * freq.view(B, chunk_len, H, 1, 1) + 
                phase.view(B, chunk_len, H, 1, 1)
            )
            
            valid_mask = torch.ones(B, chunk_len, H, S, dtype=torch.bool, device=x_flat.device)
            for dim in range(self.ndim):
                valid_mask = valid_mask & (sample_pos[..., dim] >= 0) & (sample_pos[..., dim] < spatial[dim])
            
            sample_coords = sample_pos.long()
            for dim in range(self.ndim):
                sample_coords[..., dim] = sample_coords[..., dim].clamp(0, spatial[dim] - 1)
            
            strides = [1]
            for dim in range(self.ndim - 1, 0, -1):
                strides.insert(0, strides[0] * spatial[dim])
            sample_idx = sum(sample_coords[..., dim] * strides[dim] for dim in range(self.ndim))
            
            rel_pos_nd = self.stride_grid.view(1, 1, 1, S, self.ndim) * freq.view(B, chunk_len, H, 1, 1)
            rel_pos = rel_pos_nd.norm(dim=-1)
        
        batch_idx = torch.arange(B, device=x_flat.device).view(B, 1, 1, 1).expand(B, chunk_len, H, S)
        
        # kernel_proj: linear -> RMSNorm -> SiLU
        kernel_pre = self.kernel_proj(x_chunk)
        kernel_normed = llama_rmsnorm(kernel_pre, self.kernel_gamma)
        kernel_max = F.silu(kernel_normed).view(B, chunk_len, H, K)
        
        # exponent_proj: linear -> RMSNorm -> sigmoid -> [0.5, 4.0]
        exponent_pre = self.exponent_proj(x_chunk)  # (B, chunk_len, 1)
        exponent_normed = llama_rmsnorm(exponent_pre, self.exponent_gamma)
        exponent = torch.sigmoid(exponent_normed) * 3.5 + 0.5  # (B, chunk_len, 1)
        
        # Power-law decay: distance from broadcast center normalized by L
        # rel_pos = abs(s_off * freq), shape (B, chunk_len, H, S)
        norm_dist = rel_pos.abs() / L  # normalize by sequence length
        # exponent is (B, chunk_len, 1), expand to (B, chunk_len, 1, 1) for broadcast
        power_weight = 1.0 / (1.0 + norm_dist).pow(exponent.unsqueeze(-1))  # (B, chunk_len, H, S)
        
        norm_pos = rel_pos.abs() / self.max_receptive
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
            idx_floor_h = idx_floor[:, :, h, :]
            idx_ceil_h = idx_ceil[:, :, h, :]
            w_floor_h = w_floor[:, :, h, :]
            w_ceil_h = w_ceil[:, :, h, :]
            
            k_floor_h = km_h.gather(-1, idx_floor_h)
            k_ceil_h = km_h.gather(-1, idx_ceil_h)
            kernel_h = k_floor_h * w_floor_h + k_ceil_h * w_ceil_h
            kernel_h = kernel_h * power_weight[:, :, h, :] * valid_mask_f[:, :, h, :]
            
            x_head = x_flat[..., h * D : (h + 1) * D]
            out_h = output[:, :, h * D : (h + 1) * D]
            sample_floor_h = sample_floor[:, :, h, :]
            sample_ceil_h = sample_ceil[:, :, h, :]
            sample_frac_h = sample_frac[:, :, h, :]
            
            for s in range(S):
                val_floor = x_head[batch_idx[:, :, h, s], sample_floor_h[:, :, s]]
                val_ceil = x_head[batch_idx[:, :, h, s], sample_ceil_h[:, :, s]]
                frac = sample_frac_h[:, :, s : s + 1]
                val_s = val_floor * (1.0 - frac) + val_ceil * frac
                out_h.addcmul_(kernel_h[:, :, s : s + 1], val_s)
        
        return output
    
    def _forward_triton(self, x: torch.Tensor, L: int) -> torch.Tensor:
        # Use _TelephoneAttentionTriton directly with our nn.Linear weights
        # This gets full chunked memory benefits
        return _TelephoneAttentionTriton.apply(
            x,
            self.wave_proj.weight,    # (2*H, C)
            self.wave_proj.bias,      # (2*H,)
            self.wave_gamma,          # (2*H,)
            self.kernel_proj.weight,  # (H*K, C)
            self.kernel_proj.bias,    # (H*K,)
            self.kernel_gamma,        # (H*K,)
            self.exponent_proj.weight,  # (1, C)
            self.exponent_proj.bias,    # (1,)
            self.exponent_gamma,        # (1,)
            self.out_proj.weight,     # (C, C)
            self.num_heads,
            self.max_offset,
            self.max_receptive,
            self.max_freq,
            self.min_freq,
            self.max_kernel_size,
            self.chunk_size,
            L,
        )
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        B = x.shape[0]
        spatial = x.shape[1:-1]
        C = x.shape[-1]
        L = reduce(mul, spatial, 1)
        
        x_flat = x.reshape(B, L, C)
        
        if self.use_triton and self.ndim == 1:
            output = self._forward_triton(x_flat, L)
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


TelephoneAttention1d = TelephoneAttentionND
TelephoneAttention2d = lambda *args, **kwargs: TelephoneAttentionND(*args, ndim=2, **kwargs)
TelephoneAttention3d = lambda *args, **kwargs: TelephoneAttentionND(*args, ndim=3, **kwargs)


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
    
    tel_triton = TelephoneAttentionND(
        channels=C, ndim=1, max_samples=max_samples, num_heads=num_heads, 
        checkpoint=True, use_triton=True
    ).to(device)
    tel_triton.train()
    
    tel_pytorch = TelephoneAttentionND(
        channels=C, ndim=1, max_samples=max_samples, num_heads=num_heads,
        checkpoint=True, use_triton=False
    ).to(device)
    tel_pytorch.train()
    
    sdp_attn = SDPAttention(channels=C, num_heads=num_heads).to(device)
    sdp_attn.train()
    
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    
    if HAS_TRITON:
        tel_pytorch.load_state_dict(tel_triton.state_dict())
        
        torch.manual_seed(42)
        x_test = torch.randn(2, 512, C, device=device)
        
        out_triton, _ = tel_triton(x_test)
        out_pytorch, _ = tel_pytorch(x_test)
        
        fwd_diff = (out_triton - out_pytorch).abs().max().item()
        fwd_match = fwd_diff < 1e-3
        print(f"Forward match (Triton vs PyTorch): {'PASS' if fwd_match else 'FAIL'} (max diff: {fwd_diff:.2e})")
        
        x_test_tr = x_test.clone().requires_grad_(True)
        x_test_pt = x_test.clone().requires_grad_(True)
        
        out_tr, _ = tel_triton(x_test_tr)
        out_tr.sum().backward()
        
        out_pt, _ = tel_pytorch(x_test_pt)
        out_pt.sum().backward()
        
        dx_diff = (x_test_tr.grad - x_test_pt.grad).abs().max().item()
        dx_match = dx_diff < 5e-3
        print(f"d_x match (Triton vs PyTorch): {'PASS' if dx_match else 'FAIL'} (max diff: {dx_diff:.2e})")
        
        H = num_heads
        # wave_proj outputs [freq_pre, phase_pre] so first H rows are freq, last H are phase
        freq_grad_tr = tel_triton.wave_proj.weight.grad[:H].abs().sum().item()
        freq_grad_pt = tel_pytorch.wave_proj.weight.grad[:H].abs().sum().item()
        phase_grad_tr = tel_triton.wave_proj.weight.grad[H:].abs().sum().item()
        phase_grad_pt = tel_pytorch.wave_proj.weight.grad[H:].abs().sum().item()
        kernel_grad_tr = tel_triton.kernel_proj.weight.grad.abs().sum().item()
        kernel_grad_pt = tel_pytorch.kernel_proj.weight.grad.abs().sum().item()
        exponent_grad_tr = tel_triton.exponent_proj.weight.grad.abs().sum().item()
        exponent_grad_pt = tel_pytorch.exponent_proj.weight.grad.abs().sum().item()
        
        freq_grad_match = abs(freq_grad_tr - freq_grad_pt) / (freq_grad_pt + 1e-8) < 0.2
        phase_grad_match = abs(phase_grad_tr - phase_grad_pt) / (phase_grad_pt + 1e-8) < 0.2
        kernel_grad_match = abs(kernel_grad_tr - kernel_grad_pt) / (kernel_grad_pt + 1e-8) < 0.1
        exponent_grad_match = abs(exponent_grad_tr - exponent_grad_pt) / (exponent_grad_pt + 1e-8) < 0.2
        
        print(f"freq grad non-zero: {'PASS' if freq_grad_tr > 0 else 'FAIL'} (Triton: {freq_grad_tr:.2e}, PyTorch: {freq_grad_pt:.2e})")
        print(f"phase grad non-zero: {'PASS' if phase_grad_tr > 0 else 'FAIL'} (Triton: {phase_grad_tr:.2e}, PyTorch: {phase_grad_pt:.2e})")
        print(f"kernel_proj grad non-zero: {'PASS' if kernel_grad_tr > 0 else 'FAIL'} (Triton: {kernel_grad_tr:.2e}, PyTorch: {kernel_grad_pt:.2e})")
        print(f"exponent_proj grad non-zero: {'PASS' if exponent_grad_tr > 0 else 'FAIL'} (Triton: {exponent_grad_tr:.2e}, PyTorch: {exponent_grad_pt:.2e})")
        print(f"freq grad match: {'PASS' if freq_grad_match else 'FAIL'}")
        print(f"phase grad match: {'PASS' if phase_grad_match else 'FAIL'}")
        print(f"kernel_proj grad match: {'PASS' if kernel_grad_match else 'FAIL'}")
        print(f"exponent_proj grad match: {'PASS' if exponent_grad_match else 'FAIL'}")
        
        tel_triton.zero_grad()
        tel_pytorch.zero_grad()
        del x_test, x_test_tr, x_test_pt, out_triton, out_pytorch, out_tr, out_pt
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        torch.manual_seed(42)
        x_test = torch.randn(2, 512, C, device=device, requires_grad=True)
        out, _ = tel_pytorch(x_test)
        out.sum().backward()
        
        H = num_heads
        freq_grad = tel_pytorch.wave_proj.weight.grad[:H].abs().sum().item()
        phase_grad = tel_pytorch.wave_proj.weight.grad[H:].abs().sum().item()
        kernel_grad = tel_pytorch.kernel_proj.weight.grad.abs().sum().item()
        
        print(f"freq grad non-zero: {'PASS' if freq_grad > 0 else 'FAIL'} ({freq_grad:.2e})")
        print(f"phase grad non-zero: {'PASS' if phase_grad > 0 else 'FAIL'} ({phase_grad:.2e})")
        print(f"kernel_proj grad non-zero: {'PASS' if kernel_grad > 0 else 'FAIL'} ({kernel_grad:.2e})")
        print("(Triton not available, skipping Triton vs PyTorch comparison)")
        
        tel_pytorch.zero_grad()
        del x_test, out
        if device == "cuda":
            torch.cuda.empty_cache()
    
    print("=" * 60)
    print()
    
    def bench(model, x, is_telephone=False):
        if device == "cuda":
            torch.cuda.synchronize()
        with torch.no_grad():
            for _ in range(warmup_iters):
                out = model(x)[0] if is_telephone else model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
            
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            for _ in range(bench_iters):
                out = model(x)[0] if is_telephone else model(x)
                if device == "cuda":
                    torch.cuda.synchronize()
            fwd_time = (time.perf_counter() - start) / bench_iters * 1000
            fwd_mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        return fwd_time, fwd_mem
    
    def bench_bwd(model, x, is_telephone=False):
        if device == "cuda":
            torch.cuda.synchronize()
        for _ in range(warmup_iters):
            out = model(x)[0] if is_telephone else model(x)
            out.sum().backward()
            model.zero_grad()
            if device == "cuda":
                torch.cuda.synchronize()
        
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start = time.perf_counter()
        for _ in range(bench_iters):
            out = model(x)[0] if is_telephone else model(x)
            out.sum().backward()
            model.zero_grad()
            if device == "cuda":
                torch.cuda.synchronize()
        fwd_bwd_time = (time.perf_counter() - start) / bench_iters * 1000
        fwd_bwd_mem = torch.cuda.max_memory_allocated() / 1024**2 if device == "cuda" else 0
        return fwd_bwd_time, fwd_bwd_mem
    
    def run_bench(model, name, L, is_telephone=False):
        fwd = bwd = fwd_mem = bwd_mem = "OOM"
        if device == "cuda":
            torch.cuda.empty_cache()
        try:
            x = torch.randn(B, L, C, device=device)
            fwd, fwd_mem = bench(model, x, is_telephone=is_telephone)
            del x
            
            x = torch.randn(B, L, C, device=device, requires_grad=True)
            bwd, bwd_mem = bench_bwd(model, x, is_telephone=is_telephone)
            del x
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            if device == "cuda":
                torch.cuda.empty_cache()
        return fwd, bwd, fwd_mem, bwd_mem
    
    triton_label = "Triton" if tel_triton.use_triton else "PyTorch*"
    
    print("=" * 120)
    print(f"{'':>8} | {triton_label:^20} | {'PyTorch':^20} | {'SDPA':^20}")
    print(f"{'SeqLen':>8} | {'Fwd':>4} {'Bwd':>4} {'FM':>4} {'BM':>5} | {'Fwd':>4} {'Bwd':>4} {'FM':>4} {'BM':>5} | {'Fwd':>4} {'Bwd':>4} {'FM':>4} {'BM':>5}")
    print("=" * 120)
    
    for L in seq_lengths:
        tr_fwd, tr_bwd, tr_fwd_mem, tr_bwd_mem = run_bench(tel_triton, "triton", L, is_telephone=True)
        pt_fwd, pt_bwd, pt_fwd_mem, pt_bwd_mem = run_bench(tel_pytorch, "pytorch", L, is_telephone=True)
        sd_fwd, sd_bwd, sd_fwd_mem, sd_bwd_mem = run_bench(sdp_attn, "sdpa", L, is_telephone=False)
        
        def fmt(v):
            return f"{v:>4.0f}" if isinstance(v, float) else f"{v:>4}"
        def fmt_mem(v):
            return f"{v:>4.0f}" if isinstance(v, float) else f"{v:>4}"
        
        print(f"{L:>8} | {fmt(tr_fwd)} {fmt(tr_bwd)} {fmt_mem(tr_fwd_mem)} {fmt_mem(tr_bwd_mem):>5} | {fmt(pt_fwd)} {fmt(pt_bwd)} {fmt_mem(pt_fwd_mem)} {fmt_mem(pt_bwd_mem):>5} | {fmt(sd_fwd)} {fmt(sd_bwd)} {fmt_mem(sd_fwd_mem)} {fmt_mem(sd_bwd_mem):>5}")
    
    print("=" * 120)
    print(f"TelephoneAttention params: {sum(p.numel() for p in tel_triton.parameters()):,}")
    print(f"SDPA params: {sum(p.numel() for p in sdp_attn.parameters()):,}")
    if not tel_triton.use_triton:
        print("* Triton not available, using PyTorch implementation")
    
    # Chunk size sweep at L=8192
    if HAS_TRITON:
        print()
        print("=" * 80)
        print("CHUNK SIZE SWEEP (L=8192)")
        print("=" * 80)
        print(f"{'ChunkSize':>10} | {'Fwd(ms)':>8} {'Bwd(ms)':>8} {'FwdMem':>8} {'BwdMem':>8}")
        print("-" * 80)
        
        chunk_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
        L_sweep = 8192
        
        for cs in chunk_sizes:
            model_cs = TelephoneAttentionND(
                channels=C, ndim=1, max_samples=max_samples, num_heads=num_heads,
                chunk_size=cs, use_triton=True
            ).to(device)
            model_cs.load_state_dict(tel_triton.state_dict())
            model_cs.train()
            
            fwd, bwd, fwd_mem, bwd_mem = run_bench(model_cs, f"cs{cs}", L_sweep, is_telephone=True)
            
            def fmt_f(v):
                return f"{v:>8.1f}" if isinstance(v, float) else f"{v:>8}"
            
            print(f"{cs:>10} | {fmt_f(fwd)} {fmt_f(bwd)} {fmt_f(fwd_mem)} {fmt_f(bwd_mem)}")
            
            del model_cs
            if device == "cuda":
                torch.cuda.empty_cache()
        
        print("=" * 80)
