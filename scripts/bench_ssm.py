"""Benchmark DS1 vs S4D vs Mamba on SSM litmus tests.

Tasks: delay-1, selective copy, induction head
Metrics: final loss, param count, peak memory, wall time

Requires: einops, mamba_ssm (for Mamba CUDA ops on CUDA box)
"""

import sys, os, time, math, random, hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
try:
    from torch.profiler import profile, ProfilerActivity
    HAS_PROFILER = True
except Exception:
    HAS_PROFILER = False
from einops import rearrange, repeat

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DETERMINISTIC = os.getenv("BENCH_DETERMINISTIC", "0") == "1"
if DETERMINISTIC:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
        except Exception:
            pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mamba'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 's5-pytorch'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'zoology'))

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# S4D (standalone, from s4/models/s4/s4d.py — no DropoutNd dependency)
# ---------------------------------------------------------------------------
class S4DKernel(nn.Module):
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.log_dt = nn.Parameter(log_dt)
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        return K


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, **kernel_args):
        super().__init__()
        self.h = d_model
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(self.h, N=d_state, **kernel_args)
        self.activation = nn.GELU()
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u):
        # u: (B, L, H) -> internal (B, H, L)
        u = u.transpose(-1, -2)
        L = u.size(-1)
        k = self.kernel(L=L)
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]
        y = y + u * self.D.unsqueeze(-1)
        y = self.activation(y)
        y = self.output_linear(y)
        return y.transpose(-1, -2)


# ---------------------------------------------------------------------------
# S5 (PyTorch port from s5-pytorch checkout)
# ---------------------------------------------------------------------------
from s5.s5_model import S5 as S5Module


class S5Wrapper(nn.Module):
    """Wraps S5 to accept (B, L, H) and return (B, L, H)."""
    def __init__(self, width, state_width=256):
        super().__init__()
        self.s5 = S5Module(width=width, state_width=state_width)

    def forward(self, x):
        return self.s5(x)


# ---------------------------------------------------------------------------
# Mamba (real implementation from mamba checkout)
# ---------------------------------------------------------------------------
from mamba_ssm.modules.mamba_simple import Mamba


class MambaWrapper(nn.Module):
    """Wraps Mamba to accept (B, L, D) and return (B, L, D)."""
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.mamba = Mamba(d_model=d_model, use_fast_path=True, **kwargs)

    def forward(self, x):
        return self.mamba(x)


# ---------------------------------------------------------------------------
# DS1 wrapper
# ---------------------------------------------------------------------------
from ds_moe.model import DS1, RMSNorm
from heuristic_secrets.models.backbone import SEBlock

# ---------------------------------------------------------------------------
# S6
# ---------------------------------------------------------------------------
from s6.s6 import S6 as S6Module
try:
    from s6.triton_s6 import TritonS6 as TritonS6Module
    HAS_TRITON_S6 = True
except ImportError:
    HAS_TRITON_S6 = False


class S6Wrapper(nn.Module):
    """Wraps S6 to accept (B, L, H) and return (B, L, H)."""
    def __init__(self, d_model, d_state=64, M=4, use_triton=True, **kwargs):
        super().__init__()
        if use_triton and HAS_TRITON_S6 and torch.cuda.is_available():
            self.s6 = TritonS6Module(d_model=d_model, d_state=d_state, M=M, **kwargs)
        else:
            self.s6 = S6Module(d_model=d_model, d_state=d_state, M=M, **kwargs)

    def forward(self, x):
        return self.s6(x)


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_hidden, bias=True):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden, bias=bias)
        self.w_up = nn.Linear(d_model, d_hidden, bias=bias)
        self.w_down = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class QuadConvMix(nn.Module):
    """Split channels into 4 paths: passthrough, causal conv, acausal conv, retrocausal conv.
    Recombine via SE gating."""
    def __init__(self, d_model, k=3, reduction=4):
        super().__init__()
        assert d_model % 4 == 0
        self.d_path = d_model // 4
        # causal: left-padded conv
        self.conv_causal = nn.Conv1d(self.d_path, self.d_path, k, padding=0, groups=self.d_path)
        # acausal: symmetric-padded conv
        self.conv_acausal = nn.Conv1d(self.d_path, self.d_path, k, padding=k // 2, groups=self.d_path)
        # retrocausal: right-padded conv
        self.conv_retro = nn.Conv1d(self.d_path, self.d_path, k, padding=0, groups=self.d_path)
        self.se = SEBlock(d_model, reduction=reduction)
        self.norm = RMSNorm(d_model)
        self.k = k

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        chunks = x.chunk(4, dim=-1)  # 4 x (B, L, d_path)

        p0 = chunks[0]  # passthrough

        # causal: pad left
        c1 = chunks[1].transpose(1, 2)  # (B, d_path, L)
        c1 = F.pad(c1, (self.k - 1, 0))
        c1 = self.conv_causal(c1).transpose(1, 2)

        # acausal: symmetric padding already in conv
        c2 = self.conv_acausal(chunks[2].transpose(1, 2)).transpose(1, 2)

        # retrocausal: pad right
        c3 = chunks[3].transpose(1, 2)
        c3 = F.pad(c3, (0, self.k - 1))
        c3 = self.conv_retro(c3).transpose(1, 2)

        out = torch.cat([p0, c1, c2, c3], dim=-1)  # (B, L, D)
        return self.norm(self.se(out) + x)


class MHABlock(nn.Module):
    """QuadConv → MHA."""
    def __init__(self, d_model, n_heads=4, se_reduction=4):
        super().__init__()
        self.quad_conv = QuadConvMix(d_model, k=3, reduction=se_reduction)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x):
        h = self.quad_conv(x)
        h, _ = self.attn(h, h, h)
        return h


class DS1Wrapper(nn.Module):
    def __init__(self, dim, state_dim=64, mimo_rank=4, n_iters=2, **kwargs):
        super().__init__()
        self.ds1 = DS1(dim=dim, state_dim=state_dim, mimo_rank=mimo_rank,
                        n_iters=n_iters, **kwargs)
        bank_size = DS1.bank_size(dim, state_dim, mimo_rank,
                                   out_gate=kwargs.get('out_gate', False),
                                   diff_attn=kwargs.get('diff_attn', False))
        self.bank = nn.Parameter(torch.randn(bank_size) * 0.02)

    def forward(self, x):
        return self.ds1(x, self.bank)


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------
def gen_delay(B, L, vocab_size=32, delay=1, device='cpu'):
    """Token input, target is same sequence shifted by `delay` positions."""
    x = torch.randint(0, vocab_size, (B, L), device=device)
    target = torch.zeros_like(x)
    target[:, delay:] = x[:, :L - delay]
    return x, target


def gen_selective_copy(B, L, vocab_size=32, n_markers=4, device='cpu'):
    """Token input with marker channel. Target is the n_markers marked tokens.
    Input: (B, L) tokens + (B, L) marker flags → fed as 2-token tuples via embedding.
    Output: (B, n_markers) token predictions at end of sequence."""
    tokens = torch.randint(0, vocab_size, (B, L), device=device)
    markers = torch.zeros(B, L, dtype=torch.long, device=device)
    target = torch.zeros(B, n_markers, dtype=torch.long, device=device)
    for b in range(B):
        idxs = torch.randperm(L, device=device)[:n_markers].sort().values
        markers[b, idxs] = 1
        target[b] = tokens[b, idxs]
    return tokens, markers, target


def gen_parity(B, L, device='cpu'):
    """Binary input {0,1}, target is running XOR (cumulative parity)."""
    x = torch.randint(0, 2, (B, L), device=device)
    target = x.cumsum(dim=1) % 2  # running parity
    return x, target


def gen_mod_arith(B, L, mod_base=5, device='cpu'):
    """Digit input 0..mod_base-1, target is running sum mod base."""
    x = torch.randint(0, mod_base, (B, L), device=device)
    target = x.cumsum(dim=1) % mod_base
    return x, target


def gen_induction(B, L, vocab_size=32, device='cpu'):
    half = L // 2
    pattern = torch.randint(0, vocab_size, (B, half), device=device)
    seq = torch.cat([pattern, pattern], dim=1)  # (B, L)
    input_seq = seq[:, :-1]
    target_seq = seq[:, 1:]
    return input_seq, target_seq


# ---------------------------------------------------------------------------
# Data pregeneration, disk caching, and GPU preloading
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent / '.bench_cache'


def _cache_key(task_name, n_steps, B, L, seed):
    raw = f"{task_name}_{n_steps}_{B}_{L}_{seed}"
    return hashlib.md5(raw.encode()).hexdigest()


def pregen_task_data(task_name, n_steps, B, L, seed, device='cpu'):
    """Generate all batches for a task, cache to disk, return list of GPU tensors."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(task_name, n_steps, B, L, seed)
    cache_path = CACHE_DIR / f"{task_name}_{key}.pt"

    if cache_path.exists():
        batches = torch.load(cache_path, weights_only=True)
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        batches = []
        for _ in range(n_steps + 3):  # +3 for warmup
            if task_name == 'delay':
                batches.append(gen_delay(B, L, vocab_size=32, device='cpu'))
            elif task_name == 'selective_copy':
                batches.append(gen_selective_copy(B, L, vocab_size=32, device='cpu'))
            elif task_name == 'parity':
                batches.append(gen_parity(B, L, device='cpu'))
            elif task_name == 'mod_arith':
                batches.append(gen_mod_arith(B, L, device='cpu'))
            elif task_name == 'induction':
                batches.append(gen_induction(B, L, vocab_size=32, device='cpu'))
        torch.save(batches, cache_path)

    # Preload everything to target device
    def _to(t):
        return t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t
    return [tuple(_to(t) for t in batch) for batch in batches]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_task(model, task_name, dim, n_steps=2000, lr=1e-3, B=32, L=32, device='cpu',
               preloaded_data=None):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    if task_name == 'delay':
        vocab_size = 32
        embed = nn.Embedding(vocab_size, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=lr)
    elif task_name == 'selective_copy':
        vocab_size = 32
        embed = nn.Embedding(vocab_size, dim).to(device)
        marker_embed = nn.Embedding(2, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters())
                         + list(marker_embed.parameters()) + list(head.parameters()), lr=lr)
    elif task_name == 'parity':
        vocab_size = 2
        embed = nn.Embedding(vocab_size, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=lr)
    elif task_name == 'mod_arith':
        vocab_size = 5
        embed = nn.Embedding(vocab_size, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=lr)
    elif task_name == 'induction':
        vocab_size = 32
        embed = nn.Embedding(vocab_size, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=lr)

    data = preloaded_data
    assert data is not None, "preloaded_data required — use pregen_task_data()"

    def _step(idx):
        batch = data[idx]
        if task_name == 'delay':
            inp, tgt = batch
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc
        elif task_name == 'selective_copy':
            tokens, markers, tgt = batch
            x = embed(tokens) + marker_embed(markers)
            y = head(model(x))
            n_markers = tgt.shape[1]
            y_last = y[:, -n_markers:]
            loss = F.cross_entropy(y_last.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y_last.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc
        elif task_name == 'parity':
            inp, tgt = batch
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc
        elif task_name == 'mod_arith':
            inp, tgt = batch
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc
        else:
            inp, tgt = batch
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc

    # warmup (first 3 batches)
    for i in range(3):
        _step(i)[0].backward()
        opt.zero_grad(set_to_none=True)

    # Module-level memory profiling (CUDA only, S6 only)
    if device == 'cuda' and isinstance(model, S6Wrapper):
        mem_stats = defaultdict(int)
        mem_baseline = {}
        hooks = []
        module_names = {m: n for n, m in model.named_modules()}

        def _pre_hook(module, _inputs):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            mem_baseline[id(module)] = torch.cuda.memory_allocated()

        def _post_hook(module, _inputs, _outputs):
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            base = mem_baseline.get(id(module), 0)
            delta = max(0, peak - base)
            name = module_names.get(module, module.__class__.__name__)
            mem_stats[name] = max(mem_stats[name], delta)

        for m in model.modules():
            if len(list(m.children())) == 0:
                hooks.append(m.register_forward_pre_hook(_pre_hook))
                hooks.append(m.register_forward_hook(_post_hook))

        with torch.no_grad():
            _step(0)

        for h in hooks:
            h.remove()

        # print("\n[Module Memory Peaks] (approx, per leaf module, delta)")
        # top = sorted(mem_stats.items(), key=lambda kv: kv[1], reverse=True)[:20]
        # for name, peak in top:
        #     print(f"  {name:50s} {peak / (1024**2):8.2f} MB")

    # Optional profiling for S6 on CUDA (forward + backward on one batch)
    if device == 'cuda' and isinstance(model, S6Wrapper) and HAS_PROFILER:
        for i in range(50):
            loss, _ = _step(i % 3)
            opt.zero_grad(set_to_none=True)
            loss.backward()
        torch.cuda.synchronize()
        opt.zero_grad(set_to_none=True)
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                     record_shapes=True,
                     profile_memory=True) as prof:
            loss, _ = _step(0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
        torch.cuda.synchronize()
        # print("\n[S6 profile] Forward+Backward (1 batch)")
        # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))

    mem_baseline = 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_baseline = torch.cuda.memory_allocated()

    losses = []
    t0 = time.perf_counter()

    accs = []
    for step in range(n_steps):
        loss, acc = _step(3 + step)  # offset past warmup batches

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        accs.append(acc)

    if device == 'cuda':
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    peak_mem = 0
    if device == 'cuda':
        peak_mem = (torch.cuda.max_memory_allocated() - mem_baseline) / 1024 / 1024  # MB

    initial = sum(losses[:10]) / 10
    final = sum(losses[-10:]) / 10
    final_acc = sum(accs[-10:]) / 10
    return {
        'initial': initial,
        'final': final,
        'reduction': 1.0 - final / initial if initial > 0 else 0,
        'acc': final_acc,
        'wall_s': wall,
        'peak_mem_mb': peak_mem,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())


class StackedModel(nn.Module):
    """Stack N identical layers with pre-norm residual connections."""
    def __init__(self, make_layer, n_layers, dim):
        super().__init__()
        self.layers = nn.ModuleList([make_layer() for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)

    def forward(self, x):
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
        return self.final_norm(x)


def _stack(make_layer, n_layers, dim):
    """Always wrap with pre-norm residual stacking."""
    return StackedModel(make_layer, n_layers, dim)


def make_models(dim, n_layers=1):
    """Build models with configurable depth. Param counts are per-layer (~50-57K)."""
    models = {}

    # DS1 base
    # models['DS1'] = _stack(
    #     lambda: DS1Wrapper(dim=dim, state_dim=64, mimo_rank=4, n_iters=2),
    #     n_layers, dim)

    # DS1++: DS1 + signed sparse attention
    models['DS1++'] = _stack(
        lambda: DS1Wrapper(dim=dim, state_dim=48, mimo_rank=4, n_iters=2, diff_attn=True),
        n_layers, dim)

    # S4D: ~50K params/layer
    models['S4D'] = _stack(lambda: S4D(d_model=dim, d_state=320), n_layers, dim)

    # S5: ~50K params/layer
    models['S5'] = _stack(lambda: S5Wrapper(width=dim, state_width=256), n_layers, dim)

    # Mamba: ~51K params/layer
    models['Mamba'] = _stack(lambda: MambaWrapper(d_model=dim, d_state=64, d_conv=4, expand=2), n_layers, dim)

    # S6: ~50K params/layer
    models['S6'] = _stack(
        lambda: S6Wrapper(d_model=dim, d_state=64, M=4, num_heads=4),
        n_layers, dim)

    # MHA: ~19K params/layer (QuadConv + MHA only, no MLP/Mamba to scale)
    models['MHA'] = _stack(
        lambda: MHABlock(d_model=dim, n_heads=4),
        n_layers, dim)

    return models


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    dim = 64
    n_layers = 1
    tasks = ['delay', 'selective_copy', 'induction', 'parity', 'mod_arith']
    n_steps = 2000
    B, L = 32, 32

    models_info = make_models(dim, n_layers=n_layers)
    print(f"\n{'Model':<10} {'Params':>10} {'Layers':>8}")
    print('-' * 30)
    for name, m in models_info.items():
        print(f"{name:<10} {count_params(m):>10,} {n_layers:>8}")

    print(f"\nRunning {n_steps} steps, B={B}, L={L}, dim={dim}, layers={n_layers}")
    print('=' * 90)

    header = f"{'Model':<10} {'Task':<16} {'Init':>8} {'Final':>8} {'Acc':>8} {'Wall(s)':>8} {'Mem(MB)':>9}"
    print(header)
    print('-' * 90)

    all_names = list(models_info.keys())
    for task in tasks:
        print(f"Pregenerating {task} data...")
        task_data = pregen_task_data(task, n_steps, B, L, SEED, device=DEVICE)
        print(f"  {len(task_data)} batches ready on {DEVICE}")
        for name in all_names:
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            model = make_models(dim, n_layers=n_layers)[name]
            r = train_task(model, task, dim, n_steps=n_steps, lr=1e-3, B=B, L=L, device=DEVICE,
                           preloaded_data=task_data)
            print(f"{name:<10} {task:<16} {r['initial']:>8.4f} {r['final']:>8.4f} {r['acc']:>7.1%} {r['wall_s']:>8.1f} {r['peak_mem_mb']:>8.1f}")
        del task_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print()
