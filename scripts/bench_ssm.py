"""Benchmark DS1 vs S4D vs Mamba on SSM litmus tests.

Tasks: delay-1, selective copy, induction head
Metrics: final loss, param count, peak memory, wall time

Requires: einops, mamba_ssm (for Mamba CUDA ops on CUDA box)
"""

import sys, os, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mamba'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 's5-pytorch'))

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
from ds_moe.model import DS1


class DS1Wrapper(nn.Module):
    def __init__(self, dim, state_dim=64, mimo_rank=4, n_iters=2, **kwargs):
        super().__init__()
        self.ds1 = DS1(dim=dim, state_dim=state_dim, mimo_rank=mimo_rank,
                        n_iters=n_iters, **kwargs)
        bank_size = DS1.bank_size(dim, state_dim, mimo_rank,
                                   out_gate=kwargs.get('out_gate', False))
        self.bank = nn.Parameter(torch.randn(bank_size) * 0.02)

    def forward(self, x):
        return self.ds1(x, self.bank)


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------
def gen_delay(B, L, D, delay=1, device='cpu'):
    x = torch.randn(B, L, D, device=device)
    target = torch.zeros_like(x)
    target[:, delay:, :] = x[:, :L - delay, :]
    return x, target


def gen_selective_copy(B, L, D, n_markers=4, device='cpu'):
    x = torch.randn(B, L, D, device=device)
    target = torch.zeros(B, n_markers, D, device=device)
    markers = torch.zeros(B, L, 1, device=device)
    for b in range(B):
        idxs = torch.randperm(L, device=device)[:n_markers].sort().values
        markers[b, idxs, 0] = 1.0
        target[b] = x[b, idxs]
    x = torch.cat([x, markers], dim=-1)  # (B, L, D+1)
    return x, target


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
# Training loop
# ---------------------------------------------------------------------------
def train_task(model, task_name, dim, n_steps=2000, lr=1e-3, B=32, L=32, device='cpu'):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    if task_name == 'parity':
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
    elif task_name == 'selective_copy':
        proj_in = nn.Linear(dim + 1, dim).to(device)
        readout = nn.Linear(dim, dim).to(device)
        opt = optim.Adam(list(model.parameters()) + list(proj_in.parameters()) + list(readout.parameters()), lr=lr)
    else:
        readout = nn.Linear(dim, dim).to(device)
        opt = optim.Adam(list(model.parameters()) + list(readout.parameters()), lr=lr)

    def _step(task_name):
        if task_name == 'delay':
            x, t = gen_delay(B, L, dim, device=device)
            y = readout(model(x))
            return F.mse_loss(y, t)
        elif task_name == 'selective_copy':
            x, t = gen_selective_copy(B, L, dim, device=device)
            y = readout(model(proj_in(x)))
            return F.mse_loss(y[:, -4:, :], t)
        elif task_name == 'parity':
            inp, tgt = gen_parity(B, L, device=device)
            y = head(model(embed(inp)))
            return F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
        elif task_name == 'mod_arith':
            inp, tgt = gen_mod_arith(B, L, device=device)
            y = head(model(embed(inp)))
            return F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
        else:
            inp, tgt = gen_induction(B, L, vocab_size, device=device)
            y = head(model(embed(inp)))
            return F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))

    # warmup
    for _ in range(3):
        _step(task_name).backward()
        opt.zero_grad(set_to_none=True)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    losses = []
    t0 = time.perf_counter()

    for step in range(n_steps):
        loss = _step(task_name)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    if device == 'cuda':
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    peak_mem = 0
    if device == 'cuda':
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

    initial = sum(losses[:10]) / 10
    final = sum(losses[-10:]) / 10
    return {
        'initial': initial,
        'final': final,
        'reduction': 1.0 - final / initial if initial > 0 else 0,
        'wall_s': wall,
        'peak_mem_mb': peak_mem,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())


class StackedSSM(nn.Module):
    """Stack N identical SSM layers with residual connections."""
    def __init__(self, make_layer, n_layers):
        super().__init__()
        self.layers = nn.ModuleList([make_layer() for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


def make_models(dim):
    """Build models with roughly matched param counts (~51-61K)."""
    # DS1 base: 1 layer, ~56.6K params
    ds1 = DS1Wrapper(dim=dim, state_dim=64, mimo_rank=4, n_iters=2)

    # DS1++: DS1 + differential attention (reuses B,C projections)
    ds1_pp = DS1Wrapper(dim=dim, state_dim=64, mimo_rank=4, n_iters=2,
                         diff_attn=True)

    # S4D: 3 layers × d_state=64 = ~49.9K params
    s4d = StackedSSM(lambda: S4D(d_model=dim, d_state=64), n_layers=3)

    # S5: state_width=256 = ~49.7K params
    s5 = S5Wrapper(width=dim, state_width=256)

    # Mamba: 2 layers × expand=1, d_state=64 = ~51.1K params
    mamba = StackedSSM(lambda: MambaWrapper(d_model=dim, d_state=64, d_conv=4, expand=1), n_layers=2)

    return {
        'DS1': ds1,
        'DS1++': ds1_pp,
        # 'S4D': s4d,
        # 'S5': s5,
        # 'Mamba': mamba,
    }


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    dim = 64
    tasks = ['delay', 'selective_copy', 'induction', 'parity', 'mod_arith']
    n_steps = 2000
    B, L = 32, 32

    models_info = make_models(dim)
    print(f"\n{'Model':<10} {'Params':>10}")
    print('-' * 22)
    for name, m in models_info.items():
        print(f"{name:<10} {count_params(m):>10,}")

    print(f"\nRunning {n_steps} steps, B={B}, L={L}, dim={dim}")
    print('=' * 90)

    header = f"{'Model':<10} {'Task':<16} {'Init':>8} {'Final':>8} {'Reduc%':>8} {'Wall(s)':>8} {'Mem(MB)':>9}"
    print(header)
    print('-' * 90)

    all_names = list(models_info.keys())
    for task in tasks:
        for name in all_names:
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            model = make_models(dim)[name]
            r = train_task(model, task, dim, n_steps=n_steps, lr=1e-3, B=B, L=L, device=DEVICE)
            print(f"{name:<10} {task:<16} {r['initial']:>8.4f} {r['final']:>8.4f} {r['reduction']*100:>7.1f}% {r['wall_s']:>8.1f} {r['peak_mem_mb']:>8.1f}")
        print()
