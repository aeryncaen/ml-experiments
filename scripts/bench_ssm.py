"""Benchmark DS1 vs S4D vs Mamba on SSM litmus tests.

Tasks: delay-1, selective copy, induction head
Metrics: final loss, param count, peak memory, wall time
"""

import sys, os, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange, repeat

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 's4', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mamba'))

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
# Mamba (pure-PyTorch slow path, no CUDA ops needed)
# ---------------------------------------------------------------------------
class MambaPure(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2,
                 dt_min=0.001, dt_max=0.1, dt_scale=1.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv,
                                padding=d_conv - 1, groups=self.d_inner, bias=True)
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # dt bias init
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A init (S4D real)
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), 'n -> d n', d=self.d_inner).contiguous()
        self.A_log = nn.Parameter(torch.log(A))
        self.D_skip = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        # x: (B, L, D)
        B_batch, L, _ = x.shape
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        xz = rearrange(xz, 'b l d -> b d l')
        x_inner, z = xz.chunk(2, dim=1)  # each (B, d_inner, L)

        # causal conv
        x_inner = self.act(self.conv1d(x_inner)[..., :L])

        # SSM projections
        x_dbl = rearrange(x_inner, 'b d l -> (b l) d')
        x_dbl = self.x_proj(x_dbl)
        dt, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj.weight @ dt.t()  # (d_inner, bl)
        dt = rearrange(dt, 'd (b l) -> b d l', l=L)
        dt = F.softplus(dt + self.dt_proj.bias.unsqueeze(-1))  # (B, d_inner, L)

        B_ssm = rearrange(B_ssm, '(b l) n -> b l n', l=L)
        C_ssm = rearrange(C_ssm, '(b l) n -> b l n', l=L)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # selective scan (sequential, pure PyTorch)
        y = torch.zeros_like(x_inner)  # (B, d_inner, L)
        h = torch.zeros(B_batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        for t in range(L):
            dt_t = dt[:, :, t]  # (B, d_inner)
            B_t = B_ssm[:, t, :]  # (B, d_state)
            C_t = C_ssm[:, t, :]  # (B, d_state)
            x_t = x_inner[:, :, t]  # (B, d_inner)

            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, d_inner, d_state)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (B, d_inner, d_state)
            h = h * dA + x_t.unsqueeze(-1) * dB
            y_t = (h * C_t.unsqueeze(1)).sum(-1)  # (B, d_inner)
            y[:, :, t] = y_t

        y = y + x_inner * self.D_skip.unsqueeze(-1)
        y = y * self.act(z)
        y = rearrange(y, 'b d l -> b l d')
        return self.out_proj(y)


# ---------------------------------------------------------------------------
# DS1 wrapper
# ---------------------------------------------------------------------------
from ds_moe.model import DS1


class DS1Wrapper(nn.Module):
    def __init__(self, dim, state_dim=64, mimo_rank=4, n_iters=2, **kwargs):
        super().__init__()
        self.ds1 = DS1(dim=dim, state_dim=state_dim, mimo_rank=mimo_rank,
                        n_iters=n_iters, **kwargs)
        bank_size = DS1.bank_size(dim, state_dim, mimo_rank)
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

    if task_name == 'induction':
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

    # warmup
    for _ in range(3):
        if task_name == 'delay':
            x, t = gen_delay(B, L, dim, device=device)
            y = readout(model(x))
            F.mse_loss(y, t).backward()
        elif task_name == 'selective_copy':
            x, t = gen_selective_copy(B, L, dim, device=device)
            y = readout(model(proj_in(x)))
            F.mse_loss(y[:, -4:, :], t).backward()
        else:
            inp, tgt = gen_induction(B, L, vocab_size, device=device)
            y = head(model(embed(inp)))
            F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1)).backward()
        opt.zero_grad(set_to_none=True)

    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    losses = []
    t0 = time.perf_counter()

    for step in range(n_steps):
        if task_name == 'delay':
            x, target = gen_delay(B, L, dim, device=device)
            y = readout(model(x))
            loss = F.mse_loss(y, target)
        elif task_name == 'selective_copy':
            x, target = gen_selective_copy(B, L, dim, device=device)
            y = readout(model(proj_in(x)))
            loss = F.mse_loss(y[:, -4:, :], target)
        else:
            inp, tgt = gen_induction(B, L, vocab_size, device=device)
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))

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
    """Build models with roughly matched param counts (~51-57K)."""
    # DS1: 1 layer, ~56.6K params
    ds1 = DS1Wrapper(dim=dim, state_dim=64, mimo_rank=4, n_iters=2)

    # S4D: 3 layers × d_state=64 = ~49.9K params
    s4d = StackedSSM(lambda: S4D(d_model=dim, d_state=64), n_layers=3)

    # Mamba: 2 layers × expand=1, d_state=64 = ~51.1K params
    mamba = StackedSSM(lambda: MambaPure(d_model=dim, d_state=64, d_conv=4, expand=1), n_layers=2)

    return {
        'DS1': ds1,
        'S4D': s4d,
        'Mamba': mamba,
    }


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    dim = 64
    tasks = ['delay', 'selective_copy', 'induction']
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

    for task in tasks:
        for name in ['DS1', 'S4D', 'Mamba']:
            torch.manual_seed(42)
            model = make_models(dim)[name]
            r = train_task(model, task, dim, n_steps=n_steps, lr=1e-3, B=B, L=L, device=DEVICE)
            print(f"{name:<10} {task:<16} {r['initial']:>8.4f} {r['final']:>8.4f} {r['reduction']*100:>7.1f}% {r['wall_s']:>8.1f} {r['peak_mem_mb']:>8.1f}")
        print()
