#!/usr/bin/env python3
"""Compare SwiGLU vs DifferentialSwiGLU (param-matched) on classification tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.insert(0, "src")

from heuristic_secrets.models.ripple_attention import DifferentialSwiGLU


class SwiGLU(nn.Module):
    def __init__(self, width, mult=4):
        super().__init__()
        hidden = int(width * mult * 2 / 3)
        hidden = ((hidden + 7) // 8) * 8
        self.gate = nn.Linear(width, hidden, bias=False)
        self.up = nn.Linear(width, hidden, bias=False)
        self.down = nn.Linear(hidden, width, bias=False)
    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def find_matched_mult(width, target_params, lo=2.0, hi=12.0):
    for _ in range(40):
        mid = (lo + hi) / 2
        p = count_params(DifferentialSwiGLU(width, mult=mid))
        if p < target_params:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def make_matched_diff_swiglu(width):
    target = count_params(SwiGLU(width))
    mult = find_matched_mult(width, target)
    d = DifferentialSwiGLU(width, mult=mult)
    actual = count_params(d)
    return d, actual, target


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, width, n_layers, n_classes, mlp_factory):
        super().__init__()
        self.embed = nn.Linear(in_dim, width)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.norms.append(nn.LayerNorm(width))
            self.layers.append(mlp_factory(width))
        self.norm_out = nn.LayerNorm(width)
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        x = self.embed(x)
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
        return self.head(self.norm_out(x))


def make_xor_data(n, dim=16, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    bits = (x[:, :4] > 0).astype(int)
    y = bits.sum(axis=1) % 2
    return torch.from_numpy(x), torch.from_numpy(y).long()


def make_sparse_parity_data(n, dim=32, k=6, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    bits = (x[:, :k] > 0).astype(int)
    y = bits.sum(axis=1) % 2
    return torch.from_numpy(x), torch.from_numpy(y).long()


def make_nonlinear_boundary(n, dim=16, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    y = ((x[:, :4] ** 2).sum(1) > (x[:, 4:8] ** 2).sum(1)).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def run_single(name, mlp_factory, task_fn, in_dim, width=64, n_layers=3, epochs=300, lr=1e-3, seed=42, verbose=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    x_train, y_train = task_fn(8192, dim=in_dim, seed=seed)
    x_test, y_test = task_fn(2048, dim=in_dim, seed=seed + 100)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    torch.manual_seed(seed + 1)
    model = MLPClassifier(in_dim, width, n_layers, 2, mlp_factory).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_acc = 0.0
    for ep in range(epochs):
        model.train()
        logits = model(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if (ep + 1) % 50 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(x_test).argmax(-1) == y_test).float().mean().item()
            best_acc = max(best_acc, acc)
            if verbose:
                print(f"  [{name:12s}] ep {ep+1:3d}  loss={loss.item():.4f}  test_acc={acc:.4f}  best={best_acc:.4f}")

    return best_acc, params


def run(name, mlp_factory, task_fn, in_dim, width=64, n_layers=3, epochs=300, lr=1e-3, n_runs=5):
    accs = []
    params = None
    for i in range(n_runs):
        seed = 42 + i * 1000
        acc, params = run_single(name, mlp_factory, task_fn, in_dim, width=width,
                                 n_layers=n_layers, epochs=epochs, lr=lr, seed=seed,
                                 verbose=(i == 0))
        accs.append(acc)
        print(f"  [{name:12s}] run {i+1}/{n_runs}  seed={seed}  best_acc={acc:.4f}")
    mean = np.mean(accs)
    std = np.std(accs)
    print(f"  [{name:12s}] mean={mean:.4f} ± {std:.4f}  (n={n_runs})")
    return mean, std, params


if __name__ == "__main__":
    # First show param matching works
    print("Param matching check:")
    for w in [64, 128, 256]:
        target = count_params(SwiGLU(w))
        mult = find_matched_mult(w, target)
        actual = count_params(DifferentialSwiGLU(w, mult=mult))
        print(f"  width={w:3d}  SwiGLU={target:>7,}  DiffSwiGLU={actual:>7,}  mult={mult:.3f}  delta={abs(actual-target)}")

    # Build param-matched factory
    _mult_cache = {}
    def matched_diff_factory(width):
        if width not in _mult_cache:
            target = count_params(SwiGLU(width))
            _mult_cache[width] = find_matched_mult(width, target)
        return DifferentialSwiGLU(width, mult=_mult_cache[width])

    tasks = [
        ("XOR-4 (dim=16)", make_xor_data, 16),
        ("sparse-parity-6 (dim=32)", make_sparse_parity_data, 32),
        ("quadratic boundary (dim=16)", make_nonlinear_boundary, 16),
    ]

    for task_name, task_fn, in_dim in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"{'='*60}")

        for width in [64, 128]:
            for n_layers in [2, 4]:
                print(f"\n--- width={width}, layers={n_layers} ---")
                m_s, s_s, p_s = run("SwiGLU", SwiGLU, task_fn, in_dim, width=width, n_layers=n_layers)
                m_d, s_d, p_d = run("DiffSwiGLU", matched_diff_factory, task_fn, in_dim, width=width, n_layers=n_layers)
                print(f"\n  SwiGLU:     params={p_s:>7,}  acc={m_s:.4f} ± {s_s:.4f}")
                print(f"  DiffSwiGLU: params={p_d:>7,}  acc={m_d:.4f} ± {s_d:.4f}")
                delta = m_d - m_s
                print(f"  Δ(Diff-Swi): {delta:+.4f}  {'✓ Diff wins' if delta > 0 else '✗ Swi wins' if delta < 0 else 'TIE'}")
