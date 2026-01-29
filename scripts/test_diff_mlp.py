#!/usr/bin/env python3
"""Compare SwiGLU vs DifferentialSwiGLU on tasks MLPs do well on."""

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


class MLPClassifier(nn.Module):
    def __init__(self, in_dim, width, n_layers, n_classes, mlp_cls):
        super().__init__()
        self.embed = nn.Linear(in_dim, width)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.norms.append(nn.LayerNorm(width))
            self.layers.append(mlp_cls(width))
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
    # label = XOR of signs of first 4 features
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
    # quadratic boundary: sum of squares of first 4 > sum of squares of next 4
    y = ((x[:, :4] ** 2).sum(1) > (x[:, 4:8] ** 2).sum(1)).astype(np.int64)
    return torch.from_numpy(x), torch.from_numpy(y)


def run(name, mlp_cls, task_fn, in_dim, width=64, n_layers=3, epochs=300, lr=1e-3, seed=42):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    x_train, y_train = task_fn(8192, dim=in_dim, seed=seed)
    x_test, y_test = task_fn(2048, dim=in_dim, seed=seed + 100)
    x_train, y_train = x_train.to(device), y_train.to(device)
    x_test, y_test = x_test.to(device), y_test.to(device)

    torch.manual_seed(seed + 1)
    model = MLPClassifier(in_dim, width, n_layers, 2, mlp_cls).to(device)
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
            print(f"  [{name:12s}] ep {ep+1:3d}  loss={loss.item():.4f}  test_acc={acc:.4f}  best={best_acc:.4f}")

    return best_acc, params


if __name__ == "__main__":
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
                acc_s, p_s = run("SwiGLU", SwiGLU, task_fn, in_dim, width=width, n_layers=n_layers)
                acc_d, p_d = run("DiffSwiGLU", DifferentialSwiGLU, task_fn, in_dim, width=width, n_layers=n_layers)
                print(f"  SwiGLU:     params={p_s:>7,}  acc={acc_s:.4f}")
                print(f"  DiffSwiGLU: params={p_d:>7,}  acc={acc_d:.4f}")
                winner = "DiffSwiGLU" if acc_d > acc_s else ("SwiGLU" if acc_s > acc_d else "TIE")
                print(f"  Winner: {winner}")
