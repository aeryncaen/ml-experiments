#!/usr/bin/env python3
"""Compare SwiGLU vs DifferentialSwiGLU on nonlinear function approximation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class MLPModel(nn.Module):
    def __init__(self, width, n_layers, mlp_cls):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.norms.append(nn.LayerNorm(width))
            self.layers.append(mlp_cls(width))
        self.norm_out = nn.LayerNorm(width)

    def forward(self, x):
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
        return self.norm_out(x)


def make_target_fn():
    """Random nonlinear target: a fixed 2-layer ReLU net."""
    W1 = torch.randn(64, 128)
    b1 = torch.randn(128)
    W2 = torch.randn(128, 64)
    b2 = torch.randn(64)
    def fn(x):
        h = F.relu(x @ W1 + b1)
        return h @ W2 + b2
    return fn


def run_experiment(name, mlp_cls, width=64, n_layers=4, n_train=8192, n_test=2048,
                   lr=3e-4, epochs=200, seed=42):
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    target_fn = make_target_fn()
    # move target params to device
    for obj in target_fn.__code__.co_consts:
        pass
    # just regenerate on device
    torch.manual_seed(0)
    W1 = torch.randn(width, width * 2, device=device)
    b1 = torch.randn(width * 2, device=device)
    W2 = torch.randn(width * 2, width, device=device)
    b2 = torch.randn(width, device=device)

    def target(x):
        h = F.relu(x @ W1 + b1)
        return h @ W2 + b2

    torch.manual_seed(seed)
    x_train = torch.randn(n_train, width, device=device)
    y_train = target(x_train).detach()
    x_test = torch.randn(n_test, width, device=device)
    y_test = target(x_test).detach()

    torch.manual_seed(seed + 1)
    model = MLPModel(width, n_layers, mlp_cls).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_test = float("inf")
    for ep in range(epochs):
        model.train()
        pred = model(x_train)
        loss = F.mse_loss(pred, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if (ep + 1) % 20 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                test_loss = F.mse_loss(model(x_test), y_test).item()
            best_test = min(best_test, test_loss)
            print(f"  [{name}] ep {ep+1:3d}  train={loss.item():.4f}  test={test_loss:.4f}  best={best_test:.4f}")

    return best_test, params


if __name__ == "__main__":
    print("=" * 60)
    print("Task: Learn a random 2-layer ReLU network (function approx)")
    print("=" * 60)

    for width in [64, 128]:
        for n_layers in [2, 4]:
            print(f"\n--- width={width}, layers={n_layers} ---")
            best_plain, params_plain = run_experiment(
                "SwiGLU", SwiGLU, width=width, n_layers=n_layers)
            best_diff, params_diff = run_experiment(
                "DiffSwiGLU", DifferentialSwiGLU, width=width, n_layers=n_layers)
            print(f"  SwiGLU:     params={params_plain:>8,}  best_test={best_plain:.4f}")
            print(f"  DiffSwiGLU: params={params_diff:>8,}  best_test={best_diff:.4f}")
            ratio = best_plain / max(best_diff, 1e-8)
            winner = "DiffSwiGLU" if best_diff < best_plain else "SwiGLU"
            print(f"  Winner: {winner} ({ratio:.2f}x)")
