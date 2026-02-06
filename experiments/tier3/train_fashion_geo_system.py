#!/usr/bin/env python3
"""
Tier 3: Geometry-informed training on Fashion-MNIST.

Compares:
  - baseline: standard MLP init + ERM
  - geo_system: spectral first-layer init + optional warmup shaping

This mirrors the digits prototype but on a harder objective.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import FashionMNIST

from chi2_optimizer import Chi2TrustRegion


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_masks() -> tuple[np.ndarray, np.ndarray]:
    border = np.zeros((28, 28), dtype=bool)
    border[:3, :] = True
    border[-3:, :] = True
    border[:, :3] = True
    border[:, -3:] = True

    center = np.zeros((28, 28), dtype=bool)
    center[8:20, 8:20] = True
    return border.ravel(), center.ravel()


def apply_shift(X: np.ndarray, shift_type: str, strength: float) -> np.ndarray:
    """X is standardized flattened vectors [N, 784]."""
    # Work in standardized space to avoid needing inverse transforms.
    Xs = X.copy()
    border_mask, center_mask = build_masks()

    if shift_type == "border_erase":
        Xs[:, border_mask] *= (1.0 - strength)
    elif shift_type == "center_erase":
        Xs[:, center_mask] *= (1.0 - strength)
    elif shift_type == "contrast":
        Xs = (1.0 + strength) * Xs
    else:
        raise ValueError(f"Unknown shift_type={shift_type}")

    return Xs


class MLP(nn.Module):
    def __init__(self, d_in: int = 784, h1: int = 512, h2: int = 256, d_out: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def compute_class_operator(X: np.ndarray, y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    d = X.shape[1]
    mu = X.mean(axis=0)
    M = np.zeros((d, d), dtype=np.float64)
    n = len(y)
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        pc = len(idx) / n
        muc = X[idx].mean(axis=0)
        diff = (muc - mu).reshape(-1, 1)
        M += pc * (diff @ diff.T)
    return M


def geometry_init_first_layer(fc1: nn.Linear, eigvecs: np.ndarray, eigvals: np.ndarray) -> None:
    out_dim, in_dim = fc1.weight.shape
    top = eigvecs[:, ::-1]
    vals = eigvals[::-1]
    k = min(in_dim, top.shape[1])

    W = np.zeros((out_dim, in_dim), dtype=np.float32)
    kaiming_std = np.sqrt(2.0 / in_dim)
    w_scale = vals[:k] / (vals[:k].max() + 1e-12)

    for i in range(out_dim):
        j = i % k
        sign = 1.0 if (i % 2 == 0) else -1.0
        vec = top[:, j]
        W[i] = (sign * vec * (0.5 + 0.5 * w_scale[j]) * kaiming_std).astype(np.float32)

    W += np.random.normal(0.0, kaiming_std * 0.03, size=W.shape).astype(np.float32)
    with torch.no_grad():
        fc1.weight.copy_(torch.from_numpy(W))
        fc1.bias.zero_()


def standard_init(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)


@dataclass
class RunConfig:
    epochs: int = 20
    batch_size: int = 256
    lr: float = 7e-4
    seeds: int = 3
    reg_lambda: float = 1e-3
    reg_warmup_epochs: int = 8
    max_train: int = 60000
    optimizer: str = "adam"
    trust_radius: float = 0.05
    beta2: float = 0.99
    lr_scale: float = 1.0


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    tx = torch.from_numpy(X.astype(np.float32))
    ty = torch.from_numpy(y.astype(np.int64))
    ds = torch.utils.data.TensorDataset(tx, ty)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        tx = torch.from_numpy(X.astype(np.float32)).to(device)
        logits = model(tx)
        pred = logits.argmax(dim=1).cpu().numpy()
    return float((pred == y).mean())


def train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    mode: str,
    cfg: RunConfig,
    seed: int,
    device: torch.device,
) -> dict:
    set_seed(seed)
    model = MLP().to(device)

    M = compute_class_operator(X_train, y_train)
    eigvals, eigvecs = np.linalg.eigh(M)
    k = min(64, eigvecs.shape[1])
    top = eigvecs[:, -k:]
    P_top = top @ top.T
    P_top_t = torch.from_numpy(P_top.astype(np.float32)).to(device)

    standard_init(model)
    if mode == "geo_system":
        geometry_init_first_layer(model.fc1, eigvecs, eigvals)

    if cfg.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "chi2":
        opt = Chi2TrustRegion(
            model.parameters(),
            trust_radius=cfg.trust_radius,
            beta2=cfg.beta2,
            lr_scale=cfg.lr_scale,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)
    history = []

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            if mode == "geo_system" and epoch < cfg.reg_warmup_epochs and cfg.reg_lambda > 0:
                W = model.fc1.weight
                W_proj = W @ P_top_t
                reg = torch.mean((W - W_proj) ** 2)
                loss = loss + cfg.reg_lambda * reg

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = xb.shape[0]
            total_loss += float(loss.item()) * bs
            n_seen += bs

        val_acc = accuracy(model, X_val, y_val, device)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / max(n_seen, 1), "val_acc": val_acc})

    clean_acc = accuracy(model, X_test, y_test, device)
    X_border = apply_shift(X_test, "border_erase", 0.85)
    X_center = apply_shift(X_test, "center_erase", 0.85)
    X_contrast = apply_shift(X_test, "contrast", 0.25)

    acc_border = accuracy(model, X_border, y_test, device)
    acc_center = accuracy(model, X_center, y_test, device)
    acc_contrast = accuracy(model, X_contrast, y_test, device)
    robust_avg = (acc_border + acc_center + acc_contrast) / 3.0

    return {
        "mode": mode,
        "seed": seed,
        "clean_acc": clean_acc,
        "acc_border": acc_border,
        "acc_center": acc_center,
        "acc_contrast": acc_contrast,
        "robust_avg": robust_avg,
        "history": history,
    }


def summarize(runs: list[dict], mode: str) -> dict:
    sub = [r for r in runs if r["mode"] == mode]
    keys = ["clean_acc", "acc_border", "acc_center", "acc_contrast", "robust_avg"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in sub], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    for ep in [3, 5]:
        vals = []
        for r in sub:
            h = r["history"]
            idx = min(ep - 1, len(h) - 1)
            vals.append(h[idx]["val_acc"])
        vals = np.array(vals, dtype=float)
        out[f"val_acc_ep{ep}"] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    return out


def load_fashion(data_dir: str, max_train: int | None = None):
    train_ds = FashionMNIST(root=data_dir, train=True, download=True)
    test_ds = FashionMNIST(root=data_dir, train=False, download=True)

    X_train = train_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float64) / 255.0
    y_train = train_ds.targets.numpy().astype(np.int64)
    X_test = test_ds.data.numpy().reshape(-1, 28 * 28).astype(np.float64) / 255.0
    y_test = test_ds.targets.numpy().astype(np.int64)

    if max_train is not None and max_train < len(X_train):
        X_train = X_train[:max_train]
        y_train = y_train[:max_train]

    # Split train -> train/val
    n = len(X_train)
    n_val = max(5000, int(0.1 * n))
    X_val = X_train[-n_val:]
    y_val = y_train[-n_val:]
    X_train = X_train[:-n_val]
    y_train = y_train[:-n_val]

    # Standardize per feature using train stats
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-6, 1.0, sd)

    X_train = (X_train - mu) / sd
    X_val = (X_val - mu) / sd
    X_test = (X_test - mu) / sd
    return X_train, y_train, X_val, y_val, X_test, y_test


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--reg-lambda", type=float, default=1e-3)
    parser.add_argument("--reg-warmup-epochs", type=int, default=8)
    parser.add_argument("--max-train", type=int, default=60000)
    parser.add_argument("--optimizer", type=str, choices=["adam", "chi2"], default="adam")
    parser.add_argument("--trust-radius", type=float, default=0.05)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--out", type=str, default="tier3_fashion_results.json")
    args = parser.parse_args()

    cfg = RunConfig(
        epochs=args.epochs,
        seeds=args.seeds,
        batch_size=args.batch_size,
        lr=args.lr,
        reg_lambda=args.reg_lambda,
        reg_warmup_epochs=args.reg_warmup_epochs,
        max_train=args.max_train,
        optimizer=args.optimizer,
        trust_radius=args.trust_radius,
        beta2=args.beta2,
        lr_scale=args.lr_scale,
    )

    device = get_device()
    print(f"Device: {device}")
    print(f"Optimizer: {cfg.optimizer}")

    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion(args.data_dir, args.max_train)
    print(f"Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    runs = []
    for seed in range(cfg.seeds):
        for mode in ["baseline", "geo_system"]:
            print(f"Running mode={mode}, seed={seed}")
            res = train_one(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                mode=mode,
                cfg=cfg,
                seed=seed,
                device=device,
            )
            runs.append(res)

    base = summarize(runs, "baseline")
    geo = summarize(runs, "geo_system")

    print("\n=== Summary (mean +- std across seeds) ===")
    for key in ["val_acc_ep3", "val_acc_ep5", "clean_acc", "robust_avg", "acc_border", "acc_center", "acc_contrast"]:
        b = base[key]
        g = geo[key]
        print(
            f"{key:12s} | baseline {b['mean']:.4f} +- {b['std']:.4f} "
            f"| geo_system {g['mean']:.4f} +- {g['std']:.4f} "
            f"| delta {g['mean'] - b['mean']:+.4f}"
        )

    result = {
        "config": vars(args),
        "device": str(device),
        "baseline": base,
        "geo_system": geo,
        "runs": runs,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote results: {out_path}")


if __name__ == "__main__":
    main()
