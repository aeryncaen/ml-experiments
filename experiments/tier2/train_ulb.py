"""
Train a MoE ULB on Fashion-MNIST (clean or augmented).

Patches 28x28 images into 4x4 patches (49 tokens), projects to dim,
runs through MoE ULB stack, mean-pools, classifies.

Usage:
  python train_ulb.py --data_dir data/fmnist_augmented
  python train_ulb.py --clean
"""

import argparse
import json
import math
import numpy as np
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import FashionMNIST
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from ulb import ULBBlock, ULBConfig, MoEStackedULB


class ULBClassifier(nn.Module):
    """Patch-based ULB classifier for 28x28 images."""

    def __init__(self, dim=48, n_heads=4, n_layers=2, n_experts=4, top_k=2,
                 patch_size=4, n_classes=10):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (28 // patch_size) ** 2  # 49 for 4x4
        patch_dim = patch_size * patch_size   # 16 for 4x4

        # Patch embedding
        self.patch_proj = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, dim) * 0.02)

        # ULB stack
        config = ULBConfig(d_model=dim, n_heads=n_heads, paired=True,
                           attn_mode='blend')
        self.ulb = MoEStackedULB(
            make_layer=lambda: ULBBlock(config),
            n_layers=n_layers, dim=dim,
            n_experts=n_experts, top_k=top_k)

        # Classifier head
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        # x: (B, 1, 28, 28)
        B = x.shape[0]
        p = self.patch_size

        # Extract patches: (B, 1, 28, 28) -> (B, n_patches, patch_dim)
        # Unfold H then W
        x = x.squeeze(1)  # (B, 28, 28)
        x = x.unfold(1, p, p).unfold(2, p, p)  # (B, 7, 7, 4, 4)
        x = x.contiguous().view(B, -1, p * p)   # (B, 49, 16)

        # Project + position embed
        x = self.patch_proj(x) + self.pos_embed

        # ULB
        x = self.ulb(x)  # (B, 49, dim)

        # Mean pool -> classify
        x = x.mean(dim=1)  # (B, dim)
        return self.head(x)


def load_data(data_dir=None, clean=False):
    if data_dir and not clean and os.path.exists(os.path.join(data_dir, 'X_train.npy')):
        print(f"  Loading augmented data from {data_dir}/")
        X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
        y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
        X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
        y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    else:
        print("  Loading clean Fashion-MNIST")
        train_ds = FashionMNIST('/tmp/fmnist', train=True, download=True)
        test_ds = FashionMNIST('/tmp/fmnist', train=False, download=True)
        X_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32)
        y_train = train_ds.targets.numpy()
        X_test = test_ds.data.numpy().reshape(-1, 784).astype(np.float32)
        y_test = test_ds.targets.numpy()

    X_train = (X_train / 255.0).reshape(-1, 1, 28, 28)
    X_test = (X_test / 255.0).reshape(-1, 1, 28, 28)
    return X_train, y_train, X_test, y_test


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        aux_loss = getattr(model.ulb, 'aux_loss', 0.0)
        loss = F.cross_entropy(out, y_batch) + aux_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (out.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out = model(X_batch)
        correct += (out.argmax(1) == y_batch).sum().item()
        total += len(y_batch)
    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='data/fmnist_augmented')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--dim', type=int, default=48)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--n_experts', type=int, default=4)
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\n--- Loading Data ---")
    X_train, y_train, X_test, y_test = load_data(args.data_dir, args.clean)
    tag = "clean" if args.clean else "augmented"
    print(f"  Train: {X_train.shape} ({tag}), Test: {X_test.shape}")

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = ULBClassifier(
        dim=args.dim, n_heads=args.n_heads, n_layers=args.n_layers,
        n_experts=args.n_experts, top_k=args.top_k).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n--- MoE ULB: dim={args.dim}, {args.n_layers}L, "
          f"{args.n_experts}E top-{args.top_k} ({n_params:,} params) ---")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n--- Training ({args.epochs} epochs) ---")
    best_acc = 0
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, test_loader, device)
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
        tqdm.write(f"  epoch {epoch+1:3d}  loss={train_loss:.4f}  "
                   f"train={train_acc:.2%}  val={val_acc:.2%}  best={best_acc:.2%}")

    print(f"\n{'=' * 50}")
    print(f"  Final val accuracy:  {val_acc:.2%}")
    print(f"  Best val accuracy:   {best_acc:.2%}")
    print(f"  Mode: {tag}")
    print(f"  Params: {n_params:,}")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
