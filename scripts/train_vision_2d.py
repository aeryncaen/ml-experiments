#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from heuristic_secrets.models.composed2d import (
    Model2d,
    ModelConfig2d,
    LayerConfig2d,
    HeadConfig2d,
)
from heuristic_secrets.models.train import get_device, set_seed, setup_cuda


CACHE_DIR = Path(".cache/vision_2d")


@dataclass
class DatasetConfig:
    name: str
    n_classes: int
    channels: int
    height: int
    width: int


DATASET_CONFIGS = {
    "mnist": DatasetConfig("mnist", 10, 1, 28, 28),
    "fashion-mnist": DatasetConfig("fashion-mnist", 10, 1, 28, 28),
    "cifar10": DatasetConfig("cifar10", 10, 3, 32, 32),
    "cifar100": DatasetConfig("cifar100", 100, 3, 32, 32),
}


def load_vision_dataset(name: str, data_dir: Path, train: bool):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
            if name in ["mnist", "fashion-mnist"]
            else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    if name == "mnist":
        return datasets.MNIST(data_dir, train=train, download=True, transform=transform)
    elif name == "fashion-mnist":
        return datasets.FashionMNIST(
            data_dir, train=train, download=True, transform=transform
        )
    elif name == "cifar10":
        return datasets.CIFAR10(
            data_dir, train=train, download=True, transform=transform
        )
    elif name == "cifar100":
        return datasets.CIFAR100(
            data_dir, train=train, download=True, transform=transform
        )
    raise ValueError(f"Unknown dataset: {name}")


def train_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train {epoch}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits, aux = model(images)
        loss = F.cross_entropy(logits, labels)

        aux_loss = sum(v.mean() for v in aux.values() if v.numel() > 0) * 0.01
        total_loss_batch = loss + aux_loss

        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=correct / total)

    return total_loss / total, correct / total


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Val"):
            images = images.to(device)
            labels = labels.to(device)

            logits, _ = model(images)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train 2D vision model")
    parser.add_argument(
        "--dataset", type=str, default="cifar10", choices=list(DATASET_CONFIGS.keys())
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--attn-window", type=int, default=16)
    parser.add_argument("--ssm-heads", type=int, default=4)
    parser.add_argument("--ssm-state", type=int, default=32)
    args = parser.parse_args()

    setup_cuda()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    ds_config = DATASET_CONFIGS[args.dataset]
    print(f"\nLoading {args.dataset}...")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    train_ds = load_vision_dataset(args.dataset, CACHE_DIR, train=True)
    val_ds = load_vision_dataset(args.dataset, CACHE_DIR, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"  Train: {len(train_ds):,} samples")
    print(f"  Val: {len(val_ds):,} samples")
    print(f"  Image size: {ds_config.channels}x{ds_config.height}x{ds_config.width}")

    layer_config = LayerConfig2d(
        embed_width=args.width,
        in_channels=ds_config.channels,
        dropout=0.1,
        conv_groups=max(1, args.width // 8),
        attn_heads=max(1, args.width // 8),
        attn_ffn_mult=2,
        attn_window_size=args.attn_window,
        attn_use_rope=True,
        num_attn_features=4,
        ssm_state_size=args.ssm_state,
        ssm_n_heads=args.ssm_heads,
        ssm_kernel_sizes=(3, 5, 7),
        ssm_expand=2,
        num_ssm_features=4,
        adaptive_conv=args.adaptive,
        n_adaptive_branches=3,
        adaptive_kernel_size=9,
        context_dim=16 if args.adaptive else 0,
        num_embed_features=4,
        mlp_hidden_mult=4,
        mlp_output_dim=16,
        num_hidden_features=4,
    )

    head_config = HeadConfig2d(
        head_type="classifier",
        n_classes=ds_config.n_classes,
    )

    model_config = ModelConfig2d(
        n_layers=args.n_layers,
        layer=layer_config,
        head=head_config,
    )

    model = Model2d(model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters")
    print(model_config)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            print(f"  New best: {best_acc:.4f}")

    print(f"\nBest validation accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
