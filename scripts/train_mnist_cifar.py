#!/usr/bin/env python3
import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from heuristic_secrets.models.composed2d import (
    Model2d,
    ModelConfig2d,
    LayerConfig2d,
    HeadConfig2d,
)
from heuristic_secrets.models import get_device, set_seed, setup_cuda


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


def get_transforms(ds_config: DatasetConfig, train: bool):
    if train:
        if ds_config.channels == 1:
            return transforms.Compose(
                [
                    transforms.RandomAffine(
                        degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )
    else:
        if ds_config.channels == 1:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            )


def load_vision_dataset(
    name: str, data_dir: Path, train: bool, ds_config: DatasetConfig
):
    transform = get_transforms(ds_config, train)
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


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, desc: str = "Eval"
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)

        logits, _ = model(images)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    aux_weight: float = 0.01,
    desc: str = "Training",
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=desc, leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        logits, aux = model(images)
        loss = F.cross_entropy(logits, labels)

        if aux:
            aux_loss = sum(v.mean() for v in aux.values() if v.numel() > 0)
            loss = loss + aux_weight * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct / total:.4f}")

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="mnist", choices=list(DATASET_CONFIGS.keys())
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--n-branches", type=int, default=3)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--swa", action="store_true", help="Use SWA")
    parser.add_argument(
        "--swa-start", type=float, default=0.75, help="SWA start epoch ratio"
    )
    parser.add_argument("--swa-lr", type=float, default=None, help="SWA learning rate")
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"models/{args.dataset}_2d_model.pt")

    ds_config = DATASET_CONFIGS[args.dataset]

    setup_cuda()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    attn_heads = max(1, min(4, args.width // 4))
    ssm_n_heads = max(1, min(4, args.width // 4))
    conv_groups = max(1, min(4, args.width // 4))

    layer_config = LayerConfig2d(
        embed_width=args.width,
        in_channels=ds_config.channels,
        dropout=0.1,
        conv_groups=conv_groups,
        attn_heads=attn_heads,
        attn_ffn_mult=4,
        attn_window_size=16,
        attn_use_rope=True,
        num_attn_features=4,
        ssm_state_size=32,
        ssm_n_heads=ssm_n_heads,
        ssm_kernel_sizes=(3, 5, 7),
        ssm_expand=2,
        num_ssm_features=4,
        adaptive_conv=args.adaptive,
        n_adaptive_branches=args.n_branches,
        adaptive_kernel_size=7,
        adaptive_init_sigmas=tuple(0.1 + 0.15 * i for i in range(args.n_branches))
        if args.adaptive
        else None,
        context_dim=16,
        num_embed_features=4,
        mlp_hidden_mult=4,
        mlp_output_dim=16,
        num_hidden_features=4,
    )
    head_config = HeadConfig2d(head_type="classifier", n_classes=ds_config.n_classes)
    model_config = ModelConfig2d(
        n_layers=args.n_layers, layer=layer_config, head=head_config
    )

    model: nn.Module = Model2d(model_config).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
        print("Using torch.compile")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel2d: {n_params:,} params")
    print(f"  Layers: {args.n_layers}, Width: {args.width}")
    print(
        f"  Adaptive: {args.adaptive}"
        + (f" ({args.n_branches} branches)" if args.adaptive else "")
    )

    print(f"\nLoading {args.dataset}...")
    train_dataset = load_vision_dataset(
        args.dataset, args.data_dir, train=True, ds_config=ds_config
    )
    val_dataset = load_vision_dataset(
        args.dataset, args.data_dir, train=False, ds_config=ds_config
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    print(f"  Train: {len(train_dataset):,} samples, {len(train_loader):,} batches")
    print(f"  Val: {len(val_dataset):,} samples, {len(val_loader):,} batches")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=args.lr * 0.01
    )

    swa_model = None
    swa_scheduler = None
    swa_start_epoch = int(args.epochs * args.swa_start) if args.swa else args.epochs + 1
    if args.swa:
        swa_model = AveragedModel(model)
        swa_lr = args.swa_lr if args.swa_lr else args.lr * 0.05
        swa_scheduler = SWALR(optimizer, swa_lr=swa_lr, anneal_epochs=5)

    start_epoch = 0
    best_acc = 0.0
    best_state = None

    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        print(f"  Resuming from epoch {start_epoch}, best_acc={best_acc:.4f}")

    print(f"\nTraining for {args.epochs} epochs...")
    if args.swa:
        print(f"  SWA starts at epoch {swa_start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        swa_active = args.swa and epoch >= swa_start_epoch
        desc = f"Epoch {epoch + 1}/{args.epochs}" + (" [SWA]" if swa_active else "")

        if swa_active and swa_model is not None and swa_scheduler is not None:
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, None, device, desc=desc
            )
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, device, desc=desc
            )

        val_loss, val_acc = evaluate(model, val_loader, device)

        best_str = ""
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_str = " *BEST*"

        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch + 1:3d}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"lr={lr:.2e}{best_str}"
        )

        if (epoch + 1) % 10 == 0 or epoch == args.epochs - 1:
            ckpt_path = (
                args.output.parent
                / f"{args.output.stem}_epoch{epoch + 1:03d}{args.output.suffix}"
            )
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_config": model_config.to_dict(),
                    "state_dict": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc,
                },
                ckpt_path,
            )
            print(f"  -> {ckpt_path}")

    if args.swa and swa_model is not None:
        print("\nFinalizing SWA...")
        update_bn(train_loader, swa_model, device=device)
        model.load_state_dict(swa_model.module.state_dict())
        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"SWA model: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"\nBest model: val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": model_config.to_dict(),
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
        },
        args.output,
    )
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
