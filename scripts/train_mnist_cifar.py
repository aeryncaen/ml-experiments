#!/usr/bin/env python3
import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torchvision import datasets, transforms

from heuristic_secrets.models import (
    Model,
    ModelConfig,
    LayerConfig,
    HeadConfig,
    Trainer,
    TrainConfig,
    create_binary_batches,
    get_device,
    set_seed,
    setup_cuda,
)


CACHE_DIR = Path(".cache/mnist_cifar")


@dataclass
class DatasetConfig:
    name: str
    n_classes: int
    seq_len: int
    channels: int
    height: int
    width: int


DATASET_CONFIGS = {
    "mnist": DatasetConfig("mnist", 10, 28 * 28, 1, 28, 28),
    "fashion-mnist": DatasetConfig("fashion-mnist", 10, 28 * 28, 1, 28, 28),
    "cifar10": DatasetConfig("cifar10", 10, 32 * 32 * 3, 3, 32, 32),
    "cifar100": DatasetConfig("cifar100", 100, 32 * 32 * 3, 3, 32, 32),
}


def load_vision_dataset(name: str, data_dir: Path, train: bool):
    transform = transforms.ToTensor()
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


def convert_to_bytes_dataset(dataset, ds_config: DatasetConfig):
    byte_tensors = []
    labels = []
    lengths = []

    for img, label in dataset:
        flat = (img * 255).to(torch.long).view(-1)
        byte_tensors.append(flat)
        labels.append(float(label))
        lengths.append(len(flat))

    return {
        "bytes": byte_tensors,
        "labels": labels,
        "lengths": lengths,
    }


def eval_swa_model(trainer: Trainer, val_batches: list, device: torch.device):
    from torch.optim.swa_utils import update_bn

    if not trainer.swa_active or trainer.swa_model is None:
        return None, None

    original_state = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}

    def bn_loader():
        for batch in trainer.train_data:
            yield batch.bytes.to(device)

    update_bn(bn_loader(), trainer.swa_model, device=device)
    trainer.model.load_state_dict(trainer.swa_model.module.state_dict())

    val_metrics = trainer.validate(optimize_threshold=False)

    trainer.model.load_state_dict(original_state)

    return val_metrics.accuracy, val_metrics


def cosine_schedule(
    epoch: int, total_epochs: int, start_val: float, end_val: float
) -> float:
    if total_epochs <= 1:
        return end_val
    progress = epoch / (total_epochs - 1)
    return end_val + 0.5 * (start_val - end_val) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=list(DATASET_CONFIGS.keys()),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-swa", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--ssm-kernels", type=str, default="7")
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--n-branches", type=int, default=3)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument(
        "--device", type=str, default=None, help="Device: cuda, mps, cpu"
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = Path(f"models/{args.dataset}_model.pt")

    ds_config = DATASET_CONFIGS[args.dataset]

    setup_cuda()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    ssm_kernel_sizes = tuple(int(k) for k in args.ssm_kernels.split(","))
    layer_config = LayerConfig(
        embed_width=8,
        conv_groups=2,
        attn_heads=1,
        attn_ffn_mult=2,
        num_attn_features=4,
        ssm_state_size=16,
        ssm_n_heads=2,
        ssm_expand=2,
        ssm_kernel_sizes=ssm_kernel_sizes,
        num_ssm_features=4,
        num_embed_features=4,
        num_hidden_features=4,
        num_heuristic_features=0,
        mlp_hidden_mult=4,
        mlp_output_dim=16,
        dropout=0.1,
        adaptive_conv=args.adaptive,
        n_adaptive_branches=args.n_branches,
    )
    head_config = HeadConfig(head_type="classifier", n_classes=ds_config.n_classes)
    model_config = ModelConfig(
        n_layers=args.n_layers, layer=layer_config, head=head_config
    )

    if args.adaptive:
        print(
            f"\nUsing unified architecture with ADAPTIVE conv ({args.n_branches} branches)"
        )
    else:
        print(f"\nUsing unified architecture, ssm_kernels={ssm_kernel_sizes}")

    print(f"\nLoading {args.dataset}...")
    train_dataset = load_vision_dataset(args.dataset, args.data_dir, train=True)
    val_dataset = load_vision_dataset(args.dataset, args.data_dir, train=False)

    print(f"  Converting to byte sequences...")
    train_data = convert_to_bytes_dataset(train_dataset, ds_config)
    val_data = convert_to_bytes_dataset(val_dataset, ds_config)

    print(f"  Train: {len(train_data['labels']):,} samples")
    print(f"  Val: {len(val_data['labels']):,} samples")
    print(f"  Classes: {ds_config.n_classes}, Seq len: {ds_config.seq_len}")

    train_batches = create_binary_batches(
        train_data["bytes"],
        train_data["labels"],
        train_data["lengths"],
        batch_size=args.batch_size,
        feature_tensors=None,
        shuffle=True,
        seed=args.seed,
    )
    val_batches = create_binary_batches(
        val_data["bytes"],
        val_data["labels"],
        val_data["lengths"],
        batch_size=args.eval_batch_size,
        feature_tensors=None,
        shuffle=False,
    )
    print(f"  Train batches: {len(train_batches):,}")
    print(f"  Val batches: {len(val_batches):,}")

    model = Model(model_config)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(model_config)

    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        pos_weight=1.0,
        use_swa=not args.no_swa,
        preload_to_device=False,
    )

    trainer = Trainer(
        model, train_batches, val_batches, config=train_config, device=device
    )

    start_epoch = 0
    best_acc = 0.0
    best_epoch = 0
    best_state = None

    if args.resume:
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        trainer.model.load_state_dict(ckpt["state_dict"])
        if "optimizer_state" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state"])
        if "scheduler_state" in ckpt:
            trainer.scheduler.load_state_dict(ckpt["scheduler_state"])
            trainer._scheduler_steps = ckpt.get("scheduler_steps", 0)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        best_state = ckpt.get("best_state")
        if ckpt.get("swa_active", False) and trainer.swa_model is not None:
            trainer.swa_active = True
            if "swa_state" in ckpt:
                trainer.swa_model.load_state_dict(ckpt["swa_state"])
        print(f"  Resuming from epoch {start_epoch}, best_acc={best_acc:.4f}")

    print(f"\nTraining for epochs {start_epoch + 1}-{args.epochs}...")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate(optimize_threshold=False)

        best_str = ""
        if val_metrics.accuracy > best_acc:
            best_acc = val_metrics.accuracy
            best_epoch = epoch + 1
            best_state = {
                k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
            }
            best_str = " *BEST*"

        print(
            f"Epoch {epoch + 1:2d}: train {train_metrics} | val {val_metrics}{best_str}"
        )

        epoch_path = (
            args.output.parent
            / f"{args.output.stem}_epoch{epoch + 1:02d}{args.output.suffix}"
        )
        epoch_path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(
            epoch_path,
            extra={
                "epoch": epoch,
                "optimizer_state": trainer.optimizer.state_dict(),
                "scheduler_state": trainer.scheduler.state_dict(),
                "scheduler_steps": trainer._scheduler_steps,
                "best_acc": best_acc,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "swa_active": trainer.swa_active,
                "swa_state": trainer.swa_model.state_dict()
                if trainer.swa_model
                else None,
            },
        )
        print(f"  -> {epoch_path}")

        swa_acc, swa_metrics = eval_swa_model(trainer, val_batches, device)
        if swa_acc is not None and trainer.swa_model is not None:
            print(f"         SWA: {swa_metrics} acc={swa_acc:.4f}")

            swa_path = (
                args.output.parent
                / f"{args.output.stem}_swa_epoch{epoch + 1:02d}{args.output.suffix}"
            )
            torch.save(
                {
                    "model_config": trainer.model.config.to_dict(),
                    "state_dict": trainer.swa_model.module.state_dict(),
                    "epoch": epoch,
                    "swa_acc": swa_acc,
                },
                swa_path,
            )
            print(f"         -> {swa_path}")

    if not args.no_swa:
        trainer.finalize_swa()
        final = trainer.validate(optimize_threshold=False)
        print(f"\nPost-SWA: {final}")
        swa_acc = final.accuracy

        if swa_acc > best_acc:
            best_acc = swa_acc
            best_epoch = -1
            best_state = {
                k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
            }
            print("  SWA model is new best!")

    if best_state is not None:
        trainer.model.load_state_dict(best_state)

    final = trainer.validate(optimize_threshold=False)
    best_src = f"epoch {best_epoch}" if best_epoch > 0 else "SWA"
    print(f"\nBest model ({best_src}): {final}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
