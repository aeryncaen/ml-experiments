#!/usr/bin/env python3

import argparse
import csv
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from heuristic_secrets.joint.pruned_model import PrunedDetectorConfig, PrunedSecretDetector
from heuristic_secrets.joint.pruned_train import PrunedTrainer, get_device
from heuristic_secrets.bytemasker.dataset import load_bytemasker_dataset


def create_classification_batches(
    dataset,
    batch_size: int,
    balance: bool = True,
    seed: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if seed is not None:
        random.seed(seed)
    
    if balance:
        n_secret = len(dataset.secret_lines)
        n_clean = len(dataset.clean_lines)
        n_each = min(n_secret, n_clean)
        
        secret_indices = random.sample(dataset.secret_lines, n_each)
        clean_indices = random.sample(dataset.clean_lines, n_each)
        indices = secret_indices + clean_indices
    else:
        indices = list(range(len(dataset)))
    
    random.shuffle(indices)
    
    batches = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        
        batch_bytes = [dataset.byte_tensors[j] for j in batch_indices]
        batch_labels = torch.tensor(
            [1.0 if dataset.lines[j].has_secret else 0.0 for j in batch_indices]
        )
        batch_lengths = torch.tensor([dataset.lengths[j] for j in batch_indices])
        
        padded_bytes = pad_sequence(batch_bytes, batch_first=True, padding_value=0)
        
        batches.append((padded_bytes, batch_labels, batch_lengths))
    
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--conv-depth", type=int, default=3)
    parser.add_argument("--attn-depth", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("models/pruned.pt"))
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--no-balance", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--bytemasker", type=Path, default=None, help="Init conv layers from pretrained ByteMasker")
    parser.add_argument("--validator", type=Path, default=None, help="Init attention layers from pretrained ByteValidator")
    args = parser.parse_args()

    print("Loading datasets...")
    train_dataset, train_stats = load_bytemasker_dataset(args.data_dir, "train", secrets_only=False)
    val_dataset, val_stats = load_bytemasker_dataset(args.data_dir, "val", secrets_only=False)

    print(f"\nTraining set:\n{train_stats}")
    print(f"\nValidation set:\n{val_stats}")

    print("\nCreating balanced batches...")
    train_batches = create_classification_batches(
        train_dataset, args.batch_size, balance=not args.no_balance, seed=42
    )
    val_batches = create_classification_batches(
        val_dataset, args.batch_size, balance=not args.no_balance, seed=43
    )
    print(f"Train batches: {len(train_batches)}")
    print(f"Val batches: {len(val_batches)}")

    config = PrunedDetectorConfig(
        width=args.width,
        conv_depth=args.conv_depth,
        attn_depth=args.attn_depth,
    )
    
    device = torch.device("cpu") if args.cpu else get_device()
    print(f"Device: {device}")

    if args.bytemasker or args.validator:
        print("\nLoading pretrained weights...")
        if args.bytemasker:
            print(f"  ByteMasker: {args.bytemasker}")
        if args.validator:
            print(f"  ByteValidator: {args.validator}")
        model = PrunedSecretDetector.from_pretrained(
            config,
            bytemasker_path=str(args.bytemasker) if args.bytemasker else None,
            validator_path=str(args.validator) if args.validator else None,
            device=device,
        )
    else:
        model = PrunedSecretDetector(config).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: width={config.width}, conv_depth={config.conv_depth}, attn_depth={config.attn_depth}")
    print(f"Parameters: {param_count:,}")

    trainer = PrunedTrainer(
        model=model,
        train_data=train_batches,
        val_data=val_batches,
        lr=args.lr,
        device=device,
        epochs=args.epochs,
        pos_weight=args.pos_weight,
    )

    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    epoch_metrics = []
    best_f1 = 0.0
    best_epoch = 0
    checkpoint_dir = args.output.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved_checkpoints = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(seed=epoch, epoch=epoch - 1)
        val_metrics = trainer.validate()

        is_best = val_metrics.f1 > best_f1
        if is_best:
            best_f1 = val_metrics.f1
            best_epoch = epoch

        ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "val_f1": val_metrics.f1,
            "val_accuracy": val_metrics.accuracy,
        }, ckpt_path)
        saved_checkpoints.append((ckpt_path, epoch, val_metrics.f1))

        swa_tag = " [SWA]" if trainer.swa_active else ""
        best_tag = " *" if is_best else ""
        print(
            f"Epoch {epoch:2d}{swa_tag} | "
            f"Train: {train_metrics} | "
            f"Val: {val_metrics}{best_tag}"
        )

        epoch_metrics.append({
            "epoch": epoch,
            "swa": trainer.swa_active,
            "train_loss": train_metrics.loss,
            "train_acc": train_metrics.accuracy,
            "train_f1": train_metrics.f1,
            "train_pruned_len": train_metrics.avg_pruned_len,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.accuracy,
            "val_f1": val_metrics.f1,
            "val_pruned_len": val_metrics.avg_pruned_len,
        })

    print("\nFinalizing SWA weights...")
    trainer.finalize_swa()

    print("Evaluating SWA model...")
    val_metrics = trainer.validate()

    is_best = val_metrics.f1 > best_f1
    if is_best:
        best_f1 = val_metrics.f1
        best_epoch = "swa"

    ckpt_path = checkpoint_dir / "swa_final.pt"
    torch.save({
        "epoch": "swa_final",
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "val_f1": val_metrics.f1,
        "val_accuracy": val_metrics.accuracy,
    }, ckpt_path)
    saved_checkpoints.append((ckpt_path, "swa", val_metrics.f1))

    best_tag = " *" if is_best else ""
    print(f"SWA Val: {val_metrics}{best_tag}")

    print(f"\nBest F1: {best_f1:.4f} (epoch {best_epoch})")

    best_ckpt = max(saved_checkpoints, key=lambda x: x[2])
    best_ckpt_path = best_ckpt[0]
    best_state = torch.load(best_ckpt_path, map_location=device, weights_only=False)["state_dict"]
    model.load_state_dict(best_state)

    for ckpt_path, _, _ in saved_checkpoints:
        if ckpt_path != best_ckpt_path:
            ckpt_path.unlink()
    print(f"Cleaned up checkpoints, kept: {best_ckpt_path.name}")

    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_output, "w", newline="") as f:
            fieldnames = list(epoch_metrics[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in epoch_metrics:
                writer.writerow(row)
        print(f"\nMetrics saved to {args.metrics_output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "best_f1": best_f1,
        "best_epoch": best_epoch,
    }, args.output)

    print(f"\nSaved best model (epoch {best_epoch}, F1={best_f1:.4f}) to {args.output}")


if __name__ == "__main__":
    main()
