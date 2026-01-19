#!/usr/bin/env python3

import argparse
import csv
import random
from pathlib import Path

import torch

from heuristic_secrets.bytemasker.model import ByteMaskerConfig, ByteMaskerModel
from heuristic_secrets.bytemasker.dataset import load_bytemasker_dataset, compute_stats
from heuristic_secrets.bytemasker.train import ByteMaskerTrainer, get_device


def main():
    parser = argparse.ArgumentParser(description="Train ByteMasker model")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--size", choices=["tiny", "small", "medium"], default="small")
    parser.add_argument("--output", type=Path, default=Path("models/bytemasker.pt"))
    parser.add_argument("--clean-ratio", type=float, default=1.0, help="Ratio of clean lines to secret lines per epoch")
    parser.add_argument("--pos-weight", type=float, default=None, help="Override auto-computed pos_weight")
    parser.add_argument("--pos-weight-mult", type=float, default=1.0, help="Multiplier for auto-computed pos_weight")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-output", type=Path, default=None)
    args = parser.parse_args()

    print("Loading datasets...")
    train_dataset, train_stats = load_bytemasker_dataset(args.data_dir, "train", secrets_only=False)
    val_dataset, val_stats = load_bytemasker_dataset(args.data_dir, "val", secrets_only=False)

    print(f"\nTraining set:\n{train_stats}")
    print(f"\nValidation set:\n{val_stats}")

    n_secret_lines = len(train_dataset.secret_lines)
    n_clean_available = len(train_dataset.clean_lines)
    n_clean_per_epoch = min(int(n_secret_lines * args.clean_ratio), n_clean_available)

    secret_pos_bytes = sum(
        train_dataset.mask_tensors[i].sum().item() 
        for i in train_dataset.secret_lines
    )
    secret_total_bytes = sum(train_dataset.lengths[i] for i in train_dataset.secret_lines)
    secret_neg_bytes = secret_total_bytes - secret_pos_bytes

    if n_clean_available > 0:
        avg_clean_len = sum(train_dataset.lengths[i] for i in train_dataset.clean_lines) / n_clean_available
    else:
        avg_clean_len = 0
    clean_bytes_per_epoch = n_clean_per_epoch * avg_clean_len

    total_neg_per_epoch = secret_neg_bytes + clean_bytes_per_epoch
    base_pos_weight = total_neg_per_epoch / secret_pos_bytes if secret_pos_bytes > 0 else 1.0
    pos_weight = base_pos_weight * args.pos_weight_mult

    print(f"\nTraining config:")
    print(f"  Secret lines: {n_secret_lines:,} (all used every epoch)")
    print(f"  Clean lines available: {n_clean_available:,}")
    print(f"  Clean lines per epoch: {n_clean_per_epoch:,} ({args.clean_ratio:.1f}x secrets, randomly sampled)")
    print(f"\nByte-level class balance per epoch:")
    print(f"  Positive bytes (secrets): {int(secret_pos_bytes):,}")
    print(f"  Negative bytes (from secret lines): {int(secret_neg_bytes):,}")
    print(f"  Negative bytes (from clean lines): {int(clean_bytes_per_epoch):,}")
    print(f"  Total negative: {int(total_neg_per_epoch):,}")
    print(f"  Ratio: {total_neg_per_epoch / secret_pos_bytes:.1f}:1 (neg:pos)")
    print(f"  pos_weight: {pos_weight:.2f} (base {base_pos_weight:.2f} x {args.pos_weight_mult:.1f})")

    if args.size == "tiny":
        config = ByteMaskerConfig.tiny()
    elif args.size == "small":
        config = ByteMaskerConfig.small()
    else:
        config = ByteMaskerConfig.medium()

    model = ByteMaskerModel(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel config: {config}")
    print(f"Parameters: {param_count:,}")

    device = torch.device("cpu") if args.cpu else get_device()
    print(f"Device: {device}")

    final_pos_weight = args.pos_weight if args.pos_weight is not None else pos_weight

    trainer = ByteMaskerTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        lr=args.lr,
        batch_size=args.batch_size,
        train_on_secret_lines_only=True,
        pos_weight=final_pos_weight,
        device=device,
        epochs=args.epochs,
        val_clean_ratio=args.clean_ratio,
    )
    if args.pos_weight is not None:
        print(f"Using CLI pos_weight override: {final_pos_weight:.2f}")
    print(f"SWA: starts at epoch {trainer.swa_start_epoch + 1}")

    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs (optimizing for RECALL)")
    print(f"{'='*60}\n")

    epoch_metrics = []
    best_recall = 0.0
    best_epoch = 0
    checkpoint_dir = args.output.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved_checkpoints = []

    for epoch in range(1, args.epochs + 1):
        random.seed(args.seed + epoch)
        sampled_clean = random.sample(train_dataset.clean_lines, n_clean_per_epoch)
        epoch_indices = train_dataset.secret_lines + sampled_clean
        trainer.rebuild_train_batches(epoch_indices, seed=args.seed + epoch)

        train_metrics = trainer.train_epoch(seed=epoch, epoch=epoch - 1)
        val_metrics = trainer.validate(optimize_threshold=(epoch == args.epochs))
        fp_rate = trainer.validate_fp_rate()

        is_best = val_metrics.recall > best_recall
        if is_best:
            best_recall = val_metrics.recall
            best_epoch = epoch

        ckpt_path = checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save({
            "epoch": epoch,
            "config": model.config.to_dict(),
            "state_dict": model.state_dict(),
            "val_recall": val_metrics.recall,
            "val_f1": val_metrics.f1,
            "fp_rate": fp_rate,
            "optimal_threshold": trainer.optimal_threshold,
        }, ckpt_path)
        saved_checkpoints.append((ckpt_path, epoch, val_metrics.recall))

        swa_tag = " [SWA]" if trainer.swa_active else ""
        best_tag = " *" if is_best else ""
        print(
            f"Epoch {epoch:2d}{swa_tag} | "
            f"Train: {train_metrics} | "
            f"Val: {val_metrics} | "
            f"Clean FP: {fp_rate:.2%}{best_tag}"
        )

        epoch_metrics.append({
            "epoch": epoch,
            "swa": trainer.swa_active,
            "train_loss": train_metrics.loss,
            "train_precision": train_metrics.precision,
            "train_recall": train_metrics.recall,
            "train_f1": train_metrics.f1,
            "train_fn": train_metrics.fn_rate,
            "val_loss": val_metrics.loss,
            "val_precision": val_metrics.precision,
            "val_recall": val_metrics.recall,
            "val_f1": val_metrics.f1,
            "val_fn": val_metrics.fn_rate,
            "clean_line_fp": fp_rate,
        })

    print("\nFinalizing SWA weights...")
    trainer.finalize_swa()

    print("Evaluating SWA model...")
    val_metrics = trainer.validate(optimize_threshold=True)
    fp_rate = trainer.validate_fp_rate()

    is_best = val_metrics.recall > best_recall
    if is_best:
        best_recall = val_metrics.recall
        best_epoch = "swa"

    ckpt_path = checkpoint_dir / "swa_final.pt"
    torch.save({
        "epoch": "swa_final",
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "val_recall": val_metrics.recall,
        "val_f1": val_metrics.f1,
        "fp_rate": fp_rate,
        "optimal_threshold": trainer.optimal_threshold,
    }, ckpt_path)
    saved_checkpoints.append((ckpt_path, "swa", val_metrics.recall))

    best_tag = " *" if is_best else ""
    print(f"SWA Val: {val_metrics} | Clean FP: {fp_rate:.2%}{best_tag}")

    epoch_metrics.append({
        "epoch": "swa_final",
        "swa": True,
        "train_loss": epoch_metrics[-1]["train_loss"],
        "train_precision": epoch_metrics[-1]["train_precision"],
        "train_recall": epoch_metrics[-1]["train_recall"],
        "train_f1": epoch_metrics[-1]["train_f1"],
        "train_fn": epoch_metrics[-1]["train_fn"],
        "val_loss": val_metrics.loss,
        "val_precision": val_metrics.precision,
        "val_recall": val_metrics.recall,
        "val_f1": val_metrics.f1,
        "val_fn": val_metrics.fn_rate,
        "clean_line_fp": fp_rate,
    })

    print(f"\nBest Recall: {best_recall:.4f} (epoch {best_epoch})")
    print(f"Optimal threshold: {trainer.optimal_threshold:.3f}")

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
            fieldnames = [
                "epoch", "swa", "train_loss", "train_precision", "train_recall", "train_f1",
                "train_fn", "val_loss", "val_precision", "val_recall", "val_f1", "val_fn",
                "clean_line_fp"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in epoch_metrics:
                writer.writerow(row)
        print(f"\nMetrics saved to {args.metrics_output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": model.config.to_dict(),
        "state_dict": model.state_dict(),
        "best_recall": best_recall,
        "best_epoch": best_epoch,
        "optimal_threshold": trainer.optimal_threshold,
    }, args.output)

    final_fp_rate = trainer.validate_fp_rate()
    print(f"Final FP rate on clean lines: {final_fp_rate:.2%}")
    print(f"\nSaved best model (epoch {best_epoch}, recall={best_recall:.4f}) to {args.output}")


if __name__ == "__main__":
    main()
