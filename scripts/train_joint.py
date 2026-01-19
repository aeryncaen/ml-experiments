#!/usr/bin/env python3

import argparse
import csv
import json
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from heuristic_secrets.joint import JointConfig, JointSecretDetector, JointTrainer
from heuristic_secrets.joint.train import get_device
from heuristic_secrets.bytemasker.model import ByteMaskerConfig
from heuristic_secrets.bytemasker.dataset import load_bytemasker_dataset
from heuristic_secrets.validator.model import ValidatorConfig
from heuristic_secrets.validator.features import FrequencyTable
from heuristic_secrets.data.types import ValidatorSample


def load_validator_samples(path: Path) -> list[ValidatorSample]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(ValidatorSample.from_dict(json.loads(line)))
    return samples


def create_batches(
    byte_tensors: list[torch.Tensor],
    labels: list[float],
    lengths: list[int],
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if seed is not None:
        random.seed(seed)

    indices = list(range(len(byte_tensors)))
    if shuffle:
        random.shuffle(indices)

    batches = []
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]

        batch_bytes = [byte_tensors[j] for j in batch_indices]
        batch_labels = torch.tensor([labels[j] for j in batch_indices])
        batch_lengths = torch.tensor([lengths[j] for j in batch_indices])

        padded_bytes = pad_sequence(batch_bytes, batch_first=True, padding_value=0)
        batches.append((padded_bytes, batch_labels, batch_lengths))

    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--masker-depth", type=int, default=3)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--n-clean", type=int, default=10000)
    parser.add_argument("--output", type=Path, default=Path("models/joint.pt"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--masker-checkpoint", type=Path, default=None)
    parser.add_argument("--validator-checkpoint", type=Path, default=None)
    parser.add_argument("--eval-only", action="store_true", help="Evaluate without training")
    args = parser.parse_args()

    print("Loading validator samples (secrets + false positives)...")
    train_val_samples = load_validator_samples(args.data_dir / "splits" / "validator" / "train.jsonl")
    val_val_samples = load_validator_samples(args.data_dir / "splits" / "validator" / "val.jsonl")

    train_secrets = [s for s in train_val_samples if s.label == 1]
    train_fp = [s for s in train_val_samples if s.label == 0]
    val_secrets = [s for s in val_val_samples if s.label == 1]
    val_fp = [s for s in val_val_samples if s.label == 0]

    print(f"Train: {len(train_secrets)} secrets, {len(train_fp)} false positives")
    print(f"Val: {len(val_secrets)} secrets, {len(val_fp)} false positives")

    print("\nLoading document data for clean lines...")
    train_docs, train_stats = load_bytemasker_dataset(args.data_dir, "train", max_len=args.max_len)
    val_docs, val_stats = load_bytemasker_dataset(args.data_dir, "val", max_len=args.max_len)

    n_train_clean = min(args.n_clean, len(train_docs.clean_lines))
    n_val_clean = min(args.n_clean // 5, len(val_docs.clean_lines))

    train_clean_indices = random.sample(train_docs.clean_lines, n_train_clean)
    val_clean_indices = random.sample(val_docs.clean_lines, n_val_clean)

    print(f"Using {n_train_clean} clean lines for training, {n_val_clean} for validation")

    print("\nPreparing training data...")
    train_bytes = []
    train_labels = []
    train_lengths = []

    for s in tqdm(train_secrets, desc="Secrets"):
        b = s.text.encode("utf-8")[:args.max_len]
        train_bytes.append(torch.tensor(list(b), dtype=torch.long))
        train_labels.append(1.0)
        train_lengths.append(len(b))

    for s in tqdm(train_fp, desc="False positives"):
        b = s.text.encode("utf-8")[:args.max_len]
        train_bytes.append(torch.tensor(list(b), dtype=torch.long))
        train_labels.append(0.0)
        train_lengths.append(len(b))

    for idx in tqdm(train_clean_indices, desc="Clean lines"):
        train_bytes.append(train_docs.byte_tensors[idx])
        train_labels.append(0.0)
        train_lengths.append(train_docs.lengths[idx])

    print("\nPreparing validation data...")
    val_bytes = []
    val_labels = []
    val_lengths = []

    for s in val_secrets:
        b = s.text.encode("utf-8")[:args.max_len]
        val_bytes.append(torch.tensor(list(b), dtype=torch.long))
        val_labels.append(1.0)
        val_lengths.append(len(b))

    for s in val_fp:
        b = s.text.encode("utf-8")[:args.max_len]
        val_bytes.append(torch.tensor(list(b), dtype=torch.long))
        val_labels.append(0.0)
        val_lengths.append(len(b))

    for idx in val_clean_indices:
        val_bytes.append(val_docs.byte_tensors[idx])
        val_labels.append(0.0)
        val_lengths.append(val_docs.lengths[idx])

    print(f"\nTotal train: {len(train_bytes)} ({sum(train_labels):.0f} secrets)")
    print(f"Total val: {len(val_bytes)} ({sum(val_labels):.0f} secrets)")

    secret_bytes = b"".join(s.text.encode("utf-8") for s in train_secrets)
    text_bytes = b"".join(train_docs.lines[i].bytes for i in train_clean_indices[:1000])
    text_freq = FrequencyTable.from_data(text_bytes)
    secret_freq = FrequencyTable.from_data(secret_bytes)

    print("\nCreating batches...")
    train_batches = create_batches(train_bytes, train_labels, train_lengths, args.batch_size, seed=42)
    val_batches = create_batches(val_bytes, val_labels, val_lengths, args.batch_size, shuffle=False)
    print(f"Train batches: {len(train_batches)}, Val batches: {len(val_batches)}")

    config = JointConfig(
        masker=ByteMaskerConfig(width=args.width, depth=args.masker_depth),
        validator=ValidatorConfig(width=args.width),
    )

    device = torch.device("cpu") if args.cpu else get_device()

    if args.masker_checkpoint or args.validator_checkpoint:
        print("\nLoading pretrained weights...")
        model = JointSecretDetector.from_pretrained(
            config,
            masker_path=str(args.masker_checkpoint) if args.masker_checkpoint else None,
            validator_path=str(args.validator_checkpoint) if args.validator_checkpoint else None,
            text_freq=text_freq,
            secret_freq=secret_freq,
            device=device,
        )
    else:
        model = JointSecretDetector(config, text_freq=text_freq, secret_freq=secret_freq)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: width={args.width}, masker_depth={args.masker_depth}")
    print(f"Parameters: {param_count:,}")
    print(f"Device: {device}")

    trainer = JointTrainer(
        model=model,
        train_data=train_batches,
        val_data=val_batches,
        lr=args.lr,
        device=device,
        epochs=args.epochs,
        seed=args.seed,
    )

    if args.eval_only:
        print(f"\n{'='*60}")
        print("Eval-only mode")
        print(f"{'='*60}\n")

        val_metrics = trainer.validate(optimize_threshold=True)
        print(f"Val: {val_metrics}")
        print(f"Optimal threshold: {trainer.optimal_threshold:.3f}")

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "config": model.config.to_dict(),
                "state_dict": model.state_dict(),
                "best_f1": val_metrics.f1,
                "best_epoch": "eval_only",
                "optimal_threshold": trainer.optimal_threshold,
                "text_freq": text_freq.to_dict(),
                "secret_freq": secret_freq.to_dict(),
            }, args.output)
            print(f"\nSaved model to {args.output}")
        return

    print(f"\n{'='*60}")
    print(f"Training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    epoch_metrics = []
    best_f1 = 0.0
    best_epoch = 0
    checkpoint_dir = args.output.parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    saved_checkpoints: list[tuple[Path, int | str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = trainer.train_epoch(seed=epoch, epoch=epoch - 1)
        val_metrics = trainer.validate(optimize_threshold=(epoch == args.epochs))

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
            "train_p": train_metrics.precision,
            "train_r": train_metrics.recall,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.accuracy,
            "val_f1": val_metrics.f1,
            "val_p": val_metrics.precision,
            "val_r": val_metrics.recall,
        })

    print("\nFinalizing SWA weights...")
    trainer.finalize_swa()

    print("Evaluating SWA model...")
    val_metrics = trainer.validate(optimize_threshold=True)

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
    }, ckpt_path)
    saved_checkpoints.append((ckpt_path, "swa", val_metrics.f1))

    best_tag = " *" if is_best else ""
    print(f"SWA Val: {val_metrics}{best_tag}")
    print(f"\nBest F1: {best_f1:.4f} (epoch {best_epoch})")
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
        "optimal_threshold": trainer.optimal_threshold,
        "text_freq": text_freq.to_dict(),
        "secret_freq": secret_freq.to_dict(),
    }, args.output)

    print(f"\nSaved best model (epoch {best_epoch}, F1={best_f1:.4f}) to {args.output}")


if __name__ == "__main__":
    main()
