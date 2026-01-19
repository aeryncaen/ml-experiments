#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from heuristic_secrets.validator.model import ValidatorConfig, ValidatorModel
from heuristic_secrets.validator.train import ValidatorTrainer, get_device
from heuristic_secrets.validator.features import (
    shannon_entropy,
    kolmogorov_complexity,
    char_frequency_difference,
    char_type_mix,
    FrequencyTable,
)
from heuristic_secrets.data.types import ValidatorSample


CACHE_DIR = Path(".cache/validator")


def load_validator_samples(path: Path) -> list[ValidatorSample]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(ValidatorSample.from_dict(json.loads(line)))
    return samples


def build_frequency_tables(
    samples: list[ValidatorSample],
) -> tuple[FrequencyTable, FrequencyTable, FrequencyTable]:
    all_bytes = b"".join(s.text.encode("utf-8") for s in samples)
    innocuous_bytes = b"".join(s.text.encode("utf-8") for s in samples if s.label == 0)
    secret_bytes = b"".join(s.text.encode("utf-8") for s in samples if s.label == 1)
    return (
        FrequencyTable.from_data(all_bytes),
        FrequencyTable.from_data(innocuous_bytes),
        FrequencyTable.from_data(secret_bytes),
    )


def _compute_freq_hash(text_freq: FrequencyTable, innocuous_freq: FrequencyTable, secret_freq: FrequencyTable) -> str:
    freq_str = json.dumps([text_freq.to_dict(), innocuous_freq.to_dict(), secret_freq.to_dict()], sort_keys=True)
    return hashlib.md5(freq_str.encode()).hexdigest()[:8]


def _get_cache_path(data_path: Path, freq_hash: str, max_len: int) -> Path:
    name = data_path.stem
    return CACHE_DIR / f"features_{name}_max{max_len}_{freq_hash}.pt"


def _extract_features(
    samples: list[ValidatorSample],
    text_freq: FrequencyTable,
    innocuous_freq: FrequencyTable,
    secret_freq: FrequencyTable,
    max_len: int = 512,
    show_progress: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float], list[int]]:
    byte_tensors = []
    feature_tensors = []
    labels = []
    lengths = []
    
    for sample in tqdm(samples, desc="Extracting features", disable=not show_progress):
        text_bytes = sample.text.encode("utf-8")[:max_len]
        
        features = [
            shannon_entropy(text_bytes),
            kolmogorov_complexity(text_bytes),
            char_frequency_difference(text_bytes, text_freq),
            char_frequency_difference(text_bytes, innocuous_freq),
            char_frequency_difference(text_bytes, secret_freq),
            char_type_mix(text_bytes),
        ]
        
        byte_tensors.append(torch.tensor(list(text_bytes), dtype=torch.long))
        feature_tensors.append(torch.tensor(features, dtype=torch.float32))
        labels.append(float(sample.label))
        lengths.append(len(text_bytes))
    
    return byte_tensors, feature_tensors, labels, lengths


def samples_to_tensors(
    samples: list[ValidatorSample],
    text_freq: FrequencyTable,
    innocuous_freq: FrequencyTable,
    secret_freq: FrequencyTable,
    max_len: int = 512,
    show_progress: bool = True,
    cache_path: Path | None = None,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[float], list[int]]:
    if cache_path is not None and cache_path.exists():
        print(f"  Loading cached features from {cache_path}")
        cached = torch.load(cache_path, weights_only=False)
        return cached["bytes"], cached["features"], cached["labels"], cached["lengths"]
    
    byte_tensors, feature_tensors, labels, lengths = _extract_features(
        samples, text_freq, innocuous_freq, secret_freq, max_len, show_progress
    )
    
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "bytes": byte_tensors,
            "features": feature_tensors,
            "labels": labels,
            "lengths": lengths,
        }, cache_path)
        print(f"  Cached features to {cache_path}")
    
    return byte_tensors, feature_tensors, labels, lengths


def compute_length_bins(lengths: list[int], n_bins: int = 8) -> list[int]:
    if not lengths:
        return [512]
    sorted_lens = sorted(lengths)
    n = len(sorted_lens)
    bins = []
    for i in range(1, n_bins):
        idx = int(n * i / n_bins)
        bins.append(sorted_lens[idx])
    bins.append(sorted_lens[-1])
    return sorted(set(bins))


def create_bucketed_batches(
    byte_tensors: list[torch.Tensor],
    feature_tensors: list[torch.Tensor],
    labels: list[float],
    lengths: list[int],
    batch_size: int,
    indices: list[int] | None = None,
    shuffle: bool = True,
    seed: int | None = None,
    length_bins: list[int] | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    if seed is not None:
        random.seed(seed)
    
    if indices is None:
        indices = list(range(len(byte_tensors)))
    
    if length_bins is None:
        length_bins = compute_length_bins([lengths[i] for i in indices])
    
    bins: dict[int, list[int]] = {b: [] for b in length_bins}
    bins[-1] = []
    
    for idx in indices:
        length = lengths[idx]
        for bin_max in length_bins:
            if length <= bin_max:
                bins[bin_max].append(idx)
                break
        else:
            bins[-1].append(idx)
    
    batches = []
    for bin_indices in bins.values():
        if not bin_indices:
            continue
        
        if shuffle:
            random.shuffle(bin_indices)
        
        for i in range(0, len(bin_indices), batch_size):
            batch_indices = bin_indices[i:i + batch_size]
            
            batch_bytes = [byte_tensors[j] for j in batch_indices]
            batch_features = torch.stack([feature_tensors[j] for j in batch_indices])
            batch_labels = torch.tensor([labels[j] for j in batch_indices])
            batch_lengths = torch.tensor([lengths[j] for j in batch_indices])
            
            padded_bytes = pad_sequence(batch_bytes, batch_first=True, padding_value=0)
            batches.append((padded_bytes, batch_features, batch_labels, batch_lengths))
    
    if shuffle:
        random.shuffle(batches)
    
    return batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=3)
    parser.add_argument("--ffn-mult", type=int, default=4)
    parser.add_argument("--mlp-dims", type=str, default="64,32")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--output", type=Path, default=Path("models/validator.pt"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--hard-negatives", type=Path, default=None)
    parser.add_argument("--hard-neg-ratio", type=float, default=1.0, help="Ratio of hard negatives to secrets per epoch")
    parser.add_argument("--pos-weight-mult", type=float, default=1.0, help="Multiplier for auto-computed pos_weight")
    args = parser.parse_args()

    splits_dir = args.data_dir / "splits" / "validator"
    
    print("Loading validator samples...")
    base_train_samples = load_validator_samples(splits_dir / "train.jsonl")
    val_samples = load_validator_samples(splits_dir / "val.jsonl")

    hard_neg_samples: list[ValidatorSample] = []
    if args.hard_negatives:
        print(f"Loading hard negatives from {args.hard_negatives}...")
        hard_neg_samples = load_validator_samples(args.hard_negatives)
        print(f"  {len(hard_neg_samples):,} hard negatives available")

    base_positives = [s for s in base_train_samples if s.label == 1]
    base_negatives = [s for s in base_train_samples if s.label == 0]
    n_positives = len(base_positives)
    n_hard_per_epoch = min(int(n_positives * args.hard_neg_ratio), len(hard_neg_samples))

    n_negatives_per_epoch = len(base_negatives) + n_hard_per_epoch
    base_pos_weight = n_negatives_per_epoch / n_positives if n_positives > 0 else 1.0
    pos_weight = base_pos_weight * args.pos_weight_mult

    print(f"\nTraining config:")
    print(f"  Positives (secrets): {n_positives:,}")
    print(f"  Base negatives: {len(base_negatives):,}")
    print(f"  Hard negatives available: {len(hard_neg_samples):,}")
    print(f"  Hard negatives per epoch: {n_hard_per_epoch:,} ({args.hard_neg_ratio:.1f}x secrets)")
    print(f"  Negatives per epoch: {n_negatives_per_epoch:,}")
    print(f"  Class ratio: {n_negatives_per_epoch / n_positives:.1f}:1 (neg:pos)")
    print(f"  pos_weight: {pos_weight:.2f} (base {base_pos_weight:.2f} x {args.pos_weight_mult:.1f})")
    
    all_train_for_freq = base_train_samples + hard_neg_samples[:min(1000, len(hard_neg_samples))]
    print("\nBuilding frequency tables...")
    text_freq, innocuous_freq, secret_freq = build_frequency_tables(all_train_for_freq)

    freq_hash = _compute_freq_hash(text_freq, innocuous_freq, secret_freq)
    train_cache_path = _get_cache_path(splits_dir / "train.jsonl", freq_hash, args.max_len)
    val_cache_path = _get_cache_path(splits_dir / "val.jsonl", freq_hash, args.max_len)
    hard_neg_cache_path = _get_cache_path(args.hard_negatives, freq_hash, args.max_len) if args.hard_negatives else None

    print("\nConverting base samples to tensors...")
    base_bytes, base_features, base_labels, base_lengths = samples_to_tensors(
        base_train_samples, text_freq, innocuous_freq, secret_freq, args.max_len,
        cache_path=train_cache_path
    )
    base_pos_indices = [i for i, l in enumerate(base_labels) if l == 1.0]
    base_neg_indices = [i for i, l in enumerate(base_labels) if l == 0.0]

    hard_bytes: list[torch.Tensor] = []
    hard_features: list[torch.Tensor] = []
    hard_labels: list[float] = []
    hard_lengths: list[int] = []
    if hard_neg_samples:
        print("Converting hard negatives to tensors...")
        hard_bytes, hard_features, hard_labels, hard_lengths = samples_to_tensors(
            hard_neg_samples, text_freq, innocuous_freq, secret_freq, args.max_len,
            cache_path=hard_neg_cache_path
        )

    print("\nConverting validation samples to tensors...")
    val_bytes, val_features, val_labels, val_lengths = samples_to_tensors(
        val_samples, text_freq, innocuous_freq, secret_freq, args.max_len,
        cache_path=val_cache_path
    )
    val_batches = create_bucketed_batches(
        val_bytes, val_features, val_labels, val_lengths, args.batch_size, shuffle=False
    )

    n_val_pos = sum(1 for l in val_labels if l == 1.0)
    print(f"Val: {len(val_samples):,} ({n_val_pos:,} secrets)")
    print(f"Val batches: {len(val_batches)}")

    mlp_dims = tuple(int(d) for d in args.mlp_dims.split(","))
    config = ValidatorConfig(
        width=args.width,
        depth=args.depth,
        num_heads=args.num_heads,
        ffn_mult=args.ffn_mult,
        mlp_dims=mlp_dims,
        dropout=args.dropout,
    )
    model = ValidatorModel(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"\nModel: width={config.width}, depth={config.depth}, heads={config.num_heads}, ffn_mult={config.ffn_mult}, mlp={config.mlp_dims}")
    print(f"Parameters: {param_count:,}")

    device = torch.device("cpu") if args.cpu else get_device()
    print(f"Device: {device}")

    initial_train_batches = create_bucketed_batches(
        base_bytes, base_features, base_labels, base_lengths,
        args.batch_size, indices=base_pos_indices + base_neg_indices, seed=args.seed
    )

    trainer = ValidatorTrainer(
        model=model,
        train_data=initial_train_batches,
        val_data=val_batches,
        lr=args.lr,
        device=device,
        epochs=args.epochs,
        seed=args.seed,
        pos_weight=pos_weight,
    )

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
        if hard_neg_samples and n_hard_per_epoch > 0:
            random.seed(args.seed + epoch)
            sampled_hard_indices = random.sample(range(len(hard_neg_samples)), n_hard_per_epoch)
            
            all_bytes = base_bytes + [hard_bytes[i] for i in sampled_hard_indices]
            all_features = base_features + [hard_features[i] for i in sampled_hard_indices]
            all_labels = base_labels + [hard_labels[i] for i in sampled_hard_indices]
            all_lengths = base_lengths + [hard_lengths[i] for i in sampled_hard_indices]
            
            train_batches = create_bucketed_batches(
                all_bytes, all_features, all_labels, all_lengths,
                args.batch_size, seed=args.seed + epoch
            )
            trainer.set_train_data(train_batches)

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
            "text_freq": text_freq.to_dict(),
            "innocuous_freq": innocuous_freq.to_dict(),
            "secret_freq": secret_freq.to_dict(),
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
            "train_fp": train_metrics.fp_rate,
            "train_fn": train_metrics.fn_rate,
            "val_loss": val_metrics.loss,
            "val_acc": val_metrics.accuracy,
            "val_f1": val_metrics.f1,
            "val_p": val_metrics.precision,
            "val_r": val_metrics.recall,
            "val_fp": val_metrics.fp_rate,
            "val_fn": val_metrics.fn_rate,
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
        "text_freq": text_freq.to_dict(),
        "innocuous_freq": innocuous_freq.to_dict(),
        "secret_freq": secret_freq.to_dict(),
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
        "innocuous_freq": innocuous_freq.to_dict(),
        "secret_freq": secret_freq.to_dict(),
    }, args.output)

    print(f"\nSaved best model (epoch {best_epoch}, F1={best_f1:.4f}) to {args.output}")


if __name__ == "__main__":
    main()
