#!/usr/bin/env python3
"""
Evaluate the ByteMasker -> Validator pipeline on full document lines.

Pipeline logic:
1. ByteMasker predicts which bytes are part of a secret
2. If no bytes masked -> predict "no secret"
3. If bytes masked -> extract contiguous segments, validate each
4. If ANY segment validates as secret -> predict "secret"

Outputs full metrics table matching benchmark format:
TP, FP, TN, FN, FPR, FNR, Accuracy, Precision, Recall, F1
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from heuristic_secrets.bytemasker.model import ByteMaskerConfig, ByteMaskerModel
from heuristic_secrets.bytemasker.dataset import load_bytemasker_dataset
from heuristic_secrets.validator.model import ValidatorConfig, ValidatorModel
from heuristic_secrets.validator.features import (
    FrequencyTable,
    shannon_entropy,
    kolmogorov_complexity,
    char_frequency_difference,
    char_type_mix,
)


@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def total_positive(self) -> int:
        return self.tp + self.fn

    @property
    def total_negative(self) -> int:
        return self.tn + self.fp

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def fpr(self) -> float:
        if self.total_negative == 0:
            return 0.0
        return self.fp / self.total_negative

    @property
    def fnr(self) -> float:
        if self.total_positive == 0:
            return 0.0
        return self.fn / self.total_positive

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.tp + self.tn) / self.total

    @property
    def precision(self) -> float:
        if self.tp + self.fp == 0:
            return 0.0
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self) -> float:
        if self.total_positive == 0:
            return 0.0
        return self.tp / self.total_positive

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)


def extract_segments(mask: torch.Tensor) -> list[tuple[int, int]]:
    segments = []
    in_segment = False
    start = 0

    for i, val in enumerate(mask.tolist()):
        if val and not in_segment:
            start = i
            in_segment = True
        elif not val and in_segment:
            segments.append((start, i))
            in_segment = False

    if in_segment:
        segments.append((start, len(mask)))

    return segments


def load_masker(checkpoint_path: Path, device: torch.device) -> tuple[ByteMaskerModel, float]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ByteMaskerConfig.from_dict(ckpt["config"])
    model = ByteMaskerModel(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    threshold = ckpt.get("optimal_threshold", 0.5)
    return model, threshold


def load_validator(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[ValidatorModel, float, FrequencyTable | None, FrequencyTable | None, FrequencyTable | None]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ValidatorConfig.from_dict(ckpt["config"])
    model = ValidatorModel(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    threshold = ckpt.get("optimal_threshold", 0.5)
    text_freq = FrequencyTable.from_dict(ckpt["text_freq"]) if "text_freq" in ckpt else None
    innocuous_freq = FrequencyTable.from_dict(ckpt["innocuous_freq"]) if "innocuous_freq" in ckpt else None
    secret_freq = FrequencyTable.from_dict(ckpt["secret_freq"]) if "secret_freq" in ckpt else None
    return model, threshold, text_freq, innocuous_freq, secret_freq


def main():
    parser = argparse.ArgumentParser(description="Evaluate ByteMasker -> Validator pipeline")
    parser.add_argument("--masker-checkpoint", type=Path, required=True)
    parser.add_argument("--validator-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--min-segment-len", type=int, default=4)
    parser.add_argument("--masker-threshold", type=float, default=None)
    parser.add_argument("--validator-threshold", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--name", type=str, default="Pipeline")
    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Device: {device}")

    print(f"\nLoading ByteMasker from {args.masker_checkpoint}")
    masker, masker_thresh = load_masker(args.masker_checkpoint, device)
    if args.masker_threshold is not None:
        masker_thresh = args.masker_threshold
    print(f"  Threshold: {masker_thresh:.3f}")

    print(f"\nLoading Validator from {args.validator_checkpoint}")
    validator, validator_thresh, text_freq, innocuous_freq, secret_freq = load_validator(args.validator_checkpoint, device)
    if args.validator_threshold is not None:
        validator_thresh = args.validator_threshold
    print(f"  Threshold: {validator_thresh:.3f}")

    print(f"\nLoading {args.split} documents...")
    dataset, stats = load_bytemasker_dataset(args.data_dir, args.split, secrets_only=False)
    n_secret = len(dataset.secret_lines)
    n_clean = len(dataset.clean_lines)
    print(f"  {len(dataset):,} lines ({n_secret:,} with secrets, {n_clean:,} clean)")

    if text_freq is None or innocuous_freq is None or secret_freq is None:
        print("\n  Frequency tables not in checkpoint, building from training data...")
        train_dataset, _ = load_bytemasker_dataset(args.data_dir, "train", secrets_only=False)
        all_bytes = b"".join(train_dataset.lines[i].bytes for i in range(len(train_dataset)))
        secret_bytes = b"".join(
            train_dataset.lines[i].bytes for i in train_dataset.secret_lines
        )
        clean_bytes = b"".join(
            train_dataset.lines[i].bytes for i in train_dataset.clean_lines[:len(train_dataset.secret_lines)]
        )
        text_freq = FrequencyTable.from_data(all_bytes)
        innocuous_freq = FrequencyTable.from_data(clean_bytes)
        secret_freq = FrequencyTable.from_data(secret_bytes)
        print(f"  Built from {len(train_dataset.secret_lines):,} secret lines")

    print("\nEvaluating...")
    metrics = Metrics()
    batch_size = args.batch_size
    all_indices = list(range(len(dataset)))

    for batch_start in tqdm(range(0, len(all_indices), batch_size)):
        batch_indices = all_indices[batch_start:batch_start + batch_size]

        byte_tensors = [dataset.byte_tensors[i] for i in batch_indices]
        lengths = [dataset.lengths[i] for i in batch_indices]
        has_secrets = [dataset.lines[i].has_secret for i in batch_indices]

        max_len = max(lengths) if lengths else 0
        if max_len == 0:
            continue

        padded = torch.zeros(len(batch_indices), max_len, dtype=torch.long, device=device)
        for i, t in enumerate(byte_tensors):
            if len(t) > 0:
                padded[i, :len(t)] = t.to(device)

        with torch.no_grad():
            logits = masker(padded)
            probs = torch.sigmoid(logits)
            masks = (probs >= masker_thresh).cpu()

        for i, idx in enumerate(batch_indices):
            line = dataset.lines[idx]
            line_len = lengths[i]
            actual_secret = has_secrets[i]

            if line_len == 0:
                predicted_secret = False
            else:
                mask = masks[i, :line_len]
                if not mask.any():
                    predicted_secret = False
                else:
                    segments = extract_segments(mask)
                    predicted_secret = False
                    for start, end in segments:
                        if end - start < args.min_segment_len:
                            continue
                        segment_bytes = line.bytes[start:end]
                        features = torch.tensor([[
                            shannon_entropy(segment_bytes),
                            kolmogorov_complexity(segment_bytes),
                            char_frequency_difference(segment_bytes, text_freq),
                            char_frequency_difference(segment_bytes, innocuous_freq),
                            char_frequency_difference(segment_bytes, secret_freq),
                            char_type_mix(segment_bytes),
                        ]], dtype=torch.float32, device=device)
                        seg_tensor = torch.tensor(list(segment_bytes), dtype=torch.long, device=device).unsqueeze(0)
                        seg_lengths = torch.tensor([len(segment_bytes)], dtype=torch.long, device=device)
                        with torch.no_grad():
                            val_logits = validator(features, seg_tensor, seg_lengths)
                            prob = torch.sigmoid(val_logits[0, 0]).item()
                        if prob >= validator_thresh:
                            predicted_secret = True
                            break

            if actual_secret and predicted_secret:
                metrics.tp += 1
            elif actual_secret and not predicted_secret:
                metrics.fn += 1
            elif not actual_secret and predicted_secret:
                metrics.fp += 1
            else:
                metrics.tn += 1

    print("\n" + "=" * 140)
    print(f"Results for: {args.name}")
    print("=" * 140)

    print(f"| {'Name':<17} | {'TP':>5} | {'FP':>6} | {'TN':>10} | {'FN':>5} | {'FPR':>12} | {'FNR':>12} | {'Accuracy':>12} | {'Precision':>12} | {'Recall':>12} | {'F1':>12} |")
    print(f"|{'-'*19}|{'-'*7}|{'-'*8}|{'-'*12}|{'-'*7}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*14}|")
    print(f"| **{args.name:<15}** | {metrics.tp:>5,} | {metrics.fp:>6,} | {metrics.tn:>10,} | {metrics.fn:>5,} | {metrics.fpr:>12.10f} | {metrics.fnr:>12.10f} | {metrics.accuracy:>12.10f} | {metrics.precision:>12.10f} | {metrics.recall:>12.10f} | {metrics.f1:>12.10f} |")

    print("\n" + "=" * 140)
    print("Summary:")
    print(f"  Total lines:      {metrics.total:,}")
    print(f"  Actual positives: {metrics.total_positive:,}")
    print(f"  Actual negatives: {metrics.total_negative:,}")
    print(f"  Predicted pos:    {metrics.tp + metrics.fp:,}")
    print(f"  Predicted neg:    {metrics.tn + metrics.fn:,}")


if __name__ == "__main__":
    main()
