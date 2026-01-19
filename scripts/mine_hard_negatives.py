#!/usr/bin/env python3
"""
Mine hard negatives from ByteMasker false positives.

Runs ByteMasker over clean lines, extracts segments it thinks are secrets,
and saves them as ValidatorSamples with label=0 for training the Validator.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from tqdm import tqdm

from heuristic_secrets.bytemasker.model import ByteMaskerConfig, ByteMaskerModel
from heuristic_secrets.bytemasker.dataset import load_bytemasker_dataset
from heuristic_secrets.data.types import ValidatorSample


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


def main():
    parser = argparse.ArgumentParser(description="Mine hard negatives from ByteMasker FPs")
    parser.add_argument("--masker-checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val"])
    parser.add_argument("--min-segment-len", type=int, default=4)
    parser.add_argument("--max-segment-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu", action="store_true")
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
    model, threshold = load_masker(args.masker_checkpoint, device)
    print(f"  Threshold: {threshold:.3f}")

    print(f"\nLoading {args.split} dataset...")
    dataset, stats = load_bytemasker_dataset(args.data_dir, args.split, secrets_only=False)
    print(f"  {len(dataset.clean_lines):,} clean lines to scan")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nMining false positives -> {args.output}")
    total_samples = 0
    total_lens = []
    clean_indices = dataset.clean_lines

    with open(args.output, "w") as f:
        with torch.no_grad():
            for batch_start in tqdm(range(0, len(clean_indices), args.batch_size), desc="Mining FPs"):
                batch_indices = clean_indices[batch_start:batch_start + args.batch_size]

                byte_tensors = [dataset.byte_tensors[i] for i in batch_indices]
                lengths = torch.tensor([dataset.lengths[i] for i in batch_indices])

                max_len = max(len(t) for t in byte_tensors)
                padded = torch.zeros(len(byte_tensors), max_len, dtype=torch.long)
                for i, t in enumerate(byte_tensors):
                    padded[i, :len(t)] = t

                padded = padded.to(device)
                logits = model(padded)
                probs = torch.sigmoid(logits)
                masks = (probs >= threshold).cpu()

                for i, idx in enumerate(batch_indices):
                    line = dataset.lines[idx]
                    line_len = lengths[i].item()
                    mask = masks[i, :line_len]

                    if not mask.any():
                        continue

                    segments = extract_segments(mask)
                    for start, end in segments:
                        seg_len = end - start
                        if seg_len < args.min_segment_len or seg_len > args.max_segment_len:
                            continue

                        segment_bytes = line.bytes[start:end]
                        try:
                            text = segment_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            continue

                        sample = ValidatorSample(
                            text=text,
                            label=0,
                            source="bytemasker_fp",
                            category="hard_negative",
                        )
                        f.write(json.dumps(sample.to_dict()) + "\n")
                        total_samples += 1
                        total_lens.append(len(text))

    print(f"\nWrote {total_samples:,} hard negative segments to {args.output}")
    if total_lens:
        print(f"Segment lengths: min={min(total_lens)}, max={max(total_lens)}, avg={sum(total_lens)/len(total_lens):.1f}")


if __name__ == "__main__":
    main()
