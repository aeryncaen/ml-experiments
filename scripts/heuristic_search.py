#!/usr/bin/env python3
"""
Search for optimal combination of heuristic features.
Uses the best model config from arch search: w48_both_k3579_a1_e8_mlp4x32
Tests all 2^6 - 1 = 63 non-empty subsets of the 6 heuristics.
"""

import argparse
import csv
import itertools
import random
from pathlib import Path

import torch
from tqdm import tqdm

from heuristic_secrets.models import (
    Model,
    ModelConfig,
    Trainer,
    TrainConfig,
    create_binary_batches,
    get_device,
    set_seed,
)
from heuristic_secrets.bytemasker.dataset import load_bytemasker_dataset
from heuristic_secrets.validator.features import (
    shannon_entropy,
    kolmogorov_complexity,
    char_type_mix,
    char_frequency_difference,
    FrequencyTable,
)


CACHE_DIR = Path(".cache/line_filter")

HEURISTIC_NAMES = [
    "entropy",
    "kolmogorov", 
    "text_freq",
    "innocuous_freq",
    "secret_freq",
    "char_mix",
]

HEURISTIC_FUNCS = {
    "entropy": lambda b, tf, inf, sf: shannon_entropy(b),
    "kolmogorov": lambda b, tf, inf, sf: kolmogorov_complexity(b),
    "text_freq": lambda b, tf, inf, sf: char_frequency_difference(b, tf),
    "innocuous_freq": lambda b, tf, inf, sf: char_frequency_difference(b, inf),
    "secret_freq": lambda b, tf, inf, sf: char_frequency_difference(b, sf),
    "char_mix": lambda b, tf, inf, sf: char_type_mix(b),
}


def load_freq_tables():
    cache = CACHE_DIR / "freq_tables.pt"
    if cache.exists():
        data = torch.load(cache, weights_only=False)
        return (
            FrequencyTable.from_dict(data["text"]),
            FrequencyTable.from_dict(data["innocuous"]),
            FrequencyTable.from_dict(data["secret"]),
        )
    raise FileNotFoundError(f"Run arch search first to generate {cache}")


def extract_features_subset(dataset, indices: list[int], heuristic_mask: tuple[bool, ...], 
                            text_freq, innocuous_freq, secret_freq,
                            cache_key: str, n_samples: int, seed: int):
    mask_str = "".join(str(int(b)) for b in heuristic_mask)
    cache_path = CACHE_DIR / f"heuristics_{cache_key}_n{n_samples}_s{seed}_{mask_str}.pt"
    
    if cache_path.exists():
        return torch.load(cache_path, weights_only=False)
    
    active_names = [n for n, m in zip(HEURISTIC_NAMES, heuristic_mask) if m]
    active_funcs = [HEURISTIC_FUNCS[n] for n in active_names]
    
    features = []
    for i in tqdm(indices, desc=f"Extracting {mask_str}", leave=False):
        line_bytes = dataset.lines[i].bytes
        f = torch.tensor([
            func(line_bytes, text_freq, innocuous_freq, secret_freq)
            for func in active_funcs
        ], dtype=torch.float32)
        features.append(f)
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, cache_path)
    return features


def run_search(
    heuristic_mask: tuple[bool, ...],
    train_data: tuple,
    val_data: tuple,
    train_config: TrainConfig,
    device: torch.device,
) -> dict:
    num_heuristics = sum(heuristic_mask)
    
    model_config = ModelConfig(
        task="binary",
        arch_type="both",
        embed_width=48,
        conv_kernel_sizes=(3, 5, 7, 9),
        conv_groups=3,
        attn_depth=1,
        attn_heads=4,
        num_embed_features=8,
        mlp_hidden_mult=4,
        mlp_output_dim=32,
        num_precomputed_features=num_heuristics,
    )
    
    model = Model(model_config)
    param_count = sum(p.numel() for p in model.parameters())
    
    train_bytes, train_labels, train_lengths, train_features = train_data
    val_bytes, val_labels, val_lengths, val_features = val_data
    
    train_batches = create_binary_batches(
        train_bytes, train_labels, train_lengths,
        batch_size=64,
        feature_tensors=train_features,
        seed=train_config.seed,
        show_progress=False,
    )
    val_batches = create_binary_batches(
        val_bytes, val_labels, val_lengths,
        batch_size=64,
        feature_tensors=val_features,
        shuffle=False,
        show_progress=False,
    )
    
    trainer = Trainer(model, train_batches, val_batches, config=train_config, device=device)
    
    for epoch in range(train_config.epochs):
        trainer.train_epoch(epoch)
    
    trainer.finalize_swa()
    final_metrics = trainer.validate(optimize_threshold=True)
    
    active_names = [n for n, m in zip(HEURISTIC_NAMES, heuristic_mask) if m]
    
    return {
        "heuristics": "+".join(active_names),
        "mask": "".join(str(int(b)) for b in heuristic_mask),
        "num_heuristics": num_heuristics,
        "params": param_count,
        "f1": final_metrics.f1,
        "recall": final_metrics.recall,
        "precision": final_metrics.precision,
        "threshold": trainer.optimal_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Search for optimal heuristic combination")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/heuristic_search.csv"))
    parser.add_argument("--no-swa", action="store_true")
    parser.add_argument("--max-samples", type=int, default=100_000)
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    
    print("\nLoading frequency tables...")
    text_freq, innocuous_freq, secret_freq = load_freq_tables()
    
    print("\nLoading datasets...")
    train_dataset, _ = load_bytemasker_dataset(args.data_dir, "train")
    val_dataset, _ = load_bytemasker_dataset(args.data_dir, "val")
    
    random.seed(args.seed)
    n = args.max_samples
    n_train_secret = min(n // 2, len(train_dataset.secret_lines))
    n_train_clean = min(n // 2, len(train_dataset.clean_lines))
    train_idx = random.sample(train_dataset.secret_lines, n_train_secret) + \
                random.sample(train_dataset.clean_lines, n_train_clean)
    
    n_val = n // 5
    n_val_secret = min(n_val // 2, len(val_dataset.secret_lines))
    n_val_clean = min(n_val // 2, len(val_dataset.clean_lines))
    val_idx = random.sample(val_dataset.secret_lines, n_val_secret) + \
              random.sample(val_dataset.clean_lines, n_val_clean)
    
    train_bytes = [train_dataset.byte_tensors[i] for i in train_idx]
    train_lengths = [train_dataset.lengths[i] for i in train_idx]
    train_labels = [1.0 if train_dataset.lines[i].has_secret else 0.0 for i in train_idx]
    
    val_bytes = [val_dataset.byte_tensors[i] for i in val_idx]
    val_lengths = [val_dataset.lengths[i] for i in val_idx]
    val_labels = [1.0 if val_dataset.lines[i].has_secret else 0.0 for i in val_idx]
    
    print(f"Train: {len(train_idx):,} samples, Val: {len(val_idx):,} samples")
    
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = n_neg / n_pos
    print(f"Class balance: pos_weight={pos_weight:.2f}")
    
    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        pos_weight=pos_weight,
        threshold_optimize_for="f1",
        use_swa=not args.no_swa,
        swa_start_pct=0.5,
        preload_to_device=True,
    )
    
    # Generate all non-empty subsets
    all_masks = []
    for r in range(1, 7):
        for combo in itertools.combinations(range(6), r):
            mask = tuple(i in combo for i in range(6))
            all_masks.append(mask)
    
    # Also include no heuristics as baseline
    all_masks.insert(0, (False,) * 6)
    
    random.seed(args.seed)
    random.shuffle(all_masks)
    
    print(f"\nTesting {len(all_masks)} heuristic combinations...")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["heuristics", "mask", "num_heuristics", "params", "f1", "recall", "precision", "threshold"]
    
    completed = set()
    results = []
    if args.output.exists():
        with open(args.output, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row["mask"])
                results.append(row)
        print(f"  Found {len(completed)} completed experiments")
    
    remaining = [m for m in all_masks if "".join(str(int(b)) for b in m) not in completed]
    
    need_header = not args.output.exists() or len(completed) == 0
    if need_header:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
    
    for i, mask in enumerate(remaining):
        mask_str = "".join(str(int(b)) for b in mask)
        active = [n for n, m in zip(HEURISTIC_NAMES, mask) if m]
        print(f"\n[{i+1}/{len(remaining)}] {mask_str} ({'+'.join(active) or 'none'})")
        
        if any(mask):
            train_feat = extract_features_subset(
                train_dataset, train_idx, mask, text_freq, innocuous_freq, secret_freq,
                "train", args.max_samples, args.seed
            )
            val_feat = extract_features_subset(
                val_dataset, val_idx, mask, text_freq, innocuous_freq, secret_freq,
                "val", args.max_samples // 5, args.seed
            )
        else:
            train_feat = None
            val_feat = None
        
        train_data = (train_bytes, train_labels, train_lengths, train_feat)
        val_data = (val_bytes, val_labels, val_lengths, val_feat)
        
        try:
            result = run_search(mask, train_data, val_data, train_config, device)
            results.append(result)
            
            with open(args.output, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result)
            
            print(f"  → F1={result['f1']:.4f} P={result['precision']:.4f} R={result['recall']:.4f}")
        except Exception as e:
            print(f"  → FAILED: {e}")
            continue
    
    # Sort and print final results
    for r in results:
        for k in ["f1", "precision", "recall"]:
            if isinstance(r.get(k), str):
                r[k] = float(r[k])
    
    results.sort(key=lambda x: float(x["f1"]), reverse=True)
    
    print(f"\n{'='*100}")
    print("ALL RESULTS (sorted by F1):")
    print(f"{'='*100}")
    print(f"{'Rank':<5} {'Heuristics':<50} {'N':>3} {'F1':>8} {'P':>8} {'R':>8}")
    print("-" * 100)
    for i, r in enumerate(results):
        print(f"{i+1:<5} {r['heuristics']:<50} {r['num_heuristics']:>3} {float(r['f1']):>8.4f} {float(r['precision']):>8.4f} {float(r['recall']):>8.4f}")
    
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
