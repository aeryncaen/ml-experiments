#!/usr/bin/env python3
"""
Architecture search for the line filter problem.

Task: Given a line of text, does it contain a secret? (binary classification)

Tests combinations of:
- Backbone type: conv vs attention
- Width: 16, 24, 32, 48
- Depth: 1, 2, 3, 4
- With/without precomputed heuristic features
"""

import argparse
import csv
import itertools
import random
import time
from dataclasses import dataclass
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
from heuristic_secrets.bytemasker.dataset import (
    load_bytemasker_dataset,
    LineMaskDataset,
)
from heuristic_secrets.validator.features import (
    shannon_entropy,
    kolmogorov_complexity,
    char_type_mix,
    char_frequency_difference,
    FrequencyTable,
)


CACHE_DIR = Path(".cache/line_filter")


def get_heuristic_cache_path(data_dir: Path, split: str) -> Path:
    return CACHE_DIR / f"heuristics_{split}.pt"


def get_subsample_cache_path(data_dir: Path, split: str, n_samples: int, seed: int) -> Path:
    return CACHE_DIR / f"subsample_{split}_n{n_samples}_s{seed}.pt"


def build_frequency_tables(dataset: LineMaskDataset) -> tuple[FrequencyTable, FrequencyTable, FrequencyTable]:
    all_bytes = b"".join(line.bytes for line in dataset.lines)
    text_freq = FrequencyTable.from_data(all_bytes)
    
    innocuous_bytes = b"".join(dataset.lines[i].bytes for i in dataset.clean_lines)
    innocuous_freq = FrequencyTable.from_data(innocuous_bytes)
    
    secret_bytes = b"".join(dataset.lines[i].bytes for i in dataset.secret_lines)
    secret_freq = FrequencyTable.from_data(secret_bytes)
    
    return text_freq, innocuous_freq, secret_freq


def extract_and_cache_heuristics(
    dataset: LineMaskDataset,
    cache_path: Path,
    text_freq: FrequencyTable,
    innocuous_freq: FrequencyTable,
    secret_freq: FrequencyTable,
) -> list[torch.Tensor]:
    if cache_path.exists():
        print(f"  Loading cached heuristics from {cache_path}")
        return torch.load(cache_path, weights_only=False)

    features = []
    for i in tqdm(range(len(dataset)), desc="Extracting heuristics"):
        line_bytes = dataset.lines[i].bytes
        f = torch.tensor([
            shannon_entropy(line_bytes),
            kolmogorov_complexity(line_bytes),
            char_frequency_difference(line_bytes, text_freq),
            char_frequency_difference(line_bytes, innocuous_freq),
            char_frequency_difference(line_bytes, secret_freq),
            char_type_mix(line_bytes),
        ], dtype=torch.float32)
        features.append(f)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, cache_path)
    print(f"  Cached heuristics to {cache_path}")
    return features


def load_or_create_subsample(
    data_dir: Path,
    split: str,
    n_samples: int,
    seed: int,
    text_freq: FrequencyTable,
    innocuous_freq: FrequencyTable,
    secret_freq: FrequencyTable,
) -> tuple[list[torch.Tensor], list[float], list[int], list[torch.Tensor]]:
    import random
    
    cache_path = get_subsample_cache_path(data_dir, split, n_samples, seed)
    
    if cache_path.exists():
        print(f"  Loading cached subsample from {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        return data["bytes"], data["labels"], data["lengths"], data["features"]
    
    print(f"  Creating subsample of {n_samples} for {split}...")
    dataset, _ = load_bytemasker_dataset(data_dir, split)
    
    random.seed(seed)
    n_secret = min(n_samples // 2, len(dataset.secret_lines))
    n_clean = min(n_samples // 2, len(dataset.clean_lines))
    
    secret_idx = random.sample(dataset.secret_lines, n_secret)
    clean_idx = random.sample(dataset.clean_lines, n_clean)
    indices = secret_idx + clean_idx
    
    byte_tensors = [dataset.byte_tensors[i] for i in indices]
    labels = [1.0 if dataset.lines[i].has_secret else 0.0 for i in indices]
    lengths = [dataset.lengths[i] for i in indices]
    
    features = []
    for i in tqdm(indices, desc=f"Extracting {split} features"):
        line_bytes = dataset.lines[i].bytes
        f = torch.tensor([
            shannon_entropy(line_bytes),
            kolmogorov_complexity(line_bytes),
            char_frequency_difference(line_bytes, text_freq),
            char_frequency_difference(line_bytes, innocuous_freq),
            char_frequency_difference(line_bytes, secret_freq),
            char_type_mix(line_bytes),
        ], dtype=torch.float32)
        features.append(f)
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "bytes": byte_tensors,
        "labels": labels,
        "lengths": lengths,
        "features": features,
    }, cache_path)
    print(f"  Cached subsample to {cache_path}")
    
    return byte_tensors, labels, lengths, features


@dataclass
class ArchConfig:
    arch_type: str
    embed_width: int
    conv_kernel_sizes: tuple[int, ...]
    attn_depth: int
    mlp_hidden_mult: int
    mlp_output_dim: int
    use_heuristics: bool
    num_embed_features: int = 4

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            task="binary",
            arch_type=self.arch_type,  # type: ignore
            embed_width=self.embed_width,
            conv_kernel_sizes=self.conv_kernel_sizes,
            conv_groups=max(1, self.embed_width // 16),
            attn_depth=self.attn_depth,
            attn_heads=max(1, self.embed_width // 16),
            mlp_hidden_mult=self.mlp_hidden_mult,
            mlp_output_dim=self.mlp_output_dim,
            num_precomputed_features=6 if self.use_heuristics else 0,
            num_embed_features=self.num_embed_features,
        )

    def name(self) -> str:
        if self.arch_type == "conv":
            ks = "".join(str(k) for k in self.conv_kernel_sizes)
            arch_str = f"conv_k{ks}"
        elif self.arch_type == "attn":
            arch_str = f"attn_d{self.attn_depth}"
        else:
            ks = "".join(str(k) for k in self.conv_kernel_sizes)
            arch_str = f"both_k{ks}_a{self.attn_depth}"
        h = "+heur" if self.use_heuristics else ""
        return f"w{self.embed_width}_{arch_str}_e{self.num_embed_features}_mlp{self.mlp_hidden_mult}x{self.mlp_output_dim}{h}"


def benchmark_inference(
    model: Model,
    batches: list,
    device: torch.device,
    n_runs: int = 3,
) -> float:
    model.eval()
    
    if not batches:
        return 0.0
    
    with torch.no_grad():
        for batch in batches[:2]:
            _ = model(
                batch.bytes.to(device),
                mask=None,
                precomputed=batch.features.to(device) if batch.features is not None else None,
            )
    
    total_samples = 0
    total_time = 0.0
    
    with torch.no_grad():
        for _ in range(n_runs):
            for batch in batches:
                batch_bytes = batch.bytes.to(device)
                batch_features = batch.features.to(device) if batch.features is not None else None
                
                start = time.perf_counter()
                _ = model(batch_bytes, mask=None, precomputed=batch_features)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()
                end = time.perf_counter()
                
                total_time += (end - start)
                total_samples += len(batch.bytes)
    
    return total_samples / total_time if total_time > 0 else 0.0


def run_arch_search(
    arch: ArchConfig,
    train_data: tuple,
    val_data: tuple,
    train_config: TrainConfig,
    device: torch.device,
) -> dict:
    model_config = arch.to_model_config()
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

    samples_per_sec = benchmark_inference(model, val_batches, device)

    return {
        "arch": arch.name(),
        "arch_type": arch.arch_type,
        "embed_width": arch.embed_width,
        "conv_kernels": ",".join(str(k) for k in arch.conv_kernel_sizes) if arch.conv_kernel_sizes else "",
        "attn_depth": arch.attn_depth,
        "num_embed_features": arch.num_embed_features,
        "mlp_hidden_mult": arch.mlp_hidden_mult,
        "mlp_output_dim": arch.mlp_output_dim,
        "use_heuristics": arch.use_heuristics,
        "params": param_count,
        "f1": final_metrics.f1,
        "recall": final_metrics.recall,
        "precision": final_metrics.precision,
        "threshold": trainer.optimal_threshold,
        "samples_per_sec": samples_per_sec,
    }


def main():
    parser = argparse.ArgumentParser(description="Architecture search for line filter")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/line_filter_arch_search.csv"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true", help="Run reduced search space")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for quick testing")
    parser.add_argument("--no-swa", action="store_true", help="Disable SWA for faster iteration")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu") if args.cpu else get_device()
    print(f"Device: {device}")

    freq_cache = CACHE_DIR / "freq_tables.pt"
    need_full_dataset = not args.max_samples
    train_dataset: LineMaskDataset | None = None
    
    if freq_cache.exists():
        print(f"\nLoading cached frequency tables from {freq_cache}")
        freq_data = torch.load(freq_cache, weights_only=False)
        text_freq = FrequencyTable.from_dict(freq_data["text"])
        innocuous_freq = FrequencyTable.from_dict(freq_data["innocuous"])
        secret_freq = FrequencyTable.from_dict(freq_data["secret"])
        
        if need_full_dataset:
            print("\nLoading training data...")
            train_dataset, _ = load_bytemasker_dataset(args.data_dir, "train")
    else:
        print("\nLoading training data for frequency tables...")
        train_dataset, _ = load_bytemasker_dataset(args.data_dir, "train")
        print("  Building frequency tables...")
        text_freq, innocuous_freq, secret_freq = build_frequency_tables(train_dataset)
        freq_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "text": text_freq.to_dict(),
            "innocuous": innocuous_freq.to_dict(),
            "secret": secret_freq.to_dict(),
        }, freq_cache)
        print(f"  Cached frequency tables to {freq_cache}")

    if args.max_samples:
        print(f"\nLoading/creating subsampled datasets (n={args.max_samples})...")
        train_bytes, train_labels, train_lengths, train_feat = load_or_create_subsample(
            args.data_dir, "train", args.max_samples, args.seed, text_freq, innocuous_freq, secret_freq
        )
        val_bytes, val_labels, val_lengths, val_feat = load_or_create_subsample(
            args.data_dir, "val", args.max_samples // 5, args.seed, text_freq, innocuous_freq, secret_freq
        )
        print(f"Train: {len(train_bytes)} samples, Val: {len(val_bytes)} samples")
    else:
        assert train_dataset is not None
        val_dataset, _ = load_bytemasker_dataset(args.data_dir, "val")

        print(f"Train: {len(train_dataset)} lines ({len(train_dataset.secret_lines)} with secrets)")
        print(f"Val: {len(val_dataset)} lines ({len(val_dataset.secret_lines)} with secrets)")

        train_bytes = train_dataset.byte_tensors
        train_labels = [1.0 if train_dataset.lines[i].has_secret else 0.0 for i in range(len(train_dataset))]
        train_lengths = train_dataset.lengths

        val_bytes = val_dataset.byte_tensors
        val_labels = [1.0 if val_dataset.lines[i].has_secret else 0.0 for i in range(len(val_dataset))]
        val_lengths = val_dataset.lengths

        print("\nLoading/extracting heuristic features...")
        train_feat = extract_and_cache_heuristics(
            train_dataset, get_heuristic_cache_path(args.data_dir, "train"),
            text_freq, innocuous_freq, secret_freq
        )
        val_feat = extract_and_cache_heuristics(
            val_dataset, get_heuristic_cache_path(args.data_dir, "val"),
            text_freq, innocuous_freq, secret_freq
        )

    train_data_no_feat = (train_bytes, train_labels, train_lengths, None)
    train_data_with_feat = (train_bytes, train_labels, train_lengths, train_feat)
    val_data_no_feat = (val_bytes, val_labels, val_lengths, None)
    val_data_with_feat = (val_bytes, val_labels, val_lengths, val_feat)

    n_pos = sum(1 for l in train_labels if l == 1.0)
    n_neg = sum(1 for l in train_labels if l == 0.0)
    pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    print(f"\nClass balance: {n_neg}:{n_pos} (neg:pos), pos_weight={pos_weight:.2f}")

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

    arch_types = ["conv", "attn", "both"]
    embed_widths = [16, 48]
    kernel_size_opts = [(5,), (3, 5, 7, 9)]
    attn_depths = [1, 3]
    mlp_configs = [(4, 32), (8, 128)]
    num_embed_features_opts = [0, 8]
    use_heuristics_opts = [False, True]

    configs = []
    for arch_type, embed_w, (mlp_mult, mlp_out), use_heur, num_embed in itertools.product(
        arch_types, embed_widths, mlp_configs, use_heuristics_opts, num_embed_features_opts
    ):
        if arch_type == "conv":
            for kernels in kernel_size_opts:
                configs.append(ArchConfig(
                    arch_type=arch_type,
                    embed_width=embed_w,
                    conv_kernel_sizes=kernels,
                    attn_depth=0,
                    mlp_hidden_mult=mlp_mult,
                    mlp_output_dim=mlp_out,
                    use_heuristics=use_heur,
                    num_embed_features=num_embed,
                ))
        elif arch_type == "attn":
            for attn_d in attn_depths:
                configs.append(ArchConfig(
                    arch_type=arch_type,
                    embed_width=embed_w,
                    conv_kernel_sizes=(),
                    attn_depth=attn_d,
                    mlp_hidden_mult=mlp_mult,
                    mlp_output_dim=mlp_out,
                    use_heuristics=use_heur,
                    num_embed_features=num_embed,
                ))
        else:
            for kernels, attn_d in itertools.product(kernel_size_opts, attn_depths):
                configs.append(ArchConfig(
                    arch_type=arch_type,
                    embed_width=embed_w,
                    conv_kernel_sizes=kernels,
                    attn_depth=attn_d,
                    mlp_hidden_mult=mlp_mult,
                    mlp_output_dim=mlp_out,
                    use_heuristics=use_heur,
                    num_embed_features=num_embed,
                ))

    random.seed(args.seed)
    random.shuffle(configs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    completed: set[str] = set()
    results: list[dict] = []
    fieldnames = [
        "arch", "arch_type", "embed_width", "conv_kernels", "attn_depth", "num_embed_features",
        "mlp_hidden_mult", "mlp_output_dim", "use_heuristics", "params", "f1",
        "recall", "precision", "threshold", "samples_per_sec",
    ]
    
    if args.output.exists():
        print(f"\nLoading existing results from {args.output}...")
        with open(args.output, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row['arch'])
                results.append(row)
        print(f"  Found {len(completed)} completed experiments")

    remaining = []
    for arch in configs:
        if arch.name() not in completed:
            remaining.append(arch)

    print(f"\nRunning {len(remaining)}/{len(configs)} architecture configurations...")
    print(f"{'='*70}")

    need_header = not (args.output.exists() and len(completed) > 0)
    if need_header:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for i, arch in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] {arch.name()}")

        train_data = train_data_with_feat if arch.use_heuristics else train_data_no_feat
        val_data = val_data_with_feat if arch.use_heuristics else val_data_no_feat

        try:
            result = run_arch_search(arch, train_data, val_data, train_config, device)
            results.append(result)
            
            with open(args.output, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(result)
            
            print(f"  → F1={result['f1']:.4f} P={result['precision']:.4f} R={result['recall']:.4f} | {result['samples_per_sec']:,.0f} samples/sec | {result['params']:,} params")
        except Exception as e:
            print(f"  → FAILED: {e}")
            continue

    for r in results:
        for k in ["f1", "precision", "recall", "samples_per_sec"]:
            if isinstance(r.get(k), str):
                r[k] = float(r[k])
        if isinstance(r.get("params"), str):
            r["params"] = int(r["params"])

    results.sort(key=lambda x: float(x["f1"]), reverse=True)

    print(f"\n{'='*120}")
    print("ALL RESULTS (sorted by F1):")
    print(f"{'='*120}")
    print(f"{'Rank':<5} {'Architecture':<45} {'F1':>8} {'P':>8} {'R':>8} {'Samples/s':>12} {'Params':>10}")
    print("-" * 120)
    for i, r in enumerate(results):
        print(f"{i+1:<5} {r['arch']:<45} {float(r['f1']):>8.4f} {float(r['precision']):>8.4f} {float(r['recall']):>8.4f} {float(r['samples_per_sec']):>12,.0f} {int(r['params']):>10,}")

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
