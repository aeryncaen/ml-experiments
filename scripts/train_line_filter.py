#!/usr/bin/env python3
import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch

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
from heuristic_secrets.bytemasker.dataset import (
    load_bytemasker_dataset,
    load_mixed_windows,
    load_and_split_mixed_windows,
    LineSample,
)
from heuristic_secrets.validator.features import (
    char_frequency_difference,
    FrequencyTable,
)


CACHE_DIR = Path(".cache/line_filter")


def load_or_build_text_freq_table(data_dir: Path) -> FrequencyTable:
    cache = CACHE_DIR / "freq_tables.pt"
    if cache.exists():
        data = torch.load(cache, weights_only=False)
        return FrequencyTable.from_dict(data["text"])

    print("Building frequency tables (first run only)...")
    dataset, _ = load_bytemasker_dataset(data_dir, "train")

    all_bytes = b"".join(line.bytes for line in dataset.lines)
    text_freq = FrequencyTable.from_data(all_bytes)

    innocuous_bytes = b"".join(dataset.lines[i].bytes for i in dataset.clean_lines)
    innocuous_freq = FrequencyTable.from_data(innocuous_bytes)

    secret_bytes = b"".join(dataset.lines[i].bytes for i in dataset.secret_lines)
    secret_freq = FrequencyTable.from_data(secret_bytes)

    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "text": text_freq.to_dict(),
            "innocuous": innocuous_freq.to_dict(),
            "secret": secret_freq.to_dict(),
        },
        cache,
    )
    print(f"  Cached freq tables to {cache}")

    return text_freq


def _save_dataset_npz(
    cache_path: Path,
    byte_tensors,
    labels,
    lengths,
    features,
    secret_indices,
    clean_indices,
):
    bytes_concat = torch.cat(byte_tensors)
    offsets = np.zeros(len(byte_tensors) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        bytes_concat=bytes_concat.numpy(),
        offsets=offsets,
        labels=np.array(labels, dtype=np.float32),
        lengths=np.array(lengths, dtype=np.int32),
        features=np.array(
            [f[0].item() if isinstance(f, torch.Tensor) else f for f in features],
            dtype=np.float32,
        ),
        secret_indices=np.array(secret_indices, dtype=np.int32),
        clean_indices=np.array(clean_indices, dtype=np.int32),
    )


def _load_dataset_npz(cache_path: Path):
    npz = dict(np.load(cache_path, allow_pickle=False))

    bytes_concat = torch.from_numpy(npz["bytes_concat"])
    offsets = npz["offsets"]
    byte_tensors = [
        bytes_concat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)
    ]

    return {
        "bytes": byte_tensors,
        "labels": npz["labels"].tolist(),
        "lengths": npz["lengths"].tolist(),
        "features": [torch.tensor([f], dtype=torch.float32) for f in npz["features"]],
        "secret_indices": npz["secret_indices"].tolist(),
        "clean_indices": npz["clean_indices"].tolist(),
    }


def load_dataset_with_features(data_dir: Path, split: str, text_freq: FrequencyTable):
    cache_path_npz = CACHE_DIR / f"full_{split}_v2.npz"
    cache_path_pkl = CACHE_DIR / f"full_{split}_with_text_freq.pt"

    if cache_path_npz.exists():
        print(f"  Loading cached {split} dataset from {cache_path_npz}")
        return _load_dataset_npz(cache_path_npz)

    if cache_path_pkl.exists():
        print(f"  Converting {split} pickle cache to npz...")
        data = torch.load(cache_path_pkl, weights_only=False)
        _save_dataset_npz(
            cache_path_npz,
            data["bytes"],
            data["labels"],
            data["lengths"],
            data["features"],
            data["secret_indices"],
            data["clean_indices"],
        )
        cache_path_pkl.unlink()
        print(f"  Converted to {cache_path_npz}")
        return _load_dataset_npz(cache_path_npz)

    print(f"  Loading {split} dataset...")
    dataset, _ = load_bytemasker_dataset(data_dir, split)

    print(f"  Extracting text_freq features for {len(dataset):,} samples...")
    features = [
        char_frequency_difference(dataset.lines[i].bytes, text_freq)
        for i in range(len(dataset))
    ]
    labels = [1.0 if dataset.lines[i].has_secret else 0.0 for i in range(len(dataset))]

    _save_dataset_npz(
        cache_path_npz,
        dataset.byte_tensors,
        labels,
        dataset.lengths,
        features,
        dataset.secret_lines,
        dataset.clean_lines,
    )
    print(f"  Cached to {cache_path_npz}")

    return {
        "bytes": dataset.byte_tensors,
        "labels": labels,
        "lengths": dataset.lengths,
        "features": [torch.tensor([f], dtype=torch.float32) for f in features],
        "secret_indices": dataset.secret_lines,
        "clean_indices": dataset.clean_lines,
    }


def _save_mixed_npz(
    cache_path: Path,
    windows: list[LineSample],
    features: list[float],
    all_secret_ids: set[str],
):
    byte_tensors = [torch.tensor(list(w.bytes), dtype=torch.long) for w in windows]
    lengths = [len(w.bytes) for w in windows]
    labels = [1.0 if w.has_secret else 0.0 for w in windows]

    bytes_concat = torch.cat(byte_tensors)
    offsets = np.zeros(len(byte_tensors) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)

    secret_ids_per_window = [w.secret_ids if w.secret_ids else [] for w in windows]
    secret_ids_flat = []
    sid_offsets = [0]
    for sids in secret_ids_per_window:
        secret_ids_flat.extend(sids)
        sid_offsets.append(len(secret_ids_flat))

    secret_indices = [i for i, w in enumerate(windows) if w.has_secret]
    clean_indices = [i for i, w in enumerate(windows) if not w.has_secret]

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        bytes_concat=bytes_concat.numpy(),
        offsets=np.array(offsets, dtype=np.int64),
        labels=np.array(labels, dtype=np.float32),
        lengths=np.array(lengths, dtype=np.int32),
        features=np.array(features, dtype=np.float32),
        secret_indices=np.array(secret_indices, dtype=np.int32),
        clean_indices=np.array(clean_indices, dtype=np.int32),
        secret_ids_flat=np.array(secret_ids_flat, dtype=object),
        sid_offsets=np.array(sid_offsets, dtype=np.int64),
        all_secret_ids=np.array(list(all_secret_ids), dtype=object),
    )


def _load_mixed_npz(cache_path: Path):
    npz = dict(np.load(cache_path, allow_pickle=True))

    bytes_concat = torch.from_numpy(npz["bytes_concat"])
    offsets = npz["offsets"]
    byte_tensors = [
        bytes_concat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)
    ]

    secret_ids_flat = list(npz["secret_ids_flat"])
    sid_offsets = npz["sid_offsets"]
    secret_ids_per_window = [
        secret_ids_flat[sid_offsets[i] : sid_offsets[i + 1]]
        for i in range(len(sid_offsets) - 1)
    ]

    return {
        "bytes": byte_tensors,
        "labels": npz["labels"].tolist(),
        "lengths": npz["lengths"].tolist(),
        "features": [torch.tensor([f], dtype=torch.float32) for f in npz["features"]],
        "secret_indices": npz["secret_indices"].tolist(),
        "clean_indices": npz["clean_indices"].tolist(),
        "secret_ids": secret_ids_per_window,
        "all_secret_ids": set(npz["all_secret_ids"]),
    }


def load_mixed_dataset_with_features(
    data_dir: Path, split: str, text_freq: FrequencyTable
):
    cache_path = CACHE_DIR / f"mixed_{split}_v1.npz"

    if cache_path.exists():
        print(f"  Loading cached mixed {split} from {cache_path}")
        return _load_mixed_npz(cache_path)

    print(f"  Building mixed {split} dataset (single-line + multi-line + secrets-only)...")
    windows, all_secret_ids = load_mixed_windows(data_dir, split)

    print(f"  Extracting text_freq features for {len(windows):,} windows...")
    features = [char_frequency_difference(w.bytes, text_freq) for w in windows]

    _save_mixed_npz(cache_path, windows, features, all_secret_ids)
    print(f"  Cached to {cache_path}")

    byte_tensors = [torch.tensor(list(w.bytes), dtype=torch.long) for w in windows]
    lengths = [len(w.bytes) for w in windows]
    labels = [1.0 if w.has_secret else 0.0 for w in windows]
    secret_indices = [i for i, w in enumerate(windows) if w.has_secret]
    clean_indices = [i for i, w in enumerate(windows) if not w.has_secret]
    secret_ids_per_window = [w.secret_ids if w.secret_ids else [] for w in windows]

    return {
        "bytes": byte_tensors,
        "labels": labels,
        "lengths": lengths,
        "features": [torch.tensor([f], dtype=torch.float32) for f in features],
        "secret_indices": secret_indices,
        "clean_indices": clean_indices,
        "secret_ids": secret_ids_per_window,
        "all_secret_ids": all_secret_ids,
    }


def _windows_to_data_dict(windows: list[LineSample], secret_ids_set: set[str], text_freq: FrequencyTable) -> dict:
    features = [char_frequency_difference(w.bytes, text_freq) for w in windows]
    byte_tensors = [torch.tensor(list(w.bytes), dtype=torch.long) for w in windows]
    lengths = [len(w.bytes) for w in windows]
    labels = [1.0 if w.has_secret else 0.0 for w in windows]
    secret_indices = [i for i, w in enumerate(windows) if w.has_secret]
    clean_indices = [i for i, w in enumerate(windows) if not w.has_secret]
    secret_ids_per_window = [w.secret_ids if w.secret_ids else [] for w in windows]
    categories_per_window = [w.categories if w.categories else [] for w in windows]

    return {
        "bytes": byte_tensors,
        "labels": labels,
        "lengths": lengths,
        "features": [torch.tensor([f], dtype=torch.float32) for f in features],
        "secret_indices": secret_indices,
        "clean_indices": clean_indices,
        "secret_ids": secret_ids_per_window,
        "all_secret_ids": secret_ids_set,
        "categories": categories_per_window,
    }


def load_stratified_mixed_datasets_with_features(
    data_dir: Path, text_freq: FrequencyTable, seed: int = 42
) -> tuple[dict, dict]:
    cache_path = CACHE_DIR / f"stratified_mixed_seed{seed}_v1.npz"

    if cache_path.exists():
        print(f"  Loading cached stratified mixed datasets from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        keys = ["bytes", "labels", "lengths", "features", "secret_indices", "clean_indices", "secret_ids", "all_secret_ids", "categories"]
        train_data = {}
        val_data = {}
        for k in keys:
            train_key = f"train_{k}"
            val_key = f"val_{k}"
            if train_key in data:
                train_data[k] = set(data[train_key]) if k == "all_secret_ids" else data[train_key].tolist()
            if val_key in data:
                val_data[k] = set(data[val_key]) if k == "all_secret_ids" else data[val_key].tolist()
        
        train_data["bytes"] = [torch.tensor(list(b), dtype=torch.long) for b in train_data["bytes"]]
        train_data["features"] = [torch.tensor([f], dtype=torch.float32) for f in train_data["features"]]
        val_data["bytes"] = [torch.tensor(list(b), dtype=torch.long) for b in val_data["bytes"]]
        val_data["features"] = [torch.tensor([f], dtype=torch.float32) for f in val_data["features"]]
        
        return train_data, val_data

    print("  Building stratified mixed datasets (single-line + multi-line + secrets-only)...")
    print("  This loads ALL documents and properly stratifies by category (88% train / 12% val)...")
    
    (train_windows, train_ids), (val_windows, val_ids) = load_and_split_mixed_windows(
        data_dir, seed=seed, show_progress=True
    )

    print(f"  Extracting text_freq features for {len(train_windows):,} train windows...")
    train_data = _windows_to_data_dict(train_windows, train_ids, text_freq)
    
    print(f"  Extracting text_freq features for {len(val_windows):,} val windows...")
    val_data = _windows_to_data_dict(val_windows, val_ids, text_freq)

    print(f"  Caching to {cache_path}...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    train_bytes_raw = [list(w.bytes) for w in train_windows]
    val_bytes_raw = [list(w.bytes) for w in val_windows]
    train_features_raw = [char_frequency_difference(w.bytes, text_freq) for w in train_windows]
    val_features_raw = [char_frequency_difference(w.bytes, text_freq) for w in val_windows]
    
    np.savez(
        cache_path,
        train_bytes=np.array(train_bytes_raw, dtype=object),
        train_labels=np.array(train_data["labels"]),
        train_lengths=np.array(train_data["lengths"]),
        train_features=np.array(train_features_raw),
        train_secret_indices=np.array(train_data["secret_indices"]),
        train_clean_indices=np.array(train_data["clean_indices"]),
        train_secret_ids=np.array(train_data["secret_ids"], dtype=object),
        train_all_secret_ids=np.array(list(train_ids)),
        train_categories=np.array(train_data["categories"], dtype=object),
        val_bytes=np.array(val_bytes_raw, dtype=object),
        val_labels=np.array(val_data["labels"]),
        val_lengths=np.array(val_data["lengths"]),
        val_features=np.array(val_features_raw),
        val_secret_indices=np.array(val_data["secret_indices"]),
        val_clean_indices=np.array(val_data["clean_indices"]),
        val_secret_ids=np.array(val_data["secret_ids"], dtype=object),
        val_all_secret_ids=np.array(list(val_ids)),
        val_categories=np.array(val_data["categories"], dtype=object),
    )
    print(f"  Cached stratified datasets to {cache_path}")

    return train_data, val_data


@torch.no_grad()
def validate_unique_secrets(
    model: Model,
    val_batches: list,
    all_secret_ids: set[str],
    threshold: float,
    device: torch.device,
) -> tuple[float, int, int]:
    model.eval()
    detected_secrets: set[str] = set()

    for batch in val_batches:
        batch_bytes = batch.bytes.to(device)
        mask = None
        if batch.lengths is not None:
            L = batch_bytes.shape[1]
            mask = torch.arange(L, device=device).unsqueeze(0) >= batch.lengths.to(
                device
            ).unsqueeze(1)
        features = batch.features.to(device) if batch.features is not None else None

        out = model(batch_bytes, mask=mask, precomputed=features)
        logits = out[0] if isinstance(out, tuple) else out
        probs = torch.sigmoid(logits).view(-1)
        preds = (probs >= threshold).cpu().tolist()

        for pred, sids in zip(preds, batch.secret_ids or []):
            if pred:
                detected_secrets.update(sids)

    n_detected = len(detected_secrets)
    n_total = len(all_secret_ids)
    recall = n_detected / n_total if n_total > 0 else 0.0

    return recall, n_detected, n_total


def eval_swa_model(
    trainer: Trainer,
    val_batches: list,
    all_val_secret_ids: set[str] | None,
    device: torch.device,
    mixed: bool,
):
    """Evaluate the SWA model by temporarily loading its weights, returns (recall, threshold, metrics)."""
    from torch.optim.swa_utils import update_bn

    if not trainer.swa_active or trainer.swa_model is None:
        return None, None, None

    original_state = {k: v.cpu().clone() for k, v in trainer.model.state_dict().items()}

    def bn_loader():
        for batch in trainer.train_data:
            yield batch.bytes.to(device)

    update_bn(bn_loader(), trainer.swa_model, device=device)
    trainer.model.load_state_dict(trainer.swa_model.module.state_dict())

    val_metrics = trainer.validate(optimize_threshold=True)
    threshold = trainer.optimal_threshold

    if mixed and all_val_secret_ids is not None:
        unique_recall, n_det, n_tot = validate_unique_secrets(
            trainer.model, val_batches, all_val_secret_ids, threshold, device
        )
        recall = unique_recall
    else:
        recall = val_metrics.recall
        n_det, n_tot = None, None

    trainer.model.load_state_dict(original_state)

    return recall, threshold, val_metrics


def cosine_schedule(
    epoch: int, total_epochs: int, start_val: float, end_val: float
) -> float:
    if total_epochs <= 1:
        return end_val
    progress = epoch / (total_epochs - 1)
    return end_val + 0.5 * (start_val - end_val) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/line_filter.pt"))
    parser.add_argument("--no-swa", action="store_true")
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument(
        "--pos-weight-factor",
        type=float,
        default=1.0,
        help="Multiplier for auto-calculated pos_weight",
    )
    parser.add_argument(
        "--pos-weight-decay",
        action="store_true",
        help="Cosine decay pos_weight from factor*base to base",
    )

    parser.add_argument(
        "--clean-ratio",
        type=float,
        default=1.0,
        help="Ratio of clean samples to secret samples per epoch",
    )
    parser.add_argument(
        "--hard-ratio",
        type=float,
        default=0.3,
        help="Curriculum learning hard example ratio (0 to disable)",
    )
    parser.add_argument(
        "--curriculum-start",
        type=int,
        default=2,
        help="Epoch to start curriculum learning",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Train on mixed single-line + multi-line + secrets-only windows",
    )
    parser.add_argument(
        "--quick-eval",
        action="store_true",
        help="Subsample val set like training (all secrets + equal clean)",
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="Resume from checkpoint path"
    )
    parser.add_argument(
        "--ssm",
        action="store_true",
        help="Use SSM3 (Mamba-3 style) instead of attention",
    )
    parser.add_argument(
        "--ssm-with-conv",
        action="store_true",
        help="Add DeformConv+SE to SSM3 (implies --ssm)",
    )
    parser.add_argument(
        "--unified",
        action="store_true",
        help="Use unified architecture (attn -> SSM with residuals)",
    )
    parser.add_argument(
        "--ssm-kernels",
        type=str,
        default="7",
        help="SSM conv kernel sizes: 0=no conv, 7=single, 3,5,7,9=multi-kernel",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive conv with KernelNet, deformation, SE (implies --unified)",
    )
    parser.add_argument(
        "--n-branches",
        type=int,
        default=3,
        help="Number of adaptive SSM branches (default: 3)",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=1,
        help="Number of unified layers to stack (default: 1)",
    )
    parser.add_argument(
        "--depthwise",
        action="store_true",
        help="Use depthwise conv (each channel independent) instead of grouped",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, mps, cpu (default: auto)",
    )
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument(
        "--profile-slow",
        action="store_true",
        help="Profile batches that take >3x median time",
    )
    args = parser.parse_args()

    if args.adaptive:
        args.unified = True
    if args.ssm_with_conv:
        args.ssm = True

    setup_cuda()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    if args.unified:
        ssm_kernel_sizes = tuple(int(k) for k in args.ssm_kernels.split(","))
        layer_config = LayerConfig(
            embed_width=12,
            conv_groups=2,
            attn_heads=2,
            attn_ffn_mult=2,
            num_attn_features=16,
            ssm_state_size=24,
            ssm_n_heads=4,
            ssm_expand=2,
            ssm_kernel_sizes=ssm_kernel_sizes,
            num_ssm_features=16,
            num_embed_features=16,
            num_hidden_features=16,
            num_heuristic_features=16,
            mlp_hidden_mult=16,
            mlp_output_dim=128,
            dropout=0.1,
            adaptive_conv=args.adaptive,
            n_adaptive_branches=args.n_branches,
            depthwise_conv=args.depthwise,
            use_attention=False,
        )
        head_config = HeadConfig(head_type="classifier", n_classes=2)
        model_config = ModelConfig(
            n_layers=args.n_layers, layer=layer_config, head=head_config
        )
        if args.adaptive:
            dw_str = " DEPTHWISE" if args.depthwise else ""
            print(
                f"\nUsing unified architecture with{dw_str} ADAPTIVE conv ({args.n_branches} branches)"
            )
        else:
            print(
                f"\nUsing unified architecture (attn -> SSM), ssm_kernels={ssm_kernel_sizes}"
            )
    elif args.ssm:
        arch_type = "conv_ssm"
        ssm_conv_type = "deform_se" if args.ssm_with_conv else "none"
        model_config = ModelConfig(
            task="binary",
            arch_type=arch_type,
            embed_width=48,
            conv_kernel_sizes=(3, 5, 7, 9),
            conv_groups=3,
            ssm_depth=1,
            ssm_state_size=64,
            ssm_n_heads=4,
            ssm_expand=2,
            ssm_conv_type=ssm_conv_type,
            num_embed_features=8,
            mlp_hidden_mult=4,
            mlp_output_dim=32,
            num_precomputed_features=1,
        )
        print(f"\nUsing SSM3 (Mamba-3 style), conv_type={ssm_conv_type}")
    else:
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
            num_precomputed_features=1,
        )

    print("\nLoading frequency table...")
    text_freq = load_or_build_text_freq_table(args.data_dir)

    print(f"\nLoading datasets with features...")
    if args.mixed:
        print("  Mode: MIXED (single-line + multi-line + secrets-only) with STRATIFIED splitting")
        train_data, val_data = load_stratified_mixed_datasets_with_features(
            args.data_dir, text_freq, seed=args.seed
        )
        val_secret_ids = val_data["secret_ids"]
        all_val_secret_ids = val_data["all_secret_ids"]
        all_train_secret_ids = train_data["all_secret_ids"]
        
        train_val_overlap = all_train_secret_ids & all_val_secret_ids
        if train_val_overlap:
            raise RuntimeError(f"SECRET LEAKAGE: {len(train_val_overlap)} secrets in both train and val!")
        print(f"  Train unique secrets: {len(all_train_secret_ids):,}")
        print(f"  Val unique secrets: {len(all_val_secret_ids):,}")
        print(f"  No secret leakage between train/val (verified)")
    else:
        train_data = load_dataset_with_features(args.data_dir, "train", text_freq)
        val_data = load_dataset_with_features(args.data_dir, "val", text_freq)
        val_secret_ids = None
        all_val_secret_ids = None

    n_train_secrets = len(train_data["secret_indices"])
    n_train_clean = len(train_data["clean_indices"])
    n_val_secrets = len(val_data["secret_indices"])
    print(f"Train: {n_train_secrets:,} secrets, {n_train_clean:,} clean")
    print(
        f"Val: {len(val_data['labels']):,} samples ({n_val_secrets:,} secret windows)"
    )

    n_clean_per_epoch = min(int(n_train_secrets * args.clean_ratio), n_train_clean)
    print(
        f"Per epoch: {n_train_secrets:,} secrets (all) + {n_clean_per_epoch:,} clean ({args.clean_ratio:.1f}x, resampled)"
    )

    base_pos_weight = n_clean_per_epoch / n_train_secrets
    pos_weight_start = base_pos_weight * args.pos_weight_factor
    pos_weight_end = base_pos_weight
    print(f"Class balance: base_pos_weight={base_pos_weight:.2f}")
    if args.pos_weight_decay:
        print(
            f"  pos_weight decay: {pos_weight_start:.2f} -> {pos_weight_end:.2f} (cosine)"
        )
    else:
        print(f"  pos_weight: {pos_weight_start:.2f} (fixed)")
    if args.hard_ratio > 0:
        print(
            f"Curriculum: hard_ratio={args.hard_ratio}, start_epoch={args.curriculum_start}"
        )
    else:
        print("Curriculum: disabled")

    def preload_batches(batches: list, target_device: torch.device) -> list:
        return [
            type(b)(
                bytes=b.bytes.to(target_device),
                labels=b.labels.to(target_device),
                lengths=b.lengths.to(target_device) if b.lengths is not None else None,
                features=b.features.to(target_device) if b.features is not None else None,
                secret_ids=b.secret_ids,
            )
            for b in batches
        ]

    if args.quick_eval:
        rng = random.Random(args.seed)
        categories = val_data.get("categories", [])
        
        if categories:
            cat_to_indices: dict[str, list[int]] = {}
            for idx in val_data["secret_indices"]:
                cats = categories[idx] if idx < len(categories) else []
                cat = cats[0] if cats else "unknown"
                cat_to_indices.setdefault(cat, []).append(idx)
            
            target_per_cat = max(1, n_val_secrets // (len(cat_to_indices) * 2))
            secret_sample = []
            cat_counts = {}
            for cat, indices in cat_to_indices.items():
                n_sample = min(len(indices), target_per_cat)
                cat_counts[cat] = n_sample
                secret_sample.extend(rng.sample(indices, n_sample))
            
            print(f"Quick eval stratified by category: {cat_counts}")
        else:
            secret_sample = val_data["secret_indices"]
        
        n_val_clean = len(val_data["clean_indices"])
        n_val_clean_sample = min(len(secret_sample), n_val_clean)
        val_clean_sample = rng.sample(val_data["clean_indices"], n_val_clean_sample)
        val_indices = secret_sample + val_clean_sample
        print(
            f"Quick eval: {len(secret_sample):,} secrets + {n_val_clean_sample:,} clean = {len(val_indices):,} samples"
        )

        val_batches = create_binary_batches(
            val_data["bytes"],
            val_data["labels"],
            val_data["lengths"],
            batch_size=args.eval_batch_size,
            feature_tensors=val_data["features"],
            secret_ids=val_secret_ids,
            indices=val_indices,
            shuffle=False,
        )
    else:
        val_batches = create_binary_batches(
            val_data["bytes"],
            val_data["labels"],
            val_data["lengths"],
            batch_size=args.eval_batch_size,
            feature_tensors=val_data["features"],
            secret_ids=val_secret_ids,
            shuffle=False,
        )
    
    print(f"Val batches: {len(val_batches):,}")
    print(f"  Preloading val batches to {device}...")
    val_batches = preload_batches(val_batches, device)

    def sample_train_batches(epoch_seed: int, to_device: torch.device | None = None) -> list:
        rng = random.Random(epoch_seed)
        secret_idx = train_data["secret_indices"]
        clean_idx = rng.sample(train_data["clean_indices"], n_clean_per_epoch)
        indices = secret_idx + clean_idx
        batches = create_binary_batches(
            train_data["bytes"],
            train_data["labels"],
            train_data["lengths"],
            batch_size=args.batch_size,
            feature_tensors=train_data["features"],
            indices=indices,
            seed=epoch_seed,
            show_progress=False,
        )
        if to_device is not None:
            batches = preload_batches(batches, to_device)
        return batches

    print(f"  Preloading initial train batches to {device}...")
    initial_batches = sample_train_batches(args.seed, to_device=device)

    model = Model(model_config)
    if args.compile:
        if device.type == "mps":
            print("\nSkipping torch.compile on MPS (Inductor backend not supported)")
        else:
            # Allow .item() calls to be captured in graph (for adaptive kernel size)
            torch._dynamo.config.capture_scalar_outputs = True
            model = torch.compile(model)
            print("\nUsing torch.compile (with capture_scalar_outputs=True)")
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(model_config)

    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        pos_weight=pos_weight_start,
        threshold_search=False,
        use_swa=not args.no_swa,
        preload_to_device=True,
        hard_example_ratio=args.hard_ratio,
        curriculum_start_epoch=args.curriculum_start,
    )

    trainer = Trainer(
        model, initial_batches, val_batches, config=train_config, device=device
    )

    start_epoch = 0
    best_recall = 0.0
    best_epoch = 0
    best_state = None
    best_threshold = 0.5

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
        best_recall = ckpt.get("best_recall", 0.0)
        best_epoch = ckpt.get("best_epoch", 0)
        best_state = ckpt.get("best_state")
        best_threshold = ckpt.get("best_threshold", 0.5)
        trainer.optimal_threshold = ckpt.get("optimal_threshold", 0.5)
        if ckpt.get("swa_active", False) and trainer.swa_model is not None:
            trainer.swa_active = True
            if "swa_state" in ckpt:
                trainer.swa_model.load_state_dict(ckpt["swa_state"])
        print(f"  Resuming from epoch {start_epoch}, best_recall={best_recall:.4f}")

    print(f"\nTraining for epochs {start_epoch + 1}-{args.epochs}...")

    for epoch in range(start_epoch, args.epochs):
        trainer.train_data = sample_train_batches(args.seed + epoch, to_device=device)
        trainer._batch_losses = [0.0] * len(trainer.train_data)
        if args.pos_weight_decay:
            pw = cosine_schedule(epoch, args.epochs, pos_weight_start, pos_weight_end)
            trainer.config.pos_weight = pw
        train_metrics = trainer.train_epoch(epoch, profile_slow=args.profile_slow)
        val_metrics = trainer.validate(optimize_threshold=True)

        pw_str = f" pw={trainer.config.pos_weight:.2f}" if args.pos_weight_decay else ""

        if args.mixed and all_val_secret_ids is not None:
            unique_recall, n_det, n_tot = validate_unique_secrets(
                trainer.model,
                val_batches,
                all_val_secret_ids,
                trainer.optimal_threshold,
                device,
            )
            recall_for_best = unique_recall
            recall_str = f"uR={unique_recall:.4f}({n_det}/{n_tot})"
        else:
            recall_for_best = val_metrics.recall
            recall_str = ""

        best_str = ""
        if recall_for_best > best_recall:
            best_recall = recall_for_best
            best_epoch = epoch + 1
            best_state = {
                k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
            }
            best_threshold = trainer.optimal_threshold
            best_str = " *BEST*"

        extra_str = f" {recall_str}" if recall_str else ""
        print(
            f"Epoch {epoch + 1:2d}:{pw_str} train {train_metrics} | val {val_metrics}{extra_str}{best_str}"
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
                "best_recall": best_recall,
                "best_epoch": best_epoch,
                "best_state": best_state,
                "best_threshold": best_threshold,
                "swa_active": trainer.swa_active,
                "swa_state": trainer.swa_model.state_dict()
                if trainer.swa_model
                else None,
            },
        )
        print(f"  -> {epoch_path}")

        timing_stats = trainer.get_batch_time_stats()
        if timing_stats:
            mpt_str = ""
            if "ms_per_token_mean" in timing_stats:
                mpt_str = (
                    f" | ms/tok: {timing_stats['ms_per_token_mean']:.3f}"
                    f"±{timing_stats['ms_per_token_std']:.3f}"
                )
            print(
                f"         Timing: mean={timing_stats['mean']:.3f}s "
                f"median={timing_stats['median']:.3f}s "
                f"p99={timing_stats['p99']:.3f}s "
                f"max={timing_stats['max']:.3f}s "
                f"slow={timing_stats['slow_batches']}{mpt_str}"
            )

        swa_recall, swa_threshold, swa_metrics = eval_swa_model(
            trainer, val_batches, all_val_secret_ids, device, args.mixed
        )
        if swa_recall is not None and trainer.swa_model is not None:
            swa_str = f"SWA: {swa_metrics} "
            if args.mixed:
                swa_str += f"uR={swa_recall:.4f}"
            else:
                swa_str += f"R={swa_recall:.4f}"
            print(f"         {swa_str}")

            swa_path = (
                args.output.parent
                / f"{args.output.stem}_swa_epoch{epoch + 1:02d}{args.output.suffix}"
            )
            torch.save(
                {
                    "model_config": trainer.model.config.to_dict(),
                    "state_dict": trainer.swa_model.module.state_dict(),
                    "optimal_threshold": swa_threshold,
                    "epoch": epoch,
                    "swa_recall": swa_recall,
                },
                swa_path,
            )
            print(f"         -> {swa_path}")

    if not args.no_swa:
        trainer.finalize_swa()
        final = trainer.validate(optimize_threshold=True)

        if args.mixed and all_val_secret_ids is not None:
            swa_unique_recall, n_det, n_tot = validate_unique_secrets(
                trainer.model,
                val_batches,
                all_val_secret_ids,
                trainer.optimal_threshold,
                device,
            )
            print(f"\nPost-SWA: {final} uR={swa_unique_recall:.4f}({n_det}/{n_tot})")
            swa_recall = swa_unique_recall
        else:
            print(f"\nPost-SWA: {final}")
            swa_recall = final.recall

        if swa_recall > best_recall:
            best_recall = swa_recall
            best_epoch = -1
            best_state = {
                k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
            }
            best_threshold = trainer.optimal_threshold
            print("  SWA model is new best!")

    if best_state is not None:
        trainer.model.load_state_dict(best_state)
        trainer.optimal_threshold = best_threshold or 0.5

    final = trainer.validate(optimize_threshold=True)
    best_src = f"epoch {best_epoch}" if best_epoch > 0 else "initial"

    if args.mixed and all_val_secret_ids is not None:
        final_unique_recall, n_det, n_tot = validate_unique_secrets(
            trainer.model,
            val_batches,
            all_val_secret_ids,
            trainer.optimal_threshold,
            device,
        )
        print(
            f"\nBest model ({best_src}): {final} uR={final_unique_recall:.4f}({n_det}/{n_tot})"
        )
    else:
        print(f"\nBest model ({best_src}): {final}")
    print(f"Threshold: {trainer.optimal_threshold:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
