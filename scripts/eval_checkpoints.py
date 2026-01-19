#!/usr/bin/env python3
"""Evaluate multiple checkpoints in parallel using vmap."""
import argparse
from pathlib import Path
from typing import NamedTuple

import torch
from torch.func import functional_call, stack_module_state
from tqdm import tqdm

from heuristic_secrets.models import Model, ModelConfig, get_device, create_binary_batches
from heuristic_secrets.bytemasker.dataset import load_bytemasker_dataset, load_mixed_windows
from heuristic_secrets.validator.features import char_frequency_difference, FrequencyTable


CACHE_DIR = Path(".cache/line_filter")


class EvalResult(NamedTuple):
    checkpoint: str
    threshold: float
    precision: float
    recall: float
    f1: float
    unique_recall: float | None
    unique_detected: int | None
    unique_total: int | None


def load_freq_table(data_dir: Path) -> FrequencyTable:
    cache = CACHE_DIR / "freq_tables.pt"
    if cache.exists():
        data = torch.load(cache, weights_only=False)
        return FrequencyTable.from_dict(data["text"])
    raise FileNotFoundError(f"Frequency table not found at {cache}. Run training first.")


def load_val_data(data_dir: Path, text_freq: FrequencyTable, mixed: bool):
    if mixed:
        windows, all_secret_ids = load_mixed_windows(data_dir, "val")
        byte_tensors = [torch.tensor(list(w.bytes), dtype=torch.long) for w in windows]
        lengths = [len(w.bytes) for w in windows]
        labels = [1.0 if w.has_secret else 0.0 for w in windows]
        features = [torch.tensor([char_frequency_difference(w.bytes, text_freq)], dtype=torch.float32) for w in windows]
        secret_ids = [w.secret_ids if w.secret_ids else [] for w in windows]
        secret_indices = [i for i, w in enumerate(windows) if w.has_secret]
        clean_indices = [i for i, w in enumerate(windows) if not w.has_secret]
        return {
            "bytes": byte_tensors,
            "labels": labels,
            "lengths": lengths,
            "features": features,
            "secret_ids": secret_ids,
            "secret_indices": secret_indices,
            "clean_indices": clean_indices,
            "all_secret_ids": all_secret_ids,
        }
    else:
        dataset, _ = load_bytemasker_dataset(data_dir, "val")
        features = [torch.tensor([char_frequency_difference(dataset.lines[i].bytes, text_freq)], dtype=torch.float32) 
                    for i in range(len(dataset))]
        labels = [1.0 if dataset.lines[i].has_secret else 0.0 for i in range(len(dataset))]
        secret_ids = [dataset.lines[i].secret_ids if dataset.lines[i].secret_ids else [] for i in range(len(dataset))]
        all_secret_ids: set[str] = set()
        for sids in secret_ids:
            all_secret_ids.update(sids)
        return {
            "bytes": dataset.byte_tensors,
            "labels": labels,
            "lengths": dataset.lengths,
            "features": features,
            "secret_ids": secret_ids,
            "secret_indices": dataset.secret_lines,
            "clean_indices": dataset.clean_lines,
            "all_secret_ids": all_secret_ids,
        }


def load_checkpoints(checkpoint_paths: list[Path], device: torch.device) -> tuple[Model, list[Model], list[float]]:
    """Load multiple checkpoints, return base model, list of models, and thresholds."""
    first = torch.load(checkpoint_paths[0], map_location=device, weights_only=False)
    model_config = ModelConfig.from_dict(first["model_config"])
    base_model = Model(model_config)
    
    models = []
    thresholds = []
    for path in checkpoint_paths:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = Model(model_config)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
        thresholds.append(ckpt.get("optimal_threshold", 0.5))
    
    return base_model, models, thresholds


def eval_single_model(
    model: Model,
    val_batches: list,
    threshold: float,
    device: torch.device,
    all_secret_ids: set[str] | None = None,
) -> EvalResult:
    """Evaluate a single model on validation data."""
    model.eval()
    model.to(device)
    
    all_preds = []
    all_labels = []
    detected_secrets: set[str] = set()
    
    with torch.no_grad():
        for batch in val_batches:
            batch_bytes = batch.bytes.to(device)
            mask = None
            if batch.lengths is not None:
                L = batch_bytes.shape[1]
                mask = torch.arange(L, device=device).unsqueeze(0) >= batch.lengths.to(device).unsqueeze(1)
            features = batch.features.to(device) if batch.features is not None else None
            
            logits = model(batch_bytes, mask=mask, precomputed=features)
            probs = torch.sigmoid(logits).view(-1)
            preds = (probs >= threshold).cpu()
            
            all_preds.append(preds)
            all_labels.append(batch.labels)
            
            if batch.secret_ids is not None:
                for pred, sids in zip(preds.tolist(), batch.secret_ids):
                    if pred:
                        detected_secrets.update(sids)
    
    preds = torch.cat(all_preds).float()
    labels = torch.cat(all_labels).float()
    
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    unique_recall = None
    unique_detected = None
    unique_total = None
    if all_secret_ids is not None:
        unique_detected = len(detected_secrets)
        unique_total = len(all_secret_ids)
        unique_recall = unique_detected / unique_total if unique_total > 0 else 0.0
    
    return EvalResult(
        checkpoint="",
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        unique_recall=unique_recall,
        unique_detected=unique_detected,
        unique_total=unique_total,
    )


@torch.no_grad()
def eval_batch_vmap(
    base_model: Model,
    models: list[Model],
    val_batches: list,
    thresholds: list[float],
    device: torch.device,
    all_secret_ids: set[str] | None = None,
) -> list[EvalResult]:
    n_models = len(models)
    
    params, buffers = stack_module_state(models)
    
    for k in params:
        params[k] = params[k].to(device)
    for k in buffers:
        buffers[k] = buffers[k].to(device)
    
    base_model.to(device)
    base_model.eval()
    
    def single_forward(params, buffers, x, mask, features):
        return functional_call(base_model, (params, buffers), (x,), {"mask": mask, "precomputed": features})
    
    batched_forward = torch.vmap(single_forward, in_dims=(0, 0, None, None, None))
    
    all_probs = [[] for _ in range(n_models)]
    all_labels = []
    batch_secret_ids_list = []
    
    for batch in tqdm(val_batches, desc="Evaluating", leave=False):
        batch_bytes = batch.bytes.to(device)
        mask = None
        if batch.lengths is not None:
            L = batch_bytes.shape[1]
            mask = torch.arange(L, device=device).unsqueeze(0) >= batch.lengths.to(device).unsqueeze(1)
        features = batch.features.to(device) if batch.features is not None else None
        
        logits = batched_forward(params, buffers, batch_bytes, mask, features)
        probs = torch.sigmoid(logits)
        
        for i in range(n_models):
            all_probs[i].append(probs[i].view(-1).cpu())
        
        all_labels.append(batch.labels)
        if batch.secret_ids is not None:
            batch_secret_ids_list.append(batch.secret_ids)
    
    labels = torch.cat(all_labels).float()
    
    results = []
    for i in range(n_models):
        probs = torch.cat(all_probs[i])
        threshold = thresholds[i]
        preds = (probs >= threshold).float()
        
        tp = ((preds == 1) & (labels == 1)).sum().item()
        fp = ((preds == 1) & (labels == 0)).sum().item()
        fn = ((preds == 0) & (labels == 1)).sum().item()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        unique_recall = None
        unique_detected = None
        unique_total = None
        
        if all_secret_ids is not None and batch_secret_ids_list:
            detected = set()
            idx = 0
            for batch_sids in batch_secret_ids_list:
                for sids in batch_sids:
                    if probs[idx] >= threshold:
                        detected.update(sids)
                    idx += 1
            unique_detected = len(detected)
            unique_total = len(all_secret_ids)
            unique_recall = unique_detected / unique_total if unique_total > 0 else 0.0
        
        results.append(EvalResult(
            checkpoint="",
            threshold=threshold,
            precision=precision,
            recall=recall,
            f1=f1,
            unique_recall=unique_recall,
            unique_detected=unique_detected,
            unique_total=unique_total,
        ))
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on full validation set")
    parser.add_argument("checkpoints", type=Path, nargs="+", help="Checkpoint files or glob pattern")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--parallel", type=int, default=5, help="Number of models to evaluate in parallel")
    parser.add_argument("--mixed", action="store_true", help="Use mixed single+multi-line windows")
    parser.add_argument("--no-vmap", action="store_true", help="Disable vmap, evaluate sequentially")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Device: {device}")
    
    checkpoint_paths = []
    for p in args.checkpoints:
        if p.is_file():
            checkpoint_paths.append(p)
        elif p.parent.is_dir():
            checkpoint_paths.extend(sorted(p.parent.glob(p.name)))
    
    if not checkpoint_paths:
        print("No checkpoints found")
        return
    
    checkpoint_paths = sorted(checkpoint_paths)
    print(f"Found {len(checkpoint_paths)} checkpoints")
    
    print("\nLoading validation data...")
    text_freq = load_freq_table(args.data_dir)
    val_data = load_val_data(args.data_dir, text_freq, args.mixed)
    
    val_batches = create_binary_batches(
        val_data["bytes"], val_data["labels"], val_data["lengths"],
        batch_size=args.batch_size,
        feature_tensors=val_data["features"],
        secret_ids=val_data["secret_ids"],
        shuffle=False,
    )
    print(f"Val batches: {len(val_batches)}")
    
    all_secret_ids = val_data.get("all_secret_ids")
    
    results: list[tuple[Path, EvalResult]] = []
    
    if args.no_vmap:
        print("\nEvaluating sequentially...")
        for path in tqdm(checkpoint_paths, desc="Checkpoints"):
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model_config = ModelConfig.from_dict(ckpt["model_config"])
            model = Model(model_config)
            model.load_state_dict(ckpt["state_dict"])
            threshold = ckpt.get("optimal_threshold", 0.5)
            
            result = eval_single_model(model, val_batches, threshold, device, all_secret_ids)
            results.append((path, result))
    else:
        print(f"\nEvaluating in parallel (batch size: {args.parallel})...")
        for i in tqdm(range(0, len(checkpoint_paths), args.parallel), desc="Batches"):
            batch_paths = checkpoint_paths[i:i + args.parallel]
            base_model, models, thresholds = load_checkpoints(batch_paths, device)
            
            batch_results = eval_batch_vmap(
                base_model, models, val_batches, thresholds, device, all_secret_ids
            )
            
            for path, result in zip(batch_paths, batch_results):
                results.append((path, result))
    
    print("\n" + "=" * 80)
    print("Results:")
    print("=" * 80)
    
    has_unique = all_secret_ids is not None and len(all_secret_ids) > 0
    header = f"{'Checkpoint':<40} {'Thresh':>6} {'Prec':>7} {'wRecall':>7} {'F1':>7}"
    if has_unique:
        header += f" {'uRecall':>8} {'Det/Tot':>10}"
    print(header)
    print("-" * len(header))
    
    best_recall = 0.0
    best_path = None
    
    for path, r in results:
        name = path.name[:40]
        line = f"{name:<40} {r.threshold:>6.3f} {r.precision:>7.4f} {r.recall:>7.4f} {r.f1:>7.4f}"
        if has_unique and r.unique_recall is not None:
            line += f" {r.unique_recall:>8.4f} {r.unique_detected:>4}/{r.unique_total:<5}"
            if r.unique_recall > best_recall:
                best_recall = r.unique_recall
                best_path = path
        else:
            if r.recall > best_recall:
                best_recall = r.recall
                best_path = path
        print(line)
    
    print("-" * len(header))
    metric_name = "unique_recall" if has_unique else "window_recall"
    print(f"Best {metric_name}: {best_recall:.4f} ({best_path.name if best_path else 'N/A'})")


if __name__ == "__main__":
    main()
