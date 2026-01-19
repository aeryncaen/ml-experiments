#!/usr/bin/env python3
"""Architecture search for ByteMasker model.

Tests combinations of:
- width: [32, 48, 64, 96]
- depth: [2, 3, 4]
- kernel_size: [5, 7, 9]
- groups: [2, 4, 8]
"""

import argparse
import random
from pathlib import Path
from itertools import product

import torch
from tqdm import tqdm

from heuristic_secrets.bytemasker.model import ByteMaskerConfig, ByteMaskerModel
from heuristic_secrets.bytemasker.dataset import load_bytemasker_dataset, create_bucketed_batches


WIDTHS = [32, 48, 64, 96]
DEPTHS = [2, 3, 4]
KERNEL_SIZES = [5, 7, 9]
GROUPS = [2, 4, 8]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def train_and_evaluate(
    config: ByteMaskerConfig,
    train_batches: list,
    val_batches: list,
    epochs: int,
    lr: float,
    device: torch.device,
    pos_weight: float = 5.0,
    pbar_position: int = 1,
) -> dict:
    model = ByteMaskerModel(config).to(device)
    params = count_params(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    total_steps = epochs * len(train_batches)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps, pct_start=0.2, anneal_strategy='cos'
    )
    
    for epoch in range(epochs):
        model.train()
        batch_order = list(range(len(train_batches)))
        random.shuffle(batch_order)
        
        epoch_loss = 0.0
        pbar = tqdm(batch_order, desc=f"  Epoch {epoch+1}/{epochs}", leave=False, position=pbar_position)
        for batch_idx in pbar:
            bytes_batch, masks_batch, lengths = train_batches[batch_idx]
            bytes_batch = bytes_batch.to(device)
            masks_batch = masks_batch.to(device)
            lengths = lengths.to(device)
            
            optimizer.zero_grad()
            logits = model(bytes_batch)
            
            batch_size, max_len = logits.shape
            position_mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, masks_batch, reduction='none')
            if pos_weight != 1.0:
                weight = torch.where(masks_batch == 1, pos_weight, 1.0)
                bce = bce * weight
            loss = (bce * position_mask).sum() / position_mask.sum()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    model.eval()
    total_tp, total_fp, total_fn = 0, 0, 0
    total_loss = 0.0
    
    with torch.no_grad():
        for bytes_batch, masks_batch, lengths in val_batches:
            bytes_batch = bytes_batch.to(device)
            masks_batch = masks_batch.to(device)
            lengths = lengths.to(device)
            
            logits = model(bytes_batch)
            
            batch_size, max_len = logits.shape
            position_mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
            
            bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, masks_batch, reduction='none')
            loss = (bce * position_mask).sum() / position_mask.sum()
            total_loss += loss.item()
            
            pred_binary = (torch.sigmoid(logits) >= 0.5).float()
            valid_preds = pred_binary[position_mask]
            valid_targets = masks_batch[position_mask]
            
            total_tp += ((valid_preds == 1) & (valid_targets == 1)).sum().item()
            total_fp += ((valid_preds == 1) & (valid_targets == 0)).sum().item()
            total_fn += ((valid_preds == 0) & (valid_targets == 1)).sum().item()
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "params": params,
        "loss": total_loss / len(val_batches),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def main():
    parser = argparse.ArgumentParser(description="ByteMasker architecture search")
    parser.add_argument("--data-dir", "-d", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pos-weight", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", "-o", type=Path, help="Save results to CSV")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    args = parser.parse_args()
    
    device = torch.device("cpu") if args.cpu else get_device()
    print(f"Device: {device}")
    
    print("Loading datasets...")
    train_dataset, train_stats = load_bytemasker_dataset(args.data_dir, "train", secrets_only=True)
    val_dataset, val_stats = load_bytemasker_dataset(args.data_dir, "val", secrets_only=True)
    
    print(f"Train: {train_stats}")
    print(f"Val: {val_stats}")
    
    train_batches = create_bucketed_batches(
        train_dataset, args.batch_size, indices=train_dataset.secret_lines, shuffle=False
    )
    val_batches = create_bucketed_batches(
        val_dataset, args.batch_size, indices=val_dataset.secret_lines, shuffle=False
    )
    
    print(f"Train batches: {len(train_batches)}, Val batches: {len(val_batches)}")
    
    experiments = []
    for width, depth, kernel_size, groups in product(WIDTHS, DEPTHS, KERNEL_SIZES, GROUPS):
        if width % groups != 0:
            continue
        experiments.append((width, depth, kernel_size, groups))
    
    print(f"\nRunning {len(experiments)} experiments ({args.epochs} epochs each)")
    print(f"Search space: width={WIDTHS}, depth={DEPTHS}, kernel={KERNEL_SIZES}, groups={GROUPS}")
    print("=" * 110)
    
    results = []
    
    for width, depth, kernel_size, groups in tqdm(experiments, desc="Experiments"):
        set_seed(args.seed)
        
        config = ByteMaskerConfig(
            width=width,
            depth=depth,
            kernel_size=kernel_size,
            groups=groups,
            dropout=0.1,
        )
        
        tqdm.write(f"w={width:<3} d={depth} k={kernel_size} g={groups:<2}", end=" ")
        
        metrics = train_and_evaluate(
            config=config,
            train_batches=train_batches,
            val_batches=val_batches,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            pos_weight=args.pos_weight,
        )
        
        results.append({
            "width": width,
            "depth": depth,
            "kernel_size": kernel_size,
            "groups": groups,
            **metrics,
        })
        
        tqdm.write(f"params={metrics['params']:<7} F1={metrics['f1']:.4f} P={metrics['precision']:.4f} R={metrics['recall']:.4f}")
    
    results.sort(key=lambda x: x["recall"], reverse=True)
    
    print("\n" + "=" * 110)
    print("TOP 10 BY RECALL")
    print("=" * 110)
    print(f"{'Rank':<5} {'Width':<6} {'Depth':<6} {'Kernel':<7} {'Groups':<7} {'Params':<8} {'F1':<8} {'Prec':<8} {'Recall':<8}")
    print("-" * 110)
    
    for i, r in enumerate(results[:10]):
        print(f"{i+1:<5} {r['width']:<6} {r['depth']:<6} {r['kernel_size']:<7} {r['groups']:<7} {r['params']:<8} {r['f1']:<8.4f} {r['precision']:<8.4f} {r['recall']:<8.4f}")
    
    print("\n" + "=" * 110)
    print("ANALYSIS")
    print("=" * 110)
    
    print("\nBest recall per width:")
    for width in WIDTHS:
        subset = [r for r in results if r["width"] == width]
        if subset:
            best = max(subset, key=lambda x: x["recall"])
            print(f"  width={width:<3}: R={best['recall']:.4f} F1={best['f1']:.4f} (d={best['depth']}, k={best['kernel_size']}, g={best['groups']}, params={best['params']})")
    
    print("\nBest recall per depth:")
    for depth in DEPTHS:
        subset = [r for r in results if r["depth"] == depth]
        if subset:
            best = max(subset, key=lambda x: x["recall"])
            print(f"  depth={depth}: R={best['recall']:.4f} F1={best['f1']:.4f} (w={best['width']}, k={best['kernel_size']}, g={best['groups']}, params={best['params']})")
    
    print("\nBest recall per kernel_size:")
    for kernel_size in KERNEL_SIZES:
        subset = [r for r in results if r["kernel_size"] == kernel_size]
        if subset:
            best = max(subset, key=lambda x: x["recall"])
            print(f"  kernel={kernel_size}: R={best['recall']:.4f} F1={best['f1']:.4f} (w={best['width']}, d={best['depth']}, g={best['groups']}, params={best['params']})")
    
    print("\nPareto frontier (best recall per param tier):")
    param_tiers = [(0, 10000), (10000, 25000), (25000, 50000), (50000, 100000), (100000, 200000), (200000, 500000)]
    for lo, hi in param_tiers:
        subset = [r for r in results if lo <= r["params"] < hi]
        if subset:
            best = max(subset, key=lambda x: x["recall"])
            print(f"  {lo:>6}-{hi:<6} params: R={best['recall']:.4f} F1={best['f1']:.4f} (w={best['width']}, d={best['depth']}, k={best['kernel_size']}, g={best['groups']}, params={best['params']})")
    
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            fieldnames = ["width", "depth", "kernel_size", "groups", "params", "f1", "precision", "recall", "loss", "tp", "fp", "fn"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
