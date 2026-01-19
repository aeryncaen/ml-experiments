#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.func import stack_module_state, functional_call
from tqdm import tqdm

from heuristic_secrets.models import (
    Model,
    ModelConfig,
    TrainConfig,
    create_binary_batches,
    get_device,
    set_seed,
    Metrics,
)
from heuristic_secrets.bytemasker.dataset import load_bytemasker_dataset


CACHE_DIR = Path(".cache/line_filter")


def load_or_create_subsample(data_dir: Path, split: str, n_samples: int, seed: int):
    cache_path = CACHE_DIR / f"subsample_{split}_n{n_samples}_s{seed}_nofeat.pt"
    
    if cache_path.exists():
        print(f"  Loading cached {split} subsample from {cache_path}")
        data = torch.load(cache_path, weights_only=False)
        return data["bytes"], data["labels"], data["lengths"]
    
    print(f"  Creating {split} subsample (n={n_samples})...")
    dataset, _ = load_bytemasker_dataset(data_dir, split)
    
    random.seed(seed)
    n_secret = min(n_samples // 2, len(dataset.secret_lines))
    n_clean = min(n_samples // 2, len(dataset.clean_lines))
    indices = random.sample(dataset.secret_lines, n_secret) + random.sample(dataset.clean_lines, n_clean)
    
    byte_tensors = [dataset.byte_tensors[i] for i in indices]
    labels = [1.0 if dataset.lines[i].has_secret else 0.0 for i in indices]
    lengths = [dataset.lengths[i] for i in indices]
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"bytes": byte_tensors, "labels": labels, "lengths": lengths}, cache_path)
    print(f"  Cached to {cache_path}")
    
    return byte_tensors, labels, lengths


class EnsembleTrainer:
    def __init__(self, models: nn.ModuleList, train_batches: list, val_batches: list, 
                 device: torch.device, config: TrainConfig):
        self.models = models
        self.n_members = len(models)
        self.device = device
        self.config = config
        self.train_batches = train_batches
        self.val_batches = val_batches
        
        self.optimizers = [
            torch.optim.AdamW(m.parameters(), lr=config.lr, weight_decay=config.weight_decay)
            for m in models
        ]
        
        self.pos_weight = torch.tensor([config.pos_weight], device=device)
        self.optimal_threshold = 0.5
        
    def _vmap_forward(self, x: torch.Tensor) -> torch.Tensor:
        params, buffers = stack_module_state(self.models)
        
        def call_single(params, buffers, x):
            return functional_call(self.models[0], (params, buffers), (x, None, None))
        
        return torch.vmap(call_single, in_dims=(0, 0, None), randomness='different')(params, buffers, x)
        
    def train_epoch(self, epoch: int = 0) -> Metrics:
        for m in self.models:
            m.train()
        
        cfg = self.config
        batch_indices = list(range(len(self.train_batches)))
        if cfg.seed is not None:
            random.seed(cfg.seed + epoch)
        random.shuffle(batch_indices)
        
        total_loss = 0.0
        total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
        total_samples = 0
        
        pbar = tqdm(batch_indices, desc="Training", leave=False)
        
        for batch_idx in pbar:
            batch = self.train_batches[batch_idx]
            x = batch.bytes.to(self.device)
            labels = batch.labels.to(self.device)
            
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            all_logits = []
            batch_loss = 0.0
            
            for i, (model, opt) in enumerate(zip(self.models, self.optimizers)):
                opt.zero_grad()
                logits_i = model(x, None, None)
                loss_i = loss_fn(logits_i, labels)
                loss_i.backward()
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                batch_loss += loss_i.item()
                all_logits.append(logits_i.detach())
            
            loss_val = batch_loss / self.n_members
            total_loss += loss_val
            
            with torch.no_grad():
                logits = torch.stack(all_logits)
                votes = (torch.sigmoid(logits) >= 0.5).float()
                preds = (votes.mean(dim=0) >= 0.5).float()
                tp = ((preds == 1) & (labels == 1)).sum().item()
                fp = ((preds == 1) & (labels == 0)).sum().item()
                fn = ((preds == 0) & (labels == 1)).sum().item()
                tn = ((preds == 0) & (labels == 0)).sum().item()
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn
                total_samples += len(labels)
            
            pbar.set_postfix(loss=f"{loss_val:.4f}")
        
        return self._build_metrics(total_loss, len(batch_indices), total_tp, total_fp, total_fn, total_tn, total_samples)
    
    def _build_metrics(self, total_loss: float, n_batches: int, tp: int, fp: int, fn: int, tn: int, n_samples: int) -> Metrics:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / n_samples if n_samples > 0 else 0.0
        return Metrics(
            loss=total_loss / n_batches if n_batches > 0 else 0.0,
            precision=precision,
            recall=recall,
            f1=f1,
            accuracy=accuracy,
        )
    
    @torch.no_grad()
    def validate(self, optimize_threshold: bool = False) -> Metrics:
        for m in self.models:
            m.eval()
        
        all_probs = []
        all_labels = []
        total_loss = 0.0
        
        for batch in self.val_batches:
            x = batch.bytes.to(self.device)
            labels = batch.labels.to(self.device)
            
            logits = self._vmap_forward(x)
            
            loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
            losses = torch.stack([loss_fn(logits[i], labels) for i in range(self.n_members)])
            total_loss += losses.mean().item()
            
            member_probs = torch.sigmoid(logits)
            all_probs.append(member_probs.cpu())
            all_labels.append(labels.cpu())
        
        all_probs = torch.cat(all_probs, dim=1)
        all_labels = torch.cat(all_labels)
        
        if optimize_threshold:
            best_f1 = 0.0
            best_thresh = 0.5
            for thresh in torch.linspace(0.1, 0.9, 50):
                votes = (all_probs >= thresh).float()
                preds = (votes.mean(dim=0) >= 0.5).float()
                tp = ((preds == 1) & (all_labels == 1)).sum().item()
                fp = ((preds == 1) & (all_labels == 0)).sum().item()
                fn = ((preds == 0) & (all_labels == 1)).sum().item()
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                if f1 > best_f1:
                    best_f1 = f1
                    best_thresh = thresh.item()
            self.optimal_threshold = best_thresh
        
        votes = (all_probs >= self.optimal_threshold).float()
        preds = (votes.mean(dim=0) >= 0.5).float()
        tp = ((preds == 1) & (all_labels == 1)).sum().item()
        fp = ((preds == 1) & (all_labels == 0)).sum().item()
        fn = ((preds == 0) & (all_labels == 1)).sum().item()
        tn = ((preds == 0) & (all_labels == 0)).sum().item()
        
        return self._build_metrics(total_loss, len(self.val_batches), int(tp), int(fp), int(fn), int(tn), len(all_labels))

    def save_checkpoint(self, path: Path):
        torch.save({
            "n_members": self.n_members,
            "state_dicts": [m.state_dict() for m in self.models],
            "optimal_threshold": self.optimal_threshold,
        }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/ensemble.pt"))
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--n-members", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    model_config = ModelConfig(
        task="binary",
        arch_type="both",
        embed_width=48,
        conv_kernel_sizes=(3, 5, 9, 17),
        conv_groups=3,
        attn_depth=1,
        attn_heads=4,
        num_embed_features=8,
        mlp_hidden_mult=8,
        mlp_output_dim=128,
    )

    print(f"\nLoading data...")
    train_bytes, train_labels, train_lengths = load_or_create_subsample(
        args.data_dir, "train", args.max_samples, args.seed
    )
    val_bytes, val_labels, val_lengths = load_or_create_subsample(
        args.data_dir, "val", args.max_samples // 5, args.seed
    )

    n_train_secret = sum(1 for l in train_labels if l == 1.0)
    n_val_secret = sum(1 for l in val_labels if l == 1.0)
    print(f"Train: {len(train_labels):,} samples ({n_train_secret:,} secrets)")
    print(f"Val: {len(val_labels):,} samples ({n_val_secret:,} secrets)")

    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = n_neg / n_pos
    print(f"Class balance: {int(n_neg):,}:{int(n_pos):,}, pos_weight={pos_weight:.2f}")

    train_batches = create_binary_batches(
        train_bytes, train_labels, train_lengths,
        batch_size=args.batch_size, seed=args.seed,
    )
    val_batches = create_binary_batches(
        val_bytes, val_labels, val_lengths,
        batch_size=args.batch_size, shuffle=False,
    )

    print(f"\nCreating ensemble of {args.n_members} members...")
    models = nn.ModuleList([Model(model_config) for _ in range(args.n_members)]).to(device)
    params_per_member = sum(p.numel() for p in models[0].parameters())
    print(f"Params: {params_per_member:,} per member, {params_per_member * args.n_members:,} total")
    print(model_config)

    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        pos_weight=pos_weight,
    )

    trainer = EnsembleTrainer(models, train_batches, val_batches, device, train_config)

    print(f"\nTraining for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate(optimize_threshold=(epoch == args.epochs - 1))
        print(f"Epoch {epoch+1:2d}: train {train_metrics} | val {val_metrics}")

    final = trainer.validate(optimize_threshold=True)
    print(f"\nFinal: {final}")
    print(f"Threshold: {trainer.optimal_threshold:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
