#!/usr/bin/env python3
"""
Knowledge distillation for 1D models.

Loads a teacher model from checkpoint (config embedded), trains a smaller student model
to mimic the teacher's outputs using KL divergence + optional task loss.

Example:
    python scripts/distill_1d.py \
        --teacher models/line_filter_large.pt \
        --student-width 24 \
        --student-layers 1 \
        --student-branches 2 \
        --epochs 20 \
        --output models/line_filter_small.pt
"""
import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from heuristic_secrets.models import (
    Model,
    ModelConfig,
    LayerConfig,
    HeadConfig,
    create_binary_batches,
    get_device,
    set_seed,
    setup_cuda,
)
from heuristic_secrets.bytemasker.dataset import load_mixed_windows, LineSample
from heuristic_secrets.validator.features import (
    char_frequency_difference,
    FrequencyTable,
)


CACHE_DIR = Path(".cache/line_filter")


def load_teacher(checkpoint_path: Path, device: torch.device) -> tuple[Model, dict]:
    """Load teacher model from checkpoint with embedded config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_config" not in ckpt:
        raise ValueError(f"Checkpoint {checkpoint_path} missing 'model_config'")

    config = ModelConfig.from_dict(ckpt["model_config"])
    model = Model(config)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    for param in model.parameters():
        param.requires_grad = False

    info = {
        "config": config,
        "threshold": ckpt.get("optimal_threshold", 0.5),
        "path": checkpoint_path,
    }

    return model, info


def create_student_config(args, teacher_config: ModelConfig) -> ModelConfig:
    """Create student config from CLI args, using teacher config as reference."""
    teacher_layer = teacher_config.layer

    student_layer = LayerConfig(
        embed_width=args.student_width,
        dropout=args.dropout,
        conv_groups=min(args.student_width // 8, teacher_layer.conv_groups),
        use_attention=args.student_use_attention,
        attn_heads=min(args.student_width // 8, teacher_layer.attn_heads),
        attn_ffn_mult=teacher_layer.attn_ffn_mult,
        attn_window_size=teacher_layer.attn_window_size,
        attn_use_rope=teacher_layer.attn_use_rope,
        num_attn_features=args.student_features,
        ssm_state_size=args.student_ssm_state,
        ssm_n_heads=min(args.student_width // 8, teacher_layer.ssm_n_heads),
        ssm_kernel_sizes=teacher_layer.ssm_kernel_sizes,
        ssm_expand=teacher_layer.ssm_expand,
        num_ssm_features=args.student_features,
        adaptive_conv=args.student_adaptive,
        n_adaptive_branches=args.student_branches,
        adaptive_kernel_size=teacher_layer.adaptive_kernel_size,
        adaptive_init_sigmas=teacher_layer.adaptive_init_sigmas,
        adaptive_min_sigma=teacher_layer.adaptive_min_sigma,
        adaptive_max_sigma=teacher_layer.adaptive_max_sigma,
        context_dim=min(args.student_width // 2, teacher_layer.context_dim),
        num_embed_features=args.student_features,
        mlp_hidden_mult=teacher_layer.mlp_hidden_mult,
        mlp_output_dim=args.student_mlp_dim,
        num_hidden_features=args.student_features,
        num_heuristic_features=teacher_layer.num_heuristic_features,
    )

    return ModelConfig(
        n_layers=args.student_layers,
        layer=student_layer,
        head=HeadConfig(head_type="classifier", n_classes=2),
    )


class DistillationLoss(nn.Module):
    """Combined distillation loss: KL divergence + optional task loss."""

    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.7,
        pos_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.task_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]) if pos_weight != 1.0 else None
        )

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        T = self.temperature

        student_soft = torch.sigmoid(student_logits / T)
        teacher_soft = torch.sigmoid(teacher_logits / T)

        kl_loss = F.binary_cross_entropy(
            student_soft, teacher_soft.detach(), reduction="mean"
        ) * (T * T)

        metrics = {"kl_loss": kl_loss.item()}

        if labels is not None and self.alpha < 1.0:
            if self.task_loss.pos_weight is not None:
                self.task_loss.pos_weight = self.task_loss.pos_weight.to(
                    student_logits.device
                )
            task_loss = self.task_loss(student_logits.view(-1), labels.view(-1))
            total_loss = self.alpha * kl_loss + (1 - self.alpha) * task_loss
            metrics["task_loss"] = task_loss.item()
        else:
            total_loss = kl_loss

        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics


def load_freq_table(data_dir: Path) -> FrequencyTable:
    """Load cached frequency table."""
    cache = CACHE_DIR / "freq_tables.pt"
    if not cache.exists():
        raise FileNotFoundError(
            f"Frequency table not found at {cache}. Run train_line_filter.py first."
        )
    data = torch.load(cache, weights_only=False)
    return FrequencyTable.from_dict(data["text"])


def load_dataset(data_dir: Path, split: str, text_freq: FrequencyTable) -> dict:
    """Load dataset from cache."""
    import numpy as np

    cache_path = CACHE_DIR / f"mixed_{split}_v1.npz"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Dataset cache not found at {cache_path}. Run train_line_filter.py first."
        )

    npz = dict(np.load(cache_path, allow_pickle=True))
    bytes_concat = torch.from_numpy(npz["bytes_concat"])
    offsets = npz["offsets"]
    byte_tensors = [
        bytes_concat[offsets[i] : offsets[i + 1]] for i in range(len(offsets) - 1)
    ]

    return {
        "bytes": byte_tensors,
        "labels": npz["labels"].tolist(),
        "lengths": npz["lengths"].tolist(),
        "features": [
            torch.tensor([f], dtype=torch.float32) for f in npz["features"]
        ],
        "secret_indices": npz["secret_indices"].tolist(),
        "clean_indices": npz["clean_indices"].tolist(),
    }


@torch.no_grad()
def validate(
    student: Model,
    teacher: Model,
    val_batches: list,
    device: torch.device,
    threshold: float = 0.5,
) -> dict:
    """Validate student against teacher and ground truth."""
    student.eval()

    total_kl = 0.0
    total_correct = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_samples = 0

    for batch in val_batches:
        batch_bytes = batch.bytes.to(device)
        labels = batch.labels.to(device)
        mask = None
        if batch.lengths is not None:
            L = batch_bytes.shape[1]
            mask = torch.arange(L, device=device).unsqueeze(0) >= batch.lengths.to(
                device
            ).unsqueeze(1)
        features = batch.features.to(device) if batch.features is not None else None

        student_out = student(batch_bytes, mask=mask, precomputed=features)
        student_logits = student_out[0] if isinstance(student_out, tuple) else student_out

        teacher_out = teacher(batch_bytes, mask=mask, precomputed=features)
        teacher_logits = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out

        student_soft = torch.sigmoid(student_logits / 4.0)
        teacher_soft = torch.sigmoid(teacher_logits / 4.0)
        kl = F.binary_cross_entropy(student_soft, teacher_soft, reduction="sum")
        total_kl += kl.item()

        preds = (torch.sigmoid(student_logits) >= threshold).float().view(-1)
        labels_flat = labels.view(-1)
        total_correct += (preds == labels_flat).sum().item()
        total_tp += ((preds == 1) & (labels_flat == 1)).sum().item()
        total_fp += ((preds == 1) & (labels_flat == 0)).sum().item()
        total_fn += ((preds == 0) & (labels_flat == 1)).sum().item()
        total_samples += labels_flat.numel()

    acc = total_correct / total_samples if total_samples > 0 else 0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_kl = total_kl / total_samples if total_samples > 0 else 0

    return {
        "kl": avg_kl,
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main():
    parser = argparse.ArgumentParser(description="Distill a teacher model into a smaller student")
    
    parser.add_argument("--teacher", type=Path, required=True, help="Path to teacher checkpoint")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("models/distilled.pt"))
    
    parser.add_argument("--student-width", type=int, default=24, help="Student embed width")
    parser.add_argument("--student-layers", type=int, default=1, help="Student layer count")
    parser.add_argument("--student-branches", type=int, default=2, help="Student adaptive branches")
    parser.add_argument("--student-ssm-state", type=int, default=16, help="Student SSM state size")
    parser.add_argument("--student-features", type=int, default=8, help="Student feature dimensions")
    parser.add_argument("--student-mlp-dim", type=int, default=64, help="Student MLP output dim")
    parser.add_argument("--student-adaptive", action="store_true", default=True, help="Use adaptive conv")
    parser.add_argument("--student-no-adaptive", action="store_true", help="Disable adaptive conv")
    parser.add_argument("--student-use-attention", action="store_true", help="Use attention in student")
    
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--temperature", type=float, default=4.0, help="Distillation temperature")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for KL loss (1-alpha for task loss)")
    parser.add_argument("--pos-weight", type=float, default=1.0, help="Positive class weight for task loss")
    
    parser.add_argument("--max-samples", type=int, default=100_000)
    parser.add_argument("--clean-ratio", type=float, default=1.0)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()

    if args.student_no_adaptive:
        args.student_adaptive = False

    setup_cuda()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    print(f"\nLoading teacher from {args.teacher}...")
    teacher, teacher_info = load_teacher(args.teacher, device)
    teacher_params = sum(p.numel() for p in teacher.parameters())
    print(f"  Teacher: {teacher_params:,} params")
    print(f"  Config: {teacher_info['config'].layer}")

    student_config = create_student_config(args, teacher_info["config"])
    student = Model(student_config).to(device)
    student_params = sum(p.numel() for p in student.parameters())
    print(f"\nStudent: {student_params:,} params ({student_params/teacher_params*100:.1f}% of teacher)")
    print(f"  Config: {student_config.layer}")

    print("\nLoading dataset...")
    text_freq = load_freq_table(args.data_dir)
    train_data = load_dataset(args.data_dir, "train", text_freq)
    val_data = load_dataset(args.data_dir, "val", text_freq)

    n_secrets = len(train_data["secret_indices"])
    n_clean = len(train_data["clean_indices"])
    n_clean_per_epoch = min(int(n_secrets * args.clean_ratio), n_clean)
    print(f"  Train: {n_secrets:,} secrets + {n_clean_per_epoch:,} clean/epoch")
    print(f"  Val: {len(val_data['labels']):,} samples")

    val_batches = create_binary_batches(
        val_data["bytes"],
        val_data["labels"],
        val_data["lengths"],
        batch_size=args.eval_batch_size,
        feature_tensors=val_data["features"],
        shuffle=False,
    )

    def sample_train_batches(seed: int) -> list:
        rng = random.Random(seed)
        clean_idx = rng.sample(train_data["clean_indices"], n_clean_per_epoch)
        indices = train_data["secret_indices"] + clean_idx
        return create_binary_batches(
            train_data["bytes"],
            train_data["labels"],
            train_data["lengths"],
            batch_size=args.batch_size,
            feature_tensors=train_data["features"],
            indices=indices,
            seed=seed,
            show_progress=False,
        )

    base_pos_weight = n_clean_per_epoch / n_secrets if n_secrets > 0 else 1.0
    distill_loss = DistillationLoss(
        temperature=args.temperature,
        alpha=args.alpha,
        pos_weight=base_pos_weight * args.pos_weight,
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Temperature: {args.temperature}, Alpha: {args.alpha}")

    best_f1 = 0.0
    best_state = None

    for epoch in range(args.epochs):
        student.train()
        train_batches = sample_train_batches(args.seed + epoch)

        epoch_loss = 0.0
        epoch_kl = 0.0
        epoch_task = 0.0
        n_batches = 0

        for batch in train_batches:
            batch_bytes = batch.bytes.to(device)
            labels = batch.labels.to(device)
            mask = None
            if batch.lengths is not None:
                L = batch_bytes.shape[1]
                mask = torch.arange(L, device=device).unsqueeze(0) >= batch.lengths.to(
                    device
                ).unsqueeze(1)
            features = batch.features.to(device) if batch.features is not None else None

            with torch.no_grad():
                teacher_out = teacher(batch_bytes, mask=mask, precomputed=features)
                teacher_logits = teacher_out[0] if isinstance(teacher_out, tuple) else teacher_out

            student_out = student(batch_bytes, mask=mask, precomputed=features)
            student_logits = student_out[0] if isinstance(student_out, tuple) else student_out

            loss, metrics = distill_loss(student_logits, teacher_logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            epoch_loss += metrics["total_loss"]
            epoch_kl += metrics["kl_loss"]
            epoch_task += metrics.get("task_loss", 0.0)
            n_batches += 1

        scheduler.step()

        val_metrics = validate(student, teacher, val_batches, device)

        avg_loss = epoch_loss / n_batches
        avg_kl = epoch_kl / n_batches
        avg_task = epoch_task / n_batches

        best_str = ""
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
            best_str = " *BEST*"

        print(
            f"Epoch {epoch+1:2d}: loss={avg_loss:.4f} (kl={avg_kl:.4f} task={avg_task:.4f}) | "
            f"val kl={val_metrics['kl']:.4f} P={val_metrics['precision']:.3f} "
            f"R={val_metrics['recall']:.3f} F1={val_metrics['f1']:.3f}{best_str}"
        )

    if best_state is not None:
        student.load_state_dict(best_state)

    final_metrics = validate(student, teacher, val_batches, device)
    print(f"\nFinal: P={final_metrics['precision']:.3f} R={final_metrics['recall']:.3f} F1={final_metrics['f1']:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_config": student_config.to_dict(),
            "state_dict": student.state_dict(),
            "teacher_path": str(args.teacher),
            "distillation": {
                "temperature": args.temperature,
                "alpha": args.alpha,
            },
            "metrics": final_metrics,
        },
        args.output,
    )
    print(f"Saved to {args.output}")
    print(f"  Student: {student_params:,} params ({student_params/teacher_params*100:.1f}% of teacher)")


if __name__ == "__main__":
    main()
