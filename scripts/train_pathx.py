#!/usr/bin/env python3
"""Train on Path-X benchmark using locally generated pathfinder data.

Generate data first:
    python scripts/generate_pathx.py --n-train 160000 --n-val 20000 --n-test 20000

Then train:
    python scripts/train_pathx.py --unified --adaptive
"""

import argparse
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

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


CACHE_DIR = Path(".cache/pathx")
SEQ_LEN = 256 * 256


def load_pathx_split(split: str, data_dir: Path = CACHE_DIR):
    """Load Path-X split from generated dataset.

    Expected structure:
        data_dir/{split}/imgs/{batch_id}/sample_*.png
        data_dir/{split}/metadata/{batch_id}.npy

    Metadata columns: [subpath, filename, nimg, label, ...]
    """
    split_dir = data_dir / split
    cache_path = data_dir / f"pathx_{split}_cache.npz"

    if cache_path.exists():
        print(f"  Loading cached {split} from {cache_path}")
        npz = dict(np.load(cache_path, allow_pickle=False))
        n = len(npz["labels"])
        return {
            "bytes": [torch.from_numpy(npz["pixels"][i]).long() for i in range(n)],
            "labels": npz["labels"].tolist(),
            "lengths": [SEQ_LEN] * n,
            "features": None,
            "secret_indices": npz["pos_indices"].tolist(),
            "clean_indices": npz["neg_indices"].tolist(),
        }

    metadata_dir = split_dir / "metadata"
    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {split_dir}. "
            f"Generate it first with: python scripts/generate_pathx.py"
        )

    metadata_files = sorted(metadata_dir.glob("*.npy"))
    if not metadata_files:
        raise FileNotFoundError(f"No metadata files found in {metadata_dir}")

    print(
        f"  Loading Path-X {split} from {split_dir} ({len(metadata_files)} batches)..."
    )

    pixels_list = []
    labels = []
    pos_indices = []
    neg_indices = []

    sample_idx = 0
    for meta_file in metadata_files:
        meta = np.load(meta_file, allow_pickle=True)
        for row in meta:
            subpath = str(row[0])
            filename = str(row[1])
            label = int(row[3])

            img_path = split_dir / subpath / filename
            if not img_path.exists():
                print(f"    Warning: missing {img_path}")
                continue

            img = Image.open(img_path).convert("L")
            pixels = np.array(img, dtype=np.uint8).flatten()

            pixels_list.append(pixels)
            labels.append(float(label))
            if label > 0:
                pos_indices.append(sample_idx)
            else:
                neg_indices.append(sample_idx)

            sample_idx += 1
            if sample_idx % 10000 == 0:
                print(f"    Processed {sample_idx:,} samples...")

    if not pixels_list:
        raise ValueError(f"No valid samples found in {split_dir}")

    pixels = np.stack(pixels_list)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        pixels=pixels,
        labels=np.array(labels, dtype=np.float32),
        pos_indices=np.array(pos_indices, dtype=np.int32),
        neg_indices=np.array(neg_indices, dtype=np.int32),
    )
    print(f"  Cached {len(labels):,} samples to {cache_path}")

    return {
        "bytes": [torch.from_numpy(pixels[i]).long() for i in range(len(labels))],
        "labels": labels,
        "lengths": [SEQ_LEN] * len(labels),
        "features": None,
        "secret_indices": pos_indices,
        "clean_indices": neg_indices,
    }


def eval_swa_model(trainer: Trainer, val_batches: list, device: torch.device):
    """Evaluate the SWA model by temporarily loading its weights."""
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
    acc = val_metrics.accuracy

    trainer.model.load_state_dict(original_state)

    return acc, threshold, val_metrics


def cosine_schedule(
    epoch: int, total_epochs: int, start_val: float, end_val: float
) -> float:
    if total_epochs <= 1:
        return end_val
    progress = epoch / (total_epochs - 1)
    return end_val + 0.5 * (start_val - end_val) * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser(description="Train on Path-X (Long Range Arena)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("models/pathx.pt"))
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
        "--max-clean-pool",
        type=int,
        default=500_000,
        help="Max neg samples to keep in memory for resampling",
    )
    parser.add_argument(
        "--clean-ratio",
        type=float,
        default=1.0,
        help="Ratio of neg samples to pos samples per epoch",
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
        "--quick-eval",
        action="store_true",
        help="Subsample val set like training (all pos + equal neg)",
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
    parser.add_argument("--embed-width", type=int, default=64, help="Embedding width")
    parser.add_argument(
        "--attn-window",
        type=int,
        default=64,
        help="Attention window size (64 for 65k tokens)",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=CACHE_DIR, help="Path to generated dataset"
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=1,
        help="Number of unified layers to stack (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, mps, cpu (default: auto)",
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
    print(f"Path-X: sequence length = {SEQ_LEN}")

    if args.unified:
        ssm_kernel_sizes = tuple(int(k) for k in args.ssm_kernels.split(","))
        layer_config = LayerConfig(
            embed_width=args.embed_width,
            conv_groups=4,
            attn_heads=4,
            attn_ffn_mult=2,
            attn_window_size=args.attn_window,
            num_attn_features=8,
            ssm_state_size=64,
            ssm_n_heads=4,
            ssm_expand=2,
            ssm_kernel_sizes=ssm_kernel_sizes,
            num_ssm_features=8,
            num_embed_features=8,
            num_hidden_features=16,
            num_heuristic_features=0,
            mlp_hidden_mult=4,
            mlp_output_dim=32,
            dropout=0.1,
            adaptive_conv=args.adaptive,
            n_adaptive_branches=args.n_branches,
        )
        head_config = HeadConfig(head_type="classifier", n_classes=2)
        model_config = ModelConfig(
            n_layers=args.n_layers, layer=layer_config, head=head_config
        )
        if args.adaptive:
            print(
                f"\nUsing unified architecture with ADAPTIVE conv ({args.n_branches} branches)"
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
            embed_width=args.embed_width,
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
            num_precomputed_features=0,
        )
        print(f"\nUsing SSM3 (Mamba-3 style), conv_type={ssm_conv_type}")
    else:
        model_config = ModelConfig(
            task="binary",
            arch_type="both",
            embed_width=args.embed_width,
            conv_kernel_sizes=(3, 5, 7, 9),
            conv_groups=3,
            attn_depth=1,
            attn_heads=4,
            num_embed_features=8,
            mlp_hidden_mult=4,
            mlp_output_dim=32,
            num_precomputed_features=0,
        )

    print(f"\nLoading Path-X dataset from {args.data_dir}...")
    train_data = load_pathx_split("train", args.data_dir)
    val_data = load_pathx_split("val", args.data_dir)

    n_train_pos = len(train_data["secret_indices"])
    n_train_neg = len(train_data["clean_indices"])
    n_val_pos = len(val_data["secret_indices"])
    print(f"Train: {n_train_pos:,} positive, {n_train_neg:,} negative")
    print(f"Val: {len(val_data['labels']):,} samples ({n_val_pos:,} positive)")

    if n_train_neg > args.max_clean_pool:
        print(f"Subsampling neg pool: {n_train_neg:,} -> {args.max_clean_pool:,}")
        rng = random.Random(args.seed)
        train_data["clean_indices"] = rng.sample(
            train_data["clean_indices"], args.max_clean_pool
        )
        n_train_neg = args.max_clean_pool

    n_neg_per_epoch = min(int(n_train_pos * args.clean_ratio), n_train_neg)
    print(
        f"Per epoch: {n_train_pos:,} positive (all) + {n_neg_per_epoch:,} negative ({args.clean_ratio:.1f}x, resampled)"
    )

    base_pos_weight = n_neg_per_epoch / n_train_pos
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

    if args.quick_eval:
        n_val_neg = len(val_data["clean_indices"])
        n_val_neg_sample = min(n_val_pos, n_val_neg)
        rng = random.Random(args.seed)
        val_neg_sample = rng.sample(val_data["clean_indices"], n_val_neg_sample)
        val_indices = val_data["secret_indices"] + val_neg_sample
        print(
            f"Quick eval: {n_val_pos:,} positive + {n_val_neg_sample:,} negative = {len(val_indices):,} samples"
        )

        val_batches = create_binary_batches(
            val_data["bytes"],
            val_data["labels"],
            val_data["lengths"],
            batch_size=args.eval_batch_size,
            feature_tensors=val_data["features"],
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
            shuffle=False,
        )
    print(f"Val batches: {len(val_batches):,}")

    def sample_train_batches(epoch_seed: int) -> list:
        rng = random.Random(epoch_seed)
        pos_idx = train_data["secret_indices"]
        neg_idx = rng.sample(train_data["clean_indices"], n_neg_per_epoch)
        indices = pos_idx + neg_idx
        return create_binary_batches(
            train_data["bytes"],
            train_data["labels"],
            train_data["lengths"],
            batch_size=args.batch_size,
            feature_tensors=train_data["features"],
            indices=indices,
            seed=epoch_seed,
            show_progress=False,
        )

    initial_batches = sample_train_batches(args.seed)

    model = Model(model_config)
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(model_config)

    train_config = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        pos_weight=pos_weight_start,
        threshold_optimize_for="f1",
        use_swa=not args.no_swa,
        preload_to_device=False,
        hard_example_ratio=args.hard_ratio,
        curriculum_start_epoch=args.curriculum_start,
    )

    trainer = Trainer(
        model, initial_batches, val_batches, config=train_config, device=device
    )

    start_epoch = 0
    best_acc = 0.0
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
        best_acc = ckpt.get("best_acc", ckpt.get("best_acc", 0.0))
        best_epoch = ckpt.get("best_epoch", 0)
        best_state = ckpt.get("best_state")
        best_threshold = ckpt.get("best_threshold", 0.5)
        trainer.optimal_threshold = ckpt.get("optimal_threshold", 0.5)
        if ckpt.get("swa_active", False) and trainer.swa_model is not None:
            trainer.swa_active = True
            if "swa_state" in ckpt:
                trainer.swa_model.load_state_dict(ckpt["swa_state"])
        print(f"  Resuming from epoch {start_epoch}, best_acc={best_acc:.4f}")

    print(f"\nTraining for epochs {start_epoch + 1}-{args.epochs}...")

    for epoch in range(start_epoch, args.epochs):
        trainer.train_data = sample_train_batches(args.seed + epoch)
        trainer._batch_losses = [0.0] * len(trainer.train_data)
        if args.pos_weight_decay:
            pw = cosine_schedule(epoch, args.epochs, pos_weight_start, pos_weight_end)
            trainer.config.pos_weight = pw
        train_metrics = trainer.train_epoch(epoch)
        val_metrics = trainer.validate(optimize_threshold=True)

        pw_str = f" pw={trainer.config.pos_weight:.2f}" if args.pos_weight_decay else ""

        acc_for_best = val_metrics.accuracy

        best_str = ""
        if acc_for_best > best_acc:
            best_acc = acc_for_best
            best_epoch = epoch + 1
            best_state = {
                k: v.cpu().clone() for k, v in trainer.model.state_dict().items()
            }
            best_threshold = trainer.optimal_threshold
            best_str = " *BEST*"

        print(
            f"Epoch {epoch + 1:2d}:{pw_str} train {train_metrics} | val {val_metrics} acc={acc_for_best:.4f}{best_str}"
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
                "best_acc": best_acc,
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

        swa_acc, swa_threshold, swa_metrics = eval_swa_model(
            trainer, val_batches, device
        )
        if swa_acc is not None and trainer.swa_model is not None:
            print(f"         SWA: {swa_metrics} acc={swa_acc:.4f}")

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
                    "swa_accuracy": swa_acc,
                },
                swa_path,
            )
            print(f"         -> {swa_path}")

    if not args.no_swa:
        trainer.finalize_swa()
        final = trainer.validate(optimize_threshold=True)
        swa_acc = final.accuracy
        print(f"\nPost-SWA: {final} acc={swa_acc:.4f}")

        if swa_acc > best_acc:
            best_acc = swa_acc
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
    best_src = (
        f"epoch {best_epoch}"
        if best_epoch > 0
        else "SWA"
        if best_epoch == -1
        else "initial"
    )

    print(f"\nBest model ({best_src}): {final} acc={final.accuracy:.4f}")
    print(f"Threshold: {trainer.optimal_threshold:.4f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
