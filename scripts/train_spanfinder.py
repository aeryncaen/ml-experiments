#!/usr/bin/env python3
"""Train the SpanFinder model on collected data."""

import argparse
from pathlib import Path

from heuristic_secrets.spanfinder.model import SpanFinderModel, SpanFinderConfig
from heuristic_secrets.spanfinder.train import (
    SpanFinderTrainer,
    TrainingConfig,
    set_seed,
    evaluate_spanfinder,
    get_device,
)
from heuristic_secrets.spanfinder.dataset import (
    load_spanfinder_dataset,
    create_batches,
)


def main():
    parser = argparse.ArgumentParser(description="Train SpanFinder model")
    parser.add_argument(
        "--data-dir", "-d",
        type=Path,
        default=Path("data"),
        help="Data directory",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output model path (safetensors)",
    )
    
    # Model architecture
    parser.add_argument(
        "--preset",
        type=str,
        choices=["tiny", "small", "medium"],
        default="small",
        help="Model size preset (default: small)",
    )
    parser.add_argument("--width", type=int, default=None, help="Override embed/hidden dim")
    parser.add_argument("--cnn-depth", type=int, default=None, help="Override CNN depth")
    parser.add_argument("--cnn-window", type=int, default=None, help="Override CNN window size")
    
    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in bytes")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Overlap between chunks")
    parser.add_argument("--pos-weight", type=float, default=1000.0, help="Weight for positive examples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Build model config from preset + overrides
    if args.preset == "tiny":
        config = SpanFinderConfig.tiny()
    elif args.preset == "medium":
        config = SpanFinderConfig.medium()
    else:
        config = SpanFinderConfig.small()

    if args.width is not None:
        config.width = args.width
    if args.cnn_depth is not None:
        config.cnn_depth = args.cnn_depth
    if args.cnn_window is not None:
        config.cnn_window = args.cnn_window

    device = get_device()
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print(f"Model config: width={config.width}, cnn_depth={config.cnn_depth}, cnn_window={config.cnn_window}")
    print(f"Receptive field: {config.cnn_window * config.cnn_depth} bytes")
    set_seed(args.seed)

    # Load data
    print(f"\nLoading training data from {args.data_dir}...")
    print(f"  Chunk size: {args.chunk_size}, overlap: {args.chunk_overlap}")
    train_dataset, train_stats = load_spanfinder_dataset(
        args.data_dir, split="train", 
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    val_dataset, val_stats = load_spanfinder_dataset(
        args.data_dir, split="val",
        chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )

    print(f"  Train: {train_stats.total_samples} docs -> {train_stats.total_chunks} chunks, {train_stats.total_spans} spans")
    print(f"         avg {train_stats.avg_chunks_per_sample:.1f} chunks/doc")
    print(f"  Val:   {val_stats.total_samples} docs -> {val_stats.total_chunks} chunks, {val_stats.total_spans} spans")

    # Create batches from chunked data
    print(f"\nCreating batches (batch_size={args.batch_size})...")
    train_batches = create_batches(
        train_dataset, args.batch_size, shuffle=True, seed=args.seed
    )
    val_batches = create_batches(
        val_dataset, args.batch_size, shuffle=False
    )
    print(f"  Train batches: {len(train_batches)}")
    print(f"  Val batches: {len(val_batches)}")

    # Create model
    print(f"\nCreating SpanFinder model...")
    model = SpanFinderModel(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    trainer = SpanFinderTrainer(
        model=model,
        config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            pos_weight=args.pos_weight,
            seed=args.seed,
        ),
    )

    result = trainer.train(train_batches, val_batches)

    print(f"\nTraining complete:")
    print(f"  Final train loss: {result.final_loss:.4f}")
    if result.val_loss_history:
        print(f"  Final val loss:   {result.val_loss_history[-1]:.4f}")
    print(f"  Loss reduction: {result.loss_history[0]:.4f} -> {result.final_loss:.4f}")

    # Evaluate
    print("\nEvaluating on validation set...")
    metrics = evaluate_spanfinder(model, val_batches)
    print(f"  Start - Precision: {metrics['start_precision']:.4f}, "
          f"Recall: {metrics['start_recall']:.4f}, F1: {metrics['start_f1']:.4f}")
    print(f"  End   - Precision: {metrics['end_precision']:.4f}, "
          f"Recall: {metrics['end_recall']:.4f}, F1: {metrics['end_f1']:.4f}")

    # Save model
    if args.output:
        print(f"\nSaving model to {args.output}...")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        import torch
        from safetensors.torch import save_file
        
        # Save weights
        weights_path = args.output.with_suffix(".safetensors")
        save_file(model.state_dict(), weights_path)
        
        # Save config
        meta_path = args.output.with_suffix(".meta.json")
        meta = config.to_dict()
        meta["metrics"] = {
            "start_recall": metrics["start_recall"],
            "end_recall": metrics["end_recall"],
            "start_f1": metrics["start_f1"],
            "end_f1": metrics["end_f1"],
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        print(f"  Saved: {weights_path}")
        print(f"  Saved: {meta_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
