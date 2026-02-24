#!/usr/bin/env python3
"""
Main entrypoint for the ultropt three-tier training experiment.

Usage:
    python run.py                              # standard GPT, small grid
    python run.py --ngpt                       # nGPT (hypersphere)
    python run.py --grid full                  # full grid
    python run.py --tokens 50000000            # 50M tokens
    python run.py --data-dir /path/to/tokenized --tokenizer-dir /path/to/artifacts/yamit
"""

import argparse

from experiment import run_experiment, print_summary, save_results, autodetect_device


def main():
    parser = argparse.ArgumentParser(description="ultropt 3-tier training experiment")
    # grid
    parser.add_argument("--grid", choices=["small", "full"], default="small",
                        help="Grid size: 'small' or 'full'")
    # budget
    parser.add_argument("--tokens", type=int, default=20_000_000,
                        help="Total token budget per run (default 20M)")
    parser.add_argument("--eval-every", type=int, default=2_000_000,
                        help="Evaluate every N tokens")
    # model
    parser.add_argument("--ngpt", action="store_true",
                        help="Use nGPT (normalized transformer on hypersphere)")
    parser.add_argument("--baseline-lr", type=float, default=None,
                        help="Baseline learning rate (default: 3e-3 for nGPT, 3e-4 for GPT)")
    # data
    parser.add_argument("--data-dir", type=str, default="../tokenized",
                        help="Tokenized data directory (with train/ and val/ subdirs)")
    parser.add_argument("--tokenizer-dir", type=str,
                        default="../experiments/yamit/tokenizer/artifacts/yamit",
                        help="Tokenizer artifacts directory (with artifact_meta.json)")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length (default 256)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for baseline and micro_batch_size for three-tier")
    # infra
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, mps, or cpu (auto-detected if omitted)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    parser.add_argument("--output", type=str, default="results.json",
                        help="Path to save results JSON")
    args = parser.parse_args()

    # Default baseline LR depends on model type
    baseline_lr = args.baseline_lr
    if baseline_lr is None:
        baseline_lr = 3e-3 if args.ngpt else 3e-4

    # Auto-detect device if not specified
    device = args.device if args.device else autodetect_device()
    print(f"Using device: {device}")

    results = run_experiment(
        grid=args.grid,
        total_tokens=args.tokens,
        eval_every_tokens=args.eval_every,
        baseline_lr=baseline_lr,
        device=device,
        ngpt=args.ngpt,
        data_dir=args.data_dir,
        tokenizer_dir=args.tokenizer_dir,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        compile=not args.no_compile,
    )

    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
