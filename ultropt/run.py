#!/usr/bin/env python3
"""
Main entrypoint for the Shakespeare three-tier training experiment.

Usage:
    python run.py                          # standard GPT, small grid, 2M tokens
    python run.py --ngpt                   # nGPT (hypersphere), small grid
    python run.py --grid full              # full grid
    python run.py --tokens 5000000         # 5M tokens
"""

import argparse

from experiment import run_experiment, print_summary, save_results, autodetect_device


def main():
    parser = argparse.ArgumentParser(description="Shakespeare 3-tier training experiment")
    parser.add_argument("--grid", choices=["small", "full"], default="small",
                        help="Grid size: 'small' or 'full'")
    parser.add_argument("--tokens", type=int, default=2_000_000,
                        help="Total token budget per run (default 2M)")
    parser.add_argument("--eval-every", type=int, default=200_000,
                        help="Evaluate every N tokens")
    parser.add_argument("--baseline-lr", type=float, default=None,
                        help="Baseline learning rate (default: 3e-3 for nGPT, 3e-4 for GPT)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cuda, mps, or cpu (auto-detected if omitted)")
    parser.add_argument("--ngpt", action="store_true",
                        help="Use nGPT (normalized transformer on hypersphere)")
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
    )

    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
