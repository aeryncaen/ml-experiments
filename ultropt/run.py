#!/usr/bin/env python3
"""
Main entrypoint for the ultropt training experiments.

Usage:
    # UltrOpt (default mode: baseline + UltrOpt with default config)
    python run.py ultropt

    # UltrOpt ablation (toggle each component off)
    python run.py ultropt --mode ablation

    # UltrOpt grid search
    python run.py ultropt --mode grid --grid full

    # Legacy three-tier (pre-UltrOpt)
    python run.py legacy

    # Common options
    python run.py ultropt --ngpt                 # nGPT model
    python run.py ultropt --tokens 50000000      # 50M token budget
    python run.py ultropt --no-slerp             # disable ghost SLERP
    python run.py ultropt --no-mars              # disable MARS
    python run.py ultropt --no-ns                # disable Newton-Schulz
    python run.py ultropt --no-cautious          # disable cautious masking
    python run.py ultropt --no-schedule-free     # disable Schedule-Free
"""

import argparse

from experiment import (
    run_experiment,
    run_ultropt_experiment,
    print_summary,
    save_results,
    autodetect_device,
)


def main():
    parser = argparse.ArgumentParser(description="ultropt training experiments")
    subparsers = parser.add_subparsers(dest="command", help="Experiment type")

    # --- Shared arguments ---
    def add_common_args(p):
        p.add_argument("--grid", choices=["small", "full"], default="small",
                        help="Grid size: 'small' or 'full'")
        p.add_argument("--tokens", type=int, default=20_000_000,
                        help="Total token budget per run (default 20M)")
        p.add_argument("--eval-every", type=int, default=2_000_000,
                        help="Evaluate every N tokens")
        p.add_argument("--ngpt", action="store_true",
                        help="Use nGPT (normalized transformer on hypersphere)")
        p.add_argument("--baseline-lr", type=float, default=None,
                        help="Baseline LR (default: 3e-3 for nGPT, 3e-4 for GPT)")
        p.add_argument("--data-dir", type=str, default="../tokenized",
                        help="Tokenized data directory (with train/ and val/ subdirs)")
        p.add_argument("--tokenizer-dir", type=str,
                        default="../experiments/yamit/tokenizer/artifacts/yamit",
                        help="Tokenizer artifacts directory")
        p.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length (default 256)")
        p.add_argument("--batch-size", type=int, default=64,
                        help="Batch / micro-batch size")
        p.add_argument("--device", type=str, default=None,
                        help="Device: cuda, mps, or cpu (auto-detected if omitted)")
        p.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
        p.add_argument("--output", type=str, default="results.json",
                        help="Path to save results JSON")

    # --- Legacy subcommand ---
    legacy_parser = subparsers.add_parser("legacy", help="Legacy three-tier (pre-UltrOpt)")
    add_common_args(legacy_parser)

    # --- UltrOpt subcommand ---
    ultropt_parser = subparsers.add_parser("ultropt", help="UltrOpt optimizer")
    add_common_args(ultropt_parser)
    ultropt_parser.add_argument("--mode", choices=["default", "ablation", "grid"],
                                 default="default",
                                 help="Experiment mode (default/ablation/grid)")

    # UltrOpt hyperparameter overrides
    ultropt_parser.add_argument("--lr", type=float, default=None,
                                 help="UltrOpt base LR (micro tier)")
    ultropt_parser.add_argument("--batch-lr-factor", type=float, default=None,
                                 help="Batch LR = lr * factor")
    ultropt_parser.add_argument("--super-lr-factor", type=float, default=None,
                                 help="Super LR = lr * factor")
    ultropt_parser.add_argument("--micros-per-batch", type=int, default=None,
                                 help="Micro steps per batch step")
    ultropt_parser.add_argument("--batches-per-super", type=int, default=None,
                                 help="Batch steps per super step")

    # Ablation toggles
    ultropt_parser.add_argument("--no-slerp", action="store_true",
                                 help="Disable ghost SLERP (use mean reduction)")
    ultropt_parser.add_argument("--no-mars", action="store_true",
                                 help="Disable MARS variance reduction")
    ultropt_parser.add_argument("--no-ns", action="store_true",
                                 help="Disable Newton-Schulz orthogonalization")
    ultropt_parser.add_argument("--no-cautious", action="store_true",
                                 help="Disable cautious masking")
    ultropt_parser.add_argument("--no-schedule-free", action="store_true",
                                 help="Disable Schedule-Free optimization")
    ultropt_parser.add_argument("--error-feedback", action="store_true",
                                 help="Enable error feedback on accumulators")
    ultropt_parser.add_argument("--accumulate-signal",
                                 choices=["raw", "mars_reduced", "ns_preconditioned"],
                                 default=None, help="What goes into accumulators")
    ultropt_parser.add_argument("--slerp-magnitude",
                                 choices=["mean", "median", "harmonic", "geometric"],
                                 default=None, help="SLERP magnitude restoration method")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    results = []

    # Default baseline LR depends on model type
    baseline_lr = args.baseline_lr
    if baseline_lr is None:
        baseline_lr = 3e-3 if args.ngpt else 3e-4

    device = args.device if args.device else autodetect_device()
    print(f"Using device: {device}")

    if args.command == "legacy":
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

    elif args.command == "ultropt":
        # Build overrides dict from CLI flags
        overrides = {}
        if args.lr is not None:
            overrides["lr"] = args.lr
        if args.batch_lr_factor is not None:
            overrides["batch_lr_factor"] = args.batch_lr_factor
        if args.super_lr_factor is not None:
            overrides["super_lr_factor"] = args.super_lr_factor
        if args.micros_per_batch is not None:
            overrides["micros_per_batch"] = args.micros_per_batch
        if args.batches_per_super is not None:
            overrides["batches_per_super"] = args.batches_per_super
        if args.no_slerp:
            overrides["slerp_mode"] = "mean"
        if args.no_mars:
            overrides["mars"] = False
        if args.no_ns:
            overrides["newton_schulz"] = False
        if args.no_cautious:
            overrides["cautious"] = False
        if args.no_schedule_free:
            overrides["schedule_free"] = False
        if args.error_feedback:
            overrides["error_feedback"] = True
        if args.accumulate_signal is not None:
            overrides["accumulate_signal"] = args.accumulate_signal
        if args.slerp_magnitude is not None:
            overrides["slerp_magnitude"] = args.slerp_magnitude

        results = run_ultropt_experiment(
            mode=args.mode,
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
            ultropt_overrides=overrides if overrides else None,
        )

    print_summary(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
