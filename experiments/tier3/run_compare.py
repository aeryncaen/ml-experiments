#!/usr/bin/env python3
"""
Wrapper to run Tier 3 optimizer comparisons and aggregate results.

Runs each selected dataset script twice:
  1) --optimizer adam
  2) --optimizer chi2

Then loads both JSON outputs and prints a compact comparison table.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


METRICS = [
    "val_acc_ep3",
    "val_acc_ep5",
    "clean_acc",
    "robust_avg",
    "acc_border",
    "acc_center",
    "acc_contrast",
]


def run_cmd(cmd: list[str], dry_run: bool) -> None:
    print("$ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize_pair(adam_res: dict, chi2_res: dict, label: str) -> dict:
    summary = {"dataset": label, "modes": {}}
    for mode in ["baseline", "geo_system"]:
        summary["modes"][mode] = {}
        for m in METRICS:
            a = adam_res[mode][m]["mean"]
            c = chi2_res[mode][m]["mean"]
            summary["modes"][mode][m] = {
                "adam": a,
                "chi2": c,
                "delta": c - a,
            }
    return summary


def print_table(summary: dict) -> None:
    label = summary["dataset"]
    print("\n" + "=" * 80)
    print(f"Dataset: {label}")
    print("=" * 80)
    for mode in ["baseline", "geo_system"]:
        print(f"\nMode: {mode}")
        print(f"{'metric':14s} | {'adam':>8s} | {'chi2':>8s} | {'delta':>8s}")
        print("-" * 50)
        for m in METRICS:
            row = summary["modes"][mode][m]
            print(
                f"{m:14s} | {row['adam']:8.4f} | {row['chi2']:8.4f} | {row['delta']:+8.4f}"
            )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["digits", "fashion", "both"], default="both")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--trust-radius", type=float, default=0.05)
    p.add_argument("--chi2-adaptive", action="store_true", default=True)
    p.add_argument("--no-chi2-adaptive", action="store_false", dest="chi2_adaptive")
    p.add_argument("--chi2-reject-tol", type=float, default=0.02)
    p.add_argument("--chi2-shrink", type=float, default=0.5)
    p.add_argument("--chi2-grow", type=float, default=1.01)
    p.add_argument("--chi2-min-radius", type=float, default=1e-4)
    p.add_argument("--chi2-max-radius", type=float, default=1.0)
    p.add_argument("--chi2-sigma", type=float, default=3.0)
    p.add_argument("--chi2-q-low", type=float, default=0.05)
    p.add_argument("--chi2-q-high", type=float, default=0.5)
    p.add_argument("--chi2-q-beta", type=float, default=0.9)
    p.add_argument("--reg-lambda", type=float, default=1e-3)
    p.add_argument("--reg-warmup-epochs", type=int, default=8)
    p.add_argument("--max-train", type=int, default=60000)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--out-dir", type=str, default=".")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datasets = ["digits", "fashion"] if args.dataset == "both" else [args.dataset]

    all_summaries = []

    for ds in datasets:
        if ds == "digits":
            script = root / "train_digits_geo_system.py"
            base_common = [
                sys.executable,
                str(script),
                "--epochs",
                str(args.epochs),
                "--seeds",
                str(args.seeds),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--reg-lambda",
                str(args.reg_lambda),
                "--reg-warmup-epochs",
                str(args.reg_warmup_epochs),
            ]
        else:
            script = root / "train_fashion_geo_system.py"
            base_common = [
                sys.executable,
                str(script),
                "--epochs",
                str(args.epochs),
                "--seeds",
                str(args.seeds),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--reg-lambda",
                str(args.reg_lambda),
                "--reg-warmup-epochs",
                str(args.reg_warmup_epochs),
                "--max-train",
                str(args.max_train),
                "--data-dir",
                str(args.data_dir),
            ]

        adam_out = out_dir / f"tier3_{ds}_adam_{stamp}.json"
        chi2_out = out_dir / f"tier3_{ds}_chi2_{stamp}.json"

        adam_cmd = base_common + [
            "--optimizer",
            "adam",
            "--out",
            str(adam_out),
        ]
        chi2_cmd = base_common + [
            "--optimizer",
            "chi2",
            "--trust-radius",
            str(args.trust_radius),
            "--chi2-reject-tol",
            str(args.chi2_reject_tol),
            "--chi2-shrink",
            str(args.chi2_shrink),
            "--chi2-grow",
            str(args.chi2_grow),
            "--chi2-min-radius",
            str(args.chi2_min_radius),
            "--chi2-max-radius",
            str(args.chi2_max_radius),
            "--chi2-sigma",
            str(args.chi2_sigma),
            "--chi2-q-low",
            str(args.chi2_q_low),
            "--chi2-q-high",
            str(args.chi2_q_high),
            "--chi2-q-beta",
            str(args.chi2_q_beta),
            "--out",
            str(chi2_out),
        ]
        if args.chi2_adaptive:
            chi2_cmd.append("--chi2-adaptive")
        else:
            chi2_cmd.append("--no-chi2-adaptive")

        print("\n" + "#" * 80)
        print(f"Running dataset={ds}")
        print("#" * 80)
        run_cmd(adam_cmd, args.dry_run)
        run_cmd(chi2_cmd, args.dry_run)

        if not args.dry_run:
            adam_res = load_json(adam_out)
            chi2_res = load_json(chi2_out)
            summary = summarize_pair(adam_res, chi2_res, ds)
            all_summaries.append(summary)
            print_table(summary)

    if not args.dry_run:
        combined = {
            "config": vars(args),
            "summaries": all_summaries,
        }
        combined_path = out_dir / f"tier3_compare_summary_{stamp}.json"
        combined_path.write_text(json.dumps(combined, indent=2))
        print("\nWrote combined summary:", combined_path)


if __name__ == "__main__":
    main()
