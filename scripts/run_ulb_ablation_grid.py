#!/usr/bin/env python3
"""Launch a full ULB ablation grid for overnight runs.

This script orchestrates:
1) Benchmark runs on `scripts/bench_ssm.py`
2) Optional leakage-radius checks via `scripts/check_q_peeking_causality.py`

Default grid focuses on the most important ULB-vs-standard ablations:
- Q mixing: lerp vs none vs conv2 vs conv3
- K-lerp: on vs off
- Attention path: blend vs softmax vs silu2
- Activation: learnable Swish vs fixed SiLU
- Paired heads: paired vs unpaired

Outputs:
- per-seed CSV from bench_ssm
- command log
- optional radius JSONL report
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_MODELS = [
    # Baselines
    "S4D",
    "Mamba",
    "MHA",
    # ULB base + primary ablations
    "ULBBlendP",
    "ULBBlendPQNone",
    "ULBBlendPConv2",
    "ULBBlendPConv3",
    "ULBBlendPNoK",
    "ULBSoftmaxP",
    "ULBSilu2P",
    "ULBBlendPSiLU",
    "ULBBlend",
]


def run_cmd(cmd: list[str], log_file: Path) -> int:
    line = " ".join(shlex.quote(c) for c in cmd)
    with log_file.open("a") as f:
        f.write(f"\n$ {line}\n")
    print(f"\n$ {line}")
    proc = subprocess.run(cmd, text=True)
    with log_file.open("a") as f:
        f.write(f"[exit={proc.returncode}]\n")
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="Run ULB ablation grid")
    ap.add_argument("--output-dir", type=Path, default=Path("results/ulb_ablation_grid"))
    ap.add_argument("--python", type=str, default=".venv/bin/python")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--tasks", type=str, nargs="+", default=["mixed", "induction"])
    ap.add_argument("--models", type=str, nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--max-epochs", type=int, default=32)
    ap.add_argument("--train-batches", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--early-stop-acc", type=float, default=0.99)
    ap.add_argument("--check-radius", action="store_true", help="Run leakage-radius checks after training runs")
    ap.add_argument("--clear-cache", action="store_true", help="Delete scripts/.bench_cache before running")
    ap.add_argument("--execute", action="store_true", help="Actually execute commands (default: dry-run)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    log_file = out / "commands.log"

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    plan_file = out / f"plan_{ts}.json"

    plan = {
        "timestamp": ts,
        "python": args.python,
        "seeds": args.seeds,
        "tasks": args.tasks,
        "models": args.models,
        "dim": args.dim,
        "layers": args.layers,
        "max_epochs": args.max_epochs,
        "train_batches": args.train_batches,
        "val_batches": args.val_batches,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "early_stop_acc": args.early_stop_acc,
        "check_radius": args.check_radius,
        "clear_cache": args.clear_cache,
        "execute": args.execute,
    }
    plan_file.write_text(json.dumps(plan, indent=2))

    cmds: list[list[str]] = []

    if args.clear_cache:
        cmds.append(["rm", "-rf", str(repo / "scripts" / ".bench_cache")])

    for seed in args.seeds:
        csv_path = out / f"bench_seed{seed}.csv"
        cmd = [
            args.python,
            str(repo / "scripts" / "bench_ssm.py"),
            "--models",
            *args.models,
            "--tasks",
            *args.tasks,
            "--dim",
            str(args.dim),
            "--layers",
            str(args.layers),
            "--max-epochs",
            str(args.max_epochs),
            "--train-batches",
            str(args.train_batches),
            "--val-batches",
            str(args.val_batches),
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(args.seq_len),
            "--early-stop-acc",
            str(args.early_stop_acc),
            "--seed",
            str(seed),
            "--csv",
            str(csv_path),
        ]
        cmds.append(cmd)

    if args.check_radius:
        radius_jsonl = out / "radius_checks.jsonl"
        for seed in args.seeds:
            for qmix in ["lerp", "conv2", "conv3"]:
                cmd = [
                    args.python,
                    str(repo / "scripts" / "check_q_peeking_causality.py"),
                    "--dim",
                    str(args.dim),
                    "--n-heads",
                    "4",
                    "--n-layers",
                    str(args.layers),
                    "--batch-size",
                    "8",
                    "--seq-len",
                    str(args.seq_len),
                    "--attn-mode",
                    "blend",
                    "--q-mix",
                    qmix,
                    "--paired",
                    "--device",
                    "cpu",
                    "--seed",
                    str(seed),
                ]
                cmds.append(cmd)

        # note target path for user; capture should be redirected manually in execute mode below
        with log_file.open("a") as f:
            f.write(f"\n# radius jsonl target: {radius_jsonl}\n")

    print(f"Planned {len(cmds)} commands. Plan file: {plan_file}")
    for c in cmds:
        print("$", " ".join(shlex.quote(x) for x in c))

    if not args.execute:
        print("\nDry-run only. Re-run with --execute to launch.")
        return

    # Execute sequentially; stop on first failure.
    for c in cmds:
        if args.check_radius and c[1].endswith("check_q_peeking_causality.py"):
            # Capture radius JSON to file while also printing
            proc = subprocess.run(c, text=True, capture_output=True)
            print(proc.stdout)
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                raise SystemExit(proc.returncode)
            radius_jsonl = out / "radius_checks.jsonl"
            with radius_jsonl.open("a") as f:
                f.write(proc.stdout.strip() + "\n")
            continue

        rc = run_cmd(c, log_file)
        if rc != 0:
            raise SystemExit(rc)

    print(f"\nAll commands finished. Outputs in: {out}")


if __name__ == "__main__":
    main()
