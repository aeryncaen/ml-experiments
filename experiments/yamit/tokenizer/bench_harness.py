#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path


SPEED_RE = re.compile(r"Speed:\s+([0-9.]+)\s+tokens/sec")


def run_once(
    binary: Path,
    tokenizer: Path,
    input_dir: Path,
    output_dir: Path,
    workers: int,
    val_fraction: float,
    env_overrides: dict[str, str],
) -> dict[str, float]:
    cmd = [
        str(binary),
        "--tokenizer",
        str(tokenizer),
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--workers",
        str(workers),
        "--val-fraction",
        str(val_fraction),
    ]

    env = dict(**env_overrides)
    full_env = dict(**os.environ)
    full_env.update(env)

    start = time.perf_counter()
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=full_env)
    wall = time.perf_counter() - start
    logs = proc.stdout + "\n" + proc.stderr

    speed = None
    m = SPEED_RE.search(logs)
    if m is not None:
        speed = float(m.group(1))

    meta_path = output_dir / "tokenize_meta.json"
    meta = json.loads(meta_path.read_text())
    tokens = float(meta["total_tokens"])
    elapsed = float(meta["elapsed_secs"])
    derived = tokens / elapsed if elapsed > 0 else 0.0

    return {
        "speed_log": speed or derived,
        "speed_meta": derived,
        "elapsed_secs": elapsed,
        "wall_secs": wall,
        "tokens": tokens,
    }


def summarize(values: list[float]) -> dict[str, float]:
    out = {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }
    if len(values) > 1:
        out["stdev"] = statistics.pstdev(values)
    else:
        out["stdev"] = 0.0
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Repeatable tokenizer benchmark harness")
    p.add_argument("--binary", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--workers", type=int, nargs="+", default=[8])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--val-fraction", type=float, default=0.0)
    p.add_argument(
        "--gogc",
        nargs="+",
        default=["default", "300"],
        help="GOGC values to test (use 'default' for unset)",
    )
    p.add_argument("--gomemlimit", default="")
    args = p.parse_args()

    binary = Path(args.binary).resolve()
    tokenizer = Path(args.tokenizer).resolve()
    input_dir = Path(args.input).resolve()
    run_root = Path(args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)

    all_runs: list[dict[str, object]] = []
    grouped: dict[tuple[int, str], list[float]] = {}

    for workers in args.workers:
        for gogc in args.gogc:
            key = (workers, gogc)
            grouped[key] = []
            for rep in range(1, args.repeats + 1):
                label = f"w{workers}_gogc-{gogc}_r{rep}"
                out_dir = run_root / label
                env: dict[str, str] = {}
                if gogc != "default":
                    env["GOGC"] = gogc
                if args.gomemlimit:
                    env["GOMEMLIMIT"] = args.gomemlimit

                result = run_once(
                    binary=binary,
                    tokenizer=tokenizer,
                    input_dir=input_dir,
                    output_dir=out_dir,
                    workers=workers,
                    val_fraction=args.val_fraction,
                    env_overrides=env,
                )

                grouped[key].append(result["speed_meta"])
                all_runs.append(
                    {
                        "workers": workers,
                        "gogc": gogc,
                        "repeat": rep,
                        **result,
                    }
                )
                print(
                    f"{label}: {result['speed_meta']:.0f} tok/s "
                    f"(elapsed={result['elapsed_secs']:.2f}s wall={result['wall_secs']:.2f}s)"
                )

    summary = {
        f"workers={w},gogc={g}": summarize(vals)
        for (w, g), vals in grouped.items()
    }

    report = {
        "runs": all_runs,
        "summary": summary,
    }
    report_path = run_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    print("\nSummary:")
    for k, stats in summary.items():
        print(
            f"- {k}: mean={stats['mean']:.0f}, median={stats['median']:.0f}, "
            f"min={stats['min']:.0f}, max={stats['max']:.0f}, stdev={stats['stdev']:.0f}"
        )
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
