#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _parse_csv_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_seed_list(s: str) -> list[int]:
    out = []
    for tok in _parse_csv_list(s):
        out.append(int(tok))
    if not out:
        raise ValueError("No seeds provided")
    return out


def _safe_mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(statistics.fmean(xs))


def _safe_std(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return 0.0 if xs else None
    return float(statistics.pstdev(xs))


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _collect_run_record(run_dir: Path, optimizer: str, seed: int, adapt_mode: str | None) -> dict | None:
    summary = _read_json(run_dir / "run_summary.json")
    if summary is None:
        return None
    run_cfg = _read_json(run_dir / "run_config.json") or {}
    hparams = run_cfg.get("hparams", {}) if isinstance(run_cfg, dict) else {}

    last_eval = summary.get("last_eval", {}) if isinstance(summary.get("last_eval"), dict) else {}
    last_train = summary.get("last_train", {}) if isinstance(summary.get("last_train"), dict) else {}

    return {
        "run_dir": str(run_dir),
        "optimizer": optimizer,
        "seed": seed,
        "adapt_mode": adapt_mode,
        "hparams": hparams,
        "best_val_loss": summary.get("best_val_loss"),
        "final_val_loss": last_eval.get("val_loss"),
        "final_train_loss": last_train.get("train_loss"),
        "tokens_seen": summary.get("tokens_seen"),
        "total_tokens": summary.get("total_tokens"),
        "elapsed_s": summary.get("elapsed_s"),
    }


def _aggregate(records: list[dict]) -> dict:
    by_opt: dict[str, list[dict]] = {}
    for r in records:
        opt = str(r.get("optimizer"))
        mode = r.get("adapt_mode")
        key = f"{opt}:{mode}" if mode else opt
        by_opt.setdefault(key, []).append(r)

    out: dict[str, dict] = {}
    for opt, rows in sorted(by_opt.items()):
        best_vals = [float(r["best_val_loss"]) for r in rows if isinstance(r.get("best_val_loss"), (int, float))]
        final_vals = [float(r["final_val_loss"]) for r in rows if isinstance(r.get("final_val_loss"), (int, float))]
        elapsed = [float(r["elapsed_s"]) for r in rows if isinstance(r.get("elapsed_s"), (int, float))]
        out[opt] = {
            "n_runs": len(rows),
            "best_val_loss_mean": _safe_mean(best_vals),
            "best_val_loss_std": _safe_std(best_vals),
            "final_val_loss_mean": _safe_mean(final_vals),
            "final_val_loss_std": _safe_std(final_vals),
            "elapsed_s_mean": _safe_mean(elapsed),
            "elapsed_s_std": _safe_std(elapsed),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run FineWeb trainer for multiple seeds and optimizers")
    ap.add_argument("--trainer", type=str, default="experiments/tier4/train_fineweb_vanilla.py")
    ap.add_argument("--python", type=str, default=sys.executable)
    ap.add_argument("--metrics-dir", type=str, default="experiments/tier4/runs")
    ap.add_argument("--run-prefix", type=str, default="fineweb10m")
    ap.add_argument("--optimizers", type=str, default="normuon,autonormuon,muon")
    ap.add_argument("--autonormuon-modes", type=str, default="", help="Comma-separated adapt modes for autonormuon, e.g. gnorm,mu_var")
    ap.add_argument("--seeds", type=str, default="1,2,3")
    ap.add_argument("--train-steps", type=int, default=611)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--set-env",
        action="append",
        default=[],
        help="Additional ENV overrides as KEY=VALUE (repeatable)",
    )
    args = ap.parse_args()

    metrics_dir = Path(args.metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    trainer = Path(args.trainer)
    if not trainer.exists():
        raise FileNotFoundError(f"trainer not found: {trainer}")

    optimizers = _parse_csv_list(args.optimizers)
    autonormuon_modes = _parse_csv_list(args.autonormuon_modes)
    seeds = _parse_seed_list(args.seeds)

    extra_env: dict[str, str] = {}
    for item in args.set_env:
        if "=" not in item:
            raise ValueError(f"Invalid --set-env entry (expected KEY=VALUE): {item}")
        k, v = item.split("=", 1)
        extra_env[k.strip()] = v.strip()

    records: list[dict] = []
    for opt in optimizers:
        mode_list = [None]
        if opt == "autonormuon" and autonormuon_modes:
            mode_list = autonormuon_modes

        for seed in seeds:
            for adapt_mode in mode_list:
                mode_suffix = f"_m{adapt_mode}" if adapt_mode else ""
                run_name = f"{args.run_prefix}_{opt}{mode_suffix}_s{seed}"
                run_dir = metrics_dir / run_name

                if run_dir.exists() and args.overwrite:
                    shutil.rmtree(run_dir)

                if run_dir.exists() and not args.overwrite:
                    print(f"[skip] exists: {run_name}")
                    rec = _collect_run_record(run_dir, opt, seed, adapt_mode)
                    if rec is not None:
                        records.append(rec)
                    continue

                env = os.environ.copy()
                env.update(
                    {
                        "OPTIMIZER": opt,
                        "SEED": str(seed),
                        "TRAIN_STEPS": str(args.train_steps),
                        "METRICS_ENABLED": "1",
                        "METRICS_DIR": str(metrics_dir),
                        "METRICS_RUN_NAME": run_name,
                    }
                )
                if adapt_mode:
                    env["AUTONORMUON_ADAPT_MODE"] = adapt_mode
                env.update(extra_env)

                cmd = [args.python, str(trainer)]
                extras = [f"OPTIMIZER={opt}", f"SEED={seed}", f"RUN={run_name}"]
                if adapt_mode:
                    extras.append(f"AUTONORMUON_ADAPT_MODE={adapt_mode}")
                print("[run]", " ".join(cmd), *extras)
                if args.dry_run:
                    continue

                subprocess.run(cmd, env=env, check=True)
                rec = _collect_run_record(run_dir, opt, seed, adapt_mode)
                if rec is not None:
                    records.append(rec)

    report = {
        "generated_at": datetime.now().isoformat(),
        "trainer": str(trainer),
        "python": args.python,
        "cwd": str(Path.cwd()),
        "metrics_dir": str(metrics_dir),
        "run_prefix": args.run_prefix,
        "optimizers": optimizers,
        "autonormuon_modes": autonormuon_modes,
        "seeds": seeds,
        "train_steps": args.train_steps,
        "extra_env": extra_env,
        "records": records,
        "by_optimizer": _aggregate(records),
    }

    out_path = metrics_dir / f"{args.run_prefix}_multiseed_summary.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
