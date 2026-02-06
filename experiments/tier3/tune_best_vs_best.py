#!/usr/bin/env python3
"""
Tune Adam and Chi2 optimizers and compare best-vs-best.

Runs lightweight sweeps over optimizer-specific hyperparameters, picks the
best config for each optimizer on a chosen metric/mode, then runs a final
head-to-head evaluation with more seeds.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_cmd(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def get_score(result: dict, mode: str, metric: str) -> float:
    return float(result[mode][metric]["mean"])


def make_base_cmd(dataset: str, root: Path, args) -> list[str]:
    if dataset == "digits":
        script = root / "train_digits_geo_system.py"
        return [
            sys.executable,
            str(script),
            "--epochs", str(args.epochs),
            "--seeds", str(args.seeds),
            "--batch-size", str(args.batch_size),
            "--reg-lambda", str(args.reg_lambda),
            "--reg-warmup-epochs", str(args.reg_warmup_epochs),
        ]
    script = root / "train_fashion_geo_system.py"
    return [
        sys.executable,
        str(script),
        "--epochs", str(args.epochs),
        "--seeds", str(args.seeds),
        "--batch-size", str(args.batch_size),
        "--reg-lambda", str(args.reg_lambda),
        "--reg-warmup-epochs", str(args.reg_warmup_epochs),
        "--max-train", str(args.max_train),
        "--data-dir", str(args.data_dir),
    ]


def candidate_grids():
    adam = [
        {"lr": 3e-4},
        {"lr": 7e-4},
        {"lr": 1e-3},
        {"lr": 2e-3},
    ]
    chi2 = []
    for trust_radius, reject_tol, sigma, q_low, q_high, eta_proj_scale, eta_resid_scale in itertools.product(
        [0.01, 0.03, 0.05, 0.08],
        [0.02, 0.05],
        [2.5, 3.0, 4.0],
        [-0.5, -0.2, 0.0],
        [0.2, 0.4],
        [1.1, 1.2],
        [0.8, 0.9],
    ):
        chi2.append(
            {
                "trust_radius": trust_radius,
                "chi2_reject_tol": reject_tol,
                "chi2_sigma": sigma,
                "chi2_q_low": q_low,
                "chi2_q_high": q_high,
                "chi2_q_beta": 0.9,
                "chi2_shrink": 0.5,
                "chi2_grow": 1.01,
                "chi2_min_radius": 1e-4,
                "chi2_max_radius": 1.0,
                "chi2_eta_proj_scale": eta_proj_scale,
                "chi2_eta_resid_scale": eta_resid_scale,
            }
        )
    return adam, chi2


def sweep(dataset: str, mode: str, metric: str, root: Path, out_dir: Path, args):
    adam_grid, chi2_grid = candidate_grids()

    # keep search bounded
    chi2_grid = chi2_grid[: args.max_chi2_trials]

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = make_base_cmd(dataset, root, args)

    best = {
        "adam": {"score": -1e18, "cfg": None, "path": None},
        "chi2": {"score": -1e18, "cfg": None, "path": None},
    }

    # Adam sweep
    for i, cfg in enumerate(adam_grid, 1):
        out = out_dir / f"tune_{dataset}_adam_{stamp}_{i}.json"
        cmd = base + [
            "--optimizer", "adam",
            "--lr", str(cfg["lr"]),
            "--out", str(out),
        ]
        run_cmd(cmd)
        res = load_json(out)
        score = get_score(res, mode, metric)
        print(f"[adam #{i}] score={score:.4f} cfg={cfg}")
        if score > best["adam"]["score"]:
            best["adam"] = {"score": score, "cfg": cfg, "path": str(out)}

    # Chi2 sweep
    for i, cfg in enumerate(chi2_grid, 1):
        out = out_dir / f"tune_{dataset}_chi2_{stamp}_{i}.json"
        cmd = base + [
            "--optimizer", "chi2",
            "--lr", str(args.lr_chi2),
            "--trust-radius", str(cfg["trust_radius"]),
            "--chi2-adaptive",
            "--chi2-reject-tol", str(cfg["chi2_reject_tol"]),
            "--chi2-shrink", str(cfg["chi2_shrink"]),
            "--chi2-grow", str(cfg["chi2_grow"]),
            "--chi2-min-radius", str(cfg["chi2_min_radius"]),
            "--chi2-max-radius", str(cfg["chi2_max_radius"]),
            "--chi2-sigma", str(cfg["chi2_sigma"]),
            "--chi2-q-low", str(cfg["chi2_q_low"]),
            "--chi2-q-high", str(cfg["chi2_q_high"]),
            "--chi2-q-beta", str(cfg["chi2_q_beta"]),
            "--chi2-eta-proj-scale", str(cfg["chi2_eta_proj_scale"]),
            "--chi2-eta-resid-scale", str(cfg["chi2_eta_resid_scale"]),
            "--chi2-forget-tol", str(args.chi2_forget_tol),
            "--chi2-forget-shrink", str(args.chi2_forget_shrink),
            "--out", str(out),
        ]
        cmd += ["--chi2-per-layer" if args.chi2_per_layer else "--no-chi2-per-layer"]
        cmd += ["--chi2-eta-shape" if args.chi2_eta_shape else "--no-chi2-eta-shape"]
        cmd += ["--chi2-forget-guard" if args.chi2_forget_guard else "--no-chi2-forget-guard"]
        cmd += ["--eval-best-val" if args.eval_best_val else "--eval-last"]
        run_cmd(cmd)
        res = load_json(out)
        score = get_score(res, mode, metric)
        print(f"[chi2 #{i}] score={score:.4f} cfg={cfg}")
        if score > best["chi2"]["score"]:
            best["chi2"] = {"score": score, "cfg": cfg, "path": str(out)}

    return best


def final_compare(dataset: str, best: dict, root: Path, out_dir: Path, args):
    base = make_base_cmd(dataset, root, args)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    adam_out = out_dir / f"best_{dataset}_adam_final_{stamp}.json"
    chi2_out = out_dir / f"best_{dataset}_chi2_final_{stamp}.json"

    adam_cfg = best["adam"]["cfg"]
    chi2_cfg = best["chi2"]["cfg"]

    # final eval with more seeds
    final_base = [x for x in base]
    # replace seeds/epochs with final values
    def override(cmd, key, value):
        i = cmd.index(key)
        cmd[i + 1] = str(value)

    override(final_base, "--seeds", args.final_seeds)
    override(final_base, "--epochs", args.final_epochs)

    cmd_adam = final_base + [
        "--optimizer", "adam",
        "--lr", str(adam_cfg["lr"]),
        "--out", str(adam_out),
    ]
    run_cmd(cmd_adam)

    cmd_chi2 = final_base + [
        "--optimizer", "chi2",
        "--lr", str(args.lr_chi2),
        "--trust-radius", str(chi2_cfg["trust_radius"]),
        "--chi2-adaptive",
        "--chi2-reject-tol", str(chi2_cfg["chi2_reject_tol"]),
        "--chi2-shrink", str(chi2_cfg["chi2_shrink"]),
        "--chi2-grow", str(chi2_cfg["chi2_grow"]),
        "--chi2-min-radius", str(chi2_cfg["chi2_min_radius"]),
        "--chi2-max-radius", str(chi2_cfg["chi2_max_radius"]),
        "--chi2-sigma", str(chi2_cfg["chi2_sigma"]),
        "--chi2-q-low", str(chi2_cfg["chi2_q_low"]),
        "--chi2-q-high", str(chi2_cfg["chi2_q_high"]),
        "--chi2-q-beta", str(chi2_cfg["chi2_q_beta"]),
        "--chi2-eta-proj-scale", str(chi2_cfg.get("chi2_eta_proj_scale", args.chi2_eta_proj_scale)),
        "--chi2-eta-resid-scale", str(chi2_cfg.get("chi2_eta_resid_scale", args.chi2_eta_resid_scale)),
        "--chi2-forget-tol", str(args.chi2_forget_tol),
        "--chi2-forget-shrink", str(args.chi2_forget_shrink),
        "--out", str(chi2_out),
    ]
    cmd_chi2 += ["--chi2-per-layer" if args.chi2_per_layer else "--no-chi2-per-layer"]
    cmd_chi2 += ["--chi2-eta-shape" if args.chi2_eta_shape else "--no-chi2-eta-shape"]
    cmd_chi2 += ["--chi2-forget-guard" if args.chi2_forget_guard else "--no-chi2-forget-guard"]
    cmd_chi2 += ["--eval-best-val" if args.eval_best_val else "--eval-last"]
    run_cmd(cmd_chi2)

    return adam_out, chi2_out


def print_final(dataset: str, mode: str, metric: str, adam_res: dict, chi2_res: dict):
    print("\n" + "=" * 80)
    print(f"FINAL BEST-vs-BEST ({dataset})")
    print("=" * 80)
    for m in ["val_acc_ep3", "val_acc_ep5", "clean_acc", "robust_avg", "acc_border", "acc_center", "acc_contrast"]:
        a = adam_res[mode][m]["mean"]
        c = chi2_res[mode][m]["mean"]
        print(f"{m:12s} | adam {a:.4f} | chi2 {c:.4f} | delta {c-a:+.4f}")
    print(f"\nTarget metric ({mode}/{metric}): adam={adam_res[mode][metric]['mean']:.4f}, chi2={chi2_res[mode][metric]['mean']:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["digits", "fashion"], default="fashion")
    p.add_argument("--mode", choices=["baseline", "geo_system"], default="geo_system")
    p.add_argument("--metric", choices=["clean_acc", "robust_avg", "val_acc_ep5"], default="robust_avg")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--final-epochs", type=int, default=20)
    p.add_argument("--final-seeds", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--reg-lambda", type=float, default=1e-3)
    p.add_argument("--reg-warmup-epochs", type=int, default=8)
    p.add_argument("--max-train", type=int, default=12000)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--lr-chi2", type=float, default=7e-4)
    p.add_argument("--max-chi2-trials", type=int, default=16)
    p.add_argument("--chi2-per-layer", action="store_true", default=True)
    p.add_argument("--no-chi2-per-layer", action="store_false", dest="chi2_per_layer")
    p.add_argument("--chi2-eta-shape", action="store_true", default=True)
    p.add_argument("--no-chi2-eta-shape", action="store_false", dest="chi2_eta_shape")
    p.add_argument("--chi2-eta-proj-scale", type=float, default=1.2)
    p.add_argument("--chi2-eta-resid-scale", type=float, default=0.8)
    p.add_argument("--chi2-forget-guard", action="store_true", default=True)
    p.add_argument("--no-chi2-forget-guard", action="store_false", dest="chi2_forget_guard")
    p.add_argument("--chi2-forget-tol", type=float, default=0.01)
    p.add_argument("--chi2-forget-shrink", type=float, default=0.7)
    p.add_argument("--eval-best-val", action="store_true", default=True)
    p.add_argument("--eval-last", action="store_false", dest="eval_best_val")
    p.add_argument("--out-dir", type=str, default="./results_tune")
    args = p.parse_args()

    root = Path(__file__).resolve().parent
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Tuning dataset={args.dataset}, mode={args.mode}, metric={args.metric}")
    best = sweep(args.dataset, args.mode, args.metric, root, out_dir, args)
    print("\nBest configs:")
    print(json.dumps(best, indent=2))

    adam_out, chi2_out = final_compare(args.dataset, best, root, out_dir, args)
    adam_res = load_json(adam_out)
    chi2_res = load_json(chi2_out)
    print_final(args.dataset, args.mode, args.metric, adam_res, chi2_res)

    combined = {
        "args": vars(args),
        "best": best,
        "final_adam": str(adam_out),
        "final_chi2": str(chi2_out),
    }
    summary_path = out_dir / f"best_vs_best_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(json.dumps(combined, indent=2))
    print("\nWrote:", summary_path)


if __name__ == "__main__":
    main()
