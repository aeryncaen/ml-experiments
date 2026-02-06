#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path


def bool_flag_args(k: str, v: bool) -> list[str]:
    if k == "geo_init_center":
        return [] if v else ["--geo-init-no-center"]
    if k == "geo_init_match_row_norm":
        return [] if v else ["--geo-init-no-row-norm-match"]
    if v:
        return [f"--{k.replace('_', '-')}"]
    return []


def kv_to_args(k: str, v) -> list[str]:
    if isinstance(v, bool):
        return bool_flag_args(k, v)
    return [f"--{k.replace('_', '-')}", str(v)]


def config_to_args(cfg: dict) -> list[str]:
    args = []
    for k in sorted(cfg.keys()):
        args.extend(kv_to_args(k, cfg[k]))
    return args


def candidate_space() -> dict[str, list]:
    return {
        "geo_init_method": ["bucket", "kl_bucket", "eig"],
        "geo_init_blend": [0.6, 0.8, 0.9],
        "geo_init_fullspace": [False, True],
        "geo_init_ridge": [1e-4, 1e-3, 1e-2],
        "geo_init_center": [True],
        "geo_init_match_row_norm": [True],
        "geo_attn_bias": [False, True],
        "geo_attn_bias_blend": [0.2, 0.5, 0.8],
        "geo_attn_corr_bias": [False, True],
        "geo_attn_corr_blend": [0.1, 0.2, 0.4],
        "geo_attn_corr_rank": [8, 16, 32],
        "geo_attn_corr_layers": [1, 2, 4],
        "geo_attn_corr_horizons": ["1", "1,2", "1,2,3"],
        "geo_attn_corr_horizon_weights": ["1.0", "1.0,0.5", "1.0,0.5,0.25"],
        "geo_embed_grad_shape": [False, True],
        "geo_embed_grad_rank": [8, 16, 32],
        "geo_embed_grad_perp_init": [0.05, 0.1, 0.2],
        "geo_embed_grad_hold_steps": [100, 250, 400],
        "geo_embed_grad_ramp_steps": [100, 250, 400],
        "geo_embed_reanchor_every": [0, 50],
        "geo_embed_reanchor_rho": [0.0, 0.01, 0.02],
        "geo_embed_reanchor_until_step": [0, 300],
    }


def is_valid(cfg: dict) -> bool:
    hz = [x for x in cfg["geo_attn_corr_horizons"].split(",") if x]
    hw = [x for x in cfg["geo_attn_corr_horizon_weights"].split(",") if x]
    if len(hz) != len(hw):
        return False
    if (not cfg["geo_attn_corr_bias"]) and (
        cfg["geo_attn_corr_horizons"] != "1,2" or cfg["geo_attn_corr_horizon_weights"] != "1.0,0.5"
    ):
        return False
    if (not cfg["geo_attn_bias"]) and cfg["geo_attn_bias_blend"] != 0.2:
        return False
    if not cfg["geo_embed_grad_shape"]:
        defaults = {
            "geo_embed_grad_rank": 16,
            "geo_embed_grad_perp_init": 0.1,
            "geo_embed_grad_hold_steps": 250,
            "geo_embed_grad_ramp_steps": 250,
        }
        for k, v in defaults.items():
            if cfg[k] != v:
                return False
    if cfg["geo_embed_reanchor_every"] == 0 and (
        cfg["geo_embed_reanchor_rho"] != 0.0 or cfg["geo_embed_reanchor_until_step"] != 0
    ):
        return False
    return True


def iter_grid(space: dict[str, list]):
    keys = sorted(space.keys())
    for vals in itertools.product(*[space[k] for k in keys]):
        cfg = {k: v for k, v in zip(keys, vals)}
        if is_valid(cfg):
            yield cfg


def sample_random(space: dict[str, list], n: int, rng: random.Random) -> list[dict]:
    keys = sorted(space.keys())
    out = []
    seen = set()
    attempts = 0
    while len(out) < n and attempts < n * 200:
        attempts += 1
        cfg = {k: rng.choice(space[k]) for k in keys}
        if not is_valid(cfg):
            continue
        sig = json.dumps(cfg, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(cfg)
    return out


def stable_id(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def load_ep1_metrics(result_json: Path) -> dict:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    runs = [r for r in data.get("runs", []) if r.get("mode") == "geo_system"]
    ep1_acc = []
    ep1_loss = []
    final_acc = []
    for r in runs:
        h = r.get("history", [])
        if h:
            ep1_acc.append(float(h[0]["val_acc"]))
            ep1_loss.append(float(h[0]["val_loss"]))
        final_acc.append(float(r.get("final_val_acc", float("nan"))))
    if not ep1_acc:
        raise ValueError("No geo_system epoch-1 metrics found in result JSON")
    return {
        "ep1_val_acc_mean": statistics.fmean(ep1_acc),
        "ep1_val_acc_std": statistics.pstdev(ep1_acc) if len(ep1_acc) > 1 else 0.0,
        "ep1_val_loss_mean": statistics.fmean(ep1_loss),
        "ep1_val_loss_std": statistics.pstdev(ep1_loss) if len(ep1_loss) > 1 else 0.0,
        "final_val_acc_mean": statistics.fmean(final_acc) if final_acc else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Hyperparameter/config search for best Shakespeare epoch-1 geo results")
    p.add_argument("--strategy", choices=["random", "grid"], default="random")
    p.add_argument("--max-runs", type=int, default=64)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument(
        "--runner",
        type=str,
        default="experiments/tier4/train_shakespeare_transformer.py",
    )
    p.add_argument("--out-dir", type=str, default="experiments/tier4/hparam_search_ep1")
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    leaderboard_path = out_dir / "leaderboard_ep1.json"
    best_path = out_dir / "best_ep1.json"

    common = {
        "epochs": 1,
        "steps_per_epoch": 300,
        "eval_iters": 50,
        "batch_size": 512,
        "block_size": 64,
        "d_model": 64,
        "n_head": 4,
        "n_layer": 4,
        "seeds": 5,
        "optimizer": "adam",
        "loss_type": "ce",
        "eta_shape": True,
        "eta_dist_tau": 2.0,
        "eta_min_resid_weight": 0.05,
        "geo_only": True,
    }

    space = candidate_space()
    rng = random.Random(args.seed)
    if args.strategy == "grid":
        candidates = list(itertools.islice(iter_grid(space), args.max_runs))
    else:
        candidates = sample_random(space, args.max_runs, rng)

    done_ids = set()
    trials = []
    if args.resume and trials_path.exists():
        for line in trials_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            trials.append(rec)
            done_ids.add(rec["id"])

    for i, cfg in enumerate(candidates, start=1):
        cid = stable_id(cfg)
        if cid in done_ids:
            continue
        run_dir = out_dir / f"run_{i:04d}_{cid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result_json = run_dir / "result.json"
        stdout_log = run_dir / "stdout.log"
        stderr_log = run_dir / "stderr.log"

        cmd = [args.python, args.runner]
        cmd.extend(config_to_args(common))
        cmd.extend(config_to_args(cfg))
        cmd.extend(["--out", str(result_json)])

        rec = {
            "id": cid,
            "run_index": i,
            "config": cfg,
            "command": cmd,
            "status": "pending",
        }

        if args.dry_run:
            rec["status"] = "dry_run"
        else:
            with stdout_log.open("w", encoding="utf-8") as so, stderr_log.open("w", encoding="utf-8") as se:
                rc = subprocess.run(cmd, stdout=so, stderr=se).returncode
            if rc != 0:
                rec["status"] = "failed"
                rec["returncode"] = rc
            elif not result_json.exists():
                rec["status"] = "missing_result"
            else:
                try:
                    rec["metrics"] = load_ep1_metrics(result_json)
                    rec["status"] = "ok"
                except Exception as e:
                    rec["status"] = "parse_error"
                    rec["error"] = str(e)

        with trials_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        trials.append(rec)
        done_ids.add(cid)

    ok = [t for t in trials if t.get("status") == "ok"]
    ok_sorted = sorted(
        ok,
        key=lambda t: (
            -t["metrics"]["ep1_val_acc_mean"],
            t["metrics"]["ep1_val_loss_mean"],
            -t["metrics"]["final_val_acc_mean"],
        ),
    )

    leaderboard = {
        "strategy": args.strategy,
        "requested_runs": args.max_runs,
        "completed_ok": len(ok_sorted),
        "top": ok_sorted[: min(25, len(ok_sorted))],
    }
    leaderboard_path.write_text(json.dumps(leaderboard, indent=2), encoding="utf-8")

    if ok_sorted:
        best = ok_sorted[0]
        best_path.write_text(json.dumps(best, indent=2), encoding="utf-8")
        print(f"Best ep1 val_acc: {best['metrics']['ep1_val_acc_mean']:.6f}")
        print(f"Best config id: {best['id']}")
    else:
        print("No successful runs to rank")

    print(f"Wrote trials: {trials_path}")
    print(f"Wrote leaderboard: {leaderboard_path}")
    if best_path.exists():
        print(f"Wrote best: {best_path}")


if __name__ == "__main__":
    main()
