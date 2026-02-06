#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import itertools
import json
import random
import runpy
import shlex
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


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
        "geo_init_method": ["bucket", "kl_bucket", "kl_bucket_mtp", "eig"],
        "geo_init_mtp_weights": ["1.0,0.5,0.25", "1.0,0.7,0.4", "1.0,0.5,0.25,0.125"],
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
    if cfg["geo_init_method"] != "kl_bucket_mtp" and cfg["geo_init_mtp_weights"] != "1.0,0.5,0.25":
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


def load_best_config(path: str) -> dict:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    top = data.get("top", [])
    if not top:
        raise ValueError(f"No top entries in {path}")
    return dict(top[0]["config"])


def binary_values(base: float, lo: float, hi: float, levels: int, as_int: bool) -> list:
    vals = [float(base)]
    span = hi - lo
    for i in range(1, levels + 1):
        step = span / (2 ** i)
        vals.append(max(lo, min(hi, float(base) - step)))
        vals.append(max(lo, min(hi, float(base) + step)))
    if as_int:
        out = sorted({int(round(v)) for v in vals})
    else:
        out = sorted({round(float(v), 8) for v in vals})
    return out


def build_binary_candidates(base_cfg: dict, keys: list[str], levels: int, max_runs: int, rng: random.Random) -> list[dict]:
    bounds = {
        "geo_init_blend": (0.1, 0.95, False),
        "geo_init_ridge": (1e-5, 1e-1, False),
        "geo_attn_bias_blend": (0.05, 0.95, False),
        "geo_attn_corr_blend": (0.05, 0.6, False),
        "geo_attn_corr_rank": (4, 32, True),
        "geo_attn_corr_layers": (1, 4, True),
        "geo_embed_grad_perp_init": (0.01, 0.4, False),
        "geo_embed_grad_hold_steps": (50, 500, True),
        "geo_embed_grad_ramp_steps": (50, 500, True),
        "geo_embed_grad_rank": (4, 32, True),
        "geo_embed_reanchor_rho": (0.0, 0.05, False),
    }

    value_space: dict[str, list] = {}
    for k in keys:
        if k not in base_cfg or k not in bounds:
            continue
        lo, hi, as_int = bounds[k]
        value_space[k] = binary_values(float(base_cfg[k]), lo, hi, levels, as_int)

    # Start with the base config itself.
    cands = [dict(base_cfg)]

    # Coordinate-wise binary perturbations around base.
    for k, vals in value_space.items():
        for v in vals:
            cfg = dict(base_cfg)
            cfg[k] = v
            cands.append(cfg)

    # Pairwise combinations for strongest interacting params.
    pair_keys = [k for k in ["geo_init_blend", "geo_attn_corr_blend", "geo_embed_grad_perp_init", "geo_embed_grad_hold_steps"] if k in value_space]
    for i in range(len(pair_keys)):
        for j in range(i + 1, len(pair_keys)):
            k1, k2 = pair_keys[i], pair_keys[j]
            for v1 in value_space[k1]:
                for v2 in value_space[k2]:
                    cfg = dict(base_cfg)
                    cfg[k1] = v1
                    cfg[k2] = v2
                    cands.append(cfg)

    # Deduplicate + keep valid
    seen = set()
    uniq = []
    for cfg in cands:
        s = json.dumps(cfg, sort_keys=True)
        if s in seen:
            continue
        seen.add(s)
        if is_valid(cfg):
            uniq.append(cfg)

    if len(uniq) <= max_runs:
        return uniq
    rng.shuffle(uniq)
    return uniq[:max_runs]


def load_value_counts(trials_path: Path) -> dict[str, Counter]:
    out: dict[str, Counter] = defaultdict(Counter)
    if not trials_path.exists():
        return out
    for line in trials_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") != "ok":
            continue
        cfg = rec.get("config", {})
        for k, v in cfg.items():
            out[k][json.dumps(v, sort_keys=True)] += 1
    return out


def build_binary_candidates_adaptive(
    base_cfg: dict,
    keys: list[str],
    levels: int,
    max_runs: int,
    rng: random.Random,
    space: dict[str, list],
    value_counts: dict[str, Counter],
    min_count: int,
) -> list[dict]:
    cands = build_binary_candidates(base_cfg, keys, levels, max_runs=10_000_000, rng=rng)

    # Add one-factor categorical/boolean coverage points for under-sampled values.
    for k in keys:
        if k not in space or k not in base_cfg:
            continue
        vals = space[k]
        if not vals:
            continue
        if all(isinstance(v, (int, float)) for v in vals):
            continue
        for v in vals:
            if v == base_cfg[k]:
                continue
            cnt = value_counts.get(k, Counter()).get(json.dumps(v, sort_keys=True), 0)
            if cnt <= min_count:
                cfg = dict(base_cfg)
                cfg[k] = v
                if is_valid(cfg):
                    cands.append(cfg)

    # De-duplicate.
    uniq = []
    seen = set()
    for cfg in cands:
        s = json.dumps(cfg, sort_keys=True)
        if s in seen:
            continue
        seen.add(s)
        uniq.append(cfg)

    if max_runs > 0 and len(uniq) > max_runs:
        rng.shuffle(uniq)
        return uniq[:max_runs]
    return uniq


def stable_id(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def run_with_heartbeat(
    cmd: list[str],
    stdout_log: Path,
    stderr_log: Path,
    heartbeat_sec: int,
    stream_child: bool,
) -> int:
    if stream_child:
        print("child:", shlex.join(cmd), flush=True)
        return subprocess.run(cmd).returncode

    with stdout_log.open("w", encoding="utf-8") as so, stderr_log.open("w", encoding="utf-8") as se:
        proc = subprocess.Popen(cmd, stdout=so, stderr=se)
        t0 = time.perf_counter()
        last_hb = t0
        while True:
            rc = proc.poll()
            if rc is not None:
                return rc
            now = time.perf_counter()
            if heartbeat_sec > 0 and (now - last_hb) >= heartbeat_sec:
                print(
                    f"  still running... elapsed={now - t0:.1f}s (logs: {stdout_log.name}, {stderr_log.name})",
                    flush=True,
                )
                last_hb = now
            time.sleep(0.5)


def run_inprocess(
    runner: str,
    runner_args: list[str],
    stdout_log: Path,
    stderr_log: Path,
    stream_child: bool,
) -> int:
    old_argv = sys.argv[:]
    sys.argv = [runner, *runner_args]
    try:
        if stream_child:
            print("inprocess:", runner, shlex.join(runner_args), flush=True)
            runpy.run_path(runner, run_name="__main__")
            return 0
        with stdout_log.open("w", encoding="utf-8") as so, stderr_log.open("w", encoding="utf-8") as se:
            with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
                runpy.run_path(runner, run_name="__main__")
        return 0
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    except Exception:
        with stderr_log.open("a", encoding="utf-8") as se:
            traceback.print_exc(file=se)
        return 1
    finally:
        sys.argv = old_argv


def load_runner_module(runner_path: str):
    p = Path(runner_path)
    spec = importlib.util.spec_from_file_location("tier4_runner", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load runner module from {runner_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def normalize_runner_cfg(mod, merged: dict) -> dict:
    cfg = dict(merged)
    if "geo_attn_corr_horizons" in cfg and isinstance(cfg["geo_attn_corr_horizons"], str):
        cfg["geo_attn_corr_horizons"] = tuple(mod.parse_int_list_csv(cfg["geo_attn_corr_horizons"]))
    if "geo_attn_corr_horizon_weights" in cfg and isinstance(cfg["geo_attn_corr_horizon_weights"], str):
        cfg["geo_attn_corr_horizon_weights"] = tuple(mod.parse_float_list_csv(cfg["geo_attn_corr_horizon_weights"]))
    return cfg


def key_hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def run_native_trial(
    mod,
    common: dict,
    cfg: dict,
    result_json: Path,
    stdout_log: Path,
    stderr_log: Path,
    stream_child: bool,
    ctx: dict,
) -> int:
    def _execute() -> None:
        merged = dict(common)
        merged.update(cfg)
        merged = normalize_runner_cfg(mod, merged)
        run_cfg_kwargs = {k: v for k, v in merged.items() if k in mod.RunConfig.__dataclass_fields__}
        run_cfg = mod.RunConfig(**run_cfg_kwargs)

        geo_key = (
            run_cfg.geo_init_method,
            run_cfg.geo_init_mtp_weights,
            run_cfg.d_model,
        )
        if geo_key not in ctx["geo_basis_cache"]:
            geo_disk_key = {
                "method": run_cfg.geo_init_method,
                "mtp": run_cfg.geo_init_mtp_weights,
                "d_model": run_cfg.d_model,
                "vocab_size": int(ctx["vocab_size"]),
                "train_len": int(len(ctx["train_ids"])),
            }
            geo_cache_file = ctx["cache_dir"] / f"geo_basis_{key_hash(geo_disk_key)}.npy"
            if geo_cache_file.exists():
                print(f"cache hit: geo_basis {geo_cache_file.name}")
                geo_basis = mod.np.load(geo_cache_file)
            else:
                print(f"cache miss: geo_basis {geo_cache_file.name}")
                if run_cfg.geo_init_method == "eig":
                    _, eigvecs = mod.np.linalg.eigh(ctx["op"])
                    geo_basis = eigvecs[:, -run_cfg.d_model :].astype(mod.np.float32)
                elif run_cfg.geo_init_method == "kl_bucket_mtp":
                    mtp_w = mod.parse_float_list_csv(run_cfg.geo_init_mtp_weights)
                    if not mtp_w:
                        mtp_w = [1.0, 0.5, 0.25]
                    geo_basis = mod.compute_kl_bucket_mtp_basis(ctx["train_ids"], ctx["vocab_size"], run_cfg.d_model, mtp_w)
                elif run_cfg.geo_init_method == "kl_bucket":
                    geo_basis = mod.compute_kl_bucket_basis(ctx["train_ids"], ctx["vocab_size"], run_cfg.d_model)
                else:
                    geo_basis = mod.compute_bucket_basis(ctx["train_ids"], ctx["vocab_size"], run_cfg.d_model)
                mod.np.save(geo_cache_file, geo_basis)
            ctx["geo_basis_cache"][geo_key] = geo_basis
        geo_basis = ctx["geo_basis_cache"][geo_key]

        attn_key = (
            bool(run_cfg.geo_attn_corr_bias),
            run_cfg.geo_attn_corr_rank,
            run_cfg.geo_attn_corr_layers,
            run_cfg.geo_attn_corr_horizons,
            run_cfg.geo_attn_corr_horizon_weights,
            geo_key,
        )
        if attn_key not in ctx["attn_corr_cache"]:
            attn_corr = None
            if run_cfg.geo_attn_corr_bias:
                attn_disk_key = {
                    "geo_key": geo_key,
                    "rank": run_cfg.geo_attn_corr_rank,
                    "layers": run_cfg.geo_attn_corr_layers,
                    "h": list(run_cfg.geo_attn_corr_horizons),
                    "w": list(run_cfg.geo_attn_corr_horizon_weights),
                }
                attn_cache_file = ctx["cache_dir"] / f"attn_corr_{key_hash(attn_disk_key)}.npy"
                if attn_cache_file.exists():
                    print(f"cache hit: attn_corr {attn_cache_file.name}")
                    attn_corr = mod.np.load(attn_cache_file)
                else:
                    print(f"cache miss: attn_corr {attn_cache_file.name}")
                    attn_corr = mod.compute_attn_corr_projector(
                        ctx["train_ids"],
                        ctx["vocab_size"],
                        geo_basis,
                        run_cfg.d_model,
                        run_cfg.geo_attn_corr_rank,
                        list(run_cfg.geo_attn_corr_horizons),
                        list(run_cfg.geo_attn_corr_horizon_weights),
                    )
                    mod.np.save(attn_cache_file, attn_corr)
            ctx["attn_corr_cache"][attn_key] = attn_corr
        attn_corr_projector = ctx["attn_corr_cache"][attn_key]

        runs = []
        for seed in range(run_cfg.seeds):
            print(f"Running mode=geo_system, seed={seed}")
            res = mod.train_one(
                ctx["train_ids"],
                ctx["val_ids"],
                ctx["vocab_size"],
                geo_basis,
                attn_corr_projector,
                "geo_system",
                run_cfg,
                seed,
                ctx["device"],
            )
            runs.append(res)

        out = {
            "config": merged,
            "device": str(ctx["device"]),
            "baseline": None,
            "geo_system": mod.summarize(runs, "geo_system"),
            "runs": runs,
        }
        result_json.write_text(json.dumps(out, indent=2), encoding="utf-8")

    try:
        if stream_child:
            _execute()
        else:
            with stdout_log.open("w", encoding="utf-8") as so, stderr_log.open("w", encoding="utf-8") as se:
                with contextlib.redirect_stdout(so), contextlib.redirect_stderr(se):
                    _execute()
        return 0
    except Exception:
        with stderr_log.open("a", encoding="utf-8") as se:
            traceback.print_exc(file=se)
        return 1


def short_cfg(cfg: dict) -> str:
    keys = [
        "geo_init_method",
        "geo_init_mtp_weights",
        "geo_init_blend",
        "geo_init_fullspace",
        "geo_attn_bias",
        "geo_attn_bias_blend",
        "geo_attn_corr_bias",
        "geo_attn_corr_blend",
        "geo_attn_corr_rank",
        "geo_attn_corr_layers",
        "geo_embed_grad_shape",
        "geo_embed_grad_rank",
        "geo_embed_grad_perp_init",
        "geo_embed_reanchor_every",
        "geo_embed_reanchor_rho",
    ]
    parts = []
    for k in keys:
        if k in cfg:
            parts.append(f"{k}={cfg[k]}")
    return " ".join(parts)


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
    p.add_argument("--strategy", choices=["random", "grid", "binary"], default="random")
    p.add_argument("--max-runs", type=int, default=0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--binary-base", type=str, default="leaderboard_ep1.json")
    p.add_argument(
        "--binary-keys",
        type=str,
        default="geo_init_blend,geo_init_ridge,geo_attn_corr_blend,geo_attn_corr_rank,geo_attn_corr_layers,geo_embed_grad_perp_init,geo_embed_grad_hold_steps,geo_embed_grad_ramp_steps,geo_embed_grad_rank,geo_embed_reanchor_rho",
    )
    p.add_argument("--binary-levels", type=int, default=3)
    p.add_argument("--binary-min-count", type=int, default=2)
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--steps-per-epoch", type=int, default=300)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--python", type=str, default=sys.executable)
    p.add_argument(
        "--runner",
        type=str,
        default="experiments/tier4/train_shakespeare_transformer.py",
    )
    p.add_argument("--out-dir", type=str, default="experiments/tier4/hparam_search_ep1")
    p.add_argument("--resume", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--heartbeat-sec", type=int, default=10)
    p.add_argument("--stream-child", action="store_true", default=False)
    p.add_argument("--quick", action="store_true", default=False)
    p.add_argument("--torch-compile", action="store_true", default=False)
    p.add_argument("--subprocess", action="store_true", default=False)
    p.add_argument("--runpy", action="store_true", default=False)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    trials_path = out_dir / "trials.jsonl"
    leaderboard_path = out_dir / "leaderboard_ep1.json"
    best_path = out_dir / "best_ep1.json"

    common = {
        "epochs": 1,
        "steps_per_epoch": args.steps_per_epoch,
        "eval_iters": args.eval_iters,
        "batch_size": 512,
        "block_size": 64,
        "d_model": 64,
        "n_head": 4,
        "n_layer": 4,
        "seeds": args.seeds,
        "optimizer": "adam",
        "loss_type": "ce",
        "eta_shape": True,
        "eta_dist_tau": 2.0,
        "eta_min_resid_weight": 0.05,
        "geo_only": True,
        "torch_compile": args.torch_compile,
    }
    if args.quick:
        common["steps_per_epoch"] = 120
        common["eval_iters"] = 20
        common["seeds"] = 1

    mode = "subprocess" if args.subprocess else ("runpy" if args.runpy else "native")
    runner_mod = None
    runner_ctx = None
    if mode == "native":
        print(f"Loading runner module once: {args.runner}", flush=True)
        runner_mod = load_runner_module(args.runner)
        text = runner_mod.ensure_data(Path(common.get("data_path", "./data/tinyshakespeare/input.txt")))
        stoi, _ = runner_mod.build_vocab(text)
        ids = runner_mod.encode(text, stoi)
        n = len(ids)
        n_train = int(0.9 * n)
        train_ids = ids[:n_train]
        val_ids = ids[n_train:]
        vocab_size = len(stoi)
        op = runner_mod.compute_token_operator(train_ids, vocab_size)
        runner_ctx = {
            "device": runner_mod.get_device(),
            "train_ids": train_ids,
            "val_ids": val_ids,
            "vocab_size": vocab_size,
            "op": op,
            "geo_basis_cache": {},
            "attn_corr_cache": {},
            "cache_dir": cache_dir,
        }
        print(
            f"Native context ready: device={runner_ctx['device']} train={len(train_ids)} val={len(val_ids)} vocab={vocab_size}",
            flush=True,
        )

    space = candidate_space()
    rng = random.Random(args.seed)
    print("Generating candidate configs...", flush=True)
    trials_path_for_counts = out_dir / "trials.jsonl"
    value_counts = load_value_counts(trials_path_for_counts)
    if args.strategy == "grid":
        candidates = list(iter_grid(space))
        if args.max_runs > 0:
            candidates = candidates[: args.max_runs]
    elif args.strategy == "binary":
        base_cfg = load_best_config(args.binary_base)
        binary_keys = [k.strip() for k in args.binary_keys.split(",") if k.strip()]
        candidates = build_binary_candidates_adaptive(
            base_cfg,
            keys=binary_keys,
            levels=max(1, int(args.binary_levels)),
            max_runs=args.max_runs,
            rng=rng,
            space=space,
            value_counts=value_counts,
            min_count=max(0, int(args.binary_min_count)),
        )
        print(f"Binary base from {args.binary_base}: {base_cfg}", flush=True)
    else:
        n = args.max_runs if args.max_runs > 0 else 64
        candidates = sample_random(space, n, rng)
    print(f"Generated candidates: {len(candidates)}", flush=True)

    done_ids = set()
    trials = []
    if args.resume and trials_path.exists():
        for line in trials_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            trials.append(rec)
            done_ids.add(rec["id"])

    pending = []
    for i, cfg in enumerate(candidates, start=1):
        cid = stable_id(cfg)
        if cid in done_ids:
            continue
        pending.append((i, cfg, cid))

    total_pending = len(pending)
    print(
        f"Starting search: strategy={args.strategy} max_runs={args.max_runs} pending={total_pending} resume={args.resume} mode={mode} seeds={common['seeds']} steps={common['steps_per_epoch']} eval_iters={common['eval_iters']}",
        flush=True,
    )

    progress = tqdm(total=total_pending, desc="search_ep1", unit="run") if (tqdm is not None and total_pending > 0) else None

    for i, cfg, cid in pending:
        run_dir = out_dir / f"run_{i:04d}_{cid}"
        run_dir.mkdir(parents=True, exist_ok=True)
        result_json = run_dir / "result.json"
        stdout_log = run_dir / "stdout.log"
        stderr_log = run_dir / "stderr.log"

        runner_args = []
        runner_args.extend(config_to_args(common))
        runner_args.extend(config_to_args(cfg))
        runner_args.extend(["--out", str(result_json)])
        cmd = [args.python, args.runner, *runner_args]

        rec = {
            "id": cid,
            "run_index": i,
            "config": cfg,
            "command": cmd if args.subprocess else ["inprocess", args.runner, *runner_args],
            "status": "pending",
        }

        start_t = time.perf_counter()
        print(f"[{i}/{len(candidates)}] run_id={cid} started", flush=True)
        print(f"[{i}/{len(candidates)}] config: {short_cfg(cfg)}", flush=True)

        if args.dry_run:
            rec["status"] = "dry_run"
        else:
            if mode == "subprocess":
                rc = run_with_heartbeat(
                    cmd,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    heartbeat_sec=max(0, args.heartbeat_sec),
                    stream_child=args.stream_child,
                )
            elif mode == "runpy":
                rc = run_inprocess(
                    args.runner,
                    runner_args,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    stream_child=args.stream_child,
                )
            else:
                assert runner_mod is not None and runner_ctx is not None
                rc = run_native_trial(
                    runner_mod,
                    common,
                    cfg,
                    result_json,
                    stdout_log=stdout_log,
                    stderr_log=stderr_log,
                    stream_child=args.stream_child,
                    ctx=runner_ctx,
                )
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

        elapsed = time.perf_counter() - start_t

        with trials_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        trials.append(rec)
        done_ids.add(cid)
        if progress is not None:
            progress.update(1)
        status = rec.get("status", "unknown")
        if status == "ok":
            m = rec["metrics"]
            print(
                f"[{i}/{len(candidates)}] run_id={cid} done status=ok ep1_acc={m['ep1_val_acc_mean']:.5f} ep1_loss={m['ep1_val_loss_mean']:.5f} elapsed={elapsed:.1f}s",
                flush=True,
            )
        else:
            print(f"[{i}/{len(candidates)}] run_id={cid} done status={status} elapsed={elapsed:.1f}s", flush=True)

    if progress is not None:
        progress.close()

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
