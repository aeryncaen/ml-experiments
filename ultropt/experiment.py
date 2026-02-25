"""
Grid search over optimizer hyperparameters, with baseline comparison.
Budget is in tokens so that every run sees the same amount of data.

Supports three experiment modes:
  - 'legacy'   : old three-tier (pre-UltrOpt) — for backward compat
  - 'ultropt'  : UltrOpt multi-timescale optimizer
  - 'ablation' : UltrOpt with individual components toggled off
"""

import itertools
import json
import os
import time
from dataclasses import asdict

import torch

from train import (
    BaselineConfig,
    ThreeTierConfig,
    UltrOptTrainConfig,
    train_baseline,
    train_three_tier,
    train_ultropt,
)


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

# --- Standard GPT grids ---
THREE_TIER_GRID = {
    "micro_batch_size":   [64],
    "micros_per_batch":   [2, 4, 8],
    "batches_per_super":  [2, 4],
    "ema_decay_batch":    [0.85, 0.9, 0.95],
    "ema_decay_super":    [0.9, 0.95],
    "lr_batch":           [3e-4, 1e-3, 3e-3],
    "lr_super":           [1e-5, 1e-4],
}

THREE_TIER_GRID_SMALL = {
    "micro_batch_size":   [64],
    "micros_per_batch":   [2],
    "batches_per_super":  [2, 4],
    "ema_decay_batch":    [0.85, 0.9],
    "ema_decay_super":    [0.95],
    "lr_batch":           [3e-4],
    "lr_super":           [1e-4, 3e-4],
}

# --- nGPT grids (Riemannian SGD needs much higher LRs) ---
NGPT_THREE_TIER_GRID = {
    "micro_batch_size":   [64],
    "micros_per_batch":   [2, 4],
    "batches_per_super":  [2, 4],
    "ema_decay_batch":    [0.8, 0.85, 0.9, 0.95, 0.98, 0.99],
    "ema_decay_super":    [0.9, 0.95, 0.98, 0.99],
    "lr_batch":           [0.707, 1.0, 1.414],
    "lr_super":           [0.03, 0.1, 0.3],
}

NGPT_THREE_TIER_GRID_SMALL = {
    "micro_batch_size":   [64],
    "micros_per_batch":   [2],
    "batches_per_super":  [2, 4],
    "ema_decay_batch":    [0.8, 0.85, 0.9, 0.95, 0.98, 0.99],
    "ema_decay_super":    [0.95, 0.98, 0.99],
    "lr_batch":           [0.707, 1.0, 1.414],
    "lr_super":           [0.05, 0.1, 0.3],
}


# ---------------------------------------------------------------------------
# UltrOpt grids
# ---------------------------------------------------------------------------

# Phase 1: Component ablation (small, fast)
ULTROPT_ABLATION_GRID = [
    # Each entry is a dict of overrides from defaults
    {"slerp_mode": "mean", "mars": True, "newton_schulz": True, "cautious": True, "schedule_free": True},     # no SLERP
    {"slerp_mode": "ghost_slerp", "mars": False, "newton_schulz": True, "cautious": True, "schedule_free": True},  # no MARS
    {"slerp_mode": "ghost_slerp", "mars": True, "newton_schulz": False, "cautious": True, "schedule_free": True},  # no NS
    {"slerp_mode": "ghost_slerp", "mars": True, "newton_schulz": True, "cautious": False, "schedule_free": True},  # no cautious
    {"slerp_mode": "ghost_slerp", "mars": True, "newton_schulz": True, "cautious": True, "schedule_free": False},  # no SF
    {"slerp_mode": "ghost_slerp", "mars": True, "newton_schulz": True, "cautious": True, "schedule_free": True, "error_feedback": True},  # +EF
    {"accumulate_signal": "mars_reduced"},   # accumulate mars-reduced signal
    {"accumulate_signal": "ns_preconditioned"},  # accumulate NS-preconditioned signal
]

# Phase 2: Primary hyperparameter search
ULTROPT_GRID_SMALL = {
    "lr":                [1e-3, 3e-3, 1e-2],
    "batch_lr_factor":   [0.03, 0.1, 0.3],
    "super_lr_factor":   [0.003, 0.01, 0.03],
    "micros_per_batch":  [2, 4],
    "batches_per_super": [2, 4],
}

ULTROPT_GRID_FULL = {
    "lr":                [1e-3, 3e-3, 1e-2, 3e-2],
    "batch_lr_factor":   [0.01, 0.03, 0.1, 0.3],
    "super_lr_factor":   [0.001, 0.003, 0.01, 0.03],
    "ema_decay_batch":   [0.8, 0.9, 0.95, 0.99],
    "ema_decay_super":   [0.9, 0.95, 0.99],
    "micros_per_batch":  [2, 4, 8],
    "batches_per_super": [2, 4, 8],
}

# nGPT variants (higher LRs, no weight decay)
ULTROPT_NGPT_GRID_SMALL = {
    "lr":                [3e-3, 1e-2, 3e-2, 1e-1],
    "batch_lr_factor":   [0.03, 0.1, 0.3],
    "super_lr_factor":   [0.003, 0.01, 0.03],
    "micros_per_batch":  [2, 4],
    "batches_per_super": [2, 4],
}


def autodetect_device() -> str:
    """Pick best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _grid_configs(grid: dict, base_overrides: dict | None = None):
    """Yield ThreeTierConfig for every combination in the grid."""
    keys = list(grid.keys())
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(keys, vals))
        if base_overrides:
            kw.update(base_overrides)
        yield ThreeTierConfig(**kw)


def _ultropt_grid_configs(grid: dict, base_overrides: dict | None = None):
    """Yield UltrOptTrainConfig for every combination in the grid."""
    keys = list(grid.keys())
    for vals in itertools.product(*grid.values()):
        kw = dict(zip(keys, vals))
        if base_overrides:
            kw.update(base_overrides)
        yield UltrOptTrainConfig(**kw)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_experiment(
    grid: str = "small",
    total_tokens: int = 20_000_000,
    eval_every_tokens: int = 2_000_000,
    baseline_lr: float = 3e-4,
    device: str | None = None,
    ngpt: bool = False,
    data_dir: str = "../tokenized",
    tokenizer_dir: str = "../experiments/yamit/tokenizer/artifacts/yamit",
    seq_len: int = 256,
    batch_size: int = 64,
    compile: bool = True,
):
    """
    Run baseline + grid of three-tier configs.
    All runs see exactly `total_tokens` tokens.
    Returns a list of result dicts.
    """
    if device is None:
        device = autodetect_device()
    model_tag = "nGPT" if ngpt else "GPT"
    print(f"Device: {device}")
    results = []
    base_overrides = {
        "total_tokens": total_tokens,
        "eval_every_tokens": eval_every_tokens,
        "device": device,
        "ngpt": ngpt,
        "data_dir": data_dir,
        "tokenizer_dir": tokenizer_dir,
        "seq_len": seq_len,
        "compile": compile,
    }

    # --- Baseline ---
    print("=" * 70)
    print(f"BASELINE ({model_tag})")
    print("=" * 70)
    bcfg = BaselineConfig(
        batch_size=batch_size,
        lr=baseline_lr,
        total_tokens=total_tokens,
        eval_every_tokens=eval_every_tokens,
        device=device,
        ngpt=ngpt,
        data_dir=data_dir,
        tokenizer_dir=tokenizer_dir,
        seq_len=seq_len,
        compile=compile,
    )
    blog, bmodel, tok_meta = train_baseline(bcfg)
    best_val = min(entry["val_loss"] for entry in blog)
    results.append({
        "name": "baseline",
        "config": asdict(bcfg),
        "best_val_loss": best_val,
        "final_val_loss": blog[-1]["val_loss"],
        "final_train_loss": blog[-1]["train_loss"],
        "elapsed": blog[-1]["elapsed"],
        "log": blog,
    })

    print(f"\n  Baseline best val: {best_val:.4f}\n")

    # --- Three-tier grid ---
    if ngpt:
        chosen_grid = NGPT_THREE_TIER_GRID_SMALL if grid == "small" else NGPT_THREE_TIER_GRID
    else:
        chosen_grid = THREE_TIER_GRID_SMALL if grid == "small" else THREE_TIER_GRID
    configs = list(_grid_configs(chosen_grid, base_overrides))
    print(f"\nRunning {len(configs)} three-tier configurations...\n")

    for i, cfg in enumerate(configs):
        label = (
            f"mpb{cfg.micros_per_batch}_bps{cfg.batches_per_super}_"
            f"lrB{cfg.lr_batch:.0e}_lrS{cfg.lr_super:.0e}_"
            f"db{cfg.ema_decay_batch}_ds{cfg.ema_decay_super}"
        )
        print("=" * 70)
        print(f"THREE-TIER [{i+1}/{len(configs)}]  {label}")
        print("=" * 70)

        tlog, tmodel, _ = train_three_tier(cfg)
        best_val = min(entry["val_loss"] for entry in tlog)
        results.append({
            "name": label,
            "config": asdict(cfg),
            "best_val_loss": best_val,
            "final_val_loss": tlog[-1]["val_loss"],
            "final_train_loss": tlog[-1]["train_loss"],
            "elapsed": tlog[-1]["elapsed"],
            "log": tlog,
        })

    return results


# ---------------------------------------------------------------------------
# UltrOpt experiment runner
# ---------------------------------------------------------------------------

def run_ultropt_experiment(
    mode: str = "ablation",
    grid: str = "small",
    total_tokens: int = 20_000_000,
    eval_every_tokens: int = 2_000_000,
    baseline_lr: float = 3e-4,
    device: str | None = None,
    ngpt: bool = False,
    data_dir: str = "../tokenized",
    tokenizer_dir: str = "../experiments/yamit/tokenizer/artifacts/yamit",
    seq_len: int = 256,
    batch_size: int = 64,
    compile: bool = True,
    # UltrOpt-specific overrides
    ultropt_overrides: dict | None = None,
):
    """
    Run baseline + UltrOpt experiments.

    Modes:
      'default'   : baseline + single UltrOpt with default config
      'ablation'  : baseline + UltrOpt default + each component toggled off
      'grid'      : baseline + grid search over primary hyperparams

    Returns a list of result dicts.
    """
    if device is None:
        device = autodetect_device()
    model_tag = "nGPT" if ngpt else "GPT"
    print(f"Device: {device}")
    results = []

    base_overrides = {
        "total_tokens": total_tokens,
        "eval_every_tokens": eval_every_tokens,
        "device": device,
        "ngpt": ngpt,
        "data_dir": data_dir,
        "tokenizer_dir": tokenizer_dir,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "compile": compile,
    }
    if ngpt:
        base_overrides["weight_decay"] = 0.0
        base_overrides["grad_clip"] = None
    if ultropt_overrides:
        base_overrides.update(ultropt_overrides)

    # --- Baseline (AdamW) ---
    print("=" * 70)
    print(f"BASELINE AdamW ({model_tag})")
    print("=" * 70)
    bcfg = BaselineConfig(
        batch_size=batch_size,
        lr=baseline_lr,
        total_tokens=total_tokens,
        eval_every_tokens=eval_every_tokens,
        device=device,
        ngpt=ngpt,
        data_dir=data_dir,
        tokenizer_dir=tokenizer_dir,
        seq_len=seq_len,
        compile=compile,
    )
    blog, bmodel, tok_meta = train_baseline(bcfg)
    best_val = min(entry["val_loss"] for entry in blog)
    results.append({
        "name": "baseline_adamw",
        "config": asdict(bcfg),
        "best_val_loss": best_val,
        "final_val_loss": blog[-1]["val_loss"],
        "final_train_loss": blog[-1]["train_loss"],
        "elapsed": blog[-1]["elapsed"],
        "log": blog,
    })
    print(f"\n  Baseline best val: {best_val:.4f}\n")

    if mode == "default" or mode == "ablation":
        # --- UltrOpt default config ---
        print("=" * 70)
        print(f"ULTROPT DEFAULT ({model_tag})")
        print("=" * 70)
        ucfg = UltrOptTrainConfig(**base_overrides)
        ulog, umodel, _ = train_ultropt(ucfg)
        best_val = min(entry["val_loss"] for entry in ulog)
        results.append({
            "name": "ultropt_default",
            "config": asdict(ucfg),
            "best_val_loss": best_val,
            "final_val_loss": ulog[-1]["val_loss"],
            "final_train_loss": ulog[-1]["train_loss"],
            "elapsed": ulog[-1]["elapsed"],
            "log": ulog,
        })
        print(f"\n  UltrOpt default best val: {best_val:.4f}\n")

    if mode == "ablation":
        # --- Ablation: toggle each component off ---
        for i, overrides in enumerate(ULTROPT_ABLATION_GRID):
            ablation_kw = {**base_overrides, **overrides}
            cfg = UltrOptTrainConfig(**ablation_kw)

            # Build descriptive label
            diff_parts = []
            for k, v in overrides.items():
                diff_parts.append(f"{k}={v}")
            label = "ablation_" + "_".join(diff_parts)

            print("=" * 70)
            print(f"ULTROPT ABLATION [{i+1}/{len(ULTROPT_ABLATION_GRID)}]  {label}")
            print("=" * 70)

            alog, amodel, _ = train_ultropt(cfg)
            best_val = min(entry["val_loss"] for entry in alog)
            results.append({
                "name": label,
                "config": asdict(cfg),
                "best_val_loss": best_val,
                "final_val_loss": alog[-1]["val_loss"],
                "final_train_loss": alog[-1]["train_loss"],
                "elapsed": alog[-1]["elapsed"],
                "log": alog,
            })

    elif mode == "grid":
        # --- Grid search over primary hyperparams ---
        if ngpt:
            chosen_grid = ULTROPT_NGPT_GRID_SMALL if grid == "small" else ULTROPT_NGPT_GRID_SMALL
        else:
            chosen_grid = ULTROPT_GRID_SMALL if grid == "small" else ULTROPT_GRID_FULL

        configs = list(_ultropt_grid_configs(chosen_grid, base_overrides))
        print(f"\nRunning {len(configs)} UltrOpt configurations...\n")

        for i, cfg in enumerate(configs):
            label = (
                f"lr{cfg.lr:.0e}_bf{cfg.batch_lr_factor}_sf{cfg.super_lr_factor}_"
                f"mpb{cfg.micros_per_batch}_bps{cfg.batches_per_super}"
            )
            print("=" * 70)
            print(f"ULTROPT [{i+1}/{len(configs)}]  {label}")
            print("=" * 70)

            glog, gmodel, _ = train_ultropt(cfg)
            best_val = min(entry["val_loss"] for entry in glog)
            results.append({
                "name": label,
                "config": asdict(cfg),
                "best_val_loss": best_val,
                "final_val_loss": glog[-1]["val_loss"],
                "final_train_loss": glog[-1]["train_loss"],
                "elapsed": glog[-1]["elapsed"],
                "log": glog,
            })

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(results):
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'Name':<55s} {'Best Val':>9s} {'Final Val':>10s} {'Time':>7s}")
    print("-" * 90)

    for r in sorted(results, key=lambda r: r["best_val_loss"]):
        print(
            f"{r['name']:<55s} "
            f"{r['best_val_loss']:>9.4f} "
            f"{r['final_val_loss']:>10.4f} "
            f"{r['elapsed']:>6.1f}s"
        )

    best = min(results, key=lambda r: r["best_val_loss"])
    print(f"\nBest config: {best['name']}  (val_loss = {best['best_val_loss']:.4f})")


def save_results(results, path="results.json"):
    out = []
    for r in results:
        r2 = {k: v for k, v in r.items()}
        out.append(r2)
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nResults saved to {path}")
