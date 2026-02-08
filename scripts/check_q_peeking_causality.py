#!/usr/bin/env python3
"""Check Q-peeking receptive-field leakage and KV-cache compatibility.

This script runs diagnostics tailored for local-acausal Q-peeking:

1) Future-perturbation test (per-position):
   Change tokens in suffix [split, T), then measure output drift at each
   prefix position. For local peeking, leakage should be concentrated near
   the split boundary.

2) Prefix-consistency test (decode/KV-cache proxy):
   Compare output at position t when computed from:
      a) full sequence x[:, :T]
      b) prefix-only sequence x[:, :t+1]
   For strictly causal models these match. For Q-peeking, mismatch is
   expected near boundary positions, but should be bounded in radius.

The report includes a bounded-leakage verdict:
  - "far_prefix" region: [0, split - expected_radius)
  - "boundary_band" region: [split - expected_radius, split)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Literal, cast

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from ulb import ULBBlock, ULBConfig, StackedULB


def build_model(dim: int, n_heads: int, n_layers: int, paired: bool, attn_mode: str, q_mix: str) -> torch.nn.Module:
    mode = cast(Literal["softmax", "silu2", "blend"], attn_mode)
    qmix = cast(Literal["lerp", "conv2", "conv3"], q_mix)
    cfg = ULBConfig(d_model=dim, n_heads=n_heads, paired=paired, attn_mode=mode, q_mix=qmix)
    return StackedULB(lambda: ULBBlock(cfg), n_layers=n_layers, dim=dim)


@torch.no_grad()
def run_checks(
    model: torch.nn.Module,
    x: torch.Tensor,
    split: int,
    expected_radius: int,
) -> dict:
    model.eval()

    # Full pass baseline
    y_full = model(x)

    # 1) Future perturbation check
    x_perturbed = x.clone()
    x_perturbed[:, split:, :] = torch.randn_like(x_perturbed[:, split:, :])
    y_perturbed = model(x_perturbed)
    future_d = (y_full[:, :split, :] - y_perturbed[:, :split, :]).abs()  # (B, split, D)
    future_pos_max = future_d.amax(dim=(0, 2)) if split > 0 else torch.tensor([])
    future_pos_mean = future_d.mean(dim=(0, 2)) if split > 0 else torch.tensor([])

    # 2) Prefix-consistency check (cache-compat proxy)
    per_t_max = []
    per_t_mean = []
    for t in range(x.shape[1] - 1):
        y_prefix_t = model(x[:, : t + 1, :])[:, -1, :]
        y_full_t = y_full[:, t, :]
        diff = (y_prefix_t - y_full_t).abs()
        per_t_max.append(diff.max().item())
        per_t_mean.append(diff.mean().item())

    # Region split for bounded-leakage interpretation
    far_end = max(0, split - expected_radius)

    def _slice_stats(pos_max: torch.Tensor, pos_mean: torch.Tensor, start: int, end: int) -> dict:
        if end <= start:
            return {
                "start": start,
                "end": end,
                "max_abs_diff": 0.0,
                "mean_abs_diff": 0.0,
                "positions": 0,
            }
        sm = pos_max[start:end]
        smean = pos_mean[start:end]
        return {
            "start": start,
            "end": end,
            "max_abs_diff": sm.max().item(),
            "mean_abs_diff": smean.mean().item(),
            "positions": int(end - start),
        }

    future_far = _slice_stats(future_pos_max, future_pos_mean, 0, far_end)
    future_band = _slice_stats(future_pos_max, future_pos_mean, far_end, split)

    # Empirical leakage radius from future perturbation: farthest changed pos from split
    changed = (future_pos_max > 0).nonzero(as_tuple=True)[0] if split > 0 else torch.tensor([])
    if changed.numel() == 0:
        empirical_radius = 0
    else:
        earliest_changed = int(changed.min().item())
        empirical_radius = max(0, split - earliest_changed)

    return {
        "future_perturbation": {
            "split": split,
            "prefix_max_abs_diff": future_d.max().item() if split > 0 else 0.0,
            "prefix_mean_abs_diff": future_d.mean().item() if split > 0 else 0.0,
            "per_position_max_abs_diff": future_pos_max.tolist(),
            "per_position_mean_abs_diff": future_pos_mean.tolist(),
            "far_prefix": future_far,
            "boundary_band": future_band,
            "empirical_leakage_radius": empirical_radius,
        },
        "prefix_consistency": {
            "positions_tested": len(per_t_max),
            "max_abs_diff_over_t": max(per_t_max) if per_t_max else 0.0,
            "mean_abs_diff_over_t": sum(per_t_mean) / len(per_t_mean) if per_t_mean else 0.0,
            "per_t_max_abs_diff": per_t_max,
            "per_t_mean_abs_diff": per_t_mean,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether Q-peeking breaks causality/KV-cache assumptions.")
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--attn-mode", type=str, default="blend", choices=["softmax", "silu2", "blend"])
    parser.add_argument("--q-mix", type=str, default="lerp", choices=["lerp", "conv2", "conv3"])
    parser.add_argument("--paired", action="store_true", help="Use paired-head attention.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold", type=float, default=1e-6, help="Numerical threshold for zero-diff checks.")
    parser.add_argument(
        "--expected-radius",
        type=int,
        default=None,
        help="Expected leakage radius for local peeking. Default: n_layers.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    if args.expected_radius is None:
        lookahead = 1 if args.q_mix in ("lerp", "conv2") else 2
        expected_radius = args.n_layers * lookahead
    else:
        expected_radius = max(0, args.expected_radius)

    model = build_model(
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        paired=args.paired,
        attn_mode=args.attn_mode,
        q_mix=args.q_mix,
    ).to(device)

    x = torch.randn(args.batch_size, args.seq_len, args.dim, device=device)
    split = args.seq_len // 2

    metrics = run_checks(model, x, split, expected_radius)

    strict_causal_ok = (
        metrics["future_perturbation"]["prefix_max_abs_diff"] <= args.threshold
        and metrics["prefix_consistency"]["max_abs_diff_over_t"] <= args.threshold
    )

    bounded_leakage_ok = (
        metrics["future_perturbation"]["far_prefix"]["max_abs_diff"] <= args.threshold
        and metrics["future_perturbation"]["empirical_leakage_radius"] <= expected_radius
    )

    report = {
        "config": {
            "dim": args.dim,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "attn_mode": args.attn_mode,
            "q_mix": args.q_mix,
            "paired": args.paired,
            "seed": args.seed,
            "device": str(device),
            "threshold": args.threshold,
            "expected_radius": expected_radius,
        },
        "metrics": metrics,
        "strict_causal_pass": strict_causal_ok,
        "bounded_leakage_pass": bounded_leakage_ok,
        "interpretation": {
            "strict_causal": (
                "PASS: outputs are prefix-only (strictly causal, standard KV-cache compatible)"
                if strict_causal_ok
                else "FAIL: outputs depend on future tokens"
            ),
            "bounded_leakage": (
                "PASS: leakage is boundary-local and within expected receptive field"
                if bounded_leakage_ok
                else "FAIL: leakage reaches too far into prefix (or radius larger than expected)"
            ),
        },
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
