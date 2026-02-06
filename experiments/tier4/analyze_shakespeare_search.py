#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and not math.isnan(float(x))


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    num = 0.0
    dx = 0.0
    dy = 0.0
    for a, b in zip(x, y):
        va = a - mx
        vb = b - my
        num += va * vb
        dx += va * va
        dy += vb * vb
    den = (dx * dy) ** 0.5
    return num / den if den > 0 else float("nan")


def ranks(vals: list[float]) -> list[float]:
    idx = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i + 1
        while j < len(vals) and vals[idx[j]] == vals[idx[i]]:
            j += 1
        r = 0.5 * (i + j - 1) + 1.0
        for k in range(i, j):
            out[idx[k]] = r
        i = j
    return out


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


def load_trials(search_dir: Path) -> list[dict]:
    trials_path = search_dir / "trials.jsonl"
    rows: list[dict] = []
    if trials_path.exists():
        for line in trials_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("status") != "ok":
                continue
            rows.append(
                {
                    "config": rec["config"],
                    "ep1_val_acc": float(rec["metrics"]["ep1_val_acc_mean"]),
                    "ep1_val_loss": float(rec["metrics"]["ep1_val_loss_mean"]),
                    "final_val_acc": float(rec["metrics"].get("final_val_acc_mean", float("nan"))),
                    "source": "trials.jsonl",
                }
            )
        return rows

    for p in sorted(search_dir.glob("trial_*.json")):
        data = json.loads(p.read_text(encoding="utf-8"))
        cfg = data.get("config", {})
        g = data.get("geo_system", {})
        acc = g.get("final_val_acc", {}).get("mean")
        loss = g.get("final_val_loss", {}).get("mean")
        ep1 = None
        runs = [r for r in data.get("runs", []) if r.get("mode") == "geo_system"]
        if runs and runs[0].get("history"):
            ep1 = runs[0]["history"][0].get("val_acc")
        if ep1 is None:
            ep1 = acc
        if is_number(ep1) and is_number(loss):
            rows.append(
                {
                    "config": cfg,
                    "ep1_val_acc": float(ep1),
                    "ep1_val_loss": float(loss),
                    "final_val_acc": float(acc) if is_number(acc) else float("nan"),
                    "source": p.name,
                }
            )
    return rows


def analyze(rows: list[dict], top_frac: float) -> dict:
    if not rows:
        raise ValueError("No successful runs found to analyze")

    y = [r["ep1_val_acc"] for r in rows]
    base_mean = statistics.fmean(y)

    keys = sorted({k for r in rows for k in r["config"].keys()})
    numeric_stats = {}
    categorical_stats = {}

    for k in keys:
        vals = [r["config"].get(k) for r in rows]
        num_idx = [i for i, v in enumerate(vals) if is_number(v)]
        if len(num_idx) >= max(4, int(0.6 * len(rows))):
            xv = [float(vals[i]) for i in num_idx]
            yv = [y[i] for i in num_idx]
            numeric_stats[k] = {
                "pearson": pearson(xv, yv),
                "spearman": spearman(xv, yv),
                "n": len(xv),
            }
        else:
            buckets = defaultdict(list)
            for v, t in zip(vals, y):
                buckets[str(v)].append(t)
            cat_rows = []
            for label, ys in sorted(buckets.items(), key=lambda kv: statistics.fmean(kv[1]), reverse=True):
                m = statistics.fmean(ys)
                cat_rows.append(
                    {
                        "value": label,
                        "count": len(ys),
                        "ep1_val_acc_mean": m,
                        "lift_vs_global": m - base_mean,
                    }
                )
            categorical_stats[k] = cat_rows

    ranked = sorted(rows, key=lambda r: (-r["ep1_val_acc"], r["ep1_val_loss"]))
    n_top = max(1, int(len(ranked) * top_frac))
    top = ranked[:n_top]

    suggested_grid = {}
    for k in keys:
        vals = [r["config"].get(k) for r in top]
        cnt = Counter(vals)
        if not cnt:
            continue
        if all(is_number(v) for v in cnt.keys()):
            ordered = sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
            best_vals = [float(v) for v, _ in ordered[:3]]
            best_vals = sorted(set(best_vals))
            suggested_grid[k] = best_vals
        else:
            ordered = sorted(cnt.items(), key=lambda kv: (-kv[1], str(kv[0])))
            suggested_grid[k] = [v for v, _ in ordered[:3]]

    method_blend = defaultdict(list)
    for r in rows:
        m = str(r["config"].get("geo_init_method"))
        b = str(r["config"].get("geo_init_blend"))
        method_blend[(m, b)].append(r["ep1_val_acc"])
    method_blend_table = [
        {
            "geo_init_method": m,
            "geo_init_blend": b,
            "count": len(v),
            "ep1_val_acc_mean": statistics.fmean(v),
        }
        for (m, b), v in method_blend.items()
    ]
    method_blend_table.sort(key=lambda x: x["ep1_val_acc_mean"], reverse=True)

    return {
        "n_runs": len(rows),
        "global_ep1_val_acc_mean": base_mean,
        "top_frac": top_frac,
        "top_n": n_top,
        "best_run": ranked[0],
        "top_runs": ranked[: min(20, len(ranked))],
        "numeric_correlations": numeric_stats,
        "categorical_lifts": categorical_stats,
        "method_blend_table": method_blend_table,
        "suggested_grid": suggested_grid,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze Shakespeare search and suggest refined grid")
    p.add_argument("--search-dir", type=str, default="experiments/tier4/hparam_search_ep1")
    p.add_argument("--top-frac", type=float, default=0.25)
    p.add_argument("--out", type=str, default="")
    p.add_argument("--grid-out", type=str, default="")
    args = p.parse_args()

    search_dir = Path(args.search_dir)
    rows = load_trials(search_dir)
    report = analyze(rows, top_frac=args.top_frac)

    out_path = Path(args.out) if args.out else search_dir / "correlation_report_ep1.json"
    grid_path = Path(args.grid_out) if args.grid_out else search_dir / "suggested_grid_ep1.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    grid_path.write_text(json.dumps(report["suggested_grid"], indent=2), encoding="utf-8")

    best = report["best_run"]
    print(f"Analyzed runs: {report['n_runs']}")
    print(f"Best ep1 val_acc: {best['ep1_val_acc']:.6f}")
    print(f"Best ep1 val_loss: {best['ep1_val_loss']:.6f}")
    print(f"Wrote report: {out_path}")
    print(f"Wrote suggested grid: {grid_path}")


if __name__ == "__main__":
    main()
