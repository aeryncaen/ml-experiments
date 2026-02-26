#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            out.append(json.loads(s))
        except Exception:
            continue
    return out


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _as_float(x) -> float | None:
    return float(x) if _is_num(x) else None


def _as_int(x) -> int | None:
    if isinstance(x, int):
        return x
    return int(float(x)) if _is_num(x) else None


def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _moving_average(vals: list[float], window: int) -> list[float]:
    if not vals:
        return []
    w = max(1, int(window))
    if w == 1:
        return vals[:]
    out = []
    for i in range(len(vals)):
        lo = max(0, i - w // 2)
        hi = min(len(vals), i + w // 2 + 1)
        out.append(statistics.fmean(vals[lo:hi]))
    return out


def _segment_bounds(n: int, start_frac: float, end_frac: float) -> tuple[int, int]:
    if n < 2:
        return 0, n
    a = int(max(0.0, min(1.0, start_frac)) * n)
    b = int(max(0.0, min(1.0, end_frac)) * n)
    if b <= a:
        b = min(n, a + 2)
    if b - a < 2:
        a = max(0, b - 2)
    return a, b


def _linear_fit(x: list[float], y: list[float]) -> tuple[float | None, float | None]:
    n = min(len(x), len(y))
    if n < 2:
        return None, None
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    num = 0.0
    den = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mx
        num += dx * (yi - my)
        den += dx * dx
    if den <= 0:
        return None, None
    slope = num / den
    intercept = my - slope * mx
    return slope, intercept


def _slope(x: list[float], y: list[float]) -> float | None:
    s, _ = _linear_fit(x, y)
    return s


def _residual_std(x: list[float], y: list[float]) -> float | None:
    s, b = _linear_fit(x, y)
    if s is None or b is None:
        return None
    res = [yi - (s * xi + b) for xi, yi in zip(x, y)]
    return statistics.pstdev(res) if len(res) > 1 else 0.0


def _curvature(x: list[float], y: list[float]) -> float | None:
    n = min(len(x), len(y))
    if n < 3:
        return None
    vals = []
    for i in range(1, n - 1):
        dx1 = x[i] - x[i - 1]
        dx2 = x[i + 1] - x[i]
        if dx1 <= 0 or dx2 <= 0:
            continue
        d1 = (y[i] - y[i - 1]) / dx1
        d2 = (y[i + 1] - y[i]) / dx2
        vals.append((d2 - d1) / max(0.5 * (dx1 + dx2), 1e-12))
    if not vals:
        return None
    return statistics.fmean(vals)


def _first_idx_leq(vals: list[float], threshold: float) -> int | None:
    for i, v in enumerate(vals):
        if v <= threshold:
            return i
    return None


def _first_idx_below_slope(
    x: list[float],
    y: list[float],
    start_idx: int,
    window: int,
    slope_abs_threshold: float,
    patience: int,
) -> int | None:
    n = min(len(x), len(y))
    if n < 3:
        return None
    w = max(2, int(window))
    p = max(1, int(patience))
    i = max(0, min(start_idx, max(0, n - w)))
    consec = 0
    first_ok = None
    while i <= n - w:
        s = _slope(x[i:i + w], y[i:i + w])
        ok = s is not None and abs(s) <= slope_abs_threshold
        if ok:
            if first_ok is None:
                first_ok = i
            consec += 1
            if consec >= p:
                return first_ok
        else:
            consec = 0
            first_ok = None
        i += 1
    return None


def _count_spikes(vals: list[float], factor: float, warmup: int, min_abs_delta: float) -> int:
    if len(vals) < max(2, warmup + 1):
        return 0
    c = 0
    for i in range(warmup, len(vals)):
        p = vals[i - 1]
        v = vals[i]
        if p > 0 and v > p * factor and (v - p) >= min_abs_delta:
            c += 1
    return c


def _pearson(x: list[float], y: list[float]) -> float | None:
    n = min(len(x), len(y))
    if n < 3:
        return None
    x = x[:n]
    y = y[:n]
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    den = denx * deny
    if den <= 0:
        return None
    return num / den


def _auto_summary(train_events: list[dict], mode: str, min_ratio: float) -> dict:
    if not train_events:
        return {}

    def _group_avg(ev: dict, key: str) -> float | None:
        groups = ev.get("optimizer_groups")
        if not isinstance(groups, list):
            return None
        vals = []
        for g in groups:
            if not isinstance(g, dict):
                continue
            if not bool(g.get("use_muon", False)):
                continue
            v = _as_float(g.get(key))
            if v is not None:
                vals.append(v)
        if not vals:
            return None
        return statistics.fmean(vals)

    ratio_key_map = {
        "gnorm": "ratio_gnorm",
        "mu_var": "ratio_muvar",
        "hybrid": "ratio_hybrid",
        "cv": "ratio_cv",
        "surge": "ratio_surge",
    }
    active_key = ratio_key_map.get(mode, "signal_ratio")

    fields = [
        "scheduled_lr",
        "signal_ratio",
        "lr_mult",
        "gnorm_mean",
        "gnorm_std",
        "gnorm_cv",
        "gnorm_surge",
        "ratio_gnorm",
        "ratio_muvar",
        "ratio_cv",
        "ratio_surge",
        "conflict_frac",
    ]
    series: dict[str, list[float]] = {k: [] for k in fields}
    for ev in train_events:
        for k in fields:
            v = _group_avg(ev, k)
            if v is not None and math.isfinite(v):
                series[k].append(v)
            else:
                series[k].append(float("nan"))

    def _clean(name: str) -> list[float]:
        return [v for v in series[name] if math.isfinite(v)]

    ratio_vals = _clean(active_key if active_key in series else "signal_ratio")
    lr_mult_vals = _clean("lr_mult")
    sched_vals = _clean("scheduled_lr")
    cv_vals = _clean("gnorm_cv")
    surge_vals = _clean("gnorm_surge")
    conflict_vals = _clean("conflict_frac")

    ratio_floor_hits = sum(1 for v in ratio_vals if v <= (min_ratio + 1e-6))
    ratio_ceiling_hits = sum(1 for v in ratio_vals if v >= 0.999)
    ratio_jumps = sum(1 for i in range(1, len(ratio_vals)) if abs(ratio_vals[i] - ratio_vals[i - 1]) > 0.2)
    lr_jumps = sum(1 for i in range(1, len(sched_vals)) if sched_vals[i - 1] > 0 and (sched_vals[i] / sched_vals[i - 1] > 1.5 or sched_vals[i] / sched_vals[i - 1] < 0.67))
    conflict_spikes = sum(1 for v in conflict_vals if v > 0.05)

    return {
        "active_ratio_key": active_key,
        "ratio_mean": statistics.fmean(ratio_vals) if ratio_vals else None,
        "ratio_p10": statistics.quantiles(ratio_vals, n=10)[0] if len(ratio_vals) >= 10 else None,
        "ratio_p90": statistics.quantiles(ratio_vals, n=10)[8] if len(ratio_vals) >= 10 else None,
        "ratio_floor_hit_frac": (ratio_floor_hits / len(ratio_vals)) if ratio_vals else None,
        "ratio_ceiling_hit_frac": (ratio_ceiling_hits / len(ratio_vals)) if ratio_vals else None,
        "lr_mult_mean": statistics.fmean(lr_mult_vals) if lr_mult_vals else None,
        "scheduled_lr_mean": statistics.fmean(sched_vals) if sched_vals else None,
        "gnorm_cv_mean": statistics.fmean(cv_vals) if cv_vals else None,
        "gnorm_surge_mean": statistics.fmean(surge_vals) if surge_vals else None,
        "conflict_frac_mean": statistics.fmean(conflict_vals) if conflict_vals else None,
        "corr_ratio_cv": _pearson(ratio_vals[: len(cv_vals)], cv_vals[: len(ratio_vals)]) if ratio_vals and cv_vals else None,
        "corr_ratio_surge": _pearson(ratio_vals[: len(surge_vals)], surge_vals[: len(ratio_vals)]) if ratio_vals and surge_vals else None,
        "decision_anomalies": {
            "ratio_jump_count": ratio_jumps,
            "lr_jump_count": lr_jumps,
            "conflict_spike_count": conflict_spikes,
        },
    }


def _analyze_run(run_dir: Path, args) -> dict | None:
    summary = _read_json(run_dir / "run_summary.json")
    if summary is None:
        return None
    cfg = _read_json(run_dir / "run_config.json") or {}
    hparams = cfg.get("hparams", {}) if isinstance(cfg, dict) else {}

    config_hash = hparams.get("run_config_hash")
    if not (isinstance(config_hash, str) and config_hash):
        config_hash = hashlib.sha1(json.dumps(hparams, sort_keys=True).encode("utf-8")).hexdigest()[:12] if hparams else "-"

    train_events = _read_jsonl(run_dir / "train_events.jsonl")
    eval_events = _read_jsonl(run_dir / "eval_events.jsonl")

    train_rows = []
    for e in train_events:
        loss = _as_float(e.get("train_loss"))
        if loss is None:
            continue
        st = _as_int(e.get("step"))
        tok = _as_float(e.get("tokens_seen"))
        el = _as_float(e.get("elapsed_s"))
        gnorm = _as_float(e.get("grad_norm"))
        lr = _as_float(e.get("lr"))
        train_rows.append({"step": st, "tokens": tok, "elapsed": el, "loss": loss, "gnorm": gnorm, "lr": lr, "raw": e})

    if not train_rows:
        return None

    losses = [r["loss"] for r in train_rows]
    gnorms = [r["gnorm"] for r in train_rows if r["gnorm"] is not None]
    lrs = [r["lr"] for r in train_rows if r["lr"] is not None]
    tokens = [r["tokens"] if r["tokens"] is not None else float(i) for i, r in enumerate(train_rows)]
    steps = [r["step"] if r["step"] is not None else i for i, r in enumerate(train_rows)]
    elapsed = [r["elapsed"] for r in train_rows]

    y = _moving_average(losses, args.smooth_window)
    x_mtok = [t / 1_000_000.0 for t in tokens]

    # Early/random phase
    vocab = _as_int(hparams.get("vocab_size"))
    random_ref = math.log(float(vocab)) if vocab and vocab > 1 else None
    random_exit_idx = None
    if random_ref is not None:
        threshold = random_ref * (1.0 - args.random_margin)
        random_exit_idx = _first_idx_leq(y, threshold)

    # Stable regime
    total_drop = y[0] - y[-1]
    stable_drop_idx = None
    if total_drop > 0:
        target = y[0] - args.stable_drop_frac * total_drop
        stable_drop_idx = _first_idx_leq(y, target)

    slope_start = int(len(y) * args.stable_min_frac)
    stable_slope_idx = _first_idx_below_slope(
        x_mtok,
        y,
        start_idx=slope_start,
        window=args.stable_slope_window,
        slope_abs_threshold=args.stable_slope_abs,
        patience=args.stable_slope_patience,
    )

    # Curve shape
    e0, e1 = _segment_bounds(len(y), 0.0, args.early_frac)
    m0, m1 = _segment_bounds(len(y), args.mid_start_frac, args.mid_end_frac)
    t0, t1 = _segment_bounds(len(y), max(0.0, 1.0 - args.tail_frac), 1.0)

    early_slope = _slope(x_mtok[e0:e1], y[e0:e1])
    mid_slope = _slope(x_mtok[m0:m1], y[m0:m1])
    tail_slope = _slope(x_mtok[t0:t1], y[t0:t1])
    mid_curv = _curvature(x_mtok[m0:m1], y[m0:m1])
    mid_wiggle = _residual_std(x_mtok[m0:m1], y[m0:m1])
    tail_wiggle = _residual_std(x_mtok[t0:t1], y[t0:t1])
    taper = (abs(tail_slope) / max(abs(mid_slope), 1e-12)) if (mid_slope is not None and tail_slope is not None) else None

    # End-game gain
    end_gain_abs = y[t0] - y[-1] if len(y) >= 2 else None
    end_gain_pct = (end_gain_abs / total_drop) if (end_gain_abs is not None and total_drop > 0) else None

    # Anomalies
    train_loss_spikes = _count_spikes(losses, args.loss_spike_factor, args.spike_warmup, args.loss_spike_min_abs)
    grad_spikes = _count_spikes(gnorms, args.grad_spike_factor, args.spike_warmup, args.grad_spike_min_abs)
    lr_spikes = _count_spikes(lrs, args.lr_spike_factor, args.spike_warmup, 0.0)
    tail_rebounds = 0
    for i in range(max(t0 + 1, 1), len(y)):
        if y[i] - y[i - 1] > args.tail_rebound_abs:
            tail_rebounds += 1
    nonfinite_train = sum(1 for e in train_events if not _is_num(e.get("train_loss")))
    nonfinite_grad = sum(1 for e in train_events if not _is_num(e.get("grad_norm")))
    nonfinite_eval = sum(1 for e in eval_events if not _is_num(e.get("val_loss")))

    unstable = (
        nonfinite_train > 0
        or nonfinite_grad > 0
        or nonfinite_eval > 0
        or train_loss_spikes > args.max_loss_spikes
        or grad_spikes > args.max_grad_spikes
        or tail_rebounds > args.max_tail_rebounds
    )

    # Optimizer decision diagnostics (autonormuon)
    optimizer = hparams.get("optimizer")
    adapt_mode = hparams.get("autonormuon_adapt_mode") if optimizer == "autonormuon" else None
    auto_diag = {}
    if optimizer == "autonormuon":
        auto_diag = _auto_summary(train_events, str(adapt_mode or "gnorm"), _as_float(hparams.get("autonormuon_min_ratio")) or 0.0)

    def _idx_to_step(i: int | None) -> int | None:
        return steps[i] if i is not None and 0 <= i < len(steps) else None

    def _idx_to_tokens(i: int | None) -> int | None:
        return int(tokens[i]) if i is not None and 0 <= i < len(tokens) else None

    def _idx_to_elapsed(i: int | None) -> float | None:
        return elapsed[i] if i is not None and 0 <= i < len(elapsed) else None

    best_val = _as_float(summary.get("best_val_loss"))
    final_val = None
    if isinstance(summary.get("last_eval"), dict):
        final_val = _as_float(summary["last_eval"].get("val_loss"))

    if best_val is None:
        vals = [_as_float(e.get("val_loss")) for e in eval_events]
        vals = [v for v in vals if v is not None]
        best_val = min(vals) if vals else None
        final_val = vals[-1] if vals else final_val

    _elapsed = _as_float(summary.get("elapsed_s"))
    _tokens_seen = _as_float(summary.get("tokens_seen"))
    _throughput = None
    if _elapsed is not None and _tokens_seen is not None and _elapsed > 0:
        _throughput = _tokens_seen / _elapsed

    rec = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "optimizer": optimizer,
        "adapt_mode": adapt_mode,
        "seed": _as_int(hparams.get("seed")),
        "config_hash": config_hash,
        "train_steps": _as_int(hparams.get("train_steps")),
        "batch_size": _as_int(hparams.get("batch_size")),
        "grad_accum": _as_int(hparams.get("grad_accum")),
        "seq_len": _as_int(hparams.get("seq_len")),
        "n_layer": _as_int(hparams.get("n_layer")),
        "d_model": _as_int(hparams.get("d_model")),
        "best_val_loss": best_val,
        "final_val_loss": final_val,
        "elapsed_s": _elapsed,
        "throughput_tok_s": _throughput,
        "random_loss_ref": random_ref,
        "random_exit_step": _idx_to_step(random_exit_idx),
        "random_exit_tokens": _idx_to_tokens(random_exit_idx),
        "stable_step_drop": _idx_to_step(stable_drop_idx),
        "stable_tokens_drop": _idx_to_tokens(stable_drop_idx),
        "stable_elapsed_drop_s": _idx_to_elapsed(stable_drop_idx),
        "stable_step_slope": _idx_to_step(stable_slope_idx),
        "stable_tokens_slope": _idx_to_tokens(stable_slope_idx),
        "stable_elapsed_slope_s": _idx_to_elapsed(stable_slope_idx),
        "early_slope_per_mtok": early_slope,
        "mid_slope_per_mtok": mid_slope,
        "tail_slope_per_mtok": tail_slope,
        "mid_curvature": mid_curv,
        "mid_wiggle_std": mid_wiggle,
        "tail_wiggle_std": tail_wiggle,
        "taper_ratio_tail_mid": taper,
        "end_gain_abs": end_gain_abs,
        "end_gain_frac_total": end_gain_pct,
        "train_loss_spikes": train_loss_spikes,
        "grad_norm_spikes": grad_spikes,
        "lr_spikes": lr_spikes,
        "tail_rebounds": tail_rebounds,
        "non_finite_train_loss": nonfinite_train,
        "non_finite_grad_norm": nonfinite_grad,
        "non_finite_eval_loss": nonfinite_eval,
        "unstable": unstable,
        "autonormuon_diag": auto_diag,
    }
    return rec


def _group_aggregate(rows: list[dict], group_fields: list[str]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for r in rows:
        key = tuple(r.get(f) for f in group_fields)
        buckets.setdefault(key, []).append(r)

    def _mean(vals: list[float | None]) -> float | None:
        xs = [v for v in vals if isinstance(v, (int, float))]
        return statistics.fmean(xs) if xs else None

    def _std(vals: list[float | None]) -> float | None:
        xs = [v for v in vals if isinstance(v, (int, float))]
        if not xs:
            return None
        return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    out = []
    for key, rs in buckets.items():
        rec = {f: v for f, v in zip(group_fields, key)}
        rec.update(
            {
                "n_runs": len(rs),
                "best_val_loss_mean": _mean([r.get("best_val_loss") for r in rs]),
                "best_val_loss_std": _std([r.get("best_val_loss") for r in rs]),
                "final_val_loss_mean": _mean([r.get("final_val_loss") for r in rs]),
                "stable_step_drop_mean": _mean([r.get("stable_step_drop") for r in rs]),
                "stable_step_slope_mean": _mean([r.get("stable_step_slope") for r in rs]),
                "random_exit_step_mean": _mean([r.get("random_exit_step") for r in rs]),
                "mid_slope_per_mtok_mean": _mean([r.get("mid_slope_per_mtok") for r in rs]),
                "tail_slope_per_mtok_mean": _mean([r.get("tail_slope_per_mtok") for r in rs]),
                "taper_ratio_tail_mid_mean": _mean([r.get("taper_ratio_tail_mid") for r in rs]),
                "end_gain_frac_total_mean": _mean([r.get("end_gain_frac_total") for r in rs]),
                "throughput_tok_s_mean": _mean([r.get("throughput_tok_s") for r in rs]),
                "unstable_runs": sum(1 for r in rs if r.get("unstable")),
            }
        )
        out.append(rec)

    out.sort(
        key=lambda r: (
            r["best_val_loss_mean"] if isinstance(r.get("best_val_loss_mean"), (int, float)) else float("inf"),
            r["final_val_loss_mean"] if isinstance(r.get("final_val_loss_mean"), (int, float)) else float("inf"),
            r["unstable_runs"],
        )
    )
    return out


def _markdown(rows: list[dict], groups: list[dict], group_fields: list[str]) -> str:
    lines = [
        "# FineWeb Dynamics Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Rank | Run | Opt | Mode | Best Val | Final Val | Rand Exit | Stable(drop) | Stable(slope) | End Gain % | Mid Slope | Tail Slope | Taper | tok/s | Unstable |",
        "|------|-----|-----|------|----------|-----------|-----------|--------------|---------------|------------|-----------|------------|-------|-------|----------|",
    ]
    for i, r in enumerate(rows, start=1):
        _eg = r.get("end_gain_frac_total")
        _eg_pct = (_eg * 100.0) if isinstance(_eg, (int, float)) else None
        lines.append(
            "| "
            f"{i} | {r.get('run_name')} | {r.get('optimizer') or '-'} | {r.get('adapt_mode') or '-'} | "
            f"{_fmt(r.get('best_val_loss'))} | {_fmt(r.get('final_val_loss'))} | {_fmt(r.get('random_exit_step'), nd=0)} | "
            f"{_fmt(r.get('stable_step_drop'), nd=0)} | {_fmt(r.get('stable_step_slope'), nd=0)} | "
            f"{_fmt(_eg_pct)} | "
            f"{_fmt(r.get('mid_slope_per_mtok'))} | {_fmt(r.get('tail_slope_per_mtok'))} | {_fmt(r.get('taper_ratio_tail_mid'))} | "
            f"{_fmt(r.get('throughput_tok_s'), nd=1)} | {r.get('unstable')} |"
        )

    lines += [
        "",
        "## Grouped Summary",
        "",
        "| Rank | " + " | ".join(group_fields) + " | Runs | Best Mean | Stable(drop) | Stable(slope) | Rand Exit | End Gain % | Mid Slope | Tail Slope | Taper | tok/s | Unstable |",
        "|------|" + "------|" * len(group_fields) + "------|----------|--------------|---------------|-----------|------------|-----------|------------|-------|-------|----------|",
    ]
    for i, g in enumerate(groups, start=1):
        gp = " | ".join(str(g.get(f, "-")) for f in group_fields)
        end_gain_pct = g.get("end_gain_frac_total_mean")
        end_gain_pct = (end_gain_pct * 100.0) if isinstance(end_gain_pct, (int, float)) else None
        lines.append(
            "| "
            f"{i} | {gp} | {_fmt(g.get('n_runs'), nd=0)} | {_fmt(g.get('best_val_loss_mean'))} | "
            f"{_fmt(g.get('stable_step_drop_mean'), nd=0)} | {_fmt(g.get('stable_step_slope_mean'), nd=0)} | "
            f"{_fmt(g.get('random_exit_step_mean'), nd=0)} | {_fmt(end_gain_pct)} | {_fmt(g.get('mid_slope_per_mtok_mean'))} | "
            f"{_fmt(g.get('tail_slope_per_mtok_mean'))} | {_fmt(g.get('taper_ratio_tail_mid_mean'))} | "
            f"{_fmt(g.get('throughput_tok_s_mean'), nd=1)} | {_fmt(g.get('unstable_runs'), nd=0)} |"
        )

    auto_rows = [r for r in rows if r.get("optimizer") == "autonormuon" and isinstance(r.get("autonormuon_diag"), dict)]
    if auto_rows:
        lines += [
            "",
            "## AutoNorMuon Decisions",
            "",
            "| Run | Mode | Ratio Key | Ratio Mean | Floor Hit % | Ceiling Hit % | CV Mean | Surge Mean | Conflict Mean | Ratio Jumps | LR Jumps | Conflict Spikes | Corr(ratio,cv) | Corr(ratio,surge) |",
            "|-----|------|-----------|------------|-------------|---------------|---------|------------|---------------|------------|----------|-----------------|----------------|-------------------|",
        ]
        for r in auto_rows:
            d = r.get("autonormuon_diag", {})
            an = d.get("decision_anomalies", {}) if isinstance(d.get("decision_anomalies"), dict) else {}
            floor_pct = d.get("ratio_floor_hit_frac")
            ceil_pct = d.get("ratio_ceiling_hit_frac")
            floor_pct = (floor_pct * 100.0) if isinstance(floor_pct, (int, float)) else None
            ceil_pct = (ceil_pct * 100.0) if isinstance(ceil_pct, (int, float)) else None
            lines.append(
                "| "
                f"{r.get('run_name')} | {r.get('adapt_mode') or '-'} | {d.get('active_ratio_key', '-')} | {_fmt(d.get('ratio_mean'))} | "
                f"{_fmt(floor_pct)} | {_fmt(ceil_pct)} | {_fmt(d.get('gnorm_cv_mean'))} | {_fmt(d.get('gnorm_surge_mean'))} | "
                f"{_fmt(d.get('conflict_frac_mean'))} | {_fmt(an.get('ratio_jump_count'), nd=0)} | {_fmt(an.get('lr_jump_count'), nd=0)} | "
                f"{_fmt(an.get('conflict_spike_count'), nd=0)} | {_fmt(d.get('corr_ratio_cv'))} | {_fmt(d.get('corr_ratio_surge'))} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Deep behavior analysis for FineWeb trainer JSON metrics")
    ap.add_argument("--runs-dir", type=str, default="experiments/tier4/runs")
    ap.add_argument("--pattern", type=str, default="*", help="run subdir glob")
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--out-md", type=str, default="")

    # stability anomaly knobs
    ap.add_argument("--loss-spike-factor", type=float, default=1.25)
    ap.add_argument("--grad-spike-factor", type=float, default=2.0)
    ap.add_argument("--lr-spike-factor", type=float, default=1.5)
    ap.add_argument("--spike-warmup", type=int, default=25)
    ap.add_argument("--loss-spike-min-abs", type=float, default=0.05)
    ap.add_argument("--grad-spike-min-abs", type=float, default=0.2)
    ap.add_argument("--max-loss-spikes", type=int, default=5)
    ap.add_argument("--max-grad-spikes", type=int, default=40)
    ap.add_argument("--tail-rebound-abs", type=float, default=0.03)
    ap.add_argument("--max-tail-rebounds", type=int, default=20)

    # behavior/dynamics knobs
    ap.add_argument("--smooth-window", type=int, default=11)
    ap.add_argument("--random-margin", type=float, default=0.2)
    ap.add_argument("--stable-drop-frac", type=float, default=0.8)
    ap.add_argument("--stable-min-frac", type=float, default=0.25)
    ap.add_argument("--stable-slope-window", type=int, default=25)
    ap.add_argument("--stable-slope-abs", type=float, default=0.02)
    ap.add_argument("--stable-slope-patience", type=int, default=3)
    ap.add_argument("--early-frac", type=float, default=0.2)
    ap.add_argument("--mid-start-frac", type=float, default=0.35)
    ap.add_argument("--mid-end-frac", type=float, default=0.7)
    ap.add_argument("--tail-frac", type=float, default=0.2)

    ap.add_argument(
        "--group-by",
        type=str,
        default="optimizer,adapt_mode,train_steps,batch_size,grad_accum,seq_len,n_layer,d_model",
        help="Comma-separated keys used for grouped summary",
    )

    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs-dir not found: {runs_dir}")

    run_dirs = [p for p in sorted(runs_dir.glob(args.pattern)) if p.is_dir()]
    rows = []
    for rd in run_dirs:
        rec = _analyze_run(rd, args)
        if rec is not None:
            rows.append(rec)

    rows.sort(
        key=lambda r: (
            r["best_val_loss"] if isinstance(r.get("best_val_loss"), (int, float)) else float("inf"),
            r["final_val_loss"] if isinstance(r.get("final_val_loss"), (int, float)) else float("inf"),
            bool(r.get("unstable")),
            str(r.get("run_name", "")),
        )
    )

    group_fields = [x.strip() for x in args.group_by.split(",") if x.strip()]
    groups = _group_aggregate(rows, group_fields)

    report = {
        "generated_at": datetime.now().isoformat(),
        "runs_dir": str(runs_dir),
        "pattern": args.pattern,
        "anomaly_rules": {
            "loss_spike_factor": args.loss_spike_factor,
            "grad_spike_factor": args.grad_spike_factor,
            "lr_spike_factor": args.lr_spike_factor,
            "spike_warmup": args.spike_warmup,
            "loss_spike_min_abs": args.loss_spike_min_abs,
            "grad_spike_min_abs": args.grad_spike_min_abs,
            "max_loss_spikes": args.max_loss_spikes,
            "max_grad_spikes": args.max_grad_spikes,
            "tail_rebound_abs": args.tail_rebound_abs,
            "max_tail_rebounds": args.max_tail_rebounds,
        },
        "dynamics_rules": {
            "smooth_window": args.smooth_window,
            "random_margin": args.random_margin,
            "stable_drop_frac": args.stable_drop_frac,
            "stable_min_frac": args.stable_min_frac,
            "stable_slope_window": args.stable_slope_window,
            "stable_slope_abs": args.stable_slope_abs,
            "stable_slope_patience": args.stable_slope_patience,
            "early_frac": args.early_frac,
            "mid_start_frac": args.mid_start_frac,
            "mid_end_frac": args.mid_end_frac,
            "tail_frac": args.tail_frac,
        },
        "n_runs": len(rows),
        "runs": rows,
        "group_by": group_fields,
        "groups": groups,
    }

    out_json = Path(args.out_json) if args.out_json else runs_dir / "leaderboard.json"
    out_md = Path(args.out_md) if args.out_md else runs_dir / "leaderboard.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(rows, groups, group_fields), encoding="utf-8")

    print(f"Analyzed runs: {len(rows)}")
    if rows:
        best = rows[0]
        print(
            f"Best run: {best.get('run_name')} | best_val={_fmt(best.get('best_val_loss'))} | "
            f"stable_drop_step={_fmt(best.get('stable_step_drop'), nd=0)} | stable_slope_step={_fmt(best.get('stable_step_slope'), nd=0)}"
        )
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote MD:   {out_md}")


if __name__ == "__main__":
    main()
