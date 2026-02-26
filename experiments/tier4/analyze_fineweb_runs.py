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


def _metric_good_sign(metric: str) -> float:
    m = (metric or "").lower()
    # Lower-is-better metrics should flip sign so positive means "good".
    lower_better_tokens = ("loss", "error", "latency", "time", "wer", "ppl", "perplex")
    return -1.0 if any(tok in m for tok in lower_better_tokens) else 1.0


def _is_scalar_option(v) -> bool:
    return isinstance(v, (str, int, float, bool)) or v is None


def _option_value(v):
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, int):
        return int(v)
    if isinstance(v, float):
        if math.isfinite(v):
            return float(v)
        return None
    if isinstance(v, str):
        return v
    return None


def _eval_milestones(eval_events: list[dict]) -> dict:
    vals = []
    for e in eval_events:
        vl = _as_float(e.get("val_loss"))
        if vl is None:
            continue
        vals.append(
            {
                "loss": vl,
                "step": _as_int(e.get("step")),
                "tokens": _as_int(e.get("tokens_seen")),
            }
        )
    if not vals:
        return {
            "start_eval_loss": None,
            "first_eval_loss": None,
            "mid_eval_loss": None,
            "last_eval_loss": None,
            "start_eval_step": None,
            "first_eval_step": None,
            "mid_eval_step": None,
            "last_eval_step": None,
            "start_eval_tokens": None,
            "first_eval_tokens": None,
            "mid_eval_tokens": None,
            "last_eval_tokens": None,
        }

    # "start" = eval at step 0 (random-init quality).
    start = vals[0]

    # "first/mid/last" should describe training behavior, so skip step-0 eval.
    trained = [v for v in vals if ((v.get("step") or 0) > 0) or ((v.get("tokens") or 0) > 0)]
    core = trained if trained else vals

    mid_i = (len(core) - 1) // 2
    first = core[0]
    mid = core[mid_i]
    last = core[-1]
    return {
        "start_eval_loss": start["loss"],
        "first_eval_loss": first["loss"],
        "mid_eval_loss": mid["loss"],
        "last_eval_loss": last["loss"],
        "start_eval_step": start["step"],
        "first_eval_step": first["step"],
        "mid_eval_step": mid["step"],
        "last_eval_step": last["step"],
        "start_eval_tokens": start["tokens"],
        "first_eval_tokens": first["tokens"],
        "mid_eval_tokens": mid["tokens"],
        "last_eval_tokens": last["tokens"],
    }


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
    eval_ms = _eval_milestones(eval_events)

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
    final_val = eval_ms["last_eval_loss"]
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
        "lr_scope": hparams.get("autonormuon_lr_scope") if optimizer == "autonormuon" else None,
        "gnorm_source": hparams.get("autonormuon_gnorm_source") if optimizer == "autonormuon" else None,
        "gnorm_scope": hparams.get("autonormuon_gnorm_scope") if optimizer == "autonormuon" else None,
        "gmax_scope": hparams.get("autonormuon_gmax_scope") if optimizer == "autonormuon" else None,
        "second_moment_mode": hparams.get("autonormuon_second_moment_mode") if optimizer == "autonormuon" else None,
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
        "start_eval_loss": eval_ms["start_eval_loss"],
        "first_eval_loss": eval_ms["first_eval_loss"],
        "mid_eval_loss": eval_ms["mid_eval_loss"],
        "last_eval_loss": eval_ms["last_eval_loss"],
        "start_eval_step": eval_ms["start_eval_step"],
        "first_eval_step": eval_ms["first_eval_step"],
        "mid_eval_step": eval_ms["mid_eval_step"],
        "last_eval_step": eval_ms["last_eval_step"],
        "start_eval_tokens": eval_ms["start_eval_tokens"],
        "first_eval_tokens": eval_ms["first_eval_tokens"],
        "mid_eval_tokens": eval_ms["mid_eval_tokens"],
        "last_eval_tokens": eval_ms["last_eval_tokens"],
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
        "option_values": {k: _option_value(v) for k, v in hparams.items() if _is_scalar_option(v)},
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
                "first_eval_loss_mean": _mean([r.get("first_eval_loss") for r in rs]),
                "mid_eval_loss_mean": _mean([r.get("mid_eval_loss") for r in rs]),
                "last_eval_loss_mean": _mean([r.get("last_eval_loss") for r in rs]),
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


def _option_effects(rows: list[dict], option_keys: list[str], metrics: list[str]) -> list[dict]:
    effects = []
    for key in option_keys:
        by_val: dict[str, list[dict]] = {}
        for r in rows:
            opts = r.get("option_values")
            if not isinstance(opts, dict):
                continue
            if key not in opts:
                continue
            v = opts.get(key)
            if v is None:
                continue
            by_val.setdefault(str(v), []).append(r)
        if len(by_val) < 2:
            continue

        value_stats = {}
        spreads = {}
        winners = {}
        losers = {}
        for m in metrics:
            means = []
            for v, rs in by_val.items():
                vals = []
                for r in rs:
                    rv = r.get(m)
                    if isinstance(rv, (int, float)) and math.isfinite(float(rv)):
                        vals.append(float(rv))
                mean_v = statistics.fmean(vals) if vals else None
                std_v = statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None)
                value_stats.setdefault(v, {})[m] = {
                    "n": len(vals),
                    "mean": mean_v,
                    "std": std_v,
                }
                if mean_v is not None:
                    means.append((v, mean_v))
            if means:
                means.sort(key=lambda x: x[1])
                winners[m] = means[0][0]
                losers[m] = means[-1][0]
                spreads[m] = means[-1][1] - means[0][1] if len(means) > 1 else 0.0
            else:
                winners[m] = None
                losers[m] = None
                spreads[m] = None

        # Numeric correlation against each metric if option is numeric-ish
        num_x = []
        y_map = {m: [] for m in metrics}
        numeric_ok = True
        for r in rows:
            opts = r.get("option_values")
            if not isinstance(opts, dict) or key not in opts:
                continue
            v = opts.get(key)
            if isinstance(v, bool):
                xv = 1.0 if v else 0.0
            elif isinstance(v, (int, float)) and math.isfinite(float(v)):
                xv = float(v)
            else:
                numeric_ok = False
                break
            num_x.append(xv)
            for m in metrics:
                ym = r.get(m)
                y_map[m].append(float(ym) if isinstance(ym, (int, float)) and math.isfinite(float(ym)) else float("nan"))

        direct_corrs = {}
        if numeric_ok and len(num_x) >= 3:
            for m in metrics:
                pairs = [(x, y) for x, y in zip(num_x, y_map[m]) if math.isfinite(y)]
                if len(pairs) >= 3:
                    xx, yy = zip(*pairs)
                    direct_corrs[m] = _pearson(list(xx), list(yy))
                else:
                    direct_corrs[m] = None
        else:
            for m in metrics:
                direct_corrs[m] = None

        # One-vs-rest value correlation for ALL option types (including strings)
        # Positive corr => option value tends to increase loss; negative => tends to decrease loss.
        value_corrs: dict[str, list[dict]] = {m: [] for m in metrics}
        for v in sorted(by_val.keys()):
            for m in metrics:
                xx = []
                yy = []
                for r in rows:
                    opts = r.get("option_values")
                    if not isinstance(opts, dict) or key not in opts:
                        continue
                    cur = opts.get(key)
                    if cur is None:
                        continue
                    ym = r.get(m)
                    if not (isinstance(ym, (int, float)) and math.isfinite(float(ym))):
                        continue
                    xx.append(1.0 if str(cur) == v else 0.0)
                    yy.append(float(ym))

                corr_v = None
                if len(xx) >= 3 and min(xx) < max(xx):
                    corr_v = _pearson(xx, yy)

                value_corrs[m].append(
                    {
                        "value": v,
                        "corr": corr_v,
                        "n": len(yy),
                        "n_value": int(sum(xx)),
                    }
                )

        # Single summary corr per option/metric:
        # - numeric options: direct pearson(option_value, metric)
        # - categorical options: strongest (max |corr|) one-vs-rest value corr
        corrs = {}
        good_corrs = {}
        for m in metrics:
            dc = direct_corrs.get(m)
            if isinstance(dc, (int, float)) and math.isfinite(float(dc)):
                corrs[m] = float(dc)
                good_corrs[m] = _metric_good_sign(m) * float(dc)
                continue

            best_abs_corr = None
            for rec in value_corrs.get(m, []):
                c = rec.get("corr")
                if not (isinstance(c, (int, float)) and math.isfinite(float(c))):
                    continue
                c = float(c)
                if best_abs_corr is None or abs(c) > abs(best_abs_corr):
                    best_abs_corr = c
            corrs[m] = best_abs_corr
            good_corrs[m] = (_metric_good_sign(m) * best_abs_corr) if isinstance(best_abs_corr, (int, float)) else None

        score = 0.0
        for m in metrics:
            s = spreads.get(m)
            if isinstance(s, (int, float)):
                score = max(score, float(abs(s)))

        effects.append(
            {
                "option": key,
                "n_values": len(by_val),
                "n_runs": sum(len(v) for v in by_val.values()),
                "winners": winners,
                "losers": losers,
                "spreads": spreads,
                "correlations": corrs,
                "good_correlations": good_corrs,
                "direct_correlations": direct_corrs,
                "value_correlations": value_corrs,
                "by_value": value_stats,
                "impact_score": score,
            }
        )

    effects.sort(key=lambda e: e.get("impact_score", 0.0), reverse=True)
    return effects


def _corr_rankings(option_effects: list[dict], metrics: list[str], top_k: int) -> dict:
    out = {}
    k = max(1, int(top_k))
    for m in metrics:
        good_sign = _metric_good_sign(m)
        vals = []
        for e in option_effects:
            vc = e.get("value_correlations", {}).get(m) if isinstance(e.get("value_correlations"), dict) else None
            if not isinstance(vc, list):
                continue
            for rec in vc:
                c = rec.get("corr") if isinstance(rec, dict) else None
                if isinstance(c, (int, float)) and math.isfinite(float(c)):
                    c_raw = float(c)
                    vals.append(
                        {
                            "option": e.get("option"),
                            "value": rec.get("value"),
                            "corr": good_sign * c_raw,
                            "corr_raw": c_raw,
                            "n": rec.get("n"),
                            "n_value": rec.get("n_value"),
                        }
                    )

        vals.sort(key=lambda x: x.get("corr", 0.0))
        anti = vals[:k]
        corr = vals[-k:][::-1]
        out[m] = {
            "objective": "minimize" if good_sign < 0 else "maximize",
            "correlated": corr,
            "anticorrelated": anti,
        }
    return out


def _milestone_leaders(rows: list[dict]) -> dict:
    out = {}
    for m in ("first_eval_loss", "mid_eval_loss", "last_eval_loss"):
        cands = [r for r in rows if isinstance(r.get(m), (int, float))]
        if not cands:
            out[m] = None
            continue
        best = min(cands, key=lambda r: float(r[m]))
        out[m] = {
            "run_name": best.get("run_name"),
            "optimizer": best.get("optimizer"),
            "adapt_mode": best.get("adapt_mode"),
            "value": best.get(m),
        }
    return out


def _select_option_keys(rows: list[dict], option_keys_arg: str, exclude_keys: set[str]) -> list[str]:
    if option_keys_arg.strip().lower() != "auto":
        return [k.strip() for k in option_keys_arg.split(",") if k.strip()]

    values: dict[str, set[str]] = {}
    for r in rows:
        opts = r.get("option_values")
        if not isinstance(opts, dict):
            continue
        for k, v in opts.items():
            if k in exclude_keys:
                continue
            if v is None:
                continue
            values.setdefault(k, set()).add(str(v))
    keys = [k for k, vv in values.items() if len(vv) > 1]
    keys.sort()
    return keys


def _markdown(
    rows: list[dict],
    groups: list[dict],
    group_fields: list[str],
    option_effects: list[dict],
    milestone_leaders: dict,
    corr_rankings: dict,
) -> str:
    lines = [
        "# FineWeb Dynamics Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
    ]

    if milestone_leaders:
        lines += [
            "## Milestone Leaders",
            "",
            "| Metric | Run | Optimizer | Mode | Value |",
            "|--------|-----|-----------|------|-------|",
        ]
        for m in ("first_eval_loss", "mid_eval_loss", "last_eval_loss"):
            rec = milestone_leaders.get(m)
            if rec is None:
                continue
            lines.append(
                f"| {m} | {rec.get('run_name')} | {rec.get('optimizer') or '-'} | {rec.get('adapt_mode') or '-'} | {_fmt(rec.get('value'))} |"
            )
        lines += [""]

    lines += [
        "## Per-Run",
        "",
        "| Rank | Run | Opt | Mode | First Eval | Mid Eval | Last Eval | Best Val | Rand Exit | Stable(drop) | Stable(slope) | End Gain % | Mid Slope | Tail Slope | Taper | tok/s | Unstable |",
        "|------|-----|-----|------|------------|----------|-----------|----------|-----------|--------------|---------------|------------|-----------|------------|-------|-------|----------|",
    ]
    for i, r in enumerate(rows, start=1):
        _eg = r.get("end_gain_frac_total")
        _eg_pct = (_eg * 100.0) if isinstance(_eg, (int, float)) else None
        lines.append(
            "| "
            f"{i} | {r.get('run_name')} | {r.get('optimizer') or '-'} | {r.get('adapt_mode') or '-'} | "
            f"{_fmt(r.get('first_eval_loss'))} | {_fmt(r.get('mid_eval_loss'))} | {_fmt(r.get('last_eval_loss'))} | {_fmt(r.get('best_val_loss'))} | {_fmt(r.get('random_exit_step'), nd=0)} | "
            f"{_fmt(r.get('stable_step_drop'), nd=0)} | {_fmt(r.get('stable_step_slope'), nd=0)} | "
            f"{_fmt(_eg_pct)} | "
            f"{_fmt(r.get('mid_slope_per_mtok'))} | {_fmt(r.get('tail_slope_per_mtok'))} | {_fmt(r.get('taper_ratio_tail_mid'))} | "
            f"{_fmt(r.get('throughput_tok_s'), nd=1)} | {r.get('unstable')} |"
        )

    lines += [
        "",
        "## Grouped Summary",
        "",
        "| Rank | " + " | ".join(group_fields) + " | Runs | First Mean | Mid Mean | Last Mean | Best Mean | Stable(drop) | Stable(slope) | Rand Exit | End Gain % | Mid Slope | Tail Slope | Taper | tok/s | Unstable |",
        "|------|" + "------|" * len(group_fields) + "------|------------|----------|-----------|----------|--------------|---------------|-----------|------------|-----------|------------|-------|-------|----------|",
    ]
    for i, g in enumerate(groups, start=1):
        gp = " | ".join(str(g.get(f, "-")) for f in group_fields)
        end_gain_pct = g.get("end_gain_frac_total_mean")
        end_gain_pct = (end_gain_pct * 100.0) if isinstance(end_gain_pct, (int, float)) else None
        lines.append(
            "| "
            f"{i} | {gp} | {_fmt(g.get('n_runs'), nd=0)} | {_fmt(g.get('first_eval_loss_mean'))} | {_fmt(g.get('mid_eval_loss_mean'))} | {_fmt(g.get('last_eval_loss_mean'))} | {_fmt(g.get('best_val_loss_mean'))} | "
            f"{_fmt(g.get('stable_step_drop_mean'), nd=0)} | {_fmt(g.get('stable_step_slope_mean'), nd=0)} | "
            f"{_fmt(g.get('random_exit_step_mean'), nd=0)} | {_fmt(end_gain_pct)} | {_fmt(g.get('mid_slope_per_mtok_mean'))} | "
            f"{_fmt(g.get('tail_slope_per_mtok_mean'))} | {_fmt(g.get('taper_ratio_tail_mid_mean'))} | "
            f"{_fmt(g.get('throughput_tok_s_mean'), nd=1)} | {_fmt(g.get('unstable_runs'), nd=0)} |"
        )

    if option_effects:
        lines += [
            "",
            "## Option Effects (First/Mid/Last Eval)",
            "",
            "| Option | Best@First | Worst@First | Best@Mid | Worst@Mid | Best@Last | Worst@Last | Spread@First | Spread@Mid | Spread@Last | CorrRaw@First | CorrRaw@Mid | CorrRaw@Last |",
            "|--------|------------|-------------|----------|-----------|-----------|------------|--------------|------------|-------------|--------------|------------|-------------|",
        ]
        for e in option_effects:
            w = e.get("winners", {})
            l = e.get("losers", {})
            s = e.get("spreads", {})
            c = e.get("correlations", {})
            lines.append(
                "| "
                f"{e.get('option')} | {w.get('first_eval_loss')} | {l.get('first_eval_loss')} | {w.get('mid_eval_loss')} | {l.get('mid_eval_loss')} | {w.get('last_eval_loss')} | {l.get('last_eval_loss')} | "
                f"{_fmt(s.get('first_eval_loss'))} | {_fmt(s.get('mid_eval_loss'))} | {_fmt(s.get('last_eval_loss'))} | "
                f"{_fmt(c.get('first_eval_loss'))} | {_fmt(c.get('mid_eval_loss'))} | {_fmt(c.get('last_eval_loss'))} |"
            )

    if corr_rankings:
        lines += [
            "",
            "## Correlation vs Anti-Correlation (Good-Oriented)",
            "",
        ]
        for metric in ("first_eval_loss", "mid_eval_loss", "last_eval_loss"):
            block = corr_rankings.get(metric, {}) if isinstance(corr_rankings, dict) else {}
            corr = block.get("correlated", []) if isinstance(block, dict) else []
            anti = block.get("anticorrelated", []) if isinstance(block, dict) else []
            objective = block.get("objective") if isinstance(block, dict) else None
            lines += [
                f"### {metric}",
                "",
                f"Metric objective: {objective or '-'} (positive Corr(good) means better)",
                "",
                "| Direction | Option | Value | Corr(good) | Corr(raw) | N | N(value) |",
                "|-----------|--------|-------|------------|-----------|---|----------|",
            ]
            for r in corr:
                lines.append(
                    "| "
                    f"correlated | {r.get('option')} | {r.get('value')} | {_fmt(r.get('corr'))} | {_fmt(r.get('corr_raw'))} | {_fmt(r.get('n'), nd=0)} | {_fmt(r.get('n_value'), nd=0)} |"
                )
            for r in anti:
                lines.append(
                    "| "
                    f"anticorrelated | {r.get('option')} | {r.get('value')} | {_fmt(r.get('corr'))} | {_fmt(r.get('corr_raw'))} | {_fmt(r.get('n'), nd=0)} | {_fmt(r.get('n_value'), nd=0)} |"
                )
            lines.append("")

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
        default="optimizer,adapt_mode,lr_scope,gnorm_source,gnorm_scope,gmax_scope,second_moment_mode,train_steps,batch_size,grad_accum,seq_len,n_layer,d_model",
        help="Comma-separated keys used for grouped summary",
    )
    ap.add_argument(
        "--option-keys",
        type=str,
        default="auto",
        help="Comma-separated option keys for effect analysis, or 'auto'",
    )
    ap.add_argument(
        "--option-exclude",
        type=str,
        default="seed,metrics_enabled,metrics_dir,metrics_run_name,metrics_every,metrics_flush_every,run_group,run_config_hash",
        help="Comma-separated option keys to exclude from auto option effect analysis",
    )
    ap.add_argument(
        "--corr-top-k",
        type=int,
        default=5,
        help="Top-k options to show for correlation and anti-correlation per milestone",
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

    metrics_for_options = ["first_eval_loss", "mid_eval_loss", "last_eval_loss"]
    exclude_keys = {x.strip() for x in args.option_exclude.split(",") if x.strip()}
    option_keys = _select_option_keys(rows, args.option_keys, exclude_keys)
    option_effects = _option_effects(rows, option_keys, metrics_for_options)
    corr_rankings = _corr_rankings(option_effects, metrics_for_options, top_k=args.corr_top_k)
    milestone_leaders = _milestone_leaders(rows)

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
        "option_keys": option_keys,
        "option_effects": option_effects,
        "corr_rankings": corr_rankings,
        "milestone_leaders": milestone_leaders,
    }

    out_json = Path(args.out_json) if args.out_json else runs_dir / "leaderboard.json"
    out_md = Path(args.out_md) if args.out_md else runs_dir / "leaderboard.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_markdown(rows, groups, group_fields, option_effects, milestone_leaders, corr_rankings), encoding="utf-8")

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
