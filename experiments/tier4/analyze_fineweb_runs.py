#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from dataclasses import dataclass
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


def _is_finite_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def _as_float(x) -> float | None:
    if _is_finite_number(x):
        return float(x)
    return None


def _as_int(x) -> int | None:
    if isinstance(x, int):
        return x
    if _is_finite_number(x):
        return int(float(x))
    return None


def _count_relative_spikes(
    vals: list[float],
    factor: float,
    warmup: int = 10,
    min_abs_delta: float = 0.0,
) -> int:
    if len(vals) < max(2, warmup + 1):
        return 0
    c = 0
    for i in range(warmup, len(vals)):
        prev = vals[i - 1]
        cur = vals[i]
        if prev > 0 and cur > prev * factor and (cur - prev) >= min_abs_delta:
            c += 1
    return c


def _smooth_series(vals: list[float], window: int) -> list[float]:
    n = len(vals)
    if n == 0 or window <= 1:
        return vals[:]
    w = max(1, int(window))
    out = []
    for i in range(n):
        lo = max(0, i - w // 2)
        hi = min(n, i + w // 2 + 1)
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


def _linear_slope(x: list[float], y: list[float]) -> float | None:
    n = min(len(x), len(y))
    if n < 2:
        return None
    mx = statistics.fmean(x)
    my = statistics.fmean(y)
    num = 0.0
    den = 0.0
    for xi, yi in zip(x, y):
        dx = xi - mx
        num += dx * (yi - my)
        den += dx * dx
    if den <= 0:
        return None
    return num / den


def _first_idx_leq(vals: list[float], threshold: float) -> int | None:
    for i, v in enumerate(vals):
        if v <= threshold:
            return i
    return None


@dataclass
class RunStats:
    run_name: str
    run_dir: str
    optimizer: str | None
    adapt_mode: str | None
    seed: int | None
    model_type: str | None
    train_steps: int | None
    batch_size: int | None
    grad_accum: int | None
    seq_len: int | None
    n_layer: int | None
    d_model: int | None
    config_hash: str
    best_val_loss: float | None
    final_val_loss: float | None
    final_train_loss: float | None
    final_lr: float | None
    final_grad_norm: float | None
    tokens_seen: int | None
    total_tokens: int | None
    elapsed_s: float | None
    throughput_tok_s: float | None
    random_loss_ref: float | None
    random_exit_step: int | None
    random_exit_tokens: int | None
    stable_step: int | None
    stable_tokens: int | None
    stable_elapsed_s: float | None
    early_slope_per_mtok: float | None
    mid_slope_per_mtok: float | None
    tail_slope_per_mtok: float | None
    taper_ratio_tail_mid: float | None
    mid_loss_std: float | None
    tail_loss_std: float | None
    train_events: int
    eval_events: int
    train_loss_spikes: int
    grad_norm_spikes: int
    non_finite_train_loss: int
    non_finite_grad_norm: int
    non_finite_eval_loss: int
    unstable: bool

    def as_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "run_dir": self.run_dir,
            "optimizer": self.optimizer,
            "adapt_mode": self.adapt_mode,
            "seed": self.seed,
            "model_type": self.model_type,
            "train_steps": self.train_steps,
            "batch_size": self.batch_size,
            "grad_accum": self.grad_accum,
            "seq_len": self.seq_len,
            "n_layer": self.n_layer,
            "d_model": self.d_model,
            "config_hash": self.config_hash,
            "best_val_loss": self.best_val_loss,
            "final_val_loss": self.final_val_loss,
            "final_train_loss": self.final_train_loss,
            "final_lr": self.final_lr,
            "final_grad_norm": self.final_grad_norm,
            "tokens_seen": self.tokens_seen,
            "total_tokens": self.total_tokens,
            "elapsed_s": self.elapsed_s,
            "throughput_tok_s": self.throughput_tok_s,
            "random_loss_ref": self.random_loss_ref,
            "random_exit_step": self.random_exit_step,
            "random_exit_tokens": self.random_exit_tokens,
            "stable_step": self.stable_step,
            "stable_tokens": self.stable_tokens,
            "stable_elapsed_s": self.stable_elapsed_s,
            "early_slope_per_mtok": self.early_slope_per_mtok,
            "mid_slope_per_mtok": self.mid_slope_per_mtok,
            "tail_slope_per_mtok": self.tail_slope_per_mtok,
            "taper_ratio_tail_mid": self.taper_ratio_tail_mid,
            "mid_loss_std": self.mid_loss_std,
            "tail_loss_std": self.tail_loss_std,
            "train_events": self.train_events,
            "eval_events": self.eval_events,
            "train_loss_spikes": self.train_loss_spikes,
            "grad_norm_spikes": self.grad_norm_spikes,
            "non_finite_train_loss": self.non_finite_train_loss,
            "non_finite_grad_norm": self.non_finite_grad_norm,
            "non_finite_eval_loss": self.non_finite_eval_loss,
            "unstable": self.unstable,
        }


def _extract_run_stats(
    run_dir: Path,
    loss_spike_factor: float,
    grad_spike_factor: float,
    spike_warmup: int,
    loss_spike_min_abs: float,
    grad_spike_min_abs: float,
    max_loss_spikes: int,
    max_grad_spikes: int,
    smooth_window: int,
    random_margin: float,
    stable_drop_frac: float,
    early_frac: float,
    mid_start_frac: float,
    mid_end_frac: float,
    tail_frac: float,
) -> RunStats | None:
    summary = _read_json(run_dir / "run_summary.json")
    if summary is None:
        return None

    cfg = _read_json(run_dir / "run_config.json") or {}
    hparams = cfg.get("hparams", {}) if isinstance(cfg, dict) else {}
    if isinstance(hparams.get("run_config_hash"), str) and hparams.get("run_config_hash"):
        config_hash = hparams["run_config_hash"]
    else:
        config_hash = hashlib.sha1(json.dumps(hparams, sort_keys=True).encode("utf-8")).hexdigest()[:12] if hparams else "-"

    train_events = _read_jsonl(run_dir / "train_events.jsonl")
    eval_events = _read_jsonl(run_dir / "eval_events.jsonl")

    train_losses = [float(e["train_loss"]) for e in train_events if _is_finite_number(e.get("train_loss"))]
    grad_norms = [float(e["grad_norm"]) for e in train_events if _is_finite_number(e.get("grad_norm"))]
    eval_losses = [float(e["val_loss"]) for e in eval_events if _is_finite_number(e.get("val_loss"))]

    train_steps_axis: list[int] = []
    train_tokens_axis: list[float] = []
    train_elapsed_axis: list[float | None] = []
    _n = 0
    for e in train_events:
        if not _is_finite_number(e.get("train_loss")):
            continue
        st = _as_int(e.get("step"))
        if st is None:
            st = _n
        tok = _as_float(e.get("tokens_seen"))
        if tok is None:
            tok = float(st)
        train_steps_axis.append(st)
        train_tokens_axis.append(tok)
        train_elapsed_axis.append(_as_float(e.get("elapsed_s")))
        _n += 1

    non_finite_train_loss = sum(1 for e in train_events if not _is_finite_number(e.get("train_loss")))
    non_finite_grad_norm = sum(1 for e in train_events if not _is_finite_number(e.get("grad_norm")))
    non_finite_eval_loss = sum(1 for e in eval_events if not _is_finite_number(e.get("val_loss")))

    train_loss_spikes = _count_relative_spikes(
        train_losses,
        factor=loss_spike_factor,
        warmup=spike_warmup,
        min_abs_delta=loss_spike_min_abs,
    )
    grad_norm_spikes = _count_relative_spikes(
        grad_norms,
        factor=grad_spike_factor,
        warmup=spike_warmup,
        min_abs_delta=grad_spike_min_abs,
    )

    best_val_loss = summary.get("best_val_loss")
    final_val_loss = None
    final_train_loss = None
    final_lr = None
    final_grad_norm = None
    if isinstance(summary.get("last_eval"), dict):
        final_val_loss = summary["last_eval"].get("val_loss")
    if isinstance(summary.get("last_train"), dict):
        final_train_loss = summary["last_train"].get("train_loss")
        final_lr = summary["last_train"].get("lr")
        final_grad_norm = summary["last_train"].get("grad_norm")

    if not _is_finite_number(best_val_loss) and eval_losses:
        best_val_loss = min(eval_losses)
    if not _is_finite_number(final_val_loss) and eval_losses:
        final_val_loss = eval_losses[-1]

    elapsed_s = _as_float(summary.get("elapsed_s"))
    tokens_seen = _as_int(summary.get("tokens_seen"))
    throughput_tok_s = None
    if elapsed_s is not None and tokens_seen is not None and elapsed_s > 0:
        throughput_tok_s = float(tokens_seen) / elapsed_s

    smooth_losses = _smooth_series(train_losses, smooth_window)
    x_mtok = [t / 1_000_000.0 for t in train_tokens_axis]

    random_loss_ref = None
    random_exit_step = None
    random_exit_tokens = None
    _vocab = _as_int(hparams.get("vocab_size"))
    if _vocab is not None and _vocab > 1 and smooth_losses:
        random_loss_ref = math.log(float(_vocab))
        random_threshold = random_loss_ref * (1.0 - random_margin)
        ridx = _first_idx_leq(smooth_losses, random_threshold)
        if ridx is not None:
            random_exit_step = train_steps_axis[ridx] if ridx < len(train_steps_axis) else None
            random_exit_tokens = int(train_tokens_axis[ridx]) if ridx < len(train_tokens_axis) else None

    stable_step = None
    stable_tokens = None
    stable_elapsed_s = None
    if smooth_losses:
        init_loss = smooth_losses[0]
        end_loss = smooth_losses[-1]
        total_drop = init_loss - end_loss
        if total_drop > 0:
            stable_target = init_loss - stable_drop_frac * total_drop
            sidx = _first_idx_leq(smooth_losses, stable_target)
            if sidx is not None:
                stable_step = train_steps_axis[sidx] if sidx < len(train_steps_axis) else None
                stable_tokens = int(train_tokens_axis[sidx]) if sidx < len(train_tokens_axis) else None
                stable_elapsed_s = train_elapsed_axis[sidx] if sidx < len(train_elapsed_axis) else None

    npts = len(smooth_losses)
    early_slope = None
    mid_slope = None
    tail_slope = None
    mid_loss_std = None
    tail_loss_std = None
    taper_ratio = None
    if npts >= 2:
        e0, e1 = _segment_bounds(npts, 0.0, early_frac)
        m0, m1 = _segment_bounds(npts, mid_start_frac, mid_end_frac)
        t0, t1 = _segment_bounds(npts, max(0.0, 1.0 - tail_frac), 1.0)

        early_slope = _linear_slope(x_mtok[e0:e1], smooth_losses[e0:e1])
        mid_slope = _linear_slope(x_mtok[m0:m1], smooth_losses[m0:m1])
        tail_slope = _linear_slope(x_mtok[t0:t1], smooth_losses[t0:t1])

        mid_seg = smooth_losses[m0:m1]
        tail_seg = smooth_losses[t0:t1]
        mid_loss_std = statistics.pstdev(mid_seg) if len(mid_seg) > 1 else (0.0 if mid_seg else None)
        tail_loss_std = statistics.pstdev(tail_seg) if len(tail_seg) > 1 else (0.0 if tail_seg else None)

        if mid_slope is not None and tail_slope is not None:
            taper_ratio = abs(tail_slope) / max(abs(mid_slope), 1e-12)

    unstable = (
        non_finite_train_loss > 0
        or non_finite_grad_norm > 0
        or non_finite_eval_loss > 0
        or train_loss_spikes > max_loss_spikes
        or grad_norm_spikes > max_grad_spikes
    )

    return RunStats(
        run_name=run_dir.name,
        run_dir=str(run_dir),
        optimizer=hparams.get("optimizer"),
        adapt_mode=hparams.get("autonormuon_adapt_mode") if hparams.get("optimizer") == "autonormuon" else None,
        seed=_as_int(hparams.get("seed")),
        model_type=hparams.get("model_type"),
        train_steps=_as_int(hparams.get("train_steps")),
        batch_size=_as_int(hparams.get("batch_size")),
        grad_accum=_as_int(hparams.get("grad_accum")),
        seq_len=_as_int(hparams.get("seq_len")),
        n_layer=_as_int(hparams.get("n_layer")),
        d_model=_as_int(hparams.get("d_model")),
        config_hash=config_hash,
        best_val_loss=_as_float(best_val_loss),
        final_val_loss=_as_float(final_val_loss),
        final_train_loss=_as_float(final_train_loss),
        final_lr=_as_float(final_lr),
        final_grad_norm=_as_float(final_grad_norm),
        tokens_seen=tokens_seen,
        total_tokens=_as_int(summary.get("total_tokens")),
        elapsed_s=elapsed_s,
        throughput_tok_s=throughput_tok_s,
        random_loss_ref=random_loss_ref,
        random_exit_step=random_exit_step,
        random_exit_tokens=random_exit_tokens,
        stable_step=stable_step,
        stable_tokens=stable_tokens,
        stable_elapsed_s=stable_elapsed_s,
        early_slope_per_mtok=early_slope,
        mid_slope_per_mtok=mid_slope,
        tail_slope_per_mtok=tail_slope,
        taper_ratio_tail_mid=taper_ratio,
        mid_loss_std=mid_loss_std,
        tail_loss_std=tail_loss_std,
        train_events=len(train_events),
        eval_events=len(eval_events),
        train_loss_spikes=train_loss_spikes,
        grad_norm_spikes=grad_norm_spikes,
        non_finite_train_loss=non_finite_train_loss,
        non_finite_grad_norm=non_finite_grad_norm,
        non_finite_eval_loss=non_finite_eval_loss,
        unstable=unstable,
    )


def _fmt(x, nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, int):
        return f"{x:,}"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _to_markdown(stats: list[RunStats]) -> str:
    lines = [
        "# FineWeb Run Leaderboard",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "| Rank | Run | Optimizer | Mode | Best Val | Final Val | Stable Step | Rand Exit | Mid Slope | Tail Slope | Taper | tok/s | Unstable | Spikes (loss/grad) |",
        "|------|-----|-----------|------|----------|-----------|-------------|-----------|-----------|------------|-------|-------|----------|--------------------|",
    ]
    for i, s in enumerate(stats, start=1):
        lines.append(
            "| "
            f"{i} | {s.run_name} | {s.optimizer or '-'} | {s.adapt_mode or '-'} | {_fmt(s.best_val_loss)} | {_fmt(s.final_val_loss)} | "
            f"{_fmt(s.stable_step, nd=0)} | {_fmt(s.random_exit_step, nd=0)} | {_fmt(s.mid_slope_per_mtok)} | {_fmt(s.tail_slope_per_mtok)} | "
            f"{_fmt(s.taper_ratio_tail_mid)} | {_fmt(s.throughput_tok_s, nd=1)} | {s.unstable} | "
            f"{s.train_loss_spikes}/{s.grad_norm_spikes} |"
        )
    lines.append("")
    return "\n".join(lines)


def _group_aggregate(stats: list[RunStats], group_fields: list[str]) -> list[dict]:
    buckets: dict[tuple, list[RunStats]] = {}
    for s in stats:
        key = tuple(getattr(s, f, None) for f in group_fields)
        buckets.setdefault(key, []).append(s)

    out: list[dict] = []
    for key, rows in buckets.items():
        best_vals = [r.best_val_loss for r in rows if r.best_val_loss is not None]
        final_vals = [r.final_val_loss for r in rows if r.final_val_loss is not None]
        throughputs = [r.throughput_tok_s for r in rows if r.throughput_tok_s is not None]
        stable_steps = [float(r.stable_step) for r in rows if r.stable_step is not None]
        rand_steps = [float(r.random_exit_step) for r in rows if r.random_exit_step is not None]
        mid_slopes = [r.mid_slope_per_mtok for r in rows if r.mid_slope_per_mtok is not None]
        tail_slopes = [r.tail_slope_per_mtok for r in rows if r.tail_slope_per_mtok is not None]
        taper_ratios = [r.taper_ratio_tail_mid for r in rows if r.taper_ratio_tail_mid is not None]
        unstable_n = sum(1 for r in rows if r.unstable)
        rec = {f: v for f, v in zip(group_fields, key)}
        rec.update(
            {
                "n_runs": len(rows),
                "best_val_loss_mean": statistics.fmean(best_vals) if best_vals else None,
                "best_val_loss_std": statistics.pstdev(best_vals) if len(best_vals) > 1 else (0.0 if best_vals else None),
                "final_val_loss_mean": statistics.fmean(final_vals) if final_vals else None,
                "final_val_loss_std": statistics.pstdev(final_vals) if len(final_vals) > 1 else (0.0 if final_vals else None),
                "throughput_tok_s_mean": statistics.fmean(throughputs) if throughputs else None,
                "throughput_tok_s_std": statistics.pstdev(throughputs) if len(throughputs) > 1 else (0.0 if throughputs else None),
                "stable_step_mean": statistics.fmean(stable_steps) if stable_steps else None,
                "random_exit_step_mean": statistics.fmean(rand_steps) if rand_steps else None,
                "mid_slope_per_mtok_mean": statistics.fmean(mid_slopes) if mid_slopes else None,
                "tail_slope_per_mtok_mean": statistics.fmean(tail_slopes) if tail_slopes else None,
                "taper_ratio_tail_mid_mean": statistics.fmean(taper_ratios) if taper_ratios else None,
                "unstable_runs": unstable_n,
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


def _group_markdown(groups: list[dict], group_fields: list[str]) -> str:
    hdr = [
        "Rank",
        *group_fields,
        "Runs",
        "Best Mean",
        "Final Mean",
        "Stable Step",
        "Rand Exit",
        "Mid Slope",
        "Tail Slope",
        "Taper",
        "tok/s Mean",
        "Unstable",
    ]
    sep = ["------"] * len(hdr)
    lines = ["## Grouped Summary", "", "| " + " | ".join(hdr) + " |", "| " + " | ".join(sep) + " |"]
    for i, g in enumerate(groups, start=1):
        row = [
            str(i),
            *[str(g.get(f, "-")) for f in group_fields],
            _fmt(g.get("n_runs"), nd=0),
            _fmt(g.get("best_val_loss_mean")),
            _fmt(g.get("final_val_loss_mean")),
            _fmt(g.get("stable_step_mean"), nd=0),
            _fmt(g.get("random_exit_step_mean"), nd=0),
            _fmt(g.get("mid_slope_per_mtok_mean")),
            _fmt(g.get("tail_slope_per_mtok_mean")),
            _fmt(g.get("taper_ratio_tail_mid_mean")),
            _fmt(g.get("throughput_tok_s_mean"), nd=1),
            _fmt(g.get("unstable_runs"), nd=0),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze JSON metrics runs from train_fineweb_vanilla.py")
    ap.add_argument("--runs-dir", type=str, default="experiments/tier4/runs")
    ap.add_argument("--pattern", type=str, default="*", help="subdir glob under runs-dir")
    ap.add_argument("--loss-spike-factor", type=float, default=1.25)
    ap.add_argument("--grad-spike-factor", type=float, default=2.0)
    ap.add_argument("--spike-warmup", type=int, default=25)
    ap.add_argument("--loss-spike-min-abs", type=float, default=0.05)
    ap.add_argument("--grad-spike-min-abs", type=float, default=0.2)
    ap.add_argument("--max-loss-spikes", type=int, default=5)
    ap.add_argument("--max-grad-spikes", type=int, default=40)
    ap.add_argument("--smooth-window", type=int, default=11)
    ap.add_argument("--random-margin", type=float, default=0.2)
    ap.add_argument("--stable-drop-frac", type=float, default=0.8)
    ap.add_argument("--early-frac", type=float, default=0.2)
    ap.add_argument("--mid-start-frac", type=float, default=0.4)
    ap.add_argument("--mid-end-frac", type=float, default=0.6)
    ap.add_argument("--tail-frac", type=float, default=0.2)
    ap.add_argument(
        "--group-by",
        type=str,
        default="optimizer,adapt_mode,train_steps,batch_size,grad_accum,seq_len,n_layer,d_model",
        help="Comma-separated RunStats fields to aggregate over",
    )
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--out-md", type=str, default="")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs-dir not found: {runs_dir}")

    run_dirs = [p for p in sorted(runs_dir.glob(args.pattern)) if p.is_dir()]
    stats: list[RunStats] = []
    for rd in run_dirs:
        st = _extract_run_stats(
            rd,
            loss_spike_factor=args.loss_spike_factor,
            grad_spike_factor=args.grad_spike_factor,
            spike_warmup=args.spike_warmup,
            loss_spike_min_abs=args.loss_spike_min_abs,
            grad_spike_min_abs=args.grad_spike_min_abs,
            max_loss_spikes=args.max_loss_spikes,
            max_grad_spikes=args.max_grad_spikes,
            smooth_window=args.smooth_window,
            random_margin=args.random_margin,
            stable_drop_frac=args.stable_drop_frac,
            early_frac=args.early_frac,
            mid_start_frac=args.mid_start_frac,
            mid_end_frac=args.mid_end_frac,
            tail_frac=args.tail_frac,
        )
        if st is not None:
            stats.append(st)

    def _sort_key(s: RunStats):
        best = s.best_val_loss if s.best_val_loss is not None else float("inf")
        final = s.final_val_loss if s.final_val_loss is not None else float("inf")
        return (best, final, s.unstable, s.run_name)

    stats.sort(key=_sort_key)
    group_fields = [x.strip() for x in args.group_by.split(",") if x.strip()]
    groups = _group_aggregate(stats, group_fields)

    report = {
        "generated_at": datetime.now().isoformat(),
        "runs_dir": str(runs_dir),
        "pattern": args.pattern,
        "spike_rules": {
            "loss_spike_factor": args.loss_spike_factor,
            "grad_spike_factor": args.grad_spike_factor,
            "spike_warmup": args.spike_warmup,
            "loss_spike_min_abs": args.loss_spike_min_abs,
            "grad_spike_min_abs": args.grad_spike_min_abs,
            "max_loss_spikes": args.max_loss_spikes,
            "max_grad_spikes": args.max_grad_spikes,
        },
        "dynamics_rules": {
            "smooth_window": args.smooth_window,
            "random_margin": args.random_margin,
            "stable_drop_frac": args.stable_drop_frac,
            "early_frac": args.early_frac,
            "mid_start_frac": args.mid_start_frac,
            "mid_end_frac": args.mid_end_frac,
            "tail_frac": args.tail_frac,
        },
        "n_runs": len(stats),
        "runs": [s.as_dict() for s in stats],
        "group_by": group_fields,
        "groups": groups,
    }

    out_json = Path(args.out_json) if args.out_json else runs_dir / "leaderboard.json"
    out_md = Path(args.out_md) if args.out_md else runs_dir / "leaderboard.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(stats) + "\n" + _group_markdown(groups, group_fields), encoding="utf-8")

    print(f"Analyzed runs: {len(stats)}")
    if stats:
        best = stats[0]
        print(f"Best run: {best.run_name} | best_val={_fmt(best.best_val_loss)} | final_val={_fmt(best.final_val_loss)}")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote MD:   {out_md}")


if __name__ == "__main__":
    main()
