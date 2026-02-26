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
        "| Rank | Run | Optimizer | Mode | Best Val | Final Val | Final Train | tok/s | Unstable | Spikes (loss/grad) |",
        "|------|-----|-----------|------|----------|-----------|-------------|-------|----------|--------------------|",
    ]
    for i, s in enumerate(stats, start=1):
        lines.append(
            "| "
            f"{i} | {s.run_name} | {s.optimizer or '-'} | {s.adapt_mode or '-'} | {_fmt(s.best_val_loss)} | {_fmt(s.final_val_loss)} | "
            f"{_fmt(s.final_train_loss)} | {_fmt(s.throughput_tok_s, nd=1)} | {s.unstable} | "
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
    hdr = ["Rank", *group_fields, "Runs", "Best Mean", "Best Std", "Final Mean", "Final Std", "tok/s Mean", "Unstable"]
    sep = ["------"] * len(hdr)
    lines = ["## Grouped Summary", "", "| " + " | ".join(hdr) + " |", "| " + " | ".join(sep) + " |"]
    for i, g in enumerate(groups, start=1):
        row = [
            str(i),
            *[str(g.get(f, "-")) for f in group_fields],
            _fmt(g.get("n_runs"), nd=0),
            _fmt(g.get("best_val_loss_mean")),
            _fmt(g.get("best_val_loss_std")),
            _fmt(g.get("final_val_loss_mean")),
            _fmt(g.get("final_val_loss_std")),
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
