#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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


def _count_relative_spikes(vals: list[float], factor: float, warmup: int = 10) -> int:
    if len(vals) < max(2, warmup + 1):
        return 0
    c = 0
    for i in range(warmup, len(vals)):
        prev = vals[i - 1]
        cur = vals[i]
        if prev > 0 and cur > prev * factor:
            c += 1
    return c


@dataclass
class RunStats:
    run_name: str
    run_dir: str
    optimizer: str | None
    model_type: str | None
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
            "model_type": self.model_type,
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


def _extract_run_stats(run_dir: Path, spike_factor: float) -> RunStats | None:
    summary = _read_json(run_dir / "run_summary.json")
    if summary is None:
        return None

    cfg = _read_json(run_dir / "run_config.json") or {}
    hparams = cfg.get("hparams", {}) if isinstance(cfg, dict) else {}

    train_events = _read_jsonl(run_dir / "train_events.jsonl")
    eval_events = _read_jsonl(run_dir / "eval_events.jsonl")

    train_losses = [float(e["train_loss"]) for e in train_events if _is_finite_number(e.get("train_loss"))]
    grad_norms = [float(e["grad_norm"]) for e in train_events if _is_finite_number(e.get("grad_norm"))]
    eval_losses = [float(e["val_loss"]) for e in eval_events if _is_finite_number(e.get("val_loss"))]

    non_finite_train_loss = sum(1 for e in train_events if not _is_finite_number(e.get("train_loss")))
    non_finite_grad_norm = sum(1 for e in train_events if not _is_finite_number(e.get("grad_norm")))
    non_finite_eval_loss = sum(1 for e in eval_events if not _is_finite_number(e.get("val_loss")))

    train_loss_spikes = _count_relative_spikes(train_losses, factor=spike_factor)
    grad_norm_spikes = _count_relative_spikes(grad_norms, factor=spike_factor)

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

    elapsed_s = summary.get("elapsed_s")
    tokens_seen = summary.get("tokens_seen")
    throughput_tok_s = None
    if _is_finite_number(elapsed_s) and _is_finite_number(tokens_seen) and float(elapsed_s) > 0:
        throughput_tok_s = float(tokens_seen) / float(elapsed_s)

    unstable = (
        non_finite_train_loss > 0
        or non_finite_grad_norm > 0
        or non_finite_eval_loss > 0
        or train_loss_spikes > 3
        or grad_norm_spikes > 3
    )

    return RunStats(
        run_name=run_dir.name,
        run_dir=str(run_dir),
        optimizer=hparams.get("optimizer"),
        model_type=hparams.get("model_type"),
        best_val_loss=float(best_val_loss) if _is_finite_number(best_val_loss) else None,
        final_val_loss=float(final_val_loss) if _is_finite_number(final_val_loss) else None,
        final_train_loss=float(final_train_loss) if _is_finite_number(final_train_loss) else None,
        final_lr=float(final_lr) if _is_finite_number(final_lr) else None,
        final_grad_norm=float(final_grad_norm) if _is_finite_number(final_grad_norm) else None,
        tokens_seen=int(tokens_seen) if isinstance(tokens_seen, int) else None,
        total_tokens=int(summary["total_tokens"]) if isinstance(summary.get("total_tokens"), int) else None,
        elapsed_s=float(elapsed_s) if _is_finite_number(elapsed_s) else None,
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
        "| Rank | Run | Optimizer | Best Val | Final Val | Final Train | tok/s | Unstable | Spikes (loss/grad) |",
        "|------|-----|-----------|----------|-----------|-------------|-------|----------|--------------------|",
    ]
    for i, s in enumerate(stats, start=1):
        lines.append(
            "| "
            f"{i} | {s.run_name} | {s.optimizer or '-'} | {_fmt(s.best_val_loss)} | {_fmt(s.final_val_loss)} | "
            f"{_fmt(s.final_train_loss)} | {_fmt(s.throughput_tok_s, nd=1)} | {s.unstable} | "
            f"{s.train_loss_spikes}/{s.grad_norm_spikes} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze JSON metrics runs from train_fineweb_vanilla.py")
    ap.add_argument("--runs-dir", type=str, default="experiments/tier4/runs")
    ap.add_argument("--pattern", type=str, default="*", help="subdir glob under runs-dir")
    ap.add_argument("--spike-factor", type=float, default=1.25)
    ap.add_argument("--out-json", type=str, default="")
    ap.add_argument("--out-md", type=str, default="")
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        raise FileNotFoundError(f"runs-dir not found: {runs_dir}")

    run_dirs = [p for p in sorted(runs_dir.glob(args.pattern)) if p.is_dir()]
    stats: list[RunStats] = []
    for rd in run_dirs:
        st = _extract_run_stats(rd, spike_factor=args.spike_factor)
        if st is not None:
            stats.append(st)

    def _sort_key(s: RunStats):
        best = s.best_val_loss if s.best_val_loss is not None else float("inf")
        final = s.final_val_loss if s.final_val_loss is not None else float("inf")
        return (best, final, s.unstable, s.run_name)

    stats.sort(key=_sort_key)

    report = {
        "generated_at": datetime.now().isoformat(),
        "runs_dir": str(runs_dir),
        "pattern": args.pattern,
        "spike_factor": args.spike_factor,
        "n_runs": len(stats),
        "runs": [s.as_dict() for s in stats],
    }

    out_json = Path(args.out_json) if args.out_json else runs_dir / "leaderboard.json"
    out_md = Path(args.out_md) if args.out_md else runs_dir / "leaderboard.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(stats), encoding="utf-8")

    print(f"Analyzed runs: {len(stats)}")
    if stats:
        best = stats[0]
        print(f"Best run: {best.run_name} | best_val={_fmt(best.best_val_loss)} | final_val={_fmt(best.final_val_loss)}")
    print(f"Wrote JSON: {out_json}")
    print(f"Wrote MD:   {out_md}")


if __name__ == "__main__":
    main()
