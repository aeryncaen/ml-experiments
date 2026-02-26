#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import itertools
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import tomllib


def _load_trainer_module(trainer_path: Path):
    spec = importlib.util.spec_from_file_location("tier4_fineweb_trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trainer module from {trainer_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _as_list(v):
    if isinstance(v, list):
        return v
    return [v]


def _expand_sweep(base: dict, sweep: dict) -> list[dict]:
    if not sweep:
        return [dict(base)]
    keys = sorted(sweep.keys())
    values = [_as_list(sweep[k]) for k in keys]
    out = []
    for combo in itertools.product(*values):
        d = dict(base)
        for k, v in zip(keys, combo):
            d[k] = v
        out.append(d)
    return out


def _hash_hparams(hparams: dict) -> str:
    raw = json.dumps(hparams, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _canonical_autonormuon(out: dict) -> dict:
    """Ensure AutoNorMuon defaults are present and canonical."""
    d = dict(out)
    d.setdefault("autonormuon_adaptation_scope", "neuron")
    return d


def _hash_view(hparams: dict) -> dict:
    out = dict(hparams)

    # Logging/housekeeping fields should not create new experiment identities.
    for k in (
        "metrics_enabled",
        "metrics_dir",
        "metrics_run_name",
        "metrics_every",
        "metrics_flush_every",
        "run_group",
        "run_config_hash",
    ):
        out.pop(k, None)

    # Autonormuon knobs are irrelevant for other optimizers.
    opt = str(out.get("optimizer", ""))
    if opt != "autonormuon":
        for k in list(out.keys()):
            if k.startswith("autonormuon_"):
                out.pop(k, None)
    else:
        out.setdefault("autonormuon_beta", 0.55)
        out.setdefault("autonormuon_adaptation_scope", "neuron")
        out.setdefault("autonormuon_grad_schedule", "off")
        out.setdefault("autonormuon_weight_schedule", "always")
        out.setdefault("autonormuon_ratio_pow", 1.0)
        out.setdefault("autonormuon_min_ratio", 0.0)
        out = _canonical_autonormuon(out)

    return out


def _scan_existing_hashes(metrics_dir: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not metrics_dir.exists():
        return out
    for run_dir in sorted(metrics_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg = run_dir / "run_config.json"
        if not cfg.exists():
            continue
        try:
            data = json.loads(cfg.read_text(encoding="utf-8"))
            hparams = data.get("hparams", {}) if isinstance(data, dict) else {}
            if not isinstance(hparams, dict):
                continue
            h_compat = _hash_hparams(_hash_view(hparams))
            out[h_compat] = str(run_dir)

            # Backward compatibility with prior hash styles, if present.
            h_stored = hparams.get("run_config_hash")
            if isinstance(h_stored, str) and h_stored:
                out[h_stored] = str(run_dir)
        except Exception:
            continue
    return out


def _run_name(prefix: str, hp: dict, cfg_hash: str) -> str:
    opt = str(hp.get("optimizer", "opt"))
    seed = hp.get("seed", "na")
    return f"{prefix}_{opt}_s{seed}_{cfg_hash}"


def _normalize_overrides(ov: dict) -> dict:
    out = dict(ov)
    opt = str(out.get("optimizer", ""))
    if opt != "autonormuon":
        out.pop("autonormuon_beta", None)
        out.pop("autonormuon_adaptation_scope", None)
        out.pop("autonormuon_grad_schedule", None)
        out.pop("autonormuon_weight_schedule", None)
        out.pop("autonormuon_ratio_pow", None)
        out.pop("autonormuon_min_ratio", None)
    else:
        out = _canonical_autonormuon(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run FineWeb sweeps from TOML config (in-process, no subprocess)")
    ap.add_argument("--config", type=str, required=True, help="Path to TOML sweep file")
    ap.add_argument("--trainer", type=str, default="experiments/tier4/train_fineweb_vanilla.py")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    trainer_path = Path(args.trainer)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    if not trainer_path.exists():
        raise FileNotFoundError(f"Trainer not found: {trainer_path}")

    cfg = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    runner = cfg.get("runner", {}) if isinstance(cfg, dict) else {}
    base = cfg.get("base", {}) if isinstance(cfg, dict) else {}
    sweep = cfg.get("sweep", {}) if isinstance(cfg, dict) else {}

    metrics_dir = Path(str(runner.get("metrics_dir", "experiments/tier4/runs")))
    metrics_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = str(runner.get("run_prefix", "fineweb"))
    skip_existing = bool(runner.get("skip_existing", True))
    continue_on_error = bool(runner.get("continue_on_error", False))

    trainer = _load_trainer_module(trainer_path)
    if not hasattr(trainer, "resolve_hparams") or not hasattr(trainer, "run_training"):
        raise RuntimeError("Trainer is missing resolve_hparams/run_training helpers")

    combos = _expand_sweep(base if isinstance(base, dict) else {}, sweep if isinstance(sweep, dict) else {})
    existing_hashes = _scan_existing_hashes(metrics_dir)

    report_runs: list[dict] = []
    counts_by_opt: dict[str, dict[str, int]] = {}
    seen_hashes: set[str] = set()  # dedup identical combos within this sweep
    for i, overrides in enumerate(combos, start=1):
        if not isinstance(overrides, dict):
            continue
        ov = _normalize_overrides(dict(overrides))
        ov["metrics_enabled"] = True
        ov["metrics_dir"] = str(metrics_dir)
        ov.setdefault("run_group", run_prefix)

        hp_obj = trainer.resolve_hparams(ov)
        hp = asdict(hp_obj)
        cfg_hash = _hash_hparams(_hash_view(hp))
        ov["run_config_hash"] = cfg_hash

        run_name = str(ov.get("metrics_run_name", "")).strip()
        if not run_name:
            run_name = _run_name(run_prefix, hp, cfg_hash)
        ov["metrics_run_name"] = run_name

        status = "pending"
        skip_reason = ""
        if cfg_hash in seen_hashes:
            status = "skipped"
            skip_reason = "dedup:same_sweep"
        elif skip_existing and cfg_hash in existing_hashes:
            status = "skipped"
            skip_reason = f"hash_exists:{existing_hashes[cfg_hash]}"
        elif (metrics_dir / run_name).exists() and skip_existing:
            status = "skipped"
            skip_reason = f"run_exists:{metrics_dir / run_name}"
        seen_hashes.add(cfg_hash)

        rec = {
            "index": i,
            "run_name": run_name,
            "optimizer": str(hp.get("optimizer", "")),
            "config_hash": cfg_hash,
            "status": status,
            "skip_reason": skip_reason,
            "overrides": ov,
        }
        report_runs.append(rec)
        _opt = rec["optimizer"] or "unknown"
        counts_by_opt.setdefault(_opt, {"planned": 0, "skipped": 0, "run": 0, "ok": 0, "error": 0})
        counts_by_opt[_opt]["planned"] += 1
        if status == "skipped":
            counts_by_opt[_opt]["skipped"] += 1
        else:
            counts_by_opt[_opt]["run"] += 1

        if status == "skipped" or args.dry_run:
            print(f"[{status}] {run_name} {skip_reason}")
            continue

        print(f"[run] {run_name} hash={cfg_hash}")
        try:
            trainer.run_training(ov)
            rec["status"] = "ok"
            rec["run_dir"] = str(metrics_dir / run_name)
            counts_by_opt[_opt]["ok"] += 1
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e)
            counts_by_opt[_opt]["error"] += 1
            if not continue_on_error:
                raise
        finally:
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    summary = {
        "generated_at": datetime.now().isoformat(),
        "config_path": str(cfg_path),
        "trainer_path": str(trainer_path),
        "metrics_dir": str(metrics_dir),
        "run_prefix": run_prefix,
        "skip_existing": skip_existing,
        "continue_on_error": continue_on_error,
        "dry_run": args.dry_run,
        "n_planned": len(combos),
        "counts_by_optimizer": counts_by_opt,
        "runs": report_runs,
    }
    out_path = metrics_dir / f"{run_prefix}_toml_sweep_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Plan/Result summary by optimizer:")
    for opt in sorted(counts_by_opt):
        c = counts_by_opt[opt]
        print(
            f"  {opt}: planned={c['planned']} skipped={c['skipped']} run={c['run']} ok={c['ok']} error={c['error']}"
        )
    print(f"Wrote summary: {out_path}")


if __name__ == "__main__":
    main()
