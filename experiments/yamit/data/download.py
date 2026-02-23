#!/usr/bin/env python3
"""
Download raw text from HuggingFace datasets and save as JSONL shards.

Each dataset from the registry is streamed and written to:
    {output_dir}/{dataset_name}/shard_{NNNNN}.jsonl

Each line is a JSON object: {"text": "..."}

Usage:
    # Download all stage 1 datasets, ~1B tokens worth per source
    python -m data.download --stage 1 --target-tokens 1_000_000_000 --output-dir ./raw_data

    # Download a single dataset
    python -m data.download --datasets fineweb-edu --target-tokens 500_000_000 --output-dir ./raw_data

    # Dry run — print what would be downloaded
    python -m data.download --stage 1 --target-tokens 1_000_000_000 --dry-run
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import sys
from pathlib import Path

# Add parent to path so we can import registry
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.registry import DATASETS, get_stage_mix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 characters of English text.
# This is intentionally conservative (actual ratio is ~3.5 for Qwen3).
CHARS_PER_TOKEN_ESTIMATE = 4

# How many text samples to buffer before flushing to disk.
SHARD_SIZE = 50_000


def count_existing_dataset(name: str, output_dir: Path) -> dict:
    """Count already-downloaded samples/chars/tokens for one dataset."""
    ds_dir = output_dir / name
    if not ds_dir.exists():
        return {
            "name": name,
            "shards": 0,
            "samples": 0,
            "chars": 0,
            "est_tokens": 0,
            "parse_errors": 0,
        }

    shards = sorted(ds_dir.glob("shard_*.jsonl"))
    total_chars = 0
    total_samples = 0
    parse_errors = 0

    for shard_path in shards:
        try:
            with open(shard_path) as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        parse_errors += 1
                        continue
                    text = obj.get("text")
                    if isinstance(text, str):
                        total_chars += len(text)
                        total_samples += 1
        except Exception:
            parse_errors += 1

    return {
        "name": name,
        "shards": len(shards),
        "samples": total_samples,
        "chars": total_chars,
        "est_tokens": total_chars // CHARS_PER_TOKEN_ESTIMATE,
        "parse_errors": parse_errors,
    }


def download_dataset(
    name: str,
    hf_path: str,
    hf_subset: str | None,
    text_column: str,
    target_tokens: int,
    output_dir: Path,
) -> dict:
    """Stream a single dataset and write JSONL shards.

    Returns stats dict with keys: name, shards, samples, chars, est_tokens.
    """
    ds_dir = output_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing shards to support resumption.
    existing_shards = sorted(ds_dir.glob("shard_*.jsonl"))
    if existing_shards:
        # Count existing samples and estimate tokens.
        existing_chars = 0
        existing_samples = 0
        for shard_path in existing_shards:
            with open(shard_path) as f:
                for line in f:
                    obj = json.loads(line)
                    existing_chars += len(obj["text"])
                    existing_samples += 1
        existing_tokens = existing_chars // CHARS_PER_TOKEN_ESTIMATE
        if existing_tokens >= target_tokens:
            log.info(
                f"[{name}] Already have ~{existing_tokens:,} tokens "
                f"({existing_samples:,} samples) — skipping"
            )
            return {
                "name": name,
                "shards": len(existing_shards),
                "samples": existing_samples,
                "chars": existing_chars,
                "est_tokens": existing_tokens,
                "skipped": True,
            }
        log.info(
            f"[{name}] Resuming — have ~{existing_tokens:,}/{target_tokens:,} tokens"
        )
        start_shard = len(existing_shards)
        total_chars = existing_chars
        total_samples = existing_samples
    else:
        start_shard = 0
        total_chars = 0
        total_samples = 0

    target_chars = target_tokens * CHARS_PER_TOKEN_ESTIMATE

    log.info(
        f"[{name}] Streaming from {hf_path}"
        + (f" ({hf_subset})" if hf_subset else "")
        + f" — target ~{target_tokens:,} tokens"
    )

    try:
        from datasets import load_dataset

        ds = load_dataset(
            hf_path,
            name=hf_subset,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )
    except Exception as e:
        log.error(f"[{name}] Failed to load dataset: {e}")
        return {
            "name": name,
            "shards": start_shard,
            "samples": total_samples,
            "chars": total_chars,
            "est_tokens": total_chars // CHARS_PER_TOKEN_ESTIMATE,
            "error": str(e),
        }

    shard_idx = start_shard
    buffer: list[str] = []

    def flush_buffer():
        nonlocal shard_idx, buffer
        if not buffer:
            return
        shard_path = ds_dir / f"shard_{shard_idx:05d}.jsonl"
        with open(shard_path, "w") as f:
            for text in buffer:
                f.write(json.dumps({"text": text}) + "\n")
        log.info(
            f"[{name}] Wrote {shard_path.name} ({len(buffer):,} samples)"
        )
        shard_idx += 1
        buffer = []

    for sample in ds:
        text = sample.get(text_column)
        if text is None:
            # Try common alternative column names.
            for alt in ("content", "text", "Text", "document"):
                text = sample.get(alt)
                if text is not None:
                    break
        if text is None or not isinstance(text, str) or len(text.strip()) == 0:
            continue

        text = text.strip()
        buffer.append(text)
        total_chars += len(text)
        total_samples += 1

        if len(buffer) >= SHARD_SIZE:
            flush_buffer()

        if total_chars >= target_chars:
            break

    flush_buffer()

    est_tokens = total_chars // CHARS_PER_TOKEN_ESTIMATE
    log.info(
        f"[{name}] Done — {total_samples:,} samples, "
        f"~{est_tokens:,} tokens, {shard_idx} shards"
    )
    return {
        "name": name,
        "shards": shard_idx,
        "samples": total_samples,
        "chars": total_chars,
        "est_tokens": est_tokens,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Download SmolLM3 training data as JSONL shards"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3],
        help="Download all datasets for this training stage",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Download specific datasets by name (from registry)",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        required=True,
        help="Target number of tokens to download per dataset "
        "(actual count depends on sampling weights)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Root directory for downloaded JSONL shards",
    )
    parser.add_argument(
        "--proportional",
        action="store_true",
        help="Scale target-tokens per dataset by its stage weight "
        "(so total across all datasets ≈ target-tokens)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print download plan without downloading",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of datasets to download in parallel",
    )
    parser.add_argument(
        "--count-only",
        action="store_true",
        help="Only count already-downloaded tokens/samples; do not download",
    )
    args = parser.parse_args()

    if not args.stage and not args.datasets:
        parser.error("Must specify --stage or --datasets")
    if args.count_only and args.dry_run:
        parser.error("--count-only and --dry-run are mutually exclusive")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build download plan.
    plan: list[tuple[str, str, str | None, str, int]] = []

    if args.datasets:
        ds_by_name = {ds.name: ds for ds in DATASETS}
        for name in args.datasets:
            if name not in ds_by_name:
                parser.error(
                    f"Unknown dataset '{name}'. "
                    f"Available: {sorted(ds_by_name.keys())}"
                )
            ds = ds_by_name[name]
            plan.append((
                ds.name,
                ds.hf_path,
                ds.hf_subset,
                ds.text_column,
                args.target_tokens,
            ))
    else:
        mix = get_stage_mix(args.stage)
        total_weight = sum(e.weight for e in mix)
        for entry in mix:
            if args.proportional:
                # Scale tokens by this dataset's share of the mix.
                per_ds_tokens = int(args.target_tokens * entry.weight / total_weight)
            else:
                per_ds_tokens = args.target_tokens
            plan.append((
                entry.name,
                entry.hf_path,
                entry.hf_subset,
                entry.text_column,
                per_ds_tokens,
            ))

    if args.dry_run:
        total_tokens = sum(t for _, _, _, _, t in plan)
        print(f"\nDownload plan: {len(plan)} datasets, ~{total_tokens:,} total tokens")
        print(f"{'Dataset':<30} {'HF Path':<45} {'Tokens':>15}")
        print("-" * 92)
        for name, hf_path, hf_subset, _, tokens in plan:
            path_str = hf_path + (f"/{hf_subset}" if hf_subset else "")
            print(f"{name:<30} {path_str:<45} {tokens:>15,}")
        return

    if args.count_only:
        workers = max(1, args.workers)
        log.info(
            f"Counting existing data for {len(plan)} datasets with {workers} worker(s)"
        )

        results_by_name: dict[str, dict] = {}

        def count_one(spec: tuple[str, str, str | None, str, int]) -> dict:
            name, _, _, _, _ = spec
            r = count_existing_dataset(name, output_dir)
            return r

        if workers == 1 or len(plan) == 1:
            for spec in plan:
                r = count_one(spec)
                results_by_name[r["name"]] = r
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
                future_to_name = {ex.submit(count_one, spec): spec[0] for spec in plan}
                for fut in concurrent.futures.as_completed(future_to_name):
                    name = future_to_name[fut]
                    try:
                        r = fut.result()
                    except Exception as e:
                        r = {
                            "name": name,
                            "shards": 0,
                            "samples": 0,
                            "chars": 0,
                            "est_tokens": 0,
                            "parse_errors": 1,
                            "error": str(e),
                        }
                    results_by_name[name] = r

        results = [results_by_name[name] for name, *_ in plan]

        print(f"\n{'='*86}")
        print("Existing Data Summary")
        print(f"{'='*86}")
        print(
            f"{'Dataset':<30} {'Have Tokens':>14} {'Target':>14} {'Progress':>10} {'Shards':>8}"
        )
        print("-" * 86)

        total_have = 0
        total_target = 0
        total_samples = 0
        total_shards = 0
        total_parse_errors = 0

        target_by_name = {name: tokens for name, _, _, _, tokens in plan}
        for r in results:
            target = target_by_name[r["name"]]
            have = r["est_tokens"]
            progress = (100.0 * have / target) if target > 0 else 0.0
            print(
                f"{r['name']:<30} {have:>14,} {target:>14,} {progress:>9.1f}% {r['shards']:>8,}"
            )
            total_have += have
            total_target += target
            total_samples += r["samples"]
            total_shards += r["shards"]
            total_parse_errors += r.get("parse_errors", 0)

        total_progress = (100.0 * total_have / total_target) if total_target > 0 else 0.0
        print("-" * 86)
        print(
            f"{'TOTAL':<30} {total_have:>14,} {total_target:>14,} {total_progress:>9.1f}% {total_shards:>8,}"
        )
        print(f"\nSamples: {total_samples:,}")
        if total_parse_errors > 0:
            print(f"Parse errors while counting: {total_parse_errors:,}")
        return

    # Execute downloads.
    workers = max(1, args.workers)
    log.info(
        f"Starting download for {len(plan)} datasets with {workers} worker(s)"
    )

    def run_one(spec: tuple[str, str, str | None, str, int]) -> dict:
        name, hf_path, hf_subset, text_column, target_tokens = spec
        try:
            return download_dataset(
                name=name,
                hf_path=hf_path,
                hf_subset=hf_subset,
                text_column=text_column,
                target_tokens=target_tokens,
                output_dir=output_dir,
            )
        except Exception as e:
            log.exception(f"[{name}] Unhandled error")
            return {
                "name": name,
                "shards": 0,
                "samples": 0,
                "chars": 0,
                "est_tokens": 0,
                "error": str(e),
            }

    results_by_name: dict[str, dict] = {}
    if workers == 1 or len(plan) == 1:
        for spec in plan:
            r = run_one(spec)
            results_by_name[r["name"]] = r
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            future_to_name = {
                ex.submit(run_one, spec): spec[0]
                for spec in plan
            }
            for fut in concurrent.futures.as_completed(future_to_name):
                name = future_to_name[fut]
                try:
                    r = fut.result()
                except Exception as e:
                    r = {
                        "name": name,
                        "shards": 0,
                        "samples": 0,
                        "chars": 0,
                        "est_tokens": 0,
                        "error": str(e),
                    }
                results_by_name[name] = r

    # Preserve plan order in summary output.
    results = [results_by_name[name] for name, *_ in plan]

    # Summary.
    print(f"\n{'='*70}")
    print("Download Summary")
    print(f"{'='*70}")
    total_tokens = 0
    total_samples = 0
    errors = []
    for r in results:
        status = "SKIP" if r.get("skipped") else ("ERR" if r.get("error") else "OK")
        print(
            f"  [{status:>4}] {r['name']:<30} "
            f"{r['est_tokens']:>12,} tokens  "
            f"{r['samples']:>10,} samples  "
            f"{r['shards']:>4} shards"
        )
        if r.get("error"):
            errors.append(r)
        total_tokens += r["est_tokens"]
        total_samples += r["samples"]
    print(f"\nTotal: ~{total_tokens:,} tokens, {total_samples:,} samples")

    if errors:
        print(f"\n{'='*70}")
        print(f"Errors ({len(errors)}):")
        print(f"{'='*70}")
        for r in errors:
            print(f"  {r['name']}")
            print(f"    {r['error']}")
            print()

    # Write manifest for the tokenizer to consume.
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "datasets": {
            r["name"]: {
                "shards": r["shards"],
                "samples": r["samples"],
                "est_tokens": r["est_tokens"],
            }
            for r in results
            if not r.get("error")
        }
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")


if __name__ == "__main__":
    main()
