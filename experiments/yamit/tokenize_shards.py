#!/usr/bin/env python3
"""Run YAMIT Go tokenizer from repo root without path gymnastics.

Example:
  python experiments/yamit/tokenize_shards.py \
    --input raw_data \
    --output tokenized \
    --workers 16 \
    --val-fraction 0.005
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def _resolve(repo_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    # Prefer caller CWD for relative paths; fallback to repo-root relative.
    cwd_path = Path.cwd() / p
    if cwd_path.exists():
        return cwd_path
    return repo_root / p


def _resolve_artifacts(repo_root: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    # For artifacts we default to repo-root relative path.
    return repo_root / p


def _discover_raw_roots(repo_root: Path) -> list[tuple[Path, int]]:
    counts: dict[Path, int] = {}
    for shard in repo_root.rglob("shard_*.jsonl"):
        # Expected layout: <raw_root>/<dataset>/shard_XXXXX.jsonl
        if len(shard.parents) < 2:
            continue
        raw_root = shard.parent.parent
        counts[raw_root] = counts.get(raw_root, 0) + 1
    return sorted(counts.items(), key=lambda kv: kv[1], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run YAMIT tokenizer with sane defaults")
    parser.add_argument("--input", required=True, help="Input raw JSONL directory (repo-relative or absolute)")
    parser.add_argument("--output", required=True, help="Output tokenized directory (repo-relative or absolute)")
    parser.add_argument(
        "--artifacts-dir",
        default="experiments/yamit/tokenizer/artifacts/qwen3",
        help="Tokenizer artifact directory containing tokenizer.json/surgery_map.json/id_remap.json",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.005)
    parser.add_argument("--go-bin", default="go", help="Go binary (default: go)")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    tokenizer_dir = repo_root / "experiments/yamit/tokenizer"

    input_dir = _resolve(repo_root, args.input)
    output_dir = _resolve(repo_root, args.output)
    artifacts_dir = _resolve_artifacts(repo_root, args.artifacts_dir)

    tok_go_json = artifacts_dir / "tokenizer_go.json"
    tok_json = tok_go_json if tok_go_json.exists() else (artifacts_dir / "tokenizer.json")
    surgery_map = artifacts_dir / "surgery_map.json"
    id_remap = artifacts_dir / "id_remap.json"

    missing = [p for p in (tok_json, surgery_map, id_remap) if not p.exists()]
    if missing:
        miss = "\n".join(f"  - {p}" for p in missing)
        raise SystemExit(
            "Missing tokenizer artifacts:\n"
            f"{miss}\n"
            "Generate them with build_composite_tokenizer.py first."
        )

    if not input_dir.exists():
        discovered = _discover_raw_roots(repo_root)
        if len(discovered) == 1:
            input_dir = discovered[0][0]
            print(
                f"[tokenize_shards] Input '{args.input}' not found; "
                f"auto-using discovered raw dir: {input_dir}"
            )
        else:
            msg = [f"Input directory not found: {input_dir}"]
            if discovered:
                msg.append("Discovered raw-data candidates:")
                for p, n in discovered[:6]:
                    msg.append(f"  - {p} ({n} shards)")
            msg.append("Pass --input with the correct path.")
            raise SystemExit("\n".join(msg))

    cmd = [
        args.go_bin,
        "run",
        ".",
        "--tokenizer",
        str(tok_json),
        "--surgery-map",
        str(surgery_map),
        "--id-remap",
        str(id_remap),
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--workers",
        str(args.workers),
        "--val-fraction",
        str(args.val_fraction),
    ]

    subprocess.run(cmd, cwd=tokenizer_dir, check=True)


if __name__ == "__main__":
    main()
