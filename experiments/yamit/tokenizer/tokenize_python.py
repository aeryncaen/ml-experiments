#!/usr/bin/env python3
"""Python fallback tokenizer for YAMIT shard preprocessing.

This is used when the Go tokenizer backend cannot load a tokenizer.json
pre-tokenizer regex (e.g. unsupported lookahead patterns).

Output format matches the Go tokenizer:
  - .bin: little-endian uint32 token IDs
  - .idx: little-endian uint64 byte offsets (N+1 entries for N docs)
"""

from __future__ import annotations

import argparse
import json
import random
import struct
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


_TOK = None
_SURGERY: dict[int, list[int]] = {}
_ID_REMAP: dict[int, int] | None = None


def _init_worker(tokenizer_path: str, surgery_map_path: str, id_remap_path: str | None) -> None:
    global _TOK, _SURGERY, _ID_REMAP
    from tokenizers import Tokenizer

    _TOK = Tokenizer.from_file(tokenizer_path)
    with open(surgery_map_path) as f:
        _SURGERY = {int(k): v for k, v in json.load(f).items()}

    if id_remap_path:
        with open(id_remap_path) as f:
            _ID_REMAP = {int(k): v for k, v in json.load(f).items()}
    else:
        _ID_REMAP = None


def _apply_surgery(ids: list[int]) -> list[int]:
    if not _SURGERY:
        return ids
    out: list[int] = []
    for tid in ids:
        repl = _SURGERY.get(tid)
        if repl is None:
            out.append(tid)
        else:
            out.extend(repl)
    return out


def _apply_id_remap(ids: list[int]) -> list[int]:
    if _ID_REMAP is None:
        return ids
    out: list[int] = []
    for tid in ids:
        if tid not in _ID_REMAP:
            raise RuntimeError(f"Missing remap for token id {tid}")
        out.append(_ID_REMAP[tid])
    return out


def _tokenize_file(task: tuple[str, str]) -> dict:
    input_path_str, output_base_str = task
    input_path = Path(input_path_str)
    output_base = Path(output_base_str)
    output_base.parent.mkdir(parents=True, exist_ok=True)

    bin_path = output_base.with_suffix(".bin")
    idx_path = output_base.with_suffix(".idx")

    doc_count = 0
    token_count = 0
    offset_bytes = 0

    with open(bin_path, "wb") as bin_f, open(idx_path, "wb") as idx_f, open(input_path) as in_f:
        idx_f.write(struct.pack("<Q", 0))
        for line in in_f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text")
            if not isinstance(text, str):
                continue
            text = text.strip()
            if not text:
                continue

            ids = _TOK.encode(text).ids
            ids = _apply_surgery(ids)
            ids = _apply_id_remap(ids)

            if ids:
                arr = np.asarray(ids, dtype=np.uint32)
                bin_f.write(arr.tobytes(order="C"))
                offset_bytes += int(arr.size) * 4
            idx_f.write(struct.pack("<Q", offset_bytes))
            doc_count += 1
            token_count += len(ids)

    return {
        "input": str(input_path),
        "output": str(output_base),
        "docs": doc_count,
        "tokens": token_count,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Python fallback tokenizer for YAMIT shards")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--surgery-map", required=True)
    p.add_argument("--id-remap", default="")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.005)
    args = p.parse_args()

    start = time.time()
    input_root = Path(args.input)
    output_root = Path(args.output)
    train_root = output_root / "train"
    val_root = output_root / "val"
    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    files = sorted(input_root.rglob("*.jsonl"))
    if not files:
        raise SystemExit(f"No .jsonl files found under {input_root}")

    idx = list(range(len(files)))
    rng = random.Random(42)
    rng.shuffle(idx)
    val_count = int(len(files) * args.val_fraction)
    if val_count < 1 and len(files) > 1:
        val_count = 1
    val_set = set(idx[:val_count])

    tasks: list[tuple[str, str]] = []
    for i, f in enumerate(files):
        rel = f.relative_to(input_root)
        split_root = val_root if i in val_set else train_root
        out_base = split_root / rel.parent / rel.stem
        tasks.append((str(f), str(out_base)))

    total_docs = 0
    total_tokens = 0
    errors = 0
    done = 0

    with ProcessPoolExecutor(
        max_workers=max(1, args.workers),
        initializer=_init_worker,
        initargs=(args.tokenizer, args.surgery_map, args.id_remap or None),
    ) as ex:
        futures = [ex.submit(_tokenize_file, t) for t in tasks]
        for fut in as_completed(futures):
            done += 1
            try:
                r = fut.result()
                total_docs += r["docs"]
                total_tokens += r["tokens"]
                if done % 50 == 0 or done == len(tasks):
                    print(
                        f"[tokenize_python] {done}/{len(tasks)} files | "
                        f"docs={total_docs:,} tokens={total_tokens:,}"
                    )
            except Exception as e:
                errors += 1
                print(f"[tokenize_python] ERROR: {e}")

    elapsed = time.time() - start
    print("\n[tokenize_python] Done")
    print(f"  files:   {len(files):,}")
    print(f"  docs:    {total_docs:,}")
    print(f"  tokens:  {total_tokens:,}")
    print(f"  errors:  {errors}")
    if elapsed > 0:
        print(f"  speed:   {int(total_tokens/elapsed):,} tok/s")


if __name__ == "__main__":
    main()
