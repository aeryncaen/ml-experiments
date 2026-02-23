#!/usr/bin/env python3
"""Generate a modified tokenizer for YAMIT composite embeddings.

This script takes a base HuggingFace tokenizer and produces a new tokenizer
with long tokens (> max_bytes) removed directly from the vocabulary and BPE
merges.  The modified tokenizer is self-contained: no surgery maps, no ID
remaps, no post-processing — it natively produces the correct token IDs.

Outputs:
  1) `tokenizer.json`   — modified HF-format tokenizer (long tokens removed,
                           IDs compacted, special tokens added)
  2) `token_bytes.pt`   — (final_vocab_size, max_bytes) byte-ID table
  3) `token_bytes.npy`  — same table as numpy array
  4) `artifact_meta.json` — vocabulary and special-token metadata

Algorithm:
  1. Identify all BPE vocab tokens whose byte representation exceeds max_bytes.
  2. Walk BPE merges in priority order.  Any merge whose input or output is in
     the removed set is dropped, and its output is added to the removed set
     (transitively unreachable).
  3. Surviving vocab tokens receive new contiguous IDs (preserving original
     relative order).  Added/special tokens are placed after the BPE vocab.
  4. A new <|mask|> token is appended.  Final vocab is padded to a multiple.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch


HEX_BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


# ---------------------------------------------------------------------------
# Byte decoding helpers
# ---------------------------------------------------------------------------

def _gpt2_bytes_to_unicode_decoder() -> Dict[str, int]:
    """Return GPT-2 byte-level unicode decoder map (char -> byte)."""
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]

    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    chars = [chr(c) for c in cs]
    return {ch: b for b, ch in zip(bs, chars)}


_GPT2_DECODER = _gpt2_bytes_to_unicode_decoder()


def _token_str_to_bytes(token_str: str) -> bytes:
    """Decode a BPE vocab token string to its raw byte representation."""
    m = HEX_BYTE_RE.match(token_str)
    if m:
        return bytes([int(m.group(1), 16)])

    if token_str and all(ch in _GPT2_DECODER for ch in token_str):
        return bytes(_GPT2_DECODER[ch] for ch in token_str)

    return token_str.encode("utf-8")


# ---------------------------------------------------------------------------
# Core: modify tokenizer
# ---------------------------------------------------------------------------

def _round_up_to_multiple(x: int, multiple: int) -> int:
    if multiple <= 0:
        return x
    return int(math.ceil(x / multiple) * multiple)


def build_artifacts(
    source: str,
    output_dir: Path,
    max_bytes: int,
    pad_byte_id: int,
    vocab_multiple: int,
    add_special_tokens: List[str],
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load base tokenizer.json
    # ------------------------------------------------------------------
    from transformers import AutoTokenizer

    hf_tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True)
    if not hf_tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer required")

    # We work on the raw JSON, not through the HF API, so we can
    # manipulate vocab and merges directly.
    base_json_str = hf_tokenizer.backend_tokenizer.to_str()
    tok_data = json.loads(base_json_str)

    base_vocab: Dict[str, int] = tok_data["model"]["vocab"]
    base_merges: List[List[str]] = tok_data["model"]["merges"]
    base_added: List[dict] = tok_data.get("added_tokens", [])

    base_vocab_size = len(base_vocab)

    # ------------------------------------------------------------------
    # 2. Identify tokens > max_bytes
    # ------------------------------------------------------------------
    removed_tokens: set[str] = set()
    for token_str in base_vocab:
        if len(_token_str_to_bytes(token_str)) > max_bytes:
            removed_tokens.add(token_str)

    initial_removed = len(removed_tokens)

    # ------------------------------------------------------------------
    # 3. Walk merges: drop any merge touching the removed set,
    #    propagate removals transitively.
    # ------------------------------------------------------------------
    new_merges: List[List[str]] = []
    for merge in base_merges:
        a, b = merge[0], merge[1]
        result = a + b
        if a in removed_tokens or b in removed_tokens or result in removed_tokens:
            removed_tokens.add(result)
            continue
        new_merges.append(merge)

    transitive_removed = len(removed_tokens) - initial_removed

    # ------------------------------------------------------------------
    # 4. Build surviving BPE vocab with compacted IDs
    # ------------------------------------------------------------------
    # Sort by original ID to preserve relative order.
    added_token_strs = {entry["content"] for entry in base_added}

    surviving_bpe = [
        (token_str, old_id)
        for token_str, old_id in base_vocab.items()
        if token_str not in removed_tokens and token_str not in added_token_strs
    ]
    surviving_bpe.sort(key=lambda x: x[1])

    new_vocab: Dict[str, int] = {}
    old_to_new: Dict[int, int] = {}
    for new_id, (token_str, old_id) in enumerate(surviving_bpe):
        new_vocab[token_str] = new_id
        old_to_new[old_id] = new_id

    next_id = len(surviving_bpe)

    # ------------------------------------------------------------------
    # 5. Place added/special tokens after BPE vocab
    # ------------------------------------------------------------------
    # Sort by original ID.
    sorted_added = sorted(base_added, key=lambda e: e["id"])
    new_added: List[dict] = []
    for entry in sorted_added:
        old_id = entry["id"]
        new_entry = dict(entry)
        new_entry["id"] = next_id
        old_to_new[old_id] = next_id
        new_added.append(new_entry)
        next_id += 1

    # ------------------------------------------------------------------
    # 6. Add requested new special tokens (e.g. <|mask|>)
    # ------------------------------------------------------------------
    new_special_ids: Dict[str, int] = {}
    for tok in add_special_tokens:
        if not tok:
            continue
        # Check if it already exists.
        existing_entry = None
        for entry in new_added:
            if entry["content"] == tok:
                existing_entry = entry
                break
        if existing_entry is not None:
            new_special_ids[tok] = existing_entry["id"]
        else:
            new_special_ids[tok] = next_id
            new_added.append({
                "id": next_id,
                "content": tok,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            })
            next_id += 1

    unpadded_vocab_size = next_id
    final_vocab_size = _round_up_to_multiple(unpadded_vocab_size, vocab_multiple)

    # ------------------------------------------------------------------
    # 7. Assemble modified tokenizer.json
    # ------------------------------------------------------------------
    tok_data["model"]["vocab"] = new_vocab
    tok_data["model"]["merges"] = new_merges
    tok_data["added_tokens"] = new_added

    tokenizer_json_path = output_dir / "tokenizer.json"
    with open(tokenizer_json_path, "w") as f:
        json.dump(tok_data, f, ensure_ascii=False)

    # ------------------------------------------------------------------
    # 8. Build token_bytes table
    # ------------------------------------------------------------------
    token_bytes = torch.full((final_vocab_size, max_bytes), pad_byte_id, dtype=torch.long)

    for token_str, new_id in new_vocab.items():
        raw = _token_str_to_bytes(token_str)
        assert len(raw) <= max_bytes, f"Surviving token {token_str!r} has {len(raw)} bytes > {max_bytes}"
        for i, b in enumerate(raw):
            token_bytes[new_id, i] = b

    # Special/added tokens get all-pad (no byte representation).
    # They rely on the token-private embedding path, not the shared byte path.

    token_bytes_pt = output_dir / "token_bytes.pt"
    torch.save(token_bytes, token_bytes_pt)

    import numpy as np
    token_bytes_npy = output_dir / "token_bytes.npy"
    np.save(token_bytes_npy, token_bytes.numpy())

    # ------------------------------------------------------------------
    # 9. Resolve special token IDs in the new ID space
    # ------------------------------------------------------------------
    def _resolve_special(attr_name: str) -> Optional[int]:
        old_id = getattr(hf_tokenizer, f"{attr_name}_token_id", None)
        if old_id is not None and old_id in old_to_new:
            return old_to_new[old_id]
        return None

    # ------------------------------------------------------------------
    # 10. Write metadata
    # ------------------------------------------------------------------
    meta = {
        "source": source,
        "max_bytes": max_bytes,
        "pad_byte_id": pad_byte_id,
        "base_vocab_size": base_vocab_size + len(base_added),
        "removed_long_tokens": initial_removed,
        "removed_transitive": transitive_removed,
        "removed_total": len(removed_tokens),
        "bpe_vocab_size": len(new_vocab),
        "final_vocab_size": final_vocab_size,
        "unpadded_vocab_size": unpadded_vocab_size,
        "vocab_multiple": vocab_multiple,
        "merges_base": len(base_merges),
        "merges_final": len(new_merges),
        "added_token_count": len(new_added),
        "new_special_token_ids": new_special_ids,
        "special_tokens": {
            "mask": new_special_ids.get("<|mask|>"),
            "bos": _resolve_special("bos"),
            "eos": _resolve_special("eos"),
            "pad": _resolve_special("pad"),
            "unk": _resolve_special("unk"),
        },
        "artifacts": {
            "tokenizer_json": tokenizer_json_path.name,
            "token_bytes_pt": token_bytes_pt.name,
            "token_bytes_npy": token_bytes_npy.name,
        },
    }

    meta_path = output_dir / "artifact_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a modified YAMIT tokenizer with long tokens removed from vocab/merges"
    )
    p.add_argument(
        "--source",
        required=True,
        help="HuggingFace tokenizer/model name or local tokenizer path",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write modified tokenizer + artifacts",
    )
    p.add_argument(
        "--max-bytes",
        type=int,
        default=16,
        help="Maximum bytes per token for composite slots (default: 16)",
    )
    p.add_argument(
        "--pad-byte-id",
        type=int,
        default=256,
        help="Pad byte symbol ID in token_bytes table (default: 256)",
    )
    p.add_argument(
        "--vocab-multiple",
        type=int,
        default=256,
        help="Pad final vocab size to this multiple (default: 256)",
    )
    p.add_argument(
        "--add-special-token",
        action="append",
        default=["<|mask|>"],
        help="Extra special token to reserve in final vocab (repeatable)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    extra_specials = [t for t in args.add_special_token if t]

    meta = build_artifacts(
        source=args.source,
        output_dir=output_dir,
        max_bytes=args.max_bytes,
        pad_byte_id=args.pad_byte_id,
        vocab_multiple=args.vocab_multiple,
        add_special_tokens=extra_specials,
    )

    print("\nComposite tokenizer artifacts written:")
    print(f"  output_dir          : {output_dir}")
    print(f"  base_vocab_size     : {meta['base_vocab_size']:,}")
    print(f"  removed (long)      : {meta['removed_long_tokens']:,}")
    print(f"  removed (transitive): {meta['removed_transitive']:,}")
    print(f"  removed (total)     : {meta['removed_total']:,}")
    print(f"  bpe_vocab_size      : {meta['bpe_vocab_size']:,}")
    print(f"  final_vocab_size    : {meta['final_vocab_size']:,}")
    print(f"  merges              : {meta['merges_base']:,} -> {meta['merges_final']:,}")


if __name__ == "__main__":
    main()
