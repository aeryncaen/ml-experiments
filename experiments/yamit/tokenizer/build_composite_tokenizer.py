#!/usr/bin/env python3
"""Build composite-tokenizer artifacts from a HuggingFace tokenizer.

This script prepares the artifacts required by YAMIT composite embeddings:

1) `tokenizer.json` copy from a base HF tokenizer
2) `surgery_map.json` mapping long tokens (> max_bytes) to byte-token ID sequences
3) `id_remap.json` mapping base token IDs -> pruned token IDs
4) `token_bytes.pt` table of shape [final_vocab_size, max_bytes]
5) `artifact_meta.json` with vocab/special-token metadata

Notes:
- YAMIT does not require rewriting BPE merges. Long tokens are handled by
  deterministic retokenization ("long-token surgery") at preprocessing time.
- A tokenizer is considered compatible only if full byte fallback is available
  (all 256 byte values map to tokenizer IDs, via <0xHH> tokens or equivalent).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoTokenizer


HEX_BYTE_RE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")


def _sanitize_regex_for_re2(pattern: str) -> str:
    """Best-effort regex sanitization for Go/RE2 compatibility.

    Current Qwen tokenizer issue:
      - ``(?!\\S)`` negative lookahead is unsupported in Go regexp.
        In the GPT-style pretokenizer pattern this branch already has a
        fallback ``\\s+`` alternative, so removing the lookahead keeps
        behavior close enough for tokenization.
    """
    pattern = pattern.replace("(?!\\S)", "")
    return pattern


def _walk_and_sanitize_regex(node) -> tuple[object, int]:
    """Recursively sanitize tokenizer JSON regex fields.

    Returns:
      (updated_node, n_changes)
    """
    n_changes = 0
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "Regex" and isinstance(v, str):
                nv = _sanitize_regex_for_re2(v)
                if nv != v:
                    n_changes += 1
                out[k] = nv
            else:
                vv, c = _walk_and_sanitize_regex(v)
                out[k] = vv
                n_changes += c
        return out, n_changes
    if isinstance(node, list):
        out = []
        for v in node:
            vv, c = _walk_and_sanitize_regex(v)
            out.append(vv)
            n_changes += c
        return out, n_changes
    return node, 0


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


def _decode_single_token_bytes(
    tokenizer,
    token_id: int,
    token_str: str,
    byte_decoder: Optional[Dict[str, int]],
    gpt2_decoder: Dict[str, int],
) -> bytes:
    """Token-id -> bytes using tokenizer-native byte decoders when available."""
    # Explicit <0xHH> tokens.
    m = HEX_BYTE_RE.match(token_str)
    if m:
        return bytes([int(m.group(1), 16)])

    # Tokenizer-provided byte decoder (char -> byte).
    if byte_decoder and token_str and all(ch in byte_decoder for ch in token_str):
        return bytes(byte_decoder[ch] for ch in token_str)

    # GPT2-style byte decoder fallback.
    if token_str and all(ch in gpt2_decoder for ch in token_str):
        return bytes(gpt2_decoder[ch] for ch in token_str)

    # Last-resort generic decode.
    text = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return text.encode("utf-8")


def _build_id_to_token(tokenizer, vocab_size: int) -> Dict[int, str]:
    id_to_token: Dict[int, str] = {}
    for token_id in range(vocab_size):
        tok = tokenizer.convert_ids_to_tokens(token_id)
        if tok is None:
            tok = ""
        id_to_token[token_id] = tok
    return id_to_token


def _find_byte_token_ids(
    tokenizer,
    vocab_size: int,
    id_to_token: Dict[int, str],
    byte_decoder: Optional[Dict[str, int]],
    gpt2_decoder: Dict[str, int],
) -> Dict[int, int]:
    """Find tokenizer IDs corresponding to each byte value 0..255.

    Strategy:
      1) Prefer explicit <0xHH> token names.
      2) Fallback to single-byte decode matches.
    """
    byte_to_id: Dict[int, int] = {}

    # Pass 1: explicit <0xHH> token names.
    for token_id, tok in id_to_token.items():
        m = HEX_BYTE_RE.match(tok)
        if not m:
            continue
        b = int(m.group(1), 16)
        byte_to_id[b] = token_id

    # Pass 2: fallback to single-byte decode.
    if len(byte_to_id) < 256:
        for token_id in range(vocab_size):
            if len(byte_to_id) == 256:
                break
            raw = _decode_single_token_bytes(
                tokenizer,
                token_id,
                id_to_token[token_id],
                byte_decoder=byte_decoder,
                gpt2_decoder=gpt2_decoder,
            )
            if len(raw) == 1:
                b = raw[0]
                if b not in byte_to_id:
                    byte_to_id[b] = token_id

    missing = [b for b in range(256) if b not in byte_to_id]
    if missing:
        missing_preview = ", ".join(str(b) for b in missing[:16])
        raise RuntimeError(
            "Tokenizer is incompatible with composite long-token surgery: "
            f"missing {len(missing)} byte fallback IDs. "
            f"First missing bytes: {missing_preview}"
        )
    return byte_to_id


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
    prune_long_tokens: bool,
) -> Dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("Fast tokenizer required (tokenizer.json export unsupported for slow tokenizer)")

    # Save exact tokenizer.json used for tokenization pipeline.
    tokenizer_json_path = output_dir / "tokenizer.json"
    tokenizer.backend_tokenizer.save(str(tokenizer_json_path))

    # Emit a Go/RE2-compatible variant for the Go tokenizer backend.
    tokenizer_go_json_path = output_dir / "tokenizer_go.json"
    with open(tokenizer_json_path) as f:
        tok_obj = json.load(f)
    tok_go_obj, regex_changes = _walk_and_sanitize_regex(tok_obj)
    with open(tokenizer_go_json_path, "w") as f:
        json.dump(tok_go_obj, f)

    base_vocab_size = len(tokenizer)
    id_to_token = _build_id_to_token(tokenizer, base_vocab_size)

    byte_decoder = getattr(tokenizer, "byte_decoder", None)
    if not isinstance(byte_decoder, dict):
        byte_decoder = None
    gpt2_decoder = _gpt2_bytes_to_unicode_decoder()

    byte_to_id = _find_byte_token_ids(
        tokenizer,
        base_vocab_size,
        id_to_token,
        byte_decoder=byte_decoder,
        gpt2_decoder=gpt2_decoder,
    )

    # Build surgery map and byte table for base vocab.
    surgery_map: Dict[int, List[int]] = {}
    long_token_ids: List[int] = []
    keep_token_ids: List[int] = []

    base_token_bytes = torch.full((base_vocab_size, max_bytes), pad_byte_id, dtype=torch.long)

    for token_id in range(base_vocab_size):
        raw = _decode_single_token_bytes(
            tokenizer,
            token_id,
            id_to_token[token_id],
            byte_decoder=byte_decoder,
            gpt2_decoder=gpt2_decoder,
        )
        if len(raw) > max_bytes:
            long_token_ids.append(token_id)
            surgery_map[token_id] = [byte_to_id[b] for b in raw]
            continue

        keep_token_ids.append(token_id)
        for i, b in enumerate(raw):
            base_token_bytes[token_id, i] = b

    # Build old->new ID remap.
    # - If prune_long_tokens=True: remove >max_bytes tokens from model ID space.
    # - Else: identity remap for base vocab.
    old_to_new: Dict[int, int] = {}
    new_to_old: Dict[int, int] = {}

    if prune_long_tokens:
        for new_id, old_id in enumerate(keep_token_ids):
            old_to_new[old_id] = new_id
            new_to_old[new_id] = old_id
        next_id = len(keep_token_ids)
    else:
        for old_id in range(base_vocab_size):
            old_to_new[old_id] = old_id
            new_to_old[old_id] = old_id
        next_id = base_vocab_size

    # Special token ID mapping (new IDs if token not in kept/base vocab).
    special_token_ids: Dict[str, int] = {}
    for tok in add_special_tokens:
        if not tok:
            continue
        existing = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(existing, int) and existing in old_to_new:
            special_token_ids[tok] = old_to_new[existing]
        else:
            special_token_ids[tok] = next_id
            next_id += 1

    unpadded_vocab_size = next_id
    final_vocab_size = _round_up_to_multiple(unpadded_vocab_size, vocab_multiple)

    token_bytes = torch.full((final_vocab_size, max_bytes), pad_byte_id, dtype=torch.long)

    # Copy base bytes through remap.
    for old_id, new_id in old_to_new.items():
        token_bytes[new_id] = base_token_bytes[old_id]

    # Fill bytes for newly added specials if they were assigned new IDs.
    for tok, tok_id in special_token_ids.items():
        if tok_id in new_to_old:
            continue
        raw = tok.encode("utf-8")
        if len(raw) <= max_bytes:
            for i, b in enumerate(raw):
                token_bytes[tok_id, i] = b

    # Keep surgery map in base-tokenizer ID space.
    # Pipeline order is: encode(base IDs) -> surgery(base IDs) -> id_remap(base->new IDs).
    remapped_surgery_map: Dict[int, List[int]] = {}
    for old_long_id, seq_old in surgery_map.items():
        remapped_surgery_map[old_long_id] = seq_old

    # Export artifacts.
    surgery_map_json = {str(k): v for k, v in sorted(remapped_surgery_map.items())}

    surgery_path = output_dir / "surgery_map.json"
    with open(surgery_path, "w") as f:
        json.dump(surgery_map_json, f, indent=2)

    id_remap_path = output_dir / "id_remap.json"
    with open(id_remap_path, "w") as f:
        json.dump({str(k): v for k, v in sorted(old_to_new.items())}, f, indent=2)

    token_bytes_pt = output_dir / "token_bytes.pt"
    torch.save(token_bytes, token_bytes_pt)

    token_bytes_npy = output_dir / "token_bytes.npy"
    import numpy as np

    np.save(token_bytes_npy, token_bytes.numpy())

    pruned_vocab_path = output_dir / "pruned_vocab.json"
    with open(pruned_vocab_path, "w") as f:
        json.dump(
            {
                "keep_token_ids": keep_token_ids,
                "long_token_ids": long_token_ids,
                "old_to_new_count": len(old_to_new),
            },
            f,
        )

    meta = {
        "source": source,
        "tokenizer_json": str(tokenizer_json_path.name),
        "tokenizer_go_json": str(tokenizer_go_json_path.name),
        "go_regex_sanitize_changes": regex_changes,
        "max_bytes": max_bytes,
        "pad_byte_id": pad_byte_id,
        "base_vocab_size": base_vocab_size,
        "pruned_vocab_size": len(old_to_new),
        "final_vocab_size": final_vocab_size,
        "unpadded_vocab_size": unpadded_vocab_size,
        "vocab_multiple": vocab_multiple,
        "prune_long_tokens": prune_long_tokens,
        "long_token_count": len(long_token_ids),
        "byte_token_count": len(byte_to_id),
        "special_token_ids": special_token_ids,
        "special_tokens": {
            "mask": tokenizer.mask_token,
            "bos": tokenizer.bos_token,
            "eos": tokenizer.eos_token,
            "pad": tokenizer.pad_token,
            "unk": tokenizer.unk_token,
        },
        "resolved_special_ids": {
            "mask": special_token_ids.get("<|mask|>", None),
            "bos": (
                old_to_new[tokenizer.bos_token_id]
                if getattr(tokenizer, "bos_token_id", None) in old_to_new
                else None
            ),
            "eos": (
                old_to_new[tokenizer.eos_token_id]
                if getattr(tokenizer, "eos_token_id", None) in old_to_new
                else None
            ),
            "pad": (
                old_to_new[tokenizer.pad_token_id]
                if getattr(tokenizer, "pad_token_id", None) in old_to_new
                else None
            ),
            "unk": (
                old_to_new[tokenizer.unk_token_id]
                if getattr(tokenizer, "unk_token_id", None) in old_to_new
                else None
            ),
        },
        "artifacts": {
            "surgery_map": surgery_path.name,
            "id_remap": id_remap_path.name,
            "token_bytes_pt": token_bytes_pt.name,
            "token_bytes_npy": token_bytes_npy.name,
            "pruned_vocab": pruned_vocab_path.name,
            "tokenizer_go_json": tokenizer_go_json_path.name,
        },
    }

    meta_path = output_dir / "artifact_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build YAMIT composite-tokenizer artifacts from any HF tokenizer"
    )
    p.add_argument(
        "--source",
        required=True,
        help="HuggingFace tokenizer/model name or local tokenizer path",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write tokenizer.json + composite artifacts",
    )
    p.add_argument(
        "--max-bytes",
        type=int,
        default=16,
        help="Maximum bytes per token for composite slots",
    )
    p.add_argument(
        "--pad-byte-id",
        type=int,
        default=256,
        help="Pad byte symbol ID in token_bytes table",
    )
    p.add_argument(
        "--vocab-multiple",
        type=int,
        default=256,
        help="Pad final vocab size to this multiple",
    )
    p.add_argument(
        "--add-special-token",
        action="append",
        default=["<|mask|>"],
        help="Extra special token to reserve in final vocab (repeatable)",
    )
    p.add_argument(
        "--no-prune-long-tokens",
        action="store_true",
        help="Disable long-token pruning (keeps base token ID space)",
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
        prune_long_tokens=not args.no_prune_long_tokens,
    )

    print("\nComposite tokenizer artifacts written:")
    print(f"  output_dir       : {output_dir}")
    print(f"  base_vocab_size  : {meta['base_vocab_size']:,}")
    print(f"  pruned_vocab_size: {meta['pruned_vocab_size']:,}")
    print(f"  final_vocab_size : {meta['final_vocab_size']:,}")
    print(f"  long_token_count : {meta['long_token_count']:,}")
    print(f"  go_regex_changes : {meta['go_regex_sanitize_changes']:,}")
    print(f"  surgery_map      : {meta['artifacts']['surgery_map']}")
    print(f"  id_remap         : {meta['artifacts']['id_remap']}")
    print(f"  tokenizer_go     : {meta['artifacts']['tokenizer_go_json']}")
    print(f"  token_bytes      : {meta['artifacts']['token_bytes_pt']}")


if __name__ == "__main__":
    main()
