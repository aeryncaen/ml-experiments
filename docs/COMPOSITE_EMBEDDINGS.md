# Composite Embeddings & PIT Head

## Overview

The composite embedding system factorizes token representations into
byte-level shared structure and token-specific learned parameters, using a
**matryoshka** layout that adapts slot sizes to each token's actual byte count.
On the output side, the **PIT (Parametric Inverse Transform)** head mirrors
this structure with a Cholesky-parameterized metric, enabling coupled
embed/head weight sharing through a single `CompositePITTokenInterface`.

All code lives in `experiments/tier4/train_fineweb_vanilla.py`.

## 1. Matryoshka Composite Embedding

### Core idea

Instead of one flat `d_model`-vector per token, each token's embedding is
assembled from its UTF-8 bytes. Two parameter tables are shared:

- **`byte_embed`**: `(257, byte_budget)` — 256 byte values + 1 pad entry.
  Shared across all tokens containing that byte value.
- **`token_params`**: `(V, max_tok_params)` — per-token learned parameters.

The key innovation is **matryoshka slicing**: tokens with fewer bytes get
*more dimensions per byte* (richer per-byte representations), while tokens
with more bytes get narrower slots. The byte embeddings are sliced from the
left (`[:bps]`), so the lowest-rank dimensions are always included —
analogous to Matryoshka Representation Learning.

### Layout

Given `d_model = 768`, `token_per_byte (tpb) = 8`, `max_bytes = 16`:

```
max_tok_params = 16 * tpb = 128
byte_budget    = d_model - max_tok_params = 640
```

For a token with N bytes:

```
bps       = byte_budget // N          (byte dims per slot)
tps       = max_tok_params // N       (token dims per slot)
slot_size = bps + tps
remainder = d_model - N * slot_size   (leftover token params + zero pad)
```

Each slot: `[byte_embed[byte_val][:bps] | token_params[:tps]]`

Full vector: `[slot_0 | slot_1 | ... | slot_{N-1} | remainder]`

All 128 token params are always used regardless of byte count.

### Examples (d_model=768, tpb=8)

| Bytes | bps | tps | slot | remainder | pad |
|-------|-----|-----|------|-----------|-----|
| 1     | 640 |  128| 768  | 0         | 0   |
| 3     | 213 |  42 | 255  | 3         | 1   |
| 4     | 160 |  32 | 192  | 0         | 0   |
| 6     | 106 |  21 | 127  | 6         | 4   |
| 8     |  80 |  16 | 96   | 0         | 0   |
| 16    |  40 |   8 | 48   | 0         | 0   |

### Parameter budget

With Qwen3 tokenizer (V=149,760):

- `byte_embed`: 257 × 640 = 164,480 params
- `token_params`: 149,760 × 128 = 19,169,280 params
- **Total: ~19.3M** vs standard embedding 149,760 × 768 = **115M** (5.9× compression)

### Implementation

`CompositeEmbedding` (class, `forward` method):

1. Look up each token's byte sequence and byte count N.
2. Group tokens by N (vectorized per-group loop over `_unique_n`).
3. For each group: slice `byte_embed` to `[:bps]`, slice `token_params` into
   N chunks of `tps`, concatenate per slot, flatten, handle remainder.
4. Sort-based restore to original token order (clean autograd, no in-place).

Assembly is **batch-only** — never rebuilds the full vocab table.

## 2. PIT (Parametric Inverse Transform)

### Concept

PIT couples the embedding and LM head through a shared pattern space.
Instead of separate embed/head weight matrices, a single set of
**pattern vectors** Z (assembled via matryoshka) is used for both directions,
mediated by a learned PSD metric T = L L^T (Cholesky-parameterized).

- **Embedding**: `x = T^{-1} z` (cholesky_solve in float32, then RMSNorm)
- **Projection**: `logits = (h @ T) @ Z^T + bias`

This guarantees that embed and head share the same token geometry, with T
acting as a learned inner product that can rotate/scale the space.

### `CompositePITTokenInterface`

Single nn.Module owning all shared parameters:

- `byte_memory`: `(257, byte_budget)` — nn.Parameter (not Embedding), same
  role as `byte_embed` but used with `F.embedding` for pattern assembly.
  Initialized with orthonormal rows via QR decomposition.
- `token_embed`: `(V, max_tok_params)` — per-token parameters.
- `chol_raw`: `(d_model, d_model)` — raw Cholesky factor. Diagonal entries
  pass through `softplus` to guarantee positive definiteness.
  Initialized to identity (diag = `log(expm1(1))`).
- `token_out_bias`: `(V,)` — per-token output bias.
- `embed_out_norm`: RMSNorm(d_model) applied after embedding.

Pattern assembly (`_assemble_patterns`) uses the same matryoshka logic as
`CompositeEmbedding`.

Two thin wrapper modules connect to the model:

- `CompositePITEmbedding.forward(token_ids)` → calls `interface.embed()`
- `CompositePITHead.forward(hidden)` → calls `interface.project()`

### LM_HEAD_TYPE=pit

Flat PIT head. Assembles patterns for the full vocabulary on every forward
pass. Exact logits, but O(V × d_model) compute per position.

## 3. Bucketed PIT Head

### Motivation

Full-vocab PIT is expensive for large V. The bucketed variant uses
**hierarchical softmax**: route to a bucket first, then score tokens within
that bucket only.

### Architecture (`BucketedCompositePITHead`)

Enabled via `LM_HEAD_TYPE=bucketed_pit`.

**Bucket assignment** — two modes:

- `PIT_BUCKET_MODE=hash` (default): deterministic hash
  `(token_id * 2654435761) % 2^32 % n_buckets`
- `PIT_BUCKET_MODE=semantic`: pre-computed cluster labels from
  `PIT_BUCKET_LABELS` (.npy file). Optional `PIT_BUCKET_CENTERS` for
  warm-starting the router.

**Router** — SwiGLU MLP:

```
router_logits = W_down(SiLU(W_gate(h)) * W_up(h))     # (B, T, K)
```

Hidden dim = `ceil(d_model * 8/3 / 256) * 256`.

**SiLU² correction** — learned gates (init 0 = pure softmax at start):

```
scores = softmax(logits) + α * SiLU(logits)²
```

Separate α for bucket and token levels.

### Training: `routed_cross_entropy`

Two-level exact hierarchical softmax (no approximation):

1. **Level 1 (bucket)**: CE loss on router logits vs target bucket.
2. **Level 2 (token)**: For each bucket, gather positions whose target is
   in that bucket, compute within-bucket logits via `_bucket_logits`,
   CE loss on local index. Sum over all buckets, divide by N.

```
total_loss = bucket_loss + token_loss
```

**Accuracy**: argmax across the router's top-k buckets, matching inference
behavior. For each active bucket, compute corrected token scores × bucket
confidence, track best token per position.

### Inference: `forward`

1. Route: top-k buckets per position.
2. For each active bucket: `_bucket_logits(g, b)` → corrected scores × confidence.
3. Scatter into full-vocab logits (rest = -inf).

### `_prepare_hidden` and `_bucket_logits`

- `_prepare_hidden(h)`: applies `g = h @ (L @ L^T)` — the Cholesky metric.
- `_bucket_logits(g, b)`: assembles patterns for bucket members via
  `_assemble_patterns`, computes `g @ Z^T + bias`.

## 4. Byte Table Construction

`_build_token_byte_table(vocab_size, max_bytes, pad_idx)`:

- Uses the Qwen3 tokenizer artifact at
  `experiments/yamit/tokenizer/artifacts/qwen3/token_bytes.npy`
  — shape `(149760, 16)`, pre-computed byte sequences per token.
- Tokens with fewer than `max_bytes` bytes are right-padded with `pad_idx=256`.
- Byte count distribution (Qwen3): avg 6.19, median 6, range 0–16.
  Peak at N=3 (25K tokens) and N=6 (29K tokens).

## 5. Configuration Reference

### Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPOSITE_EMBED` | `False` | Enable composite embedding |
| `COMPOSITE_TOKEN_DIMS` | `8` | Per-token dims per byte slot (tpb) |

### PIT Head

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_HEAD_TYPE` | `""` | Set to `pit` or `bucketed_pit` |
| `PIT_ORTH_INIT` | `True` | QR-orthonormal init for byte_memory |
| `PIT_EPS` | `1e-6` | Cholesky softplus epsilon |
| `PIT_MIN_DIAG` | `1e-3` | Cholesky minimum diagonal value |

### Bucketed PIT

| Variable | Default | Description |
|----------|---------|-------------|
| `PIT_N_BUCKETS` | `64` | Number of vocab buckets |
| `PIT_TOP_K` | `8` | Buckets scored per position at inference |
| `PIT_ROUTER_AUX_WEIGHT` | `0.01` | Router auxiliary loss weight |
| `PIT_BUCKET_MODE` | `hash` | `hash` or `semantic` |
| `PIT_BUCKET_LABELS` | `""` | Path to .npy cluster labels |
| `PIT_BUCKET_CENTERS` | `""` | Path to .npy cluster centers |

### Legacy / Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPOSITE_LORA` | `False` | Low-rank token adapter (additive) |
| `COMPOSITE_LORA_RANK` | `16` | LoRA rank |
| `COMPOSITE_CONV` | `False` | Residual Conv1d across byte positions |

## 6. Recommended Configuration

For training with bucketed PIT (the current default path):

```bash
COMPOSITE_EMBED=1
COMPOSITE_TOKEN_DIMS=8
LM_HEAD_TYPE=bucketed_pit
PIT_BUCKET_MODE=semantic
PIT_BUCKET_LABELS=experiments/tier4/data/gpt2wte_simcl_labels.npy
PIT_BUCKET_CENTERS=experiments/tier4/data/gpt2wte_simcl_centers.npy
PIT_N_BUCKETS=64
PIT_TOP_K=8
```
