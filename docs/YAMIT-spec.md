# YAMIT: Yet Another Minorly Improved Transformer

**Architecture Specification**  
**Status:** Draft  
**Date:** 2026-02-22

---

## 1. Overview

YAMIT combines four ideas into one trainable stack:

1. **MLA dense attention** (DeepSeek-style latent KV compression, no sparse indexer in v1).
2. **ReFusion AR+MDM objective** (joint autoregressive + masked diffusion training).
3. **Composite embeddings + Composite-PIT head** (byte-structured token representation).
4. **Quartet-style FP4 training** (native FP4 GEMMs with higher-precision safety paths).

The v1 goal is correctness and stable training on a small model, then speed via custom kernels, then production-scale training at 350M+ parameters.

---

## 2. Scope and Non-Goals

### In Scope (v1)

- Dense MLA attention with compressed KV cache.
- End-to-end ReFusion-style AR+MDM training and iterative sampler.
- Modified tokenizer generation (long tokens removed from vocab/merges) and composite byte table.
- Quartet kernel port and FP4 mixed-precision training policy.

### Explicitly Out of Scope (v1)

- DeepSeek DSA lightning indexer and sparse top-k masking.
- MoE architecture changes.
- Multi-stage post-training RL stack.

### Planned Later

- DSA indexer with dense warmup and sparse curriculum.
- Optional MoE variant.

---

## 3. Model Configurations

Two configs are defined. Model-S is the correctness and kernel bring-up target. Model-P is the production train target.

### 3.1 Model-S (Correctness)

| Parameter | Value |
|---|---:|
| target params | ~130M |
| d_model | 768 |
| n_layers | 16 |
| n_heads | 12 |
| q_compress_dim | 192 |
| kv_compress_dim | 96 |
| qk_nope_head_dim | 64 |
| qk_rope_head_dim | 32 |
| v_head_dim | 64 |
| mlp_hidden | 2048 |
| max_seq_len | 4096 |
| max_bytes_per_token | 16 |
| composite shared dims/slot | 36 |
| composite token dims/slot | 12 |

### 3.2 Model-P (Production)

| Parameter | Value |
|---|---:|
| target params | ~347M |
| d_model | 1024 |
| n_layers | 28 |
| n_heads | 16 |
| q_compress_dim | 256 |
| kv_compress_dim | 128 |
| qk_nope_head_dim | 64 |
| qk_rope_head_dim | 32 |
| v_head_dim | 64 |
| mlp_hidden | 2816 |
| max_seq_len | 4096 |
| max_bytes_per_token | 16 |
| composite shared dims/slot | 48 |
| composite token dims/slot | 16 |

### 3.3 Notes on Naming

`q_compress_dim` and `kv_compress_dim` are architectural bottlenecks in MLA, not LoRA adapters.

---

## 4. Tokenizer and Vocabulary

### 4.1 Base Tokenizer

- Base tokenizer: `Qwen/Qwen3-8B` tokenizer.
- Base vocab: 151,669 tokens.

### 4.2 Long-Token Surgery

- Tokens with byte length > 16 are disallowed by composite constraints.
- A deterministic retokenization map replaces such tokens at preprocessing time.
- The map is versioned and saved as an artifact.

### 4.3 Special Tokens

- `mask_token`: diffusion mask token.
- `bos_token`: generation start token.
- `eos_token`: end token.
- `pad_token`: standard padding token.

Final vocab is padded to nearest multiple of **256** (locked). With 151,669 base + 2 special tokens = 151,671, padded to **151,808**.

---

## 5. Composite Embedding

### 5.1 Representation

- Token embedding is split into 16 byte slots.
- Each slot has `dims_per_slot = d_model / 16`.
- Slot vector = `[shared_byte_subvector ; token_private_subvector]`.

### 5.2 Inputs and Tables

- `token_bytes[vocab_size, 16]` stores byte IDs per token with padding byte index.
- **Byte symbol count: 257** (0..255 for byte values, 256 for pad symbol).
- `num_byte_symbols = 257`.

### 5.3 Parameters

- `byte_memory[num_byte_symbols, shared_per_slot]`
- `byte_chol_raw[shared_per_slot, shared_per_slot]` (global, shared across all 16 slots)
- `token_embed[vocab_size, 16 * token_per_slot]`

### 5.4 Forward (Composite PIT Embedding)

```
byte_ids = token_bytes[token_ids]                      # (..., 16)
z_shared = byte_memory[byte_ids]                       # (..., 16, shared_per_slot)

# PIT inverse transform (embed direction = Z @ T^{-1}):
L = _byte_chol_factor(byte_chol_raw)                   # stabilized Cholesky factor (FP32)
x_shared = cholesky_solve(z_shared^T, L)^T             # (..., 16, shared_per_slot)

tok = token_embed(token_ids)                           # (..., 16 * token_per_slot)
tok = tok.view(..., 16, token_per_slot)

out = cat([x_shared, tok], dim=-1)                     # (..., 16, dims_per_slot)
return out.reshape(..., d_model)
```

The embedding applies `T^{-1}` to the byte memory vectors via `cholesky_solve`. The head applies `T` (see Section 8.2). This is the PIT duality.

### 5.5 Mask Override

When `input_id == mask_token_id`, embedding is overridden by a learned `mask_embed[d_model]`.

### 5.6 Stability

PIT Cholesky factor is stabilized via `softplus(raw_diagonal) + eps` (eps = 1e-6, configurable via `PIT_EPS`). This is a smooth lower bound, not a hard floor. The `cholesky_solve` in embedding and the `L @ L^T` matmul in the head both run in FP32.

---

## 6. MLA Dense Attention (v1)

### 6.1 Design

MLA stores compressed KV latent cache (`kv_compress_dim`) plus rope component, instead of full K/V per head.

### 6.2 Projections

- Query path:
  - `wq_a: d_model -> q_compress_dim`
  - `q_norm: RMSNorm(q_compress_dim)`
  - `wq_b: q_compress_dim -> n_heads * (qk_nope_head_dim + qk_rope_head_dim)`
  - Split output into `q_nope` and `q_pe`; apply RoPE to `q_pe` only.
- KV path:
  - `wkv_a: d_model -> kv_compress_dim + qk_rope_head_dim`
  - **Split** output into `kv_latent` (first `kv_compress_dim` dims) and `k_pe` (last `qk_rope_head_dim` dims).
  - `kv_norm: RMSNorm(kv_compress_dim)` applied to `kv_latent` only. `k_pe` **bypasses** the norm.
  - `wkv_b: kv_compress_dim -> n_heads * (qk_nope_head_dim + v_head_dim)` applied to normed `kv_latent`.
  - Apply RoPE to `k_pe`.
- Output projection:
  - `wo: n_heads * v_head_dim -> d_model`
- KV cache stores `kv_latent` and `k_pe` separately (not the expanded K/V).

### 6.3 Modes

- **Prefill mode:** materialize K/V from compressed latent for multi-token pass.
- **Decode mode:** use absorbed path to score via latent cache directly when possible.

### 6.4 Position IDs

Attention must support arbitrary `position_ids` (non-sequential order) for ReFusion slot shuffling while preserving original positions.

### 6.5 Numeric Constants

| Constant | Value |
|---|---|
| attention scale factor | `1 / sqrt(qk_nope_head_dim + qk_rope_head_dim)` = `1 / sqrt(96)` for both Model-S and Model-P |
| RoPE theta (base) | 10000.0 |
| RoPE dimensions | `qk_rope_head_dim` = 32 (applied to rope slice only) |
| RMSNorm epsilon | 1e-5 |
| attention bias | False (no bias in Q/K/V/O projections) |
| attention dropout | 0.0 |

### 6.6 RoPE

- RoPE applied to rope dimensions only.
- Context extension via linear scaling: `factor = ceil(seq_len / orig_ctx_len)` when sequence length exceeds original context.

### 6.7 DSA Placeholder

DSA indexer is disabled in v1, but attention API keeps optional sparse-mask hook for v2 integration.

---

## 7. Transformer Block

Pre-norm residual block:

1. `h = x + MLA(attn_norm(x), position_ids)`
2. `x = h + SwiGLU_MLP(mlp_norm(h))`

Model output uses final RMSNorm before LM head.

### 7.1 Weight Initialization

| Component | Method |
|---|---|
| `nn.Linear` weights | `normal_(mean=0.0, std=0.02)` |
| `nn.Linear` biases | `zeros_()` |
| `nn.Embedding` weights | `normal_(mean=0.0, std=0.02)` |
| RMSNorm weights | `fill_(1.0)` |
| PIT `byte_memory` | QR orthogonal init: `Q, _ = torch.linalg.qr(randn(257, shared_per_slot))` (default; fallback `normal_(0, 0.02)` if `orth_init=False`) |
| PIT `byte_chol_raw` | off-diagonal zeros; diagonal = `log(expm1(1.0))` ≈ 0.5414 so that `softplus(diag) = 1.0`, making `T = I` at init |
| PIT `token_embed` | `normal_(mean=0.0, std=0.02)` |
| PIT `token_out_bias` | `zeros_()` |
| composite `token_up` adapter | `zeros_()` (gate starts closed) |
| `mask_embed` | `randn(d_model) * 0.02` |

### 7.2 Head Weight Tying Policy

The composite-PIT head does **not** use standard weight tying (where LM head weight = embedding weight transposed). Instead:

- `byte_memory`, `byte_chol_raw`, and `token_embed` are **shared** between embedding and head (same Python object via `CompositePITTokenInterface`).
- `token_out_bias` is **head-only** (used in `project()` but not `embed()`).
- There is no separate `lm_head.weight` matrix.

The duality: embedding applies `T^{-1}` (via `cholesky_solve`), head applies `T` (via `L @ L^T`). Their composition cancels the metric: `(Z T^{-1})(T Z^T) = Z Z^T`.

---

## 8. Composite-PIT LM Head

### 8.1 Structure

- Hidden state `h` (shape `[B, T, d_model]`) is reshaped into 16 slots: `h_slots` (shape `[B, T, 16, dims_per_slot]`).
- Each slot is split into shared subvector (`shared_per_slot` dims) and token-private subvector (`token_per_slot` dims).

### 8.2 Logit Computation

```
# Cholesky factor with softplus-stabilized diagonal:
raw = tril(byte_chol_raw)                             # [shared_per_slot, shared_per_slot]
L = raw with diagonal replaced by softplus(raw_diag) + eps   # eps = 1e-6
T = L @ L^T                                           # PIT Gram matrix (SPD, starts as I at init)

# Shared path (PIT):
h_shared = h_slots[:, :, :, :shared_per_slot]         # [B, T, 16, shared_per_slot]
g = h_shared @ L @ L^T                                # [B, T, 16, shared_per_slot]
byte_patterns = byte_memory[token_bytes]               # [V, 16, shared_per_slot]
logits_shared = einsum('btsd,vsd->btv', g, byte_patterns)

# Token-private path (linear + bias):
h_private = h_slots[:, :, :, shared_per_slot:]         # [B, T, 16, token_per_slot]
h_tok_flat = h_private.reshape(B, T, 16 * token_per_slot)
logits_token = h_tok_flat @ token_embed.weight^T + token_out_bias   # [B, T, V]

# Final:
logits = logits_shared + logits_token                  # [B, T, V]
```

The head computes `h @ T @ Z^T` in the shared subspace, where `T = L L^T` is the PIT Gram matrix and `Z` is `byte_memory[token_bytes]`. The complementary embedding computes `Z @ T^{-1}` via `cholesky_solve` (see Section 5.4). This duality is the pseudo-inverse tying: the product `(Z T^{-1})(T Z^T) = Z Z^T` cancels the metric.

### 8.3 Parameters

- `byte_memory[257, shared_per_slot]`: shared between embedding and head.
- `byte_chol_raw[shared_per_slot, shared_per_slot]`: shared between embedding and head.
- `token_embed[vocab_size, 16 * token_per_slot]`: shared between embedding and head.
- `token_out_bias[vocab_size]`: head-only bias on the token-private path.

### 8.4 Why This Matters

- Shared byte structure gives compositional signal for denoising masked tokens (MDM benefits from byte-level structure).
- Token-private branch handles lexical specificity.
- Head parameter count is ~4x smaller than a standard `d_model x vocab_size` linear head.

### 8.5 Precision

- Shared PIT path (Cholesky factorization and solve): FP32 with `softplus(diag) + 1e-6` stabilization.
- Token-private linear path: participates in native FP4 GEMMs (same as other linear layers).

---

## 9. ReFusion Training Objective (AR+MDM)

### 9.1 Forward Process

Given sequence window with `prompt_len` and response region:

1. Sample slot size from configured set (default `[4, 8, 16, 32]`).
2. Partition response tokens into slots.
3. Sample mask probability: `t ~ Uniform(0, 1)`, then `p_mask = (1 - eps) * t + eps` where `eps = 1e-3`. This gives `p_mask ~ Uniform(0.001, 1.0)`.
4. Each slot is independently masked with probability `p_mask`; unmasked slots become AR slots.
5. Shuffle unmasked AR slots; keep masked slots ordered.
6. Replace masked slot tokens with `mask_token`.
7. Build labels:
   - AR slots: next-token labels inside slot.
   - MDM slots: original token labels at masked positions.
8. Use original token positions as `position_ids`.

For raw pretraining from scratch, `prompt_len = 0`.

### 9.2 Loss

Formulas (following ReFusion):

```
L_AR  = (1 / |AR_tokens|) * sum_{i in AR} CE(logit_i, label_i)

L_MDM = (1 / B) * sum_{i in masked} CE(logit_i, label_i) / (p_mask_i * answer_length_i)

L_total = L_AR + L_MDM
```

Where:
- `1/p_mask` is the importance weight per masked token (inverse of masking probability).
- `1/answer_length` normalizes each token's contribution by the total response length for that sample.
- `B` is the batch size.
- AR loss uses standard mean reduction.
- The two losses are summed with equal weight (no lambda coefficient).

For pretraining with `prompt_len = 0`, `answer_length` equals the full sequence length.

### 9.3 Packing Rule

Slot operations must not cross document boundaries when using packed data.

---

## 10. Diffusion Sampling

### 10.1 Algorithm

Iterative block/slot decode with verification:

1. Split generation target into serial blocks.
2. For current masked slots, run MDM prediction.
3. Select candidate slots by confidence threshold.
4. Run AR verification on candidates.
5. Accept verified prefixes, remask remainder.
6. Repeat refinement until block converges or max refinement reached.
7. Commit accepted tokens, update cache, move to next block.

### 10.2 Default Sampler Parameters

| Parameter | Default | Notes |
|---|---|---|
| slot_size | 8 | generation slot granularity |
| serial_num_blocks | 2 | number of serial blocks (`block_length = gen_length / serial_num_blocks`) |
| slot_threshold | 0.9 | minimum softmax probability of first token in slot for acceptance |
| token_threshold | 0.9 | minimum softmax probability for per-token acceptance within slot |
| max_refinement_iters | slot_size | up to `slot_size` refinement loops per block |
| temperature | 0.0 | 0 = greedy (argmax); >0 uses Gumbel sampling in float64 |
| force_accept_fallback | yes | if no slot exceeds `slot_threshold`, force-accept top-1 slot |

Slot confidence is based on the softmax probability of the **first token** in each slot. Token acceptance uses `cumprod` to find the longest contiguous accepted prefix within each slot (first token always accepted).

### 10.3 Cache

`DiffusionMLACache` stores per-layer MLA latent KV cache and rope cache with operations:

- `crop(seq_len)`
- `select_partial(indices)`
- `append(new_entries)`
- `batch_repeat(k)`
- `batch_select(indices)`

---

## 11. Precision Policy (Quartet FP4)

### 11.1 Core Rule

YAMIT is a native FP4 model — trained and deployed in 4-bit precision, no post-training quantization. All major linear-layer GEMMs run natively in FP4 via Quartet's approach; sensitive ops remain BF16/FP32.

### 11.2 Module Policy

| Component | Precision |
|---|---|
| Q/K/V/O projection GEMMs | FP4 |
| MLP up/gate/down GEMMs | FP4 |
| attention softmax | BF16/FP32 |
| RMSNorm stats | BF16/FP32 |
| RoPE ops | BF16 |
| loss/logits softmax | BF16/FP32 |
| PIT Cholesky + solve | FP32 |
| optimizer states | FP32 |

### 11.3 Quartet Path

**Forward (QuEST MXFP4):**
1. Apply Hadamard rotation: `x_had = x.view(-1, 32) @ H`, where `H` is the normalized Hadamard matrix (`H/sqrt(32)`).
2. Group into blocks of 32 elements.
3. Compute per-block scale: `std = sqrt(mean(x^2) - mean(x)^2)`, then `scale = gaussian_scale * std + 1e-8` where `gaussian_scale = 2.92247856 / 6.0 = 0.48708`.
4. Quantize to shared exponent: `shared_exp = 2^floor(log2(scale))`.
5. Round-to-nearest to FP4 levels: `{0, 0.5, 1, 1.5, 2, 3, 4, 6}` and negatives.
6. Clip mask: `mask = (|x_scaled| < 6)` saved for backward.
7. Dequantize: `x_fp4 = x_quantized * shared_exp`. Output stays in Hadamard-rotated domain.

**Backward (RTN MXFP4 with stochastic rounding):**
1. Apply Hadamard rotation to gradient error.
2. Use absmax scaling: `scale = max(|x_had|)` per block of 32.
3. Stochastic rounding to FP4 levels (probabilistic between upper/lower levels).
4. Re-randomize Hadamard matrix via random sign flips before each backward pass.
5. Both gradient matmuls quantize all operands: `grad_x = Q(error) @ Q(W^T)`, `grad_W = Q(error^T) @ Q(X^T)^T`.
6. Inverse Hadamard applied in backward autograd: `grad_input = grad_output @ H^T`.

**STE policy:**
- Forward (QuEST): clipped STE — gradient is zero for values outside FP4 representable range (masked by clip mask).
- Backward (RTN): pure STE — gradient passes through unchanged (only inverse Hadamard rotation applied).

**Constants:**
- Block size: 32 elements per quantization group.
- Hadamard dimension: 32.
- FP4 format: E2M1 with block scaling (MXFP4/NVFP4).
- Max representable FP4 value: 2.92247856 (before shared exponent).
- Aligned bias correction: multiply dequantized output by 1.009276 (optional, for AlignedAlbertTseng variant).

Kernels are ported from `Quartet/` and integrated behind runtime switch.

### 11.4 Runtime Switch

`--fp4_impl te|quartet`

- `te`: Transformer Engine fallback path.
- `quartet`: custom kernel path.

---

## 12. Data Pipeline

Token IDs must use `uint32`/`int32` storage to support 151k+ vocab.

Preprocessing: raw text -> modified Qwen3 tokenizer (long tokens removed) -> uint32 storage -> ReFusion forward process at train time.

Training details (optimizer, scheduler, data, batch size) are in `docs/plans/YAMIT-training.md`.

---

## 13. Source Component Mapping

- Base trainer target: `experiments/tier4/train_fineweb_vanilla.py`
- Quartet kernels source: `Quartet/src/models/quantization/quantizers/`
- ReFusion objective/sampler references: `ReFusion/train.py`, `ReFusion/generate.py`
- DeepSeek MLA references: `DeepSeek-V3.2-Exp/inference/model.py`

---

## 14. Future: DSA Indexer (v2)

When enabled later:

1. Dense warmup stage for indexer alignment.
2. Sparse stage with top-k token selection.
3. Detached indexer-input optimization path.
4. KL alignment loss for indexer.

This is intentionally deferred until YAMIT v1 is stable.

---

## Appendix A: Quick Param Notes

- Model-P estimate remains ~347M including composite-PIT and dense MLA stack.
- Removing indexer from v1 lowers parameter count relative to DSA-enabled design.
- Final numbers are validated by implemented model parameter report.

## Appendix B: Glossary

- **AR:** autoregressive next-token objective.
- **MDM:** masked diffusion modeling objective.
- **MLA:** multi-head latent attention with compressed KV latent cache.
- **PIT:** pseudo-inverse tying transform used in composite shared space.
- **DSA:** DeepSeek sparse attention indexer mechanism (not enabled in v1).
