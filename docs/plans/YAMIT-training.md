# YAMIT Training Plan

**Status:** Draft
**Date:** 2026-02-22

---

## 1. Baseline: SmolLM3 Training Setup

YAMIT training follows SmolLM3's recipe, scaled to our model sizes. Reference configs are at [huggingface/smollm](https://github.com/huggingface/smollm/tree/main/text/pretraining/smollm3) (nanotron YAML).

### 1.1 SmolLM3 Hyperparameters (3B model, for reference)

| Parameter | Value |
|---|---|
| optimizer | AdamW |
| betas | (0.9, 0.95) |
| weight decay | 0.1 (0.0 for embedding layers) |
| grad clip | 1.0 (max norm) |
| peak LR | 2e-4 |
| scheduler | WSD (warmup-stable-decay) |
| warmup | 2000 steps (linear) |
| decay | linear to 0 in final 10% of training |
| global batch size | 2.36M tokens |
| sequence length | 4096 |
| total tokens | 11.2T (3-stage) |
| precision | BF16 |
| framework | nanotron |
| hardware | 384 H100s, 24 days |

### 1.2 SmolLM3 Data Mix (3-stage)

| Stage | Tokens | Web | Code | Math |
|---|---|---|---|---|
| 1 (stable) | 0-8T | 85% (12% multilingual) | 12% | 3% |
| 2 (stable) | 8-10T | 75% (12% multilingual) | 15% | 10% |
| 3 (decay) | 10-11.1T | 63% (12% multilingual) | 24% | 13% |

### 1.3 SmolLM3 Architecture Choices We Do NOT Adopt

SmolLM3 uses GQA, NoPE (remove RoPE every 4th layer), and standard tied embeddings. YAMIT replaces these with MLA, full RoPE on rope dims in every layer, and composite-PIT tying. The training setup (optimizer, scheduler, data strategy) transfers; the architecture does not.

---

## 2. Scaling to YAMIT Model Sizes

SmolLM3 is 3B params trained on 11.2T tokens. YAMIT Model-S is ~130M, Model-P is ~347M. Token budgets and LR need scaling.

### 2.1 Scaling Guidelines

Quartet's scaling tables (from their experiments on 30M-3.2B models) suggest:

| Model size | LR | Tokens (Chinchilla-ish) |
|---|---|---|
| ~130M | ~6e-4 | ~3-10B |
| ~350M | ~3e-4 | ~10-20B |

SmolLM3's 2e-4 LR is for a 3B model. Smaller models generally use higher LR. The exact values need tuning — start with the Quartet scaling suggestions and adjust based on loss curves.

### 2.2 Model-S Training Config (Starting Point)

| Parameter | Value | Source |
|---|---|---|
| optimizer | AdamW, betas (0.9, 0.95) | SmolLM3 |
| weight decay | 0.1 (0.0 for embeddings and norms) | SmolLM3 |
| grad clip | 1.0 | SmolLM3 |
| peak LR | 6e-4 (tune) | Quartet 100M scaling |
| scheduler | WSD, 2000 warmup, linear decay final 10% | SmolLM3 |
| sequence length | 4096 | SmolLM3 / ReFusion slot structure |
| micro batch size | 8 (single GPU) | tier4 trainer default |
| total tokens | 3B (initial), scale up if loss hasn't plateaued | Quartet 30M baseline |

### 2.3 Model-P Training Config (Starting Point)

| Parameter | Value | Source |
|---|---|---|
| peak LR | 3e-4 (tune) | Quartet 200M scaling |
| total tokens | 10B+ | Quartet scaling |
| everything else | same as Model-S | — |

Model-P may need multi-GPU; distributed strategy decided at that time.

---

## 3. YAMIT-Specific Training Differences

### 3.1 ReFusion Dual Loss

Unlike SmolLM3's standard next-token CE, YAMIT trains with `L_total = L_AR + L_MDM` (see architecture spec Section 9). This means:

- Each training step runs the ReFusion forward process (slot partitioning, masking, shuffling).
- Two loss terms are computed and summed.
- The MDM loss uses `1/p_mask` importance weighting.
- `p_mask ~ Uniform(0.001, 1.0)` per sample, `slot_size ~ {4, 8, 16, 32}` per sample.
- For pretraining from scratch, `prompt_len = 0` (entire sequence is "response").

Intra-document masking (which SmolLM3 uses) intersects with ReFusion's slot boundary constraint: slots must not cross document boundaries in packed sequences. The trainer must enforce both.

### 3.2 Composite-PIT Embedding

The PIT Cholesky path (`cholesky_solve` in embedding, `L @ L^T` in head) must stay in FP32 regardless of mixed-precision policy. The existing tier4 trainer already handles this.

### 3.3 Native FP4 Training

The goal is a model that trains and runs natively in FP4 — no post-training quantization. The weights, activations, and gradients flow through FP4 GEMMs during training via Quartet's approach (Hadamard rotation + quantization fused into the matmul). The resulting model is a native 4-bit model at both training and inference time.

**Bring-up sequence:**
- Phase 1: BF16 to verify architecture correctness (ReFusion loss, MLA, PIT all working).
- Phase 1.5: Switch to FP4 GEMMs for Q/K/V/O projections and MLP up/gate/down. Validate loss tracks BF16 baseline.
- Phase 2: Port Quartet custom kernels for production throughput.

**FP32/BF16 safety paths** (kept out of FP4 regardless):
- Softmax, norms, RoPE, loss computation.
- PIT Cholesky factorization and solve.
- Optimizer states (Adam moments).

Initial FP4 backend: Transformer Engine. Performance backend: ported Quartet kernels.

### 3.4 No Weight Decay on Embeddings

SmolLM3 explicitly removes weight decay from embedding layers. We adopt this. For YAMIT, "embedding layers" includes:
- `byte_memory` (PIT shared byte table)
- `token_embed` (PIT token-private embedding)
- `byte_chol_raw` (PIT Cholesky factor)
- `mask_embed` (diffusion mask embedding)

These all get weight_decay=0.0 in the optimizer param groups.

---

## 4. MLA Kernel Acceleration: FlashMLA

### 4.1 What FlashMLA Provides

[FlashMLA](https://github.com/deepseek-ai/FlashMLA) is DeepSeek's optimized MLA kernel library. MIT license. Supports:

- **Dense prefill** (SM100/Blackwell): MHA mode, up to 1460 TFlops on B200.
- **Dense decode** (SM90/Hopper+): MQA absorbed mode, up to 660 TFlops on H800, 3000 GB/s memory-bound.
- **Sparse prefill/decode** (SM90+SM100): for DSA (deferred to YAMIT v2).
- FP8 KV cache for sparse decode.
- Paged KV cache with block tables.

### 4.2 Dimension Compatibility

FlashMLA is built for DeepSeek V3's head dimensions:

| Mode | FlashMLA expects | YAMIT Model-S | YAMIT Model-P |
|---|---|---|---|
| MHA prefill (head_dim_k) | 192 or 128 | 96 | 96 |
| MHA prefill (head_dim_v) | 128 | 64 | 64 |
| MQA decode (head_dim_k) | 576 | 128 | 160 |
| MQA decode (head_dim_v) | 512 | 96 | 128 |

YAMIT's dimensions are smaller than what FlashMLA's kernels are tuned for. Options:

1. **Fork and adapt tile sizes** for our smaller dimensions. The CUTLASS-based kernels should generalize, but autotuning constants will differ.
2. **Use FlashAttention 2/3** for prefill (standard flash attention works after expanding K/V from the latent). Use FlashMLA only for the absorbed decode path, if dimensions can be padded.
3. **Pad dimensions** to match FlashMLA's expected sizes. Wasteful but avoids kernel changes.

Recommendation: start with FlashAttention 2/3 for prefill during BF16 bring-up (Phase 1). Investigate FlashMLA integration as a Phase 2 performance optimization alongside the Quartet kernel port. The absorbed decode path is where FlashMLA's benefit is largest (memory-bound decode with compressed KV cache).

### 4.3 Training vs. Inference

FlashMLA provides **forward-only** kernels (no backward pass). For training, we need:
- **Prefill/forward**: FlashAttention 2/3 (has backward) or PyTorch SDPA with MLA wrappers.
- **Decode/sampling**: FlashMLA's dense decode kernel (inference only, used by diffusion sampler).

During training, the attention forward+backward runs through standard flash attention after materializing K/V from the latent. FlashMLA's training benefit is limited to the diffusion sampler's decode passes during eval/generation.

### 4.4 Native FP4 Interaction

Since YAMIT trains natively in FP4, the model's weights and KV cache are already low-precision — there's no separate quantization step for deployment. FlashMLA's FP8 KV cache mode is designed for models that store a higher-precision cache and quantize it for decode; that's not our situation. Our KV latent cache is whatever precision the compressed latent (`kv_compress_dim` + rope dims) is stored at during training. This means:

- FlashMLA's BF16 dense decode path is the relevant one (not the FP8 sparse path).
- The Quartet FP4 kernels handle the GEMM quantization; FlashMLA handles the attention score computation on the (BF16) latent cache.
- No post-training quantization needed — the shipped model IS the trained model.

### 4.5 DSA Sparse Kernels (v2)

FlashMLA includes the sparse attention kernels for DSA (token-level sparse attention with top-k selection). These are exactly what YAMIT v2's DSA indexer would use. Good to know they exist; not needed for v1.

---

## 5. Trainer Implementation Plan

### 5.1 Base

Start from `experiments/tier4/train_fineweb_vanilla.py`. It already has:
- AdamW with param group separation (decay vs. no-decay)
- Cosine LR schedule (switch to WSD)
- `torch.compile` + DDP
- Composite embedding + PIT head
- BF16 autocast
- Basic train loop with validation

### 5.2 What Needs to Be Added

| Component | Description |
|---|---|
| **WSD scheduler** | Replace cosine with warmup-stable-decay. Linear warmup → hold peak → linear decay to 0 in final 10%. |
| **ReFusion forward process** | Slot partitioning, masking, shuffling, position ID remapping. Per-sample `p_mask` and `slot_size` sampling. |
| **Dual loss** | `L_AR + L_MDM` with `1/p_mask` importance weighting and `1/answer_length` normalization. |
| **MLA attention** | Dense MLA with compressed KV latent, separate rope cache, arbitrary `position_ids`. |
| **uint32 token loading** | Extend shard reader to handle uint32 tokens (vocab > 65535). |
| **Modified Qwen3 tokenizer** | Long tokens (> 16 bytes) removed from vocab and BPE merges by tokenizer generator. |
| **Diffusion sampler** | Iterative block/slot decode with verification, for eval-time generation. |
| **DiffusionMLACache** | Per-layer latent KV cache with crop/select/append/batch ops. |
| **Checkpoint save/load** | `model.state_dict()`, `optimizer.state_dict()`, step, RNG states. The existing trainer has none. |
| **Intra-document masking** | Attention mask that prevents cross-document attention in packed sequences. Intersects with ReFusion slot boundaries. |

### 5.3 What Comes Later

| Component | Phase |
|---|---|
| TE FP4 backend | 1.5 |
| Quartet kernel port | 2 |
| FlashMLA decode integration | 2 |
| Multi-stage data mixing | 3 (Model-P) |
| Long-context extension (32k/64k) | 3 (Model-P) |

---

## 6. Open Questions

- **Token budget for Model-S**: 3B tokens is Quartet's budget for a 30M model. Our 130M model probably needs more. Need to decide: fixed budget, or train until loss plateaus?
- **WSD vs. cosine**: SmolLM3 uses WSD. Quartet uses cosine with OneCycleLR. Both work. SmolLM3's WSD is the baseline; switch to cosine if WSD causes issues.
- **Data sources**: SmolLM3 uses FineWeb-Edu, DCLM, The Stack v2, FineMath, etc. We need access to these or equivalent corpora. Tokenization must use the generated modified Qwen3 tokenizer.
- **FlashMLA dimension adaptation**: Can we pad our smaller head dimensions to match FlashMLA's expected sizes without excessive waste? Or do we fork?
- **Intra-document masking + ReFusion interaction**: SmolLM3's intra-doc masking is straightforward attention masking. ReFusion's slot shuffling changes token order. Need to verify these compose correctly.
