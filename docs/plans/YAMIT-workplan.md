# YAMIT Implementation Workplan

**Codename:** YAMIT (Yet Another Minorly Improved Transformer)  
**Status:** Active  
**Date:** 2026-02-22

---

## 1. Mission

Ship YAMIT in three progressive tracks:

1. **Correctness first** on a 100-150M model.
2. **Performance second** via custom Quartet FP4 kernels.
3. **Production training third** at 350M+.

No DSA indexer in v1.

---

## 2. Phase Summary

| Phase | Objective | Exit Gate |
|---|---|---|
| 0 | Spec freeze + environment setup | Design and runbook approved |
| 1 | Model-S correctness (BF16 first) | Stable AR+MDM + sampler correctness |
| 1.5 | Model-S FP4 bring-up | Stable FP4 and acceptable gap vs BF16 |
| 2 | Custom kernel performance pass | Throughput targets met without quality regression |
| 3 | Model-P production training | Canary and full-run readiness |
| 4 (later) | DSA indexer integration | Deferred by design |

---

## 3. Phase 0: Spec Freeze and Setup

### Deliverables

- YAMIT architecture spec approved (`docs/YAMIT-spec.md`).
- Model-S and Model-P dimensions locked.
- Tokenizer surgery policy locked (Qwen3 + long-token remap).
- FP4 policy locked (Quartet-style low precision + FP32 safety paths).

### Environment Setup

- Confirm Blackwell driver/CUDA stack on training box.
- Confirm Transformer Engine and Quartet kernel dependencies.
- Define runtime switch for FP4 backend (`te` vs `quartet`).

### Exit Gate 0

- All implementation decisions required for Phase 1 are explicit and versioned.
- Vocab size locked at 151,808 (padded to 256-multiple — see spec Section 4).
- Training plan documented (`docs/plans/YAMIT-training.md`).

---

## 4. Phase 1: Model-S Correctness Bring-Up (BF16)

### 4.1 Implementation Scope

- Generalize composite byte table build to Qwen3 tokenizer.
- Add deterministic retokenization map pipeline for long tokens.
- Add uint32 shard read/write path.
- Implement dense MLA with arbitrary `position_ids` support.
- Implement ReFusion forward process (slot split, masking, AR shuffle, labels).
- Implement dual loss (`L_AR + L_MDM`) for pretraining from scratch.
- Implement iterative diffusion sampler and MLA cache operations.

### 4.2 Required Tests

- Unit test: byte table generation and token remap invariants.
- Unit test: forward process shape and label-mask correctness.
- Unit test: `position_ids` correctness under slot shuffling.
- Unit test: diffusion cache crop/select/append semantics.
- Unit test: PIT head logit computation matches manual calculation.
- Unit test: weight initialization statistics (mean ~0, std ~0.02 for linears; norm weights = 1.0).
- Unit test: mask schedule `p_mask` samples are in `[1e-3, 1.0]` range.
- Unit test: MDM loss importance weight `1/p_mask` is applied correctly.
- Integration test: one-batch overfit.

### 4.3 Exit Gate 1

- No NaN/Inf for first 5k steps.
- `L_AR` and `L_MDM` both decrease monotonically over 500-step windows.
- Validation loss within 5% of training loss.
- Sampler emits valid outputs and terminates within `slot_size` iterations per block.
- Checkpoint save/load round-trip verified (save at step N, resume, verify step N+1 matches continuous run).

---

## 5. Phase 1.5: Model-S FP4 Bring-Up

### 5.1 Implementation Scope

- Add TE-based FP4 path as initial low-risk backend.
- Apply precision policy from spec:
  - FP4 for major GEMMs.
  - BF16/FP32 for softmax, norms, RoPE, loss.
  - FP32 for PIT Cholesky/solve and optimizer states.

### 5.2 Telemetry

- Track overflow/underflow indicators per FP4 module.
- Track gradient norm (pre-clip and post-clip) and clipping rate.
- Track finite checks for PIT factorization path (Cholesky diagonal floor hit count).
- Track per-layer activation magnitude histograms (first 100 steps, then every 500 steps).

### 5.3 Exit Gate 1.5

- Stable for at least 20k steps in FP4 mode.
- Loss gap versus BF16 baseline is within agreed threshold (default <= 5% at matched tokens).

---

## 6. Phase 2: Quartet Kernel Port and Performance

### 6.1 Implementation Scope

- Port custom kernels from `Quartet/` into YAMIT runtime.
- Wire runtime switch:
  - `--fp4_impl te`
  - `--fp4_impl quartet`
- Implement benchmarking harness with fixed batch/sequence model slices.
- Add correctness parity checks between `te` and `quartet` paths.

### 6.2 Kernel Priorities

1. Attention projections (`wq`, `wkv`, `wo`).
2. MLP projections (`gate`, `up`, `down`).
3. Quantize/dequantize + transform overhead reductions.

Kernel config: block size 32, Hadamard dim 32. Forward uses QuEST MXFP4 (RMSE clipping + deterministic RTN). Backward uses RTN MXFP4 (absmax scaling + stochastic rounding + sign-flip re-randomization). See spec Section 11.3 for full details.

### 6.3 Exit Gate 2

- Throughput target over BF16 met (initial target >= 1.4x end-to-end).
- Quartet path improves over TE FP4 baseline.
- No statistically significant quality regression versus TE FP4 at matched tokens.

---

## 7. Phase 3: Model-P Production Training (350M+)

### 7.1 Rollout Sequence

1. **Canary run** (short horizon) with full metrics.
2. **Medium validation run** to verify scaling trends.
3. **Full production run** after canary acceptance.

### 7.2 Production Readiness Checks

- Stable training (distributed strategy decided at Model-P time, not before).
- Checkpoint save/resume integrity verified.
- Evaluation and sampler throughput acceptable for run length.

### 7.3 Exit Gate 3

- Canary and medium-run criteria pass.
- No unresolved blocker in failure checklist.

---

## 8. Phase 4 (Deferred): DSA Indexer Integration

Deferred until v1 completion. Planned sequence:

1. Dense warmup for indexer-only alignment.
2. Sparse top-k stage with curriculum.
3. Detached indexer optimization and KL alignment loss.

---

## 9. Failure and Rollback Playbook

### Hard Stop Conditions

- **NaN/Inf:** 3 or more NaN/Inf events within any 100-step window after applying all mitigations (grad clip, FP32 safety paths, loss scaling). A single NaN in first 100 steps triggers investigation but is not a hard stop.
- **Tokenizer/remap inconsistency:** any token round-trip failure (encode -> remap -> decode != original) on the validation corpus.
- **PIT factorization instability:** Cholesky diagonal floor triggered on >10% of batches for 500 consecutive steps, or any non-finite output from PIT path after FP32 safeguards.
- **Sampler non-termination:** any single block fails to converge within `slot_size` refinement iterations AND the force-accept fallback fails to produce finite logits.
- **Loss divergence:** `train_loss_total` increases by >50% from its 1000-step rolling minimum and does not recover within 500 steps.
- **FP4 instability:** gradient norm exceeds 10x its 1000-step rolling mean for 50 consecutive steps under FP4.

### Rollback Rules

- Always keep a BF16-correct reference path.
- If FP4 regresses stability, revert to TE backend before disabling FP4 entirely.
- If Quartet kernel path regresses quality, run TE FP4 path while fixing kernels.

---

## 10. Metrics, Logging, and Dashboards

### Training Metrics

- `train_loss_total`
- `train_loss_ar`
- `train_loss_mdm`
- gradient norm and clipping rate
- finite-check counters (PIT and global)

### Sampler Metrics

- slot acceptance rate
- verification pass rate
- iterations per generated token
- EOS hit rate

### System Metrics

- tokens/sec
- step latency
- HBM utilization and peak memory
- kernel-level timing breakdown (when available)

---

## 11. Immediate Task Backlog

### Task Group A: Data and Tokenizer

- [ ] Implement long-token remap artifact generator for Qwen3.
- [ ] Implement uint32 shard writer and reader.
- [ ] Add validation script for tokenizer/remap consistency.

### Task Group B: Model Core

- [ ] Add Model-S config to trainer.
- [ ] Add dense MLA with arbitrary `position_ids`.
- [ ] Add ReFusion forward process and dual loss.
- [ ] Add diffusion sampler and MLA cache implementation.

### Task Group C: Precision

- [ ] Add TE FP4 backend toggle.
- [ ] Add FP32 safeguards for PIT path.
- [ ] Add FP4 finite-check telemetry.

### Task Group D: Performance

- [ ] Port Quartet kernels into YAMIT runtime module.
- [ ] Add `te` vs `quartet` parity tests.
- [ ] Add benchmark harness and reports.

---

## 12. Open Questions (Parking Lot)

- Context length for production (4k for v1; longer context requires RoPE scaling validation).
- Timing for enabling DSA indexer after production stability.
- Training config details (separate doc, follows SmolLM3 setup scaled to model size).

These are intentionally deferred so v1 execution does not block.
