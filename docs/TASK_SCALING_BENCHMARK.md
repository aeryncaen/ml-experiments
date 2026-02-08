# Task-Scaling Benchmark

A standalone benchmark for evaluating **task-scaling behavior** across sequence model architectures. The goal is to measure not just peak single-task accuracy, but how gracefully a model degrades as the number of simultaneously-required skills grows.

## Motivation

Single-task benchmarks hide a critical failure mode: models that ace one skill often catastrophically fail when asked to do several things at once. The ULB's design hypothesis is that blending causal and bounded-acausal paths yields better **multi-task floors** (worst-task accuracy) even at the cost of single-task ceilings.

This benchmark quantifies that tradeoff across three evaluation tiers.

---

## Evaluation Tiers

### Tier 1: Single Task
Train and evaluate on one task at a time. Measures raw capability ceiling per skill.

### Tier 2: Multi-Task Family
Train on all tasks within one family simultaneously (intra-batch random mixing, as in current bench_ssm mixed mode). Measures interference within related skills.

### Tier 3: Multi-Family Set
Train on tasks from multiple families simultaneously. Measures broad competence and cross-family interference/transfer.

---

## Task Taxonomy

### Family A: Memory / Recall

Tasks that require storing and retrieving specific tokens from the past.

| Task | Source | What it tests | Key params |
|------|--------|---------------|------------|
| **delay** | bench_ssm | 1-step shift register | vocab=32 |
| **selective_copy** | bench_ssm | Content-gated recall of marked tokens | n_markers=4, vocab=32+32 |
| **MQAR** | Zoology-style | Multi-query associative recall with power-law gaps | vocab=64, kv_pairs=4..32 |
| **forgetting_MQAR** | Zoology-style | Association overwriting — must return last-seen value | vocab=64, kv_pairs=4..16, updates=kv//2 |

### Family B: Pattern / Structure

Tasks that require detecting and exploiting sequential patterns.

| Task | Source | What it tests | Key params |
|------|--------|---------------|------------|
| **induction** | bench_ssm | Variable-length pattern completion (2 copies) | vocab=32, plen∈[L/3, 2L/3] |
| **compositional_MQAR** | Zoology-style | Compound-key (K1,K2) retrieval — requires attending to both | vocab=64, kv_pairs=4,9,16 (perfect squares) |

### Family C: State Tracking

Tasks that require maintaining and updating a running state.

| Task | Source | What it tests | Key params |
|------|--------|---------------|------------|
| **parity** | bench_ssm | Cumulative XOR (binary state) | binary input |
| **mod_arith** | bench_ssm | Cumulative sum mod base (multi-valued state) | mod_base=5 |

### Family D: RULER-Inspired (Multi-Hop & Aggregation)

Pure synthetic distillations of RULER task categories. No language, no pretraining — just the computational primitives.

| Task | Source | What it tests | Key params |
|------|--------|---------------|------------|
| **chain_trace** | bench_ssm (RULER variable_tracking) | Multi-hop pointer chasing through shuffled binding chains | num_chains=3, num_hops=3, vocab=32 |
| **freq_count** | bench_ssm (RULER common_words_extraction) | Global frequency counting — classify tokens as frequent/infrequent | n_targets=3, freq_thresh=5, vocab_subset=16 |

**chain_trace**: Sequence body contains `[key, value]` binding pairs for N chains of M hops each (e.g., a→b, b→c, c→d), randomly shuffled among noise tokens. Query section presents chain start tokens; target is the chain terminal. Bindings are in arbitrary order — the model must resolve transitive dependencies regardless of presentation order. This tests multi-hop composition, a capability axis not covered by any other task.

**freq_count**: Sequence body contains tokens from a vocab subset with controlled frequencies. Some tokens appear exactly `freq_thresh` times (frequent), others appear fewer times (infrequent). Query section presents both frequent and infrequent candidate tokens; target is binary (1=frequent, 0=infrequent). Tests global aggregation / counting over the full context.

### Summary: 10 tasks, 4 families

```
Family A (Recall):     delay, selective_copy, MQAR, forgetting_MQAR
Family B (Pattern):    induction, compositional_MQAR
Family C (State):      parity, mod_arith
Family D (RULER):      chain_trace, freq_count
```

---

## MQAR Integration

### Design Decision: Unified vocab, not Zoology-scale

Zoology uses vocab=8192 and up to 256 kv_pairs, which requires significantly larger models to even fit the embedding table. Our benchmark targets **under-capacity stress testing** (dim=64, ~27K params), so we adapt MQAR to our regime:

- **Vocab = 64** (same as existing bench_ssm tasks)
  - Keys: tokens `[1, 32)` — 31 unique keys
  - Values: tokens `[32, 64)` — 32 unique values
  - Token 0: filler/padding
- **kv_pairs**: 4, 8, 16, 32 (limited by vocab//2 = 31 max unique keys)
- **Power-law gaps**: `power_a=0.01` (Zoology default) — heavy short-gap bias
- **`random_non_queries=True`** — fill padding with random tokens, no zero-position cue
- **Sequence length**: Same as other tasks (default L=128)
- **Scoring**: `ignore_index=-100` on all non-answer positions (consistent with our convention)

### Forgetting MQAR
- Same vocab split as MQAR
- `num_updates = kv_pairs // 2` — half the keys get new values
- Target is always the **last-seen** value

### Compositional MQAR
- Vocab split into thirds: K1s `[1, 21)`, K2s `[21, 42)`, Values `[42, 64)`
- `kv_pairs` must be perfect square (4, 9, 16)
- Grid structure prevents single-key shortcuts

---

## Training Protocol

### Shared across all tiers

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimizer | Adam | |
| LR (single task) | 1e-4 | |
| LR (mixed) | 1e-3 | Higher LR for multi-task |
| LR schedule | 1-epoch warmup (0.1x → 1x), cosine anneal to 0.05x | |
| Max epochs | 50 | |
| Early stop | All subtask val acc > 99% | |
| Batch size | 64 | |
| Seq length | 128 | |
| Hard mining | Last 1/3 of epochs, mixed mode only | Rank-based weighting: 0.5x–2.0x |
| Seeds | 3 per configuration | Report mean ± std |

### LR Sweep (optional, for MQAR family)

Following Zoology, optionally sweep `np.logspace(-3, -1.5, 4)` = [1e-3, 3.16e-3, 1e-2, 3.16e-2] for new task types to find optimal LR per architecture. Report best-LR results.

### Model Configurations

Under-capacity regime (matched params ~27.5K):

| Model | Config |
|-------|--------|
| MHA | d_model=64, n_heads=4, 1 layer |
| Mamba | d_model=64, d_state=auto, expand=2, 1 layer |
| RWKV | d_model=64, 1 layer |
| RetNet | d_model=64, n_heads=4, 1 layer |
| Hyena | d_model=64, 1 layer |
| ULBBlendP | d_model=64, n_heads=4, paired=True, attn_mode=blend, q_mix=lerp |
| + ablations | ULBBlendPQNone, ULBBlendPConv2, ULBBlendPNoK, ULBSoftmaxP, ULBSilu2P, ULBBlendPSiLU, ULBBlend |

Multi-layer scaling: repeat at 2 and 4 layers (params scale ~linearly).

---

## Metrics

### Per-configuration outputs

For each (model, tier, task_set, seed):
- **Per-task accuracy** at convergence (or max epochs)
- **Per-task loss** (initial and best)
- **Epoch of convergence** per task

### Aggregate metrics

| Metric | Formula | What it captures |
|--------|---------|------------------|
| **Mean accuracy** | `mean(task_accs)` | Overall competence |
| **Min accuracy** | `min(task_accs)` | Worst-case / floor |
| **Geo-mean accuracy** | `exp(mean(log(task_accs)))` | Balanced competence (penalizes zeros) |
| **Interference delta** | `single_acc[t] - mixed_acc[t]` per task t | Per-task degradation from multi-tasking |
| **Mean interference** | `mean(interference_deltas)` | Average degradation |
| **Max interference** | `max(interference_deltas)` | Worst-case degradation |
| **Task-scaling slope** | Linear fit of min_acc vs log(num_tasks) | How fast floor drops with more tasks |

### Key comparisons

1. **Single vs Family**: interference within related skills
2. **Family vs Full Set**: cross-family interference
3. **Layer scaling**: does depth compensate for interference?
4. **Architecture ranking**: does the rank order change across tiers?

---

## Evaluation Sets

### Tier 1: Single Task (10 runs per model per seed)
Each of the 10 tasks trained independently.

### Tier 2: Multi-Task Family (4 runs per model per seed)
- **Family A** (4 tasks): delay + selective_copy + MQAR + forgetting_MQAR
- **Family B** (2 tasks): induction + compositional_MQAR
- **Family C** (2 tasks): parity + mod_arith
- **Family D** (2 tasks): chain_trace + freq_count

### Tier 3: Multi-Family Set (3 runs per model per seed)
- **Recall + Pattern** (6 tasks): Family A + Family B
- **Recall + RULER** (6 tasks): Family A + Family D
- **Full set** (10 tasks): Family A + Family B + Family C + Family D

### Total runs per model
- Tier 1: 10 tasks × 3 seeds = 30
- Tier 2: 4 families × 3 seeds = 12
- Tier 3: 3 sets × 3 seeds = 9
- **Total: 51 runs per model**

With 8+ models × 51 runs = 408+ training runs at 1 layer. At ~2 min/run on RTX 6000, that's ~14 hours. Multi-layer scaling (2, 4 layers) triples this to ~42 hours.

---

## Output Format

### Per-run CSV
```
model,tier,task_set,seed,task,accuracy,loss_init,loss_best,epoch_converged
ULBBlendP,tier1,delay,42,delay,0.998,3.46,0.008,12
ULBBlendP,tier1,selective_copy,42,selective_copy,0.952,3.46,0.15,50
...
ULBBlendP,tier2,family_a,42,delay,0.995,3.46,0.01,15
ULBBlendP,tier2,family_a,42,selective_copy,0.91,3.46,0.25,50
ULBBlendP,tier2,family_a,42,MQAR,0.87,3.46,0.35,50
ULBBlendP,tier2,family_a,42,forgetting_MQAR,0.82,3.46,0.42,50
...
```

### Aggregate JSON
```json
{
  "model": "ULBBlendP",
  "n_layers": 1,
  "params": 27500,
  "tier1": {
    "per_task": {"delay": {"mean": 0.998, "std": 0.001}, ...},
    "mean_acc": 0.95,
    "min_acc": 0.85,
    "geo_mean": 0.94
  },
  "tier2": {
    "family_a": {"per_task": {...}, "mean_acc": ..., "min_acc": ...},
    ...
  },
  "tier3": {
    "full_set": {"per_task": {...}, "mean_acc": ..., "min_acc": ..., "interference": {...}}
  }
}
```

### Visualization (planned)
- Heatmap: models × tasks, colored by accuracy, grouped by tier
- Spider/radar plot: per-model capability profile across all 8 tasks
- Task-scaling curve: min_acc vs num_tasks for each model
- Interference matrix: `single_acc - mixed_acc` per model × task

---

## Implementation Plan

### Phase 1: New task generators
- [x] `gen_chain_trace(B, L, num_chains, num_hops, device)` — multi-hop pointer chasing, shuffled bindings
- [x] `gen_freq_count(B, L, n_targets, freq_thresh, n_vocab_subset, device)` — global frequency counting
- [ ] `gen_mqar(B, L, kv_pairs, power_a=0.01, device)` — base MQAR with vocab=64 split
- [ ] `gen_forgetting_mqar(B, L, kv_pairs, num_updates, device)` — MQAR with value overwrites
- [ ] `gen_compositional_mqar(B, L, kv_pairs, device)` — compound-key MQAR

### Phase 2: Flexible mixed-task training
- [ ] Generalize `gen_mixed` to accept arbitrary task subsets (not just ALL_TASKS)
- [ ] Support tier-2 family groupings and tier-3 multi-family sets

### Phase 3: Runner script
- [ ] `scripts/run_task_scaling_benchmark.py` — drives all 51×N runs
- [ ] Per-run CSV output with full config columns
- [ ] Aggregate JSON summarizer
- [ ] Dry-run mode (print plan, don't execute)

### Phase 4: Analysis
- [ ] Aggregate results across seeds
- [ ] Compute interference deltas
- [ ] Generate comparison tables and plots

---

## Open Questions

1. **MQAR difficulty tuning**: With vocab=64 and kv_pairs=32, there are only 31 possible keys. Should we cap at kv_pairs=16 for the under-capacity regime?
2. **Sequence length scaling**: Should we also sweep L=64, 128, 256 to test length generalization, or fix L=128?
3. **kv_pairs as difficulty dial**: For MQAR tasks, should each kv_pairs setting be a separate "task" in the mix, or one MQAR task with variable kv_pairs per sample?
4. **Compositional MQAR vocab pressure**: With 3-way vocab split (21/21/22), we have at most 4×4=16 compound keys for kv_pairs=16. Is this enough range?
5. **chain_trace difficulty tuning**: Default num_chains=3, num_hops=3 uses 12 distinct nodes out of 32 vocab. With deeper chains or more chains, vocab pressure increases. Max feasible: 32/(num_hops+1) chains.
6. **freq_count threshold sensitivity**: freq_thresh=5 with n_targets=3 plants 15 frequent + 9 infrequent tokens in the body. Should threshold scale with L?
