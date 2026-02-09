# Task-Scaling Benchmark

A comprehensive synthetic benchmark for evaluating sequence model architectures across a wide range of computational primitives. Measures both single-task capability and multi-task scaling behavior — how gracefully a model degrades as the number of simultaneously-required skills grows.

## Design Principles

Every task in this benchmark must:

1. **Have its own synthetic dataset generator** — no external datasets, fully reproducible
2. **Not require pretraining** — models train from scratch on task data
3. **Not require language understanding** — purely symbolic, spatial, or game-based
4. **Be trainable** — learn the target operation from examples alone

The benchmark tests raw computational primitives, not language skills.

---

## Architecture: Three-Layer Sandwich

The benchmark evaluates **sequence layers** (the middle) while holding the input and output layers fixed.

### Input Layer: Lightweight Byte Encoder (~10-20K params)

Inspired by BLT (Byte Latent Transformer, arXiv:2412.09871) but drastically simplified. BLT uses a 100M-param entropy model, hash n-gram tables, and cross-attention pooling — all overkill for our purposes. We take the core insight (byte-level input with learned internal representations) and implement it minimally:

- **Byte embedding table**: 256 entries × dim (e.g., 256 × 64 = 16K params)
- **Fixed-stride patching**: pool every K bytes into one patch embedding via small learned linear or conv
- **2D sinusoidal positional embeddings**: added before patching for spatial/game tasks (chess, images, mazes, pathfinder, game of life)

This gives a universal input pipeline. Chess in PGN/FEN notation, math expressions, pixel values, board states, maze grids — all just byte sequences. One encoder for everything.

### Middle Layer: Causal Sequence Layers Under Test

This is what we're benchmarking. Always causal. The architectures under test include:

- MHA (multi-head attention, causal)
- Mamba (selective SSM)
- RWKV (time-mixing RNN)
- RetNet (retentive network)
- Hyena (long-conv)
- ULB (Universal Learning Block) + ablation variants

ULB's Q-peeking provides bounded acausal information flow (leakage radius R = n_layers) through a causal architecture. This structural advantage is especially relevant in the diffusion output tiers (see below), where the model must predict masked tokens with incomplete context — Q-peeking gives ULB forward information that strictly causal models don't have.

### Output Layer: Three Output Tiers

**Output Tier 1: CE Autoregressive**
- Standard cross-entropy, single-token prediction per position
- Lightweight, fast training
- Best for symbolic tasks with single-token-per-position targets

**Output Tier 2: Causal Masked Diffusion (LLaDA-style)**
- For tasks with complex or long structured outputs (chess moves, sorted sequences, maze paths)
- LLaDA (arXiv:2502.09992) masked diffusion, but **always causal** — no bidirectional attention
- Training: randomly mask a fraction of tokens, predict masked tokens from causal context, weight loss by 1/p_mask
- Generation: iteratively unmask by confidence, still with causal attention
- ULB doesn't need bidirectional attention to get acausal capability — Q-peeking provides it through the causal architecture

**Output Tier 3: Full Causal Masked Diffusion**
- Same as Tier 2 but applied to ALL tasks, including the ones that work fine with CE
- Tells us whether Tier 1 results transfer — do architectures that look great with CE fall apart under diffusion, or vice versa?
- Tests whether ULB's bounded acausality advantage grows when context is sparse and partial (masked positions surrounding the prediction target)

### Why Always Causal

LLaDA uses bidirectional attention because they're building a general-purpose LLM. We don't need that. The whole point of ULB is that it achieves acausal induction without acausal attention. Masked diffusion works fine with causal models — the forward process (random masking) and loss (CE on masked positions) don't depend on attention direction. The model just has to predict masked tokens from whatever context it can see to the left (plus Q-peeking's bounded forward leakage for ULB).

This keeps the comparison fair: the sequence layers are always causal across all tiers. Only the training objective and output mechanism change.

### LLaDA Integration Details

Training (Tiers 2 & 3):
```
# Forward process: mask random fraction of tokens
t = torch.rand(B)                           # mask ratio per sample
p_mask = (1 - eps) * t + eps                # avoid 0% masking
masked = torch.rand(B, L) < p_mask[:, None]
noisy_input = torch.where(masked, MASK_TOKEN, input)

# Model forward (causal sequence layers)
logits = model(noisy_input)

# Loss: CE on masked positions, importance-weighted
token_loss = CE(logits[masked], input[masked]) / p_mask[masked]
loss = token_loss.sum() / (B * L)
```

Generation:
- Start with prompt + all-mask output region
- Iteratively: predict all positions, unmask highest-confidence tokens
- Repeat for N steps (N ≈ output length for best quality)
- Remasking strategy: low-confidence positions get re-masked

---

## Task Taxonomy

### Family A: Memory / Recall

Storing and retrieving specific tokens from past context.

| Task | What it tests | Key params |
|------|---------------|------------|
| **delay** | 1-step shift register | vocab=32 |
| **selective_copy** | Content-gated recall of marked tokens | n_markers=4, vocab=32+32 |
| **MQAR** | Multi-query associative recall, power-law gaps | vocab=64, kv_pairs=4..32 |
| **forgetting_MQAR** | Association overwriting — return last-seen value | vocab=64, kv_pairs=4..16, updates=kv//2 |

### Family B: Pattern / Structure

Detecting and exploiting sequential patterns.

| Task | What it tests | Key params |
|------|---------------|------------|
| **induction** | Variable-length pattern completion (2 copies) | vocab=32, plen∈[L/3, 2L/3] |
| **compositional_MQAR** | Compound-key (K1,K2) retrieval | vocab=64, kv_pairs=4,9,16 (perfect squares) |

### Family C: Cumulative State

Maintaining and updating a running state.

| Task | What it tests | Key params |
|------|---------------|------------|
| **parity** | Cumulative XOR (binary state) | binary input |
| **mod_arith** | Cumulative sum mod base | mod_base=5 |

### Family D: Multi-Hop / Graph

Resolving transitive dependencies and graph structure.

| Task | What it tests | Key params |
|------|---------------|------------|
| **chain_trace** | Multi-hop pointer chasing, randomly shuffled bindings | num_chains=3, num_hops=3, vocab=32 |
| **reachability** | Graph connectivity (yes/no) from edge list | num_nodes, num_edges, vocab=32 |
| **shortest_path** | Hop count between two nodes | num_nodes, num_edges |
| **maze** | 2D grid connectivity with walls, flattened to 1D | grid_size, wall_density |

chain_trace: sequence body contains `[key, value]` binding pairs for N chains of M hops each (e.g., a→b, b→c, c→d), randomly shuffled among noise. Query presents chain start, target is chain terminal. Bindings in arbitrary order — model must resolve transitive dependencies regardless of presentation order.

maze: 2D grid encoded as token sequence with learned 2D sinusoidal positional embeddings. Model must learn spatial adjacency from the embeddings.

### Family E: Aggregation / Counting

Global statistics over the full sequence.

| Task | What it tests | Key params |
|------|---------------|------------|
| **freq_count** | Classify tokens as frequent/infrequent by threshold | n_targets=3, freq_thresh=5 |
| **majority** | Identify the single most common token | vocab_subset, sequence length |

### Family F: Arithmetic

Learning mathematical operations from examples.

| Task | What it tests | Key params |
|------|---------------|------------|
| **addition_carry** | Multi-digit addition with carry propagation | num_digits, base |
| **multiplication** | Multi-digit multiplication (partial products + accumulation) | num_digits, base |
| **modular_exp** | Repeated squaring — deep sequential dependency | base, exponent range, modulus |
| **GCD** | Euclidean algorithm — learning iterative/loop computation | value range |

addition_carry: carry propagation is a fundamentally sequential dependency. Each output digit depends on all less-significant digits via carries. The carry chain is conditional (depends on whether intermediate sums exceed the base), not just additive — qualitatively different from cumulative state tasks like parity/mod_arith.

### Family G: Logic / Compositional

Evaluating structured logical expressions.

| Task | What it tests | Key params |
|------|---------------|------------|
| **listops** | Nested list operations (MIN, MAX, MED, SUM_MOD) — LRA-style | nesting_depth, num_operands |
| **circuit_eval** | DAG-structured logic circuit evaluation | num_gates, gate_types, depth |
| **multi_bit_logic** | Parallel boolean ops across two input streams | op_type (AND/XOR/majority), width |

listops: from the Long Range Arena (LRA) benchmark. Expressions like `[MAX 2 9 [MIN 4 7] 0]`. Tests tree-structured compositional evaluation.

circuit_eval: generalizes boolean_formula_eval from trees to DAGs (fan-in > 1, shared subexpressions). Tests whether the model can handle non-tree dependencies.

### Family H: Routing / Permutation

Rearranging and reordering tokens.

| Task | What it tests | Key params |
|------|---------------|------------|
| **permutation_compose** | Compose two permutations — arbitrary remapping | permutation_size |
| **sorting** | Comparison-based reordering | sequence_length, vocab_range |
| **copy_reorder** | Rearrange tokens by rule (reverse subsequence, rotate, etc.) | rule_type, subsequence_length |

### Family I: Conditional Computation

Operation depends on context.

| Task | What it tests | Key params |
|------|---------------|------------|
| **conditional_op** | Context token selects which operation to apply to the rest of the sequence | num_ops, op_types |

Example: "if you see token A, the rule is XOR; if you see token B, the rule is AND." Tests dynamic computation selection — what MoE architectures are designed for.

### Family J: Spatial / Perceptual

2D structured data flattened to sequences. All use learned 2D sinusoidal positional embeddings.

| Task | What it tests | Key params |
|------|---------------|------------|
| **image_classify** | Sequential pixel classification (MNIST/CIFAR-style) | image_size, num_classes |
| **pathfinder** | LRA-style: are two points connected by a visual path? | grid_size, path_length, num_distractors |
| **game_of_life** | Predict N steps forward from a board state | grid_size, num_steps |

image_classify: pixels as token sequence, predict class. Model trains from scratch — no pretraining needed. Tests whether architecture can recover spatial structure from flat sequence + 2D positional info.

### Family K: Strategy / Games

Board games encoded as byte sequences. Board state uses 2D sinusoidal positional embeddings.

| Task | What it tests | Key params |
|------|---------------|------------|
| **chess** | Predict best/next move from board position | FEN/PGN byte encoding, training data from game databases |
| **othello** | Predict legal/best move from board state | 8×8 grid, ~65 token vocab |
| **go** | Predict next move from board position | 9×9 or 19×19 grid |

chess: board state encoded in FEN notation as bytes. 64 squares × ~13 piece states. Output is move in UCI notation (e.g., `e2e4`). Train on existing game databases (lichess, CCRL). Even at small model sizes this tests strategic reasoning and lookahead.

### Summary: ~30 tasks, 11 families

```
Family A  (Recall):          delay, selective_copy, MQAR, forgetting_MQAR
Family B  (Pattern):         induction, compositional_MQAR
Family C  (State):           parity, mod_arith
Family D  (Multi-Hop):       chain_trace, reachability, shortest_path, maze
Family E  (Aggregation):     freq_count, majority
Family F  (Arithmetic):      addition_carry, multiplication, modular_exp, GCD
Family G  (Logic):           listops, circuit_eval, multi_bit_logic
Family H  (Routing):         permutation_compose, sorting, copy_reorder
Family I  (Conditional):     conditional_op
Family J  (Spatial):         image_classify, pathfinder, game_of_life
Family K  (Games):           chess, othello, go
```

---

## MQAR Integration

Zoology uses vocab=8192 and up to 256 kv_pairs. We adapt to our small-model regime:

- **Vocab = 64** (same as existing bench_ssm tasks)
  - Keys: tokens `[1, 32)` — 31 unique keys
  - Values: tokens `[32, 64)` — 32 unique values
  - Token 0: filler/padding
- **kv_pairs**: 4, 8, 16, 32 (limited by vocab//2 = 31 max unique keys)
- **Power-law gaps**: `power_a=0.01` (Zoology default) — heavy short-gap bias
- **`random_non_queries=True`** — fill padding with random tokens
- **Sequence length**: default L=128
- **Scoring**: `ignore_index=-100` on non-answer positions

### Forgetting MQAR
- Same vocab split
- `num_updates = kv_pairs // 2`
- Target is always the last-seen value

### Compositional MQAR
- Vocab split into thirds: K1s `[1, 21)`, K2s `[21, 42)`, Values `[42, 64)`
- kv_pairs must be perfect square (4, 9, 16)
- Grid structure prevents single-key shortcuts

---

## Training Protocol

### Shared across all task-scaling tiers

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

### Model Configurations

Primary regime: small models on hard tasks (dim=64, 1 layer, param-matched to ULBBlendP at default `inner_ratio`). The point is to see which architectures make the best use of limited parameters.

| Model | Config |
|-------|--------|
| MHA | d_model=64, n_heads=4, 1 layer |
| Mamba | d_model=64, d_state=auto, expand=2, 1 layer |
| RWKV | d_model=64, 1 layer |
| RetNet | d_model=64, n_heads=4, 1 layer |
| Hyena | d_model=64, 1 layer |
| ULBBlendP | d_model=64, n_heads=4, paired=True, attn_mode=blend, q_mix=lerp |
| + ablations | ULBBlendPQNone, ULBBlendPConv2, ULBBlendPNoK, ULBSoftmaxP, ULBSilu2P, ULBBlendPSiLU, ULBBlend |

Multi-layer scaling: repeat at 2 and 4 layers.

---

## Task-Scaling Evaluation Tiers

These tiers measure multi-task scaling behavior (how many tasks at once). Orthogonal to the output tiers above.

### Tier 1: Single Task
Each task trained independently. Measures raw capability ceiling per skill.

### Tier 2: Multi-Task Family
Train on all tasks within one family simultaneously (intra-batch random mixing). Measures interference within related skills.

### Tier 3: Multi-Family Set
Train on tasks from multiple families simultaneously. Measures broad competence and cross-family interference/transfer.

---

## Metrics

### Per-configuration outputs

For each (model, output_tier, task_scaling_tier, task_set, seed):
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
5. **Output Tier 1 vs 3**: do CE results predict diffusion results? Does ULB's advantage grow under diffusion?

---

## Output Format

### Per-run CSV
```
model,output_tier,task_tier,task_set,seed,task,accuracy,loss_init,loss_best,epoch_converged
ULBBlendP,ce,single,delay,42,delay,0.998,3.46,0.008,12
ULBBlendP,ce,single,selective_copy,42,selective_copy,0.952,3.46,0.15,50
ULBBlendP,diffusion,single,delay,42,delay,0.991,3.46,0.012,18
...
```

### Visualization (planned)
- Heatmap: models × tasks, colored by accuracy, grouped by tier
- Spider/radar plot: per-model capability profile across all tasks
- Task-scaling curve: min_acc vs num_tasks for each model
- Interference matrix: `single_acc - mixed_acc` per model × task
- Output tier comparison: CE accuracy vs diffusion accuracy per model × task

---

## Implementation Plan

### Phase 1: Task generators
Existing:
- [x] delay, selective_copy, induction, parity, mod_arith (in bench_ssm.py)

New — Family A/B additions:
- [ ] gen_mqar
- [ ] gen_forgetting_mqar
- [ ] gen_compositional_mqar

New — Families D-K:
- [ ] gen_chain_trace (multi-hop pointer chasing, shuffled bindings)
- [ ] gen_reachability (graph connectivity)
- [ ] gen_shortest_path (hop count)
- [ ] gen_maze (2D grid → flat sequence)
- [ ] gen_freq_count (frequency threshold classification)
- [ ] gen_majority (most common token)
- [ ] gen_addition_carry (multi-digit addition)
- [ ] gen_multiplication (multi-digit multiplication)
- [ ] gen_modular_exp (repeated squaring)
- [ ] gen_gcd (Euclidean algorithm)
- [ ] gen_listops (nested list operations, LRA-style)
- [ ] gen_circuit_eval (DAG logic circuit)
- [ ] gen_multi_bit_logic (parallel boolean ops)
- [ ] gen_permutation_compose (permutation composition)
- [ ] gen_sorting (comparison-based reorder)
- [ ] gen_copy_reorder (rule-based rearrangement)
- [ ] gen_conditional_op (context-dependent operation)
- [ ] gen_image_classify (sequential pixel classification)
- [ ] gen_pathfinder (LRA-style path connectivity)
- [ ] gen_game_of_life (cellular automaton prediction)
- [ ] gen_chess (FEN → UCI move)
- [ ] gen_othello (board → move)
- [ ] gen_go (board → move)

### Phase 2: Byte encoder
- [ ] Byte embedding table (256 × dim)
- [ ] Fixed-stride patching (small conv/linear)
- [ ] Learned 2D sinusoidal positional embeddings for spatial/game tasks

### Phase 3: LLaDA diffusion integration
- [ ] Forward process (random masking)
- [ ] Masked CE loss with 1/p_mask weighting
- [ ] Iterative unmasking generation (confidence-based)
- [ ] All causal — no bidirectional attention changes needed

### Phase 4: Flexible mixed-task training
- [ ] Generalize gen_mixed to accept arbitrary task subsets
- [ ] Support family groupings and multi-family sets

### Phase 5: Runner script
- [ ] `scripts/run_task_scaling_benchmark.py`
- [ ] Per-run CSV output
- [ ] Aggregate JSON summarizer
- [ ] Dry-run mode

### Phase 6: Analysis
- [ ] Aggregate results across seeds
- [ ] Compute interference deltas
- [ ] Generate comparison tables and plots

---

## Open Questions

1. **MQAR difficulty tuning**: With vocab=64 and kv_pairs=32, there are only 31 possible keys. Cap at kv_pairs=16?
2. **Sequence length scaling**: Sweep L=64, 128, 256 or fix L=128?
3. **kv_pairs as difficulty dial**: Separate task per kv_pairs setting, or variable kv_pairs within one task?
4. **Compositional MQAR vocab pressure**: 3-way split (21/21/22 tokens) gives at most 4×4=16 compound keys. Enough?
5. **chain_trace difficulty**: Default 3 chains × 3 hops uses 12/32 vocab tokens. Max feasible: 32/(hops+1) chains.
6. **freq_count threshold**: Should freq_thresh scale with L?
7. **Chess data source**: Lichess game database? Stockfish-evaluated positions? Raw games ("what was played") vs engine labels ("best move")?
8. **Chess encoding**: FEN as bytes is natural, but board-as-64-tokens with 2D sinusoidal embeddings might train better. Test both?
9. **Go board size**: 9×9 (simpler, feasible at small scale) vs 19×19 (standard but huge)?
10. **Game of Life steps**: How many forward steps should the model predict? 1 step is local, N steps requires long-range planning.
11. **Maze encoding**: Row-major flattening? Hilbert curve? Token per cell with 2D sinusoidal embeddings?
12. **Scale regimes**: Primary is dim=64, 1 layer. Should we also define medium (dim=128, 2-4 layers) and large (dim=256, 4-8 layers)?
13. **Byte encoder vs token embedding**: For purely symbolic tasks (families A-I), should we go through the byte encoder or keep the current direct token embedding? Byte encoder adds uniformity, token embedding is simpler and already works.
