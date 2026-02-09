# Pool of Experts (PoE)

## Motivation

MoE stacking (`MoEStackedULB`) arranges experts in a fixed grid: N experts wide, L layers tall. Each layer has its own router, and every sample traverses exactly L layers. Depth is static.

Pool of Experts removes the grid. All N*L experts are flattened into a single pool. A learned routing chain determines both *which* experts run and *how many* hops occur. Depth becomes dynamic and input-dependent — some samples may need 2 hops, others may need 10.

## Architecture

### Components

- **Stem layer** — a non-routed ULB block that transforms raw embeddings into features suitable for routing. Runs once at the start: `x = x + stem(norm(x))`.
- **Stem router** — `Linear(dim, pool_size²)`. Mean-pools the stem output and produces logits over the router space. Kicks off the first hop.
- **Expert pool** — `pool_size` ULB blocks, each an independent expert.
- **Expert routers** — each expert has its own outbound router, `Linear(dim, pool_size²)`. After an expert runs, its router produces logits over the same space, representing that expert's vote on where the signal should go next.
- **Exit layer** — a non-routed ULB block that runs after the routing chain terminates: `x = x + exit(norm(x))`.
- **Shared hop norm** — a single RMSNorm applied before each hop's expert computation.
- **Final norm** — RMSNorm on the output.

### Router Space and Exit Slots

The router space has `pool_size²` options:

- Indices `0..pool_size-1` are **expert slots** (pool_size total)
- Indices `pool_size..pool_size²-1` are **exit slots** (pool_size*(pool_size-1) total)

A random router pick has only a **1/pool_size** chance of selecting an expert. Exit is the overwhelming default — the model must actively learn that going deeper is worth it. Depth is a cost it chooses to pay, not a default it has to escape.

### Routing

**First hop:** The stem router produces logits from the mean-pooled stem output. Top-K logits are selected and softmax-weighted. If both top-K picks land on exit slots, the model skips directly from stem to exit layer — zero expert hops.

**Each subsequent hop:**

1. **Exit ramp.** Exit logits are boosted by `exit_ramp_scale * (hop / max_hops)`. Deeper hops face increasing pressure to exit, making the max_hops cap differentiable rather than a hard cliff. Default scale is 5.0.
2. **Top-K selection.** From the current logits, select the top K entries. Softmax over the K selected logits gives merge weights.
3. **Exit check.** Samples with any exit slot in their top-K receive no expert contribution this hop. If all samples have exited, the loop stops.
4. **Expert execution.** Experts selected by still-active samples run on the full batch. Each produces an output tensor and outbound routing logits.
5. **Weighted merge.** Expert outputs are merged into a single state using the softmax weights from step 2. The outbound logits from each expert are merged with the *same weights*.
6. **Residual add.** `x = x + merged_output`.
7. **Next logits.** The weighted-merged outbound logits become the input logits for the next hop's top-K selection.

This means routing is a chain: experts collectively decide where the signal goes next. An expert that produces strong output also has more influence on the next routing decision, because its outbound logits get higher weight in the merge.

### Exit Mechanism

Exit is not a separate decision — it competes with experts in the same logit space. A sample "exits" when any exit slot appears in its top-K. That sample receives no expert contribution that hop (`x += 0`) — its state is effectively frozen from that point, only receiving the final exit layer.

The loop terminates when every sample in the batch has exited. Samples that exit early aren't forced through further expert computation — they just coast with no update while the remaining active samples continue routing. This means:

- Experts can learn to vote for exit when their processing is sufficient.
- Samples exit independently. A sample that exits at hop 1 is unaffected by other samples that continue to hop 5.
- A hard `max_hops` cap prevents runaway chains for any sample. Default is `2 * pool_size`.
- The exit ramp makes approaching max_hops increasingly expensive, so the cap is soft in practice.

### Revisiting

Experts can be selected multiple times across hops. There is no exclusion mask. If the routing chain keeps selecting expert 3, expert 3 runs every hop. This is intentional — it allows the network to learn iterative refinement through a single expert.

## Training Features

### Router Noise

Gaussian noise added to router logits during training (`router_noise_scale`, default 1.0). Encourages exploration of different routing paths early in training. Annealed linearly to 0 over epochs via `model.router_noise_scale`.

### Router Dropout

Optional (default off, `router_dropout=0.0`). When enabled, drops a random number of exit logits per sample (0 to 25% of exit slots), never touching expert logits. The count dropped is itself random per sample — not a fixed percentage. This gives experts more exposure during training without removing the exit-heavy prior entirely.

### Hop Counting

`model.last_mean_hops` tracks the true mean hops per sample across the batch. Each hop, the number of samples that didn't exit is counted; the total is divided by batch size after the loop. This is the value displayed on the progress bar and used for logging.

## Relationship to MoE Grid

| Property | MoE Grid | Pool of Experts |
|---|---|---|
| Expert count | `n_experts * n_layers` | `n_experts * n_layers` (same total) |
| Depth | Fixed `n_layers` | Dynamic, 0 to `max_hops` |
| Routing | Per-layer router | Per-expert outbound router |
| Who decides next hop | Layer l+1's router | Current hop's experts (merged vote) |
| Expert reuse | Never (grid cells are distinct) | Allowed |
| Zero-depth path | No | Yes (stem router can skip to exit) |

When invoked via `bench_ssm.py --poe`, `pool_size` is computed as `n_experts * layers`, making the total expert count identical to the MoE grid for fair comparison.

## Usage

```python
from ulb import ULBBlock, ULBConfig, PoolOfExperts

cfg = ULBConfig(d_model=64, n_heads=4, paired=True, attn_mode='blend')
make = lambda: ULBBlock(cfg)

model = PoolOfExperts(make, pool_size=8, dim=64, top_k=2, max_hops=16)

# After forward:
y = model(x)
print(model.last_mean_hops)  # mean hops per sample this forward pass
print(model.aux_loss)         # block-level aux losses from experts
```

### Tracing

```python
model.trace = True
y = model(x)
for hop in model.last_trace:
    print(hop['topk_idx'], hop['topk_weights'], hop['has_exit'])
```

### Benchmark

```bash
python scripts/bench_ssm.py --poe --n-experts 4 --layers 2 --top-k 2 \
    --models ULBBlendP --tasks mixed --save-dir out/
```

This creates a pool of 8 experts (4 * 2), max_hops=16 (2 * 8), top-k=2 per hop. `--save-dir` saves trained checkpoints for later analysis with `scripts/analyze_poe.py`.

## Implementation Notes

- Experts run on the full batch (not masked subsets) to avoid Python-loop overhead and small-kernel launches. The wasted compute on non-selected samples is cheaper than the alternative.
- `last_mean_hops` is set after each forward for logging/analysis.
- `max_hops` is a plain attribute, not a buffer — change it freely during training.
- `aux_loss` aggregates any block-level auxiliary losses from the expert ULB blocks.
- `exit_ramp_scale` is a plain attribute — adjustable at runtime.
