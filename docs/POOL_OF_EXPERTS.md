# Pool of Experts (PoE)

## Motivation

MoE stacking (`MoEStackedULB`) arranges experts in a fixed grid: N experts wide, L layers tall. Each layer has its own router, and every sample traverses exactly L layers. Depth is static.

Pool of Experts removes the grid. All N*L experts are flattened into a single pool. A learned routing chain determines both *which* experts run and *how many* hops occur. Depth becomes dynamic and input-dependent — some samples may need 2 hops, others may need 10.

## Architecture

### Components

- **Stem layer** — a non-routed ULB block that transforms raw embeddings into features suitable for routing. Runs once at the start: `x = x + stem(norm(x))`.
- **Stem router** — `Linear(dim, pool_size + 1)`. Mean-pools the stem output and produces logits over the pool plus an exit token. Kicks off the first hop.
- **Expert pool** — `pool_size` ULB blocks, each an independent expert.
- **Expert routers** — each expert has its own outbound router, `Linear(dim, pool_size + 1)`. After an expert runs, its router produces logits over the pool + exit, representing that expert's vote on where the signal should go next.
- **Exit layer** — a non-routed ULB block that runs after the routing chain terminates: `x = x + exit(norm(x))`.
- **Shared hop norm** — a single RMSNorm applied before each hop's expert computation.
- **Final norm** — RMSNorm on the output.

### Routing

The pool has `pool_size + 1` routing targets: experts `{0, 1, ..., pool_size-1}` and a special **exit token** at index `pool_size`.

**First hop:** The stem router produces logits from the mean-pooled stem output. Top-K logits are selected and softmax-weighted.

**Each subsequent hop:**

1. **Top-K selection.** From the current logits (merged from the previous hop's experts), select the top K entries. Softmax over the K selected logits gives merge weights.
2. **Exit check.** If the exit token appears anywhere in the top-K for all samples in the batch, stop.
3. **Expert execution.** All active (non-exit) experts in the top-K run on the full batch. Each produces an output tensor and outbound routing logits.
4. **Weighted merge.** Expert outputs are merged into a single state using the softmax weights from step 1. The outbound logits from each expert are merged with the *same weights*.
5. **Residual add.** `x = x + merged_output`.
6. **Next logits.** The weighted-merged outbound logits become the input logits for the next hop's top-K selection.

This means routing is a chain: experts collectively decide where the signal goes next. An expert that produces strong output also has more influence on the next routing decision, because its outbound logits get higher weight in the merge.

### Exit Mechanism

Exit is not a separate decision — it competes with experts in the same logit space. If the merged outbound logits rank exit in the top-K for every sample in the batch, the loop terminates. This means:

- Experts can learn to vote for exit when their processing is sufficient.
- Exit must win across the entire batch (conservative — no sample left behind).
- A hard `max_hops` cap prevents runaway chains. Default is `2 * pool_size`, annealable during training by setting `model.max_hops`.

### Revisiting

Experts can be selected multiple times across hops. There is no exclusion mask. If the routing chain keeps selecting expert 3, expert 3 runs every hop. This is intentional — it allows the network to learn iterative refinement through a single expert.

## Relationship to MoE Grid

| Property | MoE Grid | Pool of Experts |
|---|---|---|
| Expert count | `n_experts * n_layers` | `n_experts * n_layers` (same total) |
| Depth | Fixed `n_layers` | Dynamic, 1 to `max_hops` |
| Routing | Per-layer router | Per-expert outbound router |
| Who decides next hop | Layer l+1's router | Current hop's experts (merged vote) |
| Expert reuse | Never (grid cells are distinct) | Allowed |

When invoked via `bench_ssm.py --poe`, `pool_size` is computed as `n_experts * layers`, making the total expert count identical to the MoE grid for fair comparison.

## Usage

```python
from ulb import ULBBlock, ULBConfig, PoolOfExperts

cfg = ULBConfig(d_model=64, n_heads=4, paired=True, attn_mode='blend')
make = lambda: ULBBlock(cfg)

model = PoolOfExperts(make, pool_size=8, dim=64, top_k=2, max_hops=16)

# After forward:
y = model(x)
print(model.last_n_hops)  # how many hops this forward took
print(model.aux_loss)      # block-level aux losses from experts
```

### Benchmark

```bash
python scripts/bench_ssm.py --poe --n-experts 4 --layers 2 --top-k 2 \
    --models ULBBlendP --tasks mixed
```

This creates a pool of 8 experts (4 * 2), max_hops=16 (2 * 8), top-k=2 per hop.

## Implementation Notes

- Experts run on the full batch (not masked subsets) to avoid Python-loop overhead and small-kernel launches. The wasted compute on non-selected samples is cheaper than the alternative.
- `last_n_hops` is set after each forward for logging/analysis.
- `max_hops` is a plain attribute, not a buffer — change it freely during training for annealing.
- `aux_loss` aggregates any block-level auxiliary losses from the expert ULB blocks.
