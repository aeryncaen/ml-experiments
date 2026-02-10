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
- **Per-expert hop embeddings** — each expert has a learned embedding table `(max_hops, dim)`, initialized as `randn * 0.02`. Provides per-hop identity to the expert.
- **Per-expert hop gates** — `Linear(hop_gate_dim, 1, bias=False)` where `hop_gate_dim = min(dim, 12)`. Content-dependent gating that modulates how much the hop embedding influences the input. See [Hop Conditioning](#hop-conditioning).
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

1. **Exit ramp.** Exit logits are boosted by `exit_ramp_scale * (hop / max_hops)`. Deeper hops face increasing pressure to exit, making the max_hops cap differentiable rather than a hard cliff. Default scale is 3.0.
2. **Top-K selection.** From the current logits, select the top K entries. Softmax over the K selected logits gives initial merge weights.
3. **Partial exit and adaptive width.** Exit slots in the top-K get their weights zeroed out; the remaining real-expert weights are renormalized to sum to 1. A sample with 1 exit + 1 real expert in its top-2 continues with that single expert at full weight — this is **adaptive width**. A sample only fully exits (receives zero expert contribution) when ALL of its top-K slots are exit. If all samples have fully exited, the loop stops.
4. **Hop conditioning.** Before running each expert, the hidden state is conditioned with a content-gated hop embedding. See [Hop Conditioning](#hop-conditioning).
5. **Expert execution.** Experts selected by still-active samples run on the full batch. Each produces an output tensor and outbound routing logits.
6. **Weighted merge.** Expert outputs are merged into a single state using the renormalized weights from step 3. The outbound logits from each expert are merged with the *same weights*.
7. **Residual add.** `x = x + merged_output`.
8. **Next logits.** The weighted-merged outbound logits become the input logits for the next hop's top-K selection.

This means routing is a chain: experts collectively decide where the signal goes next. An expert that produces strong output also has more influence on the next routing decision, because its outbound logits get higher weight in the merge.

### Exit Mechanism and Adaptive Width

Exit is not a separate decision — it competes with experts in the same logit space. Exit slots in a sample's top-K get their weights zeroed; the remaining real-expert weights are renormalized to sum to 1. This creates three regimes:

- **No exit slots in top-K:** Normal operation — all K experts contribute with their softmax weights.
- **Partial exit (some exit, some expert):** The sample continues with fewer active experts, each receiving proportionally more weight. With top-k=2, picking 1 exit + 1 expert means that expert runs at full weight. This is **adaptive width** — the model narrows its expert usage when confident.
- **Full exit (ALL top-K are exit):** The sample receives zero expert contribution (`x += 0`). Its state is frozen from this point, only receiving the final exit layer.

The loop terminates when every sample in the batch has fully exited. Samples that exit early coast with no update while remaining active samples continue routing. This means:

- Experts can learn to vote for exit when their processing is sufficient.
- Samples exit independently. A sample that exits at hop 1 is unaffected by other samples that continue to hop 5.
- Depth AND width are both adaptive and input-dependent. The model can choose "1 expert deep for many hops" or "2 experts wide for fewer hops" per sample.
- A hard `max_hops` cap prevents runaway chains for any sample. Default is `2 * pool_size`.
- The exit ramp makes approaching max_hops increasingly expensive, so the cap is soft in practice.

### Revisiting

Experts can be selected multiple times across hops. There is no exclusion mask. If the routing chain keeps selecting expert 3, expert 3 runs every hop. This is intentional — it allows the network to learn iterative refinement through a single expert.

### Hop Conditioning

Each expert receives per-hop identity through a content-gated embedding, modeled after the value embedding + gate pattern from nanogpt. Before an expert runs at hop `h`:

```python
gate = sigmoid(hop_gates[eid](h_state[..., :hop_gate_dim]))   # (B, T, 1)
hop_cond = gate * hop_embeds[eid][h]                           # (B, T, dim)
expert_input = h_state + hop_cond
```

- `hop_embeds[eid]` is a `(max_hops, dim)` parameter, initialized as `randn * 0.02`.
- `hop_gates[eid]` is `Linear(hop_gate_dim, 1, bias=False)` where `hop_gate_dim = min(dim, 12)`.

The gate looks at the first `hop_gate_dim` dims of the hidden state and produces a scalar per position. This makes the conditioning content-dependent: the expert decides how much the hop embedding matters for each token based on the current hidden state. An expert revisited at hop 3 vs hop 7 sees different hop embeddings, letting it adapt its behavior across the chain.

Total parameter cost is small: 2,096 params for pool_size=4, max_hops=8, dim=64.

## Training Features

### Router Noise

Gaussian noise added to router logits during training (`router_noise_scale`, default 1.0). Encourages exploration of different routing paths early in training. Annealed linearly to 0 over epochs via `model.router_noise_scale`.

### Router Dropout

Optional (default off, `router_dropout=0.0`). When enabled, drops a random number of exit logits per sample (0 to 25% of exit slots), never touching expert logits. The count dropped is itself random per sample — not a fixed percentage. This gives experts more exposure during training without removing the exit-heavy prior entirely.

### Hop Counting

`model.last_mean_hops` tracks the true mean hops per sample across the batch. Each hop, the number of samples that didn't exit is counted; the total is divided by batch size after the loop. This is the value displayed on the progress bar and used for logging.

## Weight Sharing

Expert pools can share a fraction of their parameters across all experts via dim-slicing. For a weight `(out_dim, in_dim)` with sharing fraction `f`:

- `f * out_dim` output dimensions come from a **shared weight** — one tensor, same across all experts.
- `(1 - f) * out_dim` output dimensions come from a **private weight** — unique per expert.
- Forward: `cat([shared(x), private(x)], dim=-1)`.

Total parameters go down because the shared slice is stored once instead of N times. Shared weights train with gradients from all experts.

### Three independent sharing fractions

| Fraction | What it shares | Applies to |
|---|---|---|
| `block_shared_fraction` | Expert block internals | `up_proj`, `down_proj`, `q_proj`, `k_proj`, `v_proj`, `o_proj`, learnable swish activations |
| `router_shared_fraction` | Per-expert outbound routers | `Linear(dim, n_router_options)` per expert |
| `hop_shared_fraction` | Hop conditioning | Hop embeddings (`ParamHolder`) and hop gates (`Linear`) |

The convenience kwarg `shared_fraction` sets all three when individual fractions aren't specified.

### Implementation

- `SharedLinear` — drop-in replacement for `nn.Linear`, splits output dims into shared + private.
- `SharedLearnableSwish` — splits beta dims for learnable swish activations.
- `SharedParameter` — splits last dim of raw parameter tensors (for hop embeddings).
- `share_expert_weights(experts, fraction)` — walks all ULBBlock submodules and replaces `Linear`/`LearnableSwish` with shared variants.
- `share_linear_list(linears, fraction)` — shares across a flat `ModuleList` of `Linear` modules (routers, hop gates).
- `share_parameter_list(params, fraction)` — shares across a `ModuleList` of `ParamHolder` modules (hop embeddings).

All in `src/ulb/shared.py`.

### Parameter savings example (pool_size=4, dim=64)

```
No sharing:         388,464
Block 40%:          311,928 (-20%)
Router 40%:         387,312 (-0.3%)
Both 40%:           310,776 (-20%)
```

Block sharing dominates the savings (expert blocks hold most of the parameters). Router sharing has minimal param impact but the most behavioral impact in bench results — shared routers give all experts a baseline understanding of each other's strengths.

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

model = PoolOfExperts(make, pool_size=8, dim=64, top_k=2, max_hops=16,
                      block_shared_fraction=0.4,
                      router_shared_fraction=0.4,
                      hop_shared_fraction=0.0)

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
    --poe-block-share-fraction 0.4 --poe-router-share-fraction 0.4 \
    --poe-hop-share-fraction 0.0 \
    --models ULBBlendP --tasks mixed --save-dir out/
```

This creates a pool of 8 experts (4 * 2), max_hops=16 (2 * 8), top-k=2 per hop, with 40% block and router weight sharing. `--save-dir` saves trained checkpoints for later analysis with `scripts/analyze_poe.py`.

CLI args for weight sharing (all default to 0.0):

- `--poe-block-share-fraction` — expert block internals
- `--poe-router-share-fraction` — per-expert outbound routers
- `--poe-hop-share-fraction` — hop embeddings and gates

## Implementation Notes

- Experts run on the full batch (not masked subsets) to avoid Python-loop overhead and small-kernel launches. The wasted compute on non-selected samples is cheaper than the alternative.
- `last_mean_hops` is set after each forward for logging/analysis.
- `max_hops` is a plain attribute, not a buffer — change it freely during training.
- `aux_loss` aggregates any block-level auxiliary losses from the expert ULB blocks.
- `exit_ramp_scale` is a plain attribute — adjustable at runtime.
