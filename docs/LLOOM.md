# LLooM: Dual-Paradigm Adaptive Routing Architecture

*An LLM that's a Loom — two threads, one fabric.*

## Motivation

Two routing paradigms for neural networks excel at fundamentally different tasks:

- **Sequence-level routing through attention pools** — where entire sequences
  self-route through a shared pool of attention experts with no fixed depth —
  yields dramatically better performance on algorithmic and structured reasoning.
- **Token-level routing through MLP experts** — the familiar mixture-of-experts
  approach — excels at fuzzy tasks like language modeling, but at substantial
  cost to algorithmic and multi-task capability.

These are not a fast/slow or primary/support distinction. They are two genuinely
different reasoning systems operating across fundamentally different dimensions.
Attention performs relational, positional, and structural reasoning **over the
sequence**. MLPs perform nonlinear function composition — smooth interpolation,
pattern completion, generalization **across features at the token level**.

A critical insight: feature transformations should not be learned at the expert
level. When each expert learns its own feature basis, representations fragment
and experts cannot compose cleanly. Instead, feature transformation should be
handled through a shared pool with 50% parameter sharing, just like the
attention side.

LLooM combines both paradigms into a single architecture with a bridge between
them. The model learns not just what to compute, but which computational
paradigm to use and when to switch.

## Architecture Overview

```
                    ┌─────────────────────┐
                    │     Entry Stem       │
                    │  (Attn + SwiGLU MLP) │
                    │    + Stem Router     │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              │              ▼
     ┌────────────────┐      │     ┌────────────────┐
     │  Sequence Side  │◄────┼────►│   Token Side    │
     │  (Attn Pool)    │  bridge   │  (MLP Pool)     │
     │                 │◄────┼────►│                 │
     │  sample-routed  │     │     │  token-routed   │
     │  top-k attn     │     │     │  top-k SwiGLU   │
     │  experts        │     │     │  experts + FiLM  │
     └───────┬─────────┘     │     └───────┬─────────┘
             │               │             │
             └───────────────┼─────────────┘
                             │
                    ┌────────▼────────────┐
                    │     Exit Stem       │
                    │  (Attn + SwiGLU MLP) │
                    └────────┬────────────┘
                             │
                          output
```

### Three Components

- **Sequence Side** — A pool of attention experts. Entire samples route through
  this pool, choosing a next expert at each hop. Reasoning is sequence-level:
  positional, relational, structural.
- **Token Side** — A pool of SwiGLU MLP experts. Individual tokens route through
  this pool, choosing a next expert at each hop. Reasoning is token-level:
  nonlinear composition, distributional, fuzzy.
- **The Bridge** — A direct passthrough (no projection, no compression) that
  allows samples/tokens to cross from one side to the other mid-computation.

## Routing Mechanics

### Router Space (Both Sides)

For a pool of size P, the router space has `P + 2` options:

| Slot index | Meaning |
|------------|---------|
| `0..P-1`   | Route to expert `i` |
| `P`        | Exit |
| `P+1`      | Bridge to other side |

Single exit slot, single bridge slot, each with a configurable starting bias.
This replaces the previous single/half/squared exit-slot-count scheme with
simple scalar tuning that achieves the same functional effect.

### Sequence Side (Sample Routing)

Each attention expert has an outbound router. After processing a sample, the
router produces logits over `P+2` options, and top-k selection determines the
next action.

**Per-hop flow:**
1. Hop norm + content-gated hop embedding (per expert)
2. Banked attention dispatch for top-k experts (exit/bridge slots zeroed,
   weights renormalized over expert slots only)
3. Weighted merge across top-k + residual add
4. Outbound router logits (from merged output)
5. Apply exit bias (ramping: `exit_bias_init + exit_ramp_scale * hops_used / max_hops`)
   and bridge bias (`bridge_bias_init`, fixed)
6. Top-k selection + classify: **all** top-k are exit → sample exits;
   **all** top-k are bridge → sample bridges; otherwise → continue

Mixed top-k (e.g., one expert + one exit) dispatches to the expert and continues.

### Token Side (Token Routing + Ranked Choice Voting)

Each MLP expert has an outbound router. After processing tokens, each token's
router produces logits over `P+2` options.

**Per-hop flow:**
1. Hop norm + content-gated hop embedding (per expert)
2. Token-level router (entry router for first hop, outbound routers thereafter)
   → per-token top-k selection
3. Classify each token's top-k picks as expert/exit/bridge
4. **Token dispatch**: tokens with any expert in their top-k dispatch to those
   experts (exit/bridge weights zeroed, renormalized). Tokens with only
   exit/bridge in top-k **park** — no dispatch, state unchanged.
5. Residual add for dispatched tokens only
6. **Sticky votes**: newly parked tokens lock their top-1 vote (exit or bridge).
   Previously parked tokens keep their locked vote.
7. **Ranked Choice Vote** across all tokens in the sample:
   - Each active token's first-choice category (continue/exit/bridge) is its vote
   - Each parked token's locked category is its vote
   - Round 1: tally first choices. If any category >50% → sample action.
   - Round 2 (if no majority): eliminate category with fewest votes. Transfer
     eliminated-category voters to their next-ranked category. Recount.
   - Round 3: two categories remain, one must have ≥50%.
8. Sample action: **continue** → next hop. **exit** → done. **bridge** → cross
   to sequence side.

**Sticky votes** mean exit/bridge pressure monotonically increases within a
token-side loop as more tokens park. This provides a natural convergence
guarantee independent of the exit ramp.

### Hop Budgets

Each side has its own maximum hop count, cumulative across all visits. Attention
is O(T²) per hop; MLP hops are O(T) per token. The token side can afford
significantly more hops for the same compute budget. This creates a natural
incentive: the model learns to do as much work as it can on the cheap token side
before spending expensive attention hops.

## The Bridge

Mechanically simple. No learned projection, no norm, no compression. Raw hidden
state passes through unchanged. The receiving side's first expert, equipped with
its hop embeddings and shared parameter basis, acts as the translator.

**Seq → Tok**: Mean-pool sample state → generate FiLM conditioning → pass raw
`(B, T, D)` to token side. Token side's entry router handles first-hop routing.

**Tok → Seq**: Gather tokens back to `(B, T, D)` → pass raw to sequence side.
Sequence side's outbound routing continues.

The bridge enables multiple crossings, limited only by per-side cumulative hop
budgets.

## FiLM Conditioning (Token Side)

The fundamental limitation of token-routing is that tokens are blind to each
other. FiLM (Feature-wise Linear Modulation) conditioning gives them global
context without attention cost.

**Generated at every entry to the token side** (from sequence-side bridge or
from stem bridge):

```
sample_repr = mean_pool(x, dim=1)                  # (B, D)
film_params = film_proj(sample_repr)                # low-rank: (B, 4 * D_inner)
γ_up, β_up, γ_down, β_down = chunk(film_params, 4)
```

**Applied in every MLP expert:**
```
h = up_proj(x)                    # the query
h = γ_up * h + β_up              # FiLM on query
h = SwiGLU_internal(h)            # unconditioned knowledge
h = down_proj(h)                  # the answer
h = γ_down * h + β_down          # FiLM on answer filter
```

**Static for the entire token-side loop.** Not refreshed between hops. The MLP
internals — the expert's stored knowledge — are pure and unconditioned. The
sample context shapes how you query the databank and how you filter the
response, not the data itself.

The staleness of FiLM conditioning becomes a natural bridge-back signal: if the
token side has done enough work that the attention-derived context is no longer
adequate, the correct move is to bridge back, run attention, and receive fresh
FiLM conditioning on the next token-side visit.

## 50% Parameter Sharing Within Sides

All experts and routers within each side share 50% of their parameters. The
first half forms a common representational basis; the second half specializes.

This solves representational fragmentation: tokens can route through any sequence
of experts without encountering incompatible feature spaces, because all experts
share a common subspace. Specialization lives in the private weights.

Weights are **not** shared across sides. The two sides maintain fully
independent parameters, preserving their distinct computational identities.

## Hop Embeddings

Every expert on both sides maintains its own learned set of hop embeddings with
content-gated application:

```
gate = sigmoid(linear(h[..., :hop_gate_dim]))
h = h + gate * hop_embed[hop]
```

These encode computational lifetime — implicitly including cross-bridge history.
If a token arrives at hop 7 on the token side carrying a representation shaped
by attention processing, the hop embedding provides temporal context and the
hidden state carries the computational signature.

## Entry and Exit Stems

**Entry stem**: A full transformer block (attention sublayer + SwiGLU MLP
sublayer, each with residuals). Non-routed, shared. Processes raw embeddings
into representations rich enough for routing decisions.

**Stem router**: Produces sample-level logits over `s0..s_{S-1}, bridge, exit`:
- `s_i` → enter sequence pool at expert `i`
- `bridge` → enter token pool (token entry router handles expert selection)
- `exit` → skip pools entirely (one transformer block was enough)

**Exit stem**: Same architecture as entry stem. Non-routed, shared. Final
enrichment after all routing is done.

**Minimum compute path**: `embed → entry_stem → exit_stem → head`. Two full
transformer blocks. Everything between is adaptive.

## Router Noise Annealing

Gaussian noise injected into router logits at training start to force
exploration across all experts. Noise anneals linearly to zero over training,
allowing the model to converge on stable routing patterns.

---

## Code Design

### Base Class: `RoutingPool`

Shared routing mechanics factored into a base class. Subclasses provide the
expert dispatch and decision aggregation.

```
RoutingPool (base)
├── pool_size, top_k, max_hops, n_options
├── exit_bias_init, bridge_bias_init, exit_ramp_scale
├── router_noise_scale, shared_fraction, hop_gate_dim
├── router_bank (banked outbound routers, 50% shared)
├── hop_embeds, hop_gates (per-expert, content-gated)
├── hop_norm: RMSNorm
│
├── perturb_logits()          — noise injection
├── apply_biases()            — exit ramp + bridge bias
├── classify_topk()           — expert/exit/bridge masks
├── select_and_weight()       — top-k with exit/bridge weight zeroing
├── apply_hop_conditioning()  — content-gated hop embed
│
├── dispatch()                — ABSTRACT: expert forward pass
├── aggregate_decisions()     — ABSTRACT: sample-level continue/exit/bridge
├── prepare_bridge_out()      — ABSTRACT: format hidden states for other side
├── accept_bridge_in()        — ABSTRACT: receive hidden states, set up context

SequencePool(RoutingPool)
├── expert_bank: AttentionParamBank
├── dispatch()                — banked attention (loop over top_k, not pool_size)
├── aggregate_decisions()     — all-top-k check (vectorized)
├── prepare_bridge_out()      — identity (raw passthrough)
├── accept_bridge_in()        — identity

TokenPool(RoutingPool)
├── expert_bank: MLPParamBank
├── film_generator: low-rank projection
├── entry_router: standalone token-level router for first hop
├── dispatch()                — banked SwiGLU + FiLM (einsum-based)
├── aggregate_decisions()     — vectorized RCV with int8 vote state
├── prepare_bridge_out()      — reshape (B*T, D) → (B, T, D)
├── accept_bridge_in()        — generate FiLM, reset vote state
```

### Expert Architectures

**Sequence-side expert** (attention):
```
up_proj(SiLU) → MHA(SDPA) → norm → skip-multiply with h_up → down_proj
```

Banked as `AttentionParamBank`:
- `up_bank`: (P, D, D_inner)
- `qkv_bank`: (P, D_inner, 3 * D_inner) — fused QKV
- `o_bank`: (P, D_inner, D_inner)
- `norm_bank`: (P, D_inner) — RMSNorm weights
- `down_bank`: (P, D_inner, D)

**Token-side expert** (SwiGLU MLP):
```
gate_up_proj → FiLM(γ_up, β_up) on up → SiLU(gate) * up → down_proj → FiLM(γ_down, β_down)
```

Banked as `MLPParamBank`:
- `gate_up_bank`: (P, D, 2 * D_inner) — fused gate+up
- `down_bank`: (P, D_inner, D)

### Top-Level: `LLooM`

```
embed → entry_stem (attn + MLP) → stem_router

Main loop (fixed iterations during training):
    Sequence side runs its sub-loop (fixed max_hops, mask-gated)
    Token side runs its sub-loop (fixed max_hops, mask-gated)
    Handle bridge crossings (mask updates, FiLM generation)

exit_stem (attn + MLP) → final_norm → head
```

---

## Performance Strategy

### PyTorch Fast Path (Day 1)

1. **Fixed-iteration loops with masks.** Both sides use `for hop in range(max_hops)`.
   Active masks gate computation — done samples contribute zero but flow through
   the graph. No `while` loops, no early breaks during training. This is the
   single most important decision for `torch.compile` compatibility.

2. **No Python loops over experts.** All dispatch is banked: stacked weight
   tensors, `torch.gather` by index, batched `einsum`. The only Python loop is
   over `top_k` (2-3 iterations, fixed), not `pool_size`.

3. **Static tensor shapes.** Both sides keep tensors at full `(B, T, D)` shape.
   Inactive samples/tokens get masked results (zero delta). No dynamic tensor
   packing/unpacking.

4. **Vectorized RCV.** Vote state as `int8` tensors. Category tallies via
   `sum()` and `where()`. No per-sample Python loops.

5. **Standalone dispatch functions.** `AttentionParamBank.forward` and
   `MLPParamBank.forward` are clean tensor-in tensor-out functions with no side
   effects. Future drop-in replacement with custom ops.

6. **Profiling hooks.** Per-hop timing (routing, dispatch, merge, vote), active
   counts, mean hops, bridge frequency. `torch.cuda.Event`-based, disabled by
   default.

### Triton Path (Future)

When profiling identifies bottlenecks:

1. **First kernel target**: token-side fused dispatch (gather weights + SwiGLU +
   FiLM + weighted top-k reduction).
2. **Second**: sequence-side fused projection kernels (up_proj + QKV). Do NOT
   rewrite attention — keep FlashAttention/SDPA.
3. **Third**: token vote-tally kernel, only if routing logic is material.

Integration via `torch.library.custom_op` + `register_fake` + `register_autograd`
(same pattern as `s6/scan.py`). This keeps `torch.compile` happy — no graph
breaks at kernel boundaries.

---

## Configuration

```python
@dataclass
class LLooMConfig:
    dim: int = 64

    # Entry/exit stems
    stem_n_heads: int = 4
    stem_mlp_expansion: float = 1.75

    # Sequence side
    seq_pool_size: int = 8
    seq_top_k: int = 2
    seq_n_heads: int = 4
    seq_expansion: float = 1.75
    seq_max_hops: int = 16          # default: 2 * seq_pool_size

    # Token side
    tok_pool_size: int = 8
    tok_top_k: int = 2
    tok_expansion: float = 1.75
    tok_max_hops: int = 32          # generous — MLP hops are cheap

    # Routing
    exit_bias_init: float = 0.0
    bridge_bias_init: float = 0.0
    exit_ramp_scale: float = 3.0
    router_noise: float = 1.0
    shared_fraction: float = 0.5

    # FiLM
    film_rank: int = 16             # low-rank bottleneck

    # Hop conditioning
    hop_gate_dim: int = 12
```

---

## File Structure

```
src/lloom/
    __init__.py
    config.py           # LLooMConfig
    routing_pool.py     # RoutingPool base class
    dispatch.py         # AttentionParamBank, MLPParamBank
    sequence_pool.py    # SequencePool(RoutingPool)
    token_pool.py       # TokenPool(RoutingPool) + FiLM + RCV
    lloom.py            # LLooM top-level + stems + bridge logic

tests/lloom/
    __init__.py
    test_config.py      # Config validation, derived properties, snap_dim
    test_dispatch.py    # Both param banks: shapes, grad flow, sharing, FiLM
    test_routing_pool.py # Base routing: noise, biases, classify, hop cond
    test_sequence_pool.py # Attn dispatch, all-top-k decisions, bridge
    test_token_pool.py  # RCV voting, sticky votes, FiLM, parking, entry router
    test_lloom.py       # End-to-end fwd/bwd, stems, bridge, init, causal
```

## Implementation Phases

| Phase | File(s) | What | Dependencies |
|-------|---------|------|-------------|
| 1 | `config.py` | `LLooMConfig` dataclass | — |
| 2a | `dispatch.py` | `AttentionParamBank` | — |
| 2b | `dispatch.py` | `MLPParamBank` | — |
| 2t | `test_dispatch.py` | Dispatch tests | 2a, 2b |
| 3 | `routing_pool.py` | `RoutingPool` base class | config |
| 3t | `test_routing_pool.py` | Routing pool tests | 3 |
| 4a | `sequence_pool.py` | `SequencePool` | dispatch, routing_pool |
| 4at | `test_sequence_pool.py` | Sequence pool tests | 4a |
| 4b | `token_pool.py` | `TokenPool` + FiLM + RCV | dispatch, routing_pool |
| 4bt | `test_token_pool.py` | Token pool tests | 4b |
| 5 | `lloom.py` | `LLooM` model + stems + bridge | sequence_pool, token_pool |
| 5t | `test_lloom.py` | Integration tests | 5 |
| 6 | `bench_ssm.py` | Benchmark integration | lloom |

Phases 2a/2b are independent. Phases 4a/4b are independent.

## Testing Strategy

Every module gets unit tests written immediately after implementation and run
before moving to the next phase. Tests live in `tests/lloom/` and use `pytest`.

### Test Categories

**Gradient flow** — Every module must pass the "loss.backward() populates
all param .grad" test. Construct a minimal forward pass, compute a scalar
loss, backward, assert every `p.requires_grad` parameter has a non-None,
non-zero gradient. This catches dead code paths, detached tensors, and
in-place ops that break autograd.

**Shape correctness** — Every forward pass is tested with multiple
(B, T, D) configurations to catch broadcasting bugs and hardcoded dims.

**Router behavior** — Test that:
- Noise injection is actually stochastic (different outputs with noise > 0)
- Noise injection is deterministic when noise = 0
- Exit bias ramp increases with hop count
- Bridge bias stays fixed
- Top-k classification correctly separates expert/exit/bridge
- Exit/bridge weights are zeroed and expert weights renormalized

**Init behavior** — Test that:
- Parameter sharing creates actual weight ties (shared slice is same object
  or same storage)
- Bias inits match config (exit_bias_init, bridge_bias_init)
- Weight norms are reasonable at init (no explosion/vanishing)

**Routing decisions** — Test that:
- Sequence side: all-top-k agreement correctly triggers exit/bridge/continue
- Token side RCV: majority in round 1 exits early, elimination in round 2
  works, sticky votes accumulate monotonically, parked tokens don't dispatch

**FiLM** — Test that:
- FiLM params have correct shape (B, 4 * D_inner)
- FiLM modulation changes MLP output (not identity)
- FiLM is static across token-side hops (same conditioning reused)

**Bridge** — Test that:
- Bridge is pure passthrough (output == input, no learned params)
- FiLM is regenerated on each token-side entry
- Hop counters are cumulative across bridge crossings

**End-to-end** — Test that:
- Minimum compute path works: embed → entry_stem → exit_stem → head
- Full forward + backward completes without error
- Output shape is correct for various (B, T) inputs
- Causal masking is respected in attention operations
