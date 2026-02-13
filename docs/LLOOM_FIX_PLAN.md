# LLooM Fix Plan: Replace MLP Token Side with Feature-Attention

## Problem

The MLP (SwiGLU) token side is useless. Every model on every task learns
that attention is all you need. Models start by exploring both sides equally
(often with MORE hops on the token side), then converge to routing everything
through the sequence-side attention experts. The token side's MLP experts
can't learn to do useful work fast enough to compete.

MLPs are implicitly doing feature mixing — `up_proj` maps to a higher-dim
space, the nonlinearity does pointwise selection, `down_proj` maps back —
but it's a fixed, learned, content-independent mixing pattern. This isn't
worth the routing detour when you have dynamic, content-dependent attention
available.

## Solution

Replace the MLP experts with **feature-attention** experts. Instead of
attention over the sequence dimension (what the left side does), the right
side does attention over **features within a single token**.

This is what MLPs *try* to do — feature mixing — but with dynamic,
content-dependent attention instead of fixed learned weights.

### Architecture Symmetry

Both sides are now attention, just over different dimensions:

- **Left side (Sequence Pool)**: Attention over positions. Input `(B, T, D)`,
  QKV operates over `T`. Learns which tokens relate to which.
- **Right side (Token Pool)**: Attention over features. Input `(N, D)` flat
  tokens, each expanded to `(n_groups, group_dim)` feature groups. QKV operates
  over `n_groups`. Learns which features relate to which, conditioned on content.

The "two threads, one fabric" metaphor becomes literal: warp (sequence) and
weft (features).

### Feature-Attention Expert Flow

Each token is expanded into feature groups, which attend to each other:

```
x: (N, D)                                 # flat token bucket

h_up = SiLU(x @ up_w)                     # (N, D_feat) where D_feat = expansion * D
h = h_up.view(N, n_groups, group_dim)      # reshape into feature groups

q, k, v = (h @ qkv_w).split(group_dim)    # per-group QKV
# reshape for heads: (N, n_heads, n_groups, head_dim)
y = SDPA(q, k, v, is_causal=False)         # attend over feature groups
y = y.reshape(N, n_groups, group_dim)       # concat heads
y = y @ o_w                                # (N, n_groups, group_dim) mix across heads

y = y.view(N, D_feat)                      # flatten back
y = RMSNorm(y) * h_up                      # skip-multiply
out = y @ down_w                           # (N, D)
```

With `D=64` and `feat_expansion=4`:
- `D_feat = 256`
- `group_dim=16` gives 16 feature groups, 16x16 attention matrix per token
- `group_dim=32` gives 8 feature groups, 8x8 attention matrix per token

Feature-attention is non-causal (no ordering over features).

### Why o_proj is included

With 1 head (the default), `o_proj` is `group_dim -> group_dim` per group —
a small learned linear, slightly redundant. With >1 heads it does its real
job of mixing across heads within each group. Always included so the
architecture doesn't change shape when adjusting head count. The cost is
trivial: `group_dim^2` params per expert.

### Token-Side Dispatch Model

The token side operates on a **flat bucket of tokens** `(N, D)` where
`N = B*T`. Dispatch doesn't care about sample boundaries. Sample structure
only matters at RCV vote time: reshape votes to `(B, T)`, tally per-sample,
and gather a sample's tokens via reshape when it votes to exit/bridge.

Dispatch uses **gather-based** routing (same as sequence side): gather
selected expert weights per token, loop over `top_k` (2-3), run only the
assigned experts. NOT the old MLPParamBank pattern of looping over all
`pool_size` experts and masking.

## Changes

### 1. `config.py` — Replace MLP config with feature-attention config

**Remove:**
- `tok_expansion: float`
- `tok_inner_dim` property / `_tok_inner_dim` cache

**Add:**
- `feat_expansion: float = 4.0` — token expansion factor
- `feat_group_dim: int = 16` — feature group size
- `feat_n_heads: int = 1` — heads within feature-attention

**Add derived properties:**
- `feat_dim` -> `_snap_dim(dim * feat_expansion, feat_group_dim)`
- `feat_n_groups` -> `feat_dim // feat_group_dim`
- `feat_head_dim` -> `feat_group_dim // feat_n_heads`

**Validation:**
- `feat_dim % feat_group_dim == 0`
- `feat_group_dim % feat_n_heads == 0`

### 2. `dispatch.py` — Add `FeatureAttentionParamBank`, delete `MLPParamBank`

New class with banked weights (all shared/private split):

```
up_bank:   (P, D, D_feat)
qkv_bank:  (P, group_dim, 3 * group_dim)
o_bank:    (P, group_dim, group_dim)
norm_bank: (P, D_feat)
down_bank: (P, D_feat, D)
```

Forward: gather-based dispatch, loop over `top_k`.

Input/output: `(N, D)` flat tokens.

### 3. `token_pool.py` — Swap expert bank

Replace `MLPParamBank` with `FeatureAttentionParamBank`.

Constructor takes `feat_dim`, `feat_group_dim`, `feat_n_heads` instead of
`inner_dim`.

Everything else unchanged: routing, RCV, hop conditioning, bridge mechanics.

### 4. `lloom.py` — Wire new config params

- Update `TokenPool` construction to pass feature-attention params.
- Update `lloom_megatron_init_` to handle `FeatureAttentionParamBank` output
  projections (`down_bank`/`down_shared` and `o_bank`/`o_shared` get output
  scaling).

### 5. `__init__.py` — Update exports

- Export `FeatureAttentionParamBank`
- Remove `MLPParamBank`

### 6. `bench_ssm.py` — Update LLooM config construction

Remove `tok_expansion` references. The new config fields (`feat_expansion`,
`feat_group_dim`, `feat_n_heads`) use sensible defaults so the bench harness
may not need explicit overrides.

### 7. Tests

- `test_dispatch.py` — Add `FeatureAttentionParamBank` tests (shapes, grad
  flow, sharing, non-causal attention). Remove `MLPParamBank` tests.
- `test_token_pool.py` — Routing/RCV tests unchanged. Dispatch shape tests
  updated for new expert architecture.
- `test_lloom.py` — End-to-end tests should pass once wiring is correct.

### 8. `LLOOM.md` — Update docs

Rewrite token side description, expert architecture, remove FiLM section,
update config block and file structure.

## What is NOT changing

- Routing mechanics (`RoutingPool`, `route()`, `select_topk`, `classify_topk`)
- RCV voting
- Bridge passthrough (raw `(B, T, D)`)
- Sequence pool (untouched)
- Hop conditioning
- Stem router / stems
- Main routing loop structure in `lloom.py`
- FiLM is already removed; stays removed for now
