# Benchmark Tasks

## Overview

`bench_ssm.py` evaluates sequence models on five synthetic tasks, each targeting a different capability. Tasks can be run individually or combined in **mixed mode**, which tests a model's ability to learn disparate tasks simultaneously with a single set of weights.

All tasks share a unified vocabulary of 64 tokens. Tokens 0-31 are "normal"; tokens 32-63 are "marked" (used only by selective_copy). Sequences are generated on the fly — no fixed dataset.

## Tasks

### delay

1-step memory. The target at position `t` is the input at position `t-1`. Position 0 is masked (no valid target). This is the easiest task — any model with a single step of memory solves it.

- Input vocab: `[0, 32)`
- Scored positions: all except position 0

### selective_copy

Selective recall. The input is a sequence of random tokens, with 4 random positions "marked" by shifting their token into the 32-63 range. The target is masked everywhere except the last 4 positions, which must predict the original (un-shifted) values of the marked tokens, in order.

The model must: (1) notice which positions are marked, (2) remember those specific token values, (3) reproduce them at the end of the sequence. This tests content-addressable memory — the model can't rely on fixed positions because the markers are placed randomly.

- Input vocab: `[0, 64)` (32-63 = marked)
- Scored positions: last 4 only
- Key parameter: `n_markers=4`

### parity

Running cumulative XOR. Input is binary (`{0, 1}`), target at position `t` is the XOR of all inputs up to and including `t`. This requires maintaining a 1-bit state register across the full sequence. Every position is scored.

- Input vocab: `{0, 1}`
- Scored positions: all

### mod_arith

Running modular sum. Input is digits in `[0, 5)`, target at position `t` is `(sum of inputs 0..t) mod 5`. A harder version of parity — requires tracking a multi-bit counter (mod 5 = ~2.3 bits). Every position is scored.

- Input vocab: `[0, 5)`
- Scored positions: all
- Key parameter: `mod_base=5`

### induction

Pattern completion. A random pattern of length `plen` (between `L/3` and `2L/3`) is generated, then repeated: `[pattern, pattern]`. The input is the first `L` tokens of this doubled sequence; the target is next-token prediction (shifted by 1).

During the first copy of the pattern, the next token is unpredictable (random content). During the second copy, the model can predict the next token by recognizing it saw this exact pattern before. The theoretical ceiling for a causal model is ~50% accuracy because first-copy positions are pure guessing. Breaking past 50% requires the model to detect the repetition boundary and recall earlier tokens. Pattern length varies per sample to prevent positional shortcuts.

- Input vocab: `[0, 32)`
- Scored positions: all (but first-copy positions are inherently unpredictable)

## Mixed Mode

Mixed mode tests whether a single model can learn all five tasks simultaneously. This is harder than learning any individual task because the tasks have different input distributions, different target structures, and different scoring regions.

### How it works

Each sample in a batch is independently assigned a random task. The batch is a heterogeneous mix — sample 0 might be delay, sample 1 might be parity, sample 5 might be induction, etc. The model sees the same input format (sequence of tokens from `[0, 64)`) for all tasks and must infer from context what kind of task each sample represents.

### Data and learning rate

Mixed mode uses 5x the number of training and validation batches so each subtask sees roughly the same total samples as standalone mode. Learning rate is 10x higher (`1e-3` vs `1e-4`) to compensate for the gradient dilution across disparate tasks.

### Hard mining

When more than half the subtasks have individually converged (reached the accuracy threshold), hard mining activates. Per-sample loss is computed, samples are ranked, and the hardest samples get ~2x weight while the easiest get ~0.5x (normalized to mean 1.0). This focuses training on the lagging subtasks without dropping the converged ones entirely. Hard mining turns off again if subtasks de-converge.

### Evaluation

Each subtask is evaluated independently. Per-task accuracy is computed by masking the batch to each task's samples. The overall mixed accuracy is the **unweighted mean** of all subtask accuracies — every task matters equally regardless of how many samples it got in a given batch.

### Convergence

Mixed mode converges when **all** subtask accuracies individually reach the threshold (`0.98` by default). A model that aces 4 tasks but can't crack the 5th hasn't converged. This is deliberate — the benchmark measures whether the model can handle disparate tasks, not just average performance.

## Why This Suite

Each task isolates a different capability:

| Task | Capability |
|---|---|
| delay | Basic 1-step memory |
| selective_copy | Content-addressable recall |
| parity | 1-bit state tracking |
| mod_arith | Multi-bit state tracking |
| induction | Pattern detection and recall |

A model that solves all five in mixed mode has demonstrated: short-term memory, selective attention, state maintenance, arithmetic generalization, and pattern matching — all with shared weights. The gap between standalone and mixed performance reveals how well an architecture handles task interference.
