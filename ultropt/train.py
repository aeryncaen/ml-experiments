"""
Training loops:
  1. baseline  — standard training with a single optimizer
  2. threetier — micro-batch / batch / super-batch with EMA gradient accumulation

Budget is measured in TOKENS SEEN so that baseline and three-tier are
compared on exactly the same amount of data.
"""

import time
import math
import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from model import ShakespeareGPT, ShakespeareNGPT, ngpt_normalize_weights
from data import get_datasets, make_loader


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BaseConfig:
    """Shared training config."""
    # model
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    # data
    val_frac: float = 0.1
    # training budget — total tokens (= samples * block_size) both loops
    # will consume.  This makes comparison fair regardless of batch size.
    total_tokens: int = 2_000_000       # ~2M tokens
    eval_every_tokens: int = 200_000    # eval checkpoint interval
    eval_samples: int = 10              # val batches to average
    # model type
    ngpt: bool = False
    # device
    device: str = "mps"


@dataclass
class BaselineConfig(BaseConfig):
    """Config for vanilla single-LR training."""
    batch_size: int = 16       # same as micro_batch_size for fair comparison
    lr: float = 3e-4
    weight_decay: float = 1e-1


@dataclass
class ThreeTierConfig(BaseConfig):
    """Config for the 3-tier hierarchical training.

    Micro level  : forward/backward only, NO step. Accumulate grads via EMA.
    Batch level  : every `micros_per_batch` micros, step with lr_batch (main LR).
                   This is the primary optimizer acting on denoised gradients.
    Super level  : every `batches_per_super` batches, step with lr_super (SMALL).
                   Slow, smooth correction from long-horizon signal.

    LR hierarchy : lr_batch > lr_super  (higher freq = noisier = accumulate only,
                   lower freq = smoother = act but gently)
    """
    # micro-batch — no step, just accumulate
    micro_batch_size: int = 16
    micros_per_batch: int = 4
    ema_decay_batch: float = 0.95
    # batch — primary optimizer step
    lr_batch: float = 3e-4
    batches_per_super: int = 4
    ema_decay_super: float = 0.95
    # super — slow correction
    lr_super: float = 1e-4
    # shared
    weight_decay: float = 1e-1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device(requested: str = "mps") -> str:
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@torch.no_grad()
def estimate_val_loss(model, val_loader, device, max_batches=10):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def build_model(vocab_size, cfg: BaseConfig):
    if cfg.ngpt:
        return ShakespeareNGPT(
            vocab_size=vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
        )
    return ShakespeareGPT(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )


# ---------------------------------------------------------------------------
# Baseline training
# ---------------------------------------------------------------------------

def train_baseline(cfg: BaselineConfig):
    """Standard training loop.  Returns list of log dicts and the model."""
    device = get_device(cfg.device)
    train_ds, val_ds, tokenizer = get_datasets(cfg.block_size, cfg.val_frac)
    train_loader = make_loader(train_ds, cfg.batch_size)
    val_loader = make_loader(val_ds, cfg.batch_size, shuffle=False)

    model = build_model(tokenizer.vocab_size, cfg).to(device)
    # nGPT: no weight decay (weights are norm-constrained post-step)
    wd = 0.0 if cfg.ngpt else cfg.weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=wd)

    tokens_per_step = cfg.batch_size * cfg.block_size

    log = []
    tokens_seen = 0
    next_eval_at = cfg.eval_every_tokens
    running_loss = 0.0
    steps_since_log = 0
    t0 = time.time()

    # Step-0 eval before any training
    model.train()
    val0 = estimate_val_loss(model, val_loader, device, cfg.eval_samples)
    # quick train loss from one batch
    x0, y0 = next(iter(train_loader))
    with torch.no_grad():
        _, loss0 = model(x0.to(device), y0.to(device))
    print(f"  [baseline] step 0     | train {loss0.item():.4f} | val {val0:.4f}")

    while tokens_seen < cfg.total_tokens:
        for x, y in train_loader:
            if tokens_seen >= cfg.total_tokens:
                break

            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if cfg.ngpt:
                ngpt_normalize_weights(model)

            running_loss += loss.item()
            steps_since_log += 1
            tokens_seen += tokens_per_step

            if tokens_seen >= next_eval_at or tokens_seen >= cfg.total_tokens:
                val_loss = estimate_val_loss(model, val_loader, device, cfg.eval_samples)
                avg_train = running_loss / max(steps_since_log, 1)
                elapsed = time.time() - t0
                log.append({
                    "tokens_seen": tokens_seen,
                    "train_loss": avg_train,
                    "val_loss": val_loss,
                    "elapsed": elapsed,
                })
                print(
                    f"  [baseline] {tokens_seen/1e6:>5.2f}M tok | "
                    f"train {avg_train:.4f} | val {val_loss:.4f} | "
                    f"{elapsed:.1f}s"
                )
                running_loss = 0.0
                steps_since_log = 0
                next_eval_at += cfg.eval_every_tokens

    return log, model, tokenizer


# ---------------------------------------------------------------------------
# Three-tier training
# ---------------------------------------------------------------------------

def _zero_grads_dict(model):
    """Return a dict of zero tensors shaped like each parameter."""
    return {n: torch.zeros_like(p) for n, p in model.named_parameters()}


def _ema_accumulate(accum, new_grads, decay):
    """In-place EMA:  accum = decay * accum + (1 - decay) * new_grads."""
    for n in accum:
        accum[n].mul_(decay).add_(new_grads[n], alpha=1.0 - decay)


def _bias_corrected(accum, decay, step_count):
    """Return bias-corrected view of EMA accumulator (like Adam's m_hat).

    correction = 1 / (1 - decay^step_count)
    This compensates for the zero-initialization of the EMA.
    """
    correction = 1.0 / (1.0 - decay ** step_count)
    return {n: v * correction for n, v in accum.items()}


def _extract_grads(model):
    """Snapshot current .grad into a dict."""
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


class TierState:
    """Per-tier Adam-like state: EMA of gradients (first moment) is handled
    externally by the accumulator.  This tracks only the second moment (v)
    for adaptive per-parameter LR scaling, plus decoupled weight decay.

    Update rule at step time:
        v = beta2 * v + (1 - beta2) * accum^2
        p -= lr * accum / (sqrt(v) + eps)      # adaptive step
        p *= (1 - lr * weight_decay)            # decoupled WD
    """

    def __init__(self, model, beta2=0.999, eps=1e-8):
        self.v = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
        self.beta2 = beta2
        self.eps = eps
        self.step_count = 0

    def step(self, model, accum, lr, weight_decay):
        self.step_count += 1
        with torch.no_grad():
            for n, p in model.named_parameters():
                if n not in accum:
                    continue
                g = accum[n]
                # update second moment
                self.v[n].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                # bias correction
                v_hat = self.v[n] / (1.0 - self.beta2 ** self.step_count)
                # adaptive step (first moment = the EMA accum itself)
                p.addcdiv_(g, v_hat.sqrt().add_(self.eps), value=-lr)
                # decoupled weight decay
                if weight_decay > 0:
                    p.mul_(1.0 - lr * weight_decay)


def _project_to_tangent(grads, model):
    """Project gradients onto tangent plane of the hypersphere.

    For each weight matrix W (rows are unit-norm), the tangent projection
    of gradient G is:  G_tan = G - (G · W) * W  (per row).

    For non-matrix params (scaling params, biases), leave grads unchanged.
    """
    tangent = {}
    for n, p in model.named_parameters():
        if n not in grads:
            continue
        g = grads[n]
        # Only project 2D weight matrices (Linear, Embedding)
        # that are on the hypersphere (row-normalized).
        # Skip nGPT scaling params (marked with _no_weight_decay).
        if g.dim() >= 2 and not getattr(p, '_no_weight_decay', False):
            # dot product of each row of g with corresponding row of p
            dots = (g * p.data).sum(dim=-1, keepdim=True)
            tangent[n] = g - dots * p.data
        else:
            tangent[n] = g
    return tangent


def _sgd_step(model, grad_dict, lr):
    """Plain SGD step: p -= lr * grad."""
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in grad_dict:
                p.add_(grad_dict[n], alpha=-lr)


def train_three_tier(cfg: ThreeTierConfig):
    """
    Three-tier hierarchical training.

    For standard GPT: per-tier adaptive scaling (TierState with second moments).
    For nGPT: Riemannian optimization on the hypersphere.
      - Project gradients onto tangent plane before accumulating
      - Plain SGD steps (no Adam — the sphere regularizes scale)
      - Renormalize after each step to stay on manifold
      - EMA provides momentum in tangent space

    Micro level : forward/backward, NO step.
                  EMA-accumulate (tangent-projected for nGPT) grads into batch_accum.
    Batch level : step from smoothed gradient signal.
    Super level : gentle correction from doubly-smoothed signal.
    """
    device = get_device(cfg.device)
    train_ds, val_ds, tokenizer = get_datasets(cfg.block_size, cfg.val_frac)
    train_loader = make_loader(train_ds, cfg.micro_batch_size)
    val_loader = make_loader(val_ds, cfg.micro_batch_size, shuffle=False)

    model = build_model(tokenizer.vocab_size, cfg).to(device)

    tokens_per_micro = cfg.micro_batch_size * cfg.block_size

    # EMA accumulators
    batch_accum = _zero_grads_dict(model)
    super_accum = _zero_grads_dict(model)

    # For standard GPT: per-tier adaptive state
    batch_state = None if cfg.ngpt else TierState(model)
    super_state = None if cfg.ngpt else TierState(model)

    # EMA step counters (for bias correction in nGPT Riemannian path)
    batch_ema_steps = 0
    super_ema_steps = 0

    log = []
    tokens_seen = 0
    next_eval_at = cfg.eval_every_tokens
    micro_in_batch = 0
    batch_in_super = 0
    running_loss = 0.0
    steps_since_log = 0
    t0 = time.time()

    # Step-0 eval before any training
    model.train()
    val0 = estimate_val_loss(model, val_loader, device, cfg.eval_samples)
    x0, y0 = next(iter(train_loader))
    with torch.no_grad():
        _, loss0 = model(x0.to(device), y0.to(device))
    print(f"  [3-tier]   step 0     | train {loss0.item():.4f} | val {val0:.4f}")

    while tokens_seen < cfg.total_tokens:
        for x, y in train_loader:
            if tokens_seen >= cfg.total_tokens:
                break

            x, y = x.to(device), y.to(device)

            # --- Micro: forward/backward only, no step ---
            model.zero_grad()
            _, loss = model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Snapshot grads
            micro_grads = _extract_grads(model)

            # nGPT: project to tangent plane before accumulating
            if cfg.ngpt:
                micro_grads = _project_to_tangent(micro_grads, model)

            _ema_accumulate(batch_accum, micro_grads, cfg.ema_decay_batch)
            batch_ema_steps += 1

            running_loss += loss.item()
            steps_since_log += 1
            tokens_seen += tokens_per_micro
            micro_in_batch += 1

            # --- Batch boundary ---
            if micro_in_batch >= cfg.micros_per_batch:
                if cfg.ngpt:
                    # Riemannian: bias-correct EMA, SGD in tangent space, renormalize
                    corrected = _bias_corrected(batch_accum, cfg.ema_decay_batch, batch_ema_steps)
                    _sgd_step(model, corrected, cfg.lr_batch)
                    ngpt_normalize_weights(model)
                else:
                    batch_state.step(model, batch_accum, cfg.lr_batch, cfg.weight_decay)

                # EMA accumulate batch signal into super accumulator
                _ema_accumulate(super_accum, batch_accum, cfg.ema_decay_super)
                super_ema_steps += 1

                batch_in_super += 1
                micro_in_batch = 0

                # --- Super boundary ---
                if batch_in_super >= cfg.batches_per_super:
                    if cfg.ngpt:
                        corrected = _bias_corrected(super_accum, cfg.ema_decay_super, super_ema_steps)
                        _sgd_step(model, corrected, cfg.lr_super)
                        ngpt_normalize_weights(model)
                    else:
                        super_state.step(model, super_accum, cfg.lr_super, cfg.weight_decay)
                    batch_in_super = 0

            # --- Logging ---
            if tokens_seen >= next_eval_at or tokens_seen >= cfg.total_tokens:
                val_loss = estimate_val_loss(model, val_loader, device, cfg.eval_samples)
                avg_train = running_loss / max(steps_since_log, 1)
                elapsed = time.time() - t0
                log.append({
                    "tokens_seen": tokens_seen,
                    "train_loss": avg_train,
                    "val_loss": val_loss,
                    "elapsed": elapsed,
                })
                print(
                    f"  [3-tier]   {tokens_seen/1e6:>5.2f}M tok | "
                    f"train {avg_train:.4f} | val {val_loss:.4f} | "
                    f"{elapsed:.1f}s"
                )
                running_loss = 0.0
                steps_since_log = 0
                next_eval_at += cfg.eval_every_tokens

    return log, model, tokenizer
