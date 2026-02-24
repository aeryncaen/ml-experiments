"""
Training loops:
  1. baseline  — standard training with a single optimizer
  2. threetier — micro-batch / batch / super-batch with EMA gradient accumulation

Budget is measured in TOKENS SEEN so that baseline and three-tier are
compared on exactly the same amount of data.

Data: yamit tokenized shards (.bin/.idx).
Tokenizer metadata loaded from artifact_meta.json at runtime.
"""

import os
import time
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ShakespeareGPT, ShakespeareNGPT, ngpt_normalize_weights
from data import load_tokenizer_meta, ShardStream


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class BaseConfig:
    """Shared training config."""
    # data
    data_dir: str = "../tokenized"          # directory with train/ and val/ subdirs
    tokenizer_dir: str = "../experiments/yamit/tokenizer/artifacts/yamit"
    seq_len: int = 256                      # sequence length per sample
    # model
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.1
    # training budget — total tokens both loops will consume
    total_tokens: int = 20_000_000          # 20M tokens
    eval_every_tokens: int = 2_000_000      # eval checkpoint interval
    eval_batches: int = 10                  # val batches to average
    # model type
    ngpt: bool = False
    # device
    device: str = "auto"
    # torch.compile
    compile: bool = True


@dataclass
class BaselineConfig(BaseConfig):
    """Config for vanilla single-LR training."""
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-1


@dataclass
class ThreeTierConfig(BaseConfig):
    """Config for the 3-tier hierarchical training.

    Micro level  : forward/backward only, NO step. Accumulate grads via EMA.
    Batch level  : every `micros_per_batch` micros, step with lr_batch (main LR).
    Super level  : every `batches_per_super` batches, step with lr_super (SMALL).
    """
    # micro-batch — no step, just accumulate
    micro_batch_size: int = 256
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

def _resolve_device(requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested


def _maybe_compile(model, device: str, do_compile: bool):
    """Compile model if on CUDA and compilation requested."""
    if do_compile and device == "cuda":
        return torch.compile(model)
    return model


@torch.no_grad()
def estimate_val_loss(model, val_stream: ShardStream, device, max_batches=10):
    model.eval()
    losses = []
    val_stream.reset()
    for i in range(max_batches):
        x, y = val_stream.next_batch(torch.device(device))
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses) if losses else float("nan")


def build_model(vocab_size: int, cfg: BaseConfig):
    if cfg.ngpt:
        return ShakespeareNGPT(
            vocab_size=vocab_size,
            block_size=cfg.seq_len,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
        )
    return ShakespeareGPT(
        vocab_size=vocab_size,
        block_size=cfg.seq_len,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
    )


def _make_streams(cfg, batch_size: int, eos_token_id: int):
    """Create train and val ShardStreams."""
    train_pattern = os.path.join(cfg.data_dir, "train", "**", "*.bin")
    val_pattern = os.path.join(cfg.data_dir, "val", "**", "*.bin")
    train_stream = ShardStream(
        pattern=train_pattern,
        seq_len=cfg.seq_len,
        batch_size=batch_size,
        eos_token_id=eos_token_id,
    )
    val_stream = ShardStream(
        pattern=val_pattern,
        seq_len=cfg.seq_len,
        batch_size=batch_size,
        eos_token_id=eos_token_id,
    )
    return train_stream, val_stream


# ---------------------------------------------------------------------------
# Baseline training
# ---------------------------------------------------------------------------

def train_baseline(cfg: BaselineConfig):
    """Standard training loop.  Returns (log, model, tok_meta)."""
    device = _resolve_device(cfg.device)
    tok_meta = load_tokenizer_meta(cfg.tokenizer_dir)
    vocab_size = tok_meta["vocab_size"]
    eos_token_id = tok_meta["eos_token_id"]

    train_stream, val_stream = _make_streams(cfg, cfg.batch_size, eos_token_id)

    model = build_model(vocab_size, cfg).to(device)
    model = _maybe_compile(model, device, cfg.compile)

    # nGPT: no weight decay (weights are norm-constrained post-step)
    wd = 0.0 if cfg.ngpt else cfg.weight_decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=wd)

    tokens_per_step = cfg.batch_size * cfg.seq_len

    log = []
    tokens_seen = 0
    next_eval_at = cfg.eval_every_tokens
    running_loss = 0.0
    steps_since_log = 0
    t0 = time.time()

    # Step-0 eval before any training
    model.train()
    val0 = estimate_val_loss(model, val_stream, device, cfg.eval_batches)
    x0, y0 = train_stream.next_batch(torch.device(device))
    with torch.no_grad():
        _, loss0 = model(x0, y0)
    # undo the token consumption from the peek batch
    train_stream.pos -= train_stream.tokens_per_global_step
    print(f"  [baseline] step 0     | train {loss0.item():.4f} | val {val0:.4f}")

    while tokens_seen < cfg.total_tokens:
        x, y = train_stream.next_batch(torch.device(device))

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
            val_loss = estimate_val_loss(model, val_stream, device, cfg.eval_batches)
            avg_train = running_loss / max(steps_since_log, 1)
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / max(1, elapsed)
            log.append({
                "tokens_seen": tokens_seen,
                "train_loss": avg_train,
                "val_loss": val_loss,
                "elapsed": elapsed,
            })
            print(
                f"  [baseline] {tokens_seen/1e6:>5.2f}M tok | "
                f"train {avg_train:.4f} | val {val_loss:.4f} | "
                f"{elapsed:.1f}s | {tok_per_sec:,.0f} tok/s"
            )
            running_loss = 0.0
            steps_since_log = 0
            next_eval_at += cfg.eval_every_tokens

    return log, model, tok_meta


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
    """Return bias-corrected view of EMA accumulator (like Adam's m_hat)."""
    correction = 1.0 / (1.0 - decay ** step_count)
    return {n: v * correction for n, v in accum.items()}


def _extract_grads(model):
    """Snapshot current .grad into a dict."""
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


class TierState:
    """Per-tier Adam-like state: tracks second moment (v) for adaptive scaling.

    Update rule at step time:
        v = beta2 * v + (1 - beta2) * accum^2
        p -= lr * accum / (sqrt(v) + eps)
        p *= (1 - lr * weight_decay)
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
                self.v[n].mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)
                v_hat = self.v[n] / (1.0 - self.beta2 ** self.step_count)
                p.addcdiv_(g, v_hat.sqrt().add_(self.eps), value=-lr)
                if weight_decay > 0:
                    p.mul_(1.0 - lr * weight_decay)


def _project_to_tangent(grads, model):
    """Project gradients onto tangent plane of the hypersphere."""
    tangent = {}
    for n, p in model.named_parameters():
        if n not in grads:
            continue
        g = grads[n]
        if g.dim() >= 2 and not getattr(p, '_no_weight_decay', False):
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
    """Three-tier hierarchical training.  Returns (log, model, tok_meta)."""
    device = _resolve_device(cfg.device)
    tok_meta = load_tokenizer_meta(cfg.tokenizer_dir)
    vocab_size = tok_meta["vocab_size"]
    eos_token_id = tok_meta["eos_token_id"]

    train_stream, val_stream = _make_streams(cfg, cfg.micro_batch_size, eos_token_id)

    model = build_model(vocab_size, cfg).to(device)
    model = _maybe_compile(model, device, cfg.compile)

    tokens_per_micro = cfg.micro_batch_size * cfg.seq_len

    # EMA accumulators
    batch_accum = _zero_grads_dict(model)
    super_accum = _zero_grads_dict(model)

    # Per-tier adaptive state (standard GPT) or None (nGPT uses SGD)
    batch_state = None if cfg.ngpt else TierState(model)
    super_state = None if cfg.ngpt else TierState(model)

    # EMA step counters (for bias correction)
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

    # Step-0 eval
    model.train()
    val0 = estimate_val_loss(model, val_stream, device, cfg.eval_batches)
    x0, y0 = train_stream.next_batch(torch.device(device))
    with torch.no_grad():
        _, loss0 = model(x0, y0)
    train_stream.pos -= train_stream.tokens_per_global_step
    print(f"  [3-tier]   step 0     | train {loss0.item():.4f} | val {val0:.4f}")

    while tokens_seen < cfg.total_tokens:
        x, y = train_stream.next_batch(torch.device(device))

        # --- Micro: forward/backward only, no step ---
        model.zero_grad()
        _, loss = model(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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
                corrected = _bias_corrected(batch_accum, cfg.ema_decay_batch, batch_ema_steps)
                _sgd_step(model, corrected, cfg.lr_batch)
                ngpt_normalize_weights(model)
            else:
                batch_state.step(model, batch_accum, cfg.lr_batch, cfg.weight_decay)

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
            val_loss = estimate_val_loss(model, val_stream, device, cfg.eval_batches)
            avg_train = running_loss / max(steps_since_log, 1)
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / max(1, elapsed)
            log.append({
                "tokens_seen": tokens_seen,
                "train_loss": avg_train,
                "val_loss": val_loss,
                "elapsed": elapsed,
            })
            print(
                f"  [3-tier]   {tokens_seen/1e6:>5.2f}M tok | "
                f"train {avg_train:.4f} | val {val_loss:.4f} | "
                f"{elapsed:.1f}s | {tok_per_sec:,.0f} tok/s"
            )
            running_loss = 0.0
            steps_since_log = 0
            next_eval_at += cfg.eval_every_tokens

    return log, model, tok_meta
