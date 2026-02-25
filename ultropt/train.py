"""
Training loops:
  1. baseline   — standard training with a single optimizer (AdamW)
  2. threetier  — legacy 3-tier with EMA gradient accumulation (pre-UltrOpt)
  3. ultropt    — UltrOpt multi-timescale structure-preserving optimizer

Budget is measured in TOKENS SEEN so that baseline and UltrOpt are
compared on exactly the same amount of data.

Data: yamit tokenized shards (.bin/.idx).
Tokenizer metadata loaded from artifact_meta.json at runtime.
"""

import os
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import ShakespeareGPT, ShakespeareNGPT, ngpt_normalize_weights
from data import load_tokenizer_meta, ShardStream
from optimizer import UltrOpt, UltrOptConfig
from slerp import GhostSLERPHooks


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
    batch_size: int = 64
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
    micro_batch_size: int = 64
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


# ---------------------------------------------------------------------------
# UltrOpt training config
# ---------------------------------------------------------------------------

@dataclass
class UltrOptTrainConfig(BaseConfig):
    """Training config for UltrOpt.

    Combines BaseConfig (model/data/budget) with UltrOptConfig (optimizer).
    """
    batch_size: int = 64                      # = micro-batch size

    # --- UltrOpt optimizer hyperparameters ---
    # Learning rates
    lr: float = 3e-3
    batch_lr_factor: float = 0.1
    super_lr_factor: float = 0.01

    # Tier cadence
    micros_per_batch: int = 4
    batches_per_super: int = 4

    # EMA decays
    ema_decay_batch: float = 0.95
    ema_decay_super: float = 0.95

    # Low-rank accumulator ranks
    rank_batch: int = 32
    rank_super: int = 16
    error_feedback: bool = False

    # Channel-wise adaptive scaling
    channel_beta2: float = 0.999
    channel_eps: float = 1e-8

    # MARS variance reduction
    mars: bool = True
    mars_gamma: float = 0.025
    mars_beta1: float = 0.95
    mars_clip: Optional[float] = None

    # Newton-Schulz (micro and batch tiers only; super tier is rank-deficient)
    newton_schulz: bool = True
    ns_steps_micro: int = 4
    ns_steps_batch: int = 6
    ns_delta: float = 0.3

    # Cautious masking
    cautious: bool = True

    # Schedule-Free
    schedule_free: bool = True
    schedule_free_beta: float = 0.95
    schedule_free_scope: str = 'all'

    # Weight decay
    weight_decay: float = 0.1

    # Warmup
    warmup_steps: int = 100

    # Gradient clipping
    grad_clip: Optional[float] = None

    # SLERP gradient reduction
    slerp_mode: str = 'ghost_slerp'
    slerp_magnitude: str = 'mean'

    # Accumulate signal
    accumulate_signal: str = 'raw'

    def to_ultropt_config(self) -> UltrOptConfig:
        """Build UltrOptConfig from training config fields."""
        return UltrOptConfig(
            lr=self.lr,
            batch_lr_factor=self.batch_lr_factor,
            super_lr_factor=self.super_lr_factor,
            micros_per_batch=self.micros_per_batch,
            batches_per_super=self.batches_per_super,
            ema_decay_batch=self.ema_decay_batch,
            ema_decay_super=self.ema_decay_super,
            rank_batch=self.rank_batch,
            rank_super=self.rank_super,
            error_feedback=self.error_feedback,
            channel_beta2=self.channel_beta2,
            channel_eps=self.channel_eps,
            mars=self.mars,
            mars_gamma=self.mars_gamma,
            mars_beta1=self.mars_beta1,
            mars_clip=self.mars_clip,
            newton_schulz=self.newton_schulz,
            ns_steps_micro=self.ns_steps_micro,
            ns_steps_batch=self.ns_steps_batch,
            ns_delta=self.ns_delta,
            cautious=self.cautious,
            schedule_free=self.schedule_free,
            schedule_free_beta=self.schedule_free_beta,
            schedule_free_scope=self.schedule_free_scope,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            grad_clip=self.grad_clip,
            accumulate_signal=self.accumulate_signal,
            ngpt=self.ngpt,
        )


# ---------------------------------------------------------------------------
# UltrOpt training loop
# ---------------------------------------------------------------------------

def train_ultropt(cfg: UltrOptTrainConfig):
    """UltrOpt training loop.

    Simple structure:
      1. Forward pass
      2. loss.backward()
      3. slerp_hooks.apply(model)  — rewrite .grad with direction-preserving reduction
      4. opt.micro_step()          — handles all three tiers internally

    For eval: opt.eval_mode() before val, opt.train_mode() after.

    Returns (log, model, tok_meta).
    """
    device = _resolve_device(cfg.device)
    tok_meta = load_tokenizer_meta(cfg.tokenizer_dir)
    vocab_size = tok_meta["vocab_size"]
    eos_token_id = tok_meta["eos_token_id"]

    train_stream, val_stream = _make_streams(cfg, cfg.batch_size, eos_token_id)

    model = build_model(vocab_size, cfg).to(device)
    model = _maybe_compile(model, device, cfg.compile)

    # Build optimizer
    opt_cfg = cfg.to_ultropt_config()
    opt = UltrOpt(model, opt_cfg)

    # Build SLERP hooks
    slerp_hooks = GhostSLERPHooks(model, mode=cfg.slerp_mode, magnitude=cfg.slerp_magnitude)

    tokens_per_micro = cfg.batch_size * cfg.seq_len

    log = []
    tokens_seen = 0
    next_eval_at = cfg.eval_every_tokens
    running_loss = 0.0
    steps_since_log = 0
    t0 = time.time()

    # Memory report
    mem = opt.memory_report()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  [ultropt] params: {param_count:,} | opt state: {mem['total_elements']:,} "
          f"({mem['ratio_to_params']:.3f}x params)")

    # Step-0 eval before any training
    model.train()
    opt.eval_mode()
    val0 = estimate_val_loss(model, val_stream, device, cfg.eval_batches)
    opt.train_mode()
    x0, y0 = train_stream.next_batch(torch.device(device))
    with torch.no_grad():
        _, loss0 = model(x0, y0)
    train_stream.pos -= train_stream.tokens_per_global_step
    print(f"  [ultropt] step 0     | train {loss0.item():.4f} | val {val0:.4f}")

    while tokens_seen < cfg.total_tokens:
        x, y = train_stream.next_batch(torch.device(device))

        # Forward + backward
        _, loss = model(x, y)
        opt.zero_grad()
        loss.backward()

        # SLERP: rewrite .grad with direction-preserving reduction
        slerp_hooks.apply(model)

        # Optimizer step (handles all three tiers)
        opt.micro_step()

        running_loss += loss.item()
        steps_since_log += 1
        tokens_seen += tokens_per_micro

        # Logging
        if tokens_seen >= next_eval_at or tokens_seen >= cfg.total_tokens:
            opt.eval_mode()
            val_loss = estimate_val_loss(model, val_stream, device, cfg.eval_batches)
            opt.train_mode()
            avg_train = running_loss / max(steps_since_log, 1)
            elapsed = time.time() - t0
            tok_per_sec = tokens_seen / max(1, elapsed)
            counts = opt.tier_counts()
            log.append({
                "tokens_seen": tokens_seen,
                "train_loss": avg_train,
                "val_loss": val_loss,
                "elapsed": elapsed,
                "micro_steps": counts['micro'],
                "batch_steps": counts['batch'],
                "super_steps": counts['super'],
            })
            print(
                f"  [ultropt] {tokens_seen/1e6:>5.2f}M tok | "
                f"train {avg_train:.4f} | val {val_loss:.4f} | "
                f"{elapsed:.1f}s | {tok_per_sec:,.0f} tok/s | "
                f"tiers: {counts['micro']}/{counts['batch']}/{counts['super']}"
            )
            running_loss = 0.0
            steps_since_log = 0
            next_eval_at += cfg.eval_every_tokens

    # Cleanup hooks
    slerp_hooks.remove()

    return log, model, tok_meta
