#!/usr/bin/env python3
"""
Fully-parallelized Shakespeare geo-bias hyperparameter search.

Runs N configs simultaneously on one GPU by stacking model weights
into (N, ...) tensors and doing batched forward/backward/eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import urllib.request
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Use flash attention backend if available
try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)  # prefer flash/mem-efficient over math fallback
except Exception:
    pass


# ---------------------------------------------------------------------------
# Data helpers (same as train script)
# ---------------------------------------------------------------------------
TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def ensure_data(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=30) as r:
        txt = r.read().decode("utf-8")
    path.write_text(txt, encoding="utf-8")
    return txt


def build_vocab(text: str):
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    return np.array([stoi[c] for c in text], dtype=np.int64)


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: torch.device):
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = np.stack([data[i : i + block_size] for i in ix])
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# ---------------------------------------------------------------------------
# Geo basis builders (reused from train script logic)
# ---------------------------------------------------------------------------

def compute_bucket_basis(tokens: np.ndarray, vocab_size: int, width: int) -> np.ndarray:
    if len(tokens) < 2:
        return np.zeros((vocab_size, width), dtype=np.float32)
    basis = np.zeros((vocab_size, width), dtype=np.float64)
    cur = tokens[2:]
    prev1 = tokens[1:-1]
    prev2 = tokens[:-2]
    context_id = prev1.astype(np.int64) + vocab_size * prev2.astype(np.int64)
    buckets = context_id % width
    np.add.at(basis, (cur, buckets), 1.0)
    col_sum = basis.sum(axis=0, keepdims=True)
    basis = basis / np.maximum(col_sum, 1.0)
    basis = basis - basis.mean(axis=0, keepdims=True)
    col_norm = np.linalg.norm(basis, axis=0, keepdims=True)
    basis = basis / np.maximum(col_norm, 1e-12)
    return basis.astype(np.float32)


def compute_kl_bucket_basis(tokens: np.ndarray, vocab_size: int, width: int, eps: float = 1e-8) -> np.ndarray:
    if len(tokens) < 2:
        return np.zeros((vocab_size, width), dtype=np.float32)
    counts = np.zeros((vocab_size, width), dtype=np.float64)
    cur = tokens[2:]
    prev1 = tokens[1:-1]
    prev2 = tokens[:-2]
    context_id = prev1.astype(np.int64) + vocab_size * prev2.astype(np.int64)
    buckets = context_id % width
    np.add.at(counts, (cur, buckets), 1.0)
    col_sum = counts.sum(axis=0, keepdims=True)
    p_t_given_b = counts / np.maximum(col_sum, 1.0)
    token_sum = counts.sum(axis=1, keepdims=True)
    p_t = token_sum / np.maximum(np.sum(counts), 1.0)
    basis = np.log(p_t_given_b + eps) - np.log(p_t + eps)
    basis = basis - basis.mean(axis=0, keepdims=True)
    col_norm = np.linalg.norm(basis, axis=0, keepdims=True)
    basis = basis / np.maximum(col_norm, 1e-12)
    return basis.astype(np.float32)


def compute_kl_bucket_mtp_basis(
    tokens: np.ndarray, vocab_size: int, width: int, mtp_weights: list[float], eps: float = 1e-8,
) -> np.ndarray:
    if len(tokens) < 3:
        return np.zeros((vocab_size, width), dtype=np.float32)
    counts = np.zeros((vocab_size, width), dtype=np.float64)
    for h, w in enumerate(mtp_weights, start=1):
        if w == 0.0 or len(tokens) - h - 1 <= 0:
            continue
        cur = tokens[h + 1 :]
        prev1 = tokens[1:-h]
        prev2 = tokens[: -(h + 1)]
        context_id = prev1.astype(np.int64) + vocab_size * prev2.astype(np.int64)
        buckets = context_id % width
        np.add.at(counts, (cur, buckets), float(w))
    col_sum = counts.sum(axis=0, keepdims=True)
    p_t_given_b = counts / np.maximum(col_sum, 1.0)
    token_sum = counts.sum(axis=1, keepdims=True)
    p_t = token_sum / np.maximum(np.sum(counts), 1.0)
    basis = np.log(p_t_given_b + eps) - np.log(p_t + eps)
    basis = np.tanh(0.5 * basis)
    basis = basis - basis.mean(axis=0, keepdims=True)
    col_norm = np.linalg.norm(basis, axis=0, keepdims=True)
    basis = basis / np.maximum(col_norm, 1e-12)
    return basis.astype(np.float32)


def compute_weighted_shift_operator(
    tokens: np.ndarray, vocab_size: int, horizons: list[int], horizon_weights: list[float],
) -> np.ndarray:
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    for h, w in zip(horizons, horizon_weights):
        if h <= 0 or w == 0.0 or len(tokens) <= h:
            continue
        a = tokens[:-h]
        b = tokens[h:]
        np.add.at(counts, (a, b), w)
    sym = 0.5 * (counts + counts.T)
    total = np.sum(sym)
    if total > 0:
        sym = sym / total
    return sym


def compute_attn_corr_projector(
    tokens: np.ndarray, vocab_size: int, geo_basis: np.ndarray,
    d_model: int, rank: int, horizons: list[int], horizon_weights: list[float], eps: float = 1e-8,
) -> np.ndarray:
    op = compute_weighted_shift_operator(tokens, vocab_size, horizons, horizon_weights)
    p_tok = np.sum(op, axis=1, keepdims=True)
    denom = p_tok @ p_tok.T
    pmi = np.log(op + eps) - np.log(denom + eps)
    pmi = 0.5 * (pmi + pmi.T)
    B = np.zeros((vocab_size, d_model), dtype=np.float64)
    k = min(d_model, geo_basis.shape[1])
    B[:, :k] = geo_basis[:, :k].astype(np.float64)
    C = B.T @ pmi @ B
    C = 0.5 * (C + C.T)
    _, eigvecs = np.linalg.eigh(C)
    k_eff = max(1, min(rank, d_model))
    U = eigvecs[:, -k_eff:]
    P = U @ U.T
    return P.astype(np.float32)


def compute_model_projector(geo_basis: np.ndarray, d_model: int, k: int) -> np.ndarray:
    B = geo_basis[:, :d_model].astype(np.float64)
    cov = B.T @ B
    cov = 0.5 * (cov + cov.T)
    _, eigvecs = np.linalg.eigh(cov)
    k_eff = max(1, min(k, d_model))
    U = eigvecs[:, -k_eff:]
    P = U @ U.T
    return P.astype(np.float32)


def compute_embed_projector_from_target(target: np.ndarray, rank: int) -> np.ndarray:
    C = target.astype(np.float64).T @ target.astype(np.float64)
    C = 0.5 * (C + C.T)
    _, eigvecs = np.linalg.eigh(C)
    d = C.shape[0]
    k_eff = max(1, min(rank, d))
    U = eigvecs[:, -k_eff:]
    P = U @ U.T
    return P.astype(np.float32)


def parse_float_csv(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_int_csv(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


# ---------------------------------------------------------------------------
# Config for one parallel trial
# ---------------------------------------------------------------------------
@dataclass
class TrialConfig:
    geo_init_method: str = "kl_bucket"
    geo_init_mtp_weights: str = "1.0,0.5,0.25"
    geo_init_blend: float = 0.8
    geo_init_fullspace: bool = False
    geo_init_ridge: float = 1e-3
    geo_init_match_row_norm: bool = True
    geo_attn_bias: bool = False
    geo_attn_bias_blend: float = 0.2
    geo_attn_corr_bias: bool = False
    geo_attn_corr_blend: float = 0.1
    geo_attn_corr_rank: int = 8
    geo_attn_corr_layers: int = 2
    geo_attn_corr_horizons: str = "1,2,3"
    geo_attn_corr_horizon_weights: str = "1.0,0.5,0.25"
    geo_embed_grad_shape: bool = True
    geo_embed_grad_rank: int = 8
    geo_embed_grad_perp_init: float = 0.2
    geo_embed_grad_hold_steps: int = 100
    geo_embed_grad_ramp_steps: int = 250
    geo_embed_reanchor_every: int = 50
    geo_embed_reanchor_rho: float = 0.01
    geo_embed_reanchor_until_step: int = 0


def trial_id(cfg: TrialConfig) -> str:
    import dataclasses
    s = json.dumps(dataclasses.asdict(cfg), sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------
# Batched model: N copies of TinyGPT stacked into single tensors
# ---------------------------------------------------------------------------
class BatchedModels:
    """
    Manages N independent TinyGPT parameter sets as stacked tensors.
    All forward/backward/step ops are batched across N configs.
    """

    def __init__(
        self,
        n_configs: int,
        vocab_size: int,
        d_model: int,
        n_head: int,
        n_layer: int,
        block_size: int,
        dropout: float,
        lr: float,
        device: torch.device,
        seed: int,
        dtype: torch.dtype | None = None,
    ):
        self.N = n_configs
        self.V = vocab_size
        self.D = d_model
        self.H = n_head
        self.L = n_layer
        self.T = block_size
        self.device = device
        self.lr = lr
        self.dropout = dropout

        # Pick dtype: prefer bf16 on cuda for flash attention, else float32
        if dtype is not None:
            self.dtype = dtype
        elif device.type == "cuda" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32
        print(f"  BatchedModels dtype={self.dtype}", flush=True)

        # We'll create one reference model to get the architecture right,
        # then clone its state_dict N times into stacked tensors.
        torch.manual_seed(seed)
        ref = self._make_ref()
        ref_sd = ref.state_dict()

        # Build stacked parameters: dict[name -> (N, *shape)]
        self.params: dict[str, torch.Tensor] = {}
        self.param_names: list[str] = []
        for name, p in ref_sd.items():
            if name == "lm_head.weight":
                continue  # tied
            stacked = p.unsqueeze(0).expand(n_configs, *p.shape).clone().contiguous()
            stacked = stacked.to(device=device, dtype=self.dtype)
            stacked.requires_grad_(True)
            self.params[name] = stacked
            self.param_names.append(name)

        # Adam state
        self.m: dict[str, torch.Tensor] = {n: torch.zeros_like(self.params[n]) for n in self.param_names}
        self.v: dict[str, torch.Tensor] = {n: torch.zeros_like(self.params[n]) for n in self.param_names}
        self.step_count = 0

        # Causal mask
        self.causal_mask = torch.triu(torch.ones(block_size, block_size, device=device, dtype=torch.bool), diagonal=1)

        # Compile the forward pass
        self._compiled_forward = None
        if hasattr(torch, "compile"):
            try:
                self._compiled_forward = torch.compile(self._raw_forward, fullgraph=False)
                print(f"  torch.compile enabled for batched forward", flush=True)
            except Exception as e:
                print(f"  torch.compile failed, using eager: {e}", flush=True)

        del ref

    def _make_ref(self):
        import torch.nn as nn

        class _Block(nn.Module):
            def __init__(self, d, h, drop):
                super().__init__()
                self.ln1 = nn.LayerNorm(d)
                self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
                self.ln2 = nn.LayerNorm(d)
                self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop))

            def forward(self, x, mask):
                y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
                x = x + y
                x = x + self.mlp(self.ln2(x))
                return x

        class _GPT(nn.Module):
            def __init__(self2):
                super().__init__()
                self2.token_emb = nn.Embedding(self.V, self.D)
                self2.pos_emb = nn.Parameter(torch.zeros(1, self.T, self.D))
                self2.blocks = nn.ModuleList([_Block(self.D, self.H, self.dropout) for _ in range(self.L)])
                self2.ln_f = nn.LayerNorm(self.D)
                self2.lm_head = nn.Linear(self.D, self.V, bias=False)
                self2.lm_head.weight = self2.token_emb.weight

        return _GPT()

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        if self._compiled_forward is not None:
            return self._compiled_forward(idx)
        return self._raw_forward(idx)

    def _raw_forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, T) shared input batch
        Returns: (N, B, T, V) logits for each config
        """
        N, B, T, D, V = self.N, idx.shape[0], idx.shape[1], self.D, self.V

        # Embedding: token_emb.weight is (N, V, D)
        emb_w = self.params["token_emb.weight"]  # (N, V, D)
        tok_emb = emb_w[:, idx, :]  # (N, B, T, D)
        pos_emb = self.params["pos_emb"][:, :, :T, :]  # (N, 1, T, D)
        x = tok_emb + pos_emb  # (N, B, T, D)

        mask = self.causal_mask[:T, :T]

        for li in range(self.L):
            pf = f"blocks.{li}."
            # LayerNorm 1
            ln1_w = self.params[pf + "ln1.weight"]  # (N, D)
            ln1_b = self.params[pf + "ln1.bias"]  # (N, D)
            x_ln1 = self._batched_layernorm(x, ln1_w, ln1_b)

            # Self-attention (manual for batched weights)
            in_proj_w = self.params[pf + "attn.in_proj_weight"]  # (N, 3D, D)
            in_proj_b = self.params[pf + "attn.in_proj_bias"]  # (N, 3D)
            out_proj_w = self.params[pf + "attn.out_proj.weight"]  # (N, D, D)
            out_proj_b = self.params[pf + "attn.out_proj.bias"]  # (N, D)

            attn_out = self._batched_mha(x_ln1, in_proj_w, in_proj_b, out_proj_w, out_proj_b, mask)
            x = x + attn_out

            # LayerNorm 2
            ln2_w = self.params[pf + "ln2.weight"]
            ln2_b = self.params[pf + "ln2.bias"]
            x_ln2 = self._batched_layernorm(x, ln2_w, ln2_b)

            # MLP
            mlp0_w = self.params[pf + "mlp.0.weight"]  # (N, 4D, D)
            mlp0_b = self.params[pf + "mlp.0.bias"]  # (N, 4D)
            mlp2_w = self.params[pf + "mlp.2.weight"]  # (N, D, 4D)
            mlp2_b = self.params[pf + "mlp.2.bias"]  # (N, D)

            h = torch.einsum("nbti,noi->nbto", x_ln2, mlp0_w) + mlp0_b[:, None, None, :]
            h = F.gelu(h)
            h = torch.einsum("nbti,noi->nbto", h, mlp2_w) + mlp2_b[:, None, None, :]
            x = x + h

        # Final layernorm
        lnf_w = self.params["ln_f.weight"]
        lnf_b = self.params["ln_f.bias"]
        x = self._batched_layernorm(x, lnf_w, lnf_b)

        # LM head (tied with token_emb)
        logits = torch.einsum("nbti,nvi->nbtv", x, emb_w)  # (N, B, T, V)
        return logits

    def _batched_layernorm(self, x: torch.Tensor, w: torch.Tensor, b: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
        # x: (N, B, T, D), w: (N, D), b: (N, D)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + eps)
        return x_norm * w[:, None, None, :] + b[:, None, None, :]

    def _batched_mha(
        self,
        x: torch.Tensor,
        in_proj_w: torch.Tensor,
        in_proj_b: torch.Tensor,
        out_proj_w: torch.Tensor,
        out_proj_b: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        N, B, T, D = x.shape
        H = self.H
        head_dim = D // H

        # QKV projection: (N, B, T, 3D)
        qkv = torch.einsum("nbti,noi->nbto", x, in_proj_w) + in_proj_b[:, None, None, :]
        q, k, v = qkv.split(D, dim=-1)  # each (N, B, T, D)

        # Reshape to heads and merge N*B for SDPA: (N*B, H, T, head_dim)
        q = q.view(N * B, T, H, head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.view(N * B, T, H, head_dim).permute(0, 2, 1, 3).contiguous()
        v = v.view(N * B, T, H, head_dim).permute(0, 2, 1, 3).contiguous()

        # SDPA with causal mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (N*B, H, T, head_dim)

        # Reshape back: (N, B, T, D)
        out = out.permute(0, 2, 1, 3).reshape(N, B, T, D)

        # Output projection
        out = torch.einsum("nbti,noi->nbto", out, out_proj_w) + out_proj_b[:, None, None, :]
        return out

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits: (N, B, T, V), targets: (B, T)
        Returns: (N,) per-config mean CE loss
        """
        N, B, T, V = logits.shape
        # Reshape: (N, B*T, V) and (B*T,) broadcast
        logits_flat = logits.reshape(N, B * T, V)
        targets_flat = targets.reshape(B * T)  # shared across configs

        # Log softmax + nll per config
        log_probs = F.log_softmax(logits_flat, dim=-1)  # (N, B*T, V)
        # Gather target log probs
        target_log_probs = log_probs[:, torch.arange(B * T), targets_flat]  # (N, B*T)
        loss = -target_log_probs.mean(dim=1)  # (N,)
        return loss

    def compute_acc(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Returns (N,) per-config accuracy."""
        N, B, T, V = logits.shape
        preds = logits.argmax(dim=-1)  # (N, B, T)
        correct = (preds == targets[None, :, :]).float()
        return correct.mean(dim=(1, 2))  # (N,)

    def zero_grad(self):
        for name in self.param_names:
            if self.params[name].grad is not None:
                self.params[name].grad.zero_()

    def adam_step(self):
        self.step_count += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        bc1 = 1.0 - beta1 ** self.step_count
        bc2 = 1.0 - beta2 ** self.step_count

        with torch.no_grad():
            for name in self.param_names:
                p = self.params[name]
                if p.grad is None:
                    continue
                g = p.grad
                self.m[name].mul_(beta1).add_(g, alpha=1 - beta1)
                self.v[name].mul_(beta2).addcmul_(g, g, value=1 - beta2)
                m_hat = self.m[name] / bc1
                v_hat = self.v[name] / bc2
                p.add_(m_hat / (v_hat.sqrt() + eps), alpha=-self.lr)

    def apply_embed_geo_bias(self, config_idx: int, geo_target: torch.Tensor, blend: float):
        """Blend geo target into embedding for one config."""
        with torch.no_grad():
            E = self.params["token_emb.weight"][config_idx]
            E.copy_((1.0 - blend) * E + blend * geo_target)

    def apply_attn_bias(self, config_idx: int, P: torch.Tensor, blend: float, n_layers: int):
        """Blend attention projector into Q/K/V/O for one config."""
        b = blend
        with torch.no_grad():
            for li in range(min(n_layers, self.L)):
                pf = f"blocks.{li}."
                W = self.params[pf + "attn.in_proj_weight"][config_idx]
                D = self.D
                for s0, s1 in ((0, D), (D, 2 * D), (2 * D, 3 * D)):
                    W[s0:s1, :].copy_((1.0 - b) * W[s0:s1, :] + b * (W[s0:s1, :] @ P))
                Wo = self.params[pf + "attn.out_proj.weight"][config_idx]
                Wo.copy_((1.0 - b) * Wo + b * (Wo @ P))

    def apply_attn_corr_bias(self, config_idx: int, P_corr: torch.Tensor, blend: float, n_layers: int):
        """Blend attention correlation projector into Q/K only for one config."""
        with torch.no_grad():
            for li in range(min(n_layers, self.L)):
                pf = f"blocks.{li}."
                W = self.params[pf + "attn.in_proj_weight"][config_idx]
                D = self.D
                for s0, s1 in ((0, D), (D, 2 * D)):
                    W[s0:s1, :].copy_((1.0 - blend) * W[s0:s1, :] + blend * (W[s0:s1, :] @ P_corr))

    def apply_embed_grad_shaping(self, configs: list[TrialConfig], embed_proj_ts: list[torch.Tensor | None], step: int):
        """Per-config gradient shaping on token_emb.weight."""
        g = self.params["token_emb.weight"].grad
        if g is None:
            return
        for i, cfg in enumerate(configs):
            if not cfg.geo_embed_grad_shape or embed_proj_ts[i] is None:
                continue
            P = embed_proj_ts[i]
            gi = g[i]
            g_proj = gi @ P
            g_perp = gi - g_proj
            hold = cfg.geo_embed_grad_hold_steps
            ramp = cfg.geo_embed_grad_ramp_steps
            perp_init = cfg.geo_embed_grad_perp_init
            if step < hold:
                scale = perp_init
            elif ramp > 0 and step < hold + ramp:
                t = (step - hold) / ramp
                scale = perp_init + (1.0 - perp_init) * t
            else:
                scale = 1.0
            g[i] = g_proj + scale * g_perp

    def apply_reanchor(self, configs: list[TrialConfig], geo_targets: list[torch.Tensor], step: int):
        """Per-config reanchor pulse on token_emb."""
        with torch.no_grad():
            for i, cfg in enumerate(configs):
                if cfg.geo_embed_reanchor_every <= 0 or cfg.geo_embed_reanchor_rho <= 0.0:
                    continue
                if step > cfg.geo_embed_reanchor_until_step and cfg.geo_embed_reanchor_until_step > 0:
                    continue
                if step % cfg.geo_embed_reanchor_every == 0:
                    rho = cfg.geo_embed_reanchor_rho
                    E = self.params["token_emb.weight"][i]
                    E.copy_((1.0 - rho) * E + rho * geo_targets[i])


# ---------------------------------------------------------------------------
# Precompute all needed bases and projectors
# ---------------------------------------------------------------------------
def precompute_artifacts(
    configs: list[TrialConfig],
    train_ids: np.ndarray,
    vocab_size: int,
    d_model: int,
    device: torch.device,
) -> dict:
    """Returns cached geo bases, attn projectors, model projectors, etc."""
    geo_basis_cache: dict[tuple, np.ndarray] = {}
    attn_corr_cache: dict[tuple, np.ndarray] = {}
    model_proj_cache: dict[tuple, np.ndarray] = {}

    for cfg in configs:
        gk = (cfg.geo_init_method, cfg.geo_init_mtp_weights, d_model)
        if gk not in geo_basis_cache:
            print(f"  computing geo_basis: {gk}", flush=True)
            if cfg.geo_init_method == "kl_bucket_mtp":
                mtp_w = parse_float_csv(cfg.geo_init_mtp_weights)
                geo_basis_cache[gk] = compute_kl_bucket_mtp_basis(train_ids, vocab_size, d_model, mtp_w)
            elif cfg.geo_init_method == "kl_bucket":
                geo_basis_cache[gk] = compute_kl_bucket_basis(train_ids, vocab_size, d_model)
            elif cfg.geo_init_method == "eig":
                op = np.zeros((vocab_size, vocab_size), dtype=np.float64)
                if len(train_ids) > 1:
                    a, b = train_ids[:-1], train_ids[1:]
                    np.add.at(op, (a, b), 1.0)
                op = 0.5 * (op + op.T)
                t = np.sum(op)
                if t > 0:
                    op /= t
                _, eigvecs = np.linalg.eigh(op)
                geo_basis_cache[gk] = eigvecs[:, -d_model:].astype(np.float32)
            else:
                geo_basis_cache[gk] = compute_bucket_basis(train_ids, vocab_size, d_model)

        geo_basis = geo_basis_cache[gk]

        if cfg.geo_attn_corr_bias:
            horizons = parse_int_csv(cfg.geo_attn_corr_horizons)
            h_weights = parse_float_csv(cfg.geo_attn_corr_horizon_weights)
            ak = (gk, cfg.geo_attn_corr_rank, tuple(horizons), tuple(h_weights))
            if ak not in attn_corr_cache:
                print(f"  computing attn_corr: rank={cfg.geo_attn_corr_rank}", flush=True)
                attn_corr_cache[ak] = compute_attn_corr_projector(
                    train_ids, vocab_size, geo_basis, d_model, cfg.geo_attn_corr_rank, horizons, h_weights,
                )

        if cfg.geo_attn_bias:
            mk = (gk, 16)  # eta_topk=16 hardcoded like original
            if mk not in model_proj_cache:
                model_proj_cache[mk] = compute_model_projector(geo_basis, d_model, 16)

    return {
        "geo_basis_cache": geo_basis_cache,
        "attn_corr_cache": attn_corr_cache,
        "model_proj_cache": model_proj_cache,
    }


# ---------------------------------------------------------------------------
# Main parallel search
# ---------------------------------------------------------------------------
def run_parallel_search(
    configs: list[TrialConfig],
    train_ids: np.ndarray,
    val_ids: np.ndarray,
    vocab_size: int,
    d_model: int,
    n_head: int,
    n_layer: int,
    block_size: int,
    dropout: float,
    lr: float,
    steps: int,
    eval_iters: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> list[dict]:
    N = len(configs)
    print(f"Running {N} configs in parallel on {device}", flush=True)

    t0 = time.perf_counter()
    print("Precomputing artifacts...", flush=True)
    artifacts = precompute_artifacts(configs, train_ids, vocab_size, d_model, device)
    print(f"Artifacts ready in {time.perf_counter() - t0:.1f}s", flush=True)

    print("Initializing batched models...", flush=True)
    bm = BatchedModels(
        n_configs=N,
        vocab_size=vocab_size,
        d_model=d_model,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout,
        lr=lr,
        device=device,
        seed=seed,
    )

    # Apply per-config geo initialization
    print("Applying per-config geo bias...", flush=True)
    geo_targets: list[torch.Tensor] = []
    embed_proj_ts: list[torch.Tensor | None] = []

    for i, cfg in enumerate(configs):
        gk = (cfg.geo_init_method, cfg.geo_init_mtp_weights, d_model)
        geo_basis = artifacts["geo_basis_cache"][gk]

        # Build geo target
        geo_target_np = np.zeros((vocab_size, d_model), dtype=np.float32)
        k0 = min(d_model, geo_basis.shape[1])
        geo_target_np[:, :k0] = geo_basis[:, :k0]
        if cfg.geo_init_match_row_norm:
            E0 = bm.params["token_emb.weight"][i].detach().cpu().float().numpy()
            t_norm = np.linalg.norm(geo_target_np, axis=1, keepdims=True)
            e_norm = np.linalg.norm(E0, axis=1, keepdims=True)
            geo_target_np = geo_target_np / np.maximum(t_norm, 1e-8) * e_norm

        geo_target_t = torch.from_numpy(geo_target_np).to(device=device, dtype=bm.dtype)
        geo_targets.append(geo_target_t)

        # Blend embedding
        bm.apply_embed_geo_bias(i, geo_target_t, cfg.geo_init_blend)

        # Attention bias
        if cfg.geo_attn_bias:
            mk = (gk, 16)
            P_np = artifacts["model_proj_cache"][mk]
            P_t = torch.from_numpy(P_np).to(device=device, dtype=bm.dtype)
            bm.apply_attn_bias(i, P_t, cfg.geo_attn_bias_blend, n_layer)

        # Attention correlation bias
        if cfg.geo_attn_corr_bias:
            horizons = parse_int_csv(cfg.geo_attn_corr_horizons)
            h_weights = parse_float_csv(cfg.geo_attn_corr_horizon_weights)
            ak = (gk, cfg.geo_attn_corr_rank, tuple(horizons), tuple(h_weights))
            P_corr = torch.from_numpy(artifacts["attn_corr_cache"][ak]).to(device=device, dtype=bm.dtype)
            bm.apply_attn_corr_bias(i, P_corr, cfg.geo_attn_corr_blend, cfg.geo_attn_corr_layers)

        # Embed grad projector
        if cfg.geo_embed_grad_shape:
            P_np = compute_embed_projector_from_target(geo_target_np, cfg.geo_embed_grad_rank)
            embed_proj_ts.append(torch.from_numpy(P_np).to(device=device, dtype=bm.dtype))
        else:
            embed_proj_ts.append(None)

    print(f"Init complete in {time.perf_counter() - t0:.1f}s", flush=True)

    # Training loop
    print(f"Training {steps} steps...", flush=True)
    train_losses = torch.zeros(N, device=device)
    train_accs = torch.zeros(N, device=device)

    for step in range(steps):
        xb, yb = get_batch(train_ids, batch_size, block_size, device)

        bm.zero_grad()
        logits = bm.forward(xb)  # (N, B, T, V)
        losses = bm.compute_loss(logits, yb)  # (N,)
        total_loss = losses.sum()
        total_loss.backward()

        # Per-config grad shaping
        bm.apply_embed_grad_shaping(configs, embed_proj_ts, step)

        bm.adam_step()

        # Reanchor
        bm.apply_reanchor(configs, geo_targets, step)

        with torch.no_grad():
            train_losses += losses
            train_accs += bm.compute_acc(logits, yb)

        if (step + 1) % 50 == 0 or step == 0:
            avg_loss = train_losses.mean().item() / (step + 1)
            avg_acc = train_accs.mean().item() / (step + 1)
            best_loss = (train_losses / (step + 1)).min().item()
            elapsed = time.perf_counter() - t0
            print(
                f"  step {step+1}/{steps} avg_loss={avg_loss:.4f} avg_acc={avg_acc:.4f} best_loss={best_loss:.4f} "
                f"elapsed={elapsed:.1f}s step_avg={elapsed/(step+1)*1000:.1f}ms",
                flush=True,
            )

    train_losses /= steps
    train_accs /= steps

    # Eval
    print(f"Evaluating ({eval_iters} iters)...", flush=True)
    val_losses = torch.zeros(N, device=device)
    val_accs = torch.zeros(N, device=device)
    with torch.no_grad():
        for _ in range(eval_iters):
            xb, yb = get_batch(val_ids, batch_size, block_size, device)
            logits = bm.forward(xb)
            val_losses += bm.compute_loss(logits, yb)
            val_accs += bm.compute_acc(logits, yb)
    val_losses /= eval_iters
    val_accs /= eval_iters

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s", flush=True)

    # Build results
    results = []
    for i, cfg in enumerate(configs):
        results.append({
            "config_idx": i,
            "config": {f.name: getattr(cfg, f.name) for f in fields(cfg)},
            "config_id": trial_id(cfg),
            "train_loss": float(train_losses[i].item()),
            "train_acc": float(train_accs[i].item()),
            "val_loss": float(val_losses[i].item()),
            "val_acc": float(val_accs[i].item()),
        })

    results.sort(key=lambda r: (-r["val_acc"], r["val_loss"]))
    return results


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
def _fmt_val(v, as_int: bool):
    return int(round(v)) if as_int else round(v, 6)


# Numeric axes: (lo, hi, is_int)
BINARY_SEARCH_AXES: dict[str, tuple[float, float, bool]] = {
    "geo_init_blend":            (0.4, 0.95, False),
    "geo_embed_grad_rank":       (4, 32, True),
    "geo_embed_grad_perp_init":  (0.05, 0.4, False),
    "geo_embed_grad_hold_steps": (50, 400, True),
    "geo_embed_grad_ramp_steps": (50, 400, True),
    "geo_embed_reanchor_rho":    (0.0, 0.04, False),
}

BINARY_SEARCH_FIXED: dict[str, object] = {
    "geo_init_method": "kl_bucket",
    "geo_init_mtp_weights": "1.0,0.5,0.25",
    "geo_init_fullspace": False,
    "geo_init_ridge": 1e-3,
    "geo_init_match_row_norm": True,
    "geo_attn_bias": False,
    "geo_attn_bias_blend": 0.2,
    "geo_attn_corr_bias": False,
    "geo_attn_corr_blend": 0.1,
    "geo_attn_corr_rank": 8,
    "geo_attn_corr_layers": 2,
    "geo_attn_corr_horizons": "1,2,3",
    "geo_attn_corr_horizon_weights": "1.0,0.5,0.25",
    "geo_embed_grad_shape": True,
    "geo_embed_reanchor_every": 50,
    "geo_embed_reanchor_until_step": 0,
}


def generate_binary_level1() -> tuple[list[TrialConfig], dict[str, list]]:
    """
    Level 1: 3 values per axis (lo, mid, hi) → 3^6 = 729 configs.
    Returns (configs, axis_values) where axis_values maps key → [lo, mid, hi].
    """
    import itertools

    axis_vals: dict[str, list] = {}
    for k, (lo, hi, as_int) in BINARY_SEARCH_AXES.items():
        mid = (lo + hi) / 2.0
        axis_vals[k] = sorted({_fmt_val(lo, as_int), _fmt_val(mid, as_int), _fmt_val(hi, as_int)})

    keys = list(axis_vals.keys())
    val_lists = [axis_vals[k] for k in keys]

    total = 1
    for v in val_lists:
        total *= len(v)

    print(f"Binary search level 1: {len(keys)} axes, {total} configs", flush=True)
    for k in keys:
        print(f"  {k}: {axis_vals[k]}", flush=True)

    configs = []
    for combo in itertools.product(*val_lists):
        d = dict(BINARY_SEARCH_FIXED)
        for k, v in zip(keys, combo):
            d[k] = v
        configs.append(TrialConfig(**d))

    return configs, axis_vals


def generate_binary_refinement(
    best_config: dict,
    intervals: dict[str, tuple[float, float]],
) -> tuple[list[TrialConfig], dict[str, tuple[float, float]]]:
    """
    Narrow each axis to the half-interval containing the best value.
    The winning half keeps 2 old endpoints + 1 new midpoint = 3 values per axis.
    We already ran 2^6 = 64 combos of the 2 old values, so we run 3^6 - 2^6 = 665 new.

    Returns (new_configs, new_intervals).
    """
    import itertools

    new_intervals: dict[str, tuple[float, float]] = {}
    old_vals_per_axis: dict[str, list] = {}  # 2 surviving old values
    all_vals_per_axis: dict[str, list] = {}  # 3 values (2 old + 1 new mid)

    keys = list(BINARY_SEARCH_AXES.keys())

    for k in keys:
        lo, hi = intervals[k]
        as_int = BINARY_SEARCH_AXES[k][2]
        mid = (lo + hi) / 2.0
        best_val = best_config[k]

        # Narrow to the half containing best
        if best_val <= _fmt_val(mid, as_int):
            new_lo, new_hi = lo, mid
        else:
            new_lo, new_hi = mid, hi

        new_mid = (new_lo + new_hi) / 2.0
        new_lo_f = _fmt_val(new_lo, as_int)
        new_hi_f = _fmt_val(new_hi, as_int)
        new_mid_f = _fmt_val(new_mid, as_int)

        new_intervals[k] = (new_lo, new_hi)
        old_vals_per_axis[k] = sorted({new_lo_f, new_hi_f})
        all_vals_per_axis[k] = sorted({new_lo_f, new_mid_f, new_hi_f})

    # Already-run combos: cartesian product of just the 2 old values per axis
    already_run = set()
    old_lists = [old_vals_per_axis[k] for k in keys]
    for combo in itertools.product(*old_lists):
        already_run.add(combo)

    # New configs: full 3-value grid minus already-run
    val_lists = [all_vals_per_axis[k] for k in keys]
    new_configs = []
    for combo in itertools.product(*val_lists):
        if combo in already_run:
            continue
        d = dict(BINARY_SEARCH_FIXED)
        for k, v in zip(keys, combo):
            d[k] = v
        new_configs.append(TrialConfig(**d))

    print(f"Binary refinement: {len(new_configs)} new configs", flush=True)
    for k in keys:
        print(f"  {k}: vals={all_vals_per_axis[k]} interval=({intervals[k][0]:.6f}, {intervals[k][1]:.6f}) -> ({new_intervals[k][0]:.6f}, {new_intervals[k][1]:.6f})", flush=True)

    return new_configs, new_intervals


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Parallel batched Shakespeare geo-bias search")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-head", type=int, default=2)
    p.add_argument("--n-layer", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--data-path", type=str, default="./data/tinyshakespeare/input.txt")
    p.add_argument("--binary-levels", type=int, default=2)
    p.add_argument("--config-batch", type=int, default=64)
    p.add_argument("--top-k", type=int, default=25)
    p.add_argument("--out", type=str, default="experiments/tier4/parallel_search_results.json")
    args = p.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else (
        torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    )
    print(f"Device: {device}", flush=True)

    # Data
    print("Loading data...", flush=True)
    text = ensure_data(Path(args.data_path))
    stoi, itos = build_vocab(text)
    ids = encode(text, stoi)
    n = len(ids)
    n_train = int(0.9 * n)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    vocab_size = len(stoi)
    print(f"Vocab={vocab_size} Train={len(train_ids)} Val={len(val_ids)}", flush=True)

    def run_config_batch(cfgs: list[TrialConfig]) -> list[dict]:
        """Run a list of configs through parallel search in config-batch chunks."""
        res: list[dict] = []
        n_batches = (len(cfgs) + args.config_batch - 1) // args.config_batch
        for bi in range(n_batches):
            start = bi * args.config_batch
            end = min(start + args.config_batch, len(cfgs))
            chunk = cfgs[start:end]
            print(f"\n--- Config batch {bi+1}/{n_batches} ({len(chunk)} configs, {start}-{end-1}) ---", flush=True)
            batch_results = run_parallel_search(
                configs=chunk,
                train_ids=train_ids,
                val_ids=val_ids,
                vocab_size=vocab_size,
                d_model=args.d_model,
                n_head=args.n_head,
                n_layer=args.n_layer,
                block_size=args.block_size,
                dropout=args.dropout,
                lr=args.lr,
                steps=args.steps,
                eval_iters=args.eval_iters,
                batch_size=args.batch_size,
                seed=args.seed,
                device=device,
            )
            res.extend(batch_results)
        return res

    def print_leaderboard(results: list[dict], label: str):
        results.sort(key=lambda r: (-r["val_acc"], r["val_loss"]))
        top_k = min(args.top_k, len(results))
        print(f"\n{'='*80}")
        print(f"{label}: TOP {top_k} (of {len(results)} tested)")
        print(f"{'='*80}")
        for i, r in enumerate(results[:top_k]):
            print(
                f"#{i+1:3d} val_acc={r['val_acc']:.6f} val_loss={r['val_loss']:.4f} "
                f"train_acc={r['train_acc']:.6f} train_loss={r['train_loss']:.4f} "
                f"id={r['config_id']}"
            )
            cfg = r["config"]
            print(
                f"     blend={cfg['geo_init_blend']} "
                f"rank={cfg['geo_embed_grad_rank']} perp={cfg['geo_embed_grad_perp_init']} "
                f"hold={cfg['geo_embed_grad_hold_steps']} ramp={cfg['geo_embed_grad_ramp_steps']} "
                f"rho={cfg['geo_embed_reanchor_rho']}"
            )

    # ---- Iterative binary search (resumable) ----
    out_path = Path(args.out)
    all_results: list[dict] = []
    completed_levels = 0
    intervals: dict[str, tuple[float, float]] = {
        k: (lo, hi) for k, (lo, hi, _) in BINARY_SEARCH_AXES.items()
    }

    # Resume from previous run if output file exists
    if out_path.exists():
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        completed_levels = prev.get("completed_levels", 0)
        all_results = prev.get("all", [])
        if "intervals" in prev:
            intervals = {k: tuple(v) for k, v in prev["intervals"].items()}
        print(f"Resuming: {completed_levels} levels completed, {len(all_results)} results loaded", flush=True)
        all_results.sort(key=lambda r: (-r["val_acc"], r["val_loss"]))
        if all_results:
            print_leaderboard(all_results, f"RESUMED (level {completed_levels})")

    def save_results(level: int):
        best = all_results[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "n_configs": len(all_results),
            "strategy": "iterative_binary_search",
            "completed_levels": level,
            "binary_levels": args.binary_levels,
            "intervals": {k: list(v) for k, v in intervals.items()},
            "steps": args.steps,
            "eval_iters": args.eval_iters,
            "batch_size": args.batch_size,
            "device": str(device),
            "best": best,
            "top": all_results[:args.top_k],
            "all": all_results,
        }
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Saved results to {out_path}", flush=True)

    # Level 1
    if completed_levels < 1:
        print("\n" + "=" * 80, flush=True)
        print("BINARY SEARCH LEVEL 1", flush=True)
        print("=" * 80, flush=True)
        level1_configs, axis_vals = generate_binary_level1()
        print(f"Level 1: {len(level1_configs)} configs", flush=True)
        level1_results = run_config_batch(level1_configs)
        all_results.extend(level1_results)
        all_results.sort(key=lambda r: (-r["val_acc"], r["val_loss"]))
        completed_levels = 1
        print_leaderboard(all_results, "LEVEL 1")
        save_results(1)

    # Levels 2+: narrow and refine
    for level in range(max(2, completed_levels + 1), args.binary_levels + 1):
        print(f"\n{'='*80}", flush=True)
        print(f"BINARY SEARCH LEVEL {level}", flush=True)
        print(f"{'='*80}", flush=True)

        best_cfg = all_results[0]["config"]
        new_configs, intervals = generate_binary_refinement(best_cfg, intervals)
        if not new_configs:
            print("No new configs to evaluate, stopping.", flush=True)
            break
        print(f"Level {level}: {len(new_configs)} new configs", flush=True)
        new_results = run_config_batch(new_configs)
        all_results.extend(new_results)
        all_results.sort(key=lambda r: (-r["val_acc"], r["val_loss"]))
        completed_levels = level
        print_leaderboard(all_results, f"LEVEL {level} (cumulative)")
        save_results(level)

    # Final summary
    best = all_results[0]
    print(f"\nFINAL BEST: val_acc={best['val_acc']:.6f} val_loss={best['val_loss']:.4f}")
    print(f"BEST CONFIG: {json.dumps(best['config'], indent=2)}")


if __name__ == "__main__":
    main()
