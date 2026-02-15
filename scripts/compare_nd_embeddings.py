#!/usr/bin/env python3
"""Compare ND embedding/projection structures on MQAR.

Tests all combinations of:
  - Rank: 1D (flat linear), 2D, 3D, 4D (einsum over ND tensors)
  - Ratio: 1:3 (small x large), 1:1 (balanced), 3:1 (large x small)
  - Projection mode: einsum (true ND contraction) vs matmul (ND weight reshaped to 2D)

Architecture matches Zoology exactly: pre-norm TransformerBlock, MHA, no MLP,
LayerNorm, position embeddings, tied embed/head, 0.02 init, cosine LR.

Outputs: train loss, train acc, val loss, val acc for all models.

Usage:
    python scripts/compare_nd_embeddings.py --dim 128 --epochs 64 --save results.png
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tqdm import tqdm


VOCAB_SIZE = 8192
MOD_BASE = 5


def gen_mod_arith(B, L, device='cpu'):
    x = torch.randint(0, MOD_BASE, (B, L), device=device)
    target = x.cumsum(dim=1) % MOD_BASE
    return x, target


def gen_mqar(num_examples, seq_len, vocab_size=8192, num_kv_pairs=4, power_a=0.01, seed=42):
    """Generate multi-query associative recall data (from Zoology).

    Sequence contains key-value pairs, then queries. Model must recall the
    value associated with each queried key.

    Returns (inputs, labels) tensors. Labels are -100 except at query positions.
    """
    rng = np.random.default_rng(seed)
    context_size = num_kv_pairs * 2

    key_vocab = np.arange(1, vocab_size // 2)
    val_vocab = np.arange(vocab_size // 2, vocab_size)

    keys = np.stack([rng.choice(key_vocab, num_kv_pairs, replace=False)
                     for _ in range(num_examples)])
    values = np.stack([rng.choice(val_vocab, num_kv_pairs, replace=False)
                       for _ in range(num_examples)])

    # Build context: key val key val ...
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2] = keys
    kvs[:, 1::2] = values

    # Power-law gaps for query placement
    space = (seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1)
    p = p / p.sum()

    log_p = np.log(p)
    gumbel = rng.gumbel(size=(num_examples, space))
    gap_idx = np.argpartition(-(log_p + gumbel), num_kv_pairs, axis=1)[:, :num_kv_pairs]

    # Build query section
    queries = np.zeros((num_examples, seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gap_idx * 2, values=keys, axis=1)

    examples = np.concatenate([kvs, queries], axis=1)

    labels = np.full((num_examples, seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, gap_idx * 2 + context_size + 1, values=values, axis=1)

    inputs = torch.tensor(examples[:, :-1])
    labels = torch.tensor(labels[:, 1:])

    # Fill non-query/non-kv positions with random tokens
    mask = inputs == 0
    inputs[mask] = torch.randint(vocab_size, size=inputs.shape)[mask]

    return inputs, labels


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


# ---------------------------------------------------------------------------
# Factorization helpers
# ---------------------------------------------------------------------------

def _balanced_factor(n, parts):
    """Factor n into `parts` balanced factors."""
    if parts == 1:
        return (n,)
    target = int(round(n ** (1.0 / parts)))
    best = 2
    for f in range(2, int(n ** 0.5) + 1):
        if n % f == 0 and abs(f - target) <= abs(best - target):
            best = f
    return (best,) + _balanced_factor(n // best, parts - 1)


def factorize_dim(D, rank, ratio='1:3'):
    """Factor D into `rank` dimensions with configurable ratio.

    ratio controls the split between first dim and the rest:
      '1:3' — first dim small, rest large (ULB convention)
      '3:1' — first dim large, rest small (flipped)
      '1:1' — balanced / square-ish

    Examples for D=256, rank=2:
        1:3 -> (8, 32)
        3:1 -> (32, 8)
        1:1 -> (16, 16)
    """
    if rank == 1:
        return (D,)

    if ratio == '1:1':
        return _balanced_factor(D, rank)
    elif ratio == '1:3':
        target_nf = int((D / 3) ** 0.5)
    elif ratio == '3:1':
        target_nf = int((D * 3) ** 0.5)
    else:
        raise ValueError(f"Unknown ratio: {ratio}")

    # Search ALL factor pairs (not just up to sqrt)
    best = 1
    for s in range(2, D):
        if D % s == 0:
            if abs(s - target_nf) <= abs(best - target_nf):
                best = s
    nf = best
    desc = D // nf
    if rank == 2:
        return (nf, desc)
    return (nf,) + _balanced_factor(desc, rank - 1)


# ---------------------------------------------------------------------------
# Generic ND projection block
# ---------------------------------------------------------------------------

class BlockND(nn.Module):
    """Causal self-attention using N-dimensional tensor projections.

    Matches Zoology TransformerBlock exactly:
      - Pre-norm with LayerNorm
      - MHA: fused Wqkv -> causal attention -> out_proj
      - state_mixer = Identity (no MLP)
      - embed_dropout on layer 0, resid_dropout=0.0 elsewhere
      - out_proj gets rescaled init: std=0.02 / sqrt(2*n_layers)

    Only difference from Zoology: ND ranks use einsum projections
    instead of nn.Linear for Wqkv and out_proj.
    """

    def __init__(self, shape, n_layers=2, layer_idx=0, num_heads=1, dropout=0.1,
                 proj_mode='einsum'):
        """
        proj_mode:
          'linear'  — nn.Linear (flat matmul), used for 1D
          'einsum'  — einsum over ND-shaped weight tensors
          'matmul'  — weight stored as ND tensor but reshaped to (D,D) for matmul
        """
        super().__init__()
        self.shape = shape
        self.rank = len(shape)
        self.D = math.prod(shape)
        self.num_heads = num_heads
        self.head_dim = self.D // num_heads
        self.n_layers = n_layers
        self.proj_mode = proj_mode

        init_std = 0.02
        out_std = init_std / math.sqrt(2 * n_layers)

        if proj_mode == 'linear':
            self.Wqkv = nn.Linear(self.D, 3 * self.D)
            self.out_proj = nn.Linear(self.D, self.D)
            nn.init.normal_(self.Wqkv.weight, std=init_std)
            nn.init.zeros_(self.Wqkv.bias)
            nn.init.normal_(self.out_proj.weight, std=out_std)
            nn.init.zeros_(self.out_proj.bias)
        else:
            # Both 'einsum' and 'matmul' store weights as ND tensors
            if self.rank > 1:
                n = self.rank
                in_chars = ''.join(chr(ord('c') + i) for i in range(n))
                out_chars = ''.join(chr(ord('c') + n + i) for i in range(n))
                self.einsum_str = f'...{in_chars},{in_chars}{out_chars}->...{out_chars}'

            w_shape = tuple(shape) + tuple(shape)
            self.wq = nn.Parameter(torch.randn(*w_shape) * init_std)
            self.wk = nn.Parameter(torch.randn(*w_shape) * init_std)
            self.wv = nn.Parameter(torch.randn(*w_shape) * init_std)
            self.wo = nn.Parameter(torch.randn(*w_shape) * out_std)

        # Matching Zoology: LayerNorm, dropout
        self.norm1 = nn.LayerNorm(self.D)
        self.norm2 = nn.LayerNorm(self.D)
        self.dropout1 = nn.Dropout(dropout if layer_idx == 0 else 0.0)
        self.dropout2 = nn.Dropout(0.0)
        self.attn_dropout = dropout

    def _proj(self, x_flat, w):
        """Project x_flat (B,T,D) through ND weight w using self.proj_mode."""
        if self.proj_mode == 'einsum' and self.rank > 1:
            B, T = x_flat.shape[:2]
            x_nd = x_flat.view(B, T, *self.shape)
            return torch.einsum(self.einsum_str, x_nd, w).reshape(B, T, self.D)
        else:
            # 'matmul' mode or rank-1 einsum: reshape weight to (D,D) and matmul
            return x_flat @ w.reshape(self.D, self.D).T

    def forward(self, hidden_states, residual=None):
        # Matches Zoology TransformerBlock.forward
        B, T = hidden_states.shape[:2]
        D, NH, HD = self.D, self.num_heads, self.head_dim

        # --- sequence mixer (MHA) ---
        dropped = self.dropout1(hidden_states.reshape(B, T, D))
        residual = (dropped + residual) if residual is not None else dropped
        h = self.norm1(residual)

        if self.proj_mode == 'linear':
            qkv = self.Wqkv(h).reshape(B, T, 3, NH, HD)
            q, k, v = qkv.unbind(dim=2)
        else:
            q = self._proj(h, self.wq).reshape(B, T, NH, HD)
            k = self._proj(h, self.wk).reshape(B, T, NH, HD)
            v = self._proj(h, self.wv).reshape(B, T, NH, HD)

        q = q.transpose(1, 2)  # (B, NH, T, HD)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        context = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).contiguous().reshape(B, T, D)

        if self.proj_mode == 'linear':
            hidden_states = self.out_proj(context)
        else:
            hidden_states = self._proj(context, self.wo)

        # --- state mixer (Identity) ---
        dropped = self.dropout2(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.norm2(residual)

        return hidden_states, residual


class Model(nn.Module):
    """Matches Zoology LanguageModel + LMBackbone structure."""

    def __init__(self, shape, vocab_size=VOCAB_SIZE, n_layers=2, num_heads=1,
                 max_position_embeddings=64, dropout=0.1, proj_mode='einsum'):
        super().__init__()
        self.shape = shape
        self.rank = len(shape)
        self.D = math.prod(shape)

        # Token + position embeddings (matching Zoology TokenEmbeddings)
        self.word_embeddings = nn.Embedding(vocab_size, self.D)
        self.position_embeddings = nn.Embedding(max_position_embeddings, self.D)

        self.blocks = nn.ModuleList([
            BlockND(shape, n_layers=n_layers, layer_idx=i,
                    num_heads=num_heads, dropout=dropout, proj_mode=proj_mode)
            for i in range(n_layers)
        ])
        self.drop_f = nn.Dropout(0.0)
        self.ln_f = nn.LayerNorm(self.D)
        self.lm_head = nn.Linear(self.D, vocab_size, bias=False)

        # Tie embed and head weights (matching Zoology)
        self.lm_head.weight = self.word_embeddings.weight

        # Init all weights matching Zoology _init_weights
        self._init_weights()

    def _init_weights(self):
        init_std = 0.02
        nn.init.normal_(self.word_embeddings.weight, std=init_std)
        nn.init.normal_(self.position_embeddings.weight, std=init_std)
        # nn.Linear and nn.LayerNorm already handled by BlockND init
        # ln_f is default init (ones/zeros) which is fine

    def forward(self, x):
        B, T = x.shape
        position_ids = torch.arange(T, dtype=torch.long, device=x.device)
        h = self.word_embeddings(x) + self.position_embeddings(position_ids)

        if self.rank > 1:
            h = h.view(B, T, *self.shape)

        residual = None
        for block in self.blocks:
            h, residual = block(h, residual)

        # Final norm (matching LMBackbone)
        dropped = self.drop_f(h.reshape(B, T, self.D))
        residual = (dropped + residual) if residual is not None else dropped
        output = self.ln_f(residual)

        return self.lm_head(output)


def train(model, n_epochs, task='mqar', B=512, L=64, lr=3e-4, device='cpu', label='',
          vocab_size=8192, num_kv_pairs=4, num_train_examples=100_000,
          use_compile=False, use_amp=False, profile=False):
    model = model.to(device)

    if use_compile:
        print(f'  [{label}] Compiling model with torch.compile()...', flush=True)
        model = torch.compile(model)

    opt = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=0.0)

    # AMP setup
    amp_dtype = torch.bfloat16 if (use_amp and device == 'cuda') else None
    amp_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if amp_dtype else None

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # Gradient snapshots
    grad_snapshots = {}
    weight_snapshots = {}
    snapshot_epochs = sorted(set([0, n_epochs // 4, n_epochs // 2,
                                   3 * n_epochs // 4, n_epochs - 1]))

    # Pre-generate datasets (matching Zoology: fixed dataset, iterate in batches)
    print(f'  [{label}] Generating {num_train_examples} train + 3000 val examples...', flush=True)
    if task == 'mqar':
        train_x, train_t = gen_mqar(num_train_examples, L, vocab_size=vocab_size,
                                    num_kv_pairs=num_kv_pairs, seed=42)
        val_x, val_t = gen_mqar(3000, L, vocab_size=vocab_size,
                                num_kv_pairs=num_kv_pairs, seed=9999)
    else:
        train_x, train_t = gen_mod_arith(num_train_examples, L)
        val_x, val_t = gen_mod_arith(3000, L)

    # Move entire dataset to device once
    train_x, train_t = train_x.to(device), train_t.to(device)
    val_x, val_t = val_x.to(device), val_t.to(device)
    n_batches = num_train_examples // B

    # Helper to compute loss + acc over a dataset in batches (no mid-loop syncs)
    def eval_metrics(data_x, data_t):
        model.eval()
        total_loss = torch.zeros(1, device=device)
        total_correct = torch.zeros(1, dtype=torch.long, device=device)
        total_count = torch.zeros(1, dtype=torch.long, device=device)
        bs = min(512, len(data_x))
        with torch.no_grad():
            for i in range(0, len(data_x), bs):
                bx = data_x[i:i+bs]
                bt = data_t[i:i+bs]
                if amp_ctx is not None:
                    with amp_ctx:
                        logits = model(bx)
                else:
                    logits = model(bx)
                total_loss += F.cross_entropy(logits.reshape(-1, vocab_size), bt.reshape(-1),
                                              ignore_index=-100) * len(bx)
                preds = logits.argmax(-1)
                if task == 'mqar':
                    m = bt != -100
                    total_correct += (preds[m] == bt[m]).sum()
                    total_count += m.sum()
                else:
                    total_correct += (preds == bt).sum()
                    total_count += bt.numel()
        # Single sync point
        return (total_loss / len(data_x)).item(), (total_correct.float() / total_count.clamp(min=1)).item()

    # Register starting metrics before any training
    tl0, ta0 = eval_metrics(train_x[:min(5000, len(train_x))], train_t[:min(5000, len(train_x))])
    vl0, va0 = eval_metrics(val_x, val_t)
    train_losses.append(tl0); train_accs.append(ta0)
    val_losses.append(vl0); val_accs.append(va0)
    print(f'  [{label}] start: train_loss={tl0:.4f} train_acc={ta0:.3f} val_loss={vl0:.4f} val_acc={va0:.3f}', flush=True)

    wall_start = time.perf_counter()
    pbar = tqdm(range(n_epochs), desc=f'{label} training', leave=True)
    for epoch in pbar:
        model.train()
        epoch_loss = torch.zeros(1, device=device)
        epoch_correct = torch.zeros(1, dtype=torch.long, device=device)
        epoch_count = torch.zeros(1, dtype=torch.long, device=device)

        # Shuffle via index permutation on device (no copy)
        perm = torch.randperm(num_train_examples, device=device)
        do_snapshot = epoch in snapshot_epochs

        # Profiling timers (only on first epoch if --profile)
        do_profile = profile and epoch == 0
        if do_profile:
            t_fwd = t_bwd = t_opt = 0.0

        for batch_i in range(n_batches):
            idx = perm[batch_i * B : (batch_i + 1) * B]
            x = train_x[idx]
            t = train_t[idx]

            if do_profile and device == 'cuda':
                torch.cuda.synchronize()
                _t0 = time.perf_counter()

            if amp_ctx is not None:
                with amp_ctx:
                    logits = model(x)
                    loss = F.cross_entropy(logits.reshape(-1, vocab_size), t.reshape(-1),
                                           ignore_index=-100)
            else:
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), t.reshape(-1),
                                       ignore_index=-100)

            if do_profile and device == 'cuda':
                torch.cuda.synchronize()
                _t1 = time.perf_counter()
                t_fwd += _t1 - _t0

            opt.zero_grad()
            loss.backward()

            if do_profile and device == 'cuda':
                torch.cuda.synchronize()
                _t2 = time.perf_counter()
                t_bwd += _t2 - _t1

            # Accumulate on GPU — no sync
            with torch.no_grad():
                epoch_loss += loss.detach()
                preds = logits.argmax(-1)
                if task == 'mqar':
                    m = t != -100
                    epoch_correct += (preds[m] == t[m]).sum()
                    epoch_count += m.sum()
                else:
                    epoch_correct += (preds == t).sum()
                    epoch_count += t.numel()

            # Snapshot (first batch of snapshot epochs only)
            if do_snapshot and batch_i == 0:
                grad_snapshots[epoch] = {}
                weight_snapshots[epoch] = {}
                for name, p in model.named_parameters():
                    if p.grad is not None and ('wq' in name or 'seq_wq' in name):
                        grad_snapshots[epoch][name] = p.grad.detach().cpu().clone()
                        weight_snapshots[epoch][name] = p.detach().cpu().clone()

            opt.step()

            if do_profile and device == 'cuda':
                torch.cuda.synchronize()
                _t3 = time.perf_counter()
                t_opt += _t3 - _t2

        scheduler.step()

        if do_profile and device == 'cuda':
            torch.cuda.synchronize()
            _eval_t0 = time.perf_counter()

        # Eval on train subset + val set with final weights (not mid-training averages)
        train_sub = min(5000, num_train_examples)
        tl, ta = eval_metrics(train_x[:train_sub], train_t[:train_sub])
        train_losses.append(tl)
        train_accs.append(ta)

        vl_, va_ = eval_metrics(val_x, val_t)
        val_losses.append(vl_)
        val_accs.append(va_)

        if do_profile and device == 'cuda':
            torch.cuda.synchronize()
            _eval_t1 = time.perf_counter()
            t_eval = _eval_t1 - _eval_t0
            t_total = t_fwd + t_bwd + t_opt + t_eval
            print(f'\n  [{label}] PROFILE (epoch 0, {n_batches} batches):')
            print(f'    Forward:    {t_fwd:>7.2f}s ({t_fwd/t_total*100:>5.1f}%)')
            print(f'    Backward:   {t_bwd:>7.2f}s ({t_bwd/t_total*100:>5.1f}%)')
            print(f'    Optimizer:  {t_opt:>7.2f}s ({t_opt/t_total*100:>5.1f}%)')
            print(f'    Eval:       {t_eval:>7.2f}s ({t_eval/t_total*100:>5.1f}%)')
            print(f'    Total:      {t_total:>7.2f}s')
            print(f'    Per batch:  {(t_fwd+t_bwd+t_opt)/n_batches*1000:>7.1f}ms (fwd+bwd+opt)')
            print(flush=True)

        pbar.set_postfix(tl=f'{train_losses[-1]:.4f}', ta=f'{train_accs[-1]:.3f}',
                         vl=f'{vl_:.4f}', va=f'{va_:.3f}')

        # Early stop if solved
        if va_ > 0.99:
            print(f'  [{label}] Early stop at epoch {epoch} — val_acc={va_:.3f}', flush=True)
            break

    wall_elapsed = time.perf_counter() - wall_start
    actual_epochs = len(train_losses) - 1  # subtract initial metrics
    sec_per_epoch = wall_elapsed / max(actual_epochs, 1)
    print(f'  [{label}] Done: {actual_epochs} epochs in {wall_elapsed:.1f}s ({sec_per_epoch:.1f}s/epoch)', flush=True)

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'grad_snapshots': grad_snapshots,
        'weight_snapshots': weight_snapshots,
        'snapshot_epochs': snapshot_epochs,
        'wall_time': wall_elapsed,
        'sec_per_epoch': sec_per_epoch,
    }


def svd_spectrum(w):
    """Get normalized singular values of a weight tensor reshaped to 2D."""
    flat = w.reshape(w.shape[0], -1).float() if w.dim() > 2 else w.float()
    if w.dim() >= 4:
        half = w.dim() // 2
        rows = math.prod(w.shape[:half])
        cols = math.prod(w.shape[half:])
        flat = w.reshape(rows, cols).float()
    S = torch.linalg.svdvals(flat)
    return S / S[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dim', type=int, default=768,
                        help='Model dim (768 gives clean 1:3 ratio: 16x48)')
    parser.add_argument('--task', type=str, default='mqar', choices=['mqar', 'mod_arith'],
                        help='Task: mqar (associative recall) or mod_arith')
    parser.add_argument('--epochs', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length')
    parser.add_argument('--num-kv-pairs', type=int, default=None,
                        help='Number of key-value pairs (default: auto from seq_len, matching Zoology)')
    parser.add_argument('--num-train-examples', type=int, default=100_000,
                        help='Number of training examples (pre-generated)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (default: auto from seq_len, matching Zoology)')
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto-detect cuda/mps/cpu)')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() on model (requires PyTorch 2.0+)')
    parser.add_argument('--amp', action='store_true',
                        help='Use automatic mixed precision (bf16 on CUDA)')
    parser.add_argument('--profile', action='store_true',
                        help='Profile first model: time forward, backward, optimizer, eval separately')
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f"Using device: {args.device}")

    # Auto-compute kv_pairs and batch_size from seq_len (matching Zoology ripple_mqar.py)
    ZOOLOGY_SEQKV = {64: 4, 256: 16, 512: 32, 1024: 64, 4096: 256}
    if args.num_kv_pairs is None:
        if args.seq_len in ZOOLOGY_SEQKV:
            args.num_kv_pairs = ZOOLOGY_SEQKV[args.seq_len]
        else:
            args.num_kv_pairs = args.seq_len // 16
        print(f"Auto kv_pairs={args.num_kv_pairs} for seq_len={args.seq_len}")

    if args.batch_size is None:
        if args.seq_len <= 128:
            args.batch_size = 512
        elif args.seq_len <= 512:
            args.batch_size = 256
        elif args.seq_len <= 2048:
            args.batch_size = 128
        elif args.seq_len <= 4096:
            args.batch_size = 64
        else:
            args.batch_size = 32
        print(f"Auto batch_size={args.batch_size} for seq_len={args.seq_len}")

    D = args.dim
    vocab_size = VOCAB_SIZE if args.task == 'mqar' else MOD_BASE
    task_name = 'MQAR (associative recall)' if args.task == 'mqar' else 'Modular Arithmetic'

    # Build all model configs: (label, shape, proj_mode)
    configs = []

    # 1D baseline
    configs.append(('1D', (D,), 'linear'))

    # 2D/3D/4D with all three ratios (skip duplicates)
    seen_shapes = set()
    for rank in [2, 3, 4]:
        for ratio in ['1:3', '1:1', '3:1']:
            shape = factorize_dim(D, rank, ratio=ratio)
            key = (shape, 'einsum')
            if key in seen_shapes:
                print(f"  (skipping {rank}D-{ratio} = {'x'.join(map(str,shape))} — duplicate)")
                continue
            seen_shapes.add(key)
            label = f'{rank}D-{ratio}'
            configs.append((label, shape, 'einsum'))

    # 2D ablation: einsum vs matmul (using 1:3 ratio)
    shape_2d = factorize_dim(D, 2, ratio='1:3')
    configs.append(('2D-matmul', shape_2d, 'matmul'))

    print(f"Task: {task_name}, vocab_size={vocab_size}, seq_len={args.seq_len}, kv_pairs={args.num_kv_pairs}, batch={args.batch_size}")
    print(f"d_model = {D}")
    print(f"\nModel configs:")
    for label, shape, proj_mode in configs:
        print(f"  {label:<14} shape={'x'.join(map(str, shape)):>15}  proj={proj_mode}")

    results = {}

    def print_leaderboard():
        if not results:
            return
        W = 120
        print(f"\n{'='*W}")
        print(f"{'Model':<16} {'Shape':>15} {'Proj':>7} {'Params':>10} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'Epochs':>7} {'s/epoch':>8}")
        print(f"{'-'*W}")
        for label in sorted(results, key=lambda l: results[l]['val_accs'][-1], reverse=True):
            r = results[label]
            shape_str = 'x'.join(map(str, r['shape']))
            n_ep = len(r['train_losses']) - 1
            spe = r.get('sec_per_epoch', 0)
            print(f"{label:<16} {shape_str:>15} {r['proj_mode']:>7} {r['n_params']:>10,} {r['train_losses'][-1]:>11.4f} {r['train_accs'][-1]:>10.3f} {r['val_losses'][-1]:>10.4f} {r['val_accs'][-1]:>9.3f} {n_ep:>7} {spe:>7.1f}s")
        print(f"{'='*W}\n", flush=True)

    for label, shape, proj_mode in configs:
        torch.manual_seed(42)
        print(f"\n[{label}] Building model ({' x '.join(map(str, shape))}, proj={proj_mode})...", flush=True)
        model = Model(shape, vocab_size=vocab_size, n_layers=args.n_layers,
                      max_position_embeddings=args.seq_len, proj_mode=proj_mode)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{label}] {n_params:,} params, training {args.epochs} epochs on {args.device}...", flush=True)
        torch.manual_seed(42)  # same data sequence
        is_first = (label == configs[0][0])
        results[label] = train(model, args.epochs, task=args.task, B=args.batch_size,
                               L=args.seq_len, lr=args.lr, device=args.device, label=label,
                               vocab_size=vocab_size, num_kv_pairs=args.num_kv_pairs,
                               num_train_examples=args.num_train_examples,
                               use_compile=args.compile, use_amp=args.amp,
                               profile=(args.profile and is_first))
        results[label]['model'] = model
        results[label]['n_params'] = n_params
        results[label]['shape'] = shape
        results[label]['proj_mode'] = proj_mode
        print_leaderboard()

    all_labels = [label for label, _, _ in configs]

    # --- Final comparison ---
    max_ep = max(len(r['train_losses']) for r in results.values())

    print("\n" + "=" * 80)
    print("EPOCH-BY-EPOCH COMPARISON (every ~10 epochs)")
    print("=" * 80)
    # Print per-model blocks (too many models for one wide row)
    step = max(1, max_ep // 10)
    epochs_to_show = sorted(set(list(range(0, max_ep, step)) + [max_ep - 1]))
    for label in all_labels:
        r = results[label]
        shape_str = 'x'.join(map(str, r['shape']))
        print(f"\n  {label} ({shape_str}, proj={r['proj_mode']}):")
        print(f"  {'Epoch':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
        for ep in epochs_to_show:
            if ep < len(r['train_losses']):
                print(f"  {ep:>6} {r['train_losses'][ep]:>11.4f} {r['train_accs'][ep]:>10.3f} {r['val_losses'][ep]:>10.4f} {r['val_accs'][ep]:>9.3f}")

    print("\n" + "=" * 80)
    print("GRADIENT SVD ANALYSIS (effective rank = fraction of SVs > 10% of max)")
    print("=" * 80)
    for label in all_labels:
        r = results[label]
        snap = r['grad_snapshots']
        shape_str = 'x'.join(map(str, r['shape']))
        print(f"\n  {label} ({shape_str}):")
        for epoch in sorted(snap.keys()):
            for name, g in snap[epoch].items():
                S = svd_spectrum(g)
                eff_rank = (S > 0.1).sum().item()
                total = len(S)
                top5 = ', '.join(f'{s:.3f}' for s in S[:5].numpy())
                short = name.split('.')[-1]
                print(f"    epoch {epoch:>4} | {short:<12} | eff_rank {eff_rank}/{total} ({eff_rank/total:.1%}) | top5 SVs: [{top5}]")

    print("\n" + "=" * 80)
    print("WEIGHT SVD ANALYSIS (trained)")
    print("=" * 80)
    for label in all_labels:
        r = results[label]
        snap = r['weight_snapshots']
        final_epoch = r['snapshot_epochs'][-1]
        shape_str = 'x'.join(map(str, r['shape']))
        print(f"\n  {label} ({shape_str}):")
        if final_epoch in snap:
            for name, w in snap[final_epoch].items():
                S = svd_spectrum(w)
                eff_rank = (S > 0.1).sum().item()
                total = len(S)
                top5 = ', '.join(f'{s:.3f}' for s in S[:5].numpy())
                short = name.split('.')[-1]
                print(f"    {short:<12} | eff_rank {eff_rank}/{total} ({eff_rank/total:.1%}) | top5 SVs: [{top5}]")

    print("\n" + "=" * 80)
    print("FINAL LEADERBOARD")
    print("=" * 80)
    print_leaderboard()

    # --- Plot ---
    print("Plotting...")
    n_models = len(all_labels)

    # Layout: 2 rows — train loss + val loss on top, train acc + val acc on bottom
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    for label in all_labels:
        r = results[label]
        axes[0, 0].plot(r['train_losses'], alpha=0.8, label=label, linewidth=1.5)
        axes[0, 1].plot(r['val_losses'], alpha=0.8, label=label, linewidth=1.5)
        axes[1, 0].plot(r['train_accs'], alpha=0.8, label=label, linewidth=1.5)
        axes[1, 1].plot(r['val_accs'], alpha=0.8, label=label, linewidth=1.5)

    for ax, title in zip(axes.flat, ['Train Loss', 'Val Loss', 'Train Acc', 'Val Acc']):
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'ND Embedding/Projection Comparison on {task_name} (D={D})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
