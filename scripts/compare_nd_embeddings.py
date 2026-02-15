#!/usr/bin/env python3
"""Compare 1D, 2D, 3D, and 4D embedding/projection structures on mod_arith.

Each model uses the same transformer skeleton (QKV seq-attn + feat-attn MLP)
but with projections of different tensor rank:

  1D: flat nn.Linear(D, D) — standard transformer
  2D: einsum('...cd, cdef -> ...ef')       w: (A,B, A,B)
  3D: einsum('...cde, cdefgh -> ...fgh')   w: (A,B,C, A,B,C)
  4D: einsum('...cdef, cdefghij -> ...ghij') w: (A,B,C,D, A,B,C,D)

All models have the same d_model and approximately matched param counts.
Trains on cumulative modular arithmetic (short sequences), then compares:

  1. Training loss curves
  2. Validation accuracy curves
  3. Gradient SVD spectra (how structured are the gradients)
  4. Learned weight SVD spectra

Usage:
    python scripts/compare_nd_embeddings.py [--dim 64] [--epochs 100] [--save path.png]
"""

import argparse
import math

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


def factorize_dim(D, rank):
    """Factor D into `rank` dimensions using 1:3 ratio structure.

    Mirrors the ULB architecture convention: first dimension is the number
    of features (small), remaining dimensions form the descriptor (large).
    Target ratio is 1:3 (n_features : desc_dim) for rank 2.
    For rank > 2, the descriptor is further factored into balanced parts.

    Examples for D=4096:
        1D: (4096,)
        2D: (32, 128)       — 32 features x 128 descriptor
        3D: (32, 8, 16)     — 32 features x (8x16) descriptor
        4D: (32, 4, 4, 8)   — 32 features x (4x4x8) descriptor
    """
    if rank == 1:
        return (D,)
    # First dim: target 1:3 ratio -> n_features ≈ sqrt(D/3)
    target_nf = int((D / 3) ** 0.5)
    best = 1
    for s in range(2, int(D ** 0.5) + 1):
        if D % s == 0:
            if abs(s - target_nf) <= abs(best - target_nf):
                best = s
    nf = best
    desc = D // nf
    if rank == 2:
        return (nf, desc)
    # For rank > 2, factor descriptor into (rank-1) balanced parts
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

    def __init__(self, shape, n_layers=2, layer_idx=0, num_heads=1, dropout=0.1):
        super().__init__()
        self.shape = shape
        self.rank = len(shape)
        self.D = math.prod(shape)
        self.num_heads = num_heads
        self.head_dim = self.D // num_heads
        self.n_layers = n_layers

        init_std = 0.02
        out_std = init_std / math.sqrt(2 * n_layers)

        if self.rank == 1:
            self.Wqkv = nn.Linear(self.D, 3 * self.D)
            self.out_proj = nn.Linear(self.D, self.D)
            nn.init.normal_(self.Wqkv.weight, std=init_std)
            nn.init.zeros_(self.Wqkv.bias)
            nn.init.normal_(self.out_proj.weight, std=out_std)
            nn.init.zeros_(self.out_proj.bias)
        else:
            n = self.rank
            in_chars = ''.join(chr(ord('c') + i) for i in range(n))
            out_chars = ''.join(chr(ord('c') + n + i) for i in range(n))
            self.einsum_str = f'...{in_chars},{in_chars}{out_chars}->...{out_chars}'

            w_shape = tuple(shape) + tuple(shape)
            # Wqkv: 3 separate ND projections for Q, K, V
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

    def _proj_nd(self, x, w):
        return torch.einsum(self.einsum_str, x, w)

    def forward(self, hidden_states, residual=None):
        # Matches Zoology TransformerBlock.forward exactly
        B, T = hidden_states.shape[:2]
        D, NH, HD = self.D, self.num_heads, self.head_dim

        # --- sequence mixer (MHA) ---
        dropped = self.dropout1(hidden_states.reshape(B, T, D))
        residual = (dropped + residual) if residual is not None else dropped
        h = self.norm1(residual)

        if self.rank == 1:
            qkv = self.Wqkv(h)
            qkv = qkv.reshape(B, T, 3, NH, HD)
            q, k, v = qkv.unbind(dim=2)
            # Manual causal attention matching Zoology SelfAttention
            softmax_scale = 1.0 / math.sqrt(HD)
            scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
            causal_mask = torch.triu(
                torch.full((T, T), -10000.0, device=scores.device), 1
            )
            scores = scores + causal_mask.to(dtype=scores.dtype)
            attn = torch.softmax(scores, dim=-1, dtype=v.dtype)
            attn = F.dropout(attn, self.attn_dropout if self.training else 0.0)
            context = torch.einsum("bhts,bshd->bthd", attn, v)
            hidden_states = self.out_proj(context.reshape(B, T, D))
        else:
            h_nd = h.view(B, T, *self.shape)
            Q = self._proj_nd(h_nd, self.wq)
            K = self._proj_nd(h_nd, self.wk)
            V = self._proj_nd(h_nd, self.wv)
            q = Q.reshape(B, T, NH, HD)
            k = K.reshape(B, T, NH, HD)
            v = V.reshape(B, T, NH, HD)
            softmax_scale = 1.0 / math.sqrt(HD)
            scores = torch.einsum("bthd,bshd->bhts", q, k * softmax_scale)
            causal_mask = torch.triu(
                torch.full((T, T), -10000.0, device=scores.device), 1
            )
            scores = scores + causal_mask.to(dtype=scores.dtype)
            attn = torch.softmax(scores, dim=-1, dtype=v.dtype)
            attn = F.dropout(attn, self.attn_dropout if self.training else 0.0)
            context = torch.einsum("bhts,bshd->bthd", attn, v)
            context_nd = context.reshape(B, T, *self.shape)
            hidden_states = self._proj_nd(context_nd, self.wo).reshape(B, T, D)

        # --- state mixer (Identity) ---
        dropped = self.dropout2(hidden_states)
        residual = (dropped + residual) if residual is not None else dropped
        hidden_states = self.norm2(residual)
        # state_mixer is Identity, so hidden_states passes through

        return hidden_states, residual


class Model(nn.Module):
    """Matches Zoology LanguageModel + LMBackbone structure."""

    def __init__(self, shape, vocab_size=VOCAB_SIZE, n_layers=2, num_heads=1,
                 max_position_embeddings=64, dropout=0.1):
        super().__init__()
        self.shape = shape
        self.rank = len(shape)
        self.D = math.prod(shape)

        # Token + position embeddings (matching Zoology TokenEmbeddings)
        self.word_embeddings = nn.Embedding(vocab_size, self.D)
        self.position_embeddings = nn.Embedding(max_position_embeddings, self.D)

        self.blocks = nn.ModuleList([
            BlockND(shape, n_layers=n_layers, layer_idx=i,
                    num_heads=num_heads, dropout=dropout)
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
          vocab_size=8192, num_kv_pairs=4, num_train_examples=100_000):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=0.0)

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

    val_x, val_t = val_x.to(device), val_t.to(device)
    n_batches = num_train_examples // B

    # Helper to compute loss + acc over a dataset in batches
    def eval_metrics(data_x, data_t):
        model.eval()
        total_loss = 0
        total_correct = 0
        total_count = 0
        bs = min(512, len(data_x))
        with torch.no_grad():
            for i in range(0, len(data_x), bs):
                bx = data_x[i:i+bs].to(device) if not data_x.is_cuda and device != 'cpu' else data_x[i:i+bs]
                bt = data_t[i:i+bs].to(device) if not data_t.is_cuda and device != 'cpu' else data_t[i:i+bs]
                logits = model(bx)
                total_loss += F.cross_entropy(logits.reshape(-1, vocab_size), bt.reshape(-1),
                                              ignore_index=-100).item() * len(bx)
                preds = logits.argmax(-1)
                if task == 'mqar':
                    m = bt != -100
                    total_correct += (preds[m] == bt[m]).sum().item()
                    total_count += m.sum().item()
                else:
                    total_correct += (preds == bt).sum().item()
                    total_count += bt.numel()
        return total_loss / len(data_x), total_correct / max(total_count, 1)

    # Register starting metrics before any training
    tl0, ta0 = eval_metrics(train_x[:min(5000, len(train_x))], train_t[:min(5000, len(train_x))])
    vl0, va0 = eval_metrics(val_x, val_t)
    train_losses.append(tl0); train_accs.append(ta0)
    val_losses.append(vl0); val_accs.append(va0)
    print(f'  [{label}] start: train_loss={tl0:.4f} train_acc={ta0:.3f} val_loss={vl0:.4f} val_acc={va0:.3f}', flush=True)

    pbar = tqdm(range(n_epochs), desc=f'{label} training', leave=True)
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_count = 0

        # Shuffle training data each epoch
        perm = torch.randperm(num_train_examples)
        train_x_shuf = train_x[perm]
        train_t_shuf = train_t[perm]

        for batch_i in range(n_batches):
            x = train_x_shuf[batch_i * B : (batch_i + 1) * B].to(device)
            t = train_t_shuf[batch_i * B : (batch_i + 1) * B].to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), t.reshape(-1),
                                   ignore_index=-100)
            opt.zero_grad()
            loss.backward()

            # Accumulate train acc from training logits (free, no extra forward pass)
            with torch.no_grad():
                preds = logits.argmax(-1)
                if task == 'mqar':
                    m = t != -100
                    epoch_correct += (preds[m] == t[m]).sum().item()
                    epoch_count += m.sum().item()
                else:
                    epoch_correct += (preds == t).sum().item()
                    epoch_count += t.numel()

            # Snapshot (first batch of snapshot epochs only)
            if epoch in snapshot_epochs and batch_i == 0:
                if epoch not in grad_snapshots:
                    grad_snapshots[epoch] = {}
                    weight_snapshots[epoch] = {}
                for name, p in model.named_parameters():
                    if p.grad is not None and ('wq' in name or 'seq_wq' in name):
                        if name not in grad_snapshots[epoch]:
                            grad_snapshots[epoch][name] = p.grad.detach().cpu().clone()
                            weight_snapshots[epoch][name] = p.detach().cpu().clone()

            opt.step()
            epoch_loss += loss.item()

        scheduler.step()
        train_losses.append(epoch_loss / n_batches)
        train_accs.append(epoch_correct / max(epoch_count, 1))

        # Val eval
        vl_, va_ = eval_metrics(val_x, val_t)
        val_losses.append(vl_)
        val_accs.append(va_)

        pbar.set_postfix(tl=f'{train_losses[-1]:.4f}', ta=f'{train_accs[-1]:.3f}',
                         vl=f'{vl_:.4f}', va=f'{va_:.3f}')

        # Early stop if solved
        if va_ > 0.99:
            print(f'  [{label}] Early stop at epoch {epoch} — val_acc={va_:.3f}', flush=True)
            break

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'grad_snapshots': grad_snapshots,
        'weight_snapshots': weight_snapshots,
        'snapshot_epochs': snapshot_epochs,
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
    parser.add_argument('--dim', type=int, default=4096,
                        help='Model dim (must be factorable into 2/3/4-way)')
    parser.add_argument('--task', type=str, default='mqar', choices=['mqar', 'mod_arith'],
                        help='Task: mqar (associative recall) or mod_arith')
    parser.add_argument('--epochs', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=64, help='Sequence length')
    parser.add_argument('--num-kv-pairs', type=int, default=4,
                        help='Number of key-value pairs (mqar only)')
    parser.add_argument('--num-train-examples', type=int, default=100_000,
                        help='Number of training examples (pre-generated)')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--device', type=str, default=None,
                        help='Device (default: auto-detect cuda/mps/cpu)')
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
    print(f"Using device: {args.device}")

    D = args.dim
    vocab_size = VOCAB_SIZE if args.task == 'mqar' else MOD_BASE
    task_name = 'MQAR (associative recall)' if args.task == 'mqar' else 'Modular Arithmetic'

    # Compute factorizations
    shapes = {}
    shapes['1D'] = (D,)
    shapes['2D'] = factorize_dim(D, 2)
    shapes['3D'] = factorize_dim(D, 3)
    shapes['4D'] = factorize_dim(D, 4)

    print(f"Task: {task_name}, vocab_size={vocab_size}, seq_len={args.seq_len}")
    print(f"d_model = {D}")
    for label, shape in shapes.items():
        print(f"  {label}: {' x '.join(map(str, shape))} = {math.prod(shape)}")

    colors = {'1D': 'C0', '2D': 'C1', '3D': 'C2', '4D': 'C3'}
    results = {}

    def print_leaderboard():
        if not results:
            return
        print(f"\n{'='*90}")
        print(f"{'Model':<8} {'Shape':>15} {'Params':>10} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'Epoch':>6}")
        print(f"{'-'*90}")
        for label in sorted(results, key=lambda l: results[l]['val_accs'][-1], reverse=True):
            r = results[label]
            shape_str = 'x'.join(map(str, r['shape']))
            n_ep = len(r['train_losses']) - 1  # -1 because index 0 is pre-training
            print(f"{label:<8} {shape_str:>15} {r['n_params']:>10,} {r['train_losses'][-1]:>11.4f} {r['train_accs'][-1]:>10.3f} {r['val_losses'][-1]:>10.4f} {r['val_accs'][-1]:>9.3f} {n_ep:>6}")
        print(f"{'='*90}\n", flush=True)

    for label, shape in shapes.items():
        torch.manual_seed(42)
        print(f"\n[{label}] Building model ({' x '.join(map(str, shape))})...", flush=True)
        model = Model(shape, vocab_size=vocab_size, n_layers=args.n_layers,
                      max_position_embeddings=args.seq_len)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{label}] {n_params:,} params, training {args.epochs} epochs on {args.device}...", flush=True)
        torch.manual_seed(42)  # same data sequence
        results[label] = train(model, args.epochs, task=args.task, B=args.batch_size,
                               L=args.seq_len, lr=args.lr, device=args.device, label=label,
                               vocab_size=vocab_size, num_kv_pairs=args.num_kv_pairs,
                               num_train_examples=args.num_train_examples)
        results[label]['model'] = model
        results[label]['n_params'] = n_params
        results[label]['shape'] = shape
        print_leaderboard()

    # --- Final comparison ---
    # Find max epoch count across all models (some may early-stop)
    max_ep = max(len(r['train_losses']) for r in results.values())

    print("\n" + "=" * 120)
    print("EPOCH-BY-EPOCH COMPARISON (every ~10 epochs)")
    print("=" * 120)
    header = f"{'Epoch':>6}"
    for label in shapes:
        header += f" | {label+' tl':>8} {label+' ta':>7} {label+' vl':>8} {label+' va':>7}"
    print(header)
    print("-" * 120)
    step = max(1, max_ep // 10)
    epochs_to_show = sorted(set(list(range(0, max_ep, step)) + [max_ep - 1]))
    for ep in epochs_to_show:
        row = f"{ep:>6}"
        for label in shapes:
            r = results[label]
            if ep < len(r['train_losses']):
                row += f" | {r['train_losses'][ep]:>8.4f} {r['train_accs'][ep]:>7.3f} {r['val_losses'][ep]:>8.4f} {r['val_accs'][ep]:>7.3f}"
            else:
                row += f" | {'--':>8} {'--':>7} {'--':>8} {'--':>7}"
        print(row)

    print("\n" + "=" * 80)
    print("GRADIENT SVD ANALYSIS (effective rank = fraction of SVs > 10% of max)")
    print("=" * 80)
    for label in shapes:
        snap = results[label]['grad_snapshots']
        print(f"\n  {label} ({' x '.join(map(str, shapes[label]))}):")
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
    for label in shapes:
        snap = results[label]['weight_snapshots']
        final_epoch = results[label]['snapshot_epochs'][-1]
        print(f"\n  {label} ({' x '.join(map(str, shapes[label]))}):")
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
    rank_labels = list(shapes.keys())  # ['1D', '2D', '3D', '4D']
    n_ranks = len(rank_labels)

    # Layout: 3 rows
    #   Row 0: loss curve, val acc curve, final acc bar chart (shared across all)
    #   Row 1: one grad SVD subplot per rank (4 columns)
    #   Row 2: one weight SVD subplot per rank (4 columns)
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, n_ranks, hspace=0.4, wspace=0.3,
                           height_ratios=[1, 1, 1])

    # Row 0, col 0-1: Training loss + Val accuracy (span 2 cols each)
    ax_loss = fig.add_subplot(gs[0, :n_ranks // 2])
    for label in rank_labels:
        ax_loss.plot(results[label]['train_losses'], color=colors[label], alpha=0.8,
                     label=f"{label}", linewidth=2)
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.set_title('Training Loss')
    ax_loss.legend(fontsize=10)
    ax_loss.grid(True, alpha=0.3)

    ax_acc = fig.add_subplot(gs[0, n_ranks // 2:])
    for label in rank_labels:
        ax_acc.plot(results[label]['val_accs'], color=colors[label], alpha=0.8,
                    label=label, linewidth=2)
    ax_acc.set_xlabel('epoch')
    ax_acc.set_ylabel('accuracy')
    ax_acc.set_title('Validation Accuracy')
    ax_acc.legend(fontsize=10)
    ax_acc.grid(True, alpha=0.3)

    # Row 1: Gradient SVD — one subplot per rank
    mid_epochs = {}
    for label in rank_labels:
        snaps = results[label]['snapshot_epochs']
        mid_epochs[label] = snaps[len(snaps) // 2]

    for i, label in enumerate(rank_labels):
        ax = fig.add_subplot(gs[1, i])
        snap = results[label]['grad_snapshots']
        ep = mid_epochs[label]
        if ep in snap:
            for name, g in snap[ep].items():
                S = svd_spectrum(g)
                short = name.split('.')[-1]
                ax.semilogy(S.numpy(), linewidth=1.5, alpha=0.8, label=short)
        ax.set_title(f'{label} Grad SVD (ep {ep})', fontsize=10)
        ax.set_xlabel('SV index')
        if i == 0:
            ax.set_ylabel('normalized SV')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Row 2: Weight SVD — one subplot per rank
    for i, label in enumerate(rank_labels):
        ax = fig.add_subplot(gs[2, i])
        snap = results[label]['weight_snapshots']
        final_ep = results[label]['snapshot_epochs'][-1]
        if final_ep in snap:
            for name, w in snap[final_ep].items():
                S = svd_spectrum(w)
                short = name.split('.')[-1]
                eff = (S > 0.1).sum().item()
                ax.semilogy(S.numpy(), linewidth=1.5, alpha=0.8,
                           label=f'{short} (rank {eff})')
        ax.set_title(f'{label} Weight SVD (trained)', fontsize=10)
        ax.set_xlabel('SV index')
        if i == 0:
            ax.set_ylabel('normalized SV')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    shape_strs = [f"{l}: {'x'.join(map(str, shapes[l]))}" for l in rank_labels]
    fig.suptitle(f'1D vs 2D vs 3D vs 4D Projection Structure on {task_name} (D={D})\n'
                 f'{" | ".join(shape_strs)}',
                 fontsize=13, fontweight='bold')

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
