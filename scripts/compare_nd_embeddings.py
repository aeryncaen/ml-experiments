#!/usr/bin/env python3
"""Compare ND embedding/projection structures on Shakespeare char-level LM.

Tests all combinations of:
  - Rank: 1D (flat linear), 2D, 3D, 4D (einsum over ND tensors)
  - Ratio: 1:3 (small x large), 1:1 (balanced), 3:1 (large x small)
  - Projection mode: einsum (true ND contraction) vs matmul (ND weight reshaped to 2D)

Architecture: pre-norm TransformerBlock, MHA, no MLP,
LayerNorm, position embeddings, tied embed/head, 0.02 init, cosine LR.

Usage:
    python scripts/compare_nd_embeddings.py --save results.png --report results.md
"""

import argparse
import math
import os
import time
import urllib.request

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from tqdm import tqdm


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_PATH = os.path.join(os.path.dirname(__file__), "shakespeare.txt")


def load_shakespeare():
    """Download and return Shakespeare text, with char-level encoding."""
    if not os.path.exists(SHAKESPEARE_PATH):
        print(f"Downloading Shakespeare to {SHAKESPEARE_PATH}...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, SHAKESPEARE_PATH)
    with open(SHAKESPEARE_PATH, 'r') as f:
        text = f.read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return data, stoi, itos, len(chars)





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
            self.Wqkv = nn.Linear(self.D, 3 * self.D, bias=False)
            self.out_proj = nn.Linear(self.D, self.D, bias=False)
            nn.init.normal_(self.Wqkv.weight, std=init_std)
            nn.init.normal_(self.out_proj.weight, std=out_std)
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
            return x_flat @ w.reshape(self.D, self.D)

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

        # Force bf16 for FlashAttention dispatch in SDPA
        orig_dtype = q.dtype
        if q.dtype == torch.float32 and q.is_cuda:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        context = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        context = context.to(orig_dtype)
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
    """Transformer LM: pre-norm blocks, MHA, no MLP, tied embed/head."""

    def __init__(self, shape, vocab_size=65, n_layers=2, num_heads=1,
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


def train(model, train_data, val_data, n_steps, B, L, lr, vocab_size, device, label,
          use_compile=False, use_amp=False, profile=False, eval_every=100):
    """Step-based training on char-level Shakespeare."""
    model = model.to(device)

    if use_compile:
        print(f'  [{label}] Compiling model with torch.compile()...', flush=True)
        model = torch.compile(model)

    opt = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_steps, eta_min=0.0)

    amp_dtype = torch.bfloat16 if (use_amp and device == 'cuda') else None
    amp_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype) if amp_dtype else None

    train_data = train_data.to(device)
    val_data = val_data.to(device)

    # Metrics at eval checkpoints: list of (step, train_loss, train_acc, val_loss, val_acc)
    metrics_log = []

    # Snapshot schedule
    snapshot_steps = sorted(set([0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps - 1]))
    grad_snapshots = {}
    weight_snapshots = {}

    def get_batch(data, batch_size, seq_len):
        ix = torch.randint(len(data) - seq_len - 1, (batch_size,), device=device)
        x = torch.stack([data[i:i+seq_len] for i in ix])
        y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
        return x, y

    def eval_loss(data, n_batches=20):
        model.eval()
        total_loss = torch.zeros(1, device=device)
        total_correct = torch.zeros(1, dtype=torch.long, device=device)
        total_count = torch.zeros(1, dtype=torch.long, device=device)
        with torch.no_grad():
            for _ in range(n_batches):
                x, y = get_batch(data, B, L)
                if amp_ctx is not None:
                    with amp_ctx:
                        logits = model(x)
                else:
                    logits = model(x)
                total_loss += F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
                total_correct += (logits.argmax(-1) == y).sum()
                total_count += y.numel()
        model.train()
        return (total_loss / n_batches).item(), (total_correct.float() / total_count).item()

    # Initial metrics
    tl0, ta0 = eval_loss(train_data)
    vl0, va0 = eval_loss(val_data)
    metrics_log.append((0, tl0, ta0, vl0, va0))
    print(f'  [{label}] start: train_loss={tl0:.4f} train_acc={ta0:.3f} val_loss={vl0:.4f} val_acc={va0:.3f}', flush=True)

    wall_start = time.perf_counter()
    model.train()
    pbar = tqdm(range(n_steps), desc=f'{label}', leave=True)

    for step in pbar:
        x, y = get_batch(train_data, B, L)

        if amp_ctx is not None:
            with amp_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        else:
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))

        opt.zero_grad()
        loss.backward()

        # Snapshot
        if step in snapshot_steps:
            grad_snapshots[step] = {}
            weight_snapshots[step] = {}
            for name, p in model.named_parameters():
                is_proj = any(k in name for k in ('wq', 'wk', 'wv', 'wo', 'Wqkv', 'out_proj'))
                if p.grad is not None and is_proj:
                    grad_snapshots[step][name] = p.grad.detach().cpu().clone()
                    weight_snapshots[step][name] = p.detach().cpu().clone()

        opt.step()
        scheduler.step()

        # Eval
        if (step + 1) % eval_every == 0 or step == n_steps - 1:
            tl, ta = eval_loss(train_data)
            vl, va = eval_loss(val_data)
            metrics_log.append((step + 1, tl, ta, vl, va))
            pbar.set_postfix(tl=f'{tl:.4f}', ta=f'{ta:.3f}', vl=f'{vl:.4f}', va=f'{va:.3f}')

    wall_elapsed = time.perf_counter() - wall_start
    print(f'  [{label}] Done: {n_steps} steps in {wall_elapsed:.1f}s ({wall_elapsed/n_steps*1000:.1f}ms/step)', flush=True)

    # Extract final metrics as lists for compatibility with analysis code
    steps_list = [m[0] for m in metrics_log]
    train_losses = [m[1] for m in metrics_log]
    train_accs = [m[2] for m in metrics_log]
    val_losses = [m[3] for m in metrics_log]
    val_accs = [m[4] for m in metrics_log]

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'steps': steps_list,
        'grad_snapshots': grad_snapshots,
        'weight_snapshots': weight_snapshots,
        'wall_time': wall_elapsed,
        'ms_per_step': wall_elapsed / n_steps * 1000,
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
    parser.add_argument('--dim', type=int, default=64,
                        help='Model dim')
    parser.add_argument('--steps', type=int, default=2000, help='Training steps')
    parser.add_argument('--seq-len', type=int, default=80, help='Sequence length')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--eval-every', type=int, default=100, help='Eval every N steps')
    parser.add_argument('--save', type=str, default=None, help='Save plot to file')
    parser.add_argument('--report', type=str, default=None, help='Write markdown report to file')
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

    # Load Shakespeare
    all_data, stoi, itos, vocab_size = load_shakespeare()
    n = int(0.9 * len(all_data))
    train_data = all_data[:n]
    val_data = all_data[n:]
    print(f"Shakespeare: {len(all_data):,} chars, vocab_size={vocab_size}, "
          f"train={len(train_data):,}, val={len(val_data):,}")

    D = args.dim
    task_name = f'Shakespeare char-LM (seq_len={args.seq_len})'

    # Build all model configs: (label, shape, proj_mode)
    configs = []

    # 1D baseline
    configs.append(('1D', (D,), 'linear'))

    # 2D/3D/4D with all three ratios, both einsum and matmul (skip duplicates)
    seen = set()
    for rank in [2, 3, 4]:
        for ratio in ['1:3', '1:1', '3:1']:
            shape = factorize_dim(D, rank, ratio=ratio)
            for proj_mode in ['einsum', 'matmul']:
                key = (shape, proj_mode)
                if key in seen:
                    print(f"  (skipping {rank}D-{ratio}-{proj_mode} = {'x'.join(map(str,shape))} — duplicate)")
                    continue
                seen.add(key)
                suffix = f'-{proj_mode}' if proj_mode == 'matmul' else ''
                label = f'{rank}D-{ratio}{suffix}'
                configs.append((label, shape, proj_mode))

    print(f"Task: {task_name}, vocab_size={vocab_size}, seq_len={args.seq_len}, batch={args.batch_size}")
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
        print(f"{'Model':<20} {'Shape':>15} {'Proj':>7} {'Params':>10} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'ms/step':>8}")
        print(f"{'-'*W}")
        for label in sorted(results, key=lambda l: results[l]['val_losses'][-1]):
            r = results[label]
            shape_str = 'x'.join(map(str, r['shape']))
            mps = r.get('ms_per_step', 0)
            print(f"{label:<20} {shape_str:>15} {r['proj_mode']:>7} {r['n_params']:>10,} {r['train_losses'][-1]:>11.4f} {r['train_accs'][-1]:>10.3f} {r['val_losses'][-1]:>10.4f} {r['val_accs'][-1]:>9.3f} {mps:>7.1f}")
        print(f"{'='*W}\n", flush=True)

    for label, shape, proj_mode in configs:
        torch.manual_seed(42)
        print(f"\n[{label}] Building model ({' x '.join(map(str, shape))}, proj={proj_mode})...", flush=True)
        model = Model(shape, vocab_size=vocab_size, n_layers=args.n_layers,
                      max_position_embeddings=args.seq_len, proj_mode=proj_mode)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{label}] {n_params:,} params, {args.steps} steps on {args.device}...", flush=True)
        torch.manual_seed(42)
        is_first = (label == configs[0][0])
        results[label] = train(model, train_data, val_data, n_steps=args.steps,
                               B=args.batch_size, L=args.seq_len, lr=args.lr,
                               vocab_size=vocab_size, device=args.device, label=label,
                               use_compile=args.compile, use_amp=args.amp,
                               profile=(args.profile and is_first),
                               eval_every=args.eval_every)
        results[label]['model'] = model
        results[label]['n_params'] = n_params
        results[label]['shape'] = shape
        results[label]['proj_mode'] = proj_mode
        print_leaderboard()

    all_labels = [label for label, _, _ in configs]

    # --- Final comparison ---
    print("\n" + "=" * 80)
    print("STEP-BY-STEP COMPARISON")
    print("=" * 80)
    for label in all_labels:
        r = results[label]
        shape_str = 'x'.join(map(str, r['shape']))
        steps = r.get('steps', list(range(len(r['train_losses']))))
        print(f"\n  {label} ({shape_str}, proj={r['proj_mode']}):")
        print(f"  {'Step':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9}")
        for i, s in enumerate(steps):
            print(f"  {s:>6} {r['train_losses'][i]:>11.4f} {r['train_accs'][i]:>10.3f} {r['val_losses'][i]:>10.4f} {r['val_accs'][i]:>9.3f}")

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
                print(f"    step {epoch:>4} | {short:<12} | eff_rank {eff_rank}/{total} ({eff_rank/total:.1%}) | top5 SVs: [{top5}]")

    print("\n" + "=" * 80)
    print("WEIGHT SVD ANALYSIS (evolution over training)")
    print("=" * 80)
    for label in all_labels:
        r = results[label]
        snap = r['weight_snapshots']
        shape_str = 'x'.join(map(str, r['shape']))
        print(f"\n  {label} ({shape_str}):")
        if snap:
            for epoch in sorted(snap.keys()):
                for name, w in snap[epoch].items():
                    S = svd_spectrum(w)
                    eff_rank = (S > 0.1).sum().item()
                    total = len(S)
                    top5 = ', '.join(f'{s:.3f}' for s in S[:5].numpy())
                    short = name.split('.')[-1]
                    print(f"    step {epoch:>4} | {short:<12} | eff_rank {eff_rank}/{total} ({eff_rank/total:.1%}) | top5 SVs: [{top5}]")
        else:
            print(f"    (no snapshots captured)")

    # --- ANALYSIS ---
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Group models
    baseline = results.get('1D')
    einsum_models = {l: r for l, r in results.items() if r['proj_mode'] == 'einsum'}
    matmul_models = {l: r for l, r in results.items() if r['proj_mode'] == 'matmul'}

    def final_metrics(r):
        return {
            'val_acc': r['val_accs'][-1],
            'val_loss': r['val_losses'][-1],
            'train_acc': r['train_accs'][-1],
            'train_loss': r['train_losses'][-1],
            'final_step': r.get('steps', list(range(len(r['train_losses']))))[-1],
        }

    # 1. ND einsum vs 1D baseline
    print("\n  1. ND EINSUM vs 1D BASELINE")
    print("  " + "-" * 60)
    if baseline:
        bm = final_metrics(baseline)
        print(f"     1D baseline: val_loss={bm['val_loss']:.4f}  val_acc={bm['val_acc']:.4f}")
        for label in sorted(einsum_models):
            r = einsum_models[label]
            m = final_metrics(r)
            loss_diff = m['val_loss'] - bm['val_loss']
            acc_diff = m['val_acc'] - bm['val_acc']
            sign_l = '+' if loss_diff >= 0 else ''
            sign_a = '+' if acc_diff >= 0 else ''
            print(f"     {label:<20} val_loss={m['val_loss']:.4f} ({sign_l}{loss_diff:.4f})  "
                  f"val_acc={m['val_acc']:.4f} ({sign_a}{acc_diff:.4f})")

        best_einsum = min(einsum_models.items(), key=lambda x: x[1]['val_losses'][-1])
        worst_einsum = max(einsum_models.items(), key=lambda x: x[1]['val_losses'][-1])
        print(f"\n     Best ND einsum:    {best_einsum[0]} (val_loss={best_einsum[1]['val_losses'][-1]:.4f})")
        print(f"     Worst ND einsum:   {worst_einsum[0]} (val_loss={worst_einsum[1]['val_losses'][-1]:.4f})")

        avg_einsum_loss = np.mean([r['val_losses'][-1] for r in einsum_models.values()])
        if avg_einsum_loss < bm['val_loss'] - 0.01:
            print(f"\n     VERDICT: ND einsum outperforms 1D (avg val_loss {avg_einsum_loss:.4f} vs {bm['val_loss']:.4f})")
        elif avg_einsum_loss > bm['val_loss'] + 0.01:
            print(f"\n     VERDICT: 1D outperforms ND einsum (avg val_loss {avg_einsum_loss:.4f} vs {bm['val_loss']:.4f})")
        else:
            print(f"\n     VERDICT: No meaningful difference (avg val_loss {avg_einsum_loss:.4f} vs {bm['val_loss']:.4f})")

    # 2. Einsum vs Matmul
    print("\n  2. EINSUM vs MATMUL (same weights, different projection)")
    print("  " + "-" * 60)
    if matmul_models:
        for label, r in sorted(matmul_models.items()):
            m = final_metrics(r)
            einsum_match = None
            for el, er in einsum_models.items():
                if er['shape'] == r['shape']:
                    einsum_match = (el, er)
                    break
            if einsum_match:
                el, er = einsum_match
                em = final_metrics(er)
                loss_diff = m['val_loss'] - em['val_loss']
                print(f"     {label:<20} val_loss={m['val_loss']:.4f}  "
                      f"vs {el}: {'+' if loss_diff>=0 else ''}{loss_diff:.4f}")
            else:
                print(f"     {label:<20} val_loss={m['val_loss']:.4f}")

        avg_matmul = np.mean([r['val_losses'][-1] for r in matmul_models.values()])
        avg_einsum = np.mean([r['val_losses'][-1] for r in einsum_models.values()])
        if abs(avg_matmul - avg_einsum) < 0.01:
            print(f"\n     VERDICT: Einsum and matmul equivalent — ND structure in weights alone doesn't help")
        elif avg_einsum < avg_matmul:
            print(f"\n     VERDICT: Einsum > matmul — ND contraction matters, not just ND weight shape")
        else:
            print(f"\n     VERDICT: Matmul >= einsum — ND contraction provides no benefit over reshape")

    # 3. Ratio comparison within each rank
    print("\n  3. RATIO COMPARISON (within each rank)")
    print("  " + "-" * 60)
    for rank in [2, 3, 4]:
        rank_models = {l: r for l, r in einsum_models.items() if len(r['shape']) == rank}
        if len(rank_models) < 2:
            continue
        best = min(rank_models.items(), key=lambda x: x[1]['val_losses'][-1])
        worst = max(rank_models.items(), key=lambda x: x[1]['val_losses'][-1])
        spread = worst[1]['val_losses'][-1] - best[1]['val_losses'][-1]
        print(f"     {rank}D: best={best[0]} (val_loss={best[1]['val_losses'][-1]:.4f})  "
              f"worst={worst[0]} (val_loss={worst[1]['val_losses'][-1]:.4f})  spread={spread:.4f}")
    
    # 4. Rank comparison (best of each rank)
    print("\n  4. RANK COMPARISON (best model per rank)")
    print("  " + "-" * 60)
    for rank in [1, 2, 3, 4]:
        if rank == 1:
            if baseline:
                bm = final_metrics(baseline)
                print(f"     1D: val_loss={bm['val_loss']:.4f}  val_acc={bm['val_acc']:.4f}")
        else:
            rank_models = {l: r for l, r in einsum_models.items() if len(r['shape']) == rank}
            if rank_models:
                best = min(rank_models.items(), key=lambda x: x[1]['val_losses'][-1])
                m = final_metrics(best[1])
                print(f"     {rank}D: val_loss={m['val_loss']:.4f}  val_acc={m['val_acc']:.4f}  ({best[0]})")

    # 5. SVD effective rank comparison
    print("\n  5. WEIGHT SVD EFFECTIVE RANK (final snapshot)")
    print("  " + "-" * 60)
    for label in all_labels:
        r = results[label]
        snap = r['weight_snapshots']
        if not snap:
            continue
        last_epoch = max(snap.keys())
        eff_ranks = []
        for name, w in snap[last_epoch].items():
            S = svd_spectrum(w)
            eff_ranks.append((S > 0.1).float().mean().item())
        if eff_ranks:
            avg_eff = np.mean(eff_ranks)
            shape_str = 'x'.join(map(str, r['shape']))
            print(f"     {label:<14} ({shape_str:>12}): avg eff_rank_frac={avg_eff:.3f}")

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
        ax.set_xlabel('eval checkpoint')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'ND Embedding/Projection Comparison on {task_name} (D={D})',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {args.save}")
    else:
        plt.show()

    # --- Markdown report ---
    if args.report:
        md = []
        md.append(f"# ND Embedding/Projection Comparison")
        md.append(f"")
        md.append(f"## Config")
        md.append(f"")
        md.append(f"| Setting | Value |")
        md.append(f"|---------|-------|")
        md.append(f"| Task | {task_name} |")
        md.append(f"| d_model | {D} |")
        md.append(f"| seq_len | {args.seq_len} |")
        md.append(f"| batch_size | {args.batch_size} |")
        md.append(f"| steps | {args.steps} |")
        md.append(f"| n_layers | {args.n_layers} |")
        md.append(f"| lr | {args.lr} |")
        md.append(f"| vocab_size | {vocab_size} |")
        md.append(f"| compile | {args.compile} |")
        md.append(f"| amp | {args.amp} |")
        md.append(f"")

        if args.save:
            md.append(f"![Training curves]({args.save})")
            md.append(f"")

        # Leaderboard
        md.append(f"## Leaderboard")
        md.append(f"")
        md.append(f"| Model | Shape | Proj | Params | Train Loss | Train Acc | Val Loss | Val Acc | ms/step |")
        md.append(f"|-------|-------|------|--------|------------|-----------|----------|---------|---------|")
        for label in sorted(results, key=lambda l: results[l]['val_losses'][-1]):
            r = results[label]
            shape_str = 'x'.join(map(str, r['shape']))
            mps = r.get('ms_per_step', 0)
            md.append(f"| {label} | {shape_str} | {r['proj_mode']} | {r['n_params']:,} | "
                      f"{r['train_losses'][-1]:.4f} | {r['train_accs'][-1]:.3f} | "
                      f"{r['val_losses'][-1]:.4f} | {r['val_accs'][-1]:.3f} | {mps:.1f} |")
        md.append(f"")

        # Model configs
        md.append(f"## Model Configs")
        md.append(f"")
        md.append(f"| Model | Shape | Proj | Params |")
        md.append(f"|-------|-------|------|--------|")
        for label, shape, proj_mode in configs:
            shape_str = 'x'.join(map(str, shape))
            n_params = results[label]['n_params'] if label in results else '?'
            md.append(f"| {label} | {shape_str} | {proj_mode} | {n_params:,} |")
        md.append(f"")

        # Analysis 1: ND vs 1D
        md.append(f"## Analysis")
        md.append(f"")
        md.append(f"### 1. ND Einsum vs 1D Baseline")
        md.append(f"")
        if baseline:
            bm = final_metrics(baseline)
            md.append(f"| Model | Val Loss | vs 1D | Val Acc | vs 1D |")
            md.append(f"|-------|----------|-------|---------|-------|")
            md.append(f"| **1D (baseline)** | {bm['val_loss']:.4f} | — | {bm['val_acc']:.4f} | — |")
            for label in sorted(einsum_models):
                r = einsum_models[label]
                m = final_metrics(r)
                ld = m['val_loss'] - bm['val_loss']
                ad = m['val_acc'] - bm['val_acc']
                md.append(f"| {label} | {m['val_loss']:.4f} | {'+' if ld>=0 else ''}{ld:.4f} | "
                          f"{m['val_acc']:.4f} | {'+' if ad>=0 else ''}{ad:.4f} |")
            md.append(f"")

            avg_einsum_loss = np.mean([r['val_losses'][-1] for r in einsum_models.values()])
            if avg_einsum_loss < bm['val_loss'] - 0.01:
                md.append(f"**Verdict:** ND einsum outperforms 1D (avg val_loss {avg_einsum_loss:.4f} vs {bm['val_loss']:.4f})")
            elif avg_einsum_loss > bm['val_loss'] + 0.01:
                md.append(f"**Verdict:** 1D outperforms ND einsum (avg val_loss {avg_einsum_loss:.4f} vs {bm['val_loss']:.4f})")
            else:
                md.append(f"**Verdict:** No meaningful difference (avg val_loss {avg_einsum_loss:.4f} vs {bm['val_loss']:.4f})")
            md.append(f"")

        # Analysis 2: Einsum vs Matmul
        md.append(f"### 2. Einsum vs Matmul")
        md.append(f"")
        if matmul_models:
            md.append(f"| Matmul Model | Val Loss | vs Einsum |")
            md.append(f"|--------------|----------|-----------|")
            for label, r in sorted(matmul_models.items()):
                m = final_metrics(r)
                einsum_match = None
                for el, er in einsum_models.items():
                    if er['shape'] == r['shape']:
                        einsum_match = (el, er)
                        break
                if einsum_match:
                    el, er = einsum_match
                    em = final_metrics(er)
                    ld = m['val_loss'] - em['val_loss']
                    md.append(f"| {label} | {m['val_loss']:.4f} | "
                              f"{'+' if ld>=0 else ''}{ld:.4f} ({el}) |")
                else:
                    md.append(f"| {label} | {m['val_loss']:.4f} | — |")
            md.append(f"")

            avg_matmul = np.mean([r['val_losses'][-1] for r in matmul_models.values()])
            avg_einsum = np.mean([r['val_losses'][-1] for r in einsum_models.values()])
            if abs(avg_matmul - avg_einsum) < 0.01:
                md.append(f"**Verdict:** Einsum and matmul equivalent — ND structure in weights alone doesn't help")
            elif avg_einsum < avg_matmul:
                md.append(f"**Verdict:** Einsum > matmul — ND contraction matters, not just ND weight shape")
            else:
                md.append(f"**Verdict:** Matmul >= einsum — ND contraction provides no benefit over reshape")
            md.append(f"")

        # Analysis 3: Ratio comparison
        md.append(f"### 3. Ratio Comparison")
        md.append(f"")
        md.append(f"| Rank | Best | Best Loss | Worst | Worst Loss | Spread |")
        md.append(f"|------|------|-----------|-------|------------|--------|")
        for rank in [2, 3, 4]:
            rank_models = {l: r for l, r in einsum_models.items() if len(r['shape']) == rank}
            if len(rank_models) < 2:
                continue
            best = min(rank_models.items(), key=lambda x: x[1]['val_losses'][-1])
            worst = max(rank_models.items(), key=lambda x: x[1]['val_losses'][-1])
            spread = worst[1]['val_losses'][-1] - best[1]['val_losses'][-1]
            md.append(f"| {rank}D | {best[0]} | {best[1]['val_losses'][-1]:.4f} | "
                      f"{worst[0]} | {worst[1]['val_losses'][-1]:.4f} | {spread:.4f} |")
        md.append(f"")

        # Analysis 4: Rank comparison
        md.append(f"### 4. Rank Comparison (best per rank)")
        md.append(f"")
        md.append(f"| Rank | Model | Val Loss | Val Acc |")
        md.append(f"|------|-------|----------|---------|")
        for rank in [1, 2, 3, 4]:
            if rank == 1:
                if baseline:
                    bm = final_metrics(baseline)
                    md.append(f"| 1D | 1D | {bm['val_loss']:.4f} | {bm['val_acc']:.4f} |")
            else:
                rank_models = {l: r for l, r in einsum_models.items() if len(r['shape']) == rank}
                if rank_models:
                    best = min(rank_models.items(), key=lambda x: x[1]['val_losses'][-1])
                    m = final_metrics(best[1])
                    md.append(f"| {rank}D | {best[0]} | {m['val_loss']:.4f} | {m['val_acc']:.4f} |")
        md.append(f"")

        # Analysis 5: SVD
        md.append(f"### 5. Weight SVD Effective Rank")
        md.append(f"")
        md.append(f"| Model | Shape | Avg Eff Rank Frac |")
        md.append(f"|-------|-------|-------------------|")
        for label in all_labels:
            r = results[label]
            snap = r['weight_snapshots']
            if not snap:
                continue
            last_epoch = max(snap.keys())
            eff_ranks = []
            for name, w in snap[last_epoch].items():
                S = svd_spectrum(w)
                eff_ranks.append((S > 0.1).float().mean().item())
            if eff_ranks:
                avg_eff = np.mean(eff_ranks)
                shape_str = 'x'.join(map(str, r['shape']))
                md.append(f"| {label} | {shape_str} | {avg_eff:.3f} |")
        md.append(f"")

        # Gradient SVD
        md.append(f"### Gradient SVD (evolution)")
        md.append(f"")
        for label in all_labels:
            r = results[label]
            snap = r['grad_snapshots']
            shape_str = 'x'.join(map(str, r['shape']))
            if not snap:
                continue
            md.append(f"**{label}** ({shape_str})")
            md.append(f"")
            md.append(f"| Step | Param | Eff Rank | Total | Frac | Top 5 SVs |")
            md.append(f"|------|-------|----------|-------|------|-----------|")
            for step_k in sorted(snap.keys()):
                for name, g in snap[step_k].items():
                    S = svd_spectrum(g)
                    eff_rank = (S > 0.1).sum().item()
                    total = len(S)
                    top5 = ', '.join(f'{s:.3f}' for s in S[:5].numpy())
                    short = name.split('.')[-1]
                    md.append(f"| {step_k} | {short} | {eff_rank} | {total} | {eff_rank/total:.1%} | {top5} |")
            md.append(f"")

        # Weight SVD evolution
        md.append(f"### Weight SVD (evolution)")
        md.append(f"")
        for label in all_labels:
            r = results[label]
            snap = r['weight_snapshots']
            shape_str = 'x'.join(map(str, r['shape']))
            if not snap:
                continue
            md.append(f"**{label}** ({shape_str})")
            md.append(f"")
            md.append(f"| Step | Param | Eff Rank | Total | Frac | Top 5 SVs |")
            md.append(f"|------|-------|----------|-------|------|-----------|")
            for step_k in sorted(snap.keys()):
                for name, w in snap[step_k].items():
                    S = svd_spectrum(w)
                    eff_rank = (S > 0.1).sum().item()
                    total = len(S)
                    top5 = ', '.join(f'{s:.3f}' for s in S[:5].numpy())
                    short = name.split('.')[-1]
                    md.append(f"| {step_k} | {short} | {eff_rank} | {total} | {eff_rank/total:.1%} | {top5} |")
            md.append(f"")

        # Step-by-step
        md.append(f"### Step-by-Step Metrics")
        md.append(f"")
        for label in all_labels:
            r = results[label]
            shape_str = 'x'.join(map(str, r['shape']))
            steps = r.get('steps', list(range(len(r['train_losses']))))
            md.append(f"**{label}** ({shape_str}, proj={r['proj_mode']})")
            md.append(f"")
            md.append(f"| Step | Train Loss | Train Acc | Val Loss | Val Acc |")
            md.append(f"|------|------------|-----------|----------|---------|")
            for i, s in enumerate(steps):
                md.append(f"| {s} | {r['train_losses'][i]:.4f} | {r['train_accs'][i]:.3f} | "
                          f"{r['val_losses'][i]:.4f} | {r['val_accs'][i]:.3f} |")
            md.append(f"")

        report_text = '\n'.join(md)
        with open(args.report, 'w') as f:
            f.write(report_text)
        print(f"Saved report to {args.report}")


if __name__ == '__main__':
    main()
