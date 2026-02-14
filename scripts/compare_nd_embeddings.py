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


VOCAB_SIZE = 64
MOD_BASE = 5


def gen_mod_arith(B, L, device='cpu'):
    x = torch.randint(0, MOD_BASE, (B, L), device=device)
    target = x.cumsum(dim=1) % MOD_BASE
    return x, target


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.w = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


# ---------------------------------------------------------------------------
# Factorization helpers
# ---------------------------------------------------------------------------

def factorize_dim(D, rank):
    """Find a factorization of D into `rank` factors, as balanced as possible.

    Returns tuple of ints whose product == D.
    For rank=1, returns (D,).
    For rank=2, finds closest pair (a, b) with a <= b, a*b == D.
    For rank=3, finds (a, b, c) with a*b*c == D.
    For rank=4, finds (a, b, c, d) with a*b*c*d == D.
    """
    if rank == 1:
        return (D,)
    if rank == 2:
        best = (1, D)
        for a in range(2, int(D**0.5) + 1):
            if D % a == 0:
                best = (a, D // a)
        return best
    if rank == 3:
        best = None
        best_spread = float('inf')
        for a in range(2, int(D**(1/3)) + 2):
            if D % a != 0:
                continue
            rem = D // a
            for b in range(a, int(rem**0.5) + 1):
                if rem % b == 0:
                    c = rem // b
                    spread = max(a, b, c) - min(a, b, c)
                    if best is None or spread < best_spread:
                        best = (a, b, c)
                        best_spread = spread
        if best is None:
            # Fallback: use 2D factorization + split
            a, bc = factorize_dim(D, 2)
            b, c = factorize_dim(bc, 2)
            best = (a, b, c)
        return best
    if rank == 4:
        best = None
        best_spread = float('inf')
        for a in range(2, int(D**(1/4)) + 2):
            if D % a != 0:
                continue
            rem = D // a
            for b in range(a, int(rem**(1/3)) + 2):
                if rem % b != 0:
                    continue
                rem2 = rem // b
                for c in range(b, int(rem2**0.5) + 1):
                    if rem2 % c == 0:
                        d = rem2 // c
                        spread = max(a, b, c, d) - min(a, b, c, d)
                        if best is None or spread < best_spread:
                            best = (a, b, c, d)
                            best_spread = spread
        if best is None:
            ab, cd = factorize_dim(D, 2)
            a, b = factorize_dim(ab, 2)
            c, d = factorize_dim(cd, 2)
            best = (a, b, c, d)
        return best


# ---------------------------------------------------------------------------
# Generic ND projection block
# ---------------------------------------------------------------------------

class BlockND(nn.Module):
    """Causal self-attention + SwiGLU MLP using N-dimensional tensor projections.

    All ranks use the same architecture: QKV seq-attn + SwiGLU MLP.
    Only the projection structure changes:
      1D: nn.Linear (flat matrix multiply)
      ND: einsum with rank-2N tensor weights
    """

    def __init__(self, D, shape, n_heads=4, ffn_expand=2):
        super().__init__()
        self.D = D
        self.shape = shape
        self.rank = len(shape)
        self.n_heads = n_heads if self.rank == 1 else shape[0]
        self.head_dim = D // self.n_heads

        init_scale = D ** -0.5
        mlp_inner = int(D * ffn_expand)
        self.mlp_inner = mlp_inner

        if self.rank == 1:
            # Flat linear projections
            self.wq = nn.Linear(D, D, bias=False)
            self.wk = nn.Linear(D, D, bias=False)
            self.wv = nn.Linear(D, D, bias=False)
            self.wo = nn.Linear(D, D, bias=False)
            self.gate_proj = nn.Linear(D, mlp_inner, bias=False)
            self.up_proj = nn.Linear(D, mlp_inner, bias=False)
            self.down_proj = nn.Linear(mlp_inner, D, bias=False)
        else:
            # ND tensor projections via einsum
            n = self.rank
            in_chars = ''.join(chr(ord('c') + i) for i in range(n))
            out_chars = ''.join(chr(ord('c') + n + i) for i in range(n))
            self.einsum_str = f'...{in_chars},{in_chars}{out_chars}->...{out_chars}'

            w_shape = tuple(shape) + tuple(shape)
            self.seq_wq = nn.Parameter(torch.randn(*w_shape) * init_scale)
            self.seq_wk = nn.Parameter(torch.randn(*w_shape) * init_scale)
            self.seq_wv = nn.Parameter(torch.randn(*w_shape) * init_scale)
            self.seq_wo = nn.Parameter(torch.randn(*w_shape) * init_scale)

            # SwiGLU MLP with ND tensor projections
            # up/gate: (shape...) -> mlp_inner, down: mlp_inner -> (shape...)
            # Factor mlp_inner into same-rank shape
            mlp_shape = factorize_dim(mlp_inner, self.rank)
            self.mlp_shape = mlp_shape

            mlp_up_shape = tuple(shape) + tuple(mlp_shape)
            mlp_down_shape = tuple(mlp_shape) + tuple(shape)
            mlp_up_chars = ''.join(chr(ord('c') + n + i) for i in range(n))
            self.mlp_einsum_up = f'...{in_chars},{in_chars}{mlp_up_chars}->...{mlp_up_chars}'
            self.mlp_einsum_down = f'...{mlp_up_chars},{mlp_up_chars}{in_chars}->...{in_chars}'

            mlp_init = math.prod(mlp_shape) ** -0.5
            self.mlp_gate = nn.Parameter(torch.randn(*mlp_up_shape) * init_scale)
            self.mlp_up = nn.Parameter(torch.randn(*mlp_up_shape) * init_scale)
            self.mlp_down = nn.Parameter(torch.randn(*mlp_down_shape) * mlp_init)

        self.norm1 = RMSNorm(D)
        self.norm2 = RMSNorm(D)

    def _proj_nd(self, x, w, einsum_str=None):
        return torch.einsum(einsum_str or self.einsum_str, x, w)

    def forward(self, x):
        B, T, D = x.shape
        NH, HD = self.n_heads, self.head_dim

        if self.rank == 1:
            # --- Seq-attn with flat linear ---
            h = self.norm1(x)
            q = self.wq(h).view(B, T, NH, HD).transpose(1, 2)
            k = self.wk(h).view(B, T, NH, HD).transpose(1, 2)
            v = self.wv(h).view(B, T, NH, HD).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            y = y.transpose(1, 2).contiguous().view(B, T, D)
            x = x + self.wo(y)
            # --- SwiGLU MLP with flat linear ---
            r = self.norm2(x)
            x = x + self.down_proj(F.silu(self.gate_proj(r)) * self.up_proj(r))
            return x

        # --- ND path: same architecture, tensor projections ---
        shape = self.shape

        # Seq-attn
        h = self.norm1(x).view(B, T, *shape)
        Q = self._proj_nd(h, self.seq_wq)
        K = self._proj_nd(h, self.seq_wk)
        V = self._proj_nd(h, self.seq_wv)
        Q = Q.view(B, T, NH, -1).permute(0, 2, 1, 3)
        K = K.view(B, T, NH, -1).permute(0, 2, 1, 3)
        V = V.view(B, T, NH, -1).permute(0, 2, 1, 3)
        y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, *shape)
        y = self._proj_nd(y, self.seq_wo)
        x = x + y.reshape(B, T, D)

        # SwiGLU MLP
        r = self.norm2(x).view(B, T, *shape)
        gate = self._proj_nd(r, self.mlp_gate, self.mlp_einsum_up)
        up = self._proj_nd(r, self.mlp_up, self.mlp_einsum_up)
        h = F.silu(gate) * up
        down = self._proj_nd(h, self.mlp_down, self.mlp_einsum_down)
        x = x + down.reshape(B, T, D)

        return x


class Model(nn.Module):
    def __init__(self, D, shape, n_layers=1, n_heads=4, ffn_expand=2):
        super().__init__()
        self.D = D
        self.shape = shape
        self.embed = nn.Embedding(VOCAB_SIZE, D)
        self.blocks = nn.ModuleList([
            BlockND(D, shape, n_heads=n_heads, ffn_expand=ffn_expand)
            for _ in range(n_layers)
        ])
        self.final_norm = RMSNorm(D)
        self.head = nn.Linear(D, VOCAB_SIZE, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for block in self.blocks:
            h = block(h)
        return self.head(self.final_norm(h))


def train(model, n_epochs, B=64, L=32, lr=3e-4, device='cpu', label=''):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_accs = []

    # Gradient snapshots
    grad_snapshots = {}
    weight_snapshots = {}
    snapshot_epochs = sorted(set([0, n_epochs // 4, n_epochs // 2,
                                   3 * n_epochs // 4, n_epochs - 1]))

    val_x, val_t = gen_mod_arith(512, L, device=device)

    pbar = tqdm(range(n_epochs), desc=f'{label} training', leave=True)
    for epoch in pbar:
        model.train()
        epoch_loss = 0
        n_batches = 20

        for _ in range(n_batches):
            x, t = gen_mod_arith(B, L, device=device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), t.reshape(-1))
            opt.zero_grad()
            loss.backward()

            # Snapshot
            if epoch in snapshot_epochs:
                if epoch not in grad_snapshots:
                    grad_snapshots[epoch] = {}
                    weight_snapshots[epoch] = {}
                for name, p in model.named_parameters():
                    if p.grad is not None and ('wq' in name or 'seq_wq' in name or 'feat_wq' in name):
                        if name not in grad_snapshots[epoch]:
                            grad_snapshots[epoch][name] = p.grad.detach().cpu().clone()
                            weight_snapshots[epoch][name] = p.detach().cpu().clone()

            opt.step()
            epoch_loss += loss.item()

        train_losses.append(epoch_loss / n_batches)

        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            preds = logits.argmax(-1)
            val_acc = (preds == val_t).float().mean().item()
            val_accs.append(val_acc)

        pbar.set_postfix(train_loss=f'{train_losses[-1]:.4f}', val_acc=f'{val_acc:.3f}')

    return {
        'train_losses': train_losses,
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
    parser.add_argument('--epochs', type=int, default=100)
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

    # Compute factorizations
    shapes = {}
    shapes['1D'] = (D,)
    shapes['2D'] = factorize_dim(D, 2)
    shapes['3D'] = factorize_dim(D, 3)
    shapes['4D'] = factorize_dim(D, 4)

    print(f"d_model = {D}")
    for label, shape in shapes.items():
        print(f"  {label}: {' x '.join(map(str, shape))} = {math.prod(shape)}")

    colors = {'1D': 'C0', '2D': 'C1', '3D': 'C2', '4D': 'C3'}
    results = {}

    def print_leaderboard():
        if not results:
            return
        print(f"\n{'='*70}")
        print(f"{'Model':<8} {'Shape':>20} {'Params':>12} {'Final Loss':>12} {'Val Acc':>10}")
        print(f"{'-'*70}")
        for label in sorted(results, key=lambda l: results[l]['val_accs'][-1], reverse=True):
            r = results[label]
            shape_str = 'x'.join(map(str, r['shape']))
            print(f"{label:<8} {shape_str:>20} {r['n_params']:>12,} {r['train_losses'][-1]:>12.4f} {r['val_accs'][-1]:>10.3f}")
        print(f"{'='*70}\n", flush=True)

    for label, shape in shapes.items():
        torch.manual_seed(42)
        print(f"\n[{label}] Building model ({' x '.join(map(str, shape))})...", flush=True)
        model = Model(D, shape, n_layers=args.n_layers, ffn_expand=2)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"[{label}] {n_params:,} params, training {args.epochs} epochs on {args.device}...", flush=True)
        torch.manual_seed(42)  # same data sequence
        results[label] = train(model, args.epochs, lr=args.lr, device=args.device, label=label)
        results[label]['model'] = model
        results[label]['n_params'] = n_params
        results[label]['shape'] = shape
        print_leaderboard()

    print("FINAL RESULTS:")
    print_leaderboard()

    # --- Plot ---
    print("Plotting...")
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # (0,0): Training loss
    ax = fig.add_subplot(gs[0, 0])
    for label in shapes:
        ax.plot(results[label]['train_losses'], color=colors[label], alpha=0.8,
                label=f"{label} ({results[label]['n_params']:,}p)")
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('Training Loss')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,1): Val accuracy
    ax = fig.add_subplot(gs[0, 1])
    for label in shapes:
        ax.plot(results[label]['val_accs'], color=colors[label], alpha=0.8, label=label)
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.set_title('Validation Accuracy')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (0,2): Final val accuracy bar chart
    ax = fig.add_subplot(gs[0, 2])
    labels = list(shapes.keys())
    final_accs = [results[l]['val_accs'][-1] for l in labels]
    bars = ax.bar(labels, final_accs, color=[colors[l] for l in labels], alpha=0.8)
    for bar, acc in zip(bars, final_accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('final val accuracy')
    ax.set_title('Final Accuracy')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # (1,0): Gradient SVD spectra at mid-training
    ax = fig.add_subplot(gs[1, 0])
    for label in shapes:
        snap = results[label]['grad_snapshots']
        mid_epoch = results[label]['snapshot_epochs'][len(results[label]['snapshot_epochs']) // 2]
        if mid_epoch in snap:
            for name, g in snap[mid_epoch].items():
                S = svd_spectrum(g)
                short_name = name.split('.')[-1]
                style = '-' if 'seq' in name or 'wq' == short_name else '--'
                ax.semilogy(S.numpy(), color=colors[label], linestyle=style,
                           alpha=0.7, label=f'{label} {short_name}')
    ax.set_xlabel('singular value index')
    ax.set_ylabel('normalized SV')
    ax.set_title(f'Gradient SVD Spectra (mid-training)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (1,1): Weight SVD spectra at end of training
    ax = fig.add_subplot(gs[1, 1])
    for label in shapes:
        snap = results[label]['weight_snapshots']
        final_epoch = results[label]['snapshot_epochs'][-1]
        if final_epoch in snap:
            for name, w in snap[final_epoch].items():
                S = svd_spectrum(w)
                short_name = name.split('.')[-1]
                style = '-' if 'seq' in name or 'wq' == short_name else '--'
                ax.semilogy(S.numpy(), color=colors[label], linestyle=style,
                           alpha=0.7, label=f'{label} {short_name}')
    ax.set_xlabel('singular value index')
    ax.set_ylabel('normalized SV')
    ax.set_title('Weight SVD Spectra (trained)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # (1,2): Effective rank comparison (# SVs above 10% of max)
    ax = fig.add_subplot(gs[1, 2])
    for label in shapes:
        snap = results[label]['weight_snapshots']
        final_epoch = results[label]['snapshot_epochs'][-1]
        eff_ranks = []
        param_names = []
        if final_epoch in snap:
            for name, w in snap[final_epoch].items():
                S = svd_spectrum(w)
                eff_rank = (S > 0.1).sum().item()
                total = len(S)
                eff_ranks.append(eff_rank / total)
                param_names.append(name.split('.')[-1])
        if eff_ranks:
            x_pos = np.arange(len(eff_ranks))
            ax.bar(x_pos, eff_ranks, color=colors[label], alpha=0.7, label=label)
    ax.set_ylabel('effective rank fraction (SV > 10%)')
    ax.set_title('Weight Effective Rank')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Suptitle
    shape_strs = [f"{l}: {' x '.join(map(str, shapes[l]))}" for l in shapes]
    fig.suptitle(f'1D vs 2D vs 3D vs 4D Embeddings on mod_arith (D={D})\n'
                 f'{" | ".join(shape_strs)}',
                 fontsize=13, fontweight='bold')

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
