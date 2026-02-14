#!/usr/bin/env python3
"""Compare gradient structure of 1D vs 2D projection models trained on mod_arith.

Trains a minimal 1D model (flat Linear projections) and a 2D model (tensor einsum
projections) on the modular arithmetic task, then compares:

1. Training curves (loss, accuracy)
2. Weight structure evolution (how weights organize during training)
3. Gradient structure during training (correlation, singular spectrum)
4. Learned weight singular value spectra

Usage:
    python scripts/compare_grad_structure.py [--dim 64] [--epochs 80] [--save path.png]
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


# --- 1D Model: standard flat linear projections ---

class Block1D(nn.Module):
    """Causal self-attention with flat Linear QKV + SwiGLU MLP."""
    def __init__(self, D, n_heads=4, mlp_inner=0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = D // n_heads
        self.wq = nn.Linear(D, D, bias=False)
        self.wk = nn.Linear(D, D, bias=False)
        self.wv = nn.Linear(D, D, bias=False)
        self.wo = nn.Linear(D, D, bias=False)
        self.norm1 = RMSNorm(D)
        self.has_mlp = mlp_inner > 0
        if self.has_mlp:
            self.norm2 = RMSNorm(D)
            self.gate_proj = nn.Linear(D, mlp_inner, bias=False)
            self.up_proj = nn.Linear(D, mlp_inner, bias=False)
            self.down_proj = nn.Linear(mlp_inner, D, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        h = self.norm1(x)
        q = self.wq(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(h).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        y = self.wo(y)
        x = x + y
        if self.has_mlp:
            r = self.norm2(x)
            x = x + self.down_proj(F.silu(self.gate_proj(r)) * self.up_proj(r))
        return x


# --- 2D Model: tensor einsum projections ---

class Block2D(nn.Module):
    """Causal self-attention with 2D (C,C,C,C) tensor QKV + 2D feat-attn (no MLP)."""
    def __init__(self, D, n_heads=4):
        super().__init__()
        self.C = int(D ** 0.5)
        assert self.C * self.C == D
        C = self.C
        self.n_heads = n_heads
        self.head_dim = D // n_heads

        init_scale = D ** -0.5

        # Seq-attn QKV: 2D projections
        self.seq_wq = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.seq_wk = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.seq_wv = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.seq_wo = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.norm1 = RMSNorm(D)

        # Feat-attn QKV: 2D projections
        self.feat_wq = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.feat_wk = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.feat_wv = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.feat_wo = nn.Parameter(torch.randn(C, C, C, C) * init_scale)
        self.norm2 = RMSNorm(D)

    def _proj(self, x, w):
        return torch.einsum('...cd,cdef->...ef', x, w)

    def forward(self, x):
        B, T, D = x.shape
        C = self.C

        # --- Seq-attn ---
        h = self.norm1(x).view(B, T, C, C)
        Q = self._proj(h, self.seq_wq)
        K = self._proj(h, self.seq_wk)
        V = self._proj(h, self.seq_wv)
        # C channels as heads: (B, C, T, C)
        Q = Q.permute(0, 2, 1, 3)
        K = K.permute(0, 2, 1, 3)
        V = V.permute(0, 2, 1, 3)
        y = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        y = y.permute(0, 2, 1, 3).contiguous()
        y = self._proj(y, self.seq_wo)
        x = x + y.reshape(B, T, D)

        # --- Feat-attn ---
        h = self.norm2(x).view(B, T, C, C)
        Q = self._proj(h, self.feat_wq).reshape(B * T, 1, C, C)
        K = self._proj(h, self.feat_wk).reshape(B * T, 1, C, C)
        V = self._proj(h, self.feat_wv).reshape(B * T, 1, C, C)
        y = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
        y = y.view(B, T, C, C)
        y = self._proj(y, self.feat_wo)
        x = x + y.reshape(B, T, D)

        return x


class Model(nn.Module):
    def __init__(self, block_fn, D, n_layers=1, **block_kwargs):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D)
        self.blocks = nn.ModuleList([block_fn(D, **block_kwargs) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(D) for _ in range(n_layers)])
        self.final_norm = RMSNorm(D)
        self.head = nn.Linear(D, VOCAB_SIZE, bias=False)

    def forward(self, x):
        h = self.embed(x)
        for norm, block in zip(self.norms, self.blocks):
            h = h + block(norm(h))
        return self.head(self.final_norm(h))


def train(model, n_epochs, B=64, L=32, lr=3e-4, device='cpu'):
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    train_accs = []
    val_accs = []

    # Collect gradient snapshots at specific epochs
    grad_snapshots = {}  # epoch -> {param_name: grad_tensor}
    weight_snapshots = {}  # epoch -> {param_name: weight_tensor}
    snapshot_epochs = [0, n_epochs // 4, n_epochs // 2, 3 * n_epochs // 4, n_epochs - 1]

    # Pre-generate val data
    val_x, val_t = gen_mod_arith(512, L, device=device)

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        epoch_correct = 0
        epoch_total = 0
        n_batches = 20

        for _ in range(n_batches):
            x, t = gen_mod_arith(B, L, device=device)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), t.reshape(-1))
            opt.zero_grad()
            loss.backward()

            # Snapshot gradients
            if epoch in snapshot_epochs:
                if epoch not in grad_snapshots:
                    grad_snapshots[epoch] = {}
                    weight_snapshots[epoch] = {}
                for name, p in model.named_parameters():
                    if p.grad is not None and ('wq' in name or 'wk' in name or 'wv' in name or 'wo' in name
                                               or 'seq_wq' in name or 'feat_wq' in name):
                        if name not in grad_snapshots[epoch]:
                            grad_snapshots[epoch][name] = p.grad.detach().cpu().clone()
                            weight_snapshots[epoch][name] = p.detach().cpu().clone()

            opt.step()

            epoch_loss += loss.item()
            preds = logits.argmax(-1)
            epoch_correct += (preds == t).float().sum().item()
            epoch_total += t.numel()

        train_losses.append(epoch_loss / n_batches)
        train_accs.append(epoch_correct / epoch_total)

        # Val
        model.eval()
        with torch.no_grad():
            logits = model(val_x)
            preds = logits.argmax(-1)
            val_acc = (preds == val_t).float().mean().item()
            val_accs.append(val_acc)

        if epoch % 10 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:3d}: loss={train_losses[-1]:.4f} "
                  f"train_acc={train_accs[-1]:.3f} val_acc={val_acc:.3f}")

    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'grad_snapshots': grad_snapshots,
        'weight_snapshots': weight_snapshots,
        'snapshot_epochs': snapshot_epochs,
    }


def weight_svd_spectrum(w):
    """Get normalized singular values of a weight tensor reshaped to 2D."""
    w_flat = w.reshape(w.shape[0], -1).float() if w.dim() > 2 else w.float()
    if w.dim() == 4:
        # (C, C, C, C) -> (C*C, C*C)
        C = w.shape[0]
        w_flat = w.reshape(C * C, C * C).float()
    S = torch.linalg.svdvals(w_flat)
    return S / S[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--n-layers', type=int, default=1)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--save', type=str, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    D = args.dim
    C = int(D ** 0.5)
    assert C * C == D

    torch.manual_seed(42)

    # Build models with matched param count (approximately)
    # 1D: 4 Linear(D,D) for QKV+O = 4*D^2 = 16384, + SwiGLU MLP
    # 2D: 8 (C,C,C,C) tensors = 8*C^4 = 8*4096 = 32768
    # To match: give 1D an MLP. 4*D^2 + 3*D*mlp_inner ~ 8*C^4
    # 16384 + 192*mlp_inner ~ 32768 => mlp_inner ~ 85
    mlp_inner = max(1, round((8 * C**4 - 4 * D * D) / (3 * D)))

    model_1d = Model(Block1D, D, n_layers=args.n_layers, n_heads=4, mlp_inner=mlp_inner)
    model_2d = Model(Block2D, D, n_layers=args.n_layers, n_heads=4)

    n1 = sum(p.numel() for p in model_1d.parameters())
    n2 = sum(p.numel() for p in model_2d.parameters())
    print(f"1D model: {n1:,} params (mlp_inner={mlp_inner})")
    print(f"2D model: {n2:,} params")

    # Use same embeddings for fair comparison
    with torch.no_grad():
        model_2d.embed.weight.copy_(model_1d.embed.weight)
        model_2d.head.weight.copy_(model_1d.head.weight)

    print(f"\nTraining 1D model on mod_arith ({args.epochs} epochs)...")
    res_1d = train(model_1d, args.epochs, lr=args.lr, device=args.device)

    # Reset seed for reproducibility of data
    torch.manual_seed(42)
    print(f"\nTraining 2D model on mod_arith ({args.epochs} epochs)...")
    res_2d = train(model_2d, args.epochs, lr=args.lr, device=args.device)

    # --- Plot ---
    print("\nPlotting...")
    fig = plt.figure(figsize=(22, 14))
    gs = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.35)

    # Row 1: Training curves + final weight SVD spectra
    ax_loss = fig.add_subplot(gs[0, 0])
    ax_loss.plot(res_1d['train_losses'], 'b-', alpha=0.7, label='1D')
    ax_loss.plot(res_2d['train_losses'], 'r-', alpha=0.7, label='2D')
    ax_loss.set_xlabel('epoch')
    ax_loss.set_ylabel('loss')
    ax_loss.set_title('Training loss')
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc = fig.add_subplot(gs[0, 1])
    ax_acc.plot(res_1d['val_accs'], 'b-', alpha=0.7, label='1D')
    ax_acc.plot(res_2d['val_accs'], 'r-', alpha=0.7, label='2D')
    ax_acc.set_xlabel('epoch')
    ax_acc.set_ylabel('val accuracy')
    ax_acc.set_title('Validation accuracy')
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)

    # Weight SVD spectrum at end of training
    ax_svd = fig.add_subplot(gs[0, 2:4])
    # Get projection weights from both models
    for name, p in model_1d.named_parameters():
        if 'blocks.0.wq' in name:
            S = weight_svd_spectrum(p.detach().cpu())
            ax_svd.semilogy(S.numpy(), 'b-', alpha=0.6, label=f'1D {name.split(".")[-1]}')
        if 'blocks.0.wo' in name:
            S = weight_svd_spectrum(p.detach().cpu())
            ax_svd.semilogy(S.numpy(), 'b--', alpha=0.6, label=f'1D {name.split(".")[-1]}')
    for name, p in model_2d.named_parameters():
        if 'blocks.0.seq_wq' in name:
            S = weight_svd_spectrum(p.detach().cpu())
            ax_svd.semilogy(S.numpy(), 'r-', alpha=0.6, label=f'2D {name.split(".")[-1]}')
        if 'blocks.0.seq_wo' in name:
            S = weight_svd_spectrum(p.detach().cpu())
            ax_svd.semilogy(S.numpy(), 'r--', alpha=0.6, label=f'2D {name.split(".")[-1]}')
        if 'blocks.0.feat_wq' in name:
            S = weight_svd_spectrum(p.detach().cpu())
            ax_svd.semilogy(S.numpy(), 'm-', alpha=0.6, label=f'2D feat_{name.split(".")[-1]}')
    ax_svd.set_xlabel('singular value index')
    ax_svd.set_ylabel('normalized singular value')
    ax_svd.set_title('Learned weight SVD spectra')
    ax_svd.legend(fontsize=8)
    ax_svd.grid(True, alpha=0.3)

    # Row 2: Weight structure visualization (init vs trained)
    snap_epochs = res_1d['snapshot_epochs']

    # 1D: show wq weight as (D, D) heatmap at init and final
    ax_w1d_init = fig.add_subplot(gs[1, 0])
    w1d_init = res_1d['weight_snapshots'][snap_epochs[0]]
    w1d_key = [k for k in w1d_init if 'wq' in k][0]
    w = w1d_init[w1d_key]
    if w.dim() == 2:
        im = w.numpy()
    else:
        im = w.reshape(D, -1).numpy()
    vmax = max(abs(im.min()), abs(im.max()))
    ax_w1d_init.imshow(im, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax_w1d_init.set_title(f'1D wq @ epoch 0\n({im.shape[0]}x{im.shape[1]})', fontsize=10)

    ax_w1d_final = fig.add_subplot(gs[1, 1])
    w1d_final = res_1d['weight_snapshots'][snap_epochs[-1]]
    w = w1d_final[w1d_key]
    if w.dim() == 2:
        im = w.numpy()
    else:
        im = w.reshape(D, -1).numpy()
    vmax = max(abs(im.min()), abs(im.max()))
    ax_w1d_final.imshow(im, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax_w1d_final.set_title(f'1D wq @ epoch {snap_epochs[-1]}\n({im.shape[0]}x{im.shape[1]})', fontsize=10)

    # 2D: show seq_wq weight in block form at init and final
    ax_w2d_init = fig.add_subplot(gs[1, 2])
    w2d_init = res_2d['weight_snapshots'][snap_epochs[0]]
    w2d_key = [k for k in w2d_init if 'seq_wq' in k][0]
    w = w2d_init[w2d_key]
    # (C, C, C, C) -> block matrix (C*C, C*C) via permute
    w_block = w.view(C, C, C, C).permute(0, 2, 1, 3).reshape(C*C, C*C).numpy()
    vmax = max(abs(w_block.min()), abs(w_block.max()))
    ax_w2d_init.imshow(w_block, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    for i in range(1, C):
        ax_w2d_init.axhline(i * C - 0.5, color='k', lw=0.5, alpha=0.3)
        ax_w2d_init.axvline(i * C - 0.5, color='k', lw=0.5, alpha=0.3)
    ax_w2d_init.set_title(f'2D seq_wq @ epoch 0\n{C}x{C} blocks', fontsize=10)

    ax_w2d_final = fig.add_subplot(gs[1, 3])
    w2d_final = res_2d['weight_snapshots'][snap_epochs[-1]]
    w = w2d_final[w2d_key]
    w_block = w.view(C, C, C, C).permute(0, 2, 1, 3).reshape(C*C, C*C).numpy()
    vmax = max(abs(w_block.min()), abs(w_block.max()))
    ax_w2d_final.imshow(w_block, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    for i in range(1, C):
        ax_w2d_final.axhline(i * C - 0.5, color='k', lw=0.5, alpha=0.3)
        ax_w2d_final.axvline(i * C - 0.5, color='k', lw=0.5, alpha=0.3)
    ax_w2d_final.set_title(f'2D seq_wq @ epoch {snap_epochs[-1]}\n{C}x{C} blocks', fontsize=10)

    # Row 3: Gradient structure comparison at mid-training
    mid_epoch = snap_epochs[len(snap_epochs) // 2]

    # Gradient as heatmap
    ax_g1d = fig.add_subplot(gs[2, 0])
    g1d = res_1d['grad_snapshots'][mid_epoch]
    g1d_key = [k for k in g1d if 'wq' in k][0]
    g = g1d[g1d_key]
    if g.dim() == 2:
        gim = g.numpy()
    else:
        gim = g.reshape(D, -1).numpy()
    vmax = max(abs(gim.min()), abs(gim.max()))
    ax_g1d.imshow(gim, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax_g1d.set_title(f'1D wq gradient @ epoch {mid_epoch}', fontsize=10)

    ax_g2d = fig.add_subplot(gs[2, 1])
    g2d = res_2d['grad_snapshots'][mid_epoch]
    g2d_key = [k for k in g2d if 'seq_wq' in k][0]
    g = g2d[g2d_key]
    g_block = g.view(C, C, C, C).permute(0, 2, 1, 3).reshape(C*C, C*C).numpy()
    vmax = max(abs(g_block.min()), abs(g_block.max()))
    ax_g2d.imshow(g_block, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    for i in range(1, C):
        ax_g2d.axhline(i * C - 0.5, color='k', lw=0.5, alpha=0.3)
        ax_g2d.axvline(i * C - 0.5, color='k', lw=0.5, alpha=0.3)
    ax_g2d.set_title(f'2D seq_wq gradient @ epoch {mid_epoch}\n{C}x{C} blocks', fontsize=10)

    # Gradient SVD spectrum at mid-training
    ax_gsvd = fig.add_subplot(gs[2, 2])
    for name, g in res_1d['grad_snapshots'][mid_epoch].items():
        S = weight_svd_spectrum(g)
        label = name.split('.')[-1]
        ax_gsvd.semilogy(S.numpy(), 'b-', alpha=0.6, label=f'1D {label}')
    for name, g in res_2d['grad_snapshots'][mid_epoch].items():
        S = weight_svd_spectrum(g)
        label = name.split('.')[-1]
        color = 'r' if 'seq' in name else 'm'
        ax_gsvd.semilogy(S.numpy(), f'{color}-', alpha=0.6, label=f'2D {label}')
    ax_gsvd.set_xlabel('singular value index')
    ax_gsvd.set_ylabel('normalized singular value')
    ax_gsvd.set_title(f'Gradient SVD @ epoch {mid_epoch}', fontsize=10)
    ax_gsvd.legend(fontsize=7)
    ax_gsvd.grid(True, alpha=0.3)

    # Weight change (delta) structure: final - init
    ax_delta = fig.add_subplot(gs[2, 3])
    # 2D weight delta in block form
    w_init = w2d_init[w2d_key].view(C, C, C, C).permute(0, 2, 1, 3).reshape(C*C, C*C).numpy()
    w_final = w2d_final[w2d_key].view(C, C, C, C).permute(0, 2, 1, 3).reshape(C*C, C*C).numpy()
    delta = w_final - w_init
    vmax = max(abs(delta.min()), abs(delta.max()))
    ax_delta.imshow(delta, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='nearest')
    for i in range(1, C):
        ax_delta.axhline(i * C - 0.5, color='k', lw=0.5, alpha=0.3)
        ax_delta.axvline(i * C - 0.5, color='k', lw=0.5, alpha=0.3)
    ax_delta.set_title(f'2D seq_wq weight delta\n(trained - init)', fontsize=10)

    fig.suptitle(f'1D Linear vs 2D Einsum on mod_arith — D={D}, C={C}\n'
                 f'1D: {n1:,} params, 2D: {n2:,} params',
                 fontsize=14, fontweight='bold')

    if args.save:
        plt.savefig(args.save, dpi=150, bbox_inches='tight')
        print(f"Saved to {args.save}")
    else:
        plt.show()

    print(f"\nFinal val acc — 1D: {res_1d['val_accs'][-1]:.3f}  2D: {res_2d['val_accs'][-1]:.3f}")


if __name__ == '__main__':
    main()
