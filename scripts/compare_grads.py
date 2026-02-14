#!/usr/bin/env python3
"""Compare gradient scales between transformer and ULB-2D after 300 steps."""

import sys
sys.path.insert(0, "src")

import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics


def build_transformer():
    d_model = 768
    n_head = 12
    vocab_size = 50304
    n_layer = 12

    class RMSNorm(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(dim))
            self.eps = 1e-6
        def forward(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
            self.o = nn.Linear(d_model, d_model, bias=False)
            self.n_head = n_head
        def forward(self, x):
            B, T, D = x.shape
            qkv = self.qkv(x).reshape(B, T, 3, n_head, D//n_head).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.o(y.transpose(1, 2).reshape(B, T, D))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(d_model, 4*d_model, bias=False)
            self.fc2 = nn.Linear(4*d_model, d_model, bias=False)
        def forward(self, x):
            return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = RMSNorm(d_model)
            self.attn = Attn()
            self.ln2 = RMSNorm(d_model)
            self.mlp = MLP()
        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.mlp(self.ln2(x))
            return x

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(vocab_size, d_model)
            self.blocks = nn.ModuleList([Block() for _ in range(n_layer)])
            self.ln_f = RMSNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight = self.wte.weight
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.02)
                elif isinstance(m, nn.Embedding):
                    nn.init.normal_(m.weight, std=0.02)
        def forward(self, idx):
            x = self.wte(idx)
            for block in self.blocks:
                x = block(x)
            return self.lm_head(self.ln_f(x))

    return GPT()


def train_steps(model, n_steps, lr=3e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    B, T = 4, 128
    for step in range(n_steps):
        optimizer.zero_grad()
        idx = torch.randint(0, 50304, (B, T))
        targets = torch.randint(0, 50304, (B, T))
        logits = model(idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 100 == 0:
            print(f"    step {step}: loss={loss.item():.4f}")
    return loss.item()


def collect_grads(model):
    """Run one forward/backward on random data and return grad stats."""
    B, T = 4, 128
    model.zero_grad()
    idx = torch.randint(0, 50304, (B, T))
    targets = torch.randint(0, 50304, (B, T))
    logits = model(idx)
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    loss.backward()
    return loss.item()


def compare():
    torch.manual_seed(42)

    from ulb.transformer import CausalULB2D

    print("Building models...")
    tf = build_transformer()
    ulb = CausalULB2D(vocab_size=50304, c_h=16, c_w=48, n_layers=12, max_seq_len=2048)

    print(f"TF params:  {sum(p.numel() for p in tf.parameters()):,}")
    print(f"ULB params: {sum(p.numel() for p in ulb.parameters()):,}")

    # Train both for 300 steps
    print("\nTraining Transformer for 300 steps (lr=3e-4)...")
    train_steps(tf, 300, lr=3e-4)

    print("\nTraining ULB-2D for 300 steps (lr=3e-4)...")
    train_steps(ulb, 300, lr=3e-4)

    # Now collect grads at step 300
    print("\n" + "=" * 80)
    print("GRADIENT COMPARISON AT STEP 300")
    print("=" * 80)

    loss_tf = collect_grads(tf)
    tf_total = torch.nn.utils.clip_grad_norm_(tf.parameters(), float('inf'))

    loss_ulb = collect_grads(ulb)
    ulb_total = torch.nn.utils.clip_grad_norm_(ulb.parameters(), float('inf'))

    print(f"\n  TF loss:  {loss_tf:.4f}   ULB loss: {loss_ulb:.4f}")
    print(f"  TF grad norm: {tf_total:.4f}   ULB grad norm: {ulb_total:.4f}   ratio: {ulb_total/tf_total:.3f}x")

    # Layer-by-layer
    print(f"\n  {'Layer':<6s}  {'TF qkv':>10s}  {'TF o':>10s}  {'TF fc1':>10s}  {'TF fc2':>10s}  |  {'ULB up':>10s}  {'ULB q':>10s}  {'ULB o':>10s}  {'ULB down':>10s}")
    print("  " + "-" * 105)

    tf_all = []
    ulb_all = []

    for i in range(12):
        tf_qkv = tf.blocks[i].attn.qkv.weight.grad.pow(2).mean().sqrt().item()
        tf_o = tf.blocks[i].attn.o.weight.grad.pow(2).mean().sqrt().item()
        tf_fc1 = tf.blocks[i].mlp.fc1.weight.grad.pow(2).mean().sqrt().item()
        tf_fc2 = tf.blocks[i].mlp.fc2.weight.grad.pow(2).mean().sqrt().item()

        ulb_up = ulb.blocks[i].w_up.grad.pow(2).mean().sqrt().item()
        ulb_q = ulb.blocks[i].w_q.grad.pow(2).mean().sqrt().item()
        ulb_o = ulb.blocks[i].w_o.grad.pow(2).mean().sqrt().item()
        ulb_down = ulb.blocks[i].w_down.grad.pow(2).mean().sqrt().item()

        tf_all.extend([tf_qkv, tf_o, tf_fc1, tf_fc2])
        ulb_all.extend([ulb_up, ulb_q, ulb_o, ulb_down])

        print(f"  {i:<6d}  {tf_qkv:10.6f}  {tf_o:10.6f}  {tf_fc1:10.6f}  {tf_fc2:10.6f}  |  {ulb_up:10.6f}  {ulb_q:10.6f}  {ulb_o:10.6f}  {ulb_down:10.6f}")

    tf_mean = statistics.mean(tf_all)
    ulb_mean = statistics.mean(ulb_all)

    print(f"\n  TF avg projection grad_rms:  {tf_mean:.6f}")
    print(f"  ULB avg projection grad_rms: {ulb_mean:.6f}")
    print(f"  Ratio (ULB/TF):              {ulb_mean/tf_mean:.3f}x")

    if ulb_mean < tf_mean:
        suggested_lr = 3e-4 * (tf_mean / ulb_mean)
        print(f"\n  ULB grads are {tf_mean/ulb_mean:.1f}x smaller than TF")
        print(f"  Current LR: 3e-4")
        print(f"  Suggested LR: {suggested_lr:.1e}")
    else:
        print(f"\n  ULB grads are {ulb_mean/tf_mean:.1f}x LARGER — LR is fine or could be lower")

    # Also check the gate/bias params
    print("\n" + "=" * 80)
    print("GATE & BIAS GRAD SCALES (avg across layers)")
    print("=" * 80)
    gate_names = ['w_attn_gate', 'attn_gate_bias', 'w_feat_gate', 'feat_gate_bias',
                  'w_blend', 'blend_bias']
    for gname in gate_names:
        vals = []
        for i in range(12):
            p = dict(ulb.blocks[i].named_parameters()).get(gname)
            if p is not None and p.grad is not None:
                vals.append(p.grad.pow(2).mean().sqrt().item())
        if vals:
            print(f"  {gname:<25s}  avg_grad_rms={statistics.mean(vals):.6f}")


if __name__ == "__main__":
    compare()
