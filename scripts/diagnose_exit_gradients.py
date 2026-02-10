#!/usr/bin/env python3
"""Diagnose exit/routing behavior with pool^2 PoolOfExperts.

Uses a plain PoolOfExperts (not MaskedDiffusionPoE) with external embed+head,
same pattern as bench_ssm. This gets the pool^2 router space natively:
  pool_size real expert slots + pool_size*(pool_size-1) exit slots.
Random selection has 1/pool_size chance of picking an expert — depth must be earned.

Classifies samples into 3 categories per hop:
  - full_active: all top-k are real experts
  - partial_exit: mix of real experts and exit slots
  - full_exit: all top-k are exit (truly done)

Usage:
    python scripts/diagnose_exit_gradients.py [--device cuda]
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ulb.block import ULBBlock, ULBConfig
from ulb.stack import PoolOfExperts


VOCAB_SIZE = 64


def diagnose(device='cuda', pool_size=4, top_k=2, max_hops=8, n_steps=10,
             B=32, L=32, dim=64):
    cfg = ULBConfig(d_model=dim, n_heads=4, paired=True, attn_mode='blend', inner_ratio=1.75)
    make_layer = lambda: ULBBlock(cfg)
    model = PoolOfExperts(
        make_layer=make_layer, pool_size=pool_size, dim=dim,
        top_k=top_k, max_hops=max_hops, router_noise=1.0,
    ).to(device)

    embed = nn.Embedding(VOCAB_SIZE, dim).to(device)
    head = nn.Linear(dim, VOCAB_SIZE).to(device)

    all_params = list(model.parameters()) + list(embed.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=3e-4)

    n_router = model.n_router_options  # pool_size^2
    n_exit = n_router - pool_size

    print("=" * 80)
    print("Exit gradient diagnosis (pool^2 PoolOfExperts)")
    print(f"B={B}, L={L}, pool={pool_size}, top_k={top_k}, max_hops={max_hops}")
    print(f"Router space: {n_router} ({pool_size} expert + {n_exit} exit)")
    print("=" * 80)

    for step in range(n_steps):
        model.train()
        inp = torch.randint(0, VOCAB_SIZE, (B, L), device=device)
        tgt = torch.randint(0, VOCAB_SIZE, (B, L), device=device)

        hop_data = []
        orig_route = model.route
        orig_execute = model.execute_hop

        cumul_full_exit = torch.zeros(B, dtype=torch.bool, device=device)

        def patched_route(logits, hop):
            nonlocal cumul_full_exit
            topk_idx, topk_weights, has_exit = orig_route(logits, hop)

            with torch.no_grad():
                is_exit = topk_idx >= pool_size  # (B, top_k)
                n_exit_streams = is_exit.sum(dim=-1)  # (B,) 0..top_k

                full_active = n_exit_streams == 0
                partial_exit = (n_exit_streams > 0) & (n_exit_streams < top_k)
                full_exit = n_exit_streams == top_k

                cumul_full_exit = cumul_full_exit | full_exit

                hop_data.append({
                    'hop': hop,
                    'phase': 'route',
                    'n_full_active': full_active.sum().item(),
                    'n_partial_exit': partial_exit.sum().item(),
                    'n_full_exit': full_exit.sum().item(),
                    'n_cumul_full_exit': cumul_full_exit.sum().item(),
                    'full_active': full_active.clone(),
                    'partial_exit': partial_exit.clone(),
                    'full_exit': full_exit.clone(),
                    'cumul_full_exit': cumul_full_exit.clone(),
                })

            return topk_idx, topk_weights, has_exit

        def patched_execute(x, topk_idx, topk_weights, hop=0):
            route_info = None
            for hd in reversed(hop_data):
                if hd['hop'] == hop and hd['phase'] == 'route':
                    route_info = hd
                    break

            x.retain_grad()
            out, logits, hop_aux = orig_execute(x, topk_idx, topk_weights, hop)

            with torch.no_grad():
                fa = route_info['full_active']
                pe = route_info['partial_exit']
                fe = route_info['full_exit']
                cfe = route_info['cumul_full_exit']
                prev_full_exit = cfe & ~fe

                out_fa = out[fa].norm(dim=-1).mean().item() if fa.any() else 0.0
                out_pe = out[pe].norm(dim=-1).mean().item() if pe.any() else 0.0
                out_fe = out[fe].norm(dim=-1).mean().item() if fe.any() else 0.0
                out_prev = out[prev_full_exit].norm(dim=-1).mean().item() if prev_full_exit.any() else 0.0

            hop_data.append({
                'hop': hop,
                'phase': 'execute',
                'x_ref': x,
                'full_active': fa.clone(),
                'partial_exit': pe.clone(),
                'full_exit': fe.clone(),
                'prev_full_exit': prev_full_exit.clone(),
                'out_full_active': out_fa,
                'out_partial_exit': out_pe,
                'out_full_exit': out_fe,
                'out_prev_full_exit': out_prev,
                'n_prev_full_exit': prev_full_exit.sum().item(),
            })

            return out, logits, hop_aux

        model.route = patched_route
        model.execute_hop = patched_execute

        # Forward: embed -> PoE -> head
        x = embed(inp)
        h = model(x)
        logits_out = head(h)

        # Loss
        loss = F.cross_entropy(logits_out.reshape(-1, VOCAB_SIZE), tgt.reshape(-1))
        aux_loss = getattr(model, 'aux_loss', 0.0)
        total_loss = loss + aux_loss

        optimizer.zero_grad()
        total_loss.backward()

        # ---- Report ----
        mean_hops = model.last_mean_hops
        hops_val = mean_hops.item() if hasattr(mean_hops, 'item') else mean_hops
        print(f"\n--- Step {step} | loss={loss.item():.4f} | mean_hops={hops_val:.1f} ---")

        # Route timeline
        print(f"\n  Route decisions per hop:")
        print(f"    {'hop':>4}  {'full_act':>8}  {'partial':>8}  {'full_exit':>9}  {'cumul_done':>10}")
        for hd in hop_data:
            if hd['phase'] != 'route':
                continue
            print(f"    {hd['hop']:>4}  {hd['n_full_active']:>8}  {hd['n_partial_exit']:>8}  "
                  f"{hd['n_full_exit']:>9}  {hd['n_cumul_full_exit']:>10}")

        # Per-hop output norms and gradients
        print(f"\n  Per-hop output norms by category:")
        print(f"    {'hop':>4}  {'out_active':>10}  {'out_partial':>11}  {'out_full_ex':>11}  "
              f"{'out_prev_ex':>11}  {'n_prev':>6}  "
              f"{'grad_active':>11}  {'grad_partial':>12}  {'grad_full_ex':>12}  {'grad_prev_ex':>12}")
        for hd in hop_data:
            if hd['phase'] != 'execute':
                continue
            x_ref = hd['x_ref']
            fa = hd['full_active']
            pe = hd['partial_exit']
            fe = hd['full_exit']
            pfe = hd['prev_full_exit']

            if x_ref.grad is not None:
                g_fa = x_ref.grad[fa].norm(dim=-1).mean().item() if fa.any() else 0.0
                g_pe = x_ref.grad[pe].norm(dim=-1).mean().item() if pe.any() else 0.0
                g_fe = x_ref.grad[fe].norm(dim=-1).mean().item() if fe.any() else 0.0
                g_pfe = x_ref.grad[pfe].norm(dim=-1).mean().item() if pfe.any() else 0.0
            else:
                g_fa = g_pe = g_fe = g_pfe = float('nan')

            print(f"    {hd['hop']:>4}  {hd['out_full_active']:>10.6f}  {hd['out_partial_exit']:>11.6f}  "
                  f"{hd['out_full_exit']:>11.6f}  {hd['out_prev_full_exit']:>11.6f}  "
                  f"{hd['n_prev_full_exit']:>6}  "
                  f"{g_fa:>11.6f}  {g_pe:>12.6f}  {g_fe:>12.6f}  {g_pfe:>12.6f}")

        # Expert param gradient norms
        expert_grad_norms = []
        for i, expert in enumerate(model.experts):
            total_norm = sum(p.grad.norm().item() ** 2 for p in expert.parameters() if p.grad is not None)
            expert_grad_norms.append(total_norm ** 0.5)
        print(f"\n  Expert gradient norms: {['%.4f' % g for g in expert_grad_norms]}")

        stem_grad = model.stem_router.weight.grad.norm().item() if model.stem_router.weight.grad is not None else 0.0
        print(f"  Stem router grad norm: {stem_grad:.6f}")

        model.route = orig_route
        model.execute_hop = orig_execute
        optimizer.step()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--pool-size', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--max-hops', type=int, default=8)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--seq-len', type=int, default=32)
    parser.add_argument('--dim', type=int, default=64)
    args = parser.parse_args()
    diagnose(args.device, pool_size=args.pool_size, top_k=args.top_k,
             max_hops=args.max_hops, n_steps=args.steps, B=args.batch,
             L=args.seq_len, dim=args.dim)
