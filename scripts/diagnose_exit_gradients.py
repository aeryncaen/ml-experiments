#!/usr/bin/env python3
"""Diagnose gradient behavior for fully-exited vs active samples across hops.

With top_k=2, a sample can have one exit slot and one real expert.
We classify samples into 3 categories per hop:
  - full_active: neither top-k selection is exit (both are real experts)
  - partial_exit: exactly one top-k is exit, one is real expert
  - full_exit: both top-k selections are exit (truly done)

Only full_exit samples should get zero expert output. Partial exits still
get real computation from their non-exit expert stream.

Usage:
    python scripts/diagnose_exit_gradients.py [--device cuda]
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ulb.block import ULBConfig
from ulb.diffusion import MaskedDiffusionPoE


def diagnose(device='cuda'):
    cfg = ULBConfig(d_model=64, n_heads=4, paired=True, attn_mode='blend', inner_ratio=1.75)
    model = MaskedDiffusionPoE(
        ulb_config=cfg, vocab_size=32, max_seq_len=128,
        pool_size=4, top_k=2, max_hops=8, local_window=8, router_noise=1.0,
    ).to(device)
    pool_size = model.pool_size

    B, L_in, L_out = 32, 64, 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("=" * 80)
    print("Exit gradient diagnosis (per-stream)")
    print(f"B={B}, L_in={L_in}, L_out={L_out}, pool={pool_size}, top_k=2, max_hops=8")
    print("=" * 80)

    for step in range(5):
        model.train()
        prompt = torch.randint(0, 32, (B, L_in), device=device)
        target = torch.randint(0, 32, (B, L_out), device=device)
        t = 0.1 + 0.9 * torch.rand(B, 1, device=device)
        mask = torch.rand(B, L_out, device=device) < t
        mask[:, 0] = True

        hop_data = []
        orig_route = model.route
        orig_execute = model.execute_hop

        # Track cumulative full-exit (both streams chose exit)
        cumul_full_exit = torch.zeros(B, dtype=torch.bool, device=device)

        def patched_route(logits, hop):
            nonlocal cumul_full_exit
            topk_idx, topk_weights, has_exit = orig_route(logits, hop)

            with torch.no_grad():
                is_exit = topk_idx >= pool_size  # (B, top_k)
                n_exit_streams = is_exit.sum(dim=-1)  # (B,) 0, 1, or 2

                full_active = n_exit_streams == 0
                partial_exit = n_exit_streams == 1
                full_exit = n_exit_streams == 2

                # Only mark as truly done if BOTH streams exit
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
            # Find the latest route data for this hop
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
                prev_full_exit = cfe & ~fe  # samples that fully exited on a PREVIOUS hop

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

        # Forward
        logits_out, final_mask = model(prompt, target, mask)

        # Loss
        per_token_loss = F.cross_entropy(
            logits_out.reshape(-1, model.vocab_size),
            target.reshape(-1),
            reduction='none'
        ).reshape(B, L_out)
        masked_loss = per_token_loss * mask.float()
        per_sample_loss = masked_loss.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        loss = per_sample_loss.mean()

        optimizer.zero_grad()
        loss.backward()

        # ---- Report ----
        print(f"\n--- Step {step} | loss={loss.item():.4f} ---")

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
    args = parser.parse_args()
    diagnose(args.device)
