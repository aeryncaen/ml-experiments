#!/usr/bin/env python3
"""Diagnose gradient behavior for exited vs non-exited samples across hops.

Runs a few training steps and measures:
- Per-hop: which samples have exited, gradient norms for exited vs active
- Expert gradient contribution from exited vs active samples
- Hidden state drift for exited samples (should be zero after exit)
- Final loss contribution from exited vs active samples

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
    # Small model, enough to see routing behavior
    cfg = ULBConfig(d_model=64, n_heads=4, paired=True, attn_mode='blend', inner_ratio=1.75)
    model = MaskedDiffusionPoE(
        ulb_config=cfg, vocab_size=32, max_seq_len=128,
        pool_size=4, top_k=2, max_hops=8, local_window=8, router_noise=1.0,
    ).to(device)

    B, L_in, L_out = 32, 64, 64
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    print("=" * 70)
    print("Exit gradient diagnosis")
    print(f"B={B}, L_in={L_in}, L_out={L_out}, pool=4, top_k=2, max_hops=8")
    print("=" * 70)

    for step in range(5):
        model.train()
        prompt = torch.randint(0, 32, (B, L_in), device=device)
        target = torch.randint(0, 32, (B, L_out), device=device)
        t = 0.1 + 0.9 * torch.rand(B, 1, device=device)
        mask = torch.rand(B, L_out, device=device) < t
        mask[:, 0] = True

        # ---- Instrument the forward pass ----
        # We'll monkey-patch execute_hop to record per-sample info

        hop_data = []
        exit_status = []  # per hop, (B,) bool of cumulative exits

        orig_route = model.route
        orig_execute = model.execute_hop
        cumulative_exit = torch.zeros(B, dtype=torch.bool, device=device)

        # Track hidden states per hop for exited samples
        hidden_snapshots = {}  # hop -> x.detach() for exited samples

        def patched_route(logits, hop):
            nonlocal cumulative_exit
            topk_idx, topk_weights, has_exit = orig_route(logits, hop)
            cumulative_exit = cumulative_exit | has_exit
            exit_status.append({
                'hop': hop,
                'has_exit': has_exit.detach().cpu(),
                'cumulative_exit': cumulative_exit.detach().cpu(),
                'n_exited': cumulative_exit.sum().item(),
                'n_active': B - cumulative_exit.sum().item(),
                'topk_idx': topk_idx.detach().cpu(),
            })
            return topk_idx, topk_weights, has_exit

        def patched_execute(x, topk_idx, topk_weights, hop=0):
            # Record x norms for exited vs active BEFORE the hop
            with torch.no_grad():
                exited = cumulative_exit
                active = ~exited
                x_norm_exited = x[exited].norm(dim=-1).mean().item() if exited.any() else 0.0
                x_norm_active = x[active].norm(dim=-1).mean().item() if active.any() else 0.0

            # Run the real execute_hop with gradient tracking
            x.retain_grad()
            out, logits, hop_aux = orig_execute(x, topk_idx, topk_weights, hop)

            # Record output norms
            with torch.no_grad():
                out_norm_exited = out[exited].norm(dim=-1).mean().item() if exited.any() else 0.0
                out_norm_active = out[active].norm(dim=-1).mean().item() if active.any() else 0.0

            hop_data.append({
                'hop': hop,
                'x_norm_exited': x_norm_exited,
                'x_norm_active': x_norm_active,
                'out_norm_exited': out_norm_exited,
                'out_norm_active': out_norm_active,
                'x_ref': x,  # keep ref for grad inspection after backward
                'exited_mask': exited.clone(),
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

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # ---- Report ----
        print(f"\n--- Step {step} | loss={loss.item():.4f} ---")

        # Exit timeline
        print(f"\n  Exit timeline:")
        for es in exit_status:
            print(f"    hop {es['hop']}: {es['n_exited']:>3} exited, {es['n_active']:>3} active")

        # Per-hop output norms and gradients
        print(f"\n  Per-hop output norms (exited vs active samples):")
        print(f"    {'hop':>4}  {'out_exited':>12}  {'out_active':>12}  "
              f"{'grad_exited':>12}  {'grad_active':>12}")
        for hd in hop_data:
            x_ref = hd['x_ref']
            exited = hd['exited_mask']
            active = ~exited
            if x_ref.grad is not None:
                grad_exited = x_ref.grad[exited].norm(dim=-1).mean().item() if exited.any() else 0.0
                grad_active = x_ref.grad[active].norm(dim=-1).mean().item() if active.any() else 0.0
            else:
                grad_exited = grad_active = float('nan')

            print(f"    {hd['hop']:>4}  {hd['out_norm_exited']:>12.6f}  {hd['out_norm_active']:>12.6f}  "
                  f"{grad_exited:>12.6f}  {grad_active:>12.6f}")

        # Per-sample loss breakdown
        with torch.no_grad():
            final_exited = cumulative_exit.cpu()
            loss_exited = per_sample_loss[final_exited].mean().item() if final_exited.any() else 0.0
            loss_active = per_sample_loss[~final_exited].mean().item() if (~final_exited).any() else 0.0
            print(f"\n  Final loss: exited={loss_exited:.4f}, never-exited={loss_active:.4f}")

        # Expert param gradient norms
        expert_grad_norms = []
        for i, expert in enumerate(model.experts):
            total_norm = 0.0
            n_params = 0
            for p in expert.parameters():
                if p.grad is not None:
                    total_norm += p.grad.norm().item() ** 2
                    n_params += p.numel()
            expert_grad_norms.append(total_norm ** 0.5)
        print(f"\n  Expert gradient norms: {['%.4f' % g for g in expert_grad_norms]}")

        # Router gradient norms
        stem_grad = model.stem_router.weight.grad.norm().item() if model.stem_router.weight.grad is not None else 0.0
        print(f"  Stem router grad norm: {stem_grad:.6f}")

        # Restore
        model.route = orig_route
        model.execute_hop = orig_execute

        optimizer.step()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    diagnose(args.device)
