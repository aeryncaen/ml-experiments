#!/usr/bin/env python3
"""Train a DiffusionPoE on 2D swiss roll data.

Usage:
    python scripts/diffusion_poe.py [--dim 64] [--pool-size 4] [--top-k 2]
                                     [--epochs 200] [--batch-size 1024]
                                     [--device cpu]

The model learns to denoise corrupted swiss roll points via a single forward
pass through PoE, where routing decides the adaptive denoising schedule.
"""

import sys, argparse, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ulb.diffusion import DiffusionPoE


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def sample_swiss_roll(n: int, noise: float = 0.05) -> torch.Tensor:
    """Sample 2D swiss roll points. Returns (n, 2) normalized to ~unit variance."""
    t = 1.5 * math.pi * (1 + 2 * torch.rand(n))
    x = t * torch.cos(t)
    y = t * torch.sin(t)
    points = torch.stack([x, y], dim=-1)
    points = points + noise * torch.randn_like(points)
    # Normalize to roughly unit variance
    points = (points - points.mean(dim=0)) / points.std()
    return points


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)

    model = DiffusionPoE(
        dim=args.dim,
        input_dim=2,
        pool_size=args.pool_size,
        top_k=args.top_k,
        max_hops=args.max_hops,
        expert_expand=args.expert_expand,
        router_noise=1.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"DiffusionPoE: dim={args.dim}, pool_size={args.pool_size}, "
          f"top_k={args.top_k}, max_hops={model.max_hops}, params={n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    steps_per_epoch = args.steps_per_epoch
    best_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        # Anneal router noise
        frac = (epoch - 1) / max(args.epochs - 1, 1)
        model.router_noise_scale = 1.0 * (1 - frac)

        for step in range(steps_per_epoch):
            # Sample clean data
            x_clean = sample_swiss_roll(args.batch_size).to(device)  # (B, 2)

            # Sample noise level per sample: sigma ~ U(0.01, 1.0)
            sigma = 0.01 + 0.99 * torch.rand(args.batch_size, 1, device=device)  # (B, 1)

            # Corrupt
            noise = torch.randn_like(x_clean)
            x_noisy = x_clean + sigma * noise  # (B, 2)

            # Forward: predict epsilon (the noise that was added)
            eps_pred = model(x_noisy)  # (B, 2)

            # MSE loss on noise prediction
            loss = ((eps_pred - noise) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / steps_per_epoch
        mean_hops = model.last_mean_hops
        if isinstance(mean_hops, torch.Tensor):
            mean_hops = mean_hops.item()

        if avg_loss < best_loss:
            best_loss = avg_loss
            marker = " *"
        else:
            marker = ""

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:>4}/{args.epochs}  loss={avg_loss:.6f}  "
                  f"hops={mean_hops:.2f}  noise_scale={model.router_noise_scale:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}{marker}")

    return model


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample(model, n: int, device: torch.device, steps: int = 50) -> torch.Tensor:
    """Generate samples via iterative denoising (DDPM-style).

    Start from pure noise, denoise over `steps` evenly-spaced sigma levels.
    Each step: predict noise, subtract scaled prediction.
    """
    model.eval()
    # Sigma schedule: linear from 1.0 down to ~0
    sigmas = torch.linspace(1.0, 0.01, steps + 1, device=device)

    x = torch.randn(n, 2, device=device)  # start from pure noise

    total_hops = 0.0
    for i in range(steps):
        sigma_cur = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Predict noise at current sigma
        eps_pred = model(x)

        # Denoise: estimate x0 then re-noise to sigma_next
        x0_hat = x - sigma_cur * eps_pred
        if i < steps - 1:
            # Re-noise to next (lower) sigma level
            x = x0_hat + sigma_next * torch.randn_like(x)
        else:
            x = x0_hat

        mean_hops = model.last_mean_hops
        if isinstance(mean_hops, torch.Tensor):
            mean_hops = mean_hops.item()
        total_hops += mean_hops

    print(f"Sampling: {n} points, {steps} steps, avg hops/step={total_hops/steps:.2f}")
    return x


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize(real_points: torch.Tensor, gen_points: torch.Tensor, save_path: str):
    """Plot real vs generated distribution side by side."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    real = real_points.cpu().numpy()
    gen = gen_points.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Real
    axes[0].scatter(real[:, 0], real[:, 1], s=1, alpha=0.5, c='blue')
    axes[0].set_title('Real (Swiss Roll)')
    axes[0].set_xlim(-4, 4)
    axes[0].set_ylim(-4, 4)
    axes[0].set_aspect('equal')

    # Generated
    axes[1].scatter(gen[:, 0], gen[:, 1], s=1, alpha=0.5, c='red')
    axes[1].set_title('Generated (DiffusionPoE)')
    axes[1].set_xlim(-4, 4)
    axes[1].set_ylim(-4, 4)
    axes[1].set_aspect('equal')

    # Overlay
    axes[2].scatter(real[:, 0], real[:, 1], s=1, alpha=0.3, c='blue', label='Real')
    axes[2].scatter(gen[:, 0], gen[:, 1], s=1, alpha=0.3, c='red', label='Generated')
    axes[2].set_title('Overlay')
    axes[2].set_xlim(-4, 4)
    axes[2].set_ylim(-4, 4)
    axes[2].set_aspect('equal')
    axes[2].legend(markerscale=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved visualization → {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DiffusionPoE on 2D swiss roll')
    parser.add_argument('--dim', type=int, default=64, help='Hidden dim')
    parser.add_argument('--pool-size', type=int, default=4, help='Number of experts')
    parser.add_argument('--top-k', type=int, default=2, help='Experts per hop')
    parser.add_argument('--max-hops', type=int, default=None, help='Max routing depth')
    parser.add_argument('--expert-expand', type=float, default=4.0, help='MLP expansion ratio')
    parser.add_argument('--epochs', type=int, default=200, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=1024, help='Batch size')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--save-dir', type=str, default=None, help='Save checkpoint and plots')
    args = parser.parse_args()

    print("=" * 60)
    print("DiffusionPoE — Swiss Roll")
    print("=" * 60)

    model = train(args)
    device = next(model.parameters()).device

    # Generate samples
    gen_points = sample(model, n=5000, device=device)
    real_points = sample_swiss_roll(5000).to(device)

    # Save
    save_dir = Path(args.save_dir) if args.save_dir else Path('out/diffusion_poe')
    save_dir.mkdir(parents=True, exist_ok=True)

    visualize(real_points, gen_points, str(save_dir / 'swiss_roll.png'))

    # Save checkpoint
    ckpt_path = save_dir / 'model.pt'
    torch.save({
        'state_dict': model.state_dict(),
        'args': vars(args),
        'params': sum(p.numel() for p in model.parameters()),
    }, ckpt_path)
    print(f"Saved checkpoint → {ckpt_path}")


if __name__ == '__main__':
    main()
