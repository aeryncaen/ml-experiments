#!/usr/bin/env python3
"""Train MaskedDiffusionPoE on Shakespeare text (character-level).

LLaDA-style masked diffusion: mask a random fraction of the output,
train to predict masked tokens. PoE routing decides adaptive demasking depth.

Usage:
    python scripts/shakespeare_diffusion.py [--dim 128] [--pool-size 4]
        [--epochs 100] [--device cuda]
"""

import sys, argparse, os, math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from ulb.block import ULBConfig
from ulb.diffusion import MaskedDiffusionPoE


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_shakespeare(data_dir: str = "data") -> tuple[str, dict, dict]:
    """Download and load Shakespeare text. Returns (text, char2idx, idx2char)."""
    data_path = Path(data_dir) / "shakespeare.txt"
    if not data_path.exists():
        print(f"Downloading Shakespeare → {data_path}")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(SHAKESPEARE_URL, data_path)

    text = data_path.read_text()
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return text, char2idx, idx2char


def encode(text: str, char2idx: dict) -> torch.Tensor:
    return torch.tensor([char2idx[c] for c in text], dtype=torch.long)


def decode(ids: torch.Tensor, idx2char: dict) -> str:
    return ''.join(idx2char[i.item()] for i in ids)


class ShakespeareDataset:
    """Yields (prompt, output, target) chunks from Shakespeare text."""

    def __init__(self, data: torch.Tensor, prompt_len: int, output_len: int):
        self.data = data
        self.prompt_len = prompt_len
        self.output_len = output_len
        self.total_len = prompt_len + output_len

    def sample_batch(self, batch_size: int, device: torch.device
                     ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample random batch of (prompt, output) pairs."""
        max_start = len(self.data) - self.total_len
        starts = torch.randint(0, max_start, (batch_size,))
        prompt = torch.stack([self.data[s:s + self.prompt_len] for s in starts])
        output = torch.stack([self.data[s + self.prompt_len:s + self.total_len] for s in starts])
        return prompt.to(device), output.to(device)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)

    # Load data
    text, char2idx, idx2char = load_shakespeare()
    data = encode(text, char2idx)
    vocab_size = len(char2idx)
    print(f"Shakespeare: {len(text):,} chars, vocab_size={vocab_size}")

    # Split: 90% train, 10% val
    n_train = int(0.9 * len(data))
    train_data = data[:n_train]
    val_data = data[n_train:]
    print(f"Train: {len(train_data):,} chars, Val: {len(val_data):,} chars")

    train_ds = ShakespeareDataset(train_data, args.prompt_len, args.output_len)
    val_ds = ShakespeareDataset(val_data, args.prompt_len, args.output_len)

    # Model
    cfg = ULBConfig(
        d_model=args.dim,
        n_heads=args.n_heads,
        paired=True,
        attn_mode='blend',
        inner_ratio=args.inner_ratio,
    )
    model = MaskedDiffusionPoE(
        ulb_config=cfg,
        vocab_size=vocab_size,
        max_seq_len=args.prompt_len + args.output_len,
        pool_size=args.pool_size,
        top_k=args.top_k,
        max_hops=args.max_hops,
        local_window=args.local_window,
        router_noise=1.0,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: dim={args.dim}, heads={args.n_heads}, pool={args.pool_size}, "
          f"top_k={args.top_k}, max_hops={model.max_hops}, params={n_params:,}")

    if args.compile:
        print("Compiling full model with torch.compile...")
        model = torch.compile(model)
        print("Compilation will happen on first forward pass.")
    elif args.compile_blocks:
        n_compiled = 0
        for i, expert in enumerate(model.experts):
            model.experts[i] = torch.compile(expert)
            n_compiled += 1
        model.stem_layer = torch.compile(model.stem_layer)
        model.exit_layer = torch.compile(model.exit_layer)
        n_compiled += 2
        print(f"Compiled {n_compiled} blocks (experts + stem + exit) with torch.compile.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    save_dir = Path(args.save_dir) if args.save_dir else Path('out/shakespeare_diffusion')
    save_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = save_dir / 'best_model.pt'

    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="ep")
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        # Anneal router noise
        frac = (epoch - 1) / max(args.epochs - 1, 1)
        model.router_noise_scale = 1.0 * (1 - frac)

        for step in range(args.steps_per_epoch):
            prompt, target = train_ds.sample_batch(args.batch_size, device)

            # Random mask ratio t ~ U(0.1, 1.0) per sample
            t = 0.1 + 0.9 * torch.rand(args.batch_size, 1, device=device)
            mask = torch.rand(args.batch_size, args.output_len, device=device) < t
            # Ensure at least one mask per sample
            if not mask.any(dim=-1).all():
                mask[:, 0] = True

            logits, _ = model(prompt, target, mask)

            # Loss: CE only on originally masked positions
            # LLaDA weighting: 1/t per sample
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, model.vocab_size),
                target.reshape(-1),
                reduction='none'
            ).reshape(args.batch_size, args.output_len)

            # Mask and weight by 1/t
            masked_loss = per_token_loss * mask.float()
            per_sample_loss = masked_loss.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
            weighted_loss = (per_sample_loss / t.squeeze(-1)).mean()

            optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Track unweighted loss for reporting
            with torch.no_grad():
                epoch_loss += per_sample_loss.mean().item()
                preds = logits.argmax(dim=-1)
                correct = (preds == target) & mask
                acc = correct.sum().float() / mask.sum().float()
                epoch_acc += acc.item()

        scheduler.step()
        avg_loss = epoch_loss / args.steps_per_epoch
        avg_acc = epoch_acc / args.steps_per_epoch
        mean_hops = model.last_mean_hops
        if isinstance(mean_hops, torch.Tensor):
            mean_hops = mean_hops.item()

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = 10
        with torch.no_grad():
            for _ in range(n_val):
                prompt, target = val_ds.sample_batch(args.batch_size, device)
                t = 0.5 * torch.ones(args.batch_size, 1, device=device)  # fixed 50% mask for val
                mask = torch.rand(args.batch_size, args.output_len, device=device) < t
                mask[:, 0] = True

                logits, _ = model(prompt, target, mask)
                per_token = F.cross_entropy(
                    logits.reshape(-1, model.vocab_size),
                    target.reshape(-1),
                    reduction='none'
                ).reshape(args.batch_size, args.output_len)
                masked = per_token * mask.float()
                val_loss += (masked.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).mean().item()
                preds = logits.argmax(dim=-1)
                correct = (preds == target) & mask
                val_acc += correct.sum().float() / mask.sum().float()

        val_loss /= n_val
        val_acc = (val_acc / n_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'state_dict': model.state_dict(),
                'args': vars(args),
                'char2idx': char2idx,
                'idx2char': idx2char,
                'params': n_params,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, best_ckpt_path)
            tqdm.write(f"  [epoch {epoch}] New best val_loss={val_loss:.4f} vacc={val_acc:.1%} → saved {best_ckpt_path}")

        pbar.set_postfix(
            loss=f"{avg_loss:.3f}",
            acc=f"{avg_acc:.1%}",
            val=f"{val_loss:.3f}",
            vacc=f"{val_acc:.1%}",
            hops=f"{mean_hops:.1f}",
            rtr=f"{model.router_noise_scale:.2f}",
        )

    return model, char2idx, idx2char


# ---------------------------------------------------------------------------
# Sampling (generation)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model, prompt_text: str, gen_len: int, char2idx: dict, idx2char: dict,
             device: torch.device, n_steps: int = 20) -> str:
    """Generate text via iterative demasking.

    Start with fully masked output, iteratively unmask via forward passes
    with low-confidence remasking.
    """
    model.eval()

    prompt_ids = encode(prompt_text, char2idx).unsqueeze(0).to(device)  # (1, L_in)
    output_ids = torch.zeros(1, gen_len, dtype=torch.long, device=device)  # placeholder
    current_mask = torch.ones(1, gen_len, dtype=torch.bool, device=device)  # all masked

    for step in range(n_steps):
        # Forward pass
        logits, _ = model(prompt_ids, output_ids, current_mask)  # (1, gen_len, vocab)

        # Predict all positions
        probs = logits.softmax(dim=-1)
        pred_ids = probs.argmax(dim=-1)  # (1, gen_len)
        confidence = probs.max(dim=-1).values  # (1, gen_len)

        # Fill in predictions for masked positions
        output_ids = torch.where(current_mask, pred_ids, output_ids)

        # Determine how many to keep unmasked this step
        n_masked = current_mask.sum().item()
        if n_masked == 0:
            break

        # Unmask fraction: linear schedule from 0 to 1 over n_steps
        unmask_frac = (step + 1) / n_steps
        n_to_unmask = max(1, int(n_masked * unmask_frac))
        n_to_keep_masked = max(0, n_masked - n_to_unmask)

        if n_to_keep_masked == 0 or step == n_steps - 1:
            # Final step: unmask everything
            current_mask = torch.zeros_like(current_mask)
        else:
            # Remask the least confident among currently masked positions
            conf = confidence.clone()
            conf[~current_mask] = float('inf')  # don't remask already unmasked
            _, sorted_idx = conf.sort(dim=-1)
            # The n_to_keep_masked lowest-confidence positions stay masked
            new_mask = torch.zeros_like(current_mask)
            new_mask.scatter_(1, sorted_idx[:, :n_to_keep_masked], True)
            current_mask = new_mask

    return decode(output_ids[0], idx2char)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Shakespeare Masked Diffusion PoE')
    parser.add_argument('--dim', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--inner-ratio', type=float, default=1.75, help='Inner dim ratio')
    parser.add_argument('--pool-size', type=int, default=4, help='Expert pool size')
    parser.add_argument('--top-k', type=int, default=2, help='Experts per hop')
    parser.add_argument('--max-hops', type=int, default=None, help='Max routing depth')
    parser.add_argument('--local-window', type=int, default=16, help='Local attention window')
    parser.add_argument('--prompt-len', type=int, default=64, help='Prompt length in chars')
    parser.add_argument('--output-len', type=int, default=64, help='Output length in chars')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--save-dir', type=str, default=None, help='Save dir')
    parser.add_argument('--compile', action='store_true', help='torch.compile the full model')
    parser.add_argument('--compile-blocks', action='store_true', help='torch.compile individual expert/stem/exit blocks')
    args = parser.parse_args()

    print("=" * 60)
    print("MaskedDiffusionPoE — Shakespeare")
    print("=" * 60)

    model, char2idx, idx2char = train(args)
    device = next(model.parameters()).device

    # Generate some samples
    print("\n" + "=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)

    prompts = [
        "ROMEO:\nO, she doth teach the torches to burn bright!\n",
        "HAMLET:\nTo be, or not to be, that is the question:\n",
        "KING:\nOnce more unto the breach, dear friends,\n",
    ]

    for prompt in prompts:
        # Truncate prompt to prompt_len
        prompt = prompt[-args.prompt_len:]
        gen = generate(model, prompt, args.output_len, char2idx, idx2char, device)
        print(f"\n--- Prompt ---\n{prompt}")
        print(f"--- Generated ---\n{gen}")
        print()

    # Save
    save_dir = Path(args.save_dir) if args.save_dir else Path('out/shakespeare_diffusion')
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / 'model.pt'
    torch.save({
        'state_dict': model.state_dict(),
        'args': vars(args),
        'char2idx': char2idx,
        'idx2char': idx2char,
        'params': sum(p.numel() for p in model.parameters()),
    }, ckpt_path)
    print(f"Saved checkpoint → {ckpt_path}")


if __name__ == '__main__':
    main()
