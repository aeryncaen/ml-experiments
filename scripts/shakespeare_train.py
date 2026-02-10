#!/usr/bin/env python3
"""Unified Shakespeare trainer — supports autoregressive and diffusion modes,
with any backbone (MHA baseline, ULB-PoE, etc).

Usage:
    # AR baseline with MHA+SwiGLU
    python scripts/shakespeare_train.py --mode ar --arch mha --epochs 50

    # Diffusion with ULB-PoE
    python scripts/shakespeare_train.py --mode diffusion --arch ulb-poe --epochs 50

    # Diffusion with MHA+SwiGLU (coming later)
    # python scripts/shakespeare_train.py --mode diffusion --arch mha ...
"""

import sys, argparse, os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_shakespeare(data_dir: str = "data") -> tuple[str, dict, dict]:
    """Download and load Shakespeare text. Returns (text, char2idx, idx2char)."""
    data_path = Path(data_dir) / "shakespeare.txt"
    if not data_path.exists():
        print(f"Downloading Shakespeare -> {data_path}")
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


class TextDataset:
    """Samples contiguous chunks from encoded text."""

    def __init__(self, data: torch.Tensor, seq_len: int):
        self.data = data
        self.seq_len = seq_len

    def sample_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns (B, seq_len) token chunks."""
        max_start = len(self.data) - self.seq_len
        starts = torch.randint(0, max_start, (batch_size,))
        return torch.stack([self.data[s:s + self.seq_len] for s in starts]).to(device)


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------

def build_mha(vocab_size: int, args) -> nn.Module:
    """Build a CausalTransformer (MHA + SwiGLU baseline)."""
    from ulb.transformer import CausalTransformer
    return CausalTransformer(
        vocab_size=vocab_size,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )


def build_ulb(vocab_size: int, args) -> nn.Module:
    """Build a CausalULB (ULB blocks, same embed/head as MHA baseline)."""
    from ulb.transformer import CausalULB
    return CausalULB(
        vocab_size=vocab_size,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
        inner_ratio=args.inner_ratio,
    )


def build_ulb_poe(vocab_size: int, args) -> nn.Module:
    """Build a MaskedDiffusionPoE (ULB diffusion variant)."""
    from ulb.block import ULBConfig
    from ulb.diffusion import MaskedDiffusionPoE
    cfg = ULBConfig(
        d_model=args.dim,
        n_heads=args.n_heads,
        paired=True,
        attn_mode='blend',
        inner_ratio=args.inner_ratio,
    )
    return MaskedDiffusionPoE(
        ulb_config=cfg,
        vocab_size=vocab_size,
        max_seq_len=args.seq_len,
        pool_size=args.pool_size,
        top_k=args.top_k,
        max_hops=args.max_hops,
        local_window=args.local_window,
        router_mode=args.router_mode,
        router_noise=args.router_noise,
        block_shared_fraction=args.block_shared_fraction,
        router_shared_fraction=args.router_shared_fraction,
        hop_shared_fraction=args.hop_shared_fraction,
    )


ARCH_BUILDERS = {
    'mha': build_mha,
    'ulb': build_ulb,
    'ulb-poe': build_ulb_poe,
}


# ---------------------------------------------------------------------------
# AR training
# ---------------------------------------------------------------------------

def train_step_ar(model, batch: torch.Tensor, optimizer, grad_clip: float):
    """Standard next-token prediction step.

    batch is (B, T) tokens. Predict token[t+1] from token[0..t].
    """
    x = batch[:, :-1]   # (B, T-1) input
    y = batch[:, 1:]     # (B, T-1) target

    logits = model(x)    # (B, T-1, vocab)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()

    return loss.item(), acc


@torch.no_grad()
def val_step_ar(model, batch: torch.Tensor):
    """Validation step for AR."""
    x = batch[:, :-1]
    y = batch[:, 1:]
    logits = model(x)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    preds = logits.argmax(dim=-1)
    acc = (preds == y).float().mean().item()
    return loss.item(), acc


# ---------------------------------------------------------------------------
# Diffusion training
# ---------------------------------------------------------------------------

def train_step_diffusion(model, batch: torch.Tensor, optimizer, grad_clip: float,
                         prompt_len: int, output_len: int):
    """Masked diffusion step (LLaDA-style).

    batch is (B, T) tokens where T = prompt_len + output_len.
    """
    prompt = batch[:, :prompt_len]
    target = batch[:, prompt_len:]
    B = prompt.shape[0]

    # Random mask ratio t ~ U(0.1, 1.0) per sample
    t = 0.1 + 0.9 * torch.rand(B, 1, device=batch.device)
    mask = torch.rand(B, output_len, device=batch.device) < t
    mask[:, 0] = True  # ensure at least one masked

    logits, _ = model(prompt, target, mask)

    # CE only on masked positions, weighted by 1/t
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction='none'
    ).reshape(B, output_len)

    masked_loss = per_token_loss * mask.float()
    per_sample_loss = masked_loss.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
    weighted_loss = (per_sample_loss / t.squeeze(-1)).mean()

    optimizer.zero_grad()
    weighted_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    with torch.no_grad():
        unweighted_loss = per_sample_loss.mean().item()
        preds = logits.argmax(dim=-1)
        correct = (preds == target) & mask
        acc = correct.sum().float() / mask.sum().float()

    return unweighted_loss, acc.item()


@torch.no_grad()
def val_step_diffusion(model, batch: torch.Tensor, prompt_len: int, output_len: int):
    """Validation step for diffusion (fixed 50% mask)."""
    prompt = batch[:, :prompt_len]
    target = batch[:, prompt_len:]
    B = prompt.shape[0]

    t = 0.5 * torch.ones(B, 1, device=batch.device)
    mask = torch.rand(B, output_len, device=batch.device) < t
    mask[:, 0] = True

    logits, _ = model(prompt, target, mask)

    per_token = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
        reduction='none'
    ).reshape(B, output_len)

    masked = per_token * mask.float()
    loss = (masked.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)).mean().item()
    preds = logits.argmax(dim=-1)
    correct = (preds == target) & mask
    acc = (correct.sum().float() / mask.sum().float()).item()

    return loss, acc


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_ar(model, prompt_text: str, gen_len: int, char2idx: dict, idx2char: dict,
                device: torch.device, temperature: float = 0.8) -> str:
    """Autoregressive generation with temperature sampling."""
    model.eval()
    ids = encode(prompt_text, char2idx).unsqueeze(0).to(device)  # (1, L)
    max_ctx = model.max_seq_len  # don't exceed RoPE / pos embed range

    for _ in range(gen_len):
        ctx = ids[:, -max_ctx:]                  # sliding window
        logits = model(ctx)                      # (1, ctx_len, vocab)
        next_logits = logits[:, -1, :] / temperature  # (1, vocab)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1)    # (1, 1)
        ids = torch.cat([ids, next_id], dim=1)

    generated = ids[0, len(prompt_text):]
    return decode(generated, idx2char)


@torch.no_grad()
def generate_diffusion(model, prompt_text: str, gen_len: int, char2idx: dict,
                       idx2char: dict, device: torch.device,
                       n_steps: int = 20) -> str:
    """Iterative demasking generation."""
    model.eval()

    prompt_ids = encode(prompt_text, char2idx).unsqueeze(0).to(device)
    output_ids = torch.zeros(1, gen_len, dtype=torch.long, device=device)
    current_mask = torch.ones(1, gen_len, dtype=torch.bool, device=device)

    for step in range(n_steps):
        logits, confidence = model(prompt_ids, output_ids, current_mask)

        pred_ids = logits.argmax(dim=-1)
        output_ids = torch.where(current_mask, pred_ids, output_ids)

        n_masked = current_mask.sum().item()
        if n_masked == 0:
            break

        unmask_frac = (step + 1) / n_steps
        n_to_unmask = max(1, int(n_masked * unmask_frac))
        n_to_keep_masked = max(0, n_masked - n_to_unmask)

        if n_to_keep_masked == 0 or step == n_steps - 1:
            current_mask = torch.zeros_like(current_mask)
        else:
            conf = confidence.clone()
            conf[~current_mask] = float('inf')
            _, sorted_idx = conf.sort(dim=-1)
            new_mask = torch.zeros_like(current_mask)
            new_mask.scatter_(1, sorted_idx[:, :n_to_keep_masked], True)
            current_mask = new_mask

    return decode(output_ids[0], idx2char)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    is_diffusion = args.mode == 'diffusion'

    # Data
    text, char2idx, idx2char = load_shakespeare()
    data = encode(text, char2idx)
    vocab_size = len(char2idx)
    print(f"Shakespeare: {len(text):,} chars, vocab_size={vocab_size}")

    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    print(f"Train: {len(train_data):,} chars, Val: {len(val_data):,} chars")

    train_ds = TextDataset(train_data, args.seq_len)
    val_ds = TextDataset(val_data, args.seq_len)

    # Model
    if is_diffusion and args.arch == 'mha':
        print("ERROR: MHA diffusion not yet implemented. Use --arch ulb-poe for diffusion.")
        sys.exit(1)

    model = ARCH_BUILDERS[args.arch](vocab_size, args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Arch: {args.arch}, Mode: {args.mode}, Params: {n_params:,}")
    print(f"  dim={args.dim}, n_heads={args.n_heads}, n_layers={args.n_layers}, seq_len={args.seq_len}")
    if is_diffusion:
        print(f"  prompt_len={args.prompt_len}, output_len={args.output_len}")
        print(f"  pool_size={args.pool_size}, top_k={args.top_k}, max_hops={getattr(model, 'max_hops', 'N/A')}")
        print(f"  router_mode={args.router_mode}, router_noise={args.router_noise}")

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode=args.compile_mode)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_dir = Path(args.save_dir) if args.save_dir else Path(f'out/shakespeare_{args.arch}_{args.mode}')
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    best_ckpt_path = save_dir / 'best_model.pt'

    # Diffusion-specific
    if is_diffusion:
        prompt_len = args.prompt_len
        output_len = args.output_len
        assert prompt_len + output_len == args.seq_len, \
            f"prompt_len ({prompt_len}) + output_len ({output_len}) must equal seq_len ({args.seq_len})"

    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="ep")
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        # Anneal router noise for diffusion
        if is_diffusion and hasattr(model, 'router_noise_scale'):
            frac = (epoch - 1) / max(args.epochs - 1, 1)
            model.router_noise_scale = args.router_noise * (1 - frac)

        for step in range(args.steps_per_epoch):
            batch = train_ds.sample_batch(args.batch_size, device)

            if is_diffusion:
                loss, acc = train_step_diffusion(model, batch, optimizer, args.grad_clip,
                                                  prompt_len, output_len)
            else:
                loss, acc = train_step_ar(model, batch, optimizer, args.grad_clip)

            epoch_loss += loss
            epoch_acc += acc

        scheduler.step()
        avg_loss = epoch_loss / args.steps_per_epoch
        avg_acc = epoch_acc / args.steps_per_epoch

        # Validation
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = args.val_batches
        for _ in range(n_val):
            batch = val_ds.sample_batch(args.batch_size, device)
            if is_diffusion:
                vl, va = val_step_diffusion(model, batch, prompt_len, output_len)
            else:
                vl, va = val_step_ar(model, batch)
            val_loss += vl
            val_acc += va
        val_loss /= n_val
        val_acc /= n_val

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
            tqdm.write(f"  [epoch {epoch}] New best val_loss={val_loss:.4f} "
                       f"vacc={val_acc:.1%} -> saved {best_ckpt_path}")

        # Status bar
        postfix = dict(
            loss=f"{avg_loss:.3f}",
            acc=f"{avg_acc:.1%}",
            val=f"{val_loss:.3f}",
            vacc=f"{val_acc:.1%}",
        )
        if is_diffusion and hasattr(model, 'last_mean_hops'):
            mean_hops = model.last_mean_hops
            if isinstance(mean_hops, torch.Tensor):
                mean_hops = mean_hops.item()
            postfix['hops'] = f"{mean_hops:.1f}"
        if is_diffusion and hasattr(model, 'router_noise_scale'):
            postfix['rtr'] = f"{model.router_noise_scale:.2f}"
        pbar.set_postfix(**postfix)

    return model, char2idx, idx2char


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Shakespeare trainer (AR / Diffusion)')

    # Mode and architecture
    parser.add_argument('--mode', type=str, default='ar', choices=['ar', 'diffusion'],
                        help='Training mode: autoregressive or masked diffusion')
    parser.add_argument('--arch', type=str, default='mha', choices=list(ARCH_BUILDERS.keys()),
                        help='Model architecture')

    # Model
    parser.add_argument('--dim', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of layers (MHA) or passed to PoE')
    parser.add_argument('--inner-ratio', type=float, default=1.75, help='Inner dim ratio (ULB)')
    parser.add_argument('--seq-len', type=int, default=80, help='Total sequence length')

    # Diffusion-specific
    parser.add_argument('--prompt-len', type=int, default=64, help='Prompt length (diffusion)')
    parser.add_argument('--output-len', type=int, default=16, help='Output length (diffusion)')
    parser.add_argument('--pool-size', type=int, default=4, help='Expert pool size')
    parser.add_argument('--top-k', type=int, default=2, help='Experts per hop')
    parser.add_argument('--max-hops', type=int, default=None, help='Max routing depth')
    parser.add_argument('--local-window', type=int, default=16, help='Local attention window')
    parser.add_argument('--router-mode', type=str, default='single',
                        choices=['squared', 'single', 'half'],
                        help='Router exit slot density')
    parser.add_argument('--router-noise', type=float, default=1.0,
                        help='Starting router noise scale')
    parser.add_argument('--block-shared-fraction', type=float, default=0.0,
                        help='Expert block weight sharing fraction')
    parser.add_argument('--router-shared-fraction', type=float, default=0.0,
                        help='Router weight sharing fraction')
    parser.add_argument('--hop-shared-fraction', type=float, default=0.0,
                        help='Hop embed/gate weight sharing fraction')

    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--val-batches', type=int, default=10, help='Validation batches')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--save-dir', type=str, default=None, help='Save directory')
    parser.add_argument('--compile', action='store_true', help='torch.compile the model')
    parser.add_argument('--compile-mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')

    args = parser.parse_args()

    print("=" * 60)
    print(f"Shakespeare — {args.arch.upper()} / {args.mode.upper()}")
    print("=" * 60)

    model, char2idx, idx2char = train(args)
    device = next(model.parameters()).device

    # Generate samples
    print("\n" + "=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)

    prompts = [
        "ROMEO:\nO, she doth teach the torches to burn bright!\n",
        "HAMLET:\nTo be, or not to be, that is the question:\n",
        "KING:\nOnce more unto the breach, dear friends,\n",
    ]

    gen_len = args.output_len if args.mode == 'diffusion' else 128

    for prompt in prompts:
        # Truncate prompt to fit within model's context
        if args.mode == 'ar':
            # AR uses sliding window, but prompt should fit in one context
            max_prompt = args.seq_len - 1
        else:
            max_prompt = args.prompt_len
        prompt_text = prompt[-max_prompt:]

        if args.mode == 'ar':
            gen = generate_ar(model, prompt_text, gen_len, char2idx, idx2char, device)
        else:
            gen = generate_diffusion(model, prompt_text, gen_len, char2idx, idx2char, device)

        print(f"\n--- Prompt ---\n{prompt_text}")
        print(f"--- Generated ---\n{gen}")
        print()

    # Final save
    save_dir = Path(args.save_dir) if args.save_dir else Path(f'out/shakespeare_{args.arch}_{args.mode}')
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / 'final_model.pt'
    torch.save({
        'state_dict': model.state_dict(),
        'args': vars(args),
        'char2idx': char2idx,
        'idx2char': idx2char,
        'params': sum(p.numel() for p in model.parameters()),
    }, ckpt_path)
    print(f"Saved final checkpoint -> {ckpt_path}")


if __name__ == '__main__':
    main()
