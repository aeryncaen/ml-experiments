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

import sys, argparse, math, os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class EMA:
    """Exponential moving average of model parameters.

    Usage:
        ema = EMA(model.parameters(), decay=0.9999)
        # After each optimizer step:
        ema.update(model.parameters())
        # For validation:
        ema.store(model.parameters())   # save current weights
        ema.copy_to(model.parameters()) # load EMA weights
        # ... validate ...
        ema.restore(model.parameters()) # restore training weights
    """

    def __init__(self, parameters, decay: float = 0.9999):
        self.decay = decay
        self.num_updates = 0
        self.shadow = [p.clone().detach() for p in parameters if p.requires_grad]
        self.backup = []

    def update(self, parameters):
        self.num_updates += 1
        # Warmup: effective decay ramps up from 0 to target
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus = 1.0 - decay
        with torch.no_grad():
            for s, p in zip(self.shadow, (p for p in parameters if p.requires_grad)):
                s.sub_(one_minus * (s - p))

    def copy_to(self, parameters):
        for s, p in zip(self.shadow, (p for p in parameters if p.requires_grad)):
            p.data.copy_(s.data)

    def store(self, parameters):
        self.backup = [p.clone() for p in parameters if p.requires_grad]

    def restore(self, parameters):
        for b, p in zip(self.backup, (p for p in parameters if p.requires_grad)):
            p.data.copy_(b.data)
        self.backup = []


SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


VOCAB_SIZE = 256  # byte-level tokenization


def load_shakespeare(data_dir: str = "data") -> str:
    """Download and load Shakespeare text. Returns raw text."""
    data_path = Path(data_dir) / "shakespeare.txt"
    if not data_path.exists():
        print(f"Downloading Shakespeare -> {data_path}")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(SHAKESPEARE_URL, data_path)

    return data_path.read_text()


def encode(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode('utf-8')), dtype=torch.long)


def decode(ids: torch.Tensor) -> str:
    return bytes(ids.tolist()).decode('utf-8', errors='replace')


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

class StackedLM(nn.Module):
    """Language model wrapper: embed + stacker + head.

    Wraps any stacker (StackedULB, MoEStackedULB, PoolOfExperts) with
    token embeddings and an output head. Weight-tied.

    The stacker operates on (B, T, D) hidden states. This wrapper adds
    the embed/head bookkeeping.

    Args:
        stacker: A stacker module (StackedULB, MoEStackedULB, or PoolOfExperts).
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        max_seq_len: Maximum sequence length (stored for generation).
    """

    def __init__(self, stacker: nn.Module, vocab_size: int, dim: int, max_seq_len: int):
        super().__init__()
        self.stacker = stacker
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.token_embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(token_ids)
        x = self.stacker(x)
        return self.head(x)


# --- Plain (non-MoE, non-PoE) builders ---

def build_mha(vocab_size: int, args) -> nn.Module:
    """Build a CausalTransformer (MHA + SwiGLU baseline)."""
    from mha import CausalTransformer
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
        k_mix=args.k_mix,
        is_causal=not args.no_causal,
        embed_lerp=args.embed_lerp,
    )


# --- MoE builders ---

def build_mha_moe(vocab_size: int, args) -> nn.Module:
    """Build MoE-stacked MHA (CausalMHALayer blocks)."""
    from mha import CausalMHALayer
    from ulb.stack import MoEStackedULB
    make_layer = lambda: CausalMHALayer(
        dim=args.dim, n_heads=args.n_heads, max_seq_len=args.seq_len)
    stacker = MoEStackedULB(
        make_layer=make_layer,
        n_layers=args.n_layers,
        dim=args.dim,
        n_experts=args.n_experts,
        top_k=args.top_k,
        version=args.moe_version,
        router_mode=args.moe_router_mode,
    )
    return StackedLM(stacker, vocab_size, args.dim, args.seq_len)


def build_ulb_moe(vocab_size: int, args) -> nn.Module:
    """Build MoE-stacked ULB blocks."""
    from ulb.block import ULBBlock, ULBConfig
    from ulb.stack import MoEStackedULB
    config = ULBConfig(
        d_model=args.dim,
        n_heads=args.n_heads,
        paired=True,
        attn_mode='blend',
        inner_ratio=args.inner_ratio,
        k_mix=args.k_mix,
        is_causal=not args.no_causal,
    )
    make_layer = lambda: ULBBlock(config)
    stacker = MoEStackedULB(
        make_layer=make_layer,
        n_layers=args.n_layers,
        dim=args.dim,
        n_experts=args.n_experts,
        top_k=args.top_k,
        version=args.moe_version,
        router_mode=args.moe_router_mode,
    )
    return StackedLM(stacker, vocab_size, args.dim, args.seq_len)


# --- PoE builders (AR) ---

def build_mha_poe(vocab_size: int, args) -> nn.Module:
    """Build PoolOfExperts with CausalMHALayer blocks."""
    from mha import CausalMHALayer
    from ulb.stack import PoolOfExperts
    make_layer = lambda: CausalMHALayer(
        dim=args.dim, n_heads=args.n_heads, max_seq_len=args.seq_len)
    stacker = PoolOfExperts(
        make_layer=make_layer,
        pool_size=args.pool_size,
        dim=args.dim,
        top_k=args.top_k,
        max_hops=args.max_hops,
        router_mode=args.router_mode,
        router_noise=args.router_noise,
        block_shared_fraction=args.block_shared_fraction,
        router_shared_fraction=args.router_shared_fraction,
        hop_shared_fraction=args.hop_shared_fraction,
    )
    return StackedLM(stacker, vocab_size, args.dim, args.seq_len)


def build_ulb_poe(vocab_size: int, args) -> nn.Module:
    """Build PoolOfExperts with ULB blocks (AR mode)."""
    from ulb.block import ULBBlock, ULBConfig
    from ulb.stack import PoolOfExperts
    config = ULBConfig(
        d_model=args.dim,
        n_heads=args.n_heads,
        paired=True,
        attn_mode='blend',
        inner_ratio=args.inner_ratio,
        k_mix=args.k_mix,
        is_causal=not args.no_causal,
    )
    make_layer = lambda: ULBBlock(config)
    stacker = PoolOfExperts(
        make_layer=make_layer,
        pool_size=args.pool_size,
        dim=args.dim,
        top_k=args.top_k,
        max_hops=args.max_hops,
        router_mode=args.router_mode,
        router_noise=args.router_noise,
        block_shared_fraction=args.block_shared_fraction,
        router_shared_fraction=args.router_shared_fraction,
        hop_shared_fraction=args.hop_shared_fraction,
    )
    return StackedLM(stacker, vocab_size, args.dim, args.seq_len)


# --- Diffusion PoE builder (existing, for MaskedDiffusionPoE) ---

def build_ulb_diffusion_poe(vocab_size: int, args) -> nn.Module:
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


# --- LLaDA builders ---

def build_llada_mha(vocab_size: int, args) -> nn.Module:
    """Build LLaDA with BidirectionalTransformer backbone."""
    from mha import BidirectionalTransformer
    from ulb.transformer import LLaDAModel
    backbone = BidirectionalTransformer(
        vocab_size=vocab_size,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len + args.gen_len,
    )
    return LLaDAModel(backbone, vocab_size, args.dim,
                      time_conditioning=args.time_cond,
                      subs_parameterization=args.subs)


def build_llada_ulb(vocab_size: int, args) -> nn.Module:
    """Build LLaDA with CausalULB backbone."""
    from ulb.transformer import CausalULB, LLaDAModel
    backbone = CausalULB(
        vocab_size=vocab_size,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len + args.gen_len,
        inner_ratio=args.inner_ratio,
        k_mix=args.k_mix,
        is_causal=not args.no_causal,
        embed_lerp=args.embed_lerp,
    )
    return LLaDAModel(backbone, vocab_size, args.dim,
                      time_conditioning=args.time_cond,
                      subs_parameterization=args.subs)


def build_llada_mha_moe(vocab_size: int, args) -> nn.Module:
    """Build LLaDA with MoE-stacked BidirectionalMHALayer backbone."""
    from mha import BidirectionalMHALayer
    from ulb.stack import MoEStackedULB
    from ulb.transformer import LLaDAModel
    max_sl = args.seq_len + args.gen_len
    make_layer = lambda: BidirectionalMHALayer(
        dim=args.dim, n_heads=args.n_heads, max_seq_len=max_sl)
    stacker = MoEStackedULB(
        make_layer=make_layer,
        n_layers=args.n_layers,
        dim=args.dim,
        n_experts=args.n_experts,
        top_k=args.top_k,
        version=args.moe_version,
        router_mode=args.moe_router_mode,
    )
    backbone = StackedLM(stacker, vocab_size, args.dim, max_sl)
    return LLaDAModel(backbone, vocab_size, args.dim,
                      time_conditioning=args.time_cond,
                      subs_parameterization=args.subs)


def build_llada_ulb_moe(vocab_size: int, args) -> nn.Module:
    """Build LLaDA with MoE-stacked ULB backbone."""
    from ulb.block import ULBBlock, ULBConfig
    from ulb.stack import MoEStackedULB
    from ulb.transformer import LLaDAModel
    config = ULBConfig(
        d_model=args.dim,
        n_heads=args.n_heads,
        paired=True,
        attn_mode='blend',
        inner_ratio=args.inner_ratio,
        k_mix=args.k_mix,
        is_causal=not args.no_causal,
    )
    make_layer = lambda: ULBBlock(config)
    stacker = MoEStackedULB(
        make_layer=make_layer,
        n_layers=args.n_layers,
        dim=args.dim,
        n_experts=args.n_experts,
        top_k=args.top_k,
        version=args.moe_version,
        router_mode=args.moe_router_mode,
    )
    backbone = StackedLM(stacker, vocab_size, args.dim, args.seq_len + args.gen_len)
    return LLaDAModel(backbone, vocab_size, args.dim,
                      time_conditioning=args.time_cond,
                      subs_parameterization=args.subs)


def build_llada_mha_poe(vocab_size: int, args) -> nn.Module:
    """Build LLaDA with PoolOfExperts BidirectionalMHALayer backbone."""
    from mha import BidirectionalMHALayer
    from ulb.stack import PoolOfExperts
    from ulb.transformer import LLaDAModel
    max_sl = args.seq_len + args.gen_len
    make_layer = lambda: BidirectionalMHALayer(
        dim=args.dim, n_heads=args.n_heads, max_seq_len=max_sl)
    stacker = PoolOfExperts(
        make_layer=make_layer,
        pool_size=args.pool_size,
        dim=args.dim,
        top_k=args.top_k,
        max_hops=args.max_hops,
        router_mode=args.router_mode,
        router_noise=args.router_noise,
        block_shared_fraction=args.block_shared_fraction,
        router_shared_fraction=args.router_shared_fraction,
        hop_shared_fraction=args.hop_shared_fraction,
    )
    backbone = StackedLM(stacker, vocab_size, args.dim, max_sl)
    return LLaDAModel(backbone, vocab_size, args.dim,
                      time_conditioning=args.time_cond,
                      subs_parameterization=args.subs)


def build_llada_ulb_poe(vocab_size: int, args) -> nn.Module:
    """Build LLaDA with PoolOfExperts ULB backbone."""
    from ulb.block import ULBBlock, ULBConfig
    from ulb.stack import PoolOfExperts
    from ulb.transformer import LLaDAModel
    config = ULBConfig(
        d_model=args.dim,
        n_heads=args.n_heads,
        paired=True,
        attn_mode='blend',
        inner_ratio=args.inner_ratio,
        k_mix=args.k_mix,
        is_causal=not args.no_causal,
    )
    make_layer = lambda: ULBBlock(config)
    stacker = PoolOfExperts(
        make_layer=make_layer,
        pool_size=args.pool_size,
        dim=args.dim,
        top_k=args.top_k,
        max_hops=args.max_hops,
        router_mode=args.router_mode,
        router_noise=args.router_noise,
        block_shared_fraction=args.block_shared_fraction,
        router_shared_fraction=args.router_shared_fraction,
        hop_shared_fraction=args.hop_shared_fraction,
    )
    backbone = StackedLM(stacker, vocab_size, args.dim, args.seq_len + args.gen_len)
    return LLaDAModel(backbone, vocab_size, args.dim,
                      time_conditioning=args.time_cond,
                      subs_parameterization=args.subs)


ARCH_BUILDERS = {
    'mha': build_mha,
    'ulb': build_ulb,
    'mha-moe': build_mha_moe,
    'ulb-moe': build_ulb_moe,
    'mha-poe': build_mha_poe,
    'ulb-poe': build_ulb_poe,
    'ulb-diffusion-poe': build_ulb_diffusion_poe,
}

LLADA_BUILDERS = {
    'mha': build_llada_mha,
    'ulb': build_llada_ulb,
    'mha-moe': build_llada_mha_moe,
    'ulb-moe': build_llada_ulb_moe,
    'mha-poe': build_llada_mha_poe,
    'ulb-poe': build_llada_ulb_poe,
}


# --- MLM (Deep Hybrid) builders ---

def build_mlm_mha(vocab_size: int, args) -> nn.Module:
    """Build DeepMLM with BidirectionalMHALayer blocks."""
    from mha import BidirectionalMHALayer
    from ulb.mlm import DeepMLM
    max_sl = args.seq_len + args.gen_len
    make_layer = lambda: BidirectionalMHALayer(
        dim=args.dim, n_heads=args.n_heads, max_seq_len=max_sl)
    return DeepMLM(make_layer, args.n_layers, vocab_size, args.dim, max_sl)


def build_mlm_ulb(vocab_size: int, args) -> nn.Module:
    """Build DeepMLM with ULBBlock blocks."""
    from ulb.block import ULBBlock, ULBConfig
    from ulb.mlm import DeepMLM
    config = ULBConfig(
        d_model=args.dim,
        n_heads=args.n_heads,
        paired=True,
        inner_ratio=args.inner_ratio,
        k_mix=args.k_mix,
        is_causal=not args.no_causal,
    )
    max_sl = args.seq_len + args.gen_len
    make_layer = lambda: ULBBlock(config)
    return DeepMLM(make_layer, args.n_layers, vocab_size, args.dim, max_sl)


def build_mlm_mha_moe(vocab_size: int, args) -> nn.Module:
    """Build DeepMLMMoE with BidirectionalMHALayer experts."""
    from mha import BidirectionalMHALayer
    from ulb.mlm import DeepMLMMoE
    max_sl = args.seq_len + args.gen_len
    make_layer = lambda: BidirectionalMHALayer(
        dim=args.dim, n_heads=args.n_heads, max_seq_len=max_sl)
    return DeepMLMMoE(make_layer, args.n_layers, vocab_size, args.dim, max_sl,
                      n_experts=args.n_experts, top_k=args.top_k,
                      version=args.moe_version, router_mode=args.moe_router_mode)


def build_mlm_ulb_moe(vocab_size: int, args) -> nn.Module:
    """Build DeepMLMMoE with ULBBlock experts."""
    from ulb.block import ULBBlock, ULBConfig
    from ulb.mlm import DeepMLMMoE
    config = ULBConfig(
        d_model=args.dim,
        n_heads=args.n_heads,
        paired=True,
        inner_ratio=args.inner_ratio,
        k_mix=args.k_mix,
        is_causal=not args.no_causal,
    )
    max_sl = args.seq_len + args.gen_len
    make_layer = lambda: ULBBlock(config)
    return DeepMLMMoE(make_layer, args.n_layers, vocab_size, args.dim, max_sl,
                      n_experts=args.n_experts, top_k=args.top_k,
                      version=args.moe_version, router_mode=args.moe_router_mode)


MLM_BUILDERS = {
    'mha': build_mlm_mha,
    'ulb': build_mlm_ulb,
    'mha-moe': build_mlm_mha_moe,
    'ulb-moe': build_mlm_ulb_moe,
}


# ---------------------------------------------------------------------------
# AR training
# ---------------------------------------------------------------------------

def _get_aux_loss(model):
    """Collect aux_loss from MoE/PoE routing.

    Looks for aux_loss on model itself, model.stacker, or model.backbone.stacker.
    """
    for obj in [model, getattr(model, 'stacker', None),
                getattr(getattr(model, 'backbone', None), 'stacker', None)]:
        if obj is not None:
            aux = getattr(obj, 'aux_loss', 0.0)
            if isinstance(aux, (int, float)):
                if aux != 0.0:
                    return aux
            elif hasattr(aux, 'item'):
                return aux
    return 0.0


def train_step_ar(model, batch: torch.Tensor, optimizer, grad_clip: float):
    """Standard next-token prediction step.

    batch is (B, T) tokens. Predict token[t+1] from token[0..t].
    """
    x = batch[:, :-1]   # (B, T-1) input
    y = batch[:, 1:]     # (B, T-1) target

    logits = model(x)    # (B, T-1, vocab)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    loss = loss + _get_aux_loss(model)

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
# LLaDA training (masked diffusion, Algorithm 1 from paper)
# ---------------------------------------------------------------------------

def train_step_llada(model, batch: torch.Tensor, optimizer, grad_clip: float,
                     antithetic: bool = True):
    """LLaDA masked diffusion training step.

    Matches GSAI-ML/LLaDA GUIDELINES.md:
    - t ~ U(0,1) per sample, p_mask = (1-eps)*t + eps
    - mask each token independently with probability p_mask
    - CE on masked tokens, each divided by its p_mask
    - sum all, normalize by B*T

    Enhancements from MDLM:
    - Antithetic t-sampling: stratified t values across batch for lower variance
    - Pass t to model for time conditioning (if model supports it)
    """
    B, T = batch.shape
    eps = 1e-3

    # Sample t with optional antithetic (stratified) sampling
    if antithetic:
        # Stratified: each sample gets a different offset in [0, 1)
        offset = torch.arange(B, device=batch.device, dtype=torch.float32) / B
        t = (torch.rand(1, device=batch.device) / B + offset) % 1.0  # (B,)
    else:
        t = torch.rand(B, device=batch.device)

    # p_mask per sample: ranges from eps to 1.0
    p_mask = (1 - eps) * t + eps  # (B,)
    p_mask_expanded = p_mask[:, None].expand(B, T)  # (B, T)

    # Independent per-token masking
    mask = torch.rand(B, T, device=batch.device) < p_mask_expanded  # (B, T) bool

    logits = model(batch, mask, t)  # (B, T, vocab_size) — t for time conditioning

    # CE on masked tokens, each weighted by 1/p_mask, normalized by B*T
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch.reshape(-1),
        reduction='none'
    ).reshape(B, T)

    # 1/p_mask weighting per token, only on masked positions
    token_loss = per_token_loss[mask] / p_mask_expanded[mask]
    loss = token_loss.sum() / (B * T) + _get_aux_loss(model)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    with torch.no_grad():
        avg_loss = (per_token_loss[mask].sum() / mask.sum()).item() if mask.any() else 0.0
        preds = logits.argmax(dim=-1)
        correct = (preds == batch) & mask
        acc = (correct.sum().float() / mask.sum().float()).item() if mask.any() else 0.0

    return avg_loss, acc


@torch.no_grad()
def val_step_llada(model, batch: torch.Tensor):
    """LLaDA validation step — fixed 50% mask ratio, same normalization as training."""
    B, T = batch.shape
    p_mask = 0.5

    mask = torch.rand(B, T, device=batch.device) < p_mask

    t_val = torch.full((B,), p_mask, device=batch.device)
    logits = model(batch, mask, t_val)

    per_token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        batch.reshape(-1),
        reduction='none'
    ).reshape(B, T)

    # Match training normalization: 1/p_mask per token, sum / (B*T)
    token_loss = per_token_loss[mask] / p_mask
    loss = (token_loss.sum() / (B * T)).item() if mask.any() else 0.0
    preds = logits.argmax(dim=-1)
    correct = (preds == batch) & mask
    acc = (correct.sum().float() / mask.sum().float()).item() if mask.any() else 0.0

    return loss, acc


# ---------------------------------------------------------------------------
# MLM (Deep Hybrid) training
# ---------------------------------------------------------------------------

def train_step_mlm(model, batch: torch.Tensor, optimizer, grad_clip: float,
                   prompt_len: int, output_len: int):
    """Deep MLM training step.

    All output positions are masked. CE loss on all output positions.
    No noise schedule, no mask ratio sampling, no 1/t weighting.
    """
    prompt = batch[:, :prompt_len]
    target = batch[:, prompt_len:]
    B = prompt.shape[0]

    logits = model(prompt, target)  # (B, output_len, vocab)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
    ) + _get_aux_loss(model)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        acc = (preds == target).float().mean().item()

    return loss.item(), acc


@torch.no_grad()
def val_step_mlm(model, batch: torch.Tensor, prompt_len: int, output_len: int):
    """Deep MLM validation step."""
    prompt = batch[:, :prompt_len]
    target = batch[:, prompt_len:]

    logits = model(prompt, target)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target.reshape(-1),
    ).item()

    preds = logits.argmax(dim=-1)
    acc = (preds == target).float().mean().item()

    return loss, acc


@torch.no_grad()
def generate_mlm(model, prompt_text: str, gen_len: int,
                 device: torch.device) -> str:
    """Deep MLM generation — single forward pass.

    Args:
        model: DeepMLM model.
        prompt_text: Text prompt.
        gen_len: Number of tokens to generate.
        device: Torch device.
    """
    model.eval()

    prompt_ids = encode(prompt_text).to(device)  # (P,)
    P = prompt_ids.shape[0]
    max_T = model.max_seq_len

    # Clamp gen_len to fit
    gen_len = min(gen_len, max_T - 1)
    if P + gen_len > max_T:
        prompt_ids = prompt_ids[-(max_T - gen_len):]
        P = prompt_ids.shape[0]

    # Dummy target ids (model ignores them — all positions are masked)
    dummy_target = torch.zeros(1, gen_len, dtype=torch.long, device=device)

    logits = model(prompt_ids.unsqueeze(0), dummy_target)
    output_ids = logits.argmax(dim=-1)[0]  # (gen_len,)

    return decode(output_ids)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_ar(model, prompt_text: str, gen_len: int,
                device: torch.device, temperature: float = 0.8) -> str:
    """Autoregressive generation with temperature sampling."""
    model.eval()
    ids = encode(prompt_text).unsqueeze(0).to(device)  # (1, L)
    max_ctx = model.max_seq_len  # don't exceed RoPE / pos embed range

    for _ in range(gen_len):
        ctx = ids[:, -max_ctx:]                  # sliding window
        logits = model(ctx)                      # (1, ctx_len, vocab)
        next_logits = logits[:, -1, :] / temperature  # (1, vocab)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1)    # (1, 1)
        ids = torch.cat([ids, next_id], dim=1)

    generated = ids[0, len(encode(prompt_text)):]
    return decode(generated)


@torch.no_grad()
def generate_diffusion(model, prompt_text: str, gen_len: int,
                       device: torch.device, n_steps: int = 20) -> str:
    """Iterative demasking generation."""
    model.eval()

    prompt_ids = encode(prompt_text).unsqueeze(0).to(device)
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

    return decode(output_ids[0])


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel noise for categorical sampling (per LLaDA reference).

    Uses float64 for precision — low-precision Gumbel Max improves perplexity
    but reduces generation quality (arXiv:2409.02908).
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    return logits.exp() / ((-torch.log(noise)) ** temperature)


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Precompute how many tokens to unmask at each step.

    Distributes total masked tokens evenly across steps, with remainder
    going to the first steps (so every token is accounted for).
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)  # (B, 1)
    base = mask_num // steps
    remainder = mask_num % steps
    num_transfer = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, :remainder[i]] += 1
    return num_transfer


@torch.no_grad()
def generate_llada(model, prompt_text: str, gen_len: int,
                   device: torch.device, n_steps: int = 64,
                   temperature: float = 0.0,
                   remasking: str = 'low_confidence') -> str:
    """LLaDA generation — matches reference implementation (GSAI-ML/LLaDA).

    Iterative demasking: at each step, predict all masked tokens, then
    permanently unmask the most confident ones. Supports Gumbel noise
    sampling and low_confidence or random remasking.

    Args:
        model: LLaDAModel.
        prompt_text: Text prompt.
        gen_len: Number of tokens to generate.
        device: Torch device.
        n_steps: Number of demasking steps (must be <= gen_len).
        temperature: Gumbel noise temperature (0 = greedy).
        remasking: 'low_confidence' or 'random'.
    """
    model.eval()

    prompt_ids = encode(prompt_text).to(device)  # (P,)
    P = prompt_ids.shape[0]
    max_T = model.max_seq_len

    # Clamp gen_len so prompt + generation fits within max_seq_len
    gen_len = min(gen_len, max_T - 1)  # leave room for at least 1 prompt token
    T = P + gen_len

    # Truncate prompt if still too long
    if T > max_T:
        prompt_ids = prompt_ids[-(max_T - gen_len):]
        P = prompt_ids.shape[0]
        T = P + gen_len

    # Build initial sequence: prompt (unmasked) + gen_len masked tokens
    # Use token 0 as placeholder for masked positions
    x = torch.zeros(1, T, dtype=torch.long, device=device)
    x[0, :P] = prompt_ids
    mask = torch.zeros(1, T, dtype=torch.bool, device=device)
    mask[0, P:] = True

    # Precompute how many tokens to unmask per step
    num_transfer_tokens = _get_num_transfer_tokens(mask[:, P:], n_steps)  # (1, n_steps)

    for step in range(n_steps):
        mask_index = mask.clone()

        logits = model(x, mask)  # (1, T, vocab_size)

        # Token selection: greedy or Gumbel noise
        logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)  # (1, T)

        # Confidence of chosen tokens
        if remasking == 'low_confidence':
            p = F.softmax(logits.float(), dim=-1)
            x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)  # (1, T)
        elif remasking == 'random':
            x0_p = torch.rand(1, T, device=device)
        else:
            raise ValueError(f"Unknown remasking strategy: {remasking}")

        # Only update masked positions
        x0 = torch.where(mask_index, x0, x)
        confidence = torch.where(mask_index, x0_p, -float('inf'))

        # Transfer the top-k most confident masked tokens (permanently unmask)
        transfer_index = torch.zeros_like(x0, dtype=torch.bool)
        for j in range(confidence.shape[0]):
            k = num_transfer_tokens[j, step].item()
            if k > 0:
                _, select_index = torch.topk(confidence[j], k=k)
                transfer_index[j, select_index] = True
        x[transfer_index] = x0[transfer_index]
        # Update mask: transferred tokens are no longer masked
        mask = mask & ~transfer_index

    return decode(x[0, P:])


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device)
    is_diffusion = args.mode == 'diffusion'
    is_llada = args.mode == 'llada'
    is_mlm = args.mode == 'mlm'

    # MLM: seq_len is derived from prompt_len + output_len
    if is_mlm:
        args.seq_len = args.prompt_len + args.output_len

    # Data
    text = load_shakespeare()
    data = encode(text)
    vocab_size = VOCAB_SIZE
    print(f"Shakespeare: {len(text):,} chars, vocab_size={vocab_size}")

    n_train = int(0.9 * len(data))
    train_data, val_data = data[:n_train], data[n_train:]
    print(f"Train: {len(train_data):,} chars, Val: {len(val_data):,} chars")

    train_ds = TextDataset(train_data, args.seq_len)
    val_ds = TextDataset(val_data, args.seq_len)

    # Model
    if is_diffusion and args.arch != 'ulb-diffusion-poe':
        print(f"ERROR: Diffusion mode requires --arch ulb-diffusion-poe, got '{args.arch}'.")
        sys.exit(1)

    if is_mlm:
        if args.arch not in MLM_BUILDERS:
            print(f"ERROR: MLM mode not supported for arch '{args.arch}'. "
                  f"Available: {', '.join(MLM_BUILDERS.keys())}")
            sys.exit(1)
        model = MLM_BUILDERS[args.arch](vocab_size, args).to(device)
    elif is_llada:
        if args.arch not in LLADA_BUILDERS:
            print(f"ERROR: LLaDA mode not supported for arch '{args.arch}'. "
                  f"Available: {', '.join(LLADA_BUILDERS.keys())}")
            sys.exit(1)
        model = LLADA_BUILDERS[args.arch](vocab_size, args).to(device)
    else:
        if args.arch not in ARCH_BUILDERS:
            print(f"ERROR: Unknown arch '{args.arch}'. Available: {', '.join(ARCH_BUILDERS.keys())}")
            sys.exit(1)
        model = ARCH_BUILDERS[args.arch](vocab_size, args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Arch: {args.arch}, Mode: {args.mode}, Params: {n_params:,}")
    print(f"  dim={args.dim}, n_heads={args.n_heads}, n_layers={args.n_layers}, seq_len={args.seq_len}")
    if is_mlm:
        print(f"  prompt_len={args.prompt_len}, output_len={args.output_len}")
        print(f"  n_layers={args.n_layers} (each layer predicts + re-embeds)")
    elif is_llada:
        backbone_type = type(model.backbone).__name__
        print(f"  backbone={backbone_type}")
    elif is_diffusion:
        print(f"  prompt_len={args.prompt_len}, output_len={args.output_len}")
        print(f"  pool_size={args.pool_size}, top_k={args.top_k}, max_hops={getattr(model, 'max_hops', 'N/A')}")
        print(f"  router_mode={args.router_mode}, router_noise={args.router_noise}")
    if 'moe' in args.arch:
        print(f"  n_experts={args.n_experts}, top_k={args.top_k}, moe_version={args.moe_version}, "
              f"moe_router_mode={args.moe_router_mode}")
    if 'poe' in args.arch:
        print(f"  pool_size={args.pool_size}, top_k={args.top_k}, max_hops={args.max_hops}, "
              f"router_mode={args.router_mode}, router_noise={args.router_noise}")
        print(f"  block_shared={args.block_shared_fraction}, router_shared={args.router_shared_fraction}, "
              f"hop_shared={args.hop_shared_fraction}")

    if is_llada:
        llada_features = []
        if args.time_cond:
            llada_features.append('time_cond')
        if args.subs:
            llada_features.append('subs')
        if not args.no_antithetic:
            llada_features.append('antithetic')
        if args.ema > 0:
            llada_features.append(f'ema={args.ema}')
        if llada_features:
            print(f"  llada: {', '.join(llada_features)}")

    if args.compile:
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode=args.compile_mode)

    # EMA
    ema = None
    if args.ema > 0:
        ema = EMA(model.parameters(), decay=args.ema)
        print(f"  EMA enabled (decay={args.ema})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # LR schedule: linear warmup (1 epoch, from lr/10) then cosine decay to lr/20
    warmup_steps = args.steps_per_epoch  # 1 epoch warmup
    total_steps = args.steps_per_epoch * args.epochs
    warmup_ratio = 0.1    # start at lr * 0.1
    min_ratio = 0.05      # decay to lr * 0.05
    def lr_schedule(step):
        if step < warmup_steps:
            return warmup_ratio + (1.0 - warmup_ratio) * step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + 0.5 * (1.0 - min_ratio) * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    save_dir = Path(args.save_dir) if args.save_dir else Path(f'out/shakespeare_{args.arch}_{args.mode}')
    save_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float('inf')
    best_ckpt_path = save_dir / 'best_model.pt'

    # Diffusion / MLM -specific
    if is_diffusion or is_mlm:
        prompt_len = args.prompt_len
        output_len = args.output_len
        if is_diffusion:
            assert prompt_len + output_len == args.seq_len, \
                f"prompt_len ({prompt_len}) + output_len ({output_len}) must equal seq_len ({args.seq_len})"

    # Find the inner routing module for noise annealing and hops tracking.
    # For StackedLM wrapping PoolOfExperts: model.stacker
    # For MaskedDiffusionPoE: model itself
    # For LLaDA wrapping StackedLM: model.backbone.stacker
    _router_module = None
    if hasattr(model, 'router_noise_scale'):
        _router_module = model
    elif hasattr(model, 'stacker') and hasattr(model.stacker, 'router_noise_scale'):
        _router_module = model.stacker
    elif hasattr(model, 'backbone'):
        bb = model.backbone
        if hasattr(bb, 'router_noise_scale'):
            _router_module = bb
        elif hasattr(bb, 'stacker') and hasattr(bb.stacker, 'router_noise_scale'):
            _router_module = bb.stacker

    pbar = tqdm(range(1, args.epochs + 1), desc="Training", unit="ep")
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        # Anneal router noise for PoE models
        if _router_module is not None:
            frac = (epoch - 1) / max(args.epochs - 1, 1)
            _router_module.router_noise_scale = args.router_noise * (1 - frac)

        for step in range(args.steps_per_epoch):
            batch = train_ds.sample_batch(args.batch_size, device)

            if is_mlm:
                loss, acc = train_step_mlm(model, batch, optimizer, args.grad_clip,
                                           prompt_len, output_len)
            elif is_llada:
                loss, acc = train_step_llada(model, batch, optimizer, args.grad_clip,
                                            antithetic=not args.no_antithetic)
            elif is_diffusion:
                loss, acc = train_step_diffusion(model, batch, optimizer, args.grad_clip,
                                                  prompt_len, output_len)
            else:
                loss, acc = train_step_ar(model, batch, optimizer, args.grad_clip)

            if ema is not None:
                ema.update(model.parameters())
            scheduler.step()
            epoch_loss += loss
            epoch_acc += acc

        avg_loss = epoch_loss / args.steps_per_epoch
        avg_acc = epoch_acc / args.steps_per_epoch

        # Validation — use EMA weights if available
        if ema is not None:
            ema.store(model.parameters())
            ema.copy_to(model.parameters())
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        n_val = args.val_batches
        for _ in range(n_val):
            batch = val_ds.sample_batch(args.batch_size, device)
            if is_mlm:
                vl, va = val_step_mlm(model, batch, prompt_len, output_len)
            elif is_llada:
                vl, va = val_step_llada(model, batch)
            elif is_diffusion:
                vl, va = val_step_diffusion(model, batch, prompt_len, output_len)
            else:
                vl, va = val_step_ar(model, batch)
            val_loss += vl
            val_acc += va
        val_loss /= n_val
        val_acc /= n_val
        if ema is not None:
            ema.restore(model.parameters())

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save({
                'state_dict': model.state_dict(),
                'args': vars(args),
                'vocab_size': vocab_size,
                'params': n_params,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, best_ckpt_path)
        best_marker = " *" if is_best else ""
        tqdm.write(f"  [epoch {epoch}] val_loss={val_loss:.4f} vacc={val_acc:.1%}{best_marker}")

        # Early stop on val accuracy
        if args.early_stop_acc > 0 and val_acc >= args.early_stop_acc:
            tqdm.write(f"  [epoch {epoch}] Early stop: val_acc={val_acc:.1%} >= {args.early_stop_acc:.0%}")
            break

        # Status bar
        postfix = dict(
            loss=f"{avg_loss:.3f}",
            acc=f"{avg_acc:.1%}",
            val=f"{val_loss:.3f}",
            vacc=f"{val_acc:.1%}",
        )
        # Show hops for PoE models
        _hops_src = _router_module  # PoolOfExperts tracks last_mean_hops
        if _hops_src is not None and hasattr(_hops_src, 'last_mean_hops'):
            mean_hops = _hops_src.last_mean_hops
            if isinstance(mean_hops, torch.Tensor):
                mean_hops = mean_hops.item()
            postfix['hops'] = f"{mean_hops:.1f}"
        if _router_module is not None:
            postfix['rtr'] = f"{_router_module.router_noise_scale:.2f}"
        pbar.set_postfix(**postfix)

        # Generate a sample after each epoch
        model.eval()
        sample_prompt = "KING:\nO, "
        sample_len = 64
        try:
            with torch.no_grad():
                if is_mlm:
                    sample = generate_mlm(model, sample_prompt, sample_len, device)
                elif is_llada:
                    sample = generate_llada(model, sample_prompt, sample_len, device, n_steps=32)
                elif is_diffusion:
                    sample = generate_diffusion(model, sample_prompt, sample_len, device)
                else:
                    sample = generate_ar(model, sample_prompt, sample_len, device)
            # Show on one line, escape newlines
            preview = (sample_prompt + sample).replace('\n', '\\n')
            tqdm.write(f"  [sample] {preview}")
        except Exception:
            pass  # don't crash training on generation errors

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def interactive_generate(args):
    """Load a checkpoint and run an interactive generation loop."""
    ckpt_path = args.generate
    device = torch.device(args.device)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = argparse.Namespace(**ckpt['args'])
    vocab_size = ckpt.get('vocab_size', VOCAB_SIZE)

    # Use mode/arch from checkpoint, allow CLI overrides for gen params
    mode = saved_args.mode
    arch = saved_args.arch
    temperature = args.temperature
    gen_len = args.gen_len

    print(f"Model: {arch} / {mode}, params: {ckpt.get('params', '?'):,}")
    if 'val_acc' in ckpt:
        print(f"Checkpoint from epoch {ckpt.get('epoch', '?')}, "
              f"val_loss={ckpt.get('val_loss', '?'):.4f}, val_acc={ckpt.get('val_acc', '?'):.1%}")

    # Rebuild model from saved args
    if mode == 'mlm':
        model = MLM_BUILDERS[arch](vocab_size, saved_args).to(device)
    elif mode == 'llada':
        model = LLADA_BUILDERS[arch](vocab_size, saved_args).to(device)
    else:
        model = ARCH_BUILDERS[arch](vocab_size, saved_args).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    if mode == 'ar':
        max_prompt = saved_args.seq_len - 1
    elif mode == 'llada':
        max_prompt = saved_args.seq_len  # model has room for seq_len + gen_len
    else:
        max_prompt = saved_args.prompt_len

    print(f"\nInteractive generation — type a prompt and press Enter.")
    print(f"  mode={mode}, temperature={temperature}, gen_len={gen_len}, max_prompt={max_prompt}")
    print(f"  (Ctrl+C or empty line to quit)\n")

    while True:
        try:
            prompt_text = input("prompt> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not prompt_text:
            print("Bye.")
            break

        # Allow \n in input for multi-line prompts
        prompt_text = prompt_text.replace('\\n', '\n')

        # Truncate to fit
        if len(prompt_text) > max_prompt:
            prompt_text = prompt_text[-max_prompt:]
            print(f"  (truncated to last {max_prompt} chars)")

        if mode == 'mlm':
            gen = generate_mlm(model, prompt_text, gen_len, device)
        elif mode == 'ar':
            gen = generate_ar(model, prompt_text, gen_len,
                              device, temperature=temperature)
        elif mode == 'llada':
            gen = generate_llada(model, prompt_text, gen_len, device)
        else:
            gen = generate_diffusion(model, prompt_text, gen_len, device)

        print(f"\n{prompt_text}\033[1m{gen}\033[0m\n")


def main():
    parser = argparse.ArgumentParser(description='Shakespeare trainer (AR / Diffusion)')

    # Mode and architecture
    parser.add_argument('--generate', type=str, default=None, metavar='CKPT',
                        help='Load checkpoint and run interactive generation (skip training)')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (AR)')
    parser.add_argument('--gen-len', type=int, default=256, help='Generation length')
    parser.add_argument('--mode', type=str, default='ar', choices=['ar', 'diffusion', 'llada', 'mlm'],
                        help='Training mode: autoregressive, masked diffusion (PoE), llada, or mlm (deep hybrid)')
    parser.add_argument('--arch', type=str, default='mha', choices=list(ARCH_BUILDERS.keys()),
                        help='Model architecture: mha, ulb, mha-moe, ulb-moe, mha-poe, ulb-poe, ulb-diffusion-poe')

    # Model
    parser.add_argument('--dim', type=int, default=128, help='Model dimension')
    parser.add_argument('--n-heads', type=int, default=4, help='Attention heads')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--inner-ratio', type=float, default=1.75, help='Inner dim ratio (ULB)')
    parser.add_argument('--k-mix', type=str, default='lerp',
                        choices=['none', 'lerp', 'add', 'acausal_lerp', 'acausal_add', 'conv2', 'conv3'],
                        help='K temporal mixing mode (ULB)')
    parser.add_argument('--no-causal', action='store_true', default=False,
                        help='Disable causal mask on attention (ULB)')
    parser.add_argument('--embed-lerp', action='store_true', default=False,
                        help='Acausal lerp on token embeddings before blocks (ULB)')
    parser.add_argument('--seq-len', type=int, default=80, help='Total sequence length')

    # MoE-specific
    parser.add_argument('--n-experts', type=int, default=4, help='Number of MoE experts per layer')
    parser.add_argument('--moe-version', type=int, default=1, choices=[1, 2],
                        help='MoE routing version: 1=self-routed, 2=sender-routed')
    parser.add_argument('--moe-router-mode', type=str, default='relu', choices=['topk', 'relu'],
                        help='MoE routing mode: topk or relu (ReMoE)')

    # PoE-specific
    parser.add_argument('--pool-size', type=int, default=None, help='Expert pool size (PoE, default=n_layers)')
    parser.add_argument('--max-hops', type=int, default=None, help='Max routing depth (PoE)')
    parser.add_argument('--router-mode', type=str, default='single',
                        choices=['squared', 'single', 'half'],
                        help='Router exit slot density (PoE)')
    parser.add_argument('--router-noise', type=float, default=1.0,
                        help='Starting router noise scale (PoE)')
    parser.add_argument('--block-shared-fraction', type=float, default=0.0,
                        help='Expert block weight sharing fraction (PoE)')
    parser.add_argument('--router-shared-fraction', type=float, default=0.0,
                        help='Router weight sharing fraction (PoE)')
    parser.add_argument('--hop-shared-fraction', type=float, default=0.0,
                        help='Hop embed/gate weight sharing fraction (PoE)')

    # Shared between MoE and PoE
    parser.add_argument('--top-k', type=int, default=2, help='Experts per hop/layer (MoE/PoE)')

    # Diffusion-specific
    parser.add_argument('--prompt-len', type=int, default=64, help='Prompt length (diffusion)')
    parser.add_argument('--output-len', type=int, default=16, help='Output length (diffusion)')
    parser.add_argument('--local-window', type=int, default=16, help='Local attention window (diffusion PoE)')

    # LLaDA enhancements
    parser.add_argument('--time-cond', action='store_true', default=False,
                        help='Time conditioning: embed mask ratio and add to input (LLaDA)')
    parser.add_argument('--subs', action='store_true', default=False,
                        help='SUBS parameterization: clamp unmasked logits to one-hot (LLaDA)')
    parser.add_argument('--no-antithetic', action='store_true', default=False,
                        help='Disable antithetic (stratified) t-sampling (LLaDA)')
    parser.add_argument('--ema', type=float, default=0.0,
                        help='EMA decay (0 to disable, typical 0.9999)')

    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--steps-per-epoch', type=int, default=100, help='Steps per epoch')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--val-batches', type=int, default=10, help='Validation batches')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--early-stop-acc', type=float, default=0.99,
                        help='Stop training when val acc exceeds this (0 to disable)')
    parser.add_argument('--device', type=str, default='cpu', help='Device')
    parser.add_argument('--save-dir', type=str, default=None, help='Save directory')
    parser.add_argument('--compile', action='store_true', help='torch.compile the model')
    parser.add_argument('--compile-mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')

    args = parser.parse_args()

    # Default pool_size to n_layers if not set
    if args.pool_size is None:
        args.pool_size = args.n_layers

    # --generate mode: load checkpoint and run interactive prompt loop
    if args.generate:
        interactive_generate(args)
        return

    print("=" * 60)
    print(f"Shakespeare — {args.arch.upper()} / {args.mode.upper()}")
    print("=" * 60)

    model = train(args)
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

    if args.mode in ('diffusion', 'mlm'):
        gen_len = args.output_len
    elif args.mode == 'llada':
        gen_len = args.gen_len
    else:
        gen_len = 128

    for prompt in prompts:
        # Truncate prompt to fit within model's context
        if args.mode == 'ar':
            max_prompt = args.seq_len - 1
        elif args.mode == 'llada':
            max_prompt = args.seq_len  # model has room for seq_len + gen_len
        else:
            max_prompt = args.prompt_len
        prompt_text = prompt[-max_prompt:]

        if args.mode == 'mlm':
            gen = generate_mlm(model, prompt_text, gen_len, device)
        elif args.mode == 'ar':
            gen = generate_ar(model, prompt_text, gen_len, device)
        elif args.mode == 'llada':
            gen = generate_llada(model, prompt_text, gen_len, device)
        else:
            gen = generate_diffusion(model, prompt_text, gen_len, device)

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
        'vocab_size': VOCAB_SIZE,
        'params': sum(p.numel() for p in model.parameters()),
    }, ckpt_path)
    print(f"Saved final checkpoint -> {ckpt_path}")


if __name__ == '__main__':
    main()
