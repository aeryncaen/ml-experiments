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


NULL_CHAR = 0  # NULL symbol index (used for BOS/EOS in trigrams)
ALPHABET_SIZE = 66  # 65 Shakespeare chars + NULL
NGRAM = 2  # bigram tokenization
VOCAB_SIZE = ALPHABET_SIZE ** NGRAM  # 4,356 bigram tokens

# Character <-> index mapping (built lazily from data)
_char_to_idx: dict[str, int] = {}
_idx_to_char: dict[int, str] = {0: ''}  # NULL -> empty string


def _build_char_map(text: str) -> None:
    """Build char<->index mapping from Shakespeare text. Call once."""
    global _char_to_idx, _idx_to_char
    if _char_to_idx:
        return  # already built
    chars = sorted(set(text))
    for i, c in enumerate(chars, start=1):  # 1-indexed, 0 = NULL
        _char_to_idx[c] = i
        _idx_to_char[i] = c
    assert len(chars) == 65, f"Expected 65 unique chars, got {len(chars)}"


def _ngram_token(chars: list[int]) -> int:
    """Encode NGRAM char indices into a single token."""
    token = 0
    for c in chars:
        token = token * ALPHABET_SIZE + c
    return token


def _ungram(token: int) -> list[int]:
    """Decode a token into NGRAM char indices."""
    chars = []
    for _ in range(NGRAM):
        chars.append(token % ALPHABET_SIZE)
        token //= ALPHABET_SIZE
    return chars[::-1]


def load_shakespeare(data_dir: str = "data") -> str:
    """Download and load Shakespeare text. Returns raw text."""
    data_path = Path(data_dir) / "shakespeare.txt"
    if not data_path.exists():
        print(f"Downloading Shakespeare -> {data_path}")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(SHAKESPEARE_URL, data_path)

    text = data_path.read_text()
    _build_char_map(text)
    return text


def encode(text: str) -> torch.Tensor:
    """Encode text to trigram token IDs with BOS and EOS.

    Always bookended: [NULL, content..., NULL].
    First trigram starts with NULL (BOS-type).
    Last trigram ends with NULL (EOS-type).

    Pads with extra NULLs before the final NULL so total length
    is divisible by NGRAM.
    """
    char_ids = [_char_to_idx[c] for c in text]

    # Bookend with NULLs
    char_ids = [NULL_CHAR] + char_ids + [NULL_CHAR]

    # Pad to multiple of NGRAM
    remainder = len(char_ids) % NGRAM
    if remainder != 0:
        # Insert NULLs before the final NULL
        for _ in range(NGRAM - remainder):
            char_ids.insert(-1, NULL_CHAR)

    tokens = []
    for i in range(0, len(char_ids), NGRAM):
        tokens.append(_ngram_token(char_ids[i:i + NGRAM]))

    return torch.tensor(tokens, dtype=torch.long)


def decode(ids: torch.Tensor) -> str:
    """Decode trigram token IDs to string, dropping NULL chars."""
    chars = []
    for token in ids.tolist():
        for c in _ungram(token):
            if c != NULL_CHAR:
                chars.append(_idx_to_char[c])
    return ''.join(chars)


def _pretokenize(text: str, seq_len: int) -> list[torch.Tensor]:
    """Pretokenize text into NGRAM parity streams.

    For trigrams (NGRAM=3), creates 3 streams with content lengths that
    are 0, 1, 2 mod 3 relative to the max, producing different char
    alignments within each trigram token.

    Each stream is (N, seq_len) tensor of contiguous non-overlapping chunks,
    each bookended with BOS/EOS NULLs.
    """
    # Max content chars: seq_len tokens * NGRAM chars - 2 NULLs
    max_content = NGRAM * seq_len - 2

    def _chunkify(content_len):
        if content_len < 1:
            return torch.zeros(0, seq_len, dtype=torch.long)
        n_samples = len(text) // content_len
        chunks = []
        for i in range(n_samples):
            start = i * content_len
            content = text[start:start + content_len]
            tokens = encode(content)
            if tokens.shape[0] == seq_len:
                chunks.append(tokens)
        if chunks:
            return torch.stack(chunks)
        return torch.zeros(0, seq_len, dtype=torch.long)

    streams = []
    for offset in range(NGRAM):
        streams.append(_chunkify(max_content - offset))
    return streams


class TextDataset:
    """Pretokenized trigram dataset with NGRAM alignment parities.

    Stores NGRAM tensors of pretokenized chunks (different content lengths).
    sample_batch randomly draws from all parities.
    """

    def __init__(self, text: str, seq_len: int):
        self.seq_len = seq_len
        self.streams = _pretokenize(text, seq_len)
        counts = [len(s) for s in self.streams]
        total = sum(counts)
        desc = " + ".join(str(c) for c in counts)
        print(f"  TextDataset: {desc} = {total} pretokenized chunks "
              f"({NGRAM} parities)")

    def sample_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Returns (B, seq_len) trigram token chunks, randomly from all parities."""
        nonempty = [s for s in self.streams if len(s) > 0]
        chunks = []
        for _ in range(batch_size):
            stream = nonempty[torch.randint(0, len(nonempty), (1,)).item()]
            idx = torch.randint(0, len(stream), (1,)).item()
            chunks.append(stream[idx])
        return torch.stack(chunks).to(device)


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

        # Megatron init — infer effective depth from stacker
        n_layers = getattr(stacker, 'n_layers', None)
        if n_layers is None:
            n_layers = getattr(stacker, 'pool_size', None)
        if n_layers is None and hasattr(stacker, 'layers'):
            n_layers = len(stacker.layers)
        if n_layers is not None:
            from ulb.block import ulb_megatron_init_
            ulb_megatron_init_(self, n_layers)

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


def build_dual_mha(vocab_size: int, args) -> nn.Module:
    """Build a DualMHA (two independent MHA models, logits subtracted)."""
    from mha import DualMHA
    return DualMHA(
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


# --- TPGL (TriplePoolGraphLearner) builders ---

def _make_tpgl_stacker(args, dim: int, is_causal: bool):
    """Build a TriplePoolGraphLearner stacker from CLI args."""
    from ulb.block import ULBConfig, UniversalSequenceBlock
    from ulb.stack import TriplePoolGraphLearner
    config = ULBConfig(
        d_model=dim,
        n_heads=args.n_heads,
        paired=True,
        attn_mode='blend',
        inner_ratio=args.inner_ratio,
        k_mix=args.k_mix,
        is_causal=is_causal,
    )
    return TriplePoolGraphLearner(
        make_seq_block=lambda: UniversalSequenceBlock(config),
        seq_pool_size=args.pool_size,
        pre_pool_size=args.pre_pool_size,
        post_pool_size=args.post_pool_size,
        dim=dim,
        inner_dim=config.inner_dim,
        seq_top_k=args.top_k,
        pre_top_k=args.pre_top_k,
        post_top_k=args.post_top_k,
        max_hops=args.max_hops,
        seq_router_mode=args.router_mode,
        pre_router_mode=args.pre_router_mode,
        post_router_mode=args.post_router_mode,
        router_noise=args.router_noise,
        swish_mode=args.swish_mode,
        seq_shared_fraction=args.block_shared_fraction,
        seq_router_shared_fraction=args.router_shared_fraction,
        seq_hop_shared_fraction=args.hop_shared_fraction,
        pre_shared_fraction=args.pre_shared_fraction,
        pre_router_shared_fraction=args.pre_router_shared_fraction,
        post_shared_fraction=args.post_shared_fraction,
        post_router_shared_fraction=args.post_router_shared_fraction,
    )


def build_ulb_tpgl(vocab_size: int, args) -> nn.Module:
    """Build TriplePoolGraphLearner (AR mode)."""
    stacker = _make_tpgl_stacker(args, args.dim, is_causal=not args.no_causal)
    return StackedLM(stacker, vocab_size, args.dim, args.seq_len)


def build_llada_ulb_tpgl(vocab_size: int, args) -> nn.Module:
    """Build LLaDA with TriplePoolGraphLearner backbone."""
    from ulb.transformer import LLaDAModel
    stacker = _make_tpgl_stacker(args, args.dim, is_causal=not args.no_causal)
    backbone = StackedLM(stacker, vocab_size, args.dim, args.seq_len + args.gen_len)
    return LLaDAModel(backbone, vocab_size, args.dim,
                      time_conditioning=args.time_cond,
                      subs_parameterization=args.subs)


def build_mlm_ulb_tpgl(vocab_size: int, args) -> nn.Module:
    """Build BertMLM with TriplePoolGraphLearner backbone."""
    from ulb.mlm import BertMLM
    stacker = _make_tpgl_stacker(args, args.dim, is_causal=not args.no_causal)
    backbone = StackedLM(stacker, vocab_size, args.dim, args.seq_len)
    return BertMLM(backbone, vocab_size, args.dim, mask_prob=args.mask_prob)


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
        is_causal=not args.no_causal,
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
        dim=args.dim, n_heads=args.n_heads, max_seq_len=max_sl,
        is_causal=not args.no_causal)
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
        dim=args.dim, n_heads=args.n_heads, max_seq_len=max_sl,
        is_causal=not args.no_causal)
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


# --- LLooM builder (AR) ---

class LLooMLM(nn.Module):
    """Language model wrapper for LLooM.

    LLooM.forward() returns (output, info) but the AR training loop expects
    model(token_ids) -> logits.  This wrapper handles embed/head and stashes
    the info dict for routing stats display.
    """

    def __init__(self, lloom: nn.Module, vocab_size: int, dim: int, max_seq_len: int):
        super().__init__()
        self.lloom = lloom
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.token_embed.weight

        # Stash for routing info (updated each forward)
        self.last_info: dict = {}

        # Megatron-style init: effective depth = 2 stems + expected expert hops.
        # Use pool_size (not max_hops) as the expected depth per side: a sample
        # visits at most pool_size distinct experts before routing becomes
        # redundant, and the exit ramp pushes samples out well before max_hops.
        # The old formula (2 + seq_max_hops + tok_max_hops = 26) treated the
        # hop budget ceiling as the actual depth, crushing expert outputs with
        # out_std = 0.02/sqrt(52) ≈ 0.003, making outbound router logits ~10000x
        # smaller than the exit ramp.
        cfg = lloom.config
        n_layers = 2 + max(cfg.seq_pool_size, cfg.tok_pool_size)
        from lloom.lloom import lloom_megatron_init_
        lloom_megatron_init_(self, n_layers)

    @property
    def router_noise_scale(self) -> float:
        return self.lloom.router_noise_scale

    @router_noise_scale.setter
    def router_noise_scale(self, val: float) -> None:
        self.lloom.router_noise_scale = val

    @property
    def last_mean_hops(self) -> float:
        info = self.last_info
        rd = info.get('mean_routing_decisions', 0.0)
        if isinstance(rd, torch.Tensor):
            rd = rd.item()
        return rd

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embed(token_ids)
        x, info = self.lloom(x)
        self.last_info = {k: v.detach() if isinstance(v, torch.Tensor) else v
                          for k, v in info.items()}
        return self.head(x)


def build_lloom(vocab_size: int, args) -> nn.Module:
    """Build LLooM for AR Shakespeare training."""
    from lloom.config import LLooMConfig
    from lloom.lloom import LLooM

    cfg = LLooMConfig(
        dim=args.dim,
        max_seq_len=args.seq_len,
        stem_n_heads=args.n_heads,
        stem_mlp_expansion=args.inner_ratio,
        seq_pool_size=args.lloom_seq_pool_size,
        seq_top_k=args.lloom_seq_top_k,
        seq_n_heads=args.n_heads,
        seq_expansion=args.inner_ratio,
        seq_max_hops=args.lloom_seq_max_hops,
        tok_pool_size=args.lloom_tok_pool_size,
        tok_top_k=args.lloom_tok_top_k,
        tok_expansion=args.inner_ratio,
        tok_max_hops=args.lloom_tok_max_hops,
        exit_bias_init=None,  # auto = log(pool_size)
        bridge_bias_init=None,
        exit_ramp_scale=args.lloom_exit_ramp_scale,
        router_noise=args.router_noise,
        shared_fraction=args.lloom_shared_fraction,
        hop_gate_dim=args.lloom_hop_gate_dim,
        max_bridge_crossings=args.lloom_max_bridges,
        is_causal=not args.no_causal,
        dropout=0.0,
    )
    lloom = LLooM(cfg)
    return LLooMLM(lloom, vocab_size, args.dim, args.seq_len)


ARCH_BUILDERS = {
    'mha': build_mha,
    'dual-mha': build_dual_mha,
    'ulb': build_ulb,
    'mha-moe': build_mha_moe,
    'ulb-moe': build_ulb_moe,
    'mha-poe': build_mha_poe,
    'ulb-poe': build_ulb_poe,
    'ulb-tpgl': build_ulb_tpgl,
    'ulb-diffusion-poe': build_ulb_diffusion_poe,
    'lloom': build_lloom,
}

LLADA_BUILDERS = {
    'mha': build_llada_mha,
    'ulb': build_llada_ulb,
    'mha-moe': build_llada_mha_moe,
    'ulb-moe': build_llada_ulb_moe,
    'mha-poe': build_llada_mha_poe,
    'ulb-poe': build_llada_ulb_poe,
    'ulb-tpgl': build_llada_ulb_tpgl,
}


# --- MLM (BERT-style) builders ---

def build_mlm_mha(vocab_size: int, args) -> nn.Module:
    """Build BertMLM with BidirectionalTransformer backbone."""
    from mha import BidirectionalTransformer
    from ulb.mlm import BertMLM
    backbone = BidirectionalTransformer(
        vocab_size=vocab_size,
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
        is_causal=not args.no_causal,
    )
    return BertMLM(backbone, vocab_size, args.dim, mask_prob=args.mask_prob)


def build_mlm_ulb(vocab_size: int, args) -> nn.Module:
    """Build BertMLM with CausalULB backbone."""
    from ulb.transformer import CausalULB
    from ulb.mlm import BertMLM
    backbone = CausalULB(
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
    return BertMLM(backbone, vocab_size, args.dim, mask_prob=args.mask_prob)


def build_mlm_mha_moe(vocab_size: int, args) -> nn.Module:
    """Build BertMLM with MoE-stacked BidirectionalMHALayer backbone."""
    from mha import BidirectionalMHALayer
    from ulb.stack import MoEStackedULB
    from ulb.mlm import BertMLM
    make_layer = lambda: BidirectionalMHALayer(
        dim=args.dim, n_heads=args.n_heads, max_seq_len=args.seq_len,
        is_causal=not args.no_causal)
    stacker = MoEStackedULB(
        make_layer=make_layer, n_layers=args.n_layers, dim=args.dim,
        n_experts=args.n_experts, top_k=args.top_k,
        version=args.moe_version, router_mode=args.moe_router_mode,
    )
    backbone = StackedLM(stacker, vocab_size, args.dim, args.seq_len)
    return BertMLM(backbone, vocab_size, args.dim, mask_prob=args.mask_prob)


def build_mlm_ulb_moe(vocab_size: int, args) -> nn.Module:
    """Build BertMLM with MoE-stacked ULBBlock backbone."""
    from ulb.block import ULBBlock, ULBConfig
    from ulb.stack import MoEStackedULB
    from ulb.mlm import BertMLM
    config = ULBConfig(
        d_model=args.dim,
        n_heads=args.n_heads,
        paired=True,
        inner_ratio=args.inner_ratio,
        k_mix=args.k_mix,
        is_causal=not args.no_causal,
    )
    make_layer = lambda: ULBBlock(config)
    stacker = MoEStackedULB(
        make_layer=make_layer, n_layers=args.n_layers, dim=args.dim,
        n_experts=args.n_experts, top_k=args.top_k,
        version=args.moe_version, router_mode=args.moe_router_mode,
    )
    backbone = StackedLM(stacker, vocab_size, args.dim, args.seq_len)
    return BertMLM(backbone, vocab_size, args.dim, mask_prob=args.mask_prob)


MLM_BUILDERS = {
    'mha': build_mlm_mha,
    'ulb': build_mlm_ulb,
    'mha-moe': build_mlm_mha_moe,
    'ulb-moe': build_mlm_ulb_moe,
    'ulb-tpgl': build_mlm_ulb_tpgl,
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
# MLM (BERT-style) training
# ---------------------------------------------------------------------------

def _make_mlm_mask(B: int, T: int, mask_prob: float, gen_len: int,
                   device: torch.device) -> torch.Tensor:
    """Build MLM mask: 50% random, 50% contiguous chunk at end.

    With trigram tokens, all positions are maskable (BOS/EOS are embedded
    in trigrams, not separate tokens).
    """
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    chunk_len = min(gen_len, T)

    for i in range(B):
        if torch.rand(1).item() < 0.5 and chunk_len > 0:
            # Contiguous chunk at end
            mask[i, T - chunk_len:T] = True
        else:
            # Random masking
            rand = torch.rand(T, device=device)
            mask[i] = rand < mask_prob

    # Ensure at least one masked position per sample
    if not mask.any():
        mask[0, 0] = True

    return mask


def train_step_mlm(model, batch: torch.Tensor, optimizer, grad_clip: float,
                   mask_prob: float = 0.40):
    """MLM training step.

    50% of samples get random masking at mask_prob rate.
    50% get a single masked trigram at end-of-sequence (next-token prediction).
    CE loss on masked positions only.
    """
    B, T = batch.shape
    device = batch.device

    mask = _make_mlm_mask(B, T, mask_prob, gen_len=1, device=device)

    logits = model(batch, mask)  # (B, T, vocab)

    # MLM loss: CE on masked positions only
    loss = F.cross_entropy(
        logits[mask].reshape(-1, logits.shape[-1]),
        batch[mask].reshape(-1),
    ) + _get_aux_loss(model)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        correct = (preds == batch) & mask
        acc = (correct.sum().float() / mask.sum().float()).item() if mask.any() else 0.0

    return loss.item(), acc


@torch.no_grad()
def val_step_mlm(model, batch: torch.Tensor, mask_prob: float = 0.40):
    """MLM validation step."""
    B, T = batch.shape
    device = batch.device

    mask = _make_mlm_mask(B, T, mask_prob, gen_len=1, device=device)

    logits = model(batch, mask)

    loss = F.cross_entropy(
        logits[mask].reshape(-1, logits.shape[-1]),
        batch[mask].reshape(-1),
    ).item()

    preds = logits.argmax(dim=-1)
    correct = (preds == batch) & mask
    acc = (correct.sum().float() / mask.sum().float()).item() if mask.any() else 0.0

    return loss, acc


@torch.no_grad()
def generate_mlm(model, prompt_text: str, gen_len: int,
                 device: torch.device, tokens_per_step: int = 1) -> str:
    """AR-style MLM generation: one trigram token per forward pass.

    Each step:
      1. Build [prompt_trigrams, generated_so_far, MASK*N]
      2. Mask N positions at the end
      3. Forward pass -> argmax at all masked positions
      4. Append predicted tokens and repeat

    Prompt is left-aligned (first trigram = [NULL, first_char] = BOS).
    gen_len is number of trigram tokens to generate.
    """
    model.eval()

    prompt_ids = encode(prompt_text).to(device)  # (P,)
    max_T = model.max_seq_len

    # Trim prompt if needed
    max_prompt = max_T - tokens_per_step
    if prompt_ids.shape[0] > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]

    generated = []

    while len(generated) < gen_len:
        n_gen = len(generated)
        remaining = gen_len - n_gen
        room = max_T - (prompt_ids.shape[0] + n_gen)
        n_mask = min(tokens_per_step, remaining, room)
        if n_mask < 1:
            break

        # Total seq: prompt + generated_so_far + n_mask slots
        T = prompt_ids.shape[0] + n_gen + n_mask
        token_ids = torch.zeros(1, T, dtype=torch.long, device=device)
        token_ids[0, :prompt_ids.shape[0]] = prompt_ids
        if generated:
            token_ids[0, prompt_ids.shape[0]:prompt_ids.shape[0]+n_gen] = \
                torch.tensor(generated, dtype=torch.long, device=device)
        # mask slots are zeros (dummy)

        mask = torch.zeros(1, T, dtype=torch.bool, device=device)
        mask[0, -n_mask:] = True  # mask the last n_mask slots

        logits = model(token_ids, mask)  # (1, T, vocab)
        new_tokens = logits[0, -n_mask:].argmax(dim=-1).tolist()
        generated.extend(new_tokens)

    return decode(torch.tensor(generated[:gen_len], dtype=torch.long))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_ar(model, prompt_text: str, gen_len: int,
                device: torch.device, temperature: float = 0.8) -> str:
    """Autoregressive generation with temperature sampling (trigram vocab)."""
    model.eval()
    prompt_ids = encode(prompt_text)
    ids = prompt_ids.unsqueeze(0).to(device)  # (1, L)
    max_ctx = model.max_seq_len

    for _ in range(gen_len):
        ctx = ids[:, -max_ctx:]                  # sliding window
        logits = model(ctx)                      # (1, ctx_len, vocab)
        next_logits = logits[:, -1, :] / temperature  # (1, vocab)
        probs = F.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1)    # (1, 1)
        ids = torch.cat([ids, next_id], dim=1)

    generated = ids[0, len(prompt_ids):]
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

    # MLM: uses seq_len directly (flat sequence, no prompt/output split)

    # Data
    text = load_shakespeare()
    vocab_size = VOCAB_SIZE
    ngram_label = {1: 'char', 2: 'bigram', 3: 'trigram'}.get(NGRAM, f'{NGRAM}-gram')
    print(f"Shakespeare: {len(text):,} chars, alphabet={ALPHABET_SIZE}, "
          f"vocab_size={vocab_size} ({ngram_label})")

    n_train = int(0.9 * len(text))
    train_text, val_text = text[:n_train], text[n_train:]
    print(f"Train: {len(train_text):,} chars, Val: {len(val_text):,} chars")

    train_ds = TextDataset(train_text, args.seq_len)
    val_ds = TextDataset(val_text, args.seq_len)

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
        print(f"  mask_prob=0.15→{args.mask_prob} over first half (curriculum), gen_len=1 (single next-token)")
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
    if 'tpgl' in args.arch:
        print(f"  seq_pool={args.pool_size}, pre_pool={args.pre_pool_size}, post_pool={args.post_pool_size}")
        print(f"  seq_top_k={args.top_k}, pre_top_k={args.pre_top_k}, post_top_k={args.post_top_k}")
        print(f"  max_hops={args.max_hops}, router_noise={args.router_noise}")
        print(f"  seq_router={args.router_mode}, pre_router={args.pre_router_mode}, post_router={args.post_router_mode}")
        print(f"  seq_shared={args.block_shared_fraction}, seq_router_shared={args.router_shared_fraction}, "
              f"hop_shared={args.hop_shared_fraction}")
        print(f"  pre_shared={args.pre_shared_fraction}, pre_router_shared={args.pre_router_shared_fraction}")
        print(f"  post_shared={args.post_shared_fraction}, post_router_shared={args.post_router_shared_fraction}")
    if args.arch == 'lloom':
        print(f"  seq_pool={args.lloom_seq_pool_size} (top_k={args.lloom_seq_top_k}, max_hops={args.lloom_seq_max_hops})")
        print(f"  tok_pool={args.lloom_tok_pool_size} (top_k={args.lloom_tok_top_k}, max_hops={args.lloom_tok_max_hops})")
        print(f"  exit_ramp={args.lloom_exit_ramp_scale}, shared={args.lloom_shared_fraction}, "
              f"bridges={args.lloom_max_bridges}, router_noise={args.router_noise}")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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
    start_epoch = 0

    # Resume from checkpoint
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            print(f"  Resuming from epoch {start_epoch}")
        if 'best_val_loss' in ckpt:
            best_val_loss = ckpt['best_val_loss']
            print(f"  Best val_loss so far: {best_val_loss:.4f}")

    # Diffusion-specific
    if is_diffusion:
        prompt_len = args.prompt_len
        output_len = args.output_len
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

    pbar = tqdm(range(start_epoch + 1, args.epochs + 1), desc="Training", unit="ep")
    for epoch in pbar:
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        # Anneal router noise for PoE models
        if _router_module is not None:
            frac = (epoch - 1) / max(args.epochs - 1, 1)
            _router_module.router_noise_scale = args.router_noise * (1 - frac)

        # MLM curriculum: ramp mask_prob 15%→target over first half
        if is_mlm:
            ramp_end = max(args.epochs // 2, 1)
            frac = min((epoch - 1) / ramp_end, 1.0)
            cur_mask_prob = 0.15 + frac * (args.mask_prob - 0.15)

        step_pbar = tqdm(range(args.steps_per_epoch), desc=f"  Epoch {epoch}",
                         unit="step", leave=False)
        for step in step_pbar:
            batch = train_ds.sample_batch(args.batch_size, device)

            if is_mlm:
                loss, acc = train_step_mlm(model, batch, optimizer, args.grad_clip,
                                           mask_prob=cur_mask_prob)
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

            # Update inner progress bar with running averages
            steps_done = step + 1
            step_pbar.set_postfix(loss=f"{epoch_loss/steps_done:.4f}",
                                  acc=f"{epoch_acc/steps_done:.1%}")

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
                vl, va = val_step_mlm(model, batch, mask_prob=cur_mask_prob)
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
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args': vars(args),
                'vocab_size': vocab_size,
                'params': n_params,
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_loss': best_val_loss,
            }, best_ckpt_path)
        best_marker = " *" if is_best else ""
        mask_info = f" mask={cur_mask_prob:.0%}" if is_mlm else ""

        # LLooM-specific routing stats on the epoch summary line
        lloom_info = ""
        if hasattr(model, 'last_info') and model.last_info:
            li = model.last_info
            def _v(k):
                v = li.get(k, 0.0)
                return v.item() if isinstance(v, torch.Tensor) else float(v)
            sh = _v('mean_seq_hops')
            th = _v('mean_tok_hops')
            br = _v('mean_bridges')
            parts = [f"seq_h={sh:.1f}", f"tok_h={th:.1f}", f"br={br:.2f}"]
            if 'stem_go_seq' in li:
                gs = _v('stem_go_seq')
                gt = _v('stem_go_tok')
                ge = _v('stem_go_exit')
                parts.append(f"stem(s={gs:.0%}/t={gt:.0%}/x={ge:.0%})")
            lloom_info = " " + " ".join(parts)

        tqdm.write(f"  [epoch {epoch}] val_loss={val_loss:.4f} vacc={val_acc:.1%}{mask_info}{lloom_info}{best_marker}")

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
        if is_mlm:
            postfix['mask'] = f"{cur_mask_prob:.0%}"
        # Show hops for PoE models
        _hops_src = _router_module  # PoolOfExperts tracks last_mean_hops
        if _hops_src is not None and hasattr(_hops_src, 'last_mean_hops'):
            mean_hops = _hops_src.last_mean_hops
            if isinstance(mean_hops, torch.Tensor):
                mean_hops = mean_hops.item()
            postfix['hops'] = f"{mean_hops:.1f}"
        if _router_module is not None:
            postfix['rtr'] = f"{_router_module.router_noise_scale:.2f}"
        # LLooM: show seq/tok hops and bridge count in progress bar too
        if hasattr(model, 'last_info') and model.last_info:
            li = model.last_info
            def _v2(k):
                v = li.get(k, 0.0)
                return v.item() if isinstance(v, torch.Tensor) else float(v)
            postfix['sh'] = f"{_v2('mean_seq_hops'):.1f}"
            postfix['th'] = f"{_v2('mean_tok_hops'):.1f}"
            postfix['br'] = f"{_v2('mean_bridges'):.2f}"
        pbar.set_postfix(**postfix)

        # Generate a sample after each epoch
        model.eval()
        try:
            with torch.no_grad():
                if is_mlm:
                    def _show_mlm_sample(label, sb, m):
                        lg = model(sb, m)
                        pr = lg.argmax(dim=-1)
                        orig = sb[0]
                        res = orig.clone()
                        res[m[0]] = pr[0][m[0]]
                        nm = m.sum().item()
                        nc = ((pr[0] == orig) & m[0]).sum().item()
                        esc = lambda t: decode(t).replace('\n', '\\n')
                        # Build masked view: decode each trigram, replace masked with '___'
                        mv_parts = []
                        for j in range(orig.shape[0]):
                            if m[0, j]:
                                mv_parts.append('_' * NGRAM)
                            else:
                                mv_parts.append(decode(orig[j:j+1]))
                        mv_str = ''.join(mv_parts).replace('\n', '\\n')
                        tqdm.write(f"  [{label}] {nc}/{nm} masked correct")
                        tqdm.write(f"    orig:  ...{esc(orig)[-80:]}")
                        tqdm.write(f"    mask:  ...{mv_str[-80:]}")
                        tqdm.write(f"    recon: ...{esc(res)[-80:]}")

                    # Random masking sample (forced random, no contiguous)
                    sb1 = val_ds.sample_batch(1, device)
                    T1 = sb1.shape[1]
                    m1 = torch.rand(1, T1, device=device) < cur_mask_prob
                    if not m1.any():
                        m1[0, 0] = True
                    _show_mlm_sample("sample rand", sb1, m1)

                    # AR-style generation sample
                    sample_prompt = "KING:\nO, "
                    sample = generate_mlm(model, sample_prompt, 64, device)
                    preview = (sample_prompt + sample).replace('\n', '\\n')
                    tqdm.write(f"  [sample gen] {preview}")
                else:
                    sample_prompt = "KING:\nO, "
                    sample_len = 64
                    if is_llada:
                        sample = generate_llada(model, sample_prompt, sample_len, device, n_steps=32)
                    elif is_diffusion:
                        sample = generate_diffusion(model, sample_prompt, sample_len, device)
                    else:
                        sample = generate_ar(model, sample_prompt, sample_len, device)
                    preview = (sample_prompt + sample).replace('\n', '\\n')
                    tqdm.write(f"  [sample] {preview}")
        except Exception:
            pass  # don't crash training on generation errors

    return model, optimizer, scheduler, best_val_loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def interactive_generate(args):
    """Load a checkpoint and run an interactive generation loop."""
    # Ensure char map is built (needed for trigram encode/decode)
    load_shakespeare()

    ckpt_path = args.generate
    device = torch.device(args.device)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved_args = argparse.Namespace(**ckpt['args'])

    # Restore ngram setting from checkpoint
    global NGRAM, VOCAB_SIZE
    NGRAM = getattr(saved_args, 'ngram', 2)  # old checkpoints default to bigram
    VOCAB_SIZE = ALPHABET_SIZE ** NGRAM

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

    # max_prompt in chars (~NGRAM * seq_len since each trigram = 3 chars)
    max_prompt_chars = (saved_args.seq_len - 1) * NGRAM
    if mode == 'llada':
        max_prompt_chars = saved_args.seq_len * NGRAM

    print(f"\nInteractive generation — type a prompt and press Enter.")
    print(f"  mode={mode}, temperature={temperature}, gen_len={gen_len}, "
          f"max_prompt~={max_prompt_chars} chars")
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

        # Truncate to fit (generate functions handle exact truncation internally)
        if len(prompt_text) > max_prompt_chars:
            prompt_text = prompt_text[-max_prompt_chars:]
            print(f"  (truncated to last {max_prompt_chars} chars)")

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
    parser.add_argument('--ngram', type=int, default=1,
                        help='N-gram tokenization (1=chars/bytes, 2=bigrams, 3=trigrams)')
    parser.add_argument('--mode', type=str, default='ar', choices=['ar', 'diffusion', 'llada', 'mlm'],
                        help='Training mode: autoregressive, masked diffusion (PoE), llada, or mlm (BERT-style)')
    parser.add_argument('--arch', type=str, default='mha', choices=list(ARCH_BUILDERS.keys()),
                        help='Model architecture: mha, ulb, mha-moe, ulb-moe, mha-poe, ulb-poe, ulb-tpgl, ulb-diffusion-poe, lloom')

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
    parser.add_argument('--swish-mode', type=str, default='learnable', choices=['learnable', 'silu'],
                        help='Activation mode for ULB/databank (learnable Swish or SiLU)')

    # LLooM-specific
    parser.add_argument('--lloom-seq-pool-size', type=int, default=4,
                        help='Sequence-side expert pool size (LLooM)')
    parser.add_argument('--lloom-tok-pool-size', type=int, default=4,
                        help='Token-side expert pool size (LLooM)')
    parser.add_argument('--lloom-seq-top-k', type=int, default=2,
                        help='Sequence-side top-k (LLooM)')
    parser.add_argument('--lloom-tok-top-k', type=int, default=2,
                        help='Token-side top-k (LLooM)')
    parser.add_argument('--lloom-seq-max-hops', type=int, default=8,
                        help='Sequence-side max hops (LLooM)')
    parser.add_argument('--lloom-tok-max-hops', type=int, default=16,
                        help='Token-side max hops (LLooM)')
    parser.add_argument('--lloom-exit-ramp-scale', type=float, default=2.0,
                        help='Exit bias ramp scale (LLooM)')
    parser.add_argument('--lloom-shared-fraction', type=float, default=0.5,
                        help='Weight sharing fraction (LLooM)')
    parser.add_argument('--lloom-hop-gate-dim', type=int, default=12,
                        help='Hop gate prefix dim (LLooM)')
    parser.add_argument('--lloom-max-bridges', type=int, default=2,
                        help='Max bridge crossings per sample (LLooM)')

    # TPGL-specific (TriplePoolGraphLearner)
    parser.add_argument('--pre-pool-size', type=int, default=None,
                        help='Pre-TokenPool databank size (TPGL, default=pool_size)')
    parser.add_argument('--post-pool-size', type=int, default=None,
                        help='Post-TokenPool databank size (TPGL, default=pool_size)')
    parser.add_argument('--pre-top-k', type=int, default=2, help='Top-k for pre-TokenPool routing (TPGL)')
    parser.add_argument('--post-top-k', type=int, default=2, help='Top-k for post-TokenPool routing (TPGL)')
    parser.add_argument('--pre-router-mode', type=str, default='single',
                        choices=['squared', 'single', 'half'],
                        help='Pre-TokenPool router exit slot density (TPGL)')
    parser.add_argument('--post-router-mode', type=str, default='single',
                        choices=['squared', 'single', 'half'],
                        help='Post-TokenPool router exit slot density (TPGL)')
    parser.add_argument('--pre-shared-fraction', type=float, default=0.0,
                        help='Pre-TokenPool (up-proj) weight sharing fraction (TPGL)')
    parser.add_argument('--pre-router-shared-fraction', type=float, default=0.0,
                        help='Pre-TokenPool router weight sharing fraction (TPGL)')
    parser.add_argument('--post-shared-fraction', type=float, default=0.0,
                        help='Post-TokenPool (down-proj) weight sharing fraction (TPGL)')
    parser.add_argument('--post-router-shared-fraction', type=float, default=0.0,
                        help='Post-TokenPool router weight sharing fraction (TPGL)')

    # Shared between MoE and PoE
    parser.add_argument('--top-k', type=int, default=2, help='Experts per hop/layer (MoE/PoE/TPGL seq pool)')

    # Diffusion-specific
    parser.add_argument('--prompt-len', type=int, default=64, help='Prompt length (diffusion)')
    parser.add_argument('--output-len', type=int, default=16, help='Output length (diffusion)')
    parser.add_argument('--local-window', type=int, default=16, help='Local attention window (diffusion PoE)')

    # MLM (BERT-style)
    parser.add_argument('--mask-prob', type=float, default=0.40,
                        help='Final mask probability (MLM, default 0.40, curriculum ramps from 0.15)')

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
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--early-stop-acc', type=float, default=0.99,
                        help='Stop training when val acc exceeds this (0 to disable)')
    _default_device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    parser.add_argument('--device', type=str, default=_default_device, help='Device')
    parser.add_argument('--save-dir', type=str, default=None, help='Save directory')
    parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                        help='Resume training from checkpoint (path to .pt file)')
    parser.add_argument('--compile', action='store_true', help='torch.compile the model')
    parser.add_argument('--compile-mode', type=str, default='default',
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')

    args = parser.parse_args()

    # Set n-gram tokenization globals
    global NGRAM, VOCAB_SIZE
    NGRAM = args.ngram
    VOCAB_SIZE = ALPHABET_SIZE ** NGRAM

    # Default pool_size to n_layers if not set
    if args.pool_size is None:
        args.pool_size = args.n_layers
    # Default TPGL token pool sizes to pool_size if not set
    if args.pre_pool_size is None:
        args.pre_pool_size = args.pool_size
    if args.post_pool_size is None:
        args.post_pool_size = args.pool_size

    # --generate mode: load checkpoint and run interactive prompt loop
    if args.generate:
        interactive_generate(args)
        return

    print("=" * 60)
    print(f"Shakespeare — {args.arch.upper()} / {args.mode.upper()}")
    print("=" * 60)

    model, optimizer, scheduler, best_val_loss = train(args)
    device = next(model.parameters()).device

    # Generate samples
    print("\n" + "=" * 60)
    print("GENERATION SAMPLES")
    print("=" * 60)

    if args.mode == 'mlm':
        # AR-style generation using MLM (one token at a time)
        prompts = [
            "ROMEO:\nO, she doth teach the torches to burn bright!\n",
            "HAMLET:\nTo be, or not to be, that is the question:\n",
            "KING:\nOnce more unto the breach, dear friends,\n",
        ]
        gen_len = 128
        for prompt in prompts:
            gen = generate_mlm(model, prompt, gen_len, device)
            print(f"\n--- Prompt: {prompt.rstrip()!r} ---")
            print(f"Generated:\n{gen}")
            print()
    else:
        prompts = [
            "ROMEO:\nO, she doth teach the torches to burn bright!\n",
            "HAMLET:\nTo be, or not to be, that is the question:\n",
            "KING:\nOnce more unto the breach, dear friends,\n",
        ]

        if args.mode == 'diffusion':
            gen_len = args.output_len
        elif args.mode == 'llada':
            gen_len = args.gen_len
        else:
            gen_len = 128

        for prompt in prompts:
            if args.mode == 'ar':
                gen = generate_ar(model, prompt, gen_len, device)
            elif args.mode == 'llada':
                gen = generate_llada(model, prompt, gen_len, device)
            else:
                gen = generate_diffusion(model, prompt, gen_len, device)

            print(f"\n--- Prompt ---\n{prompt.rstrip()}")
            print(f"--- Generated ---\n{gen}")
            print()

    # Final save
    save_dir = Path(args.save_dir) if args.save_dir else Path(f'out/shakespeare_{args.arch}_{args.mode}')
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / 'final_model.pt'
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'args': vars(args),
        'vocab_size': VOCAB_SIZE,
        'params': sum(p.numel() for p in model.parameters()),
        'epoch': args.epochs,
        'best_val_loss': best_val_loss,
    }, ckpt_path)
    print(f"Saved final checkpoint -> {ckpt_path}")


if __name__ == '__main__':
    main()
