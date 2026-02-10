"""ULB language model architectures and LLaDA diffusion wrapper.

CausalULB: ULB blocks, embed/head wrapper.
LLaDAModel: Masked diffusion wrapper — works with any backbone.

Usage:
    model = CausalULB(vocab_size=256, dim=128, n_heads=4, n_layers=4, max_seq_len=256)
    logits = model(token_ids)  # (B, T, vocab_size)

    # LLaDA diffusion:
    llada = LLaDAModel(backbone, vocab_size=256, dim=128)
    logits = llada(token_ids, mask)  # (B, T, vocab_size)
"""

import torch
import torch.nn as nn


class CausalULB(nn.Module):
    """Causal language model using ULB blocks.

    Same embed/head structure as CausalTransformer but uses ULBBlock
    (pre-norm residual stacking) instead of MHA+SwiGLU.

    Args:
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of ULB blocks.
        max_seq_len: Maximum sequence length for positional embeddings.
        paired: Use paired-head attention (default True).
        attn_mode: Attention mode — 'softmax', 'silu2', or 'blend' (default 'blend').
        inner_ratio: QKV/attention inner dim as ratio of d_model (default 1.75).
    """

    def __init__(self, vocab_size: int, dim: int = 128, n_heads: int = 4,
                 n_layers: int = 4, max_seq_len: int = 256,
                 paired: bool = True, attn_mode: str = 'blend',
                 inner_ratio: float = 1.75,
                 k_mix: str = 'lerp',
                 is_causal: bool = True,
                 embed_lerp: bool = False):
        super().__init__()
        from .block import ULBBlock, ULBConfig
        from .lerp import EmbeddingLerp
        from .norm import RMSNorm

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        self.token_embed = nn.Embedding(vocab_size, dim)
        self.embed_lerp: EmbeddingLerp | None = EmbeddingLerp(dim) if embed_lerp else None

        config = ULBConfig(
            d_model=dim,
            n_heads=n_heads,
            paired=paired,
            attn_mode=attn_mode,
            inner_ratio=inner_ratio,
            k_mix=k_mix,
            is_causal=is_causal,
        )
        self.blocks = nn.ModuleList([ULBBlock(config) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_embed.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) token indices.

        Returns:
            (B, T, vocab_size) logits.
        """
        B, T = token_ids.shape
        x = self.token_embed(token_ids)
        if self.embed_lerp is not None:
            x = self.embed_lerp(x)

        for norm, block in zip(self.norms, self.blocks):
            x = x + block(norm(x))

        return self.head(self.final_norm(x))


class LLaDAModel(nn.Module):
    """LLaDA masked diffusion wrapper.

    Wraps any backbone (CausalULB, BidirectionalTransformer, etc.) and adds:
    - A learned mask token embedding
    - Masking/unmasking logic for training and generation
    - Predictions over the original vocab (not the mask token)

    The backbone must have:
    - token_embed: nn.Embedding
    - vocab_size, dim, max_seq_len attributes

    LLaDA does NOT add a mask token to the vocab. Instead, the mask token
    is a learned embedding vector that replaces masked positions at the
    embedding level before passing through the backbone's layers.

    Args:
        backbone: The underlying model (we call its layers directly
                  so we can inject mask embeddings).
        vocab_size: Original vocab size (without mask token).
        dim: Model dimension.
    """

    def __init__(self, backbone: nn.Module, vocab_size: int, dim: int):
        super().__init__()
        self.backbone = backbone
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = backbone.max_seq_len

        # Learned mask token embedding
        self.mask_embed = nn.Parameter(torch.randn(dim) * 0.02)

    def _embed_with_mask(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Embed tokens, replacing masked positions with the mask embedding.

        Args:
            token_ids: (B, T) original token indices.
            mask: (B, T) bool — True means masked (replaced with mask_embed).

        Returns:
            (B, T, D) embeddings.
        """
        x = self.backbone.token_embed(token_ids)  # (B, T, D)
        x = torch.where(mask.unsqueeze(-1), self.mask_embed, x)
        return x

    def _run_backbone_from_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone's layers on pre-computed embeddings.

        Handles:
        - TransformerBlock-based (has rope_freqs): CausalTransformer, BidirectionalTransformer
        - ULB-based (has norms + blocks): CausalULB
        - StackedLM (has stacker): MoE/PoE wrapped models

        Args:
            x: (B, T, D) embeddings.

        Returns:
            (B, T, vocab_size) logits.
        """
        bb = self.backbone

        if hasattr(bb, 'stacker'):
            # StackedLM wrapping MoE/PoE stacker — stacker handles norms+residual
            x = bb.stacker(x)
            return bb.head(x)
        elif hasattr(bb, 'rope_freqs'):
            # TransformerBlock-based (Causal or Bidirectional)
            for block in bb.blocks:
                x = block(x, bb.rope_freqs)
        elif hasattr(bb, 'norms'):
            # ULB-based
            for norm, block in zip(bb.norms, bb.blocks):
                x = x + block(norm(x))
        else:
            raise ValueError(f"Unknown backbone type: {type(bb)}")

        return bb.head(bb.final_norm(x))

    def forward(self, token_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for LLaDA.

        Args:
            token_ids: (B, T) ground-truth token indices (clean data x0).
            mask: (B, T) bool — True for masked positions.

        Returns:
            (B, T, vocab_size) logits — predictions for ALL positions,
            but loss should only be computed on masked positions.
        """
        x = self._embed_with_mask(token_ids, mask)
        return self._run_backbone_from_embeddings(x)
