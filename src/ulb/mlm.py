"""DeepMLM — Deep Hybrid Masked Language Model.

Plain bidirectional stack with prompt/output split. Masked output positions
get a learned mask embedding, unmasked get ground-truth token embeddings.
Position info comes from RoPE inside the attention layers — no absolute
positional embeddings.

Architecture:
    [prompt_tokens | mask_or_gt_tokens] → embed → stack → final logits

Training: CE loss on masked output positions.
Generation: Single forward pass with all output positions masked.
"""

from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


class DeepMLM(nn.Module):
    """Plain bidirectional MLM with prompt/output split.

    Args:
        make_layer: Callable that creates a single block (no arguments).
                    Block signature: forward(x) -> delta, where x is pre-normed.
        n_layers: Number of layers.
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        max_seq_len: Maximum total sequence length (prompt + output).
    """

    def __init__(self, make_layer, n_layers: int, vocab_size: int,
                 dim: int, max_seq_len: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers

        # Layers
        self.layers = nn.ModuleList([make_layer() for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)

        # Embeddings — no pos_embed, RoPE handles position
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.mask_embed = nn.Parameter(torch.randn(dim) * 0.02)

        # Output head (weight-tied to token_embed)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        self.output_head.weight = self.token_embed.weight

        # Stored after forward for logging
        self.aux_loss = 0.0

    def forward(self, prompt_ids: torch.Tensor,
                target_ids: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass.

        Args:
            prompt_ids: (B, P) prompt token indices.
            target_ids: (B, G) ground-truth output token indices.
            mask: (B, G) bool — True = masked (predict), False = given (ground truth).
                  If None, all output positions are masked.

        Returns:
            logits: (B, G, vocab_size) predictions at output positions.
        """
        B, P = prompt_ids.shape
        G = target_ids.shape[1]

        # Embed prompt
        prompt_x = self.token_embed(prompt_ids)  # (B, P, D)

        # Embed output: masked → mask_embed, unmasked → token_embed
        if mask is None:
            output_x = self.mask_embed.unsqueeze(0).expand(B, G, -1)
        else:
            gt_embeds = self.token_embed(target_ids)  # (B, G, D)
            mask_embeds = self.mask_embed.unsqueeze(0).expand(B, G, -1)
            output_x = torch.where(mask.unsqueeze(-1), mask_embeds, gt_embeds)

        x = torch.cat([prompt_x, output_x], dim=1)  # (B, P+G, D)

        aux = 0.0
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
            aux = aux + getattr(layer, 'aux_loss', 0.0)

        # Final prediction at output positions only
        x = self.final_norm(x)
        logits = self.output_head(x[:, P:])  # (B, G, vocab)

        self.aux_loss = aux
        return logits


class DeepMLMMoE(nn.Module):
    """Plain bidirectional MLM with MoE routing and prompt/output split.

    Args:
        make_layer: Callable that creates a single expert block.
        n_layers: Number of MoE layers.
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        max_seq_len: Maximum total sequence length (prompt + output).
        n_experts: Number of experts per layer.
        top_k: Top-k expert selection per sample.
        version: MoE routing version (1 or 2).
        router_mode: 'topk' or 'relu'.
    """

    def __init__(self, make_layer: Callable[[], nn.Module],
                 n_layers: int, vocab_size: int, dim: int, max_seq_len: int,
                 n_experts: int = 4, top_k: int = 2,
                 version: Literal[1, 2] = 1,
                 router_mode: Literal['topk', 'relu'] = 'topk',
                 **kwargs):
        super().__init__()
        from .stack import MoEStackedULB

        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len

        # MoE stacker
        self.stacker = MoEStackedULB(
            make_layer=make_layer,
            n_layers=n_layers,
            dim=dim,
            n_experts=n_experts,
            top_k=top_k,
            version=version,
            router_mode=router_mode,
        )

        # Embeddings — no pos_embed, RoPE handles position
        self.token_embed = nn.Embedding(vocab_size, dim)
        self.mask_embed = nn.Parameter(torch.randn(dim) * 0.02)

        # Output head (weight-tied)
        self.output_head = nn.Linear(dim, vocab_size, bias=False)
        self.output_head.weight = self.token_embed.weight

        self.aux_loss = 0.0

    def forward(self, prompt_ids: torch.Tensor,
                target_ids: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass with MoE routing.

        Args:
            mask: (B, G) bool — True = masked, False = given. None = all masked.
        """
        B, P = prompt_ids.shape
        G = target_ids.shape[1]
        device = prompt_ids.device

        # Embed
        prompt_x = self.token_embed(prompt_ids)

        if mask is None:
            output_x = self.mask_embed.unsqueeze(0).expand(B, G, -1)
        else:
            gt_embeds = self.token_embed(target_ids)
            mask_embeds = self.mask_embed.unsqueeze(0).expand(B, G, -1)
            output_x = torch.where(mask.unsqueeze(-1), mask_embeds, gt_embeds)

        x = torch.cat([prompt_x, output_x], dim=1)  # (B, P+G, D)

        # Run through MoE stacker
        x = self.stacker(x)

        # Final prediction at output positions only
        logits = self.output_head(x[:, P:])

        self.aux_loss = getattr(self.stacker, 'aux_loss', 0.0)
        return logits
