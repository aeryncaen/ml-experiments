"""DeepMLM — Masked Language Model with prompt/output split.

Wraps any backbone (BidirectionalTransformer, CausalULB, StackedLM, etc.)
and adds mask embedding at masked output positions. Same approach as
LLaDAModel but with an explicit prompt/output interface.

Architecture:
    [prompt_tokens | mask_or_gt_tokens] → embed with mask → backbone → logits

Training: CE loss on masked output positions.
Generation: Single forward pass with all output positions masked.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepMLM(nn.Module):
    """MLM wrapper around any backbone with prompt/output split.

    The backbone must have:
    - token_embed: nn.Embedding
    - vocab_size, dim, max_seq_len attributes

    Works with the same backbones as LLaDAModel:
    - BidirectionalTransformer (has rope_freqs, blocks)
    - CausalULB (has norms, blocks)
    - StackedLM (has stacker)

    Args:
        backbone: The underlying model.
        vocab_size: Token vocabulary size.
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

        self.aux_loss = 0.0

    def _embed_with_mask(self, prompt_ids: torch.Tensor,
                         target_ids: torch.Tensor,
                         mask: torch.Tensor | None) -> torch.Tensor:
        """Embed prompt + output, replacing masked output positions with mask_embed.

        Args:
            prompt_ids: (B, P) prompt token indices.
            target_ids: (B, G) output token indices.
            mask: (B, G) bool — True = masked. None = all masked.

        Returns:
            (B, P+G, D) embeddings.
        """
        bb = self.backbone
        B, P = prompt_ids.shape
        G = target_ids.shape[1]

        # Embed everything through the backbone's embedding
        all_ids = torch.cat([prompt_ids, target_ids], dim=1)  # (B, P+G)
        x = bb.token_embed(all_ids)  # (B, P+G, D)

        # Replace masked output positions with mask_embed
        if mask is None:
            # All output positions masked
            x[:, P:] = self.mask_embed
        else:
            x[:, P:] = torch.where(mask.unsqueeze(-1), self.mask_embed, x[:, P:])

        return x

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone layers on pre-computed embeddings. Returns logits.

        Same dispatch logic as LLaDAModel._run_backbone_from_embeddings.
        """
        bb = self.backbone

        if hasattr(bb, 'stacker'):
            x = bb.stacker(x)
            return bb.head(x)
        elif hasattr(bb, 'rope_freqs'):
            for block in bb.blocks:
                x = block(x, bb.rope_freqs)
        elif hasattr(bb, 'norms'):
            for norm, block in zip(bb.norms, bb.blocks):
                x = x + block(norm(x))
        else:
            raise ValueError(f"Unknown backbone type: {type(bb)}")

        return bb.head(bb.final_norm(x))

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
        P = prompt_ids.shape[1]

        x = self._embed_with_mask(prompt_ids, target_ids, mask)
        logits = self._run_backbone(x)

        # Collect aux_loss from backbone
        if hasattr(self.backbone, 'stacker'):
            self.aux_loss = getattr(self.backbone.stacker, 'aux_loss', 0.0)
        else:
            self.aux_loss = 0.0

        # Return only output positions
        return logits[:, P:]
