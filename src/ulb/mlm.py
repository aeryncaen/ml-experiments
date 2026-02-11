"""BERT-style Masked Language Model.

Wraps any backbone (BidirectionalTransformer, CausalULB, StackedLM, etc.)
and adds a learned mask embedding. Masks random positions plus a single
end-of-sequence token for next-token prediction.

Training: CE loss on masked positions only.
Generation: mask a region, forward pass, argmax at masked positions.
"""

import torch
import torch.nn as nn


class BertMLM(nn.Module):
    """BERT-style MLM wrapper around any backbone.

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
        mask_prob: Probability of masking each token (default 0.40).
    """

    def __init__(self, backbone: nn.Module, vocab_size: int, dim: int,
                 mask_prob: float = 0.40):
        super().__init__()
        self.backbone = backbone
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = backbone.max_seq_len
        self.mask_prob = mask_prob

        # Learned mask token embedding
        self.mask_embed = nn.Parameter(torch.randn(dim) * 0.02)

        # MLM prediction head: dense -> SiLU -> RMSNorm (ModernBERT-style)
        self.pred_dense = nn.Linear(dim, dim, bias=False)
        self.pred_act = nn.SiLU()
        self.pred_norm = nn.RMSNorm(dim)

        # Init head with trunc_normal like other input projections
        nn.init.trunc_normal_(self.pred_dense.weight, std=0.02, a=-0.04, b=0.04)

        self.aux_loss = 0.0

    def _embed_with_mask(self, token_ids: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        """Embed tokens with BERT-style corruption at masked positions.

        Of the masked positions:
        - 80% get the learned mask embedding
        - 10% get a random token's embedding
        - 10% keep their original embedding

        This prevents the model from learning that only [MASK] positions
        need prediction — it must build good representations everywhere.

        Args:
            token_ids: (B, T) token indices.
            mask: (B, T) bool — True = masked.

        Returns:
            (B, T, D) embeddings.
        """
        embed = self.backbone.token_embed
        x = embed(token_ids)  # (B, T, D)

        if self.training:
            # 80/10/10 split within masked positions
            rand = torch.rand_like(mask, dtype=torch.float32)
            # 80%: mask embedding
            use_mask_embed = mask & (rand < 0.8)
            # 10%: random token embedding
            use_random = mask & (rand >= 0.8) & (rand < 0.9)
            # 10%: keep original (no change needed)

            x = torch.where(use_mask_embed.unsqueeze(-1), self.mask_embed, x)

            if use_random.any():
                random_ids = torch.randint(0, self.vocab_size, token_ids.shape,
                                           device=token_ids.device)
                random_embeds = embed(random_ids)
                x = torch.where(use_random.unsqueeze(-1), random_embeds, x)
        else:
            # At eval/generation, just use mask embedding for all masked positions
            x = torch.where(mask.unsqueeze(-1), self.mask_embed, x)

        return x

    def _run_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run backbone layers on pre-computed embeddings. Returns hidden states.

        Same dispatch logic as LLaDAModel._run_backbone_from_embeddings.
        """
        bb = self.backbone

        if hasattr(bb, 'stacker'):
            x = bb.stacker(x)
        elif hasattr(bb, 'rope_freqs'):
            for block in bb.blocks:
                x = block(x, bb.rope_freqs)
            x = bb.final_norm(x)
        elif hasattr(bb, 'norms'):
            for norm, block in zip(bb.norms, bb.blocks):
                x = x + block(norm(x))
            x = bb.final_norm(x)
        else:
            raise ValueError(f"Unknown backbone type: {type(bb)}")

        return x

    def forward(self, token_ids: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: (B, T) token indices (ground truth).
            mask: (B, T) bool — True = masked (predict these).

        Returns:
            (B, T, vocab_size) MLM logits — loss on masked positions only.
        """
        x = self._embed_with_mask(token_ids, mask)
        h = self._run_backbone(x)

        # MLM head: dense -> SiLU -> RMSNorm -> decoder (weight-tied)
        mlm_logits = self.backbone.head(
            self.pred_norm(self.pred_act(self.pred_dense(h)))
        )

        # Collect aux_loss from backbone
        if hasattr(self.backbone, 'stacker'):
            self.aux_loss = getattr(self.backbone.stacker, 'aux_loss', 0.0)
        else:
            self.aux_loss = 0.0

        return mlm_logits
