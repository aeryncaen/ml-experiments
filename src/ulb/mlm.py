"""BERT-style Masked Language Model with progressive unmasking.

Wraps any backbone (BidirectionalTransformer, CausalULB, StackedLM, etc.)
and adds a learned mask embedding. Masks random positions across the ENTIRE
input sequence — no prompt/output split.

The last 4 layers each unmask the 25% of originally-masked positions they
are most confident about, replacing mask embeddings with softmax-weighted
token embeddings. By the final layer, all masked positions are unmasked.

Training: mask positions, progressive unmasking in last 4 layers, CE loss on masked.
Generation: same progressive unmasking, argmax at final logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BertMLM(nn.Module):
    """BERT-style MLM wrapper with progressive unmasking in the last 4 layers.

    The backbone must have:
    - token_embed: nn.Embedding
    - vocab_size, dim, max_seq_len attributes

    Works with the same backbones as LLaDAModel:
    - BidirectionalTransformer (has rope_freqs, blocks)
    - CausalULB (has norms, blocks)
    - StackedLM (has stacker) — no progressive unmasking, falls back to plain forward

    Args:
        backbone: The underlying model.
        vocab_size: Token vocabulary size.
        dim: Model dimension.
        mask_prob: Probability of masking each token (default 0.40).
        n_unmask_layers: Number of final layers that do progressive unmasking (default 4).
    """

    def __init__(self, backbone: nn.Module, vocab_size: int, dim: int,
                 mask_prob: float = 0.40, n_unmask_layers: int = 4):
        super().__init__()
        self.backbone = backbone
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = backbone.max_seq_len
        self.mask_prob = mask_prob
        self.n_unmask_layers = n_unmask_layers

        # Learned mask token embedding
        self.mask_embed = nn.Parameter(torch.randn(dim) * 0.02)

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

    def _unmask_top_k(self, x: torch.Tensor, still_masked: torch.Tensor,
                      n_to_unmask: int) -> torch.Tensor:
        """Unmask the n_to_unmask most confident positions per sample.

        Projects x to logits, computes softmax, picks top-k by max probability,
        and replaces those positions' embeddings with softmax-weighted token embeddings.

        Args:
            x: (B, T, D) hidden states after a layer.
            still_masked: (B, T) bool — True = still masked.
            n_to_unmask: Number of positions to unmask per sample.

        Returns:
            x: (B, T, D) with unmasked positions replaced.
            still_masked: (B, T) updated mask.
        """
        if n_to_unmask <= 0:
            return x, still_masked

        bb = self.backbone
        B, T, D = x.shape

        # Project to logits at current state
        logits = bb.head(bb.final_norm(x))  # (B, T, V)
        probs = F.softmax(logits, dim=-1)  # (B, T, V)

        # Confidence = max probability at each position
        confidence = probs.max(dim=-1).values  # (B, T)

        # Only consider still-masked positions; set unmasked confidence to -inf
        confidence = confidence.masked_fill(~still_masked, float('-inf'))

        # Pick top n_to_unmask per sample
        _, top_idx = confidence.topk(min(n_to_unmask, T), dim=-1)  # (B, k)

        # Softmax-weighted embedding: probs @ token_embed.weight
        # probs: (B, T, V), embed: (V, D) -> weighted: (B, T, D)
        embed_weight = bb.token_embed.weight  # (V, D)
        weighted_embeds = probs @ embed_weight  # (B, T, D)

        # Build unmask indicator for these positions
        unmask_this = torch.zeros_like(still_masked)  # (B, T)
        unmask_this.scatter_(1, top_idx, True)
        # Only unmask positions that were actually still masked
        unmask_this = unmask_this & still_masked

        # Replace embeddings at unmasked positions
        x = torch.where(unmask_this.unsqueeze(-1), weighted_embeds, x)

        # Update mask
        still_masked = still_masked & ~unmask_this

        return x, still_masked

    def _run_backbone(self, x: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
        """Run backbone layers with progressive unmasking in the last n_unmask_layers.

        For backbones with explicit blocks (BidirectionalTransformer, CausalULB):
        - Run the first (n_layers - n_unmask_layers) layers normally
        - For each of the last n_unmask_layers layers:
          1. Run the layer
          2. Unmask top 25% of originally-masked positions (by confidence)
          3. Replace their embeddings with softmax-weighted token embeddings
          4. Continue to next layer

        For StackedLM: falls back to plain forward (no progressive unmasking).

        Returns: (B, T, vocab_size) logits.
        """
        bb = self.backbone

        if hasattr(bb, 'stacker'):
            # StackedLM — opaque stacker, no layer-level access
            x = bb.stacker(x)
            return bb.head(x)

        # Get blocks and how they're called
        if hasattr(bb, 'rope_freqs'):
            # BidirectionalTransformer
            blocks = list(bb.blocks)
            def run_block(block, x):
                return block(x, bb.rope_freqs)
        elif hasattr(bb, 'norms'):
            # CausalULB
            blocks = list(zip(bb.norms, bb.blocks))
            def run_block(norm_block, x):
                norm, block = norm_block
                return x + block(norm(x))
        else:
            raise ValueError(f"Unknown backbone type: {type(bb)}")

        n_layers = len(blocks)
        n_unmask = min(self.n_unmask_layers, n_layers)
        n_normal = n_layers - n_unmask

        # Count how many positions to unmask per unmasking layer
        # Each layer gets 25% of the original masked count
        n_originally_masked = mask.sum(dim=-1)  # (B,)
        # Use the max across the batch for simplicity (positions per sample may vary)
        # Actually, do it per-sample inside _unmask_top_k via topk
        n_per_layer = (n_originally_masked.float() / n_unmask).ceil().long()  # (B,)
        # For topk we need a single k — use max across batch
        k_per_layer = n_per_layer.max().item()

        # Track which positions are still masked
        still_masked = mask.clone()

        # Run normal layers
        for i in range(n_normal):
            x = run_block(blocks[i], x)

        # Run unmasking layers — each unmasks 25% then the next layer sees the result
        for i in range(n_unmask):
            x = run_block(blocks[n_normal + i], x)
            x, still_masked = self._unmask_top_k(x, still_masked, k_per_layer)

        return bb.head(bb.final_norm(x))

    def forward(self, token_ids: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            token_ids: (B, T) token indices (ground truth).
            mask: (B, T) bool — True = masked (predict these).

        Returns:
            logits: (B, T, vocab_size) predictions at ALL positions.
                    (Loss should only be computed on masked positions.)
        """
        x = self._embed_with_mask(token_ids, mask)
        logits = self._run_backbone(x, mask)

        # Collect aux_loss from backbone
        if hasattr(self.backbone, 'stacker'):
            self.aux_loss = getattr(self.backbone.stacker, 'aux_loss', 0.0)
        else:
            self.aux_loss = 0.0

        return logits
