"""STLG — Straight-Through Latent Generator.

Causal transformer that operates on continuous latent vectors from a frozen
ByteChunkVAE. Given a sequence of byte chunks, encodes them to latents via
the frozen VAE encoder, predicts the next latent at each position, and
decodes predicted latents through the frozen VAE decoder.

Training loss: CE against target byte chunks (not MSE on latents).
No embedding lookup — continuous latent vectors go straight in.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from vae.model import ByteChunkVAE, VAEConfig, VOCAB_SIZE, PAD


@dataclass
class STLGConfig:
    d_latent: int = 128        # latent vector dim (must match VAE)
    d_model: int = 128         # transformer hidden dim (can == d_latent to skip projection)
    n_heads: int = 4
    n_layers: int = 4
    max_seq_len: int = 17      # max latent sequence length
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class CausalBlock(nn.Module):
    """Pre-norm causal transformer block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = RMSNorm(d_model)
        hidden = 4 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False, is_causal=True)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class STLG(nn.Module):
    """Straight-Through Latent Generator.

    Contains a frozen VAE (encoder + decoder) and a causal transformer.
    Forward takes raw byte chunk pieces (B, S, K), returns loss + accuracy.

    Flow:
        1. Frozen VAE encoder: chunks -> latents
        2. Causal transformer: latents[:-1] -> predicted latents for [1:]
        3. Frozen VAE decoder: predicted latents -> logits
        4. CE loss against target chunks[1:]
    """

    def __init__(self, cfg: STLGConfig, vae: ByteChunkVAE):
        super().__init__()
        self.cfg = cfg
        self.vae = vae  # frozen, no grad

        # Input projection (identity if d_latent == d_model)
        if cfg.d_latent != cfg.d_model:
            self.in_proj = nn.Linear(cfg.d_latent, cfg.d_model, bias=False)
        else:
            self.in_proj = nn.Identity()

        self.pos_embed = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.layers = nn.ModuleList([
            CausalBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)

        # Output projection back to d_latent
        if cfg.d_latent != cfg.d_model:
            self.out_proj = nn.Linear(cfg.d_model, cfg.d_latent, bias=False)
        else:
            self.out_proj = nn.Identity()

    def predict_latents(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, d_latent). Returns (B, T, d_latent) predicted next latents."""
        B, T, _ = z.shape

        h = self.in_proj(z)
        pos = torch.arange(T, device=z.device)
        h = h + self.pos_embed(pos)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        return self.out_proj(h)

    def forward(self, pieces: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pieces: (B, S, K) — S chunks per piece, K = chunk_size.

        Returns:
            loss: scalar CE on non-PAD positions
            accuracy: scalar fraction correct on non-PAD positions
        """
        B, S, K = pieces.shape

        # 1. Encode all chunks through frozen VAE encoder
        flat_chunks = pieces.reshape(B * S, K)
        with torch.no_grad():
            flat_mu, _ = self.vae.encoder(flat_chunks)
        latents = flat_mu.reshape(B, S, -1)  # (B, S, d_latent)

        # 2. Predict next latent from context
        pred_latents = self.predict_latents(latents[:, :-1])  # (B, S-1, d_latent)

        # 3. Decode predicted latents through frozen VAE decoder
        # Grad flows through pred_latents -> decoder -> logits -> loss -> back to STLG
        pred_flat = pred_latents.reshape(B * (S - 1), -1)
        logits = self.vae.decoder(pred_flat)  # (B*(S-1), K, VOCAB_SIZE)

        # 4. CE loss against target chunks
        target_chunks = pieces[:, 1:].reshape(B * (S - 1), K)
        mask = (target_chunks != PAD)
        ce = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            target_chunks.reshape(-1),
            reduction='none',
        ).reshape(target_chunks.shape)
        loss = (ce * mask).sum() / mask.sum().clamp(min=1)

        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = ((preds == target_chunks) & mask).sum()
            accuracy = correct.float() / mask.sum().clamp(min=1)

        return loss, accuracy
