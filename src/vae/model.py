"""ByteChunkVAE — compresses fixed-size byte chunks into continuous latent vectors.

Architecture:
    Encoder: byte embedding -> bidirectional transformer -> mean-pool -> mu, log_var -> z
    Decoder: z -> learned positional queries + broadcast z -> transformer -> byte logits

Vocab (259):
    0 = PAD
    1 = BOS
    2 = EOS
    3..258 = byte values 0..255
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# Special token IDs
PAD = 0
BOS = 1
EOS = 2
BYTE_OFFSET = 3  # byte value b -> token id b + 3
VOCAB_SIZE = 259  # 256 bytes + PAD + BOS + EOS


@dataclass
class VAEConfig:
    chunk_size: int = 16       # K: number of byte positions per chunk (including BOS/EOS)
    d_model: int = 256         # encoder/decoder hidden dim
    d_latent: int = 64         # bottleneck latent dim
    n_heads: int = 4           # attention heads
    enc_layers: int = 4        # encoder transformer depth
    dec_layers: int = 4        # decoder transformer depth
    beta: float = 0.01         # KL weight (β-VAE)
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (bidirectional)."""

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

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class Encoder(nn.Module):
    """Encodes a byte chunk (B, K) -> mu (B, D_latent), log_var (B, D_latent)."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.chunk_size, cfg.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.enc_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        self.to_mu = nn.Linear(cfg.d_model, cfg.d_latent)
        self.to_logvar = nn.Linear(cfg.d_model, cfg.d_latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, K) byte token IDs. Returns mu, log_var each (B, D_latent)."""
        B, K = x.shape
        pad_mask = (x == PAD)  # True where padded

        pos = torch.arange(K, device=x.device)
        h = self.embed(x) + self.pos_embed(pos)

        for layer in self.layers:
            h = layer(h, mask=pad_mask)
        h = self.norm(h)

        # Mean-pool over non-pad positions
        valid = (~pad_mask).unsqueeze(-1).float()  # (B, K, 1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)  # (B, D)

        return self.to_mu(pooled), self.to_logvar(pooled)


class Decoder(nn.Module):
    """Decodes latent z (B, D_latent) -> byte logits (B, K, VOCAB_SIZE)."""

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.chunk_size = cfg.chunk_size
        self.z_proj = nn.Linear(cfg.d_latent, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.chunk_size, cfg.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.dec_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB_SIZE)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D_latent). Returns logits (B, K, VOCAB_SIZE)."""
        B = z.shape[0]
        K = self.chunk_size

        # Broadcast z to all positions + add positional embedding
        h = self.z_proj(z).unsqueeze(1).expand(B, K, -1)  # (B, K, D)
        pos = torch.arange(K, device=z.device)
        h = h + self.pos_embed(pos)

        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)

        return self.head(h)  # (B, K, VOCAB_SIZE)


class ByteChunkVAE(nn.Module):
    """Full VAE: byte chunk -> latent -> reconstructed byte chunk.

    Input chunks are (B, K) tensors of byte token IDs:
        [BOS, byte+3, byte+3, ..., EOS, PAD, PAD, ...]

    Training returns (loss, recon_loss, kl_loss, accuracy).
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, K) byte token IDs.

        Returns:
            loss: scalar, total loss (recon + beta * KL)
            recon_loss: scalar, per-token cross-entropy on non-PAD positions
            kl_loss: scalar, KL divergence
            accuracy: scalar, fraction of non-PAD tokens correctly reconstructed
        """
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        logits = self.decoder(z)  # (B, K, VOCAB_SIZE)

        # Reconstruction loss: CE on non-PAD positions
        mask = (x != PAD)  # (B, K)
        ce = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            x.reshape(-1),
            reduction='none',
        ).reshape(x.shape)
        recon_loss = (ce * mask).sum() / mask.sum().clamp(min=1)

        # KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).sum(dim=-1).mean()

        loss = recon_loss + self.cfg.beta * kl_loss

        # Accuracy on non-PAD
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            correct = ((preds == x) & mask).sum()
            accuracy = correct.float() / mask.sum().clamp(min=1)

        return loss, recon_loss, kl_loss, accuracy

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode byte chunks to latent vectors (deterministic, uses mu)."""
        mu, _ = self.encoder(x)
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to byte token IDs (argmax)."""
        logits = self.decoder(z)
        return logits.argmax(dim=-1)

    @staticmethod
    def bytes_to_chunks(raw_bytes: bytes, chunk_size: int) -> torch.Tensor:
        """Convert raw bytes to padded chunks of token IDs.

        Each chunk: [BOS, b0+3, b1+3, ..., EOS, PAD, PAD, ...]
        Content bytes per chunk = chunk_size - 2 (for BOS/EOS).

        Returns: (N_chunks, chunk_size) long tensor.
        """
        content_len = chunk_size - 2
        n_chunks = (len(raw_bytes) + content_len - 1) // content_len
        chunks = torch.full((n_chunks, chunk_size), PAD, dtype=torch.long)

        for i in range(n_chunks):
            start = i * content_len
            end = min(start + content_len, len(raw_bytes))
            chunk_bytes = raw_bytes[start:end]
            seq_len = len(chunk_bytes) + 2  # +BOS +EOS

            chunks[i, 0] = BOS
            for j, b in enumerate(chunk_bytes):
                chunks[i, j + 1] = b + BYTE_OFFSET
            chunks[i, seq_len - 1] = EOS

        return chunks

    @staticmethod
    def chunks_to_bytes(chunks: torch.Tensor) -> bytes:
        """Convert chunk token IDs back to raw bytes. Inverse of bytes_to_chunks."""
        result = bytearray()
        for chunk in chunks:
            for tok in chunk.tolist():
                if tok == BOS or tok == PAD:
                    continue
                if tok == EOS:
                    break
                result.append(tok - BYTE_OFFSET)
        return bytes(result)
