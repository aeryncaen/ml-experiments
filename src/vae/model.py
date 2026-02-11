"""ByteChunkVAE — compresses fixed-size byte chunks into continuous latent vectors.

Architecture:
    Encoder preprocessing (before bottleneck):
        1. Byte embedding (B, K) -> (B, K, D)
        2. Acausal lerp on last quarter of dims (bidirectional neighbor blend)
        3. Full bidirectional attention with ALiBi (enc_layers deep)
        4. Mean-pool -> mu, log_var -> z

    Decoder:
        z -> broadcast + acausal lerp -> bidir attention with ALiBi -> byte logits

    ALiBi provides position information through attention bias (zero extra params,
    zero data-dependent state to compress through the bottleneck).

Vocab (259):
    0 = PAD
    1 = BOS
    2 = EOS
    3..258 = byte values 0..255
"""

from dataclasses import dataclass
import math

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
    beta: float = 0.01         # KL weight (beta-VAE)
    dropout: float = 0.0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


# ---------------------------------------------------------------------------
# ALiBi: linear attention bias (zero params, position via attention scores)
# ---------------------------------------------------------------------------

def build_alibi_bias(n_heads: int, max_len: int) -> torch.Tensor:
    """Build ALiBi bias matrix: (n_heads, max_len, max_len).

    Each head gets a different slope. Bias[i,j] = -slope * |i - j|.
    Bidirectional (symmetric) since this is not causal.
    """
    # Slopes: geometric sequence from 2^(-8/n_heads) to 2^(-8)
    # Following the ALiBi paper's head slope schedule
    ratio = 2 ** (-8.0 / n_heads)
    slopes = torch.tensor([ratio ** (i + 1) for i in range(n_heads)])  # (H,)

    pos = torch.arange(max_len)
    dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs().float()  # (L, L)
    bias = -dist.unsqueeze(0) * slopes.unsqueeze(-1).unsqueeze(-1)  # (H, L, L)
    return bias


# ---------------------------------------------------------------------------
# Embedding preprocessing: acausal lerp (no RoPE — ALiBi handles position)
# ---------------------------------------------------------------------------

class EmbeddingPreprocessor(nn.Module):
    """Acausal lerp on last quarter of embedding dims.

    Bidirectional neighbor blending so each position is informed by
    its neighbors before attention sees it. No RoPE — ALiBi provides
    position info through attention bias instead.

    d_model must be divisible by 4.
    """

    def __init__(self, d_model: int, init_bias: float = -2.0):
        super().__init__()
        assert d_model % 4 == 0, f"d_model must be divisible by 4, got {d_model}"
        self.quarter_dim = d_model // 4

        # Acausal lerp gates (last quarter of dims)
        self.gate_fwd = nn.Linear(d_model, self.quarter_dim, bias=True)
        self.gate_bwd = nn.Linear(d_model, self.quarter_dim, bias=True)
        nn.init.zeros_(self.gate_fwd.weight)
        nn.init.constant_(self.gate_fwd.bias, init_bias)
        nn.init.zeros_(self.gate_bwd.weight)
        nn.init.constant_(self.gate_bwd.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, D). Returns (B, K, D) with acausal lerp on last quarter."""
        B, K, D = x.shape
        qd = self.quarter_dim

        if K < 2:
            return x

        g_fwd = torch.sigmoid(self.gate_fwd(x))  # (B, K, qd)
        g_bwd = torch.sigmoid(self.gate_bwd(x))

        x_static = x[..., :-qd]
        x_cur = x[..., -qd:]
        x_prev = F.pad(x_cur[:, :-1], (0, 0, 1, 0))
        x_next = F.pad(x_cur[:, 1:],  (0, 0, 0, 1))
        x_mixed = (1 - g_fwd - g_bwd) * x_cur + g_fwd * x_prev + g_bwd * x_next

        return torch.cat([x_static, x_mixed], dim=-1)


# ---------------------------------------------------------------------------
# Transformer block with ALiBi (bidirectional, pre-norm)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm transformer block with ALiBi attention bias."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = RMSNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm2 = RMSNorm(d_model)
        hidden = 4 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        alibi_bias: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, K, D = x.shape
        H = self.n_heads
        hd = self.head_dim

        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, K, 3, H, hd)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (B, K, H, hd)

        # Attention scores with ALiBi bias
        q = q.transpose(1, 2)  # (B, H, K, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(hd)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, K, K)
        attn = attn + alibi_bias[:, :K, :K]  # (H, K, K) broadcast over B

        if pad_mask is not None:
            # pad_mask: (B, K) True where padded — expand to (B, 1, 1, K)
            attn = attn.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        if self.dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.dropout)

        out = torch.matmul(attn, v)  # (B, H, K, hd)
        out = out.transpose(1, 2).reshape(B, K, D)
        out = self.o_proj(out)

        x = x + out
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Encodes a byte chunk (B, K) -> mu (B, D_latent), log_var (B, D_latent).

    Pipeline: embed -> acausal lerp -> full bidirectional attention (ALiBi) ->
              mean-pool -> bottleneck.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.preprocess = EmbeddingPreprocessor(cfg.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.enc_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        self.to_mu = nn.Linear(cfg.d_model, cfg.d_latent)
        self.to_logvar = nn.Linear(cfg.d_model, cfg.d_latent)

        # ALiBi bias (registered as buffer, not a parameter)
        alibi = build_alibi_bias(cfg.n_heads, cfg.chunk_size)
        self.register_buffer('alibi_bias', alibi, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, K) byte token IDs. Returns mu, log_var each (B, D_latent)."""
        B, K = x.shape
        pad_mask = (x == PAD)

        h = self.embed(x)
        h = self.preprocess(h)

        for layer in self.layers:
            h = layer(h, self.alibi_bias, pad_mask)
        h = self.norm(h)

        # Mean-pool over non-pad positions
        valid = (~pad_mask).unsqueeze(-1).float()
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)

        return self.to_mu(pooled), self.to_logvar(pooled)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """Decodes latent z (B, D_latent) -> byte logits (B, K, VOCAB_SIZE).

    Same preprocessing as encoder: z broadcast to K positions gets
    acausal lerp + ALiBi attention.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.chunk_size = cfg.chunk_size
        self.z_proj = nn.Linear(cfg.d_latent, cfg.d_model)
        self.preprocess = EmbeddingPreprocessor(cfg.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.dropout)
            for _ in range(cfg.dec_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB_SIZE)

        alibi = build_alibi_bias(cfg.n_heads, cfg.chunk_size)
        self.register_buffer('alibi_bias', alibi, persistent=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D_latent). Returns logits (B, K, VOCAB_SIZE)."""
        B = z.shape[0]
        K = self.chunk_size

        h = self.z_proj(z).unsqueeze(1).expand(B, K, -1)  # (B, K, D)
        h = self.preprocess(h)

        for layer in self.layers:
            h = layer(h, self.alibi_bias)
        h = self.norm(h)

        return self.head(h)


# ---------------------------------------------------------------------------
# Full VAE
# ---------------------------------------------------------------------------

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

    def per_sample_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-sample reconstruction loss (no grad, for hard mining).

        Args:
            x: (B, K) byte token IDs.

        Returns:
            (B,) per-sample mean CE over non-PAD positions.
        """
        with torch.no_grad():
            mu, _ = self.encoder(x)
            logits = self.decoder(mu)  # deterministic, no reparameterize
            mask = (x != PAD)
            ce = F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE),
                x.reshape(-1),
                reduction='none',
            ).reshape(x.shape)
            # Per-sample mean CE
            per_sample = (ce * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
        return per_sample

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
