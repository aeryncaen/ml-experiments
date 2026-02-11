"""ByteChunkVAE — compresses fixed-size byte chunks into continuous latent vectors.

Architecture:
    Encoder preprocessing (before bottleneck):
        1. Byte embedding (B, K) -> (B, K, D)
        2. Fixed RoPE on first half of dims
        3. Data-dependent RoPE (cumsum angles) on second half of dims
        4. Acausal lerp on last quarter of dims (bidirectional neighbor blend)
        5. Full bidirectional attention (enc_layers deep)
        6. Mean-pool -> mu, log_var -> z

    Decoder:
        z -> broadcast + RoPE + DD-RoPE + lerp -> bidir attention -> byte logits

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
    beta: float = 0.01         # KL weight (beta-VAE)
    dropout: float = 0.0
    use_fused: bool = False    # use ULB-style fused blocks instead of transformer blocks


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


# ---------------------------------------------------------------------------
# Embedding preprocessing: fixed RoPE + DD-RoPE + acausal lerp
# ---------------------------------------------------------------------------

def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate paired dims: x[..., :d] and x[..., d:] by (cos, sin)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)


class EmbeddingPreprocessor(nn.Module):
    """Enriches embeddings with positional and temporal structure.

    Applied to (B, K, D) embeddings before the transformer layers.

    Dim layout: [fixed_x1 (D/4) | fixed_x2 (D/4) | dd_x1 (D/4) | dd_x2 (D/4)]

    1. Fixed RoPE on first half (D/2) — standard positional frequencies
    2. Data-dependent RoPE on second half (D/2) — cumsum of learned angle deltas
    3. Acausal lerp on last quarter (D/4) — bidirectional neighbor blending

    d_model must be divisible by 4.
    """

    def __init__(self, d_model: int, rope_base: float = 10000.0, init_bias: float = -2.0):
        super().__init__()
        assert d_model % 4 == 0, f"d_model must be divisible by 4, got {d_model}"
        self.d_model = d_model
        self.fixed_pairs = d_model // 4    # rotation pairs for fixed RoPE
        self.dd_pairs = d_model // 4       # rotation pairs for DD-RoPE
        self.quarter_dim = d_model // 4    # lerp channels

        # Fixed RoPE inverse frequencies
        inv_freq = 1.0 / (rope_base ** (
            torch.arange(0, self.fixed_pairs * 2, 2).float() / (d_model // 2)
        ))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # DD-RoPE: project embedding to per-position angle deltas, then cumsum
        self.dd_proj = nn.Linear(d_model, self.dd_pairs, bias=True)
        nn.init.zeros_(self.dd_proj.weight)
        nn.init.zeros_(self.dd_proj.bias)

        # Acausal lerp gates (last quarter of dims)
        self.gate_fwd = nn.Linear(d_model, self.quarter_dim, bias=True)
        self.gate_bwd = nn.Linear(d_model, self.quarter_dim, bias=True)
        nn.init.zeros_(self.gate_fwd.weight)
        nn.init.constant_(self.gate_fwd.bias, init_bias)
        nn.init.zeros_(self.gate_bwd.weight)
        nn.init.constant_(self.gate_bwd.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, D). Returns (B, K, D) with RoPE + DD-RoPE + acausal lerp."""
        B, K, D = x.shape
        fp = self.fixed_pairs
        dp = self.dd_pairs
        qd = self.quarter_dim

        # --- 1. Fixed RoPE on first half ---
        x_fixed = torch.cat([x[..., :fp], x[..., fp:2*fp]], dim=-1)  # (B, K, 2*fp)
        t = torch.arange(K, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)       # (K, fp)
        cos_f = freqs.cos()[None, :, :]              # (1, K, fp)
        sin_f = freqs.sin()[None, :, :]
        x_fixed = _apply_rotary(x_fixed, cos_f, sin_f)

        # --- 2. DD-RoPE on second half ---
        x_dd = torch.cat([x[..., 2*fp:2*fp+dp], x[..., 2*fp+dp:]], dim=-1)  # (B, K, 2*dp)
        dd_angles = self.dd_proj(x).cumsum(dim=1)   # (B, K, dp)
        x_dd = _apply_rotary(x_dd, dd_angles.cos(), dd_angles.sin())

        # Reassemble
        x = torch.cat([x_fixed[..., :fp], x_fixed[..., fp:],
                        x_dd[..., :dp], x_dd[..., dp:]], dim=-1)

        # --- 3. Acausal lerp on last quarter ---
        if K >= 2:
            g_fwd = torch.sigmoid(self.gate_fwd(x))  # (B, K, qd)
            g_bwd = torch.sigmoid(self.gate_bwd(x))

            x_static = x[..., :-qd]
            x_cur = x[..., -qd:]
            x_prev = F.pad(x_cur[:, :-1], (0, 0, 1, 0))
            x_next = F.pad(x_cur[:, 1:],  (0, 0, 0, 1))
            x_mixed = (1 - g_fwd - g_bwd) * x_cur + g_fwd * x_prev + g_bwd * x_next
            x = torch.cat([x_static, x_mixed], dim=-1)

        return x


# ---------------------------------------------------------------------------
# Transformer block (bidirectional, pre-norm)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# ULB-style fused attention+gate block (bidirectional)
# ---------------------------------------------------------------------------

class FusedBlock(nn.Module):
    """ULB-style fused attention+gate block.

    Replaces separate MHA + MLP with a single fused path:
        h_up = swish(up_proj(norm(x)))
        q, k, v = qkv_proj(h_up)
        q, k = qk_norm(q, k)
        y = SDPA(q, k, v)               # bidirectional
        y = o_proj(y)
        y = attn_norm(y) * h_up         # skip-MULTIPLY
        y = down_proj(swish(y))
        return x + y
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm = RMSNorm(d_model)

        # Gate path
        self.up_proj = nn.Linear(d_model, d_model, bias=False)
        self.down_proj = nn.Linear(d_model, d_model, bias=False)

        # QKV (all at d_model, no expansion)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # QK norm (Mamba-3 style)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        # Post-attention norm (before skip-multiply)
        self.attn_norm = RMSNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, K, D = x.shape
        H = self.n_heads
        hd = self.head_dim

        h = self.norm(x)
        h_up = F.silu(self.up_proj(h))

        q = self.q_proj(h_up).view(B, K, H, hd)
        k = self.k_proj(h_up).view(B, K, H, hd)
        v = self.v_proj(h_up).view(B, K, H, hd)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)  # (B, H, K, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1).unsqueeze(2).expand(B, H, K, K)
            attn_mask = torch.where(attn_mask, float('-inf'), 0.0)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, K, D)

        y = self.o_proj(y)
        y = self.attn_norm(y) * h_up
        y = self.down_proj(F.silu(y))

        return x + y


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Encodes a byte chunk (B, K) -> mu (B, D_latent), log_var (B, D_latent).

    Pipeline: embed -> preprocess (RoPE + DD-RoPE + acausal lerp) ->
              full bidirectional attention -> mean-pool -> bottleneck.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.preprocess = EmbeddingPreprocessor(cfg.d_model)
        Block = FusedBlock if cfg.use_fused else TransformerBlock
        block_args = (cfg.d_model, cfg.n_heads) if cfg.use_fused else (cfg.d_model, cfg.n_heads, cfg.dropout)
        self.layers = nn.ModuleList([
            Block(*block_args)
            for _ in range(cfg.enc_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        self.to_mu = nn.Linear(cfg.d_model, cfg.d_latent)
        self.to_logvar = nn.Linear(cfg.d_model, cfg.d_latent)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B, K) byte token IDs. Returns mu, log_var each (B, D_latent)."""
        B, K = x.shape
        pad_mask = (x == PAD)

        h = self.embed(x)
        h = self.preprocess(h)

        for layer in self.layers:
            h = layer(h, mask=pad_mask)
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
    RoPE + DD-RoPE + acausal lerp before transformer layers.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.chunk_size = cfg.chunk_size
        self.z_proj = nn.Linear(cfg.d_latent, cfg.d_model)
        self.preprocess = EmbeddingPreprocessor(cfg.d_model)
        Block = FusedBlock if cfg.use_fused else TransformerBlock
        block_args = (cfg.d_model, cfg.n_heads) if cfg.use_fused else (cfg.d_model, cfg.n_heads, cfg.dropout)
        self.layers = nn.ModuleList([
            Block(*block_args)
            for _ in range(cfg.dec_layers)
        ])
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, VOCAB_SIZE)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, D_latent). Returns logits (B, K, VOCAB_SIZE)."""
        B = z.shape[0]
        K = self.chunk_size

        h = self.z_proj(z).unsqueeze(1).expand(B, K, -1)  # (B, K, D)
        h = self.preprocess(h)

        for layer in self.layers:
            h = layer(h)
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
        # Tie decoder head to encoder embedding
        self.decoder.head.weight = self.encoder.embed.weight

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
