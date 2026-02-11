"""ByteChunkVAE — compresses fixed-size byte chunks into continuous latent vectors.

Architecture:
    Embedding dim layout (each quarter = D/4):
        Q1: passthrough (raw embedding, untouched)
        Q2: fixed RoPE (standard positional frequencies)
        Q3: DD-RoPE (data-dependent cumsum angles)
        Q4: acausal K-lerp (bidirectional neighbor blend)

    Encoder: embed -> preprocess (4-quarter layout) -> bidir attention -> mean-pool -> mu/logvar -> z
    Decoder: z -> broadcast -> preprocess -> bidir attention -> byte logits

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
# Embedding preprocessing: passthrough | RoPE | DD-RoPE | acausal lerp
# ---------------------------------------------------------------------------

def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate paired dims: x[..., :d] and x[..., d:] by (cos, sin)."""
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)


class EmbeddingPreprocessor(nn.Module):
    """Four-quarter preprocessing on (B, K, D) embeddings.

    Dim layout:
        Q1 [0 : D/4]       — passthrough (untouched)
        Q2 [D/4 : D/2]     — fixed RoPE (standard positional frequencies)
        Q3 [D/2 : 3D/4]    — DD-RoPE (data-dependent cumsum angle rotations)
        Q4 [3D/4 : D]      — acausal K-lerp (bidirectional neighbor blend)

    d_model must be divisible by 4.
    """

    def __init__(self, d_model: int, rope_base: float = 10000.0, init_bias: float = -2.0):
        super().__init__()
        assert d_model % 4 == 0, f"d_model must be divisible by 4, got {d_model}"
        self.d_model = d_model
        self.qd = d_model // 4  # quarter dim

        # Fixed RoPE: qd dims = qd/2 rotation pairs
        self.rope_pairs = self.qd // 2
        inv_freq = 1.0 / (rope_base ** (
            torch.arange(0, self.qd, 2).float() / self.qd
        ))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # DD-RoPE: project from full embedding to angle deltas, then cumsum
        self.dd_pairs = self.qd // 2
        self.dd_proj = nn.Linear(d_model, self.dd_pairs, bias=True)
        nn.init.zeros_(self.dd_proj.weight)
        nn.init.zeros_(self.dd_proj.bias)

        # Acausal lerp gates (on Q4)
        self.gate_fwd = nn.Linear(d_model, self.qd, bias=True)
        self.gate_bwd = nn.Linear(d_model, self.qd, bias=True)
        nn.init.zeros_(self.gate_fwd.weight)
        nn.init.constant_(self.gate_fwd.bias, init_bias)
        nn.init.zeros_(self.gate_bwd.weight)
        nn.init.constant_(self.gate_bwd.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, K, D). Returns (B, K, D) preprocessed."""
        B, K, D = x.shape
        qd = self.qd

        # Split into quarters
        q1 = x[..., :qd]              # passthrough
        q2 = x[..., qd:2*qd]         # fixed RoPE
        q3 = x[..., 2*qd:3*qd]       # DD-RoPE
        q4 = x[..., 3*qd:]           # acausal lerp

        # --- Q2: fixed RoPE ---
        rp = self.rope_pairs
        q2_pairs = torch.cat([q2[..., :rp], q2[..., rp:]], dim=-1)  # (B, K, 2*rp)
        t = torch.arange(K, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)   # (K, rp)
        q2_pairs = _apply_rotary(q2_pairs, freqs.cos()[None], freqs.sin()[None])
        q2 = torch.cat([q2_pairs[..., :rp], q2_pairs[..., rp:]], dim=-1)

        # --- Q3: DD-RoPE ---
        dp = self.dd_pairs
        q3_pairs = torch.cat([q3[..., :dp], q3[..., dp:]], dim=-1)  # (B, K, 2*dp)
        dd_angles = self.dd_proj(x).cumsum(dim=1)  # (B, K, dp)
        q3_pairs = _apply_rotary(q3_pairs, dd_angles.cos(), dd_angles.sin())
        q3 = torch.cat([q3_pairs[..., :dp], q3_pairs[..., dp:]], dim=-1)

        # --- Q4: acausal lerp ---
        if K >= 2:
            g_fwd = torch.sigmoid(self.gate_fwd(x))  # (B, K, qd)
            g_bwd = torch.sigmoid(self.gate_bwd(x))
            q4_prev = F.pad(q4[:, :-1], (0, 0, 1, 0))
            q4_next = F.pad(q4[:, 1:],  (0, 0, 0, 1))
            q4 = (1 - g_fwd - g_bwd) * q4 + g_fwd * q4_prev + g_bwd * q4_next

        return torch.cat([q1, q2, q3, q4], dim=-1)


# ---------------------------------------------------------------------------
# Fused attention block (ULB-style, with per-layer HybridRoPE + K-lerp)
# ---------------------------------------------------------------------------

class FusedBlock(nn.Module):
    """ULB-style fused attention+gate block for the VAE.

    Each layer has its own HybridRoPE (fixed + DD-RoPE) and acausal K-lerp,
    so every attention computation is position-aware and content-phase-aware.
    Bidirectional (is_causal=False).

    Flow:
        h_up = swish(up_proj(norm(x)))
        q, k, v = qkv_proj(h_up)       # d -> inner (1.75x expansion)
        q, k = qk_norm(q, k)
        k = acausal_k_lerp(k, x)       # temporal neighbor blending on last 1/4 head_dim
        dd_angles = rope.compute_dd_angles(x)
        q = rope(q, dd_angles)          # hybrid: fixed RoPE + DD-RoPE
        k = rope(k, dd_angles)
        y = SDPA(q, k, v)               # bidirectional
        y = o_proj(y)                    # inner -> d
        y = attn_norm(y) * h_up         # skip-MULTIPLY
        y = down_proj(swish(y))
        return x + y                     # residual
    """

    INNER_RATIO = 1.75

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        from ulb.rope import HybridRoPE
        from ulb.lerp import KAcausalLerp

        # Snap inner_dim to nearest multiple of n_heads*4 (head_dim divisible by 4)
        snap = n_heads * 4
        inner_dim = round(d_model * self.INNER_RATIO / snap) * snap
        assert inner_dim > 0

        self.n_heads = n_heads
        self.inner_dim = inner_dim
        self.head_dim = inner_dim // n_heads

        self.norm = RMSNorm(d_model)

        # Gate path (thin, at d_model)
        self.up_proj = nn.Linear(d_model, d_model, bias=False)
        self.down_proj = nn.Linear(d_model, d_model, bias=False)

        # QKV: d -> inner (1.75x expansion)
        self.q_proj = nn.Linear(d_model, inner_dim, bias=False)
        self.k_proj = nn.Linear(d_model, inner_dim, bias=True)
        self.v_proj = nn.Linear(d_model, inner_dim, bias=True)
        self.o_proj = nn.Linear(inner_dim, d_model, bias=False)

        # QK norm (Mamba-3 style, at head_dim)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)

        # Post-attention norm (before skip-multiply, at d_model)
        self.attn_norm = RMSNorm(d_model)

        # Per-layer HybridRoPE (fixed + DD-RoPE on QK)
        self.rope = HybridRoPE(d_model, n_heads, self.head_dim)

        # Per-layer acausal K-lerp (last 1/4 of head_dim)
        quarter_dim = self.head_dim // 4
        self.k_lerp = KAcausalLerp(d_model, n_heads, quarter_dim, init_bias=-2.0)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, K, D = x.shape
        H = self.n_heads
        hd = self.head_dim

        h = self.norm(x)

        # Gate
        h_up = F.silu(self.up_proj(h))

        # QKV from gated representation
        q = self.q_proj(h_up).view(B, K, H, hd)
        k = self.k_proj(h_up).view(B, K, H, hd)
        v = self.v_proj(h_up).view(B, K, H, hd)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # K temporal mixing (acausal lerp on last 1/4 head_dim)
        k = self.k_lerp(k, x)

        # Hybrid RoPE (fixed + data-dependent)
        dd_angles = self.rope.compute_dd_angles(x)
        q = self.rope(q, dd_angles)
        k = self.rope(k, dd_angles)

        # SDPA (bidirectional)
        q = q.transpose(1, 2)  # (B, H, K, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build attention mask from pad_mask if needed
        attn_mask = None
        if pad_mask is not None:
            attn_mask = pad_mask.unsqueeze(1).unsqueeze(2).expand(B, H, K, K)
            attn_mask = torch.where(attn_mask, float('-inf'), 0.0)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, K, self.inner_dim)

        y = self.o_proj(y)

        # Skip-multiply (not skip-add)
        y = self.attn_norm(y) * h_up

        # Down projection
        y = self.down_proj(F.silu(y))

        return x + y


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """Encodes a byte chunk (B, K) -> mu (B, D_latent), log_var (B, D_latent).

    Pipeline: embed -> preprocess (passthrough|RoPE|DD-RoPE|lerp) ->
              fused ULB-style blocks -> mean-pool -> bottleneck.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, cfg.d_model)
        self.preprocess = EmbeddingPreprocessor(cfg.d_model)
        self.layers = nn.ModuleList([
            FusedBlock(cfg.d_model, cfg.n_heads)
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
            h = layer(h, pad_mask=pad_mask)
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
    passthrough|RoPE|DD-RoPE|lerp before fused ULB-style blocks.
    """

    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.chunk_size = cfg.chunk_size
        self.z_proj = nn.Linear(cfg.d_latent, cfg.d_model)
        self.preprocess = EmbeddingPreprocessor(cfg.d_model)
        self.layers = nn.ModuleList([
            FusedBlock(cfg.d_model, cfg.n_heads)
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
