"""
YAMIT: Yet Another Minorly Improved Transformer

Architecture components:
  - MLA dense attention with compressed KV latent cache
  - Composite-PIT embedding and LM head (byte-structured, pseudo-inverse tying)
  - SwiGLU MLP, RMSNorm, RoPE
  - Support for arbitrary position_ids (ReFusion slot shuffling)

Phase 1: BF16 correctness (no FP4 yet).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class YAMITConfig:
    """YAMIT model configuration."""

    # --- vocabulary ---
    vocab_size: int = 151_808          # Qwen3 base 151,669 + 2 special + pad to 256-multiple
    num_byte_symbols: int = 257        # 0..255 byte values + 256 pad symbol
    max_bytes_per_token: int = 16      # composite slot count

    # --- transformer core ---
    d_model: int = 768
    n_layers: int = 16
    n_heads: int = 12
    mlp_hidden: int = 2048
    max_seq_len: int = 4096

    # --- MLA dimensions ---
    q_compress_dim: int = 192
    kv_compress_dim: int = 96
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64

    # --- composite-PIT ---
    shared_per_slot: int = 36
    token_per_slot: int = 12
    pit_eps: float = 1e-6

    # --- attention ---
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-5
    attn_dropout: float = 0.0

    # --- special tokens ---
    mask_token_id: int = 151_670
    eos_token_id: int = 151_643        # Qwen3 EOS
    pad_token_id: int = 151_643        # often same as EOS for Qwen3

    # --- init ---
    init_std: float = 0.02

    # --- derived (read-only) ---
    @property
    def dims_per_slot(self) -> int:
        return self.d_model // self.max_bytes_per_token

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    def __post_init__(self):
        if self.d_model % self.max_bytes_per_token != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by max_bytes_per_token "
                f"({self.max_bytes_per_token})"
            )
        if self.dims_per_slot != self.shared_per_slot + self.token_per_slot:
            raise ValueError(
                "dims_per_slot must equal shared_per_slot + token_per_slot: "
                f"{self.dims_per_slot} != {self.shared_per_slot} + {self.token_per_slot}"
            )
        if self.qk_rope_head_dim % 2 != 0:
            raise ValueError("qk_rope_head_dim must be even for RoPE")


# ── Model-S (~130M) ──────────────────────────────────────────────────────

MODEL_S = YAMITConfig(
    d_model=768,
    n_layers=16,
    n_heads=12,
    q_compress_dim=192,
    kv_compress_dim=96,
    qk_nope_head_dim=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
    mlp_hidden=2048,
    shared_per_slot=36,
    token_per_slot=12,
)

# ── Model-P (~347M) ──────────────────────────────────────────────────────

MODEL_P = YAMITConfig(
    d_model=1024,
    n_layers=28,
    n_heads=16,
    q_compress_dim=256,
    kv_compress_dim=128,
    qk_nope_head_dim=64,
    qk_rope_head_dim=32,
    v_head_dim=64,
    mlp_hidden=2816,
    shared_per_slot=48,
    token_per_slot=16,
)


@dataclass
class MLALayerCache:
    """Per-layer MLA cache entries.

    kv_latent: compressed KV latent cache [B, S, kv_compress_dim]
    k_pe:      rope key slice cache     [B, S, qk_rope_head_dim]
    """

    kv_latent: torch.Tensor
    k_pe: torch.Tensor


class DiffusionMLACache:
    """Per-layer MLA latent cache used by diffusion sampler.

    Supports operations required by the spec:
      - crop(seq_len)
      - select_partial(indices)
      - append(new_entries)
      - batch_repeat(k)
      - batch_select(indices)
    """

    def __init__(self, n_layers: int):
        self.layers: list[Optional[MLALayerCache]] = [None for _ in range(n_layers)]

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    @property
    def seq_len(self) -> int:
        for layer in self.layers:
            if layer is not None:
                return int(layer.kv_latent.shape[1])
        return 0

    def clone(self) -> "DiffusionMLACache":
        out = DiffusionMLACache(self.n_layers)
        for i, layer in enumerate(self.layers):
            if layer is None:
                out.layers[i] = None
            else:
                out.layers[i] = MLALayerCache(
                    kv_latent=layer.kv_latent.clone(),
                    k_pe=layer.k_pe.clone(),
                )
        return out

    def crop(self, seq_len: int):
        """Keep first *seq_len* cached positions."""
        for i, layer in enumerate(self.layers):
            if layer is None:
                continue
            self.layers[i] = MLALayerCache(
                kv_latent=layer.kv_latent[:, :seq_len, :],
                k_pe=layer.k_pe[:, :seq_len, :],
            )

    def select_partial(self, indices: torch.Tensor):
        """Select sequence positions from cache (same indices for all batch rows)."""
        idx = indices.long()
        for i, layer in enumerate(self.layers):
            if layer is None:
                continue
            self.layers[i] = MLALayerCache(
                kv_latent=layer.kv_latent.index_select(1, idx),
                k_pe=layer.k_pe.index_select(1, idx),
            )

    def append(self, new_entries: list[MLALayerCache]):
        """Append per-layer entries along sequence dimension."""
        if len(new_entries) != self.n_layers:
            raise ValueError(
                f"Expected {self.n_layers} cache entries, got {len(new_entries)}"
            )

        for i, entry in enumerate(new_entries):
            cur = self.layers[i]
            if cur is None:
                self.layers[i] = MLALayerCache(
                    kv_latent=entry.kv_latent,
                    k_pe=entry.k_pe,
                )
            else:
                self.layers[i] = MLALayerCache(
                    kv_latent=torch.cat([cur.kv_latent, entry.kv_latent], dim=1),
                    k_pe=torch.cat([cur.k_pe, entry.k_pe], dim=1),
                )

    def batch_repeat(self, k: int):
        """Repeat batch entries k times (interleave)."""
        if k <= 1:
            return
        for i, layer in enumerate(self.layers):
            if layer is None:
                continue
            self.layers[i] = MLALayerCache(
                kv_latent=layer.kv_latent.repeat_interleave(k, dim=0),
                k_pe=layer.k_pe.repeat_interleave(k, dim=0),
            )

    def batch_select(self, indices: torch.Tensor):
        """Select batch rows from cache."""
        idx = indices.long()
        for i, layer in enumerate(self.layers):
            if layer is None:
                continue
            self.layers[i] = MLALayerCache(
                kv_latent=layer.kv_latent.index_select(0, idx),
                k_pe=layer.k_pe.index_select(0, idx),
            )


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * rms * self.weight).to(dtype)


# ---------------------------------------------------------------------------
# RoPE utilities (arbitrary position_ids)
# ---------------------------------------------------------------------------

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension into the first half (negated)."""
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to *x* using precomputed cos/sin.

    Shapes:
        x:   (B, T, [H,] rope_dim)
        cos: (B, T, rope_dim)
        sin: (B, T, rope_dim)

    If *x* has a head dimension (4-D), cos/sin are broadcast over it.
    """
    if x.ndim == 4:
        cos = cos.unsqueeze(2)          # (B, T, 1, D)
        sin = sin.unsqueeze(2)
    return x * cos + _rotate_half(x) * sin


# ---------------------------------------------------------------------------
# Composite-PIT: shared parameter interface
# ---------------------------------------------------------------------------

class CompositePITInterface(nn.Module):
    """Shared parameters owned jointly by embedding and head.

    Parameters:
        byte_memory   : (num_byte_symbols, shared_per_slot)
        byte_chol_raw : (shared_per_slot, shared_per_slot)  -- lower-tri factor
        token_embed   : nn.Embedding(vocab_size, 16 * token_per_slot)
    """

    def __init__(self, cfg: YAMITConfig):
        super().__init__()
        S = cfg.shared_per_slot
        self.pit_eps = cfg.pit_eps

        # -- byte memory (QR orthogonal init) --
        self.byte_memory = nn.Parameter(torch.empty(cfg.num_byte_symbols, S))

        # -- PIT Cholesky factor --
        # Diagonal = log(expm1(1.0)) ≈ 0.5414 so softplus(diag) = 1.0 → T = I at init.
        self.byte_chol_raw = nn.Parameter(torch.zeros(S, S))

        # -- token-private embedding --
        self.token_embed = nn.Embedding(cfg.vocab_size, 16 * cfg.token_per_slot)

        # -- token_bytes lookup: (vocab_size, 16) uint8→int mapping --
        # Must be registered via register_token_bytes() before forward.
        self.register_buffer("token_bytes", torch.zeros(cfg.vocab_size, 16, dtype=torch.long))
        self.register_buffer("token_bytes_initialized", torch.tensor(False), persistent=False)

    def _init_weights(self, cfg: YAMITConfig):
        S = cfg.shared_per_slot
        # QR orthogonal init for byte_memory
        Q, _ = torch.linalg.qr(torch.randn(cfg.num_byte_symbols, S))
        self.byte_memory.data.copy_(Q)
        # Cholesky: off-diag zeros, diag = log(expm1(1.0))
        diag_val = math.log(math.expm1(1.0))  # ≈ 0.5414
        self.byte_chol_raw.data.zero_()
        self.byte_chol_raw.data.diagonal().fill_(diag_val)
        # Token embedding
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=cfg.init_std)

    def register_token_bytes(self, token_bytes: torch.Tensor):
        """Set the token→bytes mapping (vocab_size, 16) with values in 0..256."""
        if token_bytes.shape != self.token_bytes.shape:
            raise ValueError(
                f"token_bytes shape must be {tuple(self.token_bytes.shape)}, got {tuple(token_bytes.shape)}"
            )
        if token_bytes.min().item() < 0 or token_bytes.max().item() >= 257:
            raise ValueError("token_bytes values must be in range [0, 256]")
        self.token_bytes.copy_(token_bytes.long())
        self.token_bytes_initialized.fill_(True)

    def cholesky_factor(self) -> torch.Tensor:
        """Compute stabilised lower-triangular Cholesky factor L (FP32).

        L has softplus-stabilised diagonal: ``softplus(raw_diag) + eps``.
        """
        raw = self.byte_chol_raw.float()
        L = torch.tril(raw)
        diag = L.diagonal()
        # Replace diagonal with stabilised version.
        L = L - torch.diag_embed(diag) + torch.diag_embed(F.softplus(diag) + self.pit_eps)
        return L


# ---------------------------------------------------------------------------
# Composite-PIT Embedding
# ---------------------------------------------------------------------------

class CompositePITEmbedding(nn.Module):
    """Composite-PIT embedding with special-token support.

    Regular (BPE) tokens:
        1. Look up byte memory for each slot -> z_shared (..., 16, S_shared)
        2. Apply T^{-1} via cholesky_solve  (FP32)
        3. Look up token_embed -> tok (..., 16, S_token), gated
        4. Concatenate [x_shared ; tok * gate] -> reshape to d_model

    Special tokens (mask, EOS, PAD, etc.) have no byte structure but still
    participate in PIT.  Their shared-path patterns come from a learned
    ``special_patterns`` parameter (n_special, 16, S_shared) instead of
    byte_memory lookup.  The same T^{-1} is applied, maintaining PIT duality
    with the head (which uses T on the same patterns).
    """

    def __init__(self, pit: CompositePITInterface, cfg: YAMITConfig):
        super().__init__()
        self.pit = pit
        self.cfg = cfg

        # Token-private gate (starts closed: zeros init -> output is 0 at init).
        self.token_up_gate = nn.Parameter(torch.zeros(16 * cfg.token_per_slot))

        # Special token support — populated by register_special_tokens().
        self.register_buffer(
            "special_token_ids",
            torch.tensor([], dtype=torch.long),
            persistent=True,
        )
        self.register_buffer(
            "_special_id_to_idx",
            torch.full((cfg.vocab_size,), -1, dtype=torch.long),
            persistent=False,
        )
        # Learned shared-path patterns for special tokens (replaces byte_memory lookup).
        # Shape: (n_special, 16, S_shared).  Initialised in register_special_tokens().
        self.special_patterns = nn.Parameter(torch.empty(0, 16, cfg.shared_per_slot))

    def register_special_tokens(self, token_ids: list[int]):
        """Register token IDs that use learned patterns instead of byte lookup.

        Must be called before the first forward pass (typically right after
        model construction, using IDs from the tokenizer artifact metadata).
        """
        ids = torch.tensor(sorted(set(token_ids)), dtype=torch.long)
        n = ids.numel()
        self.special_token_ids = ids
        # Orthogonal-ish init to match byte_memory's QR init.
        S = self.cfg.shared_per_slot
        patterns = torch.randn(n, 16, S)
        for i in range(n):
            Q, _ = torch.linalg.qr(patterns[i].t())
            patterns[i] = Q.t()[:16]
        self.special_patterns = nn.Parameter(patterns)
        # Rebuild lookup.
        self._special_id_to_idx = torch.full(
            (self.cfg.vocab_size,), -1, dtype=torch.long,
            device=ids.device,
        )
        for idx, tid in enumerate(ids.tolist()):
            if tid < self.cfg.vocab_size:
                self._special_id_to_idx[tid] = idx

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) token IDs.
        Returns:
            embeddings: (B, T, d_model).
        """
        cfg = self.cfg
        B, T = input_ids.shape

        if not bool(self.pit.token_bytes_initialized.item()):
            raise RuntimeError(
                "token_bytes table is not initialized. Provide token_bytes when creating "
                "YAMIT or call model.pit.register_token_bytes(...)."
            )

        # --- shared path: get z_shared patterns --- (FP32 for PIT stability)
        byte_ids = self.pit.token_bytes[input_ids]              # (B, T, 16) long
        z_shared = self.pit.byte_memory[byte_ids].float()       # (B, T, 16, S_shared)

        # Override z_shared for special tokens with learned patterns.
        if self.special_patterns.numel() > 0:
            idx = self._special_id_to_idx[input_ids]            # (B, T)
            is_special = idx >= 0                                # (B, T)
            if is_special.any():
                safe_idx = idx.clamp(min=0)
                sp = self.special_patterns[safe_idx].float()    # (B, T, 16, S_shared)
                z_shared = torch.where(
                    is_special.unsqueeze(-1).unsqueeze(-1), sp, z_shared
                )

        # --- apply T^{-1} --- (same for all tokens)
        L = self.pit.cholesky_factor()                          # (S, S) FP32
        z_t = z_shared.transpose(-1, -2)                        # (B, T, S_shared, 16)
        x_t = torch.cholesky_solve(z_t, L)                      # (B, T, S_shared, 16)
        x_shared = x_t.transpose(-1, -2)                        # (B, T, 16, S_shared)

        # --- token-private path ---
        tok = self.pit.token_embed(input_ids)                   # (B, T, 16*S_token)
        tok = tok.view(B, T, 16, cfg.token_per_slot)
        gate = self.token_up_gate.view(16, cfg.token_per_slot)
        tok = tok * gate                                        # gate starts at 0

        # --- concatenate and reshape ---
        out = torch.cat([x_shared.to(tok.dtype), tok], dim=-1)  # (B, T, 16, dims_per_slot)
        out = out.reshape(B, T, cfg.d_model)

        return out


# ---------------------------------------------------------------------------
# Composite-PIT LM Head
# ---------------------------------------------------------------------------

class CompositePITHead(nn.Module):
    """Composite-PIT LM head.

    Logit computation:
        h  -> split into 16 slots of dims_per_slot
           -> split each slot into (shared, private)

        Shared path (PIT forward):
            T = L L^T
            g = h_shared @ T
            patterns = byte_memory[token_bytes]   (V, 16, S_shared)
              (with special_patterns substituted for special token rows)
            logits_shared = einsum(g, patterns)

        Private path:
            logits_token = h_private_flat @ token_embed.weight^T + bias

        logits = logits_shared + logits_token
    """

    def __init__(self, pit: CompositePITInterface, cfg: YAMITConfig):
        super().__init__()
        self.pit = pit
        self.cfg = cfg
        # Head-only bias on token-private path.
        self.token_out_bias = nn.Parameter(torch.zeros(cfg.vocab_size))
        # Reference to embedding's special_patterns, set by YAMIT.__init__.
        self.embed: Optional[CompositePITEmbedding] = None

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: (B, T, d_model) hidden states after final norm.
        Returns:
            logits: (B, T, vocab_size).
        """
        cfg = self.cfg
        B, T, _ = h.shape
        S_shared = cfg.shared_per_slot
        S_token = cfg.token_per_slot

        # --- split into slots ---
        h_slots = h.view(B, T, 16, cfg.dims_per_slot)
        h_shared = h_slots[:, :, :, :S_shared]                 # (B, T, 16, S_shared)
        h_private = h_slots[:, :, :, S_shared:]                # (B, T, 16, S_token)

        # --- shared path (PIT forward, FP32) ---
        L = self.pit.cholesky_factor()                          # (S, S) FP32
        T_gram = L @ L.t()                                     # (S, S)
        g = h_shared.float() @ T_gram                          # (B, T, 16, S_shared)

        # Build (V, 16, S_shared) pattern table: byte_memory for regular tokens,
        # special_patterns for special tokens.
        all_patterns = self.pit.byte_memory[self.pit.token_bytes]  # (V, 16, S_shared)
        if self.embed is not None and self.embed.special_patterns.numel() > 0:
            all_patterns = all_patterns.clone()
            for idx, tid in enumerate(self.embed.special_token_ids.tolist()):
                if tid < cfg.vocab_size:
                    all_patterns[tid] = self.embed.special_patterns[idx]

        # einsum 'btsd,vsd->btv'
        logits_shared = torch.einsum(
            "btsd,vsd->btv", g, all_patterns.float()
        )

        # --- token-private path ---
        h_tok_flat = h_private.reshape(B, T, 16 * S_token)
        logits_token = (
            F.linear(h_tok_flat, self.pit.token_embed.weight, self.token_out_bias)
        )

        logits = logits_shared.to(h.dtype) + logits_token
        return logits


# ---------------------------------------------------------------------------
# MLA Dense Attention
# ---------------------------------------------------------------------------

class MLAAttention(nn.Module):
    """Multi-Head Latent Attention (MLA).

    Query path:
        x → wq_a → q_norm → wq_b → split(q_nope, q_pe) → RoPE(q_pe)

    KV path:
        x → wkv_a → split(kv_latent, k_pe)
        kv_latent → kv_norm → wkv_b → split(k_nope, v)
        k_pe → RoPE(k_pe)   (shared across all heads)

    Dense prefill attention:
        Q = cat(q_nope, q_pe)              per head
        K = cat(k_nope, k_pe_broadcast)    per head
        V = v                              per head
        out = SDPA(Q, K, V, scale=1/√(nope+rope)) → wo

    Absorbed decode attention:
        - Scores in latent space (no full K materialization):
            (q_nope @ Wk) @ kv_cache^T + q_pe @ pe_cache^T
        - Values in latent space projected at end via Wv.
    KV cache stores (kv_latent, k_pe).
    """

    def __init__(self, cfg: YAMITConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.qk_nope_head_dim = cfg.qk_nope_head_dim
        self.qk_rope_head_dim = cfg.qk_rope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.kv_compress_dim = cfg.kv_compress_dim
        self.scale = cfg.qk_head_dim ** -0.5
        self.orig_ctx_len = cfg.max_seq_len

        # ── query path ──
        self.wq_a = nn.Linear(cfg.d_model, cfg.q_compress_dim, bias=False)
        self.q_norm = RMSNorm(cfg.q_compress_dim, eps=cfg.rms_norm_eps)
        self.wq_b = nn.Linear(
            cfg.q_compress_dim,
            cfg.n_heads * cfg.qk_head_dim,
            bias=False,
        )

        # ── KV path ──
        self.wkv_a = nn.Linear(
            cfg.d_model,
            cfg.kv_compress_dim + cfg.qk_rope_head_dim,
            bias=False,
        )
        self.kv_norm = RMSNorm(cfg.kv_compress_dim, eps=cfg.rms_norm_eps)
        self.wkv_b = nn.Linear(
            cfg.kv_compress_dim,
            cfg.n_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            bias=False,
        )

        # ── output ──
        self.wo = nn.Linear(cfg.n_heads * cfg.v_head_dim, cfg.d_model, bias=False)

        # ── RoPE inverse frequencies ──
        inv_freq = 1.0 / (
            cfg.rope_theta
            ** (torch.arange(0, cfg.qk_rope_head_dim, 2).float() / cfg.qk_rope_head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    # ------------------------------------------------------------------ RoPE
    def _rope_cos_sin(
        self, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin from arbitrary position IDs.

        Args:
            position_ids: (B, T) integer positions.
        Returns:
            cos, sin: each (B, T, rope_dim).
        """
        pos = position_ids.float()
        # Linear RoPE scaling for long context extension.
        max_pos = int(position_ids.max().item()) + 1
        if max_pos > self.orig_ctx_len:
            factor = math.ceil(max_pos / self.orig_ctx_len)
            pos = pos / factor

        # (B, T) x (D/2,) → (B, T, D/2)
        freqs = torch.einsum("bt,d->btd", pos, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)  # (B, T, rope_dim)
        return emb.cos(), emb.sin()

    def _project_q(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project queries and apply RoPE to rope slice only."""
        B, T, _ = x.shape
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(B, T, self.n_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        cos, sin = self._rope_cos_sin(position_ids)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin)
        return q_nope, q_pe

    def _project_kv_latent(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project to latent KV and rope key slice."""
        kv = self.wkv_a(x)
        kv_latent, k_pe = kv.split([self.kv_compress_dim, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_norm(kv_latent)

        cos, sin = self._rope_cos_sin(position_ids)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin)
        return kv_latent, k_pe

    def _expand_kv(
        self,
        kv_latent: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Expand latent KV to per-head K(nope)/V for dense prefill path."""
        B, T, _ = kv_latent.shape
        kv_expanded = self.wkv_b(kv_latent)
        kv_expanded = kv_expanded.view(
            B, T, self.n_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv_expanded.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return k_nope, v

    def forward_prefill(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sparse_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, MLALayerCache]:
        """Dense prefill forward; returns output and cache entries."""
        B, T, _ = x.shape

        q_nope, q_pe = self._project_q(x, position_ids)
        kv_latent, k_pe = self._project_kv_latent(x, position_ids)
        k_nope, v = self._expand_kv(kv_latent)

        q = torch.cat([q_nope, q_pe], dim=-1)
        k_pe_exp = k_pe.unsqueeze(2).expand(-1, -1, self.n_heads, -1)
        k = torch.cat([k_nope, k_pe_exp], dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_mask = mask
        if sparse_mask is not None:
            attn_mask = sparse_mask if attn_mask is None else (attn_mask + sparse_mask)

        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            is_causal=(attn_mask is None),
            dropout_p=(self.cfg.attn_dropout if self.training else 0.0),
            scale=self.scale,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.wo(out)

        cache_entry = MLALayerCache(kv_latent=kv_latent, k_pe=k_pe)
        return out, cache_entry

    def forward_decode_absorbed(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_cache: Optional[MLALayerCache],
        mask: Optional[torch.Tensor] = None,
        sparse_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, MLALayerCache]:
        """Absorbed decode path using latent KV cache.

        Expects decode chunk length 1 (single-step decode).
        """
        B, T, _ = x.shape
        if T != 1:
            raise ValueError("forward_decode_absorbed expects sequence length 1")

        q_nope, q_pe = self._project_q(x, position_ids)                  # (B,1,H,nope/rope)
        kv_new, k_pe_new = self._project_kv_latent(x, position_ids)      # (B,1,C) / (B,1,R)

        if layer_cache is None:
            kv_all = kv_new
            k_pe_all = k_pe_new
        else:
            kv_all = torch.cat([layer_cache.kv_latent, kv_new], dim=1)   # (B,S,C)
            k_pe_all = torch.cat([layer_cache.k_pe, k_pe_new], dim=1)     # (B,S,R)

        # wkv_b: [H*(nope+v), C] -> [H, nope+v, C]
        wkv_b = self.wkv_b.weight.view(
            self.n_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_compress_dim,
        )
        wk = wkv_b[:, : self.qk_nope_head_dim, :]                        # [H, nope, C]
        wv = wkv_b[:, self.qk_nope_head_dim :, :]                        # [H, v, C]

        # Absorb K projection into query: [B,1,H,nope] x [H,nope,C] -> [B,1,H,C]
        q_nope_abs = torch.einsum("bthd,hdc->bthc", q_nope, wk)

        # Scores from latent content + rope component.
        scores = (
            torch.einsum("bthc,bsc->bths", q_nope_abs, kv_all)
            + torch.einsum("bthr,bsr->bths", q_pe, k_pe_all)
        ) * self.scale

        if mask is not None:
            scores = scores + mask
        if sparse_mask is not None:
            scores = scores + sparse_mask

        attn = torch.softmax(scores.float(), dim=-1).to(q_nope.dtype)

        # Weighted sum in latent space then absorb V projection.
        latent_ctx = torch.einsum("bths,bsc->bthc", attn, kv_all)       # [B,1,H,C]
        out_heads = torch.einsum("bthc,hdc->bthd", latent_ctx, wv)      # [B,1,H,v]
        out = out_heads.reshape(B, 1, self.n_heads * self.v_head_dim)
        out = self.wo(out)

        new_entry = MLALayerCache(kv_latent=kv_new, k_pe=k_pe_new)
        return out, new_entry

    # --------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sparse_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            (B, T, d_model)
            position_ids: (B, T)   — can be non-sequential (ReFusion shuffle)
            mask:         optional attention mask; None → causal
        Returns:
            out: (B, T, d_model)
        """
        out, _ = self.forward_prefill(
            x,
            position_ids,
            mask=mask,
            sparse_mask=sparse_mask,
        )
        return out


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------

class SwiGLUMLP(nn.Module):
    def __init__(self, cfg: YAMITConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class YAMITBlock(nn.Module):
    """Pre-norm residual block:  h = x + MLA(norm(x));  out = h + MLP(norm(h))."""

    def __init__(self, cfg: YAMITConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = MLAAttention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = SwiGLUMLP(cfg)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        sparse_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), position_ids, mask, sparse_mask=sparse_mask)
        x = x + self.mlp(self.mlp_norm(x))
        return x

    def forward_with_cache(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        layer_cache: Optional[MLALayerCache],
        use_cache: bool,
        decode_mode: bool,
        mask: Optional[torch.Tensor] = None,
        sparse_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[MLALayerCache]]:
        x_norm = self.attn_norm(x)
        if decode_mode:
            attn_out, new_entry = self.attn.forward_decode_absorbed(
                x_norm,
                position_ids,
                layer_cache=layer_cache,
                mask=mask,
                sparse_mask=sparse_mask,
            )
        else:
            attn_out, new_entry = self.attn.forward_prefill(
                x_norm,
                position_ids,
                mask=mask,
                sparse_mask=sparse_mask,
            )

        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, (new_entry if use_cache else None)


# ---------------------------------------------------------------------------
# Full YAMIT Model
# ---------------------------------------------------------------------------

class YAMIT(nn.Module):
    """YAMIT transformer: MLA attention + Composite-PIT embedding/head."""

    def __init__(
        self,
        cfg: YAMITConfig,
        token_bytes: Optional[torch.Tensor] = None,
        special_token_ids: Optional[list[int]] = None,
    ):
        """
        Args:
            cfg:               model configuration.
            token_bytes:       (vocab_size, 16) int tensor mapping token IDs to byte IDs
                               (0..255 for bytes, 256 for pad).  Can be set later via
                               ``model.pit.register_token_bytes(t)``.
            special_token_ids: list of token IDs that have no byte structure
                               (e.g. mask, EOS, PAD, and other control tokens).
                               These get learned PIT patterns instead of byte
                               lookup, but still go through T^{-1}/T.
        """
        super().__init__()
        self.cfg = cfg

        # ── shared PIT interface ──
        self.pit = CompositePITInterface(cfg)

        # ── embedding & head (share pit + special patterns) ──
        self.embed = CompositePITEmbedding(self.pit, cfg)
        self.head = CompositePITHead(self.pit, cfg)
        self.head.embed = self.embed  # head reads embed.special_patterns

        # ── transformer layers ──
        self.layers = nn.ModuleList([YAMITBlock(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)

        # ── init ──
        self._init_weights()

        if token_bytes is not None:
            self.pit.register_token_bytes(token_bytes)

        if special_token_ids:
            self.embed.register_special_tokens(special_token_ids)

    # --------------------------------------------------------------- init
    def _init_weights(self):
        cfg = self.cfg
        std = cfg.init_std

        # token_out_bias
        nn.init.zeros_(self.head.token_out_bias)

        # special_embeds initialised in register_special_tokens()
        # token_up_gate already zeros

        # Linear + Embedding + RMSNorm
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

        # PIT init must come *after* the generic sweep to avoid being overwritten.
        self.pit._init_weights(cfg)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        sparse_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:    (B, T) token IDs.
            position_ids: (B, T) position indices.  Defaults to 0..T-1.
            mask:         optional attention mask (None → causal).
        Returns:
            logits: (B, T, vocab_size).
        """
        B, T = input_ids.shape

        if position_ids is None:
            position_ids = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.embed(input_ids)

        for layer in self.layers:
            x = layer(x, position_ids, mask, sparse_mask=sparse_mask)

        x = self.norm(x)
        logits = self.head(x)
        return logits

    def init_cache(self) -> DiffusionMLACache:
        return DiffusionMLACache(n_layers=self.cfg.n_layers)

    def forward_with_cache(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        cache: Optional[DiffusionMLACache] = None,
        use_cache: bool = True,
        mask: Optional[torch.Tensor] = None,
        sparse_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[DiffusionMLACache]]:
        """Cache-aware forward used by diffusion sampler.

        - If cache is empty or sequence length > 1: dense prefill path.
        - If cache has history and sequence length == 1: absorbed decode path.
        - If cache has history and sequence length > 1: decode token-by-token.
        """
        B, T = input_ids.shape

        if position_ids is None:
            if cache is not None:
                start = cache.seq_len
            else:
                start = 0
            position_ids = (
                torch.arange(start, start + T, device=input_ids.device)
                .unsqueeze(0)
                .expand(B, -1)
            )

        # Decode chunk path: step token-by-token so absorbed mode remains correct.
        if cache is not None and cache.seq_len > 0 and T > 1:
            logits_steps = []
            cur_cache = cache
            for t in range(T):
                logits_t, cur_cache = self.forward_with_cache(
                    input_ids=input_ids[:, t : t + 1],
                    position_ids=position_ids[:, t : t + 1],
                    cache=cur_cache,
                    use_cache=use_cache,
                    mask=mask,
                    sparse_mask=sparse_mask,
                )
                logits_steps.append(logits_t)
            logits = torch.cat(logits_steps, dim=1)
            return logits, cur_cache

        x = self.embed(input_ids)

        if cache is None and use_cache:
            cache = self.init_cache()

        decode_mode = cache is not None and cache.seq_len > 0 and T == 1
        new_entries: list[MLALayerCache] = []

        for i, layer in enumerate(self.layers):
            layer_cache = cache.layers[i] if cache is not None else None
            x, entry = layer.forward_with_cache(
                x,
                position_ids,
                layer_cache=layer_cache,
                use_cache=use_cache,
                decode_mode=decode_mode,
                mask=mask,
                sparse_mask=sparse_mask,
            )
            if use_cache and entry is not None:
                new_entries.append(entry)

        x = self.norm(x)
        logits = self.head(x)

        if use_cache and cache is not None:
            cache.append(new_entries)

        return logits, cache

    # --------------------------------------------------------------- utils
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_report(self) -> str:
        """Human-readable parameter breakdown."""
        lines = []
        total = 0
        for name, p in self.named_parameters():
            n = p.numel()
            total += n
            lines.append(f"  {name:60s} {str(tuple(p.shape)):>30s}  {n:>12,}")
        lines.append(f"  {'TOTAL':60s} {'':>30s}  {total:>12,}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Token-bytes table builder
# ---------------------------------------------------------------------------

def build_token_bytes_table(
    token_to_bytes: dict[int, bytes],
    vocab_size: int,
    max_bytes: int = 16,
    pad_byte_id: int = 256,
) -> torch.Tensor:
    """Build the (vocab_size, max_bytes) token→byte-IDs mapping.

    Args:
        token_to_bytes: dict mapping token_id → raw bytes.  Missing IDs and
                        tokens with >max_bytes are filled with all-pad.
        vocab_size:     total vocabulary size (including padding tokens).
        max_bytes:      slot count (default 16).
        pad_byte_id:    byte symbol used for empty slots (default 256).

    Returns:
        Tensor (vocab_size, max_bytes) of int64, values in 0..pad_byte_id.
    """
    table = torch.full((vocab_size, max_bytes), pad_byte_id, dtype=torch.long)
    for tok_id, raw in token_to_bytes.items():
        if tok_id >= vocab_size:
            continue
        if len(raw) > max_bytes:
            # Token exceeds max bytes — should not exist in a properly
            # generated tokenizer.  Fill with pad; caller should validate.
            continue
        for i, b in enumerate(raw):
            table[tok_id, i] = b
    return table


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = MODEL_S
    print(f"Model-S config:")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}, n_heads={cfg.n_heads}")
    print(f"  dims_per_slot={cfg.dims_per_slot} = shared({cfg.shared_per_slot}) + token({cfg.token_per_slot})")
    print(f"  qk_head_dim={cfg.qk_head_dim} = nope({cfg.qk_nope_head_dim}) + rope({cfg.qk_rope_head_dim})")
    print()

    # Create with random token_bytes for testing.
    token_bytes = torch.randint(0, 257, (cfg.vocab_size, 16))
    model = YAMIT(cfg, token_bytes=token_bytes)

    n_params = model.param_count()
    print(f"Total parameters: {n_params:,}")
    print()

    # Quick forward pass.
    B, T = 2, 64
    x = torch.randint(0, cfg.vocab_size, (B, T))
    pos = torch.arange(T).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        logits = model(x, position_ids=pos)
    print(f"Input:  {x.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Logits sample (mean={logits.mean().item():.4f}, std={logits.std().item():.4f})")
