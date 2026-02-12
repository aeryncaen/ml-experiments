"""LLooM configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLooMConfig:
    """Configuration for the LLooM dual-paradigm adaptive routing model.

    LLooM combines sequence-level routing through attention experts with
    token-level routing through SwiGLU MLP experts, connected by a raw
    passthrough bridge.
    """

    # ---------- core dimensions ----------
    dim: int = 64
    max_seq_len: int = 512

    # ---------- entry / exit stems ----------
    # Full transformer blocks (attention + SwiGLU MLP), non-routed, shared.
    stem_n_heads: int = 4
    stem_mlp_expansion: float = 1.75

    # ---------- sequence side (attention pool) ----------
    seq_pool_size: int = 8
    seq_top_k: int = 2
    seq_n_heads: int = 4
    seq_expansion: float = 1.75        # up_proj expansion for attention experts
    seq_max_hops: int = 16             # cumulative across all visits

    # ---------- token side (SwiGLU MLP pool) ----------
    tok_pool_size: int = 8
    tok_top_k: int = 2
    tok_expansion: float = 1.75        # gate_up expansion for MLP experts
    tok_max_hops: int = 32             # generous — MLP hops are cheap

    # ---------- routing ----------
    # exit_bias_init: None = auto = 0.0.
    # bridge_bias_init: None = auto = 0.25 — gives bridge a modest head start
    # so it can compete with the exit ramp at early hops.  Without this,
    # exit's growing ramp means samples always exit before bridging.
    exit_bias_init: float | None = None    # starting scalar bias for exit slot
    bridge_bias_init: float | None = None  # starting scalar bias for bridge slot (auto=0.25)
    exit_ramp_scale: float = 2.0       # exit_bias_init + ramp * (hops_used / max_hops)
    router_noise: float = 1.0          # gaussian noise scale, annealed to 0

    # ---------- weight sharing ----------
    # Convenience default: set shared_fraction to apply to all four categories.
    # Per-category overrides (None = use shared_fraction):
    shared_fraction: float = 0.0       # default for any unset per-category fraction
    seq_expert_shared_fraction: float | None = None  # attention expert bank weights
    tok_expert_shared_fraction: float | None = None  # MLP expert bank weights
    seq_router_shared_fraction: float | None = None  # seq-side routers + hop embeds
    tok_router_shared_fraction: float | None = None  # tok-side routers + hop embeds

    # ---------- hop conditioning ----------
    hop_gate_dim: int = 12             # prefix slice of hidden dim for gating

    # ---------- bridge ----------
    max_bridge_crossings: int = 4      # max total bridge crossings per sample

    # ---------- training ----------
    is_causal: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        # Validate dimensions
        assert self.dim > 0, f"dim must be positive, got {self.dim}"
        assert self.dim % self.stem_n_heads == 0, (
            f"dim ({self.dim}) must be divisible by stem_n_heads ({self.stem_n_heads})")

        # Validate sequence side
        assert self.seq_pool_size >= 1, f"seq_pool_size must be >= 1, got {self.seq_pool_size}"
        assert self.seq_top_k >= 1, f"seq_top_k must be >= 1, got {self.seq_top_k}"
        assert self.seq_top_k <= self.seq_pool_size, (
            f"seq_top_k ({self.seq_top_k}) must be <= seq_pool_size ({self.seq_pool_size})")
        assert self.seq_max_hops >= 1, f"seq_max_hops must be >= 1, got {self.seq_max_hops}"

        # Validate token side
        assert self.tok_pool_size >= 1, f"tok_pool_size must be >= 1, got {self.tok_pool_size}"
        assert self.tok_top_k >= 1, f"tok_top_k must be >= 1, got {self.tok_top_k}"
        assert self.tok_top_k <= self.tok_pool_size, (
            f"tok_top_k ({self.tok_top_k}) must be <= tok_pool_size ({self.tok_pool_size})")
        assert self.tok_max_hops >= 1, f"tok_max_hops must be >= 1, got {self.tok_max_hops}"

        # Validate shared fractions
        assert 0.0 <= self.shared_fraction <= 1.0, (
            f"shared_fraction must be in [0, 1], got {self.shared_fraction}")
        for name in ('seq_expert_shared_fraction', 'tok_expert_shared_fraction',
                     'seq_router_shared_fraction', 'tok_router_shared_fraction'):
            val = getattr(self, name)
            if val is not None:
                assert 0.0 <= val <= 1.0, f"{name} must be in [0, 1], got {val}"

        # Validate hop gate dim fits in hidden dim
        assert self.hop_gate_dim <= self.dim, (
            f"hop_gate_dim ({self.hop_gate_dim}) must be <= dim ({self.dim})")

        # Validate bridge crossings
        assert self.max_bridge_crossings >= 0, (
            f"max_bridge_crossings must be >= 0, got {self.max_bridge_crossings}")

        # Compute and cache inner dimensions
        self._seq_inner_dim = self._snap_dim(self.dim * self.seq_expansion, self.seq_n_heads)
        self._tok_inner_dim = self._snap_dim(self.dim * self.tok_expansion, snap=8)
        self._stem_inner_dim = self._snap_dim(self.dim * self.stem_mlp_expansion, self.stem_n_heads)

        # Validate attention head divisibility
        assert self._seq_inner_dim % self.seq_n_heads == 0, (
            f"seq_inner_dim ({self._seq_inner_dim}) must be divisible by "
            f"seq_n_heads ({self.seq_n_heads})")

    @staticmethod
    def _snap_dim(raw: float, snap: int) -> int:
        """Snap a dimension to the nearest multiple of `snap`, minimum `snap`."""
        return max(snap, round(raw / snap) * snap)

    # ---------- derived properties ----------

    @property
    def seq_inner_dim(self) -> int:
        """Inner dimension for sequence-side attention experts."""
        return self._seq_inner_dim

    @property
    def seq_head_dim(self) -> int:
        """Per-head dimension for sequence-side attention."""
        return self._seq_inner_dim // self.seq_n_heads

    @property
    def tok_inner_dim(self) -> int:
        """Inner dimension for token-side SwiGLU MLP experts."""
        return self._tok_inner_dim

    @property
    def stem_inner_dim(self) -> int:
        """Inner dimension for entry/exit stem MLP."""
        return self._stem_inner_dim

    @property
    def stem_head_dim(self) -> int:
        """Per-head dimension for stem attention."""
        return self.dim // self.stem_n_heads

    @property
    def seq_n_options(self) -> int:
        """Router output size for sequence side: pool_size + exit + bridge."""
        return self.seq_pool_size + 2

    @property
    def tok_n_options(self) -> int:
        """Router output size for token side: pool_size + exit + bridge."""
        return self.tok_pool_size + 2

    @property
    def seq_exit_idx(self) -> int:
        """Index of the exit slot in sequence-side router logits."""
        return self.seq_pool_size

    @property
    def seq_bridge_idx(self) -> int:
        """Index of the bridge slot in sequence-side router logits."""
        return self.seq_pool_size + 1

    @property
    def tok_exit_idx(self) -> int:
        """Index of the exit slot in token-side router logits."""
        return self.tok_pool_size

    @property
    def tok_bridge_idx(self) -> int:
        """Index of the bridge slot in token-side router logits."""
        return self.tok_pool_size + 1

    @property
    def stem_n_options(self) -> int:
        """Stem router output size: seq_pool_size + exit + bridge (same as seq pool)."""
        return self.seq_pool_size + 2

    @property
    def resolved_seq_expert_share(self) -> float:
        """Resolved sharing fraction for sequence-side expert banks."""
        v = self.seq_expert_shared_fraction
        return v if v is not None else self.shared_fraction

    @property
    def resolved_tok_expert_share(self) -> float:
        """Resolved sharing fraction for token-side expert banks."""
        v = self.tok_expert_shared_fraction
        return v if v is not None else self.shared_fraction

    @property
    def resolved_seq_router_share(self) -> float:
        """Resolved sharing fraction for sequence-side routers + hop embeds."""
        v = self.seq_router_shared_fraction
        return v if v is not None else self.shared_fraction

    @property
    def resolved_tok_router_share(self) -> float:
        """Resolved sharing fraction for token-side routers + hop embeds."""
        v = self.tok_router_shared_fraction
        return v if v is not None else self.shared_fraction

    @property
    def global_max_hops(self) -> int:
        """Maximum total hops across both sides (for hop embedding table size)."""
        return self.seq_max_hops + self.tok_max_hops


