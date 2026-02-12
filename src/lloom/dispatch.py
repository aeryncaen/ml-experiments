"""Banked expert dispatch for LLooM.

AttentionParamBank — sequence-side attention experts (up → QKV → SDPA → norm → skip-mul → down).
MLPParamBank — token-side SwiGLU MLP experts (gate_up → SiLU*gate → down).

Both support 50% parameter sharing (shared + private weight slices) and use
gather + einsum for dispatch — no Python loops over pool_size.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_bank(pool_size: int, in_dim: int, out_dim: int,
               shared_fraction: float) -> tuple[nn.Parameter | None, nn.Parameter, int, int]:
    """Create a shared/private weight bank.

    Returns:
        (shared_weight, private_bank, shared_out, private_out)
        shared_weight: (in_dim, shared_out) or None
        private_bank:  (pool_size, in_dim, private_out)
    """
    shared_out = round(out_dim * shared_fraction) if shared_fraction > 0 else 0
    private_out = out_dim - shared_out

    shared_weight = None
    if shared_out > 0:
        shared_weight = nn.Parameter(
            torch.randn(in_dim, shared_out) * (in_dim ** -0.5))

    private_bank = nn.Parameter(
        torch.randn(pool_size, in_dim, private_out) * (in_dim ** -0.5))

    return shared_weight, private_bank, shared_out, private_out


def _gather_weights(shared_weight: nn.Parameter | None,
                    private_bank: nn.Parameter,
                    idx: torch.Tensor,
                    shared_out: int,
                    N: int, top_k: int, in_dim: int
                    ) -> torch.Tensor:
    """Gather banked weights for selected experts.

    Args:
        idx: (N, top_k) expert indices, already clamped to valid range.

    Returns:
        (N, top_k, in_dim, out_dim) weight tensor.
    """
    # Private: gather → reshape
    priv_w = private_bank[idx.reshape(-1)].reshape(N, top_k, in_dim, -1)

    if shared_weight is not None:
        shared_w = shared_weight.unsqueeze(0).unsqueeze(0).expand(
            N, top_k, -1, -1)
        return torch.cat([shared_w, priv_w], dim=-1)
    return priv_w


def _make_1d_bank(pool_size: int, dim: int,
                  shared_fraction: float,
                  init_val: float = 1.0
                  ) -> tuple[nn.Parameter | None, nn.Parameter, int, int]:
    """Create a shared/private 1D parameter bank (e.g. norm weights, betas).

    Returns:
        (shared_param, private_bank, shared_dim, private_dim)
        shared_param: (shared_dim,) or None
        private_bank: (pool_size, private_dim)
    """
    shared_dim = round(dim * shared_fraction) if shared_fraction > 0 else 0
    private_dim = dim - shared_dim

    shared_param = None
    if shared_dim > 0:
        shared_param = nn.Parameter(torch.full((shared_dim,), init_val))

    private_bank = nn.Parameter(torch.full((pool_size, private_dim), init_val))

    return shared_param, private_bank, shared_dim, private_dim


def _gather_1d(shared_param: nn.Parameter | None,
               private_bank: nn.Parameter,
               idx: torch.Tensor,
               N: int, top_k: int
               ) -> torch.Tensor:
    """Gather 1D banked params for selected experts.

    Args:
        idx: (N, top_k) expert indices, already clamped.

    Returns:
        (N, top_k, dim) parameter tensor.
    """
    priv = private_bank[idx.reshape(-1)].reshape(N, top_k, -1)

    if shared_param is not None:
        shared = shared_param.unsqueeze(0).unsqueeze(0).expand(N, top_k, -1)
        return torch.cat([shared, priv], dim=-1)
    return priv


# ---------------------------------------------------------------------------
# AttentionParamBank
# ---------------------------------------------------------------------------

class AttentionParamBank(nn.Module):
    """Banked attention expert weights for sequence-side dispatch.

    Expert architecture (per top-k slot):
        h_up = SiLU(up_proj(x))
        q, k, v = split(qkv_proj(h_up))
        y = SDPA(q, k, v)
        y = o_proj(y)
        y = RMSNorm(y) * h_up      # skip-multiply
        out = down_proj(y)

    All weight matrices are banked: (pool_size, in, out) with optional
    shared/private split. Dispatch gathers weights for selected experts
    and loops over top_k (fixed, small: 2-3).

    Args:
        pool_size: Number of attention experts.
        dim: Model dimension (D).
        inner_dim: Expert inner dimension after up-projection.
        n_heads: Number of attention heads.
        shared_fraction: Fraction of output dim shared across experts.
        is_causal: Whether attention is causal.
    """

    def __init__(self, pool_size: int, dim: int, inner_dim: int,
                 n_heads: int, shared_fraction: float = 0.5,
                 is_causal: bool = True):
        super().__init__()
        self.pool_size = pool_size
        self.dim = dim
        self.inner_dim = inner_dim
        self.n_heads = n_heads
        self.head_dim = inner_dim // n_heads
        self.shared_fraction = shared_fraction
        self.is_causal = is_causal

        assert inner_dim % n_heads == 0, (
            f"inner_dim ({inner_dim}) must be divisible by n_heads ({n_heads})")

        # up_proj: D → D_inner
        self.up_shared, self.up_bank, self.up_shared_out, self.up_private_out = \
            _make_bank(pool_size, dim, inner_dim, shared_fraction)

        # qkv_proj: D_inner → 3 * D_inner (fused Q, K, V)
        self.qkv_shared, self.qkv_bank, self.qkv_shared_out, self.qkv_private_out = \
            _make_bank(pool_size, inner_dim, 3 * inner_dim, shared_fraction)

        # o_proj: D_inner → D_inner (output projection after attention)
        self.o_shared, self.o_bank, self.o_shared_out, self.o_private_out = \
            _make_bank(pool_size, inner_dim, inner_dim, shared_fraction)

        # norm weights: D_inner (RMSNorm per expert, before skip-multiply)
        self.norm_shared, self.norm_bank, self.norm_shared_dim, self.norm_private_dim = \
            _make_1d_bank(pool_size, inner_dim, shared_fraction, init_val=1.0)

        # down_proj: D_inner → D
        self.down_shared, self.down_bank, self.down_shared_out, self.down_private_out = \
            _make_bank(pool_size, inner_dim, dim, shared_fraction)

        self._norm_eps = 1e-6

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """RMSNorm with per-expert weight vector.

        Args:
            x: (..., D_inner)
            weight: (..., D_inner) — broadcastable norm weights.
        """
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self._norm_eps)
        return x / rms * weight

    def forward(self, x: torch.Tensor, expert_idx: torch.Tensor,
                expert_weights: torch.Tensor) -> torch.Tensor:
        """Dispatch through banked attention experts.

        Args:
            x: (B, T, D) input hidden states.
            expert_idx: (B, top_k) expert indices per sample (clamped to valid range).
            expert_weights: (B, top_k) routing weights (exit/bridge already zeroed).

        Returns:
            (B, T, D) weighted merge of expert outputs.
        """
        B, T, D = x.shape
        top_k = expert_idx.shape[1]
        safe_idx = expert_idx.clamp(max=self.pool_size - 1)

        # Gather all weights for selected experts
        up_w = _gather_weights(
            self.up_shared, self.up_bank, safe_idx,
            self.up_shared_out, B, top_k, self.dim)              # (B, K, D, D_i)
        qkv_w = _gather_weights(
            self.qkv_shared, self.qkv_bank, safe_idx,
            self.qkv_shared_out, B, top_k, self.inner_dim)       # (B, K, D_i, 3*D_i)
        o_w = _gather_weights(
            self.o_shared, self.o_bank, safe_idx,
            self.o_shared_out, B, top_k, self.inner_dim)          # (B, K, D_i, D_i)
        norm_w = _gather_1d(
            self.norm_shared, self.norm_bank, safe_idx, B, top_k) # (B, K, D_i)
        down_w = _gather_weights(
            self.down_shared, self.down_bank, safe_idx,
            self.down_shared_out, B, top_k, self.inner_dim)       # (B, K, D_i, D)

        # Process each top-k slot (fixed small loop: 2-3 iterations)
        outputs = []
        for k in range(top_k):
            # up_proj: (B, T, D) @ (B, D, D_i) → (B, T, D_i)
            h_up = torch.bmm(x, up_w[:, k])
            h_up = F.silu(h_up)

            # QKV: (B, T, D_i) @ (B, D_i, 3*D_i) → (B, T, 3*D_i)
            qkv = torch.bmm(h_up, qkv_w[:, k])
            q, kk, v = qkv.split(self.inner_dim, dim=-1)

            # Reshape for multi-head attention
            q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            kk = kk.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

            # Scaled dot-product attention
            y = F.scaled_dot_product_attention(q, kk, v, is_causal=self.is_causal)

            # Back to (B, T, D_inner)
            y = y.transpose(1, 2).contiguous().view(B, T, self.inner_dim)

            # o_proj: (B, T, D_i) @ (B, D_i, D_i) → (B, T, D_i)
            y = torch.bmm(y, o_w[:, k])

            # RMSNorm + skip-multiply
            nw = norm_w[:, k].unsqueeze(1)  # (B, 1, D_i)
            y = self._rms_norm(y, nw) * h_up

            # down_proj: (B, T, D_i) @ (B, D_i, D) → (B, T, D)
            y = torch.bmm(y, down_w[:, k])

            outputs.append(y)

        # Stack and weighted merge: (B, K, T, D) → (B, T, D)
        stacked = torch.stack(outputs, dim=1)  # (B, K, T, D)
        weights = expert_weights.unsqueeze(-1).unsqueeze(-1)  # (B, K, 1, 1)
        return (stacked * weights).sum(dim=1)


# ---------------------------------------------------------------------------
# MLPParamBank
# ---------------------------------------------------------------------------

class MLPParamBank(nn.Module):
    """Banked SwiGLU MLP expert weights for token-side dispatch.

    Expert architecture (per top-k slot):
        gate, up = split(gate_up_proj(x))
        h = SiLU(gate) * up                  # SwiGLU
        out = down_proj(h)

    All weight matrices are banked. Dispatch is fully vectorized via einsum.

    Args:
        pool_size: Number of MLP experts.
        dim: Model dimension (D).
        inner_dim: Expert inner dimension (D_inner, for gate and up each).
        shared_fraction: Fraction of output dim shared across experts.
    """

    def __init__(self, pool_size: int, dim: int, inner_dim: int,
                 shared_fraction: float = 0.5):
        super().__init__()
        self.pool_size = pool_size
        self.dim = dim
        self.inner_dim = inner_dim
        self.shared_fraction = shared_fraction

        # gate_up_proj: D → 2 * D_inner (fused gate + up)
        self.gate_up_shared, self.gate_up_bank, \
            self.gate_up_shared_out, self.gate_up_private_out = \
            _make_bank(pool_size, dim, 2 * inner_dim, shared_fraction)

        # down_proj: D_inner → D
        self.down_shared, self.down_bank, \
            self.down_shared_out, self.down_private_out = \
            _make_bank(pool_size, inner_dim, dim, shared_fraction)

    def _assemble_weight(self, shared: nn.Parameter | None,
                         private: nn.Parameter, expert_id: int
                         ) -> torch.Tensor:
        """Assemble full weight matrix for a single expert.

        Returns:
            (in_dim, out_dim) weight matrix.
        """
        priv = private[expert_id]  # (in_dim, priv_out)
        if shared is not None:
            return torch.cat([shared, priv], dim=-1)  # (in_dim, out_dim)
        return priv

    def forward(self, x: torch.Tensor, expert_idx: torch.Tensor,
                expert_weights: torch.Tensor) -> torch.Tensor:
        """Dispatch through banked SwiGLU MLP experts.

        Uses grouped dispatch: loop over pool_size experts, apply each
        expert's weights to ALL tokens via matmul, then mask to only
        accumulate for tokens assigned to that expert.  No per-token
        weight copies — memory is O(N * D_inner) for activations instead
        of O(N * D * D_inner) for gathered weight tensors.

        torch.compile friendly: fixed loops, no bool indexing, no
        dynamic shapes.

        Args:
            x: (N, D) flattened input tokens.
            expert_idx: (N, top_k) expert indices (clamped to valid range).
            expert_weights: (N, top_k) routing weights (exit/bridge zeroed).

        Returns:
            (N, D) output tokens.
        """
        N = x.shape[0]
        top_k = expert_idx.shape[1]
        safe_idx = expert_idx.clamp(max=self.pool_size - 1)

        # Output accumulator
        out = torch.zeros(N, self.dim, device=x.device, dtype=x.dtype)

        for k in range(top_k):
            slot_idx = safe_idx[:, k]       # (N,)
            slot_w = expert_weights[:, k]   # (N,)

            for e in range(self.pool_size):
                # Assemble expert weight matrices (shared + private)
                gate_up_W = self._assemble_weight(
                    self.gate_up_shared, self.gate_up_bank, e)  # (D, 2*D_i)
                down_W = self._assemble_weight(
                    self.down_shared, self.down_bank, e)  # (D_i, D)

                # Run expert on ALL tokens (matmul is cheap, no weight copies)
                gate_up = x @ gate_up_W  # (N, 2*D_i)
                gate, up = gate_up.split(self.inner_dim, dim=-1)
                h = F.silu(gate) * up  # (N, D_i)
                e_out = h @ down_W  # (N, D)

                # Mask: only accumulate for tokens assigned to this expert
                mask = (slot_idx == e).unsqueeze(-1)  # (N, 1)
                out = out + mask * slot_w[:, None] * e_out

        return out
