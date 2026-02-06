"""
Ideal stacking model with cross-layer orthogonality penalty.

Two modes:
  1. Simple: wrap IdealBlock in bench_ssm's StackedModel (external pre-norm + residual).
     Use IdealWrapper for this.
  2. Full: IdealStackedModel that passes residual objectives between layers and
     collects cross-layer Δ inner products for the orthogonality penalty.
"""

from typing import Optional, List, Dict, Any, Tuple

import torch
import torch.nn as nn

from .block import IdealBlock, IdealConfig, RMSNorm


class IdealWrapper(nn.Module):
    """
    Thin wrapper for bench_ssm compatibility.
    __init__(d_model, **kwargs), forward(x) -> x (same shape).

    bench_ssm's StackedModel handles pre-norm + residual:
        x = x + layer(norm(x))

    So this just creates an IdealBlock and forwards through it.
    The block returns the increment Δ.
    """

    def __init__(self, d_model: int, n_heads: int = 4, ffn_mult: float = 4.0, **kwargs):
        super().__init__()
        config = IdealConfig(
            d_model=d_model,
            n_heads=n_heads,
            ffn_mult=ffn_mult,
        )
        self.block = IdealBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class IdealStackedModel(nn.Module):
    """
    Full ideal stacking with cross-layer orthogonality tracking.

    Handles its own pre-norm + residual connections (don't wrap in StackedModel).

    Each layer returns Δ_l. The model:
      1. Accumulates x_{l+1} = x_l + Δ_l
      2. Tracks all Δ_l for the cross-layer penalty
      3. Optionally updates a residual objective estimate

    The orthogonality penalty is: λ * Σ_{k<l} |<Δ_l, Δ_k>|² / (||Δ_l||² ||Δ_k||² + ε)
    (normalized so scale-invariant).

    This is returned as model.ortho_penalty after forward().
    """

    def __init__(
        self,
        config: IdealConfig,
        n_layers: int,
        ortho_lambda: float = 0.1,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.ortho_lambda = ortho_lambda

        self.layers = nn.ModuleList([IdealBlock(config) for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(config.d_model) for _ in range(n_layers)])
        self.final_norm = RMSNorm(config.d_model)

        # Penalty computed during forward, consumed by caller
        self.ortho_penalty: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            x: (batch, seq_len, d_model) — final output
        """
        deltas: List[torch.Tensor] = []

        for norm, layer in zip(self.norms, self.layers):
            delta = layer(norm(x))
            x = x + delta
            deltas.append(delta)

        x = self.final_norm(x)

        # Compute cross-layer orthogonality penalty
        if self.training and len(deltas) > 1:
            penalty = torch.tensor(0.0, device=x.device, dtype=x.dtype)
            eps = 1e-8
            for i in range(len(deltas)):
                for j in range(i):
                    # Flatten to (batch, seq_len * d_model) for inner product
                    di = deltas[i].flatten(1)  # (B, L*D)
                    dj = deltas[j].flatten(1)  # (B, L*D)
                    # Cosine similarity squared, averaged over batch
                    dot = (di * dj).sum(dim=-1)  # (B,)
                    norm_i = di.pow(2).sum(dim=-1).clamp(min=eps)  # (B,)
                    norm_j = dj.pow(2).sum(dim=-1).clamp(min=eps)  # (B,)
                    cos2 = dot.pow(2) / (norm_i * norm_j)  # (B,)
                    penalty = penalty + cos2.mean()
            self.ortho_penalty = self.ortho_lambda * penalty
        else:
            self.ortho_penalty = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return x
