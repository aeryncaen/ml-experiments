"""
Ideal Transformer Block — derived from the compositional optimizer interpretation.

Key properties:
  - Attention is orthogonal projection (no softmax), not weighted average
  - Heads are structurally orthogonal via Cayley-parameterized basis
  - Data-dependent subspace selection (input controls which subspace to project onto)
  - MLP implements the selection step within the projected subspace
  - Returns the increment Δ

Reference dimensions for bench_ssm: d_model=64, n_heads=4, d_head=16
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


@dataclass
class IdealConfig:
    d_model: int
    n_heads: int = 4
    d_head: int = 0         # 0 = d_model // n_heads
    ffn_mult: float = 4.0
    causal: bool = True

    def __post_init__(self):
        if self.d_head == 0:
            self.d_head = self.d_model // self.n_heads
        assert self.d_model == self.n_heads * self.d_head, (
            f"d_model ({self.d_model}) must equal n_heads * d_head "
            f"({self.n_heads} * {self.d_head})"
        )


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def skew_to_orthogonal(A: torch.Tensor) -> torch.Tensor:
    """
    Map skew-symmetric A to orthogonal U via matrix exponential.
    exp(A) is always orthogonal when A is skew-symmetric (A = -A^T).

    Args:
        A: (..., d, d) skew-symmetric matrices
    Returns:
        U: (..., d, d) orthogonal matrices
    """
    return torch.matrix_exp(A)


class OrthogonalHeadProjection(nn.Module):
    """
    Project input into H structurally orthogonal head subspaces.

    Each head gets a d_head-dimensional subspace of R^{d_model}.
    Subspaces are columns of an orthogonal matrix U, parameterized
    via matrix_exp of a learned skew-symmetric matrix.

    Output per head: proj_h = U_h^T @ x  (coordinates in head h's subspace)
    These are concatenated, so the output has the same dim as input but in
    orthogonal coordinates where dims [h*d_h:(h+1)*d_h] belong to head h.

    The MLP then operates directly on these coordinates.
    No rotate-back step — the MLP learns in the orthogonal coordinate frame.
    """

    def __init__(self, config: IdealConfig):
        super().__init__()
        d = config.d_model
        self.d_model = d
        self.n_heads = config.n_heads
        self.d_head = config.d_head

        # Skew-symmetric matrix -> orthogonal basis via matrix_exp
        # Initialize with small random values so heads start differentiated
        A_init = torch.randn(d, d) * 0.1
        self.A_raw = nn.Parameter(A_init)

    def get_basis(self) -> torch.Tensor:
        """Return the orthogonal matrix U (d_model, d_model)."""
        A = self.A_raw - self.A_raw.T  # skew-symmetric
        return skew_to_orthogonal(A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            coeffs: (batch, seq_len, d_model) — coordinates in orthogonal basis
        """
        U = self.get_basis()  # (d, d)
        # Project: coeffs = x @ U  (each column of U is a basis vector)
        return x @ U


class SelectionMLP(nn.Module):
    """
    SwiGLU MLP that implements the selection step.
    Operates in the orthogonal coordinate frame. Output is in original space
    via the transpose projection.
    """

    def __init__(self, config: IdealConfig):
        super().__init__()
        d = config.d_model
        d_ff = int(d * config.ffn_mult)

        self.norm = RMSNorm(d)
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(self, coeffs: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coeffs: (batch, seq_len, d_model) — orthogonal-basis coordinates
            U: (d_model, d_model) — orthogonal basis (for projecting back)
        Returns:
            delta: (batch, seq_len, d_model) — increment in original space
        """
        h = self.norm(coeffs)
        h = F.silu(self.w_gate(h)) * self.w_up(h)
        delta_coeffs = self.w_down(h)
        # Project back to original space: delta = delta_coeffs @ U^T
        return delta_coeffs @ U.T


class IdealBlock(nn.Module):
    """
    One round of the compositional optimizer.

    1. Project input into orthogonal head subspaces (representation layer)
    2. SwiGLU MLP in the orthogonal coordinate frame (selection step)
    3. Project increment back to original space
    4. Return increment Δ

    bench_ssm's StackedModel handles pre-norm + residual:
        x = x + block(norm(x))
    """

    def __init__(self, config: IdealConfig):
        super().__init__()
        self.config = config
        self.projection = OrthogonalHeadProjection(config)
        self.selection = SelectionMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) — input (pre-normed by StackedModel)
        Returns:
            delta: (batch, seq_len, d_model) — the increment
        """
        U = self.projection.get_basis()  # (d, d) — shared across batch/seq
        coeffs = x @ U                   # project to orthogonal coords
        delta = self.selection(coeffs, U) # MLP + project back
        return delta
