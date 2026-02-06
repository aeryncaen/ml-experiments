"""
Sequential scan implementations for USB.

These are reference implementations - will be optimized with parallel/associative
scans later (e.g., via Triton kernels).

Supports two modes:
- elementwise: kv = k * v, state is (nheads, headdim)
- outer: kv = k ⊗ v, state is (nheads, headdim, headdim)
"""

import torch
from typing import List


def forward_scan(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    init_state: torch.Tensor,
) -> torch.Tensor:
    """
    Forward causal scan with Mamba-style discretization.
    
    Starts at t=0, moves forward. Integrates current position only.
    
    state_t = α[t] * state_{t-1} + β[t] * kv_t
    
    Args:
        kv: (batch, seq_len, nheads, headdim) for elementwise mode
            (batch, seq_len, nheads, headdim, headdim) for outer product mode
        alpha: (batch, seq_len, nheads) - decay rates in (0, 1)
        beta: (batch, seq_len, nheads) - input coefficients
        init_state: (batch, nheads, headdim) or (batch, nheads, headdim, headdim)
    
    Returns:
        states: same shape as kv - accumulated states at each position
    """
    outer_mode = kv.ndim == 5
    
    if outer_mode:
        batch, seq_len, nheads, headdim, _ = kv.shape
    else:
        batch, seq_len, nheads, headdim = kv.shape
    
    # Collect states in a list to avoid inplace ops
    state_list: List[torch.Tensor] = []
    state_tm1 = init_state

    if outer_mode:
        coef_dims = (slice(None), slice(None), None, None)
    else:
        coef_dims = (slice(None), slice(None), None)

    for t in range(seq_len):
        alpha_t = alpha[:, t][coef_dims]
        beta_t = beta[:, t][coef_dims]
        state_t = alpha_t * state_tm1 + beta_t * kv[:, t]
        state_list.append(state_t)
        state_tm1 = state_t
    
    # Stack along seq dimension
    return torch.stack(state_list, dim=1)


def backward_scan(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    init_state: torch.Tensor,
) -> torch.Tensor:
    """
    Backward causal scan with Mamba-style discretization.
    
    Starts at t=T-1 (end), moves backward. Integrates current position only.
    
    state_t = α[t] * state_{t+1} + β[t] * kv_t
    
    Args:
        kv: (batch, seq_len, nheads, headdim) for elementwise mode
            (batch, seq_len, nheads, headdim, headdim) for outer product mode
        alpha: (batch, seq_len, nheads) - decay rates in (0, 1)
        beta: (batch, seq_len, nheads) - input coefficients
        init_state: (batch, nheads, headdim) or (batch, nheads, headdim, headdim)
    
    Returns:
        states: same shape as kv - accumulated states at each position
    """
    outer_mode = kv.ndim == 5
    
    if outer_mode:
        batch, seq_len, nheads, headdim, _ = kv.shape
    else:
        batch, seq_len, nheads, headdim = kv.shape
    
    # Collect states in a list (will be reversed)
    state_list: List[torch.Tensor] = []
    state_tp1 = init_state

    if outer_mode:
        coef_dims = (slice(None), slice(None), None, None)
    else:
        coef_dims = (slice(None), slice(None), None)

    for t in range(seq_len - 1, -1, -1):
        alpha_t = alpha[:, t][coef_dims]
        beta_t = beta[:, t][coef_dims]
        state_t = alpha_t * state_tp1 + beta_t * kv[:, t]
        state_list.append(state_t)
        state_tp1 = state_t
    
    # Reverse and stack (we collected in reverse order)
    state_list.reverse()
    return torch.stack(state_list, dim=1)


def centered_scan(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    init_state: torch.Tensor,
) -> torch.Tensor:
    """
    Centered scan using a symmetric Mamba-style approximation.

    Returns the average of forward and backward scans.
    
    Args:
        kv: (batch, seq_len, nheads, headdim) for elementwise mode
            (batch, seq_len, nheads, headdim, headdim) for outer product mode
        alpha: (batch, seq_len, nheads) - decay rates in (0, 1)
        beta: (batch, seq_len, nheads) - input coefficients
        init_state: (batch, nheads, headdim) or (batch, nheads, headdim, headdim)
    
    Returns:
        states: same shape as kv - accumulated states at each position
    """
    state_fwd = forward_scan(kv, alpha, beta, init_state)
    state_bwd = backward_scan(kv, alpha, beta, init_state)
    return 0.5 * (state_fwd + state_bwd)
