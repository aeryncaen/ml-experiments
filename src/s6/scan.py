"""
Sequential scan implementations for USB.

These are reference implementations - will be optimized with parallel/associative
scans later (e.g., via Triton kernels).

Supports two modes:
- elementwise: kv = k * v, state is (nheads, headdim)
- outer: kv = k ⊗ v, state is (nheads, headdim, headdim)
"""

import torch


def forward_scan(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    epsilon: torch.Tensor,
    zeta: torch.Tensor,
    init_state: torch.Tensor,
) -> torch.Tensor:
    """
    Forward causal scan with Simpson-style discretization.
    
    Starts at t=0, moves forward. Integrates current and two previous positions.
    
    state_t = α[t] * state_{t-1}
            + δ[t] * kv_{t-2}
            + ε[t] * kv_{t-1}
            + ζ[t] * kv_t
    
    Args:
        kv: (batch, seq_len, nheads, headdim) for elementwise mode
            (batch, seq_len, nheads, headdim, headdim) for outer product mode
        alpha: (batch, seq_len, nheads) - decay rates in (0, 1)
        delta, epsilon, zeta: (batch, seq_len, nheads) - Simpson coefficients
        init_state: (batch, nheads, headdim) or (batch, nheads, headdim, headdim)
    
    Returns:
        states: same shape as kv - accumulated states at each position
    """
    outer_mode = kv.ndim == 5
    
    if outer_mode:
        batch, seq_len, nheads, headdim, _ = kv.shape
    else:
        batch, seq_len, nheads, headdim = kv.shape
    
    # Build Simpson conv term (kv_{t-2}, kv_{t-1}, kv_t)
    if outer_mode:
        coef_shape = (batch, seq_len, nheads, 1, 1)
    else:
        coef_shape = (batch, seq_len, nheads, 1)

    delta_e = delta.view(coef_shape)
    epsilon_e = epsilon.view(coef_shape)
    zeta_e = zeta.view(coef_shape)

    kv_tm1 = torch.cat([torch.zeros_like(kv[:, :1]), kv[:, :-1]], dim=1)
    kv_tm2 = torch.cat([torch.zeros_like(kv[:, :2]), kv[:, :-2]], dim=1)
    s = zeta_e * kv + epsilon_e * kv_tm1 + delta_e * kv_tm2

    # Mask-form scan (vectorized):
    # state_t = p_t * (init_state + sum_{j<=t} s_j / p_j), where p_t = prod_{i<=t} alpha_i
    # This is equivalent to the recurrence but avoids Python loops.
    alpha_safe = torch.clamp(alpha, min=1e-8)
    p = torch.cumprod(alpha_safe, dim=1)  # (b, t, h)

    if outer_mode:
        p_e = p[..., None, None]
    else:
        p_e = p[..., None]

    u = s / p_e
    u_cum = torch.cumsum(u, dim=1)
    state = p_e * u_cum

    # init_state contribution: prod(alpha_0..alpha_t) * init_state
    state = state + p_e * init_state.unsqueeze(1)
    return state


def backward_scan(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    epsilon: torch.Tensor,
    zeta: torch.Tensor,
    init_state: torch.Tensor,
) -> torch.Tensor:
    """
    Backward causal scan with Simpson-style discretization.
    
    Starts at t=T-1 (end), moves backward. Integrates current and two next positions.
    
    state_t = α[t] * state_{t+1}
            + δ[t] * kv_{t+2}
            + ε[t] * kv_{t+1}
            + ζ[t] * kv_t
    
    Args:
        kv: (batch, seq_len, nheads, headdim) for elementwise mode
            (batch, seq_len, nheads, headdim, headdim) for outer product mode
        alpha: (batch, seq_len, nheads) - decay rates in (0, 1)
        delta, epsilon, zeta: (batch, seq_len, nheads) - Simpson coefficients
        init_state: (batch, nheads, headdim) or (batch, nheads, headdim, headdim)
    
    Returns:
        states: same shape as kv - accumulated states at each position
    """
    outer_mode = kv.ndim == 5
    
    if outer_mode:
        batch, seq_len, nheads, headdim, _ = kv.shape
    else:
        batch, seq_len, nheads, headdim = kv.shape
    
    # Build Simpson conv term (kv_{t+2}, kv_{t+1}, kv_t)
    if outer_mode:
        coef_shape = (batch, seq_len, nheads, 1, 1)
    else:
        coef_shape = (batch, seq_len, nheads, 1)

    delta_e = delta.view(coef_shape)
    epsilon_e = epsilon.view(coef_shape)
    zeta_e = zeta.view(coef_shape)

    kv_tp1 = torch.cat([kv[:, 1:], torch.zeros_like(kv[:, :1])], dim=1)
    kv_tp2 = torch.cat([kv[:, 2:], torch.zeros_like(kv[:, :2])], dim=1)
    s = zeta_e * kv + epsilon_e * kv_tp1 + delta_e * kv_tp2

    # Reverse-time mask form, then flip back.
    alpha_rev = torch.flip(alpha, dims=[1])
    s_rev = torch.flip(s, dims=[1])

    alpha_safe = torch.clamp(alpha_rev, min=1e-8)
    p = torch.cumprod(alpha_safe, dim=1)  # (b, t, h)

    if outer_mode:
        p_e = p[..., None, None]
    else:
        p_e = p[..., None]

    u = s_rev / p_e
    u_cum = torch.cumsum(u, dim=1)
    state_rev = p_e * u_cum
    state_rev = state_rev + p_e * init_state.unsqueeze(1)

    return torch.flip(state_rev, dims=[1])


def centered_scan(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    delta: torch.Tensor,
    epsilon: torch.Tensor,
    zeta: torch.Tensor,
    init_state: torch.Tensor,
) -> torch.Tensor:
    """
    Centered scan using a symmetric Simpson-style approximation.

    Returns the average of forward and backward scans.
    
    Args:
        kv: (batch, seq_len, nheads, headdim) for elementwise mode
            (batch, seq_len, nheads, headdim, headdim) for outer product mode
        alpha: (batch, seq_len, nheads) - decay rates in (0, 1)
        delta, epsilon, zeta: (batch, seq_len, nheads) - Simpson coefficients
        init_state: (batch, nheads, headdim) or (batch, nheads, headdim, headdim)
    
    Returns:
        states: same shape as kv - accumulated states at each position
    """
    state_fwd = forward_scan(kv, alpha, delta, epsilon, zeta, init_state)
    state_bwd = backward_scan(kv, alpha, delta, epsilon, zeta, init_state)
    return 0.5 * (state_fwd + state_bwd)
