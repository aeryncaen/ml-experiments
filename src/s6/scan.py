"""
Sequential scan implementations for USB.

These are reference implementations - will be optimized with parallel/associative
scans later (e.g., via Triton kernels).
"""

import torch
from typing import Optional, List


def forward_scan(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    init_state: torch.Tensor,
) -> torch.Tensor:
    """
    Forward causal scan with AB-2 integration.
    
    Starts at t=0, moves forward. Integrates current position and two steps back.
    
    state_t = c0[t] * kv_t
            + c1[t] * (α[t]   * state_{t-1} + kv_{t-1})
            + c2[t] * (α[t]²  * state_{t-2} + kv_{t-2})
    
    Args:
        kv: (batch, seq_len, nheads, headdim) - K·V product
        alpha: (batch, seq_len, nheads) - decay rates in (0, 1)
        c0, c1, c2: (batch, seq_len, nheads) - AB-2 coefficients
        init_state: (batch, nheads, headdim) - learned initial state
    
    Returns:
        states: (batch, seq_len, nheads, headdim) - accumulated states at each position
    """
    batch, seq_len, nheads, headdim = kv.shape
    device = kv.device
    dtype = kv.dtype
    
    # Collect states in a list to avoid inplace ops
    state_list: List[torch.Tensor] = []
    
    # State history (need t-1 and t-2)
    state_tm1 = init_state  # (batch, nheads, headdim)
    state_tm2 = torch.zeros_like(init_state)
    
    # KV history
    kv_tm1 = torch.zeros(batch, nheads, headdim, device=device, dtype=dtype)
    kv_tm2 = torch.zeros(batch, nheads, headdim, device=device, dtype=dtype)
    
    for t in range(seq_len):
        kv_t = kv[:, t]  # (batch, nheads, headdim)
        alpha_t = alpha[:, t, :, None]  # (batch, nheads, 1)
        c0_t = c0[:, t, :, None]  # (batch, nheads, 1)
        c1_t = c1[:, t, :, None]
        c2_t = c2[:, t, :, None]
        
        # AB-2 integration
        state_t = (
            c0_t * kv_t
            + c1_t * (alpha_t * state_tm1 + kv_tm1)
            + c2_t * (alpha_t.pow(2) * state_tm2 + kv_tm2)
        )
        
        state_list.append(state_t)
        
        # Shift history
        state_tm2 = state_tm1
        state_tm1 = state_t
        kv_tm2 = kv_tm1
        kv_tm1 = kv_t
    
    # Stack along seq dimension
    return torch.stack(state_list, dim=1)


def backward_scan(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    init_state: torch.Tensor,
) -> torch.Tensor:
    """
    Backward causal scan with AB-2 integration.
    
    Starts at t=T-1 (end), moves backward. Integrates current position and two steps forward.
    
    state_t = c0[t] * kv_t
            + c1[t] * (α[t]   * state_{t+1} + kv_{t+1})
            + c2[t] * (α[t]²  * state_{t+2} + kv_{t+2})
    
    Args:
        kv: (batch, seq_len, nheads, headdim) - K·V product
        alpha: (batch, seq_len, nheads) - decay rates in (0, 1)
        c0, c1, c2: (batch, seq_len, nheads) - AB-2 coefficients
        init_state: (batch, nheads, headdim) - learned initial state
    
    Returns:
        states: (batch, seq_len, nheads, headdim) - accumulated states at each position
    """
    batch, seq_len, nheads, headdim = kv.shape
    device = kv.device
    dtype = kv.dtype
    
    # Collect states in a list (will be reversed)
    state_list: List[torch.Tensor] = []
    
    # State history (need t+1 and t+2)
    state_tp1 = init_state  # (batch, nheads, headdim)
    state_tp2 = torch.zeros_like(init_state)
    
    # KV history
    kv_tp1 = torch.zeros(batch, nheads, headdim, device=device, dtype=dtype)
    kv_tp2 = torch.zeros(batch, nheads, headdim, device=device, dtype=dtype)
    
    for t in range(seq_len - 1, -1, -1):
        kv_t = kv[:, t]  # (batch, nheads, headdim)
        alpha_t = alpha[:, t, :, None]  # (batch, nheads, 1)
        c0_t = c0[:, t, :, None]  # (batch, nheads, 1)
        c1_t = c1[:, t, :, None]
        c2_t = c2[:, t, :, None]
        
        # AB-2 integration (backward direction)
        state_t = (
            c0_t * kv_t
            + c1_t * (alpha_t * state_tp1 + kv_tp1)
            + c2_t * (alpha_t.pow(2) * state_tp2 + kv_tp2)
        )
        
        state_list.append(state_t)
        
        # Shift history
        state_tp2 = state_tp1
        state_tp1 = state_t
        kv_tp2 = kv_tp1
        kv_tp1 = kv_t
    
    # Reverse and stack (we collected in reverse order)
    state_list.reverse()
    return torch.stack(state_list, dim=1)


def centered_scan(
    kv: torch.Tensor,
    alpha: torch.Tensor,
    c0: torch.Tensor,
    c1: torch.Tensor,
    c2: torch.Tensor,
    init_state: torch.Tensor,
) -> torch.Tensor:
    """
    Centered scan with AB-2 integration.
    
    Starts at midpoint, propagates outward in both directions simultaneously.
    Integrates current position, one step back, and one step forward.
    
    state_t = c0[t] * kv_t
            + c1[t] * (α[t] * state_{t-1} + kv_{t-1})
            + c2[t] * (α[t] * state_{t+1} + kv_{t+1})
    
    Note: Both directions use α[t] (not α[t]²) since it's one step each way.
    
    Args:
        kv: (batch, seq_len, nheads, headdim) - K·V product
        alpha: (batch, seq_len, nheads) - decay rates in (0, 1)
        c0, c1, c2: (batch, seq_len, nheads) - AB-2 coefficients
        init_state: (batch, nheads, headdim) - learned initial state
    
    Returns:
        states: (batch, seq_len, nheads, headdim) - accumulated states at each position
    """
    batch, seq_len, nheads, headdim = kv.shape
    device = kv.device
    dtype = kv.dtype
    
    # Use a dict to store states by position (avoids inplace ops)
    state_dict = {}
    
    # Start from the middle
    mid = seq_len // 2
    
    # Initialize the center position
    kv_mid = kv[:, mid]
    c0_mid = c0[:, mid, :, None]
    state_dict[mid] = c0_mid * kv_mid
    
    # Expand outward from center
    # We process positions in order of distance from center
    for dist in range(1, max(mid + 1, seq_len - mid)):
        # Left side: mid - dist
        left_idx = mid - dist
        if left_idx >= 0:
            kv_t = kv[:, left_idx]
            alpha_t = alpha[:, left_idx, :, None]
            c0_t = c0[:, left_idx, :, None]
            c1_t = c1[:, left_idx, :, None]
            c2_t = c2[:, left_idx, :, None]
            
            # Get neighbors (inward = toward center)
            # t-1 is further from center (may not exist yet or be at boundary)
            # t+1 is toward center (already computed)
            kv_tm1 = kv[:, left_idx - 1] if left_idx > 0 else torch.zeros_like(kv_t)
            kv_tp1 = kv[:, left_idx + 1]
            
            state_tm1 = state_dict.get(left_idx - 1, init_state)
            state_tp1 = state_dict[left_idx + 1]  # Already computed (closer to center)
            
            state_t = (
                c0_t * kv_t
                + c1_t * (alpha_t * state_tm1 + kv_tm1)
                + c2_t * (alpha_t * state_tp1 + kv_tp1)
            )
            state_dict[left_idx] = state_t
        
        # Right side: mid + dist
        right_idx = mid + dist
        if right_idx < seq_len:
            kv_t = kv[:, right_idx]
            alpha_t = alpha[:, right_idx, :, None]
            c0_t = c0[:, right_idx, :, None]
            c1_t = c1[:, right_idx, :, None]
            c2_t = c2[:, right_idx, :, None]
            
            # Get neighbors (inward = toward center)
            # t+1 is further from center (may not exist yet or be at boundary)
            # t-1 is toward center (already computed)
            kv_tp1 = kv[:, right_idx + 1] if right_idx < seq_len - 1 else torch.zeros_like(kv_t)
            kv_tm1 = kv[:, right_idx - 1]
            
            state_tp1 = state_dict.get(right_idx + 1, init_state)
            state_tm1 = state_dict[right_idx - 1]  # Already computed (closer to center)
            
            state_t = (
                c0_t * kv_t
                + c1_t * (alpha_t * state_tm1 + kv_tm1)
                + c2_t * (alpha_t * state_tp1 + kv_tp1)
            )
            state_dict[right_idx] = state_t
    
    # Collect in order and stack
    state_list = [state_dict[i] for i in range(seq_len)]
    return torch.stack(state_list, dim=1)
