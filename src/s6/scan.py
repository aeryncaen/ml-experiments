"""Parallel associative scan for S6.

Binary operator: (a_i, b_i) ⊕ (a_j, b_j) = (a_j * a_i, a_j * b_i + b_j)
Where a = decay (complex diagonal), b = injection (complex).

Chunked scan: within each chunk, build a (K,K) decay matrix and use cuBLAS matmul
for the intra-chunk work. Carry state between chunks sequentially.
"""

import torch


def sequential_scan(alpha: torch.Tensor, inject: torch.Tensor) -> torch.Tensor:
    """Sequential scan fallback. O(L) with no parallelism.

    Args:
        alpha:  (B, L, P) — decay per position
        inject: (B, L, P) — injection per position

    Returns:
        h: (B, L, P) — all hidden states
    """
    B, L, P = alpha.shape
    h = torch.zeros(B, P, dtype=alpha.dtype, device=alpha.device)
    out = torch.empty_like(inject)
    for t in range(L):
        h = alpha[:, t] * h + inject[:, t]
        out[:, t] = h
    return out


def chunked_scan(alpha: torch.Tensor, inject: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
    """Chunked parallel scan. Intra-chunk via decay matrix matmul (cuBLAS), inter-chunk sequential.

    Args:
        alpha:  (B, L, P) — decay per position
        inject: (B, L, P) — injection per position
        chunk_size: K — positions per chunk

    Returns:
        h: (B, L, P) — all hidden states
    """
    B, L, P = alpha.shape
    device = alpha.device
    dtype = alpha.dtype
    out = torch.empty_like(inject)
    state = torch.zeros(B, P, dtype=dtype, device=device)

    for chunk_start in range(0, L, chunk_size):
        chunk_end = min(chunk_start + chunk_size, L)
        K = chunk_end - chunk_start

        a_chunk = alpha[:, chunk_start:chunk_end]   # (B, K, P)
        u_chunk = inject[:, chunk_start:chunk_end]   # (B, K, P)

        # Build (K, K) causal decay matrix per batch per state dim
        # decay[i,j] = prod_{t=j+1}^{i} alpha[t] for j < i, 1 for j == i, 0 for j > i
        # Use cumulative log-sum trick (works for complex alpha too)
        log_a = torch.log(a_chunk)  # (B, K, P)
        log_a_cumsum = torch.cumsum(log_a, dim=1)     # (B, K, P)

        # decay[i,j,p] = exp(cumsum[i,p] - cumsum[j,p]) for j <= i
        # (B, K, 1, P) - (B, 1, K, P) -> (B, K, K, P)
        log_decay = log_a_cumsum.unsqueeze(2) - log_a_cumsum.unsqueeze(1)  # (B, K, K, P)

        # Causal mask: zero out j > i
        causal = torch.tril(torch.ones(K, K, device=device, dtype=dtype))  # (K, K)
        decay = torch.exp(log_decay) * causal.unsqueeze(0).unsqueeze(-1)  # (B, K, K, P)

        # Intra-chunk: h_from_input[i] = sum_{j<=i} decay[i,j] * inject[j]
        # (B, K, K, P) @ (B, K, P) -> need einsum
        h_from_input = torch.einsum('bijk,bjk->bik', decay, u_chunk)  # (B, K, P)

        # Carry from previous chunk: decay each position's state
        # carry[i] = exp(cumsum[i]) * state = prod_{t=0}^{i} alpha[t] * state
        carry_decay = torch.exp(log_a_cumsum)  # (B, K, P)
        h_from_state = carry_decay * state.unsqueeze(1)  # (B, K, P)

        h_chunk = h_from_input + h_from_state  # (B, K, P)
        out[:, chunk_start:chunk_end] = h_chunk

        # Update state for next chunk
        state = h_chunk[:, -1]  # (B, P)

    return out


def parallel_scan(alpha: torch.Tensor, inject: torch.Tensor, chunk_size: int = 32) -> torch.Tensor:
    """Parallel scan. Currently uses sequential due to numerical issues in chunked version.
    
    The chunked scan uses log-space computation which overflows when alpha is very small
    (e.g., exp(-16) per step) and we compute cumulative products over many steps.
    TODO: Implement numerically stable chunked scan or use Triton kernel.
    """
    # Sequential is numerically stable, just slower
    return sequential_scan(alpha, inject)
