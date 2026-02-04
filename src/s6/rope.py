"""
Rotary Position Embedding (RoPE) implementations.

Includes both standard RoPE and data-dependent RoPE for USB.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _compute_rope_freqs(
    headdim: int,
    seq_len: int,
    base: float = 10000.0,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute standard RoPE frequencies.
    
    Returns:
        cos, sin: (seq_len, headdim) each
    """
    # Frequency bands: theta_i = base^(-2i/d) for i = 0, 1, ..., d/2-1
    inv_freq = 1.0 / (base ** (torch.arange(0, headdim, 2, device=device, dtype=torch.float32) / headdim))
    
    # Position indices
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    
    # Outer product: (seq_len,) x (headdim/2,) -> (seq_len, headdim/2)
    freqs = torch.outer(t, inv_freq)
    
    # Duplicate for full headdim: (seq_len, headdim)
    freqs = torch.cat([freqs, freqs], dim=-1)
    
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    
    return cos, sin


# Cache for standard RoPE frequencies
_rope_cache = {}


def apply_rope(
    x: torch.Tensor,
    seq_len: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """
    Apply standard Rotary Position Embedding.
    
    Args:
        x: (batch, seq_len, nheads, headdim)
        seq_len: sequence length
        base: RoPE base frequency
    
    Returns:
        x with RoPE applied: (batch, seq_len, nheads, headdim)
    """
    headdim = x.shape[-1]
    device = x.device
    dtype = x.dtype
    
    # Check cache
    cache_key = (headdim, seq_len, base, device, dtype)
    if cache_key not in _rope_cache:
        cos, sin = _compute_rope_freqs(headdim, seq_len, base, device, dtype)
        _rope_cache[cache_key] = (cos, sin)
    
    cos, sin = _rope_cache[cache_key]
    
    # Reshape for broadcasting: (1, seq_len, 1, headdim)
    cos = cos.view(1, seq_len, 1, headdim)
    sin = sin.view(1, seq_len, 1, headdim)
    
    # Apply rotation
    return x * cos + _rotate_half(x) * sin


def apply_data_dependent_rope(
    x: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Apply data-dependent Rotary Position Embedding.
    
    Instead of fixed frequencies based on position, frequencies are learned
    from the input content per head.
    
    Args:
        x: (batch, seq_len, nheads, headdim)
        freqs: (batch, seq_len, nheads, headdim // 2) - learned rotation frequencies per head
    
    Returns:
        x with data-dependent RoPE applied: (batch, seq_len, nheads, headdim)
    """
    batch, seq_len, nheads, headdim = x.shape
    
    # Cumulative sum of frequencies to get rotation angles
    # This creates data-dependent relative positions
    # Shape: (batch, seq_len, nheads, headdim // 2)
    angles = torch.cumsum(freqs, dim=1)
    
    # Duplicate for full headdim: (batch, seq_len, nheads, headdim)
    angles = torch.cat([angles, angles], dim=-1)
    
    cos = angles.cos()
    sin = angles.sin()
    
    # Apply rotation
    return x * cos + _rotate_half(x) * sin


class DataDependentRoPE(nn.Module):
    """
    Module wrapper for data-dependent RoPE.
    
    Each head learns its own frequency projection, allowing different heads
    to learn different positional geometries.
    """
    
    def __init__(self, d_input: int, nheads: int, headdim: int):
        super().__init__()
        self.nheads = nheads
        self.headdim = headdim
        
        # Project input to rotation frequencies
        # Output: headdim // 2 frequencies per head
        self.freq_proj = nn.Linear(d_input, nheads * (headdim // 2), bias=False)
        
        # Initialize to produce small frequencies initially
        nn.init.normal_(self.freq_proj.weight, std=0.01)
    
    def forward(self, x: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, nheads, headdim) - tensor to apply RoPE to
            hidden: (batch, seq_len, d_input) - input to project frequencies from
        
        Returns:
            x with data-dependent RoPE applied
        """
        batch, seq_len, _ = hidden.shape
        
        # Get frequencies from input
        freqs = self.freq_proj(hidden)  # (batch, seq_len, nheads * headdim // 2)
        freqs = freqs.view(batch, seq_len, self.nheads, self.headdim // 2)
        
        return apply_data_dependent_rope(x, freqs)
