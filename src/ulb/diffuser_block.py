"""ULBDiffuserBlock — ULBBlock subclass for masked diffusion.

Processes a combined [input | output] sequence where:
- Input positions: full ULB attention (Q-peek, K-lerp, causal attention)
- Output positions: K-lerp + causal attention (no Q-peek), plus acausal
  local attention over the output segment for bidirectional coherence.

Set `block.input_len = N` before calling forward to specify the boundary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import ULBBlock, ULBConfig


class ULBDiffuserBlock(ULBBlock):
    """ULBBlock adapted for masked diffusion generation.

    Changes from parent:
    - Q-peek (acausal lerp) is skipped for output positions.
    - Acausal local windowed attention is run over the output segment.

    Set self.input_len before each forward call to specify the boundary
    between input (prompt) and output (generation) positions.

    Args:
        config: ULBConfig instance.
        local_window: Half-window size for acausal local attention on output.
                      Each output position attends to ±local_window neighbors.
    """

    def __init__(self, config: ULBConfig, local_window: int = 16):
        super().__init__(config)
        self.local_window = local_window
        self.input_len = 0  # set before forward

        # Separate QKV projections for local acausal attention on output
        inner = config.inner_dim
        self.local_q_proj = nn.Linear(inner, inner, bias=False)
        self.local_k_proj = nn.Linear(inner, inner, bias=False)
        self.local_v_proj = nn.Linear(inner, inner, bias=True)
        self.local_out_norm = nn.RMSNorm(config.head_dim)

    def preprocess_qk(self, q: torch.Tensor, k: torch.Tensor, x: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Apply K-lerp everywhere, Q-peek only on input positions.

        Output positions get no Q-peek — predicted tokens don't leak forward
        into other positions' queries.
        """
        n = self.input_len

        # K-lerp: applies to full sequence (causal, safe)
        if self.k_lerp is not None:
            k = self.k_lerp(k, x)

        # Q-peek: only on input positions
        if n > 0 and (self.q_lerp is not None or self.q_conv is not None):
            q_in = q[:, :n]
            x_in = x[:, :n]
            if self.q_lerp is not None:
                q_in = self.q_lerp(q_in, x_in)
            elif self.q_conv is not None:
                q_in = self.q_conv(q_in)
            q = torch.cat([q_in, q[:, n:]], dim=1)

        dd_angles = self.rope.compute_dd_angles(x)

        blend_gate = None
        if self.config.attn_mode == 'blend':
            assert self.blend_attn is not None
            blend_gate = self.blend_attn.compute_gate(x)

        return q, k, dd_angles, blend_gate

    def _local_acausal_attention(self, h_up: torch.Tensor) -> torch.Tensor:
        """Run acausal local windowed attention over the output segment.

        Args:
            h_up: (B, T, inner) — up-projected hidden state for full sequence.

        Returns:
            (B, T_out, inner) — local attention output for output positions only.
        """
        n = self.input_len
        cfg = self.config
        h_out = h_up[:, n:]  # (B, T_out, inner)
        b, t_out, _ = h_out.shape

        if t_out == 0:
            return h_out

        head_dim = cfg.head_dim
        n_heads = cfg.n_heads

        q = self.local_q_proj(h_out).view(b, t_out, n_heads, head_dim)
        k = self.local_k_proj(h_out).view(b, t_out, n_heads, head_dim)
        v = self.local_v_proj(h_out).view(b, t_out, n_heads, head_dim)

        # Transpose to (B, H, T_out, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Build local window mask: position i attends to [i-w, i+w]
        w = self.local_window
        idx = torch.arange(t_out, device=q.device)
        # (T_out, T_out) boolean mask — True means attend
        mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= w
        # Convert to float mask for SDPA: 0 = attend, -inf = don't attend
        attn_mask = torch.where(mask, 0.0, float('-inf'))

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        y = y.transpose(1, 2)  # (B, T_out, H, D)
        y = self.local_out_norm(y)
        return y.contiguous().view(b, t_out, cfg.inner_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with acausal local attention on output segment.

        Args:
            x: (B, T, D) — pre-normed input, combined [input | output].

        Returns:
            (B, T, D) — delta for residual stream.
        """
        cfg = self.config
        b, t, d = x.shape
        n = self.input_len
        n_heads = cfg.n_heads
        head_dim = cfg.head_dim

        # --- Up projection ---
        h_up = self.up_act(self.up_proj(x))

        # --- QKV projections + norm + bias ---
        q = self.q_proj(h_up).view(b, t, n_heads, head_dim)
        k = self.k_proj(h_up).view(b, t, n_heads, head_dim)
        v = self.v_proj(h_up).view(b, t, n_heads, head_dim)
        q = self.q_norm(q) * self.q_bias
        k = self.k_norm(k) * self.k_bias

        # --- Preprocessing (Q-peek only on input) ---
        q, k, dd_angles, blend_gate = self.preprocess_qk(q, k, x)

        # --- Causal attention over full sequence ---
        y = self.attend(q, k, v, dd_angles, blend_gate)
        y = y.contiguous().view(b, t, cfg.inner_dim)

        # --- Acausal local attention on output segment ---
        if n < t:
            local_out = self._local_acausal_attention(h_up)
            y = torch.cat([y[:, :n], y[:, n:] + local_out], dim=1)

        # --- Skip-multiply (NOT skip-add) ---
        y = self.attn_norm(y) * h_up

        # --- Down projection ---
        y = self.down_proj(self.down_act(y))

        return y
