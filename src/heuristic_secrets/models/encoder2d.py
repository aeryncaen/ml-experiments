import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4 and x.shape[1] == self.weight.shape[0]:
            rms = torch.sqrt(torch.mean(x * x, dim=1, keepdim=True) + self.eps)
            return x / rms * self.weight.view(1, -1, 1, 1)
        else:
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight


def get_2d_sinusoidal_pos_embed(
    embed_dim: int, height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    half_dim = embed_dim // 2
    quarter_dim = half_dim // 2

    pos_h = torch.arange(height, device=device, dtype=dtype)
    pos_w = torch.arange(width, device=device, dtype=dtype)

    omega = torch.arange(quarter_dim, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / quarter_dim))

    h_emb = pos_h.unsqueeze(1) * omega.unsqueeze(0)
    h_emb = torch.cat([h_emb.sin(), h_emb.cos()], dim=1)

    w_emb = pos_w.unsqueeze(1) * omega.unsqueeze(0)
    w_emb = torch.cat([w_emb.sin(), w_emb.cos()], dim=1)

    h_emb = h_emb.unsqueeze(1).expand(-1, width, -1)
    w_emb = w_emb.unsqueeze(0).expand(height, -1, -1)

    return torch.cat([h_emb, w_emb], dim=-1)


class PixelEmbedding2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        dropout: float = 0.1,
        pos_base: int = 10000,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.pos_base = pos_base

        self.proj = nn.Conv2d(in_channels, width, kernel_size=1, bias=True)
        self.norm = RMSNorm2d(width)
        self.dropout = nn.Dropout(dropout)

        inv_freq_h = 1.0 / (
            pos_base ** (torch.arange(0, width // 2, 2).float() / (width // 2))
        )
        inv_freq_w = 1.0 / (
            pos_base ** (torch.arange(0, width // 2, 2).float() / (width // 2))
        )
        self.register_buffer("inv_freq_h", inv_freq_h)
        self.register_buffer("inv_freq_w", inv_freq_w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape

        emb = self.proj(x)
        emb = emb.permute(0, 2, 3, 1)
        emb = self._apply_2d_rope(emb, H, W)
        emb = self.norm(emb)
        emb = F.silu(emb)
        emb = self.dropout(emb)

        return emb

    def _apply_2d_rope(self, emb: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, _, _, C = emb.shape
        device = emb.device
        dtype = emb.dtype

        half_c = C // 2
        emb_h = emb[..., :half_c]
        emb_w = emb[..., half_c:]

        pos_h = torch.arange(H, device=device, dtype=dtype)
        inv_freq_h = self.inv_freq_h.to(dtype)
        angles_h = torch.outer(pos_h, inv_freq_h)
        cos_h, sin_h = angles_h.cos(), angles_h.sin()
        cos_h = cos_h.view(1, H, 1, -1)
        sin_h = sin_h.view(1, H, 1, -1)

        pos_w = torch.arange(W, device=device, dtype=dtype)
        inv_freq_w = self.inv_freq_w.to(dtype)
        angles_w = torch.outer(pos_w, inv_freq_w)
        cos_w, sin_w = angles_w.cos(), angles_w.sin()
        cos_w = cos_w.view(1, 1, W, -1)
        sin_w = sin_w.view(1, 1, W, -1)

        h1, h2 = emb_h[..., 0::2], emb_h[..., 1::2]
        rotated_h1 = h1 * cos_h - h2 * sin_h
        rotated_h2 = h1 * sin_h + h2 * cos_h
        emb_h = torch.stack([rotated_h1, rotated_h2], dim=-1).flatten(-2)

        w1, w2 = emb_w[..., 0::2], emb_w[..., 1::2]
        rotated_w1 = w1 * cos_w - w2 * sin_w
        rotated_w2 = w1 * sin_w + w2 * cos_w
        emb_w = torch.stack([rotated_w1, rotated_w2], dim=-1).flatten(-2)

        return torch.cat([emb_h, emb_w], dim=-1)


class PixelEmbedding2dLearned(nn.Module):
    def __init__(
        self,
        in_channels: int,
        width: int,
        max_height: int = 256,
        max_width: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.max_height = max_height
        self.max_width = max_width

        self.proj = nn.Conv2d(in_channels, width, kernel_size=1, bias=True)
        self.norm = RMSNorm2d(width)
        self.dropout = nn.Dropout(dropout)

        self.pos_embed_h = nn.Parameter(torch.randn(max_height, width // 2) * 0.02)
        self.pos_embed_w = nn.Parameter(torch.randn(max_width, width // 2) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        assert H <= self.max_height and W <= self.max_width, (
            f"Input size ({H}, {W}) exceeds max ({self.max_height}, {self.max_width})"
        )

        emb = self.proj(x)
        emb = emb.permute(0, 2, 3, 1)

        pos_h = self.pos_embed_h[:H].unsqueeze(1).expand(-1, W, -1)
        pos_w = self.pos_embed_w[:W].unsqueeze(0).expand(H, -1, -1)
        pos = torch.cat([pos_h, pos_w], dim=-1)
        emb = emb + pos.unsqueeze(0)

        emb = self.norm(emb)
        emb = F.silu(emb)
        emb = self.dropout(emb)

        return emb

    def _apply_2d_rope(self, emb: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Apply 2D RoPE-style rotary positional encoding."""
        B, _, _, C = emb.shape
        device = emb.device
        dtype = emb.dtype

        # Split embedding into height and width portions
        half_c = C // 2
        emb_h = emb[..., :half_c]  # For height encoding
        emb_w = emb[..., half_c:]  # For width encoding

        # Height positions and angles
        pos_h = torch.arange(H, device=device, dtype=dtype)
        inv_freq_h = self.inv_freq_h.to(dtype)
        angles_h = torch.outer(pos_h, inv_freq_h)  # (H, half_c // 2)
        cos_h, sin_h = angles_h.cos(), angles_h.sin()
        cos_h = cos_h.view(1, H, 1, -1)  # (1, H, 1, half_c // 2)
        sin_h = sin_h.view(1, H, 1, -1)

        # Width positions and angles
        pos_w = torch.arange(W, device=device, dtype=dtype)
        inv_freq_w = self.inv_freq_w.to(dtype)
        angles_w = torch.outer(pos_w, inv_freq_w)  # (W, half_c // 2)
        cos_w, sin_w = angles_w.cos(), angles_w.sin()
        cos_w = cos_w.view(1, 1, W, -1)  # (1, 1, W, half_c // 2)
        sin_w = sin_w.view(1, 1, W, -1)

        # Apply rotation to height portion
        h1, h2 = emb_h[..., 0::2], emb_h[..., 1::2]
        rotated_h1 = h1 * cos_h - h2 * sin_h
        rotated_h2 = h1 * sin_h + h2 * cos_h
        emb_h = torch.stack([rotated_h1, rotated_h2], dim=-1).flatten(-2)

        # Apply rotation to width portion
        w1, w2 = emb_w[..., 0::2], emb_w[..., 1::2]
        rotated_w1 = w1 * cos_w - w2 * sin_w
        rotated_w2 = w1 * sin_w + w2 * cos_w
        emb_w = torch.stack([rotated_w1, rotated_w2], dim=-1).flatten(-2)

        return torch.cat([emb_h, emb_w], dim=-1)


class PixelEmbedding2dLearned(nn.Module):
    """
    Alternative embedding with learnable 2D positional embeddings.
    Useful when input size is fixed.
    """

    def __init__(
        self,
        in_channels: int,
        width: int,
        max_height: int = 256,
        max_width: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.max_height = max_height
        self.max_width = max_width

        self.proj = nn.Conv2d(in_channels, width, kernel_size=1, bias=True)
        self.norm = RMSNorm2d(width)
        self.dropout = nn.Dropout(dropout)

        # Learnable position embeddings
        self.pos_embed_h = nn.Parameter(torch.randn(max_height, width // 2) * 0.02)
        self.pos_embed_w = nn.Parameter(torch.randn(max_width, width // 2) * 0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        assert H <= self.max_height and W <= self.max_width, (
            f"Input size ({H}, {W}) exceeds max ({self.max_height}, {self.max_width})"
        )

        emb = self.proj(x)
        emb = emb.permute(0, 2, 3, 1)  # (B, H, W, C)

        # Add learnable positional embeddings
        pos_h = self.pos_embed_h[:H].unsqueeze(1).expand(-1, W, -1)  # (H, W, C//2)
        pos_w = self.pos_embed_w[:W].unsqueeze(0).expand(H, -1, -1)  # (H, W, C//2)
        pos = torch.cat([pos_h, pos_w], dim=-1)  # (H, W, C)
        emb = emb + pos.unsqueeze(0)

        emb = self.norm(emb)
        emb = F.silu(emb)
        emb = self.dropout(emb)

        return emb
