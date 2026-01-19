from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class SpanFinderConfig:
    """Configuration for SpanFinder model.
    
    Size presets from design doc:
    | Preset | width | cnn_depth | cnn_window | Approx Params |
    |--------|-------|-----------|------------|---------------|
    | Tiny   | 32    | 2         | 2          | ~30K          |
    | Small  | 64    | 3         | 3          | ~100K         |
    | Medium | 96    | 4         | 3          | ~300K         |
    """
    vocab_size: int = 256
    width: int = 64  # embed_dim and hidden_dim
    cnn_depth: int = 3
    cnn_window: int = 3

    def to_dict(self) -> dict:
        return {
            "vocab_size": self.vocab_size,
            "width": self.width,
            "cnn_depth": self.cnn_depth,
            "cnn_window": self.cnn_window,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SpanFinderConfig":
        # Handle legacy config format
        if "embed_dim" in d and "width" not in d:
            d = d.copy()
            d["width"] = d.pop("embed_dim")
        if "hidden_dim" in d:
            d = d.copy()
            d.pop("hidden_dim")
        if "num_layers" in d and "cnn_depth" not in d:
            d = d.copy()
            d["cnn_depth"] = d.pop("num_layers")
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def tiny(cls) -> "SpanFinderConfig":
        return cls(width=32, cnn_depth=2, cnn_window=2)

    @classmethod
    def small(cls) -> "SpanFinderConfig":
        return cls(width=64, cnn_depth=3, cnn_window=3)

    @classmethod
    def medium(cls) -> "SpanFinderConfig":
        return cls(width=96, cnn_depth=4, cnn_window=3)


class ResidualConvBlock(nn.Module):
    """1D Conv block with residual connection."""

    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        # Padding to preserve sequence length
        padding = kernel_size // 2
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, channels)
        # Conv1d expects (batch, channels, seq_len)
        residual = x
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        return x + residual


class SpanFinderModel(nn.Module):
    """Byte-level CNN for span boundary prediction.
    
    Architecture:
    1. Byte embedding (256 vocab -> width dim)
    2. CNN encoder with residual connections
    3. Linear classifier (width -> 2 outputs per position)
    
    Output: [start_prob, end_prob] per byte position
    """

    def __init__(self, config: SpanFinderConfig):
        super().__init__()
        self.config = config

        # Byte embedding
        self.embedding = nn.Embedding(config.vocab_size, config.width)

        # CNN encoder with residual blocks
        self.encoder = nn.ModuleList([
            ResidualConvBlock(config.width, config.cnn_window)
            for _ in range(config.cnn_depth)
        ])

        # Output classifier: 2 values per position (start_prob, end_prob)
        self.classifier = nn.Linear(config.width, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Byte sequence tensor, shape (batch, seq_len), values 0-255
            
        Returns:
            Logits tensor, shape (batch, seq_len, 2) - [start_logit, end_logit] per position
        """
        # Embed bytes
        x = self.embedding(x)  # (batch, seq_len, width)

        # Apply CNN encoder blocks
        for block in self.encoder:
            x = block(x)

        # Classify each position
        logits = self.classifier(x)  # (batch, seq_len, 2)
        return logits

    @property
    def receptive_field(self) -> int:
        """Total context window size in bytes."""
        return self.config.cnn_window * self.config.cnn_depth
