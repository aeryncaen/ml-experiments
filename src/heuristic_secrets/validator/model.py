from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device)
        inv_freq: torch.Tensor = self.inv_freq  # type: ignore[assignment]
        freqs = torch.outer(positions.float(), inv_freq.to(device))
        return freqs.cos(), freqs.sin()


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class LightweightAttentionBlock(nn.Module):
    def __init__(self, width: int, num_heads: int, ffn_mult: int, dropout: float = 0.1):
        super().__init__()
        assert width % num_heads == 0
        self.width = width
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(width, width * ffn_mult),
            nn.GELU(),
            nn.Linear(width * ffn_mult, width),
        )
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = x.shape
        device = x.device

        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        cos, sin = self.rope(L, device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, self.width)
        x = x + self.out(out)
        x = x + self.ffn(x)
        return x


@dataclass
class ValidatorConfig:
    width: int = 32
    depth: int = 2
    num_heads: int = 2
    ffn_mult: int = 2
    num_features: int = 6
    mlp_dims: tuple[int, ...] = (32, 16)
    dropout: float = 0.1

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["mlp_dims"] = list(d["mlp_dims"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ValidatorConfig":
        if "mlp_dims" in d:
            d = d.copy()
            d["mlp_dims"] = tuple(d["mlp_dims"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class ValidatorModel(nn.Module):
    def __init__(self, config: ValidatorConfig, embedding: nn.Embedding | None = None):
        super().__init__()
        self.config = config

        self.embed = embedding if embedding is not None else nn.Embedding(256, config.width)
        
        self.blocks = nn.ModuleList([
            LightweightAttentionBlock(config.width, config.num_heads, config.ffn_mult, config.dropout)
            for _ in range(config.depth)
        ])

        self.feature_query = nn.Parameter(torch.randn(config.width))
        self.feature_out = nn.Linear(config.width, 1)
        
        self.gate_query = nn.Parameter(torch.randn(config.width))
        self.feature_gate = nn.Linear(config.width, config.num_features)
        self.gate_bias = nn.Parameter(torch.zeros(config.num_features))

        mlp_input_dim = config.num_features + 1
        mlp_layers: list[nn.Module] = []
        in_dim = mlp_input_dim
        for out_dim in config.mlp_dims:
            mlp_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            ])
            in_dim = out_dim
        mlp_layers.append(nn.Linear(in_dim, 1))
        self.classifier = nn.Sequential(*mlp_layers)

    def forward(
        self,
        features: torch.Tensor,
        byte_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, L = byte_ids.shape
        device = byte_ids.device

        mask = None
        if lengths is not None:
            mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)

        x = self.embed(byte_ids)
        for block in self.blocks:
            x = block(x, mask)

        width = self.config.width
        scale = width ** -0.5
        
        feat_q = self.feature_query.unsqueeze(0).expand(B, -1)
        feat_attn = torch.bmm(feat_q.unsqueeze(1), x.transpose(1, 2)) * scale
        if mask is not None:
            feat_attn = feat_attn.masked_fill(mask.unsqueeze(1), float('-inf'))
        feat_attn = F.softmax(feat_attn, dim=-1)
        feat_ctx = torch.bmm(feat_attn, x).squeeze(1)
        attn_feature = self.feature_out(feat_ctx)

        gate_q = self.gate_query.unsqueeze(0).expand(B, -1)
        gate_attn = torch.bmm(gate_q.unsqueeze(1), x.transpose(1, 2)) * scale
        if mask is not None:
            gate_attn = gate_attn.masked_fill(mask.unsqueeze(1), float('-inf'))
        gate_attn = F.softmax(gate_attn, dim=-1)
        gate_ctx = torch.bmm(gate_attn, x).squeeze(1)

        gate_logits = self.feature_gate(gate_ctx) + self.gate_bias
        gates = torch.sigmoid(gate_logits)
        gated_features = features * gates

        combined = torch.cat([gated_features, attn_feature], dim=-1)
        return self.classifier(combined)
