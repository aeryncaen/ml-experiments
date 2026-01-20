from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder2d import PixelEmbedding2d
from .backbone2d import (
    ContextualAttentionBlock2d,
    BiasProjection2d,
    SSMBlock3_2d,
    RMSNorm,
    MultiKernelSSMBlock2d,
    AdaptiveConvBiases2d,
    SwiGLU,
)
from .heads import ClassifierHead


@dataclass
class LayerConfig2d:
    embed_width: int = 32
    in_channels: int = 3
    dropout: float = 0.1

    conv_groups: int = 2

    use_attention: bool = True

    attn_heads: int = 2
    attn_ffn_mult: int = 4
    attn_window_size: int = 64
    attn_use_rope: bool = True
    num_attn_features: int = 4

    ssm_state_size: int = 64
    ssm_n_heads: int = 4
    ssm_kernel_sizes: tuple[int, ...] = (3, 5, 7, 9)
    ssm_expand: int = 2
    num_ssm_features: int = 4

    adaptive_conv: bool = False
    n_adaptive_branches: int = 3
    adaptive_kernel_size: int = 15
    adaptive_init_sigmas: tuple[float, ...] | None = None
    adaptive_min_sigma: float = 0.05
    adaptive_max_sigma: float = 0.5

    context_dim: int = 16
    num_embed_features: int = 4
    mlp_hidden_mult: int = 4
    mlp_output_dim: int = 16

    num_hidden_features: int = 4

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items()}
        d["ssm_kernel_sizes"] = list(d["ssm_kernel_sizes"])
        if d["adaptive_init_sigmas"] is not None:
            d["adaptive_init_sigmas"] = list(d["adaptive_init_sigmas"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LayerConfig2d":
        d = d.copy()
        if "ssm_kernel_sizes" in d:
            d["ssm_kernel_sizes"] = tuple(d["ssm_kernel_sizes"])
        if "adaptive_init_sigmas" in d and d["adaptive_init_sigmas"] is not None:
            d["adaptive_init_sigmas"] = tuple(d["adaptive_init_sigmas"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class HeadConfig2d:
    head_type: Literal["classifier"] = "classifier"
    n_classes: int = 10

    def to_dict(self) -> dict:
        return {"head_type": self.head_type, "n_classes": self.n_classes}

    @classmethod
    def from_dict(cls, d: dict) -> "HeadConfig2d":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig2d:
    n_layers: int = 1
    layer: LayerConfig2d | None = None
    head: HeadConfig2d | None = None

    def __post_init__(self):
        if self.layer is None:
            self.layer = LayerConfig2d()
        if self.head is None:
            self.head = HeadConfig2d()

    def to_dict(self) -> dict:
        return {
            "n_layers": self.n_layers,
            "layer": self.layer.to_dict() if self.layer else {},
            "head": self.head.to_dict() if self.head else {},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig2d":
        d = d.copy()
        layer_d = d.pop("layer", {})
        head_d = d.pop("head", {})
        layer = LayerConfig2d.from_dict(layer_d)
        head = HeadConfig2d.from_dict(head_d)
        return cls(
            layer=layer,
            head=head,
            **{k: v for k, v in d.items() if k in cls.__dataclass_fields__},
        )


class LearnedPooler2d(nn.Module):
    def __init__(self, width: int, num_queries: int = 1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, width))
        self.scale = width**-0.5

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        B, H, W, C = h.shape
        Q = self.queries.shape[0]

        h_flat = h.reshape(B, H * W, C)
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        attn = torch.bmm(q, h_flat.transpose(1, 2)) * self.scale
        if mask is not None:
            mask_flat = mask.reshape(B, H * W)
            attn = attn.masked_fill(mask_flat.unsqueeze(1), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        pooled = torch.bmm(attn, h_flat)
        return pooled.reshape(B, Q * C)


class EmbedPooler2d(nn.Module):
    def __init__(self, width: int, num_features: int):
        super().__init__()
        self.pooler = LearnedPooler2d(width, num_queries=1)
        self.proj = nn.Sequential(nn.Linear(width, num_features), nn.SiLU())

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        pooled = self.pooler(h, mask)
        return self.proj(pooled)


class FeatureIntegration2d(nn.Module):
    def __init__(self, width: int, num_features: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(num_features, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.norm = RMSNorm(width)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, h: torch.Tensor, features: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        feat_proj = self.proj(features).unsqueeze(1).unsqueeze(1)
        return self.norm(h + self.dropout(feat_proj))


class UnifiedLayer2d(nn.Module):
    def __init__(self, config: LayerConfig2d):
        super().__init__()
        self.config = config
        w = config.embed_width

        self.uses_adaptive_conv = config.adaptive_conv
        self.use_attention = config.use_attention

        init_sigmas = config.adaptive_init_sigmas
        if config.adaptive_conv and init_sigmas is None:
            n = config.n_adaptive_branches
            init_sigmas = tuple(
                config.adaptive_min_sigma
                + i
                * (config.adaptive_max_sigma - config.adaptive_min_sigma)
                / max(1, n - 1)
                for i in range(n)
            )

        ctx_dim = config.context_dim if config.adaptive_conv else 0
        n_branches = len(init_sigmas) if init_sigmas else len(config.ssm_kernel_sizes)

        if config.use_attention:
            self.attn_block = ContextualAttentionBlock2d(
                w,
                config.attn_heads,
                n_branches=n_branches if config.adaptive_conv else 0,
                context_dim=ctx_dim,
                ffn_mult=config.attn_ffn_mult,
                dropout=config.dropout,
                window_size=config.attn_window_size,
                use_rope=config.attn_use_rope,
            )

            if not self.uses_adaptive_conv:
                self.attn_pooler = LearnedPooler2d(w, num_queries=1)

            self.attn_proj = nn.Sequential(
                nn.Linear(w, config.num_attn_features), nn.SiLU()
            )
        else:
            if config.adaptive_conv:
                self.bias_proj = BiasProjection2d(w, n_branches, ctx_dim)
            self.input_ffn = SwiGLU(
                w, mult=config.attn_ffn_mult, dropout=config.dropout
            )

        self.multi_kernel_ssm = True
        self.ssm_block = MultiKernelSSMBlock2d(
            w,
            config.ssm_kernel_sizes,
            config.ssm_state_size,
            config.ssm_n_heads,
            config.ssm_expand,
            config.dropout,
            config.conv_groups,
            config.num_ssm_features,
            adaptive_conv=config.adaptive_conv,
            context_dim=ctx_dim,
            adaptive_kernel_size=config.adaptive_kernel_size,
            init_sigmas=init_sigmas,
            min_sigma=config.adaptive_min_sigma,
            max_sigma=config.adaptive_max_sigma,
            attn_window_size=config.attn_window_size,
            attn_use_rope=config.attn_use_rope,
            use_merge_attention=config.use_attention,
        )

        self.embed_pooler = EmbedPooler2d(w, config.num_embed_features)

        self.hidden_pooler = LearnedPooler2d(w, num_queries=1)
        self.hidden_proj = nn.Sequential(
            nn.Linear(w, config.num_hidden_features), nn.SiLU()
        )

        if config.use_attention:
            self.total_features = (
                config.num_attn_features
                + config.num_ssm_features
                + config.num_embed_features
                + config.num_hidden_features
            )
        else:
            self.total_features = (
                config.num_ssm_features
                + config.num_embed_features
                + config.num_hidden_features
            )
        self.feature_integration = FeatureIntegration2d(
            w, self.total_features, config.dropout
        )

        self._attn_pool_query = nn.Parameter(torch.randn(1, w))
        self._attn_pool_scale = w**-0.5

    def _pool_for_features(
        self, h: torch.Tensor, mask: torch.Tensor | None
    ) -> torch.Tensor:
        B, H, W, C = h.shape
        h_flat = h.reshape(B, H * W, C)
        q = self._attn_pool_query.unsqueeze(0).expand(B, -1, -1)
        attn = torch.bmm(q, h_flat.transpose(1, 2)) * self._attn_pool_scale
        if mask is not None:
            mask_flat = mask.reshape(B, H * W)
            attn = attn.masked_fill(mask_flat.unsqueeze(1), float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        return torch.bmm(attn, h_flat).squeeze(1)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.use_attention:
            if self.uses_adaptive_conv:
                h, biases, pooler_context = self.attn_block(h, mask)
                attn_features = self.attn_proj(self._pool_for_features(h, mask))
            else:
                h = self.attn_block(h, mask)
                attn_pooled = self.attn_pooler(h, mask)
                attn_features = self.attn_proj(attn_pooled)
                biases = None
                pooler_context = None
        else:
            if self.uses_adaptive_conv:
                biases, pooler_context = self.bias_proj(h, mask)
            else:
                biases = None
                pooler_context = None
            h = h + self.input_ffn(h)

        h, ssm_features, aux_losses = self.ssm_block(h, h, mask, biases, pooler_context)

        embed_features = self.embed_pooler(h, mask)

        hidden_pooled = self.hidden_pooler(h, mask)
        hidden_features = self.hidden_proj(hidden_pooled)

        if self.use_attention:
            features = torch.cat(
                [attn_features, ssm_features, embed_features, hidden_features],
                dim=-1,
            )
        else:
            features = torch.cat(
                [ssm_features, embed_features, hidden_features],
                dim=-1,
            )

        h = self.feature_integration(h, features, mask)

        return h, features, aux_losses


class UnifiedModel2d(nn.Module):
    def __init__(self, config: ModelConfig2d):
        super().__init__()
        self.config = config
        layer_config = config.layer
        head_config = config.head
        assert layer_config is not None and head_config is not None
        w = layer_config.embed_width

        self.embedding = PixelEmbedding2d(
            layer_config.in_channels, w, layer_config.dropout
        )
        self.layers = nn.ModuleList(
            [UnifiedLayer2d(layer_config) for _ in range(config.n_layers)]
        )

        self.output_pooler = LearnedPooler2d(w, num_queries=1)

        if head_config.head_type == "classifier":
            self.head = ClassifierHead(w, head_config.n_classes)
        else:
            raise ValueError(f"Unknown head type: {head_config.head_type}")

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        all_aux = {}

        h = self.embedding(x)

        for i, layer in enumerate(self.layers):
            h, features, aux = layer(h, mask)
            for k, v in aux.items():
                all_aux[f"layer{i}_{k}"] = v

        pooled = self.output_pooler(h, mask)
        output = self.head(pooled)
        return output, all_aux


class Model2d(nn.Module):
    def __init__(self, config: ModelConfig2d):
        super().__init__()
        self.config = config
        self._impl = UnifiedModel2d(config)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self._impl(x, mask)

    def predict(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            result = self.forward(x, mask)
            logits = result[0]
            return logits.argmax(dim=-1)
