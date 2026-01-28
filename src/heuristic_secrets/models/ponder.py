"""Learned internal loss + PonderNet halting for iterative refinement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class PonderableModel(Protocol):
    def embed(self, x: torch.Tensor) -> torch.Tensor: ...
    def refine(self, h: torch.Tensor) -> torch.Tensor: ...
    def decode(self, h: torch.Tensor) -> torch.Tensor: ...


class AutoSplitModel(nn.Module):
    """Splits embed->layers->norm->pool->head classifiers into three phases."""

    def __init__(self, model: nn.Module, pool_dims: tuple[int, ...] = (1, 2)):
        super().__init__()
        self.model = model
        self.pool_dims = pool_dims

    @staticmethod
    def from_classifier(model: nn.Module) -> "AutoSplitModel":
        if hasattr(model, "img_size"):
            pool_dims = (1, 2)
        elif hasattr(model, "vol_size"):
            pool_dims = (1, 2, 3)
        else:
            pool_dims = (1,)
        return AutoSplitModel(model, pool_dims=pool_dims)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        B = x.shape[0]
        if hasattr(m, "img_size"):
            x = x.view(B, *m.img_size, 1)
            return m.embed_norm(F.silu(m.patch_embed(x))) + m.pos_norm(
                F.silu(m.pos_embed)
            )
        elif hasattr(m, "vol_size"):
            x = x.permute(0, 2, 3, 1).unsqueeze(-1)
            return m.embed_norm(F.silu(m.patch_embed(x))) + m.pos_norm(
                F.silu(m.pos_embed)
            )
        else:
            return m.embed_norm(F.silu(m.embed(x.unsqueeze(-1)))) + m.pos_norm(
                F.silu(m.pos_embed)
            )

    def refine(self, h: torch.Tensor) -> torch.Tensor:
        for layer in self.model.layers:
            h = layer(h)
        return self.model.norm(h)

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        pooled = h.mean(dim=self.pool_dims)
        return self.model.head(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.refine(self.embed(x)))


class SiLUAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        h = self.norm1(x)
        q = F.silu(self.q(h)).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = F.silu(self.k(h)).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = F.silu(self.v(h)).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, L, C)
        x = x + self.out(out)
        x = x + self.mlp(self.norm2(x))
        return x


class InternalLossNetwork(nn.Module):
    def __init__(self, hidden_dim: int, n_layers: int = 1, width: int | None = None, n_heads: int = 2):
        super().__init__()
        if width is None:
            width = max(n_heads, hidden_dim // 4)
            width = width - (width % n_heads)
        self.proj_in = nn.Linear(hidden_dim, width) if hidden_dim != width else nn.Identity()
        self.blocks = nn.ModuleList([
            SiLUAttentionBlock(width, n_heads=n_heads) for _ in range(n_layers)
        ])
        self.head = nn.Linear(width, 1)
        nn.init.xavier_normal_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() > 2:
            B, C = h.shape[0], h.shape[-1]
            x = h.reshape(B, -1, C)
        else:
            x = h.unsqueeze(1)
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x.mean(dim=1)).squeeze(-1)


class HaltNetwork(nn.Module):
    def __init__(self, hidden_dim: int, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + 1, width),
            nn.ReLU(),
            nn.Linear(width, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h_pooled: torch.Tensor, loss_val: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([h_pooled, loss_val.unsqueeze(-1)], dim=-1)
        return self.net(inp).squeeze(-1)


class PonderWrapper(nn.Module):
    """Wraps any classifier with learned internal loss + PonderNet halting.

    Training: two alternating phases per batch.
      1. Inner: base model trains by minimizing L_internal (standard backprop)
      2. Outer: L_internal + halt_net train by maximizing reward (improvement - energy)

    Iteration: repeated forward passes through refine (weight-shared depth).
    Halting: L_internal value feeds halt_net, which produces PonderNet distribution.
    """

    def __init__(
        self,
        model: nn.Module,
        hidden_dim: int | None = None,
        loss_net_layers: int = 1,
        loss_net_width: int | None = None,
        halt_net_width: int = 64,
        max_steps: int = 10,
    ):
        super().__init__()

        if isinstance(model, AutoSplitModel):
            self.base = model
        elif hasattr(model, "embed") and hasattr(model, "refine") and hasattr(model, "decode"):
            self.base = model
        else:
            self.base = AutoSplitModel.from_classifier(model)

        if hidden_dim is None:
            head = self.base.model.head if hasattr(self.base, "model") else None
            if head is not None and hasattr(head, "in_features"):
                hidden_dim = head.in_features
            else:
                raise ValueError("Cannot infer hidden_dim; pass it explicitly")

        self.hidden_dim = hidden_dim
        self.max_steps = max_steps

        self.l_internal = InternalLossNetwork(hidden_dim, loss_net_layers, loss_net_width)
        self.halt_net = HaltNetwork(hidden_dim, halt_net_width)
        self.iter_norm = nn.LayerNorm(hidden_dim)
        self.residual_gate = nn.Parameter(torch.zeros(1))

    def _pool_hidden(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() > 2:
            return h.mean(dim=tuple(range(1, h.dim() - 1)))
        return h

    def forward_pondering(
        self,
        x: torch.Tensor,
        max_steps: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Forward with pondering: repeated refine passes + halting distribution.

        Used during both inner loop (to get L_internal values) and outer loop
        (to compute expected output under halting distribution for reward).
        """
        max_steps = max_steps or self.max_steps

        h = self.base.embed(x)
        h_0 = h

        halt_lambdas: list[torch.Tensor] = []
        step_logits: list[torch.Tensor] = []
        loss_values: list[torch.Tensor] = []

        for t in range(max_steps):
            h = self.iter_norm(h)
            alpha = torch.sigmoid(self.residual_gate)
            h = self.base.refine(h) + alpha * h_0

            h_pooled = self._pool_hidden(h)
            loss_val = self.l_internal(h)
            halt_logit = self.halt_net(h_pooled, loss_val)
            lam = torch.sigmoid(halt_logit)

            halt_lambdas.append(lam)
            step_logits.append(self.base.decode(h))
            loss_values.append(loss_val)

        # p(halt at t) = λ_t * Π_{i<t}(1 - λ_i)
        p_halt = []
        still_running = torch.ones_like(halt_lambdas[0])
        for lam in halt_lambdas:
            p_halt.append(still_running * lam)
            still_running = still_running * (1 - lam)
        p_halt = torch.stack(p_halt, dim=0)  # [T, B]

        all_logits = torch.stack(step_logits, dim=0)  # [T, B, C]
        expected_logits = (p_halt.unsqueeze(-1) * all_logits).sum(dim=0)  # [B, C]

        steps = torch.arange(1, max_steps + 1, device=x.device, dtype=torch.float32)
        expected_steps = (p_halt * steps.unsqueeze(-1)).sum(dim=0)  # [B]

        info = {
            "p_halt": p_halt,                                    # [T, B]
            "expected_steps": expected_steps,                    # [B]
            "loss_values": torch.stack(loss_values, dim=0),      # [T, B]
            "halt_lambdas": torch.stack(
                [l.detach() for l in halt_lambdas], dim=0
            ),                                                   # [T, B]
        }
        return expected_logits, info

    @torch.no_grad()
    def forward_inference(
        self,
        x: torch.Tensor,
        max_steps: int | None = None,
        halt_threshold: float = 0.5,
    ) -> tuple[torch.Tensor, int]:
        max_steps = max_steps or self.max_steps

        h = self.base.embed(x)
        h_0 = h

        for t in range(max_steps):
            h = self.iter_norm(h)
            alpha = torch.sigmoid(self.residual_gate)
            h = self.base.refine(h) + alpha * h_0

            h_pooled = self._pool_hidden(h)
            loss_val = self.l_internal(h)
            halt_logit = self.halt_net(h_pooled, loss_val)
            halt_prob = torch.sigmoid(halt_logit)

            if halt_prob.mean().item() > halt_threshold:
                return self.base.decode(h), t + 1

        return self.base.decode(h), max_steps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            logits, _ = self.forward_pondering(x)
            return logits
        else:
            logits, _ = self.forward_inference(x)
            return logits


# ---------------------------------------------------------------------------
# Config + reward
# ---------------------------------------------------------------------------


@dataclass
class PonderTrainConfig:
    meta_lr: float = 3e-4
    model_lr: float = 1e-3
    inner_steps: int = 1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    epochs: int = 50
    warmup_epochs: int = 2

    lambda_energy: float = 0.05
    energy_warmup_epochs: int = 10

    geometric_prior_lambda: float = 0.3
    kl_weight: float = 0.01

    gluttony_max_steps: int = 15
    starvation_max_steps: int = 3
    block_length: int = 10
    gluttony_fraction: float = 0.4

    log_interval: int = 50


# ---------------------------------------------------------------------------
# Trainer: alternating inner/outer optimization
# ---------------------------------------------------------------------------


class PonderTrainer:
    def __init__(
        self,
        ponder: PonderWrapper,
        train_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        config: PonderTrainConfig | None = None,
        flatten_input: bool = False,
        squeeze_channel: bool = True,
    ):
        self.ponder = ponder.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.config = config or PonderTrainConfig()
        self.flatten_input = flatten_input
        self.squeeze_channel = squeeze_channel

        base_params = (
            list(ponder.base.parameters())
            + list(ponder.iter_norm.parameters())
            + [ponder.residual_gate]
        )
        meta_params = (
            list(ponder.l_internal.parameters())
            + list(ponder.halt_net.parameters())
        )

        self.model_optimizer = torch.optim.AdamW(
            base_params, lr=self.config.model_lr, weight_decay=self.config.weight_decay
        )
        self.meta_optimizer = torch.optim.AdamW(
            meta_params, lr=self.config.meta_lr, weight_decay=self.config.weight_decay
        )

        total_steps = self.config.epochs * len(train_loader)
        warmup_steps = self.config.warmup_epochs * len(train_loader)

        self.model_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.model_optimizer,
            self._make_lr_lambda(total_steps, warmup_steps),
        )
        self.meta_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.meta_optimizer,
            self._make_lr_lambda(total_steps, warmup_steps),
        )

        self.best_acc = 0.0

    @staticmethod
    def _make_lr_lambda(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return lr_lambda

    def _get_regime(self, epoch: int) -> str:
        c = self.config
        block_idx = epoch // c.block_length
        cycle_len = 2
        glut_blocks = max(1, round(c.gluttony_fraction * cycle_len))
        pos_in_cycle = block_idx % cycle_len
        return "gluttony" if pos_in_cycle < glut_blocks else "starvation"

    def _get_max_steps(self, epoch: int) -> int:
        regime = self._get_regime(epoch)
        return self.config.gluttony_max_steps if regime == "gluttony" else self.config.starvation_max_steps

    def _prep_input(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        if self.flatten_input:
            images = images.view(images.size(0), -1)
        elif self.squeeze_channel and images.dim() == 4:
            images = images.squeeze(1)
        return images

    def train_epoch(self, epoch: int) -> dict:
        self.ponder.train()
        max_steps = self._get_max_steps(epoch)
        regime = self._get_regime(epoch)
        cfg = self.config

        running: dict[str, float] = {}
        count = 0

        from tqdm import tqdm
        desc = f"Epoch {epoch+1}/{cfg.epochs} [{regime[:5]} T={max_steps}]"
        pbar = tqdm(self.train_loader, desc=desc, leave=False)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = self._prep_input(images)
            labels = labels.to(self.device)

            # === INNER LOOP: train base model using L_internal as loss ===
            for _ in range(cfg.inner_steps):
                self.model_optimizer.zero_grad()
                _, info = self.ponder.forward_pondering(images, max_steps=max_steps)
                # L_internal averaged across steps, weighted by halt distribution
                inner_loss = (info["p_halt"] * info["loss_values"]).sum(dim=0).mean()
                inner_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.ponder.base.parameters(), cfg.grad_clip
                )
                self.model_optimizer.step()
                self.model_scheduler.step()

            # === OUTER LOOP: train L_internal + halt_net using reward ===
            self.meta_optimizer.zero_grad()

            # Baseline: single refine pass, no pondering (detached)
            with torch.no_grad():
                h_base = self.ponder.base.embed(images)
                baseline_logits = self.ponder.base.decode(self.ponder.base.refine(h_base))
                baseline_ce = F.cross_entropy(baseline_logits, labels, reduction="none")

            expected_logits, info = self.ponder.forward_pondering(images, max_steps=max_steps)
            ce = F.cross_entropy(expected_logits, labels, reduction="none")

            improvement = baseline_ce - ce
            expected_steps = info["expected_steps"]
            energy_scale = min(1.0, epoch / max(cfg.energy_warmup_epochs, 1))
            energy_fraction = expected_steps / max(max_steps, 1)
            reward = improvement - cfg.lambda_energy * energy_scale * energy_fraction

            # KL(p_halt || geometric prior)
            p_halt = info["p_halt"]
            T = p_halt.shape[0]
            lp = cfg.geometric_prior_lambda
            t_idx = torch.arange(T, device=p_halt.device, dtype=torch.float32)
            p_geometric = lp * ((1 - lp) ** t_idx)
            p_geometric = (p_geometric / p_geometric.sum()).unsqueeze(-1).expand_as(p_halt)
            kl = (p_halt * (torch.log(p_halt + 1e-8) - torch.log(p_geometric + 1e-8))).sum(dim=0)

            meta_loss = (-reward + cfg.kl_weight * kl).mean()
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.ponder.l_internal.parameters()) + list(self.ponder.halt_net.parameters()),
                cfg.grad_clip,
            )
            self.meta_optimizer.step()
            self.meta_scheduler.step()

            with torch.no_grad():
                preds = expected_logits.argmax(dim=-1)
                acc = (preds == labels).float().mean()
                metrics = {
                    "meta_loss": meta_loss.item(),
                    "inner_loss": inner_loss.item(),
                    "ce": ce.mean().item(),
                    "baseline_ce": baseline_ce.mean().item(),
                    "improvement": improvement.mean().item(),
                    "reward": reward.mean().item(),
                    "expected_steps": expected_steps.mean().item(),
                    "kl": kl.mean().item(),
                    "accuracy": acc.item(),
                    "l_internal_mean": info["loss_values"].mean().item(),
                }

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            count += 1

            if batch_idx % cfg.log_interval == 0:
                pbar.set_postfix(
                    ce=f"{metrics['ce']:.3f}",
                    acc=f"{metrics['accuracy']:.3f}",
                    steps=f"{metrics['expected_steps']:.1f}",
                    improv=f"{metrics['improvement']:.3f}",
                )

        return {k: v / max(count, 1) for k, v in running.items()}

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float, float]:
        self.ponder.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        total_steps = 0.0

        from tqdm import tqdm
        for images, labels in tqdm(self.test_loader, desc="Eval", leave=False):
            images = self._prep_input(images)
            labels = labels.to(self.device)

            logits, steps = self.ponder.forward_inference(images)
            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=-1) == labels).sum().item()
            total += labels.size(0)
            total_steps += steps * labels.size(0)

        n = max(total, 1)
        return total_loss / n, correct / n, total_steps / n

    def train(self, epochs: int | None = None) -> float:
        epochs = epochs or self.config.epochs
        cfg = self.config

        print(f"PonderTrainer: {epochs} epochs, device={self.device}")
        print(f"  Base model params: {sum(p.numel() for p in self.ponder.base.parameters()):,}")
        print(f"  L_internal params: {sum(p.numel() for p in self.ponder.l_internal.parameters()):,}")
        print(f"  Halt net params:   {sum(p.numel() for p in self.ponder.halt_net.parameters()):,}")
        print(f"  Inner steps per batch: {cfg.inner_steps}")
        print(f"  Gluttony steps: {cfg.gluttony_max_steps}, Starvation steps: {cfg.starvation_max_steps}")
        print(f"  Block length: {cfg.block_length} epochs, Gluttony fraction: {cfg.gluttony_fraction}")
        print()

        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch)
            test_loss, test_acc, avg_steps = self.evaluate()

            if test_acc > self.best_acc:
                self.best_acc = test_acc

            regime = self._get_regime(epoch)
            max_steps = self._get_max_steps(epoch)
            model_lr = self.model_optimizer.param_groups[0]["lr"]
            meta_lr = self.meta_optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1:3d} [{regime[:5]} T={max_steps:2d}]: "
                f"train_ce={train_metrics['ce']:.4f} base_ce={train_metrics['baseline_ce']:.4f} "
                f"improv={train_metrics['improvement']:.4f} acc={train_metrics['accuracy']:.4f} "
                f"E[steps]={train_metrics['expected_steps']:.1f} | "
                f"test_acc={test_acc:.4f} test_steps={avg_steps:.1f} | "
                f"reward={train_metrics['reward']:.3f} "
                f"lr={model_lr:.1e}/{meta_lr:.1e}"
                + (" *BEST*" if test_acc >= self.best_acc else "")
            )

        print(f"\nBest test accuracy: {self.best_acc:.4f}")
        return self.best_acc
