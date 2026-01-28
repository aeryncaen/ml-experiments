"""Learned internal loss for iterative refinement (ML3-style)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F



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
    """Learned loss function: logits [B, n_classes] → scalar loss [B]."""

    def __init__(self, n_classes: int, n_layers: int = 1, width: int | None = None, n_heads: int = 2):
        super().__init__()
        if width is None:
            width = max(n_heads, n_classes)
            width = width - (width % n_heads) if width % n_heads != 0 else width
        self.proj_in = nn.Linear(n_classes, width) if n_classes != width else nn.Identity()
        self.blocks = nn.ModuleList([
            SiLUAttentionBlock(width, n_heads=n_heads) for _ in range(n_layers)
        ])
        self.head = nn.Linear(width, 1)
        nn.init.xavier_normal_(self.head.weight, gain=0.1)
        nn.init.zeros_(self.head.bias)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """[B, n_classes] → [B]"""
        x = logits.unsqueeze(1)
        x = self.proj_in(x)
        for block in self.blocks:
            x = block(x)
        return F.softplus(self.head(x.squeeze(1)).squeeze(-1))


class PonderWrapper(nn.Module):
    """Wraps a classifier with a learned auxiliary loss.

    Base minimizes CE + predicted_loss. L_internal learns via REINFORCE
    on accuracy improvement — a secondary objective beyond CE.
    """

    def __init__(
        self,
        model: nn.Module,
        n_classes: int | None = None,
        loss_net_width: int | None = None,
        loss_net_layers: int = 1,
    ):
        super().__init__()
        self.base = model

        head = getattr(model, "head", None)
        if n_classes is None:
            if head is not None and hasattr(head, "out_features"):
                n_classes = head.out_features
            else:
                raise ValueError("Cannot infer n_classes; pass it explicitly")

        assert n_classes is not None
        self.n_classes = n_classes
        self.l_internal = InternalLossNetwork(n_classes, loss_net_layers, loss_net_width)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [B, C], predicted_loss [B])."""
        logits = self.base(x)
        predicted_loss = self.l_internal(logits)
        return logits, predicted_loss


@dataclass
class PonderTrainConfig:
    meta_lr: float = 3e-4
    model_lr: float = 1e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    epochs: int = 50
    warmup_epochs: int = 2
    meta_warmup_epochs: int = 1
    meta_wean_epochs: int = 1
    log_interval: int = 50


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

        base_params = list(ponder.base.parameters())
        meta_params = list(ponder.l_internal.parameters())

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

    def _prep_input(self, images: torch.Tensor) -> torch.Tensor:
        images = images.to(self.device)
        if self.flatten_input:
            images = images.view(images.size(0), -1)
        elif self.squeeze_channel and images.dim() == 4:
            images = images.squeeze(1)
        return images

    def _reinforce_alpha(self, epoch: int) -> float:
        cfg = self.config
        if epoch < cfg.meta_warmup_epochs:
            return 0.0
        wean_progress = (epoch - cfg.meta_warmup_epochs) / max(cfg.meta_wean_epochs, 1)
        return min(wean_progress, 1.0)

    def train_epoch(self, epoch: int) -> dict:
        self.ponder.train()
        cfg = self.config
        alpha = self._reinforce_alpha(epoch)

        running: dict[str, float] = {}
        count = 0

        from tqdm import tqdm
        phase = "CE" if alpha == 0.0 else ("WEAN" if alpha < 1.0 else "RL")
        desc = f"Epoch {epoch+1}/{cfg.epochs} [{phase} α={alpha:.2f}]"
        pbar = tqdm(self.train_loader, desc=desc, leave=False)

        for batch_idx, (images, labels) in enumerate(pbar):
            images = self._prep_input(images)
            labels = labels.to(self.device)

            with torch.no_grad():
                pre_logits = self.ponder(images)[0]
                pre_acc = (pre_logits.argmax(-1) == labels).float().mean()

            self.model_optimizer.zero_grad()
            logits, predicted_loss = self.ponder(images)
            ce = F.cross_entropy(logits, labels)
            base_loss = ce + predicted_loss.mean()
            base_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.ponder.base.parameters(), cfg.grad_clip)
            self.model_optimizer.step()
            self.model_scheduler.step()

            with torch.no_grad():
                post_logits = self.ponder(images)[0]
                post_ce = F.cross_entropy(post_logits, labels)
                post_acc = (post_logits.argmax(-1) == labels).float().mean()
                reward = post_acc - pre_acc

            self.meta_optimizer.zero_grad()
            logits_for_meta, predicted_loss_for_meta = self.ponder(images)
            per_sample_ce = F.cross_entropy(logits_for_meta, labels, reduction='none').detach()
            reinforce_loss = -(reward * predicted_loss_for_meta).mean()
            supervised_loss = F.mse_loss(predicted_loss_for_meta, per_sample_ce)
            meta_loss = alpha * reinforce_loss + (1 - alpha) * supervised_loss
            meta_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.ponder.l_internal.parameters()), cfg.grad_clip
            )
            self.meta_optimizer.step()
            self.meta_scheduler.step()

            with torch.no_grad():
                metrics = {
                    "predicted_loss": predicted_loss.mean().item(),
                    "ce": post_ce.item(),
                    "reward": reward.item(),
                    "accuracy": post_acc.item(),
                    "alpha": alpha,
                    "meta_loss": meta_loss.item(),
                }

            for k, v in metrics.items():
                running[k] = running.get(k, 0.0) + v
            count += 1

            pbar.set_postfix(
                ce=f"{metrics['ce']:.3f}",
                acc=f"{metrics['accuracy']:.3f}",
                rw=f"{metrics['reward']:.3f}",
                pl=f"{metrics['predicted_loss']:.3f}",
            )

        return {k: v / max(count, 1) for k, v in running.items()}

    @torch.no_grad()
    def evaluate(self) -> tuple[float, float]:
        self.ponder.eval()
        correct = 0
        total = 0
        total_ce = 0.0

        from tqdm import tqdm
        for images, labels in tqdm(self.test_loader, desc="Eval", leave=False):
            images = self._prep_input(images)
            labels = labels.to(self.device)

            logits, _ = self.ponder(images)
            total_ce += F.cross_entropy(logits, labels).item() * labels.size(0)
            correct += (logits.argmax(-1) == labels).sum().item()
            total += labels.size(0)

        n = max(total, 1)
        return total_ce / n, correct / n

    def train(self, epochs: int | None = None) -> float:
        epochs = epochs or self.config.epochs

        print(f"PonderTrainer: {epochs} epochs, device={self.device}")
        print(f"  Base model params: {sum(p.numel() for p in self.ponder.base.parameters()):,}")
        print(f"  L_internal params: {sum(p.numel() for p in self.ponder.l_internal.parameters()):,}")
        print(f"  LR warmup: {self.config.warmup_epochs} epochs")
        print(f"  Meta CE→RL: {self.config.meta_warmup_epochs} pure CE, {self.config.meta_wean_epochs} wean")
        print()

        for epoch in range(epochs):
            train_metrics = self.train_epoch(epoch)
            test_ce, test_acc = self.evaluate()

            if test_acc > self.best_acc:
                self.best_acc = test_acc

            model_lr = self.model_optimizer.param_groups[0]["lr"]
            meta_lr = self.meta_optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1:3d}: "
                f"pred={train_metrics['predicted_loss']:.4f} ce={train_metrics['ce']:.4f} "
                f"rw={train_metrics['reward']:.4f} "
                f"acc={train_metrics['accuracy']:.4f} α={train_metrics['alpha']:.2f} | "
                f"test_ce={test_ce:.4f} test_acc={test_acc:.4f} | "
                f"lr={model_lr:.1e}/{meta_lr:.1e}"
                + (" *BEST*" if test_acc >= self.best_acc else "")
            )

        print(f"\nBest test accuracy: {self.best_acc:.4f}")
        return self.best_acc
