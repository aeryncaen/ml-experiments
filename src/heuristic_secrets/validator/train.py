from dataclasses import dataclass
import random
import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

from heuristic_secrets.validator.model import ValidatorModel


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class TrainMetrics:
    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    fp_rate: float = 0.0
    fn_rate: float = 0.0

    def __str__(self) -> str:
        return (
            f"loss={self.loss:.4f} "
            f"acc={self.accuracy:.4f} "
            f"P={self.precision:.4f} R={self.recall:.4f} F1={self.f1:.4f} "
            f"FP%={self.fp_rate * 100:.2f} FN%={self.fn_rate * 100:.2f}"
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ValidatorTrainer:
    def __init__(
        self,
        model: ValidatorModel,
        train_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        val_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] | None = None,
        lr: float = 1e-3,
        device: torch.device | None = None,
        epochs: int = 20,
        warmup_epochs: int = 2,
        grad_clip: float = 1.0,
        hard_example_ratio: float = 0.3,
        swa_start_pct: float = 0.4,
        swa_lr: float | None = None,
        swa_anneal_epochs: int = 5,
        pos_weight: float = 1.0,
        seed: int | None = None,
        curriculum_start_epoch: int = 2,
        curriculum_end_epoch: int | None = None,
        threshold_search: bool = True,
        threshold_search_steps: int = 50,
    ):
        if seed is not None:
            set_seed(seed)
        
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_data = train_data
        self.val_data = val_data
        self.grad_clip = grad_clip
        self.max_hard_example_ratio = hard_example_ratio
        self.epochs = epochs
        self.swa_start_epoch = int(epochs * swa_start_pct)
        self.pos_weight = pos_weight
        
        self.curriculum_start_epoch = curriculum_start_epoch
        self.curriculum_end_epoch = curriculum_end_epoch or (epochs // 2)
        self.threshold_search = threshold_search
        self.threshold_search_steps = threshold_search_steps
        self.optimal_threshold = 0.5

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        num_batches = len(train_data)
        total_steps = epochs * num_batches
        warmup_steps = warmup_epochs * num_batches

        pct_start = min(warmup_steps / total_steps, 0.3) if total_steps > 0 else 0.1
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy='cos',
        )

        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(
            self.optimizer,
            swa_lr=swa_lr or lr * 0.05,
            anneal_epochs=swa_anneal_epochs,
            anneal_strategy='cos',
        )
        self.swa_active = False

        self.best_f1 = 0.0
        self.best_state: dict | None = None
        self._batch_losses: list[float] = [0.0] * len(train_data)

    def set_train_data(
        self,
        train_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        """Update training data and resize batch loss tracking."""
        self.train_data = train_data
        if len(train_data) != len(self._batch_losses):
            self._batch_losses = [0.0] * len(train_data)

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.pos_weight != 1.0:
            weight = torch.where(labels == 1, self.pos_weight, 1.0)
            return nn.functional.binary_cross_entropy_with_logits(
                logits.squeeze(-1), labels, weight=weight
            )
        return nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), labels)

    def _get_curriculum_hard_ratio(self, epoch: int) -> float:
        if epoch < self.curriculum_start_epoch:
            return 0.0
        if epoch >= self.curriculum_end_epoch:
            return self.max_hard_example_ratio
        progress = (epoch - self.curriculum_start_epoch) / (self.curriculum_end_epoch - self.curriculum_start_epoch)
        return self.max_hard_example_ratio * progress

    def train_epoch(self, seed: int | None = None, epoch: int = 0) -> TrainMetrics:
        self.model.train()

        if epoch >= self.swa_start_epoch and not self.swa_active:
            self.swa_active = True

        batch_indices = list(range(len(self.train_data)))
        if seed is not None:
            random.seed(seed)
        random.shuffle(batch_indices)

        hard_ratio = self._get_curriculum_hard_ratio(epoch)
        if epoch > 0 and hard_ratio > 0:
            n_hard = int(len(batch_indices) * hard_ratio)
            if n_hard > 0:
                hard_indices = sorted(
                    range(len(self._batch_losses)),
                    key=lambda i: self._batch_losses[i],
                    reverse=True
                )[:n_hard]
                batch_indices = batch_indices + hard_indices
                random.shuffle(batch_indices)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        total_tp, total_fp, total_fn = 0, 0, 0

        desc = "Training (SWA)" if self.swa_active else "Training"
        pbar = tqdm(batch_indices, desc=desc, leave=False)

        for batch_idx in pbar:
            bytes_batch, features_batch, labels_batch, lengths_batch = self.train_data[batch_idx]
            bytes_batch = bytes_batch.to(self.device)
            features_batch = features_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)
            lengths_batch = lengths_batch.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(features_batch, bytes_batch, lengths_batch)

            loss = self._compute_loss(logits, labels_batch)
            loss.backward()

            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.optimizer.step()

            if self.swa_active:
                self.swa_scheduler.step()
            else:
                self.scheduler.step()

            loss_val = loss.item()
            total_loss += loss_val
            self._batch_losses[batch_idx] = loss_val

            with torch.no_grad():
                preds = (torch.sigmoid(logits.squeeze(-1)) >= 0.5).float()
                total_correct += (preds == labels_batch).sum().item()
                total_samples += len(labels_batch)

                total_tp += ((preds == 1) & (labels_batch == 1)).sum().item()
                total_fp += ((preds == 1) & (labels_batch == 0)).sum().item()
                total_fn += ((preds == 0) & (labels_batch == 1)).sum().item()

            sched = self.swa_scheduler if self.swa_active else self.scheduler
            lr = sched.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")

        if self.swa_active:
            self.swa_model.update_parameters(self.model)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total_neg = total_samples - (total_tp + total_fn)
        total_pos = total_tp + total_fn
        fp_rate = total_fp / total_neg if total_neg > 0 else 0.0
        fn_rate = total_fn / total_pos if total_pos > 0 else 0.0

        return TrainMetrics(
            loss=total_loss / len(batch_indices),
            accuracy=total_correct / total_samples if total_samples > 0 else 0.0,
            precision=precision,
            recall=recall,
            f1=f1,
            fp_rate=fp_rate,
            fn_rate=fn_rate,
        )

    def finalize_swa(self) -> None:
        if not self.swa_active:
            return
        update_bn(self._get_bn_loader(), self.swa_model, device=self.device)
        self.model.load_state_dict(self.swa_model.module.state_dict())

    def _get_bn_loader(self):
        for batch in self.train_data:
            yield batch[0].to(self.device)

    @torch.no_grad()
    def validate(self, optimize_threshold: bool = False) -> TrainMetrics:
        if self.val_data is None:
            raise ValueError("No validation data provided")

        self.model.eval()

        all_probs: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []
        total_loss = 0.0

        for bytes_batch, features_batch, labels_batch, lengths_batch in self.val_data:
            bytes_batch = bytes_batch.to(self.device)
            features_batch = features_batch.to(self.device)
            labels_batch = labels_batch.to(self.device)
            lengths_batch = lengths_batch.to(self.device)

            logits = self.model(features_batch, bytes_batch, lengths_batch)
            loss = self._compute_loss(logits, labels_batch)
            total_loss += loss.item()

            probs = torch.sigmoid(logits.squeeze(-1))
            all_probs.append(probs.cpu())
            all_labels.append(labels_batch.cpu())

        all_probs_t = torch.cat(all_probs)
        all_labels_t = torch.cat(all_labels)

        if optimize_threshold and self.threshold_search:
            self.optimal_threshold = self._find_optimal_threshold(all_probs_t, all_labels_t)

        threshold = self.optimal_threshold
        preds = (all_probs_t >= threshold).float()

        total_correct = (preds == all_labels_t).sum().item()
        total_samples = len(all_labels_t)
        total_tp = ((preds == 1) & (all_labels_t == 1)).sum().item()
        total_fp = ((preds == 1) & (all_labels_t == 0)).sum().item()
        total_fn = ((preds == 0) & (all_labels_t == 1)).sum().item()

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total_neg = total_samples - (total_tp + total_fn)
        total_pos = total_tp + total_fn
        fp_rate = total_fp / total_neg if total_neg > 0 else 0.0
        fn_rate = total_fn / total_pos if total_pos > 0 else 0.0

        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        return TrainMetrics(
            loss=total_loss / len(self.val_data) if self.val_data else 0.0,
            accuracy=total_correct / total_samples if total_samples > 0 else 0.0,
            precision=precision,
            recall=recall,
            f1=f1,
            fp_rate=fp_rate,
            fn_rate=fn_rate,
        )

    def _find_optimal_threshold(self, probs: torch.Tensor, labels: torch.Tensor) -> float:
        best_f1 = 0.0
        best_threshold = 0.5
        for t in torch.linspace(0.1, 0.9, self.threshold_search_steps):
            threshold = t.item()
            preds = (probs >= threshold).float()
            tp = ((preds == 1) & (labels == 1)).sum().item()
            fp = ((preds == 1) & (labels == 0)).sum().item()
            fn = ((preds == 0) & (labels == 1)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        return best_threshold

    def save_checkpoint(self, path) -> None:
        torch.save({
            "config": self.model.config.to_dict(),
            "state_dict": self.model.state_dict(),
            "best_f1": self.best_f1,
            "best_state": self.best_state,
            "optimal_threshold": self.optimal_threshold,
        }, path)

    def load_best_model(self) -> None:
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)


def load_checkpoint(path, device: torch.device | None = None) -> ValidatorModel:
    from heuristic_secrets.validator.model import ValidatorConfig
    device = device or get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ValidatorConfig.from_dict(checkpoint["config"])
    model = ValidatorModel(config)
    state = checkpoint.get("best_state") or checkpoint["state_dict"]
    model.load_state_dict(state)
    return model.to(device)
