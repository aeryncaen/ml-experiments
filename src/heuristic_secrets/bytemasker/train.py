from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from tqdm import tqdm

from heuristic_secrets.bytemasker.model import ByteMaskerConfig, ByteMaskerModel
from heuristic_secrets.bytemasker.dataset import (
    LineMaskDataset,
    create_bucketed_batches,
    compute_stats,
)


@dataclass
class TrainMetrics:
    loss: float
    precision: float
    recall: float
    f1: float
    fn_rate: float = 0.0
    
    def __str__(self) -> str:
        return (
            f"loss={self.loss:.4f} "
            f"P={self.precision:.4f} R={self.recall:.4f} F1={self.f1:.4f} "
            f"FN%={self.fn_rate * 100:.2f}"
        )


def compute_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    lengths: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[float, float, float, float, float]:
    batch_size, max_len = preds.shape
    position_mask = torch.arange(max_len, device=preds.device).unsqueeze(0) < lengths.unsqueeze(1)
    
    pred_binary = (torch.sigmoid(preds) >= threshold).float()
    valid_preds = pred_binary[position_mask]
    valid_targets = targets[position_mask]
    
    tp = ((valid_preds == 1) & (valid_targets == 1)).sum().item()
    fp = ((valid_preds == 1) & (valid_targets == 0)).sum().item()
    fn = ((valid_preds == 0) & (valid_targets == 1)).sum().item()
    tn = ((valid_preds == 0) & (valid_targets == 0)).sum().item()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fp_rate = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fn_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return precision, recall, f1, fp_rate, fn_rate


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ByteMaskerTrainer:

    def __init__(
        self,
        model: ByteMaskerModel,
        train_dataset: LineMaskDataset,
        val_dataset: LineMaskDataset | None = None,
        lr: float = 1e-3,
        batch_size: int = 64,
        device: torch.device | None = None,
        train_on_secret_lines_only: bool = True,
        pos_weight: float = 1.0,
        epochs: int = 20,
        warmup_epochs: int = 2,
        grad_clip: float = 1.0,
        hard_example_ratio: float = 0.3,
        swa_start_pct: float = 0.4,
        swa_lr: float | None = None,
        swa_anneal_epochs: int = 5,
        curriculum_start_epoch: int = 2,
        curriculum_end_epoch: int | None = None,
        threshold_search: bool = True,
        threshold_search_steps: int = 50,
        val_clean_ratio: float = 1.0,
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.pos_weight = pos_weight
        self.grad_clip = grad_clip
        self.max_hard_example_ratio = hard_example_ratio
        self.epochs = epochs
        self.swa_start_epoch = int(epochs * swa_start_pct)
        self.curriculum_start_epoch = curriculum_start_epoch
        self.curriculum_end_epoch = curriculum_end_epoch or (epochs // 2)
        self.threshold_search = threshold_search
        self.threshold_search_steps = threshold_search_steps
        self.optimal_threshold = 0.5
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        train_indices = train_dataset.secret_lines if train_on_secret_lines_only else None
        self._train_batches = create_bucketed_batches(
            train_dataset, batch_size, indices=train_indices, shuffle=False, show_progress=False
        )
        self._initial_num_batches = len(self._train_batches)
        
        num_batches = len(self._train_batches)
        self._max_batches_per_epoch = int(num_batches * 1.5)
        total_steps = epochs * self._max_batches_per_epoch
        warmup_steps = warmup_epochs * self._max_batches_per_epoch
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=lr,
            total_steps=total_steps,
            pct_start=warmup_steps / total_steps if total_steps > 0 else 0.1,
            anneal_strategy='cos',
        )
        self._scheduler_steps = 0
        self._scheduler_total_steps = total_steps
        
        self.swa_model = AveragedModel(model)
        self.swa_scheduler = SWALR(
            self.optimizer,
            swa_lr=swa_lr or lr * 0.05,
            anneal_epochs=swa_anneal_epochs,
            anneal_strategy='cos',
        )
        self.swa_active = False
        
        self.best_f1 = 0.0
        self.best_recall = 0.0
        self.best_state = None
        self._batch_losses: list[float] = [0.0] * len(self._train_batches)
        
        self._val_batches = None
        self._val_fp_batches = None
        if val_dataset:
            self._val_batches = create_bucketed_batches(
                val_dataset, batch_size, indices=val_dataset.secret_lines, shuffle=False
            )
            n_val_clean = min(
                int(len(val_dataset.secret_lines) * val_clean_ratio),
                len(val_dataset.clean_lines)
            )
            import random
            val_clean_indices = random.sample(val_dataset.clean_lines, n_val_clean) if n_val_clean > 0 else []
            self._val_fp_batches = create_bucketed_batches(
                val_dataset, batch_size, indices=val_clean_indices, shuffle=False, show_progress=False
            ) if val_clean_indices else []

    def rebuild_train_batches(self, indices: list[int], seed: int | None = None) -> None:
        self._train_batches = create_bucketed_batches(
            self.train_dataset, self.batch_size, indices=indices, shuffle=True, seed=seed, show_progress=False
        )
        self._batch_losses = [0.0] * len(self._train_batches)
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor, position_mask: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        if self.pos_weight != 1.0:
            weight = torch.where(targets == 1, self.pos_weight, 1.0)
            bce = bce * weight
        return (bce * position_mask).sum() / position_mask.sum()

    def _get_curriculum_hard_ratio(self, epoch: int) -> float:
        if epoch < self.curriculum_start_epoch:
            return 0.0
        if epoch >= self.curriculum_end_epoch:
            return self.max_hard_example_ratio
        progress = (epoch - self.curriculum_start_epoch) / (self.curriculum_end_epoch - self.curriculum_start_epoch)
        return self.max_hard_example_ratio * progress

    def train_epoch(self, seed: int | None = None, epoch: int = 0) -> TrainMetrics:
        import random
        self.model.train()
        
        if epoch >= self.swa_start_epoch and not self.swa_active:
            self.swa_active = True
        
        batch_indices = list(range(len(self._train_batches)))
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
        total_tp, total_fp, total_fn = 0, 0, 0
        
        desc = "Training (SWA)" if self.swa_active else "Training"
        pbar = tqdm(batch_indices, desc=desc, leave=False)
        for batch_idx in pbar:
            bytes_batch, masks_batch, lengths = self._train_batches[batch_idx]
            bytes_batch = bytes_batch.to(self.device)
            masks_batch = masks_batch.to(self.device)
            lengths = lengths.to(self.device)
            
            self.optimizer.zero_grad()
            
            logits = self.model(bytes_batch)
            
            batch_size, max_len = logits.shape
            position_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            
            loss = self._compute_loss(logits, masks_batch, position_mask)
            
            loss.backward()
            
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            if self.swa_active:
                self.swa_scheduler.step()
            elif self._scheduler_steps < self._scheduler_total_steps:
                self.scheduler.step()
                self._scheduler_steps += 1
            
            loss_val = loss.item()
            total_loss += loss_val
            self._batch_losses[batch_idx] = loss_val
            
            with torch.no_grad():
                pred_binary = (torch.sigmoid(logits) >= 0.5).float()
                valid_preds = pred_binary[position_mask]
                valid_targets = masks_batch[position_mask]
                
                total_tp += ((valid_preds == 1) & (valid_targets == 1)).sum().item()
                total_fp += ((valid_preds == 1) & (valid_targets == 0)).sum().item()
                total_fn += ((valid_preds == 0) & (valid_targets == 1)).sum().item()
            
            sched = self.swa_scheduler if self.swa_active else self.scheduler
            lr = sched.get_last_lr()[0]
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{lr:.2e}")
        
        if self.swa_active:
            self.swa_model.update_parameters(self.model)
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fn_rate = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0.0
        
        return TrainMetrics(
            loss=total_loss / len(batch_indices),
            precision=precision,
            recall=recall,
            f1=f1,
            fn_rate=fn_rate,
        )
    
    def finalize_swa(self) -> None:
        if not self.swa_active:
            return
        update_bn(self._get_bn_loader(), self.swa_model, device=self.device)
        self.model.load_state_dict(self.swa_model.module.state_dict())
    
    def _get_bn_loader(self):
        for batch in self._train_batches:
            yield batch[0].to(self.device)

    @torch.no_grad()
    def validate(self, optimize_threshold: bool = False) -> TrainMetrics:
        if self._val_batches is None:
            raise ValueError("No validation dataset provided")
        
        self.model.eval()
        batches = self._val_batches
        
        all_probs: list[torch.Tensor] = []
        all_targets: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        total_loss = 0.0
        
        for bytes_batch, masks_batch, lengths in batches:
            bytes_batch = bytes_batch.to(self.device)
            masks_batch = masks_batch.to(self.device)
            lengths = lengths.to(self.device)
            
            logits = self.model(bytes_batch)
            
            batch_size, max_len = logits.shape
            position_mask = torch.arange(max_len, device=self.device).unsqueeze(0) < lengths.unsqueeze(1)
            
            loss = self._compute_loss(logits, masks_batch, position_mask)
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_probs.append(probs[position_mask].cpu())
            all_targets.append(masks_batch[position_mask].cpu())

        all_probs_t = torch.cat(all_probs)
        all_targets_t = torch.cat(all_targets)

        if optimize_threshold and self.threshold_search:
            self.optimal_threshold = self._find_optimal_threshold(all_probs_t, all_targets_t)

        threshold = self.optimal_threshold
        preds = (all_probs_t >= threshold).float()

        total_tp = ((preds == 1) & (all_targets_t == 1)).sum().item()
        total_fp = ((preds == 1) & (all_targets_t == 0)).sum().item()
        total_fn = ((preds == 0) & (all_targets_t == 1)).sum().item()
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fn_rate = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0.0
        
        if recall > self.best_recall:
            self.best_recall = recall
            self.best_f1 = f1
            self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        
        return TrainMetrics(
            loss=total_loss / len(batches) if batches else 0.0,
            precision=precision,
            recall=recall,
            f1=f1,
            fn_rate=fn_rate,
        )

    def _find_optimal_threshold(self, probs: torch.Tensor, targets: torch.Tensor, min_precision: float = 0.5) -> float:
        best_recall = 0.0
        best_threshold = 0.5
        for t in torch.linspace(0.1, 0.9, self.threshold_search_steps):
            threshold = t.item()
            preds = (probs >= threshold).float()
            tp = ((preds == 1) & (targets == 1)).sum().item()
            fp = ((preds == 1) & (targets == 0)).sum().item()
            fn = ((preds == 0) & (targets == 1)).sum().item()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if precision >= min_precision and recall > best_recall:
                best_recall = recall
                best_threshold = threshold
        return best_threshold

    @torch.no_grad()
    def validate_fp_rate(self) -> float:
        if self._val_fp_batches is None:
            raise ValueError("No validation dataset provided")
        
        self.model.eval()
        batches = self._val_fp_batches
        threshold = self.optimal_threshold
        
        lines_with_fp = 0
        total_lines = 0
        
        for bytes_batch, masks_batch, lengths in batches:
            bytes_batch = bytes_batch.to(self.device)
            lengths = lengths.to(self.device)
            
            logits = self.model(bytes_batch)
            preds = torch.sigmoid(logits) >= threshold
            
            batch_size, max_len = logits.shape
            
            for i in range(batch_size):
                valid_preds = preds[i, :lengths[i]]
                if valid_preds.any():
                    lines_with_fp += 1
                total_lines += 1
        
        return lines_with_fp / total_lines if total_lines > 0 else 0.0

    def save_checkpoint(self, path: Path) -> None:
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


def load_checkpoint(path: Path, device: torch.device | None = None) -> ByteMaskerModel:
    device = device or get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = ByteMaskerConfig.from_dict(checkpoint["config"])
    model = ByteMaskerModel(config)
    state = checkpoint.get("best_state") or checkpoint["state_dict"]
    model.load_state_dict(state)
    return model.to(device)
