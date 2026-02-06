#!/usr/bin/env python3
"""
Tier 3: Geometry-informed training intervention on a real dataset.

This script compares:
  - baseline: standard MLP initialization + ERM training
  - geo_system: geometry-informed spectral init + early spectral regularization

Dataset: sklearn digits (8x8).

Intervention:
  1) Build a class-structure operator on input space:
       M = sum_c p(c) (mu_c - mu)(mu_c - mu)^T
  2) Eigen-decompose M; use top eigenvectors to pre-bias first-layer rows.
  3) During early epochs, regularize first-layer rows to remain in the
     top-eigenspace (selection-style shaping).

This is a small, local Tier 3 prototype to validate direction before full runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from chi2_optimizer import Chi2TrustRegion


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_masks() -> tuple[np.ndarray, np.ndarray]:
    border = np.zeros((8, 8), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    center = np.zeros((8, 8), dtype=bool)
    center[2:6, 2:6] = True
    return border.ravel(), center.ravel()


def apply_shift(X: np.ndarray, shift_type: str, strength: float) -> np.ndarray:
    Xs = X.copy()
    border_mask, center_mask = build_masks()
    if shift_type == "border_erase":
        Xs[:, border_mask] *= (1.0 - strength)
    elif shift_type == "center_erase":
        Xs[:, center_mask] *= (1.0 - strength)
    elif shift_type == "contrast":
        Xs = (1.0 + strength) * Xs
    else:
        raise ValueError(f"Unknown shift_type={shift_type}")
    return np.clip(Xs, 0.0, 16.0)


class MLP(nn.Module):
    def __init__(self, d_in: int = 64, h1: int = 128, h2: int = 64, d_out: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(d_in, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def compute_class_operator(X: np.ndarray, y: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """Between-class scatter style operator M in input space."""
    d = X.shape[1]
    mu = X.mean(axis=0)
    M = np.zeros((d, d), dtype=np.float64)
    n = len(y)
    for c in range(n_classes):
        idx = np.where(y == c)[0]
        if len(idx) == 0:
            continue
        pc = len(idx) / n
        muc = X[idx].mean(axis=0)
        diff = (muc - mu).reshape(-1, 1)
        M += pc * (diff @ diff.T)
    return M


def geometry_init_first_layer(fc1: nn.Linear, eigvecs: np.ndarray, eigvals: np.ndarray) -> None:
    """
    Initialize fc1 rows using top eigendirections of class operator.
    """
    out_dim, in_dim = fc1.weight.shape
    top = eigvecs[:, ::-1]  # descending
    vals = eigvals[::-1]
    k = min(in_dim, top.shape[1])

    W = np.zeros((out_dim, in_dim), dtype=np.float32)
    # Scale target similar to kaiming-normal std
    kaiming_std = np.sqrt(2.0 / in_dim)
    # Normalize eigvalue weights
    w_scale = vals[:k] / (vals[:k].max() + 1e-12)

    for i in range(out_dim):
        j = i % k
        sign = 1.0 if (i % 2 == 0) else -1.0
        vec = top[:, j]
        W[i] = (sign * vec * (0.5 + 0.5 * w_scale[j]) * kaiming_std).astype(np.float32)

    # Add small isotropic noise to avoid rank-locking
    W += np.random.normal(0.0, kaiming_std * 0.05, size=W.shape).astype(np.float32)

    with torch.no_grad():
        fc1.weight.copy_(torch.from_numpy(W))
        fc1.bias.zero_()


def standard_init(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.zeros_(m.bias)


@dataclass
class RunConfig:
    epochs: int = 15
    batch_size: int = 128
    lr: float = 1e-3
    seeds: int = 3
    reg_lambda: float = 2e-3
    reg_warmup_epochs: int = 6
    optimizer: str = "adam"
    trust_radius: float = 0.05
    beta2: float = 0.99
    lr_scale: float = 1.0
    chi2_adaptive: bool = True
    chi2_reject_tol: float = 0.02
    chi2_shrink: float = 0.5
    chi2_grow: float = 1.01
    chi2_min_radius: float = 1e-4
    chi2_max_radius: float = 1.0
    chi2_sigma: float = 3.0
    chi2_q_low: float = 0.05
    chi2_q_high: float = 0.5
    chi2_q_beta: float = 0.9
    chi2_per_layer: bool = True
    chi2_eta_shape: bool = True
    chi2_eta_proj_scale: float = 1.2
    chi2_eta_resid_scale: float = 0.8
    chi2_eta_dist_tau: float = 2.0
    chi2_eta_min_resid_weight: float = 0.05
    chi2_forget_guard: bool = True
    chi2_forget_tol: float = 0.01
    chi2_forget_shrink: float = 0.7
    eval_best_val: bool = True
    hybrid_chi2_scale: float = 0.3


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    tx = torch.from_numpy(X.astype(np.float32))
    ty = torch.from_numpy(y.astype(np.int64))
    ds = torch.utils.data.TensorDataset(tx, ty)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        tx = torch.from_numpy(X.astype(np.float32)).to(device)
        logits = model(tx)
        pred = logits.argmax(dim=1).cpu().numpy()
    return float((pred == y).mean())


def eval_loss_acc(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device, batch_size: int = 1024) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            xb = torch.from_numpy(X[i : i + batch_size].astype(np.float32)).to(device)
            yb = torch.from_numpy(y[i : i + batch_size].astype(np.int64)).to(device)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            total_loss += float(loss.item())
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.shape[0])
    return total_loss / max(total, 1), correct / max(total, 1)


def train_one(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    mode: str,
    cfg: RunConfig,
    seed: int,
    device: torch.device,
) -> dict:
    set_seed(seed)
    model = MLP().to(device)

    # Build geometry from source training data (standardized)
    M = compute_class_operator(X_train, y_train)
    eigvals, eigvecs = np.linalg.eigh(M)
    # Top-k projector for regularization
    k = min(16, eigvecs.shape[1])
    top = eigvecs[:, -k:]
    P_top = top @ top.T
    P_top_t = torch.from_numpy(P_top.astype(np.float32)).to(device)

    if mode == "baseline":
        standard_init(model)
    elif mode == "geo_system":
        standard_init(model)
        geometry_init_first_layer(model.fc1, eigvecs, eigvals)
    else:
        raise ValueError(mode)

    is_hybrid = cfg.optimizer == "adam_chi2_hybrid"
    is_eta_follow = cfg.optimizer == "adam_eta_follow"
    is_chi2_family = cfg.optimizer in ("chi2", "adam_chi2_hybrid")
    adam_opt = None
    chi2_opt = None

    if cfg.optimizer in ("adam", "adam_eta_follow"):
        opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    elif is_chi2_family:
        if cfg.chi2_per_layer:
            param_groups = [
                {"params": list(model.fc1.parameters()), "trust_radius": cfg.trust_radius, "name": "fc1"},
                {"params": list(model.fc2.parameters()), "trust_radius": cfg.trust_radius, "name": "fc2"},
                {"params": list(model.fc3.parameters()), "trust_radius": cfg.trust_radius, "name": "fc3"},
            ]
        else:
            param_groups = model.parameters()
        chi2_opt = Chi2TrustRegion(
            param_groups,
            trust_radius=cfg.trust_radius,
            beta2=cfg.beta2,
            lr_scale=(cfg.lr_scale * cfg.hybrid_chi2_scale) if is_hybrid else cfg.lr_scale,
        )
        if is_hybrid:
            adam_opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
            opt = chi2_opt
        else:
            opt = chi2_opt
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    train_loader = make_loader(X_train, y_train, cfg.batch_size, shuffle=True)

    history = []
    params = [p for p in model.parameters() if p.requires_grad]
    chi2_ctrl = {
        "count": 0,
        "mean_delta": 0.0,
        "m2_delta": 0.0,
        "q_ema": 0.0,
    }
    chi2_group_ctrl = {}
    best_val_acc = -1.0
    best_epoch = 0
    best_state = None

    def apply_eta_shaping() -> float | None:
        if (not cfg.chi2_eta_shape) or mode != "geo_system":
            return None
        g = model.fc1.weight.grad
        if g is None:
            return None
        g_proj = g @ P_top_t
        g_res = g - g_proj
        row_dist = torch.norm(g_res, dim=1, keepdim=True)
        row_proj_norm = torch.norm(g_proj, dim=1, keepdim=True)
        rel_dist = row_dist / (row_proj_norm + 1e-12)
        resid_weight = torch.exp(-cfg.chi2_eta_dist_tau * rel_dist)
        resid_weight = torch.clamp(resid_weight, min=cfg.chi2_eta_min_resid_weight, max=1.0)
        total = float(torch.sum(g * g).item())
        proj = float(torch.sum(g_proj * g_proj).item())
        g.copy_(cfg.chi2_eta_proj_scale * g_proj + cfg.chi2_eta_resid_scale * resid_weight * g_res)
        return proj / max(total, 1e-12)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        n_seen = 0
        n_correct = 0
        n_accept = 0
        n_reject = 0
        old_loss_sum = 0.0
        new_loss_sum = 0.0
        step_l2_sum = 0.0
        step_linf_max = 0.0
        denom_sum = 0.0
        scale_sum = 0.0
        diag_count = 0
        eta_grad_sum = 0.0
        eta_grad_count = 0
        tr_start_by_group = {
            str(g.get("name", f"group_{i}")): float(g.get("trust_radius", 0.0))
            for i, g in enumerate(opt.param_groups)
        }
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = F.cross_entropy(logits, yb)

            if mode == "geo_system" and epoch < cfg.reg_warmup_epochs:
                # Spectral shaping regularizer:
                # penalize components of first-layer rows outside top eigenspace
                W = model.fc1.weight  # [h1, d]
                W_proj = W @ P_top_t
                reg = torch.mean((W - W_proj) ** 2)
                loss = loss + cfg.reg_lambda * reg

            if is_chi2_family and cfg.chi2_adaptive:
                old_loss_val = float(loss.detach().item())
                base_ref_loss = old_loss_val

                if is_hybrid:
                    if adam_opt is None or chi2_opt is None:
                        raise RuntimeError("Hybrid optimizer state not initialized")
                    adam_opt.zero_grad()
                    chi2_opt.zero_grad()
                else:
                    opt.zero_grad()
                loss.backward()
                if is_chi2_family:
                    eta_ratio = apply_eta_shaping()
                    if eta_ratio is not None:
                        eta_grad_sum += eta_ratio
                        eta_grad_count += 1
                if is_hybrid:
                    if adam_opt is None or chi2_opt is None:
                        raise RuntimeError("Hybrid optimizer state not initialized")
                    adam_opt.step()
                    with torch.no_grad():
                        logits_adam = model(xb)
                        loss_adam = F.cross_entropy(logits_adam, yb)
                        if mode == "geo_system" and epoch < cfg.reg_warmup_epochs:
                            Wa = model.fc1.weight
                            Wa_proj = Wa @ P_top_t
                            reg_adam = torch.mean((Wa - Wa_proj) ** 2)
                            loss_adam = loss_adam + cfg.reg_lambda * reg_adam
                        base_ref_loss = float(loss_adam.item())
                    backups = [p.data.clone() for p in params]
                    chi2_opt.step()
                else:
                    backups = [p.data.clone() for p in params]
                    opt.step()

                with torch.no_grad():
                    logits_new = model(xb)
                    new_loss = F.cross_entropy(logits_new, yb)
                    if mode == "geo_system" and epoch < cfg.reg_warmup_epochs:
                        Wn = model.fc1.weight
                        Wn_proj = Wn @ P_top_t
                        reg_new = torch.mean((Wn - Wn_proj) ** 2)
                        new_loss = new_loss + cfg.reg_lambda * reg_new
                    new_loss_val = float(new_loss.item())

                stats = getattr(opt, "last_stats", None)
                if stats is not None:
                    step_l2_sum += float(stats.get("step_l2", 0.0))
                    step_linf_max = max(step_linf_max, float(stats.get("step_linf", 0.0)))
                    denom_sum += float(stats.get("denom_mean", 0.0))
                    scale_sum += float(stats.get("scale", 0.0))
                    diag_count += 1
                delta_loss = new_loss_val - base_ref_loss
                pred = float((stats or {}).get("pred_linear", 0.0))
                q = (base_ref_loss - new_loss_val) / max(pred, 1e-12)
                chi2_ctrl["q_ema"] = cfg.chi2_q_beta * chi2_ctrl["q_ema"] + (1.0 - cfg.chi2_q_beta) * q

                c = chi2_ctrl["count"]
                if c > 1:
                    var = chi2_ctrl["m2_delta"] / (c - 1)
                    std = float(np.sqrt(max(var, 1e-12)))
                else:
                    std = float("inf")
                noise_bound = chi2_ctrl["mean_delta"] + cfg.chi2_sigma * std + cfg.chi2_reject_tol * abs(base_ref_loss)

                hard_bad = delta_loss > max(0.2 * abs(base_ref_loss), 1e-3)
                soft_bad = (delta_loss > noise_bound) and (chi2_ctrl["q_ema"] < cfg.chi2_q_low)
                reject = (c >= 20 and soft_bad) or hard_bad

                per_group_stats = (stats or {}).get("groups", [])
                total_pred = float(sum(max(0.0, float(gs.get("pred_linear", 0.0))) for gs in per_group_stats))
                global_linf = float((stats or {}).get("step_linf", 0.0))

                if reject:
                    for p, b in zip(params, backups):
                        p.data.copy_(b)
                    n_reject += 1
                else:
                    n_accept += 1

                for gi, group in enumerate(opt.param_groups):
                    gname = str(group.get("name", f"group_{gi}"))
                    tr = float(group.get("trust_radius", cfg.trust_radius))
                    gs = next((x for x in per_group_stats if str(x.get("name", "")) == gname), None)
                    pred_i = float((gs or {}).get("pred_linear", 0.0))
                    linf_i = float((gs or {}).get("step_linf", 0.0))
                    pred_share = pred_i / max(total_pred, 1e-12)
                    linf_share = linf_i / max(global_linf, 1e-12)

                    if gname not in chi2_group_ctrl:
                        chi2_group_ctrl[gname] = {"q_ema": 0.0}

                    local_q = q / (1.0 + 0.5 * linf_share)
                    chi2_group_ctrl[gname]["q_ema"] = (
                        cfg.chi2_q_beta * chi2_group_ctrl[gname]["q_ema"]
                        + (1.0 - cfg.chi2_q_beta) * local_q
                    )

                    if reject:
                        local_shrink = cfg.chi2_shrink ** (1.0 + 0.5 * linf_share)
                        tr = max(cfg.chi2_min_radius, tr * local_shrink)
                    else:
                        local_grow = cfg.chi2_grow * (1.0 + 0.05 * pred_share)
                        if chi2_group_ctrl[gname]["q_ema"] > cfg.chi2_q_high:
                            local_grow *= 1.02
                        tr = min(cfg.chi2_max_radius, tr * local_grow)

                    group["trust_radius"] = tr

                old_loss_sum += base_ref_loss
                new_loss_sum += new_loss_val

                x = delta_loss
                chi2_ctrl["count"] += 1
                ncnt = chi2_ctrl["count"]
                d = x - chi2_ctrl["mean_delta"]
                chi2_ctrl["mean_delta"] += d / ncnt
                d2 = x - chi2_ctrl["mean_delta"]
                chi2_ctrl["m2_delta"] += d * d2
            else:
                if is_hybrid:
                    if adam_opt is None or chi2_opt is None:
                        raise RuntimeError("Hybrid optimizer state not initialized")
                    adam_opt.zero_grad()
                    chi2_opt.zero_grad()
                else:
                    opt.zero_grad()
                loss.backward()
                if is_chi2_family or is_eta_follow:
                    eta_ratio = apply_eta_shaping()
                    if eta_ratio is not None:
                        eta_grad_sum += eta_ratio
                        eta_grad_count += 1
                if is_hybrid:
                    if adam_opt is None or chi2_opt is None:
                        raise RuntimeError("Hybrid optimizer state not initialized")
                    adam_opt.step()
                    chi2_opt.step()
                else:
                    opt.step()

            bs = xb.shape[0]
            total_loss += float(loss.item()) * bs
            n_seen += bs
            with torch.no_grad():
                pred_tr = logits.argmax(dim=1)
                n_correct += int((pred_tr == yb).sum().item())

        train_loss = total_loss / max(n_seen, 1)
        train_acc = n_correct / max(n_seen, 1)
        val_loss, val_acc = eval_loss_acc(model, X_val, y_val, device)
        print(
            f"epoch {epoch + 1:02d}/{cfg.epochs:02d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        epoch_rec = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        guard_triggered = False
        if (
            is_chi2_family
            and cfg.chi2_forget_guard
            and (epoch + 1) >= 3
            and best_epoch > 0
            and val_acc < (best_val_acc - cfg.chi2_forget_tol)
        ):
            for group in opt.param_groups:
                tr = float(group.get("trust_radius", cfg.trust_radius))
                group["trust_radius"] = max(cfg.chi2_min_radius, tr * cfg.chi2_forget_shrink)
            guard_triggered = True
        if is_chi2_family:
            tr_end_by_group = {
                str(g.get("name", f"group_{i}")): float(g.get("trust_radius", 0.0))
                for i, g in enumerate(opt.param_groups)
            }
            epoch_rec.update(
                {
                    "chi2_accept": int(n_accept),
                    "chi2_reject": int(n_reject),
                    "chi2_accept_rate": float(n_accept / max(n_accept + n_reject, 1)),
                    "chi2_trust_radius_start": float(np.mean(list(tr_start_by_group.values()))),
                    "chi2_trust_radius_end": float(np.mean(list(tr_end_by_group.values()))),
                    "chi2_trust_radius_start_by_group": tr_start_by_group,
                    "chi2_trust_radius_end_by_group": tr_end_by_group,
                    "chi2_old_loss_mean": float(old_loss_sum / max(diag_count, 1)),
                    "chi2_new_loss_mean": float(new_loss_sum / max(diag_count, 1)),
                    "chi2_step_l2_mean": float(step_l2_sum / max(diag_count, 1)),
                    "chi2_step_linf_max": float(step_linf_max),
                    "chi2_denom_mean": float(denom_sum / max(diag_count, 1)),
                    "chi2_scale_mean": float(scale_sum / max(diag_count, 1)),
                    "chi2_delta_mean": float(chi2_ctrl["mean_delta"]),
                    "chi2_delta_std": float(np.sqrt(max(chi2_ctrl["m2_delta"] / max(chi2_ctrl["count"] - 1, 1), 0.0))),
                    "chi2_q_ema": float(chi2_ctrl["q_ema"]),
                    "chi2_q_ema_by_group": {k: float(v["q_ema"]) for k, v in chi2_group_ctrl.items()},
                    "chi2_eta_grad_mean": float(eta_grad_sum / max(eta_grad_count, 1)),
                    "chi2_forget_guard_triggered": bool(guard_triggered),
                    "chi2_hybrid": bool(is_hybrid),
                    "chi2_hybrid_scale": float(cfg.hybrid_chi2_scale),
                }
            )
        elif is_eta_follow:
            epoch_rec.update({"eta_grad_mean": float(eta_grad_sum / max(eta_grad_count, 1))})
        history.append(epoch_rec)

    # Final evaluations on clean + shifts
    clean_acc_last = accuracy(model, X_test, y_test, device)
    if cfg.eval_best_val and best_state is not None:
        model.load_state_dict(best_state)

    clean_acc = accuracy(model, X_test, y_test, device)

    X_border = scaler.transform(apply_shift(scaler.inverse_transform(X_test), "border_erase", 0.85))
    X_center = scaler.transform(apply_shift(scaler.inverse_transform(X_test), "center_erase", 0.85))
    X_contrast = scaler.transform(apply_shift(scaler.inverse_transform(X_test), "contrast", 0.30))

    acc_border = accuracy(model, X_border, y_test, device)
    acc_center = accuracy(model, X_center, y_test, device)
    acc_contrast = accuracy(model, X_contrast, y_test, device)
    robust_avg = (acc_border + acc_center + acc_contrast) / 3.0

    return {
        "mode": mode,
        "seed": seed,
        "best_epoch": int(best_epoch),
        "best_val_acc": float(best_val_acc),
        "selected_eval": "best_val" if cfg.eval_best_val else "last",
        "clean_acc_last": clean_acc_last,
        "clean_acc": clean_acc,
        "acc_border": acc_border,
        "acc_center": acc_center,
        "acc_contrast": acc_contrast,
        "robust_avg": robust_avg,
        "history": history,
    }


def summarize(runs: list[dict], mode: str) -> dict:
    sub = [r for r in runs if r["mode"] == mode]
    keys = ["clean_acc", "acc_border", "acc_center", "acc_contrast", "robust_avg"]
    out = {}
    for k in keys:
        vals = np.array([r[k] for r in sub], dtype=float)
        out[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=0))}
    # Early speed proxy: val acc at epoch 3 and 5
    for ep in [3, 5]:
        vals = []
        for r in sub:
            h = r["history"]
            if len(h) >= ep:
                vals.append(h[ep - 1]["val_acc"])
        if len(vals) == 0:
            out[f"val_acc_ep{ep}"] = {"mean": float("nan"), "std": float("nan")}
        else:
            arr = np.array(vals, dtype=float)
            out[f"val_acc_ep{ep}"] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=0))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--reg-lambda", type=float, default=2e-3)
    parser.add_argument("--reg-warmup-epochs", type=int, default=6)
    parser.add_argument("--optimizer", type=str, choices=["adam", "chi2", "adam_chi2_hybrid", "adam_eta_follow"], default="adam")
    parser.add_argument("--trust-radius", type=float, default=0.05)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--lr-scale", type=float, default=1.0)
    parser.add_argument("--chi2-adaptive", action="store_true", default=True)
    parser.add_argument("--no-chi2-adaptive", action="store_false", dest="chi2_adaptive")
    parser.add_argument("--chi2-reject-tol", type=float, default=0.02)
    parser.add_argument("--chi2-shrink", type=float, default=0.5)
    parser.add_argument("--chi2-grow", type=float, default=1.01)
    parser.add_argument("--chi2-min-radius", type=float, default=1e-4)
    parser.add_argument("--chi2-max-radius", type=float, default=1.0)
    parser.add_argument("--chi2-sigma", type=float, default=3.0)
    parser.add_argument("--chi2-q-low", type=float, default=0.05)
    parser.add_argument("--chi2-q-high", type=float, default=0.5)
    parser.add_argument("--chi2-q-beta", type=float, default=0.9)
    parser.add_argument("--chi2-per-layer", action="store_true", default=True)
    parser.add_argument("--no-chi2-per-layer", action="store_false", dest="chi2_per_layer")
    parser.add_argument("--chi2-eta-shape", action="store_true", default=True)
    parser.add_argument("--no-chi2-eta-shape", action="store_false", dest="chi2_eta_shape")
    parser.add_argument("--chi2-eta-proj-scale", type=float, default=1.2)
    parser.add_argument("--chi2-eta-resid-scale", type=float, default=0.8)
    parser.add_argument("--chi2-eta-dist-tau", type=float, default=2.0)
    parser.add_argument("--chi2-eta-min-resid-weight", type=float, default=0.05)
    parser.add_argument("--chi2-forget-guard", action="store_true", default=True)
    parser.add_argument("--no-chi2-forget-guard", action="store_false", dest="chi2_forget_guard")
    parser.add_argument("--chi2-forget-tol", type=float, default=0.01)
    parser.add_argument("--chi2-forget-shrink", type=float, default=0.7)
    parser.add_argument("--eval-best-val", action="store_true", default=True)
    parser.add_argument("--eval-last", action="store_false", dest="eval_best_val")
    parser.add_argument("--hybrid-chi2-scale", type=float, default=0.3)
    parser.add_argument("--out", type=str, default="tier3_digits_results.json")
    args = parser.parse_args()

    cfg = RunConfig(
        epochs=args.epochs,
        seeds=args.seeds,
        batch_size=args.batch_size,
        lr=args.lr,
        reg_lambda=args.reg_lambda,
        reg_warmup_epochs=args.reg_warmup_epochs,
        optimizer=args.optimizer,
        trust_radius=args.trust_radius,
        beta2=args.beta2,
        lr_scale=args.lr_scale,
        chi2_adaptive=args.chi2_adaptive,
        chi2_reject_tol=args.chi2_reject_tol,
        chi2_shrink=args.chi2_shrink,
        chi2_grow=args.chi2_grow,
        chi2_min_radius=args.chi2_min_radius,
        chi2_max_radius=args.chi2_max_radius,
        chi2_sigma=args.chi2_sigma,
        chi2_q_low=args.chi2_q_low,
        chi2_q_high=args.chi2_q_high,
        chi2_q_beta=args.chi2_q_beta,
        chi2_per_layer=args.chi2_per_layer,
        chi2_eta_shape=args.chi2_eta_shape,
        chi2_eta_proj_scale=args.chi2_eta_proj_scale,
        chi2_eta_resid_scale=args.chi2_eta_resid_scale,
        chi2_eta_dist_tau=args.chi2_eta_dist_tau,
        chi2_eta_min_resid_weight=args.chi2_eta_min_resid_weight,
        chi2_forget_guard=args.chi2_forget_guard,
        chi2_forget_tol=args.chi2_forget_tol,
        chi2_forget_shrink=args.chi2_forget_shrink,
        eval_best_val=args.eval_best_val,
        hybrid_chi2_scale=args.hybrid_chi2_scale,
    )

    device = get_device()
    print(f"Device: {device}")
    print(f"Optimizer: {cfg.optimizer}")

    data = load_digits()
    X = data.data.astype(np.float64)
    y = data.target.astype(np.int64)

    # train/val/test split from source mu
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp
    )

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_val_s = scaler.transform(X_val)
    X_te_s = scaler.transform(X_te)

    runs = []
    for seed in range(cfg.seeds):
        for mode in ["baseline", "geo_system"]:
            print(f"Running mode={mode}, seed={seed}")
            res = train_one(
                X_train=X_tr_s,
                y_train=y_tr,
                X_val=X_val_s,
                y_val=y_val,
                X_test=X_te_s,
                y_test=y_te,
                scaler=scaler,
                mode=mode,
                cfg=cfg,
                seed=seed,
                device=device,
            )
            runs.append(res)

    base = summarize(runs, "baseline")
    geo = summarize(runs, "geo_system")

    print("\n=== Summary (mean +- std across seeds) ===")
    for key in ["val_acc_ep3", "val_acc_ep5", "clean_acc", "robust_avg", "acc_border", "acc_center", "acc_contrast"]:
        b = base[key]
        g = geo[key]
        print(
            f"{key:12s} | baseline {b['mean']:.4f} +- {b['std']:.4f} "
            f"| geo_system {g['mean']:.4f} +- {g['std']:.4f} "
            f"| delta {g['mean'] - b['mean']:+.4f}"
        )

    result = {
        "config": vars(args),
        "device": str(device),
        "baseline": base,
        "geo_system": geo,
        "runs": runs,
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nWrote results: {out_path}")


if __name__ == "__main__":
    main()
