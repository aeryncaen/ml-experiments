#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_data(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(TINY_SHAKESPEARE_URL, timeout=30) as r:
        txt = r.read().decode("utf-8")
    path.write_text(txt, encoding="utf-8")
    return txt


def build_vocab(text: str):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    return np.array([stoi[c] for c in text], dtype=np.int64)


def compute_token_operator(tokens: np.ndarray, vocab_size: int) -> np.ndarray:
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float64)
    if len(tokens) > 1:
        a = tokens[:-1]
        b = tokens[1:]
        np.add.at(counts, (a, b), 1.0)
    sym = 0.5 * (counts + counts.T)
    total = np.sum(sym)
    if total > 0:
        sym = sym / total
    return sym


def compute_bucket_basis(tokens: np.ndarray, vocab_size: int, width: int) -> np.ndarray:
    # Build width token-distribution vectors from deterministic context buckets.
    # Output basis shape: [vocab_size, width].
    # Bucket identity is derived from short context (prev2, prev1), then mapped to width.
    if len(tokens) < 2:
        return np.zeros((vocab_size, width), dtype=np.float32)
    basis = np.zeros((vocab_size, width), dtype=np.float64)
    cur = tokens[2:]
    prev1 = tokens[1:-1]
    prev2 = tokens[:-2]
    context_id = prev1.astype(np.int64) + vocab_size * prev2.astype(np.int64)
    buckets = context_id % width
    np.add.at(basis, (cur, buckets), 1.0)

    col_sum = basis.sum(axis=0, keepdims=True)
    basis = basis / np.maximum(col_sum, 1.0)
    basis = basis - basis.mean(axis=0, keepdims=True)
    col_norm = np.linalg.norm(basis, axis=0, keepdims=True)
    basis = basis / np.maximum(col_norm, 1e-12)
    return basis.astype(np.float32)


def compute_kl_bucket_basis(tokens: np.ndarray, vocab_size: int, width: int, eps: float = 1e-8) -> np.ndarray:
    # KL-aligned bucket basis: log-ratio features over deterministic context buckets.
    # basis[t, b] ~= log P(token=t | bucket=b) - log P(token=t)
    if len(tokens) < 2:
        return np.zeros((vocab_size, width), dtype=np.float32)

    counts = np.zeros((vocab_size, width), dtype=np.float64)
    cur = tokens[2:]
    prev1 = tokens[1:-1]
    prev2 = tokens[:-2]
    context_id = prev1.astype(np.int64) + vocab_size * prev2.astype(np.int64)
    buckets = context_id % width
    np.add.at(counts, (cur, buckets), 1.0)

    # P(token | bucket)
    col_sum = counts.sum(axis=0, keepdims=True)
    p_t_given_b = counts / np.maximum(col_sum, 1.0)

    # P(token)
    token_sum = counts.sum(axis=1, keepdims=True)
    p_t = token_sum / np.maximum(np.sum(counts), 1.0)

    basis = np.log(p_t_given_b + eps) - np.log(p_t + eps)
    basis = basis - basis.mean(axis=0, keepdims=True)
    col_norm = np.linalg.norm(basis, axis=0, keepdims=True)
    basis = basis / np.maximum(col_norm, 1e-12)
    return basis.astype(np.float32)


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask, need_weights=False)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_head: int, n_layer: int, block_size: int, dropout: float):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, block_size, d_model))
        self.blocks = nn.ModuleList([Block(d_model, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        x = self.token_emb(idx) + self.pos_emb[:, :t, :]
        mask = torch.triu(torch.ones(t, t, device=idx.device, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


@dataclass
class RunConfig:
    epochs: int = 5
    steps_per_epoch: int = 300
    eval_iters: int = 50
    batch_size: int = 64
    block_size: int = 128
    d_model: int = 128
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    lr: float = 3e-4
    seeds: int = 3
    optimizer: str = "adam"
    loss_type: str = "ce"
    loss_eps: float = 1e-6
    chi2_tail_threshold: float = 1e-4
    eta_shape: bool = True
    eta_dist_tau: float = 2.0
    eta_min_resid_weight: float = 0.05
    eta_topk: int = 16
    geo_init_method: str = "bucket"
    geo_init_blend: float = 0.7


def get_batch(data: np.ndarray, batch_size: int, block_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    x = np.stack([data[i : i + block_size] for i in ix], axis=0)
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix], axis=0)
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def token_loss(logits: torch.Tensor, y: torch.Tensor, loss_type: str, eps: float) -> torch.Tensor:
    b, t, v = logits.shape
    lflat = logits.reshape(b * t, v)
    yflat = y.reshape(b * t)
    if loss_type == "ce":
        return F.cross_entropy(lflat, yflat)
    if loss_type == "chi2":
        p = torch.softmax(lflat, dim=1)
        py = p.gather(1, yflat.view(-1, 1)).squeeze(1)
        return torch.mean(1.0 / torch.clamp(py, min=eps) - 1.0)
    if loss_type == "chi2_log2":
        p = torch.softmax(lflat, dim=1)
        py = p.gather(1, yflat.view(-1, 1)).squeeze(1)
        return torch.mean(torch.log2(1.0 / torch.clamp(py, min=eps)))
    raise ValueError(loss_type)


def eval_stats(
    model: nn.Module,
    data: np.ndarray,
    cfg: RunConfig,
    device: torch.device,
) -> dict:
    model.eval()
    losses = []
    accs = []
    inv_py_sum = 0.0
    inv_py_max = 0.0
    low_count = 0
    n_tok = 0
    with torch.no_grad():
        for _ in range(cfg.eval_iters):
            xb, yb = get_batch(data, cfg.batch_size, cfg.block_size, device)
            logits = model(xb)
            losses.append(float(token_loss(logits, yb, cfg.loss_type, cfg.loss_eps).item()))
            pred = logits.argmax(dim=-1)
            accs.append(float((pred == yb).float().mean().item()))
            if cfg.loss_type in ("chi2", "chi2_log2"):
                lflat = logits.reshape(-1, logits.size(-1))
                yflat = yb.reshape(-1)
                p = torch.softmax(lflat, dim=1)
                py = p.gather(1, yflat.view(-1, 1)).squeeze(1)
                inv = 1.0 / torch.clamp(py, min=cfg.loss_eps)
                inv_py_sum += float(torch.sum(inv).item())
                inv_py_max = max(inv_py_max, float(torch.max(inv).item()))
                low_count += int((py < cfg.chi2_tail_threshold).sum().item())
                n_tok += int(yflat.shape[0])
    out = {
        "loss": float(np.mean(losses)),
        "acc": float(np.mean(accs)),
        "ppl": float(math.exp(min(np.mean(losses), 20.0))) if cfg.loss_type == "ce" else float("nan"),
    }
    if cfg.loss_type in ("chi2", "chi2_log2"):
        out.update(
            {
                "inv_py_mean": inv_py_sum / max(n_tok, 1),
                "inv_py_max": inv_py_max,
                "py_low_frac": low_count / max(n_tok, 1),
            }
        )
    return out


def train_one(train_ids: np.ndarray, val_ids: np.ndarray, vocab_size: int, geo_basis: np.ndarray, mode: str, cfg: RunConfig, seed: int, device: torch.device) -> dict:
    set_seed(seed)
    model = TinyGPT(vocab_size, cfg.d_model, cfg.n_head, cfg.n_layer, cfg.block_size, cfg.dropout).to(device)
    if mode == "geo_system":
        with torch.no_grad():
            E = model.token_emb.weight
            k = min(E.shape[1], geo_basis.shape[1])
            B = torch.from_numpy(geo_basis[:, :k]).to(device=device, dtype=E.dtype)
            E[:, :k] = (1.0 - cfg.geo_init_blend) * E[:, :k] + cfg.geo_init_blend * B
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if geo_basis.shape[1] >= cfg.d_model:
        P_top_t = torch.from_numpy(geo_basis[:, : cfg.d_model].astype(np.float32)).to(device)
    else:
        pad = np.zeros((vocab_size, cfg.d_model), dtype=np.float32)
        pad[:, : geo_basis.shape[1]] = geo_basis
        P_top_t = torch.from_numpy(pad).to(device)
    history = []
    best_val = float("inf")
    best_epoch = 0
    best_state = None

    for ep in range(cfg.epochs):
        model.train()
        train_loss_sum = 0.0
        train_acc_sum = 0.0
        eta_raw_sum = 0.0
        eta_shaped_sum = 0.0
        eta_w_sum = 0.0
        eta_cnt = 0
        for _ in range(cfg.steps_per_epoch):
            xb, yb = get_batch(train_ids, cfg.batch_size, cfg.block_size, device)
            logits = model(xb)
            loss = token_loss(logits, yb, cfg.loss_type, cfg.loss_eps)
            opt.zero_grad()
            loss.backward()

            if cfg.optimizer == "adam_eta_follow" and cfg.eta_shape:
                g = model.token_emb.weight.grad
                if g is not None:
                    g_proj = P_top_t @ g
                    g_res = g - g_proj
                    row_dist = torch.norm(g_res, dim=1, keepdim=True)
                    row_proj = torch.norm(g_proj, dim=1, keepdim=True)
                    rel = row_dist / (row_proj + 1e-12)
                    w = torch.exp(-cfg.eta_dist_tau * rel)
                    w = torch.clamp(w, min=cfg.eta_min_resid_weight, max=1.0)
                    total = float(torch.sum(g * g).item())
                    proj = float(torch.sum(g_proj * g_proj).item())
                    g.copy_(g_proj + w * g_res)
                    g2 = model.token_emb.weight.grad
                    if g2 is None:
                        g2 = g
                    g2_proj = P_top_t @ g2
                    total2 = float(torch.sum(g2 * g2).item())
                    proj2 = float(torch.sum(g2_proj * g2_proj).item())
                    eta_raw_sum += proj / max(total, 1e-12)
                    eta_shaped_sum += proj2 / max(total2, 1e-12)
                    eta_w_sum += float(torch.mean(w).item())
                    eta_cnt += 1

            opt.step()
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                train_acc_sum += float((pred == yb).float().mean().item())
            train_loss_sum += float(loss.item())

        train_loss = train_loss_sum / cfg.steps_per_epoch
        train_acc = train_acc_sum / cfg.steps_per_epoch
        val = eval_stats(model, val_ids, cfg, device)
        rec = {
            "epoch": ep + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["acc"],
            "val_ppl": val["ppl"],
        }
        if cfg.loss_type in ("chi2", "chi2_log2"):
            rec.update(
                {
                    "chi2_val_inv_py_mean": val["inv_py_mean"],
                    "chi2_val_inv_py_max": val["inv_py_max"],
                    "chi2_val_py_low_frac": val["py_low_frac"],
                }
            )
        if eta_cnt > 0:
            rec.update(
                {
                    "eta_grad_raw_mean": eta_raw_sum / eta_cnt,
                    "eta_grad_shaped_mean": eta_shaped_sum / eta_cnt,
                    "eta_resid_weight_mean": eta_w_sum / eta_cnt,
                }
            )
        history.append(rec)
        print(
            f"epoch {ep + 1:02d}/{cfg.epochs:02d} train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val['loss']:.4f} val_acc={val['acc']:.4f}"
        )
        if cfg.loss_type in ("chi2", "chi2_log2"):
            print(
                f"  chi2_tail val_inv_py_mean={val['inv_py_mean']:.3f} val_inv_py_max={val['inv_py_max']:.3f} "
                f"val_py_low_frac={val['py_low_frac']:.4f}"
            )
        if val["loss"] < best_val:
            best_val = float(val["loss"])
            best_epoch = ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final = eval_stats(model, val_ids, cfg, device)
    return {
        "mode": mode,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "final_val_loss": final["loss"],
        "final_val_acc": final["acc"],
        "final_val_ppl": final["ppl"],
        "history": history,
    }


def summarize(runs: list[dict], mode: str) -> dict:
    sub = [r for r in runs if r["mode"] == mode]
    out = {}
    for k in ["final_val_loss", "final_val_acc", "final_val_ppl"]:
        vals = np.array([r[k] for r in sub], dtype=float)
        out[k] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}
    for ep in [3, 5]:
        vals_loss = []
        vals_acc = []
        for r in sub:
            h = r["history"]
            if len(h) >= ep:
                vals_loss.append(h[ep - 1]["val_loss"])
                vals_acc.append(h[ep - 1]["val_acc"])
        out[f"val_loss_ep{ep}"] = {
            "mean": float(np.mean(vals_loss)) if vals_loss else float("nan"),
            "std": float(np.std(vals_loss)) if vals_loss else float("nan"),
        }
        out[f"val_acc_ep{ep}"] = {
            "mean": float(np.mean(vals_acc)) if vals_acc else float("nan"),
            "std": float(np.std(vals_acc)) if vals_acc else float("nan"),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", type=str, default="./data/tinyshakespeare/input.txt")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--steps-per-epoch", type=int, default=300)
    p.add_argument("--eval-iters", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-head", type=int, default=4)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--optimizer", type=str, choices=["adam", "adam_eta_follow"], default="adam")
    p.add_argument("--loss-type", type=str, choices=["ce", "chi2", "chi2_log2"], default="ce")
    p.add_argument("--loss-eps", type=float, default=1e-6)
    p.add_argument("--chi2-tail-threshold", type=float, default=1e-4)
    p.add_argument("--eta-shape", action="store_true", default=True)
    p.add_argument("--no-eta-shape", action="store_false", dest="eta_shape")
    p.add_argument("--eta-dist-tau", type=float, default=2.0)
    p.add_argument("--eta-min-resid-weight", type=float, default=0.05)
    p.add_argument("--eta-topk", type=int, default=16)
    p.add_argument("--geo-init-method", type=str, choices=["bucket", "kl_bucket", "eig"], default="bucket")
    p.add_argument("--geo-init-blend", type=float, default=0.3)
    p.add_argument("--out", type=str, default="tier4_shakespeare_results.json")
    args = p.parse_args()

    cfg = RunConfig(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        eval_iters=args.eval_iters,
        batch_size=args.batch_size,
        block_size=args.block_size,
        d_model=args.d_model,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        lr=args.lr,
        seeds=args.seeds,
        optimizer=args.optimizer,
        loss_type=args.loss_type,
        loss_eps=args.loss_eps,
        chi2_tail_threshold=args.chi2_tail_threshold,
        eta_shape=args.eta_shape,
        eta_dist_tau=args.eta_dist_tau,
        eta_min_resid_weight=args.eta_min_resid_weight,
        eta_topk=args.eta_topk,
        geo_init_method=args.geo_init_method,
        geo_init_blend=args.geo_init_blend,
    )

    device = get_device()
    print(f"Device: {device}")
    print(f"Optimizer: {cfg.optimizer} | Loss: {cfg.loss_type}")

    text = ensure_data(Path(args.data_path))
    stoi, _ = build_vocab(text)
    ids = encode(text, stoi)
    n = len(ids)
    n_train = int(0.9 * n)
    train_ids = ids[:n_train]
    val_ids = ids[n_train:]
    vocab_size = len(stoi)
    print(f"Chars={n} vocab={vocab_size} train={len(train_ids)} val={len(val_ids)}")

    op = compute_token_operator(train_ids, vocab_size)
    if cfg.geo_init_method == "eig":
        _, eigvecs = np.linalg.eigh(op)
        geo_basis = eigvecs[:, -cfg.d_model :].astype(np.float32)
    elif cfg.geo_init_method == "kl_bucket":
        geo_basis = compute_kl_bucket_basis(train_ids, vocab_size, cfg.d_model)
    else:
        geo_basis = compute_bucket_basis(train_ids, vocab_size, cfg.d_model)

    runs = []
    for seed in range(cfg.seeds):
        for mode in ["baseline", "geo_system"]:
            print(f"Running mode={mode}, seed={seed}")
            res = train_one(train_ids, val_ids, vocab_size, geo_basis, mode, cfg, seed, device)
            runs.append(res)

    base = summarize(runs, "baseline")
    geo = summarize(runs, "geo_system")
    print("\n=== Summary (mean +- std across seeds) ===")
    for key in [
        "val_loss_ep3",
        "val_acc_ep3",
        "val_loss_ep5",
        "val_acc_ep5",
        "final_val_loss",
        "final_val_acc",
        "final_val_ppl",
    ]:
        b = base[key]
        g = geo[key]
        print(f"{key:14s} | baseline {b['mean']:.4f} +- {b['std']:.4f} | geo_system {g['mean']:.4f} +- {g['std']:.4f} | delta {g['mean'] - b['mean']:+.4f}")

    out = {
        "config": vars(args),
        "device": str(device),
        "baseline": base,
        "geo_system": geo,
        "runs": runs,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote results: {out_path}")


if __name__ == "__main__":
    main()
