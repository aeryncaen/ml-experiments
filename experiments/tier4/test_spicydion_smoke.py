#!/usr/bin/env python3
"""Head-to-head comparison: SpicyDion vs Muon vs AdamW on a MiniGPT model.

Model: d_model=256, 4 layers, 4 heads, vocab=512, seq_len=128
Data:  Deterministic sequences with learnable structure (shifted copy task)
Steps: 500
All optimizers get the same cosine LR schedule.
"""
import math
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from experiments.tier4.spicydion import SpicyDion
from experiments.tier4.muon import SingleDeviceMuonWithAuxAdam
from experiments.tier4.normuon import SingleDeviceNorMuonWithAuxAdam
from experiments.tier4.autonormuon import AutoNorMuon
from dion.dion2 import Dion2
from experiments.tier4.spicy_adam import SpicyAdam
from experiments.tier4.gnorm_scheduler import GnormScheduler


# ── Model ──────────────────────────────────────────────────────────────────

class MiniGPT(nn.Module):
    def __init__(self, vocab=512, d_model=256, n_heads=4, n_layers=4, seq_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                "attn_qkv": nn.Linear(d_model, 3 * d_model, bias=False),
                "attn_out": nn.Linear(d_model, d_model, bias=False),
                "ln1": nn.LayerNorm(d_model),
                "ffn_up": nn.Linear(d_model, 4 * d_model, bias=False),
                "ffn_down": nn.Linear(4 * d_model, d_model, bias=False),
                "ln2": nn.LayerNorm(d_model),
            }))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.n_heads = n_heads
        self.d_model = d_model
        self.seq_len = seq_len
        self._init_weights(n_layers)

    def _init_weights(self, n_layers):
        """Scaled init: N(0, 1/sqrt(d_model)) for most, residual scaling for output projections."""
        init_std = 1.0 / math.sqrt(self.d_model)
        residual_std = init_std / math.sqrt(2 * n_layers)
        # Embeddings
        nn.init.normal_(self.embed.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.pos.weight, mean=0.0, std=init_std)
        # Transformer layers
        for layer in self.layers:
            nn.init.normal_(layer["attn_qkv"].weight, mean=0.0, std=init_std)
            nn.init.normal_(layer["attn_out"].weight, mean=0.0, std=residual_std)
            nn.init.normal_(layer["ffn_up"].weight, mean=0.0, std=init_std)
            nn.init.normal_(layer["ffn_down"].weight, mean=0.0, std=residual_std)
        # Output head
        nn.init.normal_(self.head.weight, mean=0.0, std=init_std)

    def forward(self, x):
        B, T = x.shape
        h = self.embed(x) + self.pos(torch.arange(T, device=x.device))
        for layer in self.layers:
            r = layer["ln1"](h)
            qkv = layer["attn_qkv"](r)
            q, k, v = qkv.chunk(3, dim=-1)
            nh, dk = self.n_heads, self.d_model // self.n_heads
            q = q.view(B, T, nh, dk).transpose(1, 2)
            k = k.view(B, T, nh, dk).transpose(1, 2)
            v = v.view(B, T, nh, dk).transpose(1, 2)
            att = (q @ k.transpose(-2, -1)) * (dk ** -0.5)
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            att = att.masked_fill(mask, float('-inf'))
            att = F.softmax(att, dim=-1)
            out = (att @ v).transpose(1, 2).contiguous().view(B, T, -1)
            h = h + layer["attn_out"](out)
            r = layer["ln2"](h)
            h = h + layer["ffn_down"](F.gelu(layer["ffn_up"](r)))
        return self.head(self.ln_f(h))

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ── Data ───────────────────────────────────────────────────────────────────

def make_batch(batch_size, seq_len, vocab, device, rng):
    """Copy-with-shift task: first half random, second half = first + offset."""
    half = seq_len // 2
    prefix = torch.randint(0, vocab // 2, (batch_size, half), generator=rng, device=device)
    shift = torch.randint(1, 8, (batch_size, 1), generator=rng, device=device)
    suffix = (prefix + shift) % vocab
    x = torch.cat([prefix, suffix], dim=1)
    return x[:, :-1], x[:, 1:]


# ── Cosine LR helper ──────────────────────────────────────────────────────

def cosine_lr(base_lr, step, total_steps):
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))

def pressure_lr(base_lr, step, total_steps, k=4.0):
    return base_lr / (1.0 + k * step / total_steps)


# ── Optimizer factories ───────────────────────────────────────────────────

def split_params(model):
    muon_params, adam_params = [], []
    for name, p in model.named_parameters():
        if p.ndim >= 2 and "embed" not in name and "head" not in name and "ln" not in name:
            muon_params.append(p)
        else:
            adam_params.append(p)
    return muon_params, adam_params


class SpicyDionWithAdam:
    """Wrapper: SpicyDion for 2D params + plain AdamW for the rest."""

    def __init__(self, model, muon_lr=0.02, adam_lr=3e-4, fraction=1.0, total_steps=2000):
        muon_p, adam_p = split_params(model)
        self.spicy = SpicyDion(
            muon_p,
            distributed_mesh=None,
            lr=muon_lr,
            fraction=fraction,
            ef_decay=0.95,
            weight_decay=0,
            adjust_lr="spectral_norm",
            total_steps=total_steps,
        )
        self.adam = torch.optim.AdamW(adam_p, lr=adam_lr, weight_decay=0)
        self.param_groups = self.spicy.param_groups + self.adam.param_groups

    def zero_grad(self):
        self.spicy.zero_grad()
        self.adam.zero_grad()

    def step(self):
        self.spicy.step()
        self.adam.step()

    def effective_muon_lr(self):
        vals = []
        for p in self.spicy.param_groups[0]["params"]:
            st = self.spicy.state.get(p, None)
            if not st:
                continue
            v = st.get("gnorm_last_lr")
            if isinstance(v, torch.Tensor):
                vals.append(float(v.detach().item()))
        if not vals:
            return float(self.spicy.param_groups[0]["lr"])
        return float(sum(vals) / len(vals))

    def gnorm_stats(self, step=0, total_steps=2000):
        ratios = []
        for p in self.spicy.param_groups[0]["params"]:
            st = self.spicy.state.get(p, None)
            if not st:
                continue
            ema = st.get("gnorm_ema")
            gmax = st.get("gnorm_max")
            if isinstance(ema, torch.Tensor) and isinstance(gmax, torch.Tensor) and ema.numel() > 0:
                ratio = ema / gmax.clamp(min=1e-12)
                ratios.append(ratio.reshape(-1).float())
        if not ratios:
            return None
        r = torch.cat(ratios)
        # Compute per-neuron LR using the same pressure formula as the optimizer
        k = 1.50  # must match spicydion.py
        base_lr = float(self.spicy.param_groups[0].get("initial_lr",
                         self.spicy.param_groups[0]["lr"]))
        pressure = k * (step / total_steps)
        lr_per_neuron = (base_lr * r * (r / (r + pressure).clamp(min=1e-12))).clamp(min=1e-3 * base_lr)
        pcts = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        q = torch.tensor(pcts, device=r.device)
        r_pct = torch.quantile(r, q)
        lr_pct = torch.quantile(lr_per_neuron, q)
        return {
            "n_neurons": int(r.numel()),
            "ratio_p0": float(r_pct[0]), "ratio_p10": float(r_pct[1]),
            "ratio_p25": float(r_pct[2]), "ratio_p50": float(r_pct[3]),
            "ratio_p75": float(r_pct[4]), "ratio_p90": float(r_pct[5]),
            "ratio_p100": float(r_pct[6]),
            "lr_p0": float(lr_pct[0]), "lr_p10": float(lr_pct[1]),
            "lr_p25": float(lr_pct[2]), "lr_p50": float(lr_pct[3]),
            "lr_p75": float(lr_pct[4]), "lr_p90": float(lr_pct[5]),
            "lr_p100": float(lr_pct[6]),
            "pressure": pressure,
        }

    def grad_norm_stats(self):
        """Raw gradient norm stats for SpicyDion muon params."""
        mat_l2 = []
        row_l2 = []
        for p in self.spicy.param_groups[0]["params"]:
            g = p.grad
            if g is None:
                continue
            gf = g.detach().float()
            mat_l2.append(gf.norm().reshape(1))
            if gf.ndim >= 2:
                row_l2.append(gf.norm(dim=-1).reshape(-1))
        if not mat_l2:
            return None
        mat = torch.cat(mat_l2)
        out = {
            "grad_mat_l2_med": float(mat.median().item()),
            "grad_mat_l2_mean": float(mat.mean().item()),
            "grad_mat_l2_max": float(mat.max().item()),
        }
        if row_l2:
            rows = torch.cat(row_l2)
            out["grad_row_l2_med"] = float(rows.median().item())
            out["grad_row_l2_mean"] = float(rows.mean().item())
            out["grad_row_l2_max"] = float(rows.max().item())
        return out


class SpicyDionWithScheduler:
    """Wrapper: SpicyDion + GnormScheduler for self-scheduled LR."""

    def __init__(self, model, muon_lr=0.02, adam_lr=3e-4, fraction=1.0, total_steps=2000):
        muon_p, adam_p = split_params(model)
        self.spicy = SpicyDion(
            muon_p,
            distributed_mesh=None,
            lr=muon_lr,
            fraction=fraction,
            ef_decay=0.95,
            weight_decay=0,
            adjust_lr="spectral_norm",
            total_steps=total_steps,
        )
        self.adam = torch.optim.AdamW(adam_p, lr=adam_lr, weight_decay=0)
        self.param_groups = self.spicy.param_groups + self.adam.param_groups
        self.scheduler = GnormScheduler(base_lr=muon_lr, n_layers=4)
        self._adam_lr = adam_lr
        self._step_num = 0

    def zero_grad(self):
        self.spicy.zero_grad()
        self.adam.zero_grad()

    def step(self):
        # Compute grad norm for scheduler
        all_params = []
        for g in self.param_groups:
            all_params.extend(g["params"])
        gnorm = torch.cat([p.grad.flatten() for p in all_params if p.grad is not None]).norm().item()

        # Run scheduler
        sched = self.scheduler.step(gnorm, self._step_num)
        sched_lr = sched["lr"]
        aa = 1.0 if sched["adaptive_active"] else 0.0

        # Set LR on param groups
        for pg in self.spicy.param_groups:
            pg["lr"] = sched_lr
            pg["scheduled_lr"] = sched_lr
            pg["_adaptive_active"] = aa
        for pg in self.adam.param_groups:
            pg["lr"] = sched_lr / 3.0

        self.spicy.step()
        self.adam.step()
        self._step_num += 1

    def effective_muon_lr(self):
        return self.scheduler.current_lr

    def gnorm_stats(self, step=0, total_steps=2000):
        ratios = []
        for p in self.spicy.param_groups[0]["params"]:
            st = self.spicy.state.get(p, None)
            if not st:
                continue
            ema = st.get("gnorm_ema")
            gmax = st.get("gnorm_max")
            if isinstance(ema, torch.Tensor) and isinstance(gmax, torch.Tensor) and ema.numel() > 0:
                ratio = ema / gmax.clamp(min=1e-12)
                ratios.append(ratio.reshape(-1).float())
        if not ratios:
            return None
        r = torch.cat(ratios)
        base_lr = self.scheduler.current_lr
        lr_per_neuron = base_lr * r
        pcts = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        q = torch.tensor(pcts, device=r.device)
        r_pct = torch.quantile(r, q)
        lr_pct = torch.quantile(lr_per_neuron, q)
        return {
            "n_neurons": int(r.numel()),
            "ratio_p0": float(r_pct[0]), "ratio_p10": float(r_pct[1]),
            "ratio_p25": float(r_pct[2]), "ratio_p50": float(r_pct[3]),
            "ratio_p75": float(r_pct[4]), "ratio_p90": float(r_pct[5]),
            "ratio_p100": float(r_pct[6]),
            "lr_p0": float(lr_pct[0]), "lr_p10": float(lr_pct[1]),
            "lr_p25": float(lr_pct[2]), "lr_p50": float(lr_pct[3]),
            "lr_p75": float(lr_pct[4]), "lr_p90": float(lr_pct[5]),
            "lr_p100": float(lr_pct[6]),
            "pressure": 0.0,
            "phase": self.scheduler.phase,
            "gnorm_ratio": self.scheduler.gnorm_ratio,
            "tap_count": self.scheduler.tap_count,
        }

    def grad_norm_stats(self):
        mat_l2 = []
        row_l2 = []
        for p in self.spicy.param_groups[0]["params"]:
            g = p.grad
            if g is None:
                continue
            gf = g.detach().float()
            mat_l2.append(gf.norm().reshape(1))
            if gf.ndim >= 2:
                row_l2.append(gf.norm(dim=-1).reshape(-1))
        if not mat_l2:
            return None
        mat = torch.cat(mat_l2)
        out = {
            "grad_mat_l2_med": float(mat.median().item()),
            "grad_mat_l2_mean": float(mat.mean().item()),
            "grad_mat_l2_max": float(mat.max().item()),
        }
        if row_l2:
            rows = torch.cat(row_l2)
            out["grad_row_l2_med"] = float(rows.median().item())
            out["grad_row_l2_mean"] = float(rows.mean().item())
            out["grad_row_l2_max"] = float(rows.max().item())
        return out


class Dion2WithAdam:
    """Wrapper: Dion2 for 2D params + plain AdamW for the rest."""
    def __init__(self, model, muon_lr=0.02, adam_lr=3e-4, fraction=0.25):
        muon_p, adam_p = split_params(model)
        self.dion2 = Dion2(
            muon_p,
            distributed_mesh=None,
            lr=muon_lr,
            fraction=fraction,
            ef_decay=0.95,
            weight_decay=0,
            adjust_lr="spectral_norm",
        )
        self.adam = torch.optim.AdamW(adam_p, lr=adam_lr, weight_decay=0)
        # Expose param_groups for LR scheduling
        self.param_groups = self.dion2.param_groups + self.adam.param_groups
        self._dion2_groups = len(self.dion2.param_groups)

    def zero_grad(self):
        self.dion2.zero_grad()
        self.adam.zero_grad()

    def step(self):
        self.dion2.step()
        self.adam.step()
        # Sync param_groups back after LR changes
        for i, g in enumerate(self.param_groups):
            if i < self._dion2_groups:
                self.dion2.param_groups[i]["lr"] = g["lr"]
            else:
                self.adam.param_groups[i - self._dion2_groups]["lr"] = g["lr"]


class AutoNorMuonWrapper:
    """Wrapper for AutoNorMuon that exposes gnorm diagnostics like SpicyDionWithAdam."""

    def __init__(self, model, muon_lr=0.02, adam_lr=3e-4, total_steps=2000):
        muon_p, adam_p = split_params(model)
        self.opt = AutoNorMuon([
            {"params": muon_p, "use_muon": True, "lr": muon_lr, "momentum": 0.95, "weight_decay": 0},
            {"params": adam_p, "use_muon": False, "lr": adam_lr, "weight_decay": 0},
        ], total_steps=total_steps)
        self.param_groups = self.opt.param_groups
        self._muon_params = muon_p

    def zero_grad(self):
        self.opt.zero_grad()

    def step(self):
        self.opt.step()

    def effective_muon_lr(self):
        for g in self.opt.param_groups:
            if g.get("use_muon"):
                v = g.get("scheduled_lr")
                if v is not None:
                    return float(v) if not isinstance(v, torch.Tensor) else float(v.item())
        return 0.0

    def gnorm_stats(self, **kwargs):
        for g in self.opt.param_groups:
            if g.get("use_muon"):
                out = {}
                for key in ("gnorm_ema", "gnorm_max", "gnorm_median", "gnorm_mean",
                            "gnorm_std", "ratio_gnorm", "signal_ratio", "lr_mult",
                            "scheduled_lr"):
                    v = g.get(key)
                    if v is not None:
                        out[key] = float(v) if not isinstance(v, torch.Tensor) else float(v.item())
                # Map to the names the print loop expects
                return {
                    "ratio_med": out.get("ratio_gnorm", 0.0),
                    "ratio_mean": out.get("signal_ratio", 0.0),
                    "ema_med": out.get("gnorm_ema", 0.0),
                    "gmax_med": out.get("gnorm_max", 0.0),
                    "units_med": out.get("gnorm_median", 0.0),
                    "signal_med": out.get("gnorm_mean", 0.0),
                    "denom_med": out.get("gnorm_std", 0.0),
                }
        return None

    def grad_norm_stats(self):
        mat_l2 = []
        row_l2 = []
        for p in self._muon_params:
            g = p.grad
            if g is None:
                continue
            gf = g.detach().float()
            mat_l2.append(gf.norm().reshape(1))
            if gf.ndim >= 2:
                row_l2.append(gf.norm(dim=-1).reshape(-1))
        if not mat_l2:
            return None
        mat = torch.cat(mat_l2)
        out = {
            "grad_mat_l2_med": float(mat.median().item()),
            "grad_mat_l2_mean": float(mat.mean().item()),
            "grad_mat_l2_max": float(mat.max().item()),
        }
        if row_l2:
            rows = torch.cat(row_l2)
            out["grad_row_l2_med"] = float(rows.median().item())
            out["grad_row_l2_mean"] = float(rows.mean().item())
            out["grad_row_l2_max"] = float(rows.max().item())
        return out


def make_autonormuon(model, muon_lr=0.02, adam_lr=3e-4, total_steps=2000):
    return AutoNorMuonWrapper(model, muon_lr=muon_lr, adam_lr=adam_lr, total_steps=total_steps)


def make_dion2(model, muon_lr=0.02, adam_lr=3e-4, fraction=0.25):
    return Dion2WithAdam(model, muon_lr=muon_lr, adam_lr=adam_lr, fraction=fraction)


def make_spicydion(model, muon_lr=0.02, adam_lr=3e-4, fraction=0.25, total_steps=2000):
    return SpicyDionWithAdam(model, muon_lr=muon_lr, adam_lr=adam_lr, fraction=fraction, total_steps=total_steps)

def make_spicydion_sched(model, muon_lr=0.02, adam_lr=3e-4, fraction=1.0, total_steps=2000):
    return SpicyDionWithScheduler(model, muon_lr=muon_lr, adam_lr=adam_lr, fraction=fraction, total_steps=total_steps)


def make_muon(model, muon_lr=0.02, adam_lr=3e-4):
    muon_p, adam_p = split_params(model)
    return SingleDeviceMuonWithAuxAdam([
        {"params": muon_p, "use_muon": True, "lr": muon_lr, "weight_decay": 0},
        {"params": adam_p, "use_muon": False, "lr": adam_lr, "weight_decay": 0},
    ])

def make_normuon(model, muon_lr=0.02, adam_lr=3e-4):
    muon_p, adam_p = split_params(model)
    return SingleDeviceNorMuonWithAuxAdam([
        {"params": muon_p, "use_muon": True, "lr": muon_lr, "weight_decay": 0},
        {"params": adam_p, "use_muon": False, "lr": adam_lr, "weight_decay": 0},
    ])


def make_adamw(model, lr=3e-4):
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0)

def make_spicy_adam(model, lr=3e-4):
    return SpicyAdam(model.parameters(), lr=lr, weight_decay=0)


# ── Training loop ─────────────────────────────────────────────────────────

def train_run(name, model, optimizer, steps, batch_size, seq_len, vocab, device,
              muon_lr=0.02, adam_lr=3e-4):
    rng = torch.Generator(device=device)
    rng.manual_seed(123)

    losses = []
    t0 = time.time()
    last_grad_stats = None
    for step in range(steps):
        # IMPORTANT: Do NOT cosine-schedule SpicyDion externally.
        # SpicyDion uses its own adaptive LR logic from gnorm ratio.
        if name.endswith("-P"):
            # Pressure schedule: lr = base / (1 + k*t/T)
            for g in optimizer.param_groups:
                # Use initial LR from the group itself
                if "initial_lr" not in g:
                    g["initial_lr"] = g["lr"]
                g["lr"] = pressure_lr(g["initial_lr"], step, steps)
        elif name in ("Dion2", "Muon", "NorMuon"):
            cf = cosine_lr(1.0, step, steps)
            for g in optimizer.param_groups:
                if g.get("use_muon") is True or g.get("algorithm") == "dion2":
                    g["lr"] = muon_lr * cf
                else:
                    g["lr"] = adam_lr * cf
        elif name in ("AdamW", "SpicyAdam"):
            cf = cosine_lr(1.0, step, steps)
            for g in optimizer.param_groups:
                g["lr"] = adam_lr * cf
        elif name.startswith("AutoNor") or name == "Spicy-Sched":
            # AutoNorMuon / SpicyDion+GnormScheduler handle LR internally.
            pass
        else:
            # SpicyDion computes its own cosine ceiling internally.
            # Only cosine-schedule the auxiliary Adam groups externally.
            # SpicyDion handles distribution internally; schedule comes from outside.
            cf = cosine_lr(1.0, step, steps)
            for g in optimizer.param_groups:
                algo = g.get("algorithm", "")
                if algo == "spicydion":
                    g["lr"] = muon_lr * cf
                elif algo == "adamw":
                    g["lr"] = adam_lr * cf

        x, targets = make_batch(batch_size, seq_len, vocab, device, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()

        if hasattr(optimizer, "grad_norm_stats"):
            last_grad_stats = optimizer.grad_norm_stats()

        optimizer.step()

        lv = loss.item()
        losses.append(lv)

        if step % 100 == 0 or step == steps - 1:
            elapsed = time.time() - t0
            base_lr = float(optimizer.param_groups[0]["lr"])
            if hasattr(optimizer, "gnorm_stats") and hasattr(optimizer.gnorm_stats, "__call__"):
                stats = optimizer.gnorm_stats(step=step, total_steps=steps)
                if stats is not None and "ratio_p50" in stats:
                    if "phase" in stats:
                        tc = stats.get("tap_count", 0)
                        tc_str = f"(t={tc})" if tc > 0 else ""
                        phase_str = f" | {stats['phase']}{tc_str} gr={stats['gnorm_ratio']:.3f}"
                    else:
                        phase_str = ""
                    print(f"  [{name:12s}] step {step:4d} | loss={lv:.4f} | P={stats['pressure']:.3f}{phase_str} | {elapsed:.1f}s")
                    print(f"    ratio  p0={stats['ratio_p0']:.3f} p10={stats['ratio_p10']:.3f}"
                          f" p25={stats['ratio_p25']:.3f} p50={stats['ratio_p50']:.3f}"
                          f" p75={stats['ratio_p75']:.3f} p90={stats['ratio_p90']:.3f}"
                          f" p100={stats['ratio_p100']:.3f}")
                    print(f"    eff_lr p0={stats['lr_p0']:.5f} p10={stats['lr_p10']:.5f}"
                          f" p25={stats['lr_p25']:.5f} p50={stats['lr_p50']:.5f}"
                          f" p75={stats['lr_p75']:.5f} p90={stats['lr_p90']:.5f}"
                          f" p100={stats['lr_p100']:.5f}")
                    if last_grad_stats is not None:
                        print(f"    grad_l2 med={last_grad_stats['grad_mat_l2_med']:.4f}"
                              f" row_med={last_grad_stats.get('grad_row_l2_med', 0.0):.4f}")
                else:
                    eff_lr = float(optimizer.effective_muon_lr()) if hasattr(optimizer, "effective_muon_lr") else base_lr
                    print(f"  [{name:12s}] step {step:4d} | loss={lv:.4f} | lr={eff_lr:.5f} | {elapsed:.1f}s")
            elif hasattr(optimizer, "effective_muon_lr"):
                eff_lr = float(optimizer.effective_muon_lr())
                print(f"  [{name:12s}] step {step:4d} | loss={lv:.4f} | lr={eff_lr:.5f} | {elapsed:.1f}s")
            else:
                print(f"  [{name:12s}] step {step:4d} | loss={lv:.4f} | lr={base_lr:.5f} | {elapsed:.1f}s")

    elapsed = time.time() - t0

    bad = False
    for pname, p in model.named_parameters():
        if torch.isnan(p).any() or torch.isinf(p).any():
            print(f"  [{name}] WARNING: NaN/Inf in {pname}")
            bad = True

    return {
        "name": name,
        "losses": losses,
        "best": min(losses),
        "final": losses[-1],
        "initial": losses[0],
        "elapsed": elapsed,
        "diverged": bad or (max(losses[-20:]) > losses[0] + 0.5),
    }


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    vocab, d_model, n_heads, n_layers, seq_len = 65, 256, 4, 4, 64
    batch_size, steps = 8, 2000
    muon_lr, adam_lr = 0.02, 3e-4

    print(f"Device: {device}")
    print(f"Model: d_model={d_model}, layers={n_layers}, heads={n_heads}, "
          f"vocab={vocab}, seq_len={seq_len}")

    tmp = MiniGPT(vocab, d_model, n_heads, n_layers, seq_len)
    muon_p, adam_p = split_params(tmp)
    print(f"Params: {tmp.param_count():,} total "
          f"({sum(p.numel() for p in muon_p):,} muon, "
          f"{sum(p.numel() for p in adam_p):,} adam)")
    print(f"Training: {steps} steps, batch={batch_size}, seq_len={seq_len}")
    print("LR: muon/dion2 use cosine schedule; spicydion uses fixed base + adaptive gnorm LR")
    del tmp
    print()

    # 2000-step side-by-side benchmark.
    results = []
    for name, make_opt in [
         ("Spicy-Sched", lambda m: make_spicydion_sched(m, muon_lr=muon_lr, adam_lr=adam_lr, fraction=1.0, total_steps=steps)),
         ("Spicy-P", lambda m: make_spicydion(m, muon_lr=muon_lr, adam_lr=adam_lr, fraction=1.0, total_steps=steps)),
         #("AutoNorMuon", lambda m: make_autonormuon(m, muon_lr=muon_lr, adam_lr=adam_lr, total_steps=steps)),
         #("NorMuon", lambda m: make_normuon(m, muon_lr=muon_lr, adam_lr=adam_lr)),
         #("NorMuon-P", lambda m: make_normuon(m, muon_lr=muon_lr, adam_lr=adam_lr)),
         #("AdamW", lambda m: make_adamw(m, lr=adam_lr)),
         #("AdamW-P", lambda m: make_adamw(m, lr=adam_lr)),
         #("SpicyAdam-P", lambda m: make_spicy_adam(m, lr=2e-4)),
    ]:
        torch.manual_seed(42)
        model = MiniGPT(vocab, d_model, n_heads, n_layers, seq_len).to(device)
        opt = make_opt(model)
        r = train_run(name, model, opt, steps, batch_size, seq_len, vocab, device,
                      muon_lr=muon_lr, adam_lr=adam_lr)
        results.append(r)
        print()

    print("=" * 65)
    print(f"{'Optimizer':>12s} | {'Initial':>8s} | {'Best':>8s} | {'Final':>8s} | {'Time':>6s} | Status")
    print("-" * 65)
    for r in results:
        status = "DIVERGED" if r["diverged"] else "ok"
        print(f"{r['name']:>12s} | {r['initial']:8.4f} | {r['best']:8.4f} | "
              f"{r['final']:8.4f} | {r['elapsed']:5.1f}s | {status}")
    print("=" * 65)

    # single optimizer run, no delta reporting


if __name__ == "__main__":
    main()
