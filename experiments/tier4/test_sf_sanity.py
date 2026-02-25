#!/usr/bin/env python3
"""Sanity check: train a tiny transformer on Shakespeare with AutoNorMuon."""
import sys, time, math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Tiny GPT model
# ---------------------------------------------------------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_head, n_layer, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(seq_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # tie weights
        self.lm_head.weight = self.wte.weight
        # Megatron-style init
        self._megatron_init(d_model, n_layer)

    def _megatron_init(self, d_model, n_layer):
        """Megatron-style init: N(0, 1/sqrt(fan_in)), residual outputs scaled by 1/sqrt(2*n_layer)."""
        import math
        res_scale = 1.0 / math.sqrt(2 * n_layer)
        # Embeddings: N(0, 1/sqrt(d_model))
        nn.init.normal_(self.wte.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        nn.init.normal_(self.wpe.weight, mean=0.0, std=1.0 / math.sqrt(d_model))
        for block in self.blocks:
            # Attention
            fan_in = d_model
            nn.init.normal_(block.attn.qkv.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=res_scale / math.sqrt(fan_in))
            # MLP
            nn.init.normal_(block.mlp.fc1.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
            mlp_fan_in = 4 * d_model
            nn.init.normal_(block.mlp.fc2.weight, mean=0.0, std=res_scale / math.sqrt(mlp_fan_in))

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.wte(idx) + self.wpe(pos)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


class Block(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


# ---------------------------------------------------------------------------
# Shakespeare dataset (character-level)
# ---------------------------------------------------------------------------
def get_shakespeare(seq_len, device):
    """Download tiny shakespeare, return train/val tensors."""
    import urllib.request, os
    path = os.path.join(os.path.dirname(__file__), "tiny_shakespeare.txt")
    if not os.path.exists(path):
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        print(f"Downloading {url} ...")
        urllib.request.urlretrieve(url, path)
    text = open(path).read()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=device)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)


def get_batch(data, batch_size, seq_len):
    ix = torch.randint(len(data) - seq_len - 1, (batch_size,))
    x = torch.stack([data[i:i+seq_len] for i in ix])
    y = torch.stack([data[i+1:i+seq_len+1] for i in ix])
    return x, y


# ---------------------------------------------------------------------------
# Build optimizer for a given variant
# ---------------------------------------------------------------------------
_EMBED_NAMES = {"wte", "lm_head", "wpe"}

def build_optimizer(model, variant, total_steps, muon_lr=0.02, adam_lr=2e-4, wd=0.0):
    """Returns (optimizer, is_auto, retract_params).

    is_auto=True for autonormuon (manages its own LR internally).
    retract_params: list of 2D params to retract externally (None for autonormuon,
    which retracts internally).
    """
    muon_params = []
    adam_decay = []
    adam_nodecay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1:
            adam_nodecay.append(p)
        elif any(k in name for k in _EMBED_NAMES):
            adam_decay.append(p)
        else:
            muon_params.append(p)

    # Strip schedule suffix to pick optimizer class
    opt_key = variant.replace("_cosine", "")

    if opt_key == "autonormuon":
        from autonormuon import AutoNorMuon
        groups = [
            dict(params=muon_params, lr=muon_lr, momentum=0.95, beta2=0.95, weight_decay=wd, use_muon=True),
            dict(params=adam_decay, lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=wd, use_muon=False),
            dict(params=adam_nodecay, lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0, use_muon=False),
        ]
        groups = [g for g in groups if g["params"]]
        # AutoNorMuon retracts internally — no external retraction needed
        return AutoNorMuon(groups, total_steps=total_steps), True, None

    elif opt_key == "normuon":
        from normuon import SingleDeviceNorMuonWithAuxAdam
        groups = [
            dict(params=muon_params, lr=muon_lr, momentum=0.95, beta2=0.95, weight_decay=wd, use_muon=True),
            dict(params=adam_decay, lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=wd, use_muon=False),
            dict(params=adam_nodecay, lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0, use_muon=False),
        ]
        groups = [g for g in groups if g["params"]]
        return SingleDeviceNorMuonWithAuxAdam(groups), False, muon_params

    elif opt_key == "muon":
        from muon import SingleDeviceMuonWithAuxAdam
        groups = [
            dict(params=muon_params, lr=muon_lr, momentum=0.95, weight_decay=wd, use_muon=True),
            dict(params=adam_decay, lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=wd, use_muon=False),
            dict(params=adam_nodecay, lr=adam_lr, betas=(0.9, 0.95), eps=1e-10, weight_decay=0.0, use_muon=False),
        ]
        groups = [g for g in groups if g["params"]]
        return SingleDeviceMuonWithAuxAdam(groups), False, muon_params

    else:
        raise ValueError(f"Unknown variant: {variant}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def _default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def train(variant, steps=5000, d_model=256, n_head=4, n_layer=4, seq_len=32,
          batch_size=32, muon_lr=0.02, adam_lr=2e-4, device=None, **kwargs):
    if device is None:
        device = _default_device()
    use_cosine = variant.endswith("_cosine")
    sched_label = "cosine" if use_cosine else ("internal" if variant == "autonormuon" else "static")
    print(f"\n{'='*60}")
    print(f"  {variant}  |  {steps} steps  |  d={d_model} L={n_layer} H={n_head} T={seq_len}")
    print(f"  muon_lr={muon_lr}  adam_lr={adam_lr}  schedule={sched_label}  device={device}")
    print(f"{'='*60}")

    dev = torch.device(device)
    torch.manual_seed(42)

    train_data, val_data, vocab_size = get_shakespeare(seq_len, dev)
    model = TinyGPT(vocab_size, d_model, n_head, n_layer, seq_len).to(dev)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_params:,}  vocab: {vocab_size}")

    optimizer, is_auto, retract_params = build_optimizer(model, variant, total_steps=steps,
                                                          muon_lr=muon_lr, adam_lr=adam_lr)

    # LR schedule for non-auto variants: cosine decay or static (constant)
    def get_lr_factor(step):
        if use_cosine:
            return 0.5 * (1.0 + math.cos(math.pi * step / steps))
        return 1.0

    # Collect base LRs
    base_lrs = [g["lr"] for g in optimizer.param_groups]

    n_groups = len(optimizer.param_groups)
    gnorm_log = {gi: [] for gi in range(n_groups)}
    group_names = []
    for gi, pg in enumerate(optimizer.param_groups):
        group_names.append(f"{'muon' if pg.get('use_muon') else 'adam'}_{gi}")

    t0 = time.time()
    log_interval = 500
    val_losses = []

    for step in range(steps):
        # Validation
        if step % log_interval == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                vl_sum = 0.0
                for _ in range(20):
                    xv, yv = get_batch(val_data, batch_size, seq_len)
                    _, vl = model(xv, yv)
                    vl_sum += vl.item()
                val_loss = vl_sum / 20
            val_losses.append((step, val_loss))
            elapsed = time.time() - t0
            lr_display = optimizer.param_groups[0].get("scheduled_lr", optimizer.param_groups[0]["lr"])
            print(f"  step {step:5d} | val_loss {val_loss:.4f} | lr {lr_display:.2e} | {elapsed:.1f}s")
            model.train()

        # LR schedule for non-auto variants
        if not is_auto:
            factor = get_lr_factor(step)
            for pg, blr in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = blr * factor

        # Training step
        x, y = get_batch(train_data, batch_size, seq_len)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Row retraction: unit-norm each row of 2D weight matrices
        if retract_params is not None:
            with torch.no_grad():
                for p in retract_params:
                    p.div_(p.norm(dim=-1, keepdim=True).clamp(min=1e-8))

        # Collect every step
        for gi, pg in enumerate(optimizer.param_groups):
            gnorm_log[gi].append({
                "step": step,
                "median": pg.get("gnorm_median", 0),
                "fast": pg.get("gnorm_fast", 0),
                "slow": pg.get("gnorm_slow", 0),
                "ratio_raw": pg.get("gnorm_ratio_raw", 1.0),
                "ratio": pg.get("gnorm_ratio", 1.0),
                "loss": loss.item(),
            })

    # Final val
    model.eval()
    with torch.no_grad():
        vl_sum = 0.0
        for _ in range(20):
            xv, yv = get_batch(val_data, batch_size, seq_len)
            _, vl = model(xv, yv)
            vl_sum += vl.item()
        final_val = vl_sum / 20
    val_losses.append((steps, final_val))
    elapsed = time.time() - t0
    print(f"  step {steps:5d} | val_loss {final_val:.4f} | FINAL | {elapsed:.1f}s")

    # Write CSV
    import csv, os
    csv_path = os.path.join(os.path.dirname(__file__), f"gnorm_{variant}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "group", "median", "fast", "slow", "ratio_raw", "ratio", "loss"])
        for gi in range(n_groups):
            gname = group_names[gi]
            for row in gnorm_log[gi]:
                w.writerow([row["step"], gname, row["median"], row["fast"],
                            row["slow"], row["ratio_raw"], row["ratio"], row["loss"]])
    print(f"  wrote {csv_path} ({sum(len(v) for v in gnorm_log.values())} rows)")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(n_groups, 3, figsize=(18, 4 * n_groups), squeeze=False)
        for gi in range(n_groups):
            gname = group_names[gi]
            rows = gnorm_log[gi]
            steps_arr = [r["step"] for r in rows]
            medians = [r["median"] for r in rows]
            fasts = [r["fast"] for r in rows]
            slows = [r["slow"] for r in rows]
            ratios_raw = [r["ratio_raw"] for r in rows]
            ratios = [r["ratio"] for r in rows]
            losses = [r["loss"] for r in rows]

            # Col 0: raw median + both EMAs
            ax = axes[gi][0]
            ax.plot(steps_arr, medians, alpha=0.3, linewidth=0.5, label="median (raw)")
            ax.plot(steps_arr, fasts, linewidth=1, label="fast EMA")
            ax.plot(steps_arr, slows, linewidth=1, label="slow EMA")
            ax.set_title(f"{gname}: grad norm")
            ax.set_xlabel("step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Col 1: raw ratio + normalized ratio
            ax = axes[gi][1]
            ax.plot(steps_arr, ratios_raw, linewidth=0.6, alpha=0.5, color="tab:orange", label="raw (fast/slow)")
            ax.plot(steps_arr, ratios, linewidth=1, color="tab:red", label="normalized (/ max)")
            ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
            ax.set_title(f"{gname}: ratio")
            ax.set_xlabel("step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Col 2: training loss
            ax = axes[gi][2]
            ax.plot(steps_arr, losses, alpha=0.3, linewidth=0.5, color="tab:green")
            _smooth = []
            _ema = losses[0] if losses else 0
            for l in losses:
                _ema = 0.99 * _ema + 0.01 * l
                _smooth.append(_ema)
            ax.plot(steps_arr, _smooth, linewidth=1, color="tab:green", label="loss (ema)")
            ax.set_title(f"{gname}: train loss")
            ax.set_xlabel("step")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(os.path.dirname(__file__), f"gnorm_{variant}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  wrote {plot_path}")
    except ImportError:
        print("  matplotlib not available, skipping plot")

    return val_losses


# ---------------------------------------------------------------------------
# Main — comparison grid
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="*", help="specific variants to run (overrides grid)")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Megatron-style LR: 0.1 / sqrt(R) where R = n_layer * 2
    R = args.n_layer * 2
    auto_lr = 0.1 / math.sqrt(R)

    # Standard hand-tuned LRs
    STD_MUON_LR = 0.02
    STD_ADAM_LR = 2e-4

    if args.variants:
        # Manual mode: run exactly what's requested with default LRs
        results = {}
        for v in args.variants:
            results[v] = train(v, steps=args.steps, n_layer=args.n_layer, d_model=args.d_model,
                               n_head=args.n_head, seq_len=args.seq_len, batch_size=args.batch_size,
                               device=args.device)
    else:
        # Grid mode: normuon baselines at standard LR vs autonormuon at derived LR
        grid = [
            # (label, variant, muon_lr, adam_lr)
            ("normuon_static_std",  "normuon",        STD_MUON_LR, STD_ADAM_LR),
            ("normuon_cosine_std",  "normuon_cosine", STD_MUON_LR, STD_ADAM_LR),
            ("autonormuon_auto",    "autonormuon",    auto_lr,     auto_lr),
        ]
        results = {}
        for label, variant, mlr, alr in grid:
            results[label] = train(variant, steps=args.steps, n_layer=args.n_layer,
                                   d_model=args.d_model, n_head=args.n_head, seq_len=args.seq_len,
                                   batch_size=args.batch_size, muon_lr=mlr, adam_lr=alr,
                                   device=args.device)

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    if not args.variants:
        print(f"  n_layer={args.n_layer} R={R} auto_lr={auto_lr:.6f}")
        print(f"  std_muon_lr={STD_MUON_LR} std_adam_lr={STD_ADAM_LR}")
    print(f"{'='*70}")
    print(f"  {'variant':<25} {'muon_lr':>10} {'adam_lr':>10} {'final_val':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*12}")
    for label, vals in results.items():
        final = vals[-1][1]
        # Recover LRs from label or defaults
        if not args.variants:
            row = next(r for r in grid if r[0] == label)
            print(f"  {label:<25} {row[2]:>10.6f} {row[3]:>10.6f} {final:>12.4f}")
        else:
            print(f"  {label:<25} {'':>10} {'':>10} {final:>12.4f}")
    print()
