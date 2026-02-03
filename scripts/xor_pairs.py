"""
XOR-in-d-dimensions with configurable group size k.
  k=2: XOR of adjacent pairs (easy)
  k=3: XOR of adjacent triplets (harder)
  k=4+: increasingly difficult

x = random_binary(d), y[i] = XOR(x[k*i], x[k*i+1], ..., x[k*i+k-1])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


def make_sparse_indices(d: int, k: int) -> torch.Tensor:
    """Generate random index assignments for sparse parity: [d_out, k]."""
    d_out = d // k
    return torch.stack([torch.randperm(d)[:k] for _ in range(d_out)])


def make_xor_data(d: int, k: int, n: int, indices: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate n samples. d must be divisible by k. Output has d//k bits.
    If indices is provided: sparse parity mode — each output bit depends on k
    randomly chosen input bits (not adjacent).
    """
    assert d % k == 0, f"d={d} must be divisible by k={k}"
    x = torch.randint(0, 2, (n, d), dtype=torch.float32)
    d_out = d // k
    if indices is None:
        groups = x.view(n, d_out, k)
        y = (groups.sum(dim=-1) % 2).float()
    else:
        y = torch.zeros(n, d_out)
        for i in range(d_out):
            y[:, i] = (x[:, indices[i]].sum(dim=-1) % 2).float()
    return x, y


class LearnedAct(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2) * 0.5)

    @staticmethod
    def rms_norm(x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x):
        w = self.w.clamp(0.05, 1.0)
        return self.rms_norm(F.silu(x)) + w[0] * self.rms_norm(F.relu(x)) + w[1] * self.rms_norm(torch.tanh(x))


class LearnedAttnAct(nn.Module):
    """Linear attention as an activation function.
    φ(Q) @ (φ(K)^T @ V) where φ = learned act mix.
    Reshapes hidden dim into [n_heads, head_dim] for attention. Input/output: [batch, hidden]."""
    def __init__(self, hidden, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.wq = nn.Linear(hidden, hidden)
        self.wk = nn.Linear(hidden, hidden)
        self.wv = nn.Linear(hidden, hidden)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.phi = LearnedAct()

    @staticmethod
    def rms_norm(x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x):
        B = x.shape[0]
        Q = self.phi(self.q_norm(self.wq(x).view(B, self.n_heads, self.head_dim)))
        K = self.phi(self.k_norm(self.wk(x).view(B, self.n_heads, self.head_dim)))
        V = self.wv(x).view(B, self.n_heads, self.head_dim)
        # linear attention: φ(Q) @ (φ(K)^T @ V)
        KtV = torch.einsum('bni,bnj->bij', K, V)
        out = torch.einsum('bni,bij->bnj', Q, KtV)
        out = self.rms_norm(out)
        return out.reshape(B, -1)


class SwiGLUBlock(nn.Module):
    def __init__(self, width, ffn_mult=4, learned_act=False, differential=None, learned_attn_act=False):
        super().__init__()
        hidden = int(width * ffn_mult)
        self.differential = differential  # None, 'partial', or 'full'
        self.half_hidden = hidden // 2
        self.w1 = nn.Linear(width, hidden)
        self.w2 = nn.Linear(width, hidden)
        self.w3 = nn.Linear(hidden, width)
        if learned_attn_act:
            self.act = LearnedAttnAct(hidden)
        elif learned_act:
            self.act = LearnedAct()
        else:
            self.act = F.silu

        if differential in ('partial', 'full'):
            self.lam = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def rms_norm(x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x):
        if self.differential == 'partial':
            # shared gate/up, split hidden for differential readout
            h = self.rms_norm(self.act(self.w1(x))) * self.w2(x)
            h1, h2 = h[..., :self.half_hidden], h[..., self.half_hidden:]
            w3_1, w3_2 = self.w3.weight[:, :self.half_hidden], self.w3.weight[:, self.half_hidden:]
            out1 = F.linear(h1, w3_1, self.w3.bias)
            out2 = F.linear(h2, w3_2)
            return out1 - self.lam * out2

        elif self.differential == 'full':
            # full gate/up path, then split hidden for differential readout
            g = self.act(self.w1(x))
            u = self.w2(x)
            g1, g2 = g[..., :self.half_hidden], g[..., self.half_hidden:]
            u1, u2 = u[..., :self.half_hidden], u[..., self.half_hidden:]
            h1 = self.rms_norm(g1) * u1
            h2 = self.rms_norm(g2) * u2
            w3_1, w3_2 = self.w3.weight[:, :self.half_hidden], self.w3.weight[:, self.half_hidden:]
            out1 = F.linear(h1, w3_1, self.w3.bias)
            out2 = F.linear(h2, w3_2)
            return out1 - self.lam * out2

        h = self.rms_norm(self.act(self.w1(x))) * self.w2(x)
        return self.w3(h)


def make_mlp(d_in: int, d_out: int, width: int, depth: int = 2, ffn_mult: int = 4,
             learned_act: bool = False, differential: str | None = None,
             learned_attn_act: bool = False) -> nn.Module:
    layers: list[nn.Module] = [nn.Linear(d_in, width)]
    for _ in range(depth):
        layers.append(SwiGLUBlock(width, ffn_mult, learned_act, differential, learned_attn_act))
    layers.append(nn.Linear(width, d_out))
    return nn.Sequential(*layers)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(d: int = 64, k: int = 3, width: int = 64, n_train: int = 100_000,
          n_test: int = 10_000, depth: int = 2, ffn_mult: int = 4, lr: float = 1e-3,
          epochs: int = 100, batch_size: int = 512, hard: bool = False,
          learned_act: bool = False, differential: str | None = None,
          learned_attn_act: bool = False):
    device = get_device()
    d_out = d // k
    indices = make_sparse_indices(d, k) if hard else None
    if hard:
        print(f"sparse parity indices:\n{indices}")
    x_train, y_train = make_xor_data(d, k, n_train, indices)
    x_test, y_test = make_xor_data(d, k, n_test, indices)
    x_test, y_test = x_test.to(device), y_test.to(device)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)

    model = make_mlp(d, d_out, width, depth, ffn_mult, learned_act, differential, learned_attn_act).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"d={d} k={k} d_out={d_out} width={width} ffn_mult={ffn_mult} depth={depth} params={n_params:,} device={device}")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()
    steps_per_epoch = len(train_loader)
    warmup_steps = steps_per_epoch  # 1 epoch warmup

    def get_lr(step):
        if step < warmup_steps:
            return lr * (step + 1) / warmup_steps
        return lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: (step + 1) / warmup_steps if step < warmup_steps else 1.0)

    pbar = tqdm(range(epochs), desc="training")
    for epoch in pbar:
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb) * 5.0
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / n_train

        model.eval()
        with torch.no_grad():
            test_logits = model(x_test)
            val_loss = loss_fn(test_logits, y_test).item() * 5.0
            test_preds = (test_logits > 0).float()
            bit_acc = (test_preds == y_test).float().mean().item()
            full_acc = (test_preds == y_test).all(dim=1).float().mean().item()

        pbar.set_postfix(loss=f"{avg_loss:.4f}", val=f"{val_loss:.4f}", bit=f"{bit_acc:.4f}", full=f"{full_acc:.4f}")

        if (epoch + 1) % 20 == 0 or epoch == 0:
            tqdm.write(f"epoch {epoch+1:3d}  loss {avg_loss:.4f}  val {val_loss:.4f}  "
                       f"bit_acc {bit_acc:.4f}  full_acc {full_acc:.4f}")

        if full_acc >= 0.99:
            tqdm.write(f"early stop at epoch {epoch+1}  full_acc {full_acc:.4f}")
            return epoch + 1

    return epochs


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--d", type=int, default=64, help="input dimensionality")
    p.add_argument("--k", type=int, default=3, help="group size for XOR (2=pairs, 3=triplets, ...)")
    p.add_argument("--width", type=int, default=64, help="model internal width (independent of d)")
    p.add_argument("--ffn-mult", type=int, default=4)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--n-train", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hard", action="store_true", help="sparse parity: random index assignments instead of adjacent groups")
    p.add_argument("--learned-act", action="store_true", help="learned activation: silu + w0*relu + w1*tanh")
    p.add_argument("--differential-partial", action="store_true", help="shared gate/up, differential down readout")
    p.add_argument("--differential-full", action="store_true", help="fully independent second SwiGLU path")
    p.add_argument("--learned-attn-act", action="store_true", help="linear attention with learned act as kernel replaces activation")
    p.add_argument("--runs", type=int, default=3, help="number of runs to average")
    args = p.parse_args()

    differential = 'full' if args.differential_full else ('partial' if args.differential_partial else None)

    results = []
    for run in range(args.runs):
        print(f"\n=== run {run+1}/{args.runs} ===")
        ep = train(d=args.d, k=args.k, width=args.width, ffn_mult=args.ffn_mult,
                   depth=args.depth, epochs=args.epochs, n_train=args.n_train,
                   lr=args.lr, hard=args.hard, learned_act=args.learned_act,
                   differential=differential, learned_attn_act=args.learned_attn_act)
        results.append(ep)

    print(f"\n=== results: {results}  avg epochs to converge: {sum(results)/len(results):.1f} ===")
