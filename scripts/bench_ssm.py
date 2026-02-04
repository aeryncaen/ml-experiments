"""Benchmark DS1 vs S4D vs Mamba vs USB on SSM litmus tests.

Tasks: delay-1, selective copy, induction head
Metrics: final loss, param count, peak memory, wall time

Requires: einops, mamba_ssm (for Mamba CUDA ops on CUDA box), tqdm
"""

import sys, os, time, math, random, hashlib, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import defaultdict
try:
    from torch.profiler import profile, ProfilerActivity
    HAS_PROFILER = True
except Exception:
    HAS_PROFILER = False
from einops import rearrange, repeat
from tqdm import tqdm

DETERMINISTIC = os.getenv("BENCH_DETERMINISTIC", "0") == "1"
if DETERMINISTIC:
    # Must be set before any CUDA context is created
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if DETERMINISTIC:
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if torch.cuda.is_available():
        try:
            torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
        except Exception:
            pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'mamba'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 's5-pytorch'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'zoology'))

DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

# ---------------------------------------------------------------------------
# S4D (standalone, from s4/models/s4/s4d.py — no DropoutNd dependency)
# ---------------------------------------------------------------------------
class S4DKernel(nn.Module):
    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.log_dt = nn.Parameter(log_dt)
        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), 'n -> h n', h=H)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)

    def forward(self, L):
        dt = torch.exp(self.log_dt)
        C = torch.view_as_complex(self.C)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA) - 1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real
        return K


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, **kernel_args):
        super().__init__()
        self.h = d_model
        self.D = nn.Parameter(torch.randn(self.h))
        self.kernel = S4DKernel(self.h, N=d_state, **kernel_args)
        self.activation = nn.GELU()
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u):
        # u: (B, L, H) -> internal (B, H, L)
        u = u.transpose(-1, -2)
        L = u.size(-1)
        k = self.kernel(L=L)
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]
        y = y + u * self.D.unsqueeze(-1)
        y = self.activation(y)
        y = self.output_linear(y)
        return y.transpose(-1, -2)


# ---------------------------------------------------------------------------
# S5 (PyTorch port from s5-pytorch checkout)
# ---------------------------------------------------------------------------
try:
    from s5.s5_model import S5 as S5Module
    HAS_S5 = True
except ImportError:
    HAS_S5 = False
    S5Module = None


class S5Wrapper(nn.Module):
    """Wraps S5 to accept (B, L, H) and return (B, L, H)."""
    def __init__(self, width, state_width=256):
        super().__init__()
        if not HAS_S5:
            raise ImportError("S5 not available. Install s5-pytorch.")
        self.s5 = S5Module(width=width, state_width=state_width)

    def forward(self, x):
        return self.s5(x)


# ---------------------------------------------------------------------------
# Mamba (real implementation from mamba checkout)
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    Mamba = None


class MambaWrapper(nn.Module):
    """Wraps Mamba to accept (B, L, D) and return (B, L, D)."""
    def __init__(self, d_model, **kwargs):
        super().__init__()
        if not HAS_MAMBA:
            raise ImportError("Mamba not available. Install mamba_ssm.")
        self.mamba = Mamba(d_model=d_model, use_fast_path=True, **kwargs)

    def forward(self, x):
        return self.mamba(x)


# ---------------------------------------------------------------------------
# DS1 wrapper
# ---------------------------------------------------------------------------
from ds_moe.model import DS1, RMSNorm
from heuristic_secrets.models.backbone import SEBlock

# ---------------------------------------------------------------------------
# S6 / USB (Unified Sequence Block)
# ---------------------------------------------------------------------------
from s6.usb_block import USBBlock, USBConfig


class USBWrapper(nn.Module):
    """Wraps USB to accept (B, L, H) and return (B, L, H)."""
    def __init__(self, d_model, headdim=64, expansion_factor=2, attention_type="full", 
                 lowrank_factor=1.5, linear_feature_map="elu", **kwargs):
        super().__init__()
        config = USBConfig(
            d_model=d_model,
            headdim=headdim,
            expansion_factor=expansion_factor,
            attention_type=attention_type,  # type: ignore
            lowrank_factor=lowrank_factor,
            linear_feature_map=linear_feature_map,  # type: ignore
        )
        self.usb = USBBlock(config)

    def forward(self, x):
        return self.usb(x)


# Legacy S6 wrapper for backward compatibility (now uses USB)
S6Wrapper = USBWrapper


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_hidden, bias=True):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_hidden, bias=bias)
        self.w_up = nn.Linear(d_model, d_hidden, bias=bias)
        self.w_down = nn.Linear(d_hidden, d_model, bias=bias)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class QuadConvMix(nn.Module):
    """Split channels into 4 paths: passthrough, causal conv, acausal conv, retrocausal conv.
    Recombine via SE gating."""
    def __init__(self, d_model, k=3, reduction=4):
        super().__init__()
        assert d_model % 4 == 0
        self.d_path = d_model // 4
        # causal: left-padded conv
        self.conv_causal = nn.Conv1d(self.d_path, self.d_path, k, padding=0, groups=self.d_path)
        # acausal: symmetric-padded conv
        self.conv_acausal = nn.Conv1d(self.d_path, self.d_path, k, padding=k // 2, groups=self.d_path)
        # retrocausal: right-padded conv
        self.conv_retro = nn.Conv1d(self.d_path, self.d_path, k, padding=0, groups=self.d_path)
        self.se = SEBlock(d_model, reduction=reduction)
        self.norm = RMSNorm(d_model)
        self.k = k

    def forward(self, x):
        # x: (B, L, D)
        B, L, D = x.shape
        chunks = x.chunk(4, dim=-1)  # 4 x (B, L, d_path)

        p0 = chunks[0]  # passthrough

        # causal: pad left
        c1 = chunks[1].transpose(1, 2)  # (B, d_path, L)
        c1 = F.pad(c1, (self.k - 1, 0))
        c1 = self.conv_causal(c1).transpose(1, 2)

        # acausal: symmetric padding already in conv
        c2 = self.conv_acausal(chunks[2].transpose(1, 2)).transpose(1, 2)

        # retrocausal: pad right
        c3 = chunks[3].transpose(1, 2)
        c3 = F.pad(c3, (0, self.k - 1))
        c3 = self.conv_retro(c3).transpose(1, 2)

        out = torch.cat([p0, c1, c2, c3], dim=-1)  # (B, L, D)
        return self.norm(self.se(out) + x)


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
    """Linear attention as activation: φ(Q) @ (φ(K)^T @ V) with learned act as φ."""
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
        *leading, D = x.shape
        flat = x.reshape(-1, D)
        B = flat.shape[0]
        Q = self.phi(self.q_norm(self.wq(flat).view(B, self.n_heads, self.head_dim)))
        K = self.phi(self.k_norm(self.wk(flat).view(B, self.n_heads, self.head_dim)))
        V = self.wv(flat).view(B, self.n_heads, self.head_dim)
        KtV = torch.einsum('bni,bnj->bij', K, V)
        out = torch.einsum('bni,bij->bnj', Q, KtV)
        out = self.rms_norm(out)
        return out.reshape(*leading, D)


class LearnedAttnActMLP(nn.Module):
    """SwiGLU MLP with LearnedAttnAct as the activation. (B, L, D) -> (B, L, D)."""
    def __init__(self, d_model, ffn_mult=4, n_heads=4):
        super().__init__()
        hidden = int(d_model * ffn_mult)
        self.w1 = nn.Linear(d_model, hidden)
        self.w2 = nn.Linear(d_model, hidden)
        self.w3 = nn.Linear(hidden, d_model)
        self.act = LearnedAttnAct(hidden, n_heads)

    @staticmethod
    def rms_norm(x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x):
        return self.w3(self.rms_norm(self.act(self.w1(x))) * self.w2(x))


class MHABlock(nn.Module):
    """QuadConv → MHA."""
    def __init__(self, d_model, n_heads=4, se_reduction=4):
        super().__init__()
        self.quad_conv = QuadConvMix(d_model, k=3, reduction=se_reduction)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x):
        h = self.quad_conv(x)
        h, _ = self.attn(h, h, h)
        return h


class DS1Wrapper(nn.Module):
    def __init__(self, dim, state_dim=64, mimo_rank=4, n_iters=2, **kwargs):
        super().__init__()
        self.ds1 = DS1(dim=dim, state_dim=state_dim, mimo_rank=mimo_rank,
                        n_iters=n_iters, **kwargs)
        bank_size = DS1.bank_size(dim, state_dim, mimo_rank,
                                   out_gate=kwargs.get('out_gate', False),
                                   diff_attn=kwargs.get('diff_attn', False))
        self.bank = nn.Parameter(torch.randn(bank_size) * 0.02)

    def forward(self, x):
        return self.ds1(x, self.bank)


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------
def gen_delay(B, L, vocab_size=32, delay=1, device='cpu'):
    """Token input, target is same sequence shifted by `delay` positions."""
    x = torch.randint(0, vocab_size, (B, L), device=device)
    target = torch.zeros_like(x)
    target[:, delay:] = x[:, :L - delay]
    return x, target


def gen_selective_copy(B, L, vocab_size=32, n_markers=4, device='cpu'):
    """Token input with marker channel. Target is the n_markers marked tokens.
    Input: (B, L) tokens + (B, L) marker flags → fed as 2-token tuples via embedding.
    Output: (B, n_markers) token predictions at end of sequence."""
    tokens = torch.randint(0, vocab_size, (B, L), device=device)
    markers = torch.zeros(B, L, dtype=torch.long, device=device)
    target = torch.zeros(B, n_markers, dtype=torch.long, device=device)
    for b in range(B):
        idxs = torch.randperm(L, device=device)[:n_markers].sort().values
        markers[b, idxs] = 1
        target[b] = tokens[b, idxs]
    return tokens, markers, target


def gen_parity(B, L, device='cpu'):
    """Binary input {0,1}, target is running XOR (cumulative parity)."""
    x = torch.randint(0, 2, (B, L), device=device)
    target = x.cumsum(dim=1) % 2  # running parity
    return x, target


def gen_mod_arith(B, L, mod_base=5, device='cpu'):
    """Digit input 0..mod_base-1, target is running sum mod base."""
    x = torch.randint(0, mod_base, (B, L), device=device)
    target = x.cumsum(dim=1) % mod_base
    return x, target


def gen_induction(B, L, vocab_size=32, device='cpu'):
    half = L // 2
    pattern = torch.randint(0, vocab_size, (B, half), device=device)
    seq = torch.cat([pattern, pattern], dim=1)  # (B, L)
    input_seq = seq[:, :-1]
    target_seq = seq[:, 1:]
    return input_seq, target_seq


# ---------------------------------------------------------------------------
# Data pregeneration, disk caching, and GPU preloading
# ---------------------------------------------------------------------------
CACHE_DIR = Path(__file__).parent / '.bench_cache'


def _cache_key(task_name, n_steps, B, L, seed):
    raw = f"{task_name}_{n_steps}_{B}_{L}_{seed}"
    return hashlib.md5(raw.encode()).hexdigest()


def pregen_task_data(task_name, n_train_batches, n_val_batches, B, L, seed, device='cpu'):
    """Generate train and val batches for a task, cache to disk, return dict with train/val lists."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(task_name, n_train_batches + n_val_batches, B, L, seed)
    cache_path = CACHE_DIR / f"{task_name}_{key}.pt"

    if cache_path.exists():
        cached = torch.load(cache_path, weights_only=True)
        train_batches = cached['train']
        val_batches = cached['val']
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        def gen_batch(task_name):
            if task_name == 'delay':
                return gen_delay(B, L, vocab_size=32, device='cpu')
            elif task_name == 'selective_copy':
                return gen_selective_copy(B, L, vocab_size=32, device='cpu')
            elif task_name == 'parity':
                return gen_parity(B, L, device='cpu')
            elif task_name == 'mod_arith':
                return gen_mod_arith(B, L, device='cpu')
            elif task_name == 'induction':
                return gen_induction(B, L, vocab_size=32, device='cpu')
        
        train_batches = []
        for _ in tqdm(range(n_train_batches), desc=f"Generating {task_name} train", leave=False):
            train_batches.append(gen_batch(task_name))
        
        val_batches = []
        for _ in tqdm(range(n_val_batches), desc=f"Generating {task_name} val", leave=False):
            val_batches.append(gen_batch(task_name))
        
        torch.save({'train': train_batches, 'val': val_batches}, cache_path)

    # Preload everything to target device
    def _to(t):
        return t.to(device, non_blocking=True) if isinstance(t, torch.Tensor) else t
    
    return {
        'train': [tuple(_to(t) for t in batch) for batch in train_batches],
        'val': [tuple(_to(t) for t in batch) for batch in val_batches],
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_task(model, task_name, dim, max_epochs=100, lr=1e-3, B=32, L=32, device='cpu',
               preloaded_data=None, early_stop_acc=0.99):
    """Train with epochs and early stopping when val accuracy exceeds threshold."""
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)

    if task_name == 'delay':
        vocab_size = 32
        embed = nn.Embedding(vocab_size, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=lr)
    elif task_name == 'selective_copy':
        vocab_size = 32
        embed = nn.Embedding(vocab_size, dim).to(device)
        marker_embed = nn.Embedding(2, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters())
                         + list(marker_embed.parameters()) + list(head.parameters()), lr=lr)
    elif task_name == 'parity':
        vocab_size = 2
        embed = nn.Embedding(vocab_size, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=lr)
    elif task_name == 'mod_arith':
        vocab_size = 5
        embed = nn.Embedding(vocab_size, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=lr)
    elif task_name == 'induction':
        vocab_size = 32
        embed = nn.Embedding(vocab_size, dim).to(device)
        head = nn.Linear(dim, vocab_size).to(device)
        opt = optim.Adam(list(model.parameters()) + list(embed.parameters()) + list(head.parameters()), lr=lr)

    assert preloaded_data is not None, "preloaded_data required — use pregen_task_data()"
    train_data = preloaded_data['train']
    val_data = preloaded_data['val']

    def _forward(batch):
        if task_name == 'delay':
            inp, tgt = batch
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc
        elif task_name == 'selective_copy':
            tokens, markers, tgt = batch
            x = embed(tokens) + marker_embed(markers)
            y = head(model(x))
            n_markers = tgt.shape[1]
            y_last = y[:, -n_markers:]
            loss = F.cross_entropy(y_last.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y_last.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc
        elif task_name == 'parity':
            inp, tgt = batch
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc
        elif task_name == 'mod_arith':
            inp, tgt = batch
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc
        else:  # induction
            inp, tgt = batch
            y = head(model(embed(inp)))
            loss = F.cross_entropy(y.reshape(-1, vocab_size), tgt.reshape(-1))
            with torch.no_grad():
                acc = (y.reshape(-1, vocab_size).argmax(-1) == tgt.reshape(-1)).float().mean().item()
            return loss, acc

    def _eval_val():
        model.eval()
        val_accs = []
        val_losses = []
        with torch.no_grad():
            for batch in val_data:
                loss, acc = _forward(batch)
                val_losses.append(loss.item())
                val_accs.append(acc)
        model.train()
        return sum(val_losses) / len(val_losses), sum(val_accs) / len(val_accs)

    mem_baseline = 0
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        mem_baseline = torch.cuda.memory_allocated()

    t0 = time.perf_counter()
    
    n_train = len(train_data)
    total_steps = 0
    final_epoch = 0
    final_val_acc = 0.0
    final_val_loss = 0.0
    initial_loss = None
    
    epoch_pbar = tqdm(range(max_epochs), desc="Epochs", leave=False, ncols=100)
    for epoch in epoch_pbar:
        # Shuffle train indices each epoch
        train_indices = torch.randperm(n_train).tolist()
        
        epoch_losses = []
        epoch_accs = []
        
        step_pbar = tqdm(train_indices, desc=f"Epoch {epoch+1}", leave=False, ncols=80)
        for idx in step_pbar:
            batch = train_data[idx]
            loss, acc = _forward(batch)
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            
            epoch_losses.append(loss.item())
            epoch_accs.append(acc)
            total_steps += 1
            
            step_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.1%}")
        
        # Record initial loss from first epoch
        if initial_loss is None:
            initial_loss = sum(epoch_losses[:10]) / min(10, len(epoch_losses))
        
        # Evaluate on validation set
        val_loss, val_acc = _eval_val()
        final_epoch = epoch + 1
        final_val_acc = val_acc
        final_val_loss = val_loss
        
        epoch_pbar.set_postfix(val_loss=f"{val_loss:.4f}", val_acc=f"{val_acc:.1%}")
        
        # Early stopping
        if val_acc >= early_stop_acc:
            break

    if device == 'cuda':
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0

    peak_mem = 0
    if device == 'cuda':
        peak_mem = (torch.cuda.max_memory_allocated() - mem_baseline) / 1024 / 1024  # MB

    return {
        'initial': initial_loss if initial_loss else 0.0,
        'final': final_val_loss,
        'acc': final_val_acc,
        'epochs': final_epoch,
        'steps': total_steps,
        'wall_s': wall,
        'peak_mem_mb': peak_mem,
        'converged': final_val_acc >= early_stop_acc,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def count_params(model):
    return sum(p.numel() for p in model.parameters())


class StackedModel(nn.Module):
    """Stack N identical layers with pre-norm residual connections."""
    def __init__(self, make_layer, n_layers, dim):
        super().__init__()
        self.layers = nn.ModuleList([make_layer() for _ in range(n_layers)])
        self.norms = nn.ModuleList([RMSNorm(dim) for _ in range(n_layers)])
        self.final_norm = RMSNorm(dim)

    def forward(self, x):
        for norm, layer in zip(self.norms, self.layers):
            x = x + layer(norm(x))
        return self.final_norm(x)


def _stack(make_layer, n_layers, dim):
    """Always wrap with pre-norm residual stacking."""
    return StackedModel(make_layer, n_layers, dim)


def make_models(dim, n_layers=1, requested_models=None):
    """Build models with configurable depth. Param counts are per-layer (~50-57K).
    
    If requested_models is provided, only build those models (skips unavailable ones with warning).
    """
    models = {}
    
    def try_add(name, make_fn):
        if requested_models is not None and name not in requested_models:
            return
        try:
            models[name] = _stack(make_fn, n_layers, dim)
        except ImportError as e:
            print(f"  Warning: {name} not available ({e})")
        except Exception as e:
            print(f"  Warning: {name} failed to initialize ({e})")

    # DS1++: DS1 + signed sparse attention
    try_add('DS1++', lambda: DS1Wrapper(dim=dim, state_dim=48, mimo_rank=4, n_iters=2, diff_attn=True))

    # S4D: ~50K params/layer
    try_add('S4D', lambda: S4D(d_model=dim, d_state=320))

    # S5: ~50K params/layer
    try_add('S5', lambda: S5Wrapper(width=dim, state_width=256))

    # Mamba: ~51K params/layer
    try_add('Mamba', lambda: MambaWrapper(d_model=dim, d_state=64, d_conv=4, expand=2))

    # USB (Unified Sequence Block, formerly S6): fused SSM + attention
    try_add('USB', lambda: USBWrapper(d_model=dim, headdim=32, expansion_factor=2))
    
    # USB-LR: USB with low-rank attention (O(L^1.5) instead of O(L^2))
    try_add('USB-LR', lambda: USBWrapper(d_model=dim, headdim=32, expansion_factor=2, 
                                          attention_type="lowrank", lowrank_factor=1.5))
    
    # USB-Lin: USB with linear attention (O(L·d²) instead of O(L²·d))
    try_add('USB-Lin', lambda: USBWrapper(d_model=dim, headdim=32, expansion_factor=2,
                                           attention_type="linear", linear_feature_map="elu"))

    # MHA: ~19K params/layer (QuadConv + MHA only, no MLP/Mamba to scale)
    try_add('MHA', lambda: MHABlock(d_model=dim, n_heads=4))

    # LearnedAttnAct MLP: SwiGLU with linear attention activation
    try_add('AttnActMLP', lambda: LearnedAttnActMLP(d_model=dim, ffn_mult=4, n_heads=4))

    return models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark SSM models on litmus tests')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile on models')
    parser.add_argument('--compile-mode', type=str, default='default', 
                        choices=['default', 'reduce-overhead', 'max-autotune'],
                        help='torch.compile mode')
    parser.add_argument('--dim', type=int, default=64, help='Model dimension')
    parser.add_argument('--layers', type=int, default=1, help='Number of layers')
    parser.add_argument('--max-epochs', type=int, default=100, help='Maximum training epochs')
    parser.add_argument('--train-batches', type=int, default=100, help='Number of training batches per epoch')
    parser.add_argument('--val-batches', type=int, default=20, help='Number of validation batches')
    parser.add_argument('--early-stop-acc', type=float, default=0.99, help='Early stop when val acc exceeds this')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-len', type=int, default=32, help='Sequence length')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Specific models to test (default: all)')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='Specific tasks to test (default: all)')
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if args.compile:
        print(f"torch.compile enabled (mode={args.compile_mode})")
    
    dim = args.dim
    n_layers = args.layers
    all_tasks = ['delay', 'selective_copy', 'induction', 'parity', 'mod_arith']
    tasks = args.tasks if args.tasks else all_tasks
    max_epochs = args.max_epochs
    n_train = args.train_batches
    n_val = args.val_batches
    B, L = args.batch_size, args.seq_len

    requested = args.models  # None means all available
    models_info = make_models(dim, n_layers=n_layers, requested_models=requested)
    all_names = list(models_info.keys())
    
    if not all_names:
        print("No models available to test!")
        sys.exit(1)
    
    if requested:
        missing = set(requested) - set(all_names)
        if missing:
            print(f"Warning: Requested models not available: {missing}")
    
    print(f"\n{'Model':<10} {'Params':>10} {'Layers':>8}")
    print('-' * 30)
    for name in all_names:
        m = models_info[name]
        print(f"{name:<10} {count_params(m):>10,} {n_layers:>8}")

    print(f"\nMax {max_epochs} epochs, {n_train} train batches, {n_val} val batches, B={B}, L={L}, dim={dim}, layers={n_layers}")
    print(f"Early stop at >{args.early_stop_acc:.0%} val accuracy")
    print('=' * 100)

    header = f"{'Model':<10} {'Task':<16} {'Init':>8} {'Final':>8} {'Val Acc':>8} {'Epochs':>7} {'Wall(s)':>8} {'Mem(MB)':>9} {'Status':>10}"
    print(header)
    print('-' * 100)

    task_pbar = tqdm(tasks, desc="Tasks", position=0)
    for task in task_pbar:
        task_pbar.set_description(f"Task: {task}")
        print(f"\nPregenerating {task} data...")
        task_data = pregen_task_data(task, n_train, n_val, B, L, SEED, device=DEVICE)
        print(f"  {len(task_data['train'])} train + {len(task_data['val'])} val batches ready on {DEVICE}")
        
        model_pbar = tqdm(all_names, desc="Models", position=1, leave=False)
        for name in model_pbar:
            model_pbar.set_description(f"Model: {name}")
            random.seed(SEED)
            np.random.seed(SEED)
            torch.manual_seed(SEED)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(SEED)
            model = make_models(dim, n_layers=n_layers, requested_models=[name])[name]
            
            # Optionally compile the model
            if args.compile:
                model = torch.compile(model, mode=args.compile_mode)
            
            r = train_task(model, task, dim, max_epochs=max_epochs, lr=1e-3, B=B, L=L, device=DEVICE,
                           preloaded_data=task_data, early_stop_acc=args.early_stop_acc)
            status = "CONVERGED" if r['converged'] else "MAX_EPOCH"
            tqdm.write(f"{name:<10} {task:<16} {r['initial']:>8.4f} {r['final']:>8.4f} {r['acc']:>7.1%} {r['epochs']:>7} {r['wall_s']:>8.1f} {r['peak_mem_mb']:>8.1f} {status:>10}")
        del task_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
