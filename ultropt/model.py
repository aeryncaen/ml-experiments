"""
Small character-level GPT for Shakespeare experiments.
Includes both standard GPT and nGPT (normalized transformer on the hypersphere).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(
                1, 1, block_size, block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ShakespeareGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    ):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # weight tying
        self.tok_emb.weight = self.head.weight

        self.apply(self._init_weights)
        # scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, f"Sequence length {T} > block_size {self.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# nGPT: Normalized Transformer on the Hypersphere
# (Loshchilov et al. 2025)
# ---------------------------------------------------------------------------

def _unit_norm(x, dim=-1):
    """Normalize to unit norm along dim."""
    return F.normalize(x, p=2, dim=dim, eps=1e-8)


def _ngpt_scale_param(shape, init_val, scale_val):
    """Create a scaling parameter with the nGPT init/scale trick.

    Stored as `scale_val`, but forward multiplies by `init_val / scale_val`
    so actual value starts at init_val while Adam sees magnitude scale_val.
    """
    p = nn.Parameter(torch.full(shape, scale_val))
    p._ngpt_init = init_val
    p._ngpt_scale = scale_val
    p._no_weight_decay = True
    return p


def _ngpt_actual(p):
    """Recover actual value: param * (init / scale)."""
    return p * (p._ngpt_init / p._ngpt_scale)


class NGPTRotary(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False,
        )
        self.seq_len_cached = 0
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        t = x.shape[1]
        if self.cos_cached is None or t != self.seq_len_cached:
            self.seq_len_cached = t
            inv_freq = self.inv_freq.to(device=x.device)
            tt = torch.arange(t, device=x.device, dtype=inv_freq.dtype)
            freqs = torch.outer(tt, inv_freq)
            self.cos_cached = freqs.cos()[None, :, None, :]
            self.sin_cached = freqs.sin()[None, :, None, :]
        return self.cos_cached, self.sin_cached


def _apply_rotary(x, cos, sin):
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)


class NGPTSelfAttention(nn.Module):
    """nGPT attention: QK unit-norm + learned s_qk scaling, RoPE."""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head

        self.q = nn.Linear(n_embd, n_embd, bias=False)
        self.k = nn.Linear(n_embd, n_embd, bias=False)
        self.v = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        self.rotary = NGPTRotary(self.head_dim)

        # s_qk: per head_dim scaling, init=1.0, stored at 1/sqrt(d)
        s_scale = 1.0 / math.sqrt(n_embd)
        self.s_qk = _ngpt_scale_param((self.head_dim,), 1.0, s_scale)

    def forward(self, x):
        B, T, C = x.size()
        q = self.q(x).view(B, T, self.n_head, self.head_dim)
        k = self.k(x).view(B, T, self.n_head, self.head_dim)
        v = self.v(x).view(B, T, self.n_head, self.head_dim)

        # nGPT: unit-norm Q, K then scale
        s_qk = _ngpt_actual(self.s_qk)
        q = _unit_norm(q, dim=-1) * s_qk
        k = _unit_norm(k, dim=-1) * s_qk

        # RoPE
        cos, sin = self.rotary(q)
        q = _apply_rotary(q, cos, sin)
        k = _apply_rotary(k, cos, sin)

        # SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2).contiguous()
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class NGPTMLP(nn.Module):
    """SwiGLU MLP for nGPT with s_u, s_v scaling."""

    def __init__(self, n_embd):
        super().__init__()
        hidden = int(n_embd * 8 / 3)
        # Snap to multiple of 8 for small models
        hidden = ((hidden + 7) // 8) * 8
        self.gate_proj = nn.Linear(n_embd, hidden, bias=False)
        self.up_proj = nn.Linear(n_embd, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, n_embd, bias=False)

        self.s_u = _ngpt_scale_param((hidden,), 1.0, 1.0)
        self.s_v = _ngpt_scale_param((hidden,), 1.0, 1.0)
        self._sqrt_d = math.sqrt(n_embd)

    def forward(self, x):
        u = self.up_proj(x) * _ngpt_actual(self.s_u)
        v = self.gate_proj(x) * _ngpt_actual(self.s_v) * self._sqrt_d
        return self.down_proj(F.silu(v) * u)


class NGPTBlock(nn.Module):
    """nGPT block: LERP updates on the hypersphere.

    h = Norm(h + alpha_A * (Norm(attn(h)) - h))
    h = Norm(h + alpha_M * (Norm(mlp(h)) - h))
    """

    def __init__(self, n_embd, n_head, block_size, n_layer):
        super().__init__()
        self.attn = NGPTSelfAttention(n_embd, n_head, block_size)
        self.mlp = NGPTMLP(n_embd)

        # Eigen learning rates (alpha)
        alpha_init = 1.0 / n_layer
        alpha_scale = 1.0 / math.sqrt(n_embd)
        self.alpha_attn = _ngpt_scale_param((n_embd,), alpha_init, alpha_scale)
        self.alpha_mlp = _ngpt_scale_param((n_embd,), alpha_init, alpha_scale)

    def forward(self, x):
        # Attention LERP
        h_a = _unit_norm(self.attn(x))
        alpha_a = _ngpt_actual(self.alpha_attn).abs()
        x = _unit_norm(x + alpha_a * (h_a - x))
        # MLP LERP
        h_m = _unit_norm(self.mlp(x))
        alpha_m = _ngpt_actual(self.alpha_mlp).abs()
        x = _unit_norm(x + alpha_m * (h_m - x))
        return x


def ngpt_normalize_weights(model):
    """Post-optimizer-step: normalize all weight matrices to unit norm.

    nn.Linear weight is (out, in) — normalize along dim=1 (input dim).
    nn.Embedding weight is (vocab, dim) — normalize along dim=1.
    Skip parameters marked with _no_weight_decay (scaling params).
    """
    skip_ids = set()
    for m in model.modules():
        for p in m.parameters(recurse=False):
            if getattr(p, '_no_weight_decay', False):
                skip_ids.add(id(p))

    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, nn.Linear):
                if id(m.weight) in skip_ids:
                    continue
                m.weight.div_(m.weight.norm(dim=1, keepdim=True).clamp(min=1e-8))
            elif isinstance(m, nn.Embedding):
                m.weight.div_(m.weight.norm(dim=1, keepdim=True).clamp(min=1e-8))


class ShakespeareNGPT(nn.Module):
    """nGPT: Normalized GPT on the hypersphere.

    - All hidden states live on the unit hypersphere
    - LERP updates instead of residual additions
    - No LayerNorm/RMSNorm — normalization is structural
    - No weight decay — weights are norm-constrained post-step
    - Learned s_z scaling on logits
    """

    def __init__(
        self,
        vocab_size,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.0,  # nGPT typically doesn't use dropout
    ):
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)

        self.blocks = nn.ModuleList(
            [NGPTBlock(n_embd, n_head, block_size, n_layer) for _ in range(n_layer)]
        )
        # No final LayerNorm — hidden state is already unit-norm
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.head.weight

        # s_z: per-token logit scaling
        s_z_scale = 1.0 / math.sqrt(n_embd)
        self.s_z = _ngpt_scale_param((vocab_size,), 1.0, s_z_scale)

        # Init weights then normalize
        self.apply(self._init_weights)
        ngpt_normalize_weights(self)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size

        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(torch.arange(T, device=idx.device))
        x = _unit_norm(tok_emb + pos_emb)  # project onto hypersphere

        for block in self.blocks:
            x = block(x)

        # x is already unit-norm
        logits = self.head(x)
        logits = logits * _ngpt_actual(self.s_z)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
