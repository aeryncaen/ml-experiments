"""DS1 SSM behavioral tests: validates DS1 exhibits SSM-like properties.

Tests whether DS1 can learn fundamental SSM tasks:
1. Shift/delay: output[t] = input[t-k] (memory)
2. Selective copying: copy marked tokens to output (selective recall)
3. Induction heads: repeat pattern completion (in-context learning)

These are the standard SSM litmus tests. If DS1 can't learn these,
it's not functioning as an SSM regardless of architecture.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ds_moe.model import DS1


DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


def _make_trainable_ds1(dim, state_dim=32, mimo_rank=2, n_iters=2, **kwargs):
    """Creates a DS1 with trainable bank weights + linear readout."""
    ds1 = DS1(dim=dim, state_dim=state_dim, mimo_rank=mimo_rank,
              n_iters=n_iters, **kwargs).to(DEVICE)
    bank_size = DS1.bank_size(dim, state_dim, mimo_rank)
    bank = nn.Parameter(torch.randn(bank_size, device=DEVICE) * 0.02)
    return ds1, bank


def _train_loop(ds1, bank, readout, train_fn, n_steps=500, lr=1e-3):
    """Generic training loop. train_fn(batch_idx) -> (x, target).
    Returns final loss.
    """
    params = [bank] + list(ds1.parameters()) + list(readout.parameters())
    opt = optim.Adam(params, lr=lr)
    losses = []
    for step in range(n_steps):
        x, target = train_fn(step)
        x, target = x.to(DEVICE), target.to(DEVICE)
        y = ds1(x, bank)
        pred = readout(y)
        loss = nn.functional.mse_loss(pred, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses[-1], losses


class TestDS1Delay:
    """Can DS1 learn output[t] = input[t-k]? This is the most basic SSM test.
    A system with memory should be able to delay a signal by k steps.
    """

    def test_learns_delay(self):
        D = 64
        L = 32
        B = 32
        delay = 1
        ds1, bank = _make_trainable_ds1(D, state_dim=64, mimo_rank=4)
        readout = nn.Linear(D, D).to(DEVICE)

        def gen_batch(step):
            x = torch.randn(B, L, D)
            target = torch.zeros_like(x)
            target[:, delay:, :] = x[:, :L - delay, :]
            return x, target

        final_loss, losses = _train_loop(ds1, bank, readout, gen_batch, n_steps=2000, lr=1e-3)
        initial_loss = sum(losses[:10]) / 10
        assert final_loss < initial_loss * 0.5, (
            f"DS1 failed to learn delay=1: "
            f"initial={initial_loss:.4f}, final={final_loss:.4f}"
        )


class TestDS1SelectiveCopy:
    """Can DS1 selectively copy marked tokens?

    Input: sequence of (token, marker) pairs.
    marker=1 means "remember this token".
    Output: the remembered tokens in order, packed at the start.

    This tests selective memory — the SSM must gate what it stores
    based on input content, not just position.
    """

    def test_learns_selective_copy(self):
        D = 16
        L = 32
        B = 32
        n_marked = 4
        ds1, bank = _make_trainable_ds1(D, state_dim=32, mimo_rank=2,
                                        diff_inject=True, diff_readout=True)
        readout = nn.Linear(D, D).to(DEVICE)

        def gen_batch(step):
            tokens = torch.randn(B, L, D - 1)
            markers = torch.zeros(B, L, 1)
            positions = torch.stack([
                torch.randperm(L)[:n_marked] for _ in range(B)
            ])
            for b in range(B):
                markers[b, positions[b], 0] = 1.0

            x = torch.cat([tokens, markers], dim=-1)
            target = torch.zeros(B, L, D)
            for b in range(B):
                sorted_pos = positions[b].sort().values
                for i, p in enumerate(sorted_pos):
                    if i < L:
                        target[b, i, :D - 1] = tokens[b, p]
                        target[b, i, D - 1] = 1.0
            return x, target

        final_loss, losses = _train_loop(ds1, bank, readout, gen_batch, n_steps=2000, lr=1e-3)
        initial_loss = sum(losses[:10]) / 10
        assert final_loss < initial_loss * 0.9, (
            f"DS1 failed selective copy: "
            f"initial={initial_loss:.4f}, final={final_loss:.4f}. "
            f"Note: selective copy is hard for a single SSM — "
            f"full model with attention should do much better."
        )


class TestDS1InductionHead:
    """Can DS1 complete repeated patterns?

    Input: [A B C ... A B ?]
    Target: C (the token that followed A B last time)

    This tests in-context pattern matching — the SSM must
    recognize "I've seen this prefix before" and recall what followed.
    """

    def test_learns_pattern_completion(self):
        vocab_size = 16
        D = 16
        seq_len = 64
        B = 32
        ds1, bank = _make_trainable_ds1(D, state_dim=32, mimo_rank=2)
        embed = nn.Embedding(vocab_size, D).to(DEVICE)
        head = nn.Linear(D, vocab_size).to(DEVICE)

        def gen_batch(step):
            pattern_len = 8
            pattern = torch.randint(0, vocab_size, (B, pattern_len))
            n_repeats = seq_len // pattern_len
            tokens = pattern.repeat(1, n_repeats)[:, :seq_len]
            x_emb = embed(tokens.to(DEVICE))
            target_tokens = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1).to(DEVICE)
            return x_emb, target_tokens

        params = [bank] + list(ds1.parameters()) + list(embed.parameters()) + list(head.parameters())
        opt = optim.Adam(params, lr=1e-3)
        losses = []

        for step in range(1000):
            x_emb, target = gen_batch(step)
            y = ds1(x_emb, bank)
            logits = head(y)
            loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        final_loss = sum(losses[-20:]) / 20
        initial_loss = sum(losses[:20]) / 20

        assert final_loss < initial_loss * 0.3, (
            f"DS1 failed induction head: "
            f"initial={initial_loss:.4f}, final={final_loss:.4f}"
        )

    def test_accuracy_on_pattern_completion(self):
        vocab_size = 8
        D = 16
        pattern_len = 4
        seq_len = 32
        B = 64
        ds1, bank = _make_trainable_ds1(D, state_dim=32, mimo_rank=2,
                                        diff_inject=True, diff_readout=True, bc_norm=True)
        embed = nn.Embedding(vocab_size, D).to(DEVICE)
        head = nn.Linear(D, vocab_size).to(DEVICE)

        params = [bank] + list(ds1.parameters()) + list(embed.parameters()) + list(head.parameters())
        opt = optim.Adam(params, lr=1e-3)

        for step in range(1500):
            pattern = torch.randint(0, vocab_size, (B, pattern_len))
            n_repeats = seq_len // pattern_len
            tokens = pattern.repeat(1, n_repeats)
            x_emb = embed(tokens.to(DEVICE))
            target = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1).to(DEVICE)

            y = ds1(x_emb, bank)
            logits = head(y)
            loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), target.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            pattern = torch.randint(0, vocab_size, (256, pattern_len))
            tokens = pattern.repeat(1, n_repeats)
            x_emb = embed(tokens.to(DEVICE))
            target = torch.cat([tokens[:, 1:], tokens[:, :1]], dim=1).to(DEVICE)
            y = ds1(x_emb, bank)
            logits = head(y)
            preds = logits.argmax(dim=-1)
            acc = (preds == target).float().mean().item()

        assert acc > 0.7, f"DS1 pattern completion accuracy {acc:.2%} < 70%"


class TestDS1PositionalSensitivity:
    """Verify DS1 is position-sensitive: same content at different positions
    should produce different outputs (due to positional RoPE on theta).
    """

    def test_different_positions_different_output(self):
        D = 16
        L = 16
        ds1 = DS1(dim=D, state_dim=16, mimo_rank=2, n_iters=2).to(DEVICE)
        bank_size = DS1.bank_size(D, 16, 2)
        bank = torch.randn(bank_size, device=DEVICE) * 0.1

        x = torch.zeros(1, L, D, device=DEVICE)
        token = torch.randn(D, device=DEVICE)
        x[0, 2, :] = token
        y1 = ds1(x, bank)

        x2 = torch.zeros(1, L, D, device=DEVICE)
        x2[0, 8, :] = token
        y2 = ds1(x2, bank)

        assert not torch.allclose(y1, y2, atol=1e-6), (
            "DS1 produced identical output for same token at different positions — "
            "positional RoPE may not be working"
        )


class TestDS1Determinism:
    def test_same_input_same_output(self):
        D = 16
        ds1 = DS1(dim=D, state_dim=16, mimo_rank=2).to(DEVICE)
        bank = torch.randn(DS1.bank_size(D, 16, 2), device=DEVICE)
        x = torch.randn(2, 8, D, device=DEVICE)
        ds1.eval()
        with torch.no_grad():
            y1 = ds1(x, bank)
            y2 = ds1(x, bank)
        torch.testing.assert_close(y1, y2)
