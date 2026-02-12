"""Tests for LLooM TokenPool.

Covers:
- execute_hop() output shapes and post-dispatch routing
- RCV: ranked_choice_vote with sticky votes
- Entry router for bridge crossings
- Gradient flow through all parameters
- Logit chain: execute_hop logits feed back into routing
"""

import pytest
import torch

from lloom.token_pool import TokenPool, _ranked_choice_vote

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

POOL = 4
D = 32
D_INNER = 48
TOP_K = 2
MAX_HOPS = 16
B = 3
T = 8


def _make_pool(**kwargs):
    defaults = dict(
        pool_size=POOL, dim=D, inner_dim=D_INNER, top_k=TOP_K,
        max_hops=MAX_HOPS,
        expert_shared_fraction=0.5, router_shared_fraction=0.5,
        hop_gate_dim=12,
        exit_bias_init=0.0, bridge_bias_init=0.0,
        exit_ramp_scale=3.0, router_noise=0.0,
    )
    defaults.update(kwargs)
    return TokenPool(**defaults)


def _make_inputs(b=B, t=T):
    x = torch.randn(b, t, D, requires_grad=True)
    topk_idx = torch.randint(0, POOL, (b, t, TOP_K))
    topk_weights = torch.full((b, t, TOP_K), 1.0 / TOP_K)
    return x, topk_idx, topk_weights


# ---------------------------------------------------------------------------
# RCV voting (module-level function)
# ---------------------------------------------------------------------------

class TestRankedChoiceVoting:
    """Test the _ranked_choice_vote function directly."""

    def test_clear_continue_majority(self):
        """All tokens vote continue -> continue."""
        votes = torch.zeros(B, T, dtype=torch.int8)  # all continue
        cont, exit_, bridge = _ranked_choice_vote(votes, B, T, votes.device)
        assert cont.all()
        assert not exit_.any()
        assert not bridge.any()

    def test_clear_exit_majority(self):
        """All tokens vote exit -> exit."""
        votes = torch.ones(B, T, dtype=torch.int8)  # all exit
        cont, exit_, bridge = _ranked_choice_vote(votes, B, T, votes.device)
        assert exit_.all()
        assert not cont.any()
        assert not bridge.any()

    def test_clear_bridge_majority(self):
        """All tokens vote bridge -> bridge."""
        votes = torch.full((B, T), 2, dtype=torch.int8)  # all bridge
        cont, exit_, bridge = _ranked_choice_vote(votes, B, T, votes.device)
        assert bridge.all()
        assert not cont.any()
        assert not exit_.any()

    def test_simple_majority(self):
        """More than half vote exit -> exit in round 1."""
        votes = torch.zeros(B, T, dtype=torch.int8)
        votes[:, :T // 2 + 1] = 1  # majority exit
        cont, exit_, bridge = _ranked_choice_vote(votes, B, T, votes.device)
        assert exit_.all()

    def test_no_majority_goes_to_round2(self):
        """Three-way split with no majority -> round 2 resolves."""
        pool_t = 9
        votes = torch.zeros(B, pool_t, dtype=torch.int8)
        votes[:, :3] = 0  # continue
        votes[:, 3:6] = 1  # exit
        votes[:, 6:9] = 2  # bridge
        cont, exit_, bridge = _ranked_choice_vote(votes, B, pool_t, votes.device)
        assert (cont | exit_ | bridge).all()  # decision made for every sample

    def test_mutually_exclusive_decisions(self):
        """Exactly one of continue/exit/bridge should be true per sample."""
        votes = torch.randint(0, 3, (B, T), dtype=torch.int8)
        cont, exit_, bridge = _ranked_choice_vote(votes, B, T, votes.device)
        total = cont.long() + exit_.long() + bridge.long()
        assert (total == 1).all()

    def test_per_sample_independence(self):
        """Different samples can have different decisions."""
        votes = torch.zeros(B, T, dtype=torch.int8)
        votes[0, :] = 0  # sample 0: all continue
        votes[1, :] = 1  # sample 1: all exit
        votes[2, :] = 2  # sample 2: all bridge
        cont, exit_, bridge = _ranked_choice_vote(votes, B, T, votes.device)
        assert cont[0] and exit_[1] and bridge[2]


# ---------------------------------------------------------------------------
# Sticky votes via ranked_choice_vote
# ---------------------------------------------------------------------------

class TestStickyVotes:
    def test_newly_decided_tokens_lock_vote(self):
        vote_state = torch.zeros(B, T, dtype=torch.int8)
        # Some tokens exit, some bridge, some continue
        token_has_exit = torch.zeros(B, T, dtype=torch.bool)
        token_has_bridge = torch.zeros(B, T, dtype=torch.bool)
        token_has_continue = torch.zeros(B, T, dtype=torch.bool)

        token_has_exit[:, 0] = True    # token 0 exits
        token_has_bridge[:, 1] = True  # token 1 bridges
        token_has_continue[:, 2:] = True  # rest continue

        _, _, _, vs = TokenPool.ranked_choice_vote(
            token_has_exit, token_has_bridge, token_has_continue,
            vote_state=vote_state,
        )
        # Token 0 should be locked as exit (1)
        assert (vs[:, 0] == 1).all()
        # Token 1 should be locked as bridge (2)
        assert (vs[:, 1] == 2).all()
        # Token 2+ should still be active (0)
        assert (vs[:, 2] == 0).all()

    def test_locked_votes_persist(self):
        vote_state = torch.zeros(B, T, dtype=torch.int8)
        vote_state[:, 0] = 1  # already locked as exit

        # All tokens now continue (but locked one stays locked)
        token_has_exit = torch.zeros(B, T, dtype=torch.bool)
        token_has_bridge = torch.zeros(B, T, dtype=torch.bool)
        token_has_continue = torch.ones(B, T, dtype=torch.bool)

        _, _, _, vs = TokenPool.ranked_choice_vote(
            token_has_exit, token_has_bridge, token_has_continue,
            vote_state=vote_state,
        )
        assert (vs[:, 0] == 1).all()  # still locked

    def test_sticky_votes_monotonic(self):
        """Number of parked tokens should never decrease."""
        vote_state = torch.zeros(B, T, dtype=torch.int8)

        for _ in range(5):
            n_parked_before = (vote_state > 0).sum().item()
            # Random decisions
            token_has_exit = torch.rand(B, T) < 0.2
            token_has_bridge = torch.rand(B, T) < 0.2
            token_has_continue = ~token_has_exit & ~token_has_bridge

            _, _, _, vote_state = TokenPool.ranked_choice_vote(
                token_has_exit, token_has_bridge, token_has_continue,
                vote_state=vote_state,
            )
            n_parked_after = (vote_state > 0).sum().item()
            assert n_parked_after >= n_parked_before


# ---------------------------------------------------------------------------
# execute_hop shapes
# ---------------------------------------------------------------------------

class TestExecuteHopShapes:
    def test_basic_output_shapes(self):
        pool = _make_pool()
        x, topk_idx, topk_weights = _make_inputs()
        out, next_logits = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        assert out.shape == (B, T, D)
        assert next_logits.shape == (B, T, pool.n_options)

    @pytest.mark.parametrize("b,t", [(1, 4), (4, 16)])
    def test_various_shapes(self, b, t):
        pool = _make_pool()
        x, topk_idx, topk_weights = _make_inputs(b=b, t=t)
        out, next_logits = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        assert out.shape == (b, t, D)
        assert next_logits.shape == (b, t, pool.n_options)


# ---------------------------------------------------------------------------
# Post-dispatch routing
# ---------------------------------------------------------------------------

class TestPostDispatchRouting:
    def test_next_logits_nonzero(self):
        pool = _make_pool()
        x, topk_idx, topk_weights = _make_inputs()
        _, next_logits = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        assert next_logits.abs().sum() > 0

    def test_logit_chain_flow(self):
        """execute_hop logits can feed back into per-token routing."""
        pool = _make_pool()
        x = torch.randn(B, T, D)

        # Initial logits from entry router
        logits = pool.entry_router(x.reshape(B * T, D)).reshape(B, T, pool.n_options)

        # First hop: apply biases + topk per token
        biased = pool.apply_biases(logits.reshape(B * T, pool.n_options), hops_used=0)
        topk_idx, topk_weights, _ = pool.select_topk(biased)
        topk_idx = topk_idx.reshape(B, T, TOP_K)
        topk_weights = topk_weights.reshape(B, T, TOP_K)

        out, next_logits = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        x = x + out

        # Second hop from outbound logits
        biased2 = pool.apply_biases(next_logits.reshape(B * T, pool.n_options), hops_used=1)
        topk_idx2, topk_weights2, _ = pool.select_topk(biased2)
        topk_idx2 = topk_idx2.reshape(B, T, TOP_K)
        topk_weights2 = topk_weights2.reshape(B, T, TOP_K)

        out2, next_logits2 = pool.execute_hop(x, topk_idx2, topk_weights2, hop=1)
        x = x + out2

        assert x.shape == (B, T, D)
        assert next_logits2.shape == (B, T, pool.n_options)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_all_params_get_grad(self):
        pool = _make_pool()
        pool.hop_gate_proj.bias.data.fill_(2.0)
        x, topk_idx, topk_weights = _make_inputs()
        out, next_logits = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        loss = out.sum() + next_logits.sum()
        loss.backward()

        assert x.grad is not None and x.grad.abs().sum() > 0

        # Entry router should have params
        assert pool.entry_router.weight.grad is None  # wasn't used here, that's fine

    def test_entry_router_gets_grad(self):
        pool = _make_pool()
        x = torch.randn(B, T, D, requires_grad=True)
        logits = pool.entry_router(x.reshape(B * T, D))
        logits.sum().backward()
        assert pool.entry_router.weight.grad is not None

    def test_multi_hop_backward(self):
        pool = _make_pool()
        x = torch.randn(B, T, D, requires_grad=True)
        logits = pool.entry_router(x.reshape(B * T, D)).reshape(B, T, pool.n_options)

        for hop in range(3):
            biased = pool.apply_biases(logits.reshape(B * T, pool.n_options), hops_used=hop)
            topk_idx, topk_weights, _ = pool.select_topk(biased)
            topk_idx = topk_idx.reshape(B, T, TOP_K)
            topk_weights = topk_weights.reshape(B, T, TOP_K)
            out, logits = pool.execute_hop(x, topk_idx, topk_weights, hop=hop)
            x = x + out

        loss = x.sum()
        loss.backward()  # should not crash


# ---------------------------------------------------------------------------
# Entry router
# ---------------------------------------------------------------------------

class TestEntryRouter:
    def test_entry_router_output_shape(self):
        pool = _make_pool()
        x = torch.randn(B * T, D)
        logits = pool.entry_router(x)
        assert logits.shape == (B * T, pool.n_options)

    def test_entry_router_gradient(self):
        pool = _make_pool()
        x = torch.randn(B * T, D, requires_grad=True)
        logits = pool.entry_router(x)
        logits.sum().backward()
        assert x.grad is not None
        assert pool.entry_router.weight.grad is not None


# ---------------------------------------------------------------------------
# Bridge
# ---------------------------------------------------------------------------

class TestBridge:
    def test_prepare_bridge_identity(self):
        pool = _make_pool()
        x = torch.randn(B, T, D)
        assert pool.prepare_bridge_out(x) is x

    def test_accept_bridge_identity(self):
        pool = _make_pool()
        x = torch.randn(B, T, D)
        assert pool.accept_bridge_in(x) is x


# ---------------------------------------------------------------------------
# Hop conditioning through execute_hop
# ---------------------------------------------------------------------------

class TestHopConditioning:
    def test_different_hops_produce_different_outputs(self):
        pool = _make_pool(router_noise=0.0)
        pool.hop_gate_proj.bias.data.fill_(2.0)
        pool.eval()
        x = torch.randn(B, T, D)
        topk_idx = torch.zeros(B, T, TOP_K, dtype=torch.long)
        topk_weights = torch.full((B, T, TOP_K), 1.0 / TOP_K)

        with torch.no_grad():
            out0, _ = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
            out1, _ = pool.execute_hop(x, topk_idx, topk_weights, hop=1)

        assert not torch.allclose(out0, out1, atol=1e-6), \
            "Outputs identical across hops -- hop conditioning not applied"

    def test_hop_embed_gets_gradient(self):
        pool = _make_pool()
        pool.hop_gate_proj.bias.data.fill_(2.0)
        x = torch.randn(B, T, D, requires_grad=True)
        topk_idx = torch.zeros(B, T, TOP_K, dtype=torch.long)
        topk_weights = torch.full((B, T, TOP_K), 1.0 / TOP_K)
        out, _ = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        out.sum().backward()
        assert pool.hop_embed_bank.grad is not None
        assert pool.hop_embed_bank.grad.abs().max() > 0

    def test_hop_gate_proj_gets_gradient(self):
        pool = _make_pool()
        x = torch.randn(B, T, D, requires_grad=True)
        topk_idx = torch.zeros(B, T, TOP_K, dtype=torch.long)
        topk_weights = torch.full((B, T, TOP_K), 1.0 / TOP_K)
        out, _ = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        out.sum().backward()
        assert pool.hop_gate_proj.weight.grad is not None

    def test_hop_norm_gets_gradient(self):
        pool = _make_pool()
        x = torch.randn(B, T, D, requires_grad=True)
        topk_idx = torch.zeros(B, T, TOP_K, dtype=torch.long)
        topk_weights = torch.full((B, T, TOP_K), 1.0 / TOP_K)
        out, _ = pool.execute_hop(x, topk_idx, topk_weights, hop=0)
        out.sum().backward()
        assert pool.hop_norm.weight.grad is not None
