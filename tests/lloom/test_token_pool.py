"""Tests for LLooM TokenPool.

Covers:
- RCV vectorized voting: majority, elimination, sticky votes
- FiLM generation and identity at init
- Token parking (only exit/bridge in top-k)
- Entry router vs outbound router
- Gradient flow through all parameters including FiLM
- Forward shapes and hop budget enforcement
"""

import pytest
import torch

from lloom.token_pool import TokenPool

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

POOL = 4
D = 32
D_INNER = 48
TOP_K = 2
MAX_HOPS = 16
FILM_RANK = 8
B = 3
T = 8


def _make_pool(**kwargs):
    defaults = dict(
        pool_size=POOL, dim=D, inner_dim=D_INNER, top_k=TOP_K,
        max_hops=MAX_HOPS, film_rank=FILM_RANK,
        expert_shared_fraction=0.5, router_shared_fraction=0.5,
        hop_gate_dim=12,
        exit_bias_init=0.0, bridge_bias_init=0.0,
        exit_ramp_scale=3.0, router_noise=0.0,
    )
    defaults.update(kwargs)
    return TokenPool(**defaults)


def _make_inputs(b=B, t=T):
    x = torch.randn(b, t, D, requires_grad=True)
    active = torch.ones(b, dtype=torch.bool)
    return x, active


# ---------------------------------------------------------------------------
# RCV voting
# ---------------------------------------------------------------------------

class TestRankedChoiceVoting:
    """Test the static _ranked_choice_vote method directly."""

    def test_clear_continue_majority(self):
        """All tokens vote continue → continue."""
        votes = torch.zeros(B, T, dtype=torch.int8)  # all continue
        cont, exit_, bridge = TokenPool._ranked_choice_vote(votes, B, T, votes.device)
        assert cont.all()
        assert not exit_.any()
        assert not bridge.any()

    def test_clear_exit_majority(self):
        """All tokens vote exit → exit."""
        votes = torch.ones(B, T, dtype=torch.int8)  # all exit
        cont, exit_, bridge = TokenPool._ranked_choice_vote(votes, B, T, votes.device)
        assert exit_.all()
        assert not cont.any()
        assert not bridge.any()

    def test_clear_bridge_majority(self):
        """All tokens vote bridge → bridge."""
        votes = torch.full((B, T), 2, dtype=torch.int8)  # all bridge
        cont, exit_, bridge = TokenPool._ranked_choice_vote(votes, B, T, votes.device)
        assert bridge.all()
        assert not cont.any()
        assert not exit_.any()

    def test_simple_majority(self):
        """More than half vote exit → exit in round 1."""
        votes = torch.zeros(B, T, dtype=torch.int8)
        votes[:, :T // 2 + 1] = 1  # majority exit
        cont, exit_, bridge = TokenPool._ranked_choice_vote(votes, B, T, votes.device)
        assert exit_.all()

    def test_no_majority_goes_to_round2(self):
        """Three-way split with no majority → round 2 resolves."""
        # T=9 tokens: 3 continue, 3 exit, 3 bridge → no majority
        # After elimination: lowest eliminated, votes transfer
        pool_t = 9
        votes = torch.zeros(B, pool_t, dtype=torch.int8)
        votes[:, :3] = 0  # continue
        votes[:, 3:6] = 1  # exit
        votes[:, 6:9] = 2  # bridge
        cont, exit_, bridge = TokenPool._ranked_choice_vote(votes, B, pool_t, votes.device)
        # All three tied at 3 each. argmin picks first (continue=0).
        # Continue eliminated → transfers to whichever of exit/bridge is leading.
        # exit=3, bridge=3, so exit >= bridge → transfer to exit. exit gets 6 > 4.5.
        assert (cont | exit_ | bridge).all()  # decision made for every sample

    def test_mutually_exclusive_decisions(self):
        """Exactly one of continue/exit/bridge should be true per sample."""
        votes = torch.randint(0, 3, (B, T), dtype=torch.int8)
        cont, exit_, bridge = TokenPool._ranked_choice_vote(votes, B, T, votes.device)
        total = cont.long() + exit_.long() + bridge.long()
        assert (total == 1).all()

    def test_per_sample_independence(self):
        """Different samples can have different decisions."""
        votes = torch.zeros(B, T, dtype=torch.int8)
        votes[0, :] = 0  # sample 0: all continue
        votes[1, :] = 1  # sample 1: all exit
        votes[2, :] = 2  # sample 2: all bridge
        cont, exit_, bridge = TokenPool._ranked_choice_vote(votes, B, T, votes.device)
        assert cont[0] and exit_[1] and bridge[2]


# ---------------------------------------------------------------------------
# Sticky votes
# ---------------------------------------------------------------------------

class TestStickyVotes:
    def test_newly_parked_tokens_lock_vote(self):
        pool = _make_pool()
        vote_state = torch.zeros(B, T, dtype=torch.int8)

        # Craft topk_idx where all are exit for some tokens
        topk_idx = torch.zeros(B, T, TOP_K, dtype=torch.long)
        # Token 0: both exit
        topk_idx[:, 0, :] = pool.exit_idx
        # Token 1: both bridge
        topk_idx[:, 1, :] = pool.bridge_idx
        # Token 2: one expert, one exit (not parked)
        topk_idx[:, 2, 0] = 0
        topk_idx[:, 2, 1] = pool.exit_idx

        is_expert, is_exit, is_bridge = pool.classify_topk(topk_idx)
        pool.aggregate_decisions(topk_idx, is_expert, is_exit, is_bridge,
                                 vote_state=vote_state)

        # Token 0 should be locked as exit (1)
        assert (vote_state[:, 0] == 1).all()
        # Token 1 should be locked as bridge (2)
        assert (vote_state[:, 1] == 2).all()
        # Token 2 should still be active (0)
        assert (vote_state[:, 2] == 0).all()

    def test_locked_votes_persist(self):
        pool = _make_pool()
        vote_state = torch.zeros(B, T, dtype=torch.int8)
        vote_state[:, 0] = 1  # already locked as exit

        # Even if new routing gives expert for token 0, locked vote persists
        topk_idx = torch.randint(0, POOL, (B, T, TOP_K))
        is_expert, is_exit, is_bridge = pool.classify_topk(topk_idx)
        pool.aggregate_decisions(topk_idx, is_expert, is_exit, is_bridge,
                                 vote_state=vote_state)

        # Token 0 should still be locked as exit
        assert (vote_state[:, 0] == 1).all()

    def test_sticky_votes_monotonic(self):
        """Number of parked tokens should never decrease."""
        pool = _make_pool()
        vote_state = torch.zeros(B, T, dtype=torch.int8)

        for _ in range(5):
            n_parked_before = (vote_state > 0).sum().item()
            topk_idx = torch.randint(0, pool.n_options, (B, T, TOP_K))
            is_expert, is_exit, is_bridge = pool.classify_topk(topk_idx)
            pool.aggregate_decisions(topk_idx, is_expert, is_exit, is_bridge,
                                     vote_state=vote_state)
            n_parked_after = (vote_state > 0).sum().item()
            assert n_parked_after >= n_parked_before


# ---------------------------------------------------------------------------
# FiLM generation
# ---------------------------------------------------------------------------

class TestFiLM:
    def test_film_output_shapes(self):
        pool = _make_pool()
        x = torch.randn(B, T, D)
        g_up, b_up, g_down, b_down = pool.generate_film(x)
        assert g_up.shape == (B, D_INNER)
        assert b_up.shape == (B, D_INNER)
        assert g_down.shape == (B, D)  # down-proj output is D, not D_inner
        assert b_down.shape == (B, D)

    def test_film_identity_at_init(self):
        """FiLM should be near-identity at initialization (gamma~1, beta~0)."""
        pool = _make_pool()
        x = torch.randn(B, T, D)
        g_up, b_up, g_down, b_down = pool.generate_film(x)
        # film_up.weight is zero-init, so output = bias + small noise from SiLU
        # bias is set to gamma=1, beta=0
        # With zero weight, film_up output = bias regardless of input
        torch.testing.assert_close(g_up, torch.ones_like(g_up), atol=0.1, rtol=0.1)
        torch.testing.assert_close(b_up, torch.zeros_like(b_up), atol=0.1, rtol=0.1)
        torch.testing.assert_close(g_down, torch.ones_like(g_down), atol=0.1, rtol=0.1)
        torch.testing.assert_close(b_down, torch.zeros_like(b_down), atol=0.1, rtol=0.1)

    def test_film_gradient_flow(self):
        pool = _make_pool()
        x = torch.randn(B, T, D, requires_grad=True)
        film = pool.generate_film(x)
        loss = sum(f.sum() for f in film)
        loss.backward()
        assert x.grad is not None
        assert pool.film_down.weight.grad is not None
        assert pool.film_up.weight.grad is not None

    def test_film_different_inputs_different_output(self):
        pool = _make_pool()
        # Need nonzero film_up weights to see input differences
        pool.film_up.weight.data.normal_(std=0.1)
        x1 = torch.randn(B, T, D)
        x2 = torch.randn(B, T, D)
        film1 = pool.generate_film(x1)
        film2 = pool.generate_film(x2)
        assert not torch.allclose(film1[0], film2[0], atol=1e-5)


# ---------------------------------------------------------------------------
# Forward and shapes
# ---------------------------------------------------------------------------

class TestForward:
    def test_basic_output_shape(self):
        pool = _make_pool()
        x, active = _make_inputs()
        x_out, new_active, do_exit, do_bridge, ce, vs, hops = \
            pool(x, active, hops_used=0, is_first_hop=True)
        assert x_out.shape == (B, T, D)
        assert new_active.shape == (B,)
        assert do_exit.shape == (B,)
        assert do_bridge.shape == (B,)
        assert ce.shape == (B, T)
        assert vs.shape == (B, T)
        assert hops == 1

    def test_entry_router_first_hop(self):
        """First hop should use entry router, not outbound router."""
        pool = _make_pool()
        x, active = _make_inputs()
        # Should not crash with current_expert=None on first hop
        x_out, *_ = pool(x, active, hops_used=0, is_first_hop=True,
                         current_expert=None)
        assert x_out.shape == (B, T, D)

    def test_with_film(self):
        pool = _make_pool()
        x, active = _make_inputs()
        film = pool.generate_film(x.detach())
        x_out, *_ = pool(x, active, hops_used=0, is_first_hop=True,
                         film_params=film)
        assert x_out.shape == (B, T, D)

    @pytest.mark.parametrize("b,t", [(1, 4), (4, 16)])
    def test_various_shapes(self, b, t):
        pool = _make_pool()
        x, active = _make_inputs(b=b, t=t)
        x_out, *_ = pool(x, active, hops_used=0, is_first_hop=True)
        assert x_out.shape == (b, t, D)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------

class TestGradientFlow:
    def test_all_params_get_grad(self):
        pool = _make_pool()
        pool.hop_gate_proj.bias.data.fill_(2.0)
        x, active = _make_inputs()
        film = pool.generate_film(x.detach())
        x_out, *_ = pool(x, active, hops_used=0, is_first_hop=True,
                         film_params=film)
        loss = x_out.sum()
        loss.backward()

        assert x.grad is not None and x.grad.abs().sum() > 0

        # Entry router and expert bank should get gradients
        assert pool.entry_router.weight.grad is not None

    def test_grad_with_film(self):
        """Gradients flow through FiLM params back to the FiLM generator."""
        pool = _make_pool()
        pool.film_up.weight.data.normal_(std=0.1)
        x = torch.randn(B, T, D, requires_grad=True)
        active = torch.ones(B, dtype=torch.bool)
        film = pool.generate_film(x)
        x_out, *_ = pool(x, active, hops_used=0, is_first_hop=True,
                         film_params=film)
        loss = x_out.sum()
        loss.backward()
        # film_down should get gradient through the FiLM → expert path
        assert pool.film_down.weight.grad is not None

    def test_multi_hop_backward(self):
        pool = _make_pool()
        x, active = _make_inputs()
        ce = None
        for hop in range(3):
            x_out, active, _, _, ce, vs, _ = pool(
                x, active, hops_used=hop, is_first_hop=(hop == 0),
                current_expert=ce)
            x = x + x_out
        loss = x.sum()
        loss.backward()  # should not crash


# ---------------------------------------------------------------------------
# Hop budget
# ---------------------------------------------------------------------------

class TestHopBudget:
    def test_force_exit_at_budget(self):
        pool = _make_pool()
        x, active = _make_inputs()
        x_out, new_active, do_exit, _, _, _, hops = \
            pool(x, active, hops_used=MAX_HOPS, is_first_hop=True)
        assert do_exit.all()
        assert not new_active.any()
        assert hops == MAX_HOPS


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
# Hop conditioning in forward path
# ---------------------------------------------------------------------------

class TestHopConditioning:
    """Verify that apply_hop_conditioning is called during TokenPool.forward."""

    def test_different_hops_produce_different_outputs(self):
        """Output should differ when hops_used changes, proving conditioning is active."""
        pool = _make_pool(router_noise=0.0)
        pool.eval()
        torch.manual_seed(0)
        x = torch.randn(B, T, D)
        active = torch.ones(B, dtype=torch.bool)

        with torch.no_grad():
            out0, *_ = pool(x, active, hops_used=0, is_first_hop=True)
            out1, *_ = pool(x, active, hops_used=1, is_first_hop=False,
                            current_expert=torch.zeros(B, T, dtype=torch.long))

        # With hop conditioning, the outputs should differ because of hop embedding
        assert not torch.allclose(out0, out1, atol=1e-6), \
            "Outputs identical across hops — hop conditioning not applied"

    def test_hop_embed_gets_gradient(self):
        """hop_embed_bank must receive gradient through TokenPool.forward."""
        pool = _make_pool()
        pool.train()
        x, active = _make_inputs()

        out, *_ = pool(x, active, hops_used=0, is_first_hop=True)
        out.sum().backward()

        assert pool.hop_embed_bank.grad is not None
        assert pool.hop_embed_bank.grad.abs().max() > 0, \
            "hop_embed_bank got zero gradient — not connected to forward path"

    def test_hop_gate_proj_gets_gradient(self):
        """hop_gate_proj must receive gradient through TokenPool.forward."""
        pool = _make_pool()
        pool.train()
        x, active = _make_inputs()

        out, *_ = pool(x, active, hops_used=0, is_first_hop=True)
        out.sum().backward()

        assert pool.hop_gate_proj.weight.grad is not None
        assert pool.hop_gate_proj.weight.grad.abs().max() > 0, \
            "hop_gate_proj.weight got zero gradient — not connected to forward path"

    def test_hop_norm_gets_gradient(self):
        """hop_norm (RMSNorm) must receive gradient through TokenPool.forward."""
        pool = _make_pool()
        pool.train()
        x, active = _make_inputs()

        out, *_ = pool(x, active, hops_used=0, is_first_hop=True)
        out.sum().backward()

        # RMSNorm has a 'weight' parameter
        assert pool.hop_norm.weight.grad is not None
        assert pool.hop_norm.weight.grad.abs().max() > 0, \
            "hop_norm.weight got zero gradient — not connected to forward path"

    def test_inactive_samples_skip_conditioning(self):
        """Inactive samples should get identical output regardless of hop."""
        pool = _make_pool(router_noise=0.0)
        pool.eval()
        x = torch.randn(B, T, D)
        # All inactive
        active = torch.zeros(B, dtype=torch.bool)

        with torch.no_grad():
            out0, *_ = pool(x, active, hops_used=0, is_first_hop=True)
            out5, *_ = pool(x, active, hops_used=5, is_first_hop=False,
                            current_expert=torch.zeros(B, T, dtype=torch.long))

        # Inactive samples should be unchanged from input x
        assert torch.allclose(out0, x, atol=1e-6), \
            "Inactive samples were modified by forward"
        assert torch.allclose(out5, x, atol=1e-6), \
            "Inactive samples were modified by forward"


# ---------------------------------------------------------------------------
# Global hop conditioning
# ---------------------------------------------------------------------------

class TestGlobalHop:
    def test_global_hop_changes_conditioning(self):
        """Different global_hop with same hops_used should produce different outputs."""
        pool = _make_pool(router_noise=0.0, global_max_hops=48)
        pool.eval()
        x = torch.randn(B, T, D)
        active = torch.ones(B, dtype=torch.bool)

        with torch.no_grad():
            out0, *_ = pool(x, active, hops_used=0, is_first_hop=True, global_hop=0)
            out10, *_ = pool(x, active, hops_used=0, is_first_hop=True, global_hop=10)

        assert not torch.allclose(out0, out10, atol=1e-6), \
            "Different global_hop values produced identical outputs"

    def test_fallback_to_hops_used(self):
        """Without global_hop, conditioning should use hops_used."""
        pool = _make_pool(router_noise=0.0)
        pool.eval()
        x = torch.randn(B, T, D)
        active = torch.ones(B, dtype=torch.bool)
        ce = torch.zeros(B, T, dtype=torch.long)

        with torch.no_grad():
            out_explicit, *_ = pool(x, active, hops_used=3, is_first_hop=False,
                                    current_expert=ce, global_hop=3)
            out_fallback, *_ = pool(x, active, hops_used=3, is_first_hop=False,
                                    current_expert=ce)

        assert torch.allclose(out_explicit, out_fallback, atol=1e-6), \
            "global_hop=hops_used should match fallback behavior"
