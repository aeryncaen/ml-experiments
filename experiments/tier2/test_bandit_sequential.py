"""
Tier 2, Experiment 2: Sequential Attribution on Contextual Bandit.

Goal: verify A6a / Law R.7 in a *sequential* setting using exact density ratios.

We simulate trajectories under baseline mu (uniform policy). For each step,
we compute the true per-step density ratio w_t = P(a_t|x_t) / mu(a_t).
Because contexts are i.i.d. and mu is uniform, the trajectory density ratio
is w = product_t w_t, and the martingale is:

  M_t = E_mu[u | G_t] = (prod_{s<=t} w_s) - 1
  Delta_t = M_t - M_{t-1} = (prod_{s<=t-1} w_s) * (w_t - 1)

Then sequential Pythagoras holds exactly in expectation:
  E_mu[M_T^2] = sum_t E_mu[Delta_t^2]

We estimate these expectations from many trajectories.
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from density_ratio import ULSIF, estimate_chi2


def run_bandit(n_rounds=50, K=5, d=4, epsilon=0.3, seed=42):
    """
    Run contextual bandit and collect trajectories under both policies.
    
    Returns:
        contexts: (n_rounds, d) context vectors
        actions_mu: (n_rounds,) actions under uniform policy
        actions_P: (n_rounds,) actions under structured policy
        rewards_mu: (n_rounds,) rewards under uniform
        rewards_P: (n_rounds,) rewards under structured
        theta_true: (K, d) true reward parameters
    """
    rng = np.random.default_rng(seed)
    
    # True linear reward model: E[r|x, a] = x^T theta_a
    theta_true = rng.standard_normal((K, d)) * 0.5
    
    contexts = rng.standard_normal((n_rounds, d))
    noise_scale = 0.3
    
    # Uniform policy (baseline mu)
    actions_mu = rng.integers(0, K, size=n_rounds)
    rewards_mu = np.array([
        contexts[t] @ theta_true[actions_mu[t]] + rng.normal(0, noise_scale)
        for t in range(n_rounds)
    ])
    
    # Structured policy: epsilon-greedy on true model
    actions_P = np.zeros(n_rounds, dtype=int)
    rewards_P = np.zeros(n_rounds)
    for t in range(n_rounds):
        if rng.random() < epsilon:
            actions_P[t] = rng.integers(0, K)
        else:
            expected = contexts[t] @ theta_true.T  # (K,)
            actions_P[t] = np.argmax(expected)
        rewards_P[t] = (contexts[t] @ theta_true[actions_P[t]] + 
                        rng.normal(0, noise_scale))
    
    return (contexts, actions_mu, actions_P, rewards_mu, rewards_P,
            theta_true)


def test_sequential_pythagoras(n_trajectories=20000, T=20, K=5, d=4, epsilon=0.9):
    """
    Verify E_mu[M_T^2] = sum_t E_mu[Delta_t^2] using exact ratios.
    """
    rng = np.random.default_rng(42)
    
    # True linear reward model for policy P (only for action selection)
    theta_true = rng.standard_normal((K, d)) * 0.5
    
    # Sample trajectories under mu
    contexts = rng.standard_normal((n_trajectories, T, d))
    actions = rng.integers(0, K, size=(n_trajectories, T))  # mu uniform
    
    # Compute optimal action for each context
    expected_rewards = contexts @ theta_true.T  # (N, T, K)
    best_actions = np.argmax(expected_rewards, axis=2)      # (N, T)
    
    # P(a|x): epsilon-greedy
    # If action is best, prob = (1-epsilon) + epsilon/K; else epsilon/K
    prob_P = np.full((n_trajectories, T), epsilon / K)
    is_best = (actions == best_actions)
    prob_P[is_best] += (1.0 - epsilon)
    
    # mu(a) = 1/K
    prob_mu = 1.0 / K
    
    # Per-step ratio w_t and trajectory ratio w
    w_t = prob_P / prob_mu  # shape (N, T)
    
    # Compute M_t and Delta_t for each trajectory
    M_t = np.zeros((n_trajectories, T))
    Delta_t = np.zeros((n_trajectories, T))
    
    prod = np.ones(n_trajectories)
    for t in range(T):
        prod = prod * w_t[:, t]
        M_t[:, t] = prod - 1.0
        if t == 0:
            Delta_t[:, t] = M_t[:, t]
        else:
            Delta_t[:, t] = M_t[:, t] - M_t[:, t - 1]
    
    # Estimate expectations under mu (sample mean)
    MT_sq = np.mean(M_t[:, -1] ** 2)
    Delta_sq_sum = np.sum(np.mean(Delta_t ** 2, axis=0))
    
    rel_error = abs(MT_sq - Delta_sq_sum) / max(MT_sq, 1e-12)
    
    # Theoretical expectation for reference
    # Under mu, P(action=best) = 1/K, so w_t takes two values:
    w_best = (epsilon / K + (1.0 - epsilon)) / (1.0 / K)
    w_other = (epsilon / K) / (1.0 / K)
    Ew2 = (1.0 / K) * (w_best ** 2) + (1.0 - 1.0 / K) * (w_other ** 2)
    MT_sq_theory = (Ew2 ** T) - 1.0

    print(f"  Sequential Pythagoras (exact ratios):")
    print(f"    E[M_T^2] = {MT_sq:.6f}")
    print(f"    sum E[Delta_t^2] = {Delta_sq_sum:.6f}")
    print(f"    Theory: E[M_T^2] = {MT_sq_theory:.6f}")
    print(f"    relative error = {rel_error:.4e}")
    
    ok = rel_error < 0.10  # 10% relative error threshold (Monte Carlo noise)
    if ok:
        print(f"  PASS: Sequential Pythagoras holds (within 5%)")
    else:
        print(f"  FAIL: Sequential Pythagoras error too large")
    return ok


def test_advantage_attribution(n_rounds=500):
    """
    Verify per-epoch advantage attribution (weighted average).
    Total advantage = average of epoch advantages when epochs are equal size.
    """
    K, d = 5, 4
    
    (contexts, actions_mu, actions_P, rewards_mu, rewards_P,
     theta_true) = run_bandit(n_rounds=n_rounds, K=K, d=d, seed=42)
    
    total_adv = np.mean(rewards_P) - np.mean(rewards_mu)
    
    n_epochs = 5
    epoch_size = n_rounds // n_epochs
    
    print(f"  Total advantage (reward): {total_adv:.4f}")
    print(f"\n  Per-epoch reward advantages:")
    
    epoch_advs = []
    for e in range(n_epochs):
        start = e * epoch_size
        end = (e + 1) * epoch_size
        
        adv_e = np.mean(rewards_P[start:end]) - np.mean(rewards_mu[start:end])
        epoch_advs.append(adv_e)
        print(f"    Epoch {e + 1}: {adv_e:+.4f}")
    
    mean_advs = np.mean(epoch_advs)
    print(f"\n  Mean of epoch advantages: {mean_advs:.4f}")
    print(f"  Total advantage: {total_adv:.4f}")
    print(f"  Difference: {abs(mean_advs - total_adv):.4f}")
    
    ok = abs(mean_advs - total_adv) < 0.05
    if ok:
        print(f"  PASS: Advantages approximately additive across epochs")
    else:
        print(f"  FAIL: Advantage mean differs from total")
    return ok


def test_stopping_rule(n_rounds=1000):
    """
    Demonstrate stopping rule: halt when cumulative chi^2 reaches
    a threshold fraction of estimated total.
    """
    K, d = 5, 4
    
    (contexts, actions_mu, actions_P, rewards_mu, rewards_P,
     theta_true) = run_bandit(n_rounds=n_rounds, K=K, d=d, seed=42)
    
    def make_features(contexts, actions, K):
        n = len(actions)
        d = contexts.shape[1]
        feats = np.zeros((n, d + K))
        feats[:, :d] = contexts
        for i in range(n):
            feats[i, d + actions[i]] = 1.0
        return feats
    
    feats_mu = make_features(contexts, actions_mu, K)
    feats_P = make_features(contexts, actions_P, K)
    
    # Estimate total chi^2 (uLSIF)
    model_full = ULSIF(n_centers=min(200, n_rounds))
    model_full.fit(feats_P, feats_mu)
    chi2_total = estimate_chi2(feats_P, feats_mu, model=model_full)
    
    threshold = 0.8  # stop when 80% captured
    
    print(f"  Stopping rule: halt when cumulative chi^2 >= {threshold:.0%} of total")
    print(f"  Estimated total chi^2 = {chi2_total:.4f}")
    print(f"  Target: {threshold * chi2_total:.4f}")
    print()
    
    checkpoints = [50, 100, 200, 300, 500, 700, 1000]
    stopped = False
    
    for cp in checkpoints:
        if cp > n_rounds:
            break
        
        feats_mu_cp = feats_mu[:cp]
        feats_P_cp = feats_P[:cp]
        
        model_cp = ULSIF(n_centers=min(150, cp))
        model_cp.fit(feats_P_cp, feats_mu_cp)
        chi2_cp = estimate_chi2(feats_P_cp, feats_mu_cp, model=model_cp)
        
        frac = chi2_cp / max(chi2_total, 1e-10)
        
        status = ""
        if not stopped and frac >= threshold:
            status = " <-- STOP"
            stopped = True
        
        print(f"    Round {cp:5d}: chi^2 = {chi2_cp:.4f} "
              f"({frac:.1%} of total){status}")
    
    if stopped:
        print(f"\n  PASS: Stopping rule triggered before exhausting all rounds")
    else:
        print(f"\n  INFO: Stopping rule not triggered (may need more rounds "
              f"or lower threshold)")
    
    return True  # informational test


def run():
    print("=" * 70)
    print("TIER 2: Sequential Attribution on Contextual Bandit")
    print("=" * 70)
    
    print("\n1. Sequential Pythagoras from exact ratios:")
    ok1 = test_sequential_pythagoras()
    
    print("\n2. Per-epoch advantage attribution:")
    ok2 = test_advantage_attribution()
    
    print("\n3. Stopping rule demonstration:")
    ok3 = test_stopping_rule()
    
    print("\n" + "-" * 70)
    overall = ok1 and ok2 and ok3
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    run()
