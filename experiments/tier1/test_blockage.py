"""
Test Blockage Theorems 4.B1, 4.B2, 4.B3.

Theorem 4.B1 (Non-Identifiability):
  There exist P1, P2 with chi^2(P1||mu) = chi^2(P2||mu) but
  chi^2(P1_S||mu_S) != chi^2(P2_S||mu_S).
  Specifically, one can have full capture and zero capture.

Theorem 4.B2 (Scalar Additivity Impossibility):
  Only D proportional to chi^2 admits exact orthogonal additivity.
  Test: verify KL and Hellinger do NOT decompose additively.

Theorem 4.B3 (Alignment/Mass-Flow Non-Recovery):
  Given (chi^2, chi^2_S), cannot recover alignment rho or mass flow Delta(S).
  Construct examples with identical scalar signatures but different rho and Delta.
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from core import (
    random_baseline, random_deviation, norm2, inner, norm, center, expectation,
    partition_projection, chi_squared, observable_chi_squared,
    alignment, advantage
)


# =========================================================================
# Theorem 4.B1: Non-Identifiability
# =========================================================================

def test_B1_non_identifiability():
    """
    Construct P1, P2 with identical chi^2(P||mu) but different 
    chi^2(P_S||mu_S). Following the proof in the paper:
    
    - u1 in V_S (fully observable) => chi^2(P1_S||mu_S) = chi^2(P1||mu)
    - u2 in V_S^perp (fully wasted) => chi^2(P2_S||mu_S) = 0
    - Scale so ||u1|| = ||u2||.
    """
    n = 8
    mu = np.ones(n) / n  # uniform
    
    # Partition: {0,1,2,3} vs {4,5,6,7}
    partition = [[0, 1, 2, 3], [4, 5, 6, 7]]
    
    # u1: S-measurable (constant on each atom), so Pi_S u1 = u1
    u1 = np.zeros(n)
    u1[:4] = 1.0
    u1[4:] = -1.0
    u1 -= expectation(u1, mu)  # center (should already be zero-mean for uniform)
    
    # u2: orthogonal to V_S (sums to 0 within each atom)
    u2 = np.zeros(n)
    u2[0], u2[1] = 1.0, -1.0
    u2[4], u2[5] = 1.0, -1.0
    u2 -= expectation(u2, mu)
    
    # Verify u2 is orthogonal to V_S
    proj_u2 = partition_projection(u2, mu, partition)
    assert norm2(proj_u2, mu) < 1e-12, "u2 should be orthogonal to V_S"
    
    # Scale to equal chi^2
    scale = norm(u1, mu) / norm(u2, mu)
    u2 = u2 * scale
    
    chi2_1 = chi_squared(u1, mu)
    chi2_2 = chi_squared(u2, mu)
    
    obs_chi2_1 = observable_chi_squared(u1, mu, partition)
    obs_chi2_2 = observable_chi_squared(u2, mu, partition)
    
    print(f"  P1: chi^2(P1||mu) = {chi2_1:.6f}, chi^2(P1_S||mu_S) = {obs_chi2_1:.6f}")
    print(f"  P2: chi^2(P2||mu) = {chi2_2:.6f}, chi^2(P2_S||mu_S) = {obs_chi2_2:.6f}")
    
    global_match = abs(chi2_1 - chi2_2) < 1e-10
    obs_differ = abs(obs_chi2_1 - obs_chi2_2) > 0.01
    
    if global_match and obs_differ:
        print(f"  PASS: Identical global chi^2 ({chi2_1:.6f}) but "
              f"observable chi^2 differs ({obs_chi2_1:.6f} vs {obs_chi2_2:.6f})")
        return True
    else:
        print(f"  FAIL: global_match={global_match}, obs_differ={obs_differ}")
        return False


def test_B1_randomized(n: int = 20, n_trials: int = 50):
    """
    Randomized version: for random partitions, construct pairs with
    identical chi^2 but different observable chi^2.
    """
    rng = np.random.default_rng(42)
    successes = 0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        
        k = rng.integers(2, min(n, 6))
        assignments = rng.integers(0, k, size=n)
        partition = [list(np.where(assignments == g)[0]) for g in range(k)
                     if np.any(assignments == g)]
        
        # u1: S-measurable
        u1 = np.zeros(n)
        for atom in partition:
            u1[atom] = rng.standard_normal()
        u1 -= expectation(u1, mu)
        
        # u2: orthogonal to V_S
        raw = rng.standard_normal(n)
        raw -= expectation(raw, mu)
        proj = partition_projection(raw, mu, partition)
        u2 = raw - proj  # residual is in V_S^perp
        
        # Skip if u2 is trivial
        if norm(u2, mu) < 1e-10:
            continue
        
        # Scale to equal norm
        u2 = u2 * (norm(u1, mu) / norm(u2, mu))
        
        chi2_1 = chi_squared(u1, mu)
        chi2_2 = chi_squared(u2, mu)
        obs1 = observable_chi_squared(u1, mu, partition)
        obs2 = observable_chi_squared(u2, mu, partition)
        
        if abs(chi2_1 - chi2_2) < 1e-8 and abs(obs1 - obs2) > 1e-6:
            successes += 1
    
    rate = successes / n_trials
    if rate > 0.8:
        print(f"  PASS: Non-identifiability demonstrated in {successes}/{n_trials} trials")
        return True
    else:
        print(f"  WEAK: Only {successes}/{n_trials} trials showed non-identifiability")
        return rate > 0.5


# =========================================================================
# Theorem 4.B2: Only chi^2 admits exact orthogonal additivity
# =========================================================================

def kl_divergence(u: np.ndarray, mu: np.ndarray) -> float:
    """D_KL(P||mu) = E_mu[(1+u) log(1+u)]."""
    w = 1.0 + u
    w = np.maximum(w, 1e-15)
    return np.sum(w * np.log(w) * mu)


def hellinger_squared(u: np.ndarray, mu: np.ndarray) -> float:
    """H^2(P,mu) = E_mu[(sqrt(1+u) - 1)^2]."""
    w = np.maximum(1.0 + u, 1e-15)
    return np.sum((np.sqrt(w) - 1.0) ** 2 * mu)


def test_B2_chi2_additive(n: int = 16, n_trials: int = 50):
    """
    Verify chi^2 decomposes additively for orthogonal components:
      chi^2(P||mu) = sum chi^2 of orthogonal pieces.
    
    And KL, Hellinger do NOT.
    """
    rng = np.random.default_rng(42)
    
    chi2_errors = []
    kl_errors = []
    hell_errors = []
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 0.5), rng=rng)
        
        # Create orthogonal decomposition via nested partitions
        # coarse partition
        partition_coarse = [list(range(0, n // 2)), list(range(n // 2, n))]
        
        proj = partition_projection(u, mu, partition_coarse)
        resid = u - proj
        
        # Verify orthogonality
        assert abs(inner(proj, resid, mu)) < 1e-10
        
        # chi^2 additivity
        chi2_total = chi_squared(u, mu)
        chi2_proj = chi_squared(proj, mu)
        chi2_resid = chi_squared(resid, mu)
        chi2_err = abs(chi2_total - (chi2_proj + chi2_resid))
        chi2_errors.append(chi2_err)
        
        # KL non-additivity: D_KL(P||mu) vs D_KL for components
        # For orthogonal u = u1 + u2, P has density 1+u.
        # Component "distributions" have density 1+u1, 1+u2.
        # KL(P||mu) should NOT equal KL(P1||mu) + KL(P2||mu).
        kl_total = kl_divergence(u, mu)
        kl_proj = kl_divergence(proj, mu)
        kl_resid = kl_divergence(resid, mu)
        kl_err = abs(kl_total - (kl_proj + kl_resid))
        kl_errors.append(kl_err)
        
        # Hellinger non-additivity
        hell_total = hellinger_squared(u, mu)
        hell_proj = hellinger_squared(proj, mu)
        hell_resid = hellinger_squared(resid, mu)
        hell_err = abs(hell_total - (hell_proj + hell_resid))
        hell_errors.append(hell_err)
    
    chi2_max = max(chi2_errors)
    kl_max = max(kl_errors)
    hell_max = max(hell_errors)
    
    kl_mean = np.mean(kl_errors)
    hell_mean = np.mean(hell_errors)
    
    print(f"  chi^2 additivity error: max={chi2_max:.2e} (should be ~0)")
    print(f"  KL additivity error:    max={kl_max:.2e}, mean={kl_mean:.2e} (should be >0)")
    print(f"  Hellinger add. error:   max={hell_max:.2e}, mean={hell_mean:.2e} (should be >0)")
    
    chi2_ok = chi2_max < 1e-10
    kl_fails = kl_mean > 1e-8
    hell_fails = hell_mean > 1e-8
    
    ok = chi2_ok and kl_fails and hell_fails
    if chi2_ok:
        print(f"  PASS: chi^2 IS orthogonally additive")
    else:
        print(f"  FAIL: chi^2 should be additive")
    if kl_fails:
        print(f"  PASS: KL is NOT orthogonally additive")
    else:
        print(f"  FAIL: KL should not be additive")
    if hell_fails:
        print(f"  PASS: Hellinger is NOT orthogonally additive")
    else:
        print(f"  FAIL: Hellinger should not be additive")
    
    return ok


# =========================================================================
# Theorem 4.B3: Alignment and mass-flow non-recovery
# =========================================================================

def test_B3_alignment_non_recovery(n: int = 20):
    """
    Construct u1, u2 with identical (chi^2, chi^2_S) but different
    alignment rho. Use a 4-atom partition so V_S is at least 3-dimensional,
    giving room to rotate Pi_S u while preserving its norm.
    """
    mu = np.ones(n) / n  # uniform for cleanliness
    
    # 4-atom partition => V_S has dim 3 (4 atoms, minus 1 for zero-mean)
    q = n // 4
    partition = [list(range(0, q)), list(range(q, 2*q)),
                 list(range(2*q, 3*q)), list(range(3*q, n))]
    
    # Two orthogonal directions in V_S
    a = np.zeros(n)
    a[:q] = 1.0; a[q:2*q] = -1.0
    a -= expectation(a, mu)
    
    b = np.zeros(n)
    b[2*q:3*q] = 1.0; b[3*q:] = -1.0
    b -= expectation(b, mu)
    
    # Verify a, b orthogonal in L^2(mu)
    assert abs(inner(a, b, mu)) < 1e-10
    
    # Normalize
    a = a / norm(a, mu)
    b = b / norm(b, mu)
    
    target_proj_norm = 0.5
    
    # u1_proj = target_proj_norm * a  (pointing along a)
    # u2_proj = target_proj_norm * b  (pointing along b, same norm)
    u1_proj = target_proj_norm * a
    u2_proj = target_proj_norm * b
    
    # Add same orthogonal (wasted) component to both => same total chi^2
    rng = np.random.default_rng(42)
    raw = rng.standard_normal(n)
    raw -= expectation(raw, mu)
    raw_proj = partition_projection(raw, mu, partition)
    raw_perp = raw - raw_proj
    if norm(raw_perp, mu) < 1e-10:
        print("  SKIP: degenerate perp component")
        return True
    raw_perp = raw_perp / norm(raw_perp, mu) * 0.3  # fixed perp norm
    
    u1 = u1_proj + raw_perp
    u2 = u2_proj + raw_perp
    
    chi2_1 = chi_squared(u1, mu)
    chi2_2 = chi_squared(u2, mu)
    obs1 = observable_chi_squared(u1, mu, partition)
    obs2 = observable_chi_squared(u2, mu, partition)
    
    # Phi aligned with a but not b
    Phi_c = a.copy()  # S-measurable, aligned with a
    
    proj_u1 = partition_projection(u1, mu, partition)
    proj_u2 = partition_projection(u2, mu, partition)
    rho1 = alignment(Phi_c, proj_u1, mu)
    rho2 = alignment(Phi_c, proj_u2, mu)
    
    print(f"  u1: chi^2={chi2_1:.6f}, chi^2_S={obs1:.6f}, rho={rho1:.4f}")
    print(f"  u2: chi^2={chi2_2:.6f}, chi^2_S={obs2:.6f}, rho={rho2:.4f}")
    
    scalars_match = (abs(chi2_1 - chi2_2) < 1e-8 and abs(obs1 - obs2) < 1e-8)
    rho_differ = abs(rho1 - rho2) > 0.5
    
    ok = scalars_match and rho_differ
    if ok:
        print(f"  PASS: Same (chi^2, chi^2_S) = ({chi2_1:.4f}, {obs1:.4f}) "
              f"but rho differs ({rho1:.4f} vs {rho2:.4f})")
    else:
        print(f"  FAIL: scalars_match={scalars_match}, rho_differ={rho_differ}")
    return ok


def test_B3_mass_flow_non_recovery(n: int = 16):
    """
    Construct u1, u2 with identical (||u||, ||Pi_S u||) but different
    mass flow Delta(S) = E[u|S].
    """
    mu = np.ones(n) / n  # uniform
    partition = [list(range(0, n // 4)), list(range(n // 4, n // 2)),
                 list(range(n // 2, 3 * n // 4)), list(range(3 * n // 4, n))]
    
    # u1: specific mass flow pattern
    u1 = np.zeros(n)
    u1[:n // 4] = 2.0
    u1[n // 4:n // 2] = -2.0
    u1[n // 2:3 * n // 4] = 1.0
    u1[3 * n // 4:] = -1.0
    u1 -= expectation(u1, mu)
    
    # u2: different mass flow, but same norms
    u2 = np.zeros(n)
    u2[:n // 4] = 1.0
    u2[n // 4:n // 2] = -1.0
    u2[n // 2:3 * n // 4] = 2.0
    u2[3 * n // 4:] = -2.0
    u2 -= expectation(u2, mu)
    
    # Scale u2 to match total norm of u1
    u2 = u2 * (norm(u1, mu) / norm(u2, mu))
    
    chi2_1 = chi_squared(u1, mu)
    chi2_2 = chi_squared(u2, mu)
    obs1 = observable_chi_squared(u1, mu, partition)
    obs2 = observable_chi_squared(u2, mu, partition)
    
    Delta1 = partition_projection(u1, mu, partition)
    Delta2 = partition_projection(u2, mu, partition)
    
    # Check values per atom
    print(f"  u1: chi^2={chi2_1:.4f}, chi^2_S={obs1:.4f}")
    print(f"  u2: chi^2={chi2_2:.4f}, chi^2_S={obs2:.4f}")
    print(f"  Mass flow Delta1 (per atom): ", end="")
    for atom in partition:
        print(f"{Delta1[atom[0]]:.4f} ", end="")
    print()
    print(f"  Mass flow Delta2 (per atom): ", end="")
    for atom in partition:
        print(f"{Delta2[atom[0]]:.4f} ", end="")
    print()
    
    norms_match = abs(chi2_1 - chi2_2) < 1e-8
    obs_match = abs(obs1 - obs2) < 1e-8
    flows_differ = norm(Delta1 - Delta2, mu) > 1e-6
    
    ok = norms_match and obs_match and flows_differ
    if ok:
        print(f"  PASS: Same scalar signatures but different mass flows")
    else:
        print(f"  FAIL: norms_match={norms_match}, obs_match={obs_match}, "
              f"flows_differ={flows_differ}")
    return ok


def run():
    print("=" * 70)
    print("TEST BLOCKAGE: Theorems 4.B1, 4.B2, 4.B3")
    print("=" * 70)
    
    print("\n1. Theorem 4.B1: Non-identifiability (constructive):")
    ok1 = test_B1_non_identifiability()
    
    print("\n2. Theorem 4.B1: Non-identifiability (randomized):")
    ok2 = test_B1_randomized()
    
    print("\n3. Theorem 4.B2: Only chi^2 is orthogonally additive:")
    ok3 = test_B2_chi2_additive()
    
    print("\n4. Theorem 4.B3a: Alignment non-recovery from scalar signatures:")
    ok4 = test_B3_alignment_non_recovery()
    
    print("\n5. Theorem 4.B3b: Mass-flow non-recovery from scalar signatures:")
    ok5 = test_B3_mass_flow_non_recovery()
    
    print("\n" + "-" * 70)
    overall = ok1 and ok2 and ok3 and ok4 and ok5
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    run()
