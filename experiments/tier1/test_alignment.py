"""
Test Law R.6: Alignment Limit and Three-Factor Factorization.

Claim (Theorem 4.6 / Lemma 4.2):
  |E_P[Phi] - E_mu[Phi]| <= ||Phi^o||_2 * sqrt(chi^2(P_S||mu_S))
  
  with equality iff Phi^o || Pi_S u.
  
  Equivalently:
    Advantage = ||Phi^o|| * ||Pi_S u|| * rho
  where rho = <Phi^o, Pi_S u> / (||Phi^o|| * ||Pi_S u||) in [-1, 1].

Tests:
  1. Three-factor identity holds exactly
  2. Cauchy-Schwarz bound holds (|rho| <= 1)
  3. Bound is tight when Phi^o proportional to Pi_S u
  4. Misalignment diagnosis: |rho| << 1 when Phi^o nearly orthogonal to Pi_S u
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from core import (
    random_baseline, random_deviation, random_performance,
    norm, norm2, inner, center, expectation,
    partition_projection, chi_squared, observable_chi_squared,
    alignment, advantage
)


def test_three_factor_identity(n: int = 20, n_trials: int = 200):
    """
    Verify Adv = ||Phi^o|| * ||Pi_S u|| * rho exactly.
    """
    rng = np.random.default_rng(42)
    max_error = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        
        k = rng.integers(2, min(n, 8))
        assignments = rng.integers(0, k, size=n)
        partition = [list(np.where(assignments == g)[0]) for g in range(k)
                     if np.any(assignments == g)]
        
        # S-measurable Phi
        Phi = np.zeros(n)
        for atom in partition:
            Phi[atom] = rng.standard_normal()
        Phi_c = center(Phi, mu)
        
        proj_u = partition_projection(u, mu, partition)
        
        # Direct advantage
        adv = inner(Phi_c, u, mu)
        
        # Three-factor decomposition
        norm_Phi = norm(Phi_c, mu)
        norm_proj_u = norm(proj_u, mu)
        
        if norm_Phi < 1e-12 or norm_proj_u < 1e-12:
            continue
        
        rho = alignment(Phi_c, proj_u, mu)
        three_factor = norm_Phi * norm_proj_u * rho
        
        error = abs(adv - three_factor)
        max_error = max(max_error, error)
        
        if error > 1e-10:
            print(f"  FAIL trial {trial}: adv={adv:.12f}, "
                  f"||Phi^o||*||Pi u||*rho={three_factor:.12f}")
            return False
    
    print(f"  PASS: Three-factor identity exact across {n_trials} trials "
          f"(max error: {max_error:.2e})")
    return True


def test_cauchy_schwarz_bound(n: int = 20, n_trials: int = 200):
    """
    Verify |Adv| <= ||Phi^o|| * ||Pi_S u|| (i.e., |rho| <= 1).
    """
    rng = np.random.default_rng(42)
    max_rho = 0.0
    violations = 0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        Phi = random_performance(n, rng)
        
        k = rng.integers(2, min(n, 8))
        assignments = rng.integers(0, k, size=n)
        partition = [list(np.where(assignments == g)[0]) for g in range(k)
                     if np.any(assignments == g)]
        
        # Make Phi S-measurable
        Phi_s = np.zeros(n)
        for atom in partition:
            Phi_s[atom] = np.mean(Phi[atom])
        Phi_c = center(Phi_s, mu)
        
        proj_u = partition_projection(u, mu, partition)
        
        adv = abs(inner(Phi_c, proj_u, mu))
        bound = norm(Phi_c, mu) * norm(proj_u, mu)
        
        if adv > bound + 1e-10:
            violations += 1
        
        if bound > 1e-12:
            rho_val = adv / bound
            max_rho = max(max_rho, rho_val)
    
    if violations == 0:
        print(f"  PASS: |Adv| <= ||Phi^o||*||Pi_S u|| in all {n_trials} trials "
              f"(max |rho| = {max_rho:.6f})")
        return True
    else:
        print(f"  FAIL: {violations} violations of Cauchy-Schwarz")
        return False


def test_bound_tightness(n: int = 20, n_trials: int = 50):
    """
    Verify bound is tight when Phi^o is proportional to Pi_S u.
    """
    rng = np.random.default_rng(42)
    max_gap = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        
        k = rng.integers(2, min(n, 6))
        assignments = rng.integers(0, k, size=n)
        partition = [list(np.where(assignments == g)[0]) for g in range(k)
                     if np.any(assignments == g)]
        
        proj_u = partition_projection(u, mu, partition)
        
        if norm(proj_u, mu) < 1e-10:
            continue
        
        # Set Phi^o = alpha * Pi_S u (perfect alignment)
        alpha = rng.uniform(0.5, 3.0)
        Phi_c = alpha * proj_u  # already zero-mean since proj_u is
        
        adv = inner(Phi_c, proj_u, mu)
        bound = norm(Phi_c, mu) * norm(proj_u, mu)
        
        # Should be tight: adv == bound
        gap = abs(adv - bound) / max(bound, 1e-15)
        max_gap = max(max_gap, gap)
        
        rho = alignment(Phi_c, proj_u, mu)
        if abs(rho - 1.0) > 1e-8:
            print(f"  FAIL trial {trial}: rho={rho:.10f} should be 1.0")
            return False
    
    print(f"  PASS: Bound tight when Phi^o || Pi_S u (max gap: {max_gap:.2e}, "
          f"rho = 1.0)")
    return True


def test_misalignment_diagnosis(n: int = 20):
    """
    Demonstrate: same ||Phi^o|| and ||Pi_S u|| but different rho
    yield very different advantages.
    
    Need >= 4-atom partition so V_S has dimension >= 3, allowing
    construction of orthogonal directions within V_S.
    """
    rng = np.random.default_rng(42)
    mu = np.ones(n) / n  # uniform for clean construction
    
    # 4-atom partition => dim(V_S) = 3
    q = n // 4
    partition = [list(range(0, q)), list(range(q, 2*q)),
                 list(range(2*q, 3*q)), list(range(3*q, n))]
    
    u = random_deviation(mu, scale=1.0, rng=rng)
    proj_u = partition_projection(u, mu, partition)
    
    if norm(proj_u, mu) < 1e-10:
        print(f"  SKIP: trivial projection")
        return True
    
    # Case 1: Phi^o aligned with Pi_S u
    Phi_aligned = proj_u / norm(proj_u, mu) * 2.0
    
    # Case 2: Phi^o orthogonal to Pi_S u (within V_S)
    # Build an S-measurable function, then Gram-Schmidt out proj_u
    Phi_raw = np.zeros(n)
    for atom in partition:
        Phi_raw[atom] = rng.standard_normal()
    Phi_raw -= expectation(Phi_raw, mu)
    # Gram-Schmidt: remove component along proj_u
    ip = inner(Phi_raw, proj_u, mu) / norm2(proj_u, mu)
    Phi_orth = Phi_raw - ip * proj_u
    
    if norm(Phi_orth, mu) < 1e-10:
        # Extremely unlikely with 4 atoms, but handle gracefully
        print(f"  SKIP: could not construct orthogonal Phi in V_S")
        return True
    
    Phi_orth = Phi_orth / norm(Phi_orth, mu) * 2.0
    
    # Case 3: Phi^o partially aligned
    raw_partial = 0.5 * Phi_aligned + 0.5 * Phi_orth
    Phi_partial = raw_partial / norm(raw_partial, mu) * 2.0
    
    cases = [
        ("Aligned", Phi_aligned),
        ("Orthogonal", Phi_orth),
        ("Partial (50/50)", Phi_partial),
    ]
    
    print(f"  ||Pi_S u|| = {norm(proj_u, mu):.4f}")
    print()
    
    for name, Phi_c in cases:
        rho_val = alignment(Phi_c, proj_u, mu)
        adv = inner(Phi_c, proj_u, mu)
        bound = norm(Phi_c, mu) * norm(proj_u, mu)
        efficiency = abs(adv) / max(bound, 1e-15)
        
        print(f"  {name:20s}: ||Phi^o||={norm(Phi_c, mu):.4f}, "
              f"rho={rho_val:+.4f}, Adv={adv:+.6f}, "
              f"bound={bound:.6f}, efficiency={efficiency:.2%}")
    
    rho_aligned = alignment(Phi_aligned, proj_u, mu)
    rho_orth = alignment(Phi_orth, proj_u, mu)
    
    ok = abs(rho_aligned - 1.0) < 1e-6 and abs(rho_orth) < 1e-6
    if ok:
        print(f"\n  PASS: Alignment diagnosis works (rho=1 for aligned, "
              f"rho~0 for orthogonal)")
    else:
        print(f"\n  FAIL: rho_aligned={rho_aligned:.8f}, rho_orth={rho_orth:.8f}")
    return ok


def run():
    print("=" * 70)
    print("TEST R.6: Alignment Limit and Three-Factor Factorization")
    print("=" * 70)
    
    print("\n1. Three-factor identity Adv = ||Phi^o|| * ||Pi_S u|| * rho:")
    ok1 = test_three_factor_identity()
    
    print("\n2. Cauchy-Schwarz bound |rho| <= 1:")
    ok2 = test_cauchy_schwarz_bound()
    
    print("\n3. Bound tightness when Phi^o || Pi_S u:")
    ok3 = test_bound_tightness()
    
    print("\n4. Misalignment diagnosis (same norms, different rho):")
    ok4 = test_misalignment_diagnosis()
    
    print("\n" + "-" * 70)
    overall = ok1 and ok2 and ok3 and ok4
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    run()
