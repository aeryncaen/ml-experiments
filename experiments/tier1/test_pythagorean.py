"""
Test Law R.3: Pythagorean Decomposition.

Claim (Theorem 4.4 / Law R.3):
  chi^2(P||mu) = chi^2(P_S||mu_S) + ||u^perp||^2
  
  where u^perp = (I - Pi_S) u is the orthogonal residual.
  This is EXACT equality, not an inequality.

Also tests:
  - Law R.4 (Data-Processing): chi^2(P_S||mu_S) <= chi^2(P||mu)
  - Corollary 4.2.1: observable advantage = <Phi^o, Pi_S u>
  - Orthogonality: <Pi_S u, u^perp> = 0
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from core import (
    random_baseline, random_deviation, random_performance,
    norm2, inner, center, partition_projection, chi_squared,
    observable_chi_squared, advantage, expectation
)


def test_pythagorean_exact(n: int = 20, n_trials: int = 100):
    """Verify chi^2(P||mu) = chi^2(P_S||mu_S) + ||u^perp||^2 exactly."""
    rng = np.random.default_rng(42)
    max_error = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        
        # Random partition into k groups
        k = rng.integers(2, min(n, 8))
        assignments = rng.integers(0, k, size=n)
        partition = []
        for g in range(k):
            atoms = list(np.where(assignments == g)[0])
            if atoms:
                partition.append(atoms)
        
        # Compute decomposition
        total_chi2 = chi_squared(u, mu)
        proj_u = partition_projection(u, mu, partition)
        obs_chi2 = norm2(proj_u, mu)
        u_perp = u - proj_u
        wasted = norm2(u_perp, mu)
        
        # Pythagorean identity
        error = abs(total_chi2 - (obs_chi2 + wasted))
        max_error = max(max_error, error)
        
        if error > 1e-10:
            print(f"  FAIL trial {trial}: chi2={total_chi2:.12f}, "
                  f"obs+waste={obs_chi2 + wasted:.12f}, error={error:.2e}")
            return False
    
    print(f"  PASS: Pythagorean exact across {n_trials} trials "
          f"(max error: {max_error:.2e})")
    return True


def test_orthogonality(n: int = 20, n_trials: int = 100):
    """Verify <Pi_S u, u^perp> = 0."""
    rng = np.random.default_rng(77)
    max_error = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        
        k = rng.integers(2, min(n, 8))
        assignments = rng.integers(0, k, size=n)
        partition = [list(np.where(assignments == g)[0]) for g in range(k)
                     if np.any(assignments == g)]
        
        proj_u = partition_projection(u, mu, partition)
        u_perp = u - proj_u
        
        ip = abs(inner(proj_u, u_perp, mu))
        max_error = max(max_error, ip)
        
        if ip > 1e-10:
            print(f"  FAIL trial {trial}: <Pi_S u, u^perp> = {ip:.2e}")
            return False
    
    print(f"  PASS: Orthogonality <Pi_S u, u^perp> = 0 across {n_trials} trials "
          f"(max: {max_error:.2e})")
    return True


def test_data_processing(n: int = 20, n_trials: int = 100):
    """Verify chi^2(P_S||mu_S) <= chi^2(P||mu) with equality iff u in V_S."""
    rng = np.random.default_rng(99)
    all_pass = True
    
    violations = 0
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        
        k = rng.integers(2, min(n, 8))
        assignments = rng.integers(0, k, size=n)
        partition = [list(np.where(assignments == g)[0]) for g in range(k)
                     if np.any(assignments == g)]
        
        total = chi_squared(u, mu)
        obs = observable_chi_squared(u, mu, partition)
        
        if obs > total + 1e-10:
            violations += 1
    
    if violations > 0:
        print(f"  FAIL: Data processing violated in {violations}/{n_trials} trials")
        all_pass = False
    else:
        print(f"  PASS: Data processing chi^2(P_S||mu_S) <= chi^2(P||mu) "
              f"in all {n_trials} trials")
    
    # Test equality when u is already S-measurable
    mu = random_baseline(n, rng)
    partition = [[0, 1, 2], [3, 4, 5], list(range(6, n))]
    # Construct u that IS S-measurable (constant on each atom)
    u_meas = np.zeros(n)
    for atom in partition:
        val = rng.standard_normal()
        u_meas[atom] = val
    # Center
    u_meas -= expectation(u_meas, mu)
    
    total = chi_squared(u_meas, mu)
    obs = observable_chi_squared(u_meas, mu, partition)
    gap = abs(total - obs)
    if gap < 1e-10:
        print(f"  PASS: Equality when u is S-measurable (gap={gap:.2e})")
    else:
        print(f"  FAIL: Should be equal for S-measurable u (gap={gap:.2e})")
        all_pass = False
    
    return all_pass


def test_observable_advantage_isolation(n: int = 20, n_trials: int = 100):
    """
    Corollary 4.2.1: For S-measurable Phi,
      E_P[Phi] - E_mu[Phi] = <Phi^o, Pi_S u>
    and the orthogonal part u^perp contributes nothing.
    """
    rng = np.random.default_rng(55)
    max_error = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        
        k = rng.integers(2, min(n, 6))
        assignments = rng.integers(0, k, size=n)
        partition = [list(np.where(assignments == g)[0]) for g in range(k)
                     if np.any(assignments == g)]
        
        # S-measurable Phi: constant on each atom
        Phi = np.zeros(n)
        for atom in partition:
            Phi[atom] = rng.standard_normal()
        
        Phi_c = center(Phi, mu)
        proj_u = partition_projection(u, mu, partition)
        u_perp = u - proj_u
        
        # Full advantage
        adv_full = inner(Phi_c, u, mu)
        # Observable advantage
        adv_obs = inner(Phi_c, proj_u, mu)
        # Wasted contribution (should be 0)
        adv_wasted = inner(Phi_c, u_perp, mu)
        
        error1 = abs(adv_full - adv_obs)
        error2 = abs(adv_wasted)
        max_error = max(max_error, error1, error2)
        
        if error1 > 1e-10 or error2 > 1e-10:
            print(f"  FAIL trial {trial}: adv_full={adv_full:.12f}, "
                  f"adv_obs={adv_obs:.12f}, adv_wasted={adv_wasted:.2e}")
            return False
    
    print(f"  PASS: Observable advantage isolation across {n_trials} trials "
          f"(max error: {max_error:.2e})")
    return True


def run():
    print("=" * 70)
    print("TEST R.3: Pythagorean Decomposition + Data Processing")
    print("=" * 70)
    
    print("\n1. Pythagorean identity (exact equality):")
    ok1 = test_pythagorean_exact()
    
    print("\n2. Orthogonality <Pi_S u, u^perp> = 0:")
    ok2 = test_orthogonality()
    
    print("\n3. Data processing inequality chi^2(P_S||mu_S) <= chi^2(P||mu):")
    ok3 = test_data_processing()
    
    print("\n4. Observable advantage isolation (Corollary 4.2.1):")
    ok4 = test_observable_advantage_isolation()
    
    print("\n" + "-" * 70)
    overall = ok1 and ok2 and ok3 and ok4
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    run()
