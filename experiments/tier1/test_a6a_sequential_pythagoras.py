"""
Test A6a: Sequential Pythagoras.

Claim (Axiom A6a / Law R.7 Part B):
  For filtration (G_t) with martingale M_t = E_mu[u|G_t] and
  increments Delta_t = M_t - M_{t-1}:
  
    ||M_T||^2 = sum_t ||Delta_t||^2    (EXACT in L^2)
  
  This should FAIL in L^p for p != 2 (Lemma 3.2).

Method:
  - Construct discrete Omega with |Omega| = 16, random mu.
  - Construct filtration: trivial -> binary -> 4-way -> 8-way -> full.
  - Compute martingale increments Delta_t.
  - In L^2: verify ||M_T||^2 == sum ||Delta_t||^2 (exact, up to float eps).
  - In L^p for p in {1, 1.5, 3, 4, inf}: verify INEQUALITY (not equality).
  
Also tests Lemma 3.2 directly on binary partition.
"""

import numpy as np
import sys
sys.path.insert(0, '.')
from core import (
    random_baseline, random_deviation, norm2, norm, lp_norm,
    conditional_expectation, inner, expectation
)


def build_nested_filtration(n: int):
    """Build a nested filtration by successive binary splits.
    Returns list of partitions, from trivial to full."""
    filtration = []
    # Start with trivial partition
    current = [list(range(n))]
    filtration.append(current)
    
    # Successively split each atom in half
    while max(len(atom) for atom in current) > 1:
        new_partition = []
        for atom in current:
            if len(atom) > 1:
                mid = len(atom) // 2
                new_partition.append(atom[:mid])
                new_partition.append(atom[mid:])
            else:
                new_partition.append(atom)
        current = new_partition
        filtration.append(current)
    
    return filtration


def test_sequential_pythagoras_l2(n: int = 16, n_trials: int = 50):
    """Test that ||M_T||^2 = sum ||Delta_t||^2 in L^2(mu)."""
    rng = np.random.default_rng(42)
    max_error = 0.0
    
    for trial in range(n_trials):
        mu = random_baseline(n, rng)
        u = random_deviation(mu, scale=rng.uniform(0.1, 2.0), rng=rng)
        
        filtration = build_nested_filtration(n)
        
        # Compute martingale and increments
        M_prev = np.zeros(n)
        delta_norms_sq = []
        
        for partition in filtration[1:]:  # skip trivial
            M_t = conditional_expectation(u, mu, partition)
            delta = M_t - M_prev
            delta_norms_sq.append(norm2(delta, mu))
            M_prev = M_t.copy()
        
        M_T = M_prev  # final martingale
        lhs = norm2(M_T, mu)  # ||M_T||^2
        rhs = sum(delta_norms_sq)  # sum ||Delta_t||^2
        
        error = abs(lhs - rhs)
        max_error = max(max_error, error)
        
        if error > 1e-10:
            print(f"  FAIL trial {trial}: ||M_T||^2={lhs:.12f}, "
                  f"sum||Delta_t||^2={rhs:.12f}, error={error:.2e}")
            return False
    
    print(f"  PASS: L^2 Sequential Pythagoras exact across {n_trials} trials "
          f"(max error: {max_error:.2e})")
    return True


def test_sequential_pythagoras_lp_fails(n: int = 16, n_trials: int = 50):
    """Test that ||M_T||^p != sum ||Delta_t||^p for p != 2."""
    rng = np.random.default_rng(123)
    p_values = [1.0, 1.5, 3.0, 4.0]
    
    results = {}
    for p in p_values:
        violations = 0
        max_gap = 0.0
        
        for trial in range(n_trials):
            mu = random_baseline(n, rng)
            u = random_deviation(mu, scale=rng.uniform(0.3, 1.5), rng=rng)
            
            filtration = build_nested_filtration(n)
            
            M_prev = np.zeros(n)
            delta_norms_p = []
            
            for partition in filtration[1:]:
                M_t = conditional_expectation(u, mu, partition)
                delta = M_t - M_prev
                lp_val = lp_norm(delta, mu, p)
                delta_norms_p.append(lp_val ** p)
                M_prev = M_t.copy()
            
            M_T = M_prev
            lhs = lp_norm(M_T, mu, p) ** p
            rhs = sum(delta_norms_p)
            
            gap = abs(lhs - rhs) / max(abs(lhs), 1e-15)
            if gap > 1e-8:
                violations += 1
                max_gap = max(max_gap, gap)
        
        results[p] = (violations, max_gap)
        status = "PASS (fails as expected)" if violations > 0 else "UNEXPECTED: equality held"
        print(f"  p={p:.1f}: {violations}/{n_trials} trials show inequality, "
              f"max relative gap={max_gap:.4f} -- {status}")
    
    # p=2 should have zero violations (tested above); all other p should have violations
    all_fail = all(results[p][0] > 0 for p in p_values)
    return all_fail


def test_lemma_3_2_binary():
    """
    Direct test of Lemma 3.2: binary partition {A,B} with mu(A)=mu(B)=1/2.
    x = (1,1), P1 projects onto first coordinate, P2 onto second.
    
    A6a requires: ||x||^2 = ||P1 x||^2 + ||P2 x||^2
    In l^p: ||(1,1)||_p^2 = 2^{2/p}, but ||P1 x||_p^2 + ||P2 x||_p^2 = 1+1 = 2.
    Equality iff p=2.
    """
    mu = np.array([0.5, 0.5])
    x = np.array([1.0, 1.0])
    
    # Partitions: {{0}} and {{1}} are the two coordinate projections
    P1_x = np.array([1.0, 0.0])  # project onto atom {0}
    P2_x = np.array([0.0, 1.0])  # project onto atom {1}
    
    print("  Lemma 3.2 (binary partition, x=(1,1), mu=(0.5,0.5)):")
    
    all_pass = True
    for p in [1.0, 1.5, 2.0, 3.0, 4.0]:
        lhs = lp_norm(x, mu, p) ** 2
        rhs = lp_norm(P1_x, mu, p) ** 2 + lp_norm(P2_x, mu, p) ** 2
        gap = abs(lhs - rhs)
        exact = gap < 1e-12
        mark = "==" if exact else "!="
        expected = "PASS" if (exact == (p == 2.0)) else "FAIL"
        print(f"    p={p:.1f}: ||x||_p^2 = {lhs:.6f}, "
              f"sum||P_i x||_p^2 = {rhs:.6f}  [{mark}]  {expected}")
        if expected == "FAIL":
            all_pass = False
    
    return all_pass


def run():
    print("=" * 70)
    print("TEST A6a: Sequential Pythagoras (Axiom A6a / Lemma 3.2)")
    print("=" * 70)
    
    print("\n1. L^2 Sequential Pythagoras (should be EXACT):")
    ok1 = test_sequential_pythagoras_l2()
    
    print("\n2. L^p Sequential Pythagoras for p != 2 (should FAIL):")
    ok2 = test_sequential_pythagoras_lp_fails()
    
    print("\n3. Lemma 3.2: Binary partition Pythagoras:")
    ok3 = test_lemma_3_2_binary()
    
    print("\n" + "-" * 70)
    overall = ok1 and ok2 and ok3
    print(f"OVERALL: {'PASS' if overall else 'FAIL'}")
    return overall


if __name__ == "__main__":
    run()
