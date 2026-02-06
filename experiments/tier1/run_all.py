#!/usr/bin/env python3
"""
Tier 1: Mathematical Verification of Geometric Optimization Framework.

Runs all tests verifying the paper's core claims on finite discrete spaces
where ground truth is known exactly.

Tests:
  1. A6a: Sequential Pythagoras (exact in L^2, fails in L^p for p!=2)
  2. R.3: Pythagorean decomposition (exact usable/wasted split)
  3. B1-B3: Blockage theorems (scalar non-identifiability)
  4. S.2: Selection radius law (sqrt(rho) scaling)
  5. R.6: Alignment limit (three-factor factorization)
  6. R.7+R.5: Spectral conservation and budget
  7. Thm 5.2: f-divergence local collapse
  8. S.3+S.4: Selection bottleneck and additivity
"""

import sys
import time

sys.path.insert(0, '.')

from test_a6a_sequential_pythagoras import run as run_a6a
from test_pythagorean import run as run_pythagorean
from test_blockage import run as run_blockage
from test_selection_radius import run as run_selection_radius
from test_alignment import run as run_alignment
from test_spectral_conservation import run as run_spectral
from test_fdivergence_collapse import run as run_fdivergence
from test_selection_bottleneck_additivity import run as run_selection_laws


def main():
    print("=" * 70)
    print(" TIER 1: Mathematical Verification of Geometric Optimization")
    print(" Paper: 'Geometric Optimization: Axiomatizing Representation")
    print("         and Selection in L^2(mu)'")
    print("=" * 70)
    print()
    
    tests = [
        ("A6a: Sequential Pythagoras", run_a6a),
        ("R.3: Pythagorean Decomposition", run_pythagorean),
        ("B1-B3: Blockage Theorems", run_blockage),
        ("S.2: Selection Radius Law", run_selection_radius),
        ("R.6: Alignment Limit", run_alignment),
        ("R.7+R.5: Spectral Conservation", run_spectral),
        ("Thm 5.2: f-Divergence Collapse", run_fdivergence),
        ("S.3+S.4: Bottleneck + Additivity", run_selection_laws),
    ]
    
    results = {}
    total_start = time.time()
    
    for name, test_fn in tests:
        start = time.time()
        try:
            ok = test_fn()
        except Exception as e:
            print(f"\nERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            ok = False
        elapsed = time.time() - start
        results[name] = (ok, elapsed)
        print(f"\n{'=' * 70}\n")
    
    total_elapsed = time.time() - total_start
    
    # Summary
    print("=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print()
    
    for name, (ok, elapsed) in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name} ({elapsed:.1f}s)")
    
    n_pass = sum(1 for ok, _ in results.values() if ok)
    n_total = len(results)
    
    print()
    print(f"  {n_pass}/{n_total} test suites passed ({total_elapsed:.1f}s total)")
    print()
    
    if n_pass == n_total:
        print("  ALL TESTS PASS: Framework's mathematical claims verified on")
        print("  finite discrete spaces with exact ground truth.")
    else:
        failed = [name for name, (ok, _) in results.items() if not ok]
        print(f"  FAILURES: {', '.join(failed)}")
    
    return n_pass == n_total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
