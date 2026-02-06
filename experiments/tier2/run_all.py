#!/usr/bin/env python3
"""
Tier 2: Empirical Verification from Samples.

Tests the framework on continuous distributions with finite samples,
where density ratios must be estimated and decompositions are approximate.

Experiments:
  1. Pythagorean decomposition from samples (Gaussian shifts)
  2. Sequential attribution on contextual bandit
  3. Distribution shift diagnosis via geometric diagnostics
"""

import sys
import time

sys.path.insert(0, '.')

from test_pythagorean_samples import run as run_pythagorean
from test_bandit_sequential import run as run_bandit
from test_shift_diagnosis import run as run_shift
from test_real_dataset_digits import run as run_digits


def main():
    print("=" * 70)
    print(" TIER 2: Empirical Verification from Samples")
    print(" Density ratio estimation -> geometric diagnostics")
    print("=" * 70)
    print()
    
    tests = [
        ("Pythagorean from Samples", run_pythagorean),
        ("Sequential Bandit Attribution", run_bandit),
        ("Distribution Shift Diagnosis", run_shift),
        ("Real Dataset (Digits)", run_digits),
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
    
    return n_pass == n_total


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
