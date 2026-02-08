"""
Automatic Region Relevance Discovery for Image Classification.

Given a classifier and dataset, automatically discover which spatial regions
of the input are most relevant to classification — without manually specifying
candidate regions.

Four approaches compared:
  A) Patch sweep: sliding window, compute eta per patch position
  B) Per-pixel sweep: corrupt one pixel at a time, compute eta for each
  C) Learned mask: optimize a continuous mask to maximize eta
  D) Density ratio decomposition: attribute importance to input dims from
     the fitted kernel expansion

All produce a spatial relevance heatmap over the 8x8 digit grid.

Dataset: sklearn.datasets.load_digits (8x8 grayscale handwritten digits)
"""

import numpy as np
import sys
sys.path.insert(0, '.')

from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from density_ratio import ULSIF, estimate_chi2, estimate_observable_chi2


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

def setup(random_state=42):
    """Load digits, train classifier, return everything needed."""
    data = load_digits()
    X = data.data.astype(float)
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=4000, penalty='l2', solver='lbfgs')
    clf.fit(X_train_s, y_train)

    acc = np.mean(clf.predict(X_test_s) == y_test)
    print(f"Baseline accuracy: {acc:.2%}")

    return X_test, X_test_s, y_test, clf, scaler


def compute_eta(X_clean_s, X_corrupt_s, clf, n_centers=200, n_bins=14):
    """
    Compute eta = chi2_S / chi2 for a corruption.

    Returns (chi2, chi2_S, eta). Returns (0, 0, 0) if shift is negligible.
    """
    model = ULSIF(n_centers=min(n_centers, len(X_clean_s)))
    model.fit(X_corrupt_s, X_clean_s)
    chi2 = estimate_chi2(X_corrupt_s, X_clean_s, model=model)

    if chi2 < 1e-10:
        return 0.0, 0.0, 0.0

    S_P = clf.decision_function(X_corrupt_s)
    S_mu = clf.decision_function(X_clean_s)
    chi2_S = estimate_observable_chi2(X_corrupt_s, X_clean_s, S_P, S_mu,
                                      n_bins=n_bins)
    chi2_S = min(max(chi2_S, 0.0), chi2)
    eta = chi2_S / max(chi2, 1e-12)
    return chi2, chi2_S, eta


# ---------------------------------------------------------------------------
# Method A: Patch sweep
# ---------------------------------------------------------------------------

def method_a_patch_sweep(X_test, X_test_s, y_test, clf, scaler,
                         patch_size=2, strength=0.85, n_centers=150):
    """
    Slide a patch across the 8x8 grid. For each position, erase that patch
    and compute eta. Assign eta to all pixels in the patch (accumulate and
    average for overlapping patches).
    """
    H, W = 8, 8
    n_pixels = H * W
    eta_accum = np.zeros((H, W))
    count_accum = np.zeros((H, W))
    chi2_map = np.zeros((H, W))
    chi2_S_map = np.zeros((H, W))

    positions = []
    for r in range(H - patch_size + 1):
        for c in range(W - patch_size + 1):
            positions.append((r, c))

    print(f"  Patch sweep: {len(positions)} positions, patch_size={patch_size}")

    for i, (r, c) in enumerate(positions):
        # Build mask for this patch
        mask = np.zeros((H, W), dtype=bool)
        mask[r:r+patch_size, c:c+patch_size] = True
        mask_flat = mask.ravel()

        # Corrupt
        X_corrupt = X_test.copy()
        X_corrupt[:, mask_flat] *= (1.0 - strength)
        X_corrupt = np.clip(X_corrupt, 0.0, 16.0)
        X_corrupt_s = scaler.transform(X_corrupt)

        chi2, chi2_S, eta = compute_eta(X_test_s, X_corrupt_s, clf,
                                         n_centers=n_centers)

        eta_accum[r:r+patch_size, c:c+patch_size] += eta
        chi2_map[r:r+patch_size, c:c+patch_size] += chi2
        chi2_S_map[r:r+patch_size, c:c+patch_size] += chi2_S
        count_accum[r:r+patch_size, c:c+patch_size] += 1

        if (i + 1) % 10 == 0 or i == len(positions) - 1:
            print(f"    [{i+1}/{len(positions)}]")

    # Average over overlapping patches
    count_accum = np.maximum(count_accum, 1)
    eta_map = eta_accum / count_accum

    return eta_map


# ---------------------------------------------------------------------------
# Method B: Per-pixel sweep
# ---------------------------------------------------------------------------

def method_b_pixel_sweep(X_test, X_test_s, y_test, clf, scaler,
                         strength=0.85, n_centers=150):
    """
    Corrupt one pixel at a time and compute eta for each.
    """
    H, W = 8, 8
    n_pixels = H * W
    eta_map = np.zeros(n_pixels)
    chi2_map = np.zeros(n_pixels)
    chi2_S_map = np.zeros(n_pixels)

    print(f"  Per-pixel sweep: {n_pixels} pixels")

    for px in range(n_pixels):
        X_corrupt = X_test.copy()
        X_corrupt[:, px] *= (1.0 - strength)
        X_corrupt = np.clip(X_corrupt, 0.0, 16.0)
        X_corrupt_s = scaler.transform(X_corrupt)

        chi2, chi2_S, eta = compute_eta(X_test_s, X_corrupt_s, clf,
                                         n_centers=n_centers)
        eta_map[px] = eta
        chi2_map[px] = chi2
        chi2_S_map[px] = chi2_S

        if (px + 1) % 16 == 0 or px == n_pixels - 1:
            print(f"    [{px+1}/{n_pixels}]")

    return eta_map.reshape(H, W)


# ---------------------------------------------------------------------------
# Method C: Learned mask
# ---------------------------------------------------------------------------

def method_c_learned_mask(X_test, X_test_s, y_test, clf, scaler,
                          n_centers=150, n_steps=100, lr=0.05,
                          strength=0.85, l1_weight=0.01):
    """
    Optimize a continuous mask M in [0, 1]^64 that, when used to erase pixels,
    maximizes eta (chi2_S / chi2) while being sparse (L1 penalty on mask area).

    Actually, maximizing eta directly is tricky because it requires refitting
    ULSIF at each step. Instead, we use a proxy: maximize accuracy drop
    per unit of mask area, using the classifier's loss as a differentiable
    surrogate.

    Simpler approach: parameterize mask as logits, apply sigmoid, use
    finite-difference estimation of eta gradient.

    Simplest viable approach: coordinate descent. For each pixel, estimate
    the marginal gain in eta from including it in the mask. Greedily build
    the mask pixel by pixel.
    """
    H, W = 8, 8
    n_pixels = H * W

    print(f"  Learned mask (greedy coordinate selection, {n_steps} steps)")

    # Start with per-pixel eta estimates (we can reuse method B's logic)
    # to warm-start, but let's do it fresh for a clean comparison.

    # Phase 1: estimate per-pixel marginal eta
    pixel_etas = np.zeros(n_pixels)
    for px in range(n_pixels):
        X_corrupt = X_test.copy()
        X_corrupt[:, px] *= (1.0 - strength)
        X_corrupt = np.clip(X_corrupt, 0.0, 16.0)
        X_corrupt_s = scaler.transform(X_corrupt)
        _, _, eta = compute_eta(X_test_s, X_corrupt_s, clf,
                                n_centers=n_centers)
        pixel_etas[px] = eta

    # Phase 2: greedily grow mask, adding the pixel that most increases eta
    mask = np.zeros(n_pixels, dtype=bool)
    mask_etas = []  # eta at each mask size
    mask_order = []  # order pixels were added

    n_to_add = min(n_steps, n_pixels)
    remaining = set(range(n_pixels))

    # Sort by marginal eta descending for initial ordering
    ranked = sorted(remaining, key=lambda px: pixel_etas[px], reverse=True)

    for step in range(n_to_add):
        best_px = -1
        best_eta = -1.0

        # Evaluate top candidates (not all remaining — too expensive)
        # Use marginal eta ranking to check top-k candidates
        candidates = [px for px in ranked if px in remaining][:8]

        for px in candidates:
            trial_mask = mask.copy()
            trial_mask[px] = True

            X_corrupt = X_test.copy()
            X_corrupt[:, trial_mask] *= (1.0 - strength)
            X_corrupt = np.clip(X_corrupt, 0.0, 16.0)
            X_corrupt_s = scaler.transform(X_corrupt)

            _, _, eta = compute_eta(X_test_s, X_corrupt_s, clf,
                                     n_centers=n_centers)

            if eta > best_eta:
                best_eta = eta
                best_px = px

        if best_px < 0:
            break

        mask[best_px] = True
        remaining.discard(best_px)
        mask_etas.append(best_eta)
        mask_order.append(best_px)

        if (step + 1) % 10 == 0 or step == n_to_add - 1:
            print(f"    Step {step+1}: added pixel {best_px}, "
                  f"mask size={mask.sum()}, eta={best_eta:.4f}")

    # Build importance map: pixel rank (earlier = more important)
    importance = np.zeros(n_pixels)
    for rank, px in enumerate(mask_order):
        importance[px] = 1.0 - (rank / max(len(mask_order) - 1, 1))

    return importance.reshape(H, W), mask_order, mask_etas


# ---------------------------------------------------------------------------
# Method D: Density ratio decomposition
# ---------------------------------------------------------------------------

def method_d_decomposition(X_test, X_test_s, y_test, clf, scaler,
                           strength=0.85, n_centers=200):
    """
    Fit ULSIF once on a global corruption (erase all pixels uniformly by
    strength), then decompose the density ratio into per-dimension
    contributions using the kernel gradient.

    For RBF kernel k(x, c) = exp(-||x-c||^2 / (2*sigma^2)):
        dk/dx_i = -(x_i - c_i) / sigma^2 * k(x, c)

    The density ratio is w(x) = sum_l alpha_l * k(x, c_l).
    The gradient wrt input dimension i:
        dw/dx_i = sum_l alpha_l * (-(x_i - c_l_i) / sigma^2) * k(x, c_l)

    Per-dimension importance: average |dw/dx_i| over mu samples,
    indicating how sensitive the density ratio is to each pixel.
    """
    H, W = 8, 8
    n_pixels = H * W

    print(f"  Density ratio decomposition")

    # Global corruption: erase all pixels
    X_corrupt = X_test.copy()
    X_corrupt *= (1.0 - strength)
    X_corrupt = np.clip(X_corrupt, 0.0, 16.0)
    X_corrupt_s = scaler.transform(X_corrupt)

    # Fit ULSIF on the global shift
    model = ULSIF(n_centers=min(n_centers, len(X_test_s)))
    model.fit(X_corrupt_s, X_test_s)

    # Compute per-dimension gradient of w(x) at each mu sample
    # w(x) = sum_l alpha_l * k(x, c_l)
    # dw/dx_i = sum_l alpha_l * (-(x_i - c_l_i)/sigma^2) * k(x, c_l)
    sigma = model.sigma_
    centers = model.centers_  # (n_c, d)
    alpha = model.alpha_      # (n_c,)

    K = np.exp(-np.sum((X_test_s[:, None, :] - centers[None, :, :]) ** 2,
                       axis=2) / (2.0 * sigma ** 2))  # (n_mu, n_c)

    # Gradient: for each sample i, dimension j
    # dw/dx_j[i] = sum_l alpha_l * (-(X[i,j] - C[l,j]) / sigma^2) * K[i,l]
    # = -1/sigma^2 * sum_l alpha_l * (X[i,j] - C[l,j]) * K[i,l]

    # Vectorized: (n_mu, n_c, d) differences
    diff = X_test_s[:, None, :] - centers[None, :, :]  # (n_mu, n_c, d)
    # Weight by alpha and kernel: alpha_l * K[i,l]
    weights = alpha[None, :] * K  # (n_mu, n_c)
    # Gradient per dim: -1/sigma^2 * sum_l weights[i,l] * diff[i,l,j]
    grad_w = -1.0 / (sigma ** 2) * np.einsum('il,ilj->ij', weights, diff)
    # (n_mu, d)

    # Per-dimension importance: mean |gradient| over mu samples
    importance = np.mean(np.abs(grad_w), axis=0)  # (d,)

    # Also compute per-dimension importance weighted by the density ratio
    # deviation: |dw/dx_i| * |w(x) - 1| to emphasize where the ratio
    # is actually deviating
    w_mu = model.predict(X_test_s)
    u_mu = np.abs(w_mu - 1.0)
    importance_weighted = np.mean(np.abs(grad_w) * u_mu[:, None], axis=0)

    return importance.reshape(H, W), importance_weighted.reshape(H, W)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def print_heatmap(name, heatmap):
    """Print an 8x8 heatmap as ASCII art."""
    H, W = heatmap.shape
    vmin, vmax = heatmap.min(), heatmap.max()
    if vmax - vmin < 1e-12:
        normalized = np.zeros_like(heatmap)
    else:
        normalized = (heatmap - vmin) / (vmax - vmin)

    chars = ' ░▒▓█'
    print(f"\n  {name}:")
    print(f"  (range: {vmin:.4f} to {vmax:.4f})")
    print(f"  ┌{'──' * W}┐")
    for r in range(H):
        row = '  │'
        for c in range(W):
            idx = min(int(normalized[r, c] * (len(chars) - 1)), len(chars) - 1)
            row += chars[idx] * 2
        row += '│'
        print(row)
    print(f"  └{'──' * W}┘")


def print_comparison(results):
    """Print all heatmaps side by side and correlation matrix."""
    names = list(results.keys())
    maps = [results[n].ravel() for n in names]

    print("\n" + "=" * 60)
    print("CORRELATION MATRIX")
    print("=" * 60)
    print(f"  {'':20s}", end='')
    for n in names:
        print(f" {n[:10]:>10s}", end='')
    print()

    for i, n1 in enumerate(names):
        print(f"  {n1:20s}", end='')
        for j, n2 in enumerate(names):
            corr = np.corrcoef(maps[i], maps[j])[0, 1]
            print(f" {corr:10.3f}", end='')
        print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 60)
    print("AUTOMATIC REGION RELEVANCE DISCOVERY")
    print("=" * 60)

    X_test, X_test_s, y_test, clf, scaler = setup()
    results = {}

    # Method A: Patch sweep
    print("\n--- Method A: Patch Sweep ---")
    eta_a = method_a_patch_sweep(X_test, X_test_s, y_test, clf, scaler,
                                  patch_size=2)
    results['A_patch'] = eta_a
    print_heatmap("Patch sweep eta", eta_a)

    # Method B: Per-pixel sweep
    print("\n--- Method B: Per-Pixel Sweep ---")
    eta_b = method_b_pixel_sweep(X_test, X_test_s, y_test, clf, scaler)
    results['B_pixel'] = eta_b
    print_heatmap("Per-pixel eta", eta_b)

    # Method C: Learned mask (greedy)
    print("\n--- Method C: Learned Mask (Greedy) ---")
    importance_c, mask_order, mask_etas = method_c_learned_mask(
        X_test, X_test_s, y_test, clf, scaler, n_steps=32)
    results['C_greedy'] = importance_c
    print_heatmap("Greedy mask importance", importance_c)

    # Method D: Density ratio decomposition
    print("\n--- Method D: Density Ratio Decomposition ---")
    importance_d, importance_d_weighted = method_d_decomposition(
        X_test, X_test_s, y_test, clf, scaler)
    results['D_grad'] = importance_d
    results['D_grad_wt'] = importance_d_weighted
    print_heatmap("DR gradient importance", importance_d)
    print_heatmap("DR gradient importance (weighted)", importance_d_weighted)

    # Comparison
    print_comparison(results)

    # Sanity check: all methods should agree that center > border
    border_mask = np.zeros((8, 8), dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    center_mask = np.zeros((8, 8), dtype=bool)
    center_mask[2:6, 2:6] = True

    print("\n" + "=" * 60)
    print("CENTER vs BORDER RELEVANCE")
    print("=" * 60)
    print(f"  {'Method':20s} {'Center mean':>12s} {'Border mean':>12s} {'Ratio':>8s} {'Pass':>6s}")
    print(f"  {'-' * 60}")

    all_pass = True
    for name, hmap in results.items():
        cm = hmap[center_mask].mean()
        bm = hmap[border_mask].mean()
        ratio = cm / max(bm, 1e-12)
        ok = cm > bm
        all_pass = all_pass and ok
        print(f"  {name:20s} {cm:12.4f} {bm:12.4f} {ratio:8.2f}x {'PASS' if ok else 'FAIL':>6s}")

    print(f"\n  Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    return all_pass


if __name__ == '__main__':
    run()
