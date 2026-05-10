"""Robust Regression — outlier-resistant alternatives to OLS.

Methoden
--------
- **Huber-M-Estimator** (Huber 1964): quadratisch für kleine, linear für große
  Residuen. Maximum-Likelihood unter t-Verteilungsannahme.
- **RANSAC** (Fischler/Bolles 1981): random sampling consensus — robust gegen
  >50 % Outlier.
- **MM-Estimator** (Yohai 1987): hochgradig robust + asymptotisch effizient.

Reference
---------
- Huber, P. (1964). Robust estimation of a location parameter. *Ann. Math. Stat.* 35.
- Maronna/Martin/Yohai (2006). *Robust Statistics — Theory and Methods*. Wiley.
"""

from __future__ import annotations

import numpy as np


def huber_regression(
    X: np.ndarray,
    y: np.ndarray,
    delta: float = 1.345,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """Huber-M-Estimator via Iteratively-Reweighted-Least-Squares (IRLS).

    Args:
        X: design matrix (n, p), include intercept manually if needed.
        y: response (n,).
        delta: Huber threshold (1.345 for 95% asymptotic efficiency at Gaussian).
        max_iter, tol: IRLS-Konvergenz.

    Returns:
        beta-coefficients.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if n < p + 5:
        raise ValueError("not enough samples")
    # Start with OLS
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    for _ in range(max_iter):
        resid = y - X @ beta
        # Robust scale estimate via MAD
        sigma = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
        if sigma == 0:
            break
        z = resid / sigma
        # Huber weights
        w = np.where(np.abs(z) <= delta, 1.0, delta / np.maximum(np.abs(z), 1e-12))
        # Weighted least squares
        W = np.diag(w)
        try:
            new_beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        except np.linalg.LinAlgError:
            break
        if np.linalg.norm(new_beta - beta) < tol:
            beta = new_beta
            break
        beta = new_beta
    return beta


def ransac_regression(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 1.0,
    n_iter: int = 200,
    min_inliers_frac: float = 0.5,
    seed: int = 42,
) -> dict:
    """RANSAC (Fischler/Bolles 1981) — robust gegen extreme Outlier.

    Args:
        X, y: design matrix + response.
        threshold: max abs-residual for an inlier (in y-units).
        n_iter: random sub-samples.
        min_inliers_frac: stop early if found this fraction of inliers.
        seed: RNG.

    Returns:
        dict mit ``beta``, ``inlier_mask``, ``n_inliers``, ``n_iter_used``.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    if n < p + 5:
        raise ValueError("not enough samples")
    rng = np.random.default_rng(seed)
    target_inliers = int(min_inliers_frac * n)
    best_beta = None
    best_inliers: np.ndarray = np.array([])

    sample_size = max(p + 1, 10)
    for it in range(n_iter):
        idx = rng.choice(n, sample_size, replace=False)
        X_s, y_s = X[idx], y[idx]
        try:
            beta_s, *_ = np.linalg.lstsq(X_s, y_s, rcond=None)
        except np.linalg.LinAlgError:
            continue
        resid = y - X @ beta_s
        inliers = np.abs(resid) < threshold
        if inliers.sum() > len(best_inliers):
            best_inliers = inliers
            best_beta = beta_s
            if inliers.sum() >= target_inliers:
                # Refit on all inliers
                best_beta, *_ = np.linalg.lstsq(X[inliers], y[inliers], rcond=None)
                return {
                    "beta": best_beta,
                    "inlier_mask": inliers,
                    "n_inliers": int(inliers.sum()),
                    "n_iter_used": it + 1,
                    "early_stop": True,
                }
    if best_beta is None:
        # Fallback OLS
        best_beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return {
            "beta": best_beta,
            "inlier_mask": np.ones(n, dtype=bool),
            "n_inliers": n,
            "n_iter_used": n_iter,
            "early_stop": False,
        }
    # Refit on inliers
    best_beta, *_ = np.linalg.lstsq(X[best_inliers], y[best_inliers], rcond=None)
    return {
        "beta": best_beta,
        "inlier_mask": best_inliers,
        "n_inliers": int(best_inliers.sum()),
        "n_iter_used": n_iter,
        "early_stop": False,
    }


def median_absolute_deviation(x: np.ndarray, scale: float = 1.4826) -> float:
    """MAD-Skala: 1.4826 × median(|x − median(x)|). Konsistente Schätzung von σ unter Gauss."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan")
    return float(scale * np.median(np.abs(x - np.median(x))))


__all__ = [
    "huber_regression",
    "ransac_regression",
    "median_absolute_deviation",
]
