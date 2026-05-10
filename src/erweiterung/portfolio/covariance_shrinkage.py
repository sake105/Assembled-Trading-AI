"""Covariance-Shrinkage-Estimators für stabilere Portfolio-Konstruktion.

Theorie
-------
Sample-Covariance ist unbiased aber sehr ungenau bei N >~ T/3.
Shrinkage zu strukturiertem Target reduziert MSE bei minimaler Bias.

Methoden
--------
1. **Ledoit-Wolf (2004)**: Schrumpft zu skalierter Identität:
   Σ̂ = (1-α) S + α (avg-diag) I, mit α* analytisch optimal.
2. **Ledoit-Wolf-Quadratic-Form (2017)**: nichtlineare Eigenvalue-Schrumpfung.
3. **Constant-Correlation-Target (Ledoit-Wolf 2003)**:
   Schrumpft zu Matrix mit konstanten Korrelationen.
4. **RIE — Rotation-Invariant-Estimator (Bun-Bouchaud-Potters 2017)**:
   Optimal in spektrum-shrinkage Sinne unter RMT.

Reference
---------
- Ledoit, O. & Wolf, M. (2004). Honey, I shrunk the Sample Covariance Matrix.
  *J. Portfolio Management* 30.
- Ledoit, O. & Wolf, M. (2003). Improved Estimation of the Covariance Matrix
  of Stock Returns. *J. Empirical Finance* 10.
- Bun, J., Bouchaud, J.-P. & Potters, M. (2017). Cleaning large correlation
  matrices: tools from RMT. *Physics Reports*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ledoit_wolf_shrinkage(returns: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf (2004) shrinkage to scaled identity.

    Args:
        returns: DataFrame (T, N).

    Returns:
        (Σ_shrunk, alpha).  α ∈ [0, 1] = shrinkage intensity.
    """
    X = returns.dropna(how="any").values
    T, N = X.shape
    if T < 5 or N < 2:
        raise ValueError("need T>=5, N>=2")

    Xc = X - X.mean(axis=0)
    sample = (Xc.T @ Xc) / T
    diag_mean = float(np.trace(sample) / N)
    target = diag_mean * np.eye(N)

    # Optimal shrinkage intensity (Ledoit-Wolf formula)
    # π estimator
    Y = Xc**2
    pi_mat = (Y.T @ Y) / T - sample**2
    pi = float(pi_mat.sum())

    # γ estimator (distance from target)
    gamma = float(np.sum((sample - target) ** 2))

    if gamma <= 0:
        alpha = 0.0
    else:
        alpha = float(np.clip(pi / (T * gamma), 0.0, 1.0))

    shrunk = (1 - alpha) * sample + alpha * target
    return (
        pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns),
        float(alpha),
    )


def constant_correlation_target_shrinkage(
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf (2003) shrinkage to constant-correlation target.

    Target: Matrix mit gemeinsamem ρ̄ = mean off-diagonal correlation.
    """
    X = returns.dropna(how="any").values
    T, N = X.shape
    if T < 5 or N < 2:
        raise ValueError("need T>=5, N>=2")

    Xc = X - X.mean(axis=0)
    sample = (Xc.T @ Xc) / T
    diag = np.diag(sample)
    std = np.sqrt(diag)
    corr = sample / np.outer(std, std)

    # Mean off-diagonal correlation
    off_mask = ~np.eye(N, dtype=bool)
    rho_bar = float(corr[off_mask].mean())

    target_corr = np.full((N, N), rho_bar)
    np.fill_diagonal(target_corr, 1.0)
    target = target_corr * np.outer(std, std)

    # Heuristic alpha = min(1, K/T) where K depends on dimensionality
    alpha = float(min(1.0, N / max(T, 1)))
    shrunk = (1 - alpha) * sample + alpha * target
    return (
        pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns),
        alpha,
    )


def rie_clip_eigenvalues(
    returns: pd.DataFrame, T_over_N: float | None = None
) -> pd.DataFrame:
    """Marchenko-Pastur-basierte Eigenvalue-Clipping (light-RIE).

    Idee: in einer Sample-Cov-Matrix sind viele kleine Eigenvalues nur Rauschen.
    Replace alle Eigenvalues < λ_+ (oberer MP-Edge) mit ihrem Mittelwert.

    Args:
        returns: DataFrame (T, N).
        T_over_N: ratio, default = T/N from data.

    Returns:
        Bereinigte Cov-Matrix.
    """
    X = returns.dropna(how="any").values
    T, N = X.shape
    if T_over_N is None:
        T_over_N = T / N
    Xc = X - X.mean(axis=0)
    sample = (Xc.T @ Xc) / T
    diag = np.diag(sample)
    std = np.sqrt(np.maximum(diag, 1e-12))
    corr = sample / np.outer(std, std)

    eigvals, eigvecs = np.linalg.eigh(corr)
    # Marchenko-Pastur upper edge for corr matrix:
    q = 1.0 / T_over_N  # q = N/T
    lambda_plus = (1 + np.sqrt(q)) ** 2

    # Replace small eigenvalues (<= lambda_plus) by their mean
    mask = eigvals <= lambda_plus
    if mask.sum() > 0 and (~mask).sum() > 0:
        mean_noise = eigvals[mask].mean()
        eigvals_clean = np.where(mask, mean_noise, eigvals)
        # Re-normalize so trace is preserved
        eigvals_clean *= eigvals.sum() / eigvals_clean.sum()
    else:
        eigvals_clean = eigvals

    corr_clean = eigvecs @ np.diag(eigvals_clean) @ eigvecs.T
    # Symmetrize
    corr_clean = 0.5 * (corr_clean + corr_clean.T)
    cov_clean = corr_clean * np.outer(std, std)
    return pd.DataFrame(cov_clean, index=returns.columns, columns=returns.columns)


__all__ = [
    "ledoit_wolf_shrinkage",
    "constant_correlation_target_shrinkage",
    "rie_clip_eigenvalues",
]
