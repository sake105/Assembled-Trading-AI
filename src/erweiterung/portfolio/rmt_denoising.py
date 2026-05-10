"""Random-Matrix-Theory Cov-Denoising (Laloux et al. 1999, Bun-Bouchaud 2017).

Theorie
-------
Bei N Assets und T historischen Returns mit q = N/T > 0:
- Sample-Eigenvalues sind nach Marchenko-Pastur verteilt:
  ρ(λ) = √((λ_+ - λ)(λ - λ_-)) / (2π σ² q λ)
  mit  λ_± = σ² (1 ± √q)².
- Alle Sample-Eigenvalues innerhalb [λ_-, λ_+] sind **Rauschen**.
- Eigenvalues > λ_+ sind echte Signale.

Methoden
--------
1. **Eigenvalue-Clipping**: replace alle Eigenvalues < λ_+ durch deren Mittel.
2. **Eigenvalue-Substitution**: replace durch optimal-MP-shrinkage Form
   (Ledoit-Péché 2011, Bun-Bouchaud-Potters 2016).

Reference
---------
- Laloux, L., Cizeau, P., Bouchaud, J.-P. & Potters, M. (1999). Noise Dressing
  of Financial Correlation Matrices. *Phys. Rev. Lett.* 83.
- Bun, J., Bouchaud, J.-P. & Potters, M. (2017). Cleaning large correlation
  matrices: tools from RMT. *Physics Reports* 666.
- Lopez de Prado, M. (2019). A Robust Estimator of the Efficient Frontier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class MPFit:
    lambda_minus: float
    lambda_plus: float
    sigma_sq: float
    q: float  # N/T ratio


def fit_marchenko_pastur(eigvals: np.ndarray, T: int, N: int) -> MPFit:
    """Fit Marchenko-Pastur edges to the bulk of sample eigenvalues.

    Args:
        eigvals: array of sorted sample-correlation eigenvalues.
        T: number of time observations.
        N: number of assets.

    Returns:
        MPFit with λ_-, λ_+, σ², q.
    """
    q = N / T
    # σ² from mean of "bulk" — exclude top-K largest as signals
    # heuristic: σ² ≈ mean of eigvals truncated above estimated edge
    sigma_sq_init = float(np.mean(eigvals))
    lambda_plus_init = sigma_sq_init * (1 + np.sqrt(q)) ** 2

    # iterate: refine σ² as mean of eigvals below current edge
    for _ in range(10):
        bulk = eigvals[eigvals <= lambda_plus_init]
        if len(bulk) < 2:
            break
        new_sigma_sq = float(np.mean(bulk) / (1 - q))  # adjusted estimate
        lambda_plus_new = new_sigma_sq * (1 + np.sqrt(q)) ** 2
        if abs(lambda_plus_new - lambda_plus_init) < 1e-6:
            sigma_sq_init = new_sigma_sq
            lambda_plus_init = lambda_plus_new
            break
        sigma_sq_init = new_sigma_sq
        lambda_plus_init = lambda_plus_new

    lambda_minus = sigma_sq_init * (1 - np.sqrt(q)) ** 2
    return MPFit(
        lambda_minus=lambda_minus,
        lambda_plus=lambda_plus_init,
        sigma_sq=sigma_sq_init,
        q=q,
    )


def denoise_correlation_eigenvalue_clipping(corr: pd.DataFrame, T: int) -> pd.DataFrame:
    """Eigenvalue-Clipping nach RMT-Bulk.

    Args:
        corr: Korrelationsmatrix.
        T: Time-Observations für q = N/T.

    Returns:
        Bereinigte Korrelationsmatrix.
    """
    N = corr.shape[0]
    eigvals, eigvecs = np.linalg.eigh(corr.values)
    mp = fit_marchenko_pastur(eigvals, T, N)

    mask = eigvals <= mp.lambda_plus
    if mask.sum() == 0 or (~mask).sum() == 0:
        return corr.copy()
    # Replace small eigvals with mean of noise-bulk so trace preserved
    noise_mean = eigvals[mask].mean()
    eigvals_clean = np.where(mask, noise_mean, eigvals)
    # Renormalize so trace == N (corr-matrix trace)
    eigvals_clean *= N / eigvals_clean.sum()

    corr_clean = eigvecs @ np.diag(eigvals_clean) @ eigvecs.T
    corr_clean = 0.5 * (corr_clean + corr_clean.T)
    # Ensure diagonal == 1
    d = 1.0 / np.sqrt(np.diag(corr_clean))
    corr_clean = (d[:, None] * corr_clean) * d[None, :]
    return pd.DataFrame(corr_clean, index=corr.index, columns=corr.columns)


def denoise_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Convenience: take returns, compute clipped cov-matrix."""
    X = returns.dropna(how="any").values
    T, N = X.shape
    Xc = X - X.mean(axis=0)
    sample = (Xc.T @ Xc) / T
    diag = np.diag(sample)
    std = np.sqrt(np.maximum(diag, 1e-12))
    corr = sample / np.outer(std, std)
    corr_df = pd.DataFrame(corr, index=returns.columns, columns=returns.columns)
    corr_clean = denoise_correlation_eigenvalue_clipping(corr_df, T=T)
    cov_clean = corr_clean.values * np.outer(std, std)
    return pd.DataFrame(cov_clean, index=returns.columns, columns=returns.columns)


def signal_to_noise_ratio(eigvals: np.ndarray, T: int, N: int) -> float:
    """Anteil der "Signal"-Eigenvalues an Total-Varianz."""
    mp = fit_marchenko_pastur(eigvals, T, N)
    signal_eigvals = eigvals[eigvals > mp.lambda_plus]
    total = eigvals.sum()
    if total == 0:
        return float("nan")
    return float(signal_eigvals.sum() / total)


__all__ = [
    "MPFit",
    "fit_marchenko_pastur",
    "denoise_correlation_eigenvalue_clipping",
    "denoise_covariance",
    "signal_to_noise_ratio",
]
