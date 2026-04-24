"""Stressed VaR and RMT Covariance Cleaning (M25 Tasks 25.6 + 25.7).

Implements:
1. Random Matrix Theory (RMT) covariance cleaning via Marchenko-Pastur
2. Stressed covariance from crisis periods
3. Stressed VaR computation
4. Combined total_risk = 0.5 * normal_VaR + 0.5 * stressed_VaR

Reference:
    Marchenko & Pastur (1967), Laloux et al. (1999) for RMT
    Basel III for Stressed VaR methodology
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RMTResult:
    """Result of RMT covariance cleaning."""
    cleaned_covariance: np.ndarray
    n_signal_eigenvalues: int
    n_noise_eigenvalues: int
    noise_threshold: float       # Marchenko-Pastur upper bound
    explained_signal_ratio: float  # Fraction of variance from signal


@dataclass
class StressedVaRResult:
    """Result of stressed VaR computation."""
    normal_var: float
    stressed_var: float
    combined_var: float          # 0.5 * normal + 0.5 * stressed
    stress_period: str
    stress_multiplier: float     # stressed / normal ratio


def marchenko_pastur_bounds(
    n_obs: int,
    n_assets: int,
    sigma_sq: float = 1.0,
) -> tuple[float, float]:
    """Compute Marchenko-Pastur distribution bounds.

    Args:
        n_obs: Number of observations (T).
        n_assets: Number of assets (N).
        sigma_sq: Noise variance (default 1.0 for correlation matrix).

    Returns:
        (lambda_min, lambda_max) bounds.
    """
    q = n_obs / n_assets
    lambda_max = sigma_sq * (1 + 1 / np.sqrt(q)) ** 2
    lambda_min = sigma_sq * max(0, (1 - 1 / np.sqrt(q)) ** 2)
    return lambda_min, lambda_max


def clean_covariance_rmt(
    returns: pd.DataFrame | np.ndarray,
    method: str = "clip",
) -> RMTResult:
    """Clean covariance matrix using Random Matrix Theory (Task 25.6).

    Separates eigenvalues into signal (above Marchenko-Pastur bound)
    and noise (below), then reconstructs with cleaned noise eigenvalues.

    Args:
        returns: (T, N) return matrix.
        method: "clip" (clip noise to average) or "zero" (set noise to zero).

    Returns:
        RMTResult with cleaned covariance.
    """
    ret = np.asarray(returns, dtype=float)
    T, N = ret.shape

    if T < N:
        logger.warning("[RMT] T=%d < N=%d — insufficient data for reliable cleaning", T, N)

    # Correlation matrix
    corr = np.corrcoef(ret.T)
    std = ret.std(axis=0)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # MP bounds
    _, lambda_max = marchenko_pastur_bounds(T, N)

    # Separate signal and noise
    n_signal = int(np.sum(eigenvalues > lambda_max))
    n_noise = N - n_signal

    # Clean noise eigenvalues
    cleaned_eigenvalues = eigenvalues.copy()
    noise_mask = eigenvalues <= lambda_max

    if method == "clip":
        # Replace noise eigenvalues with their average
        noise_avg = eigenvalues[noise_mask].mean() if noise_mask.any() else 1.0
        cleaned_eigenvalues[noise_mask] = noise_avg
    elif method == "zero":
        # Set noise eigenvalues to small positive value (for invertibility)
        cleaned_eigenvalues[noise_mask] = 0.01

    # Reconstruct correlation matrix
    cleaned_corr = eigenvectors @ np.diag(cleaned_eigenvalues) @ eigenvectors.T

    # Ensure valid correlation matrix
    np.fill_diagonal(cleaned_corr, 1.0)
    cleaned_corr = np.clip(cleaned_corr, -1, 1)

    # Convert back to covariance
    D = np.diag(std)
    cleaned_cov = D @ cleaned_corr @ D

    signal_var = eigenvalues[~noise_mask].sum() / eigenvalues.sum() if eigenvalues.sum() > 0 else 0

    logger.info("[RMT] Cleaned: %d signal + %d noise eigenvalues (threshold=%.4f)",
                n_signal, n_noise, lambda_max)

    return RMTResult(
        cleaned_covariance=cleaned_cov,
        n_signal_eigenvalues=n_signal,
        n_noise_eigenvalues=n_noise,
        noise_threshold=round(lambda_max, 4),
        explained_signal_ratio=round(signal_var, 4),
    )


def compute_stressed_covariance(
    returns: pd.DataFrame,
    stress_start: str = "2008-09-01",
    stress_end: str = "2009-03-31",
) -> np.ndarray:
    """Compute covariance from a stress period (Task 25.7).

    Args:
        returns: Full return DataFrame with DatetimeIndex.
        stress_start: Start of stress period.
        stress_end: End of stress period.

    Returns:
        Stressed covariance matrix.
    """
    try:
        stress_data = returns.loc[stress_start:stress_end]
    except KeyError:
        logger.warning("[StressedVaR] Stress period not in data — using worst 60-day window")
        # Fallback: find worst 60-day rolling return period
        port_ret = returns.mean(axis=1)
        rolling_ret = port_ret.rolling(60).sum()
        worst_end = rolling_ret.idxmin()
        if worst_end is not None:
            worst_start = worst_end - pd.Timedelta(days=90)
            stress_data = returns.loc[worst_start:worst_end]
        else:
            stress_data = returns.tail(60)

    if len(stress_data) < 20:
        logger.warning("[StressedVaR] Stress period too short (%d rows), using full data", len(stress_data))
        return returns.cov().values

    return stress_data.cov().values


def compute_parametric_var(
    weights: np.ndarray,
    covariance: np.ndarray,
    confidence: float = 0.99,
    horizon_days: int = 1,
    portfolio_value: float = 1_000_000.0,
) -> float:
    """Compute parametric VaR.

    Args:
        weights: Portfolio weights.
        covariance: Covariance matrix (daily).
        confidence: Confidence level.
        horizon_days: VaR horizon in days.
        portfolio_value: Portfolio value for dollar VaR.

    Returns:
        VaR in dollars.
    """
    from scipy.stats import norm
    z = norm.ppf(confidence)

    port_var = float(weights @ covariance @ weights)
    port_vol = np.sqrt(port_var * horizon_days)
    var = z * port_vol * portfolio_value

    return round(var, 2)


def compute_stressed_var(
    weights: np.ndarray,
    returns: pd.DataFrame,
    confidence: float = 0.99,
    portfolio_value: float = 1_000_000.0,
    stress_start: str = "2008-09-01",
    stress_end: str = "2009-03-31",
    use_rmt: bool = True,
) -> StressedVaRResult:
    """Compute normal VaR, stressed VaR, and combined (Task 25.7).

    total_risk = 0.5 * normal_VaR + 0.5 * stressed_VaR (Basel III)

    Args:
        weights: Portfolio weights.
        returns: Full return DataFrame.
        confidence: VaR confidence level.
        portfolio_value: Portfolio value.
        stress_start: Start of stress period.
        stress_end: End of stress period.
        use_rmt: Apply RMT cleaning to normal covariance.

    Returns:
        StressedVaRResult.
    """
    # Normal covariance
    if use_rmt:
        rmt_result = clean_covariance_rmt(returns)
        normal_cov = rmt_result.cleaned_covariance
    else:
        normal_cov = returns.cov().values

    # Stressed covariance
    stressed_cov = compute_stressed_covariance(returns, stress_start, stress_end)

    # Ensure dimensions match
    n = len(weights)
    if normal_cov.shape[0] != n:
        normal_cov = normal_cov[:n, :n]
    if stressed_cov.shape[0] != n:
        stressed_cov = stressed_cov[:n, :n]

    # Compute VaR
    try:
        normal_var = compute_parametric_var(weights, normal_cov, confidence, 1, portfolio_value)
        stressed_var = compute_parametric_var(weights, stressed_cov, confidence, 1, portfolio_value)
    except ImportError:
        # No scipy — use the local z-table (resolves the requested confidence
        # level exactly for common alphas and interpolates otherwise). The
        # prior binary fallback (2.326 for ≥0.99, else 1.645) silently
        # upgraded a 0.975 request to 95% precision.
        from src.assembled_core.risk.var_methods import _z_from_alpha

        z = _z_from_alpha(confidence)
        normal_vol = np.sqrt(float(weights @ normal_cov @ weights))
        stressed_vol = np.sqrt(float(weights @ stressed_cov @ weights))
        normal_var = round(z * normal_vol * portfolio_value, 2)
        stressed_var = round(z * stressed_vol * portfolio_value, 2)

    combined = 0.5 * normal_var + 0.5 * stressed_var
    multiplier = stressed_var / max(normal_var, 1.0)

    logger.info("[StressedVaR] Normal=$%,.0f, Stressed=$%,.0f, Combined=$%,.0f (%.1fx)",
                normal_var, stressed_var, combined, multiplier)

    return StressedVaRResult(
        normal_var=normal_var,
        stressed_var=stressed_var,
        combined_var=round(combined, 2),
        stress_period=f"{stress_start} to {stress_end}",
        stress_multiplier=round(multiplier, 2),
    )


__all__ = [
    "RMTResult",
    "StressedVaRResult",
    "marchenko_pastur_bounds",
    "clean_covariance_rmt",
    "compute_stressed_covariance",
    "compute_parametric_var",
    "compute_stressed_var",
]
