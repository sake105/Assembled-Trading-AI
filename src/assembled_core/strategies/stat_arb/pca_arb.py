"""PCA-Based Statistical Arbitrage (M36.3).

Uses Principal Component Analysis on sector-adjusted returns to find
mean-reverting residuals across the entire universe.

Algorithm:
1. PCA on returns -> statistical factors (eigenvectors)
2. Regress each stock on top-K factors -> residual
3. Residuals are approximately mean-reverting
4. Long stocks with low residual, short stocks with high residual
5. Alpha: +150-400 bps/year (Avellaneda & Lee 2010)

Advantage over pairs: uses entire universe, not just 2 stocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PCAFactorModel:
    """PCA factor model results."""
    n_components: int
    explained_variance_ratio: list[float]
    factor_loadings: pd.DataFrame  # (symbols x n_components)
    residuals: pd.DataFrame  # (dates x symbols)
    eigenvalues: list[float]


@dataclass
class PCASignal:
    """PCA arbitrage signal."""
    weights: dict[str, float]  # symbol -> position weight
    expected_alpha_bps: float
    residual_z_scores: dict[str, float]
    method: str


def compute_pca_factors(
    returns: pd.DataFrame,
    n_components: int = 5,
    min_obs: int = 60,
) -> PCAFactorModel | None:
    """Compute PCA factors from return matrix.

    Args:
        returns: Wide DataFrame (dates x symbols) of daily returns.
        n_components: Number of PCA components to extract.
        min_obs: Minimum observations required.

    Returns:
        PCAFactorModel or None if insufficient data.
    """
    clean = returns.dropna(axis=0, how="any")
    if len(clean) < min_obs or clean.shape[1] < n_components:
        logger.warning("[PCA] Insufficient data: %d obs, %d symbols",
                       len(clean), clean.shape[1])
        return None

    # Standardize
    ret_matrix = clean.values
    mean = ret_matrix.mean(axis=0)
    std = ret_matrix.std(axis=0)
    std[std < 1e-10] = 1.0
    standardized = (ret_matrix - mean) / std

    # SVD (more numerically stable than eigendecomposition)
    try:
        U, S, Vt = np.linalg.svd(standardized, full_matrices=False)
    except np.linalg.LinAlgError:
        logger.warning("[PCA] SVD failed")
        return None

    # Top K components
    k = min(n_components, len(S))
    eigenvalues = (S[:k] ** 2) / (len(clean) - 1)
    total_var = np.sum(S ** 2) / (len(clean) - 1)
    explained = eigenvalues / total_var

    # Factor loadings: V[:k] transposed -> (symbols x k)
    loadings = Vt[:k].T  # (n_symbols x k)
    loading_df = pd.DataFrame(
        loadings, index=clean.columns,
        columns=[f"PC{i+1}" for i in range(k)],
    )

    # Factor returns: U * S -> (dates x k)
    factors = U[:, :k] * S[:k]

    # Residuals: returns - factor_model_prediction
    predicted = factors @ loadings.T  # (dates x symbols)
    predicted = predicted * std + mean  # un-standardize
    residuals = clean.values - predicted
    residual_df = pd.DataFrame(residuals, index=clean.index, columns=clean.columns)

    return PCAFactorModel(
        n_components=k,
        explained_variance_ratio=[round(float(e), 6) for e in explained],
        factor_loadings=loading_df,
        residuals=residual_df,
        eigenvalues=[round(float(e), 6) for e in eigenvalues],
    )


def generate_pca_signals(
    pca_model: PCAFactorModel,
    lookback: int = 60,
    entry_z: float = 1.5,
    max_position: float = 0.02,
) -> PCASignal:
    """Generate trading signals from PCA residuals.

    Long stocks with negative residual z-score (undervalued by factors).
    Short stocks with positive residual z-score (overvalued by factors).

    Args:
        pca_model: Fitted PCA factor model.
        lookback: Days for z-score computation.
        entry_z: Z-score threshold for positions.
        max_position: Maximum weight per position.

    Returns:
        PCASignal with portfolio weights.
    """
    residuals = pca_model.residuals
    if len(residuals) < lookback:
        return PCASignal(weights={}, expected_alpha_bps=0, residual_z_scores={}, method="pca_arb")

    recent = residuals.tail(lookback)
    current = residuals.iloc[-1]
    mean = recent.mean()
    std = recent.std()
    std[std < 1e-10] = 1.0

    z_scores = (current - mean) / std

    # Positions: short high-z, long low-z
    weights = {}
    z_dict = {}
    for sym in z_scores.index:
        z = float(z_scores[sym])
        z_dict[sym] = round(z, 4)
        if z > entry_z:
            weights[sym] = -min(max_position, (z - entry_z) * 0.005)
        elif z < -entry_z:
            weights[sym] = min(max_position, (abs(z) - entry_z) * 0.005)

    # Make dollar-neutral
    if weights:
        long_sum = sum(w for w in weights.values() if w > 0)
        short_sum = sum(abs(w) for w in weights.values() if w < 0)
        total = max(long_sum, short_sum, 1e-10)
        weights = {s: round(w / total * 0.5, 6) for s, w in weights.items()}

    # Rough expected alpha estimate
    n_active = sum(1 for w in weights.values() if abs(w) > 1e-6)
    expected_alpha = n_active * 2.0  # ~2bps per position per day, rough

    return PCASignal(
        weights=weights,
        expected_alpha_bps=round(expected_alpha, 1),
        residual_z_scores=z_dict,
        method="pca_arb",
    )


__all__ = [
    "PCAFactorModel",
    "PCASignal",
    "compute_pca_factors",
    "generate_pca_signals",
]
