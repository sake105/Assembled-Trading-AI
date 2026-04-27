"""Synthetic Data Generator (Plan 10.10).

Generates synthetic price data based on historical crisis templates for stress testing.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Crisis templates: (name, duration_days, mean_daily_return, vol_daily)
CRISIS_TEMPLATES: dict[str, tuple[int, float, float]] = {
    "2008_gfc": (252, -0.0020, 0.035),
    "2020_covid": (30, -0.0080, 0.050),
    "2000_dotcom": (504, -0.0008, 0.020),
    "1987_crash": (5, -0.0400, 0.080),
    "2022_rate_shock": (180, -0.0010, 0.018),
}


def generate_crisis_returns(
    template: str = "2008_gfc",
    n_assets: int = 10,
    correlation: float = 0.7,
    seed: int = 42,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Generate synthetic crisis return series.

    Args:
        template: Crisis template name.
        n_assets: Number of assets to simulate.
        correlation: Pairwise correlation during crisis.
        seed: Random seed.
        scale: Magnitude multiplier (2.0 = twice as severe).

    Returns:
        DataFrame of daily returns (rows=days, columns=asset names).
    """
    np.random.seed(seed)

    if template not in CRISIS_TEMPLATES:
        raise ValueError(f"Unknown template: {template}. Available: {list(CRISIS_TEMPLATES.keys())}")

    n_days, mean_ret, vol = CRISIS_TEMPLATES[template]
    mean_ret *= scale
    vol *= scale

    # Correlated multivariate normal
    corr_matrix = np.full((n_assets, n_assets), correlation)
    np.fill_diagonal(corr_matrix, 1.0)
    cov = np.outer(np.full(n_assets, vol), np.full(n_assets, vol)) * corr_matrix

    try:
        L = np.linalg.cholesky(cov + np.eye(n_assets) * 1e-10)
    except np.linalg.LinAlgError:
        L = np.eye(n_assets) * vol

    z = np.random.standard_normal((n_days, n_assets))
    returns = mean_ret + z @ L.T

    columns = [f"ASSET_{i}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=columns)


def generate_normal_returns(
    n_days: int = 252,
    n_assets: int = 10,
    mean_annual: float = 0.08,
    vol_annual: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic normal-market return series.

    Args:
        n_days: Number of trading days.
        n_assets: Number of assets.
        mean_annual: Annualized mean return.
        vol_annual: Annualized volatility.
        seed: Random seed.

    Returns:
        DataFrame of daily returns.
    """
    np.random.seed(seed)
    daily_mean = mean_annual / 252
    daily_vol = vol_annual / np.sqrt(252)

    returns = np.random.normal(daily_mean, daily_vol, (n_days, n_assets))
    columns = [f"ASSET_{i}" for i in range(n_assets)]
    return pd.DataFrame(returns, columns=columns)


__all__ = ["CRISIS_TEMPLATES", "generate_crisis_returns", "generate_normal_returns"]
