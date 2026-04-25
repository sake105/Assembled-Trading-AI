"""Cointegration Engine for Pairs Trading (M36.1).

Implements:
- Engle-Granger 2-Step: OLS Regression -> ADF Test on residuals
- Rolling cointegration with configurable windows
- Pair screening within same sector
- Half-life estimation via Ornstein-Uhlenbeck fit

Output: PairCandidate with hedge_ratio, half_life, p_value.

References:
    Engle & Granger (1987) "Co-Integration and Error Correction"
    Avellaneda & Lee (2010) "Statistical Arbitrage in the US Equities Market"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from scipy import stats as sp_stats  # noqa: F401
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class PairCandidate:
    """A cointegrated pair candidate."""
    stock_a: str
    stock_b: str
    hedge_ratio: float
    half_life: float  # days
    p_value: float  # cointegration p-value
    correlation: float
    spread_std: float
    sector: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05


def _ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> float:
    """Compute OLS hedge ratio: y = beta * x + epsilon."""
    x_mean = x.mean()
    y_mean = y.mean()
    beta = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-10)
    return float(beta)


def _adf_test_pvalue(residuals: np.ndarray) -> float:
    """Approximate ADF test p-value for cointegration residuals.

    Uses augmented Dickey-Fuller: delta_z_t = alpha + phi * z_{t-1} + e_t
    (with intercept to handle non-zero-mean spreads).
    If phi < 0, residuals are mean-reverting.
    """
    if len(residuals) < 30:
        return 1.0

    z = residuals
    dz = np.diff(z)
    z_lag = z[:-1]
    n = len(dz)

    # OLS with intercept: dz = alpha + phi * z_lag
    # Using normal equations
    X = np.column_stack([np.ones(n), z_lag])
    try:
        beta = np.linalg.lstsq(X, dz, rcond=None)[0]
    except np.linalg.LinAlgError:
        return 1.0

    phi = beta[1]
    residual = dz - X @ beta
    sse = float(np.sum(residual ** 2))
    se_phi = np.sqrt(sse / max(n - 2, 1) / (np.sum((z_lag - z_lag.mean()) ** 2) + 1e-10))

    if se_phi < 1e-12:
        return 1.0

    t_stat = phi / se_phi

    # Approximate critical values for ADF with intercept (MacKinnon 1994)
    # 1%: -3.43, 5%: -2.86, 10%: -2.57
    if t_stat < -3.43:
        return 0.01
    elif t_stat < -2.86:
        return 0.05
    elif t_stat < -2.57:
        return 0.10
    elif t_stat < -1.95:
        return 0.20
    else:
        return min(1.0, 0.5 * np.exp(0.5 * t_stat))


def _estimate_half_life(spread: np.ndarray) -> float:
    """Estimate half-life via Ornstein-Uhlenbeck model.

    OU model: dz = theta * (mu - z) * dt + sigma * dW
    Half-life = -ln(2) / ln(1 + theta) ≈ ln(2) / theta
    """
    if len(spread) < 30:
        return float("inf")

    z = spread
    dz = np.diff(z)
    z_lag = z[:-1]
    z_mean = z.mean()

    # OLS: dz = theta * (z_lag - mean)
    centered = z_lag - z_mean
    theta = float(np.sum(dz * centered) / (np.sum(centered ** 2) + 1e-10))

    if theta >= 0:
        return float("inf")  # not mean-reverting

    half_life = -np.log(2) / theta
    return max(1.0, float(half_life))


def test_cointegration(
    prices_a: pd.Series,
    prices_b: pd.Series,
    symbol_a: str = "A",
    symbol_b: str = "B",
    sector: str = "",
) -> PairCandidate:
    """Test cointegration between two price series.

    Args:
        prices_a: Price series for stock A.
        prices_b: Price series for stock B.
        symbol_a: Symbol name for A.
        symbol_b: Symbol name for B.
        sector: Sector label.

    Returns:
        PairCandidate with test results.
    """
    # Align
    common = prices_a.index.intersection(prices_b.index)
    if len(common) < 60:
        return PairCandidate(
            stock_a=symbol_a, stock_b=symbol_b,
            hedge_ratio=0.0, half_life=float("inf"),
            p_value=1.0, correlation=0.0, spread_std=0.0,
            sector=sector,
        )

    pa = prices_a.reindex(common).values.astype(float)
    pb = prices_b.reindex(common).values.astype(float)

    # Hedge ratio
    hedge_ratio = _ols_hedge_ratio(pa, pb)

    # Spread
    spread = pa - hedge_ratio * pb

    # ADF test on spread
    p_value = _adf_test_pvalue(spread)

    # Half-life
    half_life = _estimate_half_life(spread)

    # Correlation
    corr = float(np.corrcoef(pa, pb)[0, 1])

    return PairCandidate(
        stock_a=symbol_a,
        stock_b=symbol_b,
        hedge_ratio=round(hedge_ratio, 6),
        half_life=round(half_life, 2),
        p_value=round(p_value, 6),
        correlation=round(corr, 4),
        spread_std=round(float(np.std(spread)), 6),
        sector=sector,
    )


def screen_pairs(
    prices: pd.DataFrame,
    sector_mapping: dict[str, str] | None = None,
    same_sector_only: bool = True,
    max_half_life: float = 60.0,
    max_p_value: float = 0.05,
) -> list[PairCandidate]:
    """Screen all pairs for cointegration.

    Args:
        prices: DataFrame with symbols as columns, dates as index.
        sector_mapping: Symbol -> sector mapping.
        same_sector_only: Only test intra-sector pairs.
        max_half_life: Maximum half-life filter.
        max_p_value: Maximum p-value filter.

    Returns:
        List of significant PairCandidates, sorted by p-value.
    """
    symbols = list(prices.columns)
    n = len(symbols)
    candidates = []

    for i in range(n):
        for j in range(i + 1, n):
            s_a, s_b = symbols[i], symbols[j]

            # Sector filter
            if same_sector_only and sector_mapping:
                sec_a = sector_mapping.get(s_a, "")
                sec_b = sector_mapping.get(s_b, "")
                if sec_a != sec_b or not sec_a:
                    continue

            sector = (sector_mapping or {}).get(s_a, "")
            pair = test_cointegration(
                prices[s_a], prices[s_b],
                symbol_a=s_a, symbol_b=s_b,
                sector=sector,
            )

            if pair.p_value <= max_p_value and pair.half_life <= max_half_life:
                candidates.append(pair)

    candidates.sort(key=lambda c: c.p_value)
    logger.info("[Cointegration] Screened %d pairs, %d significant",
                n * (n - 1) // 2, len(candidates))
    return candidates


__all__ = [
    "PairCandidate",
    "test_cointegration",
    "screen_pairs",
]
