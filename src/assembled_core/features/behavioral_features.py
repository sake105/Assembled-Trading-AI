"""Behavioral Finance Factors (M41).

Implements academically-validated behavioral biases as alpha factors:

1. Disposition Effect / Capital Gains Overhang (Grinblatt & Han 2005)
2. Anchoring Bias / 52-Week High Proximity (George & Hwang 2004)
3. Investor Attention / Abnormal Volume (Da Costa et al. 2015)
4. Lottery Preference / MAX Effect (Bali et al. 2011)
5. Overconfidence / Abnormal Turnover (Odean 1999)

Combined alpha: +200-500 bps/year across all behavioral factors.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def capital_gains_overhang(
    prices: pd.Series,
    volumes: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """Capital Gains Overhang — Disposition Effect proxy.

    CGO = (price - avg_cost_basis) / price
    where avg_cost_basis is volume-weighted average price.

    High CGO → lots of paper gains → less selling pressure → momentum reinforced.

    Args:
        prices: Daily close prices.
        volumes: Daily trading volume.
        lookback: Lookback window for VWAP cost basis.

    Returns:
        CGO series.

    Reference: Grinblatt & Han (2005), Frazzini (2006)
    Alpha: +60-120 bps/year
    """
    vwap = (prices * volumes).rolling(lookback, min_periods=20).sum() / volumes.rolling(
        lookback, min_periods=20
    ).sum().replace(0, np.nan)
    cgo = (prices - vwap) / prices.replace(0, np.nan)
    return cgo.fillna(0)


def anchoring_52w_high(
    prices: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """52-Week High Proximity — Anchoring Bias.

    proximity = price / 52w_high
    Stocks near 52W high tend to break higher (George & Hwang 2004).

    Args:
        prices: Daily close prices.
        lookback: Window for high computation (default 252 = 1 year).

    Returns:
        Proximity ratio (0-1, higher = closer to 52W high).

    Reference: George & Hwang (2004)
    Alpha: +40-80 bps/year
    """
    high_52w = prices.rolling(lookback, min_periods=20).max()
    proximity = prices / high_52w.replace(0, np.nan)
    return proximity.fillna(0)


def round_number_proximity(
    prices: pd.Series,
    levels: list[float] | None = None,
) -> pd.Series:
    """Round Number Proximity — Anchoring Bias.

    Measures how close price is to the nearest round number.
    Stocks just below round numbers have higher breakout probability.

    Args:
        prices: Daily close prices.
        levels: Round number levels. Default: [10, 25, 50, 100, 200, 500].

    Returns:
        Proximity score (0-1, higher = closer to round number from below).
    """
    levels = levels or [10, 25, 50, 100, 200, 500]

    def _proximity(price: float) -> float:
        if price <= 0:
            return 0.0
        min_dist = float("inf")
        for lvl in levels:
            # Distance to nearest round number
            nearest = round(price / lvl) * lvl
            dist = (nearest - price) / lvl  # positive = below round number
            if 0 < dist < min_dist:
                min_dist = dist
        return 1.0 / (1.0 + min_dist * 10) if min_dist < float("inf") else 0.0

    return prices.apply(_proximity)


def abnormal_volume(
    volumes: pd.Series,
    lookback: int = 60,
) -> pd.Series:
    """Abnormal Volume — Investor Attention proxy.

    Volume spike > 3x average signals attention-driven overreaction.

    Args:
        volumes: Daily trading volume.
        lookback: Window for average volume.

    Returns:
        Abnormal volume ratio (>1 = above average).

    Reference: Da Costa et al. (2015), Joseph et al. (2011)
    Alpha: +50-100 bps/year (mean-reversion after spikes)
    """
    avg_vol = volumes.rolling(lookback, min_periods=20).mean()
    return (volumes / avg_vol.replace(0, np.nan)).fillna(1.0)


def max_effect(
    returns: pd.Series,
    lookback: int = 20,
) -> pd.Series:
    """MAX Effect — Lottery Preference Factor.

    MAX = maximum daily return over last N days.
    Investors overpay for lottery-like stocks (high MAX).
    Short high-MAX, long low-MAX.

    Args:
        returns: Daily returns.
        lookback: Window for MAX computation (default 20 = 1 month).

    Returns:
        Rolling MAX return.

    Reference: Bali, Cakici & Whitelaw (2011)
    Alpha: +80-150 bps/year (short high-MAX)
    """
    return returns.rolling(lookback, min_periods=5).max().fillna(0)


def abnormal_turnover(
    volumes: pd.Series,
    shares_outstanding: float,
    lookback: int = 60,
) -> pd.Series:
    """Abnormal Turnover — Overconfidence proxy.

    High turnover relative to recent average signals overconfidence.

    Args:
        volumes: Daily trading volume.
        shares_outstanding: Total shares outstanding.
        lookback: Window for average turnover.

    Returns:
        Abnormal turnover ratio.

    Reference: Odean (1999)
    Alpha: +30-60 bps/year (short high-turnover after earnings)
    """
    turnover = volumes / max(shares_outstanding, 1)
    avg_turnover = turnover.rolling(lookback, min_periods=20).mean()
    return (turnover / avg_turnover.replace(0, np.nan)).fillna(1.0)


def compute_behavioral_composite(
    prices: pd.Series,
    volumes: pd.Series,
    returns: pd.Series,
    shares_outstanding: float = 1e8,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """Compute composite behavioral factor score.

    Combines all behavioral factors into a single score.
    Higher = more behavioral alpha potential.

    Args:
        prices: Daily close prices.
        volumes: Daily trading volume.
        returns: Daily returns.
        shares_outstanding: Total shares.
        weights: Factor weights. Default: equal weight.

    Returns:
        Composite z-score.
    """
    factors = {}
    factors["cgo"] = capital_gains_overhang(prices, volumes)
    factors["anchor_52w"] = anchoring_52w_high(prices)
    factors["abn_vol"] = abnormal_volume(volumes)
    factors["max_effect"] = -max_effect(returns)  # negative: short high-MAX
    factors["abn_turnover"] = -abnormal_turnover(volumes, shares_outstanding)

    weights = weights or {k: 1.0 for k in factors}

    df = pd.DataFrame(factors)
    # Z-score each factor
    z_scores = (df - df.rolling(252, min_periods=60).mean()) / (
        df.rolling(252, min_periods=60).std() + 1e-10
    )

    # Weighted composite
    composite = sum(weights.get(k, 0) * z_scores[k] for k in factors) / sum(
        weights.values()
    )

    return composite.fillna(0)


__all__ = [
    "capital_gains_overhang",
    "anchoring_52w_high",
    "round_number_proximity",
    "abnormal_volume",
    "max_effect",
    "abnormal_turnover",
    "compute_behavioral_composite",
]
