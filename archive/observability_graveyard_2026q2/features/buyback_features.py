"""Buyback Announcement Alpha (M18 Task 18.5).

Tracks share buyback announcements and computes post-announcement drift.
Ikenberry, Lakonishok & Vermaelen (1995): +80-150 bps over 6 months.

Buyback announcements signal management believes shares are undervalued.
Post-announcement drift is one of the most persistent corporate event anomalies.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_buyback_from_shares(
    shares_outstanding: pd.Series,
    threshold_pct: float = -0.5,
) -> pd.Series:
    """Detect buyback events from declining shares outstanding.

    Args:
        shares_outstanding: Quarterly shares outstanding time series.
        threshold_pct: Minimum quarterly decrease (%) to flag as buyback.
            Default -0.5% (half a percent decline).

    Returns:
        Boolean series: True where buyback detected.
    """
    pct_change = shares_outstanding.pct_change()
    return pct_change < (threshold_pct / 100.0)


def compute_buyback_yield(
    shares_outstanding: pd.Series,
    market_cap: pd.Series,
    lookback: int = 4,
) -> pd.Series:
    """Compute annualized buyback yield.

    buyback_yield = (shares_retired * avg_price) / market_cap

    Args:
        shares_outstanding: Quarterly shares outstanding.
        market_cap: Market cap at each quarter.
        lookback: Quarters to look back (default 4 = 1 year).

    Returns:
        Buyback yield series (positive = net buyback).
    """
    shares_change = shares_outstanding.diff(lookback)
    # Negative change = buybacks, positive = issuance
    buyback_shares = -shares_change.clip(upper=0)
    avg_price = market_cap / shares_outstanding.replace(0, np.nan)
    buyback_value = buyback_shares * avg_price
    return (buyback_value / market_cap.replace(0, np.nan)).fillna(0)


def compute_buyback_completion_rate(
    announced_amount: float,
    actual_repurchased: float,
) -> float:
    """Compute buyback program completion rate.

    High completion → management committed → stronger signal.

    Args:
        announced_amount: Announced buyback size ($).
        actual_repurchased: Actual amount repurchased ($).

    Returns:
        Completion rate (0-1+, can exceed 1 if over-bought).
    """
    if announced_amount <= 0:
        return 0.0
    return actual_repurchased / announced_amount


def post_buyback_drift(
    prices: pd.Series,
    buyback_dates: list[pd.Timestamp],
    horizon: int = 126,
) -> pd.DataFrame:
    """Compute post-buyback-announcement drift.

    Args:
        prices: Daily price series.
        buyback_dates: Dates of buyback announcements.
        horizon: Forward return horizon in trading days (default 126 ≈ 6 months).

    Returns:
        DataFrame with columns [announcement_date, forward_return].
    """
    results = []
    for date in buyback_dates:
        if date not in prices.index:
            # Find nearest trading day
            idx = prices.index.searchsorted(date)
            if idx >= len(prices.index):
                continue
            date = prices.index[idx]

        loc = prices.index.get_loc(date)
        end_loc = min(loc + horizon, len(prices) - 1)
        if end_loc <= loc:
            continue

        fwd_return = (prices.iloc[end_loc] / prices.iloc[loc]) - 1.0
        results.append({
            "announcement_date": date,
            "forward_return": fwd_return,
        })

    return pd.DataFrame(results)


def build_buyback_features(
    prices: pd.Series,
    shares_outstanding: pd.Series | None = None,
    market_cap: pd.Series | None = None,
    buyback_dates: list[pd.Timestamp] | None = None,
    lookback: int = 252,
) -> pd.DataFrame:
    """Build buyback-related alpha features.

    Args:
        prices: Daily close prices.
        shares_outstanding: Quarterly/monthly shares outstanding (optional).
        market_cap: Market cap series (optional).
        buyback_dates: Known buyback announcement dates (optional).
        lookback: Lookback for feature computation.

    Returns:
        DataFrame with buyback features indexed by date.
    """
    features = pd.DataFrame(index=prices.index)

    # Feature 1: Recent buyback flag (from shares outstanding decline)
    if shares_outstanding is not None and len(shares_outstanding) > 1:
        # Reindex to daily, forward fill
        shares_daily = shares_outstanding.reindex(prices.index, method="ffill")
        shares_change = shares_daily.pct_change(min(20, len(shares_daily) - 1))
        features["buyback_flag"] = (shares_change < -0.001).astype(float)
        features["shares_change_pct"] = shares_change.fillna(0)
    else:
        features["buyback_flag"] = 0.0
        features["shares_change_pct"] = 0.0

    # Feature 2: Buyback yield (annualized)
    if shares_outstanding is not None and market_cap is not None:
        shares_daily = shares_outstanding.reindex(prices.index, method="ffill")
        mcap_daily = market_cap.reindex(prices.index, method="ffill")
        features["buyback_yield"] = compute_buyback_yield(
            shares_daily, mcap_daily, lookback=min(lookback, len(shares_daily) - 1)
        )
    else:
        features["buyback_yield"] = 0.0

    # Feature 3: Post-announcement proximity
    if buyback_dates:
        days_since = pd.Series(np.nan, index=prices.index)
        for bd in sorted(buyback_dates):
            mask = prices.index >= bd
            days_since[mask] = (prices.index[mask] - bd).days
        # Decay: exp(-days/60) — strongest in first ~60 trading days
        features["buyback_recency"] = np.exp(-days_since.fillna(9999) / 60.0)
    else:
        features["buyback_recency"] = 0.0

    # Feature 4: Price vs. buyback price (if price fell since announcement → stronger signal)
    if buyback_dates:
        buyback_price = np.nan
        price_vs_buyback = pd.Series(0.0, index=prices.index)
        for bd in sorted(buyback_dates):
            if bd in prices.index:
                buyback_price = prices.loc[bd]
            else:
                idx = prices.index.searchsorted(bd)
                if idx < len(prices.index):
                    buyback_price = prices.iloc[idx]
            if not np.isnan(buyback_price):
                mask = prices.index >= bd
                price_vs_buyback[mask] = (prices[mask] / buyback_price) - 1.0
        features["price_vs_buyback"] = price_vs_buyback
    else:
        features["price_vs_buyback"] = 0.0

    return features


__all__ = [
    "detect_buyback_from_shares",
    "compute_buyback_yield",
    "compute_buyback_completion_rate",
    "post_buyback_drift",
    "build_buyback_features",
]
