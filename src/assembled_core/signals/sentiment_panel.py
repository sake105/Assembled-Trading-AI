"""Sentiment Panel — Fear & Greed replication from free data.

From 13_FREE_MODULE.md §13.10.
Contrarian overlay: Retail panic → accumulate. Extreme greed → reduce.

Components (all free):
  - CBOE Put/Call Ratio (CBOE public CSVs)
  - HYG/LQD spread proxy (FRED: BAMLH0A0HYM2 / BAMLC0A0CM)
  - VIX (FRED: VIXCLS)
  - 127-day SPY momentum (yfinance)
  - UMich Consumer Sentiment (FRED: UMCSENT)

Score range: 0–100 (100 = extreme fear, contrarian buy signal)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _zscore(s: pd.Series, window: int = 252) -> pd.Series:
    roll = s.rolling(window=window, min_periods=max(window // 4, 20))
    return (s - roll.mean()) / roll.std().replace(0, np.nan)


def compute_sentiment_panel(
    cboe_put_call: pd.Series,
    hy_spread: pd.Series,
    vix: pd.Series,
    spy_127d_return: pd.Series,
    umich_sentiment: pd.Series | None = None,
    lookback: int = 252,
) -> pd.Series:
    """Compute composite sentiment panel score.

    Args:
        cboe_put_call: CBOE equity put/call ratio
        hy_spread: HY OAS spread (FRED: BAMLH0A0HYM2)
        vix: VIX daily close
        spy_127d_return: 127-day SPY return (trailing momentum)
        umich_sentiment: UMich Consumer Sentiment Index (optional, weekly)
        lookback: Z-score window

    Returns:
        Series with sentiment scores 0–100. 100 = extreme fear.
    """
    common = cboe_put_call.index
    for s in [hy_spread, vix, spy_127d_return]:
        common = common.intersection(s.index)

    if len(common) == 0:
        return pd.Series(dtype=float)

    components = {
        "put_call": _zscore(cboe_put_call.loc[common], lookback),
        "hy_spread": _zscore(hy_spread.loc[common], lookback),
        "vix": _zscore(vix.loc[common], lookback),
        # Negative momentum → fear → positive z-score for fear index
        "spy_momentum": -_zscore(spy_127d_return.loc[common], lookback),
    }

    if umich_sentiment is not None:
        umich_aligned = umich_sentiment.reindex(common, method="ffill")
        # Low consumer sentiment = more fear → invert
        components["umich"] = -_zscore(umich_aligned, lookback)

    n_components = len(components)
    composite = sum(components.values()) / n_components
    # Map to 0-100 scale: composite z-score clamped [-3, +3] → [0, 100]
    score = 50 + (10 * composite).clip(-50, 50)
    score.name = "sentiment_fear_score"
    return score


def sentiment_multiplier(score: float) -> float:
    """Return long-bias multiplier based on sentiment score.

    Extreme fear (score > 80) → 1.2 (accumulate)
    Normal (20–80) → 1.0
    Extreme greed (score < 20) → 0.7 (reduce)
    """
    if score > 80:
        return 1.2
    if score < 20:
        return 0.7
    return 1.0


def latest_sentiment_score(fred_client: object, spy_return_127d: float) -> float:
    """Compute latest sentiment score from FRED data.

    Args:
        fred_client: fredapi.Fred instance.
        spy_return_127d: Pre-computed 127-day SPY return (float).

    Returns:
        Sentiment score 0-100. Returns 50.0 on failure.
    """
    try:
        vix = fred_client.get_series("VIXCLS")
        hy = fred_client.get_series("BAMLH0A0HYM2")
        umich = fred_client.get_series("UMCSENT")
        # CBOE put/call via local daily download (CBOE public CSV)
        # Use VIX as proxy if CBOE unavailable
        common_idx = vix.index.intersection(hy.index)
        if len(common_idx) < 20:
            return 50.0

        vix_z = float(_zscore(vix).iloc[-1])
        hy_z = float(_zscore(hy).iloc[-1])
        spy_z = -float((spy_return_127d - 0.0) / 0.15) if abs(spy_return_127d) < 2 else 0.0

        umich_z = 0.0
        if umich is not None and len(umich) > 20:
            umich_z = -float(_zscore(umich).iloc[-1])

        composite = (vix_z + hy_z + spy_z + umich_z) / 4.0
        return float(50 + max(-50, min(50, composite * 10)))
    except Exception as exc:
        logger.debug("Sentiment panel from FRED failed: %s", exc)
        return 50.0


__all__ = [
    "compute_sentiment_panel",
    "sentiment_multiplier",
    "latest_sentiment_score",
]
