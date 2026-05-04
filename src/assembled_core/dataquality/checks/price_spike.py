"""Adaptive price-spike detection. From 37_DATA_QUALITY_GATE.md §3.1."""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_price_spikes(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    price_col: str = "close",
    timestamp_col: str = "timestamp",
    max_abs_return_1bar: float = 0.30,
    z_score_threshold: float = 6.0,
    adaptive: bool = True,
) -> pd.DataFrame:
    """Return rows flagged as price spikes.

    Checks:
      - Absolute 1-bar return > threshold (adaptive per price level)
      - 20-bar rolling z-score of return > z_score_threshold
    """
    suspects: list[pd.DataFrame] = []

    for ticker, group in df.groupby(ticker_col, sort=False):
        group = group.sort_values(timestamp_col).copy()

        threshold = max_abs_return_1bar
        if adaptive:
            avg_price = group[price_col].mean()
            if avg_price < 5:
                threshold = 0.50
            elif avg_price < 20:
                threshold = 0.35

        ret = group[price_col].pct_change(fill_method=None)
        mask_abs = (ret.abs() > threshold).fillna(False)

        roll_mean = ret.rolling(20, min_periods=5).mean()
        roll_std = ret.rolling(20, min_periods=5).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            z = (ret - roll_mean) / roll_std.replace(0, np.nan)
        mask_z = (z.abs() > z_score_threshold).fillna(False)

        combined = mask_abs | mask_z
        if not combined.any():
            continue

        flagged = group[combined].copy()
        flagged["reason"] = "price_spike"
        flagged["spike_return"] = ret[combined]
        flagged["spike_z"] = z[combined]
        suspects.append(flagged)

    if suspects:
        return pd.concat(suspects, ignore_index=True)
    return pd.DataFrame()
