"""Volume anomaly detection. From 37_DATA_QUALITY_GATE.md §3.3."""

from __future__ import annotations

import pandas as pd


def detect_volume_anomalies(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    timestamp_col: str = "timestamp",
    spike_multiple: float = 20.0,
    zero_volume_tolerance: int = 5,
) -> pd.DataFrame:
    """Return rows flagged for volume spikes or zero-volume runs."""
    suspects: list[pd.DataFrame] = []

    for ticker, group in df.groupby(ticker_col, sort=False):
        group = group.sort_values(timestamp_col).copy()

        med20 = group["volume"].rolling(20, min_periods=5).median()
        spike_mask = ((med20 > 0) & (group["volume"] > spike_multiple * med20)).fillna(
            False
        )

        is_zero = (group["volume"] == 0).astype(int)
        zero_run = is_zero.rolling(
            zero_volume_tolerance, min_periods=zero_volume_tolerance
        ).sum()
        zero_run_mask = (zero_run >= zero_volume_tolerance).fillna(False)

        combined = spike_mask | zero_run_mask
        if not combined.any():
            continue

        flagged = group[combined].copy()
        # Assign reason — spike takes priority
        flagged["reason"] = "zero_volume_run"
        flagged.loc[
            spike_mask[spike_mask].index.intersection(flagged.index), "reason"
        ] = "volume_spike"
        flagged["vol_ratio"] = (group["volume"] / med20.clip(lower=1))[combined]
        suspects.append(flagged)

    if suspects:
        return pd.concat(suspects, ignore_index=True)
    return pd.DataFrame()
