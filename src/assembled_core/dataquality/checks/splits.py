"""Unadjusted split detection heuristic. From 37_DATA_QUALITY_GATE.md §3.4."""

from __future__ import annotations

import pandas as pd


def detect_unadjusted_splits(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    timestamp_col: str = "timestamp",
    price_col: str = "close",
    drop_threshold: float = 0.40,
    adjacent_bars: int = 3,
) -> pd.DataFrame:
    """Heuristic: 1-bar drop > drop_threshold with no recovery → likely unadjusted split."""
    suspects: list[dict] = []

    for ticker, group in df.groupby(ticker_col, sort=False):
        group = group.sort_values(timestamp_col).reset_index(drop=True)
        ret = group[price_col].pct_change()

        for pos in ret[ret < -drop_threshold].index:
            if pos == 0:
                continue
            # Check recovery in the next adjacent_bars bars
            post_window = group.iloc[pos : pos + adjacent_bars + 1][price_col]
            drop_close = group.iloc[pos][price_col]
            max_post = post_window.max()
            recovery = (max_post - drop_close) / (drop_close + 1e-9)

            if recovery < 0.10:
                suspects.append(
                    {
                        "ticker": ticker,
                        "timestamp": group.iloc[pos][timestamp_col],
                        "close": drop_close,
                        "ret_1bar": round(float(ret.iloc[pos]), 4),
                        "recovery_pct": round(float(recovery), 4),
                        "reason": "possible_unadjusted_split",
                    }
                )

    return pd.DataFrame(suspects)
