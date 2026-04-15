"""Short Interest Features — signals from FINRA short data.

Implements:
    - short_pct_float: Short interest as % of float
    - short_ratio: Days to cover (short interest / avg volume)
    - short_squeeze_score: Combined signal for squeeze potential
    - short_momentum: Change in short interest
    - short_utilization: Shares shorted / shares available to borrow

References:
    Desai et al. (2002), Asquith et al. (2005)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core short interest metrics
# ---------------------------------------------------------------------------


def compute_short_pct_float(
    short_interest: float,
    shares_float: float,
) -> float:
    """Short interest as percentage of float.

    Args:
        short_interest: Total shares sold short.
        shares_float: Total shares in public float.

    Returns:
        Short interest ratio (0.0 to 1.0+). Values > 0.20 are elevated.
    """
    if shares_float <= 0:
        return 0.0
    return short_interest / shares_float


def compute_short_ratio(
    short_interest: float,
    avg_daily_volume: float,
) -> float:
    """Days-to-cover ratio (short interest / average daily volume).

    Args:
        short_interest: Total shares sold short.
        avg_daily_volume: Average daily trading volume (20-day typical).

    Returns:
        Days to cover. Values > 5 indicate crowded short.
    """
    if avg_daily_volume <= 0:
        return 0.0
    return short_interest / avg_daily_volume


def compute_short_squeeze_score(
    short_pct_float: float,
    short_ratio: float,
    short_momentum: float = 0.0,
    cost_to_borrow: float = 0.0,
) -> float:
    """Composite short squeeze potential score.

    Higher values indicate greater squeeze risk.

    Args:
        short_pct_float: Short interest / float (0-1+).
        short_ratio: Days to cover.
        short_momentum: Recent change in short interest (positive = increasing).
        cost_to_borrow: Annual cost to borrow (0-1+).

    Returns:
        Score in [0, 1] range. > 0.7 = high squeeze risk.
    """
    # Normalize components to 0-1
    si_score = min(short_pct_float / 0.30, 1.0)  # 30% float = max
    dtc_score = min(short_ratio / 10.0, 1.0)  # 10 days = max
    mom_score = min(max(short_momentum, 0) / 0.10, 1.0)  # 10% increase = max
    ctb_score = min(cost_to_borrow / 0.50, 1.0)  # 50% annual = max

    # Weighted combination
    score = 0.35 * si_score + 0.30 * dtc_score + 0.20 * mom_score + 0.15 * ctb_score
    return round(float(score), 4)


# ---------------------------------------------------------------------------
# Panel-level feature builder
# ---------------------------------------------------------------------------


def build_short_interest_features(
    short_data: pd.DataFrame,
    symbol_col: str = "symbol",
    si_col: str = "short_interest",
    float_col: str = "shares_float",
    volume_col: str = "avg_volume",
    date_col: str = "settlement_date",
) -> pd.DataFrame:
    """Build short interest features from a panel of short data.

    Args:
        short_data: DataFrame with short interest data per symbol per date.
        symbol_col: Symbol column name.
        si_col: Short interest column.
        float_col: Shares float column.
        volume_col: Average daily volume column.
        date_col: Settlement date column.

    Returns:
        DataFrame with added feature columns:
        - si_pct_float: Short interest as % of float
        - si_days_to_cover: Days to cover ratio
        - si_momentum_2w: 2-week change in short interest
        - si_squeeze_score: Composite squeeze score
    """
    if short_data.empty:
        return short_data.copy()

    df = short_data.copy()
    df = df.sort_values([symbol_col, date_col])

    # Short % float
    si = df[si_col].fillna(0).values
    fl = df[float_col].fillna(0).values
    df["si_pct_float"] = np.where(fl > 0, si / fl, 0.0)

    # Days to cover
    vol = df[volume_col].fillna(0).values
    df["si_days_to_cover"] = np.where(vol > 0, si / vol, 0.0)

    # Short interest momentum (percent change)
    df["si_momentum_2w"] = df.groupby(symbol_col)[si_col].pct_change(periods=1).fillna(0.0)

    # Squeeze score
    df["si_squeeze_score"] = df.apply(
        lambda row: compute_short_squeeze_score(
            short_pct_float=row.get("si_pct_float", 0),
            short_ratio=row.get("si_days_to_cover", 0),
            short_momentum=row.get("si_momentum_2w", 0),
        ),
        axis=1,
    )

    logger.info("[ShortInterest] Built features for %d rows, %d symbols",
                len(df), df[symbol_col].nunique())
    return df


def get_short_interest_feature_names() -> list[str]:
    """Return list of short interest feature column names."""
    return ["si_pct_float", "si_days_to_cover", "si_momentum_2w", "si_squeeze_score"]
