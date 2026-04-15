"""Analyst Revision Momentum Features.

Signals from sell-side analyst estimate revisions — EPS, revenue, target price.
Based on Chan et al. (1996): Post-revision drift.

Features:
    - eps_revision_1m: Net EPS revision (up - down) in last 30 days
    - revenue_revision_1m: Net revenue revision in last 30 days
    - target_price_change: % change in consensus target price
    - revision_breadth: Fraction of analysts revising up vs down
    - estimate_dispersion: StdDev of estimates / mean (uncertainty)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_eps_revision_score(
    estimates: pd.DataFrame,
    symbol: str,
    as_of: pd.Timestamp,
    lookback_days: int = 30,
    date_col: str = "date",
    symbol_col: str = "symbol",
    estimate_col: str = "eps_estimate",
    direction_col: str = "revision_direction",
) -> dict:
    """Compute EPS revision momentum for a single symbol.

    Args:
        estimates: DataFrame of analyst estimate revisions.
        symbol: Target symbol.
        as_of: Reference date for PIT safety.
        lookback_days: Window for recent revisions.
        date_col: Date column name.
        symbol_col: Symbol column name.
        estimate_col: EPS estimate column.
        direction_col: Revision direction column (up/down/unchanged).

    Returns:
        Dict with eps_revision_1m, revision_breadth, estimate_dispersion.
    """
    if estimates.empty:
        return {"eps_revision_1m": 0.0, "revision_breadth": 0.0, "estimate_dispersion": 0.0}

    mask = (
        (estimates[symbol_col] == symbol)
        & (pd.to_datetime(estimates[date_col]) <= as_of)
        & (pd.to_datetime(estimates[date_col]) >= as_of - pd.Timedelta(days=lookback_days))
    )
    recent = estimates.loc[mask]

    if recent.empty:
        return {"eps_revision_1m": 0.0, "revision_breadth": 0.0, "estimate_dispersion": 0.0}

    # Revision breadth
    if direction_col in recent.columns:
        ups = (recent[direction_col].str.lower() == "up").sum()
        downs = (recent[direction_col].str.lower() == "down").sum()
        total = ups + downs
        breadth = (ups - downs) / total if total > 0 else 0.0
    else:
        breadth = 0.0

    # Estimate dispersion
    if estimate_col in recent.columns:
        vals = recent[estimate_col].dropna()
        mean_est = vals.mean()
        std_est = vals.std()
        dispersion = (std_est / abs(mean_est)) if abs(mean_est) > 1e-9 else 0.0
    else:
        dispersion = 0.0

    return {
        "eps_revision_1m": float(breadth),
        "revision_breadth": float(breadth),
        "estimate_dispersion": float(dispersion),
    }


def compute_target_price_change(
    current_target: float,
    previous_target: float,
) -> float:
    """Compute percentage change in consensus target price.

    Args:
        current_target: Current consensus target price.
        previous_target: Previous consensus target price (e.g. 30 days ago).

    Returns:
        Percentage change (-1 to +1+ range).
    """
    if previous_target <= 0:
        return 0.0
    return (current_target - previous_target) / previous_target


def build_analyst_features(
    estimates_df: pd.DataFrame,
    symbols: list[str],
    as_of: pd.Timestamp,
    lookback_days: int = 30,
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> pd.DataFrame:
    """Build analyst revision features for a list of symbols.

    Args:
        estimates_df: Panel of analyst estimates with revisions.
        symbols: List of symbols to compute features for.
        as_of: PIT-safe reference date.
        lookback_days: Lookback window for revisions.
        symbol_col: Symbol column name.
        date_col: Date column name.

    Returns:
        DataFrame indexed by symbol with analyst feature columns.
    """
    if estimates_df.empty or not symbols:
        return pd.DataFrame(
            columns=["eps_revision_1m", "revision_breadth",
                     "estimate_dispersion", "target_price_change"],
        )

    rows = []
    for sym in symbols:
        scores = compute_eps_revision_score(
            estimates_df, sym, as_of, lookback_days,
            date_col=date_col, symbol_col=symbol_col,
        )
        scores["symbol"] = sym
        scores["target_price_change"] = 0.0  # Requires target price data
        rows.append(scores)

    result = pd.DataFrame(rows)
    if "symbol" in result.columns:
        result = result.set_index("symbol")

    logger.info("[AnalystFeatures] Built features for %d symbols as of %s",
                len(result), as_of)
    return result


def get_analyst_feature_names() -> list[str]:
    """Return list of analyst feature column names."""
    return [
        "eps_revision_1m",
        "revision_breadth",
        "estimate_dispersion",
        "target_price_change",
    ]
