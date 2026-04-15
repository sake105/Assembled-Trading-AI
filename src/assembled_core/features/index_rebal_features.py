"""Index Rebalancing Front-Running Features.

Tracks S&P 500 and Russell 2000 additions/deletions to predict
forced index-fund buying/selling pressure.

Reference: Petajisto (2011) — Index rebalancing creates predictable demand.

Features:
    - index_addition_flag: 1 if added to index, -1 if deleted
    - predicted_demand_pct: Estimated demand as % of float
    - rebal_window_flag: 1 if within rebalancing window
    - index_demand_score: Composite signal combining all factors
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Major index AUM estimates (simplified, in billions USD)
_INDEX_AUM = {
    "SP500": 7_000e9,
    "RUSSELL2000": 500e9,
    "RUSSELL1000": 2_000e9,
}


def compute_predicted_demand(
    market_cap: float,
    index_weight: float,
    index_aum: float,
    shares_float: float,
    current_price: float,
) -> float:
    """Estimate forced buying/selling as % of float.

    When a stock is added to an index, passive funds must buy.
    Demand = index_weight * index_total_AUM / (shares_float * price)

    Args:
        market_cap: Company market cap.
        index_weight: Expected weight in index (0-1).
        index_aum: Total assets tracking the index.
        shares_float: Total shares in public float.
        current_price: Current share price.

    Returns:
        Predicted demand as fraction of float (0-1+).
    """
    if shares_float <= 0 or current_price <= 0:
        return 0.0

    demand_dollars = index_weight * index_aum
    demand_shares = demand_dollars / current_price
    return demand_shares / shares_float


def build_index_rebal_features(
    changes_df: pd.DataFrame,
    prices_df: pd.DataFrame | None = None,
    symbol_col: str = "symbol",
    date_col: str = "effective_date",
    action_col: str = "action",
    index_col: str = "index_name",
) -> pd.DataFrame:
    """Build index rebalancing features from change announcements.

    Args:
        changes_df: DataFrame of index changes with columns:
            symbol, effective_date, action (add/delete), index_name.
        prices_df: Optional price panel for market cap / float estimates.
        symbol_col: Symbol column name.
        date_col: Effective date column.
        action_col: Action column (add/delete/addition/deletion).
        index_col: Index name column.

    Returns:
        DataFrame with rebalancing feature columns per symbol per date.
    """
    if changes_df.empty:
        return pd.DataFrame(columns=[
            symbol_col, date_col, "index_addition_flag",
            "predicted_demand_pct", "rebal_window_flag", "index_demand_score",
        ])

    df = changes_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Classify action
    action_map = {"add": 1, "addition": 1, "delete": -1, "deletion": -1, "removal": -1}
    df["index_addition_flag"] = df[action_col].str.lower().map(action_map).fillna(0).astype(float)

    # Rebalancing window: 5 trading days before effective date
    rows = []
    for _, row in df.iterrows():
        eff_date = row[date_col]
        sym = row[symbol_col]
        flag = row["index_addition_flag"]
        index_name = row.get(index_col, "SP500")
        aum = _INDEX_AUM.get(str(index_name).upper(), _INDEX_AUM["SP500"])

        # Create window: T-5 to T
        window_dates = pd.bdate_range(end=eff_date, periods=6)
        for wd in window_dates:
            days_to_rebal = (eff_date - wd).days
            # Predicted demand (simplified: assume equal weight)
            n_constituents = 500 if "500" in str(index_name) else 2000
            est_weight = 1.0 / n_constituents
            est_demand = est_weight * aum  # demand in dollars

            rows.append({
                symbol_col: sym,
                date_col: wd,
                "index_addition_flag": flag,
                "predicted_demand_pct": est_weight * 100,  # simplified
                "rebal_window_flag": 1.0,
                "days_to_rebal": days_to_rebal,
                "index_demand_score": float(flag) * max(0, 1 - days_to_rebal / 6),
            })

    result = pd.DataFrame(rows)
    logger.info("[IndexRebal] Built features for %d events, %d window-rows",
                len(df), len(result))
    return result


def get_index_rebal_feature_names() -> list[str]:
    """Return list of index rebalancing feature column names."""
    return [
        "index_addition_flag",
        "predicted_demand_pct",
        "rebal_window_flag",
        "index_demand_score",
    ]
