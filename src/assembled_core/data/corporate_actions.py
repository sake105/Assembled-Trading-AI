"""Corporate actions (splits, dividends) stub module."""

from __future__ import annotations

import pandas as pd


def load_corporate_actions(path: str | None = None) -> pd.DataFrame:
    """Load corporate actions from CSV. Returns empty DataFrame if missing."""
    if path is None:
        return pd.DataFrame(columns=["symbol", "date", "action_type", "factor"])
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame(columns=["symbol", "date", "action_type", "factor"])


def apply_splits_for_research_prices(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """Add close_research column with split-adjusted prices for research (returns/features).

    Prices before a split date are multiplied by 1/split_ratio; close is unchanged for trading.
    """
    result = prices.copy()
    if "close" not in result.columns:
        result["close_research"] = pd.Series(dtype=float)
        return result

    if actions.empty:
        result["close_research"] = result["close"]
        return result

    required = {"symbol", "action_type", "effective_date", "split_ratio"}
    missing = required - set(actions.columns)
    if missing:
        raise ValueError(f"actions missing required columns: {sorted(missing)}")
    if (actions["action_type"] != "SPLIT").any():
        raise ValueError("actions must contain only SPLIT actions")

    # Normalize timestamps for comparison
    ts_col = "timestamp" if "timestamp" in result.columns else result.columns[0]
    result_ts = pd.to_datetime(result[ts_col], utc=True)
    actions = actions.copy()
    actions["effective_date"] = pd.to_datetime(actions["effective_date"], utc=True)

    # Vectorized split-adjustment: for each price row, multiply by the product of
    # 1/split_ratio for all future splits of that symbol.
    # Pre-group splits by symbol so we avoid repeated O(n_actions) scans per row.
    splits_by_symbol: dict[str, pd.DataFrame] = {
        sym: grp.sort_values("effective_date")
        for sym, grp in actions.groupby("symbol")
    }

    result = result.copy()
    result_ts = result_ts.reset_index(drop=True)
    result = result.reset_index(drop=True)

    # Build the cumulative adjustment factor per row in one vectorized pass per symbol
    adj_factors = pd.Series(1.0, index=result.index, dtype=float)

    for sym, sym_splits in splits_by_symbol.items():
        sym_mask = result["symbol"] == sym
        if not sym_mask.any():
            continue
        sym_idx = result.index[sym_mask]
        sym_ts = result_ts[sym_idx]

        for _, split_row in sym_splits.iterrows():
            eff_date = split_row["effective_date"]
            ratio = float(split_row["split_ratio"])
            if ratio <= 0:
                continue
            # Rows before the split effective date get divided by split_ratio
            pre_split = sym_ts < eff_date
            adj_factors.loc[sym_idx[pre_split]] /= ratio

    result["close_research"] = result["close"] * adj_factors
    return result


def compute_dividend_cashflows(
    positions: pd.DataFrame,
    actions: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute dividend cashflows for positions (ledger-ready events)."""
    required_pos = {"symbol", "qty"}
    required_act = {"symbol", "action_type", "effective_date", "dividend_cash"}
    if required_pos - set(positions.columns):
        raise ValueError(
            f"positions missing required columns: {sorted(required_pos - set(positions.columns))}"
        )
    if actions.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "cashflow_type", "amount"])
    if required_act - set(actions.columns):
        raise ValueError(
            f"actions missing required columns: {sorted(required_act - set(actions.columns))}"
        )
    if (actions["action_type"] != "DIVIDEND").any():
        raise ValueError("actions must contain only DIVIDEND actions")

    actions = actions.copy()
    actions["effective_date"] = pd.to_datetime(actions["effective_date"], utc=True)
    if as_of is not None:
        a = pd.Timestamp(as_of)
        as_of_utc = (
            a.tz_convert("UTC") if a.tzinfo is not None else a.tz_localize("UTC")
        )
        actions = actions[actions["effective_date"] <= as_of_utc]

    rows = []
    for _, pos in positions.iterrows():
        sym, qty = pos["symbol"], float(pos["qty"])
        for _, act in actions[actions["symbol"] == sym].iterrows():
            rows.append(
                {
                    "timestamp": act["effective_date"],
                    "symbol": sym,
                    "cashflow_type": "DIVIDEND",
                    "amount": qty * float(act["dividend_cash"]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "cashflow_type", "amount"])
    return out.sort_values("timestamp").reset_index(drop=True)


def adjust_prices_for_splits(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """Adjust the close column for stock splits (backward adjustment).

    Prices recorded *before* a split date are multiplied by ``1/split_ratio``
    so that the adjusted series is continuous across the split event.

    Requires actions DataFrame with columns:
        ``symbol``, ``action_type`` (must be "SPLIT"), ``effective_date``, ``split_ratio``

    Requires prices DataFrame with columns:
        ``symbol``, ``close``, and a timestamp column (``timestamp`` preferred,
        otherwise the first column is used as the time axis).

    Args:
        prices: Price DataFrame. Returns a copy with ``close`` adjusted.
        actions: Corporate actions DataFrame. Only SPLIT rows are processed.

    Returns:
        Copy of *prices* with ``close`` backward-adjusted for all splits found
        in *actions*. Returns *prices* unchanged (not a copy) if *actions* is empty.
        Returns a copy unchanged if required columns are missing in *actions*.

    Note:
        Only modifies the ``close`` column. Other price columns (open, high, low,
        volume) are preserved without adjustment.
    """
    if actions.empty:
        return prices

    required = {"symbol", "action_type", "effective_date", "split_ratio"}
    missing = required - set(actions.columns)
    if missing:
        # Cannot apply — return copy unchanged rather than raising (defensive)
        return prices.copy()

    split_actions = actions[actions["action_type"] == "SPLIT"].copy()
    if split_actions.empty:
        return prices.copy()

    if "close" not in prices.columns:
        return prices.copy()

    result = prices.copy()

    # Determine timestamp column
    ts_col = "timestamp" if "timestamp" in result.columns else result.columns[0]
    result_ts = pd.to_datetime(result[ts_col], utc=True, errors="coerce")
    split_actions["effective_date"] = pd.to_datetime(
        split_actions["effective_date"], utc=True
    )

    # Apply each split: for rows of the same symbol before the split date,
    # multiply close by 1/split_ratio (backward adjustment).
    for _, s in split_actions.iterrows():
        sym = s["symbol"]
        eff_date = s["effective_date"]
        ratio = float(s["split_ratio"])
        if ratio <= 0:
            continue

        sym_mask = result["symbol"] == sym
        before_mask = result_ts < eff_date
        apply_mask = sym_mask & before_mask

        result.loc[apply_mask, "close"] = result.loc[apply_mask, "close"] / ratio

    return result
