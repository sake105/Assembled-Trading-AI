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

    def research_close(row: pd.Series) -> float:
        sym = row["symbol"]
        t = result_ts.loc[row.name]
        close = row["close"]
        sym_splits = actions[actions["symbol"] == sym].sort_values("effective_date")
        factor = 1.0
        for _, s in sym_splits.iterrows():
            if s["effective_date"] > t:
                factor *= 1.0 / float(s["split_ratio"])
        return close * factor

    result["close_research"] = result.apply(research_close, axis=1)
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
        raise ValueError(f"positions missing required columns: {sorted(required_pos - set(positions.columns))}")
    if actions.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "cashflow_type", "amount"])
    if required_act - set(actions.columns):
        raise ValueError(f"actions missing required columns: {sorted(required_act - set(actions.columns))}")
    if (actions["action_type"] != "DIVIDEND").any():
        raise ValueError("actions must contain only DIVIDEND actions")

    actions = actions.copy()
    actions["effective_date"] = pd.to_datetime(actions["effective_date"], utc=True)
    if as_of is not None:
        a = pd.Timestamp(as_of)
        as_of_utc = a.tz_convert("UTC") if a.tzinfo is not None else a.tz_localize("UTC")
        actions = actions[actions["effective_date"] <= as_of_utc]

    rows = []
    for _, pos in positions.iterrows():
        sym, qty = pos["symbol"], float(pos["qty"])
        for _, act in actions[actions["symbol"] == sym].iterrows():
            rows.append({
                "timestamp": act["effective_date"],
                "symbol": sym,
                "cashflow_type": "DIVIDEND",
                "amount": qty * float(act["dividend_cash"]),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "cashflow_type", "amount"])
    return out.sort_values("timestamp").reset_index(drop=True)


def adjust_prices_for_splits(
    prices: pd.DataFrame,
    actions: pd.DataFrame,
) -> pd.DataFrame:
    """Adjust prices for stock splits (no-op if no actions)."""
    if actions.empty:
        return prices
    return prices.copy()
