"""Congressional trading features module (Phase 6 Skeleton).

This module provides functions to compute features from congressional trading events.
Currently provides skeleton implementations with simple aggregation logic.

B2 Integration: This module now supports PIT-safe filtering using disclosure_date.
Events are filtered to only include those disclosed by as_of, preventing look-ahead bias.

Zukünftige Integration:
- Party-weighted features (D vs R)
- Politician influence scores
- Sector-level aggregations
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.assembled_core.data.latency import (
    apply_source_latency,
    ensure_event_schema,
    filter_events_as_of,
)


def add_congress_features(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    disclosure_latency_days: int = 45,  # audit C4-082 — data.source_latencies.CONGRESS_DAYS
) -> pd.DataFrame:
    """Add congressional trading features to price DataFrame (PIT-safe).

    Computes features like:
    - congress_trade_count_60d: Number of congressional trades in last 60 days
    - congress_total_amount_60d: Total trade amount in USD in last 60 days
    - congress_trade_count_90d: Number of trades in last 90 days
    - congress_total_amount_90d: Total trade amount in last 90 days

    B2 PIT Safety:
    - If as_of is provided, only events with disclosure_date <= as_of are used
    - If disclosure_date is missing, it is derived from timestamp + disclosure_latency_days
    - This ensures features are "blind" to events not yet disclosed

    Args:
        prices: DataFrame with columns: timestamp (UTC), symbol, close (and optionally other price columns)
        events: DataFrame from congress_trades_ingest.load_congress_sample() with columns:
            timestamp, symbol, politician, party, amount
            Optional: event_date, disclosure_date (if missing, derived from timestamp)
        as_of: Optional point-in-time cutoff (pd.Timestamp, UTC)
            Only events with disclosure_date <= as_of are used. If None, all events are used.
        disclosure_latency_days: Minimum days between event_date and disclosure_date (default: 45).
            House PTR under STOCK Act requires filing within 45 days; Senate within 30 days.
            Used if disclosure_date is missing from the events data.

    Returns:
        Copy of prices DataFrame with additional columns:
        - congress_trade_count_60d: Trade count in last 60 days
        - congress_total_amount_60d: Total amount in last 60 days
        - congress_trade_count_90d: Trade count in last 90 days
        - congress_total_amount_90d: Total amount in last 90 days

    Raises:
        KeyError: If required columns are missing in prices or events
    """
    # Validate inputs
    required_price_cols = ["timestamp", "symbol", "close"]
    for col in required_price_cols:
        if col not in prices.columns:
            raise KeyError(f"Required column '{col}' not found in prices DataFrame")

    result = prices.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    # Ensure event schema (timestamp, symbol required)
    events = ensure_event_schema(
        events, required_cols=["timestamp", "symbol"], strict=False
    )

    # Ensure events have disclosure_date (derive from timestamp if missing)
    if "disclosure_date" not in events.columns:
        events = apply_source_latency(
            events,
            days=disclosure_latency_days,
            event_date_col="event_date",
            timestamp_col="timestamp",
        )

    # Apply PIT-safe filtering if as_of is provided
    if as_of is not None:
        if isinstance(as_of, pd.Timestamp):
            events = filter_events_as_of(
                events, as_of, disclosure_col="disclosure_date"
            )

    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    # Initialize feature columns
    result["congress_trade_count_60d"] = 0
    result["congress_total_amount_60d"] = 0.0
    result["congress_trade_count_90d"] = 0
    result["congress_total_amount_90d"] = 0.0

    # Pre-group events by symbol to avoid O(N*M) per-symbol filter
    _events_by_sym = {sym: grp for sym, grp in events.groupby("symbol", sort=False)}

    td60_ns = int(pd.Timedelta(days=60).value)
    td90_ns = int(pd.Timedelta(days=90).value)

    # Group by symbol for efficient processing
    for symbol, symbol_prices in result.groupby("symbol", sort=False):
        symbol_events = _events_by_sym.get(symbol, pd.DataFrame())
        if symbol_events.empty:
            continue

        # Determine window time column and PIT flag once per symbol
        window_time_col = (
            "event_date" if "event_date" in symbol_events.columns else "timestamp"
        )
        has_pit = "disclosure_date" in symbol_events.columns
        has_amount = "amount" in symbol_events.columns

        ev_time_ns = symbol_events[window_time_col].values.astype("int64")
        ev_amounts = (
            symbol_events["amount"].values
            if has_amount
            else np.zeros(len(symbol_events))
        )
        if has_pit:
            ev_disclose_ns = symbol_events["disclosure_date"].values.astype("int64")
        else:
            ev_disclose_ns = None

        price_ts_ns = symbol_prices["timestamp"].values.astype("int64")
        n_prices = len(price_ts_ns)

        tc60 = np.zeros(n_prices, dtype=np.int64)
        ta60 = np.zeros(n_prices)
        tc90 = np.zeros(n_prices, dtype=np.int64)
        ta90 = np.zeros(n_prices)

        for i, pt_ns in enumerate(price_ts_ns):
            if ev_disclose_ns is not None:
                _ns_per_day = 86_400_000_000_000
                pt_day_ns = (pt_ns // _ns_per_day) * _ns_per_day
                ev_day_ns = (ev_disclose_ns // _ns_per_day) * _ns_per_day
                pit = ev_day_ns <= pt_day_ns
            else:
                pit = np.ones(len(ev_time_ns), dtype=bool)

            w60 = pit & (ev_time_ns <= pt_ns) & (ev_time_ns > pt_ns - td60_ns)
            w90 = pit & (ev_time_ns <= pt_ns) & (ev_time_ns > pt_ns - td90_ns)

            tc60[i] = int(w60.sum())
            ta60[i] = ev_amounts[w60].sum()
            tc90[i] = int(w90.sum())
            ta90[i] = ev_amounts[w90].sum()

        result.loc[symbol_prices.index, "congress_trade_count_60d"] = tc60
        result.loc[symbol_prices.index, "congress_total_amount_60d"] = ta60
        result.loc[symbol_prices.index, "congress_trade_count_90d"] = tc90
        result.loc[symbol_prices.index, "congress_total_amount_90d"] = ta90

    return result


# ---------------------------------------------------------------------------
# Congress Trading Alpha Extensions (Plan 3.7)
# ---------------------------------------------------------------------------


def compute_congress_net_buy_score(
    trades_df: "pd.DataFrame",
    window_days: int = 30,
    committee_weight: float = 2.0,
    committee_members: set | None = None,
) -> dict[str, float]:
    """Compute net-buy score from Congress trading data.

    Uses disclosure_date (not trade_date) for PIT safety.

    Args:
        trades_df: DataFrame with columns: symbol, amount, type (buy/sell),
            disclosure_date, member_id.
        window_days: Rolling window.
        committee_weight: Weight multiplier for committee members.
        committee_members: Set of member IDs on relevant committees.

    Returns:
        Symbol -> net buy score.
    """

    if trades_df is None or trades_df.empty:
        return {}

    df = trades_df.copy()
    df["_amount"] = (
        pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
        if "amount" in df.columns
        else 0.0
    )
    # Three-branch sign: buy/purchase -> +1, sell/sale -> -1, unknown/None -> 0
    # (neutral). A two-branch where would fabricate a directional SELL sign for
    # Exchange/unknown/missing-side rows (fail-open). Neutral rows contribute 0.
    if "type" in df.columns:
        _side = df["type"].astype(str).str.lower()
        df["_sign"] = np.where(
            _side.isin(("buy", "purchase")),
            1.0,
            np.where(_side.isin(("sell", "sale")), -1.0, 0.0),
        )
    else:
        df["_sign"] = 0.0
    df["_weight"] = 1.0
    if committee_members and "member_id" in df.columns:
        df["_weight"] = np.where(
            df["member_id"].isin(committee_members), committee_weight, 1.0
        )
    df["_net"] = df["_amount"] * df["_sign"] * df["_weight"]
    scores_series = df.groupby("symbol")["_net"].sum().round(2)
    return {str(sym): float(v) for sym, v in scores_series.items()}
