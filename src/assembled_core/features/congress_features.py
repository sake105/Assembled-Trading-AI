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
    disclosure_latency_days: int = 10,
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
        disclosure_latency_days: Number of days between event_date and disclosure_date (default: 10)
            Used if disclosure_date is missing (typical for Congress PTR: 10-30 days delay)

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
            events = filter_events_as_of(events, as_of, disclosure_col="disclosure_date")

    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    # Initialize feature columns
    result["congress_trade_count_60d"] = 0
    result["congress_total_amount_60d"] = 0.0
    result["congress_trade_count_90d"] = 0
    result["congress_total_amount_90d"] = 0.0

    # Group by symbol for efficient processing
    for symbol in result["symbol"].unique():
        symbol_mask = result["symbol"] == symbol
        symbol_prices = result[symbol_mask].copy()

        # Get events for this symbol
        symbol_events = events[events["symbol"] == symbol].copy()

        if symbol_events.empty:
            continue

        # For each price row, compute features based on events in rolling windows
        # PIT: Use disclosure_date for filtering, but event_date/timestamp for window calculation
        for idx in symbol_prices.index:
            price_time = symbol_prices.loc[idx, "timestamp"]

            # Apply per-row PIT filtering (if as_of is per-row)
            row_events = symbol_events.copy()

            # Filter by disclosure_date <= price_time (PIT-safe)
            if "disclosure_date" in row_events.columns:
                row_events = row_events[
                    row_events["disclosure_date"] <= price_time.normalize()
                ].copy()

            # Use event_date or timestamp for window calculation
            window_time_col = "event_date" if "event_date" in row_events.columns else "timestamp"

            # 60-day window (based on event_date, but only disclosed events)
            window_60d = row_events[
                (row_events[window_time_col] <= price_time)
                & (row_events[window_time_col] > price_time - pd.Timedelta(days=60))
            ]
            result.loc[idx, "congress_trade_count_60d"] = len(window_60d)
            result.loc[idx, "congress_total_amount_60d"] = window_60d["amount"].sum() if "amount" in window_60d.columns and len(window_60d) > 0 else 0.0

            # 90-day window
            window_90d = row_events[
                (row_events[window_time_col] <= price_time)
                & (row_events[window_time_col] > price_time - pd.Timedelta(days=90))
            ]
            result.loc[idx, "congress_trade_count_90d"] = len(window_90d)
            result.loc[idx, "congress_total_amount_90d"] = window_90d["amount"].sum() if "amount" in window_90d.columns and len(window_90d) > 0 else 0.0

    return result
