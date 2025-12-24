"""Insider trading features module (Phase 6 Skeleton).

This module provides functions to compute features from insider trading events.
Currently provides skeleton implementations with simple aggregation logic.

B2 Integration: This module now supports PIT-safe filtering using disclosure_date.
Events are filtered to only include those disclosed by as_of, preventing look-ahead bias.

Zukünftige Integration:
- Erweiterte Features: Insider buy/sell ratios, role-weighted signals, etc.
- Time-weighted features (recent trades more important)
- Cross-symbol correlations
"""

from __future__ import annotations

import pandas as pd

from src.assembled_core.data.latency import (
    apply_source_latency,
    ensure_event_schema,
    filter_events_as_of,
)


def add_insider_features(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    disclosure_latency_days: int = 2,
) -> pd.DataFrame:
    """Add insider trading features to price DataFrame (PIT-safe).

    Computes features like:
    - insider_net_buy_20d: Net shares bought (positive) or sold (negative) in last 20 days
    - insider_trade_count_20d: Number of insider trades in last 20 days
    - insider_net_buy_60d: Net shares bought/sold in last 60 days
    - insider_trade_count_60d: Number of insider trades in last 60 days

    B2 PIT Safety:
    - If as_of is provided, only events with disclosure_date <= as_of are used
    - If disclosure_date is missing, it is derived from timestamp + disclosure_latency_days
    - This ensures features are "blind" to events not yet disclosed

    Args:
        prices: DataFrame with columns: timestamp (UTC), symbol, close (and optionally other price columns)
        events: DataFrame from insider_ingest.load_insider_sample() with columns:
            timestamp, symbol, trades_count, net_shares, role
            Optional: event_date, disclosure_date (if missing, derived from timestamp)
        as_of: Optional point-in-time cutoff (pd.Timestamp, UTC)
            Only events with disclosure_date <= as_of are used. If None, all events are used.
        disclosure_latency_days: Number of days between event_date and disclosure_date (default: 2)
            Used if disclosure_date is missing (typical for Form 4 filings: T+2)

    Returns:
        Copy of prices DataFrame with additional columns:
        - insider_net_buy_20d: Net shares bought in last 20 days
        - insider_trade_count_20d: Trade count in last 20 days
        - insider_net_buy_60d: Net shares bought in last 60 days
        - insider_trade_count_60d: Trade count in last 60 days

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
        # Determine max as_of from prices (if as_of not provided, use max price timestamp)
        if isinstance(as_of, pd.Timestamp):
            events = filter_events_as_of(events, as_of, disclosure_col="disclosure_date")
        else:
            # If as_of is per-price-row, we filter per row (handled in loop below)
            pass

    # For per-row filtering (if as_of varies per price row), filter in the loop
    # For now, assume as_of is a single timestamp (or None)
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    # Ensure required event columns exist (with defaults if needed)
    if "net_shares" not in events.columns:
        events["net_shares"] = 0.0
    if "trades_count" not in events.columns:
        events["trades_count"] = 1

    # Initialize feature columns
    result["insider_net_buy_20d"] = 0.0
    result["insider_trade_count_20d"] = 0
    result["insider_net_buy_60d"] = 0.0
    result["insider_trade_count_60d"] = 0

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
            # For now, use pre-filtered events (as_of was single timestamp)
            row_events = symbol_events.copy()

            # Filter by disclosure_date <= price_time (PIT-safe)
            if "disclosure_date" in row_events.columns:
                row_events = row_events[
                    row_events["disclosure_date"] <= price_time.normalize()
                ].copy()

            # Use event_date or timestamp for window calculation
            window_time_col = "event_date" if "event_date" in row_events.columns else "timestamp"
            
            # 20-day window (based on event_date, but only disclosed events)
            window_20d = row_events[
                (row_events[window_time_col] <= price_time)
                & (row_events[window_time_col] > price_time - pd.Timedelta(days=20))
            ]
            result.loc[idx, "insider_net_buy_20d"] = window_20d["net_shares"].sum() if "net_shares" in window_20d.columns else 0.0
            result.loc[idx, "insider_trade_count_20d"] = len(window_20d)

            # 60-day window
            window_60d = row_events[
                (row_events[window_time_col] <= price_time)
                & (row_events[window_time_col] > price_time - pd.Timedelta(days=60))
            ]
            result.loc[idx, "insider_net_buy_60d"] = window_60d["net_shares"].sum() if "net_shares" in window_60d.columns else 0.0
            result.loc[idx, "insider_trade_count_60d"] = len(window_60d)

    return result
