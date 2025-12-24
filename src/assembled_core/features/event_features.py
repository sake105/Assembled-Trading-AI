"""Minimal event feature builder module (B2 Reference Implementation).

This module provides a minimal, PIT-safe event feature builder as a reference
implementation for B2 integration. It demonstrates how to use the latency helpers
for building event-based features with disclosure_date filtering.

This is a reference implementation for other event feature builders to follow.
"""

from __future__ import annotations

import pandas as pd

from src.assembled_core.data.latency import (
    apply_source_latency,
    ensure_event_schema,
    filter_events_as_of,
)


def build_event_feature_panel(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    lookback_days: int = 30,
    disclosure_latency_days: int = 2,
    feature_prefix: str = "event",
) -> pd.DataFrame:
    """Build event feature panel from events DataFrame (PIT-safe reference implementation).

    This function demonstrates the B2 pattern for building event-based features:
    1. Ensure event schema (validate required columns)
    2. Derive disclosure_date if missing (apply latency)
    3. Filter events by disclosure_date <= as_of (PIT-safe)
    4. Aggregate events into features

    Args:
        events_df: Event DataFrame with columns:
            - timestamp (required): Event timestamp
            - symbol (required): Symbol/ticker
            - event_date (optional): When event happened
            - disclosure_date (optional): When event becomes observable
            - value (optional): Event value for aggregation
        prices_df: Price DataFrame with columns: timestamp (UTC), symbol, close
            Features are joined to this DataFrame
        as_of: Optional point-in-time cutoff (pd.Timestamp, UTC)
            Only events with disclosure_date <= as_of are used. If None, all events are used.
        lookback_days: Number of days to look back for event aggregation (default: 30)
        disclosure_latency_days: Days between event_date and disclosure_date (default: 2)
            Used if disclosure_date is missing
        feature_prefix: Prefix for feature column names (default: "event")

    Returns:
        DataFrame with same rows as prices_df, plus additional columns:
        - {feature_prefix}_count_{lookback_days}d: Number of events in lookback window
        - {feature_prefix}_sum_{lookback_days}d: Sum of event values in lookback window
        - {feature_prefix}_mean_{lookback_days}d: Mean of event values in lookback window

    Raises:
        KeyError: If required columns are missing
    """
    # Validate inputs
    required_price_cols = ["timestamp", "symbol"]
    missing_price_cols = [c for c in required_price_cols if c not in prices_df.columns]
    if missing_price_cols:
        raise KeyError(f"Missing required columns in prices_df: {missing_price_cols}")

    result = prices_df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    # Step 1: Ensure event schema (validate required columns)
    events = ensure_event_schema(
        events_df, required_cols=["timestamp", "symbol"], strict=False
    )

    if events.empty:
        # Return prices with NaN features
        result[f"{feature_prefix}_count_{lookback_days}d"] = 0
        result[f"{feature_prefix}_sum_{lookback_days}d"] = 0.0
        result[f"{feature_prefix}_mean_{lookback_days}d"] = pd.NA
        return result

    # Step 2: Derive disclosure_date if missing (apply latency)
    if "disclosure_date" not in events.columns:
        events = apply_source_latency(
            events,
            days=disclosure_latency_days,
            event_date_col="event_date",
            timestamp_col="timestamp",
        )

    # Step 3: Filter events by disclosure_date <= as_of (PIT-safe)
    if as_of is not None:
        events = filter_events_as_of(events, as_of, disclosure_col="disclosure_date")

    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)

    # Step 4: Aggregate events into features (per symbol, per price timestamp)
    # Initialize feature columns
    result[f"{feature_prefix}_count_{lookback_days}d"] = 0
    result[f"{feature_prefix}_sum_{lookback_days}d"] = 0.0
    result[f"{feature_prefix}_mean_{lookback_days}d"] = pd.NA

    # Determine value column for aggregation
    value_col = "value" if "value" in events.columns else None
    if value_col is None:
        # If no value column, just count events
        value_col = None

    # Determine time column for window calculation (prefer event_date, fallback to timestamp)
    window_time_col = "event_date" if "event_date" in events.columns else "timestamp"

    # Group by symbol for efficient processing
    for symbol in result["symbol"].unique():
        symbol_mask = result["symbol"] == symbol
        symbol_prices = result[symbol_mask].copy()

        # Get events for this symbol (already PIT-filtered)
        symbol_events = events[events["symbol"] == symbol].copy()

        if symbol_events.empty:
            continue

        # For each price row, compute features based on events in rolling window
        for idx in symbol_prices.index:
            price_time = symbol_prices.loc[idx, "timestamp"]

            # Apply per-row PIT filtering (disclosure_date <= price_time)
            row_events = symbol_events[
                symbol_events["disclosure_date"] <= price_time.normalize()
            ].copy()

            # Filter by lookback window (event_date/timestamp <= price_time and > price_time - lookback)
            window_events = row_events[
                (row_events[window_time_col] <= price_time)
                & (row_events[window_time_col] > price_time - pd.Timedelta(days=lookback_days))
            ].copy()

            # Compute features
            result.loc[idx, f"{feature_prefix}_count_{lookback_days}d"] = len(window_events)

            if value_col and value_col in window_events.columns:
                result.loc[idx, f"{feature_prefix}_sum_{lookback_days}d"] = window_events[
                    value_col
                ].sum()
                result.loc[idx, f"{feature_prefix}_mean_{lookback_days}d"] = window_events[
                    value_col
                ].mean()
            else:
                result.loc[idx, f"{feature_prefix}_sum_{lookback_days}d"] = 0.0
                result.loc[idx, f"{feature_prefix}_mean_{lookback_days}d"] = pd.NA

    return result

