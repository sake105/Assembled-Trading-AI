"""Minimal event feature builder module (B2 Reference Implementation, Sprint 10.B).

This module provides PIT-safe event feature builders using the standardized
Alt-Data Event Contract (Sprint 10.A). All features respect disclosure_date
filtering to prevent look-ahead bias.

This is a reference implementation for other event feature builders to follow.

Sprint 11.E1: Added vectorized implementation option via method parameter.
"""

from __future__ import annotations

import pandas as pd

from src.assembled_core.data.altdata.contract import (
    filter_events_pit,
    normalize_alt_events,
)

# Import vectorized implementations (Sprint 11.E1)
try:
    from src.assembled_core.features.event_features_vectorized import (
        add_disclosure_count_feature_vectorized,
        build_event_feature_panel_vectorized,
    )
except ImportError:
    # Fallback if vectorized module not available
    add_disclosure_count_feature_vectorized = None
    build_event_feature_panel_vectorized = None


def build_event_feature_panel(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    lookback_days: int = 30,
    feature_prefix: str = "event",
    method: str = "legacy",
) -> pd.DataFrame:
    """Build event feature panel from events DataFrame (PIT-safe, Sprint 10.B).

    This function uses the standardized Alt-Data Event Contract (Sprint 10.A):
    1. Normalize events to contract schema (normalize_alt_events)
    2. Filter events by disclosure_date <= as_of (filter_events_pit, PIT-safe)
    3. Aggregate events into features (based on disclosure_date window)

    Args:
        events_df: Event DataFrame (must have symbol, event_date, disclosure_date)
            Optional: effective_date, event_type, source, value
        prices_df: Price DataFrame with columns: timestamp (UTC), symbol, close
            Features are joined to this DataFrame
        as_of: Point-in-time cutoff (pd.Timestamp, UTC, required)
            Only events with disclosure_date <= as_of are used.
        lookback_days: Number of days to look back for event aggregation (default: 30)
            Window is based on disclosure_date (not event_date)
        feature_prefix: Prefix for feature column names (default: "event")
        method: Implementation method (default: "legacy")
            - "legacy": Original nested loop implementation (Sprint 10.B)
            - "vectorized": Vectorized implementation using merge_asof + rolling (Sprint 11.E1)

    Returns:
        DataFrame with same rows as prices_df, plus additional columns:
        - {feature_prefix}_count_{lookback_days}d: Number of events in lookback window
        - {feature_prefix}_sum_{lookback_days}d: Sum of event values in lookback window
        - {feature_prefix}_mean_{lookback_days}d: Mean of event values in lookback window

    Raises:
        ValueError: If required columns are missing or as_of is None
    """
    if as_of is None:
        raise ValueError("as_of is required for PIT-safe feature building")

    # Route to vectorized implementation if requested (Sprint 11.E1)
    if method == "vectorized":
        if build_event_feature_panel_vectorized is None:
            raise ImportError(
                "Vectorized implementation not available. "
                "Use method='legacy' or ensure event_features_vectorized module is available."
            )
        return build_event_feature_panel_vectorized(
            events_df, prices_df, as_of, lookback_days, feature_prefix
        )

    # Legacy implementation (default)

    # Validate inputs
    required_price_cols = ["timestamp", "symbol"]
    missing_price_cols = [c for c in required_price_cols if c not in prices_df.columns]
    if missing_price_cols:
        raise KeyError(f"Missing required columns in prices_df: {missing_price_cols}")

    result = prices_df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    # Step 1: Normalize events to contract schema
    try:
        events = normalize_alt_events(events_df)
    except ValueError:
        # If normalization fails, return prices with zero features
        result[f"{feature_prefix}_count_{lookback_days}d"] = 0
        result[f"{feature_prefix}_sum_{lookback_days}d"] = 0.0
        result[f"{feature_prefix}_mean_{lookback_days}d"] = pd.NA
        return result

    if events.empty:
        # Return prices with zero features
        result[f"{feature_prefix}_count_{lookback_days}d"] = 0
        result[f"{feature_prefix}_sum_{lookback_days}d"] = 0.0
        result[f"{feature_prefix}_mean_{lookback_days}d"] = pd.NA
        return result

    # Step 2: Filter events by disclosure_date <= as_of (PIT-safe)
    events = filter_events_pit(events, as_of)

    # Step 3: Aggregate events into features (per symbol, per price timestamp)
    # Initialize feature columns
    result[f"{feature_prefix}_count_{lookback_days}d"] = 0
    result[f"{feature_prefix}_sum_{lookback_days}d"] = 0.0
    result[f"{feature_prefix}_mean_{lookback_days}d"] = pd.NA

    # Determine value column for aggregation
    value_col = "value" if "value" in events.columns else None

    # Group by symbol for efficient processing
    for symbol in result["symbol"].unique():
        symbol_mask = result["symbol"] == symbol
        symbol_prices = result[symbol_mask].copy()

        # Get events for this symbol (already PIT-filtered globally)
        symbol_events = events[events["symbol"] == symbol].copy()

        if symbol_events.empty:
            continue

        # For each price row, compute features based on events in rolling window
        # Window is based on disclosure_date (PIT-safe)
        for idx in symbol_prices.index:
            price_time = symbol_prices.loc[idx, "timestamp"]
            price_time_normalized = price_time.normalize()

            # Apply per-row PIT filtering (disclosure_date <= price_time)
            # This is redundant if as_of is set correctly, but ensures safety
            row_events = symbol_events[
                symbol_events["disclosure_date"] <= price_time_normalized
            ].copy()

            # Filter by lookback window (disclosure_date in [price_time - lookback, price_time])
            window_events = row_events[
                (row_events["disclosure_date"] <= price_time_normalized)
                & (row_events["disclosure_date"] > price_time_normalized - pd.Timedelta(days=lookback_days))
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


def add_disclosure_count_feature(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    window_days: int = 30,
    out_col: str = "alt_disclosure_count_30d_v1",
    as_of: pd.Timestamp | None = None,
    method: str = "legacy",
) -> pd.DataFrame:
    """Add disclosure count feature to prices DataFrame (PIT-safe, Sprint 10.B).

    For each (symbol, timestamp) in prices, counts events whose disclosure_date
    falls within the window [timestamp - window_days, timestamp].

    This feature is PIT-safe: only events with disclosure_date <= as_of (or price timestamp)
    are counted, preventing look-ahead bias.

    Args:
        prices: Price DataFrame with columns: timestamp (UTC), symbol, ...
        events: Event DataFrame (must have symbol, event_date, disclosure_date)
            Will be normalized to Alt-Data Event Contract
        window_days: Lookback window in days (default: 30)
        out_col: Output column name (default: "alt_disclosure_count_30d_v1")
        as_of: Optional point-in-time cutoff (pd.Timestamp, UTC)
            If provided, events are filtered globally. Otherwise, per-row filtering
            is applied (disclosure_date <= price timestamp)
        method: Implementation method (default: "legacy")
            - "legacy": Original nested loop implementation (Sprint 10.B)
            - "vectorized": Vectorized implementation using cross-join + filtering (Sprint 11.E1)

    Returns:
        DataFrame with additional column {out_col} containing event counts

    Raises:
        ValueError: If required columns are missing
    """
    # Route to vectorized implementation if requested (Sprint 11.E1)
    if method == "vectorized":
        if add_disclosure_count_feature_vectorized is None:
            raise ImportError(
                "Vectorized implementation not available. "
                "Use method='legacy' or ensure event_features_vectorized module is available."
            )
        return add_disclosure_count_feature_vectorized(
            prices, events, window_days=window_days, out_col=out_col, as_of=as_of
        )

    # Legacy implementation (default)
    result = prices.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    # Normalize events to contract schema
    try:
        events_normalized = normalize_alt_events(events)
    except ValueError:
        # If normalization fails, return prices with zero feature
        result[out_col] = 0
        return result

    if events_normalized.empty:
        result[out_col] = 0
        return result

    # Initialize feature column
    result[out_col] = 0

    # If as_of is provided, filter globally (more efficient)
    if as_of is not None:
        events_normalized = filter_events_pit(events_normalized, as_of)

    # Group by symbol for efficient processing
    for symbol in result["symbol"].unique():
        symbol_mask = result["symbol"] == symbol
        symbol_prices = result[symbol_mask].copy()

        # Get events for this symbol
        symbol_events = events_normalized[events_normalized["symbol"] == symbol].copy()

        if symbol_events.empty:
            continue

        # For each price row, count events in disclosure_date window
        for idx in symbol_prices.index:
            price_time = symbol_prices.loc[idx, "timestamp"]
            price_time_normalized = price_time.normalize()

            # Apply per-row PIT filtering (disclosure_date <= price_time)
            if as_of is None:
                row_events = symbol_events[
                    symbol_events["disclosure_date"] <= price_time_normalized
                ].copy()
            else:
                # Already filtered globally
                row_events = symbol_events.copy()

            # Filter by lookback window (disclosure_date in [price_time - window, price_time])
            window_events = row_events[
                (row_events["disclosure_date"] <= price_time_normalized)
                & (row_events["disclosure_date"] > price_time_normalized - pd.Timedelta(days=window_days))
            ].copy()

            # Count events
            result.loc[idx, out_col] = len(window_events)

    return result

