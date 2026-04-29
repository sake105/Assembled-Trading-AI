"""Vectorized event feature builder module (Sprint 11.E1).

This module provides vectorized implementations of event feature builders using
pandas merge_asof and cumulative counting instead of nested loops.

Key Functions:
    - build_event_feature_panel_vectorized(): Vectorized version of build_event_feature_panel
    - add_disclosure_count_feature_vectorized(): Vectorized version of add_disclosure_count_feature

Design Principles:
    - PIT-safe: Uses disclosure_date filtering (same as legacy)
    - Deterministic: Same input -> same output (explicit sorting)
    - Vectorized: O(N log N) per symbol using merge_asof
    - Compatible: Same output schema as legacy implementation
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.assembled_core.data.altdata.contract import (
    filter_events_pit,
    normalize_alt_events,
)


def build_event_feature_panel_vectorized(
    events_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int = 30,
    feature_prefix: str = "event",
) -> pd.DataFrame:
    """Build event feature panel using vectorized operations (Sprint 11.E1).

    This is a vectorized implementation of build_event_feature_panel that uses:
    1. Per-row PIT filtering (disclosure_date <= row.timestamp)
    2. Cumulative counting via merge_asof (O(N log N) per symbol)
    3. Window subtraction (cum_at_t - cum_at_start)

    Args:
        events_df: Event DataFrame (must have symbol, event_date, disclosure_date)
            Optional: effective_date, event_type, source, value
        prices_df: Price DataFrame with columns: timestamp (UTC), symbol, close
        as_of: Point-in-time cutoff (pd.Timestamp, UTC, required)
        lookback_days: Number of days to look back for event aggregation (default: 30)
        feature_prefix: Prefix for feature column names (default: "event")

    Returns:
        DataFrame with same rows as prices_df, plus additional columns:
        - {feature_prefix}_count_{lookback_days}d: Number of events in lookback window
        - {feature_prefix}_sum_{lookback_days}d: Sum of event values in lookback window
        - {feature_prefix}_mean_{lookback_days}d: Mean of event values in lookback window

    Raises:
        ValueError: If required columns are missing
    """
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
        result[f"{feature_prefix}_count_{lookback_days}d"] = 0.0
        result[f"{feature_prefix}_sum_{lookback_days}d"] = 0.0
        result[f"{feature_prefix}_mean_{lookback_days}d"] = np.full(len(result), np.nan)
        return result

    if events.empty:
        # Return prices with zero features
        result[f"{feature_prefix}_count_{lookback_days}d"] = 0.0
        result[f"{feature_prefix}_sum_{lookback_days}d"] = 0.0
        result[f"{feature_prefix}_mean_{lookback_days}d"] = np.full(len(result), np.nan)
        return result

    # Step 2: Filter events by disclosure_date <= as_of (PIT-safe, global).
    # latency_days=0: vectorized mirror of event_features.build_event_feature_panel;
    # the Alt-Data Event Contract already embeds vendor-side latency in
    # disclosure_date. P0 A5 (Deep Run v2, 2026-04-18).
    events = filter_events_pit(events, as_of, latency_days=0)

    # Step 3: Initialize feature columns (use float to avoid LossySetitemError with NaN)
    result[f"{feature_prefix}_count_{lookback_days}d"] = 0.0
    result[f"{feature_prefix}_sum_{lookback_days}d"] = 0.0
    result[f"{feature_prefix}_mean_{lookback_days}d"] = np.full(len(result), np.nan)

    # Determine value column for aggregation
    value_col = "value" if "value" in events.columns else None

    # Step 4: Process per symbol (O(N log N) per symbol)
    for symbol in result["symbol"].unique():
        symbol_mask = result["symbol"] == symbol
        symbol_prices = result[symbol_mask].copy()

        # Get events for this symbol (already PIT-filtered globally)
        symbol_events = events[events["symbol"] == symbol].copy()

        if symbol_events.empty:
            continue

        # Compute features for this symbol using vectorized approach
        features = _compute_features_for_symbol_vectorized(
            symbol_prices,
            symbol_events,
            lookback_days,
            feature_prefix,
            value_col,
        )

        # Assign features back to result (convert to float to avoid LossySetitemError)
        result.loc[symbol_mask, f"{feature_prefix}_count_{lookback_days}d"] = features[
            "count"
        ].values.astype(np.float64)
        result.loc[symbol_mask, f"{feature_prefix}_sum_{lookback_days}d"] = features[
            "sum"
        ].values.astype(np.float64)
        result.loc[symbol_mask, f"{feature_prefix}_mean_{lookback_days}d"] = (
            pd.to_numeric(features["mean"], errors="coerce").values
        )

    # Final deterministic sort (same as legacy)
    result = result.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )

    return result


def _compute_features_for_symbol_vectorized(
    prices_group: pd.DataFrame,
    events: pd.DataFrame,
    lookback_days: int,
    feature_prefix: str,
    value_col: str | None,
) -> pd.DataFrame:
    """Compute features for a single symbol using vectorized merge_asof approach.

    This implements the exact Legacy semantics:
    - Per-row PIT filtering: disclosure_date <= price_time_normalized
    - Window: disclosure_date > price_time_normalized - lookback_days (strict >)
    - Uses cumulative counting via merge_asof for O(N log N) performance
    """
    # Normalize timestamps (same as legacy: price_time.normalize())
    prices_sorted = prices_group.sort_values("timestamp", kind="mergesort").reset_index(
        drop=True
    )
    prices_sorted["timestamp_normalized"] = prices_sorted["timestamp"].dt.normalize()
    prices_sorted["_row_id"] = range(len(prices_sorted))

    # Prepare events: normalize disclosure_date and sort
    events_sorted = events.copy()
    events_sorted["disclosure_date_normalized"] = events_sorted[
        "disclosure_date"
    ].dt.normalize()
    events_sorted = events_sorted.sort_values(
        "disclosure_date_normalized", kind="mergesort"
    ).reset_index(drop=True)

    # Build cumulative counts/sums per disclosure_date
    # Group by disclosure_date to handle multiple events on same day
    if value_col and value_col in events_sorted.columns:
        events_cum = (
            events_sorted.groupby("disclosure_date_normalized", sort=False)
            .agg(
                {
                    "disclosure_date": "count",  # Count events per day
                    value_col: "sum",  # Sum of values per day
                }
            )
            .reset_index()
        )
        events_cum.columns = ["disclosure_date_normalized", "event_count", "value_sum"]
    else:
        events_cum = (
            events_sorted.groupby("disclosure_date_normalized", sort=False)
            .agg(
                {
                    "disclosure_date": "count",
                }
            )
            .reset_index()
        )
        events_cum.columns = ["disclosure_date_normalized", "event_count"]
        events_cum["value_sum"] = 0.0

    # Compute cumulative sums (cumulative count/sum up to each disclosure_date)
    events_cum["cum_count"] = events_cum["event_count"].cumsum()
    events_cum["cum_sum"] = events_cum["value_sum"].cumsum()

    # For each price timestamp, find cumulative count/sum at that time
    # Using merge_asof with direction="backward" (disclosure_date <= timestamp)
    merged_at_t = pd.merge_asof(
        prices_sorted[["timestamp_normalized", "_row_id"]],
        events_cum[["disclosure_date_normalized", "cum_count", "cum_sum"]],
        left_on="timestamp_normalized",
        right_on="disclosure_date_normalized",
        direction="backward",
        allow_exact_matches=True,
    )

    # For window start (price_time - lookback_days), find cumulative count/sum
    # Legacy uses strict >, so we need events <= (timestamp - lookback_days)
    prices_sorted["window_start"] = prices_sorted[
        "timestamp_normalized"
    ] - pd.Timedelta(days=lookback_days)
    merged_at_start = pd.merge_asof(
        prices_sorted[["window_start", "_row_id"]],
        events_cum[["disclosure_date_normalized", "cum_count", "cum_sum"]],
        left_on="window_start",
        right_on="disclosure_date_normalized",
        direction="backward",
        allow_exact_matches=True,
    )

    # Compute window features: cum_at_t - cum_at_start
    # Handle NaNs (no events before timestamp or window_start)
    features = pd.DataFrame(
        {
            "_row_id": prices_sorted["_row_id"],
            "count": (
                merged_at_t["cum_count"].fillna(0)
                - merged_at_start["cum_count"].fillna(0)
            ).astype(int),
            "sum": (
                merged_at_t["cum_sum"].fillna(0.0)
                - merged_at_start["cum_sum"].fillna(0.0)
            ),
        }
    )

    # Compute mean: sum / count (if count > 0)
    # Legacy computes mean from window_events[value_col].mean()
    # This is equivalent to sum / count (mean of values in window)
    if value_col and value_col in events_sorted.columns:
        # Mean = sum / count (if count > 0)
        # Use np.nan instead of pd.NA to match legacy dtype (object with NaN)
        features["mean"] = features["sum"] / features["count"].replace(0, np.nan)
        features["mean"] = features["mean"].where(features["count"] > 0, np.nan)
        # Ensure float64 dtype for consistent NaN handling (not pd.NA)
        features["mean"] = pd.to_numeric(features["mean"], errors="coerce").astype(
            "float64"
        )
    else:
        features["mean"] = pd.array([np.nan] * len(features), dtype="float64")

    # Ensure count >= 0 (should be, but safety check)
    features["count"] = features["count"].clip(lower=0)

    # Restore original order via _row_id
    features = features.sort_values("_row_id", kind="mergesort").reset_index(drop=True)

    return features


def add_disclosure_count_feature_vectorized(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    *,
    window_days: int = 30,
    out_col: str = "alt_disclosure_count_30d_v1",
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Add disclosure count feature using vectorized operations (Sprint 11.E1).

    This is a vectorized implementation of add_disclosure_count_feature that uses
    merge_asof + cumulative counting instead of nested loops.

    Args:
        prices: Price DataFrame with columns: timestamp (UTC), symbol, ...
        events: Event DataFrame (must have symbol, event_date, disclosure_date)
        window_days: Lookback window in days (default: 30)
        out_col: Output column name (default: "alt_disclosure_count_30d_v1")
        as_of: Optional point-in-time cutoff (pd.Timestamp, UTC)
            If provided, events are filtered globally. Otherwise, per-row filtering
            is applied (disclosure_date <= price timestamp)

    Returns:
        DataFrame with additional column {out_col} containing event counts

    Raises:
        ValueError: If required columns are missing
    """
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

    # If as_of is provided, filter globally (more efficient).
    # latency_days=0: generic Alt-Data Event Contract. P0 A5 (Deep Run v2).
    if as_of is not None:
        events_normalized = filter_events_pit(events_normalized, as_of, latency_days=0)

    # Initialize feature column
    result[out_col] = 0

    # Process per symbol
    for symbol in result["symbol"].unique():
        symbol_mask = result["symbol"] == symbol
        symbol_prices = result[symbol_mask].copy()

        # Get events for this symbol
        symbol_events = events_normalized[events_normalized["symbol"] == symbol].copy()

        if symbol_events.empty:
            continue

        # Compute count for this symbol using vectorized approach
        counts = _compute_count_for_symbol_vectorized(
            symbol_prices,
            symbol_events,
            window_days,
            as_of,
        )

        # Assign counts back to result
        result.loc[symbol_mask, out_col] = counts.values

    # Final deterministic sort (same as legacy)
    result = result.sort_values(["symbol", "timestamp"], kind="mergesort").reset_index(
        drop=True
    )

    return result


def _compute_count_for_symbol_vectorized(
    prices_group: pd.DataFrame,
    events: pd.DataFrame,
    window_days: int,
    as_of: pd.Timestamp | None,
) -> pd.Series:
    """Compute count feature for a single symbol using vectorized merge_asof approach.

    This implements the exact Legacy semantics:
    - Per-row PIT filtering (if as_of is None): disclosure_date <= price_time_normalized
    - Window: disclosure_date > price_time_normalized - window_days (strict >)
    - Uses cumulative counting via merge_asof for O(N log N) performance
    """
    # Normalize timestamps (same as legacy: price_time.normalize())
    prices_sorted = prices_group.sort_values("timestamp", kind="mergesort").reset_index(
        drop=True
    )
    prices_sorted["timestamp_normalized"] = prices_sorted["timestamp"].dt.normalize()
    prices_sorted["_row_id"] = range(len(prices_sorted))

    # Prepare events: normalize disclosure_date and sort
    events_sorted = events.copy()
    events_sorted["disclosure_date_normalized"] = events_sorted[
        "disclosure_date"
    ].dt.normalize()
    events_sorted = events_sorted.sort_values(
        "disclosure_date_normalized", kind="mergesort"
    ).reset_index(drop=True)

    # If as_of is None, we need per-row PIT filtering
    # This means we can't pre-filter events globally
    # Instead, we'll filter during merge_asof by only including events <= timestamp
    # But merge_asof already does this with direction="backward"
    # So we just need to ensure events are sorted and use merge_asof correctly

    # Build cumulative counts per disclosure_date
    events_cum = (
        events_sorted.groupby("disclosure_date_normalized", sort=False)
        .agg(
            {
                "disclosure_date": "count",
            }
        )
        .reset_index()
    )
    events_cum.columns = ["disclosure_date_normalized", "event_count"]

    # Compute cumulative sum
    events_cum["cum_count"] = events_cum["event_count"].cumsum()

    # For each price timestamp, find cumulative count at that time
    # Using merge_asof with direction="backward" (disclosure_date <= timestamp)
    # If as_of is None, this effectively does per-row PIT filtering
    merged_at_t = pd.merge_asof(
        prices_sorted[["timestamp_normalized", "_row_id"]],
        events_cum[["disclosure_date_normalized", "cum_count"]],
        left_on="timestamp_normalized",
        right_on="disclosure_date_normalized",
        direction="backward",
        allow_exact_matches=True,
    )

    # For window start (price_time - window_days), find cumulative count
    # Legacy uses strict >, so we need events <= (timestamp - window_days)
    prices_sorted["window_start"] = prices_sorted[
        "timestamp_normalized"
    ] - pd.Timedelta(days=window_days)
    merged_at_start = pd.merge_asof(
        prices_sorted[["window_start", "_row_id"]],
        events_cum[["disclosure_date_normalized", "cum_count"]],
        left_on="window_start",
        right_on="disclosure_date_normalized",
        direction="backward",
        allow_exact_matches=True,
    )

    # Compute window count: cum_at_t - cum_at_start
    counts = (
        merged_at_t["cum_count"].fillna(0) - merged_at_start["cum_count"].fillna(0)
    ).astype(int)

    # Ensure count >= 0
    counts = counts.clip(lower=0)

    # Restore original order via _row_id
    counts_df = pd.DataFrame(
        {
            "_row_id": prices_sorted["_row_id"],
            "count": counts,
        }
    )
    counts_df = counts_df.sort_values("_row_id", kind="mergesort").reset_index(
        drop=True
    )

    return counts_df["count"]
