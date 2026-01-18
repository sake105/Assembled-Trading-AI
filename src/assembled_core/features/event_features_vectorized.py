"""Vectorized event feature builder module (Sprint 11.E1).

This module provides vectorized implementations of event feature builders using
pandas merge_asof and rolling window operations instead of nested loops.

Key Functions:
    - build_event_feature_panel_vectorized(): Vectorized version of build_event_feature_panel
    - add_disclosure_count_feature_vectorized(): Vectorized version of add_disclosure_count_feature

Design Principles:
    - PIT-safe: Uses disclosure_date filtering (same as legacy)
    - Deterministic: Same input -> same output (explicit sorting)
    - Vectorized: No Python loops, uses pandas operations
    - Compatible: Same output schema as legacy implementation
"""

from __future__ import annotations

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
    1. Daily aggregation of events (per symbol, per disclosure_date)
    2. merge_asof to join events to prices (disclosure_date <= timestamp)
    3. Rolling window statistics (count, sum, mean) via groupby+rolling

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

    # Step 3: Daily aggregation (per symbol, per disclosure_date)
    # Aggregate events by (symbol, disclosure_date) to get daily counts/sums
    agg_dict = {}
    if "value" in events.columns:
        agg_dict["value"] = ["count", "sum", "mean"]
    else:
        # If no value column, just count events
        agg_dict["event_date"] = "count"  # Use any column for count

    events_daily = events.groupby(["symbol", "disclosure_date"]).agg(agg_dict).reset_index()

    # Flatten column names if multi-level
    if isinstance(events_daily.columns, pd.MultiIndex):
        events_daily.columns = ["_".join(col).strip("_") if col[1] else col[0] for col in events_daily.columns.values]
    else:
        # Single-level columns
        if "value" not in events.columns:
            # Rename count column
            events_daily = events_daily.rename(columns={"event_date": "value_count"})

    # Ensure we have the columns we need
    if "value" in events.columns:
        # Rename aggregated columns
        events_daily = events_daily.rename(columns={
            "value_count": "event_count",
            "value_sum": "event_sum",
            "value_mean": "event_mean",
        })
    else:
        # Only count available
        events_daily = events_daily.rename(columns={"value_count": "event_count"})
        events_daily["event_sum"] = 0.0
        events_daily["event_mean"] = pd.NA

    # Step 4: Sort prices for processing
    result_sorted = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Step 5: Note: merge_asof could be used here, but we use cross-join approach
    # in _compute_features_for_symbol for better window handling

    # Step 6: Compute rolling window statistics using vectorized operations
    # Use groupby + apply with vectorized helper function
    # Initialize feature columns first
    result_sorted[f"{feature_prefix}_count_{lookback_days}d"] = 0
    result_sorted[f"{feature_prefix}_sum_{lookback_days}d"] = 0.0
    result_sorted[f"{feature_prefix}_mean_{lookback_days}d"] = pd.NA

    # Store original sorted index
    original_sorted_index = result_sorted.index.copy()

    # Compute features per symbol group
    # group_keys=False ensures index is preserved within groups
    result_grouped = result_sorted.groupby("symbol", group_keys=False).apply(
        lambda g: _compute_features_for_symbol(
            g, events, lookback_days, feature_prefix, as_of
        )
    )

    # result_grouped should have same index as result_sorted (group_keys=False)
    # Reindex to ensure alignment
    result_grouped = result_grouped.reindex(original_sorted_index)

    # Copy feature columns directly
    result_sorted[f"{feature_prefix}_count_{lookback_days}d"] = result_grouped[f"{feature_prefix}_count_{lookback_days}d"].fillna(0).astype(int)
    result_sorted[f"{feature_prefix}_sum_{lookback_days}d"] = result_grouped[f"{feature_prefix}_sum_{lookback_days}d"].fillna(0.0)
    result_sorted[f"{feature_prefix}_mean_{lookback_days}d"] = result_grouped[f"{feature_prefix}_mean_{lookback_days}d"]

    # Step 7: Merge back to original result to preserve original index order
    # Create mapping from original index to sorted index
    result_original = result.copy()
    result_original["_sort_key"] = result_original["symbol"].astype(str) + "_" + result_original["timestamp"].astype(str)
    result_sorted["_sort_key"] = result_sorted["symbol"].astype(str) + "_" + result_sorted["timestamp"].astype(str)

    # Merge on sort_key to align
    result = result_original.merge(
        result_sorted[["_sort_key", f"{feature_prefix}_count_{lookback_days}d",
                       f"{feature_prefix}_sum_{lookback_days}d",
                       f"{feature_prefix}_mean_{lookback_days}d"]],
        on="_sort_key",
        how="left",
        suffixes=("", "_new"),
    )

    # Use new columns, fill missing with defaults
    count_col = f"{feature_prefix}_count_{lookback_days}d_new"
    sum_col = f"{feature_prefix}_sum_{lookback_days}d_new"
    mean_col = f"{feature_prefix}_mean_{lookback_days}d_new"

    if count_col in result.columns:
        result[f"{feature_prefix}_count_{lookback_days}d"] = result[count_col].fillna(0).astype(int)
    else:
        result[f"{feature_prefix}_count_{lookback_days}d"] = 0

    if sum_col in result.columns:
        result[f"{feature_prefix}_sum_{lookback_days}d"] = result[sum_col].fillna(0.0)
    else:
        result[f"{feature_prefix}_sum_{lookback_days}d"] = 0.0

    if mean_col in result.columns:
        result[f"{feature_prefix}_mean_{lookback_days}d"] = result[mean_col]
    else:
        result[f"{feature_prefix}_mean_{lookback_days}d"] = pd.NA

    # Drop temporary columns
    result = result.drop(columns=[col for col in result.columns if col.endswith("_new") or col == "_sort_key"], errors="ignore")

    # Final deterministic sort
    result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    return result


def _compute_features_for_symbol(
    prices_group: pd.DataFrame,
    events: pd.DataFrame,
    lookback_days: int,
    feature_prefix: str,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Compute features for a single symbol using vectorized operations."""
    symbol = prices_group["symbol"].iloc[0]
    symbol_events = events[events["symbol"] == symbol].copy()

    # Store original index to preserve row order
    original_index = prices_group.index.copy()

    if symbol_events.empty:
        prices_group[f"{feature_prefix}_count_{lookback_days}d"] = 0
        prices_group[f"{feature_prefix}_sum_{lookback_days}d"] = 0.0
        prices_group[f"{feature_prefix}_mean_{lookback_days}d"] = pd.NA
        return prices_group

    # For each price timestamp, we need to count events in window
    # Use vectorized approach: cross-join prices with events, then filter
    prices_expanded = prices_group[["timestamp", "symbol"]].copy()
    prices_expanded["timestamp_normalized"] = prices_expanded["timestamp"].dt.normalize()
    # Add row identifier to preserve original row order
    if "_row_id" in prices_group.columns:
        prices_expanded["_row_id"] = prices_group["_row_id"]
    else:
        prices_expanded["_row_id"] = prices_expanded.index

    # Cross join with events (vectorized)
    events_for_symbol = symbol_events[["disclosure_date", "value"]].copy() if "value" in symbol_events.columns else symbol_events[["disclosure_date"]].copy()
    events_for_symbol["disclosure_date_normalized"] = events_for_symbol["disclosure_date"].dt.normalize()

    # Use merge to create all combinations, then filter
    cross = prices_expanded.assign(key=1).merge(
        events_for_symbol.assign(key=1),
        on="key",
        suffixes=("_price", "_event"),
    ).drop("key", axis=1)

    # Vectorized filtering: disclosure_date <= timestamp AND in window
    mask = (
        (cross["disclosure_date_normalized"] <= cross["timestamp_normalized"])
        & (cross["disclosure_date_normalized"] > cross["timestamp_normalized"] - pd.Timedelta(days=lookback_days))
    )

    # Aggregate per row_id (not timestamp, to handle duplicate timestamps)
    if "value" in events_for_symbol.columns:
        features = cross[mask].groupby("_row_id").agg({
            "disclosure_date": "count",
            "value": ["sum", "mean"],
        })
        if isinstance(features.columns, pd.MultiIndex):
            features.columns = ["count", "sum", "mean"]
        else:
            features.columns = ["count", "sum", "mean"]
        features = features.reset_index()
    else:
        features = cross[mask].groupby("_row_id").agg({
            "disclosure_date": "count",
        }).reset_index()
        features.columns = ["_row_id", "count"]
        features["sum"] = 0.0
        features["mean"] = pd.NA

    # Merge back to prices_group using row_id
    # First, ensure prices_group has _row_id column
    if "_row_id" not in prices_group.columns:
        prices_group["_row_id"] = prices_group.index

    prices_group = prices_group.merge(
        features,
        on="_row_id",
        how="left",
        suffixes=("", "_feature"),
    )

    # Fill missing values
    prices_group[f"{feature_prefix}_count_{lookback_days}d"] = prices_group["count"].fillna(0).astype(int)
    if "value" in events_for_symbol.columns:
        prices_group[f"{feature_prefix}_sum_{lookback_days}d"] = prices_group["sum"].fillna(0.0)
        prices_group[f"{feature_prefix}_mean_{lookback_days}d"] = prices_group["mean"]
    else:
        prices_group[f"{feature_prefix}_sum_{lookback_days}d"] = 0.0
        prices_group[f"{feature_prefix}_mean_{lookback_days}d"] = pd.NA

    # Drop temporary columns
    prices_group = prices_group.drop(columns=["count", "sum", "mean", "_row_id"], errors="ignore")

    # Restore original index
    prices_group.index = original_index

    return prices_group


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
    cross-join + vectorized filtering instead of nested loops.

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

    # If as_of is provided, filter globally (more efficient)
    if as_of is not None:
        events_normalized = filter_events_pit(events_normalized, as_of)

    # Group by symbol and compute features
    result_grouped = result.sort_values(["symbol", "timestamp"]).groupby("symbol", group_keys=False).apply(
        lambda g: _compute_count_for_symbol(
            g, events_normalized, window_days, out_col, as_of
        )
    )

    # Merge back to original result
    result = result.merge(
        result_grouped[[out_col]],
        left_index=True,
        right_index=True,
        how="left",
        suffixes=("", "_new"),
    )

    # Use new column, fill missing with 0
    new_col = f"{out_col}_new"
    if new_col in result.columns:
        result[out_col] = result[new_col].fillna(0).astype(int)
    else:
        result[out_col] = 0

    # Drop temporary column
    result = result.drop(columns=[f"{out_col}_new"], errors="ignore")

    # Final deterministic sort
    result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    return result


def _compute_count_for_symbol(
    prices_group: pd.DataFrame,
    events: pd.DataFrame,
    window_days: int,
    out_col: str,
    as_of: pd.Timestamp | None,
) -> pd.DataFrame:
    """Compute count feature for a single symbol using vectorized operations."""
    symbol = prices_group["symbol"].iloc[0]
    symbol_events = events[events["symbol"] == symbol].copy()

    # Store original index
    original_index = prices_group.index.copy()

    if symbol_events.empty:
        prices_group[out_col] = 0
        return prices_group

    # Cross join prices with events (vectorized)
    prices_expanded = prices_group[["timestamp"]].copy()
    prices_expanded["timestamp_normalized"] = prices_expanded["timestamp"].dt.normalize()
    prices_expanded["_row_id"] = prices_expanded.index

    events_for_symbol = symbol_events[["disclosure_date"]].copy()
    events_for_symbol["disclosure_date_normalized"] = events_for_symbol["disclosure_date"].dt.normalize()

    # Cross join
    cross = prices_expanded.assign(key=1).merge(
        events_for_symbol.assign(key=1),
        on="key",
        suffixes=("_price", "_event"),
    ).drop("key", axis=1)

    # Vectorized filtering
    if as_of is None:
        # Per-row PIT filtering: disclosure_date <= timestamp
        mask = (
            (cross["disclosure_date_normalized"] <= cross["timestamp_normalized"])
            & (cross["disclosure_date_normalized"] > cross["timestamp_normalized"] - pd.Timedelta(days=window_days))
        )
    else:
        # Already filtered globally, just check window
        mask = (
            (cross["disclosure_date_normalized"] <= cross["timestamp_normalized"])
            & (cross["disclosure_date_normalized"] > cross["timestamp_normalized"] - pd.Timedelta(days=window_days))
        )

    # Aggregate per row_id (to handle duplicate timestamps correctly)
    features = cross[mask].groupby("_row_id").agg({
        "disclosure_date": "count",
    }).reset_index()
    features.columns = ["_row_id", "count"]

    # Merge back to prices_group
    # Ensure _row_id exists in prices_group
    if "_row_id" not in prices_group.columns:
        prices_group["_row_id"] = prices_group.index

    prices_group = prices_group.merge(
        features,
        on="_row_id",
        how="left",
    )

    # Fill missing values
    prices_group[out_col] = prices_group["count"].fillna(0).astype(int)

    # Drop temporary columns
    prices_group = prices_group.drop(columns=["count", "_row_id"], errors="ignore")

    # Restore original index
    prices_group.index = original_index

    return prices_group
