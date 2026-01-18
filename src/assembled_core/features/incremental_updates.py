# src/assembled_core/features/incremental_updates.py
"""Incremental Feature Updates (Sprint 5 / F3).

This module provides functions for incremental feature updates, allowing
computation of only the last session(s) instead of full recompute.

Key Functions:
    - compute_only_last_session(): Compute features for last session only
    - compute_last_N_sessions(): Compute features for last N sessions
    - filter_prices_for_incremental(): Filter prices for incremental update

Use Case:
    Daily EOD pipeline can use incremental updates to:
    1. Load existing factors from store
    2. Compute only last session (or last N sessions)
    3. Append to store (mode="append" with deduplication)
    
This avoids full recompute of historical features, improving performance.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import pandas as pd

logger = logging.getLogger(__name__)


def filter_prices_for_incremental(
    prices: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    window_days: int = 1,
) -> pd.DataFrame:
    """Filter prices for incremental update (last session or last N sessions).

    Args:
        prices: Price DataFrame with columns: timestamp, symbol, ...
        as_of: Optional cutoff timestamp (default: max timestamp in prices)
        window_days: Number of days to include (default: 1 = last session only)

    Returns:
        Filtered DataFrame with only last session(s) (sorted by timestamp, symbol)

    Notes:
        - If as_of is None, uses prices["timestamp"].max() as cutoff
        - Filters to include only data within window_days before as_of
        - Ensures timestamp is UTC-aware
        - Sorts by (timestamp, symbol) for deterministic processing
    """
    if prices.empty:
        return prices.copy()

    # Ensure timestamp is UTC-aware
    if not pd.api.types.is_datetime64_any_dtype(prices["timestamp"]):
        prices = prices.copy()
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    elif prices["timestamp"].dtype.tz is None:
        prices = prices.copy()
        prices["timestamp"] = prices["timestamp"].dt.tz_localize("UTC")

    # Determine cutoff (as_of or max timestamp)
    if as_of is None:
        as_of = prices["timestamp"].max()
    elif isinstance(as_of, str):
        as_of = pd.Timestamp(as_of, tz="UTC")
    elif as_of.tz is None:
        as_of = as_of.tz_localize("UTC")

    # Filter to last window_days
    cutoff_date = as_of.date() if hasattr(as_of, "date") else pd.Timestamp(as_of).date()
    start_date = cutoff_date - pd.Timedelta(days=window_days - 1)
    
    # Filter prices within window
    filtered = prices[
        (prices["timestamp"].dt.date >= start_date) & (prices["timestamp"] <= as_of)
    ].copy()

    # Sort for deterministic processing
    filtered = filtered.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    logger.debug(
        f"Incremental filter: as_of={as_of.date()}, window_days={window_days}, "
        f"filtered {len(filtered)} rows from {len(prices)} total rows"
    )

    return filtered


def compute_only_last_session(
    prices: pd.DataFrame,
    builder_fn: Callable[[pd.DataFrame], pd.DataFrame],
    builder_kwargs: dict[str, Any] | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute features for last session only (incremental update).

    Args:
        prices: Price DataFrame with columns: timestamp, symbol, ...
        builder_fn: Function to build features (e.g., add_all_features)
        builder_kwargs: Optional kwargs passed to builder_fn
        as_of: Optional cutoff timestamp (default: max timestamp in prices)

    Returns:
        DataFrame with features for last session only

    Notes:
        - Filters prices to last session (window_days=1)
        - Computes features only for filtered data
        - Use with store_factors_parquet(..., mode="append") for incremental update
    """
    if builder_kwargs is None:
        builder_kwargs = {}

    # Filter to last session
    prices_filtered = filter_prices_for_incremental(prices, as_of=as_of, window_days=1)

    if prices_filtered.empty:
        logger.warning("No prices found for last session, returning empty DataFrame")
        return pd.DataFrame()

    # Compute features for last session
    factors = builder_fn(prices_filtered, **builder_kwargs)

    logger.info(f"Computed features for last session: {len(factors)} rows")
    return factors


def compute_last_N_sessions(
    prices: pd.DataFrame,
    builder_fn: Callable[[pd.DataFrame], pd.DataFrame],
    window_days: int,
    builder_kwargs: dict[str, Any] | None = None,
    as_of: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute features for last N sessions (incremental update).

    Args:
        prices: Price DataFrame with columns: timestamp, symbol, ...
        builder_fn: Function to build features (e.g., add_all_features)
        window_days: Number of days to include (e.g., 5 for last 5 sessions)
        builder_kwargs: Optional kwargs passed to builder_fn
        as_of: Optional cutoff timestamp (default: max timestamp in prices)

    Returns:
        DataFrame with features for last N sessions

    Notes:
        - Filters prices to last N sessions
        - Computes features only for filtered data
        - Use with store_factors_parquet(..., mode="append") for incremental update
    """
    if builder_kwargs is None:
        builder_kwargs = {}

    # Filter to last N sessions
    prices_filtered = filter_prices_for_incremental(prices, as_of=as_of, window_days=window_days)

    if prices_filtered.empty:
        logger.warning(f"No prices found for last {window_days} sessions, returning empty DataFrame")
        return pd.DataFrame()

    # Compute features for last N sessions
    factors = builder_fn(prices_filtered, **builder_kwargs)

    logger.info(f"Computed features for last {window_days} sessions: {len(factors)} rows")
    return factors
