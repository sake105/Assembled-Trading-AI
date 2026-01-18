"""Factor store integration for feature building.

This module provides build_or_load_factors() which checks the factor store
for cached features before computing them, enabling performance improvements
for repeated runs.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.assembled_core.data.factor_store import (
    compute_universe_key,
    load_factors,
    store_factors,
)
from src.assembled_core.features.ta_features import add_all_features

logger = logging.getLogger(__name__)


def build_or_load_factors(
    prices: pd.DataFrame,
    factor_group: str = "core_ta",
    freq: str = "1d",
    universe_key: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    as_of: str | pd.Timestamp | None = None,
    force_rebuild: bool = False,
    builder_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    builder_kwargs: dict[str, Any] | None = None,
    factors_root: Path | None = None,
) -> pd.DataFrame:
    """Build or load factors from cache.

    High-level API that:
    1. Checks factor store for existing factors
    2. If cache hit and complete: load and return
    3. If cache miss or incomplete: compute factors (e.g., add_all_features), store, return

    Args:
        prices: Price DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        factor_group: Factor group name (e.g., "core_ta", "vol_liquidity")
        freq: Trading frequency ("1d" or "5min")
        universe_key: Optional universe key (if None, computed from prices["symbol"].unique())
        start_date: Optional start date for loading (if None, uses prices timestamp range)
        end_date: Optional end date for loading (if None, uses prices timestamp range)
        as_of: Optional point-in-time cutoff (PIT-safe filtering)
        force_rebuild: If True, always recompute and overwrite cache
        builder_fn: Optional function to build factors (default: add_all_features)
        builder_kwargs: Optional kwargs passed to builder_fn (e.g., ma_windows, atr_window)
        factors_root: Optional root directory for factor store

    Returns:
        DataFrame with factors (columns: timestamp, symbol, date, <feature_columns>)
        Sorted by timestamp, then symbol

    Notes:
        - If universe_key is None, it's computed from prices["symbol"].unique()
        - If start_date/end_date are None, they're inferred from prices timestamp range
        - Cache hit is logged with "cache_hit", cache miss with "cache_miss"
        - Factors are stored with year partitioning and metadata
    """
    # Compute universe_key if not provided
    if universe_key is None:
        symbols = sorted(prices["symbol"].unique().tolist())
        universe_key = compute_universe_key(symbols=symbols)

    # Infer date range from prices if not provided
    if start_date is None:
        start_date = prices["timestamp"].min()
    if end_date is None:
        end_date = prices["timestamp"].max()

    # Convert to Timestamps if strings
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date, tz="UTC")
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date, tz="UTC")
    if as_of is not None and isinstance(as_of, str):
        as_of = pd.Timestamp(as_of, tz="UTC")

    # Try to load from cache first (unless force_rebuild)
    if not force_rebuild:
        cached_factors = load_factors(
            factor_group=factor_group,
            freq=freq,
            universe_key=universe_key,
            start_date=start_date,
            end_date=end_date if as_of is None else None,
            as_of=as_of,
            factors_root=factors_root,
        )

        if cached_factors is not None and not cached_factors.empty:
            # Check if cache covers the full date range (simple check: all requested dates present)
            cached_start = cached_factors["timestamp"].min()
            cached_end = cached_factors["timestamp"].max()
            pit_cutoff = as_of if as_of is not None else end_date

            # Cache hit if it covers the requested range (with tolerance for PIT filtering)
            if cached_start <= start_date and cached_end >= pit_cutoff:
                logger.info(
                    f"[cache_hit] Factors loaded from store: {factor_group}/{freq}/{universe_key}, "
                    f"date_range=[{cached_start.date()}, {cached_end.date()}], rows={len(cached_factors)}"
                )
                
                # Apply PIT filtering if as_of is provided (even on cache hit, load_factors may return full range)
                if as_of is not None:
                    cached_factors = cached_factors[cached_factors["timestamp"] <= as_of].copy()
                
                return cached_factors
            else:
                logger.debug(
                    f"[cache_partial] Cached factors exist but don't cover full range: "
                    f"requested=[{start_date.date()}, {pit_cutoff.date()}], "
                    f"cached=[{cached_start.date()}, {cached_end.date()}], will compute missing"
                )

    # Cache miss or incomplete: compute factors
    logger.info(
        f"[cache_miss] Computing factors: {factor_group}/{freq}/{universe_key}, "
        f"date_range=[{start_date.date()}, {end_date.date()}]"
    )

    # Use default builder if not provided
    if builder_fn is None:
        builder_fn = add_all_features

    # Prepare builder kwargs (default: empty dict)
    if builder_kwargs is None:
        builder_kwargs = {}

    # Compute factors
    # Note: For incremental updates, prices should already be filtered to last session(s)
    factors_df = builder_fn(prices, **builder_kwargs)

    # Ensure required columns are present
    required_cols = ["timestamp", "symbol"]
    missing_cols = [col for col in required_cols if col not in factors_df.columns]
    if missing_cols:
        raise ValueError(f"Builder function must return DataFrame with columns: {missing_cols}")

    # Store factors in cache
    # Use append mode (not overwrite) to avoid rewriting existing year partitions
    # In backtest mode, this ensures only missing partitions are written per timestamp
    store_factors(
        df=factors_df,
        factor_group=factor_group,
        freq=freq,
        universe_key=universe_key,
        mode="overwrite" if force_rebuild else "append",
        factors_root=factors_root,
        metadata={
            "builder_fn": builder_fn.__name__ if hasattr(builder_fn, "__name__") else str(builder_fn),
            "builder_kwargs": builder_kwargs,
        },
    )

    logger.info(f"[cache_stored] Factors stored to cache: {len(factors_df)} rows")

    # Apply PIT filtering if as_of is provided (before returning)
    if as_of is not None:
        factors_df = factors_df[factors_df["timestamp"] <= as_of].copy()

    return factors_df

