"""Precomputed index for efficient snapshot extraction in backtests.

This module provides an optimized index structure for extracting the latest row
per symbol up to a given timestamp (as_of) without expensive groupby/sort operations
on every snapshot query.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PrecomputedPanelIndex:
    """Precomputed index for efficient snapshot extraction.
    
    This index structure enables O(S log N) snapshot extraction (where S = number
    of symbols, N = max rows per symbol) instead of O(N log N) groupby operations.
    
    Attributes:
        symbols: List of unique symbols (sorted)
        timestamps_by_symbol: Dictionary mapping symbol -> sorted numpy array of timestamps
        row_idx_by_symbol: Dictionary mapping symbol -> numpy array of row indices
            into the original DataFrame (matching order of timestamps_by_symbol)
        _last_seen_pointers: Internal cache for monotonic as_of queries (optional optimization)
            Dictionary mapping symbol -> last seen index (for monotonically increasing as_of)
    """
    
    symbols: list[str]
    timestamps_by_symbol: dict[str, np.ndarray]
    row_idx_by_symbol: dict[str, np.ndarray]
    _last_seen_pointers: dict[str, int] | None = None
    
    def __post_init__(self) -> None:
        """Initialize last seen pointers cache."""
        if self._last_seen_pointers is None:
            self._last_seen_pointers = {sym: 0 for sym in self.symbols}


def build_panel_index(
    df: pd.DataFrame,
    *,
    symbol_col: str = "symbol",
    ts_col: str = "timestamp",
) -> PrecomputedPanelIndex:
    """Build a precomputed index from a DataFrame.
    
    This function creates an index structure that allows efficient snapshot
    extraction (latest row per symbol <= as_of) without expensive groupby
    operations.
    
    Args:
        df: DataFrame with columns: symbol_col, ts_col, ... (must be sorted by symbol, timestamp)
        symbol_col: Name of the symbol column (default: "symbol")
        ts_col: Name of the timestamp column (default: "timestamp")
    
    Returns:
        PrecomputedPanelIndex with symbols, timestamps_by_symbol, and row_idx_by_symbol
    
    Note:
        The input DataFrame should be sorted by symbol, then timestamp for optimal performance.
        If not sorted, the function will sort it internally (which is a one-time cost).
    """
    # Ensure DataFrame is sorted by symbol, then timestamp
    if not df.empty:
        # Check if already sorted (optimization: avoid sorting if already sorted)
        sorted_check = df[[symbol_col, ts_col]].sort_values([symbol_col, ts_col], ignore_index=True)
        if not sorted_check.equals(df[[symbol_col, ts_col]].reset_index(drop=True)):
            df = df.sort_values([symbol_col, ts_col]).reset_index(drop=True)
    
    # Extract unique symbols (sorted)
    symbols = sorted(df[symbol_col].unique().tolist())
    
    # Build index structures
    timestamps_by_symbol: dict[str, np.ndarray] = {}
    row_idx_by_symbol: dict[str, np.ndarray] = {}
    
    for symbol in symbols:
        # Get all rows for this symbol
        mask = df[symbol_col] == symbol
        symbol_rows = df[mask]
        
        if symbol_rows.empty:
            # Empty symbol (shouldn't happen, but handle gracefully)
            timestamps_by_symbol[symbol] = np.array([], dtype='datetime64[ns, UTC]')
            row_idx_by_symbol[symbol] = np.array([], dtype=np.int64)
            continue
        
        # Extract timestamps (convert to numpy array for efficient binary search)
        timestamps_series = symbol_rows[ts_col]
        
        # Ensure timestamps are timezone-aware UTC (convert if needed)
        if timestamps_series.dtype.tz is None:
            timestamps_series = pd.to_datetime(timestamps_series, utc=True)
        elif timestamps_series.dtype.tz != pd.Timestamp.utcnow().tz:
            timestamps_series = timestamps_series.dt.tz_convert('UTC')
        
        # Convert to numpy datetime64 array for efficient binary search
        timestamps = timestamps_series.values
        
        # Get row indices (original DataFrame indices)
        row_indices = np.arange(len(df))[mask]
        
        # Store sorted arrays (they should already be sorted due to input sorting)
        timestamps_by_symbol[symbol] = timestamps
        row_idx_by_symbol[symbol] = row_indices
    
    return PrecomputedPanelIndex(
        symbols=symbols,
        timestamps_by_symbol=timestamps_by_symbol,
        row_idx_by_symbol=row_idx_by_symbol,
        _last_seen_pointers={sym: 0 for sym in symbols},
    )


def snapshot_as_of(
    df: pd.DataFrame,
    index: PrecomputedPanelIndex,
    as_of: pd.Timestamp,
    *,
    use_monotonic_optimization: bool = True,
) -> pd.DataFrame:
    """Extract snapshot (latest row per symbol <= as_of) using precomputed index.
    
    This function efficiently extracts the latest row per symbol that is <= as_of
    using binary search on precomputed sorted arrays, avoiding expensive groupby
    operations.
    
    Args:
        df: Original DataFrame (must match the one used to build the index)
        index: PrecomputedPanelIndex built from df
        as_of: Maximum allowed timestamp (pd.Timestamp, timezone-aware or naive).
            If naive, will be normalized to UTC. If timezone-aware, will be converted to UTC.
            Semantics: returns the latest row per symbol where timestamp <= as_of.
        use_monotonic_optimization: If True, use cached pointers for monotonically
            increasing as_of queries (default: True). This enables O(S) performance
            for sequential queries instead of O(S log N).
    
    Returns:
        DataFrame with one row per symbol (latest row <= as_of).
        Columns: same as input df, sorted by symbol.
        
        Edge cases:
        - If as_of is before the first timestamp for a symbol, that symbol is excluded
          (no rows <= as_of for that symbol).
        - If as_of exactly matches a timestamp, that row is included (<= comparison).
        - Missing days/gaps: Returns the last available row <= as_of for each symbol.
        - Missing symbols: Only symbols present in the original DataFrame are included.
    
    Performance:
        - Without monotonic optimization: O(S log N) where S = symbols, N = max rows per symbol
        - With monotonic optimization (monotonic as_of): O(S) per query
        - Traditional groupby approach: O(N log N) per query
    
    Note:
        Timezone handling: All timestamps are normalized to UTC for consistent comparison.
        Naive timestamps are assumed to be UTC. Timezone-aware timestamps are converted to UTC.
    """
    if df.empty or not index.symbols:
        return pd.DataFrame(columns=df.columns)
    
    # Ensure as_of is timezone-aware UTC
    if as_of.tz is None:
        as_of = pd.to_datetime(as_of, utc=True)
    elif as_of.tz != pd.Timestamp.utcnow().tz:
        as_of = as_of.tz_convert('UTC')
    
    # Convert as_of to numpy-compatible timestamp for comparison
    as_of_np = np.datetime64(as_of.to_numpy())
    
    # Collect rows for snapshot
    snapshot_rows = []
    
    for symbol in index.symbols:
        timestamps = index.timestamps_by_symbol[symbol]
        row_indices = index.row_idx_by_symbol[symbol]
        
        if len(timestamps) == 0:
            continue
        
        # Find latest row <= as_of using binary search
        if use_monotonic_optimization and index._last_seen_pointers is not None:
            # Start search from last seen pointer (monotonic optimization)
            start_idx = index._last_seen_pointers[symbol]
            if start_idx >= len(timestamps):
                start_idx = len(timestamps) - 1
            
            # Find first index where timestamp > as_of (starting from last_seen)
            # This is more efficient if as_of is monotonically increasing
            search_slice = timestamps[start_idx:]
            if len(search_slice) > 0:
                idx_in_slice = np.searchsorted(search_slice, as_of_np, side='right')
                found_idx = start_idx + idx_in_slice - 1
            else:
                found_idx = len(timestamps) - 1
        else:
            # Standard binary search (no monotonic optimization)
            found_idx = np.searchsorted(timestamps, as_of_np, side='right') - 1
        
        # Clamp to valid range
        if found_idx < 0:
            # No row <= as_of for this symbol
            continue
        
        if found_idx >= len(timestamps):
            found_idx = len(timestamps) - 1
        
        # Update last seen pointer for monotonic optimization
        if use_monotonic_optimization and index._last_seen_pointers is not None:
            index._last_seen_pointers[symbol] = found_idx
        
        # Get the row index and append to snapshot
        row_idx = row_indices[found_idx]
        snapshot_rows.append(row_idx)
    
    if not snapshot_rows:
        return pd.DataFrame(columns=df.columns)
    
    # Extract rows and create snapshot DataFrame
    snapshot_df = df.iloc[snapshot_rows].copy()
    
    # Ensure deterministic sorting by symbol
    snapshot_df = snapshot_df.sort_values("symbol").reset_index(drop=True)
    
    return snapshot_df

