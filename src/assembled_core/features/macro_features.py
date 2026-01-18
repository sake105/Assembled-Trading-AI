"""Macro feature builder module (Sprint 11.E3).

Provides PIT-safe macro features using availability_ts filtering.
"""

from __future__ import annotations

import pandas as pd

from src.assembled_core.data.macro.contract import filter_macro_pit, normalize_macro_releases


def add_latest_macro_value(
    panel_index: pd.DataFrame,
    macro_df: pd.DataFrame,
    as_of: pd.Timestamp,
    *,
    series_id: str,
    out_col: str | None = None,
) -> pd.DataFrame:
    """Add latest macro value to panel index (PIT-safe, Sprint 11.E3).

    For each row in panel_index, adds the latest available macro value
    for the given series_id, where available_ts <= as_of.

    This feature is PIT-safe: only macro data with available_ts <= as_of
    is used, preventing look-ahead bias.

    Args:
        panel_index: Panel DataFrame with columns: timestamp (UTC), symbol, ...
        macro_df: Macro releases DataFrame (must have series_id, available_ts, value)
        as_of: Point-in-time cutoff (pd.Timestamp, UTC)
        series_id: Macro series identifier to use
        out_col: Output column name (default: f"macro_{series_id}_latest")

    Returns:
        Panel DataFrame with additional column containing latest macro value

    Raises:
        ValueError: If required columns are missing
    """
    if "timestamp" not in panel_index.columns:
        raise ValueError("panel_index must have 'timestamp' column")

    result = panel_index.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    # Normalize macro data
    try:
        macro_normalized = normalize_macro_releases(macro_df)
    except ValueError:
        # If normalization fails, return panel with NaN values
        out_col = out_col or f"macro_{series_id}_latest"
        result[out_col] = pd.NA
        return result

    # Filter by series_id
    macro_series = macro_normalized[macro_normalized["series_id"] == series_id].copy()

    if macro_series.empty:
        out_col = out_col or f"macro_{series_id}_latest"
        result[out_col] = pd.NA
        return result

    # Apply PIT filtering
    macro_pit = filter_macro_pit(macro_series, as_of)

    if macro_pit.empty:
        out_col = out_col or f"macro_{series_id}_latest"
        result[out_col] = pd.NA
        return result

    # Sort by available_ts for merge_asof
    macro_sorted = macro_pit.sort_values("available_ts").reset_index(drop=True)

    # Use merge_asof to join latest available value to each timestamp
    # Store original index for mapping back
    original_index = result.index.copy()
    result_sorted = result.sort_values("timestamp").reset_index(drop=True)

    merged = pd.merge_asof(
        result_sorted,
        macro_sorted[["available_ts", "value"]],
        left_on="timestamp",
        right_on="available_ts",
        direction="backward",  # available_ts <= timestamp
        allow_exact_matches=True,
    )

    # Set output column
    out_col = out_col or f"macro_{series_id}_latest"
    merged[out_col] = merged["value"]

    # Restore original index
    merged.index = original_index

    # Assign values back to result
    result[out_col] = merged[out_col]

    return result
