"""Macro feature builder module (Sprint 11.E3).

Provides PIT-safe macro features using availability_ts filtering.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.assembled_core.data.macro.contract import (
    filter_macro_pit,
    normalize_macro_releases,
)


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
        result[out_col] = np.nan
        return result

    # Filter by series_id
    macro_series = macro_normalized[macro_normalized["series_id"] == series_id].copy()

    if macro_series.empty:
        out_col = out_col or f"macro_{series_id}_latest"
        result[out_col] = np.nan
        return result

    # Apply PIT filtering
    macro_pit = filter_macro_pit(macro_series, as_of)

    if macro_pit.empty:
        out_col = out_col or f"macro_{series_id}_latest"
        result[out_col] = np.nan
        return result

    # Sort by available_ts for merge_asof
    macro_sorted = macro_pit.sort_values("available_ts", kind="mergesort").reset_index(
        drop=True
    )

    # Use merge_asof to join latest available value to each timestamp
    # Implement stable row mapping to preserve original order
    panel_in = result.copy()
    panel_in["_row_id"] = np.arange(len(panel_in), dtype=np.int64)

    # Sort panel for merge_asof (stable sort: timestamp, then _row_id)
    panel_sorted = panel_in.sort_values(
        ["timestamp", "_row_id"], kind="mergesort"
    ).reset_index(drop=True)

    # Perform merge_asof
    merged = pd.merge_asof(
        panel_sorted,
        macro_sorted[["available_ts", "value"]],
        left_on="timestamp",
        right_on="available_ts",
        direction="backward",  # available_ts <= timestamp
        allow_exact_matches=True,
    )

    # Set output column
    out_col = out_col or f"macro_{series_id}_latest"
    merged[out_col] = merged["value"]

    # Map back to original rows using _row_id
    # Merge on _row_id to restore original order
    result = panel_in.merge(
        merged[["_row_id", out_col]],
        on="_row_id",
        how="left",
        validate="one_to_one",
    )

    # Sort by _row_id to restore original order, then drop helper column
    result = (
        result.sort_values("_row_id", kind="mergesort")
        .drop(columns=["_row_id"])
        .reset_index(drop=True)
    )

    return result


# ── 3.8  Extended FRED Macro Feature Config ──────────────────────────
# series_id → (friendly name, publication lag in days)
EXTENDED_FRED_SERIES: dict[str, tuple[str, int]] = {
    "UNRATE": ("unemployment_rate", 30),
    "ICSA": ("initial_claims", 7),
    "UMCSENT": ("consumer_sentiment", 14),
    "HOUST": ("housing_starts", 30),
    "INDPRO": ("industrial_production", 30),
    "PCEPI": ("pce_inflation", 30),
    "T10Y2Y": ("yield_curve_10y2y", 1),
    "DFF": ("fed_funds_rate", 1),
}


def compute_diffusion_index(
    macro_values: dict[str, pd.Series],
    momentum_window: int = 3,
) -> pd.Series:
    """Compute macro diffusion index: % of indicators with positive momentum.

    Args:
        macro_values: Dict of series_id → time-series of values.
        momentum_window: Number of periods for momentum calculation.

    Returns:
        Series with diffusion index (0-1 range).
    """
    if not macro_values:
        return pd.Series(dtype=float)

    combined = pd.DataFrame(macro_values)
    if combined.empty:
        return pd.Series(dtype=float)

    momentum = combined.diff(momentum_window)
    positive_frac = (momentum > 0).sum(axis=1) / momentum.notna().sum(axis=1)
    return positive_frac.fillna(0.5)
