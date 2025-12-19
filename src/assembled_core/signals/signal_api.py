"""Signal API for standardized signal representation and validation.

This module provides:
- SignalMetadata: Metadata dataclass for signal frames
- normalize_signals: Normalize signal values (zscore, rank, none)
- make_signal_frame: Create standardized SignalFrame from raw scores
- validate_signal_frame: Validate SignalFrame for correctness and PIT-safety

SignalFrame Contract:
- Index: pd.DatetimeIndex (UTC-aware)
- Required columns: symbol, signal_value
- Optional columns: weight_target, as_of, side, source
- No duplicate (timestamp, symbol) pairs
- PIT-safe: as_of >= timestamp for all rows (if as_of present)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.assembled_core.qa.point_in_time_checks import (
    PointInTimeViolationError,
    check_features_pit_safe,
)

logger = logging.getLogger(__name__)

# Type alias for SignalFrame (documented in module docstring)
# SignalFrame = pd.DataFrame with:
# - Index: pd.DatetimeIndex (UTC-aware)
# - Required columns: symbol, signal_value
# - Optional columns: weight_target, as_of, side, source


@dataclass
class SignalMetadata:
    """Metadata for a SignalFrame.
    
    Attributes:
        strategy_name: Name of the strategy (e.g., "trend_baseline", "ml_alpha")
        freq: Trading frequency ("1d" or "5min")
        universe_name: Optional name of the universe (e.g., "ai_tech", "macro_world_etfs")
        as_of: Optional point-in-time timestamp (when were signals generated?)
        horizon_days: Optional forward-looking horizon for signal evaluation (default: None)
        source: Optional signal source (e.g., "research_playbook", "eod_pipeline")
        notes: Optional notes or description
    """
    strategy_name: str
    freq: Literal["1d", "5min"]
    universe_name: str | None = None
    as_of: pd.Timestamp | None = None
    horizon_days: int | None = None
    source: str | None = None
    notes: str | None = None


def normalize_signals(
    signals: pd.DataFrame,
    value_col: str = "signal_value",
    method: str = "zscore",
    clip: float | None = 5.0,
) -> pd.DataFrame:
    """Normalize signal values in a SignalFrame.
    
    Args:
        signals: DataFrame with signal values (must contain value_col and timestamp index)
        value_col: Name of signal value column (default: "signal_value")
        method: Normalization method:
            - "zscore": Z-score normalization (mean=0, std=1) per timestamp
            - "rank": Rank normalization (ranks scaled to [-0.5, 0.5]) per timestamp
            - "none": No normalization (return as-is)
        clip: Optional clipping value (e.g., 5.0 for zscore = clip to [-5, 5])
            Applied after normalization. None = no clipping.
    
    Returns:
        DataFrame with normalized signal values (same structure as input)
        Original DataFrame is not modified (copy returned)
    
    Raises:
        ValueError: If value_col not found or method not recognized
    
    Note:
        Normalization is performed per timestamp (cross-sectional) to ensure
        signals are comparable across symbols at each time point.
        For zscore: (x - mean) / std per timestamp
        For rank: Ranks scaled to [-0.5, 0.5] per timestamp
    """
    if value_col not in signals.columns:
        raise ValueError(f"Signal column '{value_col}' not found. Available columns: {list(signals.columns)}")
    
    if method not in ("zscore", "rank", "none"):
        raise ValueError(f"Unknown normalization method: {method}. Must be one of: zscore, rank, none")
    
    # Always return a copy to avoid modifying input
    result = signals.copy()
    
    if method == "none":
        # No normalization, but still apply clipping if requested
        if clip is not None:
            result[value_col] = result[value_col].clip(-clip, clip)
        return result
    
    # Group by timestamp (index) for cross-sectional normalization
    if not isinstance(result.index, pd.DatetimeIndex):
        raise ValueError("Signals DataFrame must have DatetimeIndex for normalization")
    
    # Extract signal values
    signal_values = result[value_col].copy()
    
    if method == "zscore":
        # Z-score normalization per timestamp
        # Group by timestamp, compute mean and std, then normalize
        normalized = signal_values.groupby(result.index).transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 1e-10 else x - x.mean()
        )
        result[value_col] = normalized
    elif method == "rank":
        # Rank normalization: ranks scaled to [-0.5, 0.5] per timestamp
        # Rank from 0 to n-1, then scale to [-0.5, 0.5]
        def rank_normalize(x):
            if len(x) == 1:
                return pd.Series([0.0], index=x.index)
            ranks = x.rank(method="average")
            # Scale to [-0.5, 0.5]
            n = len(x)
            normalized_ranks = (ranks - (n + 1) / 2) / n
            return normalized_ranks
        
        normalized = signal_values.groupby(result.index).transform(rank_normalize)
        result[value_col] = normalized
    
    # Apply clipping if requested
    if clip is not None:
        result[value_col] = result[value_col].clip(-clip, clip)
    
    return result


def make_signal_frame(
    raw_scores: pd.DataFrame,
    meta: SignalMetadata,
    value_col: str = "score",
    method: str = "zscore",
    clip: float | None = 5.0,
) -> pd.DataFrame:
    """Create a standardized SignalFrame from raw scores.
    
    Args:
        raw_scores: DataFrame with raw signal scores
            Required columns: symbol, value_col
            Index: pd.DatetimeIndex (UTC-aware)
        meta: SignalMetadata with strategy information
        value_col: Name of raw score column (default: "score")
        method: Normalization method (default: "zscore")
        clip: Optional clipping value after normalization (default: 5.0)
    
    Returns:
        SignalFrame DataFrame with columns:
        - symbol: Symbol name
        - signal_value: Normalized signal value
        - as_of: Point-in-time timestamp (if meta.as_of is set)
        - strategy_name: Strategy name (if useful as column)
        - freq: Frequency (if useful as column)
        Sorted by timestamp, then symbol.
    
    Raises:
        ValueError: If raw_scores format is invalid or missing required columns
    """
    # Validate input format
    if not isinstance(raw_scores.index, pd.DatetimeIndex):
        raise ValueError(f"raw_scores must have DatetimeIndex, got {type(raw_scores.index)}")
    
    required_cols = ["symbol", value_col]
    missing = [col for col in required_cols if col not in raw_scores.columns]
    if missing:
        raise ValueError(f"raw_scores missing required columns: {missing}")
    
    # Create copy to avoid modifying input
    signal_df = raw_scores[[value_col, "symbol"]].copy()
    
    # Rename value_col to signal_value
    if value_col != "signal_value":
        signal_df = signal_df.rename(columns={value_col: "signal_value"})
    
    # Normalize signals
    signal_df = normalize_signals(
        signal_df,
        value_col="signal_value",
        method=method,
        clip=clip,
    )
    
    # Add metadata columns
    if meta.as_of is not None:
        signal_df["as_of"] = meta.as_of
    
    # Ensure proper column order: symbol, signal_value, then optional columns
    cols = ["symbol", "signal_value"]
    if "as_of" in signal_df.columns:
        cols.append("as_of")
    
    # Add any other columns that were in raw_scores (e.g., side, weight_target)
    other_cols = [c for c in signal_df.columns if c not in cols]
    cols.extend(other_cols)
    
    signal_df = signal_df[cols]
    
    # Sort by timestamp, then symbol
    signal_df = signal_df.sort_values(["symbol"], kind="stable")
    
    return signal_df


def validate_signal_frame(
    signal_df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    strict: bool = False,
    feature_source: str | None = None,
) -> None:
    """Validate a SignalFrame for correctness and PIT-safety.
    
    Args:
        signal_df: SignalFrame to validate
        as_of: Optional maximum allowed timestamp (for PIT-check)
            If None, uses signal_df["as_of"].max() if "as_of" column exists
        strict: If True, raise ValueError/PointInTimeViolationError on violations.
                If False, log warnings.
        feature_source: Optional string describing signal source (for logging)
    
    Raises:
        ValueError: If strict=True and validation fails:
            - Index is not DatetimeIndex
            - Missing required columns (symbol, signal_value)
            - Duplicate (timestamp, symbol) pairs
            - All signal_value are NaN
        PointInTimeViolationError: If strict=True and PIT violation detected
    
    Checks:
        1. Formale Checks: Index-Typ, Spalten, Duplikate
        2. PIT-Checks: max(timestamp) <= as_of (via check_features_pit_safe)
        3. Data Quality: signal_value not all NaN
    """
    # Check 1: Index is DatetimeIndex
    if not isinstance(signal_df.index, pd.DatetimeIndex):
        error_msg = f"SignalFrame index must be DatetimeIndex, got {type(signal_df.index)}"
        if strict:
            raise ValueError(error_msg)
        logger.warning("VALIDATE SIGNAL FRAME: %s", error_msg)
        return
    
    # Check 2: Required columns
    required_cols = ["symbol", "signal_value"]
    missing = [col for col in required_cols if col not in signal_df.columns]
    if missing:
        error_msg = f"SignalFrame missing required columns: {missing}"
        if strict:
            raise ValueError(error_msg)
        logger.warning("VALIDATE SIGNAL FRAME: %s", error_msg)
        return
    
    # Check 3: No duplicate (timestamp, symbol) pairs
    # Create a DataFrame with timestamp (from index) and symbol
    check_df = pd.DataFrame({
        "timestamp": signal_df.index,
        "symbol": signal_df["symbol"].values,
    })
    duplicate_mask = check_df.duplicated(subset=["timestamp", "symbol"], keep=False)
    if duplicate_mask.any():
        n_duplicates = duplicate_mask.sum()
        error_msg = f"SignalFrame contains {n_duplicates} duplicate (timestamp, symbol) pairs"
        if strict:
            raise ValueError(error_msg)
        logger.warning("VALIDATE SIGNAL FRAME: %s", error_msg)
        return
    
    # Check 4: signal_value not all NaN
    if signal_df["signal_value"].isna().all():
        error_msg = "SignalFrame signal_value column contains only NaN values"
        if strict:
            raise ValueError(error_msg)
        logger.warning("VALIDATE SIGNAL FRAME: %s", error_msg)
        return
    
    # Check 5: PIT-safety check
    # Determine as_of: use parameter if provided, else try to get from "as_of" column
    pit_as_of = as_of
    if pit_as_of is None and "as_of" in signal_df.columns:
        # Use max as_of from DataFrame
        pit_as_of = signal_df["as_of"].max()
    
    if pit_as_of is not None:
        # Convert signal_df to format expected by check_features_pit_safe
        # It expects a DataFrame with "timestamp" column, but we have timestamp as index
        # Create a simple DataFrame with timestamp column from index
        pit_check_df = pd.DataFrame({
            "timestamp": signal_df.index,
        })
        
        try:
            is_pit_safe = check_features_pit_safe(
                pit_check_df,
                as_of=pit_as_of,
                timestamp_col="timestamp",
                strict=strict,
                feature_source=feature_source or "SignalFrame",
            )
            if not is_pit_safe and strict:
                # check_features_pit_safe already raised PointInTimeViolationError
                # This should not be reached, but just in case
                raise PointInTimeViolationError("PIT violation detected in SignalFrame")
        except PointInTimeViolationError:
            # Re-raise if strict mode
            if strict:
                raise
            # Otherwise, warning was already logged by check_features_pit_safe

