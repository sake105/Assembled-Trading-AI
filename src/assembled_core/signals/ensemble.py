"""Ensemble layer for combining rule-based signals with meta-model confidence scores.

This module provides functions to apply meta-model confidence scores to rule-based signals,
either as a filter (removing low-confidence signals) or as a scaling factor (adjusting
position sizes based on confidence).

Example:
    >>> from src.assembled_core.signals.ensemble import apply_meta_filter, apply_meta_scaling
    >>> from src.assembled_core.signals.meta_model import load_meta_model
    >>> import pandas as pd
    >>>
    >>> # Load meta-model
    >>> meta_model = load_meta_model("models/meta/trend_baseline_meta.joblib")
    >>>
    >>> # Load signals and features
    >>> signals = pd.DataFrame({
    ...     "timestamp": [...],
    ...     "symbol": [...],
    ...     "direction": ["LONG", "FLAT", ...],
    ...     "score": [0.8, 0.2, ...]
    ... })
    >>> features = prices_with_features[meta_model.feature_names]
    >>>
    >>> # Apply meta-filter (remove signals with confidence < 0.5)
    >>> filtered_signals = apply_meta_filter(
    ...     signals=signals,
    ...     meta_model=meta_model,
    ...     features=features,
    ...     min_confidence=0.5
    ... )
    >>>
    >>> # Or apply meta-scaling (scale positions by confidence)
    >>> scaled_signals = apply_meta_scaling(
    ...     signals=signals,
    ...     meta_model=meta_model,
    ...     features=features,
    ...     min_confidence=0.3,
    ...     max_scaling=1.0
    ... )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.assembled_core.signals.meta_model import MetaModel

logger = logging.getLogger(__name__)


def apply_meta_filter(
    signals: pd.DataFrame,
    meta_model: MetaModel,
    features: pd.DataFrame,
    min_confidence: float = 0.5,
    join_keys: list[str] | None = None,
) -> pd.DataFrame:
    """Apply meta-model confidence filter to signals.

    This function:
    1. Computes confidence scores for each signal using the meta-model
    2. Filters out signals with confidence < min_confidence (sets direction to "FLAT")
    3. Adds a `meta_confidence` column to the signals DataFrame

    Args:
        signals: DataFrame with columns: timestamp, symbol, direction (and optionally score)
            Must have index or join_keys that can be used to join with features
        meta_model: Trained MetaModel instance
        features: DataFrame with feature columns matching meta_model.feature_names
            Must have same index/join_keys as signals
        min_confidence: Minimum confidence threshold (default: 0.5)
            Signals with confidence < min_confidence will be set to "FLAT"
        join_keys: Optional list of column names to join signals and features
            If None, uses index alignment (signals and features must have same index)

    Returns:
        DataFrame with same structure as input signals, plus:
        - `meta_confidence`: Confidence score ∈ [0, 1] for each signal
        - `direction`: Updated to "FLAT" for signals with confidence < min_confidence

    Raises:
        ValueError: If required columns are missing or join fails
    """
    if signals.empty:
        return signals.copy()

    # Validate inputs
    required_signal_cols = ["direction"]
    missing = [c for c in required_signal_cols if c not in signals.columns]
    if missing:
        raise ValueError(f"Signals DataFrame missing required columns: {missing}")

    # Make a copy to avoid modifying original
    result = signals.copy()

    # Compute confidence scores
    try:
        confidence_scores = meta_model.predict_proba(features)
    except Exception as e:
        logger.error(f"Failed to compute confidence scores: {e}")
        raise ValueError(f"Failed to compute confidence scores: {e}") from e

    # Join confidence scores with signals
    # Since features and signals are aligned (same order from merge), use direct assignment
    if len(result) == len(confidence_scores):
        # Direct alignment (same order from merge)
        result["meta_confidence"] = confidence_scores.values
    elif join_keys is not None:
        # Try to merge using join_keys
        if all(k in signals.columns for k in join_keys):
            # Reset index if needed
            if features.index.names != [None]:
                features_reset = features.reset_index()
            else:
                features_reset = features.copy()

            # Create confidence DataFrame with join keys
            confidence_df = pd.DataFrame(
                {
                    "meta_confidence": confidence_scores.values,
                }
            )

            # Add join keys from features if available
            for k in join_keys:
                if k in features_reset.columns:
                    confidence_df[k] = features_reset[k].values
                elif k in features.index.names:
                    confidence_df[k] = features.index.get_level_values(k).values

            # Merge
            result = result.merge(confidence_df, on=join_keys, how="left")
        else:
            raise ValueError(
                f"Cannot join on keys {join_keys}: not all keys present in signals"
            )
    else:
        # Use index alignment (signals and features must have same index)
        if len(result) != len(confidence_scores):
            raise ValueError(
                f"Signals and features must have same length for index alignment. "
                f"Got {len(result)} signals and {len(confidence_scores)} confidence scores"
            )
        result["meta_confidence"] = confidence_scores.values

    # Fill NaN confidence scores with 0.0 (shouldn't happen, but handle gracefully)
    result["meta_confidence"] = result["meta_confidence"].fillna(0.0)

    # Apply filter: set direction to "FLAT" for signals with confidence < min_confidence
    # Only filter LONG signals (keep FLAT signals as-is)
    long_mask = result["direction"] == "LONG"
    low_confidence_mask = result["meta_confidence"] < min_confidence

    filtered_count = (long_mask & low_confidence_mask).sum()
    if filtered_count > 0:
        logger.info(
            f"Filtering {filtered_count} signals with confidence < {min_confidence}"
        )
        result.loc[long_mask & low_confidence_mask, "direction"] = "FLAT"

    return result


def apply_meta_scaling(
    signals: pd.DataFrame,
    meta_model: MetaModel,
    features: pd.DataFrame,
    min_confidence: float = 0.0,
    max_scaling: float = 1.0,
    join_keys: list[str] | None = None,
    scale_score: bool = True,
) -> pd.DataFrame:
    """Apply meta-model confidence scaling to signals.

    This function:
    1. Computes confidence scores for each signal using the meta-model
    2. Scales signal scores (or positions) by confidence score
    3. Filters out signals with confidence < min_confidence (sets direction to "FLAT")
    4. Adds a `meta_confidence` column and optionally `final_score` column

    Args:
        signals: DataFrame with columns: timestamp, symbol, direction (and optionally score)
            Must have index or join_keys that can be used to join with features
        meta_model: Trained MetaModel instance
        features: DataFrame with feature columns matching meta_model.feature_names
            Must have same index/join_keys as signals
        min_confidence: Minimum confidence threshold (default: 0.0)
            Signals with confidence < min_confidence will be set to "FLAT"
        max_scaling: Maximum scaling factor (default: 1.0)
            Confidence scores are clipped to [0, max_scaling] before scaling
        join_keys: Optional list of column names to join signals and features
            If None, uses index alignment (signals and features must have same index)
        scale_score: If True, scale the `score` column by confidence (default: True)
            If False, only add `meta_confidence` column without scaling

    Returns:
        DataFrame with same structure as input signals, plus:
        - `meta_confidence`: Confidence score ∈ [0, 1] for each signal
        - `final_score`: Scaled score (if scale_score=True and original score exists)
        - `direction`: Updated to "FLAT" for signals with confidence < min_confidence

    Raises:
        ValueError: If required columns are missing or join fails
    """
    if signals.empty:
        return signals.copy()

    # Validate inputs
    required_signal_cols = ["direction"]
    missing = [c for c in required_signal_cols if c not in signals.columns]
    if missing:
        raise ValueError(f"Signals DataFrame missing required columns: {missing}")

    # Make a copy to avoid modifying original
    result = signals.copy()

    # Compute confidence scores
    try:
        confidence_scores = meta_model.predict_proba(features)
    except Exception as e:
        logger.error(f"Failed to compute confidence scores: {e}")
        raise ValueError(f"Failed to compute confidence scores: {e}") from e

    # Join confidence scores with signals (same logic as apply_meta_filter)
    if join_keys is not None:
        # Join on specified keys
        if all(k in signals.columns for k in join_keys):
            # Reset index if needed
            if features.index.names != [None]:
                features_reset = features.reset_index()
            else:
                features_reset = features.copy()

            # Merge confidence scores
            confidence_df = pd.DataFrame(
                {
                    "meta_confidence": confidence_scores.values,
                    **{
                        k: features_reset[k].values
                        for k in join_keys
                        if k in features_reset.columns
                    },
                }
            )

            result = result.merge(confidence_df, on=join_keys, how="left")
        else:
            raise ValueError(
                f"Cannot join on keys {join_keys}: not all keys present in signals"
            )
    else:
        # Use index alignment
        if len(result) != len(confidence_scores):
            raise ValueError(
                f"Signals and features must have same length for index alignment. "
                f"Got {len(result)} signals and {len(confidence_scores)} confidence scores"
            )
        result["meta_confidence"] = confidence_scores.values

    # Fill NaN confidence scores with 0.0
    result["meta_confidence"] = result["meta_confidence"].fillna(0.0)

    # Clip confidence to [0, max_scaling] for scaling
    result["meta_confidence_clipped"] = result["meta_confidence"].clip(0.0, max_scaling)

    # Apply filter: set direction to "FLAT" for signals with confidence < min_confidence
    long_mask = result["direction"] == "LONG"
    low_confidence_mask = result["meta_confidence"] < min_confidence

    filtered_count = (long_mask & low_confidence_mask).sum()
    if filtered_count > 0:
        logger.info(
            f"Filtering {filtered_count} signals with confidence < {min_confidence}"
        )
        result.loc[long_mask & low_confidence_mask, "direction"] = "FLAT"

    # Scale scores if requested
    if scale_score and "score" in result.columns:
        # Scale original score by confidence
        result["final_score"] = result["score"] * result["meta_confidence_clipped"]
        logger.info(f"Scaled {len(result)} signal scores by meta-confidence")
    elif scale_score:
        # If no score column, create one from confidence
        result["final_score"] = result["meta_confidence_clipped"]
        logger.info(
            "Created final_score from meta-confidence (no original score column)"
        )

    # Clean up temporary column
    if "meta_confidence_clipped" in result.columns:
        result = result.drop(columns=["meta_confidence_clipped"])

    return result
