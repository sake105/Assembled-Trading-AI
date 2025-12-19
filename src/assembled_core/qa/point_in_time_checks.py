"""Point-in-Time (PIT) safety checks for Alt-Data features.

This module provides runtime guards to detect look-ahead bias in feature
computation. These checks can be enabled in strict QA mode to catch
point-in-time violations early.
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class PointInTimeViolationError(Exception):
    """Raised when a point-in-time safety check fails.
    
    This indicates that features contain data from timestamps after
    the `as_of` date, which would cause look-ahead bias.
    """
    pass


def check_features_pit_safe(
    features_df: pd.DataFrame,
    as_of: pd.Timestamp | None,
    timestamp_col: str = "timestamp",
    strict: bool = False,
    feature_source: str | None = None,
) -> bool:
    """Check that all feature timestamps are <= as_of (point-in-time safe).
    
    Args:
        features_df: DataFrame with features (must contain timestamp_col)
        as_of: Maximum allowed timestamp (None = no check)
        timestamp_col: Name of timestamp column (default: "timestamp")
        strict: If True, raise PointInTimeViolationError on violation.
                If False, log warning and return False.
        feature_source: Optional string describing feature source (for logging)
    
    Returns:
        True if all timestamps <= as_of, False otherwise
    
    Raises:
        PointInTimeViolationError: If strict=True and violation detected
    
    Example:
        >>> features = pd.DataFrame({
        ...     "timestamp": pd.date_range("2024-01-01", "2024-01-10", tz="UTC"),
        ...     "factor_mom": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ... })
        >>> check_features_pit_safe(features, pd.Timestamp("2024-01-05", tz="UTC"))
        False  # Some timestamps > as_of
        >>> check_features_pit_safe(features, pd.Timestamp("2024-01-15", tz="UTC"))
        True  # All timestamps <= as_of
    """
    if as_of is None:
        # No check requested
        return True
    
    if timestamp_col not in features_df.columns:
        logger.warning(
            "PIT check skipped: timestamp column '%s' not found in features DataFrame",
            timestamp_col,
        )
        return True
    
    # Normalize as_of to UTC if needed
    as_of_ts = pd.to_datetime(as_of, utc=True).normalize()
    
    # Get all timestamps from features
    timestamps = pd.to_datetime(features_df[timestamp_col], utc=True)
    
    # Find violations (timestamps > as_of)
    violations = timestamps > as_of_ts
    
    if violations.any():
        n_violations = violations.sum()
        max_timestamp = timestamps.max()
        min_violation = timestamps[violations].min()
        
        source_info = f" (source: {feature_source})" if feature_source else ""
        
        error_msg = (
            f"Point-in-Time violation detected{source_info}:\n"
            f"  - as_of: {as_of_ts}\n"
            f"  - Max feature timestamp: {max_timestamp}\n"
            f"  - First violation: {min_violation}\n"
            f"  - Number of violations: {n_violations} / {len(features_df)}"
        )
        
        if strict:
            raise PointInTimeViolationError(error_msg)
        else:
            logger.warning("PIT CHECK: %s", error_msg)
            return False
    
    # All timestamps are <= as_of
    logger.debug(
        "PIT check passed: all %d timestamps <= %s%s",
        len(features_df),
        as_of_ts,
        f" (source: {feature_source})" if feature_source else "",
    )
    return True


def check_altdata_events_pit_safe(
    events_df: pd.DataFrame,
    as_of: pd.Timestamp | None,
    disclosure_date_col: str = "disclosure_date",
    event_source: str | None = None,
    strict: bool = False,
) -> bool:
    """Check that all event disclosure dates are <= as_of.
    
    This is a specialized check for Alt-Data event DataFrames that have
    explicit disclosure_date columns.
    
    Args:
        events_df: DataFrame with Alt-Data events (must contain disclosure_date_col)
        as_of: Maximum allowed disclosure_date (None = no check)
        disclosure_date_col: Name of disclosure_date column (default: "disclosure_date")
        event_source: Optional string describing event source (for logging)
        strict: If True, raise PointInTimeViolationError on violation.
                If False, log warning and return False.
    
    Returns:
        True if all disclosure_date <= as_of, False otherwise
    
    Raises:
        PointInTimeViolationError: If strict=True and violation detected
    """
    if as_of is None:
        return True
    
    if disclosure_date_col not in events_df.columns:
        logger.debug(
            "PIT check skipped: disclosure_date column '%s' not found in events DataFrame",
            disclosure_date_col,
        )
        return True
    
    # Normalize as_of to UTC if needed
    as_of_ts = pd.to_datetime(as_of, utc=True).normalize()
    
    # Get all disclosure dates
    disclosure_dates = pd.to_datetime(events_df[disclosure_date_col], utc=True)
    
    # Find violations (disclosure_date > as_of)
    violations = disclosure_dates > as_of_ts
    
    if violations.any():
        n_violations = violations.sum()
        max_disclosure = disclosure_dates.max()
        min_violation = disclosure_dates[violations].min()
        
        source_info = f" (source: {event_source})" if event_source else ""
        
        error_msg = (
            f"Point-in-Time violation in events{source_info}:\n"
            f"  - as_of: {as_of_ts}\n"
            f"  - Max disclosure_date: {max_disclosure}\n"
            f"  - First violation: {min_violation}\n"
            f"  - Number of violations: {n_violations} / {len(events_df)}"
        )
        
        if strict:
            raise PointInTimeViolationError(error_msg)
        else:
            logger.warning("PIT CHECK: %s", error_msg)
            return False
    
    logger.debug(
        "PIT check passed: all %d events have disclosure_date <= %s%s",
        len(events_df),
        as_of_ts,
        f" (source: {event_source})" if event_source else "",
    )
    return True


def validate_feature_builder_pit_safe(
    features_df: pd.DataFrame,
    as_of: pd.Timestamp | None,
    builder_name: str,
    strict: bool = False,
) -> bool:
    """Convenience wrapper for checking feature builder output.
    
    This function is intended to be called after building Alt-Data features
    to ensure they respect the as_of parameter.
    
    Args:
        features_df: DataFrame with features from a feature builder
        as_of: Maximum allowed timestamp (should match builder's as_of parameter)
        builder_name: Name of feature builder (e.g., "build_earnings_surprise_factors")
        strict: If True, raise PointInTimeViolationError on violation.
                If False, log warning and return False.
    
    Returns:
        True if all timestamps <= as_of, False otherwise
    
    Raises:
        PointInTimeViolationError: If strict=True and violation detected
    """
    return check_features_pit_safe(
        features_df=features_df,
        as_of=as_of,
        timestamp_col="timestamp",
        strict=strict,
        feature_source=builder_name,
    )

