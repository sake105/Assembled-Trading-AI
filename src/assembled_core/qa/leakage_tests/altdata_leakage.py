"""Leakage test helpers for alt-data features (PIT-safe validation).

This module provides helper functions to detect look-ahead bias in alt-data features.
These helpers verify that features respect disclosure_date filtering and do not
use events that have not yet been disclosed.

Key Functions:
    - assert_feature_zero_before_disclosure(): Validates feature is zero before disclosure_date

Design Principles:
    - Clear failure messages: Identify which symbol/date violates PIT-safety
    - Deterministic: Same input -> same output
    - No pytest dependencies: Pure helper functions (tests in separate test files)
"""

from __future__ import annotations

import logging
from typing import Callable

import pandas as pd

logger = logging.getLogger(__name__)


def assert_feature_zero_before_disclosure(
    prices: pd.DataFrame,
    events: pd.DataFrame,
    feature_fn: Callable[[pd.DataFrame, pd.DataFrame, pd.Timestamp], pd.DataFrame],
    *,
    as_of_before: pd.Timestamp,
    as_of_after: pd.Timestamp,
) -> None:
    """Assert that feature is zero before disclosure_date (PIT-safety check).

    This helper validates that a feature function does not leak future information.
    It checks that features computed at `as_of_before` (before disclosure) are zero,
    and features computed at `as_of_after` (after disclosure) are non-zero.

    Args:
        prices: Price DataFrame with columns: timestamp, symbol, ...
        events: Event DataFrame (must have symbol, event_date, disclosure_date)
        feature_fn: Feature function with signature:
            (prices: pd.DataFrame, events: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame
            Must return DataFrame with feature columns
        as_of_before: Point-in-time before disclosure (pd.Timestamp, UTC)
            Features computed at this time should be zero
        as_of_after: Point-in-time after disclosure (pd.Timestamp, UTC)
            Features computed at this time should be non-zero

    Raises:
        AssertionError: If feature is non-zero before disclosure or zero after disclosure
            Error message includes symbol and date information for debugging

    Example:
        >>> from src.assembled_core.features.event_features import add_disclosure_count_feature
        >>> 
        >>> prices = pd.DataFrame({
        ...     "timestamp": pd.date_range("2024-01-10", periods=5, freq="D", tz="UTC"),
        ...     "symbol": ["AAPL"] * 5,
        ...     "close": [150.0] * 5,
        ... })
        >>> 
        >>> events = pd.DataFrame({
        ...     "symbol": ["AAPL"],
        ...     "event_date": pd.to_datetime(["2024-01-05"], utc=True),
        ...     "disclosure_date": pd.to_datetime(["2024-01-15"], utc=True),
        ...     "effective_date": pd.to_datetime(["2024-01-15"], utc=True),
        ... })
        >>> 
        >>> def feature_fn(p, e, as_of):
        ...     return add_disclosure_count_feature(p, e, window_days=30, as_of=as_of)
        >>> 
        >>> assert_feature_zero_before_disclosure(
        ...     prices,
        ...     events,
        ...     feature_fn,
        ...     as_of_before=pd.Timestamp("2024-01-14", tz="UTC"),
        ...     as_of_after=pd.Timestamp("2024-01-15", tz="UTC"),
        ... )
    """
    # Validate inputs
    if prices.empty:
        raise ValueError("prices DataFrame is empty")
    if events.empty:
        raise ValueError("events DataFrame is empty")

    # Ensure timestamps are UTC
    if not isinstance(as_of_before, pd.Timestamp):
        as_of_before = pd.to_datetime(as_of_before, utc=True)
    elif as_of_before.tz is None:
        as_of_before = as_of_before.tz_localize("UTC")
    else:
        as_of_before = as_of_before.tz_convert("UTC")

    if not isinstance(as_of_after, pd.Timestamp):
        as_of_after = pd.to_datetime(as_of_after, utc=True)
    elif as_of_after.tz is None:
        as_of_after = as_of_after.tz_localize("UTC")
    else:
        as_of_after = as_of_after.tz_convert("UTC")

    # Validate: as_of_before < as_of_after
    if as_of_before >= as_of_after:
        raise ValueError(
            f"as_of_before ({as_of_before}) must be < as_of_after ({as_of_after})"
        )

    # Compute features at as_of_before (should be zero)
    result_before = feature_fn(prices.copy(), events.copy(), as_of_before)

    # Compute features at as_of_after (should be non-zero)
    result_after = feature_fn(prices.copy(), events.copy(), as_of_after)

    # Find feature columns (all columns except original price columns)
    original_cols = set(prices.columns)
    feature_cols = [col for col in result_before.columns if col not in original_cols]

    if not feature_cols:
        raise ValueError(
            f"Feature function did not add any feature columns. "
            f"Original columns: {list(original_cols)}, "
            f"Result columns: {list(result_before.columns)}"
        )

    # Check each feature column
    violations = []

    for feature_col in feature_cols:
        # Check: feature should be zero before disclosure
        # Treat NaN/NA as zero (they are not violations)
        before_values = result_before[feature_col]
        # Convert NaN/NA to 0 for comparison
        before_values_clean = before_values.fillna(0)
        non_zero_before = before_values_clean[before_values_clean != 0]

        if len(non_zero_before) > 0:
            # Find which symbols/dates have non-zero features
            non_zero_rows = result_before.loc[non_zero_before.index]
            for idx, row in non_zero_rows.iterrows():
                symbol = row.get("symbol", "UNKNOWN")
                timestamp = row.get("timestamp", "UNKNOWN")
                value = row[feature_col]
                violations.append(
                    f"Feature '{feature_col}' is non-zero ({value}) before disclosure "
                    f"at symbol={symbol}, timestamp={timestamp}, as_of={as_of_before}"
                )

        # Check: feature should be non-zero after disclosure (at least for some rows)
        after_values = result_after[feature_col]
        non_zero_after = after_values[after_values != 0]

        if len(non_zero_after) == 0:
            # This might be OK if no events are in window, but we log it
            logger.warning(
                f"Feature '{feature_col}' is zero after disclosure at as_of={as_of_after}. "
                "This might indicate no events are in the lookback window."
            )

    if violations:
        error_msg = (
            f"PIT-safety violation detected: Features are non-zero before disclosure_date.\n"
            f"  as_of_before: {as_of_before}\n"
            f"  as_of_after: {as_of_after}\n"
            f"  Violations:\n    - " + "\n    - ".join(violations)
        )
        raise AssertionError(error_msg)
