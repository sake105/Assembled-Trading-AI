"""Tests for Point-in-Time safety checks (B2 Runtime Guards).

These tests verify that the PIT safety guards correctly detect
look-ahead bias violations in feature computation.
"""

from __future__ import annotations

import os
import pytest

import pandas as pd

from src.assembled_core.qa.point_in_time_checks import (
    PointInTimeViolationError,
    check_altdata_events_pit_safe,
    check_features_pit_safe,
    validate_feature_builder_pit_safe,
)

pytestmark = pytest.mark.advanced


@pytest.fixture
def sample_features_with_future_data() -> pd.DataFrame:
    """Create sample features DataFrame with intentional future data.

    Returns:
        DataFrame with timestamps from 2024-01-01 to 2024-01-15,
        where some timestamps are intentionally after as_of=2024-01-10
    """
    dates = pd.date_range(start="2024-01-01", end="2024-01-15", freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * len(dates),
            "factor_mom": [0.1 * i for i in range(len(dates))],
        }
    )


@pytest.fixture
def sample_events_with_future_disclosure() -> pd.DataFrame:
    """Create sample Alt-Data events with intentional future disclosure dates.

    Returns:
        DataFrame with events where some disclosure_date > as_of
    """
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "disclosure_date": pd.date_range(
                "2024-01-05", "2024-01-14", freq="D", tz="UTC"
            ),
            "event_type": ["earnings"] * 10,
        }
    )


def test_check_features_pit_safe_passes():
    """Test that PIT check passes when all timestamps <= as_of."""
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "factor_mom": [0.1 * i for i in range(10)],
        }
    )

    as_of = pd.Timestamp("2024-01-10", tz="UTC")

    result = check_features_pit_safe(
        features_df=features,
        as_of=as_of,
        strict=False,
    )

    assert result is True


def test_check_features_pit_safe_detects_violation_non_strict(
    sample_features_with_future_data,
):
    """Test that PIT check detects violations in non-strict mode (logs warning)."""
    as_of = pd.Timestamp("2024-01-10", tz="UTC")

    result = check_features_pit_safe(
        features_df=sample_features_with_future_data,
        as_of=as_of,
        strict=False,
        feature_source="test_builder",
    )

    assert result is False  # Violation detected, but no exception raised


def test_check_features_pit_safe_detects_violation_strict(
    sample_features_with_future_data,
):
    """Test that PIT check raises exception in strict mode."""
    as_of = pd.Timestamp("2024-01-10", tz="UTC")

    with pytest.raises(PointInTimeViolationError) as exc_info:
        check_features_pit_safe(
            features_df=sample_features_with_future_data,
            as_of=as_of,
            strict=True,
            feature_source="test_builder",
        )

    error_msg = str(exc_info.value)
    assert "Point-in-Time violation" in error_msg
    assert "test_builder" in error_msg
    assert "2024-01-10" in error_msg  # as_of
    assert "2024-01-15" in error_msg or "2024-01-11" in error_msg  # violation timestamp


def test_check_features_pit_safe_no_as_of():
    """Test that PIT check passes when as_of is None."""
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", "2024-01-15", freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 15,
            "factor_mom": [0.1 * i for i in range(15)],
        }
    )

    result = check_features_pit_safe(
        features_df=features,
        as_of=None,
        strict=True,
    )

    assert result is True


def test_check_altdata_events_pit_safe_passes():
    """Test that Alt-Data events PIT check passes when all disclosure_date <= as_of."""
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "disclosure_date": pd.date_range(
                "2024-01-01", "2024-01-10", freq="D", tz="UTC"
            ),
            "event_type": ["earnings"] * 10,
        }
    )

    as_of = pd.Timestamp("2024-01-10", tz="UTC")

    result = check_altdata_events_pit_safe(
        events_df=events,
        as_of=as_of,
        strict=False,
    )

    assert result is True


def test_check_altdata_events_pit_safe_detects_violation_strict(
    sample_events_with_future_disclosure,
):
    """Test that Alt-Data events PIT check raises exception in strict mode."""
    as_of = pd.Timestamp("2024-01-10", tz="UTC")

    with pytest.raises(PointInTimeViolationError) as exc_info:
        check_altdata_events_pit_safe(
            events_df=sample_events_with_future_disclosure,
            as_of=as_of,
            strict=True,
            event_source="test_earnings_events",
        )

    error_msg = str(exc_info.value)
    assert "Point-in-Time violation" in error_msg
    assert "test_earnings_events" in error_msg
    assert "disclosure_date" in error_msg.lower()


def test_validate_feature_builder_pit_safe():
    """Test convenience wrapper for feature builder validation."""
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "factor_mom": [0.1 * i for i in range(10)],
        }
    )

    as_of = pd.Timestamp("2024-01-10", tz="UTC")

    result = validate_feature_builder_pit_safe(
        features_df=features,
        as_of=as_of,
        builder_name="test_build_factors",
        strict=False,
    )

    assert result is True


def test_validate_feature_builder_pit_safe_with_violation():
    """Test that feature builder validation detects violations."""
    features = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", "2024-01-15", freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 15,
            "factor_mom": [0.1 * i for i in range(15)],
        }
    )

    as_of = pd.Timestamp("2024-01-10", tz="UTC")

    with pytest.raises(PointInTimeViolationError) as exc_info:
        validate_feature_builder_pit_safe(
            features_df=features,
            as_of=as_of,
            builder_name="test_build_factors",
            strict=True,
        )

    error_msg = str(exc_info.value)
    assert "test_build_factors" in error_msg


def test_pit_check_missing_timestamp_column():
    """Test that PIT check handles missing timestamp column gracefully."""
    features = pd.DataFrame(
        {
            "symbol": ["AAPL"] * 10,
            "factor_mom": [0.1 * i for i in range(10)],
            # Missing "timestamp" column
        }
    )

    as_of = pd.Timestamp("2024-01-10", tz="UTC")

    # Should log warning and return True (skip check)
    result = check_features_pit_safe(
        features_df=features,
        as_of=as_of,
        strict=False,
    )

    assert result is True


def test_pit_check_missing_disclosure_date_column():
    """Test that Alt-Data events check handles missing disclosure_date gracefully."""
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "event_type": ["earnings"] * 10,
            # Missing "disclosure_date" column
        }
    )

    as_of = pd.Timestamp("2024-01-10", tz="UTC")

    # Should log debug and return True (skip check)
    result = check_altdata_events_pit_safe(
        events_df=events,
        as_of=as_of,
        strict=False,
    )

    assert result is True


def test_feature_builder_guard_with_strict_mode():
    """Test that feature builder guards catch violations when ASSEMBLED_STRICT_PIT_CHECKS is enabled.

    This test simulates what happens when a feature builder is called with
    prices that extend beyond as_of, which would cause the guard to detect
    a violation (features with timestamps > as_of).

    Note: The feature builder correctly filters events by disclosure_date,
    but the resulting features DataFrame still contains price timestamps
    that extend beyond as_of. The guard should catch this.
    """
    from src.assembled_core.features.altdata_earnings_insider_factors import (
        build_earnings_surprise_factors,
    )

    # Create prices with timestamps up to 2024-01-15 (beyond as_of=2024-01-10)
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", "2024-01-15", freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 15,
            "close": [150.0 + i * 0.1 for i in range(15)],
        }
    )

    # Create events with disclosure_date <= as_of (correctly filtered)
    events = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 10,
            "event_type": ["earnings"] * 10,
            "event_date": pd.date_range("2024-01-01", "2024-01-10", freq="D", tz="UTC"),
            "disclosure_date": pd.date_range(
                "2024-01-01", "2024-01-10", freq="D", tz="UTC"
            ),
            "eps_actual": [2.0] * 10,
            "eps_estimate": [1.9] * 10,
            "eps_surprise_pct": [5.0] * 10,
        }
    )

    # Set as_of to 2024-01-10 (prices extend to 2024-01-15)
    as_of = pd.Timestamp("2024-01-10", tz="UTC")

    # Enable strict PIT checks
    original_env = os.environ.get("ASSEMBLED_STRICT_PIT_CHECKS", None)
    try:
        os.environ["ASSEMBLED_STRICT_PIT_CHECKS"] = "true"

        # The feature builder will create factors for all price timestamps,
        # including those > as_of. The guard should catch this violation.
        with pytest.raises(PointInTimeViolationError) as exc_info:
            build_earnings_surprise_factors(
                events_earnings=events,
                prices=prices,
                as_of=as_of,
            )

        error_msg = str(exc_info.value)
        assert "build_earnings_surprise_factors" in error_msg
        assert "2024-01-10" in error_msg  # as_of
        assert "2024-01-15" in error_msg or "2024-01-11" in error_msg  # violation

    finally:
        # Restore original environment
        if original_env is None:
            os.environ.pop("ASSEMBLED_STRICT_PIT_CHECKS", None)
        else:
            os.environ["ASSEMBLED_STRICT_PIT_CHECKS"] = original_env
