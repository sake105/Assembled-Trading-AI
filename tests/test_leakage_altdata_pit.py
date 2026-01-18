"""Leakage tests for alt-data features (PIT-safety validation, Sprint 10.C).

These tests verify that alt-data features do not leak future information.
Features must be zero before disclosure_date and non-zero after disclosure_date.

Tests are mandatory and run in CI (no markers, always executed).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.features.event_features import (
    add_disclosure_count_feature,
    build_event_feature_panel,
)
from src.assembled_core.qa.leakage_tests.altdata_leakage import (
    assert_feature_zero_before_disclosure,
)


def test_future_event_inserted_feature_remains_zero() -> None:
    """Test that event with disclosure far in future -> feature remains 0 before disclosure."""
    # Create synthetic prices
    dates = pd.date_range("2024-01-10", periods=10, freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    # Create event with disclosure_date far in future
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-05"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-02-01"], utc=True),  # Far in future
        "effective_date": pd.to_datetime(["2024-02-01"], utc=True),
    })

    # Define feature function
    def feature_fn(p, e, as_of):
        return add_disclosure_count_feature(p, e, window_days=30, as_of=as_of)

    # Assert: feature is zero before disclosure, non-zero after
    assert_feature_zero_before_disclosure(
        prices,
        events,
        feature_fn,
        as_of_before=pd.Timestamp("2024-01-31", tz="UTC"),  # Before disclosure
        as_of_after=pd.Timestamp("2024-02-01", tz="UTC"),  # On disclosure
    )


def test_late_arrival_event_date_old_disclosure_late() -> None:
    """Test late arrival: event_date old, disclosure_date late -> 0 before, >0 after."""
    # Create synthetic prices
    dates = pd.date_range("2024-01-10", periods=10, freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    # Create event with old event_date but late disclosure_date (late arrival)
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-01"], utc=True),  # Old
        "disclosure_date": pd.to_datetime(["2024-01-15"], utc=True),  # Late
        "effective_date": pd.to_datetime(["2024-01-15"], utc=True),
    })

    # Define feature function
    def feature_fn(p, e, as_of):
        return add_disclosure_count_feature(p, e, window_days=30, as_of=as_of)

    # Assert: feature is zero before disclosure, non-zero after
    assert_feature_zero_before_disclosure(
        prices,
        events,
        feature_fn,
        as_of_before=pd.Timestamp("2024-01-14", tz="UTC"),  # Before disclosure
        as_of_after=pd.Timestamp("2024-01-15", tz="UTC"),  # On disclosure
    )


def test_multiple_symbols_only_affected_symbol_rises() -> None:
    """Test multiple symbols: only affected symbol rises, others remain 0."""
    # Create synthetic prices for two symbols
    dates = pd.date_range("2024-01-10", periods=5, freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates.tolist() * 2,
        "symbol": ["AAPL"] * 5 + ["MSFT"] * 5,
        "close": [150.0] * 10,
    })

    # Create event only for AAPL (not MSFT)
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-05"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-15"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-15"], utc=True),
    })

    # Define feature function
    def feature_fn(p, e, as_of):
        return add_disclosure_count_feature(p, e, window_days=30, as_of=as_of)

    # Assert: feature is zero before disclosure for both symbols
    # After disclosure, only AAPL should have non-zero feature
    assert_feature_zero_before_disclosure(
        prices,
        events,
        feature_fn,
        as_of_before=pd.Timestamp("2024-01-14", tz="UTC"),  # Before disclosure
        as_of_after=pd.Timestamp("2024-01-15", tz="UTC"),  # On disclosure
    )

    # Additional check: MSFT should remain zero even after disclosure
    result_after = feature_fn(prices.copy(), events.copy(), pd.Timestamp("2024-01-15", tz="UTC"))
    msft_features = result_after[result_after["symbol"] == "MSFT"]["alt_disclosure_count_30d_v1"]
    assert (msft_features == 0).all(), "MSFT should have zero features (no events)"


def test_build_event_feature_panel_leakage() -> None:
    """Test that build_event_feature_panel does not leak future information."""
    # Create synthetic prices
    dates = pd.date_range("2024-01-10", periods=10, freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    # Create events with different disclosure dates
    events = pd.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "event_date": pd.to_datetime(["2024-01-05", "2024-01-06"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-12", "2024-01-18"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-12", "2024-01-18"], utc=True),
        "value": [1000.0, 2000.0],
    })

    # Define feature function
    def feature_fn(p, e, as_of):
        return build_event_feature_panel(e, p, as_of=as_of, lookback_days=30)

    # Assert: feature is zero before disclosure, non-zero after
    assert_feature_zero_before_disclosure(
        prices,
        events,
        feature_fn,
        as_of_before=pd.Timestamp("2024-01-11", tz="UTC"),  # Before first disclosure
        as_of_after=pd.Timestamp("2024-01-12", tz="UTC"),  # On first disclosure
    )


def test_leakage_detection_clear_error_message() -> None:
    """Test that leakage detection provides clear error messages."""
    # Create synthetic prices
    dates = pd.date_range("2024-01-10", periods=5, freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 5,
        "close": [150.0] * 5,
    })

    # Create event with late disclosure
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-05"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-15"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-15"], utc=True),
    })

    # Define a "leaky" feature function (uses event_date instead of disclosure_date)
    def leaky_feature_fn(p, e, as_of):
        # This is intentionally leaky for testing error messages
        # In real code, this would be a bug
        result = p.copy()
        result["leaky_feature"] = 0

        # Leak: use event_date instead of disclosure_date
        for idx, row in result.iterrows():
            price_time = row["timestamp"]
            symbol = row["symbol"]
            symbol_events = e[e["symbol"] == symbol]
            # BUG: Using event_date instead of disclosure_date
            leaky_events = symbol_events[
                symbol_events["event_date"] <= price_time.normalize()
            ]
            result.loc[idx, "leaky_feature"] = len(leaky_events)

        return result

    # This should raise AssertionError with clear message
    with pytest.raises(AssertionError) as exc_info:
        assert_feature_zero_before_disclosure(
            prices,
            events,
            leaky_feature_fn,
            as_of_before=pd.Timestamp("2024-01-14", tz="UTC"),
            as_of_after=pd.Timestamp("2024-01-15", tz="UTC"),
        )

    # Verify error message contains useful information
    error_msg = str(exc_info.value)
    assert "PIT-safety violation" in error_msg
    assert "symbol" in error_msg.lower() or "AAPL" in error_msg
    assert "timestamp" in error_msg.lower() or "2024-01" in error_msg
