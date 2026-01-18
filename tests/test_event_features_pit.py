"""Tests for PIT-safe event features (Sprint 10.B).

Tests verify:
1. Feature is zero before disclosure_date
2. Feature > 0 after disclosure_date (in window)
3. Timezone invariance (naive vs aware input identical)
4. PIT filtering respects disclosure_date <= as_of
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


def test_feature_zero_before_disclosure() -> None:
    """Test that feature is zero before disclosure_date."""
    # Create synthetic prices (10 days, 1 symbol)
    dates = pd.date_range("2024-01-10", periods=10, freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    # Create event with early event_date but late disclosure_date
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-05"], utc=True),  # Early
        "disclosure_date": pd.to_datetime(["2024-01-15"], utc=True),  # Late (after some prices)
        "effective_date": pd.to_datetime(["2024-01-15"], utc=True),
    })

    # Add feature with window_days=30
    result = add_disclosure_count_feature(
        prices,
        events,
        window_days=30,
        as_of=dates[-1],  # Use last date as as_of
    )

    # Verify: feature is zero for prices before disclosure_date (2024-01-15)
    # Prices on 2024-01-10 to 2024-01-14 should have feature = 0
    before_disclosure = result[result["timestamp"] < pd.Timestamp("2024-01-15", tz="UTC")]
    assert (before_disclosure["alt_disclosure_count_30d_v1"] == 0).all(), (
        "Feature should be zero before disclosure_date"
    )


def test_feature_positive_after_disclosure() -> None:
    """Test that feature > 0 after disclosure_date (in window)."""
    # Create synthetic prices (10 days, 1 symbol)
    dates = pd.date_range("2024-01-10", periods=10, freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    # Create event with early event_date but late disclosure_date
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-05"], utc=True),  # Early
        "disclosure_date": pd.to_datetime(["2024-01-15"], utc=True),  # Late (after some prices)
        "effective_date": pd.to_datetime(["2024-01-15"], utc=True),
    })

    # Add feature with window_days=30
    result = add_disclosure_count_feature(
        prices,
        events,
        window_days=30,
        as_of=dates[-1],  # Use last date as as_of
    )

    # Verify: feature > 0 for prices on/after disclosure_date (2024-01-15)
    # Prices on 2024-01-15 to 2024-01-19 should have feature = 1 (within 30-day window)
    after_disclosure = result[result["timestamp"] >= pd.Timestamp("2024-01-15", tz="UTC")]
    assert (after_disclosure["alt_disclosure_count_30d_v1"] > 0).any(), (
        "Feature should be > 0 after disclosure_date (in window)"
    )
    # Specifically, price on 2024-01-15 should have feature = 1
    price_on_disclosure = result[result["timestamp"] == pd.Timestamp("2024-01-15", tz="UTC")]
    assert len(price_on_disclosure) == 1
    assert price_on_disclosure.iloc[0]["alt_disclosure_count_30d_v1"] == 1


def test_timezone_invariance() -> None:
    """Test that naive and tz-aware inputs produce identical results."""
    # Create prices with naive timestamps
    dates_naive = pd.date_range("2024-01-10", periods=5, freq="D")
    prices_naive = pd.DataFrame({
        "timestamp": dates_naive,
        "symbol": ["AAPL"] * 5,
        "close": [150.0] * 5,
    })

    # Create prices with tz-aware timestamps
    dates_tz = pd.date_range("2024-01-10", periods=5, freq="D", tz="UTC")
    prices_tz = pd.DataFrame({
        "timestamp": dates_tz,
        "symbol": ["AAPL"] * 5,
        "close": [150.0] * 5,
    })

    # Create events (tz-aware)
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-05"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-12"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-12"], utc=True),
    })

    # Add feature to both
    result_naive = add_disclosure_count_feature(
        prices_naive,
        events,
        window_days=30,
        as_of=pd.Timestamp("2024-01-14", tz="UTC"),
    )

    result_tz = add_disclosure_count_feature(
        prices_tz,
        events,
        window_days=30,
        as_of=pd.Timestamp("2024-01-14", tz="UTC"),
    )

    # Verify: feature values are identical
    pd.testing.assert_series_equal(
        result_naive["alt_disclosure_count_30d_v1"],
        result_tz["alt_disclosure_count_30d_v1"],
        check_names=False,
    )


def test_pit_filtering_respects_as_of() -> None:
    """Test that PIT filtering respects disclosure_date <= as_of."""
    # Create synthetic prices
    dates = pd.date_range("2024-01-10", periods=10, freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    # Create events with different disclosure dates
    events = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "AAPL"],
        "event_date": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-12", "2024-01-14", "2024-01-18"], utc=True),  # Second event on as_of
        "effective_date": pd.to_datetime(["2024-01-12", "2024-01-14", "2024-01-18"], utc=True),
    })

    # Add feature with as_of = 2024-01-14 (includes first two events)
    result = add_disclosure_count_feature(
        prices,
        events,
        window_days=30,
        as_of=pd.Timestamp("2024-01-14", tz="UTC"),
    )

    # Verify: First two events are counted (disclosure_date <= 2024-01-14)
    # Price on 2024-01-14 should have feature = 2 (both events in window)
    price_on_as_of = result[result["timestamp"] == pd.Timestamp("2024-01-14", tz="UTC")]
    assert len(price_on_as_of) == 1
    assert price_on_as_of.iloc[0]["alt_disclosure_count_30d_v1"] == 2

    # Price on 2024-01-15 should still have feature = 2 (third event not disclosed yet)
    price_after_as_of = result[result["timestamp"] == pd.Timestamp("2024-01-15", tz="UTC")]
    assert len(price_after_as_of) == 1
    assert price_after_as_of.iloc[0]["alt_disclosure_count_30d_v1"] == 2


def test_build_event_feature_panel_pit_safe() -> None:
    """Test that build_event_feature_panel is PIT-safe."""
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

    # Build features with as_of = 2024-01-14 (before second event disclosure)
    result = build_event_feature_panel(
        events,
        prices,
        as_of=pd.Timestamp("2024-01-14", tz="UTC"),
        lookback_days=30,
    )

    # Verify: Only first event is counted (disclosure_date <= 2024-01-14)
    # Price on 2024-01-14 should have count = 1
    price_on_as_of = result[result["timestamp"] == pd.Timestamp("2024-01-14", tz="UTC")]
    assert len(price_on_as_of) == 1
    assert price_on_as_of.iloc[0]["event_count_30d"] == 1
    assert price_on_as_of.iloc[0]["event_sum_30d"] == 1000.0


def test_build_event_feature_panel_requires_as_of() -> None:
    """Test that build_event_feature_panel requires as_of parameter."""
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-10", periods=5, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "close": [150.0] * 5,
    })

    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-05"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-12"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-12"], utc=True),
    })

    # Should raise ValueError if as_of is None
    with pytest.raises(ValueError, match="as_of is required"):
        build_event_feature_panel(events, prices, as_of=None)


def test_window_based_on_disclosure_date() -> None:
    """Test that window is based on disclosure_date, not event_date."""
    # Create synthetic prices
    dates = pd.date_range("2024-01-10", periods=10, freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 10,
        "close": [150.0] * 10,
    })

    # Create event with early event_date but late disclosure_date
    events = pd.DataFrame({
        "symbol": ["AAPL"],
        "event_date": pd.to_datetime(["2024-01-01"], utc=True),  # Very early
        "disclosure_date": pd.to_datetime(["2024-01-15"], utc=True),  # Late
        "effective_date": pd.to_datetime(["2024-01-15"], utc=True),
    })

    # Add feature with window_days=5 (small window)
    result = add_disclosure_count_feature(
        prices,
        events,
        window_days=5,
        as_of=dates[-1],
    )

    # Verify: Feature appears only when disclosure_date is in window
    # Price on 2024-01-15: disclosure_date is in [2024-01-10, 2024-01-15] -> count = 1
    # Price on 2024-01-20: disclosure_date is in [2024-01-15, 2024-01-20] -> count = 1
    # Price on 2024-01-21: disclosure_date is NOT in [2024-01-16, 2024-01-21] -> count = 0

    # Check price on 2024-01-15
    price_15 = result[result["timestamp"] == pd.Timestamp("2024-01-15", tz="UTC")]
    assert len(price_15) == 1
    assert price_15.iloc[0]["alt_disclosure_count_30d_v1"] == 1

    # Check price on 2024-01-20 (if exists)
    if len(result[result["timestamp"] == pd.Timestamp("2024-01-20", tz="UTC")]) > 0:
        price_20 = result[result["timestamp"] == pd.Timestamp("2024-01-20", tz="UTC")]
        assert price_20.iloc[0]["alt_disclosure_count_30d_v1"] == 1


def test_multiple_symbols() -> None:
    """Test that feature works with multiple symbols."""
    # Create synthetic prices for two symbols
    dates = pd.date_range("2024-01-10", periods=5, freq="D", tz="UTC")
    prices = pd.DataFrame({
        "timestamp": dates.tolist() * 2,
        "symbol": ["AAPL"] * 5 + ["MSFT"] * 5,
        "close": [150.0] * 10,
    })

    # Create events for both symbols
    events = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        "event_date": pd.to_datetime(["2024-01-05", "2024-01-06"], utc=True),
        "disclosure_date": pd.to_datetime(["2024-01-12", "2024-01-13"], utc=True),
        "effective_date": pd.to_datetime(["2024-01-12", "2024-01-13"], utc=True),
    })

    # Add feature
    result = add_disclosure_count_feature(
        prices,
        events,
        window_days=30,
        as_of=dates[-1],
    )

    # Verify: Both symbols have features
    assert "alt_disclosure_count_30d_v1" in result.columns
    aapl_features = result[result["symbol"] == "AAPL"]["alt_disclosure_count_30d_v1"]
    msft_features = result[result["symbol"] == "MSFT"]["alt_disclosure_count_30d_v1"]

    # Both should have at least one non-zero value
    assert (aapl_features > 0).any(), "AAPL should have non-zero features"
    assert (msft_features > 0).any(), "MSFT should have non-zero features"
