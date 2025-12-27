"""Tests for PIT-safe event feature builders (B2 integration)."""

from __future__ import annotations

import pandas as pd

from src.assembled_core.features.congress_features import add_congress_features
from src.assembled_core.features.event_features import build_event_feature_panel
from src.assembled_core.features.insider_features import add_insider_features


def test_build_event_feature_panel_pit_safe() -> None:
    """Test that build_event_feature_panel enforces PIT safety."""
    # Create events: event on 2025-01-05, disclosed on 2025-01-08
    events = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-01-05", tz="UTC")],
        "event_date": [pd.Timestamp("2025-01-05", tz="UTC")],
        "disclosure_date": [pd.Timestamp("2025-01-08", tz="UTC")],
        "symbol": ["AAPL"],
        "value": [100.0],
    })

    # Create prices: dates before and after disclosure
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-06", periods=5, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "close": [100.0] * 5,
    })

    # Build features with as_of=2025-01-07 (before disclosure)
    features_before = build_event_feature_panel(
        events, prices, as_of=pd.Timestamp("2025-01-07", tz="UTC"), lookback_days=30
    )

    # Event should not be visible (disclosure_date > as_of)
    assert features_before["event_count_30d"].sum() == 0, (
        "Feature must be blind to events not yet disclosed"
    )

    # Build features with as_of=2025-01-08 (on disclosure)
    features_on = build_event_feature_panel(
        events, prices, as_of=pd.Timestamp("2025-01-08", tz="UTC"), lookback_days=30
    )

    # Event should now be visible (disclosure_date <= as_of)
    assert features_on["event_count_30d"].sum() > 0, (
        "Event should be visible after disclosure_date"
    )


def test_add_insider_features_pit_safe() -> None:
    """Test that add_insider_features enforces PIT safety."""
    # Create insider events: trade on 2025-01-05, disclosed on 2025-01-08 (T+3)
    events = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-01-05", tz="UTC")],
        "symbol": ["AAPL"],
        "net_shares": [1000.0],
        "trades_count": [1],
    })

    # Create prices: dates before and after disclosure
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-06", periods=5, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 5,
        "close": [100.0] * 5,
    })

    # Build features with as_of=2025-01-07 (before disclosure, default latency=2 days)
    # Default disclosure_latency_days=2 means disclosure_date = timestamp + 2 days = 2025-01-07
    features_before = add_insider_features(
        prices, events, as_of=pd.Timestamp("2025-01-06", tz="UTC"), disclosure_latency_days=2
    )

    # Event disclosed on 2025-01-07, so as_of=2025-01-06 should exclude it
    # (disclosure_date 2025-01-07 > as_of 2025-01-06)
    assert features_before["insider_net_buy_20d"].iloc[0] == 0.0, (
        "Feature must be blind before disclosure"
    )

    # Build features with as_of=2025-01-07 (on disclosure)
    features_on = add_insider_features(
        prices, events, as_of=pd.Timestamp("2025-01-07", tz="UTC"), disclosure_latency_days=2
    )

    # Event should now be visible (disclosure_date <= as_of)
    assert features_on["insider_net_buy_20d"].iloc[-1] == 1000.0, (
        "Event should be visible after disclosure_date"
    )


def test_add_congress_features_pit_safe() -> None:
    """Test that add_congress_features enforces PIT safety."""
    # Create congress events: trade on 2025-01-05, disclosed on 2025-01-15 (T+10)
    events = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-01-05", tz="UTC")],
        "symbol": ["AAPL"],
        "amount": [50000.0],
    })

    # Create prices: dates before and after disclosure
    prices = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-06", periods=15, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 15,
        "close": [100.0] * 15,
    })

    # Build features with as_of=2025-01-10 (before disclosure, default latency=10 days)
    # Default disclosure_latency_days=10 means disclosure_date = timestamp + 10 days = 2025-01-15
    features_before = add_congress_features(
        prices, events, as_of=pd.Timestamp("2025-01-10", tz="UTC"), disclosure_latency_days=10
    )

    # Event disclosed on 2025-01-15, so as_of=2025-01-10 should exclude it
    assert features_before["congress_trade_count_60d"].iloc[0] == 0, (
        "Feature must be blind before disclosure"
    )
    assert features_before["congress_total_amount_60d"].iloc[0] == 0.0

    # Build features with as_of=2025-01-15 (on disclosure)
    features_on = add_congress_features(
        prices, events, as_of=pd.Timestamp("2025-01-15", tz="UTC"), disclosure_latency_days=10
    )

    # Event should now be visible (disclosure_date <= as_of)
    assert features_on["congress_trade_count_60d"].iloc[-1] > 0, (
        "Event should be visible after disclosure_date"
    )
    assert features_on["congress_total_amount_60d"].iloc[-1] == 50000.0


def test_build_event_feature_panel_with_explicit_disclosure_date() -> None:
    """Test build_event_feature_panel with explicit disclosure_date."""
    events = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-01-05", tz="UTC")],
        "event_date": [pd.Timestamp("2025-01-05", tz="UTC")],
        "disclosure_date": [pd.Timestamp("2025-01-08", tz="UTC")],  # Explicit disclosure
        "symbol": ["AAPL"],
        "value": [100.0],
    })

    prices = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-07", periods=3, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 3,
        "close": [100.0] * 3,
    })

    # as_of before disclosure
    features = build_event_feature_panel(
        events, prices, as_of=pd.Timestamp("2025-01-07", tz="UTC"), lookback_days=30
    )

    # Should exclude event (disclosure_date 2025-01-08 > as_of 2025-01-07)
    assert features["event_count_30d"].sum() == 0


def test_build_event_feature_panel_derives_disclosure_date() -> None:
    """Test build_event_feature_panel derives disclosure_date if missing."""
    events = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-01-05", tz="UTC")],
        "symbol": ["AAPL"],
        "value": [100.0],
        # No disclosure_date - should be derived from timestamp + latency
    })

    prices = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-06", periods=3, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 3,
        "close": [100.0] * 3,
    })

    # disclosure_latency_days=2 means disclosure_date = timestamp + 2 = 2025-01-07
    features = build_event_feature_panel(
        events,
        prices,
        as_of=pd.Timestamp("2025-01-07", tz="UTC"),
        lookback_days=30,
        disclosure_latency_days=2,
    )

    # Event should be visible (derived disclosure_date 2025-01-07 <= as_of 2025-01-07)
    assert features["event_count_30d"].iloc[-1] > 0


def test_event_features_backward_compatible_no_as_of() -> None:
    """Test that event features work without as_of (backward compatibility)."""
    events = pd.DataFrame({
        "timestamp": [pd.Timestamp("2025-01-05", tz="UTC")],
        "symbol": ["AAPL"],
        "net_shares": [1000.0],
        "trades_count": [1],
    })

    prices = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-06", periods=3, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 3,
        "close": [100.0] * 3,
    })

    # Should work without as_of (uses all events)
    features = add_insider_features(prices, events, as_of=None)

    assert "insider_net_buy_20d" in features.columns
    assert features["insider_net_buy_20d"].iloc[-1] == 1000.0

