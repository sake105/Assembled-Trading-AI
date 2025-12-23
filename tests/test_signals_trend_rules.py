# tests/test_signals_trend_rules.py
"""Tests for trend-following signal rules."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.signals.rules_trend import (
    generate_trend_signals,
    generate_trend_signals_from_prices,
)


def create_sample_price_data_with_ma() -> pd.DataFrame:
    """Create sample price data with moving averages already computed."""
    from datetime import datetime, timedelta

    base = datetime(2025, 1, 1, 0, 0, 0)
    symbols = ["AAPL", "MSFT"]
    data = []

    for sym in symbols:
        price_base = 100.0 if sym == "AAPL" else 200.0
        for i in range(50):
            ts = base + timedelta(days=i)
            close_p = price_base + i * 0.1

            # Create artificial MA values where ma_20 > ma_50 for some periods
            ma_20 = close_p + (5.0 if i > 25 else -5.0)  # ma_20 > ma_50 after day 25
            ma_50 = close_p - 2.0

            volume = 1000000.0 + i * 10000.0

            data.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "close": close_p,
                    "ma_20": ma_20,
                    "ma_50": ma_50,
                    "volume": volume,
                }
            )

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def create_sample_price_data_simple() -> pd.DataFrame:
    """Create simple sample price data without MAs."""
    from datetime import datetime, timedelta

    base = datetime(2025, 1, 1, 0, 0, 0)
    symbols = ["AAPL"]
    data = []

    for sym in symbols:
        price_base = 100.0
        for i in range(50):
            ts = base + timedelta(days=i)
            close_p = price_base + i * 0.1
            volume = 1000000.0 + i * 10000.0

            data.append(
                {"timestamp": ts, "symbol": sym, "close": close_p, "volume": volume}
            )

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_generate_trend_signals_long_condition():
    """Test that LONG signals are generated when ma_fast > ma_slow."""
    df = create_sample_price_data_with_ma()
    signals = generate_trend_signals(df, ma_fast=20, ma_slow=50)

    # Check output columns
    assert "timestamp" in signals.columns
    assert "symbol" in signals.columns
    assert "direction" in signals.columns
    assert "score" in signals.columns

    # Check LONG signals exist where ma_20 > ma_50
    long_signals = signals[signals["direction"] == "LONG"]
    assert len(long_signals) > 0, "Should have some LONG signals when ma_20 > ma_50"

    # Check FLAT signals exist where ma_20 <= ma_50
    flat_signals = signals[signals["direction"] == "FLAT"]
    assert len(flat_signals) > 0, "Should have some FLAT signals when ma_20 <= ma_50"

    # Check scores are in [0, 1]
    assert (signals["score"] >= 0).all() and (signals["score"] <= 1.0).all(), (
        "Scores should be in [0, 1]"
    )

    # Check LONG signals have positive scores
    assert (long_signals["score"] > 0).all(), "LONG signals should have positive scores"

    # Check FLAT signals have zero scores
    assert (flat_signals["score"] == 0.0).all(), "FLAT signals should have zero scores"


def test_generate_trend_signals_from_prices():
    """Test generating signals directly from prices."""
    df = create_sample_price_data_simple()
    signals = generate_trend_signals_from_prices(df, ma_fast=20, ma_slow=50)

    # Check output columns
    assert "timestamp" in signals.columns
    assert "symbol" in signals.columns
    assert "direction" in signals.columns
    assert "score" in signals.columns

    # Check we have some signals (after MA windows are filled)
    assert len(signals) > 0, "Should generate signals"

    # Check directions are valid
    assert signals["direction"].isin(["LONG", "FLAT"]).all(), (
        "Directions should be LONG or FLAT"
    )


def test_generate_trend_signals_volume_filter():
    """Test volume filtering in signal generation."""
    df = create_sample_price_data_with_ma()

    # Test with high volume threshold (should filter out most signals)
    signals_high_threshold = generate_trend_signals(
        df,
        ma_fast=20,
        ma_slow=50,
        volume_threshold=5000000.0,  # Very high threshold
    )

    # Test with low volume threshold (should allow more signals)
    signals_low_threshold = generate_trend_signals(
        df,
        ma_fast=20,
        ma_slow=50,
        volume_threshold=500000.0,  # Low threshold
    )

    # Low threshold should have more or equal LONG signals
    long_high = len(
        signals_high_threshold[signals_high_threshold["direction"] == "LONG"]
    )
    long_low = len(signals_low_threshold[signals_low_threshold["direction"] == "LONG"])

    assert long_low >= long_high, (
        "Lower volume threshold should allow more LONG signals"
    )


def test_generate_trend_signals_no_volume():
    """Test signal generation without volume column."""
    df = create_sample_price_data_with_ma()
    df = df.drop(columns=["volume"])

    signals = generate_trend_signals(df, ma_fast=20, ma_slow=50)

    # Should still work without volume
    assert len(signals) > 0, "Should generate signals without volume"
    assert signals["direction"].isin(["LONG", "FLAT"]).all(), (
        "Directions should be valid"
    )


def test_generate_trend_signals_missing_columns():
    """Test that KeyError is raised when required columns are missing."""
    df = create_sample_price_data_with_ma()
    df = df.drop(columns=["close"])

    with pytest.raises(KeyError, match="Missing required columns"):
        generate_trend_signals(df, ma_fast=20, ma_slow=50)


def test_generate_trend_signals_ma_auto_compute():
    """Test that MAs are computed automatically if not present."""
    df = create_sample_price_data_simple()

    # Remove MAs if they exist
    df = df.drop(
        columns=[col for col in df.columns if col.startswith("ma_")], errors="ignore"
    )

    # Generate signals - should compute MAs automatically
    signals = generate_trend_signals(df, ma_fast=20, ma_slow=50)

    assert len(signals) > 0, "Should generate signals after auto-computing MAs"
    assert signals["direction"].isin(["LONG", "FLAT"]).all(), (
        "Directions should be valid"
    )
