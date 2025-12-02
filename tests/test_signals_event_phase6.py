"""Tests for Phase 6 event-based signals (Insider + Shipping).

These tests verify that event-based signal generation works correctly
with insider trading and shipping route features.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.assembled_core.signals.rules_event_insider_shipping import generate_event_signals

pytestmark = pytest.mark.phase6


@pytest.fixture
def sample_prices_with_features() -> pd.DataFrame:
    """Create sample price DataFrame with event features."""
    now = datetime.now(timezone.utc)
    symbols = ["AAPL", "MSFT"]
    
    data = []
    for symbol in symbols:
        for i in range(5):
            data.append({
                "timestamp": now - timedelta(days=5-i),
                "symbol": symbol,
                "close": 100.0 + i * 0.5,
                "open": 100.0 + i * 0.5,
                "high": 101.0 + i * 0.5,
                "low": 99.0 + i * 0.5,
                "volume": 1000000,
                # Insider features
                "insider_net_buy_20d": 2000.0 if i < 2 else -1500.0,  # First 2: buy, rest: sell
                "insider_trade_count_20d": 3 if i < 2 else 2,
                # Shipping features
                "shipping_congestion_score_7d": 20.0 if i < 2 else 80.0,  # First 2: low, rest: high
                "shipping_ships_count_7d": 10.0 if i < 2 else 50.0,
            })
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def bullish_features() -> pd.DataFrame:
    """Create DataFrame with bullish features (strong insider buy + low congestion)."""
    now = datetime.now(timezone.utc)
    
    data = [{
        "timestamp": now,
        "symbol": "AAPL",
        "close": 100.0,
        "insider_net_buy_20d": 5000.0,  # Strong buy
        "shipping_congestion_score_7d": 15.0,  # Low congestion
    }]
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture
def bearish_features() -> pd.DataFrame:
    """Create DataFrame with bearish features (strong insider sell + high congestion)."""
    now = datetime.now(timezone.utc)
    
    data = [{
        "timestamp": now,
        "symbol": "AAPL",
        "close": 100.0,
        "insider_net_buy_20d": -5000.0,  # Strong sell
        "shipping_congestion_score_7d": 85.0,  # High congestion
    }]
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.fixture
def neutral_features() -> pd.DataFrame:
    """Create DataFrame with neutral features."""
    now = datetime.now(timezone.utc)
    
    data = [{
        "timestamp": now,
        "symbol": "AAPL",
        "close": 100.0,
        "insider_net_buy_20d": 100.0,  # Weak buy
        "shipping_congestion_score_7d": 50.0,  # Neutral congestion
    }]
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


@pytest.mark.phase6
@pytest.mark.unit
def test_generate_event_signals_returns_dataframe(sample_prices_with_features):
    """Test that generate_event_signals returns a DataFrame with correct columns."""
    result = generate_event_signals(sample_prices_with_features)
    
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(sample_prices_with_features)
    assert "timestamp" in result.columns
    assert "symbol" in result.columns
    assert "direction" in result.columns
    assert "score" in result.columns


@pytest.mark.phase6
@pytest.mark.unit
def test_generate_event_signals_bullish_combination(bullish_features):
    """Test that bullish combination (strong insider buy + low congestion) generates LONG signal."""
    result = generate_event_signals(bullish_features)
    
    assert len(result) == 1
    assert result["direction"].iloc[0] == "LONG"
    assert result["score"].iloc[0] > 0.0  # Non-zero score


@pytest.mark.phase6
@pytest.mark.unit
def test_generate_event_signals_bearish_combination(bearish_features):
    """Test that bearish combination (strong insider sell + high congestion) generates SHORT signal."""
    result = generate_event_signals(bearish_features)
    
    assert len(result) == 1
    assert result["direction"].iloc[0] == "SHORT"
    assert result["score"].iloc[0] > 0.0  # Non-zero score


@pytest.mark.phase6
@pytest.mark.unit
def test_generate_event_signals_neutral_combination(neutral_features):
    """Test that neutral combination generates FLAT signal."""
    result = generate_event_signals(neutral_features)
    
    assert len(result) == 1
    assert result["direction"].iloc[0] == "FLAT"
    assert result["score"].iloc[0] == 0.0  # Zero score for FLAT


@pytest.mark.phase6
@pytest.mark.unit
def test_generate_event_signals_mixed_signals(sample_prices_with_features):
    """Test that mixed signals generate appropriate results."""
    result = generate_event_signals(sample_prices_with_features)
    
    # First 2 rows per symbol should be LONG (strong buy + low congestion)
    # Rest should be SHORT (strong sell + high congestion)
    assert len(result) == 10  # 2 symbols * 5 rows
    
    # Check signal distribution
    long_count = (result["direction"] == "LONG").sum()
    short_count = (result["direction"] == "SHORT").sum()
    flat_count = (result["direction"] == "FLAT").sum()
    
    # Should have some LONG and SHORT signals
    assert long_count > 0
    assert short_count > 0
    
    # Scores should be non-zero for non-FLAT signals
    non_flat = result[result["direction"] != "FLAT"]
    if len(non_flat) > 0:
        assert (non_flat["score"] > 0.0).all()


@pytest.mark.phase6
@pytest.mark.unit
def test_generate_event_signals_missing_required_columns():
    """Test that missing required columns raise KeyError."""
    df = pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc)],
        "symbol": ["AAPL"],
        # Missing 'close'
    })
    
    with pytest.raises(KeyError, match="Missing required columns"):
        generate_event_signals(df)


@pytest.mark.phase6
@pytest.mark.unit
def test_generate_event_signals_missing_feature_columns():
    """Test that missing feature columns raise KeyError with helpful message."""
    df = pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc)],
        "symbol": ["AAPL"],
        "close": [100.0],
        # Missing feature columns
    })
    
    with pytest.raises(KeyError, match="Missing required feature columns"):
        generate_event_signals(df)


@pytest.mark.phase6
@pytest.mark.unit
def test_generate_event_signals_missing_one_feature_column():
    """Test that missing one feature column raises KeyError."""
    df = pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc)],
        "symbol": ["AAPL"],
        "close": [100.0],
        "insider_net_buy_20d": [1000.0],
        # Missing shipping_congestion_score_7d
    })
    
    with pytest.raises(KeyError, match="Missing required feature columns"):
        generate_event_signals(df)


@pytest.mark.phase6
@pytest.mark.unit
def test_generate_event_signals_handles_nan_values():
    """Test that NaN values in features are handled gracefully."""
    now = datetime.now(timezone.utc)
    
    df = pd.DataFrame({
        "timestamp": [now, now + timedelta(days=1)],
        "symbol": ["AAPL", "AAPL"],
        "close": [100.0, 101.0],
        "insider_net_buy_20d": [1000.0, pd.NA],  # One NaN
        "shipping_congestion_score_7d": [20.0, pd.NA],  # One NaN
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    result = generate_event_signals(df)
    
    assert len(result) == 2
    # First row should have signal (non-NaN features)
    assert result["direction"].iloc[0] in ["LONG", "FLAT", "SHORT"]
    # Second row should be FLAT (NaN features filled with neutral values)
    assert result["direction"].iloc[1] == "FLAT"


@pytest.mark.phase6
@pytest.mark.unit
def test_generate_event_signals_custom_weights():
    """Test that custom weights affect signal generation."""
    now = datetime.now(timezone.utc)
    
    df = pd.DataFrame({
        "timestamp": [now],
        "symbol": ["AAPL"],
        "close": [100.0],
        "insider_net_buy_20d": 2000.0,  # Strong buy
        "shipping_congestion_score_7d": 25.0,  # Low congestion
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    # Default weights should generate LONG
    result_default = generate_event_signals(df)
    assert result_default["direction"].iloc[0] == "LONG"
    
    # Zero shipping weight should still generate LONG (insider alone)
    result_no_shipping = generate_event_signals(df, shipping_weight=0.0)
    assert result_no_shipping["direction"].iloc[0] == "LONG"
    
    # Zero insider weight should generate LONG (shipping alone, low congestion)
    result_no_insider = generate_event_signals(df, insider_weight=0.0)
    assert result_no_insider["direction"].iloc[0] == "LONG"


@pytest.mark.phase6
@pytest.mark.unit
def test_generate_event_signals_invalid_weights():
    """Test that negative weights raise ValueError."""
    now = datetime.now(timezone.utc)
    
    df = pd.DataFrame({
        "timestamp": [now],
        "symbol": ["AAPL"],
        "close": [100.0],
        "insider_net_buy_20d": [1000.0],
        "shipping_congestion_score_7d": [30.0],
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    with pytest.raises(ValueError, match="weights must be non-negative"):
        generate_event_signals(df, insider_weight=-1.0)
    
    with pytest.raises(ValueError, match="weights must be non-negative"):
        generate_event_signals(df, shipping_weight=-1.0)


@pytest.mark.phase6
@pytest.mark.unit
def test_generate_event_signals_signal_values():
    """Test that signal values are only -1, 0, or 1."""
    now = datetime.now(timezone.utc)
    
    # Create various combinations
    data = []
    for insider in [-5000, -1000, 0, 1000, 5000]:
        for congestion in [10, 30, 50, 70, 90]:
            data.append({
                "timestamp": now,
                "symbol": "AAPL",
                "close": 100.0,
                "insider_net_buy_20d": float(insider),
                "shipping_congestion_score_7d": float(congestion),
            })
    
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    
    result = generate_event_signals(df)
    
    # All directions should be "LONG", "FLAT", or "SHORT"
    assert result["direction"].isin(["LONG", "FLAT", "SHORT"]).all()
    
    # Scores should be in [0, 1]
    assert (result["score"] >= 0.0).all()
    assert (result["score"] <= 1.0).all()

