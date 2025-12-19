"""Tests for Signal API module (Phase A, Sprint A2).

Tests the signal_api module: SignalMetadata, normalize_signals, make_signal_frame, validate_signal_frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.qa.point_in_time_checks import PointInTimeViolationError
from src.assembled_core.signals.signal_api import (
    SignalMetadata,
    make_signal_frame,
    normalize_signals,
    validate_signal_frame,
)


@pytest.fixture
def sample_raw_scores() -> pd.DataFrame:
    """Create a sample raw scores DataFrame."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    data = []
    for date in dates:
        for symbol in symbols:
            # Create varying scores
            base_score = 10.0 if symbol == "AAPL" else 5.0 if symbol == "MSFT" else 8.0
            score = base_score + np.random.randn() * 2.0
            data.append({"symbol": symbol, "score": score})
    
    df = pd.DataFrame(data)
    df.index = pd.DatetimeIndex(dates.repeat(len(symbols)), tz="UTC")
    return df


@pytest.fixture
def sample_signal_metadata() -> SignalMetadata:
    """Create a sample SignalMetadata."""
    return SignalMetadata(
        strategy_name="test_strategy",
        freq="1d",
        universe_name="test_universe",
        as_of=pd.Timestamp("2024-01-10", tz="UTC"),
        source="test_source",
    )


@pytest.mark.advanced
def test_normalize_signals_zscore_basic():
    """Test zscore normalization of signals."""
    # Create simple signal DataFrame
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    signals = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"] * 3,
        "signal_value": [10.0, 5.0, 8.0, 12.0, 6.0, 9.0, 11.0, 7.0, 10.0],
    })
    signals.index = pd.DatetimeIndex(dates.repeat(3), tz="UTC")
    
    # Normalize with zscore
    normalized = normalize_signals(signals, method="zscore", clip=None)
    
    # Check that mean is approximately 0 and std is approximately 1 per timestamp
    for date in dates:
        date_signals = normalized.loc[date, "signal_value"]
        assert abs(date_signals.mean()) < 1e-10, f"Mean should be 0 for {date}, got {date_signals.mean()}"
        assert abs(date_signals.std() - 1.0) < 1e-10, f"Std should be 1 for {date}, got {date_signals.std()}"
    
    # Check that original DataFrame was not modified
    assert "signal_value" in signals.columns
    assert signals["signal_value"].iloc[0] == 10.0  # Original value unchanged


@pytest.mark.advanced
def test_normalize_signals_rank_basic():
    """Test rank normalization of signals."""
    # Create simple signal DataFrame
    dates = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    signals = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"] * 2,
        "signal_value": [10.0, 5.0, 8.0, 12.0, 6.0, 9.0],
    })
    signals.index = pd.DatetimeIndex(dates.repeat(3), tz="UTC")
    
    # Normalize with rank
    normalized = normalize_signals(signals, method="rank", clip=None)
    
    # Check that ranks are in [-0.5, 0.5] range
    for date in dates:
        date_signals = normalized.loc[date, "signal_value"]
        assert (date_signals >= -0.5).all(), f"Ranks should be >= -0.5 for {date}"
        assert (date_signals <= 0.5).all(), f"Ranks should be <= 0.5 for {date}"
        # Check that mean is approximately 0 (ranks centered)
        assert abs(date_signals.mean()) < 1e-10, f"Mean should be 0 for {date}"


@pytest.mark.advanced
def test_normalize_signals_clipping():
    """Test clipping after normalization."""
    dates = pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC")
    signals = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "GOOGL"],
        "signal_value": [100.0, 5.0, 8.0],  # AAPL has extreme value
    })
    signals.index = pd.DatetimeIndex(dates.repeat(3), tz="UTC")
    
    # Normalize with zscore and clip at 5.0
    normalized = normalize_signals(signals, method="zscore", clip=5.0)
    
    # Check that all values are in [-5, 5] range
    assert (normalized["signal_value"] >= -5.0).all()
    assert (normalized["signal_value"] <= 5.0).all()


@pytest.mark.advanced
def test_make_signal_frame_basic(sample_raw_scores, sample_signal_metadata):
    """Test creating SignalFrame from raw scores."""
    signal_frame = make_signal_frame(
        sample_raw_scores,
        sample_signal_metadata,
        value_col="score",
        method="zscore",
    )
    
    # Check required columns
    assert "symbol" in signal_frame.columns
    assert "signal_value" in signal_frame.columns
    
    # Check that score column was renamed to signal_value
    assert "score" not in signal_frame.columns
    
    # Check that as_of column was added
    assert "as_of" in signal_frame.columns
    assert (signal_frame["as_of"] == sample_signal_metadata.as_of).all()
    
    # Check that index is DatetimeIndex
    assert isinstance(signal_frame.index, pd.DatetimeIndex)
    
    # Check that values are normalized (mean ~0, std ~1 per timestamp)
    for date in signal_frame.index.unique():
        date_signals = signal_frame.loc[date, "signal_value"]
        if len(date_signals) > 1:
            assert abs(date_signals.mean()) < 0.1, f"Mean should be close to 0 for {date}"


@pytest.mark.advanced
def test_validate_signal_frame_passes_for_valid_input():
    """Test validation passes for valid SignalFrame."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC")
    # Create unique (timestamp, symbol) pairs by building list explicitly
    data = []
    for date in dates:
        data.append({"symbol": "AAPL", "signal_value": 0.5, "as_of": pd.Timestamp("2024-01-10", tz="UTC")})
        data.append({"symbol": "MSFT", "signal_value": -0.3, "as_of": pd.Timestamp("2024-01-10", tz="UTC")})
    
    signal_df = pd.DataFrame(data)
    signal_df.index = pd.DatetimeIndex(dates.repeat(2), tz="UTC")
    
    # Should not raise any exceptions
    validate_signal_frame(signal_df, strict=True)


@pytest.mark.advanced
def test_validate_signal_frame_detects_duplicates():
    """Test validation detects duplicate (timestamp, symbol) pairs."""
    dates = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    signal_df = pd.DataFrame({
        "symbol": ["AAPL", "AAPL", "MSFT"],  # Duplicate AAPL on first date
        "signal_value": [0.5, 0.6, -0.3],  # Two values for AAPL
    })
    signal_df.index = pd.DatetimeIndex([dates[0], dates[0], dates[1]], tz="UTC")
    
    # Should raise ValueError in strict mode
    with pytest.raises(ValueError, match="duplicate"):
        validate_signal_frame(signal_df, strict=True)
    
    # Should log warning in non-strict mode
    validate_signal_frame(signal_df, strict=False)


@pytest.mark.advanced
def test_validate_signal_frame_missing_columns():
    """Test validation detects missing required columns."""
    dates = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    signal_df = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"],
        # Missing signal_value column
    })
    signal_df.index = pd.DatetimeIndex(dates, tz="UTC")
    
    # Should raise ValueError in strict mode
    with pytest.raises(ValueError, match="missing required columns"):
        validate_signal_frame(signal_df, strict=True)


@pytest.mark.advanced
def test_validate_signal_frame_with_as_of_and_pit_guard():
    """Test PIT-safety check with as_of parameter."""
    # Create signals with timestamps AFTER as_of (PIT violation)
    dates = pd.date_range("2024-01-05", periods=3, freq="D", tz="UTC")  # Start after as_of
    # Create unique (timestamp, symbol) pairs
    data = []
    for date in dates:
        data.append({"symbol": "AAPL", "signal_value": 0.5})
        data.append({"symbol": "MSFT", "signal_value": -0.3})
    
    signal_df = pd.DataFrame(data)
    signal_df.index = pd.DatetimeIndex(dates.repeat(2), tz="UTC")
    
    as_of = pd.Timestamp("2024-01-04", tz="UTC")  # Before all signal timestamps
    
    # Should raise PointInTimeViolationError in strict mode
    with pytest.raises(PointInTimeViolationError):
        validate_signal_frame(signal_df, as_of=as_of, strict=True)
    
    # Should log warning in non-strict mode
    validate_signal_frame(signal_df, as_of=as_of, strict=False, feature_source="test_signals")


@pytest.mark.advanced
def test_normalize_signals_none_method():
    """Test normalization with method='none'."""
    dates = pd.date_range("2024-01-01", periods=2, freq="D", tz="UTC")
    signals = pd.DataFrame({
        "symbol": ["AAPL", "MSFT"] * 2,
        "signal_value": [10.0, 5.0, 12.0, 6.0],
    })
    signals.index = pd.DatetimeIndex(dates.repeat(2), tz="UTC")
    original_values = signals["signal_value"].copy()
    
    # Normalize with method='none'
    normalized = normalize_signals(signals, method="none", clip=None)
    
    # Values should be unchanged
    pd.testing.assert_series_equal(normalized["signal_value"], original_values)


@pytest.mark.advanced
def test_make_signal_frame_with_custom_method(sample_raw_scores, sample_signal_metadata):
    """Test make_signal_frame with different normalization methods."""
    # Test with rank method
    signal_frame = make_signal_frame(
        sample_raw_scores,
        sample_signal_metadata,
        value_col="score",
        method="rank",
        clip=None,
    )
    
    # Check that values are in rank range [-0.5, 0.5]
    assert (signal_frame["signal_value"] >= -0.5).all()
    assert (signal_frame["signal_value"] <= 0.5).all()

