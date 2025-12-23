# tests/test_features_ta.py
"""Tests for technical analysis features."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase4

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.features.ta_features import (
    add_all_features,
    add_atr,
    add_log_returns,
    add_moving_averages,
    add_rsi,
)


def create_sample_price_data() -> pd.DataFrame:
    """Create sample price data for testing."""
    from datetime import datetime, timedelta

    base = datetime(2025, 1, 1, 0, 0, 0)
    symbols = ["AAPL", "MSFT"]
    data = []

    for sym in symbols:
        price_base = 100.0 if sym == "AAPL" else 200.0
        for i in range(50):
            ts = base + timedelta(days=i)
            open_p = price_base + i * 0.1
            high_p = open_p + 0.5
            low_p = open_p - 0.3
            close_p = open_p + 0.2
            volume = 1000000.0 + i * 10000.0

            data.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "open": open_p,
                    "high": high_p,
                    "low": low_p,
                    "close": close_p,
                    "volume": volume,
                }
            )

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def test_add_log_returns():
    """Test log returns computation."""
    df = create_sample_price_data()
    df = add_log_returns(df)

    # Check column exists
    assert "log_return" in df.columns

    # Check first value is NaN (no previous price)
    first_log_return = (
        df.sort_values(["symbol", "timestamp"])
        .groupby("symbol")["log_return"]
        .nth(0)  # wirklich erste Zeile, auch wenn NaN
    )
    assert first_log_return.isna().all(), "First log return should be NaN"

    # Check subsequent values are finite
    # Alle Werte außer der ersten Zeile pro Symbol sollten nicht-NaN sein
    subsequent = (
        df.sort_values(["symbol", "timestamp"])
        .groupby("symbol")["log_return"]
        .nth(slice(1, None))
    )
    assert subsequent.notna().all(), "Subsequent log returns should be finite"

    # Check log returns are reasonable (not infinite)
    assert np.isfinite(df["log_return"].fillna(0)).all(), "Log returns should be finite"


def test_add_moving_averages():
    """Test moving averages computation."""
    df = create_sample_price_data()
    df = add_moving_averages(df, windows=(20, 50))

    # Check columns exist
    assert "ma_20" in df.columns
    assert "ma_50" in df.columns

    # Check values are finite
    assert df["ma_20"].notna().any(), "ma_20 should have non-NaN values"
    assert df["ma_50"].notna().any(), "ma_50 should have non-NaN values"

    # Check MA values are close to price (for first few rows, MA ≈ price)
    first_rows = df.groupby("symbol").head(5)
    assert (first_rows["ma_20"] - first_rows["close"]).abs().max() < 10.0, (
        "Early MA values should be close to price"
    )

    # Check MA is smoother than price (lower std)
    ma_20_std = df.groupby("symbol")["ma_20"].std()
    close_std = df.groupby("symbol")["close"].std()
    assert (ma_20_std < close_std).all(), "MA should be smoother (lower std) than price"


def test_add_atr():
    """Test ATR computation."""
    df = create_sample_price_data()
    df = add_atr(df, window=14)

    # Check column exists
    assert "atr_14" in df.columns

    # Check ATR values are positive
    assert (df["atr_14"] >= 0).all(), "ATR should be non-negative"

    # Check ATR is finite
    assert df["atr_14"].notna().any(), "ATR should have non-NaN values"

    # Check ATR is reasonable (should be less than price range)
    for sym in df["symbol"].unique():
        sym_df = df[df["symbol"] == sym]
        price_range = sym_df["high"].max() - sym_df["low"].min()
        max_atr = sym_df["atr_14"].max()
        assert max_atr < price_range, f"ATR should be less than price range for {sym}"


def test_add_rsi():
    """Test RSI computation."""
    df = create_sample_price_data()
    df = add_rsi(df, window=14)

    # Check column exists
    assert "rsi_14" in df.columns

    # Check first value is NaN (no previous prices)
    first_rsi = (
        df.sort_values(["symbol", "timestamp"])
        .groupby("symbol")["rsi_14"]
        .nth(0)  # wirklich erste Zeile, auch wenn NaN
    )
    assert first_rsi.isna().all(), "First RSI should be NaN"

    # Check subsequent values are in [0, 100]
    # Alle Werte außer der ersten Zeile pro Symbol sollten nicht-NaN sein
    subsequent = (
        df.sort_values(["symbol", "timestamp"])
        .groupby("symbol")["rsi_14"]
        .nth(slice(1, None))
    )
    valid_rsi = subsequent.dropna()
    if len(valid_rsi) > 0:
        assert (valid_rsi >= 0).all() and (valid_rsi <= 100).all(), (
            "RSI should be in [0, 100]"
        )


def test_add_all_features():
    """Test adding all features at once."""
    df = create_sample_price_data()
    df = add_all_features(
        df, ma_windows=(20, 50), atr_window=14, rsi_window=14, include_rsi=True
    )

    # Check all feature columns exist
    assert "log_return" in df.columns
    assert "ma_20" in df.columns
    assert "ma_50" in df.columns
    assert "atr_14" in df.columns
    assert "rsi_14" in df.columns

    # Check no columns are completely NaN
    feature_cols = ["log_return", "ma_20", "ma_50", "atr_14", "rsi_14"]
    for col in feature_cols:
        assert df[col].notna().any(), (
            f"Feature column '{col}' should have non-NaN values"
        )


def test_add_log_returns_missing_column():
    """Test that KeyError is raised when price column is missing."""
    df = create_sample_price_data()
    df = df.drop(columns=["close"])

    with pytest.raises(KeyError, match="Price column"):
        add_log_returns(df, price_col="close")


def test_add_moving_averages_missing_column():
    """Test that KeyError is raised when price column is missing."""
    df = create_sample_price_data()
    df = df.drop(columns=["close"])

    with pytest.raises(KeyError, match="Price column"):
        add_moving_averages(df, price_col="close")


def test_add_atr_missing_columns():
    """Test that KeyError is raised when required columns are missing."""
    df = create_sample_price_data()
    df = df.drop(columns=["high"])

    with pytest.raises(KeyError, match="Missing required columns"):
        add_atr(df)
