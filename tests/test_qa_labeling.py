"""Tests for qa.labeling module."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase7

from src.assembled_core.qa.labeling import label_daily_records, label_trades


@pytest.fixture
def sample_trades_with_pnl() -> pd.DataFrame:
    """Create sample trades with pre-computed P&L."""
    dates = pd.date_range("2020-01-01", periods=5, freq="D", tz="UTC")

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * 5,
            "pnl_pct": [0.05, -0.01, 0.03, 0.01, -0.02],  # Mix of positive and negative
            "side": ["BUY"] * 5,
            "qty": [10.0] * 5,
            "price": [100.0] * 5,
        }
    )


@pytest.fixture
def sample_trades_without_pnl() -> pd.DataFrame:
    """Create sample trades without P&L (for reconstruction)."""
    dates = pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC")

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * 3,
            "side": ["BUY", "BUY", "SELL"],
            "qty": [10.0, 10.0, 10.0],
            "price": [100.0, 105.0, 110.0],
        }
    )


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create sample price data for P&L reconstruction."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")

    # Create upward trend with some noise
    base_prices = [100.0 + i * 2.0 + np.random.normal(0, 1) for i in range(20)]

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": ["AAPL"] * 20,
            "close": base_prices,
            "open": [p * 0.99 for p in base_prices],
            "high": [p * 1.02 for p in base_prices],
            "low": [p * 0.98 for p in base_prices],
            "volume": [1000000.0] * 20,
        }
    )


@pytest.fixture
def sample_equity_curve() -> pd.DataFrame:
    """Create sample equity curve for daily labeling."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC")

    # Create equity curve with upward trend
    equity_values = [10000.0 + i * 50.0 for i in range(30)]

    return pd.DataFrame(
        {
            "timestamp": dates,
            "equity": equity_values,
        }
    )


def test_label_trades_with_pnl(sample_trades_with_pnl: pd.DataFrame):
    """Test label_trades with pre-computed P&L."""
    result = label_trades(
        trades=sample_trades_with_pnl, success_threshold=0.02, horizon_days=10
    )

    # Check that label column was added
    assert "label" in result.columns
    assert "horizon_days" in result.columns
    assert "pnl_pct" in result.columns

    # Check labels: pnl_pct >= 0.02 should be 1, else 0
    assert result["label"].iloc[0] == 1  # 0.05 >= 0.02
    assert result["label"].iloc[1] == 0  # -0.01 < 0.02
    assert result["label"].iloc[2] == 1  # 0.03 >= 0.02
    assert result["label"].iloc[3] == 0  # 0.01 < 0.02
    assert result["label"].iloc[4] == 0  # -0.02 < 0.02

    # Check horizon_days
    assert (result["horizon_days"] == 10).all()

    # Check that original columns are preserved
    assert "symbol" in result.columns
    assert "timestamp" in result.columns


def test_label_trades_without_pnl(
    sample_trades_without_pnl: pd.DataFrame, sample_prices: pd.DataFrame
):
    """Test label_trades with P&L reconstruction from prices."""
    result = label_trades(
        trades=sample_trades_without_pnl,
        prices=sample_prices,
        success_threshold=0.02,
        horizon_days=5,
    )

    # Check that label column was added
    assert "label" in result.columns
    assert "horizon_days" in result.columns
    assert "pnl_pct" in result.columns
    assert "close_time" in result.columns

    # Check that P&L was computed
    assert not result["pnl_pct"].isna().all()

    # Check that labels are binary (0 or 1)
    assert result["label"].isin([0, 1]).all()

    # Check horizon_days
    assert (result["horizon_days"] == 5).all()


def test_label_trades_empty():
    """Test label_trades with empty DataFrame."""
    empty_trades = pd.DataFrame(columns=["timestamp", "symbol", "pnl_pct"])
    result = label_trades(empty_trades)

    assert result.empty
    assert "label" in result.columns or result.empty


def test_label_trades_missing_pnl_no_prices(sample_trades_without_pnl: pd.DataFrame):
    """Test that label_trades raises error if pnl_pct missing and prices not provided."""
    with pytest.raises(KeyError, match="Cannot compute P&L"):
        label_trades(trades=sample_trades_without_pnl, prices=None)


def test_label_trades_missing_required_columns():
    """Test that label_trades raises error if required columns are missing."""
    invalid_trades = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            # Missing timestamp
        }
    )

    with pytest.raises(ValueError, match="timestamp"):
        label_trades(invalid_trades)


def test_label_daily_records(sample_equity_curve: pd.DataFrame):
    """Test label_daily_records on equity curve."""
    result = label_daily_records(
        df=sample_equity_curve,
        horizon_days=5,
        success_threshold=0.01,  # 1% threshold
        price_col="equity",
    )

    # Check that label column was added
    assert "label" in result.columns

    # Check that labels are binary (0, 1, or NaN for last rows)
    valid_labels = result["label"].dropna()
    assert valid_labels.isin([0, 1]).all()

    # Check that last few rows may have NaN (insufficient forward data)
    # This is expected behavior
    assert len(result) == len(sample_equity_curve)


def test_label_daily_records_custom_price_col():
    """Test label_daily_records with custom price column."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC"),
            "custom_price": [100.0 + i * 2.0 for i in range(20)],
        }
    )

    result = label_daily_records(
        df=df, horizon_days=3, success_threshold=0.01, price_col="custom_price"
    )

    assert "label" in result.columns
    assert "custom_price" in result.columns


def test_label_daily_records_empty():
    """Test label_daily_records with empty DataFrame."""
    empty_df = pd.DataFrame(columns=["timestamp", "equity"])
    result = label_daily_records(empty_df)

    assert result.empty
    assert "label" in result.columns or result.empty


def test_label_daily_records_missing_columns():
    """Test that label_daily_records raises error if required columns are missing."""
    invalid_df = pd.DataFrame(
        {
            "equity": [10000.0],
            # Missing timestamp
        }
    )

    with pytest.raises(ValueError, match="timestamp"):
        label_daily_records(invalid_df)

    invalid_df2 = pd.DataFrame(
        {
            "timestamp": [datetime.now(timezone.utc)],
            # Missing equity column
        }
    )

    with pytest.raises(ValueError, match="equity"):
        label_daily_records(invalid_df2, price_col="equity")


def test_label_trades_with_close_time(sample_trades_with_pnl: pd.DataFrame):
    """Test label_trades with explicit close_time column."""
    trades = sample_trades_with_pnl.copy()
    trades["close_time"] = trades["timestamp"] + pd.Timedelta(days=15)

    result = label_trades(trades, horizon_days=10, success_threshold=0.02)

    # Should use provided close_time
    assert "close_time" in result.columns
    assert (result["close_time"] == trades["close_time"]).all()


def test_label_daily_records_edge_cases():
    """Test label_daily_records with edge cases."""
    # Equity curve with no growth (all labels should be 0)
    flat_equity = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC"),
            "equity": [10000.0] * 20,
        }
    )

    result = label_daily_records(flat_equity, horizon_days=5, success_threshold=0.01)

    # All valid labels should be 0 (no growth)
    valid_labels = result["label"].dropna()
    if len(valid_labels) > 0:
        assert (valid_labels == 0).all()
