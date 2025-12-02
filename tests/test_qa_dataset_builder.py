"""Tests for qa.dataset_builder module."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase7

from src.assembled_core.qa.dataset_builder import build_ml_dataset_from_backtest, save_ml_dataset


@pytest.fixture
def sample_prices_with_features() -> pd.DataFrame:
    """Create sample prices DataFrame with TA and Event features."""
    dates = pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC")
    
    # Create prices with some trend
    base_prices = [100.0 + i * 2.0 for i in range(20)]
    
    df = pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 20,
        "close": base_prices,
        "open": [p * 0.99 for p in base_prices],
        "high": [p * 1.02 for p in base_prices],
        "low": [p * 0.98 for p in base_prices],
        "volume": [1000000.0] * 20,
    })
    
    # Add TA features
    df["ma_20"] = df["close"].rolling(window=5, min_periods=1).mean()  # Simplified
    df["ma_50"] = df["close"].rolling(window=5, min_periods=1).mean()  # Simplified
    df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
    df["rsi_14"] = 50.0 + np.random.normal(0, 10, 20)  # Dummy RSI values
    
    # Add Event features
    df["insider_net_buy_20d"] = np.random.normal(1000, 500, 20)
    df["insider_trade_count_20d"] = np.random.randint(0, 10, 20)
    df["shipping_congestion_score_7d"] = np.random.uniform(20, 80, 20)
    df["shipping_ships_count_7d"] = np.random.randint(10, 50, 20)
    
    return df


@pytest.fixture
def sample_trades() -> pd.DataFrame:
    """Create sample trades DataFrame."""
    dates = pd.date_range("2020-01-05", periods=5, freq="D", tz="UTC")
    
    return pd.DataFrame({
        "timestamp": dates,
        "symbol": ["AAPL"] * 5,
        "side": ["BUY"] * 5,
        "qty": [10.0] * 5,
        "price": [100.0 + i * 2.0 for i in range(5)],
        "pnl_pct": [0.05, -0.01, 0.03, 0.01, -0.02],  # Mix of positive and negative
    })


def test_build_ml_dataset_from_backtest(sample_prices_with_features: pd.DataFrame, sample_trades: pd.DataFrame):
    """Test build_ml_dataset_from_backtest with TA and Event features."""
    dataset = build_ml_dataset_from_backtest(
        prices_with_features=sample_prices_with_features,
        trades=sample_trades,
        label_horizon_days=10,
        success_threshold=0.02,
        feature_prefixes=("ma_", "insider_", "shipping_", "rsi_", "log_return")
    )
    
    # Check that dataset has correct number of rows (one per trade)
    assert len(dataset) == len(sample_trades)
    
    # Check that label column exists and contains 0/1
    assert "label" in dataset.columns
    assert dataset["label"].isin([0, 1]).all()
    
    # Check that at least one TA feature is present
    ta_features = [col for col in dataset.columns if col.startswith(("ma_", "rsi_", "log_return", "atr_"))]
    assert len(ta_features) > 0, "Dataset should contain at least one TA feature"
    
    # Check that at least one Event feature is present
    event_features = [col for col in dataset.columns if col.startswith(("insider_", "shipping_", "congress_", "news_"))]
    assert len(event_features) > 0, "Dataset should contain at least one Event feature"
    
    # Check that metadata columns are present
    assert "timestamp" in dataset.columns
    assert "symbol" in dataset.columns
    
    # Check that pnl_pct is present (from label_trades)
    assert "pnl_pct" in dataset.columns


def test_build_ml_dataset_empty_trades(sample_prices_with_features: pd.DataFrame):
    """Test build_ml_dataset_from_backtest with empty trades."""
    empty_trades = pd.DataFrame(columns=["timestamp", "symbol", "pnl_pct"])
    
    dataset = build_ml_dataset_from_backtest(
        prices_with_features=sample_prices_with_features,
        trades=empty_trades,
    )
    
    assert dataset.empty


def test_build_ml_dataset_empty_prices(sample_trades: pd.DataFrame):
    """Test build_ml_dataset_from_backtest with empty prices."""
    empty_prices = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    
    dataset = build_ml_dataset_from_backtest(
        prices_with_features=empty_prices,
        trades=sample_trades,
    )
    
    assert dataset.empty


def test_build_ml_dataset_feature_filtering(sample_prices_with_features: pd.DataFrame, sample_trades: pd.DataFrame):
    """Test that feature_prefixes correctly filters features."""
    # Test with only insider features
    dataset_insider = build_ml_dataset_from_backtest(
        prices_with_features=sample_prices_with_features,
        trades=sample_trades,
        feature_prefixes=("insider_",)
    )
    
    # Should only have insider features
    feature_cols = [col for col in dataset_insider.columns 
                   if col not in ["label", "timestamp", "symbol", "pnl_pct", "horizon_days", "close_time", "side", "qty", "price"]]
    assert all(col.startswith("insider_") for col in feature_cols), (
        f"Should only have insider features, but found: {feature_cols}"
    )
    
    # Test with only TA features (using known patterns)
    dataset_ta = build_ml_dataset_from_backtest(
        prices_with_features=sample_prices_with_features,
        trades=sample_trades,
        feature_prefixes=("ma_", "rsi_", "log_return")
    )
    
    # Should have TA features
    ta_cols = [col for col in dataset_ta.columns 
              if col.startswith(("ma_", "rsi_", "log_return"))]
    assert len(ta_cols) > 0, "Should have TA features"


def test_build_ml_dataset_trades_without_pnl(sample_prices_with_features: pd.DataFrame):
    """Test build_ml_dataset_from_backtest with trades that need P&L reconstruction."""
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-05", periods=3, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 3,
        "side": ["BUY"] * 3,
        "qty": [10.0] * 3,
        "price": [100.0, 105.0, 110.0],
    })
    
    dataset = build_ml_dataset_from_backtest(
        prices_with_features=sample_prices_with_features,
        trades=trades,
        feature_prefixes=("ma_", "insider_")
    )
    
    # Should have labels computed from reconstructed P&L
    assert "label" in dataset.columns
    assert "pnl_pct" in dataset.columns
    assert dataset["label"].isin([0, 1]).all()


def test_save_ml_dataset(sample_prices_with_features: pd.DataFrame, sample_trades: pd.DataFrame, tmp_path: Path):
    """Test save_ml_dataset function."""
    dataset = build_ml_dataset_from_backtest(
        prices_with_features=sample_prices_with_features,
        trades=sample_trades,
        feature_prefixes=("ma_", "insider_", "shipping_")
    )
    
    output_path = tmp_path / "ml_dataset.parquet"
    save_ml_dataset(dataset, output_path)
    
    # Check that file was created
    assert output_path.exists()
    
    # Check that file can be read back
    loaded = pd.read_parquet(output_path)
    assert len(loaded) == len(dataset)
    assert "label" in loaded.columns


def test_save_ml_dataset_empty(tmp_path: Path):
    """Test that save_ml_dataset raises error for empty DataFrame."""
    empty_df = pd.DataFrame()
    
    with pytest.raises(ValueError, match="empty"):
        save_ml_dataset(empty_df, tmp_path / "empty.parquet")


def test_build_ml_dataset_missing_columns():
    """Test that build_ml_dataset_from_backtest raises error if required columns are missing."""
    invalid_prices = pd.DataFrame({
        "symbol": ["AAPL"],
        # Missing timestamp
    })
    
    trades = pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc)],
        "symbol": ["AAPL"],
        "pnl_pct": [0.05],
    })
    
    with pytest.raises(ValueError, match="timestamp"):
        build_ml_dataset_from_backtest(invalid_prices, trades)


def test_build_ml_dataset_label_distribution(sample_prices_with_features: pd.DataFrame):
    """Test that labels are correctly distributed based on P&L."""
    # Create trades with known P&L
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-05", periods=4, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 4,
        "pnl_pct": [0.05, 0.01, -0.01, 0.03],  # 2 successful (>= 0.02), 2 unsuccessful
    })
    
    dataset = build_ml_dataset_from_backtest(
        prices_with_features=sample_prices_with_features,
        trades=trades,
        success_threshold=0.02,
        feature_prefixes=("ma_",)
    )
    
    # Check label distribution
    label_counts = dataset["label"].value_counts()
    assert 1 in label_counts.index, "Should have at least one successful trade (label=1)"
    assert 0 in label_counts.index, "Should have at least one unsuccessful trade (label=0)"
    
    # Verify labels match P&L
    assert dataset[dataset["pnl_pct"] >= 0.02]["label"].all() == 1, "Trades with pnl_pct >= 0.02 should be labeled 1"
    assert dataset[dataset["pnl_pct"] < 0.02]["label"].all() == 0, "Trades with pnl_pct < 0.02 should be labeled 0"

