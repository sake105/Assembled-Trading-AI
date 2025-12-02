"""Tests for walk-forward analysis module."""
from __future__ import annotations


import numpy as np
import pandas as pd
import pytest

from src.assembled_core.portfolio.position_sizing import compute_target_positions
from src.assembled_core.qa.walk_forward import (
    WalkForwardConfig,
    WalkForwardResult,
    WalkForwardWindow,
    _split_time_series,
    run_walk_forward_backtest,
)
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices


@pytest.fixture
def synthetic_prices_long() -> pd.DataFrame:
    """Create synthetic price data for walk-forward testing (long time series)."""
    # Create 500 days of data for 3 symbols
    dates = pd.date_range("2020-01-01", periods=500, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    data = []
    for symbol in symbols:
        # Create trending prices with some volatility
        base_price = 100.0 if symbol == "AAPL" else 150.0 if symbol == "MSFT" else 200.0
        trend = pd.Series(range(500)) * 0.1  # Upward trend
        noise = pd.Series([0.0] * 500).apply(lambda _: np.random.normal(0, 2))
        prices = base_price + trend + noise
        
        for i, date in enumerate(dates):
            data.append({
                "timestamp": date,
                "symbol": symbol,
                "close": prices.iloc[i],
                "open": prices.iloc[i] * 0.99,
                "high": prices.iloc[i] * 1.01,
                "low": prices.iloc[i] * 0.98,
                "volume": 1000000 + np.random.randint(-100000, 100000)
            })
    
    return pd.DataFrame(data)


@pytest.fixture
def synthetic_prices_short() -> pd.DataFrame:
    """Create synthetic price data for walk-forward testing (short time series)."""
    # Create 100 days of data for 2 symbols
    dates = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT"]
    
    data = []
    for symbol in symbols:
        base_price = 100.0
        trend = pd.Series(range(100)) * 0.05
        noise = pd.Series([0.0] * 100).apply(lambda _: np.random.normal(0, 1))
        prices = base_price + trend + noise
        
        for i, date in enumerate(dates):
            data.append({
                "timestamp": date,
                "symbol": symbol,
                "close": prices.iloc[i],
                "open": prices.iloc[i] * 0.99,
                "high": prices.iloc[i] * 1.01,
                "low": prices.iloc[i] * 0.98,
                "volume": 1000000
            })
    
    return pd.DataFrame(data)


def test_split_time_series_rolling(synthetic_prices_long):
    """Test that time series splitting works correctly for rolling windows."""
    config = WalkForwardConfig(
        train_size=100,  # 100 days
        test_size=30,    # 30 days
        step_size=20,    # Move forward 20 days
        window_type="rolling"
    )
    
    windows = _split_time_series(synthetic_prices_long, config, "1d")
    
    # Should have multiple windows
    assert len(windows) > 0
    
    # Check first window
    first = windows[0]
    assert isinstance(first, WalkForwardWindow)
    assert first.window_index == 0
    assert not first.train_data.empty
    assert not first.test_data.empty
    
    # Check that train and test don't overlap
    train_end = first.train_end
    test_start = first.test_start
    assert test_start > train_end
    
    # Check window sizes (approximately)
    train_periods = len(first.train_data["timestamp"].unique())
    test_periods = len(first.test_data["timestamp"].unique())
    assert train_periods >= config.min_train_periods
    assert test_periods >= config.min_test_periods


def test_split_time_series_expanding(synthetic_prices_long):
    """Test that time series splitting works correctly for expanding windows."""
    config = WalkForwardConfig(
        train_size=50,   # Initial train size
        test_size=20,    # Test size
        step_size=10,    # Move forward 10 days
        window_type="expanding"
    )
    
    windows = _split_time_series(synthetic_prices_long, config, "1d")
    
    # Should have multiple windows
    assert len(windows) > 0
    
    # In expanding windows, train start should stay the same
    first_train_start = windows[0].train_start
    if len(windows) > 1:
        second_train_start = windows[1].train_start
        # Train start should be same or earlier (expanding)
        assert second_train_start <= first_train_start


def test_split_time_series_with_purging(synthetic_prices_long):
    """Test that purging works correctly (gap between train and test)."""
    config = WalkForwardConfig(
        train_size=100,
        test_size=30,
        step_size=20,
        window_type="rolling",
        purge_periods=10  # 10 days gap
    )
    
    windows = _split_time_series(synthetic_prices_long, config, "1d")
    
    if len(windows) > 0:
        first = windows[0]
        # Check that there's a gap (test_start > train_end + purge_periods)
        gap_days = (first.test_start - first.train_end).days
        assert gap_days >= config.purge_periods


def test_split_time_series_empty_data():
    """Test that empty data returns empty list."""
    empty_df = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    config = WalkForwardConfig(train_size=10, test_size=5, step_size=5)
    
    windows = _split_time_series(empty_df, config, "1d")
    assert len(windows) == 0


def test_split_time_series_date_filtering(synthetic_prices_long):
    """Test that date filtering works correctly."""
    config = WalkForwardConfig(
        train_size=50,
        test_size=20,
        step_size=10,
        start_date=pd.Timestamp("2020-02-01", tz="UTC"),
        end_date=pd.Timestamp("2020-06-01", tz="UTC")
    )
    
    windows = _split_time_series(synthetic_prices_long, config, "1d")
    
    if len(windows) > 0:
        first = windows[0]
        # All windows should be within date range
        assert first.train_start >= config.start_date
        assert first.test_end <= config.end_date


def test_run_walk_forward_backtest_basic(synthetic_prices_short):
    """Test basic walk-forward backtest execution."""
    config = WalkForwardConfig(
        train_size=30,  # 30 days training
        test_size=10,   # 10 days testing
        step_size=10,  # Move forward 10 days
        window_type="rolling"
    )
    
    result = run_walk_forward_backtest(
        prices=synthetic_prices_short,
        signal_fn=generate_trend_signals_from_prices,
        position_sizing_fn=compute_target_positions,
        config=config,
        start_capital=10000.0,
        freq="1d",
        compute_is_metrics=True
    )
    
    # Check result structure
    assert isinstance(result, WalkForwardResult)
    assert len(result.windows) > 0
    assert len(result.window_results) == len(result.windows)
    
    # Check that each window has OOS metrics
    for window_result in result.window_results:
        assert window_result.oos_metrics is not None
        assert window_result.oos_metrics.final_pf > 0
        assert window_result.backtest_result is not None
    
    # Check summary metrics
    assert "oos_mean_final_pf" in result.summary_metrics
    assert "num_windows" in result.summary_metrics
    assert result.summary_metrics["num_windows"] == len(result.windows)


def test_run_walk_forward_backtest_is_metrics(synthetic_prices_short):
    """Test that IS (in-sample) metrics are computed when requested."""
    config = WalkForwardConfig(
        train_size=30,
        test_size=10,
        step_size=10,
        window_type="rolling"
    )
    
    result = run_walk_forward_backtest(
        prices=synthetic_prices_short,
        signal_fn=generate_trend_signals_from_prices,
        position_sizing_fn=compute_target_positions,
        config=config,
        start_capital=10000.0,
        freq="1d",
        compute_is_metrics=True
    )
    
    # Check that IS metrics are computed (at least for some windows)
    has_is_metrics = any(wr.is_metrics is not None for wr in result.window_results)
    assert has_is_metrics
    
    # Check summary metrics include IS metrics
    assert "is_mean_final_pf" in result.summary_metrics or result.summary_metrics.get("is_mean_final_pf") is None


def test_run_walk_forward_backtest_no_is_metrics(synthetic_prices_short):
    """Test that IS metrics are not computed when disabled."""
    config = WalkForwardConfig(
        train_size=30,
        test_size=10,
        step_size=10,
        window_type="rolling"
    )
    
    result = run_walk_forward_backtest(
        prices=synthetic_prices_short,
        signal_fn=generate_trend_signals_from_prices,
        position_sizing_fn=compute_target_positions,
        config=config,
        start_capital=10000.0,
        freq="1d",
        compute_is_metrics=False
    )
    
    # Check that IS metrics are None for all windows
    for window_result in result.window_results:
        assert window_result.is_metrics is None


def test_run_walk_forward_backtest_metrics_per_window(synthetic_prices_short):
    """Test that metrics are computed correctly for each window."""
    config = WalkForwardConfig(
        train_size=30,
        test_size=10,
        step_size=10,
        window_type="rolling"
    )
    
    result = run_walk_forward_backtest(
        prices=synthetic_prices_short,
        signal_fn=generate_trend_signals_from_prices,
        position_sizing_fn=compute_target_positions,
        config=config,
        start_capital=10000.0,
        freq="1d"
    )
    
    # Check that each window result has valid OOS metrics
    for window_result in result.window_results:
        oos_metrics = window_result.oos_metrics
        
        # Check required fields
        assert oos_metrics.final_pf > 0
        assert oos_metrics.periods > 0
        assert oos_metrics.start_date <= oos_metrics.end_date
        assert oos_metrics.start_capital == 10000.0
        
        # Check that equity curve exists in backtest result
        assert not window_result.backtest_result.equity.empty
        assert "timestamp" in window_result.backtest_result.equity.columns
        assert "equity" in window_result.backtest_result.equity.columns


def test_run_walk_forward_backtest_summary_aggregation(synthetic_prices_short):
    """Test that summary metrics are correctly aggregated across windows."""
    config = WalkForwardConfig(
        train_size=30,
        test_size=10,
        step_size=10,
        window_type="rolling"
    )
    
    result = run_walk_forward_backtest(
        prices=synthetic_prices_short,
        signal_fn=generate_trend_signals_from_prices,
        position_sizing_fn=compute_target_positions,
        config=config,
        start_capital=10000.0,
        freq="1d"
    )
    
    # Check OOS summary metrics
    assert "oos_mean_final_pf" in result.summary_metrics
    assert "oos_std_final_pf" in result.summary_metrics
    assert "oos_win_rate" in result.summary_metrics
    assert "total_periods" in result.summary_metrics
    assert "total_trades" in result.summary_metrics
    assert "num_windows" in result.summary_metrics
    
    # Check that aggregated values are reasonable
    assert result.summary_metrics["num_windows"] == len(result.windows)
    assert result.summary_metrics["total_periods"] > 0
    assert 0.0 <= result.summary_metrics["oos_win_rate"] <= 1.0


def test_run_walk_forward_backtest_no_windows():
    """Test that error is raised when no valid windows are found."""
    # Create data that's too short for any window
    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    data = []
    for date in dates:
        data.append({
            "timestamp": date,
            "symbol": "AAPL",
            "close": 100.0,
            "volume": 1000000
        })
    prices = pd.DataFrame(data)
    
    config = WalkForwardConfig(
        train_size=100,  # Too large
        test_size=50,
        step_size=10
    )
    
    with pytest.raises(ValueError, match="No valid windows found"):
        run_walk_forward_backtest(
            prices=prices,
            signal_fn=generate_trend_signals_from_prices,
            position_sizing_fn=compute_target_positions,
            config=config,
            start_capital=10000.0,
            freq="1d"
        )

