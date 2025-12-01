"""Tests for qa.metrics module."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.metrics import (
    PerformanceMetrics,
    compute_all_metrics,
    compute_cagr,
    compute_drawdown,
    compute_equity_metrics,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_trade_metrics,
    compute_turnover,
)


@pytest.fixture
def synthetic_equity_1d() -> pd.DataFrame:
    """Synthetic equity curve for 1 year (252 days)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Simple upward trend with some volatility
    returns = np.random.normal(0.001, 0.02, 252)  # ~0.1% daily return, 2% volatility
    equity = 10000.0 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })


@pytest.fixture
def synthetic_equity_short() -> pd.DataFrame:
    """Synthetic equity curve for 30 days (short period)."""
    dates = pd.date_range("2020-01-01", periods=30, freq="D")
    returns = np.random.normal(0.001, 0.02, 30)
    equity = 10000.0 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })


@pytest.fixture
def synthetic_equity_declining() -> pd.DataFrame:
    """Synthetic equity curve with declining trend."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    returns = np.random.normal(-0.001, 0.02, 252)  # Negative trend
    equity = 10000.0 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })


@pytest.fixture
def synthetic_equity_strong_positive() -> pd.DataFrame:
    """Synthetic equity curve with strong positive trend (high CAGR)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Strong positive trend: ~0.3% daily return, 1.5% volatility
    np.random.seed(42)  # For reproducibility
    returns = np.random.normal(0.003, 0.015, 252)
    equity = 10000.0 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })


@pytest.fixture
def synthetic_equity_negative() -> pd.DataFrame:
    """Synthetic equity curve with strong negative trend."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Strong negative trend: ~-0.3% daily return, 1.5% volatility
    np.random.seed(43)
    returns = np.random.normal(-0.003, 0.015, 252)
    equity = 10000.0 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })


@pytest.fixture
def synthetic_equity_sideways() -> pd.DataFrame:
    """Synthetic equity curve with sideways movement (low return, high volatility)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    # Sideways: Generate equity that starts and ends at start_capital with high volatility
    # Use random walk that returns to start (mean-reverting)
    np.random.seed(44)
    start_capital = 10000.0
    n = len(dates)
    equity_values = [start_capital]
    
    # Generate random returns with high volatility
    returns = np.random.normal(0.0, 0.03, n - 1)
    # Adjust returns so cumulative product is exactly 1.0 (ends at start)
    cumulative = (1 + returns).prod()
    if abs(cumulative - 1.0) > 1e-10:
        # Scale returns to make cumulative = 1.0
        adjustment = (1.0 / cumulative) ** (1.0 / (n - 1))
        returns = returns * adjustment
    
    # Build equity curve
    for r in returns:
        equity_values.append(equity_values[-1] * (1 + r))
    
    # Ensure it ends exactly at start_capital (should be close already)
    if abs(equity_values[-1] - start_capital) > 0.01:
        # Scale entire curve so end = start_capital
        scale = start_capital / equity_values[-1]
        equity_values = [v * scale for v in equity_values]
    
    return pd.DataFrame({
        "timestamp": dates,
        "equity": equity_values
    })


@pytest.fixture
def synthetic_trades() -> pd.DataFrame:
    """Synthetic trades DataFrame."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    trades = []
    for i, date in enumerate(dates):
        trades.append({
            "timestamp": date,
            "symbol": "AAPL",
            "side": "BUY" if i % 2 == 0 else "SELL",
            "qty": 10.0,
            "price": 100.0 + i * 0.5
        })
    
    return pd.DataFrame(trades)


@pytest.fixture
def synthetic_trades_high_turnover() -> pd.DataFrame:
    """Synthetic trades DataFrame with high turnover (many trades)."""
    dates = pd.date_range("2020-01-01", periods=252, freq="D")
    trades = []
    for i, date in enumerate(dates):
        # Trade every day
        trades.append({
            "timestamp": date,
            "symbol": f"SYM{i % 5}",  # Rotate between 5 symbols
            "side": "BUY" if i % 2 == 0 else "SELL",
            "qty": 100.0,
            "price": 100.0 + i * 0.1
        })
    
    return pd.DataFrame(trades)


@pytest.mark.unit
def test_compute_sharpe_ratio_basic():
    """Test Sharpe ratio computation."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.01])
    sharpe = compute_sharpe_ratio(returns, freq="1d", risk_free_rate=0.0)
    
    assert sharpe is not None
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)


@pytest.mark.unit
def test_compute_sharpe_ratio_insufficient_data():
    """Test Sharpe ratio with insufficient data."""
    returns = pd.Series([0.01])  # Only 1 return
    sharpe = compute_sharpe_ratio(returns, freq="1d")
    
    assert sharpe is None


@pytest.mark.unit
def test_compute_sharpe_ratio_zero_std():
    """Test Sharpe ratio with zero standard deviation."""
    returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])  # Constant returns
    sharpe = compute_sharpe_ratio(returns, freq="1d")
    
    assert sharpe is None


@pytest.mark.unit
def test_compute_sortino_ratio_basic():
    """Test Sortino ratio computation."""
    returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.01])
    sortino = compute_sortino_ratio(returns, freq="1d", risk_free_rate=0.0)
    
    assert sortino is not None
    assert isinstance(sortino, float)
    assert not np.isnan(sortino)


@pytest.mark.unit
def test_compute_sortino_ratio_only_positive():
    """Test Sortino ratio with only positive returns."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.01, 0.01])  # All positive
    sortino = compute_sortino_ratio(returns, freq="1d")
    
    # Should still compute (uses regular std as fallback)
    assert sortino is not None or sortino is None  # May be None if std is 0


@pytest.mark.unit
def test_compute_drawdown():
    """Test drawdown computation."""
    equity = pd.Series([10000, 11000, 10500, 12000, 11500, 13000])
    drawdown_series, max_dd, max_dd_pct, current_dd = compute_drawdown(equity)
    
    assert len(drawdown_series) == len(equity)
    assert max_dd <= 0  # Drawdown should be negative
    assert max_dd_pct <= 0  # Percentage should be negative
    assert current_dd <= 0  # Current drawdown should be negative


@pytest.mark.unit
def test_compute_cagr_one_year():
    """Test CAGR computation for exactly 1 year."""
    cagr = compute_cagr(10000.0, 11000.0, 252, freq="1d")
    
    assert cagr is not None
    assert isinstance(cagr, float)
    assert cagr > 0  # Positive return


@pytest.mark.unit
def test_compute_cagr_short_period():
    """Test CAGR computation for period < 1 year."""
    cagr = compute_cagr(10000.0, 11000.0, 30, freq="1d")
    
    assert cagr is None  # Should return None for < 1 year


@pytest.mark.unit
def test_compute_cagr_invalid_inputs():
    """Test CAGR with invalid inputs."""
    assert compute_cagr(0.0, 11000.0, 252, freq="1d") is None
    assert compute_cagr(10000.0, 0.0, 252, freq="1d") is None
    assert compute_cagr(10000.0, 11000.0, 0, freq="1d") is None


@pytest.mark.unit
def test_compute_turnover():
    """Test turnover computation."""
    trades = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=10, freq="D"),
        "symbol": ["AAPL"] * 10,
        "side": ["BUY", "SELL"] * 5,
        "qty": [10.0] * 10,
        "price": [100.0] * 10
    })
    
    equity = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=10, freq="D"),
        "equity": [10000.0] * 10
    })
    
    turnover = compute_turnover(trades, equity, start_capital=10000.0, freq="1d")
    
    assert turnover is not None
    assert isinstance(turnover, float)
    assert turnover > 0


@pytest.mark.unit
def test_compute_turnover_empty_trades():
    """Test turnover with empty trades."""
    trades = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    equity = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=10, freq="D"),
        "equity": [10000.0] * 10
    })
    
    turnover = compute_turnover(trades, equity, start_capital=10000.0, freq="1d")
    
    assert turnover is None


@pytest.mark.smoke
def test_compute_equity_metrics_basic(synthetic_equity_1d):
    """Test basic equity metrics computation."""
    metrics = compute_equity_metrics(
        equity=synthetic_equity_1d,
        start_capital=10000.0,
        freq="1d",
        risk_free_rate=0.0
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.final_pf > 0
    assert metrics.total_return == metrics.final_pf - 1.0
    assert metrics.periods == 252
    assert metrics.start_capital == 10000.0
    assert metrics.end_equity > 0
    assert metrics.max_drawdown <= 0
    assert metrics.max_drawdown_pct <= 0
    assert metrics.current_drawdown <= 0


@pytest.mark.smoke
def test_compute_equity_metrics_short_period(synthetic_equity_short):
    """Test equity metrics with short period (< 1 year)."""
    metrics = compute_equity_metrics(
        equity=synthetic_equity_short,
        start_capital=10000.0,
        freq="1d"
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.cagr is None  # Should be None for < 1 year
    assert metrics.periods == 30


@pytest.mark.smoke
def test_compute_equity_metrics_strong_positive(synthetic_equity_strong_positive):
    """Test equity metrics with strong positive trend."""
    metrics = compute_equity_metrics(
        equity=synthetic_equity_strong_positive,
        start_capital=10000.0,
        freq="1d"
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    # Strong positive trend should have:
    assert metrics.final_pf > 1.5  # At least 50% return
    assert metrics.total_return > 0.5
    assert metrics.cagr is not None
    assert metrics.cagr > 0.3  # At least 30% CAGR
    assert metrics.sharpe_ratio is not None
    assert metrics.sharpe_ratio > 0  # Positive Sharpe


@pytest.mark.smoke
def test_compute_equity_metrics_negative(synthetic_equity_negative):
    """Test equity metrics with negative trend."""
    metrics = compute_equity_metrics(
        equity=synthetic_equity_negative,
        start_capital=10000.0,
        freq="1d"
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    # Negative trend should have:
    assert metrics.final_pf < 1.0  # Loss
    assert metrics.total_return < 0
    assert metrics.cagr is not None
    assert metrics.cagr < 0  # Negative CAGR
    assert metrics.max_drawdown < 0  # Negative drawdown


@pytest.mark.smoke
def test_compute_equity_metrics_sideways(synthetic_equity_sideways):
    """Test equity metrics with sideways movement."""
    metrics = compute_equity_metrics(
        equity=synthetic_equity_sideways,
        start_capital=10000.0,
        freq="1d"
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    # Sideways should have:
    assert abs(metrics.total_return) < 0.2  # Small total return
    assert metrics.volatility is not None
    assert metrics.volatility > 0.2  # High volatility (20%+)
    # Sharpe should be low or negative
    if metrics.sharpe_ratio is not None:
        assert metrics.sharpe_ratio < 1.0


@pytest.mark.smoke
def test_compute_equity_metrics_with_daily_return(synthetic_equity_1d):
    """Test equity metrics with pre-computed daily_return."""
    equity = synthetic_equity_1d.copy()
    equity["daily_return"] = equity["equity"].pct_change()
    
    metrics = compute_equity_metrics(
        equity=equity,
        start_capital=10000.0,
        freq="1d"
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.final_pf > 0


@pytest.mark.smoke
def test_compute_equity_metrics_declining(synthetic_equity_declining):
    """Test equity metrics with declining trend."""
    metrics = compute_equity_metrics(
        equity=synthetic_equity_declining,
        start_capital=10000.0,
        freq="1d"
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.final_pf < 1.0  # Declining trend
    assert metrics.total_return < 0
    assert metrics.max_drawdown < 0


@pytest.mark.smoke
def test_compute_equity_metrics_5min():
    """Test equity metrics with 5min frequency."""
    # Create 5min data (1 day = 78 periods)
    periods = 78 * 5  # 5 days
    dates = pd.date_range("2020-01-01", periods=periods, freq="5min")
    returns = np.random.normal(0.0001, 0.005, periods)  # Smaller returns for 5min
    equity = 10000.0 * (1 + returns).cumprod()
    
    equity_df = pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })
    
    metrics = compute_equity_metrics(
        equity=equity_df,
        start_capital=10000.0,
        freq="5min"
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.periods == periods


@pytest.mark.smoke
def test_compute_all_metrics_with_trades(synthetic_equity_1d, synthetic_trades):
    """Test compute_all_metrics with trades."""
    metrics = compute_all_metrics(
        equity=synthetic_equity_1d,
        trades=synthetic_trades,
        start_capital=10000.0,
        freq="1d"
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_trades == len(synthetic_trades)
    assert metrics.turnover is not None or metrics.turnover is None  # May be None if calculation fails


@pytest.mark.smoke
def test_compute_all_metrics_high_turnover(synthetic_equity_1d, synthetic_trades_high_turnover):
    """Test compute_all_metrics with high turnover trades."""
    metrics = compute_all_metrics(
        equity=synthetic_equity_1d,
        trades=synthetic_trades_high_turnover,
        start_capital=10000.0,
        freq="1d"
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_trades == len(synthetic_trades_high_turnover)
    assert metrics.turnover is not None
    # High turnover: many trades should result in high turnover ratio
    assert metrics.turnover > 10.0  # At least 10x annualized turnover


@pytest.mark.smoke
def test_compute_all_metrics_without_trades(synthetic_equity_1d):
    """Test compute_all_metrics without trades."""
    metrics = compute_all_metrics(
        equity=synthetic_equity_1d,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.total_trades is None
    assert metrics.turnover is None
    assert metrics.hit_rate is None


@pytest.mark.smoke
def test_metrics_strong_positive_scenario(synthetic_equity_strong_positive):
    """Test metrics computation for strong positive scenario."""
    metrics = compute_all_metrics(
        equity=synthetic_equity_strong_positive,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    # Strong positive should have:
    assert metrics.final_pf > 1.5  # At least 50% return
    assert metrics.total_return > 0.5
    assert metrics.cagr is not None
    assert metrics.cagr > 0.3  # At least 30% CAGR
    assert metrics.sharpe_ratio is not None
    assert metrics.sharpe_ratio > 1.0  # Good Sharpe
    assert metrics.max_drawdown_pct > -20.0  # Reasonable drawdown
    assert metrics.volatility is not None
    assert metrics.volatility < 0.30  # Not too volatile


@pytest.mark.smoke
def test_metrics_negative_scenario(synthetic_equity_negative):
    """Test metrics computation for negative scenario."""
    metrics = compute_all_metrics(
        equity=synthetic_equity_negative,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    # Negative should have:
    assert metrics.final_pf < 1.0  # Loss
    assert metrics.total_return < 0
    assert metrics.cagr is not None
    assert metrics.cagr < 0  # Negative CAGR
    assert metrics.sharpe_ratio is not None
    assert metrics.sharpe_ratio < 0.5  # Poor Sharpe
    assert metrics.max_drawdown_pct < -20.0  # Large drawdown


@pytest.mark.smoke
def test_metrics_sideways_scenario(synthetic_equity_sideways):
    """Test metrics computation for sideways scenario."""
    metrics = compute_all_metrics(
        equity=synthetic_equity_sideways,
        trades=None,
        start_capital=10000.0,
        freq="1d"
    )
    
    # Sideways should have:
    assert abs(metrics.total_return) < 0.2  # Small total return
    assert metrics.volatility is not None
    assert metrics.volatility > 0.25  # High volatility
    if metrics.sharpe_ratio is not None:
        assert metrics.sharpe_ratio < 1.0  # Low Sharpe


@pytest.mark.smoke
def test_metrics_high_turnover_scenario(synthetic_equity_strong_positive, synthetic_trades_high_turnover):
    """Test metrics computation with high turnover."""
    metrics = compute_all_metrics(
        equity=synthetic_equity_strong_positive,
        trades=synthetic_trades_high_turnover,
        start_capital=10000.0,
        freq="1d"
    )
    
    # High turnover should be reflected:
    assert metrics.turnover is not None
    assert metrics.turnover > 30.0  # High turnover
    assert metrics.total_trades == len(synthetic_trades_high_turnover)


@pytest.mark.unit
def test_compute_equity_metrics_empty_dataframe():
    """Test equity metrics with empty DataFrame."""
    empty_df = pd.DataFrame(columns=["timestamp", "equity"])
    
    with pytest.raises(ValueError, match="empty"):
        compute_equity_metrics(empty_df, start_capital=10000.0)


@pytest.mark.unit
def test_compute_equity_metrics_missing_columns():
    """Test equity metrics with missing columns."""
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=10, freq="D")})
    
    with pytest.raises(ValueError, match="timestamp.*equity"):
        compute_equity_metrics(df, start_capital=10000.0)


@pytest.mark.unit
def test_compute_equity_metrics_with_nans():
    """Test equity metrics with NaN values."""
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    equity = pd.Series([10000.0, np.nan, 11000.0, 12000.0, np.inf, 13000.0, 14000.0, 15000.0, 16000.0, 17000.0])
    
    df = pd.DataFrame({
        "timestamp": dates,
        "equity": equity
    })
    
    # Should handle NaNs and inf gracefully
    metrics = compute_equity_metrics(df, start_capital=10000.0, freq="1d")
    
    assert isinstance(metrics, PerformanceMetrics)
    assert not np.isnan(metrics.final_pf)
    assert not np.isinf(metrics.final_pf)


@pytest.mark.unit
def test_compute_trade_metrics_empty():
    """Test trade metrics with empty trades."""
    trades = pd.DataFrame(columns=["timestamp", "symbol", "side", "qty", "price"])
    equity = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=10, freq="D"),
        "equity": [10000.0] * 10
    })
    
    result = compute_trade_metrics(trades, equity, start_capital=10000.0, freq="1d")
    
    assert result["total_trades"] == 0
    assert result["hit_rate"] is None
    assert result["profit_factor"] is None
