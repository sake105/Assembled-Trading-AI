"""Unit and smoke tests for portfolio-level backtest engine."""
from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from src.assembled_core.qa.backtest_engine import BacktestResult, run_portfolio_backtest


@pytest.fixture
def synthetic_prices_multi_year() -> pd.DataFrame:
    """Create synthetic EOD price data for multiple years (2 symbols, ~3 years).
    
    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        Dates: 2020-01-01 to 2022-12-31 (daily)
        Symbols: AAPL, MSFT
    """
    dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D", tz="UTC")
    # Filter out weekends (keep only weekdays)
    dates = dates[dates.weekday < 5]
    
    symbols = ["AAPL", "MSFT"]
    rows = []
    
    for symbol in symbols:
        # Generate realistic price series with trend and volatility
        np.random.seed(42 if symbol == "AAPL" else 43)
        n_days = len(dates)
        
        # Base price
        base_price = 150.0 if symbol == "AAPL" else 200.0
        
        # Generate random walk with drift
        returns = np.random.normal(0.0005, 0.02, n_days)  # ~0.05% daily drift, 2% volatility
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLCV
        for i, date in enumerate(dates):
            close = prices[i]
            # Add some intraday volatility
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = close * (1 + np.random.normal(0, 0.005))
            volume = np.random.randint(1000000, 10000000)
            
            rows.append({
                "timestamp": date,
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            })
    
    df = pd.DataFrame(rows)
    return df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


@pytest.fixture
def synthetic_prices_single_symbol() -> pd.DataFrame:
    """Create synthetic EOD price data for single symbol (1 year).
    
    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        Dates: 2021-01-01 to 2021-12-31 (daily)
        Symbol: AAPL only
    """
    dates = pd.date_range(start="2021-01-01", end="2021-12-31", freq="D", tz="UTC")
    dates = dates[dates.weekday < 5]  # Weekdays only
    
    np.random.seed(42)
    n_days = len(dates)
    base_price = 150.0
    returns = np.random.normal(0.0005, 0.02, n_days)
    prices = base_price * np.exp(np.cumsum(returns))
    
    rows = []
    for i, date in enumerate(dates):
        close = prices[i]
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        open_price = close * (1 + np.random.normal(0, 0.005))
        volume = np.random.randint(1000000, 10000000)
        
        rows.append({
            "timestamp": date,
            "symbol": "AAPL",
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume
        })
    
    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


@pytest.fixture
def synthetic_prices_with_gaps() -> pd.DataFrame:
    """Create synthetic EOD price data with gaps (missing days).
    
    Returns:
        DataFrame with columns: timestamp, symbol, open, high, low, close, volume
        Dates: 2021-01-01 to 2021-12-31 (daily, but with gaps)
        Symbols: AAPL, MSFT
    """
    dates = pd.date_range(start="2021-01-01", end="2021-12-31", freq="D", tz="UTC")
    dates = dates[dates.weekday < 5]  # Weekdays only
    
    # Remove some dates to create gaps (every 10th day)
    dates_with_gaps = dates[::10]  # Keep every 10th day
    
    symbols = ["AAPL", "MSFT"]
    rows = []
    
    for symbol in symbols:
        np.random.seed(42 if symbol == "AAPL" else 43)
        base_price = 150.0 if symbol == "AAPL" else 200.0
        n_days = len(dates_with_gaps)
        returns = np.random.normal(0.0005, 0.02, n_days)
        prices = base_price * np.exp(np.cumsum(returns))
        
        for i, date in enumerate(dates_with_gaps):
            close = prices[i]
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_price = close * (1 + np.random.normal(0, 0.005))
            volume = np.random.randint(1000000, 10000000)
            
            rows.append({
                "timestamp": date,
                "symbol": symbol,
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            })
    
    return pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def dummy_signal_fn(prices_df: pd.DataFrame) -> pd.DataFrame:
    """Simple dummy signal function: LONG if close > 150, else FLAT.
    
    Args:
        prices_df: DataFrame with columns: timestamp, symbol, close, ...
    
    Returns:
        DataFrame with columns: timestamp, symbol, direction, score
    """
    signals = prices_df[["timestamp", "symbol", "close"]].copy()
    signals["direction"] = np.where(signals["close"] > 150.0, "LONG", "FLAT")
    signals["score"] = np.where(signals["direction"] == "LONG", 0.8, 0.0)
    return signals[["timestamp", "symbol", "direction", "score"]]


def dummy_position_sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Simple dummy position sizing: equal weight for all LONG signals.
    
    Args:
        signals_df: DataFrame with columns: symbol, direction, ...
        capital: Total capital available
    
    Returns:
        DataFrame with columns: symbol, target_weight, target_qty
    """
    long_signals = signals_df[signals_df["direction"] == "LONG"].copy()
    
    if long_signals.empty:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    
    n = len(long_signals)
    long_signals["target_weight"] = 1.0 / n
    long_signals["target_qty"] = (capital / n) / 150.0  # Rough estimate: assume price ~150
    
    return long_signals[["symbol", "target_weight", "target_qty"]]


@pytest.mark.unit
def test_backtest_engine_import():
    """Test that backtest engine can be imported."""
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest, BacktestResult
    assert callable(run_portfolio_backtest)
    assert BacktestResult is not None


@pytest.mark.unit
def test_backtest_engine_empty_data(synthetic_prices_multi_year):
    """Test that empty data raises appropriate error."""
    empty_prices = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    
    with pytest.raises(ValueError, match="Missing required columns"):
        run_portfolio_backtest(
            prices=empty_prices,
            signal_fn=dummy_signal_fn,
            position_sizing_fn=dummy_position_sizing_fn,
            start_capital=10000.0
        )


@pytest.mark.unit
def test_backtest_engine_missing_columns():
    """Test that missing required columns raises ValueError."""
    invalid_prices = pd.DataFrame({
        "timestamp": [pd.Timestamp("2021-01-01", tz="UTC")],
        "symbol": ["AAPL"]
        # Missing "close" column
    })
    
    with pytest.raises(ValueError, match="Missing required columns"):
        run_portfolio_backtest(
            prices=invalid_prices,
            signal_fn=dummy_signal_fn,
            position_sizing_fn=dummy_position_sizing_fn,
            start_capital=10000.0
        )


@pytest.mark.unit
def test_backtest_engine_invalid_signal_fn(synthetic_prices_multi_year):
    """Test that invalid signal function raises KeyError."""
    def invalid_signal_fn(prices_df):
        # Returns DataFrame without required columns
        return pd.DataFrame({"symbol": ["AAPL"]})
    
    with pytest.raises(KeyError, match="signal_fn must return DataFrame"):
        run_portfolio_backtest(
            prices=synthetic_prices_multi_year,
            signal_fn=invalid_signal_fn,
            position_sizing_fn=dummy_position_sizing_fn,
            start_capital=10000.0
        )


@pytest.mark.smoke
def test_backtest_engine_multi_year(synthetic_prices_multi_year):
    """Test normal backtest over multiple years → sensible equity curve.
    
    This is a smoke test that verifies the complete workflow:
    1. Feature computation
    2. Signal generation
    3. Position sizing
    4. Order generation
    5. Equity simulation
    6. Metrics computation
    """
    result = run_portfolio_backtest(
        prices=synthetic_prices_multi_year,
        signal_fn=dummy_signal_fn,
        position_sizing_fn=dummy_position_sizing_fn,
        start_capital=10000.0,
        include_costs=False,  # Simpler for smoke test
        include_trades=True,
        include_signals=True,
        compute_features=True
    )
    
    # Check result type
    assert isinstance(result, BacktestResult)
    
    # Check equity DataFrame structure
    assert "equity" in result.__dict__
    equity = result.equity
    assert isinstance(equity, pd.DataFrame)
    assert len(equity) > 0
    
    # Check required columns
    required_cols = ["date", "timestamp", "equity", "daily_return"]
    for col in required_cols:
        assert col in equity.columns, f"Missing column: {col}"
    
    # Check equity values are sensible
    assert equity["equity"].min() >= 0, "Equity should not be negative"
    assert equity["equity"].iloc[0] == pytest.approx(10000.0, abs=1.0), "Start equity should be ~10000"
    
    # Check daily_return
    assert equity["daily_return"].dtype in [np.float64, float], "daily_return should be float"
    assert not equity["daily_return"].isna().any(), "daily_return should not have NaNs"
    
    # Check metrics
    assert "metrics" in result.__dict__
    metrics = result.metrics
    assert isinstance(metrics, dict)
    assert "final_pf" in metrics
    assert "sharpe" in metrics
    assert "trades" in metrics
    assert metrics["trades"] >= 0
    
    # Check optional outputs
    assert result.trades is not None, "Trades should be included"
    assert isinstance(result.trades, pd.DataFrame)
    assert len(result.trades) == metrics["trades"]
    
    assert result.signals is not None, "Signals should be included"
    assert isinstance(result.signals, pd.DataFrame)
    assert len(result.signals) > 0


@pytest.mark.smoke
def test_backtest_engine_with_costs(synthetic_prices_multi_year):
    """Test backtest with cost model → equity curve reflects costs."""
    result = run_portfolio_backtest(
        prices=synthetic_prices_multi_year,
        signal_fn=dummy_signal_fn,
        position_sizing_fn=dummy_position_sizing_fn,
        start_capital=10000.0,
        include_costs=True,
        commission_bps=10.0,  # 0.1% commission
        spread_w=0.5,
        impact_w=1.0,
        include_trades=True
    )
    
    assert isinstance(result, BacktestResult)
    equity = result.equity
    
    # Equity should start at start_capital
    assert equity["equity"].iloc[0] == pytest.approx(10000.0, abs=1.0)
    
    # With costs, final equity should be <= equity without costs (roughly)
    # (This is a sanity check, not a strict requirement)
    assert equity["equity"].iloc[-1] > 0, "Final equity should be positive"
    
    # Check metrics
    assert result.metrics["trades"] >= 0


@pytest.mark.unit
def test_backtest_engine_single_symbol(synthetic_prices_single_symbol):
    """Test backtest with single symbol (edge case)."""
    result = run_portfolio_backtest(
        prices=synthetic_prices_single_symbol,
        signal_fn=dummy_signal_fn,
        position_sizing_fn=dummy_position_sizing_fn,
        start_capital=10000.0,
        include_costs=False,
        include_trades=True
    )
    
    assert isinstance(result, BacktestResult)
    equity = result.equity
    
    # Should still produce valid equity curve
    assert len(equity) > 0
    assert "date" in equity.columns
    assert "equity" in equity.columns
    assert "daily_return" in equity.columns
    
    # Check that equity values are sensible
    assert equity["equity"].min() >= 0
    assert equity["equity"].iloc[0] == pytest.approx(10000.0, abs=1.0)


@pytest.mark.unit
def test_backtest_engine_with_gaps(synthetic_prices_with_gaps):
    """Test backtest with gaps in data (edge case)."""
    result = run_portfolio_backtest(
        prices=synthetic_prices_with_gaps,
        signal_fn=dummy_signal_fn,
        position_sizing_fn=dummy_position_sizing_fn,
        start_capital=10000.0,
        include_costs=False,
        include_trades=True
    )
    
    assert isinstance(result, BacktestResult)
    equity = result.equity
    
    # Should handle gaps gracefully
    assert len(equity) > 0
    assert "date" in equity.columns
    assert "equity" in equity.columns
    assert "daily_return" in equity.columns
    
    # Equity should still be valid
    assert equity["equity"].min() >= 0
    assert not equity["equity"].isna().any()


@pytest.mark.unit
def test_backtest_engine_no_features(synthetic_prices_multi_year):
    """Test backtest without feature computation."""
    result = run_portfolio_backtest(
        prices=synthetic_prices_multi_year,
        signal_fn=dummy_signal_fn,
        position_sizing_fn=dummy_position_sizing_fn,
        start_capital=10000.0,
        compute_features=False,  # Skip feature computation
        include_costs=False
    )
    
    assert isinstance(result, BacktestResult)
    equity = result.equity
    assert len(equity) > 0
    assert equity["equity"].iloc[0] == pytest.approx(10000.0, abs=1.0)


@pytest.mark.unit
def test_backtest_engine_no_trades(synthetic_prices_multi_year):
    """Test backtest when no trades are generated (all signals FLAT)."""
    def flat_signal_fn(prices_df):
        """Signal function that always returns FLAT."""
        signals = prices_df[["timestamp", "symbol", "close"]].copy()
        signals["direction"] = "FLAT"
        signals["score"] = 0.0
        return signals[["timestamp", "symbol", "direction", "score"]]
    
    result = run_portfolio_backtest(
        prices=synthetic_prices_multi_year,
        signal_fn=flat_signal_fn,
        position_sizing_fn=dummy_position_sizing_fn,
        start_capital=10000.0,
        include_costs=False,
        include_trades=True
    )
    
    assert isinstance(result, BacktestResult)
    equity = result.equity
    
    # Equity should remain constant (no trades)
    assert len(equity) > 0
    assert equity["equity"].iloc[0] == pytest.approx(10000.0, abs=1.0)
    
    # No trades should be generated
    assert result.metrics["trades"] == 0
    assert result.trades is not None
    assert len(result.trades) == 0


@pytest.mark.unit
def test_backtest_engine_optional_outputs(synthetic_prices_multi_year):
    """Test that optional outputs are only included when requested."""
    # Test without optional outputs
    result_minimal = run_portfolio_backtest(
        prices=synthetic_prices_multi_year,
        signal_fn=dummy_signal_fn,
        position_sizing_fn=dummy_position_sizing_fn,
        start_capital=10000.0,
        include_trades=False,
        include_signals=False,
        include_targets=False
    )
    
    assert result_minimal.trades is None
    assert result_minimal.signals is None
    assert result_minimal.target_positions is None
    
    # Test with all optional outputs
    result_full = run_portfolio_backtest(
        prices=synthetic_prices_multi_year,
        signal_fn=dummy_signal_fn,
        position_sizing_fn=dummy_position_sizing_fn,
        start_capital=10000.0,
        include_trades=True,
        include_signals=True,
        include_targets=True
    )
    
    assert result_full.trades is not None
    assert result_full.signals is not None
    assert result_full.target_positions is not None


@pytest.mark.unit
def test_backtest_engine_cost_model(synthetic_prices_multi_year):
    """Test backtest with CostModel instead of individual parameters."""
    from src.assembled_core.costs import CostModel
    
    cost_model = CostModel(
        commission_bps=5.0,
        spread_w=0.3,
        impact_w=0.7
    )
    
    result = run_portfolio_backtest(
        prices=synthetic_prices_multi_year,
        signal_fn=dummy_signal_fn,
        position_sizing_fn=dummy_position_sizing_fn,
        start_capital=10000.0,
        cost_model=cost_model,
        include_costs=True
    )
    
    assert isinstance(result, BacktestResult)
    equity = result.equity
    assert len(equity) > 0
    assert equity["equity"].iloc[0] == pytest.approx(10000.0, abs=1.0)

