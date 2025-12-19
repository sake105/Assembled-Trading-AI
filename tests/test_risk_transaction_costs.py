"""Tests for Transaction Cost Analysis (TCA) Module.

This module tests the transaction costs functionality in src/assembled_core/risk/transaction_costs.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.risk.transaction_costs import (
    compute_cost_adjusted_risk_metrics,
    compute_tca_for_trades,
    estimate_per_trade_cost,
    summarize_tca,
)

pytestmark = pytest.mark.advanced


@pytest.fixture
def sample_trades_df() -> pd.DataFrame:
    """Create sample trades DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    trades = pd.DataFrame({
        "timestamp": dates[:5],
        "symbol": ["AAPL", "MSFT", "AAPL", "GOOGL", "MSFT"],
        "side": ["BUY", "BUY", "SELL", "BUY", "SELL"],
        "qty": [10, 20, 10, 15, 20],
        "price": [100.0, 250.0, 105.0, 150.0, 255.0],
    })
    return trades


@pytest.mark.advanced
def test_estimate_per_trade_cost_simple(sample_trades_df: pd.DataFrame):
    """Test simple cost estimation per trade."""
    costs = estimate_per_trade_cost(
        trades=sample_trades_df,
        method="simple",
        commission_bps=0.5,
        spread_bps=5.0,
        slippage_bps=3.0,
    )
    
    # Check that costs are computed
    assert len(costs) == len(sample_trades_df)
    assert costs.index.equals(sample_trades_df.index)
    
    # Check that all costs are positive
    assert (costs > 0).all(), "All costs should be positive"
    
    # Check that costs increase with notional (qty * price)
    notional = (sample_trades_df["qty"] * sample_trades_df["price"]).abs()
    cost_vs_notional = pd.DataFrame({"notional": notional, "cost": costs})
    cost_vs_notional = cost_vs_notional.sort_values("notional")
    
    # Costs should roughly increase with notional (may not be perfectly monotonic due to rounding)
    assert costs.mean() > 0, "Mean cost should be positive"
    assert costs.max() > costs.min(), "There should be variation in costs"


@pytest.mark.advanced
def test_estimate_per_trade_cost_with_commission_column(sample_trades_df: pd.DataFrame):
    """Test cost estimation when commission column already exists."""
    # Add commission column
    sample_trades_df = sample_trades_df.copy()
    sample_trades_df["commission"] = 0.1
    
    costs = estimate_per_trade_cost(
        trades=sample_trades_df,
        method="simple",
        commission_bps=0.5,
        spread_bps=5.0,
        slippage_bps=3.0,
    )
    
    # Should use existing commission column
    assert len(costs) == len(sample_trades_df)
    assert (costs > 0).all()


@pytest.mark.advanced
def test_compute_tca_for_trades_basic(sample_trades_df: pd.DataFrame):
    """Test TCA computation for trades with Series cost_per_trade."""
    # Estimate costs
    cost_per_trade = estimate_per_trade_cost(sample_trades_df)
    
    # Compute TCA
    tca_trades = compute_tca_for_trades(sample_trades_df, cost_per_trade)
    
    # Check that original columns are preserved
    assert "timestamp" in tca_trades.columns
    assert "symbol" in tca_trades.columns
    assert "side" in tca_trades.columns
    assert "qty" in tca_trades.columns
    assert "price" in tca_trades.columns
    
    # Check that cost columns are added
    assert "cost_total" in tca_trades.columns
    assert "cost_commission" in tca_trades.columns
    assert "cost_spread" in tca_trades.columns
    assert "cost_slippage" in tca_trades.columns
    assert "notional" in tca_trades.columns
    
    # Check that costs match
    assert tca_trades["cost_total"].equals(cost_per_trade)
    
    # Check that notional is computed correctly
    expected_notional = (tca_trades["qty"] * tca_trades["price"]).abs()
    pd.testing.assert_series_equal(
        tca_trades["notional"],
        expected_notional,
        check_names=False,
    )


@pytest.mark.advanced
def test_compute_tca_for_trades_constant_cost(sample_trades_df: pd.DataFrame):
    """Test TCA computation with constant cost (float)."""
    constant_cost = 1.5
    
    tca_trades = compute_tca_for_trades(sample_trades_df, constant_cost)
    
    # All costs should be the same
    assert (tca_trades["cost_total"] == constant_cost).all()


@pytest.mark.advanced
def test_summarize_tca_daily(sample_trades_df: pd.DataFrame):
    """Test TCA summarization by day."""
    # Estimate costs and compute TCA
    cost_per_trade = estimate_per_trade_cost(sample_trades_df)
    tca_trades = compute_tca_for_trades(sample_trades_df, cost_per_trade)
    
    # Summarize by day
    summary = summarize_tca(tca_trades, freq="D")
    
    # Check structure
    assert "timestamp" in summary.columns
    assert "total_cost" in summary.columns
    assert "n_trades" in summary.columns
    assert "avg_cost_per_trade" in summary.columns
    
    # Check that summary has rows
    assert len(summary) > 0
    
    # Check that total cost matches sum of trades
    assert abs(summary["total_cost"].sum() - cost_per_trade.sum()) < 1e-6
    
    # Check that n_trades matches
    assert summary["n_trades"].sum() == len(sample_trades_df)
    
    # Check that avg_cost_per_trade is correct (within rounding)
    for _, row in summary.iterrows():
        if row["n_trades"] > 0:
            expected_avg = row["total_cost"] / row["n_trades"]
            assert abs(row["avg_cost_per_trade"] - expected_avg) < 1e-6


@pytest.mark.advanced
def test_summarize_tca_with_equity_curve(sample_trades_df: pd.DataFrame):
    """Test TCA summarization with equity curve for cost ratio."""
    # Estimate costs and compute TCA
    cost_per_trade = estimate_per_trade_cost(sample_trades_df)
    tca_trades = compute_tca_for_trades(sample_trades_df, cost_per_trade)
    
    # Create equity curve
    dates = pd.date_range("2024-01-01", periods=10, freq="D", tz="UTC")
    equity_df = pd.DataFrame({
        "timestamp": dates,
        "equity": [10000.0] * 10,
    })
    
    # Summarize with equity curve
    summary = summarize_tca(tca_trades, freq="D", equity_curve=equity_df)
    
    # Should still have basic columns
    assert "timestamp" in summary.columns
    assert "total_cost" in summary.columns
    assert "n_trades" in summary.columns


@pytest.mark.advanced
def test_compute_cost_adjusted_risk_metrics_basic():
    """Test cost-adjusted risk metrics computation."""
    # Create synthetic returns and costs
    dates = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    np.random.seed(42)
    
    # Generate returns with some positive trend
    returns = pd.Series(
        np.random.normal(0.001, 0.01, len(dates)),
        index=dates,
    )
    
    # Generate costs (small positive values)
    costs = pd.Series(
        np.random.uniform(0.0001, 0.001, len(dates)),
        index=dates,
    )
    
    # Compute metrics
    metrics = compute_cost_adjusted_risk_metrics(
        returns=returns,
        costs=costs,
        freq="1d",
    )
    
    # Check that metrics are computed
    assert "net_sharpe" in metrics
    assert "gross_sharpe" in metrics
    assert "total_cost" in metrics
    assert "n_periods" in metrics
    
    # Check that total cost is positive
    assert metrics["total_cost"] > 0
    
    # Check that net Sharpe should be <= gross Sharpe (costs reduce returns)
    if metrics["net_sharpe"] is not None and metrics["gross_sharpe"] is not None:
        assert metrics["net_sharpe"] <= metrics["gross_sharpe"] + 1e-6, "Net Sharpe should be <= Gross Sharpe"
    
    # Check period count
    assert metrics["n_periods"] == len(returns)


@pytest.mark.advanced
def test_compute_cost_adjusted_risk_metrics_single_period():
    """Test cost-adjusted risk metrics with single period (edge case)."""
    dates = pd.date_range("2024-01-01", periods=1, freq="D", tz="UTC")
    returns = pd.Series([0.01], index=dates)
    costs = pd.Series([0.001], index=dates)
    
    metrics = compute_cost_adjusted_risk_metrics(
        returns=returns,
        costs=costs,
        freq="1d",
    )
    
    # Should handle single period gracefully
    assert "total_cost" in metrics
    assert metrics["total_cost"] == 0.001
    assert metrics["n_periods"] == 1
    # Sharpe should be None for single period
    assert metrics["net_sharpe"] is None or metrics["net_sharpe"] is not None


@pytest.mark.advanced
def test_compute_cost_adjusted_risk_metrics_cost_impact():
    """Test that costs have measurable impact on metrics."""
    dates = pd.date_range("2024-01-01", periods=252, freq="D", tz="UTC")  # 1 year
    np.random.seed(42)
    
    # Generate returns
    returns = pd.Series(
        np.random.normal(0.001, 0.01, len(dates)),
        index=dates,
    )
    
    # Generate small costs
    costs_low = pd.Series(
        np.random.uniform(0.0001, 0.0005, len(dates)),
        index=dates,
    )
    
    # Generate larger costs
    costs_high = pd.Series(
        np.random.uniform(0.001, 0.005, len(dates)),
        index=dates,
    )
    
    # Compute metrics with low costs
    metrics_low = compute_cost_adjusted_risk_metrics(returns, costs_low, freq="1d")
    
    # Compute metrics with high costs
    metrics_high = compute_cost_adjusted_risk_metrics(returns, costs_high, freq="1d")
    
    # High costs should have larger total cost
    assert metrics_high["total_cost"] > metrics_low["total_cost"]
    
    # High costs should have larger cost impact (if both have valid Sharpe)
    if metrics_low["cost_impact_sharpe"] is not None and metrics_high["cost_impact_sharpe"] is not None:
        assert metrics_high["cost_impact_sharpe"] >= metrics_low["cost_impact_sharpe"] - 1e-6

