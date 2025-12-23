"""Tests for portfolio risk metrics computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.risk_metrics import compute_portfolio_risk_metrics

pytestmark = pytest.mark.phase8


@pytest.fixture
def linear_equity() -> pd.Series:
    """Create equity that linearly increases (low volatility, no drawdown)."""
    return pd.Series([10000.0, 10100.0, 10200.0, 10300.0, 10400.0, 10500.0])


@pytest.fixture
def equity_with_crash() -> pd.Series:
    """Create equity with a clear crash (high drawdown)."""
    return pd.Series(
        [
            10000.0,  # Start
            10100.0,  # +1%
            10200.0,  # +2%
            10300.0,  # +3%
            8000.0,  # Crash: -22.3% from peak
            8100.0,  # Recovery
            8200.0,  # Recovery
        ]
    )


@pytest.fixture
def equity_with_volatility() -> pd.Series:
    """Create equity with known volatility pattern."""
    # Generate returns with known std dev
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 100)  # Mean 0.1%, Std 2%
    equity_values = [10000.0]
    for ret in returns:
        equity_values.append(equity_values[-1] * (1 + ret))
    return pd.Series(equity_values)


def test_risk_metrics_linear_equity(linear_equity):
    """Test risk metrics for linear equity (low volatility, no drawdown)."""
    metrics = compute_portfolio_risk_metrics(linear_equity, freq="1d")

    # Should have low volatility
    assert metrics["daily_vol"] is not None
    assert metrics["daily_vol"] >= 0
    assert metrics["ann_vol"] is not None
    assert metrics["ann_vol"] >= 0

    # No drawdown (equity only increases)
    assert metrics["max_drawdown"] == 0.0

    # VaR/ES should be computed (at least 5 points)
    assert metrics["var_95"] is not None
    assert metrics["es_95"] is not None


def test_risk_metrics_equity_with_crash(equity_with_crash):
    """Test risk metrics for equity with crash (high drawdown)."""
    metrics = compute_portfolio_risk_metrics(equity_with_crash, freq="1d")

    # Should have volatility
    assert metrics["daily_vol"] is not None
    assert metrics["ann_vol"] is not None

    # Should have significant drawdown (from 10300 to 8000 = -2300)
    assert metrics["max_drawdown"] < 0
    assert metrics["max_drawdown"] <= -2000  # At least -2000

    # VaR/ES should be computed
    assert metrics["var_95"] is not None
    assert metrics["es_95"] is not None
    # VaR and ES should be negative (losses)
    assert metrics["var_95"] <= 0
    assert metrics["es_95"] <= 0


def test_risk_metrics_equity_with_volatility(equity_with_volatility):
    """Test risk metrics for equity with known volatility pattern."""
    metrics = compute_portfolio_risk_metrics(equity_with_volatility, freq="1d")

    # Should have volatility
    assert metrics["daily_vol"] is not None
    assert metrics["daily_vol"] > 0
    assert metrics["ann_vol"] is not None
    assert metrics["ann_vol"] > 0

    # Annualized vol should be higher than daily vol
    assert metrics["ann_vol"] > metrics["daily_vol"]

    # VaR/ES should be computed
    assert metrics["var_95"] is not None
    assert metrics["es_95"] is not None


def test_risk_metrics_empty_equity():
    """Test risk metrics with empty equity series."""
    empty_equity = pd.Series([], dtype=float)
    metrics = compute_portfolio_risk_metrics(empty_equity, freq="1d")

    assert metrics["daily_vol"] is None
    assert metrics["ann_vol"] is None
    assert metrics["max_drawdown"] == 0.0
    assert metrics["var_95"] is None
    assert metrics["es_95"] is None


def test_risk_metrics_single_point():
    """Test risk metrics with only one data point."""
    single_equity = pd.Series([10000.0])
    metrics = compute_portfolio_risk_metrics(single_equity, freq="1d")

    assert metrics["daily_vol"] is None
    assert metrics["ann_vol"] is None
    assert metrics["max_drawdown"] == 0.0
    assert metrics["var_95"] is None
    assert metrics["es_95"] is None


def test_risk_metrics_two_points():
    """Test risk metrics with only two data points."""
    two_equity = pd.Series([10000.0, 10100.0])
    metrics = compute_portfolio_risk_metrics(two_equity, freq="1d")

    # With 2 points, we get 1 return, which is not enough for std dev
    # So daily_vol should be None
    assert metrics["daily_vol"] is None
    assert metrics["ann_vol"] is None

    # But not VaR/ES (need at least 5 points)
    assert metrics["var_95"] is None
    assert metrics["es_95"] is None

    # But max_drawdown should be computed
    assert metrics["max_drawdown"] is not None


def test_risk_metrics_dataframe_input():
    """Test risk metrics with DataFrame input (equity column)."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
            "equity": [10000.0 + i * 100 for i in range(10)],
        }
    )

    metrics = compute_portfolio_risk_metrics(df, freq="1d")

    assert metrics["daily_vol"] is not None
    assert metrics["ann_vol"] is not None
    assert metrics["var_95"] is not None
    assert metrics["es_95"] is not None


def test_risk_metrics_dataframe_missing_column():
    """Test risk metrics with DataFrame missing equity column."""
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="D"),
            "value": [10000.0 + i * 100 for i in range(10)],
        }
    )

    with pytest.raises(ValueError, match="equity"):
        compute_portfolio_risk_metrics(df, freq="1d")


def test_risk_metrics_freq_5min():
    """Test risk metrics with 5min frequency (different annualization)."""
    equity = pd.Series([10000.0 + i * 10 for i in range(100)])
    metrics = compute_portfolio_risk_metrics(equity, freq="5min")

    assert metrics["daily_vol"] is not None
    assert metrics["ann_vol"] is not None

    # Annualized vol should be higher for 5min (more periods per year)
    assert metrics["ann_vol"] > metrics["daily_vol"]


def test_risk_metrics_var_es_negative():
    """Test that VaR and ES are negative (losses) for typical equity curves."""
    # Equity with some volatility and potential losses
    np.random.seed(123)
    returns = np.random.normal(-0.0005, 0.015, 50)  # Slight negative drift
    equity_values = [10000.0]
    for ret in returns:
        equity_values.append(equity_values[-1] * (1 + ret))
    equity = pd.Series(equity_values)

    metrics = compute_portfolio_risk_metrics(equity, freq="1d")

    # VaR and ES should be negative (worst-case losses)
    if metrics["var_95"] is not None:
        assert metrics["var_95"] <= 0
    if metrics["es_95"] is not None:
        assert metrics["es_95"] <= 0
        # ES should be more negative than VaR (expected loss in tail is worse)
        if metrics["var_95"] is not None:
            assert metrics["es_95"] <= metrics["var_95"]


def test_risk_metrics_es_worse_than_var():
    """Test that ES (Expected Shortfall) is worse than VaR for same confidence level."""
    # Create equity with clear tail risk
    equity = pd.Series(
        [
            10000.0,
            10050.0,
            10100.0,
            10000.0,
            9500.0,  # Some volatility
            9000.0,
            8500.0,
            8000.0,  # Crash
            8200.0,
            8400.0,
            8600.0,  # Recovery
        ]
    )

    metrics = compute_portfolio_risk_metrics(equity, freq="1d")

    if metrics["var_95"] is not None and metrics["es_95"] is not None:
        # ES should be more negative (worse) than VaR
        assert metrics["es_95"] <= metrics["var_95"]


def test_risk_metrics_handles_nan():
    """Test that risk metrics handle NaN values gracefully."""
    equity = pd.Series([10000.0, np.nan, 10100.0, 10200.0, np.nan, 10300.0])
    metrics = compute_portfolio_risk_metrics(equity, freq="1d")

    # Should still compute metrics (NaN values are forward-filled)
    assert metrics["max_drawdown"] is not None
    # May or may not have volatility depending on how many valid points remain


def test_risk_metrics_handles_inf():
    """Test that risk metrics handle infinite values gracefully."""
    equity = pd.Series([10000.0, np.inf, 10100.0, 10200.0, -np.inf, 10300.0])
    metrics = compute_portfolio_risk_metrics(equity, freq="1d")

    # Should still compute metrics (inf values are replaced with NaN and forward-filled)
    assert metrics["max_drawdown"] is not None
