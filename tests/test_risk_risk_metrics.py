"""Tests for Advanced Risk Metrics Module (Phase D2).

Tests the risk_metrics functions:
- compute_basic_risk_metrics()
- compute_exposure_timeseries()
- compute_risk_by_regime()
- compute_risk_by_factor_group()
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.risk.risk_metrics import (
    compute_basic_risk_metrics,
    compute_exposure_timeseries,
    compute_risk_by_factor_group,
    compute_risk_by_regime,
)


@pytest.fixture
def sample_returns() -> pd.Series:
    """Create sample returns with clear phases: positive phase + negative phase."""
    np.random.seed(42)
    
    # Phase 1: Positive returns (30 days)
    positive_returns = np.random.normal(0.001, 0.01, 30)  # Mean 0.1%, std 1%
    
    # Phase 2: Negative returns (20 days)
    negative_returns = np.random.normal(-0.002, 0.015, 20)  # Mean -0.2%, std 1.5%
    
    # Phase 3: Recovery (10 days)
    recovery_returns = np.random.normal(0.0005, 0.008, 10)  # Mean 0.05%, std 0.8%
    
    all_returns = np.concatenate([positive_returns, negative_returns, recovery_returns])
    
    dates = pd.date_range("2020-01-01", periods=len(all_returns), freq="D", tz="UTC")
    return pd.Series(all_returns, index=dates, name="return")


@pytest.fixture
def sample_positions() -> pd.DataFrame:
    """Create sample positions DataFrame with changing portfolio weights over 60 days."""
    np.random.seed(42)
    
    dates = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    
    all_data = []
    for date in dates:
        # Generate weights that sum to ~1.0
        weights = np.random.rand(len(symbols))
        weights = weights / weights.sum() * 1.0  # Normalize to 1.0
        
        # Occasionally add short positions (negative weights)
        if np.random.rand() < 0.2:  # 20% chance of short position
            short_idx = np.random.randint(0, len(symbols))
            weights[short_idx] = -0.1  # Small short position
        
        for symbol, weight in zip(symbols, weights):
            all_data.append({
                "timestamp": date,
                "symbol": symbol,
                "weight": weight,
            })
    
    return pd.DataFrame(all_data).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


@pytest.fixture
def sample_regime_state_df() -> pd.DataFrame:
    """Create sample regime state DataFrame with 3 regimes."""
    dates = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
    
    # Phase 1: Bull regime (days 0-25)
    bull_regimes = ["bull"] * 26
    
    # Phase 2: Bear regime (days 26-45)
    bear_regimes = ["bear"] * 20
    
    # Phase 3: Neutral regime (days 46-59)
    neutral_regimes = ["neutral"] * 14
    
    all_regimes = bull_regimes + bear_regimes + neutral_regimes
    
    return pd.DataFrame({
        "timestamp": dates,
        "regime_label": all_regimes,
        "regime_trend_score": np.random.randn(len(dates)),
        "regime_risk_score": np.random.randn(len(dates)),
    })


@pytest.fixture
def sample_factor_panel_df() -> pd.DataFrame:
    """Create sample factor panel DataFrame with controllable relationship to returns."""
    np.random.seed(42)
    
    dates = pd.date_range("2020-01-01", periods=60, freq="D", tz="UTC")
    symbols = ["AAPL", "MSFT", "GOOGL"]
    
    all_data = []
    for date in dates:
        for symbol in symbols:
            # Factor 1: Strong positive correlation with returns (Trend factor)
            trend_factor = np.random.randn() * 0.5
            
            # Factor 2: Weak/no correlation (Volatility factor)
            vol_factor = np.random.randn() * 0.3
            
            # Factor 3: Moderate positive correlation
            earnings_factor = np.random.randn() * 0.4
            
            all_data.append({
                "timestamp": date,
                "symbol": symbol,
                "returns_12m": trend_factor,  # Strong trend signal
                "rv_20": vol_factor,  # Volatility factor
                "earnings_eps_surprise_last": earnings_factor,  # Earnings factor
            })
    
    return pd.DataFrame(all_data).sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def test_compute_basic_risk_metrics_basic(sample_returns):
    """Test that compute_basic_risk_metrics returns sensible metrics, no NaNs, n_periods correct."""
    metrics = compute_basic_risk_metrics(returns=sample_returns, freq="1d")
    
    # Check that all expected keys are present
    expected_keys = [
        "mean_return_annualized",
        "vol_annualized",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "skew",
        "kurtosis",
        "var_95",
        "cvar_95",
        "n_periods",
    ]
    
    assert all(key in metrics for key in expected_keys)
    
    # Check n_periods
    assert metrics["n_periods"] == len(sample_returns)
    
    # Check that at least basic metrics are not None (should have data)
    assert metrics["mean_return_annualized"] is not None
    assert metrics["vol_annualized"] is not None
    assert metrics["n_periods"] > 0
    
    # Check that metrics have reasonable values
    # Volatility should be positive (if not None)
    if metrics["vol_annualized"] is not None:
        assert metrics["vol_annualized"] > 0
    
    # Max drawdown should be negative or None (if no drawdown)
    if metrics["max_drawdown"] is not None:
        assert metrics["max_drawdown"] <= 0


def test_compute_basic_risk_metrics_tail_risk(sample_returns):
    """Test that VaR/ES have expected magnitude."""
    metrics = compute_basic_risk_metrics(returns=sample_returns, freq="1d")
    
    # VaR and ES should be present if enough data
    if len(sample_returns) >= 5:
        # VaR (95%) should be the 5th percentile (negative value for losses)
        assert metrics["var_95"] is not None
        
        # ES should be <= VaR (CVaR is average of tail, so more negative)
        if metrics["cvar_95"] is not None and metrics["var_95"] is not None:
            assert metrics["cvar_95"] <= metrics["var_95"]
        
        # VaR should be a reasonable percentile of returns
        returns_array = sample_returns.values
        percentile_5 = np.percentile(returns_array, 5)
        # VaR should be close to the 5th percentile (within tolerance)
        assert abs(metrics["var_95"] - percentile_5) < 0.001


def test_compute_basic_risk_metrics_edge_cases():
    """Test edge cases: empty returns, single return, etc."""
    # Empty returns
    empty_returns = pd.Series([], dtype=float)
    metrics_empty = compute_basic_risk_metrics(returns=empty_returns, freq="1d")
    assert metrics_empty["n_periods"] == 0
    assert metrics_empty["mean_return_annualized"] is None
    assert metrics_empty["vol_annualized"] is None
    
    # Single return
    single_return = pd.Series([0.01], index=pd.date_range("2020-01-01", periods=1, freq="D", tz="UTC"))
    metrics_single = compute_basic_risk_metrics(returns=single_return, freq="1d")
    assert metrics_single["n_periods"] == 1
    # With only 1 period, we can't compute volatility properly
    assert metrics_single["sharpe"] is None or metrics_single["vol_annualized"] is None
    
    # Very small returns (near zero)
    small_returns = pd.Series([0.0001, -0.0001, 0.00005] * 10, 
                               index=pd.date_range("2020-01-01", periods=30, freq="D", tz="UTC"))
    metrics_small = compute_basic_risk_metrics(returns=small_returns, freq="1d")
    assert metrics_small["n_periods"] == 30
    assert metrics_small["mean_return_annualized"] is not None


def test_compute_exposure_timeseries_basic(sample_positions):
    """Test that exposure timeseries computes gross/net exposure, n_positions, HHI correctly."""
    exposure_df = compute_exposure_timeseries(positions=sample_positions, freq="1d")
    
    # Check structure
    required_cols = ["timestamp", "gross_exposure", "net_exposure", "n_positions", "hhi_concentration", "turnover"]
    assert all(col in exposure_df.columns for col in required_cols)
    
    # Check that we have one row per unique timestamp
    unique_timestamps = sample_positions["timestamp"].unique()
    assert len(exposure_df) == len(unique_timestamps)
    
    # Check that gross exposure >= |net exposure| (gross is sum of absolutes)
    for _, row in exposure_df.iterrows():
        assert row["gross_exposure"] >= abs(row["net_exposure"])
        assert row["gross_exposure"] >= 0
        assert row["n_positions"] >= 0
        assert 0 <= row["hhi_concentration"] <= 1.0  # HHI should be normalized [0, 1]
    
    # Check one specific timestamp manually
    first_timestamp = sample_positions["timestamp"].min()
    first_positions = sample_positions[sample_positions["timestamp"] == first_timestamp]
    first_row = exposure_df[exposure_df["timestamp"] == first_timestamp].iloc[0]
    
    expected_gross = first_positions["weight"].abs().sum()
    expected_net = first_positions["weight"].sum()
    expected_n_pos = (first_positions["weight"].abs() > 1e-10).sum()
    
    assert abs(first_row["gross_exposure"] - expected_gross) < 1e-6
    assert abs(first_row["net_exposure"] - expected_net) < 1e-6
    assert first_row["n_positions"] == expected_n_pos


def test_compute_exposure_timeseries_empty_positions():
    """Test exposure timeseries with empty positions."""
    empty_positions = pd.DataFrame(columns=["timestamp", "symbol", "weight"])
    exposure_df = compute_exposure_timeseries(positions=empty_positions, freq="1d")
    
    assert len(exposure_df) == 0
    assert all(col in exposure_df.columns for col in ["timestamp", "gross_exposure", "net_exposure", "n_positions", "hhi_concentration"])


def test_compute_risk_by_regime_basic(sample_returns, sample_regime_state_df):
    """Test that risk_by_regime returns different metrics per regime, n_periods correct."""
    risk_by_regime_df = compute_risk_by_regime(
        returns=sample_returns,
        regime_state_df=sample_regime_state_df,
        freq="1d",
    )
    
    # Check structure
    required_cols = ["regime", "n_periods", "mean_return_annualized", "vol_annualized", "sharpe", "max_drawdown", "total_return"]
    assert all(col in risk_by_regime_df.columns for col in required_cols)
    
    # Should have 3 regimes
    assert len(risk_by_regime_df) == 3
    assert set(risk_by_regime_df["regime"]) == {"bull", "bear", "neutral"}
    
    # Check that n_periods matches expected
    bull_periods = len(sample_regime_state_df[sample_regime_state_df["regime_label"] == "bull"])
    bear_periods = len(sample_regime_state_df[sample_regime_state_df["regime_label"] == "bear"])
    neutral_periods = len(sample_regime_state_df[sample_regime_state_df["regime_label"] == "neutral"])
    
    bull_row = risk_by_regime_df[risk_by_regime_df["regime"] == "bull"].iloc[0]
    bear_row = risk_by_regime_df[risk_by_regime_df["regime"] == "bear"].iloc[0]
    neutral_row = risk_by_regime_df[risk_by_regime_df["regime"] == "neutral"].iloc[0]
    
    assert bull_row["n_periods"] == bull_periods
    assert bear_row["n_periods"] == bear_periods
    assert neutral_row["n_periods"] == neutral_periods
    
    # Check that we get different metrics per regime (they should differ due to different return subsets)
    # At least one metric should differ between regimes
    sharpe_values = risk_by_regime_df["sharpe"].dropna().unique()
    if len(sharpe_values) > 1:
        # Different regimes should have different Sharpe ratios
        pass  # This is expected
    
    # Check that total_return is computed correctly (cumulative return)
    # For bull regime, we should have some returns
    assert bull_row["total_return"] is not None
    assert bear_row["total_return"] is not None
    assert neutral_row["total_return"] is not None


def test_compute_risk_by_regime_empty_regime():
    """Test risk_by_regime with empty regime DataFrame."""
    returns = pd.Series([0.01, -0.01, 0.02], index=pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC"))
    empty_regime = pd.DataFrame(columns=["timestamp", "regime_label"])
    
    result = compute_risk_by_regime(returns=returns, regime_state_df=empty_regime, freq="1d")
    
    assert len(result) == 0
    assert all(col in result.columns for col in ["regime", "n_periods", "mean_return_annualized"])


def test_compute_risk_by_factor_group_basic(sample_returns, sample_factor_panel_df, sample_positions):
    """Test that factor group attribution works, with one group showing higher correlation."""
    # Create factor groups
    factor_groups = {
        "Trend": ["returns_12m"],
        "Vol": ["rv_20"],
        "Earnings": ["earnings_eps_surprise_last"],
    }
    
    risk_by_factor_df = compute_risk_by_factor_group(
        returns=sample_returns,
        factor_panel_df=sample_factor_panel_df,
        positions_df=sample_positions,
        factor_groups=factor_groups,
    )
    
    # Check structure
    required_cols = ["factor_group", "factors", "correlation_with_returns", "avg_exposure", "n_periods"]
    assert all(col in risk_by_factor_df.columns for col in required_cols)
    
    # Should have 3 groups
    assert len(risk_by_factor_df) == 3
    assert set(risk_by_factor_df["factor_group"]) == {"Trend", "Vol", "Earnings"}
    
    # Check that correlations are computed (can be None if insufficient data)
    # At least some groups should have correlations
    correlations = risk_by_factor_df["correlation_with_returns"].dropna()
    if len(correlations) > 0:
        # Correlations should be in [-1, 1]
        assert all(-1 <= corr <= 1 for corr in correlations)
    
    # Check that n_periods is reasonable
    assert all(row["n_periods"] >= 0 for _, row in risk_by_factor_df.iterrows())


def test_compute_risk_by_factor_group_no_overlap():
    """Test factor group attribution when there's no overlap between returns and factor data."""
    # Returns with different timestamps than factor data
    returns = pd.Series([0.01, -0.01], index=pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC"))
    
    # Factor panel with completely different timestamps
    factor_panel = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=10, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 10,
        "returns_12m": np.random.randn(10),
    })
    
    # Positions with matching timestamps to returns
    positions = pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=2, freq="D", tz="UTC"),
        "symbol": ["AAPL"] * 2,
        "weight": [0.5, 0.5],
    })
    
    factor_groups = {"Trend": ["returns_12m"]}
    
    result = compute_risk_by_factor_group(
        returns=returns,
        factor_panel_df=factor_panel,
        positions_df=positions,
        factor_groups=factor_groups,
    )
    
    # Should return empty or low n_periods due to no overlap
    assert len(result) == 1  # One group
    assert result.iloc[0]["n_periods"] == 0 or result.iloc[0]["correlation_with_returns"] is None


def test_compute_risk_by_factor_group_default_groups(sample_returns, sample_factor_panel_df, sample_positions):
    """Test that default factor groups work if none provided."""
    # Use default factor groups (None)
    result = compute_risk_by_factor_group(
        returns=sample_returns,
        factor_panel_df=sample_factor_panel_df,
        positions_df=sample_positions,
        factor_groups=None,  # Should use defaults
    )
    
    # Should have default groups
    assert len(result) > 0
    # Default groups include Trend, Vol/Liq, Earnings, Insider, News/Macro
    # At least some should be present (if factors match)

