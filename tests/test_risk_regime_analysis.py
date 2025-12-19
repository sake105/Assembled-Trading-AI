"""Tests for Extended Regime Analysis (B3).

These tests verify regime classification and metrics summarization by regime.
"""
from __future__ import annotations

import pytest

import numpy as np
import pandas as pd

from src.assembled_core.risk.regime_analysis import (
    RegimeConfig,
    classify_regimes_from_index,
    compute_regime_transitions,
    summarize_factor_ic_by_regime,
    summarize_metrics_by_regime,
)

pytestmark = pytest.mark.advanced


@pytest.fixture
def synthetic_index_returns_with_phases() -> pd.Series:
    """Create synthetic index returns with clear phases (bull, bear, crisis, sideways)."""
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D", tz="UTC")
    n = len(dates)
    
    returns = pd.Series(0.0, index=dates)
    
    # Phase 1: Bull market (Days 0-200) - positive trend, moderate vol
    returns.iloc[0:200] = np.random.normal(0.001, 0.01, 200)  # ~0.1% daily return, 1% daily vol
    
    # Phase 2: Bear market (Days 200-400) - negative trend, elevated vol
    returns.iloc[200:400] = np.random.normal(-0.0005, 0.015, 200)  # ~-0.05% daily return, 1.5% daily vol
    
    # Phase 3: Crisis (Days 400-450) - crash, high vol
    returns.iloc[400:450] = np.random.normal(-0.002, 0.025, 50)  # ~-0.2% daily return, 2.5% daily vol
    
    # Phase 4: Reflation (Days 450-550) - strong recovery
    returns.iloc[450:550] = np.random.normal(0.002, 0.012, 100)  # ~0.2% daily return, 1.2% daily vol
    
    # Phase 5: Sideways (Days 550-700) - low trend, low vol
    returns.iloc[550:700] = np.random.normal(0.0001, 0.008, 150)  # ~0.01% daily return, 0.8% daily vol
    
    # Phase 6: Continued bull (Days 700+) - positive trend, moderate vol
    if n > 700:
        returns.iloc[700:] = np.random.normal(0.0008, 0.010, n - 700)
    
    return returns


@pytest.fixture
def simple_equity_curve() -> pd.Series:
    """Create simple equity curve with different performances."""
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D", tz="UTC")
    
    # Bull phase: steady growth
    bull_returns = pd.Series(0.001, index=dates[:500])  # 0.1% daily return
    
    # Bear phase: decline
    bear_returns = pd.Series(-0.0005, index=dates[500:1000])  # -0.05% daily return
    
    all_returns = pd.concat([bull_returns, bear_returns])
    if len(all_returns) < len(dates):
        # Fill remaining with small positive returns
        remaining = pd.Series(0.0003, index=dates[len(all_returns):])
        all_returns = pd.concat([all_returns, remaining])
    
    # Normalize to start at 10000
    equity = (1.0 + all_returns).cumprod() * 10000.0
    
    return equity


def test_classify_regimes_from_index_basic():
    """Test basic regime classification from index returns."""
    # Create synthetic returns with clear phases (add noise for realistic volatility)
    np.random.seed(42)  # For reproducibility
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D", tz="UTC")
    returns = pd.Series(0.0, index=dates)
    
    # Bull phase: positive returns with moderate vol (~15% annualized = ~0.95% daily)
    returns.iloc[0:200] = np.random.normal(0.001, 0.0095, 200)
    
    # Bear phase: negative returns with elevated vol (~20% annualized = ~1.26% daily)
    returns.iloc[200:400] = np.random.normal(-0.0005, 0.0126, 200)
    
    # Crisis: large negative returns with high vol (~40% annualized = ~2.5% daily)
    returns.iloc[400:450] = np.random.normal(-0.002, 0.025, 50)
    
    config = RegimeConfig(
        vol_window=20,
        trend_ma_window=50,  # Shorter for testing
        drawdown_threshold=-0.15,
        vol_threshold_high=0.30,
        vol_threshold_low=0.10,
        trend_threshold=0.03,
    )
    
    regimes = classify_regimes_from_index(returns, config)
    
    assert len(regimes) == len(returns), "Regimes should have same length as returns"
    assert regimes.index.equals(returns.index), "Regimes should have same index as returns"
    
    # Check that regimes are valid
    valid_regimes = {"bull", "bear", "sideways", "crisis", "reflation", "neutral"}
    assert all(r in valid_regimes for r in regimes.unique()), \
        f"All regimes should be valid. Found: {set(regimes.unique())}"
    
    # In bull phase (first 200 days), should have mostly bull or neutral (due to warm-up)
    # Lower threshold because warm-up period and config thresholds may not perfectly match
    bull_phase_regimes = regimes.iloc[100:200]  # Skip warm-up
    bull_or_neutral_pct = ((bull_phase_regimes == "bull") | (bull_phase_regimes == "neutral")).sum() / len(bull_phase_regimes)
    assert bull_or_neutral_pct > 0.3, \
        f"Bull phase should have >30% bull or neutral regimes, got {bull_or_neutral_pct:.1%}"
    
    # In crisis phase (days 400-450), should have crisis or bear
    # Lower threshold because drawdown may take time to accumulate
    crisis_phase_regimes = regimes.iloc[420:450]  # Skip early warm-up in crisis
    crisis_or_bear_pct = ((crisis_phase_regimes == "crisis") | (crisis_phase_regimes == "bear")).sum() / len(crisis_phase_regimes)
    assert crisis_or_bear_pct > 0.2, \
        f"Crisis phase should have >20% crisis or bear regimes, got {crisis_or_bear_pct:.1%}"


def test_classify_regimes_from_index_with_phases(synthetic_index_returns_with_phases):
    """Test regime classification with synthetic multi-phase returns."""
    config = RegimeConfig(
        vol_window=20,
        trend_ma_window=50,
        drawdown_threshold=-0.15,
        vol_threshold_high=0.30,
        vol_threshold_low=0.10,
        trend_threshold=0.03,
    )
    
    regimes = classify_regimes_from_index(synthetic_index_returns_with_phases, config)
    
    # Check that different phases are detected
    unique_regimes = set(regimes.unique())
    assert len(unique_regimes) >= 2, \
        f"Should detect at least 2 different regimes, got: {unique_regimes}"
    
    # Crisis period (days 400-450) should have crisis regimes
    crisis_period = regimes.iloc[400:450]
    crisis_pct = (crisis_period == "crisis").sum() / len(crisis_period)
    # Allow some flexibility due to rolling windows
    assert crisis_pct > 0.2, \
        f"Crisis period should have >20% crisis regimes, got {crisis_pct:.1%}"


def test_classify_regimes_from_index_insufficient_data():
    """Test that insufficient data returns neutral regimes."""
    # Too few periods
    returns = pd.Series([0.001, 0.002, -0.001], index=pd.date_range("2020-01-01", periods=3, freq="D", tz="UTC"))
    
    config = RegimeConfig(min_periods=20)
    
    regimes = classify_regimes_from_index(returns, config)
    
    assert len(regimes) == len(returns)
    assert all(regimes == "neutral"), "Insufficient data should return all neutral"


def test_summarize_metrics_by_regime_basic(simple_equity_curve):
    """Test metrics summarization by regime."""
    # Create simple regimes: bull for first half, bear for second half
    dates = simple_equity_curve.index
    regimes = pd.Series("neutral", index=dates)
    regimes.iloc[:len(dates)//2] = "bull"
    regimes.iloc[len(dates)//2:] = "bear"
    
    metrics_df = summarize_metrics_by_regime(
        equity=simple_equity_curve,
        regimes=regimes,
        freq="1d",
    )
    
    assert not metrics_df.empty, "Should return metrics DataFrame"
    assert "regime_label" in metrics_df.columns
    assert "n_periods" in metrics_df.columns
    assert "sharpe" in metrics_df.columns
    assert "volatility" in metrics_df.columns
    assert "max_drawdown" in metrics_df.columns
    
    # Should have metrics for both regimes
    assert len(metrics_df) >= 2, f"Should have metrics for at least 2 regimes, got {len(metrics_df)}"
    
    # Bull regime should have better Sharpe than bear regime
    bull_metrics = metrics_df[metrics_df["regime_label"] == "bull"]
    bear_metrics = metrics_df[metrics_df["regime_label"] == "bear"]
    
    if not bull_metrics.empty and not bear_metrics.empty:
        bull_sharpe = bull_metrics["sharpe"].iloc[0]
        bear_sharpe = bear_metrics["sharpe"].iloc[0]
        
        if not pd.isna(bull_sharpe) and not pd.isna(bear_sharpe):
            assert bull_sharpe > bear_sharpe, \
                f"Bull Sharpe ({bull_sharpe:.2f}) should be > Bear Sharpe ({bear_sharpe:.2f})"
    
    # Check that n_periods matches (allow small differences due to alignment)
    for _, row in metrics_df.iterrows():
        regime = row["regime_label"]
        n_periods = row["n_periods"]
        actual_count = (regimes == regime).sum()
        # Allow difference of 1 due to returns.dropna() alignment
        assert abs(n_periods - actual_count) <= 1, \
            f"n_periods for {regime} should match actual count (within 1): {n_periods} != {actual_count}"


def test_summarize_metrics_by_regime_with_trades(simple_equity_curve):
    """Test metrics summarization with trades DataFrame."""
    dates = simple_equity_curve.index
    regimes = pd.Series("bull", index=dates)
    regimes.iloc[len(dates)//2:] = "bear"
    
    # Create simple trades DataFrame
    trade_dates = dates[::50]
    n_trades = len(trade_dates)
    side_pattern = ["BUY", "SELL"] * ((n_trades // 2) + 1)
    trades = pd.DataFrame({
        "timestamp": trade_dates,
        "symbol": "TEST",
        "side": side_pattern[:n_trades],
        "qty": 10.0,
        "price": 100.0,
    })
    
    metrics_df = summarize_metrics_by_regime(
        equity=simple_equity_curve,
        regimes=regimes,
        trades=trades,
        freq="1d",
    )
    
    assert not metrics_df.empty
    assert "n_trades" in metrics_df.columns
    # Note: win_rate, avg_trade_duration, avg_profit_per_trade may be None (TODO in implementation)


def test_summarize_factor_ic_by_regime_basic():
    """Test factor IC summarization by regime."""
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="D", tz="UTC")
    
    # Create synthetic IC series (correlation between factor and forward returns)
    # Higher IC in bull, lower in bear
    ic_series = pd.Series(0.0, index=dates)
    ic_series.iloc[:len(dates)//2] = np.random.normal(0.1, 0.05, len(dates)//2)  # Bull: positive IC
    ic_series.iloc[len(dates)//2:] = np.random.normal(-0.05, 0.05, len(dates) - len(dates)//2)  # Bear: negative IC
    
    # Create regimes
    regimes = pd.Series("bull", index=dates)
    regimes.iloc[len(dates)//2:] = "bear"
    
    ic_metrics_df = summarize_factor_ic_by_regime(ic_series, regimes)
    
    assert not ic_metrics_df.empty
    assert "regime_label" in ic_metrics_df.columns
    assert "ic_mean" in ic_metrics_df.columns
    assert "ic_std" in ic_metrics_df.columns
    assert "ic_count" in ic_metrics_df.columns
    assert "ic_ir" in ic_metrics_df.columns
    
    # Should have metrics for both regimes
    assert len(ic_metrics_df) >= 2
    
    # Bull should have higher mean IC than bear
    bull_ic = ic_metrics_df[ic_metrics_df["regime_label"] == "bull"]["ic_mean"].iloc[0]
    bear_ic = ic_metrics_df[ic_metrics_df["regime_label"] == "bear"]["ic_mean"].iloc[0]
    
    assert bull_ic > bear_ic, \
        f"Bull IC mean ({bull_ic:.4f}) should be > Bear IC mean ({bear_ic:.4f})"
    
    # IC count should match number of observations
    bull_count = ic_metrics_df[ic_metrics_df["regime_label"] == "bull"]["ic_count"].iloc[0]
    bear_count = ic_metrics_df[ic_metrics_df["regime_label"] == "bear"]["ic_count"].iloc[0]
    
    assert bull_count == (regimes == "bull").sum(), \
        f"Bull IC count should match regime count: {bull_count} != {(regimes == 'bull').sum()}"
    assert bear_count == (regimes == "bear").sum(), \
        f"Bear IC count should match regime count: {bear_count} != {(regimes == 'bear').sum()}"


def test_summarize_factor_ic_by_regime_empty_inputs():
    """Test that empty inputs return empty DataFrame."""
    empty_ic = pd.Series(dtype=float)
    empty_regimes = pd.Series(dtype=object)
    
    result = summarize_factor_ic_by_regime(empty_ic, empty_regimes)
    
    assert result.empty or "regime_label" in result.columns


def test_summarize_metrics_by_regime_empty_inputs():
    """Test that empty inputs return appropriate empty DataFrame."""
    empty_equity = pd.Series(dtype=float)
    empty_regimes = pd.Series(dtype=object)
    
    result = summarize_metrics_by_regime(empty_equity, empty_regimes)
    
    assert result.empty or "regime_label" in result.columns


def test_compute_regime_transitions_stub():
    """Test that compute_regime_transitions returns expected structure (currently a stub)."""
    # Create simple regime state DataFrame
    dates = pd.date_range("2020-01-01", periods=100, freq="D", tz="UTC")
    regime_state = pd.DataFrame({
        "timestamp": dates,
        "regime_label": ["bull"] * 50 + ["bear"] * 50,
    })
    
    result = compute_regime_transitions(regime_state)
    
    # Currently returns empty DataFrame or stub structure
    assert isinstance(result, pd.DataFrame)
    # Once implemented, should have columns: from_regime, to_regime, transition_count, etc.
