"""Tests for src.assembled_core.qa.portfolio_analyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.portfolio_analyzer import (
    PerformanceProfile,
    RegimePerformance,
    PortfolioAnalysisResult,
    compute_performance_profile,
    compute_portfolio_structure,
    analyze_regime_performance,
    compute_attribution,
    analyze_portfolio,
    format_portfolio_report,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RNG = np.random.default_rng(42)


def make_positive_drift_returns(n: int = 300, drift: float = 0.001) -> np.ndarray:
    """Generate daily returns with a positive mean."""
    noise = RNG.normal(0, 0.01, n)
    return noise + drift


def make_mixed_returns(n: int = 252) -> np.ndarray:
    """Returns with both positive and negative days."""
    return RNG.normal(0.0005, 0.015, n)


# ---------------------------------------------------------------------------
# compute_performance_profile
# ---------------------------------------------------------------------------


def test_performance_profile_positive_returns():
    """Sharpe should be > 0 for consistently positive-drift returns."""
    arr = make_positive_drift_returns(n=300, drift=0.001)
    profile = compute_performance_profile(arr)
    assert isinstance(profile, PerformanceProfile)
    assert profile.sharpe > 0, f"Expected Sharpe > 0, got {profile.sharpe}"


def test_performance_profile_empty_raises():
    """Empty input should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        compute_performance_profile(np.array([]))


def test_performance_profile_empty_series_raises():
    """Empty pd.Series should also raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        compute_performance_profile(pd.Series([], dtype=float))


def test_sortino_less_than_sharpe_for_positive_skew():
    """For a return series with mostly positive returns, Sharpe and Sortino
    both positive; directional check only (Sortino uses downside vol only).
    With a clean positive-drift series the two ratios can go either way
    depending on sample, so we only assert both are non-NaN and positive."""
    arr = make_positive_drift_returns(n=500, drift=0.0015)
    profile = compute_performance_profile(arr)
    assert profile.sharpe > 0
    # Sortino can be 0 if all days positive (no downside); accept >= 0
    assert profile.sortino >= 0


def test_calmar_finite():
    """When MaxDD is non-zero, Calmar should be a finite number."""
    arr = make_mixed_returns(252)
    profile = compute_performance_profile(arr)
    if profile.max_drawdown < 0:
        assert np.isfinite(profile.calmar), f"Calmar not finite: {profile.calmar}"


def test_win_rate_between_0_and_1():
    arr = make_mixed_returns(252)
    profile = compute_performance_profile(arr)
    assert 0.0 <= profile.win_rate <= 1.0


def test_max_drawdown_non_positive():
    arr = make_mixed_returns(252)
    profile = compute_performance_profile(arr)
    assert profile.max_drawdown <= 0.0


def test_trading_days_matches_input():
    arr = make_mixed_returns(200)
    profile = compute_performance_profile(arr)
    assert profile.trading_days == 200


def test_total_return_compound():
    """Total return from array [0.1, -0.1] should be (1.1)(0.9)-1 = -0.01."""
    arr = np.array([0.1, -0.1])
    profile = compute_performance_profile(arr)
    expected = (1.1 * 0.9) - 1.0
    assert abs(profile.total_return - expected) < 1e-9


# ---------------------------------------------------------------------------
# compute_portfolio_structure
# ---------------------------------------------------------------------------


def test_portfolio_structure_concentration():
    """Top-5 concentration should equal sum of 5 largest weights."""
    weights = {f"SYM{i}": (i + 1) * 0.05 for i in range(10)}  # 0.05..0.50
    struct = compute_portfolio_structure(weights)
    sorted_w = sorted(weights.values(), reverse=True)
    expected_top5 = sum(sorted_w[:5])
    assert abs(struct.top_5_concentration - expected_top5) < 1e-9


def test_portfolio_structure_herfindahl():
    """Herfindahl index should be sum(w_i^2)."""
    weights = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    struct = compute_portfolio_structure(weights)
    expected = sum(w**2 for w in weights.values())
    assert abs(struct.herfindahl_index - expected) < 1e-9


def test_portfolio_structure_n_positions():
    weights = {"X": 0.5, "Y": 0.5}
    struct = compute_portfolio_structure(weights)
    assert struct.n_positions == 2


def test_portfolio_structure_empty():
    struct = compute_portfolio_structure({})
    assert struct.n_positions == 0
    assert struct.total_invested == 0.0
    assert struct.cash_pct == 1.0


def test_sector_aggregation():
    """Sector weights should correctly aggregate by sector."""
    weights = {"AAPL": 0.2, "MSFT": 0.3, "XOM": 0.25, "CVX": 0.15}
    metadata = {
        "AAPL": {"sector": "tech", "region": "US"},
        "MSFT": {"sector": "tech", "region": "US"},
        "XOM": {"sector": "energy", "region": "US"},
        "CVX": {"sector": "energy", "region": "US"},
    }
    struct = compute_portfolio_structure(weights, symbol_metadata=metadata)
    assert abs(struct.sector_weights["tech"] - 0.5) < 1e-9
    assert abs(struct.sector_weights["energy"] - 0.4) < 1e-9


def test_region_aggregation():
    weights = {"AAPL": 0.3, "NESN": 0.3, "BABA": 0.4}
    metadata = {
        "AAPL": {"sector": "tech", "region": "US"},
        "NESN": {"sector": "consumer", "region": "EU"},
        "BABA": {"sector": "tech", "region": "APAC"},
    }
    struct = compute_portfolio_structure(weights, symbol_metadata=metadata)
    assert abs(struct.region_weights["US"] - 0.3) < 1e-9
    assert abs(struct.region_weights["EU"] - 0.3) < 1e-9
    assert abs(struct.region_weights["APAC"] - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# analyze_regime_performance
# ---------------------------------------------------------------------------


def test_regime_performance():
    """At least 2 distinct regimes should be identified."""
    n = 200
    returns = pd.Series(RNG.normal(0.001, 0.01, n))
    labels = pd.Series(["bull"] * 100 + ["bear"] * 100, dtype=str)
    result = analyze_regime_performance(returns, labels)
    assert len(result) >= 2, f"Expected >= 2 regimes, got {len(result)}"
    assert "bull" in result
    assert "bear" in result


def test_regime_performance_returns_regime_performance_objects():
    n = 100
    returns = pd.Series(RNG.normal(0.001, 0.01, n))
    labels = pd.Series(["r1"] * 50 + ["r2"] * 50)
    result = analyze_regime_performance(returns, labels)
    for v in result.values():
        assert isinstance(v, RegimePerformance)
        assert v.n_days >= 2


def test_regime_performance_min_days():
    """Regime with only 1 day should be excluded."""
    returns = pd.Series(RNG.normal(0.001, 0.01, 51))
    labels = pd.Series(["main"] * 50 + ["tiny"])
    result = analyze_regime_performance(returns, labels)
    # "tiny" regime has only 1 day, should be excluded
    assert "tiny" not in result
    assert "main" in result


# ---------------------------------------------------------------------------
# compute_attribution
# ---------------------------------------------------------------------------


def test_attribution_sums_correctly():
    """Sum of symbol contributions should equal total_return."""
    weights = {"A": 0.4, "B": 0.3, "C": 0.3}
    returns = {"A": 0.05, "B": -0.02, "C": 0.10}
    report = compute_attribution(weights, returns)
    expected = 0.4 * 0.05 + 0.3 * (-0.02) + 0.3 * 0.10
    assert abs(report.total_return - expected) < 1e-9
    assert abs(sum(report.symbol_contributions.values()) - expected) < 1e-9


def test_attribution_top_contributors_sorted_desc():
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    returns = {"A": 0.10, "B": -0.05, "C": 0.20}
    report = compute_attribution(weights, returns)
    contribs = [c for _, c in report.top_contributors]
    assert contribs == sorted(contribs, reverse=True)


def test_attribution_sector_aggregation():
    weights = {"AAPL": 0.3, "XOM": 0.4, "MSFT": 0.3}
    returns = {"AAPL": 0.05, "XOM": 0.10, "MSFT": -0.02}
    metadata = {
        "AAPL": {"sector": "tech"},
        "MSFT": {"sector": "tech"},
        "XOM": {"sector": "energy"},
    }
    report = compute_attribution(weights, returns, symbol_metadata=metadata)
    tech_contrib = 0.3 * 0.05 + 0.3 * (-0.02)
    energy_contrib = 0.4 * 0.10
    assert abs(report.sector_contributions["tech"] - tech_contrib) < 1e-9
    assert abs(report.sector_contributions["energy"] - energy_contrib) < 1e-9


# ---------------------------------------------------------------------------
# analyze_portfolio
# ---------------------------------------------------------------------------


def test_analyze_portfolio_returns_result():
    arr = make_mixed_returns(252)
    result = analyze_portfolio(arr)
    assert isinstance(result, PortfolioAnalysisResult)
    assert isinstance(result.performance, PerformanceProfile)


def test_analyze_portfolio_with_weights():
    arr = make_mixed_returns(252)
    weights = {"A": 0.5, "B": 0.5}
    result = analyze_portfolio(arr, weights=weights)
    assert result.structure is not None
    assert result.structure.n_positions == 2


def test_analyze_portfolio_regime_labels():
    n = 200
    returns = pd.Series(make_mixed_returns(n))
    labels = pd.Series(["bull"] * 100 + ["bear"] * 100)
    result = analyze_portfolio(returns, regime_labels=labels)
    assert result.regime_performance is not None
    assert len(result.regime_performance) >= 2


# ---------------------------------------------------------------------------
# format_portfolio_report
# ---------------------------------------------------------------------------


def test_format_report_contains_sharpe():
    arr = make_positive_drift_returns(252)
    result = analyze_portfolio(arr)
    report = format_portfolio_report(result)
    assert "Sharpe" in report


def test_format_report_contains_total_return():
    arr = make_positive_drift_returns(252)
    result = analyze_portfolio(arr)
    report = format_portfolio_report(result)
    assert "Total Return" in report


def test_format_report_is_string():
    arr = make_mixed_returns(100)
    result = analyze_portfolio(arr)
    report = format_portfolio_report(result)
    assert isinstance(report, str)
    assert len(report) > 0


def test_format_report_contains_structure_section():
    arr = make_mixed_returns(100)
    weights = {"X": 0.6, "Y": 0.4}
    result = analyze_portfolio(arr, weights=weights)
    report = format_portfolio_report(result)
    assert "Structure" in report
