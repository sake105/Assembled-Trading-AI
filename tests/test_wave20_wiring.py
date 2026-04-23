"""Tests for wave-20 module wiring into trading_cycle.py.

Covers:
  Step 4.93 — portfolio.robust_optimizer (compute_robust_weights)
  Step 8.5  — qa.deflated_sharpe (deflated_sharpe)
  Step 8.6  — risk.factor_exposures (compute_factor_exposures, summarize_factor_exposures)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.portfolio.robust_optimizer import (
    compute_robust_weights,
    RobustOptResult,
)
from src.assembled_core.qa.deflated_sharpe import deflated_sharpe, DSRResult
from src.assembled_core.risk.factor_exposures import (
    compute_factor_exposures,
    summarize_factor_exposures,
    FactorExposureConfig,
)


# ---------------------------------------------------------------------------
# compute_robust_weights (Step 4.93)
# ---------------------------------------------------------------------------

def _make_mu_cov(n: int = 5, seed: int = 0):
    rng = np.random.default_rng(seed)
    mu = pd.Series(rng.normal(0.0005, 0.001, n), index=[f"S{i}" for i in range(n)])
    raw = rng.standard_normal((100, n))
    cov = pd.DataFrame(np.cov(raw.T), index=mu.index, columns=mu.index)
    return mu, cov


def test_robust_weights_returns_result():
    mu, cov = _make_mu_cov()
    result = compute_robust_weights(mu, cov)
    assert isinstance(result, RobustOptResult)


def test_robust_weights_sum_to_one():
    mu, cov = _make_mu_cov()
    result = compute_robust_weights(mu, cov, long_only=True)
    total = sum(result.weights.values())
    assert abs(total - 1.0) < 1e-4


def test_robust_weights_long_only_non_negative():
    mu, cov = _make_mu_cov()
    result = compute_robust_weights(mu, cov, long_only=True)
    for w in result.weights.values():
        assert w >= -1e-6


def test_robust_weights_has_symbols():
    mu, cov = _make_mu_cov(n=4)
    syms = [f"S{i}" for i in range(4)]
    result = compute_robust_weights(mu, cov, symbols=syms)
    assert set(result.weights.keys()) == set(syms)


def test_robust_weights_method_str():
    mu, cov = _make_mu_cov()
    result = compute_robust_weights(mu, cov)
    assert isinstance(result.method, str)
    assert len(result.method) > 0


def test_robust_weights_max_weight_respected():
    mu, cov = _make_mu_cov(n=6)
    result = compute_robust_weights(mu, cov, max_weight=0.30, long_only=True)
    for w in result.weights.values():
        assert w <= 0.30 + 1e-4


def test_robust_weights_portfolio_vol_positive():
    mu, cov = _make_mu_cov()
    result = compute_robust_weights(mu, cov)
    assert result.portfolio_volatility >= 0.0


# ---------------------------------------------------------------------------
# deflated_sharpe (Step 8.5)
# ---------------------------------------------------------------------------

def _make_returns(n: int = 100, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n))


def test_dsr_returns_result():
    returns = _make_returns()
    result = deflated_sharpe(returns)
    assert isinstance(result, DSRResult)


def test_dsr_sharpe_observed_is_float():
    returns = _make_returns()
    result = deflated_sharpe(returns)
    assert isinstance(result.sharpe_observed, float)


def test_dsr_probability_in_01():
    returns = _make_returns()
    result = deflated_sharpe(returns)
    p = result.deflated_sharpe_probability
    assert 0.0 <= p <= 1.0 + 1e-9


def test_dsr_n_obs_matches_input():
    returns = _make_returns(n=80)
    result = deflated_sharpe(returns)
    assert result.n_observations == 80


def test_dsr_n_trials_recorded():
    returns = _make_returns()
    result = deflated_sharpe(returns, n_trials=10)
    assert result.n_trials == 10


def test_dsr_passes_5pct_is_bool():
    returns = _make_returns()
    result = deflated_sharpe(returns)
    assert isinstance(result.passes_5pct, bool)


def test_dsr_strong_returns_high_sr():
    # Strong positive drift with small noise → high SR
    rng = np.random.default_rng(42)
    returns = pd.Series(0.005 + rng.normal(0, 0.001, 120))
    result = deflated_sharpe(returns)
    assert result.sharpe_observed > 2.0


def test_dsr_as_dict_has_keys():
    returns = _make_returns()
    result = deflated_sharpe(returns)
    d = result.as_dict()
    for key in ["sharpe_observed", "sharpe_threshold", "deflated_sharpe_probability", "passes_5pct"]:
        assert key in d


# ---------------------------------------------------------------------------
# compute_factor_exposures + summarize_factor_exposures (Step 8.6)
# ---------------------------------------------------------------------------

pytest.importorskip("sklearn", reason="sklearn required for factor_exposures")


def _make_strategy_and_factors(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    market = pd.Series(rng.normal(0.0005, 0.01, n), index=idx, name="strategy")
    strategy = market + rng.normal(0.0002, 0.003, n)
    strategy.name = "strategy"
    factors = pd.DataFrame({"market": market.values}, index=idx)
    return strategy, factors


def test_factor_exposures_returns_df():
    strategy, factors = _make_strategy_and_factors()
    result = compute_factor_exposures(strategy, factors)
    assert isinstance(result, pd.DataFrame)


def test_factor_exposures_has_beta_market():
    strategy, factors = _make_strategy_and_factors()
    result = compute_factor_exposures(strategy, factors)
    assert "beta_market" in result.columns


def test_factor_exposures_has_r2():
    strategy, factors = _make_strategy_and_factors()
    result = compute_factor_exposures(strategy, factors)
    assert "r2" in result.columns


def test_factor_exposures_r2_in_range():
    strategy, factors = _make_strategy_and_factors()
    result = compute_factor_exposures(strategy, factors)
    r2_valid = result["r2"].dropna()
    assert (r2_valid >= 0.0).all() and (r2_valid <= 1.0 + 1e-9).all()


def test_summarize_factor_exposures_returns_df():
    strategy, factors = _make_strategy_and_factors()
    exposures = compute_factor_exposures(strategy, factors)
    summary = summarize_factor_exposures(exposures)
    assert isinstance(summary, pd.DataFrame)


def test_summarize_has_factor_column():
    strategy, factors = _make_strategy_and_factors()
    exposures = compute_factor_exposures(strategy, factors)
    summary = summarize_factor_exposures(exposures)
    if not summary.empty:
        assert "factor" in summary.columns


def test_summarize_market_beta_positive_for_correlated():
    # standardize_factors=True scales betas to unit-variance factor; just check sign
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    market = pd.Series(rng.normal(0.0005, 0.01, 80), index=idx)
    strategy = market * 1.2 + rng.normal(0, 0.001, 80)
    strategy.name = "strategy"
    factors = pd.DataFrame({"market": market.values}, index=idx)
    exposures = compute_factor_exposures(strategy, factors)
    summary = summarize_factor_exposures(exposures)
    if not summary.empty and "factor" in summary.columns:
        mkt_row = summary[summary["factor"] == "market"]
        if len(mkt_row) > 0:
            assert float(mkt_row["mean_beta"].iloc[0]) > 0.0
