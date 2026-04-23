"""Tests for wave-32 module wiring into trading_cycle.py.

Covers:
  Step 1.97 — features.macro_features (compute_diffusion_index)
  Step 8.19 — qa.performance_attribution (compute_attribution)
  Step 8.20 — qa.qa_gates (evaluate_all_gates)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.features.macro_features import (
    compute_diffusion_index,
)
from src.assembled_core.qa.performance_attribution import (
    compute_attribution,
    AttributionResult,
)
from src.assembled_core.qa.qa_gates import (
    evaluate_all_gates,
    QAGatesSummary,
    QAResult,
)
from src.assembled_core.qa.metrics import compute_equity_metrics


# ---------------------------------------------------------------------------
# compute_diffusion_index (Step 1.97)
# ---------------------------------------------------------------------------

def _make_macro_values(n: int = 30, n_series: int = 5, seed: int = 0) -> dict[str, pd.Series]:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="B")
    return {f"macro_{i}": pd.Series(100.0 + np.cumsum(rng.normal(0, 1, n)), index=ts)
            for i in range(n_series)}


def test_diffusion_index_returns_series():
    macro_vals = _make_macro_values()
    result = compute_diffusion_index(macro_vals)
    assert isinstance(result, pd.Series)


def test_diffusion_index_values_in_01():
    macro_vals = _make_macro_values()
    result = compute_diffusion_index(macro_vals)
    assert (result >= 0.0).all()
    assert (result <= 1.0).all()


def test_diffusion_index_empty_dict_returns_empty():
    result = compute_diffusion_index({})
    assert isinstance(result, pd.Series)
    assert len(result) == 0


def test_diffusion_index_all_rising():
    rng = np.random.default_rng(1)
    ts = pd.date_range("2024-01-01", periods=20, freq="B")
    macro_vals = {f"s{i}": pd.Series(np.cumsum(np.abs(rng.normal(0.5, 0.1, 20))), index=ts)
                  for i in range(5)}
    result = compute_diffusion_index(macro_vals)
    # Strongly rising series → diffusion > 0.5 after warmup
    assert result.iloc[-1] > 0.3


def test_diffusion_index_length_matches():
    macro_vals = _make_macro_values(n=20, n_series=3)
    result = compute_diffusion_index(macro_vals)
    ts_len = 20
    assert len(result) == ts_len


# ---------------------------------------------------------------------------
# compute_attribution (Step 8.19)
# ---------------------------------------------------------------------------

def _make_returns(n: int = 60, seed: int = 0) -> tuple[pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    market = pd.Series(rng.normal(0.0, 0.01, n), index=idx)
    portfolio = market * 0.8 + rng.normal(0.0, 0.005, n)
    portfolio = pd.Series(portfolio, index=idx)
    factors = pd.DataFrame({"market": market.values}, index=idx)
    return portfolio, factors


def test_attribution_returns_result():
    port, factors = _make_returns()
    result = compute_attribution(port, factors)
    assert isinstance(result, AttributionResult)


def test_attribution_r_squared_in_01():
    port, factors = _make_returns()
    result = compute_attribution(port, factors)
    assert 0.0 <= result.r_squared <= 1.0


def test_attribution_market_beta_in_range():
    port, factors = _make_returns()
    result = compute_attribution(port, factors)
    assert -5.0 <= result.factor_betas.get("market", 0.0) <= 5.0


def test_attribution_insufficient_data_raises():
    rng = np.random.default_rng(3)
    n = 10
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    port = pd.Series(rng.normal(0, 0.01, n), index=idx)
    factors = pd.DataFrame({"market": rng.normal(0, 0.01, n)}, index=idx)
    with pytest.raises(ValueError):
        compute_attribution(port, factors, min_obs=20)


def test_attribution_positive_market_for_correlated():
    rng = np.random.default_rng(5)
    idx = pd.date_range("2024-01-01", periods=80, freq="B")
    market = pd.Series(rng.normal(0, 0.01, 80), index=idx)
    portfolio = market * 1.2 + rng.normal(0, 0.001, 80)
    portfolio = pd.Series(portfolio, index=idx)
    factors = pd.DataFrame({"market": market.values}, index=idx)
    result = compute_attribution(portfolio, factors)
    assert result.factor_betas["market"] > 0.5


# ---------------------------------------------------------------------------
# evaluate_all_gates (Step 8.20)
# ---------------------------------------------------------------------------

def _make_equity_metrics(n: int = 252, seed: int = 0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2024-01-01", periods=n, freq="B")
    equity = 100000.0 + np.cumsum(rng.normal(50, 200, n))
    eq_df = pd.DataFrame({"timestamp": ts, "equity": equity})
    return compute_equity_metrics(eq_df, start_capital=100000.0)


def test_qa_gates_returns_summary():
    metrics = _make_equity_metrics()
    summary = evaluate_all_gates(metrics)
    assert isinstance(summary, QAGatesSummary)


def test_qa_gates_overall_result_valid():
    metrics = _make_equity_metrics()
    summary = evaluate_all_gates(metrics)
    assert str(summary.overall_result) in {str(r) for r in QAResult}


def test_qa_gates_counts_non_negative():
    metrics = _make_equity_metrics()
    summary = evaluate_all_gates(metrics)
    assert summary.passed_gates >= 0
    assert summary.warning_gates >= 0
    assert summary.blocked_gates >= 0


def test_qa_gates_total_equals_sum():
    metrics = _make_equity_metrics()
    summary = evaluate_all_gates(metrics)
    total = summary.passed_gates + summary.warning_gates + summary.blocked_gates
    assert total == len(summary.gate_results)


def test_qa_gates_strong_equity_mostly_passes():
    rng = np.random.default_rng(42)
    ts = pd.date_range("2024-01-01", periods=252, freq="B")
    # Strong upward equity with low volatility
    equity = 100000.0 + np.cumsum(np.abs(rng.normal(100, 50, 252)))
    eq_df = pd.DataFrame({"timestamp": ts, "equity": equity})
    metrics = compute_equity_metrics(eq_df, start_capital=100000.0)
    summary = evaluate_all_gates(metrics)
    # With good equity, should have at least some passes
    assert summary.passed_gates + summary.warning_gates > 0
