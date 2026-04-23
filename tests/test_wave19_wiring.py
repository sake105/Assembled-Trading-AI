"""Tests for wave-19 module wiring into trading_cycle.py.

Covers:
  Step 4.87 — portfolio.stress_test_constraints (evaluate_stress_scenarios)
  Step 8.3  — qa.drawdown_decomposition (decompose_drawdown)
  Step 8.4  — qa.benchmark_metrics (compute_benchmark_metrics)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.portfolio.stress_test_constraints import (
    evaluate_stress_scenarios,
    StressTestResult,
    build_scenario_return_matrix,
)
from src.assembled_core.qa.drawdown_decomposition import (
    decompose_drawdown,
    find_worst_drawdown,
    DrawdownDecompositionReport,
)
from src.assembled_core.qa.benchmark_metrics import (
    compute_benchmark_metrics,
    BenchmarkMetrics,
)


# ---------------------------------------------------------------------------
# evaluate_stress_scenarios (Step 4.87)
# ---------------------------------------------------------------------------

def _make_equal_weights(symbols: list[str]) -> dict[str, float]:
    w = 1.0 / len(symbols)
    return {s: w for s in symbols}


def test_stress_returns_dataclass():
    syms = ["AAPL", "MSFT", "XOM"]
    weights = _make_equal_weights(syms)
    sector_map = {"AAPL": "Technology", "MSFT": "Technology", "XOM": "Energy"}
    result = evaluate_stress_scenarios(weights, syms, sector_map)
    assert isinstance(result, StressTestResult)


def test_stress_has_scenario_losses():
    syms = ["A", "B", "C"]
    weights = _make_equal_weights(syms)
    sector_map = {s: "Unknown" for s in syms}
    result = evaluate_stress_scenarios(weights, syms, sector_map)
    assert len(result.scenario_losses) > 0


def test_stress_worst_loss_is_most_negative():
    syms = ["A", "B"]
    weights = _make_equal_weights(syms)
    sector_map = {s: "Unknown" for s in syms}
    result = evaluate_stress_scenarios(weights, syms, sector_map)
    assert result.worst_loss == min(result.scenario_losses.values())


def test_stress_covid_crash_present():
    syms = ["A", "B"]
    weights = _make_equal_weights(syms)
    sector_map = {s: "Unknown" for s in syms}
    result = evaluate_stress_scenarios(weights, syms, sector_map)
    assert "covid_crash" in result.scenario_losses


def test_stress_all_within_floors_bool():
    syms = ["A", "B"]
    weights = _make_equal_weights(syms)
    sector_map = {s: "Unknown" for s in syms}
    result = evaluate_stress_scenarios(weights, syms, sector_map)
    assert isinstance(result.all_within_floors, bool)


def test_stress_violated_scenarios_is_list():
    syms = ["A", "B"]
    weights = _make_equal_weights(syms)
    sector_map = {s: "Unknown" for s in syms}
    result = evaluate_stress_scenarios(weights, syms, sector_map)
    assert isinstance(result.violated_scenarios, list)


def test_stress_energy_heavy_covid_worse():
    syms = ["XOM", "CVX", "SLB"]
    weights = _make_equal_weights(syms)
    sector_map = {s: "Energy" for s in syms}
    result = evaluate_stress_scenarios(weights, syms, sector_map)
    # Energy sectors get hit hard in covid (-0.45) vs default
    assert result.scenario_losses["covid_crash"] < -0.30


def test_stress_array_weights():
    syms = ["A", "B", "C"]
    weights = np.array([0.5, 0.3, 0.2])
    sector_map = {s: "Unknown" for s in syms}
    result = evaluate_stress_scenarios(weights, syms, sector_map)
    assert isinstance(result, StressTestResult)


# ---------------------------------------------------------------------------
# decompose_drawdown (Step 8.3)
# ---------------------------------------------------------------------------

def _make_portfolio_and_factors(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    # Portfolio returns with a drawdown in the middle
    returns = list(rng.normal(0.001, 0.01, 20)) + [-0.02] * 15 + list(rng.normal(0.0005, 0.01, n - 35))
    port = pd.Series(returns[:n])
    market = pd.DataFrame({"market": list(rng.normal(0.0005, 0.01, n))[:n]})
    return port, market


def test_decompose_returns_report():
    port, factors = _make_portfolio_and_factors()
    report = decompose_drawdown(port, factors)
    assert isinstance(report, DrawdownDecompositionReport)


def test_decompose_has_drawdown():
    port, factors = _make_portfolio_and_factors()
    report = decompose_drawdown(port, factors)
    assert report.drawdown.max_drawdown <= 0.0


def test_decompose_drawdown_duration_positive():
    port, factors = _make_portfolio_and_factors()
    report = decompose_drawdown(port, factors)
    assert report.drawdown.duration >= 0


def test_decompose_alpha_is_float():
    port, factors = _make_portfolio_and_factors()
    report = decompose_drawdown(port, factors)
    assert isinstance(report.alpha_during_dd, float)


def test_decompose_r_squared_in_range():
    port, factors = _make_portfolio_and_factors()
    report = decompose_drawdown(port, factors)
    assert -0.1 <= report.r_squared <= 1.0 + 1e-9


def test_decompose_short_series_returns_report():
    # Very short series — should return partial report without crashing
    port = pd.Series([-0.01, -0.02, -0.01, 0.005, -0.005])
    factors = pd.DataFrame({"market": [0.0] * 5})
    report = decompose_drawdown(port, factors)
    assert isinstance(report, DrawdownDecompositionReport)


def test_find_worst_drawdown_returns_drawdown():
    rng = np.random.default_rng(1)
    returns = pd.Series(rng.normal(0.001, 0.01, 80))
    dd = find_worst_drawdown(returns)
    assert dd.max_drawdown <= 0.0
    assert dd.duration >= 0


# ---------------------------------------------------------------------------
# compute_benchmark_metrics (Step 8.4)
# ---------------------------------------------------------------------------

def _make_port_bench(n: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    bench = pd.Series(rng.normal(0.0003, 0.008, n))
    port = bench + rng.normal(0.0001, 0.003, n)
    return port, bench


def test_benchmark_metrics_returns_dataclass():
    port, bench = _make_port_bench()
    result = compute_benchmark_metrics(port, bench)
    assert isinstance(result, BenchmarkMetrics)


def test_benchmark_metrics_beta_defined():
    port, bench = _make_port_bench()
    result = compute_benchmark_metrics(port, bench)
    assert result.beta is not None


def test_benchmark_metrics_beta_positive_correlated():
    rng = np.random.default_rng(5)
    bench = pd.Series(rng.normal(0.0005, 0.01, 80))
    port = bench * 1.2 + rng.normal(0, 0.001, 80)
    result = compute_benchmark_metrics(port, bench)
    assert result.beta is not None and result.beta > 0.5


def test_benchmark_metrics_tracking_error_positive():
    port, bench = _make_port_bench()
    result = compute_benchmark_metrics(port, bench)
    if result.tracking_error is not None:
        assert result.tracking_error >= 0.0


def test_benchmark_metrics_too_short_returns_none_fields():
    port = pd.Series([0.01, -0.01, 0.005, -0.003, 0.002])
    bench = pd.Series([0.008, -0.009, 0.004, -0.002, 0.001])
    result = compute_benchmark_metrics(port, bench)
    assert result.alpha is None


def test_benchmark_metrics_identical_series_zero_te():
    rng = np.random.default_rng(3)
    bench = pd.Series(rng.normal(0.0003, 0.01, 80))
    result = compute_benchmark_metrics(bench, bench)
    if result.tracking_error is not None:
        assert result.tracking_error < 1e-6
