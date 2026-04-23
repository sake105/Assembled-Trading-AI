"""Tests for wave-17 module wiring into trading_cycle.py.

Covers:
  Step 4.86 — risk.regime_costs (estimate_regime_costs)
  Step 5.5  — qa.risk_metrics (compute_portfolio_risk_metrics)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.risk.regime_costs import (
    estimate_regime_costs,
    RegimeCostEstimate,
)
from src.assembled_core.qa.risk_metrics import compute_portfolio_risk_metrics


# ---------------------------------------------------------------------------
# estimate_regime_costs (Step 4.86)
# ---------------------------------------------------------------------------

def test_regime_costs_returns_dataclass():
    result = estimate_regime_costs(trade_value=100_000.0, adv=50_000_000.0)
    assert isinstance(result, RegimeCostEstimate)


def test_regime_costs_normal_has_positive_bps():
    result = estimate_regime_costs(100_000.0, 50_000_000.0, regime="normal")
    assert result.total_cost_bps > 0


def test_regime_costs_crisis_higher_than_normal():
    normal = estimate_regime_costs(100_000.0, 50_000_000.0, regime="normal")
    crisis = estimate_regime_costs(100_000.0, 50_000_000.0, regime="crisis")
    assert crisis.total_cost_bps > normal.total_cost_bps


def test_regime_costs_calm_lower_than_normal():
    normal = estimate_regime_costs(100_000.0, 50_000_000.0, regime="normal")
    calm = estimate_regime_costs(100_000.0, 50_000_000.0, regime="calm")
    assert calm.total_cost_bps <= normal.total_cost_bps


def test_regime_costs_fill_rate_in_range():
    result = estimate_regime_costs(100_000.0, 50_000_000.0)
    assert 0.0 <= result.expected_fill_rate <= 1.0


def test_regime_costs_urgency_increases_cost():
    low = estimate_regime_costs(100_000.0, 50_000_000.0, urgency=0.0)
    high = estimate_regime_costs(100_000.0, 50_000_000.0, urgency=1.0)
    assert high.total_cost_bps >= low.total_cost_bps


def test_regime_costs_vix_increases_cost():
    normal = estimate_regime_costs(100_000.0, 50_000_000.0, vix_level=20.0)
    stressed = estimate_regime_costs(100_000.0, 50_000_000.0, vix_level=40.0)
    assert stressed.total_cost_bps >= normal.total_cost_bps


def test_regime_costs_result_regime_matches():
    result = estimate_regime_costs(100_000.0, 50_000_000.0, regime="crisis")
    assert result.regime == "crisis"


# ---------------------------------------------------------------------------
# compute_portfolio_risk_metrics (Step 5.5)
# ---------------------------------------------------------------------------

def _make_equity(n: int = 60, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.01, n)
    return pd.Series(100.0 * np.cumprod(1 + returns))


def test_risk_metrics_returns_dict():
    equity = _make_equity()
    result = compute_portfolio_risk_metrics(equity)
    assert isinstance(result, dict)


def test_risk_metrics_has_required_keys():
    equity = _make_equity()
    result = compute_portfolio_risk_metrics(equity)
    for key in ["daily_vol", "ann_vol", "max_drawdown", "var_95", "es_95"]:
        assert key in result, f"Missing: {key}"


def test_risk_metrics_ann_vol_positive():
    equity = _make_equity(n=100)
    result = compute_portfolio_risk_metrics(equity)
    if result["ann_vol"] is not None:
        assert result["ann_vol"] > 0


def test_risk_metrics_max_drawdown_non_positive():
    equity = _make_equity(n=100)
    result = compute_portfolio_risk_metrics(equity)
    assert result["max_drawdown"] <= 0.0


def test_risk_metrics_empty_series():
    result = compute_portfolio_risk_metrics(pd.Series(dtype=float))
    assert result["daily_vol"] is None
    assert result["max_drawdown"] == 0.0


def test_risk_metrics_declining_equity_has_drawdown():
    # Steadily declining equity
    equity = pd.Series([100.0, 95.0, 90.0, 85.0, 80.0, 75.0])
    result = compute_portfolio_risk_metrics(equity)
    assert result["max_drawdown"] < 0


def test_risk_metrics_dataframe_input():
    df = pd.DataFrame({"equity": _make_equity(50).values})
    result = compute_portfolio_risk_metrics(df)
    assert isinstance(result, dict)
    assert "daily_vol" in result
