"""Tests for wave-21 module wiring into trading_cycle.py.

Covers:
  Step 4.88 — portfolio.regime_portfolio (blend_regime_templates)
  Step 4.94 — qa.reverse_stress (reverse_stress_test)
  Step 8.7  — qa.scenario_simulator (run_stress_test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.portfolio.regime_portfolio import (
    blend_regime_templates,
    REGIME_TEMPLATES,
)
from src.assembled_core.qa.reverse_stress import (
    reverse_stress_test,
    ReverseStressResult,
)
from src.assembled_core.qa.scenario_simulator import (
    run_stress_test,
    simulate_vol_spike_scenario,
    simulate_crash_scenario,
    StressTestReport,
    ScenarioResult,
)


# ---------------------------------------------------------------------------
# blend_regime_templates (Step 4.88)
# ---------------------------------------------------------------------------

def test_blend_regime_templates_returns_dict():
    probs = {"bull": 1.0}
    result = blend_regime_templates(probs)
    assert isinstance(result, dict)


def test_blend_regime_templates_sums_to_one():
    probs = {"bull": 0.6, "sideways": 0.4}
    result = blend_regime_templates(probs)
    total = sum(result.values())
    assert abs(total - 1.0) < 1e-4


def test_blend_regime_templates_non_negative():
    probs = {"bear": 0.5, "crisis": 0.5}
    result = blend_regime_templates(probs)
    for w in result.values():
        assert w >= 0.0


def test_blend_regime_templates_all_regimes():
    for regime in REGIME_TEMPLATES:
        result = blend_regime_templates({regime: 1.0})
        assert len(result) > 0


def test_blend_regime_templates_zero_prob_fallback():
    result = blend_regime_templates({})
    assert isinstance(result, dict)


def test_blend_regime_templates_crisis_has_bonds():
    result = blend_regime_templates({"crisis": 1.0})
    assert "bonds_treasury" in result
    assert result["bonds_treasury"] > 0.2


def test_blend_regime_templates_bull_has_equity():
    result = blend_regime_templates({"bull": 1.0})
    equity_keys = [k for k in result if "equity" in k]
    assert len(equity_keys) > 0


# ---------------------------------------------------------------------------
# reverse_stress_test (Step 4.94)
# ---------------------------------------------------------------------------

pytest.importorskip("scipy", reason="scipy required for reverse_stress_test")


def _make_weights_cov(n: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    w = np.ones(n) / n
    raw = rng.standard_normal((60, n))
    cov = np.cov(raw.T)
    return w, cov


def test_reverse_stress_returns_result():
    w, cov = _make_weights_cov()
    result = reverse_stress_test(w, cov, target_loss=-0.20)
    assert isinstance(result, ReverseStressResult)


def test_reverse_stress_has_shock_vector():
    w, cov = _make_weights_cov()
    result = reverse_stress_test(w, cov, target_loss=-0.20)
    assert len(result.shock_vector) == len(w)


def test_reverse_stress_target_loss_recorded():
    w, cov = _make_weights_cov()
    result = reverse_stress_test(w, cov, target_loss=-0.15)
    assert result.target_loss == -0.15


def test_reverse_stress_shock_norm_non_negative():
    w, cov = _make_weights_cov()
    result = reverse_stress_test(w, cov, target_loss=-0.20)
    assert result.shock_norm >= 0.0


def test_reverse_stress_converged_is_bool():
    w, cov = _make_weights_cov()
    result = reverse_stress_test(w, cov, target_loss=-0.20)
    assert isinstance(result.converged, bool)


def test_reverse_stress_plausibility_positive():
    w, cov = _make_weights_cov()
    result = reverse_stress_test(w, cov, target_loss=-0.20)
    assert result.plausibility_score >= 0.0


# ---------------------------------------------------------------------------
# run_stress_test / scenario_simulator (Step 8.7)
# ---------------------------------------------------------------------------

def _make_baseline(n: int = 80, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0005, 0.01, n))


def test_run_stress_test_returns_report():
    baseline = _make_baseline()
    result = run_stress_test(baseline, include_correlation=False)
    assert isinstance(result, StressTestReport)


def test_run_stress_test_has_scenarios():
    baseline = _make_baseline()
    result = run_stress_test(baseline, include_correlation=False)
    assert len(result.scenarios) >= 1


def test_run_stress_test_worst_scenario_str():
    baseline = _make_baseline()
    result = run_stress_test(baseline, include_correlation=False)
    assert isinstance(result.worst_scenario, str)


def test_run_stress_test_worst_cvar_negative():
    baseline = _make_baseline()
    result = run_stress_test(baseline, include_correlation=False)
    assert result.worst_cvar < 0.0


def test_simulate_vol_spike_returns_result():
    baseline = _make_baseline()
    result = simulate_vol_spike_scenario(baseline, vol_multiplier=3.0)
    assert isinstance(result, ScenarioResult)
    assert result.scenario_name == "VolSpike"


def test_simulate_crash_has_negative_mean():
    baseline = _make_baseline()
    result = simulate_crash_scenario(baseline, crash_magnitude=-0.10)
    assert result.mean_return < 0.0


def test_run_stress_with_portfolio_returns():
    rng = np.random.default_rng(1)
    baseline = pd.Series(rng.normal(0.0005, 0.01, 80))
    port_rets = pd.DataFrame(rng.normal(0.0005, 0.01, (80, 3)), columns=["A", "B", "C"])
    result = run_stress_test(baseline, portfolio_returns=port_rets, include_correlation=True)
    assert isinstance(result, StressTestReport)
    assert len(result.scenarios) >= 2
