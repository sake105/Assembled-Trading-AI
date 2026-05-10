"""Tests for erweiterung.risk."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.risk import (
    correlation_breakdown,
    dynamic_drawdown_control,
    tail_risk_evt,
)


def test_gpd_fit_basic():
    rng = np.random.default_rng(42)
    losses = rng.exponential(1.0, 1000)
    fit = tail_risk_evt.fit_gpd(losses, threshold_quantile=0.95)
    assert fit.n_excess > 30
    assert np.isfinite(fit.xi)
    assert fit.beta > 0


def test_var_evt_consistency():
    rng = np.random.default_rng(0)
    # Heavy-tail Student-t-like sample
    losses = np.abs(rng.standard_t(df=4, size=2000))
    fit = tail_risk_evt.fit_gpd(losses, threshold_quantile=0.90)
    var_99 = tail_risk_evt.var_evt(fit, len(losses), alpha=0.99)
    var_95 = tail_risk_evt.var_evt(fit, len(losses), alpha=0.95)
    assert var_99 > var_95


def test_estimate_tail_metrics():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 1000))
    out = tail_risk_evt.estimate_tail_metrics(returns, alpha=0.99)
    assert "var_hist" in out
    assert "cvar_evt" in out


def test_compute_drawdown():
    eq = pd.Series([100, 110, 120, 100, 90, 110])
    dd = dynamic_drawdown_control.compute_running_drawdown(eq)
    assert dd.iloc[-2] == pytest_approx(-0.25)


def pytest_approx(x, tol=1e-6):
    class _Approx:
        def __eq__(self, other):
            return abs(other - x) < tol

    return _Approx()


def test_cppi_floor_factor():
    # No drawdown -> factor = 1 (capped)
    assert dynamic_drawdown_control.cppi_floor_factor(0.0, max_dd=0.2) > 0
    # At max_dd -> factor 0
    assert dynamic_drawdown_control.cppi_floor_factor(-0.20, max_dd=0.2) == 0
    # Beyond max_dd -> 0
    assert dynamic_drawdown_control.cppi_floor_factor(-0.30, max_dd=0.2) == 0


def test_vol_targeted_leverage():
    # rv = target -> 1.0
    lev = dynamic_drawdown_control.vol_targeted_leverage(0.15, target_vol=0.15)
    assert abs(lev - 1.0) < 1e-9
    # rv = 0.30 (high) -> 0.5
    lev = dynamic_drawdown_control.vol_targeted_leverage(0.30, target_vol=0.15)
    assert abs(lev - 0.5) < 1e-9


def test_combined_dd_vol(synthetic_returns):
    eq = (1 + synthetic_returns.iloc[:, 0]).cumprod()
    out = dynamic_drawdown_control.combined_dd_vol_control(
        eq, synthetic_returns.iloc[:, 0], vol_window=30, target_vol=0.15
    )
    assert (out >= 0).all()


def test_apc_basic(synthetic_returns):
    apc = correlation_breakdown.average_pairwise_correlation(synthetic_returns)
    assert -1 <= apc <= 1


def test_rolling_apc(synthetic_returns):
    apc = correlation_breakdown.rolling_apc(synthetic_returns, window=60)
    assert (apc.dropna().between(-1, 1)).all()


def test_first_eigenvalue_share(synthetic_returns):
    s = correlation_breakdown.first_eigenvalue_share(synthetic_returns)
    assert 0 < s < 1


def test_crisis_score_bounded(synthetic_returns):
    out = correlation_breakdown.crisis_score(synthetic_returns, window=60)
    assert "crisis_score" in out.columns
    assert out["crisis_score"].dropna().between(0, 1).all()
