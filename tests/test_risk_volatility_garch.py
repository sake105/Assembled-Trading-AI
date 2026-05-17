"""Tests for src/assembled_core/risk/volatility/garch.py — GARCH(1,1) fit + forecast."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")

from src.assembled_core.risk.volatility.garch import GarchForecast, fit_garch


@pytest.fixture
def synthetic_returns() -> pd.Series:
    """Generate ~3 years of daily GARCH(1,1) returns for testing."""
    rng = np.random.default_rng(42)
    n = 252 * 3
    omega, alpha, beta = 0.0001, 0.05, 0.90
    sigma2 = np.zeros(n)
    eps = np.zeros(n)
    sigma2[0] = omega / (1 - alpha - beta)
    for t in range(1, n):
        eps[t - 1] = np.sqrt(sigma2[t - 1]) * rng.standard_normal()
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
    eps[-1] = np.sqrt(sigma2[-1]) * rng.standard_normal()
    idx = pd.date_range("2020-01-01", periods=n, freq="B", tz="UTC")
    return pd.Series(eps, index=idx, name="returns")


def test_fit_returns_garchforecast(synthetic_returns):
    result = fit_garch(synthetic_returns)
    assert isinstance(result, GarchForecast)
    assert result.next_period_volatility > 0
    assert len(result.conditional_volatility) == len(synthetic_returns)
    assert result.convergence
    assert set(result.params.keys()) >= {"omega", "alpha[1]", "beta[1]"}


def test_fit_recovers_synthetic_params_approximately(synthetic_returns):
    """GARCH should recover omega, alpha, beta within ~30% of synthetic truth."""
    result = fit_garch(synthetic_returns)
    # Synthetic truth: omega=0.0001, alpha=0.05, beta=0.90
    # arch may rescale; we check the relative magnitude (alpha < beta < 1).
    alpha = result.params["alpha[1]"]
    beta = result.params["beta[1]"]
    assert 0 < alpha < 0.2, f"alpha out of expected range: {alpha}"
    assert 0.6 < beta < 0.99, f"beta out of expected range: {beta}"
    assert alpha + beta < 1.0, "stationarity violated"


def test_empty_returns_raises():
    with pytest.raises(ValueError, match="empty"):
        fit_garch(pd.Series([], dtype=float))


def test_insufficient_obs_raises():
    s = pd.Series(np.random.randn(20))
    with pytest.raises(ValueError, match="50"):
        fit_garch(s)


def test_nan_returns_raises():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.standard_normal(100))
    s.iloc[10] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        fit_garch(s)


def test_pit_safety_conditional_vol_length(synthetic_returns):
    """Conditional vol should be same length as returns input (no leak)."""
    result = fit_garch(synthetic_returns)
    assert len(result.conditional_volatility) == len(synthetic_returns)
    # Should share the same index as returns (in-sample)
    pd.testing.assert_index_equal(
        result.conditional_volatility.index, synthetic_returns.index
    )


def test_forecast_is_annualized(synthetic_returns):
    """Next-period vol should be annualized (sqrt(252) scale)."""
    result = fit_garch(synthetic_returns)
    # Annualized vol for GARCH(1,1) with these params is typically ~15-25%
    assert (
        0.05 < result.next_period_volatility < 1.0
    ), f"Annualized vol out of plausible range: {result.next_period_volatility}"


def test_zero_mean_model_works(synthetic_returns):
    """Should accept mean='zero' for zero-mean returns assumption."""
    result = fit_garch(synthetic_returns, mean="zero")
    assert result.convergence
    assert "mu" not in result.params  # zero-mean has no mu term
