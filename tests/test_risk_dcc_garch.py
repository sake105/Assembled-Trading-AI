"""Tests for DCC-GARCH (Engle 2002) + cDCC (Aielli 2013) — C4-072 closure."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("arch")
pytest.importorskip("scipy")

from src.assembled_core.risk.dcc_garch import (
    DCCResult,
    current_covariance,
    fit_dcc_garch,
)


def _correlated_garch_returns(
    n_periods: int = 500,
    n_vars: int = 3,
    rho_target: float = 0.6,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic correlated GARCH(1,1) returns."""
    rng = np.random.default_rng(seed)
    # Common factor + idiosyncratic noise
    common = rng.standard_normal(n_periods)
    sigma2 = np.zeros((n_periods, n_vars))
    eps = np.zeros((n_periods, n_vars))
    omega, alpha, beta = 0.0001, 0.05, 0.90
    for j in range(n_vars):
        sigma2[0, j] = omega / (1 - alpha - beta)
        for t in range(1, n_periods):
            shock = (
                np.sqrt(rho_target) * common[t]
                + np.sqrt(1 - rho_target) * rng.standard_normal()
            )
            eps[t - 1, j] = np.sqrt(sigma2[t - 1, j]) * shock
            sigma2[t, j] = omega + alpha * eps[t - 1, j] ** 2 + beta * sigma2[t - 1, j]
        eps[-1, j] = np.sqrt(sigma2[-1, j]) * rng.standard_normal()
    return pd.DataFrame(eps, columns=[f"asset_{i}" for i in range(n_vars)])


def _independent_garch_returns(
    n_periods: int = 500, n_vars: int = 3, seed: int = 0
) -> pd.DataFrame:
    """Independent GARCH(1,1) series (low cross-correlation)."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.0001, 0.05, 0.90
    eps = np.zeros((n_periods, n_vars))
    sigma2 = np.zeros((n_periods, n_vars))
    for j in range(n_vars):
        sigma2[0, j] = omega / (1 - alpha - beta)
        for t in range(1, n_periods):
            eps[t - 1, j] = np.sqrt(sigma2[t - 1, j]) * rng.standard_normal()
            sigma2[t, j] = omega + alpha * eps[t - 1, j] ** 2 + beta * sigma2[t - 1, j]
        eps[-1, j] = np.sqrt(sigma2[-1, j]) * rng.standard_normal()
    return pd.DataFrame(eps, columns=[f"asset_{i}" for i in range(n_vars)])


# ---------------------------------------------------------------------------
# fit_dcc_garch — standard DCC
# ---------------------------------------------------------------------------


def test_returns_dcc_result_with_all_fields():
    df = _correlated_garch_returns(n_periods=300, n_vars=3)
    result = fit_dcc_garch(df, method="dcc")
    assert isinstance(result, DCCResult)
    assert 0 < result.a < 1
    assert 0 < result.b < 1
    assert result.a + result.b < 1.0
    assert result.q_bar.shape == (3, 3)
    assert result.conditional_volatilities.shape == (300, 3)
    assert len(result.conditional_correlations) == 300
    assert len(result.conditional_covariance) == 300
    assert result.standardized_residuals.shape == (300, 3)
    assert result.n_obs == 300
    assert result.n_vars == 3
    assert result.method == "dcc"


def test_dcc_recovers_positive_correlation():
    """DCC on synthetic correlated data → R_bar off-diagonals positive."""
    df = _correlated_garch_returns(n_periods=500, n_vars=3, rho_target=0.6, seed=1)
    result = fit_dcc_garch(df, method="dcc")
    # Off-diagonal of q_bar should be substantially positive
    off_diag = result.q_bar - np.diag(np.diag(result.q_bar))
    assert off_diag.sum() > 0
    # Mean off-diagonal correlation should be > 0.2
    mean_off = off_diag.sum() / (3 * 2)
    assert mean_off > 0.2, f"Expected positive mean off-diag, got {mean_off:.3f}"


def test_dcc_diagonal_of_correlation_is_one():
    """R_t diagonal must always be 1.0 by construction."""
    df = _correlated_garch_returns(n_periods=300, n_vars=3)
    result = fit_dcc_garch(df, method="dcc")
    for r_t in result.conditional_correlations[:5]:
        np.testing.assert_allclose(np.diag(r_t), 1.0, atol=1e-6)


def test_dcc_covariance_is_symmetric_positive():
    """H_t must be symmetric and positive-diagonal."""
    df = _correlated_garch_returns(n_periods=300, n_vars=3)
    result = fit_dcc_garch(df, method="dcc")
    h_last = result.conditional_covariance[-1]
    np.testing.assert_allclose(h_last, h_last.T, atol=1e-9)
    assert (np.diag(h_last) > 0).all()


def test_dcc_stationarity_constraint():
    """α + β < 1 must always hold (optimiser bounds + post-check)."""
    df = _correlated_garch_returns(n_periods=400, n_vars=3, seed=7)
    result = fit_dcc_garch(df, method="dcc")
    assert result.a + result.b < 1.0


# ---------------------------------------------------------------------------
# fit_dcc_garch — cDCC (Aielli 2013)
# ---------------------------------------------------------------------------


def test_cdcc_returns_dcc_result_with_method_flag():
    df = _correlated_garch_returns(n_periods=300, n_vars=3)
    result = fit_dcc_garch(df, method="cdcc")
    assert result.method == "cdcc"
    assert isinstance(result, DCCResult)


def test_cdcc_recovers_positive_correlation():
    df = _correlated_garch_returns(n_periods=500, n_vars=3, rho_target=0.6, seed=2)
    result = fit_dcc_garch(df, method="cdcc")
    off_diag = result.q_bar - np.diag(np.diag(result.q_bar))
    mean_off = off_diag.sum() / (3 * 2)
    assert mean_off > 0.2


def test_dcc_vs_cdcc_q_bar_differs():
    """cDCC's Q̄ should differ from standard DCC's (Aielli correction applied)."""
    df = _correlated_garch_returns(n_periods=500, n_vars=3, seed=3)
    r_dcc = fit_dcc_garch(df, method="dcc")
    r_cdcc = fit_dcc_garch(df, method="cdcc")
    # The off-diagonals should be similar in direction but not identical
    assert not np.allclose(r_dcc.q_bar, r_cdcc.q_bar, atol=1e-6), (
        "cDCC correction should produce different Q̄ than standard DCC"
    )


# ---------------------------------------------------------------------------
# current_covariance
# ---------------------------------------------------------------------------


def test_current_covariance_returns_dataframe():
    df = _correlated_garch_returns(n_periods=300, n_vars=3)
    result = fit_dcc_garch(df)
    h_now = current_covariance(result)
    assert isinstance(h_now, pd.DataFrame)
    assert h_now.shape == (3, 3)
    assert list(h_now.columns) == result.column_names
    assert list(h_now.index) == result.column_names


def test_current_covariance_symmetric():
    df = _correlated_garch_returns(n_periods=300, n_vars=3)
    result = fit_dcc_garch(df)
    h_now = current_covariance(result)
    np.testing.assert_allclose(h_now.to_numpy(), h_now.to_numpy().T, atol=1e-9)


# ---------------------------------------------------------------------------
# Input validation / edge cases
# ---------------------------------------------------------------------------


def test_rejects_single_variable():
    df = pd.DataFrame({"only_one": np.random.randn(200)})
    with pytest.raises(ValueError, match="≥2 variables"):
        fit_dcc_garch(df)


def test_rejects_short_series():
    df = _correlated_garch_returns(n_periods=50, n_vars=2)
    with pytest.raises(ValueError, match="100"):
        fit_dcc_garch(df)


def test_rejects_unknown_method():
    """F-stage1-dcc-7 regression: typo like 'dcc-garch' must raise, not silently
    fall through to standard DCC."""
    df = _correlated_garch_returns(n_periods=200, n_vars=2)
    with pytest.raises(ValueError, match="method must be 'dcc' or 'cdcc'"):
        fit_dcc_garch(df, method="dcc-garch")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="method must be 'dcc' or 'cdcc'"):
        fit_dcc_garch(df, method="DCC")  # type: ignore[arg-type]


def test_stationarity_constraint_enforced_via_slsqp():
    """F-stage1-dcc-3 + F-senior-2 regression: SLSQP with constraint should
    return params satisfying stationarity AND converge cleanly (no snap) for
    well-behaved synthetic data.

    Asserts BOTH stationarity AND `converged is True` — a regression where
    SLSQP always falls back to snap (defeating the fix) would NOT silently
    pass this test.
    """
    df = _correlated_garch_returns(n_periods=400, n_vars=3, rho_target=0.7, seed=42)
    result = fit_dcc_garch(df, method="dcc")
    # Stationarity: a + b < 1 must always hold
    assert result.a + result.b < 1.0
    # F-senior-2: also assert converged=True — snap branch sets it False
    # and recomputes ll. If this ever flips to False, SLSQP failed and we
    # need to investigate (the test will surface the regression).
    assert result.converged is True, (
        f"SLSQP expected to converge cleanly on this synthetic seed; "
        f"converged={result.converged} suggests fall-back to snap-branch — "
        f"investigate F-stage1-dcc-3 regression. a={result.a}, b={result.b}"
    )


# ---------------------------------------------------------------------------
# Integration with portfolio.covariance.estimate_covariance
# ---------------------------------------------------------------------------


def test_estimate_covariance_dcc_garch_routes_to_real_module():
    """estimate_covariance(method='dcc_garch') must now route to fit_dcc_garch,
    not silently fall through to sample covariance (C4-072 closure)."""
    from src.assembled_core.portfolio.covariance import estimate_covariance

    df = _correlated_garch_returns(n_periods=300, n_vars=3, rho_target=0.6, seed=5)
    cov_dcc = estimate_covariance(df, method="dcc_garch")
    cov_sample = estimate_covariance(df, method="sample")

    assert isinstance(cov_dcc, pd.DataFrame)
    assert cov_dcc.shape == (3, 3)
    # DCC covariance should NOT equal sample covariance (proves the route)
    assert not np.allclose(cov_dcc.to_numpy(), cov_sample.to_numpy(), atol=1e-9), (
        "DCC-GARCH should produce different cov than sample (silent-stub regression test)"
    )


def test_estimate_covariance_cdcc_routes_to_real_module():
    from src.assembled_core.portfolio.covariance import estimate_covariance

    df = _correlated_garch_returns(n_periods=300, n_vars=3, rho_target=0.6, seed=6)
    cov_cdcc = estimate_covariance(df, method="cdcc")
    assert isinstance(cov_cdcc, pd.DataFrame)
    assert cov_cdcc.shape == (3, 3)
