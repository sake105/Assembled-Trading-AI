"""Tests for C2-025 / C2-026 — PLR and Causal Forest.

Synthetic data is constructed with known causal parameters so that we can
assert recovery.

PLR DGP (Robinson 1988 form):
    D = 0.5 * X[:,0] + noise_D
    Y = 2.0 * D + X[:,0] + noise_Y
    True theta = 2.0

Causal Forest DGP:
    D = 0.5 * X[:,0] + noise_D   (binary-ish after rounding is NOT done here)
    Y = 2.5 * D + X[:,0] + noise_Y
    True ATE = 2.5
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# causal_ml requires scikit-learn; skip entire module if not installed.
# Convention: matches tests/regression/test_deflated_sharpe.py pattern.
pytest.importorskip("sklearn")

from src.assembled_core.signals.causal_ml import (
    CausalForestResult,
    PLRResult,
    fit_causal_forest,
    fit_plr,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 400
TRUE_THETA = 2.0
TRUE_ATE = 2.5
RNG_SEED = 42


@pytest.fixture(scope="module")
def plr_data():
    """Synthetic data with known PLR theta = 2.0."""
    rng = np.random.default_rng(RNG_SEED)
    p = 5
    X = rng.normal(0, 1, (N, p))
    noise_D = rng.normal(0, 0.5, N)
    noise_Y = rng.normal(0, 0.5, N)
    D = 0.5 * X[:, 0] + noise_D
    Y = TRUE_THETA * D + X[:, 0] + noise_Y
    return Y, D, X


@pytest.fixture(scope="module")
def cf_data():
    """Synthetic data with known ATE = 2.5."""
    rng = np.random.default_rng(RNG_SEED + 1)
    p = 5
    X = rng.normal(0, 1, (N, p))
    noise_D = rng.normal(0, 0.3, N)
    noise_Y = rng.normal(0, 0.5, N)
    D = 0.5 * X[:, 0] + noise_D
    Y = TRUE_ATE * D + X[:, 0] + noise_Y
    return Y, D, X


# ---------------------------------------------------------------------------
# PLR tests
# ---------------------------------------------------------------------------


def test_plr_theta_close_to_true(plr_data):
    """PLR theta should be within ±0.5 of the true value 2.0."""
    Y, D, X = plr_data
    result = fit_plr(Y, D, X, n_folds=5)
    assert abs(result.theta - TRUE_THETA) < 0.5, (
        f"theta={result.theta:.4f} too far from true={TRUE_THETA}"
    )


def test_plr_result_fields_present(plr_data):
    """PLRResult must expose all required fields."""
    Y, D, X = plr_data
    result = fit_plr(Y, D, X)
    assert isinstance(result, PLRResult)
    assert hasattr(result, "theta")
    assert hasattr(result, "se")
    assert hasattr(result, "t_stat")
    assert hasattr(result, "pvalue")
    assert hasattr(result, "n_obs")
    assert hasattr(result, "n_folds")
    assert hasattr(result, "method")


def test_plr_t_stat_reasonably_sized(plr_data):
    """For a strong signal, |t_stat| should be > 2 (roughly significant)."""
    Y, D, X = plr_data
    result = fit_plr(Y, D, X)
    if not (math.isnan(result.t_stat) or math.isinf(result.t_stat)):
        assert abs(result.t_stat) > 2.0, (
            f"|t_stat|={abs(result.t_stat):.2f} expected > 2 for strong signal"
        )


def test_plr_pvalue_significant(plr_data):
    """p-value should be < 0.05 for a strong synthetic treatment effect."""
    Y, D, X = plr_data
    result = fit_plr(Y, D, X)
    if not (math.isnan(result.pvalue) or math.isinf(result.pvalue)):
        assert result.pvalue < 0.05, f"pvalue={result.pvalue:.4f} expected < 0.05"


def test_plr_works_with_2_folds(plr_data):
    """fit_plr must run without error with n_folds=2."""
    Y, D, X = plr_data
    result = fit_plr(Y, D, X, n_folds=2)
    assert isinstance(result, PLRResult)
    assert not math.isnan(result.theta)


def test_plr_small_dataset():
    """PLR must handle n=20 without crashing."""
    rng = np.random.default_rng(99)
    n = 20
    X = rng.normal(0, 1, (n, 2))
    D = 0.5 * X[:, 0] + rng.normal(0, 0.3, n)
    Y = 2.0 * D + rng.normal(0, 0.5, n)
    result = fit_plr(Y, D, X, n_folds=2)
    assert isinstance(result, PLRResult)


def test_plr_n_obs_correct(plr_data):
    """PLRResult.n_obs must match the input length."""
    Y, D, X = plr_data
    result = fit_plr(Y, D, X)
    assert result.n_obs == len(Y)


def test_plr_n_folds_stored(plr_data):
    """PLRResult.n_folds must match the requested value."""
    Y, D, X = plr_data
    result = fit_plr(Y, D, X, n_folds=3)
    assert result.n_folds == 3


# ---------------------------------------------------------------------------
# Causal Forest tests
# ---------------------------------------------------------------------------


def test_cf_cate_array_length(cf_data):
    """CATEs must be an array of length n."""
    Y, D, X = cf_data
    result = fit_causal_forest(Y, D, X)
    assert isinstance(result.cate, np.ndarray)
    assert len(result.cate) == N


def test_cf_ate_close_to_true(cf_data):
    """ATE should be within ±1.5 of true value 2.5 (both econml and fallback)."""
    Y, D, X = cf_data
    result = fit_causal_forest(Y, D, X, random_state=RNG_SEED)
    if result.converged:
        assert abs(result.ate - TRUE_ATE) < 1.5, (
            f"ATE={result.ate:.4f} too far from true={TRUE_ATE}"
        )


def test_cf_converged_when_sklearn_available(cf_data):
    """converged should be True when at least sklearn is available."""
    from src.assembled_core.signals.causal_ml import HAS_SKLEARN

    Y, D, X = cf_data
    result = fit_causal_forest(Y, D, X)
    if HAS_SKLEARN:
        assert result.converged, "Expected converged=True with sklearn available"


def test_cf_result_fields_present(cf_data):
    """CausalForestResult must expose all required fields."""
    Y, D, X = cf_data
    result = fit_causal_forest(Y, D, X)
    assert isinstance(result, CausalForestResult)
    assert hasattr(result, "cate")
    assert hasattr(result, "ate")
    assert hasattr(result, "ate_se")
    assert hasattr(result, "n_obs")
    assert hasattr(result, "method")
    assert hasattr(result, "converged")


def test_cf_deterministic(cf_data):
    """Same random_state → same ATE."""
    Y, D, X = cf_data
    r1 = fit_causal_forest(Y, D, X, random_state=7)
    r2 = fit_causal_forest(Y, D, X, random_state=7)
    if r1.converged and r2.converged:
        assert abs(r1.ate - r2.ate) < 1e-10, (
            f"Non-deterministic ATE: {r1.ate} vs {r2.ate}"
        )


def test_cf_n_obs_correct(cf_data):
    """CausalForestResult.n_obs must match input length."""
    Y, D, X = cf_data
    result = fit_causal_forest(Y, D, X)
    assert result.n_obs == N


def test_cf_method_string_non_empty(cf_data):
    """method field must be a non-empty string."""
    Y, D, X = cf_data
    result = fit_causal_forest(Y, D, X)
    assert isinstance(result.method, str)
    assert len(result.method) > 0


def test_cf_ate_se_non_negative(cf_data):
    """ate_se must be >= 0 when converged."""
    Y, D, X = cf_data
    result = fit_causal_forest(Y, D, X)
    if result.converged and not math.isnan(result.ate_se):
        assert result.ate_se >= 0.0


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------


def test_can_import_both_functions():
    """Both fit_plr and fit_causal_forest must be importable from the module."""
    from src.assembled_core.signals.causal_ml import fit_causal_forest, fit_plr  # noqa: F401

    assert callable(fit_plr)
    assert callable(fit_causal_forest)
