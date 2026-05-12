"""Wave 19 — KNOWN_ISSUES §8 follow-on helpers.

Covers:
    * qa.conformal_cross — Cross-Conformal (audit C2-033)
    * ml.bayesian_model_averaging — BMA (audit C2-058)
    * signals.regime_conditional_ensemble — regime-aware ensemble (audit C2-055)
    * risk.vol_targeting_ewma — EWMA forecast variant (audit C2-066)
    * attribution.brinson_multi_period — Cariño linking (audit C4-077)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ===========================================================================
# Cross-Conformal
# ===========================================================================


class _LinearMock:
    """Trivial regressor: predicts the calibration mean of y_train."""

    def __init__(self) -> None:
        self._mu: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LinearMock":
        self._mu = float(np.mean(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(len(X), self._mu)


def test_cross_conformal_basic_shape() -> None:
    from src.assembled_core.qa.conformal_cross import (
        fit_cross_conformal,
        predict_with_intervals_cross,
    )

    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 3))
    y = rng.normal(size=100)
    model, scores = fit_cross_conformal(_LinearMock(), X, y, n_folds=5)
    intervals = predict_with_intervals_cross(model, scores, X[:20], alpha=0.1)
    assert intervals.lower.shape == (20,)
    assert intervals.upper.shape == (20,)
    assert intervals.n_calibration == 100  # pooled = T
    assert intervals.n_folds == 5
    assert (intervals.upper >= intervals.lower).all()


def test_cross_conformal_rejects_too_few_folds() -> None:
    from src.assembled_core.qa.conformal_cross import fit_cross_conformal

    X = np.zeros((10, 1))
    y = np.zeros(10)
    with pytest.raises(ValueError):
        fit_cross_conformal(_LinearMock(), X, y, n_folds=1)


def test_cross_conformal_with_purge_horizon() -> None:
    from src.assembled_core.qa.conformal_cross import fit_cross_conformal

    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 2))
    y = rng.normal(size=120)
    _, scores = fit_cross_conformal(_LinearMock(), X, y, n_folds=4, purge_horizon=3)
    assert scores.size == 120


# ===========================================================================
# Bayesian Model Averaging
# ===========================================================================


def test_bma_weights_uniform_priors_sum_to_one() -> None:
    from src.assembled_core.ml.bayesian_model_averaging import bma_weights

    rng = np.random.default_rng(0)
    y = rng.normal(size=100)
    preds = [y + rng.normal(scale=s, size=100) for s in (0.5, 0.7, 1.0)]
    res = bma_weights(preds, y)
    assert np.isclose(res.weights.sum(), 1.0)
    # The lowest-error model (scale=0.5) should dominate.
    assert int(np.argmax(res.weights)) == 0


def test_bma_effective_n_models_diagnostic() -> None:
    from src.assembled_core.ml.bayesian_model_averaging import bma_weights

    # Two IDENTICAL predictions → posterior is symmetric, eff_n = 2.0.
    rng = np.random.default_rng(1)
    y = rng.normal(size=80)
    preds_same = y + rng.normal(scale=0.5, size=80)
    res = bma_weights([preds_same, preds_same], y)
    assert res.effective_n_models == pytest.approx(2.0, abs=1e-9)
    # Concentrated case: one model dominates → eff_n close to 1.
    great = y + rng.normal(scale=0.01, size=80)
    awful = y + rng.normal(scale=10.0, size=80)
    res2 = bma_weights([great, awful], y)
    assert res2.effective_n_models == pytest.approx(1.0, abs=1e-3)


def test_bma_bic_scoring_uses_complexities() -> None:
    from src.assembled_core.ml.bayesian_model_averaging import bma_weights

    rng = np.random.default_rng(2)
    y = rng.normal(size=100)
    # Two models, model A has higher complexity → BIC penalty should
    # shift weight toward B even with similar fit quality.
    preds = [y + rng.normal(scale=0.5, size=100), y + rng.normal(scale=0.5, size=100)]
    res = bma_weights(preds, y, scoring_rule="bic", model_complexities=[100, 1])
    assert res.weights[1] > res.weights[0]


def test_bma_predict_combines_correctly() -> None:
    from src.assembled_core.ml.bayesian_model_averaging import bma_predict

    p1 = np.array([1.0, 2.0, 3.0])
    p2 = np.array([4.0, 5.0, 6.0])
    out = bma_predict(np.array([0.5, 0.5]), [p1, p2])
    np.testing.assert_allclose(out, [2.5, 3.5, 4.5])


def test_bma_rejects_bad_inputs() -> None:
    from src.assembled_core.ml.bayesian_model_averaging import bma_weights

    with pytest.raises(ValueError):
        bma_weights([np.array([1.0, 2.0])], np.array([1.0, 2.0]))  # M < 2
    with pytest.raises(ValueError):
        bma_weights(
            [np.array([1.0]), np.array([1.0])],
            np.array([1.0, 2.0]),  # T mismatch
        )


# ===========================================================================
# Regime-conditional ensemble
# ===========================================================================


def test_regime_ensemble_uses_regime_weights() -> None:
    from src.assembled_core.signals.regime_conditional_ensemble import (
        conditional_ensemble,
    )

    p1 = np.array([1.0, 2.0, 3.0])
    p2 = np.array([10.0, 20.0, 30.0])
    weights = {
        "bull": np.array([1.0, 0.0]),  # only model A
        "bear": np.array([0.0, 1.0]),  # only model B
    }
    bull_out = conditional_ensemble(
        model_predictions=[p1, p2],
        current_regime="bull",
        per_regime_weights=weights,
    )
    bear_out = conditional_ensemble(
        model_predictions=[p1, p2],
        current_regime="bear",
        per_regime_weights=weights,
    )
    np.testing.assert_allclose(bull_out.combined, p1)
    np.testing.assert_allclose(bear_out.combined, p2)
    assert not bull_out.fell_back
    assert not bear_out.fell_back


def test_regime_ensemble_falls_back_to_uniform() -> None:
    from src.assembled_core.signals.regime_conditional_ensemble import (
        conditional_ensemble,
    )

    p1 = np.array([1.0, 1.0])
    p2 = np.array([3.0, 3.0])
    out = conditional_ensemble(
        model_predictions=[p1, p2],
        current_regime="unknown_regime",
        per_regime_weights={"bull": np.array([1.0, 0.0])},
    )
    np.testing.assert_allclose(out.combined, [2.0, 2.0])  # uniform mean
    assert out.fell_back is True
    assert out.regime_used == "uniform"


def test_regime_dispersion_zero_when_weights_identical() -> None:
    from src.assembled_core.signals.regime_conditional_ensemble import regime_dispersion

    w = np.array([0.5, 0.5])
    assert regime_dispersion({"r1": w, "r2": w, "r3": w}) == 0.0


def test_regime_dispersion_positive_when_weights_differ() -> None:
    from src.assembled_core.signals.regime_conditional_ensemble import regime_dispersion

    d = regime_dispersion(
        {
            "bull": np.array([1.0, 0.0]),
            "bear": np.array([0.0, 1.0]),
        }
    )
    assert d > 1.0  # max L2 distance for unit-sum probability vectors is √2


# ===========================================================================
# EWMA vol-targeting
# ===========================================================================


def test_ewma_vol_forecast_smoke() -> None:
    from src.assembled_core.risk.vol_targeting_ewma import ewma_vol_forecast

    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(scale=0.01, size=200))
    out = ewma_vol_forecast(ret)
    assert out.forecast_vol_annual > 0.0
    assert out.last_observation_count == 200
    assert out.lambda_used == 0.94


def test_ewma_vol_returns_nan_below_min_obs() -> None:
    from src.assembled_core.risk.vol_targeting_ewma import ewma_vol_forecast

    ret = pd.Series([0.001, -0.002, 0.001])
    out = ewma_vol_forecast(ret, min_observations=30)
    assert np.isnan(out.forecast_vol_annual)


def test_ewma_vol_rejects_bad_lambda() -> None:
    from src.assembled_core.risk.vol_targeting_ewma import ewma_vol_forecast

    ret = pd.Series(np.zeros(100))
    with pytest.raises(ValueError):
        ewma_vol_forecast(ret, lambda_=1.5)


def test_ewma_scale_factor_clamps() -> None:
    from src.assembled_core.risk.vol_targeting_ewma import compute_ewma_scale_factor

    # forecast 40%, target 20%, default cap 1.5 → raw 0.5 ≤ 1.5 ✔
    assert compute_ewma_scale_factor(0.40, 0.20) == pytest.approx(0.5)
    # forecast 5%, target 20%, default cap 1.5 → raw 4.0 clamped to 1.5
    assert compute_ewma_scale_factor(0.05, 0.20) == 1.5
    # forecast 0 -> safe fallback 1.0
    assert compute_ewma_scale_factor(0.0, 0.20) == 1.0


# ===========================================================================
# Brinson multi-period Cariño linking
# ===========================================================================


def test_carino_link_reconciles_geometric_active_return() -> None:
    from src.assembled_core.attribution.brinson_multi_period import (
        link_multi_period_attribution,
        reconciliation_residual,
    )

    n = 12
    rng = np.random.default_rng(0)
    port_ret = pd.Series(rng.normal(0.01, 0.04, n))
    bench_ret = pd.Series(rng.normal(0.005, 0.04, n))
    # Synthetic single-period attribution whose row-sum equals port_ret - bench_ret.
    active = port_ret - bench_ret
    single = pd.DataFrame(
        {
            "allocation": 0.5 * active,
            "selection": 0.3 * active,
            "interaction": 0.2 * active,
            "active_total": active,
        }
    )
    linked = link_multi_period_attribution(single, port_ret, bench_ret)
    residual = reconciliation_residual(linked, port_ret, bench_ret)
    assert abs(residual) < 1e-10


def test_carino_link_handles_zero_active_return() -> None:
    from src.assembled_core.attribution.brinson_multi_period import (
        carino_link_coefficients,
    )

    port = pd.Series([0.01, 0.02, -0.01])
    bench = port.copy()  # exact match each period
    k = carino_link_coefficients(port, bench)
    # When port == bench every period, coefficients should be finite
    # and the multi-period decomposition trivially reconciles (since
    # single-period attribution is zero everywhere).
    assert np.isfinite(k).all()


def test_carino_link_columns_preserved() -> None:
    from src.assembled_core.attribution.brinson_multi_period import (
        link_multi_period_attribution,
    )

    port_ret = pd.Series([0.01, 0.02])
    bench_ret = pd.Series([0.005, 0.015])
    single = pd.DataFrame({"allocation": [0.001, 0.002], "selection": [0.004, 0.003]})
    linked = link_multi_period_attribution(single, port_ret, bench_ret)
    assert list(linked.columns) == ["allocation", "selection"]
