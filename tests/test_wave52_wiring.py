"""Tests for wave-52 module wiring into trading_cycle.py.

Covers:
  Step 8.49 — ml.gaussian_process (FactorGPR / GPRResult)
  Step 8.50 — ml.automl (run_automl / AutoMLResult)
  Step 8.51 — ml.causal_inference (screen_factors_causal / CausalEffectResult)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.gaussian_process import FactorGPR, GPRResult, SKLEARN_GP_AVAILABLE
from src.assembled_core.ml.automl import (
    AutoMLResult,
    ModelCandidate,
    run_automl,
    SKLEARN_AVAILABLE,
)
from src.assembled_core.ml.causal_inference import (
    CausalEffectResult,
    screen_factors_causal,
    granger_causality_test,
    GrangerResult,
)


# ---------------------------------------------------------------------------
# FactorGPR (Step 8.49)
# ---------------------------------------------------------------------------

def test_factor_gpr_creates():
    gpr = FactorGPR()
    assert isinstance(gpr, FactorGPR)


def test_factor_gpr_defaults():
    gpr = FactorGPR()
    assert gpr.length_scale > 0
    assert gpr.noise_level > 0
    assert not gpr._fitted


def test_factor_gpr_custom_params():
    gpr = FactorGPR(length_scale=2.0, noise_level=0.5, n_restarts=1)
    assert gpr.length_scale == 2.0
    assert gpr.noise_level == 0.5


def test_factor_gpr_fit_predict_sklearn():
    pytest.importorskip("sklearn", reason="scikit-learn required")
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (30, 3))
    y = rng.normal(0, 1, 30)
    gpr = FactorGPR(n_restarts=0)
    gpr.fit(X, y)
    assert gpr._fitted
    result = gpr.predict(X)
    assert isinstance(result, GPRResult)


def test_gpr_result_has_mean_std():
    pytest.importorskip("sklearn", reason="scikit-learn required")
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (30, 3))
    y = rng.normal(0, 1, 30)
    gpr = FactorGPR(n_restarts=0)
    gpr.fit(X, y)
    result = gpr.predict(X[:5])
    assert hasattr(result, "mean")
    assert hasattr(result, "std")
    assert len(result.mean) == 5


def test_sklearn_gp_available_flag():
    assert isinstance(SKLEARN_GP_AVAILABLE, bool)


# ---------------------------------------------------------------------------
# AutoML (Step 8.50)
# ---------------------------------------------------------------------------

def test_sklearn_available_flag():
    assert isinstance(SKLEARN_AVAILABLE, bool)


def test_model_candidate_creates():
    mc = ModelCandidate(
        model_type="ridge", params={"alpha": 1.0},
        ic_mean=0.05, ic_std=0.02, ic_ir=2.5,
        n_features=3, feature_names=["f1", "f2", "f3"], rank=1,
    )
    assert mc.model_type == "ridge"
    assert mc.rank == 1


def test_run_automl_no_sklearn_returns_result():
    if SKLEARN_AVAILABLE:
        pytest.skip("sklearn installed — testing no-sklearn path not applicable")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (30, 3)), columns=["f1", "f2", "f3"])
    y = pd.Series(rng.normal(0, 1, 30))
    result = run_automl(X, y)
    assert isinstance(result, AutoMLResult)


def test_run_automl_with_sklearn():
    pytest.importorskip("sklearn", reason="scikit-learn required")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (80, 3)), columns=["f1", "f2", "f3"])
    y = pd.Series(rng.normal(0, 1, 80))
    result = run_automl(X, y, model_types=["ridge"], n_folds=3)
    assert isinstance(result, AutoMLResult)
    assert result.n_models_evaluated >= 1


def test_run_automl_result_has_best_model():
    pytest.importorskip("sklearn", reason="scikit-learn required")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (80, 3)), columns=["f1", "f2", "f3"])
    y = pd.Series(rng.normal(0, 1, 80))
    result = run_automl(X, y, model_types=["ridge"], n_folds=3)
    assert isinstance(result.best_model, ModelCandidate)


# ---------------------------------------------------------------------------
# causal_inference (Step 8.51)
# ---------------------------------------------------------------------------

def _make_factor_return_df(n: int = 50) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    factor_df = pd.DataFrame(
        rng.normal(0, 1, (n, 3)),
        index=idx,
        columns=["momentum", "value", "quality"],
    )
    returns = pd.Series(rng.normal(0, 0.01, n), index=idx, name="ret")
    return factor_df, returns


def test_screen_factors_causal_returns_list():
    factor_df, returns = _make_factor_return_df()
    results = screen_factors_causal(factor_df, returns)
    assert isinstance(results, list)


def test_screen_factors_causal_one_per_factor():
    factor_df, returns = _make_factor_return_df()
    results = screen_factors_causal(factor_df, returns)
    assert len(results) == len(factor_df.columns)


def test_screen_factors_causal_result_type():
    factor_df, returns = _make_factor_return_df()
    results = screen_factors_causal(factor_df, returns)
    for r in results:
        assert isinstance(r, CausalEffectResult)


def test_screen_factors_causal_has_ate():
    factor_df, returns = _make_factor_return_df()
    results = screen_factors_causal(factor_df, returns)
    for r in results:
        assert hasattr(r, "ate")
        assert hasattr(r, "p_value")


def test_granger_causality_returns_result():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 50)
    y = rng.normal(0, 1, 50)
    result = granger_causality_test(x, y)
    assert isinstance(result, GrangerResult)


def test_causal_effect_result_is_significant_bool():
    factor_df, returns = _make_factor_return_df()
    results = screen_factors_causal(factor_df, returns)
    for r in results:
        assert isinstance(r.is_significant, bool)
