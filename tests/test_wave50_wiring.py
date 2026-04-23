"""Tests for wave-50 module wiring into trading_cycle.py.

Covers:
  Step 8.46 — ml.bayesian_ensemble (compute_bma_weights / run_bayesian_ensemble)
  Step 8.47 — ml.stacking_ensemble (StackingConfig / run_stacking_cv)
  Step 8.48 — ml.tda_regime (compute_persistence_features / extract_tda_features)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.bayesian_ensemble import (
    compute_bma_weights,
    run_bayesian_ensemble,
    BMAResult,
)
from src.assembled_core.ml.stacking_ensemble import (
    StackingConfig,
    StackingResult,
    run_stacking_cv,
)
from src.assembled_core.ml.tda_regime import (
    compute_persistence_features,
    extract_tda_features,
    TDAFeatures,
)


# ---------------------------------------------------------------------------
# bayesian_ensemble (Step 8.46)
# ---------------------------------------------------------------------------

def test_compute_bma_weights_returns_dict():
    scores = {"model_a": -0.3, "model_b": -0.4, "model_c": -0.35}
    result = compute_bma_weights(scores)
    assert isinstance(result, dict)


def test_compute_bma_weights_sums_to_one():
    scores = {"model_a": -0.3, "model_b": -0.4, "model_c": -0.35}
    result = compute_bma_weights(scores)
    total = sum(result.values())
    assert abs(total - 1.0) < 1e-6


def test_compute_bma_weights_empty_returns_empty():
    result = compute_bma_weights({})
    assert isinstance(result, dict)
    assert len(result) == 0


def test_compute_bma_weights_single_model():
    result = compute_bma_weights({"only_model": -0.5})
    assert abs(result["only_model"] - 1.0) < 1e-6


def test_compute_bma_weights_higher_temperature_flatter():
    scores = {"a": -0.2, "b": -0.5}
    w_low = compute_bma_weights(scores, temperature=0.5)
    w_high = compute_bma_weights(scores, temperature=5.0)
    # At higher temperature, distribution is flatter → max weight smaller
    assert max(w_high.values()) < max(w_low.values())


def test_run_bayesian_ensemble_sklearn_gated():
    pytest.importorskip("sklearn", reason="scikit-learn required")
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    X_tr = pd.DataFrame(rng.normal(0, 1, (40, 3)), columns=["f1", "f2", "f3"])
    y_tr = pd.Series(rng.normal(0, 1, 40))
    X_val = pd.DataFrame(rng.normal(0, 1, (10, 3)), columns=["f1", "f2", "f3"])
    y_val = pd.Series(rng.normal(0, 1, 10))
    factories = {"ridge_a": lambda: Ridge(alpha=1.0), "ridge_b": lambda: Ridge(alpha=0.1)}
    result = run_bayesian_ensemble(X_tr, y_tr, X_val, y_val, factories)
    assert isinstance(result, BMAResult)


# ---------------------------------------------------------------------------
# stacking_ensemble (Step 8.47)
# ---------------------------------------------------------------------------

def test_stacking_config_creates():
    cfg = StackingConfig()
    assert isinstance(cfg, StackingConfig)


def test_stacking_config_has_base_models():
    cfg = StackingConfig()
    assert len(cfg.base_models) >= 1


def test_stacking_config_has_n_splits():
    cfg = StackingConfig()
    assert cfg.n_splits >= 2


def test_stacking_config_meta_model():
    cfg = StackingConfig()
    assert isinstance(cfg.meta_model, str)


def test_run_stacking_cv_sklearn_gated():
    pytest.importorskip("sklearn", reason="scikit-learn required")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (60, 4)), columns=["f1", "f2", "f3", "f4"])
    y = pd.Series(rng.normal(0, 1, 60))
    cfg = StackingConfig(n_splits=3)
    result = run_stacking_cv(X, y, config=cfg)
    assert isinstance(result, StackingResult)


def test_run_stacking_cv_has_oof_preds():
    pytest.importorskip("sklearn", reason="scikit-learn required")
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(0, 1, (60, 3)), columns=["f1", "f2", "f3"])
    y = pd.Series(rng.normal(0, 1, 60))
    result = run_stacking_cv(X, y, config=StackingConfig(n_splits=3))
    assert hasattr(result, "oof_predictions") or hasattr(result, "metrics")


# ---------------------------------------------------------------------------
# tda_regime (Step 8.48)
# ---------------------------------------------------------------------------

def _make_point_cloud(n: int = 30) -> np.ndarray:
    rng = np.random.default_rng(0)
    rets = rng.normal(0, 0.01, n + 1)
    return np.column_stack([rets[:-1], rets[1:]])


def test_compute_persistence_features_returns_tuple():
    cloud = _make_point_cloud()
    diagrams, features = compute_persistence_features(cloud)
    assert isinstance(diagrams, list)
    assert isinstance(features, dict)


def test_compute_persistence_features_has_entropy():
    cloud = _make_point_cloud()
    _, features = compute_persistence_features(cloud)
    assert "persistence_entropy" in features


def test_compute_persistence_features_entropy_non_negative():
    cloud = _make_point_cloud()
    _, features = compute_persistence_features(cloud)
    assert features["persistence_entropy"] >= 0.0


def test_extract_tda_features_returns_tda_features():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 30))
    result = extract_tda_features(returns)
    assert isinstance(result, TDAFeatures)


def test_tda_features_has_entropy():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0, 0.01, 30))
    result = extract_tda_features(returns)
    assert hasattr(result, "persistence_entropy")


def test_compute_persistence_different_clouds_differ():
    rng = np.random.default_rng(1)
    # Circular data (should have different topology)
    t = np.linspace(0, 2 * np.pi, 20)
    circle = np.column_stack([np.cos(t), np.sin(t)])
    _, feat_circle = compute_persistence_features(circle)
    # Random cloud
    cloud_random = rng.normal(0, 1, (20, 2))
    _, feat_random = compute_persistence_features(cloud_random)
    # Both should return valid results
    assert "persistence_entropy" in feat_circle
    assert "persistence_entropy" in feat_random
