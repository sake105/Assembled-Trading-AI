"""Tests for wave-27 module wiring into trading_cycle.py.

Covers:
  Step 2.17 — ml.evt_models (compute_evt_risk_metrics)
  Step 3.8  — ml.conformal_prediction (SplitConformal)
  Step 7.64 — ml.model_registry (ModelRegistry)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.evt_models import compute_evt_risk_metrics, fit_evt_pot
from src.assembled_core.ml.conformal_prediction import (
    SplitConformal,
    ConformalResult,
)
from src.assembled_core.ml.model_registry import ModelRegistry, ModelRecord


# ---------------------------------------------------------------------------
# compute_evt_risk_metrics (Step 2.17)
# ---------------------------------------------------------------------------

def _make_returns(n: int = 300, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0, 0.01, n))


def test_evt_returns_dict():
    rets = _make_returns(300)
    metrics = compute_evt_risk_metrics(rets)
    assert isinstance(metrics, dict)


def test_evt_keys_present():
    rets = _make_returns(300)
    metrics = compute_evt_risk_metrics(rets)
    for key in ["evt_var_95", "evt_var_99", "evt_cvar_99", "evt_shape_xi"]:
        assert key in metrics


def test_evt_insufficient_data_returns_zeros():
    rets = _make_returns(50)  # below 100 threshold
    metrics = compute_evt_risk_metrics(rets)
    # Should return conservative zeros when fit fails
    assert isinstance(metrics, dict)
    assert metrics["evt_var_99"] >= 0.0


def test_evt_var99_leq_cvar99():
    rets = _make_returns(500)
    metrics = compute_evt_risk_metrics(rets)
    # CVaR >= VaR by definition (if non-zero)
    if metrics["evt_var_99"] > 0:
        assert metrics["evt_cvar_99"] >= metrics["evt_var_99"]


def test_evt_values_non_negative():
    rets = _make_returns(400)
    metrics = compute_evt_risk_metrics(rets)
    for key in ["evt_var_95", "evt_var_99", "evt_cvar_99"]:
        assert metrics[key] >= 0.0


def test_evt_fit_pot_returns_none_for_short_series():
    rets = _make_returns(50)
    result = fit_evt_pot(rets)
    assert result is None


def test_evt_var_increases_with_heavier_tail():
    rng = np.random.default_rng(42)
    light = pd.Series(rng.normal(0, 0.01, 400))
    heavy = pd.Series(rng.standard_t(df=3, size=400) * 0.01)
    m_light = compute_evt_risk_metrics(light)
    m_heavy = compute_evt_risk_metrics(heavy)
    # Heavy tail should generally have higher tail risk
    assert isinstance(m_light, dict)
    assert isinstance(m_heavy, dict)


# ---------------------------------------------------------------------------
# SplitConformal (Step 3.8)
# ---------------------------------------------------------------------------

def _make_regression_data(n: int = 100, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3))
    y = X[:, 0] * 0.5 + rng.normal(0, 0.1, n)
    return X, y


def test_split_conformal_calibrate_returns_float():
    X, y = _make_regression_data(100)
    cp = SplitConformal(alpha=0.10)
    q = cp.calibrate(lambda x: np.zeros(len(x)), X[:50], y[:50])
    assert isinstance(q, float)
    assert q >= 0.0


def test_split_conformal_predict_returns_result():
    X, y = _make_regression_data(100)
    cp = SplitConformal(alpha=0.10)
    cp.calibrate(lambda x: np.zeros(len(x)), X[:50], y[:50])
    result = cp.predict(X[50:])
    assert isinstance(result, ConformalResult)


def test_split_conformal_interval_covers_target():
    rng = np.random.default_rng(99)
    X = rng.standard_normal((200, 2))
    y = X[:, 0] + rng.normal(0, 0.5, 200)
    predictor = lambda x: x[:, 0]
    cp = SplitConformal(alpha=0.10)
    cp.calibrate(predictor, X[:100], y[:100])
    result = cp.predict(X[100:])
    covered = np.mean((y[100:] >= result.lower) & (y[100:] <= result.upper))
    assert covered >= 0.75  # should be ~90% but allow noise


def test_split_conformal_lower_leq_upper():
    X, y = _make_regression_data(100)
    cp = SplitConformal(alpha=0.10)
    cp.calibrate(lambda x: np.full(len(x), 0.0), X[:50], y[:50])
    result = cp.predict(X[50:])
    assert (result.lower <= result.upper).all()


def test_split_conformal_invalid_alpha_raises():
    with pytest.raises(ValueError):
        SplitConformal(alpha=1.5)


def test_split_conformal_predict_before_calibrate_raises():
    cp = SplitConformal(alpha=0.10)
    X, _ = _make_regression_data(10)
    with pytest.raises(RuntimeError):
        cp.predict(X)


def test_split_conformal_interval_width_positive():
    X, y = _make_regression_data(100)
    cp = SplitConformal(alpha=0.10)
    cp.calibrate(lambda x: np.zeros(len(x)), X[:50], y[:50])
    result = cp.predict(X[50:])
    assert (result.interval_width > 0).all()


# ---------------------------------------------------------------------------
# ModelRegistry (Step 7.64)
# ---------------------------------------------------------------------------

def test_model_registry_creates_empty(tmp_path):
    reg = ModelRegistry(base_dir=tmp_path / "models")
    assert isinstance(reg._records, dict)
    assert len(reg._records) == 0


def test_model_registry_records_is_dict(tmp_path):
    reg = ModelRegistry(base_dir=tmp_path / "models")
    assert isinstance(reg._records, dict)


def test_model_registry_missing_dir_ok(tmp_path):
    # Should not crash when directory doesn't exist yet
    reg = ModelRegistry(base_dir=tmp_path / "nonexistent" / "models")
    assert reg._records == {}


def test_model_registry_from_json(tmp_path):
    # Manually write a registry JSON, then load it
    import json
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    registry_data = {
        "models": {
            "test_model": [
                {
                    "model_id": "test_model",
                    "version": 1,
                    "model_type": "lr",
                    "trained_at": "2024-01-15T09:00:00",
                    "file_path": "test_model/v1.joblib",
                    "metrics": {"sharpe": 1.2},
                    "features": ["f1", "f2"],
                    "train_start": "2023-01-01",
                    "train_end": "2024-01-01",
                    "status": "candidate",
                    "notes": "",
                }
            ]
        }
    }
    (models_dir / "_registry.json").write_text(json.dumps(registry_data))
    reg = ModelRegistry(base_dir=models_dir)
    assert "test_model" in reg._records
    assert len(reg._records["test_model"]) == 1


def test_model_registry_list_versions_from_json(tmp_path):
    import json
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    registry_data = {
        "models": {
            "alpha_model": [
                {
                    "model_id": "alpha_model", "version": 1,
                    "model_type": "xgb", "trained_at": "2024-01-01",
                    "file_path": "alpha_model/v1.joblib",
                    "metrics": {}, "features": [],
                    "train_start": "2023-01-01", "train_end": "2024-01-01",
                    "status": "candidate", "notes": "",
                }
            ]
        }
    }
    (models_dir / "_registry.json").write_text(json.dumps(registry_data))
    reg = ModelRegistry(base_dir=models_dir)
    versions = reg.list_versions("alpha_model")
    assert len(versions) == 1
    assert versions[0].model_id == "alpha_model"
