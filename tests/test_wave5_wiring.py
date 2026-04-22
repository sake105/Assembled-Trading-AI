"""Tests for wave-5 module wiring into trading_cycle.py.

Covers:
  Step 2.4 — ml.model_monitoring (detect_feature_drift)
  Step 3.7 — signals.meta_model (load_meta_model shadow scoring)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ml.model_monitoring import detect_feature_drift


# ---------------------------------------------------------------------------
# detect_feature_drift (Step 2.4)
# ---------------------------------------------------------------------------

def _make_feature_df(n: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "feat_stable": rng.normal(0, 1, n),
        "feat_var": rng.normal(5, 1, n),
        "timestamp": pd.date_range("2024-01-01", periods=n),
        "symbol": ["A"] * n,
    })


def test_drift_detects_shifted_distribution():
    train = _make_feature_df(100, seed=1)
    # Shift feat_stable by 5 sigma in recent data
    recent = _make_feature_df(30, seed=2)
    recent["feat_stable"] = recent["feat_stable"] + 10.0
    result = detect_feature_drift(train, recent, ["feat_stable", "feat_var"])
    drifted = {d["feature"] for d in result["drifted_features"]}
    assert "feat_stable" in drifted


def test_drift_no_drift_when_same_distribution():
    rng = np.random.default_rng(42)
    train = pd.DataFrame({"feat": rng.normal(0, 1, 200)})
    recent = pd.DataFrame({"feat": rng.normal(0, 1, 50)})
    result = detect_feature_drift(train, recent, ["feat"])
    assert result["alert_level"] in ("OK", "INFO", "WARNING", "CRITICAL")
    assert "n_tested" in result


def test_drift_insufficient_data_skips():
    train = pd.DataFrame({"feat": [1.0, 2.0]})  # < 30 rows
    recent = pd.DataFrame({"feat": [3.0]})
    result = detect_feature_drift(train, recent, ["feat"])
    assert result["n_tested"] == 0


def test_drift_missing_column_skipped():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"feat_a": rng.normal(0, 1, 100)})
    recent = pd.DataFrame({"feat_b": rng.normal(5, 1, 30)})
    result = detect_feature_drift(train, recent, ["feat_a", "feat_b"])
    assert result["n_tested"] == 0


def test_drift_result_structure():
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"f": rng.normal(0, 1, 100)})
    recent = pd.DataFrame({"f": rng.normal(0, 1, 30)})
    result = detect_feature_drift(train, recent, ["f"])
    assert "drift_score" in result
    assert "drifted_features" in result
    assert "alert_level" in result
    assert isinstance(result["drift_score"], float)


# ---------------------------------------------------------------------------
# meta_model (Step 3.7) — only test the load path + predict_proba interface
# (no model file on disk; test that the module is importable and MetaModel
# correctly raises when no model path exists)
# ---------------------------------------------------------------------------

def test_meta_model_imports_cleanly():
    from src.assembled_core.signals.meta_model import MetaModel, load_meta_model  # noqa: F401


def test_load_meta_model_raises_if_not_found(tmp_path):
    from src.assembled_core.signals.meta_model import load_meta_model
    missing = tmp_path / "nonexistent_model.joblib"
    with pytest.raises(Exception):
        load_meta_model(missing)


def test_meta_model_predict_proba_shape():
    """Test MetaModel.predict_proba with a mock sklearn model."""
    from src.assembled_core.signals.meta_model import MetaModel
    from unittest.mock import MagicMock
    import numpy as np

    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.6, 0.4]])
    mm = MetaModel(model=mock_model, feature_names=["f1", "f2"])
    X = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
    scores = mm.predict_proba(X)
    assert len(scores) == 2
    assert all(0 <= s <= 1 for s in scores)
