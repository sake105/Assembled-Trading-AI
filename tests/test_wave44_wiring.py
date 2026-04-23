"""Tests for wave-44 module wiring into trading_cycle.py.

Covers:
  Step 8.40 — ops.compare (compare_summaries)
  Step 8.41 — ops.experiment_runner (deep_merge_policy)
  Step 8.42 — ml.explainability (compute_model_feature_importance)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.ops.compare import compare_summaries
from src.assembled_core.ops.experiment_runner import deep_merge_policy


# ---------------------------------------------------------------------------
# compare_summaries (Step 8.40)
# ---------------------------------------------------------------------------

def _write_summary(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_compare_summaries_returns_dict(tmp_path):
    a = tmp_path / "exp_a" / "summary.json"
    b = tmp_path / "exp_b" / "summary.json"
    _write_summary(a, {"total_return": 0.12, "max_drawdown": -0.08})
    _write_summary(b, {"total_return": 0.15, "max_drawdown": -0.06})
    result = compare_summaries(a, b)
    assert isinstance(result, dict)


def test_compare_summaries_has_schema_version(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_summary(a, {"total_return": 0.10})
    _write_summary(b, {"total_return": 0.12})
    result = compare_summaries(a, b)
    assert "schema_version" in result


def test_compare_summaries_missing_a_raises(tmp_path):
    b = tmp_path / "b.json"
    _write_summary(b, {"total_return": 0.10})
    with pytest.raises(FileNotFoundError):
        compare_summaries(tmp_path / "nope.json", b)


def test_compare_summaries_missing_b_raises(tmp_path):
    a = tmp_path / "a.json"
    _write_summary(a, {"total_return": 0.10})
    with pytest.raises(FileNotFoundError):
        compare_summaries(a, tmp_path / "nope.json")


def test_compare_summaries_has_delta(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _write_summary(a, {"total_return": 0.10})
    _write_summary(b, {"total_return": 0.15})
    result = compare_summaries(a, b)
    assert "delta" in result or "a" in result


def test_compare_summaries_equal_files(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    data = {"total_return": 0.10, "max_drawdown": -0.05}
    _write_summary(a, data)
    _write_summary(b, data)
    result = compare_summaries(a, b)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# deep_merge_policy (Step 8.41)
# ---------------------------------------------------------------------------

def test_deep_merge_returns_dict():
    result = deep_merge_policy({"a": 1}, {"b": 2})
    assert isinstance(result, dict)


def test_deep_merge_combines_keys():
    result = deep_merge_policy({"a": 1, "b": 2}, {"b": 99, "c": 3})
    assert result["a"] == 1
    assert result["b"] == 99
    assert result["c"] == 3


def test_deep_merge_nested():
    base = {"risk": {"max_dd": 0.15, "vol": 0.10}}
    overrides = {"risk": {"max_dd": 0.20}}
    result = deep_merge_policy(base, overrides)
    assert result["risk"]["max_dd"] == 0.20
    assert result["risk"]["vol"] == 0.10


def test_deep_merge_empty_base():
    result = deep_merge_policy({}, {"x": 1})
    assert result["x"] == 1


def test_deep_merge_empty_overrides():
    result = deep_merge_policy({"x": 1}, {})
    assert result["x"] == 1


def test_deep_merge_does_not_mutate_base():
    base = {"a": 1}
    deep_merge_policy(base, {"b": 2})
    assert "b" not in base


# ---------------------------------------------------------------------------
# compute_model_feature_importance (Step 8.42)
# ---------------------------------------------------------------------------

pytest.importorskip("sklearn", reason="scikit-learn required for explainability")

from src.assembled_core.ml.explainability import (  # noqa: E402
    compute_model_feature_importance,
    compute_permutation_importance,
)


def _make_ridge():
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (50, 4))
    y = rng.normal(0, 1, 50)
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model, ["f1", "f2", "f3", "f4"], X, y


def test_compute_model_importance_returns_df():
    model, names, X, y = _make_ridge()
    result = compute_model_feature_importance(model, names)
    assert isinstance(result, pd.DataFrame)


def test_compute_model_importance_columns():
    model, names, X, y = _make_ridge()
    result = compute_model_feature_importance(model, names)
    assert "feature" in result.columns
    assert "importance" in result.columns


def test_compute_model_importance_length():
    model, names, X, y = _make_ridge()
    result = compute_model_feature_importance(model, names)
    assert len(result) == 4


def test_compute_model_importance_sorted():
    model, names, X, y = _make_ridge()
    result = compute_model_feature_importance(model, names)
    importances = result["importance"].values
    assert all(importances[i] >= importances[i + 1] for i in range(len(importances) - 1))


def test_compute_permutation_importance_returns_df():
    model, names, X, y = _make_ridge()
    X_df = pd.DataFrame(X, columns=names)
    y_s = pd.Series(y)
    result = compute_permutation_importance(model, X_df, y_s)
    assert isinstance(result, pd.DataFrame)
