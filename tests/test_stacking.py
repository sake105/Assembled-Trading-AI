"""Tests for ML stacking ensemble module."""

from __future__ import annotations

import pytest

pytest.importorskip("src.assembled_core.ml.stacking")
import pytest
import numpy as np
import pandas as pd

sklearn = pytest.importorskip("sklearn")

from src.assembled_core.ml.stacking import (
    StackedEnsemble,
    build_default_stack,
    enforce_ensemble_diversity,
)
from src.assembled_core.ml.factor_models import MLModelConfig, MLExperimentConfig


def _synthetic_panel(n: int = 500, k: int = 5, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic factor panel for stacking tests."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    rows = []
    for sym in ["AAPL", "MSFT", "GOOG"]:
        for i, d in enumerate(dates):
            features = {f"f{j}": rng.normal(0, 1) for j in range(k)}
            target = 0.02 * features["f0"] + 0.01 * features["f1"] + rng.normal(0, 0.05)
            rows.append(
                {"timestamp": d, "symbol": sym, "fwd_return_5d": target, **features}
            )
    return pd.DataFrame(rows)


@pytest.mark.fast
class TestStackedEnsemble:
    def test_fit_predict(self):
        panel = _synthetic_panel(n=200)
        configs = [
            MLModelConfig(name="ridge", model_type="ridge", params={"alpha": 1.0}),
            MLModelConfig(name="lasso", model_type="lasso", params={"alpha": 0.01}),
        ]
        exp = MLExperimentConfig(
            label_col="fwd_return_5d",
            feature_cols=[f"f{i}" for i in range(5)],
            n_splits=3,
            min_train_samples=30,
            standardize=True,
        )
        stack = StackedEnsemble(base_configs=configs, meta_alpha=1.0)
        stack.fit(panel, exp)
        assert stack._is_fitted

        X_test = panel[[f"f{i}" for i in range(5)]].iloc[:10]
        preds = stack.predict(X_test)
        assert len(preds) == 10
        assert all(np.isfinite(preds))

    def test_not_fitted_raises(self):
        stack = StackedEnsemble(
            base_configs=[
                MLModelConfig(name="ridge", model_type="ridge", params={"alpha": 1.0})
            ],
        )
        with pytest.raises(RuntimeError, match="fitted"):
            stack.predict(np.zeros((5, 3)))

    def test_empty_base_configs_raises(self):
        stack = StackedEnsemble(base_configs=[])
        panel = _synthetic_panel(n=100)
        exp = MLExperimentConfig(label_col="fwd_return_5d", n_splits=2)
        with pytest.raises(ValueError, match="empty"):
            stack.fit(panel, exp)

    def test_predict_with_confidence(self):
        panel = _synthetic_panel(n=200)
        configs = [
            MLModelConfig(name="ridge", model_type="ridge", params={"alpha": 1.0}),
        ]
        exp = MLExperimentConfig(
            label_col="fwd_return_5d",
            feature_cols=[f"f{i}" for i in range(5)],
            n_splits=3,
            min_train_samples=30,
        )
        stack = StackedEnsemble(base_configs=configs)
        stack.fit(panel, exp)
        X_test = panel[[f"f{i}" for i in range(5)]].iloc[:5]
        preds, lo, hi = stack.predict_with_confidence(X_test, confidence_level=0.9)
        assert len(preds) == 5
        assert all(lo <= preds)
        assert all(preds <= hi)

    def test_diversity_report(self):
        panel = _synthetic_panel(n=200)
        configs = [
            MLModelConfig(name="ridge", model_type="ridge", params={"alpha": 1.0}),
            MLModelConfig(name="lasso", model_type="lasso", params={"alpha": 0.01}),
        ]
        exp = MLExperimentConfig(
            label_col="fwd_return_5d",
            feature_cols=[f"f{i}" for i in range(5)],
            n_splits=3,
            min_train_samples=30,
        )
        stack = StackedEnsemble(base_configs=configs)
        stack.fit(panel, exp)
        X = panel[[f"f{i}" for i in range(5)]].iloc[:50]
        report = stack.diversity_report(X)
        assert report.shape == (2, 2)


@pytest.mark.fast
class TestBuildDefaultStack:
    def test_creates_ensemble(self):
        stack = build_default_stack(include_boosting=False)
        assert isinstance(stack, StackedEnsemble)
        assert len(stack.base_configs) >= 2  # at least ridge + lasso

    def test_with_boosting(self):
        stack = build_default_stack(include_boosting=True)
        assert isinstance(stack, StackedEnsemble)
        # Should have at least the linear models
        names = [c.name for c in stack.base_configs]
        assert "ridge" in names
        assert "random_forest" in names


@pytest.mark.fast
class TestEnforceDiversity:
    def test_diverse_models(self):
        rng = np.random.default_rng(42)
        oof = rng.normal(0, 1, (100, 3))
        result = enforce_ensemble_diversity(oof, max_correlation=0.8)
        assert result["diverse"] is True
        assert len(result["recommendations"]) == 0

    def test_correlated_models(self):
        rng = np.random.default_rng(42)
        base = rng.normal(0, 1, 100)
        oof = np.column_stack([base, base + rng.normal(0, 0.01, 100)])
        result = enforce_ensemble_diversity(oof, max_correlation=0.8)
        assert result["max_pair_correlation"] > 0.9
        assert result["diverse"] is False

    def test_single_model(self):
        oof = np.random.randn(50, 1)
        result = enforce_ensemble_diversity(oof)
        assert result["diverse"] is True
