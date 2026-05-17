"""Tests for M32: AutoML — Automated Model Selection and Feature Engineering."""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

pytest.importorskip("src.assembled_core.ml.automl")
from src.assembled_core.ml.automl import (
    AutoMLResult,
    compute_ic,
    select_features_mi,
    time_series_cv_split,
    run_automl,
)


@pytest.fixture
def synthetic_features():
    """Synthetic feature matrix with signal and noise."""
    rng = np.random.default_rng(42)
    n = 300
    return pd.DataFrame(
        {
            "signal_1": rng.normal(0, 1, n),
            "signal_2": rng.normal(0, 1, n),
            "noise_1": rng.normal(0, 1, n),
            "noise_2": rng.normal(0, 1, n),
            "noise_3": rng.normal(0, 1, n),
        }
    )


@pytest.fixture
def synthetic_target(synthetic_features):
    """Target with known relationship to signal features."""
    rng = np.random.default_rng(42)
    n = len(synthetic_features)
    return pd.Series(
        0.03 * synthetic_features["signal_1"].values
        + 0.02 * synthetic_features["signal_2"].values
        + rng.normal(0, 0.05, n),
        index=synthetic_features.index,
    )


@pytest.mark.fast
class TestComputeIC:
    def test_perfect_prediction(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic = compute_ic(x, x)
        assert ic == pytest.approx(1.0, abs=0.01)

    def test_inverse_prediction(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic = compute_ic(x, -x)
        assert ic == pytest.approx(-1.0, abs=0.01)

    def test_random_near_zero(self):
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 100)
        y = rng.normal(0, 1, 100)
        ic = compute_ic(x, y)
        assert abs(ic) < 0.3

    def test_short_data(self):
        ic = compute_ic(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert ic == 0.0


@pytest.mark.fast
class TestFeatureSelection:
    def test_selects_signal_features(self, synthetic_features, synthetic_target):
        selected = select_features_mi(
            synthetic_features, synthetic_target, max_features=3
        )
        assert len(selected) <= 3
        assert len(selected) > 0
        # Signal features should rank higher
        assert "signal_1" in selected or "signal_2" in selected

    def test_max_features_respected(self, synthetic_features, synthetic_target):
        selected = select_features_mi(
            synthetic_features, synthetic_target, max_features=2
        )
        assert len(selected) <= 2

    def test_empty_features(self, synthetic_target):
        selected = select_features_mi(pd.DataFrame(), synthetic_target)
        assert selected == []


@pytest.mark.fast
class TestTimeSeriesCV:
    def test_basic_splits(self):
        splits = time_series_cv_split(200, n_folds=5)
        assert len(splits) > 0
        for train_idx, test_idx in splits:
            # Train should come before test (temporal ordering)
            assert train_idx.max() < test_idx.min()

    def test_purge_gap(self):
        splits = time_series_cv_split(200, n_folds=3, purge_gap=10)
        for train_idx, test_idx in splits:
            gap = test_idx.min() - train_idx.max()
            assert gap >= 10

    def test_short_data_fewer_splits(self):
        splits = time_series_cv_split(50, n_folds=5, min_train_size=30)
        assert len(splits) <= 5


@pytest.mark.fast
class TestRunAutoML:
    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="sklearn required"),
        reason="sklearn not available",
    )
    def test_basic_automl(self, synthetic_features, synthetic_target):
        result = run_automl(
            synthetic_features,
            synthetic_target,
            model_types=["ridge", "lasso"],
            max_features=3,
            n_folds=3,
        )
        assert isinstance(result, AutoMLResult)
        assert result.n_models_evaluated > 0
        assert result.best_model.rank == 1
        assert len(result.selected_features) > 0

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="sklearn required"),
        reason="sklearn not available",
    )
    def test_all_candidates_ranked(self, synthetic_features, synthetic_target):
        result = run_automl(
            synthetic_features,
            synthetic_target,
            model_types=["ridge"],
            n_folds=3,
        )
        ranks = [c.rank for c in result.all_candidates]
        assert ranks == sorted(ranks)

    def test_insufficient_data(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        y = pd.Series([0.01, 0.02, 0.03])
        result = run_automl(X, y)
        assert result.best_model.model_type == "none"

    @pytest.mark.skipif(
        not pytest.importorskip("sklearn", reason="sklearn required"),
        reason="sklearn not available",
    )
    def test_ic_ir_ordering(self, synthetic_features, synthetic_target):
        result = run_automl(
            synthetic_features,
            synthetic_target,
            model_types=["ridge", "lasso"],
            n_folds=3,
        )
        if len(result.all_candidates) >= 2:
            # Should be sorted by IC-IR descending
            assert result.all_candidates[0].ic_ir >= result.all_candidates[-1].ic_ir
