"""Tests for M19a: Conformal Prediction."""

from __future__ import annotations

import pytest
import numpy as np

pytest.importorskip('src.assembled_core.ml.conformal_prediction')
from src.assembled_core.ml.conformal_prediction import (
    ConformalResult,
    SplitConformal,
    AdaptiveConformal,
    evaluate_coverage,
)


def _simple_predictor(X):
    """Linear predictor for testing."""
    return X[:, 0] * 2 + 1


def _difficulty_fn(X):
    """Difficulty increases with feature magnitude."""
    return np.abs(X[:, 0]) + 0.5


@pytest.fixture
def calibration_data():
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, 3))
    y = X[:, 0] * 2 + 1 + rng.normal(0, 0.5, 200)
    return X, y


@pytest.fixture
def test_data():
    rng = np.random.default_rng(99)
    X = rng.normal(0, 1, (100, 3))
    y = X[:, 0] * 2 + 1 + rng.normal(0, 0.5, 100)
    return X, y


@pytest.mark.phase12
class TestSplitConformal:
    def test_calibrate(self, calibration_data):
        X, y = calibration_data
        cp = SplitConformal(alpha=0.10)
        q = cp.calibrate(_simple_predictor, X, y)
        assert q > 0
        assert cp.is_calibrated

    def test_predict(self, calibration_data, test_data):
        X_cal, y_cal = calibration_data
        X_test, y_test = test_data
        cp = SplitConformal(alpha=0.10)
        cp.calibrate(_simple_predictor, X_cal, y_cal)
        result = cp.predict(X_test)

        assert isinstance(result, ConformalResult)
        assert len(result.predictions) == len(X_test)
        assert len(result.lower) == len(X_test)
        assert len(result.upper) == len(X_test)
        assert all(result.lower <= result.upper)
        assert result.coverage_target == pytest.approx(0.90)

    def test_coverage_approximately_valid(self, calibration_data, test_data):
        X_cal, y_cal = calibration_data
        X_test, y_test = test_data
        cp = SplitConformal(alpha=0.10)
        cp.calibrate(_simple_predictor, X_cal, y_cal)
        result = cp.predict(X_test)
        metrics = evaluate_coverage(result, y_test)
        # Coverage should be approximately >= 0.90 (may vary with sample)
        assert metrics["actual_coverage"] >= 0.80  # relaxed for small sample

    def test_predict_before_calibrate_raises(self):
        cp = SplitConformal(alpha=0.10)
        with pytest.raises(RuntimeError, match="calibrate"):
            cp.predict(np.array([[1, 2, 3]]))

    def test_invalid_alpha(self):
        with pytest.raises(ValueError, match="alpha"):
            SplitConformal(alpha=0.0)
        with pytest.raises(ValueError, match="alpha"):
            SplitConformal(alpha=1.0)

    def test_narrower_intervals_with_higher_alpha(self, calibration_data, test_data):
        X_cal, y_cal = calibration_data
        X_test, _ = test_data

        cp_tight = SplitConformal(alpha=0.50)
        cp_tight.calibrate(_simple_predictor, X_cal, y_cal)
        result_tight = cp_tight.predict(X_test)

        cp_wide = SplitConformal(alpha=0.05)
        cp_wide.calibrate(_simple_predictor, X_cal, y_cal)
        result_wide = cp_wide.predict(X_test)

        assert result_tight.interval_width.mean() < result_wide.interval_width.mean()


@pytest.mark.phase12
class TestAdaptiveConformal:
    def test_calibrate_and_predict(self, calibration_data, test_data):
        X_cal, y_cal = calibration_data
        X_test, _ = test_data

        cp = AdaptiveConformal(alpha=0.10)
        cp.calibrate(_simple_predictor, _difficulty_fn, X_cal, y_cal)
        result = cp.predict(X_test)

        assert isinstance(result, ConformalResult)
        assert len(result.predictions) == len(X_test)
        # Intervals should vary in width
        assert result.interval_width.std() > 0

    def test_harder_regions_wider(self, calibration_data, test_data):
        X_cal, y_cal = calibration_data
        cp = AdaptiveConformal(alpha=0.10)
        cp.calibrate(_simple_predictor, _difficulty_fn, X_cal, y_cal)

        # Easy region (small features)
        X_easy = np.zeros((10, 3))
        # Hard region (large features)
        X_hard = np.ones((10, 3)) * 5

        result_easy = cp.predict(X_easy)
        result_hard = cp.predict(X_hard)

        assert result_hard.interval_width.mean() > result_easy.interval_width.mean()


@pytest.mark.phase12
class TestEvaluateCoverage:
    def test_perfect_coverage(self):
        result = ConformalResult(
            predictions=np.array([1.0, 2.0, 3.0]),
            lower=np.array([0.0, 1.0, 2.0]),
            upper=np.array([2.0, 3.0, 4.0]),
            interval_width=np.array([2.0, 2.0, 2.0]),
            coverage_target=0.90,
            calibration_score=1.0,
        )
        y_true = np.array([1.0, 2.0, 3.0])
        metrics = evaluate_coverage(result, y_true)
        assert metrics["actual_coverage"] == 1.0
        assert metrics["pct_covered"] == 100.0

    def test_no_coverage(self):
        result = ConformalResult(
            predictions=np.array([1.0]),
            lower=np.array([10.0]),
            upper=np.array([20.0]),
            interval_width=np.array([10.0]),
            coverage_target=0.90,
            calibration_score=5.0,
        )
        metrics = evaluate_coverage(result, np.array([0.0]))
        assert metrics["actual_coverage"] == 0.0
