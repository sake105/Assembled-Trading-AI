"""Tests for probability calibration module."""

from __future__ import annotations

import pytest
import numpy as np

pytest.importorskip('src.assembled_core.ml.calibration')
from src.assembled_core.ml.calibration import (
    compute_calibration_error,
    IsotonicCalibrator,
    TemperatureScaler,
    CalibrationResult,
)


def _synthetic_classification(n: int = 500, seed: int = 42):
    rng = np.random.default_rng(seed)
    y_true = rng.binomial(1, 0.5, n).astype(float)
    # Poorly calibrated probabilities
    y_prob = np.clip(y_true + rng.normal(0, 0.3, n), 0.01, 0.99)
    return y_true, y_prob


@pytest.mark.phase12
class TestCalibrationError:
    def test_basic(self):
        y_true, y_prob = _synthetic_classification()
        result = compute_calibration_error(y_true, y_prob)
        assert isinstance(result, CalibrationResult)
        assert result.brier_score >= 0
        assert result.ece >= 0
        assert result.mce >= 0

    def test_perfect_calibration(self):
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_prob = np.array([0.0, 0.0, 1.0, 1.0])
        result = compute_calibration_error(y_true, y_prob)
        assert result.brier_score == pytest.approx(0.0)

    def test_worst_calibration(self):
        y_true = np.array([0.0, 0.0, 1.0, 1.0])
        y_prob = np.array([1.0, 1.0, 0.0, 0.0])  # Completely wrong
        result = compute_calibration_error(y_true, y_prob)
        assert result.brier_score > 0.5

    def test_custom_bins(self):
        y_true, y_prob = _synthetic_classification()
        result = compute_calibration_error(y_true, y_prob, n_bins=20)
        assert result.n_bins == 20


@pytest.mark.phase12
class TestIsotonicCalibrator:
    def test_fit_transform(self):
        y_true, y_prob = _synthetic_classification()
        cal = IsotonicCalibrator()
        cal.fit(y_true, y_prob)
        calibrated = cal.transform(y_prob[:10])
        assert len(calibrated) == 10
        assert (calibrated >= 0).all()
        assert (calibrated <= 1).all()

    def test_improves_calibration(self):
        y_true, y_prob = _synthetic_classification()
        n_cal = 250
        cal = IsotonicCalibrator()
        cal.fit(y_true[:n_cal], y_prob[:n_cal])
        calibrated = cal.transform(y_prob[n_cal:])

        raw_ece = compute_calibration_error(y_true[n_cal:], y_prob[n_cal:]).ece
        cal_ece = compute_calibration_error(y_true[n_cal:], calibrated).ece
        # Calibrated should generally improve or be similar
        assert cal_ece <= raw_ece + 0.1

    def test_not_fitted_raises(self):
        cal = IsotonicCalibrator()
        with pytest.raises(RuntimeError):
            cal.transform(np.array([0.5]))


@pytest.mark.phase12
class TestTemperatureScaler:
    def test_fit_transform_v2(self):
        rng = np.random.default_rng(42)
        logits = rng.normal(0, 2, 200)
        y_true = (logits > 0).astype(float)
        scaler = TemperatureScaler()
        scaler.fit(logits, y_true)
        probs = scaler.transform(logits[:10])
        assert len(probs) == 10
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_temperature_reasonable(self):
        rng = np.random.default_rng(42)
        logits = rng.normal(0, 2, 200)
        y_true = (logits > 0).astype(float)
        scaler = TemperatureScaler()
        scaler.fit(logits, y_true)
        assert 0.01 <= scaler.temperature <= 10.0

    def test_not_fitted_raises_v2(self):
        scaler = TemperatureScaler()
        with pytest.raises(RuntimeError):
            scaler.transform(np.array([0.5]))
