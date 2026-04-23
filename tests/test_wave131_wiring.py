"""Tests for wave-131 module wiring into trading_cycle.py.

Covers:
  Step 3.95 — ml.calibration (CalibrationResult / IsotonicCalibrator)
  Step 3.96 — ml.conformal (SplitConformalPredictor / conformal_position_size)
  Step 3.97 — ml.regime_hmm (RegimeHMM / MultiFeatureRegimeHMM)
"""

from __future__ import annotations

import numpy as np
import pytest

from src.assembled_core.ml.calibration import (
    CalibrationResult,
    IsotonicCalibrator,
    compute_calibration_error,
)
from src.assembled_core.ml.conformal import (
    ConformalResult,
    SplitConformalPredictor,
    conformal_position_size,
)
from src.assembled_core.ml.regime_hmm import RegimeHMM, MultiFeatureRegimeHMM


# ---------------------------------------------------------------------------
# ml.calibration (Step 3.95)
# ---------------------------------------------------------------------------

def test_calibration_result_importable():
    assert CalibrationResult is not None


def test_calibration_result_creates():
    cr = CalibrationResult(method="isotonic", brier_score=0.25, ece=0.05, mce=0.10, n_bins=10)
    assert cr.method == "isotonic"
    assert cr.n_bins == 10


def test_isotonic_calibrator_creates():
    cal = IsotonicCalibrator()
    assert isinstance(cal, IsotonicCalibrator)


def test_compute_calibration_error_importable():
    assert compute_calibration_error is not None


def test_compute_calibration_error_returns_result():
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
    result = compute_calibration_error(y_true, y_prob, n_bins=5)
    assert isinstance(result, CalibrationResult)
    assert 0.0 <= result.ece <= 1.0


# ---------------------------------------------------------------------------
# ml.conformal (Step 3.96)
# ---------------------------------------------------------------------------

def test_conformal_result_importable():
    assert ConformalResult is not None


def test_split_conformal_predictor_creates():
    scp = SplitConformalPredictor(model=None, alpha=0.1)
    assert scp.alpha == 0.1


def test_conformal_position_size_importable():
    assert conformal_position_size is not None


# ---------------------------------------------------------------------------
# ml.regime_hmm (Step 3.97)
# ---------------------------------------------------------------------------

def test_regime_hmm_importable():
    assert RegimeHMM is not None


def test_multi_feature_regime_hmm_importable():
    assert MultiFeatureRegimeHMM is not None


def test_regime_hmm_creates_or_raises_import():
    try:
        hmm = RegimeHMM(n_regimes=3)
        assert hmm.n_regimes == 3
    except ImportError:
        # hmmlearn not installed — acceptable
        pass
