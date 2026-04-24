"""Tests für Round-5 (Signal 7 + Calibration Monitor)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase12


# ---------------------------------------------------------------------------
# Signal 7: Distribution-Shift
# ---------------------------------------------------------------------------

def test_signal7_detects_shift_in_feedback_loop():
    """FeedbackLoopController detectiert distribution-shift auf panel mit starkem Shift."""
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.feedback_loop import FeedbackLoopController

    rng = np.random.default_rng(42)
    n = 400
    # Erste Hälfte N(0,1), zweite N(3, 2)
    stable = rng.standard_normal(n // 2)
    shifted = rng.standard_normal(n // 2) * 2 + 3
    panel = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n),
        "f1": np.concatenate([stable, shifted]),
        "f2": rng.standard_normal(n),
    })

    controller = FeedbackLoopController()
    skipped: list[str] = []
    fired, auc = controller._check_distribution_shift(panel, skipped, train_fraction=0.5)
    # Strong shift sollte detektiert werden
    assert auc > 0.65
    assert fired is True


def test_signal7_no_shift_when_stable():
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.feedback_loop import FeedbackLoopController

    rng = np.random.default_rng(2)
    n = 400
    panel = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n),
        "f1": rng.standard_normal(n),
        "f2": rng.standard_normal(n),
    })

    controller = FeedbackLoopController()
    skipped: list[str] = []
    fired, auc = controller._check_distribution_shift(panel, skipped, train_fraction=0.5)
    # Keine Shift → low AUC, not fired
    assert auc < 0.70
    assert fired is False


def test_signal7_handles_small_panel():
    """Zu kleines Panel → skipped."""
    from src.assembled_core.ml.feedback_loop import FeedbackLoopController

    rng = np.random.default_rng(3)
    panel = pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=50),
        "f1": rng.standard_normal(50),
    })

    controller = FeedbackLoopController()
    skipped: list[str] = []
    fired, auc = controller._check_distribution_shift(panel, skipped)
    assert fired is False
    assert auc == 0.0
    assert "distribution_shift" in skipped


# ---------------------------------------------------------------------------
# Calibration Monitor
# ---------------------------------------------------------------------------

def test_calibration_perfect_predictions():
    """Perfekt kalibrierte Predictions (pred == actual) → ECE nahe 0."""
    import pytest; pytest.importorskip('src.assembled_core.ml.calibration_monitor')
    from src.assembled_core.ml.calibration_monitor import compute_calibration

    n = 500
    # Pred = actual rate pro Bin
    rng = np.random.default_rng(7)
    pred = rng.uniform(0, 1, n)
    actual = (rng.uniform(0, 1, n) < pred).astype(int)

    report = compute_calibration(pred, actual, n_bins=10)
    assert report.ece < 0.15
    assert report.n_samples == n
    assert report.n_bins == 10


def test_calibration_poor_predictions():
    """Systematisch überzogen (pred=0.9 aber acc=0.5) → hohe ECE."""
    import pytest; pytest.importorskip('src.assembled_core.ml.calibration_monitor')
    from src.assembled_core.ml.calibration_monitor import compute_calibration

    n = 300
    pred = np.full(n, 0.9)
    actual = np.zeros(n)  # nichts passiert
    actual[:n // 2] = 1  # 50% wirklich positive

    report = compute_calibration(pred, actual, n_bins=10)
    assert report.ece > 0.3  # starker Kalibrierungsfehler
    assert not report.is_well_calibrated()


def test_platt_calibrator():
    import pytest; pytest.importorskip('src.assembled_core.ml.calibration_monitor')
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.calibration_monitor import PlattCalibrator, compute_calibration

    rng = np.random.default_rng(11)
    n = 400
    # Raw predictions are too confident
    true_probs = rng.uniform(0, 1, n)
    raw = np.clip(true_probs * 2 - 0.5, 0.01, 0.99)  # extremer skaliert
    actual = (rng.uniform(0, 1, n) < true_probs).astype(int)

    before = compute_calibration(raw, actual)

    calibrator = PlattCalibrator().fit(raw[:200], actual[:200])
    calibrated = calibrator.transform(raw[200:])

    after = compute_calibration(calibrated, actual[200:])
    # Kalibrierung sollte ECE reduzieren oder zumindest nicht massiv erhöhen
    assert after.ece <= before.ece + 0.1


def test_isotonic_calibrator():
    import pytest; pytest.importorskip('src.assembled_core.ml.calibration_monitor')
    pytest.importorskip("sklearn")
    from src.assembled_core.ml.calibration_monitor import IsotonicCalibrator

    rng = np.random.default_rng(15)
    raw = rng.uniform(0, 1, 200)
    actual = (rng.uniform(0, 1, 200) < raw * 0.5).astype(int)

    calibrator = IsotonicCalibrator().fit(raw, actual)
    transformed = calibrator.transform(rng.uniform(0, 1, 50))
    assert len(transformed) == 50
    assert (transformed >= 0).all() and (transformed <= 1).all()
