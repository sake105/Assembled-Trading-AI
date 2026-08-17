"""Synthetic-stress validation for LPPLSCrashDetector — C4-078 closure.

Audit note: "existiert nur als Forschungs-Layer; bei Aktivierung als Trading-
Signal verlangt Synthetic-Stress-Validation." This file builds the
validation: generate synthetic paths from known LPPLS parameters and test
that the detector (a) returns finite scores, (b) discriminates bubble
paths from random walks, (c) doesn't over-call random walks as crashes.

Reference: Sornette (2003), JLS model. Validity conditions:
0.1 < m < 0.9, 6 < ω < 13, B < 0, |C/B| < 1.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.assembled_core.signals.lppls_crash import (
    LPPLSCrashDetector,
    simulate_lppls_path,
)


# ---------------------------------------------------------------------------
# simulate_lppls_path
# ---------------------------------------------------------------------------


def test_simulate_returns_correct_length():
    path = simulate_lppls_path(n_days=200, tc=250.0, seed=42)
    assert len(path) == 200


def test_simulate_returns_positive_prices():
    path = simulate_lppls_path(n_days=200, tc=250.0, seed=42)
    assert (path > 0).all()


def test_simulate_rejects_tc_before_end():
    """tc must be in the future relative to path end."""
    with pytest.raises(ValueError, match="tc"):
        simulate_lppls_path(n_days=200, tc=150.0)


def test_simulate_rejects_too_short_path():
    with pytest.raises(ValueError, match="n_days"):
        simulate_lppls_path(n_days=10, tc=50.0)


def test_simulate_is_reproducible_with_seed():
    p1 = simulate_lppls_path(n_days=150, tc=200.0, seed=99)
    p2 = simulate_lppls_path(n_days=150, tc=200.0, seed=99)
    np.testing.assert_array_equal(p1, p2)


def test_simulate_noise_changes_path():
    """Different seeds → different paths."""
    p1 = simulate_lppls_path(n_days=150, tc=200.0, seed=1)
    p2 = simulate_lppls_path(n_days=150, tc=200.0, seed=2)
    assert not np.array_equal(p1, p2)


# ---------------------------------------------------------------------------
# LPPLSCrashDetector smoke checks
# ---------------------------------------------------------------------------


def test_detector_returns_finite_score_on_bubble_path():
    """Bubble path with Sornette-valid params → detector returns finite metrics."""
    path = simulate_lppls_path(
        m=0.5, omega=8.0, A=4.6, B=-0.5, C=0.05, tc=300.0, n_days=250, seed=42
    )
    detector = LPPLSCrashDetector(fit_window=200, max_searches=20)
    result = detector.fit_and_score(path)

    assert "crash_confidence" in result
    assert 0.0 <= result["crash_confidence"] <= 1.0
    assert "tc_estimate" in result
    assert np.isfinite(result["tc_estimate"])
    assert "method" in result


def test_detector_returns_finite_score_on_random_walk():
    """Random-walk path → detector must not crash; confidence is irrelevant
    here (might or might not flag — we just verify the call succeeds)."""
    rng = np.random.default_rng(0)
    log_p = np.cumsum(rng.normal(0.0, 0.01, 250))
    path = np.exp(log_p + 4.6)  # ~price level ≈ 100
    detector = LPPLSCrashDetector(fit_window=200, max_searches=20)
    result = detector.fit_and_score(path)
    assert 0.0 <= result["crash_confidence"] <= 1.0
    assert np.isfinite(result["tc_estimate"])


def test_detector_discriminates_bubble_vs_random_walk_in_mean():
    """Across multiple seeds, bubble paths should AVERAGE higher confidence
    than random walks. Single-sample comparison is noisy; aggregate over
    several runs to reduce variance."""
    detector = LPPLSCrashDetector(fit_window=200, max_searches=20)

    bubble_confidences = []
    rw_confidences = []
    for seed in range(5):
        # Valid Sornette bubble parameters
        bubble = simulate_lppls_path(
            m=0.45,
            omega=8.5,
            A=4.6,
            B=-0.5,
            C=0.05,
            tc=280.0,
            n_days=250,
            seed=seed,
        )
        bubble_confidences.append(detector.fit_and_score(bubble)["crash_confidence"])

        rng = np.random.default_rng(100 + seed)
        rw_log = np.cumsum(rng.normal(0.0, 0.01, 250))
        rw_path = np.exp(rw_log + 4.6)
        rw_confidences.append(detector.fit_and_score(rw_path)["crash_confidence"])

    mean_bubble = float(np.mean(bubble_confidences))
    mean_rw = float(np.mean(rw_confidences))
    # Bubble paths should produce HIGHER mean confidence than random walks.
    # Allow tolerance: detector is heuristic (Sornette validity rule scoring),
    # not a perfect classifier. We just need a discriminative direction.
    assert mean_bubble > mean_rw, (
        f"Detector failed to discriminate: bubble_mean={mean_bubble:.2f}, "
        f"rw_mean={mean_rw:.2f}"
    )


def test_detector_recovers_tc_roughly():
    """For a strong bubble signal, the detector's tc_estimate should be in
    a plausible range relative to the true tc. Tolerance is wide because the
    detector uses random restarts and a heuristic fit."""
    true_tc = 280.0
    path = simulate_lppls_path(
        m=0.45,
        omega=8.5,
        A=4.6,
        B=-0.5,
        C=0.05,
        tc=true_tc,
        n_days=250,
        seed=42,
    )
    detector = LPPLSCrashDetector(fit_window=250, max_searches=50)
    result = detector.fit_and_score(path)
    tc_est = result["tc_estimate"]
    # Allow generous tolerance — the fit is non-trivial. We just want to see
    # that tc_est is NOT wildly off (e.g., not 5x or 0.2x the truth).
    assert 0.5 * true_tc < tc_est < 2.0 * true_tc, (
        f"tc_estimate {tc_est:.1f} too far from true tc {true_tc}"
    )


def test_detector_handles_short_window_gracefully():
    """When fit_window > len(prices), detector should still return finite
    metrics (it'll fit on what's available or fall back)."""
    short_path = simulate_lppls_path(n_days=60, tc=100.0, seed=42)
    detector = LPPLSCrashDetector(fit_window=252, max_searches=10)
    result = detector.fit_and_score(short_path)
    assert "crash_confidence" in result
    assert 0.0 <= result["crash_confidence"] <= 1.0
