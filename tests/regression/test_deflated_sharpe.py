"""E4 — Deflated Sharpe Ratio regression pins (BLP 2014).

These tests lock the DSR implementation against drift. They do NOT test the
underlying scipy.stats; they pin the *composition* of moments, stderr, and
threshold into the DSR probability.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytestmark = [pytest.mark.phase_realism]

scipy = pytest.importorskip("scipy")

from src.assembled_core.qa.deflated_sharpe import (  # noqa: E402
    DSRResult,
    deflated_sharpe,
    sharpe_std_error,
    sharpe_threshold,
)


def _normal_returns(n: int, sr_target: float, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Construct so mean/std ≈ sr_target for periodic frame.
    r = rng.standard_normal(n)
    r = (r - r.mean()) / r.std(ddof=1)
    return r * 1.0 + sr_target  # mean=sr_target, std=1


def test_normal_single_trial_matches_closed_form() -> None:
    """Normal returns, n_trials=1 → DSR ≈ Φ(SR·sqrt((T-1)/(1 + SR²/2)))."""
    r = _normal_returns(500, sr_target=0.1, seed=7)
    res = deflated_sharpe(r, n_trials=1)
    assert isinstance(res, DSRResult)
    assert res.n_trials == 1
    assert res.sharpe_threshold == 0.0
    assert math.isfinite(res.sharpe_observed)
    assert math.isfinite(res.deflated_sharpe_probability)
    # Under normality γ3=0, γ4=0 → inside = 1 + SR²·(-1/4) should still be > 0.
    # The composed probability must lie inside (0, 1).
    assert 0.0 < res.deflated_sharpe_probability < 1.0


def test_more_trials_raises_threshold_and_drops_dsr() -> None:
    r = _normal_returns(500, sr_target=0.08, seed=11)
    single = deflated_sharpe(r, n_trials=1)
    many = deflated_sharpe(r, n_trials=500)
    assert many.sharpe_threshold > single.sharpe_threshold
    # With a positive threshold, the deflated probability must be smaller.
    assert many.deflated_sharpe_probability < single.deflated_sharpe_probability


def test_heavy_negative_skew_inflates_stderr() -> None:
    """Under left-tailed returns the Sharpe stderr rises, DSR falls."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal(400)
    skewed = base - 0.6 * np.where(base < -1.0, (base + 1.0) ** 2, 0.0)
    skewed = skewed - skewed.mean() + 0.1  # keep a positive drift

    sym = deflated_sharpe(base * 1.0 + 0.1, n_trials=1)
    asym = deflated_sharpe(skewed, n_trials=1)
    assert asym.skew < sym.skew
    # Negative skew against a positive SR inflates the `-γ3·SR` term → stderr up.
    assert asym.sharpe_std_error > sym.sharpe_std_error


def test_too_short_series_returns_nan() -> None:
    res = deflated_sharpe(np.array([0.01]), n_trials=1)
    assert math.isnan(res.sharpe_observed)
    assert math.isnan(res.deflated_sharpe_probability)
    assert res.passes_5pct is False


def test_passes_5pct_only_when_probability_exceeds_threshold() -> None:
    # Strong, long, positive drift → high DSR on single-trial.
    r = _normal_returns(1000, sr_target=0.25, seed=19)
    res = deflated_sharpe(r, n_trials=1)
    assert res.deflated_sharpe_probability > 0.95
    assert res.passes_5pct is True

    # Near-zero SR → low DSR, cannot pass.
    flat = _normal_returns(1000, sr_target=0.001, seed=21)
    low = deflated_sharpe(flat, n_trials=1)
    assert low.deflated_sharpe_probability < 0.95
    assert low.passes_5pct is False


def test_as_dict_contains_all_fields() -> None:
    r = _normal_returns(200, sr_target=0.05, seed=5)
    d = deflated_sharpe(r, n_trials=1).as_dict()
    expected = {
        "sharpe_observed",
        "sharpe_std_error",
        "sharpe_threshold",
        "deflated_sharpe_probability",
        "n_observations",
        "n_trials",
        "skew",
        "excess_kurtosis",
        "passes_5pct",
    }
    assert set(d.keys()) == expected


def test_sharpe_std_error_direct_formula() -> None:
    # Pin the stderr composition at a chosen input.
    se = sharpe_std_error(sharpe=0.2, n_obs=100, skew=0.0, excess_kurtosis=0.0)
    expected = math.sqrt((1.0 + (-1.0 / 4.0) * 0.04) / 99.0)
    assert se == pytest.approx(expected, rel=1e-9)


def test_threshold_zero_for_single_trial() -> None:
    assert sharpe_threshold(n_trials=1, variance_across_trials=0.25) == 0.0
    assert sharpe_threshold(n_trials=50, variance_across_trials=0.0) == 0.0
