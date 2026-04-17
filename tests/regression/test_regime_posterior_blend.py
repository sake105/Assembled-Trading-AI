"""F2 — Regime-posterior blending + EWMA smoothing regression pins."""

from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.phase_depth]

from src.assembled_core.signals.regime.hmm_posterior import (  # noqa: E402
    DEFAULT_HALF_LIFE_DAYS,
    _alpha_for_half_life,
    blend_weights_by_regime_posterior,
    smooth_posterior,
)


def test_blend_matches_weighted_sum() -> None:
    base = {
        "bull": {"mom": 0.6, "qual": 0.4},
        "bear": {"mom": 0.0, "qual": 1.0},
    }
    post = {"bull": 0.75, "bear": 0.25}
    res = blend_weights_by_regime_posterior(post, base)
    assert res.weights["mom"] == pytest.approx(0.75 * 0.6 + 0.25 * 0.0)
    assert res.weights["qual"] == pytest.approx(0.75 * 0.4 + 0.25 * 1.0)


def test_step_vs_smoothed_prevents_whipsaw() -> None:
    prev = {"bull": 1.0, "bear": 0.0}
    new = {"bull": 0.0, "bear": 1.0}  # violent flip
    smoothed = smooth_posterior(new, prev, half_life_days=DEFAULT_HALF_LIFE_DAYS)
    # With half-life 5d, alpha ≈ 0.1294; after one step bull still dominant.
    assert smoothed["bull"] > smoothed["bear"]
    # And still sums to 1.
    assert sum(smoothed.values()) == pytest.approx(1.0)


def test_smoother_converges_to_new_over_time() -> None:
    prev = {"bull": 1.0, "bear": 0.0}
    new = {"bull": 0.0, "bear": 1.0}
    for _ in range(40):
        prev = smooth_posterior(new, prev, half_life_days=5.0)
    assert prev["bear"] > 0.99


def test_posterior_renormalises_on_large_drift() -> None:
    # Sum = 2.0 → must be renormalised back to 1.0 before blending.
    res = blend_weights_by_regime_posterior(
        {"bull": 1.0, "bear": 1.0},
        {"bull": {"f": 1.0}, "bear": {"f": 0.0}},
    )
    assert sum(res.posterior_used.values()) == pytest.approx(1.0)
    assert res.weights["f"] == pytest.approx(0.5)


def test_zero_posterior_raises() -> None:
    with pytest.raises(ValueError):
        blend_weights_by_regime_posterior(
            {"bull": 0.0, "bear": 0.0},
            {"bull": {"f": 1.0}, "bear": {"f": 0.0}},
        )


def test_missing_regime_in_base_is_skipped() -> None:
    # No ``bear`` entry in base_weights → bear posterior contribution is 0.
    res = blend_weights_by_regime_posterior(
        {"bull": 0.3, "bear": 0.7},
        {"bull": {"f": 1.0}},
    )
    assert res.weights["f"] == pytest.approx(0.3)


def test_alpha_half_life_formula() -> None:
    alpha = _alpha_for_half_life(5.0)
    assert alpha == pytest.approx(1.0 - math.exp(math.log(0.5) / 5.0))
    # Edge cases.
    assert _alpha_for_half_life(0.0) == 1.0
    assert _alpha_for_half_life(-3.0) == 1.0
