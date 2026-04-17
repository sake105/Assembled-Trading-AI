"""F1 — IC-decay-weighted factor combination regression pins."""

from __future__ import annotations

import math

import pytest

pytestmark = [pytest.mark.phase_depth]

from src.assembled_core.strategies.ic_decay_weights import (  # noqa: E402
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_MAX_W_PER_FACTOR,
    compute_ic_decay_weights,
)


def test_equal_positive_ic_gives_equal_weights() -> None:
    res = compute_ic_decay_weights({"a": 0.05, "b": 0.05, "c": 0.05})
    assert res.fallback_used is False
    assert pytest.approx(sum(res.weights.values()), rel=1e-9) == 1.0
    assert res.weights["a"] == pytest.approx(res.weights["b"])
    assert res.weights["a"] == pytest.approx(res.weights["c"])


def test_negative_ic_gets_zero_before_normalisation() -> None:
    res = compute_ic_decay_weights({"good": 0.04, "bad": -0.1})
    assert res.weights.get("bad", 0.0) == 0.0
    assert res.weights["good"] == pytest.approx(1.0)
    assert res.raw_weights["bad"] == 0.0


def test_lag_reduces_weight_via_exp_decay() -> None:
    snapshot = {"a": 0.04, "b": 0.04}
    res = compute_ic_decay_weights(
        snapshot,
        lags={"a": 0.0, "b": 30.0},
        half_lives={"a": 30.0, "b": 30.0},
    )
    # a: raw = 0.04 * exp(0) = 0.04; b: raw = 0.04 * exp(-1) ≈ 0.01472.
    assert res.weights["a"] > res.weights["b"]
    raw_a = 0.04
    raw_b = 0.04 * math.exp(-1.0)
    total = raw_a + raw_b
    assert res.weights["a"] == pytest.approx(raw_a / total, rel=1e-6)
    assert res.weights["b"] == pytest.approx(raw_b / total, rel=1e-6)


def test_cap_is_respected() -> None:
    snapshot = {"big": 5.0, "small": 0.01}
    res = compute_ic_decay_weights(snapshot, max_w_per_factor=0.25)
    # `big` clipped to cap; `small` keeps raw value.
    assert res.weights["big"] <= 1.0
    # After cap: big=0.25 raw, small=0.01 raw → big normalises to 0.25/0.26.
    expected_big = 0.25 / (0.25 + 0.01)
    assert res.weights["big"] == pytest.approx(expected_big, rel=1e-6)


def test_all_non_positive_uses_fallback() -> None:
    res = compute_ic_decay_weights(
        {"a": -0.01, "b": 0.0, "c": float("nan")},
        fallback_weights={"a": 0.5, "b": 0.3, "c": 0.2},
    )
    assert res.fallback_used is True
    assert res.weights == {"a": 0.5, "b": 0.3, "c": 0.2}


def test_missing_half_life_defaults_to_30d() -> None:
    snapshot = {"x": 0.04}
    res_default = compute_ic_decay_weights(snapshot, lags={"x": DEFAULT_HALF_LIFE_DAYS})
    # With lag == default half-life → raw = IC * exp(-1) ≈ 0.0147
    expected_raw = 0.04 * math.exp(-1.0)
    assert res_default.raw_weights["x"] == pytest.approx(expected_raw, rel=1e-6)
    # Single factor → always 1.0 after normalisation.
    assert res_default.weights["x"] == pytest.approx(1.0)


def test_default_cap_constant_is_25pct() -> None:
    assert DEFAULT_MAX_W_PER_FACTOR == 0.25
