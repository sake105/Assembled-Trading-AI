"""Tests for vol targeting — M6-T01/T02.

Covers:
- compute_realized_vol: edge cases, min_observations, annualization
- compute_vol_scale_factor: clamping, nan/zero inputs, normal case
- apply_vol_targeting_to_weights: scaling, empty inputs
- compute_vol_targeting_result: disabled, insufficient data, high/low vol
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

pytestmark = pytest.mark.phase12

from src.assembled_core.risk.vol_targeting import (
    apply_vol_targeting_to_weights,
    compute_realized_vol,
    compute_vol_scale_factor,
    compute_vol_targeting_result,
)


# ---------------------------------------------------------------------------
# compute_realized_vol
# ---------------------------------------------------------------------------


class TestComputeRealizedVol:
    def test_none_input_returns_nan(self):
        result = compute_realized_vol(None)  # type: ignore[arg-type]
        assert math.isnan(result)

    def test_empty_series_returns_nan(self):
        result = compute_realized_vol(pd.Series([], dtype=float))
        assert math.isnan(result)

    def test_too_few_observations_returns_nan(self):
        # Only 3 observations, min_observations=5
        returns = pd.Series([0.01, -0.01, 0.02])
        result = compute_realized_vol(returns, min_observations=5)
        assert math.isnan(result)

    def test_sufficient_observations_returns_float(self):
        returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.015, 0.003])
        result = compute_realized_vol(returns, lookback_days=6, min_observations=5)
        assert isinstance(result, float)
        assert result > 0.0

    def test_constant_returns_vol_is_zero(self):
        # Constant returns have zero std
        returns = pd.Series([0.01] * 20)
        result = compute_realized_vol(returns, lookback_days=20, min_observations=5)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_annualization_uses_factor(self):
        # Daily std = 0.01; annualized = 0.01 * sqrt(252)
        returns = pd.Series([0.01, -0.01] * 15)  # 30 observations
        daily_vol = compute_realized_vol(
            returns, lookback_days=30, annualize_factor=1.0
        )
        annual_vol = compute_realized_vol(
            returns, lookback_days=30, annualize_factor=252.0
        )
        assert annual_vol == pytest.approx(daily_vol * (252.0**0.5), rel=1e-6)

    def test_uses_only_lookback_tail(self):
        # First 10 elements are high vol, last 10 are low vol
        high_vol_part = pd.Series([0.10, -0.10] * 5)
        low_vol_part = pd.Series([0.001, -0.001] * 5)
        returns = pd.concat([high_vol_part, low_vol_part], ignore_index=True)

        vol_all = compute_realized_vol(returns, lookback_days=20, min_observations=5)
        vol_recent = compute_realized_vol(returns, lookback_days=10, min_observations=5)
        # vol_recent should be lower (using only quiet tail)
        assert vol_recent < vol_all

    def test_nan_values_dropped(self):
        returns = pd.Series([float("nan"), 0.01, -0.01, 0.02, -0.01, 0.015])
        result = compute_realized_vol(returns, lookback_days=10, min_observations=5)
        assert not math.isnan(result)
        assert result > 0.0


# ---------------------------------------------------------------------------
# compute_vol_scale_factor
# ---------------------------------------------------------------------------


class TestComputeVolScaleFactor:
    def test_nan_realized_vol_returns_one(self):
        result = compute_vol_scale_factor(float("nan"), target_vol=0.20)
        assert result == pytest.approx(1.0)

    def test_zero_realized_vol_returns_one(self):
        result = compute_vol_scale_factor(0.0, target_vol=0.20)
        assert result == pytest.approx(1.0)

    def test_zero_target_vol_returns_one(self):
        result = compute_vol_scale_factor(0.15, target_vol=0.0)
        assert result == pytest.approx(1.0)

    def test_normal_scale_down(self):
        # realized=0.30, target=0.20 → scale = 20/30 ≈ 0.667
        result = compute_vol_scale_factor(0.30, target_vol=0.20)
        assert result == pytest.approx(0.20 / 0.30, rel=1e-6)

    def test_normal_scale_up(self):
        # realized=0.10, target=0.20 → scale = 20/10 = 2.0, capped at max_scale=1.5
        result = compute_vol_scale_factor(0.10, target_vol=0.20, max_scale=1.5)
        assert result == pytest.approx(1.5)

    def test_clamped_to_min_scale(self):
        # realized=1.0, target=0.05 → raw=0.05, clamped at min_scale=0.10
        result = compute_vol_scale_factor(1.0, target_vol=0.05, min_scale=0.10)
        assert result == pytest.approx(0.10)

    def test_exactly_at_target_returns_one(self):
        result = compute_vol_scale_factor(0.20, target_vol=0.20)
        assert result == pytest.approx(1.0)

    def test_custom_max_scale(self):
        result = compute_vol_scale_factor(0.05, target_vol=0.20, max_scale=3.0)
        assert result == pytest.approx(min(3.0, 0.20 / 0.05))


# ---------------------------------------------------------------------------
# apply_vol_targeting_to_weights
# ---------------------------------------------------------------------------


class TestApplyVolTargetingToWeights:
    def test_empty_dict_returns_empty(self):
        result = apply_vol_targeting_to_weights({}, scale_factor=0.8)
        assert result == {}

    def test_scale_factor_one_unchanged(self):
        weights = {"A": 0.3, "B": 0.2}
        result = apply_vol_targeting_to_weights(weights, scale_factor=1.0)
        assert result["A"] == pytest.approx(0.3)
        assert result["B"] == pytest.approx(0.2)

    def test_scale_down(self):
        weights = {"GLD": 0.20, "TLT": 0.15}
        result = apply_vol_targeting_to_weights(weights, scale_factor=0.5)
        assert result["GLD"] == pytest.approx(0.10)
        assert result["TLT"] == pytest.approx(0.075)

    def test_does_not_mutate_input(self):
        weights = {"X": 0.25}
        original_copy = dict(weights)
        apply_vol_targeting_to_weights(weights, scale_factor=0.7)
        assert weights == original_copy

    def test_all_symbols_scaled(self):
        weights = {"A": 0.1, "B": 0.2, "C": 0.3}
        result = apply_vol_targeting_to_weights(weights, scale_factor=2.0)
        for sym in weights:
            assert result[sym] == pytest.approx(weights[sym] * 2.0)


# ---------------------------------------------------------------------------
# compute_vol_targeting_result
# ---------------------------------------------------------------------------


def _policy_enabled(
    target_vol: float = 0.20,
    lookback_days: int = 20,
    min_scale: float = 0.0,
    max_scale: float = 1.5,
) -> dict:
    return {
        "vol_targeting": {
            "enabled": True,
            "target_vol_annual": target_vol,
            "lookback_days": lookback_days,
            "min_scale": min_scale,
            "max_scale": max_scale,
            "annualize_factor": 252.0,
            "min_observations": 5,
        }
    }


class TestComputeVolTargetingResult:
    def test_disabled_returns_defaults(self):
        curve = pd.Series([1.0, 1.01, 1.02])
        policy = {"vol_targeting": {"enabled": False}}
        scale, realized, target = compute_vol_targeting_result(curve, policy)
        assert scale == pytest.approx(1.0)
        assert math.isnan(realized)
        assert math.isnan(target)

    def test_none_equity_curve_returns_scale_one(self):
        scale, realized, target = compute_vol_targeting_result(None, _policy_enabled())  # type: ignore[arg-type]
        assert scale == pytest.approx(1.0)
        assert math.isnan(realized)

    def test_empty_equity_curve_returns_scale_one(self):
        scale, _, _ = compute_vol_targeting_result(
            pd.Series([], dtype=float), _policy_enabled()
        )
        assert scale == pytest.approx(1.0)

    def test_short_equity_curve_returns_scale_one(self):
        # Only 2 observations, need 5+ after pct_change
        curve = pd.Series([1.0, 1.01])
        scale, _, _ = compute_vol_targeting_result(curve, _policy_enabled())
        assert scale == pytest.approx(1.0)

    def test_high_vol_curve_scales_down(self):
        # Build a high-vol daily equity curve: ±5% each day
        import numpy as np

        rng = [1.0]
        returns = [0.05, -0.05] * 30  # 60 bars, 5% vol
        for r in returns:
            rng.append(rng[-1] * (1 + r))
        curve = pd.Series(rng)
        # target=0.10, realized≈0.05*sqrt(252)≈0.79 → scale ≈ 0.10/0.79 ≈ 0.13
        scale, realized, target = compute_vol_targeting_result(
            curve, _policy_enabled(target_vol=0.10)
        )
        assert scale < 1.0, f"Expected scale < 1.0, got {scale}"
        assert realized > target, "Realized vol should exceed target"

    def test_low_vol_curve_scales_up_capped(self):
        # Very flat equity curve → very low realized vol → scale hits max_scale
        rng = [1.0]
        returns = [0.0001, -0.0001] * 30
        for r in returns:
            rng.append(rng[-1] * (1 + r))
        curve = pd.Series(rng)
        scale, realized, _ = compute_vol_targeting_result(
            curve, _policy_enabled(target_vol=0.20, max_scale=1.5)
        )
        assert scale == pytest.approx(1.5)

    def test_now_idx_limits_history(self):
        # Equity curve with 50 bars; set now_idx=10 → uses only first 11 bars
        rng = [1.0]
        for i in range(50):
            rng.append(rng[-1] * (1 + 0.01 * (-1 if i % 2 else 1)))
        curve = pd.Series(rng)
        # Using all history
        scale_all, _, _ = compute_vol_targeting_result(
            curve, _policy_enabled(lookback_days=20)
        )
        # Using only first 11 bars (now_idx=10) → only 10 returns, < min_observations=5
        # so it should still succeed (10 >= 5)
        scale_early, _, _ = compute_vol_targeting_result(
            curve, _policy_enabled(lookback_days=10), now_idx=10
        )
        # Both should be valid floats (not nan)
        assert not math.isnan(scale_all)
        assert not math.isnan(scale_early)
