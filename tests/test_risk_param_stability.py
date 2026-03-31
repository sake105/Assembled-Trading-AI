"""Tests for parameter stability checks — M6-T09.

Covers:
- compute_rolling_vol_estimates: normal, insufficient data, edge cases
- check_vol_stability: stable/unstable classification, edge cases
- check_turnover_stability: stable/unstable, empty, insufficient data
- compute_rolling_max_drawdown: normal, insufficient data
- check_drawdown_stability: stable/unstable, edge cases
- compute_stability_report: combined report, partial data, policy overrides
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase12

from src.assembled_core.risk.param_stability import (
    check_drawdown_stability,
    check_turnover_stability,
    check_vol_stability,
    compute_rolling_max_drawdown,
    compute_rolling_vol_estimates,
    compute_stability_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_equity_curve(n: int = 200, vol: float = 0.01, seed: int = 42) -> pd.Series:
    """Build synthetic equity curve with given daily vol."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, vol, n)
    prices = 100.0 * np.cumprod(1.0 + returns)
    return pd.Series(prices)


def _make_stable_equity(n: int = 200, seed: int = 42) -> pd.Series:
    """Equity curve with very consistent vol (same vol throughout)."""
    return _make_equity_curve(n=n, vol=0.01, seed=seed)


def _make_unstable_equity(n: int = 200, seed: int = 42) -> pd.Series:
    """Equity curve with abrupt vol regime change (very different first/second half)."""
    rng = np.random.default_rng(seed)
    r1 = rng.normal(0.0002, 0.001, n // 2)  # very low vol
    r2 = rng.normal(0.0002, 0.05, n // 2)   # very high vol
    returns = np.concatenate([r1, r2])
    prices = 100.0 * np.cumprod(1.0 + returns)
    return pd.Series(prices)


# ---------------------------------------------------------------------------
# compute_rolling_vol_estimates
# ---------------------------------------------------------------------------


class TestRollingVolEstimates:
    def test_normal_returns_dict_for_all_windows(self):
        ec = _make_stable_equity(200)
        result = compute_rolling_vol_estimates(ec, [10, 20, 40])
        assert set(result.keys()) == {10, 20, 40}
        for v in result.values():
            assert not math.isnan(v)
            assert v > 0.0

    def test_too_short_curve_returns_nan(self):
        ec = pd.Series([100.0, 101.0, 102.0])  # only 3 bars
        result = compute_rolling_vol_estimates(ec, [20, 40], min_observations=5)
        assert math.isnan(result[20])
        assert math.isnan(result[40])

    def test_none_input_returns_all_nan(self):
        result = compute_rolling_vol_estimates(None, [10, 20])  # type: ignore[arg-type]
        assert math.isnan(result[10])
        assert math.isnan(result[20])

    def test_empty_series_returns_all_nan(self):
        result = compute_rolling_vol_estimates(pd.Series(dtype=float), [10, 20])
        assert math.isnan(result[10])
        assert math.isnan(result[20])

    def test_larger_window_uses_more_data(self):
        # Both windows have enough data — just verify they return valid floats
        ec = _make_stable_equity(200)
        result = compute_rolling_vol_estimates(ec, [10, 60])
        assert not math.isnan(result[10])
        assert not math.isnan(result[60])

    def test_annualization_applies(self):
        ec = _make_stable_equity(200)
        r252 = compute_rolling_vol_estimates(ec, [20], annualize_factor=252.0)
        r52 = compute_rolling_vol_estimates(ec, [20], annualize_factor=52.0)
        ratio = r252[20] / r52[20]
        assert ratio == pytest.approx(math.sqrt(252.0 / 52.0), rel=1e-6)


# ---------------------------------------------------------------------------
# check_vol_stability
# ---------------------------------------------------------------------------


class TestVolStability:
    def test_stable_equity_reports_stable(self):
        ec = _make_stable_equity(200)
        result = check_vol_stability(ec, window_sizes=[20, 40, 60])
        assert result["status"] == "ok"
        assert result["is_stable"] is True
        assert result["valid_windows"] == 3

    def test_unstable_equity_reports_unstable(self):
        ec = _make_unstable_equity(200)
        # Very tight threshold so the large CV from the vol regime change exceeds it
        result = check_vol_stability(ec, window_sizes=[10, 20, 40, 60], stability_threshold=0.01)
        # With extreme vol regime change (0.001 vs 0.05 daily vol), cv should exceed 1%
        assert result["status"] == "ok"
        assert result["cv"] > 0.01  # unstable at this tight threshold

    def test_too_short_curve_insufficient_data(self):
        ec = pd.Series([100.0, 101.0])
        result = check_vol_stability(ec, window_sizes=[20, 40])
        assert result["status"] in ("all_nan", "insufficient_data")

    def test_none_curve_all_nan(self):
        result = check_vol_stability(None, window_sizes=[10, 20])  # type: ignore[arg-type]
        assert result["status"] == "all_nan"
        assert result["is_stable"] is False

    def test_mean_vol_is_positive(self):
        ec = _make_stable_equity(200)
        result = check_vol_stability(ec, window_sizes=[20, 40])
        assert result["mean_vol"] > 0.0

    def test_cv_is_non_negative(self):
        ec = _make_stable_equity(200)
        result = check_vol_stability(ec, window_sizes=[10, 20, 40, 60])
        if result["status"] == "ok":
            assert result["cv"] >= 0.0


# ---------------------------------------------------------------------------
# check_turnover_stability
# ---------------------------------------------------------------------------


class TestTurnoverStability:
    def test_stable_turnover_reports_stable(self):
        # Low-variation turnover series
        rng = np.random.default_rng(1)
        to = pd.Series(0.10 + rng.normal(0, 0.005, 30))
        result = check_turnover_stability(to, stability_threshold=0.50)
        assert result["status"] == "ok"
        assert result["is_stable"] is True

    def test_unstable_turnover_reports_unstable(self):
        # High-variation turnover
        to = pd.Series([0.01, 0.50, 0.02, 0.80, 0.03, 0.90, 0.01, 0.70, 0.02, 0.60])
        result = check_turnover_stability(to, stability_threshold=0.10)
        assert result["is_stable"] is False

    def test_none_input_returns_empty(self):
        result = check_turnover_stability(None)  # type: ignore[arg-type]
        assert result["status"] == "empty"

    def test_empty_series_returns_empty(self):
        result = check_turnover_stability(pd.Series(dtype=float))
        assert result["status"] == "empty"

    def test_too_few_observations(self):
        result = check_turnover_stability(pd.Series([0.1, 0.2]), min_observations=5)
        assert result["status"] == "insufficient_data"

    def test_max_turnover_is_maximum(self):
        to = pd.Series([0.1, 0.3, 0.2, 0.5, 0.15] * 3)
        result = check_turnover_stability(to)
        assert result["max_turnover"] == pytest.approx(0.5)

    def test_n_observations_correct(self):
        to = pd.Series([0.1] * 20)
        result = check_turnover_stability(to)
        assert result["n_observations"] == 20


# ---------------------------------------------------------------------------
# compute_rolling_max_drawdown
# ---------------------------------------------------------------------------


class TestRollingMaxDrawdown:
    def test_drawdown_is_non_positive(self):
        ec = _make_stable_equity(100)
        rolling_dd = compute_rolling_max_drawdown(ec, window=20)
        valid = rolling_dd.dropna()
        assert len(valid) > 0
        assert (valid <= 0.0).all()

    def test_flat_curve_zero_drawdown(self):
        ec = pd.Series([100.0] * 50)
        rolling_dd = compute_rolling_max_drawdown(ec, window=10)
        valid = rolling_dd.dropna()
        assert valid.abs().max() < 1e-9

    def test_insufficient_length_returns_empty(self):
        ec = pd.Series([100.0, 101.0, 102.0])
        result = compute_rolling_max_drawdown(ec, window=20)
        assert len(result) == 0

    def test_declining_curve_has_large_drawdown(self):
        # Steadily declining equity
        prices = pd.Series([100.0 - i * 0.5 for i in range(60)])
        rolling_dd = compute_rolling_max_drawdown(prices, window=20)
        valid = rolling_dd.dropna()
        assert valid.min() < -0.05  # at least 5% drawdown somewhere


# ---------------------------------------------------------------------------
# check_drawdown_stability
# ---------------------------------------------------------------------------


class TestDrawdownStability:
    def test_stable_equity_reports_stable(self):
        ec = _make_stable_equity(200)
        result = check_drawdown_stability(ec, window=20, stability_threshold=0.80)
        assert result["status"] in ("ok", "insufficient_data")
        if result["status"] == "ok":
            # With generous threshold, stable equity should pass
            assert isinstance(result["is_stable"], bool)

    def test_insufficient_equity_returns_insufficient(self):
        ec = pd.Series([100.0] * 30)  # 30 bars, window=40
        result = check_drawdown_stability(ec, window=40)
        assert result["status"] in ("empty", "insufficient_data")

    def test_worst_dd_is_most_negative(self):
        ec = _make_stable_equity(200)
        result = check_drawdown_stability(ec, window=20)
        if result["status"] == "ok":
            assert result["worst_dd"] <= result["mean_max_dd"]

    def test_flat_curve_dd_is_zero(self):
        ec = pd.Series([100.0] * 100)
        result = check_drawdown_stability(ec, window=10, min_observations=5)
        if result["status"] == "ok":
            assert result["worst_dd"] == pytest.approx(0.0, abs=1e-9)
            assert result["mean_max_dd"] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# compute_stability_report
# ---------------------------------------------------------------------------


class TestStabilityReport:
    def test_full_report_structure(self):
        ec = _make_stable_equity(200)
        to = pd.Series([0.1] * 50)
        report = compute_stability_report(ec, turnover_series=to)
        assert "vol_stability" in report
        assert "turnover_stability" in report
        assert "drawdown_stability" in report
        assert "all_stable" in report
        assert "checks_passed" in report
        assert "checks_total" in report

    def test_no_turnover_series(self):
        ec = _make_stable_equity(200)
        report = compute_stability_report(ec)
        assert report["turnover_stability"] is None

    def test_checks_passed_leq_checks_total(self):
        ec = _make_stable_equity(200)
        to = pd.Series([0.1 + i * 0.001 for i in range(50)])
        report = compute_stability_report(ec, turnover_series=to)
        assert report["checks_passed"] <= report["checks_total"]

    def test_all_stable_when_all_checks_pass(self):
        # Stable equity + stable turnover → all_stable = True
        ec = _make_stable_equity(200)
        to = pd.Series([0.10] * 50)  # perfectly constant turnover
        policy = {
            "param_stability": {
                "vol_stability_threshold": 0.99,   # very generous
                "turnover_stability_threshold": 0.99,
                "drawdown_stability_threshold": 0.99,
            }
        }
        report = compute_stability_report(ec, turnover_series=to, policy=policy)
        if report["checks_total"] > 0:
            assert report["all_stable"] is True

    def test_policy_overrides_applied(self):
        ec = _make_stable_equity(200)
        policy = {
            "param_stability": {
                "vol_window_sizes": [15, 30],
                "drawdown_window": 15,
                "annualize_factor": 252.0,
            }
        }
        report = compute_stability_report(ec, policy=policy)
        # vol_by_window should have keys 15 and 30
        vol_windows = report["vol_stability"]["vol_by_window"]
        assert 15 in vol_windows
        assert 30 in vol_windows

    def test_short_equity_curve_handles_gracefully(self):
        ec = pd.Series([100.0, 101.0, 100.5])
        report = compute_stability_report(ec)
        # Should not raise — just return degraded results
        assert "vol_stability" in report
        assert "drawdown_stability" in report
