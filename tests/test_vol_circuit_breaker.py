"""Tests for risk/circuit_breaker.VolCircuitBreaker (Sprint 4 / Plan C28)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest; pytest.importorskip("src.assembled_core.risk.circuit_breaker")
from src.assembled_core.risk.circuit_breaker import VolCircuitBreaker  # noqa: E402


def test_insufficient_history_no_trip() -> None:
    vcb = VolCircuitBreaker(short_window=5, long_window=60, ratio_threshold=2.0)
    # only 30 bars < long_window=60
    assert vcb.check_returns([0.001] * 30) is False
    assert vcb.trip_count == 0
    assert vcb.is_tripped is False


def test_calm_regime_does_not_trip() -> None:
    vcb = VolCircuitBreaker(short_window=5, long_window=60, ratio_threshold=2.0)
    # steady low-vol returns → ratio ~ 1.0
    calm = [0.001, -0.001, 0.0012, -0.0011, 0.0009] * 20  # 100 bars
    assert vcb.check_returns(calm) is False
    assert vcb.trip_count == 0
    assert 0.0 < vcb.last_ratio < 2.0


def test_vol_spike_trips() -> None:
    vcb = VolCircuitBreaker(short_window=5, long_window=60, ratio_threshold=2.0)
    # 60 calm bars followed by 5 wild bars
    calm = [0.001, -0.001] * 30  # 60 bars
    spike = [0.05, -0.06, 0.055, -0.058, 0.052]
    series = calm + spike
    assert vcb.check_returns(series) is True
    assert vcb.trip_count == 1
    assert vcb.is_tripped is True
    assert vcb.last_ratio >= 2.0


def test_zero_long_vol_does_not_divide_by_zero() -> None:
    vcb = VolCircuitBreaker(short_window=5, long_window=60, ratio_threshold=2.0)
    flat = [0.0] * 100
    assert vcb.check_returns(flat) is False
    assert vcb.trip_count == 0


def test_reset_clears_state() -> None:
    vcb = VolCircuitBreaker(short_window=5, long_window=60, ratio_threshold=2.0)
    calm = [0.001, -0.001] * 30
    spike = [0.05, -0.06, 0.055, -0.058, 0.052]
    vcb.check_returns(calm + spike)
    assert vcb.is_tripped is True
    vcb.reset()
    assert vcb.is_tripped is False
    assert vcb.last_ratio == 0.0
    # trip_count is historical, not cleared by reset
    assert vcb.trip_count == 1


def test_get_state_returns_expected_keys() -> None:
    vcb = VolCircuitBreaker(short_window=5, long_window=60, ratio_threshold=2.5)
    state = vcb.get_state()
    for key in (
        "is_tripped",
        "trip_count",
        "last_ratio",
        "tripped_at",
        "short_window",
        "long_window",
        "ratio_threshold",
        "cooldown_minutes",
    ):
        assert key in state
    assert state["ratio_threshold"] == 2.5


def test_invalid_windows_are_noop() -> None:
    vcb = VolCircuitBreaker(short_window=1, long_window=60, ratio_threshold=2.0)
    assert vcb.check_returns([0.01] * 100) is False
