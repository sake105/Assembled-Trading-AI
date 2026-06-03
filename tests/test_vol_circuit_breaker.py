"""Tests for risk/circuit_breaker.VolCircuitBreaker (Sprint 4 / Plan C28)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest

pytest.importorskip("src.assembled_core.risk.circuit_breaker")
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


# --- B-risk-3: cooldown PIT-correctness (is_tripped_at as_of) ---

from datetime import datetime, timedelta, timezone  # noqa: E402

from src.assembled_core.risk.circuit_breaker import CircuitBreaker  # noqa: E402


def test_circuitbreaker_is_tripped_at_uses_as_of_not_wallclock() -> None:
    """Replay: a historical trip is still 'in cooldown' when as_of is just
    after the trip, but 'expired' when wall-clock-now (2026) is used."""
    cb = CircuitBreaker(drop_threshold_pct=3.0, window_minutes=15, cooldown_minutes=30)
    t0 = datetime(2022, 3, 7, 14, 0, tzinfo=timezone.utc)
    cb.observe(price=100.0, timestamp=t0)
    assert cb.observe(price=96.0, timestamp=t0 + timedelta(minutes=1)) is True  # -4%

    # 10 min after the trip in replay time -> still in 30-min cooldown
    as_of_in = t0 + timedelta(minutes=11)
    assert cb.is_tripped_at(as_of_in) is True
    # 45 min after the trip in replay time -> cooldown expired
    as_of_out = t0 + timedelta(minutes=46)
    assert cb.is_tripped_at(as_of_out) is False

    # Wall-clock-now (years later) WOULD wrongly say "expired" — the legacy
    # property still does that, proving why as_of is needed for replay.
    assert cb.is_tripped is False
    # Default (no as_of) == legacy wall-clock behaviour, unchanged.
    assert cb.is_tripped_at(None) is cb.is_tripped


def test_circuitbreaker_is_tripped_at_naive_as_of_treated_utc() -> None:
    cb = CircuitBreaker(drop_threshold_pct=3.0, window_minutes=15, cooldown_minutes=30)
    t0 = datetime(2022, 3, 7, 14, 0, tzinfo=timezone.utc)
    cb.observe(price=100.0, timestamp=t0)
    cb.observe(price=96.0, timestamp=t0 + timedelta(minutes=1))
    naive_in = datetime(2022, 3, 7, 14, 11)  # naive -> treated as UTC
    assert cb.is_tripped_at(naive_in) is True


def test_circuitbreaker_not_tripped_when_never_tripped() -> None:
    cb = CircuitBreaker()
    assert cb.is_tripped_at(datetime(2022, 3, 7, tzinfo=timezone.utc)) is False
    assert cb.is_tripped_at(None) is False


def test_volcircuitbreaker_is_tripped_at_uses_as_of() -> None:
    """VolCircuitBreaker cooldown is PIT-correct via is_tripped_at(as_of)."""
    vcb = VolCircuitBreaker(short_window=5, long_window=60, ratio_threshold=2.0)
    calm = [0.001, -0.001] * 30
    spike = [0.05, -0.06, 0.055, -0.058, 0.052]
    assert vcb.check_returns(calm + spike) is True
    # _tripped_at was set to wall-clock now() inside check_returns; emulate a
    # replay by reading cooldown relative to that trip instant.
    trip_at = vcb._tripped_at  # noqa: SLF001 - test introspection
    assert trip_at is not None
    assert vcb.is_tripped_at(trip_at + timedelta(minutes=5)) is True
    assert vcb.is_tripped_at(trip_at + timedelta(minutes=999)) is False
    # Default path unchanged (just-tripped -> wall-clock still in cooldown)
    assert vcb.is_tripped_at(None) is vcb.is_tripped
