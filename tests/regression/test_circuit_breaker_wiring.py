"""Tier-1 wiring — verify risk.circuit_breaker is consumed by the trading
cycle through ``_evaluate_circuit_breaker``.

The gate is policy-flag-guarded (``policy.risk.circuit_breaker.enabled``)
and defaults to OFF. When ON and a sequence of intraday equity/benchmark
observations shows a window drop > threshold, a decision dict is returned.
"""

from __future__ import annotations

import types
from datetime import datetime, timedelta, timezone

import pytest

pytestmark = [pytest.mark.phase12]

from src.assembled_core.pipeline.trading_cycle import (  # noqa: E402
    _evaluate_circuit_breaker,
)


def _crash_observations() -> list[dict]:
    base = datetime(2026, 4, 18, 14, 30, tzinfo=timezone.utc)
    # 5% drop within 15min window → well above the default 3% trigger.
    return [
        {"timestamp": base, "price": 100.0},
        {"timestamp": base + timedelta(minutes=2), "price": 99.5},
        {"timestamp": base + timedelta(minutes=5), "price": 98.0},
        {"timestamp": base + timedelta(minutes=10), "price": 94.5},
    ]


def _quiet_observations() -> list[dict]:
    base = datetime(2026, 4, 18, 14, 30, tzinfo=timezone.utc)
    return [
        {"timestamp": base + timedelta(minutes=i), "price": 100.0 + 0.02 * i}
        for i in range(10)
    ]


def _make_ctx_result(obs: list[dict]) -> tuple:
    ctx = types.SimpleNamespace(intraday_equity_observations=None)
    result = types.SimpleNamespace(meta={"intraday_equity_observations": obs})
    return ctx, result


def test_circuit_breaker_disabled_returns_none() -> None:
    ctx, result = _make_ctx_result(_crash_observations())
    assert _evaluate_circuit_breaker(ctx, result, policy={}) is None


def test_circuit_breaker_enabled_quiet_session_passes() -> None:
    ctx, result = _make_ctx_result(_quiet_observations())
    decision = _evaluate_circuit_breaker(
        ctx,
        result,
        policy={
            "risk": {
                "circuit_breaker": {
                    "enabled": True,
                    "drop_threshold_pct": 3.0,
                    "window_minutes": 15,
                }
            }
        },
    )
    assert decision is None


def test_circuit_breaker_enabled_crash_trips() -> None:
    ctx, result = _make_ctx_result(_crash_observations())
    decision = _evaluate_circuit_breaker(
        ctx,
        result,
        policy={
            "risk": {
                "circuit_breaker": {
                    "enabled": True,
                    "drop_threshold_pct": 3.0,
                    "window_minutes": 15,
                }
            }
        },
    )
    assert decision is not None
    assert decision["breach"] is True
    assert decision["reason"] == "intraday_circuit_breaker_trip"
    assert decision["tripped_on"]["trip_count"] >= 1
