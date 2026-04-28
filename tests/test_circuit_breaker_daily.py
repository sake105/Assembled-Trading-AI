"""Tests for the daily circuit-breaker evaluator (Sprint 1 / W4b)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline.trading_cycle_shared import _evaluate_circuit_breaker_daily


def _mk_prices(ref_closes: list[float], symbol: str = "SPY") -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(ref_closes), freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": [symbol] * len(ref_closes),
            "close": ref_closes,
        }
    )


POLICY_ON = {
    "circuit_breaker": {
        "enabled": True,
        "reference_symbol": "SPY",
        "drop_threshold_pct": 3.0,
    }
}


def test_disabled_policy_returns_none() -> None:
    prices = _mk_prices([100, 95])  # huge drop
    assert _evaluate_circuit_breaker_daily(prices, {"circuit_breaker": {"enabled": False}}) is None


def test_no_trip_on_small_drop() -> None:
    prices = _mk_prices([100, 99])  # -1 %
    assert _evaluate_circuit_breaker_daily(prices, POLICY_ON) is None


def test_trip_on_large_drop() -> None:
    prices = _mk_prices([100, 96])  # -4 %
    r = _evaluate_circuit_breaker_daily(prices, POLICY_ON)
    assert r is not None
    assert r["reference_symbol"] == "SPY"
    assert r["drop_pct"] > 3.0
    assert "SPY" in r["reason"]


def test_trip_at_exact_threshold() -> None:
    prices = _mk_prices([100, 97])  # -3.00 %
    r = _evaluate_circuit_breaker_daily(prices, POLICY_ON)
    assert r is not None


def test_missing_reference_returns_none() -> None:
    prices = _mk_prices([100, 50], symbol="AAPL")  # no SPY row
    assert _evaluate_circuit_breaker_daily(prices, POLICY_ON) is None


def test_single_bar_returns_none() -> None:
    prices = _mk_prices([100])
    assert _evaluate_circuit_breaker_daily(prices, POLICY_ON) is None


def test_empty_prices_returns_none() -> None:
    empty = pd.DataFrame(columns=["timestamp", "symbol", "close"])
    assert _evaluate_circuit_breaker_daily(empty, POLICY_ON) is None


def test_as_of_filter_excludes_future_bars() -> None:
    # SPY: two bars, the second is a -4% crash. as_of pinned to the
    # first bar → no crash visible yet → no trip.
    prices = _mk_prices([100, 96])
    as_of = prices["timestamp"].iloc[0]
    assert _evaluate_circuit_breaker_daily(prices, POLICY_ON, as_of=as_of) is None


def test_custom_reference_symbol() -> None:
    prices = _mk_prices([200, 190], symbol="QQQ")  # -5 %
    policy = {
        "circuit_breaker": {
            "enabled": True,
            "reference_symbol": "QQQ",
            "drop_threshold_pct": 3.0,
        }
    }
    r = _evaluate_circuit_breaker_daily(prices, policy)
    assert r is not None
    assert r["reference_symbol"] == "QQQ"


def test_upward_move_never_trips() -> None:
    prices = _mk_prices([100, 110])
    assert _evaluate_circuit_breaker_daily(prices, POLICY_ON) is None
