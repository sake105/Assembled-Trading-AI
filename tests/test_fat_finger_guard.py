"""Tests for execution/fat_finger_guard.py (Sprint 4 / Plan C29)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.execution.fat_finger_guard import (  # noqa: E402
    apply_fat_finger_guard,
    apply_fat_finger_guard_from_policy,
)


def _orders(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["symbol", "qty", "price"])


def test_empty_orders_pass_through() -> None:
    df = pd.DataFrame(columns=["symbol", "qty", "price"])
    filtered, reasons = apply_fat_finger_guard(df, max_notional_usd=1000.0)
    assert filtered.empty
    assert reasons == []


def test_notional_cap_rejects_oversized_order() -> None:
    orders = _orders(
        [
            ("AAA", 100.0, 10.0),  # notional 1000
            ("BBB", 500.0, 50.0),  # notional 25000
        ]
    )
    filtered, reasons = apply_fat_finger_guard(orders, max_notional_usd=10_000.0)
    assert list(filtered["symbol"]) == ["AAA"]
    assert len(reasons) == 1
    assert "BBB" in reasons[0]
    assert "notional" in reasons[0]


def test_qty_multiple_rejects_vs_history() -> None:
    orders = _orders(
        [
            ("AAA", 50.0, 10.0),  # 2x history_max=25 → rejected (3x cap)
            ("BBB", 200.0, 10.0),  # 4x history_max=50 → rejected (3x cap)
            ("CCC", 10.0, 10.0),  # 1x history_max=10 → ok
        ]
    )
    filtered, reasons = apply_fat_finger_guard(
        orders,
        max_qty_multiple=3.0,
        history_qty_by_symbol={"AAA": 25.0, "BBB": 50.0, "CCC": 10.0},
    )
    # BBB triggers 4x > 3x; AAA is exactly 2x (below cap) so keeps.
    assert list(filtered["symbol"]) == ["AAA", "CCC"]
    assert "BBB" in " ".join(reasons)


def test_both_checks_combined() -> None:
    orders = _orders(
        [
            ("AAA", 100.0, 10.0),  # notional 1000, history=50 -> 2x, ok
            ("BBB", 1000.0, 500.0),  # notional 500000 -> rejected by notional
            (
                "CCC",
                200.0,
                10.0,
            ),  # notional 2000, history=10 -> 20x, rejected by multiple
        ]
    )
    filtered, reasons = apply_fat_finger_guard(
        orders,
        max_notional_usd=10_000.0,
        max_qty_multiple=3.0,
        history_qty_by_symbol={"AAA": 50.0, "CCC": 10.0},
    )
    assert list(filtered["symbol"]) == ["AAA"]
    assert len(reasons) == 2


def test_missing_history_symbol_passes_multiple_check() -> None:
    orders = _orders([("AAA", 999.0, 1.0)])
    filtered, reasons = apply_fat_finger_guard(
        orders,
        max_qty_multiple=3.0,
        history_qty_by_symbol={},  # no history → skip multiple check
    )
    assert len(filtered) == 1
    assert reasons == []


def test_zero_history_does_not_divide_by_zero() -> None:
    orders = _orders([("AAA", 100.0, 1.0)])
    filtered, reasons = apply_fat_finger_guard(
        orders,
        max_qty_multiple=3.0,
        history_qty_by_symbol={"AAA": 0.0},  # zero history is skipped
    )
    assert len(filtered) == 1
    assert reasons == []


def test_from_policy_disabled_passes_through() -> None:
    orders = _orders([("AAA", 1_000_000.0, 1.0)])
    filtered, reasons = apply_fat_finger_guard_from_policy(
        orders, {"fat_finger_guard": {"enabled": False}}
    )
    assert len(filtered) == 1
    assert reasons == []


def test_from_policy_enabled_enforces_cap() -> None:
    orders = _orders([("AAA", 100.0, 5_000.0)])  # 500k notional
    policy = {
        "fat_finger_guard": {
            "enabled": True,
            "max_notional_usd": 100_000.0,
        }
    }
    filtered, reasons = apply_fat_finger_guard_from_policy(orders, policy)
    assert filtered.empty
    assert len(reasons) == 1
