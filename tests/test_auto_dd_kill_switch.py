"""Tests for the auto-drawdown kill-switch evaluator (Sprint 1 / C7)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingCycleResult,
    _evaluate_auto_dd_kill_switch,
)


def _mk_ctx(current: float | None, peak: float | None) -> SimpleNamespace:
    return SimpleNamespace(current_equity=current, peak_equity=peak, hwm_equity=peak)


def _mk_result(dd_meta: float | None = None) -> TradingCycleResult:
    r = TradingCycleResult()
    if dd_meta is not None:
        r.meta["drawdown_pct"] = dd_meta
    return r


POLICY_DEFAULT: dict = {
    "drawdown_policy": {
        "auto_kill_enabled": True,
        "levels": {"soft": -0.08, "hard": -0.12, "kill": -0.18},
    }
}


def test_no_trigger_when_drawdown_above_soft() -> None:
    # Only down 5% — below soft threshold of -8%
    ctx = _mk_ctx(current=95.0, peak=100.0)
    assert _evaluate_auto_dd_kill_switch(ctx, _mk_result(), POLICY_DEFAULT) is None


def test_soft_level_triggers_50pct_throttle() -> None:
    ctx = _mk_ctx(current=91.0, peak=100.0)  # -9 % → soft
    d = _evaluate_auto_dd_kill_switch(ctx, _mk_result(), POLICY_DEFAULT)
    assert d is not None
    assert d["level"] == "soft"
    assert d["throttle_allowed_pct"] == 0.5
    assert d["drawdown"] < -0.08


def test_hard_level_triggers_20pct_throttle() -> None:
    ctx = _mk_ctx(current=86.0, peak=100.0)  # -14 % → hard
    d = _evaluate_auto_dd_kill_switch(ctx, _mk_result(), POLICY_DEFAULT)
    assert d is not None
    assert d["level"] == "hard"
    assert d["throttle_allowed_pct"] == 0.2


def test_kill_level_blocks_all() -> None:
    ctx = _mk_ctx(current=80.0, peak=100.0)  # -20 % → kill
    d = _evaluate_auto_dd_kill_switch(ctx, _mk_result(), POLICY_DEFAULT)
    assert d is not None
    assert d["level"] == "kill"
    assert d["throttle_allowed_pct"] == 0.0


def test_staircase_is_monotone() -> None:
    # Stricter drawdown must never mean more-allowed throttle.
    last_allowed = 1.0
    for current in (95, 91, 86, 80):
        ctx = _mk_ctx(current=float(current), peak=100.0)
        d = _evaluate_auto_dd_kill_switch(ctx, _mk_result(), POLICY_DEFAULT)
        allowed = 1.0 if d is None else d["throttle_allowed_pct"]
        assert allowed <= last_allowed
        last_allowed = allowed


def test_disabled_policy_returns_none() -> None:
    ctx = _mk_ctx(current=70.0, peak=100.0)  # deep drawdown
    policy = {"drawdown_policy": {"auto_kill_enabled": False}}
    assert _evaluate_auto_dd_kill_switch(ctx, _mk_result(), policy) is None


def test_falls_back_to_meta_drawdown_when_equity_missing() -> None:
    ctx = _mk_ctx(current=None, peak=None)
    d = _evaluate_auto_dd_kill_switch(
        ctx, _mk_result(dd_meta=-0.15), POLICY_DEFAULT
    )
    assert d is not None
    assert d["level"] == "hard"


def test_no_equity_no_meta_returns_none() -> None:
    ctx = _mk_ctx(current=None, peak=None)
    assert _evaluate_auto_dd_kill_switch(ctx, _mk_result(), POLICY_DEFAULT) is None


def test_custom_thresholds_respected() -> None:
    ctx = _mk_ctx(current=96.0, peak=100.0)  # -4 %
    policy = {
        "drawdown_policy": {
            "auto_kill_enabled": True,
            "levels": {"soft": -0.03, "hard": -0.06, "kill": -0.10},
        }
    }
    d = _evaluate_auto_dd_kill_switch(ctx, _mk_result(), policy)
    assert d is not None
    assert d["level"] == "soft"


def test_reason_contains_drawdown_pct() -> None:
    ctx = _mk_ctx(current=80.0, peak=100.0)
    d = _evaluate_auto_dd_kill_switch(ctx, _mk_result(), POLICY_DEFAULT)
    assert d is not None
    assert "drawdown=-20.00%" in d["reason"]
    assert "kill" in d["reason"]
