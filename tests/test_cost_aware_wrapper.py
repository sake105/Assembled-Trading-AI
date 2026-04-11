"""Tests for portfolio/cost_aware_wrapper.py (Sprint 3 / Plan W12)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.portfolio.cost_aware_wrapper import (  # noqa: E402
    apply_cost_aware_from_policy,
    apply_cost_aware_wrapper,
)


def test_empty_targets_return_empty() -> None:
    adjusted, reasons = apply_cost_aware_wrapper({}, None)
    assert adjusted == {}
    assert reasons == []


def test_no_turnover_returns_copy_without_reasons() -> None:
    tgt = {"AAA": 0.4, "BBB": 0.3}
    cur = dict(tgt)  # identical → zero turnover
    adjusted, reasons = apply_cost_aware_wrapper(tgt, cur, penalty_factor=1.0)
    assert adjusted == tgt
    assert adjusted is not tgt
    assert reasons == []


def test_penalty_zero_is_noop() -> None:
    tgt = {"AAA": 0.4, "BBB": 0.3}
    cur = {"AAA": 0.0, "BBB": 0.0}
    adjusted, reasons = apply_cost_aware_wrapper(tgt, cur, penalty_factor=0.0)
    assert adjusted == tgt
    assert reasons == []


def test_shrinks_deltas_toward_current_weights() -> None:
    tgt = {"AAA": 0.4, "BBB": 0.3}
    cur = {"AAA": 0.0, "BBB": 0.0}
    # Very high cost per symbol → meaningful shrink
    adjusted, reasons = apply_cost_aware_wrapper(
        tgt,
        cur,
        cost_bps_per_symbol={"AAA": 500.0, "BBB": 500.0},
        penalty_factor=0.5,
    )
    # Expected shrink math:
    #   turnover = 0.7, weighted_cost = 0.7 * 500/1e4 = 0.035
    #   raw_shrink = 0.5 * 0.035 / 0.7 = 0.025
    #   shrink = 0.975
    assert reasons  # must report action
    assert 0.9 < adjusted["AAA"] / tgt["AAA"] < 1.0
    assert 0.9 < adjusted["BBB"] / tgt["BBB"] < 1.0
    assert adjusted["AAA"] < tgt["AAA"]
    assert adjusted["BBB"] < tgt["BBB"]


def test_extreme_penalty_can_reach_floor() -> None:
    tgt = {"AAA": 0.4}
    cur = {"AAA": 0.0}
    adjusted, reasons = apply_cost_aware_wrapper(
        tgt,
        cur,
        cost_bps_per_symbol={"AAA": 10_000.0},  # 100% cost
        penalty_factor=100.0,
        min_shrink=0.0,
    )
    # shrink should clamp to 0, so adjusted = current (no trade)
    assert reasons
    assert abs(adjusted["AAA"]) < 1e-9


def test_respects_min_shrink_floor() -> None:
    tgt = {"AAA": 0.5}
    cur = {"AAA": 0.0}
    adjusted, reasons = apply_cost_aware_wrapper(
        tgt,
        cur,
        cost_bps_per_symbol={"AAA": 10_000.0},
        penalty_factor=100.0,
        min_shrink=0.5,  # never shrink below 50% of the trade
    )
    assert reasons
    # shrink clamped to 0.5, so adjusted should be 0.5 * 0.5 = 0.25
    assert abs(adjusted["AAA"] - 0.25) < 1e-9


def test_from_policy_disabled_returns_copy() -> None:
    tgt = {"AAA": 0.4}
    cur = {"AAA": 0.0}
    adjusted, reasons = apply_cost_aware_from_policy(
        tgt, cur, {"cost_aware_wrapper": {"enabled": False}}
    )
    assert adjusted == tgt
    assert adjusted is not tgt
    assert reasons == []


def test_from_policy_derives_penalty_from_invested_pct() -> None:
    tgt = {"AAA": 0.3, "BBB": 0.3}
    cur = {"AAA": 0.0, "BBB": 0.0}
    policy = {"cost_aware_wrapper": {"enabled": True}}
    adjusted, reasons = apply_cost_aware_from_policy(
        tgt,
        cur,
        policy,
        cost_bps_per_symbol={"AAA": 100.0, "BBB": 100.0},
        current_invested_pct=0.8,  # → penalty_factor = 0.4
    )
    assert reasons  # enabled + penalty > 0 + turnover > 0
    # Shrink should be mild (cost only 100 bps), but non-zero
    assert adjusted["AAA"] < tgt["AAA"]
    assert adjusted["AAA"] > 0.9 * tgt["AAA"]
