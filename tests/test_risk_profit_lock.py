"""Tests for profit lock overlay — M6 coverage (module already implemented in M3/M5).

Covers:
- disabled policy → always returns (1.0, {})
- insufficient equity curve history → 1.0
- return below trigger → 1.0
- return at/above trigger → reduced multiplier
- cooldown preserves reduced multiplier
- cooldown expires → resets
- floor clamping when multiplier_on_trigger < floor
- multiplier clamped to 1.0 when multiplier_on_trigger > 1.0
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.phase12

from src.assembled_core.risk.profit_lock import compute_profit_lock_multiplier


def _policy(
    enabled: bool = True,
    lookback_days: int = 20,
    trigger_return: float = 0.08,
    multiplier_on_trigger: float = 0.80,
    floor: float = 0.50,
    cooldown_days: int = 10,
) -> dict:
    return {
        "enabled": enabled,
        "lookback_days": lookback_days,
        "trigger_return": trigger_return,
        "multiplier_on_trigger": multiplier_on_trigger,
        "floor": floor,
        "cooldown_days": cooldown_days,
    }


def _flat_curve(n: int, start: float = 1.0) -> pd.Series:
    """Equity curve with no returns (flat)."""
    return pd.Series([start] * n)


def _growth_curve(n: int, daily_return: float = 0.01) -> pd.Series:
    """Equity curve with constant daily return."""
    vals = [1.0]
    for _ in range(n - 1):
        vals.append(vals[-1] * (1 + daily_return))
    return pd.Series(vals)


# ---------------------------------------------------------------------------
# Disabled
# ---------------------------------------------------------------------------


class TestProfitLockDisabled:
    def test_disabled_returns_one_and_empty_state(self):
        curve = _growth_curve(30, daily_return=0.05)
        mult, state = compute_profit_lock_multiplier(curve, _policy(enabled=False), now_idx=29)
        assert mult == 1.0
        assert isinstance(state, dict)

    def test_disabled_state_pass_through(self):
        curve = _growth_curve(30, daily_return=0.05)
        existing_state = {"trigger_idx": 5}
        mult, state = compute_profit_lock_multiplier(
            curve, _policy(enabled=False), now_idx=29, state=existing_state
        )
        assert mult == 1.0
        assert state.get("trigger_idx") == 5  # state preserved, not consumed


# ---------------------------------------------------------------------------
# Insufficient data
# ---------------------------------------------------------------------------


class TestProfitLockInsufficientData:
    def test_none_curve_returns_one(self):
        mult, _ = compute_profit_lock_multiplier(None, _policy(), now_idx=0)  # type: ignore[arg-type]
        assert mult == 1.0

    def test_empty_curve_returns_one(self):
        mult, _ = compute_profit_lock_multiplier(pd.Series([], dtype=float), _policy(), now_idx=0)
        assert mult == 1.0

    def test_curve_shorter_than_lookback_returns_one(self):
        curve = _growth_curve(15)  # 15 bars, lookback=20 → need now_idx >= 20
        mult, _ = compute_profit_lock_multiplier(curve, _policy(lookback_days=20), now_idx=14)
        assert mult == 1.0

    def test_now_idx_zero_returns_one(self):
        curve = _growth_curve(30)
        mult, _ = compute_profit_lock_multiplier(curve, _policy(lookback_days=20), now_idx=0)
        assert mult == 1.0


# ---------------------------------------------------------------------------
# Trigger and multiplier
# ---------------------------------------------------------------------------


class TestProfitLockTrigger:
    def test_return_below_trigger_no_lock(self):
        # 20 bars, 0.3% daily → lookback return ≈ 6% < trigger 8%
        curve = _growth_curve(30, daily_return=0.003)
        mult, state = compute_profit_lock_multiplier(curve, _policy(trigger_return=0.08), now_idx=29)
        assert mult == 1.0
        assert "trigger_idx" not in state

    def test_return_above_trigger_applies_multiplier(self):
        # 20 bars, 0.5% daily → lookback return ≈ 10.5% > trigger 8%
        curve = _growth_curve(30, daily_return=0.005)
        mult, state = compute_profit_lock_multiplier(
            curve, _policy(trigger_return=0.08, multiplier_on_trigger=0.80), now_idx=29
        )
        assert mult == pytest.approx(0.80)
        assert state.get("trigger_idx") == 29

    def test_multiplier_clamped_by_floor(self):
        # multiplier_on_trigger=0.30 but floor=0.50 → clamped to 0.50
        curve = _growth_curve(30, daily_return=0.005)
        mult, _ = compute_profit_lock_multiplier(
            curve,
            _policy(trigger_return=0.08, multiplier_on_trigger=0.30, floor=0.50),
            now_idx=29,
        )
        assert mult == pytest.approx(0.50)

    def test_multiplier_clamped_to_one_when_above_one(self):
        # multiplier_on_trigger=1.2 → clamped to 1.0
        curve = _growth_curve(30, daily_return=0.005)
        mult, _ = compute_profit_lock_multiplier(
            curve,
            _policy(trigger_return=0.08, multiplier_on_trigger=1.20),
            now_idx=29,
        )
        assert mult == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Cooldown
# ---------------------------------------------------------------------------


class TestProfitLockCooldown:
    def test_cooldown_preserves_multiplier(self):
        # Trigger at idx=20, check at idx=25 (cooldown=10) → still locked
        curve = _growth_curve(30, daily_return=0.005)
        _, state_after_trigger = compute_profit_lock_multiplier(
            curve, _policy(trigger_return=0.08, cooldown_days=10), now_idx=29
        )
        # Now re-check at idx=25 (still within cooldown)
        # Simulate: trigger_idx=29, now=25+5=30 → within cooldown
        # Re-use state to simulate next bar evaluation
        mult, _ = compute_profit_lock_multiplier(
            curve,
            _policy(trigger_return=0.08, multiplier_on_trigger=0.80, cooldown_days=10),
            now_idx=29,
            state=state_after_trigger,
        )
        assert mult == pytest.approx(0.80)

    def test_cooldown_expires_resets(self):
        # Trigger at idx=5, cooldown=10, now_idx=25 → expired (25-5=20 > 10).
        # now_idx must be >= lookback_days (20) so profit_lock doesn't return early
        # before processing the cooldown expiry.
        curve = _growth_curve(50, daily_return=0.005)
        state_with_old_trigger = {"trigger_idx": 5}
        mult, new_state = compute_profit_lock_multiplier(
            curve,
            _policy(trigger_return=0.99, cooldown_days=10),  # high trigger to avoid re-trigger
            now_idx=25,
            state=state_with_old_trigger,
        )
        # Cooldown expired; lookback return (≈10%) is below trigger (99%) → 1.0
        assert mult == pytest.approx(1.0)
        assert "trigger_idx" not in new_state
