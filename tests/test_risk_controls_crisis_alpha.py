"""Unit tests for T4.3 — crisis_alpha PAUSE kill-switch in pre-trade checks.

Tests:
    1. test_watch_state_allows_orders      — WATCH → (True, "OK")
    2. test_pause_state_blocks_orders      — PAUSE → (False, "BLOCKED...")
    3. test_disabled_policy_skips_check    — enabled=False → always (True, "OK")
    4. test_missing_state_fails_open       — no state available → (True, ...)
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

pytestmark = pytest.mark.phase12

from src.assembled_core.execution.risk_controls import (
    check_crisis_alpha_kill_switch,
    filter_orders_with_risk_controls,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx_with_state(state: str) -> SimpleNamespace:
    """Return a minimal ctx object carrying a crisis_alpha_state string."""
    return SimpleNamespace(crisis_alpha_state=state)


def _ctx_with_meta_state(state: str) -> SimpleNamespace:
    """Return a ctx with state carried in the meta dict (fallback 2)."""
    return SimpleNamespace(meta={"crisis_alpha_state": state})


def _ctx_empty() -> SimpleNamespace:
    """Return a ctx with no crisis state information at all."""
    return SimpleNamespace()


def _sample_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "GOOGL"],
            "side": ["BUY", "BUY"],
            "qty": [100, 50],
            "price": [150.0, 2800.0],
        }
    )


def _crisis_alpha_policy(enabled: bool) -> dict:
    return {"intel": {"crisis_alpha": {"enabled": enabled}}}


# ---------------------------------------------------------------------------
# Tests for check_crisis_alpha_kill_switch (standalone function)
# ---------------------------------------------------------------------------


class TestCheckCrisisAlphaKillSwitch:
    """Direct unit tests for check_crisis_alpha_kill_switch."""

    @pytest.mark.parametrize("state", ["NORMAL", "WATCH", "ACTIVE", "COOLDOWN"])
    def test_allowed_states_return_true(self, state: str) -> None:
        ctx = _ctx_with_state(state)
        allowed, reason = check_crisis_alpha_kill_switch(ctx)
        assert allowed is True
        assert reason == "OK"

    def test_watch_state_allows_orders(self) -> None:
        """T4.3 test 1: WATCH → (True, 'OK')."""
        ctx = _ctx_with_state("WATCH")
        allowed, reason = check_crisis_alpha_kill_switch(ctx)
        assert allowed is True
        assert reason == "OK"

    def test_pause_state_blocks_orders(self) -> None:
        """T4.3 test 2: PAUSE → (False, ...) with 'BLOCKED' in reason."""
        ctx = _ctx_with_state("PAUSE")
        allowed, reason = check_crisis_alpha_kill_switch(ctx)
        assert allowed is False
        assert "BLOCKED" in reason
        assert "PAUSE" in reason

    def test_pause_state_via_meta_blocks_orders(self) -> None:
        """PAUSE carried in ctx.meta also blocks."""
        ctx = _ctx_with_meta_state("PAUSE")
        allowed, reason = check_crisis_alpha_kill_switch(ctx)
        assert allowed is False
        assert "BLOCKED" in reason

    def test_missing_state_fails_open(self) -> None:
        """T4.3 test 4: no state available → (True, ...) fail-open."""
        ctx = _ctx_empty()
        allowed, reason = check_crisis_alpha_kill_switch(ctx)
        assert allowed is True
        # Reason should indicate no state was found
        assert "no crisis state" in reason.lower() or reason == "OK"

    def test_crisis_state_record_object_with_state_attr(self) -> None:
        """ctx.crisis_alpha_state can be a CrisisStateRecord-like object."""

        class _FakeRecord:
            state = "PAUSE"

        ctx = SimpleNamespace(crisis_alpha_state=_FakeRecord())
        allowed, reason = check_crisis_alpha_kill_switch(ctx)
        assert allowed is False
        assert "BLOCKED" in reason

    def test_state_case_insensitive(self) -> None:
        """State comparison is case-insensitive for robustness."""
        ctx = _ctx_with_state("pause")  # lowercase
        allowed, reason = check_crisis_alpha_kill_switch(ctx)
        assert allowed is False

        ctx2 = _ctx_with_state("watch")
        allowed2, _ = check_crisis_alpha_kill_switch(ctx2)
        assert allowed2 is True

    def test_empty_string_state_fails_open(self) -> None:
        """An empty string state is treated as missing → fail-open."""
        ctx = SimpleNamespace(crisis_alpha_state="")
        allowed, reason = check_crisis_alpha_kill_switch(ctx)
        assert allowed is True


# ---------------------------------------------------------------------------
# Tests for filter_orders_with_risk_controls integration
# ---------------------------------------------------------------------------


class TestFilterOrdersWithCrisisAlphaGate:
    """Integration tests: crisis_alpha gate inside filter_orders_with_risk_controls."""

    def test_disabled_policy_skips_check_allows_all(self) -> None:
        """T4.3 test 3: enabled=False → check skipped, orders pass through."""
        policy = _crisis_alpha_policy(enabled=False)
        # Even with PAUSE state, if disabled the gate must be skipped
        ctx = _ctx_with_state("PAUSE")
        orders = _sample_orders()
        filtered, result = filter_orders_with_risk_controls(
            orders,
            enable_pre_trade_checks=False,
            enable_kill_switch=False,
            policy=policy,
            crisis_alpha_ctx=ctx,
        )
        # Orders should pass through (gate disabled)
        assert len(filtered) == len(orders)
        assert result.kill_switch_engaged is False

    def test_enabled_policy_pause_blocks_all_orders(self) -> None:
        """enabled=True + PAUSE state → all orders blocked."""
        policy = _crisis_alpha_policy(enabled=True)
        ctx = _ctx_with_state("PAUSE")
        orders = _sample_orders()
        filtered, result = filter_orders_with_risk_controls(
            orders,
            enable_pre_trade_checks=False,
            enable_kill_switch=False,
            policy=policy,
            crisis_alpha_ctx=ctx,
        )
        assert len(filtered) == 0
        assert result.total_orders_before == 2
        assert result.total_orders_after == 0
        assert result.kill_switch_engaged is True

    def test_enabled_policy_watch_allows_orders(self) -> None:
        """enabled=True + WATCH state → orders pass through gate."""
        policy = _crisis_alpha_policy(enabled=True)
        ctx = _ctx_with_state("WATCH")
        orders = _sample_orders()
        filtered, result = filter_orders_with_risk_controls(
            orders,
            enable_pre_trade_checks=False,
            enable_kill_switch=False,
            policy=policy,
            crisis_alpha_ctx=ctx,
        )
        assert len(filtered) == len(orders)
        assert result.kill_switch_engaged is False

    def test_no_ctx_with_enabled_policy_falls_through(self) -> None:
        """enabled=True but no crisis_alpha_ctx → gate not invoked, orders pass."""
        policy = _crisis_alpha_policy(enabled=True)
        orders = _sample_orders()
        filtered, result = filter_orders_with_risk_controls(
            orders,
            enable_pre_trade_checks=False,
            enable_kill_switch=False,
            policy=policy,
            crisis_alpha_ctx=None,  # no ctx provided
        )
        # No ctx → gate not triggered → orders allowed
        assert len(filtered) == len(orders)

    def test_no_policy_defaults_to_disabled(self) -> None:
        """No policy dict → gate defaults to disabled (fail-safe)."""
        ctx = _ctx_with_state("PAUSE")
        orders = _sample_orders()
        filtered, result = filter_orders_with_risk_controls(
            orders,
            enable_pre_trade_checks=False,
            enable_kill_switch=False,
            policy=None,  # no policy
            crisis_alpha_ctx=ctx,
        )
        # No policy → disabled → orders pass
        assert len(filtered) == len(orders)
        assert result.kill_switch_engaged is False

    @pytest.mark.parametrize("state", ["NORMAL", "WATCH", "ACTIVE", "COOLDOWN"])
    def test_all_non_pause_states_allow_orders(self, state: str) -> None:
        """All non-PAUSE states must allow orders when gate is enabled."""
        policy = _crisis_alpha_policy(enabled=True)
        ctx = _ctx_with_state(state)
        orders = _sample_orders()
        filtered, result = filter_orders_with_risk_controls(
            orders,
            enable_pre_trade_checks=False,
            enable_kill_switch=False,
            policy=policy,
            crisis_alpha_ctx=ctx,
        )
        assert len(filtered) == len(orders), f"state={state} should allow orders"
        assert result.kill_switch_engaged is False
