"""Property-based tests for the order-lifecycle FSM, kill-switch FSM, and
generic PIT-safety invariants (audit E-003 / E-004 / E-005).

These tests complement ``test_hypothesis_property.py`` which already covers
qa/, portfolio/, and risk/ — here we add the safety-critical state machines
plus a generic PIT property the audit names as non-negotiable.

Markers:
    - No optional-dep guard needed: Hypothesis is in the dev extras
      (already used by the existing property test).
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st


# ---------------------------------------------------------------------------
# E-003 — Order-FSM
# ---------------------------------------------------------------------------


_NON_TERMINAL = ["CREATED", "VALIDATED", "SUBMITTED", "PARTIAL_FILL"]
_TERMINAL = ["FILLED", "CANCELLED", "REJECTED"]


@given(
    transitions=st.lists(
        st.sampled_from(_NON_TERMINAL + _TERMINAL),
        min_size=0,
        max_size=8,
    )
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.too_slow])
def test_order_fsm_invalid_transitions_raise(transitions: list[str]) -> None:
    """Any sequence of transitions either follows _VALID_TRANSITIONS or raises ValueError.

    The state machine never silently swallows an illegal transition.
    """
    from src.assembled_core.execution.order_lifecycle import (
        OrderLifecycleTracker,
        OrderState,
        _VALID_TRANSITIONS,
    )

    tracker = OrderLifecycleTracker()
    oid = tracker.create(symbol="TST", side="BUY", quantity=1.0)

    current = OrderState.CREATED
    for step in transitions:
        target = OrderState(step)
        allowed = _VALID_TRANSITIONS.get(current, set())
        if target in allowed:
            tracker.transition(oid, target)
            current = target
        else:
            with pytest.raises(ValueError):
                tracker.transition(oid, target)
            # Stop on first illegal — terminal states have empty allowed sets,
            # so any further attempt would also raise.
            break


@given(target=st.sampled_from(_TERMINAL))
@settings(max_examples=30)
def test_order_fsm_terminal_states_are_absorbing(target: str) -> None:
    """Once in FILLED/CANCELLED/REJECTED, any further transition must raise."""
    from src.assembled_core.execution.order_lifecycle import (
        OrderLifecycleTracker,
        OrderState,
    )

    tracker = OrderLifecycleTracker()
    oid = tracker.create(symbol="TST", side="BUY", quantity=1.0)

    # Drive to a terminal state via the shortest legal path.
    if target == "REJECTED":
        tracker.transition(oid, OrderState.REJECTED)
    else:
        tracker.transition(oid, OrderState.VALIDATED)
        tracker.transition(oid, OrderState.SUBMITTED)
        tracker.transition(oid, OrderState(target))

    # Every onward transition must raise.
    for s in OrderState:
        with pytest.raises(ValueError):
            tracker.transition(oid, s)


# ---------------------------------------------------------------------------
# E-004 — Kill-Switch FSM
# ---------------------------------------------------------------------------


@pytest.fixture()
def isolated_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run kill-switch tests against a temp state/audit dir.

    The fixture sets the override env vars and clears them after the test,
    so the property tests do not touch the real ops/ directory.
    """
    with tempfile.TemporaryDirectory() as tmp:
        monkeypatch.setenv(
            "ASSEMBLED_KILL_SWITCH_STATE", os.path.join(tmp, "state.json")
        )
        monkeypatch.setenv(
            "ASSEMBLED_KILL_SWITCH_AUDIT", os.path.join(tmp, "audit.jsonl")
        )
        monkeypatch.setenv(
            "ASSEMBLED_KILL_SWITCH_SENTINEL", os.path.join(tmp, ".sentinel")
        )
        monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)
        yield


@given(throttle=st.floats(min_value=0.0, max_value=1.0, allow_nan=False))
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_kill_switch_activate_then_deactivate(
    throttle: float, isolated_kill_switch: None
) -> None:
    """activate(throttle) -> engaged=True; deactivate -> engaged=False, throttle=1.0."""
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
        get_kill_switch_state,
    )

    activate_kill_switch(throttle_pct=throttle, reason="prop-test", actor="hypothesis")
    s1 = get_kill_switch_state()
    assert s1["engaged"] is True
    # activate clamps throttle to [0,1]; matches our generator anyway
    assert 0.0 <= s1["throttle_pct"] <= 1.0

    deactivate_kill_switch(reason="prop-test", actor="hypothesis")
    s2 = get_kill_switch_state()
    assert s2["engaged"] is False
    assert s2["throttle_pct"] == 1.0


@given(
    n_activations=st.integers(min_value=1, max_value=4),
    throttle=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_kill_switch_repeated_activation_idempotent(
    n_activations: int, throttle: float, isolated_kill_switch: None
) -> None:
    """Repeated activations remain engaged; final throttle equals last activation."""
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        get_kill_switch_state,
    )

    for i in range(n_activations):
        activate_kill_switch(throttle_pct=throttle, reason=f"prop-{i}", actor="ht")

    s = get_kill_switch_state()
    assert s["engaged"] is True
    assert abs(s["throttle_pct"] - throttle) < 1e-9


# ---------------------------------------------------------------------------
# E-005 — Generic PIT property
# ---------------------------------------------------------------------------


def _simple_rolling_mean(prices: pd.Series, window: int = 5) -> pd.Series:
    """Reference PIT-safe rolling mean.

    Used as the canonical example of a PIT-safe feature: feature[i] depends
    only on prices[0..i], never on prices[i+1..].
    """
    return prices.rolling(window=window, min_periods=1).mean()


@given(
    prices=st.lists(
        st.floats(
            min_value=1.0, max_value=500.0, allow_nan=False, allow_infinity=False
        ),
        min_size=20,
        max_size=120,
    ),
    cut_frac=st.floats(min_value=0.3, max_value=0.9, allow_nan=False),
)
@settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
def test_pit_safety_rolling_mean(prices: list[float], cut_frac: float) -> None:
    """A PIT-safe feature applied on prices[:cut] == feature applied on full prices, sliced to :cut.

    This is the canonical PIT property the audit names (E-005): no future
    information may leak into past feature values. Used as a reference
    implementation; specific features should be wired to this same shape.
    """
    arr = pd.Series(prices, dtype=float)
    cut = max(5, int(len(arr) * cut_frac))
    assume(cut < len(arr))

    feat_full = _simple_rolling_mean(arr).iloc[:cut]
    feat_prefix = _simple_rolling_mean(arr.iloc[:cut])

    # NaN-safe equality
    pd.testing.assert_series_equal(
        feat_full.reset_index(drop=True),
        feat_prefix.reset_index(drop=True),
        check_dtype=False,
        atol=1e-9,
        rtol=0,
    )


@given(
    n=st.integers(min_value=10, max_value=80),
)
@settings(max_examples=20, suppress_health_check=[HealthCheck.too_slow])
def test_pit_safety_pct_change(n: int) -> None:
    """pct_change is PIT-safe: row i depends only on prices[i-1..i]."""
    rng = np.random.default_rng(seed=1234)
    prices = pd.Series(rng.uniform(50, 200, size=n))

    full_feat = prices.pct_change()
    cut = max(2, n // 2)
    prefix_feat = prices.iloc[:cut].pct_change()

    pd.testing.assert_series_equal(
        full_feat.iloc[:cut].reset_index(drop=True),
        prefix_feat.reset_index(drop=True),
        check_dtype=False,
        atol=1e-12,
        rtol=0,
    )
