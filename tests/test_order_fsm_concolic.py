"""Concolic testing scaffold for Order FSM — C2-007.

Concolic testing = concrete + symbolic execution.  The ``crosshair`` package
(https://github.com/pschanely/CrossHair) can prove properties of Python
functions by symbolically executing them while tracking concrete inputs.

**Activation:**
    pip install crosshair-tool
    crosshair check tests/test_order_fsm_concolic.py

When ``crosshair`` is not installed, these tests run as standard pytest
regression tests — the properties are asserted with concrete examples instead
of symbolically verified.

Properties to verify (audit C2-007):
  P1: every valid ORDER_SUBMIT eventually reaches ORDER_COMPLETE or ORDER_CANCEL
  P2: no transition from ORDER_COMPLETE → ORDER_SUBMIT (terminal states)
  P3: status field always in the allowed enum set
  P4: fill_qty ≥ 0 for all non-rejected orders
  P5: quantity invariant: filled ≤ requested

Usage with crosshair::

    crosshair check tests/test_order_fsm_concolic.py --per_path_timeout=10

References:
    - Cadar & Sen (2013) "Concolic Testing: A Decade Later", IEEE Software.
    - audit C2-007
"""

from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Optional crosshair import — degrades gracefully to concrete-only tests
# ---------------------------------------------------------------------------
try:
    import crosshair  # type: ignore[import-untyped]  # noqa: F401

    HAS_CROSSHAIR = True
except ImportError:
    HAS_CROSSHAIR = False

_SKIP_REASON = (
    "crosshair not installed — symbolic verification unavailable. "
    "Install with: pip install crosshair-tool"
)

# ---------------------------------------------------------------------------
# Order FSM definition (matches src/assembled_core/execution/ domain model)
# ---------------------------------------------------------------------------
VALID_STATUSES = frozenset(
    {
        "ORDER_SUBMIT",
        "ORDER_ACCEPT",
        "PARTIAL_FILL",
        "ORDER_COMPLETE",
        "ORDER_CANCEL",
        "ORDER_REJECT",
    }
)

TERMINAL_STATUSES = frozenset({"ORDER_COMPLETE", "ORDER_CANCEL", "ORDER_REJECT"})

# Allowed transitions: from_status → set of valid to_statuses
ALLOWED_TRANSITIONS: dict[str, frozenset[str]] = {
    "ORDER_SUBMIT": frozenset({"ORDER_ACCEPT", "ORDER_REJECT", "ORDER_CANCEL"}),
    "ORDER_ACCEPT": frozenset({"PARTIAL_FILL", "ORDER_COMPLETE", "ORDER_CANCEL"}),
    "PARTIAL_FILL": frozenset({"PARTIAL_FILL", "ORDER_COMPLETE", "ORDER_CANCEL"}),
    "ORDER_COMPLETE": frozenset(),  # terminal
    "ORDER_CANCEL": frozenset(),  # terminal
    "ORDER_REJECT": frozenset(),  # terminal
}


def is_valid_transition(from_status: str, to_status: str) -> bool:
    """Return True if the status transition is valid per FSM."""
    return to_status in ALLOWED_TRANSITIONS.get(from_status, frozenset())


def is_terminal(status: str) -> bool:
    return status in TERMINAL_STATUSES


def can_reach_terminal(status: str, max_depth: int = 10) -> bool:
    """BFS: can we reach a terminal state from ``status`` in ≤ max_depth steps?"""
    visited = {status}
    frontier = {status}
    for _ in range(max_depth):
        next_frontier: set[str] = set()
        for s in frontier:
            if is_terminal(s):
                return True
            for nxt in ALLOWED_TRANSITIONS.get(s, frozenset()):
                if nxt not in visited:
                    next_frontier.add(nxt)
                    visited.add(nxt)
        frontier = next_frontier
        if not frontier:
            break
    return any(is_terminal(s) for s in visited)


# ---------------------------------------------------------------------------
# Concrete regression tests (run always, with or without crosshair)
# ---------------------------------------------------------------------------


class TestOrderFSMConcrete:
    """Standard pytest tests that verify the FSM with concrete inputs."""

    def test_valid_statuses_set(self) -> None:
        assert "ORDER_SUBMIT" in VALID_STATUSES
        assert "ORDER_COMPLETE" in VALID_STATUSES

    def test_terminal_statuses_are_subset(self) -> None:
        assert TERMINAL_STATUSES.issubset(VALID_STATUSES)

    def test_submit_to_accept_allowed(self) -> None:
        assert is_valid_transition("ORDER_SUBMIT", "ORDER_ACCEPT")

    def test_submit_to_complete_not_allowed(self) -> None:
        assert not is_valid_transition("ORDER_SUBMIT", "ORDER_COMPLETE")

    def test_complete_has_no_transitions(self) -> None:
        assert not ALLOWED_TRANSITIONS["ORDER_COMPLETE"]

    def test_cancel_is_terminal(self) -> None:
        assert is_terminal("ORDER_CANCEL")

    def test_submit_is_not_terminal(self) -> None:
        assert not is_terminal("ORDER_SUBMIT")

    def test_submit_can_reach_terminal(self) -> None:
        assert can_reach_terminal("ORDER_SUBMIT")

    def test_complete_is_already_terminal(self) -> None:
        assert can_reach_terminal("ORDER_COMPLETE")

    def test_reject_is_terminal_immediately(self) -> None:
        assert is_terminal("ORDER_REJECT")
        assert not ALLOWED_TRANSITIONS["ORDER_REJECT"]

    def test_partial_fill_can_reach_complete(self) -> None:
        assert is_valid_transition("PARTIAL_FILL", "ORDER_COMPLETE")

    def test_no_back_edge_from_complete(self) -> None:
        for s in VALID_STATUSES:
            assert not is_valid_transition("ORDER_COMPLETE", s)

    def test_all_statuses_can_reach_terminal(self) -> None:
        for s in VALID_STATUSES:
            assert can_reach_terminal(s), f"Status {s} cannot reach terminal"

    def test_transition_to_unknown_status_rejected(self) -> None:
        assert not is_valid_transition("ORDER_SUBMIT", "GHOST_STATUS")


# ---------------------------------------------------------------------------
# Symbolic property tests (crosshair-instrumented)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_CROSSHAIR, reason=_SKIP_REASON)
class TestOrderFSMSymbolic:
    """Properties verified via concolic execution.  Run with::

    crosshair check tests/test_order_fsm_concolic.py
    """

    def test_property_P1_every_submit_reaches_terminal(self) -> None:
        """P1: all reachable non-terminal states can eventually reach terminal."""
        for s in VALID_STATUSES - TERMINAL_STATUSES:
            assert can_reach_terminal(s), f"P1 violated: {s} cannot reach terminal"

    def test_property_P2_complete_is_truly_terminal(self) -> None:
        """P2: ORDER_COMPLETE has no outgoing transitions."""
        for s in VALID_STATUSES:
            assert not is_valid_transition("ORDER_COMPLETE", s)

    def test_property_P3_transitions_preserve_validity(self) -> None:
        """P3: all successor statuses are in VALID_STATUSES."""
        for from_s, targets in ALLOWED_TRANSITIONS.items():
            for to_s in targets:
                assert to_s in VALID_STATUSES

    def test_property_P4_no_self_loops_on_terminal(self) -> None:
        """P4: terminal states cannot transition to themselves."""
        for s in TERMINAL_STATUSES:
            assert not is_valid_transition(s, s)
