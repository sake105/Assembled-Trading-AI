"""Structural guard: state machine must run BEFORE intel load in ingest_data.

This is a PIT-firewall regression guard, not a behavioural test. In
``ingest_data`` (``src/assembled_core/pipeline/trading_cycle_v2.py``) the risk
state machine (``compute_next_state``) is deliberately called BEFORE
``_load_intel``.

Why the order is load-bearing (see the comment block at the top of
``_load_intel``): the geo / disclosures sources ``_load_intel`` reads are NOT
as_of-indexed — ``news_geo`` (``crisis_state.json``) and ``disclosures_triggers``
(``triggers_latest.json`` via ``load_disclosures_triggers(path)``) are single
live "latest" snapshots with no PIT filtering. Only ``market_stress`` is
PIT-guarded. Reordering ``_load_intel`` ahead of ``compute_next_state`` would
inject TODAY's live snapshot into every historical bar's risk-state transitions
(WATCH->ACTIVE/PAUSE) = a backtest look-ahead on the risk-state path
(anti-pattern E-002), in the most sensitive component.

This test parses the source with ``ast`` (no import / execution of the trading
cycle) and asserts the call order, so a future silent reorder fails loudly
instead of introducing a latent look-ahead.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "assembled_core"
    / "pipeline"
    / "trading_cycle_v2.py"
)


def _ingest_data_node() -> ast.FunctionDef:
    """Return the ``ingest_data`` FunctionDef node, or fail if it moved/renamed."""
    source = _MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(_MODULE_PATH))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "ingest_data":
            return node
    pytest.fail(
        "ingest_data() not found at module top level in trading_cycle_v2.py — "
        "structure changed; re-check the PIT-firewall ordering guard."
    )


def _called_func_name(call: ast.Call) -> str | None:
    """Return the simple callee name for a Call node (``foo`` or ``a.foo`` -> 'foo')."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _call_linenos(func_node: ast.FunctionDef, target_name: str) -> list[int]:
    return sorted(
        call.lineno
        for call in ast.walk(func_node)
        if isinstance(call, ast.Call) and _called_func_name(call) == target_name
    )


def test_compute_next_state_called_in_ingest_data() -> None:
    """Sanity: the state-machine call exists where the firewall depends on it."""
    node = _ingest_data_node()
    assert _call_linenos(node, "compute_next_state"), (
        "compute_next_state no longer called inside ingest_data — PIT-firewall "
        "ordering assumption broken."
    )


def test_load_intel_called_in_ingest_data() -> None:
    """Sanity: _load_intel is invoked from ingest_data (the ordering anchor)."""
    node = _ingest_data_node()
    assert _call_linenos(node, "_load_intel"), (
        "_load_intel no longer called inside ingest_data — PIT-firewall ordering "
        "anchor moved; re-verify state-machine vs intel-load ordering."
    )


def test_state_machine_runs_before_load_intel() -> None:
    """PIT FIREWALL: every compute_next_state call must precede _load_intel.

    If a refactor moves _load_intel ahead of (or interleaved with) the state
    machine, this assertion fails — flagging a potential look-ahead on the
    risk-state path before it can ship silently.
    """
    node = _ingest_data_node()

    cns_lines = _call_linenos(node, "compute_next_state")
    load_intel_lines = _call_linenos(node, "_load_intel")

    assert cns_lines, "compute_next_state not called in ingest_data"
    assert load_intel_lines, "_load_intel not called in ingest_data"

    last_state_machine = max(cns_lines)
    first_load_intel = min(load_intel_lines)

    assert last_state_machine < first_load_intel, (
        "PIT FIREWALL VIOLATION: compute_next_state (last call line "
        f"{last_state_machine}) must run BEFORE _load_intel (first call line "
        f"{first_load_intel}) in ingest_data. _load_intel reads NON-as_of live "
        "snapshots (crisis_state.json / triggers_latest.json); feeding them into "
        "the risk state machine on historical bars is a look-ahead (E-002). Do "
        "not reorder."
    )
