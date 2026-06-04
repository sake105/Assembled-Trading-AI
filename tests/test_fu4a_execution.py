"""FU-4a execution follow-up tests.

Covers two execution-area follow-ups:

FIX 1 — ``is_duplicate_error`` false-positive tightening (idempotency.py):
    the order/id guard was a bare ``"order" in msg`` substring that also matched
    look-alikes ("border", "reorder", "recorder", "ordering", "disorder"). It is
    now a word-boundary match. These tests pin BOTH directions: every real broker
    duplicate signature still classifies True, while look-alike substrings and
    generic/transient failures classify False.

FIX 3 — partial-throttle pass-through (kill_switch.guard_orders_with_kill_switch):
    with the kill switch engaged at 0 < throttle_pct < 1, orders are RETURNED
    with quantities scaled by throttle_pct (count preserved when the floored qty
    stays >= 1 share), not dropped — distinct from the block-all (0.0) path and
    the unchanged pass-through (1.0) path. The audit record reflects
    throttled-not-blocked (action="GUARD", throttle_pct == the fractional value).

No production behaviour is changed by FIX 3 (test-only). FIX 1's only behaviour
change is the look-alike-substring tightening; real signatures stay byte-identical.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# FIX 1 — is_duplicate_error word-boundary tightening
# ---------------------------------------------------------------------------

# Real broker duplicate/idempotency rejections — MUST classify True. These are
# the four accepted signatures the recovery path in broker_adapter relies on.
_REAL_DUPLICATE_SIGNATURES = [
    # (1)+(2): explicit "duplicate" + an order/id reference.
    "400 Bad Request: duplicate client_order_id ata-abc123",
    "duplicate order",
    "Duplicate Order rejected by broker",
    # (3): "... already exists" guarded by an order/id reference.
    "order already exists",
    "422 Unprocessable Entity: client_order_id ata-deadbeef already exists",
    "422: order with that client_order_id already exists",
    # (4): Alpaca's same-intent resting-order 422.
    "potential wash trade detected",
    "403 Forbidden: potential wash trade",
]

# Benign look-alikes + generic/transient errors — MUST classify False. The first
# block proves the word-boundary tightening (substrings of "order" no longer
# count); the second block proves generic failures never masquerade as duplicates.
_NON_DUPLICATE_MESSAGES = [
    # Look-alike substrings of "order" combined with duplicate/already-exists.
    "reorder cancelled — duplicate request",
    "border crossing already exists in route table",
    "recording error: duplicate frame already exists",
    "ordering service duplicate already exists",
    "disorder in queue, entry already exists",
    # Generic / transient failures (no order/id reference at all).
    "503 service unavailable",
    "timeout while contacting broker",
    "connection reset by peer",
    "429 too many requests: rate limit exceeded",
    "insufficient buying power",
    # "duplicate" / "already exists" with NO order reference — must not fire.
    "duplicate session token",
    "account record already exists",
]


@pytest.mark.parametrize("msg", _REAL_DUPLICATE_SIGNATURES)
def test_is_duplicate_error_matches_real_broker_signatures(msg: str) -> None:
    from src.assembled_core.execution.idempotency import is_duplicate_error

    assert is_duplicate_error(msg) is True, f"real duplicate not matched: {msg!r}"


@pytest.mark.parametrize("msg", _NON_DUPLICATE_MESSAGES)
def test_is_duplicate_error_rejects_lookalikes_and_transients(msg: str) -> None:
    from src.assembled_core.execution.idempotency import is_duplicate_error

    assert is_duplicate_error(msg) is False, f"false-positive on: {msg!r}"


def test_is_duplicate_error_case_insensitive_real_signature() -> None:
    """The classifier lower-cases the message; mixed-case real signatures match."""
    from src.assembled_core.execution.idempotency import is_duplicate_error

    assert is_duplicate_error("DUPLICATE CLIENT_ORDER_ID ata-XYZ") is True
    assert is_duplicate_error("Potential Wash Trade") is True


def test_is_duplicate_error_standalone_order_word_still_counts() -> None:
    """The word-boundary guard must still accept the STANDALONE word 'order'
    (the regression risk is over-tightening so that real 'order already exists'
    stops matching)."""
    from src.assembled_core.execution.idempotency import is_duplicate_error

    assert is_duplicate_error("the order already exists on file") is True
    assert is_duplicate_error("duplicate order book entry") is True


# ---------------------------------------------------------------------------
# FIX 3 — partial-throttle pass-through (0 < throttle_pct < 1)
# ---------------------------------------------------------------------------


def _isolate_kill_switch(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Point all kill-switch state at tmp_path so the test never touches real
    output/ops files and is order-independent."""
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_STATE", str(tmp_path / "state.json"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_AUDIT", str(tmp_path / "audit.jsonl"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_SENTINEL", str(tmp_path / ".sentinel"))
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)
    monkeypatch.setenv("OPERATOR_KILL_TOKEN", "t-token")


def test_guard_orders_partial_throttle_scales_and_preserves(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """0 < throttle_pct < 1: orders are RETURNED with qty scaled by throttle_pct,
    count preserved when floored qty stays >= 1 share (NOT block-all, NOT
    pass-through). Scaling is floor-with-sign per the documented whole-share
    semantics: qty 40 * 0.25 = 10; qty -80 * 0.25 = -20.
    """
    _isolate_kill_switch(monkeypatch, tmp_path)

    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
        guard_orders_with_kill_switch,
    )

    activate_kill_switch(throttle_pct=0.25, reason="partial-throttle", actor="t")
    orders = pd.DataFrame(
        [
            {"symbol": "AAA", "qty": 40.0, "side": "BUY"},  # 40 * .25 = 10
            {"symbol": "BBB", "qty": -80.0, "side": "SELL"},  # -80 * .25 = -20
            {"symbol": "CCC", "qty": 8.0, "side": "BUY"},  # 8 * .25 = 2
        ]
    )
    result = guard_orders_with_kill_switch(orders)

    # Count preserved — none floored to zero, so nothing dropped.
    assert len(result) == 3, "partial throttle must NOT drop orders that floor >= 1"
    by_sym = {row["symbol"]: row["qty"] for _, row in result.iterrows()}
    assert by_sym["AAA"] == 10.0
    assert by_sym["BBB"] == -20.0  # short sign preserved
    assert by_sym["CCC"] == 2.0

    deactivate_kill_switch(reason="done", actor="t", operator_token="t-token")


def test_guard_orders_partial_throttle_audit_reflects_throttled_not_blocked(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """The audit record for a partial throttle must carry the FRACTIONAL
    throttle_pct (throttled), not a 0.0 block-all marker, and keep the order
    count under the GUARD action."""
    _isolate_kill_switch(monkeypatch, tmp_path)
    audit_path = tmp_path / "audit.jsonl"

    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
        guard_orders_with_kill_switch,
    )

    activate_kill_switch(throttle_pct=0.5, reason="partial", actor="t")
    orders = pd.DataFrame([{"symbol": "AAA", "qty": 10.0, "side": "BUY"}])
    result = guard_orders_with_kill_switch(orders)
    assert len(result) == 1 and result["qty"].iloc[0] == 5.0

    records = [
        json.loads(line)
        for line in audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    guard_recs = [r for r in records if r.get("action") == "GUARD"]
    assert guard_recs, "guard_orders must append a GUARD audit record"
    last_guard = guard_recs[-1]
    assert last_guard["throttle_pct"] == 0.5, "audit must reflect fractional throttle"
    assert last_guard["throttle_pct"] != 0.0, "must NOT look like a block-all"
    assert last_guard["orders_count"] == 1

    deactivate_kill_switch(reason="done", actor="t", operator_token="t-token")


def test_guard_orders_block_all_vs_partial_vs_passthrough_contrast(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Contrast the three engaged regimes on the SAME order set:
    0.0 -> empty frame; 0 < p < 1 -> scaled+preserved; 1.0 -> unchanged.
    """
    from src.assembled_core.execution.kill_switch import (
        activate_kill_switch,
        deactivate_kill_switch,
        guard_orders_with_kill_switch,
    )

    base = pd.DataFrame(
        [
            {"symbol": "AAA", "qty": 20.0, "side": "BUY"},
            {"symbol": "BBB", "qty": 16.0, "side": "BUY"},
        ]
    )

    # block-all (0.0)
    _isolate_kill_switch(monkeypatch, tmp_path / "block")
    activate_kill_switch(throttle_pct=0.0, reason="block", actor="t")
    blocked = guard_orders_with_kill_switch(base.copy())
    assert blocked.empty
    deactivate_kill_switch(reason="d", actor="t", operator_token="t-token")

    # partial (0.5)
    _isolate_kill_switch(monkeypatch, tmp_path / "partial")
    activate_kill_switch(throttle_pct=0.5, reason="partial", actor="t")
    partial = guard_orders_with_kill_switch(base.copy())
    assert len(partial) == 2
    assert sorted(partial["qty"].tolist()) == [8.0, 10.0]
    deactivate_kill_switch(reason="d", actor="t", operator_token="t-token")

    # pass-through (1.0)
    _isolate_kill_switch(monkeypatch, tmp_path / "pass")
    activate_kill_switch(throttle_pct=1.0, reason="pass", actor="t")
    passed = guard_orders_with_kill_switch(base.copy())
    assert len(passed) == 2
    assert sorted(passed["qty"].tolist()) == [16.0, 20.0]
    deactivate_kill_switch(reason="d", actor="t", operator_token="t-token")
