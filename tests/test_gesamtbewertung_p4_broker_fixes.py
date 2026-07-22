"""Regression guards for the 2026-07-22 GESAMTBEWERTUNG P4 broker-path fixes.

K4  — convert_broker_fills_to_ledger_format books ANY executed quantity
      (partially_filled / timed-out / cancelled with filled_qty > 0), not
      only status=="filled". Silently dropped partial fills were a
      systematic ledger-vs-broker drift source.
K2b — execute_via_broker cancels timed-out orders IN-RUN (per-order
      cancel) and refreshes their status once so a last-moment partial
      fill still reaches the ledger. Root cause of the 2026-07-14
      after-hours fills / 2026-07-20 reconcile halt.
W8  — the order-lifecycle log is wired into the REAL broker path
      (SUBMITTED / FILLED / REJECTED / PARTIAL_FILL|CANCELLED events).
K3  — AlpacaAdapter now exposes cancel_order (feature-detected by the
      preflight stale-order cleanup).
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.execution.broker_adapter import AlpacaAdapter, BrokerOrder
from src.assembled_core.execution.broker_execution import (
    convert_broker_fills_to_ledger_format,
    execute_via_broker,
)

pytestmark = pytest.mark.fast


def _order(
    order_id: str,
    symbol: str,
    status: str,
    qty: float = 10.0,
    filled_qty: float = 0.0,
    filled_avg_price: float | None = None,
) -> BrokerOrder:
    return BrokerOrder(
        order_id=order_id,
        symbol=symbol,
        side="buy",
        qty=qty,
        order_type="market",
        status=status,
        filled_qty=filled_qty,
        filled_avg_price=filled_avg_price,
    )


# ---------------------------------------------------------------------------
# K4 — partial fills reach the ledger
# ---------------------------------------------------------------------------


def test_k4_partially_filled_order_is_booked(tmp_path):
    fills = convert_broker_fills_to_ledger_format(
        [
            _order(
                "o1",
                "AAPL",
                "partially_filled",
                qty=10,
                filled_qty=4,
                filled_avg_price=210.5,
            )
        ],
        intent_store_path=str(tmp_path / "intents.jsonl"),
    )
    assert fills == [{"symbol": "AAPL", "side": "BUY", "qty": 4, "price": 210.5}]


def test_k4_timed_out_with_partial_fill_is_booked(tmp_path):
    fills = convert_broker_fills_to_ledger_format(
        [
            _order(
                "o2", "MSFT", "accepted", qty=10, filled_qty=2.5, filled_avg_price=99.0
            )
        ],
        intent_store_path=str(tmp_path / "intents.jsonl"),
    )
    assert fills == [{"symbol": "MSFT", "side": "BUY", "qty": 2.5, "price": 99.0}]


def test_k4_nothing_executed_is_not_booked(tmp_path):
    fills = convert_broker_fills_to_ledger_format(
        [
            _order("o3", "TSLA", "accepted", filled_qty=0.0),
            _order("o4", "NVDA", "rejected", filled_qty=0.0),
        ],
        intent_store_path=str(tmp_path / "intents.jsonl"),
    )
    assert fills == []


def test_k4_partial_without_price_is_skipped(tmp_path):
    # A partial fill without a price must NOT book a zero-cost fill.
    fills = convert_broker_fills_to_ledger_format(
        [
            _order(
                "o5", "AMD", "partially_filled", filled_qty=3.0, filled_avg_price=None
            )
        ],
        intent_store_path=str(tmp_path / "intents.jsonl"),
    )
    assert fills == []


# ---------------------------------------------------------------------------
# K2b + W8 — in-run cancel of timed-out orders, lifecycle wiring
# ---------------------------------------------------------------------------


class _FakeAdapter:
    """Simulates the timeout path honestly: the order stays 'accepted'
    (non-terminal) during the fill poll, and only AFTER a cancel attempt
    does get_order_status report ``refresh_status``. (The first version
    returned the terminal refresh status already during the poll — which
    only worked because of the E-055 canceled/cancelled drift.)"""

    def __init__(self):
        self.cancelled: list[str] = []
        self.cancel_attempted = False
        self.refresh_status = "canceled"
        self.refresh_filled_qty = 0.0
        self.refresh_price: float | None = None
        self._n = 0

    def submit_market_order(self, symbol, side, qty, client_order_id=None):
        self._n += 1
        return _order(f"fake-{self._n}", symbol, "accepted", qty=qty)

    def get_order_status(self, order_id: str) -> BrokerOrder:
        status = self.refresh_status if self.cancel_attempted else "accepted"
        return _order(
            order_id,
            "ZZZ",
            status,
            filled_qty=self.refresh_filled_qty,
            filled_avg_price=self.refresh_price,
        )

    def cancel_order(self, order_id: str) -> bool:
        self.cancel_attempted = True
        self.cancelled.append(order_id)
        return True


def _run(adapter, tmp_path, monkeypatch, lifecycle_path):
    import src.assembled_core.ops.order_lifecycle_log as lc

    # Stage-1 M1: raising=True — a wrong constant name here silently no-ops
    # the patch and the test contaminates the REAL C1 audit journal
    # (happened 2026-07-22 with "DEFAULT_LOG_PATH"; cleaned up same day).
    monkeypatch.setattr(lc, "DEFAULT_LIFECYCLE_LOG_PATH", lifecycle_path)
    orders_df = pd.DataFrame([{"symbol": "ZZZ", "side": "BUY", "qty": 7.0}])
    return execute_via_broker(
        adapter,
        orders_df,
        dry_run=False,
        timeout_s=0.1,
        poll_interval_s=0.05,
        intent_store_path=str(tmp_path / "intents.jsonl"),
    )


def test_k2b_timed_out_orders_are_cancelled_in_run(tmp_path, monkeypatch):
    adapter = _FakeAdapter()
    result = _run(adapter, tmp_path, monkeypatch, tmp_path / "lifecycle.jsonl")
    assert len(adapter.cancelled) == 1  # the timed-out order was cancelled
    assert result.fills_for_ledger == []  # nothing executed


def test_k2b_last_moment_partial_fill_reaches_ledger(tmp_path, monkeypatch):
    adapter = _FakeAdapter()
    adapter.refresh_status = "canceled"
    adapter.refresh_filled_qty = 3.0
    adapter.refresh_price = 55.5
    result = _run(adapter, tmp_path, monkeypatch, tmp_path / "lifecycle.jsonl")
    assert adapter.cancelled  # cancel happened
    # The post-cancel refresh captured the partial fill -> booked (K4+K2b).
    assert result.fills_for_ledger == [
        {"symbol": "ZZZ", "side": "BUY", "qty": 3.0, "price": 55.5}
    ]


def test_w8_lifecycle_events_written_for_broker_path(tmp_path, monkeypatch):
    import src.assembled_core.ops.order_lifecycle_log as lc

    events: list[tuple] = []

    def _capture(event_type, order_id, symbol, side, qty, **kw):
        events.append((event_type, symbol))

    import src.assembled_core.execution.broker_execution as be

    monkeypatch.setattr(lc, "append_lifecycle_event", _capture)  # raising default
    # Patching lc's module attribute works because broker_execution does a
    # function-local `from ...order_lifecycle_log import append_lifecycle_event`
    # at call time, which re-binds from lc's (patched) namespace.
    adapter = _FakeAdapter()
    orders_df = pd.DataFrame([{"symbol": "ZZZ", "side": "BUY", "qty": 7.0}])
    be.execute_via_broker(
        adapter,
        orders_df,
        dry_run=False,
        timeout_s=0.1,
        poll_interval_s=0.05,
        intent_store_path=str(tmp_path / "intents.jsonl"),
    )
    kinds = [e[0] for e in events]
    assert "SUBMITTED" in kinds
    assert any(k in kinds for k in ("CANCELLED", "PARTIAL_FILL"))


def test_k2b_cancel_raises_but_race_fill_is_still_booked(tmp_path, monkeypatch):
    # Cancel fails (order already filled at the broker) -> the bounded
    # re-poll sees status=filled and the full fill reaches the ledger,
    # labelled FILLED (not PARTIAL_FILL) in the journal.
    class _RaceAdapter(_FakeAdapter):
        """Order stays 'accepted' during the poll (-> timeout), the cancel
        attempt raises (already terminal at the broker), and only AFTER the
        cancel attempt does the refresh reveal the race-winning full fill."""

        def __init__(self):
            super().__init__()
            self.cancel_attempted = False
            self.refresh_status = "accepted"

        def cancel_order(self, order_id: str) -> bool:
            self.cancel_attempted = True
            raise RuntimeError("order already in terminal state (422)")

        def get_order_status(self, order_id: str) -> BrokerOrder:
            if self.cancel_attempted:
                return _order(
                    order_id,
                    "ZZZ",
                    "filled",
                    qty=7.0,
                    filled_qty=7.0,
                    filled_avg_price=42.0,
                )
            return _order(order_id, "ZZZ", "accepted", qty=7.0)

    adapter = _RaceAdapter()

    import src.assembled_core.ops.order_lifecycle_log as lc

    events: list[tuple] = []
    monkeypatch.setattr(
        lc,
        "append_lifecycle_event",
        lambda event_type, order_id, symbol, side, qty, **kw: events.append(
            (event_type, kw.get("reason"))
        ),
    )
    orders_df = pd.DataFrame([{"symbol": "ZZZ", "side": "BUY", "qty": 7.0}])
    result = execute_via_broker(
        adapter,
        orders_df,
        dry_run=False,
        timeout_s=0.1,
        poll_interval_s=0.05,
        intent_store_path=str(tmp_path / "intents.jsonl"),
    )
    assert result.fills_for_ledger == [
        {"symbol": "ZZZ", "side": "BUY", "qty": 7.0, "price": 42.0}
    ]
    assert ("FILLED", "filled_during_cancel_race") in events


def test_k2b_adapter_without_cancel_order_books_partial_and_logs(tmp_path, monkeypatch):
    class _NoCancelAdapter(_FakeAdapter):
        # Feature-detection must not find a callable cancel_order here —
        # the inherited method is masked (callable(None) is False).
        cancel_order = None

    adapter = _NoCancelAdapter()
    adapter.refresh_status = "accepted"  # still live, never terminal
    adapter.refresh_filled_qty = 2.0
    adapter.refresh_price = 10.0
    result = _run(adapter, tmp_path, monkeypatch, tmp_path / "lifecycle.jsonl")
    # Partial still booked; order remains live (backstop = reconcile halt).
    assert result.fills_for_ledger == [
        {"symbol": "ZZZ", "side": "BUY", "qty": 2.0, "price": 10.0}
    ]


def test_e055_broker_canceled_single_l_is_terminal(tmp_path, monkeypatch):
    # E-055: Alpaca emits "canceled" (single-l). Before the fix this status
    # was not in the terminal set -> the order polled until timeout and
    # landed in timed_out; now it must be terminal and categorised rejected.
    class _CanceledAdapter(_FakeAdapter):
        def submit_market_order(self, symbol, side, qty, client_order_id=None):
            self._n += 1
            return _order(f"fake-{self._n}", symbol, "canceled", qty=qty)

    adapter = _CanceledAdapter()
    result = _run(adapter, tmp_path, monkeypatch, tmp_path / "lifecycle.jsonl")
    assert len(result.rejected) == 1
    assert result.timed_out == []
    assert adapter.cancelled == []  # no cancel attempt on a terminal order


# ---------------------------------------------------------------------------
# W11b (Stage-1 B1 regression) — turnover-gate crash must degrade, not raise
# ---------------------------------------------------------------------------


def test_w11b_turnover_gate_crash_degrades_instead_of_raising():
    from types import SimpleNamespace

    import logging

    from src.assembled_core.pipeline._tc_sizing import _sp_apply_turnover_gate

    target = pd.DataFrame(
        {"symbol": ["AAA"], "target_qty": [100.0], "target_weight": [0.5]}
    )
    ctx = SimpleNamespace(current_positions=None, capital=100_000.0)
    policy = {"turnover_budget": {"enabled": True, "cap": "boom"}}  # forces crash
    meta: dict = {}
    out = _sp_apply_turnover_gate(
        target,
        ctx,
        None,
        None,
        policy,
        logging.getLogger("test_w11b"),
        meta,
    )
    # No raise (fail-open kept), positions unchanged, degradation OBSERVABLE.
    assert out is not None and len(out) == 1
    degraded = meta.get("degraded_steps") or []
    assert any("turnover_budget_gate" in str(d) for d in degraded), (
        f"degraded_steps missing turnover_budget_gate: {meta}"
    )


# ---------------------------------------------------------------------------
# K3 — adapter exposes per-order cancel (feature-detection contract)
# ---------------------------------------------------------------------------


def test_k3_alpaca_adapter_has_cancel_order():
    assert callable(getattr(AlpacaAdapter, "cancel_order", None))
