"""Tests for execution/order_management.py (spec 33_EXECUTION_ORDERMANAGEMENT)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock


from assembled_core.execution.order_management import (
    BarSnapshot,
    ExecutionCostModel,
    ExitManager,
    OrderStatusStream,
    PartialFillPolicy,
    PositionRecord,
    handle_rejection,
    has_recent_loss_close,
    position_reconcile_before_signal,
    reconcile_cash,
    reconcile_positions,
    submit_with_idempotency,
)

# ---------------------------------------------------------------------------
# submit_with_idempotency
# ---------------------------------------------------------------------------


def _intent(
    signal_id="sig-001",
    symbol="AAPL",
    side="buy",
    qty=100,
    order_type="market",
    tif="day",
    limit_price=None,
):
    from types import SimpleNamespace
    from assembled_core.execution.idempotency import compute_intent_hash

    intent_hash = compute_intent_hash(symbol, side, qty, order_type, limit_price)
    return SimpleNamespace(
        signal_id=signal_id,
        intent_hash=intent_hash,
        symbol=symbol,
        qty=qty,
        side=side,
        order_type=order_type,
        tif=tif,
        limit_price=limit_price,
    )


class TestSubmitWithIdempotency:
    def test_submitted_on_success(self):
        client = MagicMock()
        client.submit_order.return_value = {"id": "order-1"}
        status, resp, err = submit_with_idempotency(client, _intent())
        assert status == "submitted"
        assert resp["id"] == "order-1"
        assert err is None

    def test_already_submitted_on_duplicate(self):
        client = MagicMock()
        client.submit_order.side_effect = Exception("duplicate client_order_id")
        client.get_order_by_client_order_id.return_value = {
            "id": "order-1",
            "status": "filled",
        }
        status, resp, err = submit_with_idempotency(client, _intent())
        assert status == "already_submitted"
        assert resp["status"] == "filled"

    def test_rejected_on_insufficient_funds(self):
        client = MagicMock()
        client.submit_order.side_effect = Exception("insufficient buying power")
        status, resp, err = submit_with_idempotency(client, _intent())
        assert status == "rejected"
        assert "insufficient" in err

    def test_rejected_on_pdt(self):
        client = MagicMock()
        client.submit_order.side_effect = Exception("pattern day trading 403")
        status, resp, err = submit_with_idempotency(client, _intent())
        assert status == "rejected"
        assert "pdt" in err


# ---------------------------------------------------------------------------
# PartialFillPolicy
# ---------------------------------------------------------------------------


class TestPartialFillPolicy:
    def _now(self):
        return datetime.now(timezone.utc)

    def test_complete(self):
        assert PartialFillPolicy.classify(self._now(), 100, 100) == "complete"

    def test_wait_within_window(self):
        assert (
            PartialFillPolicy.classify(self._now() - timedelta(seconds=30), 100, 40)
            == "wait"
        )

    def test_partial_accepted_above_ratio(self):
        assert (
            PartialFillPolicy.classify(self._now() - timedelta(seconds=200), 100, 60)
            == "partial_accepted"
        )

    def test_partial_failed_below_ratio(self):
        assert (
            PartialFillPolicy.classify(self._now() - timedelta(seconds=200), 100, 20)
            == "partial_failed"
        )

    def test_cancel_threshold_constant(self):
        assert PartialFillPolicy.CANCEL_AFTER_SECONDS == 120

    def test_min_fill_ratio_constant(self):
        assert PartialFillPolicy.MIN_FILL_RATIO == 0.5


# ---------------------------------------------------------------------------
# position_reconcile_before_signal
# ---------------------------------------------------------------------------


class TestPositionReconcile:
    def test_no_delta_returns_none(self):
        assert position_reconcile_before_signal("AAPL", 100, 100) is None

    def test_tiny_delta_returns_none(self):
        assert (
            position_reconcile_before_signal("AAPL", 100, 99.5, min_trade_size=1.0)
            is None
        )

    def test_small_pct_delta_returns_none(self):
        # 5% delta < 10% threshold
        assert position_reconcile_before_signal("AAPL", 100, 95) is None

    def test_large_delta_returns_order(self):
        result = position_reconcile_before_signal("AAPL", 100, 70)
        assert result is not None
        assert result["symbol"] == "AAPL"
        assert abs(result["delta"] - 30) < 0.01

    def test_negative_delta(self):
        result = position_reconcile_before_signal("AAPL", 50, 100)
        assert result is not None
        assert result["delta"] < 0


# ---------------------------------------------------------------------------
# ExecutionCostModel
# ---------------------------------------------------------------------------


class TestExecutionCostModel:
    def setup_method(self):
        self.model = ExecutionCostModel()

    def test_buy_fill_above_close(self):
        bar = BarSnapshot(close=100.0, realized_vol_20d=0.02, adv=50_000_000)
        fill = self.model.estimate_fill("buy", "AAPL", 1000, bar)
        assert fill > 100.0

    def test_sell_fill_below_close(self):
        bar = BarSnapshot(close=100.0, realized_vol_20d=0.02, adv=50_000_000)
        fill = self.model.estimate_fill("sell", "AAPL", 1000, bar)
        assert fill < 100.0

    def test_large_cap_lower_spread(self):
        bar_large = BarSnapshot(close=100.0, realized_vol_20d=0.01, adv=200_000_000)
        bar_small = BarSnapshot(close=100.0, realized_vol_20d=0.01, adv=5_000_000)
        fill_large = self.model.estimate_fill("buy", "A", 100, bar_large)
        fill_small = self.model.estimate_fill("buy", "B", 100, bar_small)
        assert fill_large < fill_small

    def test_high_participation_increases_fill(self):
        bar_light = BarSnapshot(close=100.0, adv=100_000_000)
        bar_heavy = BarSnapshot(close=100.0, adv=1_000_000)
        fill_light = self.model.estimate_fill("buy", "X", 100, bar_light)
        fill_heavy = self.model.estimate_fill("buy", "X", 100, bar_heavy)
        assert fill_heavy > fill_light

    def test_spread_large_cap(self):
        bar = BarSnapshot(close=100.0, realized_vol_20d=0.0, adv=200_000_000)
        spread = ExecutionCostModel._get_spread("X", bar)
        assert abs(spread - 1.0) < 0.01

    def test_spread_small_cap(self):
        bar = BarSnapshot(close=100.0, realized_vol_20d=0.0, adv=1_000_000)
        spread = ExecutionCostModel._get_spread("X", bar)
        assert abs(spread - 10.0) < 0.01


# ---------------------------------------------------------------------------
# handle_rejection
# ---------------------------------------------------------------------------


class TestHandleRejection:
    def test_known_reason_returns_action(self):
        result = handle_rejection("insufficient_buying_power", "AAPL")
        assert result["action"] == "pause_symbol_30min"
        assert result["symbol"] == "AAPL"

    def test_unknown_reason_log_and_skip(self):
        result = handle_rejection("unknown_reason", "MSFT")
        assert result["action"] == "log_and_skip"

    def test_wash_sale_action(self):
        result = handle_rejection("wash_sale_block", "TSLA")
        assert result["action"] == "mark_wash_sale"

    def test_pdt_action(self):
        result = handle_rejection("pdt_restriction", "AAPL")
        assert result["action"] == "alert_pdt_counter_bug"

    def test_short_not_available(self):
        result = handle_rejection("short_not_available", "GME")
        assert result["action"] == "cache_unshortable_24h"


# ---------------------------------------------------------------------------
# has_recent_loss_close
# ---------------------------------------------------------------------------


class TestHasRecentLossClose:
    def _closed(self, symbol, days_ago, pnl):
        return {
            "symbol": symbol,
            "closed_at": datetime.now(timezone.utc) - timedelta(days=days_ago),
            "realized_pnl": pnl,
        }

    def test_recent_loss_detected(self):
        rows = [self._closed("AAPL", 10, -500.0)]
        assert has_recent_loss_close("AAPL", rows, days=30) is True

    def test_old_loss_not_detected(self):
        rows = [self._closed("AAPL", 40, -500.0)]
        assert has_recent_loss_close("AAPL", rows, days=30) is False

    def test_recent_gain_not_detected(self):
        rows = [self._closed("AAPL", 5, 200.0)]
        assert has_recent_loss_close("AAPL", rows, days=30) is False

    def test_different_symbol_not_detected(self):
        rows = [self._closed("MSFT", 5, -100.0)]
        assert has_recent_loss_close("AAPL", rows, days=30) is False

    def test_empty_positions(self):
        assert has_recent_loss_close("AAPL", [], days=30) is False


# ---------------------------------------------------------------------------
# reconcile_positions / reconcile_cash
# ---------------------------------------------------------------------------


class TestReconcilePositions:
    def test_no_drift(self):
        broker = [{"symbol": "AAPL", "qty": 100}]
        internal = [{"symbol": "AAPL", "qty": 100}]
        assert reconcile_positions(broker, internal) == []

    def test_drift_detected(self):
        broker = [{"symbol": "AAPL", "qty": 100}]
        internal = [{"symbol": "AAPL", "qty": 90}]
        drifts = reconcile_positions(broker, internal)
        assert len(drifts) == 1
        assert drifts[0]["symbol"] == "AAPL"
        assert abs(drifts[0]["delta"] - 10) < 1e-6

    def test_broker_has_extra_position(self):
        broker = [{"symbol": "AAPL", "qty": 100}, {"symbol": "MSFT", "qty": 50}]
        internal = [{"symbol": "AAPL", "qty": 100}]
        drifts = reconcile_positions(broker, internal)
        assert any(d["symbol"] == "MSFT" for d in drifts)

    def test_internal_has_ghost_position(self):
        broker = [{"symbol": "AAPL", "qty": 100}]
        internal = [{"symbol": "AAPL", "qty": 100}, {"symbol": "GOOG", "qty": 30}]
        drifts = reconcile_positions(broker, internal)
        assert any(d["symbol"] == "GOOG" for d in drifts)


class TestReconcileCash:
    def test_no_drift_v2(self):
        assert reconcile_cash(10000.0, 10000.0) is None

    def test_small_drift_within_tolerance(self):
        assert reconcile_cash(10000.50, 10000.0) is None

    def test_large_drift_detected(self):
        result = reconcile_cash(10050.0, 10000.0)
        assert result is not None
        assert abs(result["delta"] - 50.0) < 1e-6

    def test_negative_drift(self):
        result = reconcile_cash(9990.0, 10000.0)
        assert result is not None
        assert result["delta"] < 0


# ---------------------------------------------------------------------------
# ExitManager
# ---------------------------------------------------------------------------


def _pos(symbol, entry, stop=None, pt=None, days=None, opened_days_ago=0, side="long"):
    return PositionRecord(
        symbol=symbol,
        qty=100,
        avg_entry_price=entry,
        stop_price=stop,
        profit_target_price=pt,
        max_holding_days=days,
        opened_at=datetime.now(timezone.utc) - timedelta(days=opened_days_ago),
        side=side,
    )


class TestExitManager:
    def test_stop_hit_long(self):
        mgr = ExitManager()
        signals = mgr.check_exits([_pos("AAPL", entry=100, stop=90)], {"AAPL": 85.0})
        assert len(signals) == 1
        assert signals[0].exit_reason == "stop_hit"

    def test_stop_not_hit(self):
        mgr = ExitManager()
        signals = mgr.check_exits([_pos("AAPL", entry=100, stop=90)], {"AAPL": 95.0})
        assert signals == []

    def test_profit_target_hit(self):
        mgr = ExitManager()
        signals = mgr.check_exits([_pos("AAPL", entry=100, pt=120)], {"AAPL": 125.0})
        assert len(signals) == 1
        assert signals[0].exit_reason == "pt_hit"

    def test_vertical_barrier_hit(self):
        mgr = ExitManager()
        signals = mgr.check_exits(
            [_pos("AAPL", entry=100, days=10, opened_days_ago=11)], {"AAPL": 102.0}
        )
        assert len(signals) == 1
        assert signals[0].exit_reason == "vertical_barrier"

    def test_vertical_barrier_not_yet(self):
        mgr = ExitManager()
        signals = mgr.check_exits(
            [_pos("AAPL", entry=100, days=10, opened_days_ago=5)], {"AAPL": 102.0}
        )
        assert signals == []

    def test_no_price_skipped(self):
        mgr = ExitManager()
        signals = mgr.check_exits([_pos("AAPL", entry=100, stop=90)], {})
        assert signals == []

    def test_short_stop_hit(self):
        mgr = ExitManager()
        signals = mgr.check_exits(
            [_pos("AAPL", entry=100, stop=110, side="short")], {"AAPL": 115.0}
        )
        assert len(signals) == 1
        assert signals[0].exit_reason == "stop_hit"

    def test_multiple_positions(self):
        mgr = ExitManager()
        positions = [
            _pos("AAPL", entry=100, stop=90),
            _pos("MSFT", entry=200, pt=250),
            _pos("GOOG", entry=150, days=5, opened_days_ago=3),
        ]
        prices = {"AAPL": 88.0, "MSFT": 260.0, "GOOG": 155.0}
        signals = mgr.check_exits(positions, prices)
        assert len(signals) == 2
        reasons = {s.exit_reason for s in signals}
        assert "stop_hit" in reasons
        assert "pt_hit" in reasons


# ---------------------------------------------------------------------------
# OrderStatusStream
# ---------------------------------------------------------------------------


class TestOrderStatusStream:
    def test_is_not_running_by_default(self):
        stream = OrderStatusStream()
        assert stream.is_running() is False

    def test_apply_known_event(self):
        stream = OrderStatusStream()
        event = {"event": "fill", "order": {"symbol": "AAPL", "status": "filled"}}
        result = stream.apply_event(event)
        assert result == "fill"

    def test_apply_unknown_event_returns_none(self):
        stream = OrderStatusStream()
        result = stream.apply_event({"event": "something_new", "order": {}})
        assert result is None

    def test_apply_partial_fill(self):
        stream = OrderStatusStream()
        result = stream.apply_event({"event": "partial_fill", "order": {}})
        assert result == "partial_fill"

    def test_poll_interval_constant(self):
        assert OrderStatusStream.POLL_INTERVAL_SECONDS == 30

    def test_reconcile_interval_constant(self):
        assert OrderStatusStream.RECONCILE_INTERVAL_SECONDS == 300
