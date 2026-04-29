"""Tests for execution/order_management.py (spec 33_EXECUTION_ORDERMANAGEMENT)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from assembled_core.execution.order_management import (
    BarSnapshot,
    ExecutionCostModel,
    PartialFillPolicy,
    position_reconcile_before_signal,
    submit_with_idempotency,
)


# ---------------------------------------------------------------------------
# submit_with_idempotency
# ---------------------------------------------------------------------------

def _intent(signal_id="sig-001", symbol="AAPL", side="buy", qty=100,
            order_type="market", tif="day", limit_price=None):
    from types import SimpleNamespace
    from assembled_core.execution.idempotency import compute_intent_hash
    intent_hash = compute_intent_hash(symbol, side, qty, order_type, limit_price)
    return SimpleNamespace(
        signal_id=signal_id, intent_hash=intent_hash, symbol=symbol,
        qty=qty, side=side, order_type=order_type, tif=tif, limit_price=limit_price,
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
        client.get_order_by_client_order_id.return_value = {"id": "order-1", "status": "filled"}
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
        assert PartialFillPolicy.classify(
            self._now() - timedelta(seconds=30), 100, 40
        ) == "wait"

    def test_partial_accepted_above_ratio(self):
        assert PartialFillPolicy.classify(
            self._now() - timedelta(seconds=200), 100, 60
        ) == "partial_accepted"

    def test_partial_failed_below_ratio(self):
        assert PartialFillPolicy.classify(
            self._now() - timedelta(seconds=200), 100, 20
        ) == "partial_failed"

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
        assert position_reconcile_before_signal("AAPL", 100, 99.5, min_trade_size=1.0) is None

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
