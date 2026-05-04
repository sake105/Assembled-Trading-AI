"""Tests for PDT execution modules (spec 41_PDT_REGEL_INTRADAY_MARGIN)."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace


def _noon_utc() -> datetime:
    """Return today's noon UTC at call time — avoids midnight-crossing when the
    module is imported hours before the tests run (e.g. long full-suite runs)."""
    return datetime.now(timezone.utc).replace(
        hour=12, minute=0, second=0, microsecond=0
    )


from assembled_core.execution.pdt_tracker import DayTrade, PDTTracker
from assembled_core.execution.round_trip_detector import RoundTripDetector
from assembled_core.execution.order_gate import OrderDecision, OrderGate
from assembled_core.execution.migration_detector import PDTMigrationDetector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trade(ticker="AAPL", side="long", days_ago=0, qty=100, entry=100.0, exit_=105.0):
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return DayTrade(
        ticker=ticker,
        open_timestamp=ts,
        close_timestamp=ts + timedelta(hours=1),
        side=side,
        quantity=qty,
        entry_price=entry,
        exit_price=exit_,
    )


def _fill(ticker, side, qty=100, price=100.0, ts=None):
    if ts is None:
        ts = datetime.now(timezone.utc)
    return SimpleNamespace(
        ticker=ticker, side=side, quantity=qty, price=price, timestamp=ts
    )


# ---------------------------------------------------------------------------
# DayTrade
# ---------------------------------------------------------------------------


class TestDayTrade:
    def test_trade_date(self):
        t = _trade()
        assert isinstance(t.trade_date, date)

    def test_pnl_long(self):
        t = DayTrade(
            "A",
            datetime.now(tz=timezone.utc),
            datetime.now(tz=timezone.utc),
            "long",
            100,
            100.0,
            110.0,
        )
        assert abs(t.pnl - 1000.0) < 0.01

    def test_pnl_short(self):
        t = DayTrade(
            "A",
            datetime.now(tz=timezone.utc),
            datetime.now(tz=timezone.utc),
            "short",
            100,
            110.0,
            100.0,
        )
        assert abs(t.pnl - 1000.0) < 0.01


# ---------------------------------------------------------------------------
# PDTTracker
# ---------------------------------------------------------------------------


class TestPDTTracker:
    def test_no_trades_count_zero(self):
        tracker = PDTTracker(account_equity=10_000.0)
        assert tracker.count_recent_day_trades() == 0

    def test_record_and_count(self):
        tracker = PDTTracker(account_equity=10_000.0)
        tracker.record_day_trade(_trade(days_ago=0))
        tracker.record_day_trade(_trade(ticker="MSFT", days_ago=1))
        assert tracker.count_recent_day_trades() == 2

    def test_old_trades_excluded(self):
        tracker = PDTTracker(account_equity=10_000.0)
        tracker.record_day_trade(_trade(days_ago=20))  # outside 5-day window
        assert tracker.count_recent_day_trades() == 0

    def test_would_violate_pdt_at_limit(self):
        tracker = PDTTracker(account_equity=10_000.0)
        for i in range(3):
            tracker.record_day_trade(_trade(ticker=f"T{i}", days_ago=0))
        assert tracker.would_violate_pdt() is True

    def test_would_not_violate_below_limit(self):
        tracker = PDTTracker(account_equity=10_000.0)
        for i in range(2):
            tracker.record_day_trade(_trade(ticker=f"T{i}", days_ago=0))
        assert tracker.would_violate_pdt() is False

    def test_above_25k_never_violates(self):
        tracker = PDTTracker(account_equity=30_000.0)
        for i in range(5):
            tracker.record_day_trade(_trade(ticker=f"T{i}", days_ago=0))
        assert tracker.would_violate_pdt() is False

    def test_disabled_never_violates(self):
        tracker = PDTTracker(account_equity=5_000.0, enabled=False)
        for i in range(5):
            tracker.record_day_trade(_trade(ticker=f"T{i}", days_ago=0))
        assert tracker.would_violate_pdt() is False

    def test_days_until_reset_zero_when_no_trades(self):
        tracker = PDTTracker(account_equity=10_000.0)
        assert tracker.days_until_pdt_reset() == 0

    def test_days_until_reset_positive_with_trades(self):
        tracker = PDTTracker(account_equity=10_000.0)
        tracker.record_day_trade(_trade(days_ago=1))
        days = tracker.days_until_pdt_reset()
        assert days >= 0


# ---------------------------------------------------------------------------
# RoundTripDetector
# ---------------------------------------------------------------------------


class TestRoundTripDetector:
    def _make_detector(self, equity=10_000.0):
        tracker = PDTTracker(account_equity=equity)
        return tracker, RoundTripDetector(tracker)

    def test_buy_open_no_day_trade(self):
        tracker, detector = self._make_detector()
        result = detector.on_fill(_fill("AAPL", "buy"))
        assert result is None
        assert tracker.count_recent_day_trades() == 0

    def test_buy_then_sell_same_day_is_day_trade(self):
        tracker, detector = self._make_detector()
        ts = _noon_utc()
        detector.on_fill(_fill("AAPL", "buy", ts=ts))
        trade = detector.on_fill(
            _fill("AAPL", "sell", price=110.0, ts=ts + timedelta(hours=1))
        )
        assert trade is not None
        assert trade.ticker == "AAPL"
        assert trade.side == "long"
        assert tracker.count_recent_day_trades() == 1

    def test_buy_different_day_no_day_trade(self):
        tracker, detector = self._make_detector()
        yesterday = _noon_utc() - timedelta(days=1)
        today = _noon_utc()
        detector.on_fill(_fill("AAPL", "buy", ts=yesterday))
        trade = detector.on_fill(_fill("AAPL", "sell", ts=today))
        assert trade is None
        assert tracker.count_recent_day_trades() == 0

    def test_adding_to_position_not_day_trade(self):
        tracker, detector = self._make_detector()
        ts = _noon_utc()
        detector.on_fill(_fill("AAPL", "buy", qty=100, ts=ts))
        result = detector.on_fill(
            _fill("AAPL", "buy", qty=50, price=105.0, ts=ts + timedelta(minutes=30))
        )
        assert result is None
        assert tracker.count_recent_day_trades() == 0

    def test_partial_close_one_day_trade(self):
        tracker, detector = self._make_detector()
        ts = _noon_utc()
        detector.on_fill(_fill("AAPL", "buy", qty=500, ts=ts))
        detector.on_fill(_fill("AAPL", "sell", qty=100, ts=ts + timedelta(hours=1)))
        detector.on_fill(_fill("AAPL", "sell", qty=400, ts=ts + timedelta(hours=2)))
        assert tracker.count_recent_day_trades() == 1

    def test_short_round_trip(self):
        tracker, detector = self._make_detector()
        ts = _noon_utc()
        detector.on_fill(_fill("TSLA", "sell", ts=ts))  # short
        trade = detector.on_fill(
            _fill("TSLA", "buy", price=95.0, ts=ts + timedelta(hours=2))
        )
        assert trade is not None
        assert trade.side == "short"


# ---------------------------------------------------------------------------
# OrderGate
# ---------------------------------------------------------------------------


class TestOrderGate:
    def _gate(self, equity=10_000.0, n_existing_trades=0):
        tracker = PDTTracker(account_equity=equity)
        detector = RoundTripDetector(tracker)
        for i in range(n_existing_trades):
            t = _trade(ticker=f"SYM{i}", days_ago=0)
            tracker.record_day_trade(t)
        return OrderGate(tracker, detector), tracker, detector

    def test_allowed_when_no_open_position(self):
        gate, _, _ = self._gate()
        result = gate.check_order("AAPL", "sell", 100)
        assert result.decision == OrderDecision.ALLOWED

    def test_allowed_when_open_position_but_below_limit(self):
        gate, tracker, detector = self._gate(n_existing_trades=2)
        detector.on_fill(_fill("AAPL", "buy", ts=_noon_utc()))
        result = gate.check_order("AAPL", "sell", 100)
        assert result.decision == OrderDecision.ALLOWED

    def test_blocked_pdt_at_limit(self):
        gate, tracker, detector = self._gate(n_existing_trades=3)
        detector.on_fill(_fill("AAPL", "buy", ts=_noon_utc()))
        result = gate.check_order("AAPL", "sell", 100)
        assert result.decision == OrderDecision.BLOCKED_PDT
        assert result.suggested_action is not None

    def test_allowed_above_25k_equity(self):
        gate, _, detector = self._gate(equity=30_000.0, n_existing_trades=5)
        detector.on_fill(_fill("AAPL", "buy", ts=_noon_utc()))
        result = gate.check_order("AAPL", "sell", 100)
        assert result.decision == OrderDecision.ALLOWED

    def test_disabled_tracker_always_allowed(self):
        tracker = PDTTracker(account_equity=5_000.0, enabled=False)
        for i in range(5):
            tracker.record_day_trade(_trade(ticker=f"T{i}", days_ago=0))
        detector = RoundTripDetector(tracker)
        gate = OrderGate(tracker, detector)
        detector.on_fill(_fill("AAPL", "buy", ts=_noon_utc()))
        result = gate.check_order("AAPL", "sell", 100)
        assert result.decision == OrderDecision.ALLOWED


# ---------------------------------------------------------------------------
# PDTMigrationDetector
# ---------------------------------------------------------------------------


class TestPDTMigrationDetector:
    def test_no_data_not_migrated(self):
        md = PDTMigrationDetector()
        assert md.likely_migrated() is False

    def test_blocks_present_not_migrated(self):
        md = PDTMigrationDetector()
        for _ in range(5):
            md.record_fourth_day_trade_attempt()
            md.record_pdt_block()
        assert md.likely_migrated() is False

    def test_attempts_without_blocks_migrated(self):
        md = PDTMigrationDetector()
        for _ in range(3):
            md.record_fourth_day_trade_attempt()
        assert md.likely_migrated() is True

    def test_old_attempts_excluded(self):
        md = PDTMigrationDetector(observation_window_days=7)
        old = datetime.now(timezone.utc) - timedelta(days=30)
        for _ in range(5):
            md.record_fourth_day_trade_attempt(old)
        assert md.likely_migrated() is False

    def test_record_pdt_block_resets_signal(self):
        md = PDTMigrationDetector()
        for _ in range(3):
            md.record_fourth_day_trade_attempt()
        md.record_pdt_block()
        assert md.likely_migrated() is False
