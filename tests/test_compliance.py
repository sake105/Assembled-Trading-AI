"""Tests for src/assembled_core/compliance/."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone


from src.assembled_core.compliance.gdpr import (
    anonymize_news_headline,
    pseudonymize_user,
    should_retain,
)
from src.assembled_core.compliance.pdt import (
    PDT_EQUITY_THRESHOLD_USD,
    can_day_trade,
    count_day_trades,
)
from src.assembled_core.compliance.rate_limits import (
    ALLOWED_PERSONAL_USE,
    PROHIBITED_SOURCES,
    get_min_delay_seconds,
)
from src.assembled_core.compliance.tax_report import (
    EFFECTIVE_TAX_RATE,
    SPARER_PAUSCHBETRAG_EUR,
    summarize_closed_lots,
)

# ---------------------------------------------------------------------------
# GDPR helpers
# ---------------------------------------------------------------------------


class TestPseudonymizeUser:
    def test_deterministic(self):
        h1 = pseudonymize_user("user123")
        h2 = pseudonymize_user("user123")
        assert h1 == h2

    def test_different_users_differ(self):
        assert pseudonymize_user("alice") != pseudonymize_user("bob")

    def test_output_length_16(self):
        assert len(pseudonymize_user("test_user")) == 16

    def test_hex_string(self):
        result = pseudonymize_user("hello")
        assert all(c in "0123456789abcdef" for c in result)

    def test_salt_changes_result(self, monkeypatch):
        monkeypatch.setenv("ATA_PSEUDO_SALT", "salt_one")
        h1 = pseudonymize_user("user_x")
        monkeypatch.setenv("ATA_PSEUDO_SALT", "salt_two")
        h2 = pseudonymize_user("user_x")
        assert h1 != h2


class TestShouldRetain:
    def test_recent_record_retained(self):
        recent = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
        assert should_retain(recent, retention_days=365) is True

    def test_old_record_not_retained(self):
        old = (datetime.now(timezone.utc) - timedelta(days=400)).isoformat()
        assert should_retain(old, retention_days=365) is False

    def test_exactly_at_boundary(self):
        # exactly 365 days ago — should still be retained (age == limit)
        boundary = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        assert should_retain(boundary, retention_days=365) is True


class TestAnonymizeHeadline:
    def test_deterministic_v2(self):
        h = "Apple CEO retires"
        assert anonymize_news_headline(h) == anonymize_news_headline(h)

    def test_different_headlines_differ(self):
        assert anonymize_news_headline("A") != anonymize_news_headline("B")

    def test_returns_64_char_hex(self):
        result = anonymize_news_headline("test")
        assert len(result) == 64


# ---------------------------------------------------------------------------
# PDT rule
# ---------------------------------------------------------------------------


def _make_trade(d: date, symbol: str, side: str) -> dict:
    return {"date": d, "symbol": symbol, "side": side}


class TestCountDayTrades:
    def test_no_trades_zero(self):
        assert count_day_trades([]) == 0

    def test_single_buy_no_daytrade(self):
        trades = [_make_trade(date.today(), "AAPL", "buy")]
        assert count_day_trades(trades) == 0

    def test_buy_and_sell_same_day_is_daytrade(self):
        today = date.today()
        trades = [
            _make_trade(today, "AAPL", "buy"),
            _make_trade(today, "AAPL", "sell"),
        ]
        assert count_day_trades(trades, reference_date=today) == 1

    def test_buy_and_sell_different_days_not_daytrade(self):
        today = date.today()
        yesterday = today - timedelta(days=1)
        trades = [
            _make_trade(yesterday, "AAPL", "buy"),
            _make_trade(today, "AAPL", "sell"),
        ]
        assert count_day_trades(trades, reference_date=today) == 0

    def test_multiple_symbols_multiple_daytraders(self):
        today = date.today()
        trades = [
            _make_trade(today, "AAPL", "buy"),
            _make_trade(today, "AAPL", "sell"),
            _make_trade(today, "TSLA", "buy"),
            _make_trade(today, "TSLA", "sell"),
        ]
        assert count_day_trades(trades, reference_date=today) == 2

    def test_outside_window_excluded(self):
        today = date.today()
        old_day = today - timedelta(days=10)
        trades = [
            _make_trade(old_day, "AAPL", "buy"),
            _make_trade(old_day, "AAPL", "sell"),
        ]
        assert count_day_trades(trades, reference_date=today, window_days=5) == 0


class TestCanDayTrade:
    def test_above_threshold_always_ok(self):
        allowed, reason = can_day_trade(30_000.0, [])
        assert allowed is True
        assert reason == "pdt_threshold_met"

    def test_below_threshold_zero_trades_ok(self):
        allowed, reason = can_day_trade(10_000.0, [])
        assert allowed is True
        assert "pdt_ok" in reason

    def test_below_threshold_limit_reached(self):
        today = date.today()
        trades = []
        for symbol in ["AAPL", "TSLA", "MSFT"]:
            trades += [
                _make_trade(today, symbol, "buy"),
                _make_trade(today, symbol, "sell"),
            ]
        allowed, reason = can_day_trade(10_000.0, trades, reference_date=today)
        assert allowed is False
        assert "pdt_limit_reached" in reason

    def test_threshold_constant(self):
        assert PDT_EQUITY_THRESHOLD_USD == 25_000.0


# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------


class TestRateLimits:
    def test_yfinance_in_allowed(self):
        assert "yfinance" in ALLOWED_PERSONAL_USE

    def test_linkedin_prohibited(self):
        assert "linkedin" in PROHIBITED_SOURCES

    def test_get_min_delay_positive_for_known_source(self):
        delay = get_min_delay_seconds("sec_edgar")
        assert delay > 0.0

    def test_get_min_delay_zero_for_unknown(self):
        assert get_min_delay_seconds("some_unknown_source") == 0.0

    def test_stooq_delay_reasonable(self):
        # max 10 req/min → 6 s between requests
        assert abs(get_min_delay_seconds("stooq") - 6.0) < 0.01


# ---------------------------------------------------------------------------
# Tax report
# ---------------------------------------------------------------------------


class TestTaxReportSummary:
    def _make_lot(self, pnl_eur: float, year: int = 2025) -> dict:
        return {
            "realized_pnl_eur": pnl_eur,
            "trade_date": date(year, 6, 15),
        }

    def test_empty_lots(self):
        summary = summarize_closed_lots([], year=2025)
        assert summary.trade_count == 0
        assert summary.total_realized_pnl_eur == 0.0

    def test_only_wins(self):
        lots = [self._make_lot(500.0), self._make_lot(800.0)]
        summary = summarize_closed_lots(lots, year=2025)
        assert summary.wins_count == 2
        assert summary.losses_count == 0
        assert abs(summary.total_realized_pnl_eur - 1300.0) < 0.01

    def test_taxable_after_pauschbetrag(self):
        lots = [self._make_lot(2000.0)]
        summary = summarize_closed_lots(lots, year=2025)
        expected_taxable = 2000.0 - SPARER_PAUSCHBETRAG_EUR
        assert abs(summary.taxable_pnl_eur - expected_taxable) < 0.01

    def test_below_pauschbetrag_no_tax(self):
        lots = [self._make_lot(500.0)]
        summary = summarize_closed_lots(lots, year=2025)
        assert summary.taxable_pnl_eur == 0.0
        assert summary.estimated_tax_eur == 0.0

    def test_estimated_tax_rate(self):
        lots = [self._make_lot(5000.0)]
        summary = summarize_closed_lots(lots, year=2025)
        expected_tax = (5000.0 - SPARER_PAUSCHBETRAG_EUR) * EFFECTIVE_TAX_RATE
        assert abs(summary.estimated_tax_eur - expected_tax) < 0.01

    def test_wrong_year_excluded(self):
        lots_2024 = [self._make_lot(1000.0, year=2024)]
        lots_2025 = [self._make_lot(500.0, year=2025)]
        summary = summarize_closed_lots(lots_2024 + lots_2025, year=2025)
        assert summary.trade_count == 1
        assert abs(summary.total_realized_pnl_eur - 500.0) < 0.01

    def test_loss_year_note(self):
        lots = [self._make_lot(-300.0)]
        summary = summarize_closed_lots(lots, year=2025)
        assert any("Verlust" in note for note in summary.notes)

    def test_effective_tax_rate_constant(self):
        # 25% * 1.055 = 26.375%
        assert abs(EFFECTIVE_TAX_RATE - 0.26375) < 1e-6
