"""Regression tests for Iteration-11 fixes.

Covers:
  Fix 1 — earnings_calendar_source.py: datetime.today() -> datetime.now(tz=UTC)
  Fix 2 — drawdown_decomposition.py: active drawdown at series end flushed
  Fix 4 — paper_track.py: enable_risk_controls warning + geo-risk multiplier
    (pending risk-execution-reviewer assessment)
"""

from __future__ import annotations

from datetime import timezone

import pandas as pd
import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Fix 1 — earnings_calendar_source.py: UTC-aware datetime
# ---------------------------------------------------------------------------


class TestEarningsCalendarUtcDatetime:
    """fetch_calendar must use UTC-aware datetime, not naive datetime.today()."""

    def test_fetch_calendar_uses_utc_aware_dates(self, monkeypatch):
        """Dates passed to yfinance / finnhub must be UTC-aware, not naive."""
        from src.assembled_core.data.sources.earnings_calendar_source import (
            EarningsCalendarSource,
        )

        received_start: list = []
        received_end: list = []

        source = EarningsCalendarSource()

        def _fake_fetch_yf(sym, start_date, end_date):
            received_start.append(start_date)
            received_end.append(end_date)
            return []

        monkeypatch.setattr(source, "_fetch_yfinance", _fake_fetch_yf)

        source.fetch_calendar(["AAPL"], days_ahead=30)

        assert received_start, "Expected _fetch_yfinance to be called at least once"
        start = received_start[0]
        end = received_end[0]

        assert start.tzinfo is not None, (
            f"start_date must be tz-aware, got naive: {start}"
        )
        assert end.tzinfo is not None, f"end_date must be tz-aware, got naive: {end}"
        assert start.tzinfo == timezone.utc or str(start.tzinfo) in (
            "UTC",
            "utc",
        ), f"Expected UTC timezone, got: {start.tzinfo}"

    def test_end_date_is_in_future(self, monkeypatch):
        """end_date must be approximately days_ahead from now."""
        from datetime import datetime

        from src.assembled_core.data.sources.earnings_calendar_source import (
            EarningsCalendarSource,
        )

        received_end: list = []
        source = EarningsCalendarSource()
        monkeypatch.setattr(
            source, "_fetch_yfinance", lambda s, st, e: received_end.append(e) or []
        )

        source.fetch_calendar(["AAPL"], days_ahead=60)

        now = datetime.now(tz=timezone.utc)
        end = received_end[0]
        delta_days = (end - now).days
        assert 55 <= delta_days <= 65, (
            f"end_date should be ~60 days in the future, got {delta_days} days"
        )


# ---------------------------------------------------------------------------
# Fix 2 — drawdown_decomposition.py: flush active DD at series end
# ---------------------------------------------------------------------------


class TestFindAllDrawdownsFlushesAtEnd:
    """Active drawdown at series end must be captured, not silently dropped."""

    def _make_returns(self, values: list[float]) -> pd.Series:
        return pd.Series(values, dtype=float)

    def test_drawdown_ending_at_series_end_is_captured(self):
        """DD that starts mid-series and never recovers must appear in result."""
        from src.assembled_core.qa.drawdown_decomposition import find_all_drawdowns

        # Equity rises, then falls to end of series — never recovers to peak
        # dd_arr will be negative at the end
        returns = self._make_returns(
            [
                0.01,
                0.01,
                0.01,
                0.01,
                0.01,  # rising phase: builds peak
                -0.05,
                -0.03,
                -0.04,
                -0.02,
                -0.03,
            ]  # falling phase: DD never closes
        )
        drawdowns = find_all_drawdowns(returns, min_depth=0.03, min_duration=2)
        assert len(drawdowns) >= 1, (
            "Expected at least 1 drawdown — the open DD at series end must be flushed"
        )
        last_dd = drawdowns[-1]
        assert last_dd.max_drawdown < 0, "Drawdown value must be negative"

    def test_drawdown_that_recovers_mid_series_still_found(self):
        """A DD that recovers mid-series must also be captured (existing behavior)."""
        from src.assembled_core.qa.drawdown_decomposition import find_all_drawdowns

        returns = self._make_returns(
            [0.01, -0.05, -0.04, 0.08, 0.01]  # recovers by index 3
        )
        drawdowns = find_all_drawdowns(returns, min_depth=0.03, min_duration=1)
        assert len(drawdowns) >= 1, "Recovered DD must still be found"

    def test_no_drawdown_returns_empty_list(self):
        """Monotonically rising series produces no drawdowns."""
        from src.assembled_core.qa.drawdown_decomposition import find_all_drawdowns

        returns = self._make_returns([0.01] * 10)
        drawdowns = find_all_drawdowns(returns, min_depth=0.01, min_duration=1)
        assert drawdowns == [], f"Expected empty list, got {drawdowns}"

    def test_two_drawdowns_one_open_at_end(self):
        """Two DDs: first recovers, second ends at series end — both found."""
        from src.assembled_core.qa.drawdown_decomposition import find_all_drawdowns

        returns = self._make_returns(
            # DD1: recovers
            [
                0.02,
                0.02,
                -0.06,
                -0.04,
                0.10,
                0.02,
                # DD2: never recovers
                -0.06,
                -0.05,
                -0.03,
            ]
        )
        drawdowns = find_all_drawdowns(returns, min_depth=0.04, min_duration=1)
        assert len(drawdowns) == 2, (
            f"Expected 2 drawdowns (one recovered, one open at end), got {len(drawdowns)}"
        )
