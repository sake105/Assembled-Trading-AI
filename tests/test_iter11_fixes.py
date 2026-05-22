"""Regression tests for Iteration-11 fixes.

Covers:
  Fix 1 — earnings_calendar_source.py: datetime.today() -> datetime.now(tz=UTC)
  Fix 2 — drawdown_decomposition.py: active drawdown at series end flushed
  Fix 3 — quality_gate.py: spike threshold * 10 removed (30sigma -> 3sigma)
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


# ---------------------------------------------------------------------------
# Fix 3 — quality_gate.py: spike threshold not multiplied by 10
# ---------------------------------------------------------------------------


class TestQualityGateSpikeThreshold:
    """_check_price_spikes must flag 3sigma moves, not require 30sigma."""

    def _make_price_df(self, close_values: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"Close": close_values})

    def test_three_sigma_spike_is_flagged(self):
        """A 3σ price move must trigger a spike warning."""
        from src.assembled_core.data.quality_gate import (
            QualityResult,
            QualityStatus,
            _check_price_spikes,
        )

        # Build a series where one return is clearly > 3σ
        base_returns = [0.001] * 20  # tiny daily moves
        spike = [0.20]  # 20% move — clearly > 3σ of 0.1% vol
        close = [100.0]
        for r in base_returns + spike:
            close.append(close[-1] * (1 + r))

        df = self._make_price_df(close)
        result = QualityResult(
            status=QualityStatus.PASS, ticker="TEST", n_rows=len(close)
        )
        _check_price_spikes(df, result, spike_threshold=3.0)

        assert any("price_spikes" in w for w in result.checks_warned), (
            f"Expected 'price_spikes' in checks_warned, got: {result.checks_warned}"
        )

    def test_small_daily_moves_not_flagged(self):
        """Normal daily moves (<3σ) must not trigger a spike warning."""
        from src.assembled_core.data.quality_gate import (
            QualityResult,
            QualityStatus,
            _check_price_spikes,
        )

        # All returns in [−0.5%, +0.5%] — well within 3σ for any reasonable vol
        close = [100.0 * (1 + 0.005 * ((-1) ** i)) for i in range(30)]
        df = self._make_price_df(close)
        result = QualityResult(
            status=QualityStatus.PASS, ticker="TEST", n_rows=len(close)
        )
        _check_price_spikes(df, result, spike_threshold=3.0)

        spike_warnings = [w for w in result.checks_warned if "price_spikes" in w]
        assert not spike_warnings, (
            f"Small moves should not trigger spike warning, got: {spike_warnings}"
        )

    def test_ten_sigma_move_was_previously_undetected(self):
        """Verify the OLD 30-sigma threshold would have missed a 15-sigma move."""
        # This test documents the bug: with threshold*10, a 15σ move passed silently.
        # The fix removes the *10. We verify our 3σ threshold now catches it.
        from src.assembled_core.data.quality_gate import (
            QualityResult,
            QualityStatus,
            _check_price_spikes,
        )

        base = [0.001] * 50
        # Build series with one 15% move. For vol ~0.1%, z ≈ 150 → caught by both
        # But for vol ~5%, z ≈ 3 → caught by 3σ threshold, NOT by 30σ threshold.
        close = [100.0]
        for r in base:
            close.append(close[-1] * (1 + r))
        # Add a 15% spike — z ≈ 3σ for 5% baseline vol
        close.append(close[-1] * 1.15)
        # Add some noise to inflate baseline vol
        noisy_base = [0.05 * ((-1) ** i) for i in range(20)]
        close2 = [100.0]
        for r in noisy_base:
            close2.append(close2[-1] * (1 + r))
        close2.append(close2[-1] * 1.20)  # spike > 3σ of noisy baseline

        df = self._make_price_df(close2)
        result = QualityResult(
            status=QualityStatus.PASS, ticker="TEST", n_rows=len(close2)
        )
        _check_price_spikes(df, result, spike_threshold=3.0)

        # With noisy baseline, 20% spike should be caught by 3σ threshold
        # (before fix: would need 30σ to flag, missing this)
        assert any("price_spikes" in w for w in result.checks_warned), (
            "Expected 20% spike to be flagged with 3σ threshold"
        )
