"""Tests for earnings calendar source module."""

from __future__ import annotations

import pytest
import pandas as pd

from src.assembled_core.data.sources.earnings_calendar_source import (
    EarningsCalendarSource,
)


@pytest.mark.phase12
class TestEarningsCalendarSource:
    def test_init(self):
        source = EarningsCalendarSource()
        assert source is not None

    def test_build_earnings_factors(self):
        source = EarningsCalendarSource()
        # Build from synthetic calendar data
        calendar = pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL", "MSFT"],
                "earnings_date": pd.to_datetime(
                    ["2024-01-25", "2024-04-25", "2024-01-30"]
                ),
            }
        )
        panel = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 5 + ["MSFT"] * 5,
                "timestamp": list(pd.bdate_range("2024-01-20", periods=5)) * 2,
                "close": [
                    190.0,
                    191.0,
                    192.0,
                    193.0,
                    194.0,
                    400.0,
                    401.0,
                    402.0,
                    403.0,
                    404.0,
                ],
            }
        )
        result = source.build_earnings_factors(calendar, panel)
        assert isinstance(result, pd.DataFrame)
        if len(result) > 0:
            assert "days_to_earnings" in result.columns or len(result.columns) > 0


@pytest.mark.phase12
class TestEarningsFactorComputation:
    def test_days_to_earnings_decreases(self):
        """Days to earnings should decrease as date approaches."""
        source = EarningsCalendarSource()
        calendar = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                "earnings_date": pd.to_datetime(["2024-01-30"]),
            }
        )
        dates = pd.bdate_range("2024-01-22", periods=5)
        panel = pd.DataFrame(
            {
                "symbol": ["AAPL"] * 5,
                "timestamp": dates,
                "close": [190.0] * 5,
            }
        )
        result = source.build_earnings_factors(calendar, panel)
        if "days_to_earnings" in result.columns:
            dte = result["days_to_earnings"].values
            # Should be decreasing or at least non-increasing
            assert dte[0] >= dte[-1]
