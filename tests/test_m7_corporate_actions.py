"""Tests for M7-T02: adjust_prices_for_splits real implementation.

Covers:
- Empty actions → original returned
- 2:1 split: pre-split prices halved, post-split unchanged
- Multiple symbols: only target symbol adjusted
- Multiple splits: both applied correctly
- Missing required columns in actions → copy returned unchanged
- Missing close column → copy returned unchanged
- Invalid split_ratio (zero) → skipped
- Backward adjustment does not affect post-split prices
"""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.phase12

from src.assembled_core.data.corporate_actions import adjust_prices_for_splits


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prices(symbol: str, closes: list[float], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=len(closes), freq="D", tz="UTC")
    return pd.DataFrame({
        "timestamp": dates,
        "symbol": [symbol] * len(closes),
        "close": closes,
    })


def _split_action(symbol: str, date: str, ratio: float) -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": [symbol],
        "action_type": ["SPLIT"],
        "effective_date": [pd.Timestamp(date, tz="UTC")],
        "split_ratio": [ratio],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAdjustPricesForSplits:
    def test_empty_actions_returns_original(self):
        prices = _make_prices("AAPL", [200.0, 205.0, 100.0])
        actions = pd.DataFrame(columns=["symbol", "action_type", "effective_date", "split_ratio"])
        result = adjust_prices_for_splits(prices, actions)
        # Returns original reference (not a copy)
        assert result is prices

    def test_2_to_1_split_halves_presplit_prices(self):
        # Days 1-2 before split, days 3-5 after; split on day 3 (2:1)
        prices = _make_prices("AAPL", [200.0, 205.0, 100.0, 102.0, 104.0], "2024-01-01")
        actions = _split_action("AAPL", "2024-01-03", 2.0)
        result = adjust_prices_for_splits(prices, actions)

        # Pre-split (days 1-2): prices should be halved
        assert result["close"].iloc[0] == pytest.approx(100.0)
        assert result["close"].iloc[1] == pytest.approx(102.5)
        # Post-split (days 3-5): prices unchanged
        assert result["close"].iloc[2] == pytest.approx(100.0)
        assert result["close"].iloc[3] == pytest.approx(102.0)
        assert result["close"].iloc[4] == pytest.approx(104.0)

    def test_split_does_not_create_fake_crash(self):
        # Verify adjusted returns are smooth across split
        prices = _make_prices("AAPL", [200.0, 204.0, 102.0, 104.04], "2024-01-01")
        actions = _split_action("AAPL", "2024-01-03", 2.0)
        result = adjust_prices_for_splits(prices, actions)
        # Adjusted: [100, 102, 102, 104.04] → return on day 3 ≈ 0% (no fake crash)
        r = result["close"].pct_change().dropna()
        assert abs(r.iloc[1]) < 0.01  # return around split ≈ 0%

    def test_only_target_symbol_adjusted(self):
        p_aapl = _make_prices("AAPL", [200.0, 205.0, 100.0])
        p_msft = _make_prices("MSFT", [300.0, 305.0, 310.0])
        prices = pd.concat([p_aapl, p_msft], ignore_index=True)
        actions = _split_action("AAPL", "2024-01-03", 2.0)
        result = adjust_prices_for_splits(prices, actions)

        msft_result = result[result["symbol"] == "MSFT"]["close"].tolist()
        assert msft_result == pytest.approx([300.0, 305.0, 310.0])

    def test_3_to_1_split(self):
        prices = _make_prices("XYZ", [300.0, 303.0, 101.0, 103.0], "2024-01-01")
        actions = _split_action("XYZ", "2024-01-03", 3.0)
        result = adjust_prices_for_splits(prices, actions)
        assert result["close"].iloc[0] == pytest.approx(100.0)
        assert result["close"].iloc[1] == pytest.approx(101.0)
        assert result["close"].iloc[2] == pytest.approx(101.0)

    def test_multiple_splits_applied_sequentially(self):
        # Two splits on same symbol
        prices = _make_prices("A", [400.0, 200.0, 100.0, 50.0], "2024-01-01")
        actions = pd.DataFrame({
            "symbol": ["A", "A"],
            "action_type": ["SPLIT", "SPLIT"],
            "effective_date": [pd.Timestamp("2024-01-02", tz="UTC"),
                               pd.Timestamp("2024-01-03", tz="UTC")],
            "split_ratio": [2.0, 2.0],
        })
        result = adjust_prices_for_splits(prices, actions)
        # Day 1 (before both splits): 400 / 2 / 2 = 100
        assert result["close"].iloc[0] == pytest.approx(100.0)
        # Day 2 (after first split, before second): 200 / 2 = 100
        assert result["close"].iloc[1] == pytest.approx(100.0)
        # Day 3+ (after both splits): unchanged
        assert result["close"].iloc[2] == pytest.approx(100.0)

    def test_missing_required_column_returns_copy_unchanged(self):
        prices = _make_prices("AAPL", [200.0, 100.0])
        # actions missing split_ratio column
        actions = pd.DataFrame({
            "symbol": ["AAPL"],
            "action_type": ["SPLIT"],
            "effective_date": [pd.Timestamp("2024-01-02", tz="UTC")],
            # missing split_ratio
        })
        result = adjust_prices_for_splits(prices, actions)
        # Returns copy but unchanged
        assert result["close"].tolist() == pytest.approx([200.0, 100.0])

    def test_missing_close_column_returns_copy_unchanged(self):
        prices = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "symbol": ["A"] * 3,
            "open": [100.0, 200.0, 300.0],  # no close column
        })
        actions = _split_action("A", "2024-01-02", 2.0)
        result = adjust_prices_for_splits(prices, actions)
        assert "open" in result.columns
        assert result["open"].tolist() == pytest.approx([100.0, 200.0, 300.0])

    def test_zero_ratio_skipped(self):
        prices = _make_prices("A", [200.0, 100.0])
        actions = pd.DataFrame({
            "symbol": ["A"],
            "action_type": ["SPLIT"],
            "effective_date": [pd.Timestamp("2024-01-02", tz="UTC")],
            "split_ratio": [0.0],  # invalid
        })
        result = adjust_prices_for_splits(prices, actions)
        # Zero ratio skipped → prices unchanged
        assert result["close"].tolist() == pytest.approx([200.0, 100.0])

    def test_returns_copy_not_original(self):
        prices = _make_prices("A", [200.0, 100.0])
        actions = _split_action("A", "2024-01-02", 2.0)
        result = adjust_prices_for_splits(prices, actions)
        assert result is not prices
