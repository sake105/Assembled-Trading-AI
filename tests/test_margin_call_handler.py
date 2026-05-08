"""Tests for risk/margin_call_handler.py — Item 42.

Verifies: when margin_call=True, positions are sorted by notional (ascending)
and the bottom 50% are returned for closure.
"""

from __future__ import annotations


from src.assembled_core.risk.margin_call_handler import handle_margin_call


def _make_state(
    positions: dict[str, float], prices: dict[str, float] | None = None
) -> dict:
    return {
        "margin_call": True,
        "equity": 50_000.0,
        "maintenance_required": 60_000.0,
        "margin_call_amount": 10_000.0,
        "positions": positions,
        "prices": prices or {},
    }


class TestHandleMarginCall:
    def test_returns_bottom_50_pct_by_notional(self):
        """4 positions → bottom 2 (lowest notional) returned."""
        positions = {
            "AAPL": 10.0,  # notional = 10 * 200 = 2000
            "MSFT": 5.0,  # notional = 5  * 100 = 500   ← lowest
            "NVDA": 2.0,  # notional = 2  * 50  = 100   ← 2nd lowest
            "TSLA": 20.0,  # notional = 20 * 300 = 6000
        }
        prices = {"AAPL": 200.0, "MSFT": 100.0, "NVDA": 50.0, "TSLA": 300.0}
        state = _make_state(positions, prices)

        to_close = handle_margin_call(state)

        assert len(to_close) == 2
        assert set(to_close) == {"NVDA", "MSFT"}

    def test_rounds_up_to_at_least_one(self):
        """Single position → 1 symbol returned even though 50% of 1 = 0.5."""
        positions = {"AAPL": 10.0}
        state = _make_state(positions, {"AAPL": 150.0})

        to_close = handle_margin_call(state)

        assert len(to_close) == 1
        assert to_close[0] == "AAPL"

    def test_empty_positions_returns_empty_list(self):
        state = _make_state({})
        to_close = handle_margin_call(state)
        assert to_close == []

    def test_no_positions_key_returns_empty_list(self):
        state = {
            "margin_call": True,
            "equity": 10_000.0,
            "maintenance_required": 15_000.0,
            "margin_call_amount": 5_000.0,
        }
        to_close = handle_margin_call(state)
        assert to_close == []

    def test_skips_zero_quantity_positions(self):
        """Positions with qty=0 must not be included."""
        positions = {"AAPL": 10.0, "MSFT": 0.0}
        prices = {"AAPL": 200.0, "MSFT": 100.0}
        state = _make_state(positions, prices)

        to_close = handle_margin_call(state)

        assert "MSFT" not in to_close
        assert len(to_close) == 1

    def test_short_positions_handled_by_absolute_value(self):
        """Short positions (negative qty) are sorted by |notional|."""
        positions = {
            "AAPL": -5.0,  # |notional| = 5  * 200 = 1000  ← lowest
            "TSLA": -20.0,  # |notional| = 20 * 300 = 6000
        }
        prices = {"AAPL": 200.0, "TSLA": 300.0}
        state = _make_state(positions, prices)

        to_close = handle_margin_call(state)

        assert len(to_close) == 1
        assert to_close[0] == "AAPL"

    def test_no_adapter_returns_all_flagged(self):
        """Without broker adapter, returns symbols caller must act on."""
        positions = {"AAPL": 5.0, "MSFT": 3.0, "GOOG": 7.0, "AMZN": 2.0}
        prices = {"AAPL": 100.0, "MSFT": 200.0, "GOOG": 300.0, "AMZN": 50.0}
        state = _make_state(positions, prices)

        to_close = handle_margin_call(state, adapter=None)

        assert len(to_close) == 2  # bottom 50% of 4

    def test_custom_close_fraction(self):
        """close_fraction=0.25 with 4 positions → 1 symbol."""
        positions = {sym: float(i + 1) for i, sym in enumerate(["A", "B", "C", "D"])}
        prices = {sym: 100.0 for sym in positions}
        state = _make_state(positions, prices)

        to_close = handle_margin_call(state, close_fraction=0.25)

        assert len(to_close) == 1

    def test_exported_from_risk_init(self):
        """handle_margin_call must be importable from the risk package."""
        from src.assembled_core.risk import handle_margin_call as hmc  # noqa: F401

        assert callable(hmc)
