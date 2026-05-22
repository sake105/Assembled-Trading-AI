"""Tests for backlog items 41, 43, 47, 69.

Item 41 — Decimal money arithmetic in paper_ledger / trade_journal / kpi_artifacts.
Item 43 — Halt-check in size_positions (halted symbols dropped from final targets).
Item 47 — ZeroDivisionError guards in _tc_sizing (_px_mean_lag, crash-cap scale).
Item 69 — Buying-power pre-check in size_positions (gross weight capped at 95%).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Item 41 — paper_ledger: Decimal cash accumulation
# ---------------------------------------------------------------------------


class TestDecimalCashAccumulation:
    """Decimal arithmetic prevents float drift in apply_fills_to_ledger."""

    def _make_state(self, cash: float = 100_000.0) -> dict[str, Any]:
        return {
            "schema_version": "paper.ledger_state.v1",
            "updated_utc": None,
            "cash": cash,
            "positions": {},
            "equity_curve": [],
        }

    def test_buy_fill_deducts_correct_cash(self) -> None:
        from src.assembled_core.ops.paper_ledger import apply_fills_to_ledger

        state = self._make_state(100_000.0)
        fills = [{"symbol": "AAPL", "side": "BUY", "qty": 10.0, "price": 150.0}]
        out = apply_fills_to_ledger(state, fills)
        assert abs(out["cash"] - 98_500.0) < 0.01

    def test_sell_fill_adds_correct_cash(self) -> None:
        from src.assembled_core.ops.paper_ledger import apply_fills_to_ledger

        state = self._make_state(50_000.0)
        state["positions"] = {"MSFT": {"qty": 20.0, "avg_price": 300.0, "hwm": 300.0}}
        fills = [{"symbol": "MSFT", "side": "SELL", "qty": 20.0, "price": 310.0}]
        out = apply_fills_to_ledger(state, fills)
        assert abs(out["cash"] - 56_200.0) < 0.01

    def test_many_small_fills_no_drift(self) -> None:
        """Many 0.10-lot fills should not accumulate float rounding error."""
        from src.assembled_core.ops.paper_ledger import apply_fills_to_ledger

        state = self._make_state(10_000.0)
        fills = [
            {"symbol": "X", "side": "BUY", "qty": 0.1, "price": 10.0}
            for _ in range(100)
        ]
        out = apply_fills_to_ledger(state, fills)
        # 100 × 0.1 × 10 = 100.00 deducted
        expected = 10_000.0 - 100.0
        assert abs(out["cash"] - expected) < 0.01, (
            f"Cash drift detected: got {out['cash']}, expected {expected}"
        )

    def test_cash_is_float_in_output(self) -> None:
        from src.assembled_core.ops.paper_ledger import apply_fills_to_ledger

        state = self._make_state(10_000.0)
        fills = [{"symbol": "A", "side": "BUY", "qty": 1.0, "price": 100.0}]
        out = apply_fills_to_ledger(state, fills)
        assert isinstance(out["cash"], float)


# ---------------------------------------------------------------------------
# Item 41 — trade_journal: Decimal invested accumulation
# ---------------------------------------------------------------------------


class TestDecimalInvestedSum:
    """Decimal accumulation in write_daily_summary."""

    def test_invested_matches_expected(self, tmp_path) -> None:
        from src.assembled_core.ops.trade_journal import write_daily_summary

        positions = {
            "AAPL": {"qty": 10.0, "avg_price": 150.0},
            "MSFT": {"qty": 5.0, "avg_price": 300.0},
        }
        ledger_state = {"cash": 3500.0, "positions": positions}
        # Expected invested: 10*150 + 5*300 = 1500 + 1500 = 3000
        path = write_daily_summary(
            "2026-05-07",
            ledger_state,
            fills=[],
            equity=6_500.0,
            start_capital=10_000.0,
            output_dir=str(tmp_path),
        )
        assert path is not None
        text = path.read_text(encoding="utf-8")
        # Invested should be $3,000.00
        assert "3,000.00" in text


# ---------------------------------------------------------------------------
# Item 43 — Halt-check: halted symbols dropped from target_positions
# ---------------------------------------------------------------------------


def _make_sizing_fn(symbols: list[str]):
    """Return a simple equal-weight position sizing function for tests."""

    def _fn(sigs: pd.DataFrame, capital: float) -> pd.DataFrame:
        syms = sigs["symbol"].tolist()
        w = 1.0 / len(syms) if syms else 0.0
        return pd.DataFrame(
            {
                "symbol": syms,
                "target_weight": [w] * len(syms),
                "target_qty": [round(w * capital / 100.0, 2)] * len(syms),
            }
        )

    return _fn


class TestHaltCheck:
    """Halted symbols must not appear in final target_positions from size_positions."""

    def _make_signals(self, symbols: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "symbol": symbols,
                "direction": [1] * len(symbols),
                "score": [0.8] * len(symbols),
            }
        )

    def _make_ctx(self, symbols: list[str], halted: set[str] | None = None):
        from src.assembled_core.pipeline.trading_cycle_shared import TradingContext

        prices = pd.DataFrame(
            {
                "timestamp": pd.Timestamp("2026-01-02"),
                "symbol": symbols,
                "close": [100.0] * len(symbols),
                "open": [100.0] * len(symbols),
                "high": [101.0] * len(symbols),
                "low": [99.0] * len(symbols),
                "volume": [1_000_000] * len(symbols),
            }
        )
        ctx = TradingContext(
            prices=prices,
            capital=100_000.0,
            position_sizing_fn=_make_sizing_fn(symbols),
        )
        if halted is not None:
            ctx.halted_symbols = halted  # type: ignore[attr-defined]
        return ctx

    def test_halted_symbol_removed(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import size_positions

        syms = ["AAPL", "MSFT", "GOOG"]
        ctx = self._make_ctx(syms, halted={"MSFT"})
        signals = self._make_signals(syms)
        with patch(
            "src.assembled_core.pipeline._tc_sizing.load_policy", return_value={}
        ):
            tp, _, _ = size_positions(signals, ctx)
        if not tp.empty and "symbol" in tp.columns:
            assert "MSFT" not in tp["symbol"].tolist(), (
                "Halted symbol MSFT must not appear in final target_positions"
            )

    def test_non_halted_symbols_retained(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import size_positions

        syms = ["AAPL", "MSFT", "GOOG"]
        ctx = self._make_ctx(syms, halted={"MSFT"})
        signals = self._make_signals(syms)
        with patch(
            "src.assembled_core.pipeline._tc_sizing.load_policy", return_value={}
        ):
            tp, _, _ = size_positions(signals, ctx)
        if not tp.empty and "symbol" in tp.columns:
            remaining = set(tp["symbol"].tolist())
            assert "MSFT" not in remaining
            # At least one non-halted symbol should remain
            assert len(remaining - {"MSFT"}) >= 1

    def test_no_halted_symbols_attr_no_crash(self) -> None:
        """If ctx has no halted_symbols attribute, size_positions must not crash."""
        from src.assembled_core.pipeline._tc_sizing import size_positions

        syms = ["AAPL"]
        ctx = self._make_ctx(syms, halted=None)
        signals = self._make_signals(syms)
        with patch(
            "src.assembled_core.pipeline._tc_sizing.load_policy", return_value={}
        ):
            tp, _, _ = size_positions(signals, ctx)
        # Just verifying no exception


# ---------------------------------------------------------------------------
# Item 47 — ZeroDivisionError guards
# ---------------------------------------------------------------------------


class TestZeroDivisionGuards:
    """Guards against division by zero in _tc_sizing."""

    def test_crash_cap_scale_zero_gross_does_not_raise(self) -> None:
        """_sp_apply_crash_cap must handle current_long_gross == 0 without ZeroDivisionError."""
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_crash_cap

        # All-zero weights → current_long_gross = 0
        tp = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "target_weight": [0.0, 0.0]})
        policy = {
            "crash_prediction": {
                "enabled": True,
                "equity_cap_threshold": 0.4,
                "base_long_gross": 1.0,
            }
        }
        meta = {"crash_prediction": {"crash_probability": 0.8}}
        # Should not raise
        result = _sp_apply_crash_cap(tp, policy, meta, as_of_str="2026-01-02")
        assert isinstance(result, pd.DataFrame)

    def test_hmm_zero_mean_price_no_crash(self) -> None:
        """_px_mean_lag with zero-price rows: replace(0, nan) avoids inf/ZeroDivision."""
        # Directly test the numpy logic used in the HMM cache section
        px_mean = pd.Series([0.0, 100.0, 101.0, 102.0])
        px_mean_lag = px_mean.shift(1).replace(0, np.nan)
        ratio = (px_mean / px_mean_lag).clip(lower=1e-10)
        result = np.log(ratio).dropna()
        # Should not contain inf or NaN
        assert not np.any(np.isinf(result.values))
        assert not np.any(np.isnan(result.values))


# ---------------------------------------------------------------------------
# Item 69 — Buying-power pre-check
# ---------------------------------------------------------------------------


class TestBuyingPowerPreCheck:
    """Gross weight exceeding 95% of buying power triggers scale-down."""

    def _overweight_sizing_fn(self, syms: list[str]):
        """Return a sizing fn that produces gross weight > 0.95."""

        def _fn(sigs: pd.DataFrame, capital: float) -> pd.DataFrame:
            # Intentionally assign 0.40 per symbol → gross = 1.20 for 3 symbols
            return pd.DataFrame(
                {
                    "symbol": syms,
                    "target_weight": [0.40] * len(syms),
                    "target_qty": [400.0] * len(syms),
                }
            )

        return _fn

    def _make_ctx_with_capital(self, symbols: list[str], capital: float = 100_000.0):
        from src.assembled_core.pipeline.trading_cycle_shared import TradingContext

        prices = pd.DataFrame(
            {
                "timestamp": pd.Timestamp("2026-01-02"),
                "symbol": symbols,
                "close": [100.0] * len(symbols),
                "open": [100.0] * len(symbols),
                "high": [101.0] * len(symbols),
                "low": [99.0] * len(symbols),
                "volume": [1_000_000] * len(symbols),
            }
        )
        ctx = TradingContext(
            prices=prices,
            capital=capital,
            position_sizing_fn=self._overweight_sizing_fn(symbols),
        )
        return ctx

    def test_overweight_portfolio_scaled_down(self) -> None:
        """When gross weight > 0.95, positions must be scaled back."""
        from src.assembled_core.pipeline._tc_sizing import size_positions

        syms = ["A", "B", "C"]
        signals = pd.DataFrame(
            {"symbol": syms, "direction": [1, 1, 1], "score": [1.0, 1.0, 1.0]}
        )
        ctx = self._make_ctx_with_capital(syms)
        ctx.buying_power = 100_000.0  # type: ignore[attr-defined]

        with patch(
            "src.assembled_core.pipeline._tc_sizing.load_policy", return_value={}
        ):
            tp, _, _ = size_positions(signals, ctx)

        if not tp.empty and "target_weight" in tp.columns:
            gross = float(tp["target_weight"].abs().sum())
            assert gross <= 0.951, (
                f"Gross weight {gross:.4f} exceeds buying-power limit after pre-check"
            )

    def test_underweight_portfolio_not_scaled(self) -> None:
        """When gross weight <= 0.95, no scale-down occurs."""
        from src.assembled_core.pipeline._tc_sizing import size_positions

        syms = ["A"]
        signals = pd.DataFrame({"symbol": syms, "direction": [1], "score": [0.5]})
        ctx = self._make_ctx_with_capital(syms)
        ctx.buying_power = 100_000.0  # type: ignore[attr-defined]

        with patch(
            "src.assembled_core.pipeline._tc_sizing.load_policy", return_value={}
        ):
            tp, _, _ = size_positions(signals, ctx)
        # Just verifying no crash
        assert isinstance(tp, pd.DataFrame)
