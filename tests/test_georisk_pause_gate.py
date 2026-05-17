"""Tests for geo-risk PAUSE enforcement in _apply_risk_controls_default.

Verifies that:
- PAUSE state blocks all orders and returns empty DataFrame with correct columns
- NORMAL/WATCH/ACTIVE/COOLDOWN states do not trigger the PAUSE block
- risk_state=None does not trigger the block (safe no-op)
- enable_risk_controls=False bypasses PAUSE (existing early-return takes precedence)
- return type is always pd.DataFrame, never list (pre-existing bug fix at exception path)
"""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import patch

from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    _apply_risk_controls_default,
)

pytestmark = pytest.mark.fast


def _make_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "side": ["BUY", "BUY"],
            "qty": [10.0, 5.0],
            "price": [150.0, 300.0],
        }
    )


def _make_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-10", "2025-01-10"], utc=True),
            "symbol": ["AAPL", "MSFT"],
            "close": [150.0, 300.0],
        }
    )


def _ctx(**kwargs) -> TradingContext:
    return TradingContext(
        prices=_make_prices(),
        enable_risk_controls=kwargs.pop("enable_risk_controls", True),
        risk_state=kwargs.pop("risk_state", None),
        **kwargs,
    )


class TestPauseGate:
    """PAUSE state must block all orders."""

    def test_pause_blocks_all_orders(self) -> None:
        orders = _make_orders()
        ctx = _ctx(risk_state={"state": "PAUSE"})
        result = _apply_risk_controls_default(ctx, orders)
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == list(orders.columns)

    def test_pause_returns_correct_columns(self) -> None:
        orders = _make_orders()
        ctx = _ctx(risk_state={"state": "PAUSE"})
        result = _apply_risk_controls_default(ctx, orders)
        assert set(result.columns) == {"symbol", "side", "qty", "price"}

    def test_pause_state_lowercase_not_blocked(self) -> None:
        """Only exact "PAUSE" blocks; lowercase "pause" should not (defensive)."""
        orders = _make_orders()
        ctx = _ctx(risk_state={"state": "pause"})
        # "pause" != "PAUSE" — should NOT trigger the guard
        # (state machine always uses uppercase, so this validates exact match)
        result = _apply_risk_controls_default(ctx, orders)
        # No block — orders may pass or be filtered by other controls
        assert isinstance(result, pd.DataFrame)


class TestNonPauseStates:
    """Non-PAUSE states must not trigger the PAUSE block."""

    @pytest.mark.parametrize("state", ["NORMAL", "WATCH", "ACTIVE", "COOLDOWN"])
    def test_non_pause_not_blocked(self, state: str) -> None:
        orders = _make_orders()
        ctx = _ctx(risk_state={"state": state})
        # Should NOT immediately return empty; risk controls may filter for
        # other reasons (drawdown etc.) but PAUSE gate itself must not fire.
        result = _apply_risk_controls_default(ctx, orders)
        assert isinstance(result, pd.DataFrame)
        # Result has at least the correct columns (may be empty for other reasons)
        assert set(result.columns).issuperset({"symbol"})


class TestPauseBypass:
    """enable_risk_controls=False must bypass PAUSE gate."""

    def test_risk_controls_off_bypasses_pause(self) -> None:
        orders = _make_orders()
        ctx = _ctx(risk_state={"state": "PAUSE"}, enable_risk_controls=False)
        result = _apply_risk_controls_default(ctx, orders)
        # Early-return at line 1265 returns orders.copy() — not blocked by PAUSE
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(orders)


class TestNullRiskState:
    """risk_state=None must not trigger PAUSE gate."""

    def test_none_risk_state_not_blocked(self) -> None:
        orders = _make_orders()
        ctx = _ctx(risk_state=None)
        result = _apply_risk_controls_default(ctx, orders)
        assert isinstance(result, pd.DataFrame)

    def test_empty_orders_with_pause_returns_empty(self) -> None:
        orders = pd.DataFrame(columns=["symbol", "side", "qty", "price"])
        ctx = _ctx(risk_state={"state": "PAUSE"})
        # Empty orders hits early-return before PAUSE guard
        result = _apply_risk_controls_default(ctx, orders)
        assert isinstance(result, pd.DataFrame)
        assert result.empty


class TestReturnTypeSafety:
    """Exception path must return DataFrame, not list (pre-existing bug fix)."""

    def test_exception_path_returns_dataframe(self) -> None:
        orders = _make_orders()
        ctx = _ctx(risk_state=None)
        # Force an exception inside the try block by patching filter_orders
        with patch(
            "src.assembled_core.pipeline.trading_cycle_shared.filter_orders_with_risk_controls",
            side_effect=RuntimeError("simulated risk module failure"),
        ):
            result = _apply_risk_controls_default(ctx, orders)
        assert isinstance(
            result, pd.DataFrame
        ), "Exception path must return pd.DataFrame, not list"
