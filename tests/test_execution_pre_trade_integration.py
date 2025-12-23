"""Integration tests for pre-trade checks and kill switch in order flow."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

pytestmark = pytest.mark.phase10

from src.assembled_core.execution.order_generation import generate_orders_from_targets
from src.assembled_core.execution.risk_controls import filter_orders_with_risk_controls
from src.assembled_core.execution.safe_bridge import write_safe_orders_csv
from src.assembled_core.execution.pre_trade_checks import PreTradeConfig


@pytest.fixture
def sample_target_positions() -> pd.DataFrame:
    """Create sample target positions."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "GOOGL", "MSFT"],
            "target_weight": [0.33, 0.33, 0.34],
            "target_qty": [100.0, 50.0, 200.0],
        }
    )


@pytest.fixture
def sample_prices() -> pd.DataFrame:
    """Create sample prices."""
    from datetime import datetime

    base = datetime(2024, 1, 1, tzinfo=None)
    return pd.DataFrame(
        {
            "timestamp": [base] * 3,
            "symbol": ["AAPL", "GOOGL", "MSFT"],
            "close": [150.0, 2500.0, 300.0],
        }
    )


class TestOrderFlowWithRiskControls:
    """Integration tests for order flow with risk controls."""

    def test_order_flow_respects_pre_trade_checks(
        self, sample_target_positions, sample_prices, tmp_path: Path
    ):
        """Test that order flow respects pre-trade checks and blocks oversized orders."""
        # Generate orders
        orders = generate_orders_from_targets(
            sample_target_positions, prices=sample_prices
        )

        # Apply risk controls with strict limits
        config = PreTradeConfig(max_notional_per_symbol=10000.0)  # Very low limit

        # AAPL: 100 * 150 = 15000 (> 10000) - blocked
        # GOOGL: 50 * 2500 = 125000 (> 10000) - blocked
        # MSFT: 200 * 300 = 60000 (> 10000) - blocked

        filtered, result = filter_orders_with_risk_controls(
            orders,
            pre_trade_config=config,
            enable_pre_trade_checks=True,
            enable_kill_switch=False,
        )

        # All orders should be blocked due to size limits
        assert len(filtered) == 0
        assert result.total_orders_before == 3
        assert result.total_orders_after == 0
        assert result.pre_trade_result is not None
        assert not result.pre_trade_result.is_ok

        # Write to SAFE CSV - should be empty
        safe_path = tmp_path / "orders_test.csv"
        write_safe_orders_csv(
            filtered,
            output_path=safe_path,
            date=None,
            price_type="MARKET",
            comment="Test",
        )

        # Read back and verify empty
        written_df = pd.read_csv(safe_path)
        assert len(written_df) == 0

    def test_order_flow_respects_kill_switch(
        self, sample_target_positions, sample_prices, tmp_path: Path
    ):
        """Test that order flow respects kill switch and blocks all orders when engaged."""
        # Generate orders
        orders = generate_orders_from_targets(
            sample_target_positions, prices=sample_prices
        )

        # Set kill switch via environment variable
        with patch.dict(os.environ, {"ASSEMBLED_KILL_SWITCH": "1"}, clear=False):
            filtered, result = filter_orders_with_risk_controls(
                orders,
                enable_pre_trade_checks=False,  # Disable pre-trade checks for this test
                enable_kill_switch=True,
            )

        # All orders should be blocked by kill switch
        assert len(filtered) == 0
        assert result.kill_switch_engaged is True
        assert result.total_orders_before == 3
        assert result.total_orders_after == 0

        # Write to SAFE CSV - should be empty
        safe_path = tmp_path / "orders_test.csv"
        write_safe_orders_csv(
            filtered,
            output_path=safe_path,
            date=None,
            price_type="MARKET",
            comment="Test",
        )

        # Read back and verify empty
        written_df = pd.read_csv(safe_path)
        assert len(written_df) == 0

    def test_order_flow_passes_valid_orders(
        self, sample_target_positions, sample_prices, tmp_path: Path
    ):
        """Test that valid orders pass through risk controls."""
        # Generate orders
        orders = generate_orders_from_targets(
            sample_target_positions, prices=sample_prices
        )

        # Apply risk controls with lenient limits
        config = PreTradeConfig(max_notional_per_symbol=1000000.0)  # High limit

        filtered, result = filter_orders_with_risk_controls(
            orders,
            pre_trade_config=config,
            enable_pre_trade_checks=True,
            enable_kill_switch=False,
        )

        # All orders should pass
        assert len(filtered) == len(orders)
        assert result.total_orders_before == 3
        assert result.total_orders_after == 3
        assert result.pre_trade_result is not None
        assert result.pre_trade_result.is_ok

    def test_order_flow_combined_controls(self, sample_target_positions, sample_prices):
        """Test that both pre-trade checks and kill switch work together."""
        # Generate orders
        orders = generate_orders_from_targets(
            sample_target_positions, prices=sample_prices
        )

        # Set kill switch AND strict pre-trade limits
        config = PreTradeConfig(max_notional_per_symbol=10000.0)

        with patch.dict(os.environ, {"ASSEMBLED_KILL_SWITCH": "1"}, clear=False):
            filtered, result = filter_orders_with_risk_controls(
                orders,
                pre_trade_config=config,
                enable_pre_trade_checks=True,
                enable_kill_switch=True,
            )

        # All orders should be blocked (either by pre-trade checks or kill switch)
        assert len(filtered) == 0
        assert result.kill_switch_engaged is True
        assert result.total_orders_after == 0

    def test_order_flow_empty_orders_passes(self):
        """Test that empty orders pass through risk controls."""
        empty_orders = pd.DataFrame(
            columns=["timestamp", "symbol", "side", "qty", "price"]
        )

        filtered, result = filter_orders_with_risk_controls(
            empty_orders, enable_pre_trade_checks=True, enable_kill_switch=True
        )

        assert len(filtered) == 0
        assert result.total_orders_before == 0
        assert result.total_orders_after == 0
        assert result.pre_trade_result is not None
        assert result.pre_trade_result.is_ok  # Empty orders are OK
