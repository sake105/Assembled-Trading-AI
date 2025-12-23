"""Tests for execution.pre_trade_checks module."""

from __future__ import annotations

import pandas as pd
import pytest

pytestmark = pytest.mark.phase10

from src.assembled_core.execution.pre_trade_checks import (
    PreTradeConfig,
    run_pre_trade_checks,
)
from src.assembled_core.qa.qa_gates import QAGateResult, QAGatesSummary, QAResult


@pytest.fixture
def sample_orders() -> pd.DataFrame:
    """Create sample orders DataFrame."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "GOOGL", "MSFT"],
            "side": ["BUY", "BUY", "SELL"],
            "qty": [100, 50, 75],
            "price": [150.0, 2500.0, 300.0],
        }
    )


@pytest.fixture
def sample_portfolio() -> pd.DataFrame:
    """Create sample portfolio DataFrame."""
    return pd.DataFrame(
        {
            "symbol": ["AAPL", "GOOGL", "MSFT"],
            "qty": [50, 25, 100],
            "weight": [0.1, 0.15, 0.25],
        }
    )


@pytest.fixture
def qa_status_ok() -> QAGatesSummary:
    """Create QA status with OK result."""
    return QAGatesSummary(
        overall_result=QAResult.OK,
        passed_gates=5,
        warning_gates=0,
        blocked_gates=0,
        gate_results=[],
    )


@pytest.fixture
def qa_status_block() -> QAGatesSummary:
    """Create QA status with BLOCK result."""
    return QAGatesSummary(
        overall_result=QAResult.BLOCK,
        passed_gates=2,
        warning_gates=1,
        blocked_gates=2,
        gate_results=[
            QAGateResult(
                gate_name="sharpe_ratio",
                result=QAResult.BLOCK,
                reason="Sharpe ratio too low",
                details={"sharpe_ratio": 0.3},
            )
        ],
    )


class TestPreTradeChecks:
    """Tests for run_pre_trade_checks function."""

    def test_pre_trade_checks_empty_orders_is_ok(self):
        """Test that empty orders list passes checks."""
        orders = pd.DataFrame(columns=["symbol", "side", "qty", "price"])

        result, filtered = run_pre_trade_checks(orders, config=PreTradeConfig())

        assert result.is_ok is True
        assert len(result.blocked_reasons) == 0
        assert len(filtered) == 0
        assert result.summary["total_orders"] == 0
        assert result.summary["passed_orders"] == 0

    def test_pre_trade_checks_block_position_too_large(self, sample_orders):
        """Test that orders exceeding max_notional_per_symbol are blocked."""
        config = PreTradeConfig(max_notional_per_symbol=10000.0)

        # AAPL: 100 * 150 = 15000 (> 10000) - should be blocked
        # GOOGL: 50 * 2500 = 125000 (> 10000) - should be blocked
        # MSFT: 75 * 300 = 22500 (> 10000) - should be blocked

        result, filtered = run_pre_trade_checks(sample_orders, config=config)

        assert result.is_ok is False
        assert len(result.blocked_reasons) > 0
        assert any(
            "max_notional_per_symbol" in reason for reason in result.blocked_reasons
        )
        assert len(filtered) == 0  # All orders blocked

    def test_pre_trade_checks_allow_small_positions(self):
        """Test that orders within limits pass checks."""
        orders = pd.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL"],
                "side": ["BUY", "BUY"],
                "qty": [10, 2],
                "price": [150.0, 2500.0],
            }
        )
        config = PreTradeConfig(max_notional_per_symbol=10000.0)

        # AAPL: 10 * 150 = 1500 (< 10000) - OK
        # GOOGL: 2 * 2500 = 5000 (< 10000) - OK

        result, filtered = run_pre_trade_checks(orders, config=config)

        assert result.is_ok is True
        assert len(result.blocked_reasons) == 0
        assert len(filtered) == 2  # All orders passed

    def test_pre_trade_checks_respect_gross_exposure_limit(self, sample_orders):
        """Test that orders exceeding max_gross_exposure are blocked."""
        config = PreTradeConfig(max_gross_exposure=50000.0)

        # Total gross exposure: 100*150 + 50*2500 + 75*300 = 15000 + 125000 + 22500 = 162500
        # This exceeds 50000, so all orders should be blocked

        result, filtered = run_pre_trade_checks(sample_orders, config=config)

        assert result.is_ok is False
        assert len(result.blocked_reasons) > 0
        assert any("max_gross_exposure" in reason for reason in result.blocked_reasons)
        assert len(filtered) == 0  # All orders blocked

    def test_pre_trade_checks_respect_qa_block(self, sample_orders, qa_status_block):
        """Test that orders are blocked when QA status is BLOCK."""
        config = PreTradeConfig()

        result, filtered = run_pre_trade_checks(
            sample_orders, qa_status=qa_status_block, config=config
        )

        assert result.is_ok is False
        assert len(result.blocked_reasons) > 0
        assert any("QA_BLOCK" in reason for reason in result.blocked_reasons)
        assert len(filtered) == 0
        assert result.summary.get("qa_blocked") is True

    def test_pre_trade_checks_pass_with_qa_ok(self, sample_orders, qa_status_ok):
        """Test that orders pass when QA status is OK."""
        config = PreTradeConfig()  # No limits

        result, filtered = run_pre_trade_checks(
            sample_orders, qa_status=qa_status_ok, config=config
        )

        assert result.is_ok is True
        assert len(result.blocked_reasons) == 0
        assert len(filtered) == len(sample_orders)

    def test_pre_trade_checks_partial_block(self):
        """Test that only exceeding orders are blocked, others pass."""
        orders = pd.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL"],
                "side": ["BUY", "BUY"],
                "qty": [10, 200],  # GOOGL exceeds limit
                "price": [150.0, 2500.0],
            }
        )
        config = PreTradeConfig(max_notional_per_symbol=10000.0)

        # AAPL: 10 * 150 = 1500 (< 10000) - OK
        # GOOGL: 200 * 2500 = 500000 (> 10000) - BLOCKED

        result, filtered = run_pre_trade_checks(orders, config=config)

        assert result.is_ok is False  # Some orders blocked
        assert len(result.blocked_reasons) > 0
        assert len(filtered) == 1  # Only AAPL passes
        assert filtered["symbol"].iloc[0] == "AAPL"

    def test_pre_trade_checks_no_prices_skips_notional_check(self):
        """Test that checks are skipped when prices are missing."""
        orders = pd.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL"],
                "side": ["BUY", "BUY"],
                "qty": [100, 50],
                # No price column
            }
        )
        config = PreTradeConfig(max_notional_per_symbol=10000.0)

        # Without prices, notional check is skipped
        result, filtered = run_pre_trade_checks(orders, config=config)

        # Should pass (no way to check notional without prices)
        assert result.is_ok is True
        assert len(filtered) == 2

    def test_pre_trade_checks_missing_required_columns(self):
        """Test that missing required columns are handled."""
        orders = pd.DataFrame(
            {
                "symbol": ["AAPL"],
                # Missing qty column
            }
        )

        result, filtered = run_pre_trade_checks(orders, config=PreTradeConfig())

        assert result.is_ok is False
        assert len(result.blocked_reasons) > 0
        assert any(
            "missing required columns" in reason.lower()
            for reason in result.blocked_reasons
        )

    def test_pre_trade_checks_combined_limits(self):
        """Test multiple limits together."""
        orders = pd.DataFrame(
            {
                "symbol": ["AAPL", "GOOGL"],
                "side": ["BUY", "BUY"],
                "qty": [100, 50],
                "price": [150.0, 2500.0],
            }
        )
        config = PreTradeConfig(
            max_notional_per_symbol=20000.0,  # Both symbols pass this
            max_gross_exposure=50000.0,  # But total exceeds this
        )

        result, filtered = run_pre_trade_checks(orders, config=config)

        # Gross exposure check should block all orders
        assert result.is_ok is False
        assert any("max_gross_exposure" in reason for reason in result.blocked_reasons)
        assert len(filtered) == 0

    def test_pre_trade_checks_no_config_no_limits(self, sample_orders):
        """Test that with no config (or None limits), all orders pass."""
        config = PreTradeConfig()  # All limits None

        result, filtered = run_pre_trade_checks(sample_orders, config=config)

        assert result.is_ok is True
        assert len(result.blocked_reasons) == 0
        assert len(filtered) == len(sample_orders)

    def test_pre_trade_checks_summary_metrics(self, sample_orders):
        """Test that summary contains expected metrics."""
        config = PreTradeConfig(max_gross_exposure=1000000.0)  # High limit, should pass

        result, filtered = run_pre_trade_checks(sample_orders, config=config)

        assert "total_orders" in result.summary
        assert "passed_orders" in result.summary
        assert "blocked_orders" in result.summary
        assert result.summary["total_orders"] == len(sample_orders)
        assert result.summary["passed_orders"] == len(filtered)
        assert result.summary["blocked_orders"] == len(sample_orders) - len(filtered)
