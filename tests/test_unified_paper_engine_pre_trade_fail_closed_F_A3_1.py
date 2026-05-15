"""Regression test for F-A3-1: unified_paper_engine pre_trade_checks fail-CLOSED.

R3 audit (F-A3-1): _apply_risk_controls wrapped run_pre_trade_checks in
try/except: logger.warning + continue with UNFILTERED orders. A dependency
drift, data-shape error, or transient bug in pre_trade_checks would silently
bypass all pre-trade safeguards.

R4 fix (880cb38): on exception, log ERROR + replace orders with empty
DataFrame (same columns) — equivalent to "all rejected". Aligned with
risk_controls L545 fail-closed path.

R6 test backfill: closes F-C4-N-5 process-gap for F-A3-1.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

pytestmark = [pytest.mark.unit]


def _orders_df(n: int = 3) -> pd.DataFrame:
    """Build an N-row orders DataFrame with the standard schema."""
    return pd.DataFrame(
        [
            {
                "symbol": f"SYM{i}",
                "side": "BUY",
                "qty": 100.0,
                "price": 50.0 + i,
            }
            for i in range(n)
        ]
    )


def test_pre_trade_exception_returns_empty_orders_F_A3_1() -> None:
    """When run_pre_trade_checks raises, _apply_risk_controls must return
    empty DataFrame (all-rejected), NOT the unfiltered input.
    """
    from src.assembled_core.execution.unified_paper_engine import (
        UnifiedPaperConfig,
        UnifiedPaperEngine,
    )

    eng = UnifiedPaperEngine(UnifiedPaperConfig(seed_capital=10000.0))
    orders = _orders_df(3)

    # Patch run_pre_trade_checks to raise. The patch target must match the
    # imported name inside unified_paper_engine (it imports the name into
    # the module namespace).
    with patch(
        "src.assembled_core.execution.unified_paper_engine.run_pre_trade_checks",
        side_effect=RuntimeError("simulated pre_trade failure"),
    ):
        result = eng._apply_risk_controls(orders)

    assert result.empty, (
        "F-A3-1 regression: pre_trade_checks exception must yield empty orders "
        f"(fail-closed), got {len(result)} rows"
    )
    # Same columns preserved (so downstream loop sees expected schema)
    assert list(result.columns) == list(orders.columns)


def test_pre_trade_success_passes_filtered_orders_F_A3_1() -> None:
    """Normal-path: when run_pre_trade_checks returns successfully, orders flow through."""
    from src.assembled_core.execution.unified_paper_engine import (
        UnifiedPaperConfig,
        UnifiedPaperEngine,
    )

    eng = UnifiedPaperEngine(UnifiedPaperConfig(seed_capital=10000.0))
    orders = _orders_df(3)

    class _FakeResult:
        is_ok = True
        blocked_reasons: list[str] = []

    # Return tuple (result, filtered_orders) — the documented return shape
    with patch(
        "src.assembled_core.execution.unified_paper_engine.run_pre_trade_checks",
        return_value=(_FakeResult(), orders.iloc[:2].copy()),
    ):
        result = eng._apply_risk_controls(orders)

    # Filtered orders survive
    assert len(result) == 2


def test_pre_trade_exception_logs_error_F_A3_1(caplog) -> None:
    """Fail-closed path must log ERROR with exc_info (not silent or WARN)."""
    import logging

    from src.assembled_core.execution.unified_paper_engine import (
        UnifiedPaperConfig,
        UnifiedPaperEngine,
    )

    eng = UnifiedPaperEngine(UnifiedPaperConfig(seed_capital=10000.0))
    orders = _orders_df(2)

    with caplog.at_level(logging.ERROR):
        with patch(
            "src.assembled_core.execution.unified_paper_engine.run_pre_trade_checks",
            side_effect=ValueError("xyz simulated"),
        ):
            eng._apply_risk_controls(orders)

    # Must contain an ERROR-level message mentioning the exception
    error_records = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert error_records, "F-A3-1: must log at ERROR on pre_trade exception"
    assert any("pre_trade_checks" in r.getMessage() for r in error_records)
    assert any("xyz simulated" in r.getMessage() for r in error_records)
