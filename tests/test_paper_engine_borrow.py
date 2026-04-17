"""Phase 5 regression tests for borrow-cost accrual.

Covers:

* ``compute_borrow_cost``: long/zero-qty/zero-rate produce 0.0
* short at 500bps on 1M notional for 1 day ≈ 500/10_000/365 * 1M
* ``BorrowRateTable``: overrides, htb_symbols, default rate lookup
* engine with ``enable_borrow_costs=False`` → cash unchanged
* engine with enabled borrow and a short → cash reduced by expected amount
* long-only portfolio under enabled borrow → cash unchanged
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.execution.borrow_costs import (
    EASY_TO_BORROW_BPS,
    HARD_TO_BORROW_BPS,
    BorrowRateTable,
    compute_borrow_cost,
    compute_borrow_cost_for_positions,
)
from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


# --- compute_borrow_cost -----------------------------------------------------


def test_compute_borrow_cost_long_position_is_zero() -> None:
    assert compute_borrow_cost(100.0, 50.0, 500.0) == 0.0


def test_compute_borrow_cost_zero_price_is_zero() -> None:
    assert compute_borrow_cost(-100.0, 0.0, 500.0) == 0.0


def test_compute_borrow_cost_zero_rate_is_zero() -> None:
    assert compute_borrow_cost(-100.0, 50.0, 0.0) == 0.0


def test_compute_borrow_cost_short_500bps() -> None:
    """Short 100 shares @ $100 = $10_000 notional, 500bps annual, 1 day/365."""
    cost = compute_borrow_cost(-100.0, 100.0, 500.0, days_held=1)
    expected = 10_000.0 * 0.05 / 365
    assert cost == pytest.approx(expected)


def test_compute_borrow_cost_scales_with_days_held() -> None:
    c1 = compute_borrow_cost(-100.0, 100.0, 500.0, days_held=1)
    c10 = compute_borrow_cost(-100.0, 100.0, 500.0, days_held=10)
    assert c10 == pytest.approx(c1 * 10.0)


# --- BorrowRateTable ---------------------------------------------------------


def test_rate_table_default_and_htb() -> None:
    t = BorrowRateTable(htb_symbols={"GME"})
    assert t.rate_bps("AAPL") == EASY_TO_BORROW_BPS
    assert t.rate_bps("GME") == HARD_TO_BORROW_BPS


def test_rate_table_override_beats_htb() -> None:
    t = BorrowRateTable(overrides={"GME": 1200.0}, htb_symbols={"GME"})
    assert t.rate_bps("GME") == 1200.0


# --- compute_borrow_cost_for_positions ---------------------------------------


def test_cost_for_positions_only_shorts() -> None:
    positions = {"AAPL": 100.0, "GME": -50.0, "TSLA": 0.0}
    prices = {"AAPL": 150.0, "GME": 20.0, "TSLA": 200.0}
    table = BorrowRateTable(htb_symbols={"GME"})
    costs = compute_borrow_cost_for_positions(positions, prices, table)
    assert "AAPL" not in costs
    assert "TSLA" not in costs
    assert "GME" in costs
    assert costs["GME"] == pytest.approx(
        compute_borrow_cost(-50.0, 20.0, HARD_TO_BORROW_BPS)
    )


# --- engine integration ------------------------------------------------------


def _make_engine(tmp_path: Path, *, enable_borrow_costs: bool) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        enable_borrow_costs=enable_borrow_costs,
        htb_symbols=("GME",),
        run_id="borrow_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 1_000_000.0, "positions": {}, "cost_basis": {}}
    return eng


def test_engine_borrow_disabled_no_cash_change(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path, enable_borrow_costs=False)
    eng._state["positions"]["GME"] = -100.0
    prices = pd.DataFrame([{"symbol": "GME", "close": 20.0}])
    total = eng._apply_borrow_costs("2025-01-15", prices)
    assert total == 0.0
    assert eng._state["cash"] == 1_000_000.0


def test_engine_borrow_enabled_on_short_reduces_cash(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path, enable_borrow_costs=True)
    eng._state["positions"]["GME"] = -100.0
    prices = pd.DataFrame([{"symbol": "GME", "close": 20.0}])
    total = eng._apply_borrow_costs("2025-01-15", prices)
    expected = compute_borrow_cost(-100.0, 20.0, HARD_TO_BORROW_BPS)
    assert total == pytest.approx(expected)
    assert eng._state["cash"] == pytest.approx(1_000_000.0 - expected)
    # History appended
    hist = eng._state["borrow_cost_history"]
    assert len(hist) == 1
    assert hist[0]["date"] == "2025-01-15"


def test_engine_borrow_long_only_noop(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path, enable_borrow_costs=True)
    eng._state["positions"]["AAPL"] = 100.0
    prices = pd.DataFrame([{"symbol": "AAPL", "close": 150.0}])
    total = eng._apply_borrow_costs("2025-01-15", prices)
    assert total == 0.0
    assert eng._state["cash"] == 1_000_000.0
