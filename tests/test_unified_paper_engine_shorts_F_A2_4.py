"""Regression tests for F-A2-4: unified_paper_engine._update_positions shorts.

Sister to tests/test_ops_paper_ledger.py F-A-1 regressions. Verifies the
same 9-branch short logic (BUY/SELL × long/short/cover/oversell/add) holds
in the unified_paper_engine implementation.

Process-gap note: F-A2-4 was BLOCKER-grade but the original fix (commit
d8d1034) shipped without targeted tests. R4 audit (F-C4-N-5) flagged this.
This file closes that gap.
"""

from __future__ import annotations

import pandas as pd
import pytest


pytestmark = [pytest.mark.unit]


def _engine_with_state(cash: float, positions: dict, cost_basis: dict):
    """Build a UnifiedPaperEngine with the given starting state.

    Uses minimal config; the _update_positions method only touches
    self._state and self.config.seed_capital so we don't need a full engine.
    """
    from src.assembled_core.execution.unified_paper_engine import (
        UnifiedPaperConfig,
        UnifiedPaperEngine,
    )

    eng = UnifiedPaperEngine(UnifiedPaperConfig(seed_capital=cash))
    eng._state["cash"] = cash
    eng._state["positions"] = dict(positions)
    eng._state["cost_basis"] = dict(cost_basis)
    return eng


def _fill(symbol: str, side: str, qty: float, fill_price: float) -> pd.DataFrame:
    """Single-fill DataFrame matching unified_paper_engine._update_positions schema."""
    return pd.DataFrame(
        [
            {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "fill_qty": qty,
                "fill_price": fill_price,
                "notional": qty * fill_price,
                "status": "filled",
            }
        ]
    )


def test_buy_long_add_F_A2_4() -> None:
    """BUY on existing long: qty adds, weighted avg cost."""
    eng = _engine_with_state(10000.0, {"A": 10.0}, {"A": 90.0})
    eng._update_positions(_fill("A", "BUY", 5.0, 100.0))
    assert eng._state["positions"]["A"] == 15.0
    # (10*90 + 5*100) / 15 = (900 + 500) / 15 ≈ 93.33
    assert abs(eng._state["cost_basis"]["A"] - 93.333333) < 1e-4
    assert eng._state["cash"] == 9500.0


def test_sell_long_partial_F_A2_4() -> None:
    """SELL part of long: qty reduces, cost_basis preserved."""
    eng = _engine_with_state(10000.0, {"A": 10.0}, {"A": 90.0})
    eng._update_positions(_fill("A", "SELL", 4.0, 100.0))
    assert eng._state["positions"]["A"] == 6.0
    assert eng._state["cost_basis"]["A"] == 90.0
    assert eng._state["cash"] == 10400.0


def test_sell_long_full_close_F_A2_4() -> None:
    """SELL exact qty: position and cost_basis popped."""
    eng = _engine_with_state(10000.0, {"A": 10.0}, {"A": 90.0})
    eng._update_positions(_fill("A", "SELL", 10.0, 100.0))
    assert "A" not in eng._state["positions"]
    assert "A" not in eng._state["cost_basis"]
    assert eng._state["cash"] == 11000.0


def test_sell_oversell_flips_long_to_short_F_A2_4() -> None:
    """SELL qty > long: full close + opens short for overflow at fill price.

    Regression for F-A2-4 BLOCKER: previously credited only sold_qty=long_qty
    and silently dropped the overflow shares.
    """
    eng = _engine_with_state(10000.0, {"A": 5.0}, {"A": 90.0})
    eng._update_positions(_fill("A", "SELL", 8.0, 100.0))
    # Cash credited for FULL 8 shares
    assert eng._state["cash"] == 10800.0
    # Short opened for overflow 3 shares at fill price
    assert eng._state["positions"]["A"] == -3.0
    assert eng._state["cost_basis"]["A"] == 100.0


def test_sell_open_short_on_zero_F_A2_4() -> None:
    """SELL on zero position: opens short, cash credited.

    Regression for F-A2-4 BLOCKER: previously silently dropped.
    """
    eng = _engine_with_state(10000.0, {}, {})
    eng._update_positions(_fill("A", "SELL", 10.0, 100.0))
    assert eng._state["cash"] == 11000.0
    assert eng._state["positions"]["A"] == -10.0
    assert eng._state["cost_basis"]["A"] == 100.0


def test_sell_add_to_short_weighted_avg_F_A2_4() -> None:
    """SELL adds to existing short: weighted short avg."""
    eng = _engine_with_state(10000.0, {"A": -10.0}, {"A": 100.0})
    eng._update_positions(_fill("A", "SELL", 10.0, 110.0))
    assert eng._state["cash"] == 11100.0
    assert eng._state["positions"]["A"] == -20.0
    # Weighted: (10*100 + 10*110) / 20 = 105
    assert eng._state["cost_basis"]["A"] == 105.0


def test_buy_cover_short_partial_F_A2_4() -> None:
    """BUY covers part of short: short avg preserved."""
    eng = _engine_with_state(10000.0, {"A": -10.0}, {"A": 100.0})
    eng._update_positions(_fill("A", "BUY", 4.0, 90.0))
    assert eng._state["cash"] == 10000.0 - 4 * 90  # 9640
    assert eng._state["positions"]["A"] == -6.0
    assert eng._state["cost_basis"]["A"] == 100.0


def test_buy_cover_short_exact_F_A2_4() -> None:
    """BUY exactly covers short: position+cost_basis popped."""
    eng = _engine_with_state(10000.0, {"A": -10.0}, {"A": 100.0})
    eng._update_positions(_fill("A", "BUY", 10.0, 90.0))
    assert eng._state["cash"] == 10000.0 - 10 * 90  # 9100
    assert "A" not in eng._state["positions"]
    assert "A" not in eng._state["cost_basis"]


def test_buy_cover_short_and_flip_to_long_F_A2_4() -> None:
    """BUY covers short + opens long: new long at fill price (not blended)."""
    eng = _engine_with_state(10000.0, {"A": -10.0}, {"A": 100.0})
    eng._update_positions(_fill("A", "BUY", 15.0, 90.0))
    assert eng._state["cash"] == 10000.0 - 15 * 90  # 8650
    assert eng._state["positions"]["A"] == 5.0
    # Cost-basis is REPLACED at fill price (not blended with short avg)
    assert eng._state["cost_basis"]["A"] == 90.0


def test_sell_residual_short_takes_open_short_branch_B_exec_4() -> None:
    """B-exec-4: a residual short like -1e-12 must be treated as effectively flat.

    The SELL/short-open branch used ``if current_qty == 0`` (exact float ==).
    A residual short (e.g. -1e-12 left after a near-exact cover) would slip past
    the == 0 check into the weighted-short-avg branch, blending the new fill
    against a stale/garbage prior short cost. With the tolerance compare
    (abs(current_qty) <= 1e-8, matching the sibling SELL branch) the new short
    cost_basis is set cleanly to the fill price, not contaminated by the stale
    cost. Here the stale cost is deliberately garbage (99999) to make a wrong
    weighted-blend visible.
    """
    eng = _engine_with_state(10000.0, {"A": -1e-12}, {"A": 99999.0})
    eng._update_positions(_fill("A", "SELL", 10.0, 100.0))
    # Cash credited for the full 10 shares.
    assert eng._state["cash"] == 11000.0
    # Position is the new short (residual is negligible).
    assert abs(eng._state["positions"]["A"] - (-10.0)) < 1e-6
    # Cost-basis is the clean fill price — NOT blended with the stale 99999 cost.
    assert eng._state["cost_basis"]["A"] == 100.0
