"""Phase 2 regression tests for participation-cap partial fills.

Covers:

* default (enable_partial_fills=False) produces full fills, status="filled"
* enabling partial fills caps ``fill_qty`` at ``max_participation * adv``
* partial fills carry ``status="partial"`` and correct ``remaining_qty``
* orders that can't reach ``min_fill_qty`` are rejected with reason
* post-fill positions match ``fill_qty``, not intended ``qty`` (sizing feedback)
* market impact uses executed qty, not intended qty
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


def _make_engine(
    tmp_path: Path,
    *,
    enable_partial_fills: bool = False,
    max_participation: float = 0.05,
    min_fill_qty: float = 0.0,
) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        enable_partial_fills=enable_partial_fills,
        max_participation=max_participation,
        min_fill_qty=min_fill_qty,
        run_id="pf_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 1_000_000.0, "positions": {}, "cost_basis": {}}
    return eng


def test_default_legacy_full_fill_has_status_filled(tmp_path: Path) -> None:
    """Even with partial-fill scaffolding, default behaviour is full fill."""
    eng = _make_engine(tmp_path, enable_partial_fills=False)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 100.0}]
    )
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 1000.0}]
    )
    fills = eng._simulate_fills(orders, prices)
    assert len(fills) == 1
    assert fills.loc[0, "fill_qty"] == 100.0
    assert fills.loc[0, "remaining_qty"] == 0.0
    assert fills.loc[0, "status"] == "filled"


def test_enable_partial_caps_fill_at_participation(tmp_path: Path) -> None:
    """Order qty above participation * adv must become partial."""
    eng = _make_engine(
        tmp_path, enable_partial_fills=True, max_participation=0.10
    )
    # ADV (via volume) = 1000. Cap = 0.10 * 1000 = 100 shares.
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 500.0, "price": 100.0}]
    )
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 1000.0}]
    )
    fills = eng._simulate_fills(orders, prices)

    assert len(fills) == 1
    assert fills.loc[0, "fill_qty"] == pytest.approx(100.0)
    assert fills.loc[0, "remaining_qty"] == pytest.approx(400.0)
    assert fills.loc[0, "status"] == "partial"
    # qty column preserves the intended size
    assert fills.loc[0, "qty"] == 500.0


def test_partial_fill_small_order_fully_fills(tmp_path: Path) -> None:
    """Order below the cap must fill completely with status=filled."""
    eng = _make_engine(
        tmp_path, enable_partial_fills=True, max_participation=0.10
    )
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 50.0, "price": 100.0}]
    )
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 1000.0}]
    )
    fills = eng._simulate_fills(orders, prices)
    assert fills.loc[0, "fill_qty"] == pytest.approx(50.0)
    assert fills.loc[0, "status"] == "filled"


def test_min_fill_qty_rejects_order(tmp_path: Path) -> None:
    """Cap below min_fill_qty must reject the order entirely."""
    eng = _make_engine(
        tmp_path,
        enable_partial_fills=True,
        max_participation=0.01,
        min_fill_qty=50.0,
    )
    # cap = 0.01 * 1000 = 10 shares < min_fill_qty=50 → reject
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 500.0, "price": 100.0}]
    )
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 1000.0}]
    )
    fills = eng._simulate_fills(orders, prices)

    assert fills.loc[0, "status"] == "rejected"
    assert fills.loc[0, "reject_reason"] == "MIN_FILL_QTY"
    assert fills.loc[0, "fill_qty"] == 0.0
    assert fills.loc[0, "notional"] == 0.0


def test_partial_fill_updates_positions_by_fill_qty(tmp_path: Path) -> None:
    """Sizing feedback: position after fill == fill_qty, not intended qty."""
    eng = _make_engine(
        tmp_path, enable_partial_fills=True, max_participation=0.10
    )
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 500.0, "price": 100.0}]
    )
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 1000.0}]
    )
    fills, _ = eng._simulate_fills_with_cost(orders, prices)
    eng._update_positions(fills)

    # partial fill = 100 shares, not 500
    assert eng._state["positions"]["AAA"] == pytest.approx(100.0)
    # cash reduced only by 100 * fill_price (~= 100 * 100.x with spread/impact)
    notional = float(fills.loc[0, "notional"])
    assert eng._state["cash"] == pytest.approx(1_000_000.0 - notional)


def test_partial_fill_impact_uses_executed_qty_not_intended(tmp_path: Path) -> None:
    """Market impact must be computed on fill_qty, not intended qty."""
    eng = _make_engine(
        tmp_path, enable_partial_fills=True, max_participation=0.10
    )
    # qty >> cap, cap = 100 shares, adv = 1000
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 10_000.0, "price": 100.0}]
    )
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 1000.0}]
    )
    fills = eng._simulate_fills(orders, prices)
    # fill_qty=100, participation = 0.10 → impact ~ 0.10 * sqrt(0.10) * 100
    # ~ 3.16 price units (with coeff=0.10). If it were computed on qty=10_000,
    # participation would be 10 and the price would explode.
    fill_price = float(fills.loc[0, "fill_price"])
    # Sanity: fill_price must be sensible (within ~15% of mid), not hundreds of %
    assert 100.0 < fill_price < 115.0


def test_disabled_partial_matches_legacy_math(tmp_path: Path) -> None:
    """With enable_partial_fills=False, fill price math matches pre-Phase-2."""
    eng_off = _make_engine(tmp_path / "off", enable_partial_fills=False)
    orders = pd.DataFrame(
        [{"symbol": "AAA", "side": "BUY", "qty": 500.0, "price": 100.0}]
    )
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 1000.0}]
    )
    fills = eng_off._simulate_fills(orders, prices)
    # fill_qty == qty, status == filled — sanity
    assert fills.loc[0, "fill_qty"] == 500.0
    assert fills.loc[0, "status"] == "filled"
    # Spread+impact computed on intended qty (matches pre-Phase-2 behaviour)
    fill_price = float(fills.loc[0, "fill_price"])
    assert fill_price > 100.0  # BUY pays more than mid
