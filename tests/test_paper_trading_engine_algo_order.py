"""Tests for PaperTradingEngine.submit_algo_order (TWAP/VWAP slicing).

Regression coverage for the latent-correctness fix of ``submit_algo_order``:
the previous implementation called ``scheduler.schedule(total_quantity=...,
reference_price=...)``, which does not match the real scheduler signature
``schedule(symbol, total_qty, side, start_time, end_time, ...)`` and therefore
raised ``TypeError`` on every invocation. The method had no live/paper caller,
so this was dead code; the fix wires the correct signature so the method works.

These tests call the method end-to-end and assert it slices, fills, and
conserves quantity.
"""

from __future__ import annotations

import pytest

from src.assembled_core.execution.paper_trading_engine import PaperTradingEngine


@pytest.mark.fast
def test_submit_algo_order_twap_slices_and_fills() -> None:
    engine = PaperTradingEngine(initial_cash=1_000_000.0)

    filled = engine.submit_algo_order(
        symbol="AAPL",
        side="BUY",
        total_quantity=1000.0,
        price=150.0,
        algo="TWAP",
        n_slices=5,
    )

    # The scheduler caps slices at min(n_slices, int(total_qty)); for 1000 units
    # and 5 requested slices we expect exactly 5 filled slice orders.
    assert len(filled) == 5
    assert all(o.status == "FILLED" for o in filled)
    assert all(o.symbol == "AAPL" for o in filled)
    assert all(o.side == "BUY" for o in filled)
    assert all(o.source == "ALGO" for o in filled)
    # Each slice fills at the reference price (paper fills immediately).
    assert all(o.price == 150.0 for o in filled)

    # Total filled quantity is conserved (TWAP randomizes per-slice but preserves
    # the sum). Allow a tiny float tolerance.
    total_filled = sum(o.quantity for o in filled)
    assert total_filled == pytest.approx(1000.0, abs=1e-6)


@pytest.mark.fast
def test_submit_algo_order_vwap_runs_without_error() -> None:
    engine = PaperTradingEngine(initial_cash=1_000_000.0)

    filled = engine.submit_algo_order(
        symbol="MSFT",
        side="SELL",
        total_quantity=500.0,
        price=300.0,
        algo="VWAP",
        n_slices=4,
        participation_rate=0.10,
    )

    assert len(filled) == 4
    assert all(o.status == "FILLED" for o in filled)
    assert all(o.side == "SELL" for o in filled)
    total_filled = sum(o.quantity for o in filled)
    assert total_filled == pytest.approx(500.0, abs=1e-6)


@pytest.mark.fast
def test_submit_algo_order_deterministic_client_ids_unique() -> None:
    engine = PaperTradingEngine(initial_cash=1_000_000.0)

    filled = engine.submit_algo_order(
        symbol="NVDA",
        side="BUY",
        total_quantity=300.0,
        price=120.0,
        algo="TWAP",
        n_slices=3,
    )

    client_ids = [o.client_order_id for o in filled]
    # Each slice gets a unique deterministic client_order_id.
    assert len(set(client_ids)) == len(client_ids)
    assert all(cid for cid in client_ids)
