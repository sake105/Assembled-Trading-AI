"""Phase 0 regression tests for UnifiedPaperEngine.

These tests lock in the partial-fill-aware behavior of ``_update_positions``
and ``_write_ledger_events``. They cover:

* legacy full-fill path (no fill_qty / status columns) stays bit-identical
* fill_qty < qty correctly drives both cash and position math
* status == "rejected" is treated as a no-op for accounting and is written to
  the ledger as a REJECT event (qty=0, cash_delta=0)
* event_id is unique across side when the same symbol trades BUY+SELL on one day
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)


def _make_engine(tmp_path: Path, seed: float = 100_000.0) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=seed,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        run_id="test_run",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": seed, "positions": {}, "cost_basis": {}}
    return eng


# --- _update_positions ------------------------------------------------------


def test_update_positions_legacy_full_fill_unchanged(tmp_path: Path) -> None:
    """Without fill_qty/status columns, behaviour matches the pre-Phase-0 path."""
    eng = _make_engine(tmp_path)
    fills = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "side": "BUY",
                "qty": 10.0,
                "fill_price": 100.0,
                "notional": 1000.0,
            },
        ]
    )

    eng._update_positions(fills)

    assert eng._state["positions"]["AAA"] == 10.0
    assert eng._state["cost_basis"]["AAA"] == 100.0
    assert eng._state["cash"] == pytest.approx(99_000.0)


def test_update_positions_uses_fill_qty_when_partial(tmp_path: Path) -> None:
    """Partial fills must move only fill_qty shares and fill_qty * price cash."""
    eng = _make_engine(tmp_path)
    fills = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "side": "BUY",
                "qty": 10.0,
                "fill_qty": 3.0,
                "fill_price": 100.0,
                "status": "partial",
            },
        ]
    )

    eng._update_positions(fills)

    assert eng._state["positions"]["AAA"] == 3.0
    assert eng._state["cost_basis"]["AAA"] == pytest.approx(100.0)
    assert eng._state["cash"] == pytest.approx(100_000.0 - 300.0)


def test_update_positions_skips_rejected(tmp_path: Path) -> None:
    """Rejected rows must leave cash and positions completely untouched."""
    eng = _make_engine(tmp_path)
    fills = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "side": "BUY",
                "qty": 10.0,
                "fill_qty": 0.0,
                "fill_price": 100.0,
                "status": "rejected",
                "reject_reason": "INSUFFICIENT_CASH",
            },
        ]
    )

    eng._update_positions(fills)

    assert "AAA" not in eng._state["positions"]
    assert eng._state["cash"] == pytest.approx(100_000.0)


def test_update_positions_partial_sell_respects_fill_qty(tmp_path: Path) -> None:
    """A partial SELL reduces position by fill_qty, not the intended qty."""
    eng = _make_engine(tmp_path)
    eng._state["positions"]["AAA"] = 10.0
    eng._state["cost_basis"]["AAA"] = 100.0

    fills = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "side": "SELL",
                "qty": 10.0,
                "fill_qty": 4.0,
                "fill_price": 110.0,
                "status": "partial",
            },
        ]
    )
    eng._update_positions(fills)

    assert eng._state["positions"]["AAA"] == pytest.approx(6.0)
    assert eng._state["cash"] == pytest.approx(100_000.0 + 4.0 * 110.0)


# --- _write_ledger_events ---------------------------------------------------


def _canonical_ledger_path(eng: UnifiedPaperEngine) -> Path:
    """Canonical ledger artifact path written by the re-activated store.

    Re-activation (2026-06-04, Option A) moved the write from the dead per-day
    ``ledger_<date>.parquet`` to the canonical
    ``ledger_<run_id>/ledger_events.parquet`` layout that the accounting chain
    reads. These tests assert the FILL/REJECT/event_id mapping, which is
    unchanged; only the output location moved.
    """
    return (
        eng.config.ledger_dir / f"ledger_{eng.config.run_id}" / "ledger_events.parquet"
    )


def _read_ledger(path: Path) -> pd.DataFrame:
    assert path.exists(), f"ledger file missing: {path}"
    return pd.read_parquet(path)


def test_ledger_records_fill_qty_for_partial(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    fills = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "side": "BUY",
                "qty": 10.0,
                "fill_qty": 3.0,
                "fill_price": 100.0,
                "status": "partial",
            },
        ]
    )
    eng._write_ledger_events(fills, "2025-01-15")

    df = _read_ledger(_canonical_ledger_path(eng))
    assert len(df) == 1
    assert df.loc[0, "event_type"] == "FILL"
    assert df.loc[0, "qty"] == pytest.approx(3.0)
    assert df.loc[0, "cash_delta"] == pytest.approx(-300.0)


def test_ledger_records_rejected_as_reject_event(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    fills = pd.DataFrame(
        [
            {
                "symbol": "AAA",
                "side": "BUY",
                "qty": 10.0,
                "fill_qty": 0.0,
                "fill_price": 100.0,
                "status": "rejected",
                "reject_reason": "INSUFFICIENT_CASH",
            },
        ]
    )
    eng._write_ledger_events(fills, "2025-01-15")

    df = _read_ledger(_canonical_ledger_path(eng))
    assert len(df) == 1
    assert df.loc[0, "event_type"] == "REJECT"
    assert df.loc[0, "qty"] == 0.0
    assert df.loc[0, "cash_delta"] == 0.0
    assert df.loc[0, "reject_reason"] == "INSUFFICIENT_CASH"


def test_ledger_event_id_distinguishes_buy_and_sell_same_symbol(
    tmp_path: Path,
) -> None:
    """Without side in event_id, multi-leg days would collide."""
    eng = _make_engine(tmp_path)
    fills = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 5.0, "fill_price": 100.0},
            {"symbol": "AAA", "side": "SELL", "qty": 2.0, "fill_price": 102.0},
        ]
    )
    eng._write_ledger_events(fills, "2025-01-15")

    df = _read_ledger(_canonical_ledger_path(eng))
    assert len(df) == 2
    assert df["event_id"].nunique() == 2


def test_ledger_legacy_full_fill_unchanged(tmp_path: Path) -> None:
    """Absent fill_qty/status must default to full FILL matching legacy output."""
    eng = _make_engine(tmp_path)
    fills = pd.DataFrame(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 10.0, "fill_price": 100.0},
        ]
    )
    eng._write_ledger_events(fills, "2025-01-15")

    df = _read_ledger(_canonical_ledger_path(eng))
    assert df.loc[0, "event_type"] == "FILL"
    assert df.loc[0, "qty"] == pytest.approx(10.0)
    assert df.loc[0, "cash_delta"] == pytest.approx(-1000.0)
