"""END-TO-END coverage for the re-activated paper-engine ledger-events write.

Context
-------
Commit b6109960 re-activated ``_write_ledger_events`` so the engine now calls
``store_ledger_events_parquet(df_events, ledger_dir, run_id, mode="append")`` and
produces the CANONICAL ``<ledger_dir>/ledger_<run_id>/ledger_events.parquet``
artifact (atomic tmp->rename, dedup-by-event_id) that the accounting chain reads.

The sibling unit suite ``test_unified_paper_engine_ledger_write.py`` covers
``_write_ledger_events`` in ISOLATION (it calls the method directly). This file
is the GENUINE END-TO-END test: it drives the REAL ``run_paper_day`` lifecycle
(Step 4/5 order generation -> Step 6 risk controls -> Step 7 REAL
``_simulate_fills_with_cost`` -> Step 8 REAL ``_write_ledger_events`` -> Step 9
REAL ``_update_positions`` -> Step 12 REAL state persist) and then proves the
engine-written artifact round-trips through the accounting readers.

What is REAL vs STUBBED in this drive
-------------------------------------
REAL (exercised, not mocked):
  * ``UnifiedPaperEngine.run_paper_day`` — the actual EOD entry point.
  * Risk-controls path (kill-switch / fat-finger / pre-trade are config-DISABLED,
    not mocked — the real ``_apply_risk_controls`` branches execute and pass the
    orders through, exactly as the existing engine fixtures disable them).
  * ``_simulate_fills_with_cost`` — real spread + Almgren-Chriss impact + cash gate.
  * ``_write_ledger_events`` — real Step-8 ledger write (the path under test).
  * ``store_ledger_events_parquet`` — real canonical parquet store (atomic,
    dedup-by-event_id, append mode).
  * ``_update_positions`` — real average-cost accounting + cash.
  * State persistence to JSON.

STUBBED (the single, minimal, documented injection seam):
  * ``_generate_orders`` is overridden in a thin subclass. The base method is a
    deliberate no-op stub (returns an empty frame) whose own docstring says
    "Subclasses or callers should override this method or inject orders". This
    is the ONLY way to feed orders into the real cycle without standing up the
    full external signal/sizing/data stack — and it is the engine's documented
    extension point, NOT a bypass of the ledger path. Prices are passed directly
    to ``run_paper_day`` (its public ``prices=`` parameter), so no broker and no
    data-layer loader are involved.

The four assertions
-------------------
1. The canonical ``ledger_<run_id>/ledger_events.parquet`` EXISTS after a real
   ``run_paper_day`` that produced fills (written by the engine path, never by a
   direct ``_write_ledger_events`` call).
2. ROUND-TRIP: the artifact is read back by the ACCOUNTING readers
   (``load_ledger_events_parquet``) and fed to the accounting
   ``build_positions_from_ledger`` — FILL events present, REQUIRED_COLUMNS
   present, event_id values match what the engine wrote, and the reconstructed
   positions/cash match the engine's own book (downstream-consumable proof).
3. A multi-day drive (two real EOD cycles, same run_id, append mode) ACCUMULATES
   and DEDUPS correctly through the real path.
4. The cycle still completes and positions/cash are correct — the ledger write
   is additive and did not perturb the engine result.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.accounting.ledger import REQUIRED_COLUMNS
from src.assembled_core.accounting.ledger_store import (
    ledger_base_path,
    load_ledger_events_parquet,
)
from src.assembled_core.accounting.position_engine import (
    build_positions_from_ledger,
)
from src.assembled_core.execution import unified_paper_engine as upe
from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)

RUN_ID = "e2e_ledger_run"


class _InjectingEngine(UnifiedPaperEngine):
    """Real engine with orders injected via the documented ``_generate_orders`` seam.

    Only ``_generate_orders`` is overridden — every other lifecycle step
    (risk controls, fill simulation, ledger write, position update, state
    persist) is the production implementation. Orders are keyed by ISO date so a
    multi-day drive can vary the order set per cycle.
    """

    def __init__(
        self, config: UnifiedPaperConfig, orders_by_date: dict[str, pd.DataFrame]
    ) -> None:
        super().__init__(config)
        self._orders_by_date = orders_by_date

    def _generate_orders(
        self, as_of_date: str, prices: pd.DataFrame
    ) -> pd.DataFrame | None:
        return self._orders_by_date.get(as_of_date, pd.DataFrame())


def _make_engine(
    tmp_path: Path,
    orders_by_date: dict[str, pd.DataFrame],
    *,
    seed: float = 1_000_000.0,
    enable_reconciliation: bool = False,
) -> _InjectingEngine:
    """Build a real engine wired to a tmp ledger dir.

    Risk gates that depend on global/process state (kill-switch file,
    fat-finger history) are config-disabled so the drive is deterministic and
    environment-independent — exactly as the existing engine test fixtures do.
    The disabled gates are NOT mocked: the real ``_apply_risk_controls`` still
    runs and passes the orders through.
    """
    cfg = UnifiedPaperConfig(
        seed_capital=seed,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=enable_reconciliation,
        enable_kill_switch=False,
        enable_fat_finger=False,
        enable_borrow_costs=False,
        enable_corporate_actions=False,
        enable_tca=False,
        enable_attribution=False,
        enable_manifest=False,
        run_id=RUN_ID,
    )
    return _InjectingEngine(cfg, orders_by_date)


def _canonical_path(eng: UnifiedPaperEngine) -> Path:
    return (
        ledger_base_path(eng.config.ledger_dir, eng.config.run_id)
        / "ledger_events.parquet"
    )


def _orders(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _prices(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Assertion 1 + 2 — real cycle writes the canonical artifact; it round-trips.
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_real_run_paper_day_writes_canonical_ledger_and_roundtrips(
    tmp_path: Path,
) -> None:
    """Drive the REAL ``run_paper_day``; assert the canonical artifact exists and
    round-trips through the accounting readers.
    """
    day = "2025-03-03"
    orders = _orders(
        [
            {"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 50.0},
            {"symbol": "BBB", "side": "BUY", "qty": 40.0, "price": 100.0},
        ]
    )
    # Large ADV (volume) keeps impact small and avoids partial-fill paths;
    # the default engine config does full fills anyway.
    prices = _prices(
        [
            {"symbol": "AAA", "close": 50.0, "volume": 5_000_000.0},
            {"symbol": "BBB", "close": 100.0, "volume": 5_000_000.0},
        ]
    )

    eng = _make_engine(tmp_path, {day: orders})

    # No direct _write_ledger_events call anywhere — this is the real EOD path.
    result = eng.run_paper_day(day, prices=prices)

    # The real cycle generated fills and completed.
    assert result.status == "success"
    assert result.n_orders == 2
    assert result.n_fills == 2

    # --- Assertion 1: canonical artifact exists, written by the engine path ---
    path = _canonical_path(eng)
    assert path.exists(), f"engine did not write canonical ledger artifact: {path}"
    assert path.parent.name == f"ledger_{RUN_ID}"
    assert path.name == "ledger_events.parquet"
    # The engine cached the path it actually wrote (proves Step 8 ran).
    assert eng._last_ledger_path == path

    # --- Assertion 2: round-trip via the accounting reader -------------------
    # Read with the SAME reader the accounting chain uses (not a bare read_parquet).
    events = load_ledger_events_parquet(eng.config.ledger_dir, eng.config.run_id)
    assert len(events) == 2
    for col in REQUIRED_COLUMNS:
        assert col in events.columns, f"missing required column: {col}"
    assert set(events["event_type"]) == {"FILL"}
    assert set(events["symbol"]) == {"AAA", "BBB"}

    # event_id values are exactly what the engine emits (run/day/symbol/side/type).
    expected_event_ids = {
        f"{RUN_ID}_{day}_AAA_BUY_FILL",
        f"{RUN_ID}_{day}_BBB_BUY_FILL",
    }
    assert set(events["event_id"]) == expected_event_ids
    assert events["event_id"].nunique() == 2

    # Signed qty / cash_delta agree with BUY semantics (qty +, cash_delta -).
    aaa = events[events["symbol"] == "AAA"].iloc[0]
    assert aaa["qty"] == pytest.approx(100.0)
    assert aaa["cash_delta"] < 0.0  # BUY debits cash

    # Downstream-consumability: feed the engine-written canonical events into the
    # accounting position engine and confirm it reconstructs the SAME book the
    # engine holds. This proves the canonical layout is actually consumed by the
    # accounting chain, not just file-shaped.
    rebuilt = build_positions_from_ledger(
        events,
        start_cash=eng.config.seed_capital,
        missing_price_policy="zero",
    )
    rebuilt_pos = {
        str(r["symbol"]): float(r["qty"]) for _, r in rebuilt["positions_df"].iterrows()
    }
    assert rebuilt_pos.get("AAA") == pytest.approx(100.0)
    assert rebuilt_pos.get("BBB") == pytest.approx(40.0)
    # Cash reconstructed from the canonical FILL cash_delta == engine's own cash.
    assert rebuilt["cash_balance"] == pytest.approx(float(eng._state["cash"]))


# ---------------------------------------------------------------------------
# Assertion 4 — the ledger write is additive: engine result is unperturbed.
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_real_cycle_positions_and_cash_unperturbed_by_ledger_write(
    tmp_path: Path,
) -> None:
    """The engine's positions/cash after the real cycle must equal the fill math,
    independent of the additive ledger write.
    """
    day = "2025-03-03"
    orders = _orders([{"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 50.0}])
    prices = _prices([{"symbol": "AAA", "close": 50.0, "volume": 5_000_000.0}])

    eng = _make_engine(tmp_path, {day: orders}, seed=1_000_000.0)
    result = eng.run_paper_day(day, prices=prices)

    assert result.status == "success"
    assert result.n_fills == 1

    # Position is the full fill (default config = full fill).
    assert eng._state["positions"]["AAA"] == pytest.approx(100.0)

    # Cash = seed - notional. notional = 100 * fill_price; fill_price >= mid for a
    # BUY (spread + impact push it up), so cash must be strictly below seed minus
    # the mid-notional, and the position result must be internally consistent with
    # the ledger cash_delta the engine wrote.
    seed = 1_000_000.0
    mid_notional = 100.0 * 50.0
    assert eng._state["cash"] < seed - mid_notional + 1e-6
    assert eng._state["cash"] > seed - mid_notional * 1.05  # sane cost bound

    # The cash debit recorded in the canonical ledger equals the engine's cash move.
    events = load_ledger_events_parquet(eng.config.ledger_dir, eng.config.run_id)
    ledger_cash_delta = float(events["cash_delta"].sum())
    assert (seed + ledger_cash_delta) == pytest.approx(float(eng._state["cash"]))


# ---------------------------------------------------------------------------
# Assertion 3 — multi-day drive accumulates + dedups through the real path.
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_two_real_eod_cycles_same_run_accumulate_and_dedup(tmp_path: Path) -> None:
    """Two real ``run_paper_day`` cycles on the same engine/run_id must accumulate
    distinct days and dedup a re-run of an identical day (append mode).
    """
    day1 = "2025-03-03"
    day2 = "2025-03-04"
    orders1 = _orders([{"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 50.0}])
    orders2 = _orders([{"symbol": "BBB", "side": "BUY", "qty": 30.0, "price": 100.0}])
    prices = _prices(
        [
            {"symbol": "AAA", "close": 50.0, "volume": 5_000_000.0},
            {"symbol": "BBB", "close": 100.0, "volume": 5_000_000.0},
        ]
    )

    eng = _make_engine(tmp_path, {day1: orders1, day2: orders2}, seed=1_000_000.0)

    # Day 1 — real cycle.
    r1 = eng.run_paper_day(day1, prices=prices)
    assert r1.n_fills == 1
    events_after_d1 = load_ledger_events_parquet(
        eng.config.ledger_dir, eng.config.run_id
    )
    assert len(events_after_d1) == 1

    # Day 2 — real cycle, SAME run_id; append accumulates.
    r2 = eng.run_paper_day(day2, prices=prices)
    assert r2.n_fills == 1
    events_after_d2 = load_ledger_events_parquet(
        eng.config.ledger_dir, eng.config.run_id
    )
    assert len(events_after_d2) == 2
    assert set(events_after_d2["event_id"]) == {
        f"{RUN_ID}_{day1}_AAA_BUY_FILL",
        f"{RUN_ID}_{day2}_BBB_BUY_FILL",
    }

    # Re-drive day 1 through the REAL path again: same deterministic event_id →
    # append-mode dedup keeps the row count at 2 (idempotent re-run).
    r1_again = eng.run_paper_day(day1, prices=prices)
    assert r1_again.n_fills == 1
    events_after_rerun = load_ledger_events_parquet(
        eng.config.ledger_dir, eng.config.run_id
    )
    assert len(events_after_rerun) == 2
    assert events_after_rerun["event_id"].nunique() == 2

    # Accumulated canonical events still reconstruct a consistent book downstream.
    rebuilt = build_positions_from_ledger(
        events_after_rerun,
        start_cash=eng.config.seed_capital,
        missing_price_policy="zero",
    )
    rebuilt_pos = {
        str(r["symbol"]): float(r["qty"]) for _, r in rebuilt["positions_df"].iterrows()
    }
    assert rebuilt_pos.get("AAA") == pytest.approx(100.0)
    assert rebuilt_pos.get("BBB") == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Bonus — reconcile observability flag reads the REAL engine-written artifact.
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_reconcile_ledger_exists_true_through_real_cycle(tmp_path: Path) -> None:
    """With reconciliation enabled, the real cycle's reconcile step must see the
    engine-written canonical artifact (observability flag only, no verdict gate).
    """
    day = "2025-03-03"
    orders = _orders([{"symbol": "AAA", "side": "BUY", "qty": 100.0, "price": 50.0}])
    prices = _prices([{"symbol": "AAA", "close": 50.0, "volume": 5_000_000.0}])

    eng = _make_engine(
        tmp_path, {day: orders}, seed=1_000_000.0, enable_reconciliation=True
    )

    result = eng.run_paper_day(day, prices=prices)
    assert result.n_fills == 1

    # The artifact the engine wrote is what reconcile inspects.
    assert _canonical_path(eng).exists()
    verdict = eng._run_reconciliation(day)
    assert verdict is not None
    assert verdict["reconcile"]["ledger_exists"] is True


# ---------------------------------------------------------------------------
# Re-activation contract guard (kept here so the e2e suite is self-contained).
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_has_ledger_flag_is_true() -> None:
    """If this flips False the engine silently skips Step 8 — the whole e2e is moot."""
    assert upe._HAS_LEDGER is True
