"""Phase 1.5 regression tests for the determinism backbone.

Covers:

* ``derive_seed`` is stable across instances for the same ``run_id`` + date
* ``make_rng`` produces reproducible byte sequences with the same inputs
* different base seeds produce different streams
* ``RunSnapshot`` round-trips through the filesystem preserving seed, prices,
  signals, and context
* ``UnifiedPaperEngine._rng`` returns a Generator seeded from its config
* enabling ``replay_snapshot_dir`` writes manifest + prices.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)
from src.assembled_core.ops.replay_snapshot import (
    RunSnapshot,
    derive_seed,
    make_rng,
)


# --- derive_seed / make_rng --------------------------------------------------


def test_derive_seed_is_stable_across_calls() -> None:
    a = derive_seed("paper", "2025-01-15", None)
    b = derive_seed("paper", "2025-01-15", None)
    assert a == b
    assert 0 <= a < (1 << 63)


def test_derive_seed_changes_with_base_seed() -> None:
    a = derive_seed("paper", "2025-01-15", None)
    b = derive_seed("paper", "2025-01-15", 42)
    c = derive_seed("paper", "2025-01-15", 43)
    assert a != b != c
    assert a != c


def test_derive_seed_changes_with_date_and_run_id() -> None:
    a = derive_seed("paper", "2025-01-15", 42)
    b = derive_seed("paper", "2025-01-16", 42)
    c = derive_seed("other", "2025-01-15", 42)
    assert a != b
    assert a != c


def test_make_rng_reproducible() -> None:
    rng1 = make_rng("paper", "2025-01-15", 42)
    rng2 = make_rng("paper", "2025-01-15", 42)
    x1 = rng1.standard_normal(10)
    x2 = rng2.standard_normal(10)
    np.testing.assert_array_equal(x1, x2)


def test_make_rng_differs_for_different_seeds() -> None:
    rng1 = make_rng("paper", "2025-01-15", 1)
    rng2 = make_rng("paper", "2025-01-15", 2)
    x1 = rng1.standard_normal(10)
    x2 = rng2.standard_normal(10)
    assert not np.array_equal(x1, x2)


# --- RunSnapshot round-trip --------------------------------------------------


def test_run_snapshot_round_trip_preserves_fields(tmp_path: Path) -> None:
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 1000.0}]
    )
    signals = pd.DataFrame([{"symbol": "AAA", "signal": 1.0}])
    snap = RunSnapshot(
        run_id="paper",
        as_of_date="2025-01-15",
        seed=42,
        prices=prices,
        signals=signals,
        context={"regime": "normal"},
    )
    path = snap.save(tmp_path)

    loaded = RunSnapshot.load(path)
    assert loaded.run_id == "paper"
    assert loaded.as_of_date == "2025-01-15"
    assert loaded.seed == 42
    assert loaded.context == {"regime": "normal"}
    pd.testing.assert_frame_equal(
        loaded.prices.reset_index(drop=True), prices.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        loaded.signals.reset_index(drop=True), signals.reset_index(drop=True)
    )


def test_run_snapshot_without_signals(tmp_path: Path) -> None:
    prices = pd.DataFrame([{"symbol": "AAA", "close": 100.0}])
    snap = RunSnapshot(
        run_id="paper",
        as_of_date="2025-01-15",
        seed=None,
        prices=prices,
    )
    path = snap.save(tmp_path)
    loaded = RunSnapshot.load(path)
    assert loaded.signals is None
    assert loaded.seed is None


def test_run_snapshot_rng_matches_make_rng(tmp_path: Path) -> None:
    snap = RunSnapshot(
        run_id="paper",
        as_of_date="2025-01-15",
        seed=7,
        prices=pd.DataFrame([{"symbol": "AAA", "close": 100.0}]),
    )
    a = snap.rng().standard_normal(5)
    b = make_rng("paper", "2025-01-15", 7).standard_normal(5)
    np.testing.assert_array_equal(a, b)


# --- Engine integration ------------------------------------------------------


def test_engine_rng_is_seeded_from_run_id_and_date(tmp_path: Path) -> None:
    cfg = UnifiedPaperConfig(
        seed_capital=100_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        run_id="paper",
        random_seed=42,
    )
    eng = UnifiedPaperEngine(cfg)
    rng_a = eng._rng("2025-01-15")
    rng_b = eng._rng("2025-01-15")
    # Two calls return independent Generator instances but the same seed.
    assert rng_a is not rng_b
    np.testing.assert_array_equal(
        rng_a.standard_normal(4),
        rng_b.standard_normal(4),
    )

    # Different day → different stream.
    rng_c = eng._rng("2025-01-16")
    assert not np.array_equal(
        eng._rng("2025-01-15").standard_normal(4),
        rng_c.standard_normal(4),
    )


def test_engine_writes_replay_snapshot_when_dir_set(tmp_path: Path) -> None:
    snap_dir = tmp_path / "snapshots"
    cfg = UnifiedPaperConfig(
        seed_capital=100_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        run_id="paper",
        random_seed=42,
        replay_snapshot_dir=snap_dir,
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 100_000.0, "positions": {}, "cost_basis": {}}

    prices = pd.DataFrame([{"symbol": "AAA", "close": 100.0, "volume": 1000.0}])
    eng._maybe_save_replay_snapshot("2025-01-15", prices, context={"regime": "normal"})

    path = snap_dir / "paper" / "2025-01-15"
    assert (path / "manifest.json").exists()
    assert (path / "prices.parquet").exists()

    loaded = RunSnapshot.load(path)
    assert loaded.seed == 42
    assert loaded.context == {"regime": "normal"}


def test_engine_replay_snapshot_noop_when_dir_none(tmp_path: Path) -> None:
    cfg = UnifiedPaperConfig(
        seed_capital=100_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        run_id="paper",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 100_000.0, "positions": {}, "cost_basis": {}}
    # No dir set → should be a silent noop.
    eng._maybe_save_replay_snapshot(
        "2025-01-15",
        pd.DataFrame([{"symbol": "AAA", "close": 100.0}]),
    )
    # Nothing to assert except: no crash and nothing in tmp_path beyond the
    # created state/ledger/lifecycle dirs.
    assert not (tmp_path / "snapshots").exists()
