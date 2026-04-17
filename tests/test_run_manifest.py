"""Phase 8 tests for RunManifest and cross-run index.

Covers:

* ``write_run_manifest`` creates per-day JSON and ``manifest.latest.json``
* Only existing artifact paths are listed in ``artifacts``
* ``compute_config_hash`` is deterministic and stable across calls
* ``append_run_index`` appends + replaces + sorts deterministically
* Two runs produce two rows; same (run_id, date) replaces in place
* Engine integration: runs write a manifest and add an index row
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from src.assembled_core.execution.unified_paper_engine import (
    UnifiedPaperConfig,
    UnifiedPaperEngine,
)
from src.assembled_core.ops.run_index import INDEX_COLUMNS, append_run_index
from src.assembled_core.ops.run_manifest import (
    compute_config_hash,
    write_run_manifest,
)


# --- run_manifest ------------------------------------------------------------


def test_write_manifest_creates_latest_pointer(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.parquet"
    ledger.write_bytes(b"x")  # exists
    missing = tmp_path / "does_not_exist.json"

    out = write_run_manifest(
        run_id="r1",
        date="2025-01-15",
        started_at_utc="2025-01-15T12:00:00+00:00",
        status="success",
        config={"a": 1, "b": "two"},
        artifacts={"ledger": ledger, "ghost": missing},
        metrics={"total_return": 0.01},
        manifests_dir=tmp_path / "manifests",
    )
    assert out.exists()
    latest = out.parent / "manifest.latest.json"
    assert latest.exists()
    payload = json.loads(out.read_text())
    assert payload["run_id"] == "r1"
    assert payload["date"] == "2025-01-15"
    assert "ledger" in payload["artifacts"]
    assert "ghost" not in payload["artifacts"]  # missing artifact dropped
    assert payload["metrics"]["total_return"] == 0.01


def test_compute_config_hash_deterministic() -> None:
    cfg = {"a": 1, "b": "two", "c": [1, 2, 3]}
    h1 = compute_config_hash(cfg)
    h2 = compute_config_hash(cfg)
    assert h1 == h2
    assert len(h1) == 16
    # Changing config changes the hash
    cfg2 = {"a": 2, "b": "two", "c": [1, 2, 3]}
    assert compute_config_hash(cfg2) != h1


# --- run_index --------------------------------------------------------------


def test_append_run_index_appends_and_sorts(tmp_path: Path) -> None:
    idx = tmp_path / "index.csv"
    append_run_index(
        run_id="r1", date="2025-01-16", status="success",
        metrics={"final_equity": 1_010_000.0, "total_return": 0.01, "n_fills": 3,
                 "avg_cost_bps": 5.0},
        git_sha="abc", config_hash="c1",
        manifest_path=tmp_path / "m1.json", index_path=idx,
    )
    append_run_index(
        run_id="r1", date="2025-01-15", status="success",
        metrics={"final_equity": 1_000_000.0, "total_return": 0.0, "n_fills": 0,
                 "avg_cost_bps": 0.0},
        git_sha="abc", config_hash="c1",
        manifest_path=tmp_path / "m2.json", index_path=idx,
    )
    with open(idx, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["date"] for r in rows] == ["2025-01-15", "2025-01-16"]
    assert set(INDEX_COLUMNS).issubset(rows[0].keys())


def test_append_run_index_replaces_same_run_date(tmp_path: Path) -> None:
    idx = tmp_path / "index.csv"
    append_run_index(
        run_id="r1", date="2025-01-15", status="success",
        metrics={"final_equity": 1_000_000.0}, git_sha="", config_hash="",
        manifest_path=tmp_path / "m.json", index_path=idx,
    )
    append_run_index(
        run_id="r1", date="2025-01-15", status="error",
        metrics={"final_equity": 999_000.0}, git_sha="", config_hash="",
        manifest_path=tmp_path / "m.json", index_path=idx,
    )
    with open(idx, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["final_equity"] == "999000.0"


# --- engine integration ------------------------------------------------------


def _make_engine(tmp_path: Path) -> UnifiedPaperEngine:
    cfg = UnifiedPaperConfig(
        seed_capital=1_000_000.0,
        state_dir=tmp_path / "state",
        ledger_dir=tmp_path / "ledger",
        lifecycle_dir=tmp_path / "lifecycle",
        tca_dir=tmp_path / "tca",
        manifests_dir=tmp_path / "manifests",
        run_index_path=tmp_path / "manifests" / "index.csv",
        enable_reconciliation=False,
        enable_kill_switch=False,
        enable_fat_finger=False,
        run_id="manifest_test",
    )
    eng = UnifiedPaperEngine(cfg)
    eng._state = {"cash": 1_000_000.0, "positions": {}, "cost_basis": {}}
    return eng


def test_engine_writes_manifest_and_index(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    eng._write_manifest_and_index(
        as_of_date="2025-01-15",
        run_started_utc="2025-01-15T12:00:00+00:00",
        status="success",
        equity_after=1_000_000.0,
        total_return=0.0,
        n_fills=0,
        total_cost_bps=0.0,
    )
    manifest = tmp_path / "manifests" / "manifest_test" / "manifest_2025-01-15.json"
    assert manifest.exists()
    latest = tmp_path / "manifests" / "manifest_test" / "manifest.latest.json"
    assert latest.exists()
    idx = tmp_path / "manifests" / "index.csv"
    assert idx.exists()
    with open(idx, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    assert rows[0]["run_id"] == "manifest_test"
    assert rows[0]["status"] == "success"


def test_engine_run_paper_day_writes_manifest(tmp_path: Path) -> None:
    eng = _make_engine(tmp_path)
    # Minimal, orderless run_paper_day: default _generate_orders returns empty.
    prices = pd.DataFrame(
        [{"symbol": "AAA", "close": 100.0, "volume": 1_000_000.0}]
    )
    eng.run_paper_day("2025-01-15", prices=prices)
    manifest = tmp_path / "manifests" / "manifest_test" / "manifest_2025-01-15.json"
    assert manifest.exists()
    payload = json.loads(manifest.read_text())
    assert payload["run_id"] == "manifest_test"
    assert payload["date"] == "2025-01-15"
    assert payload["status"] == "success"
    assert payload["phase_versions"]["paper_engine"] == "phase8"
