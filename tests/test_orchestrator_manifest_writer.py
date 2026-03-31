"""Tests for orchestrator manifest writer determinism and portable paths."""

from __future__ import annotations

from pathlib import Path

import json

from src.assembled_core.pipeline.orchestrator import (
    _manifest_path_str,
    _write_manifest_json,
)


def test_write_manifest_twice_byte_identical(tmp_path: Path) -> None:
    """Writing the same manifest twice should be byte-identical."""
    base = tmp_path
    manifest_path = base / "run_manifest_1d.json"

    # Create a deterministic manifest (no runtime timestamps in this unit test).
    manifest = {
        "freq": "1d",
        "start_capital": 10000.0,
        "ledger_pack_path": _manifest_path_str(
            base / "ledger_run" / "ledger_events.parquet", base_dir=base
        ),
        "reconcile_report_path": _manifest_path_str(
            base / "reconcile" / "reconcile_2025-01-01.json", base_dir=base
        ),
        "reconciliation_ok": True,
        "qa_report_path": None,
        "timestamps": {
            "started": "2025-01-01T00:00:00",
            "finished": "2025-01-01T00:00:01",
        },
        "failure": False,
    }

    _write_manifest_json(manifest_path, manifest)
    b1 = manifest_path.read_bytes()

    _write_manifest_json(manifest_path, manifest)
    b2 = manifest_path.read_bytes()

    assert b1 == b2


def test_manifest_paths_are_relative_and_posix(tmp_path: Path) -> None:
    """Manifest paths should be relative (to output dir) and use forward slashes."""
    base = tmp_path
    ledger_dir = base / "ledger_run"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_file = ledger_dir / "ledger_events.parquet"
    ledger_file.write_bytes(b"dummy")

    rel = _manifest_path_str(ledger_file, base_dir=base)
    assert rel == "ledger_run/ledger_events.parquet"
    assert "\\" not in rel

    manifest_path = base / "run_manifest_1d.json"
    manifest = {
        "freq": "1d",
        "ledger_pack_path": rel,
    }
    _write_manifest_json(manifest_path, manifest)

    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert loaded["ledger_pack_path"] == "ledger_run/ledger_events.parquet"
