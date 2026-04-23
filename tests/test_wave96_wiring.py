"""Tests for wave-96 module wiring into trading_cycle.py.

Covers:
  Step 7.82 — accounting.broker_snapshot_importer (import_broker_snapshot)
  Step 7.83 — accounting.broker_snapshot_store (store_broker_snapshot_json)
  Step 7.84 — accounting.evidence_index (write_evidence_index_json)
"""

from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.assembled_core.accounting.broker_snapshot_importer import (
    import_broker_snapshot,
    load_external_broker_snapshot,
)
from src.assembled_core.accounting.broker_snapshot_store import (
    broker_snapshot_base_path,
    store_broker_snapshot_json,
)
from src.assembled_core.accounting.evidence_index import write_evidence_index_json


# ---------------------------------------------------------------------------
# broker_snapshot_importer (Step 7.82)
# ---------------------------------------------------------------------------

def test_import_broker_snapshot_importable():
    assert import_broker_snapshot is not None


def test_load_external_broker_snapshot_importable():
    assert load_external_broker_snapshot is not None


def test_load_external_broker_snapshot_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_external_broker_snapshot(tmp_path / "missing.json")


# ---------------------------------------------------------------------------
# broker_snapshot_store (Step 7.83)
# ---------------------------------------------------------------------------

def test_broker_snapshot_base_path_returns_path(tmp_path):
    path = broker_snapshot_base_path(tmp_path, "run_001")
    assert isinstance(path, Path)


def test_broker_snapshot_base_path_contains_run_id(tmp_path):
    path = broker_snapshot_base_path(tmp_path, "run_001")
    assert "run_001" in str(path)


def test_store_broker_snapshot_importable():
    assert store_broker_snapshot_json is not None


# ---------------------------------------------------------------------------
# evidence_index (Step 7.84)
# ---------------------------------------------------------------------------

def test_write_evidence_index_json_creates_file(tmp_path):
    path = write_evidence_index_json(
        output_dir=tmp_path,
        run_id="test_run",
        as_of_date="2024-06-01",
        paths={"broker_snapshot_path": None, "ledger_pack_path": None},
    )
    assert path.exists()


def test_write_evidence_index_json_valid_json(tmp_path):
    path = write_evidence_index_json(
        output_dir=tmp_path,
        run_id="test_run",
        as_of_date="2024-06-01",
        paths={"broker_snapshot_path": None},
    )
    data = json.loads(path.read_text())
    assert "run_id" in data
    assert data["run_id"] == "test_run"
