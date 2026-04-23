"""Tests for wave-97 module wiring into trading_cycle.py.

Covers:
  Step 7.85 — accounting.evidence_pack (_sha256_bytes / _sha256_file)
  Step 7.86 — accounting.ledger_integration (build_ledger_from_trades)
  Step 7.87 — accounting.ledger_store (ledger_base_path / store_ledger_events_parquet)
"""

from __future__ import annotations

import pytest
import pandas as pd
from pathlib import Path

from src.assembled_core.accounting.evidence_pack import _sha256_bytes, _sha256_file
from src.assembled_core.accounting.ledger_integration import build_ledger_from_trades
from src.assembled_core.accounting.ledger_store import (
    ledger_base_path,
    list_ledger_runs,
    store_ledger_events_parquet,
)


# ---------------------------------------------------------------------------
# evidence_pack (Step 7.85)
# ---------------------------------------------------------------------------

def test_sha256_bytes_returns_hex():
    result = _sha256_bytes(b"hello")
    assert isinstance(result, str)
    assert len(result) == 64  # SHA-256 hex is 64 chars


def test_sha256_bytes_deterministic():
    r1 = _sha256_bytes(b"test data")
    r2 = _sha256_bytes(b"test data")
    assert r1 == r2


def test_sha256_bytes_different_inputs():
    r1 = _sha256_bytes(b"aaa")
    r2 = _sha256_bytes(b"bbb")
    assert r1 != r2


def test_sha256_file_importable():
    assert _sha256_file is not None


# ---------------------------------------------------------------------------
# ledger_integration (Step 7.86)
# ---------------------------------------------------------------------------

def test_build_ledger_from_trades_importable():
    assert build_ledger_from_trades is not None


def test_build_ledger_from_trades_signature():
    import inspect
    sig = inspect.signature(build_ledger_from_trades)
    params = list(sig.parameters)
    assert "orders_df" in params
    assert "trades_df" in params
    assert "run_id" in params


# ---------------------------------------------------------------------------
# accounting.ledger_store (Step 7.87)
# ---------------------------------------------------------------------------

def test_ledger_base_path_returns_path(tmp_path):
    path = ledger_base_path(tmp_path, "run_001")
    assert isinstance(path, Path)


def test_ledger_base_path_contains_run_id(tmp_path):
    path = ledger_base_path(tmp_path, "run_test")
    assert "run_test" in str(path)


def test_list_ledger_runs_empty_dir(tmp_path):
    runs = list_ledger_runs(tmp_path)
    assert isinstance(runs, list)
    assert len(runs) == 0


def test_store_ledger_events_parquet_importable():
    assert store_ledger_events_parquet is not None
