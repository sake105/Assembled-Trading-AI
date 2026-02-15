"""Tests for pack manifest files[] schema guard (path, sha256, size_bytes, source_type)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_pack import _validate_manifest_files_schema


def test_validate_manifest_files_schema_ok() -> None:
    """Valid files list passes."""
    _validate_manifest_files_schema(
        [
            {
                "path": "ledger_run/ledger_events.parquet",
                "sha256": "abc123",
                "size_bytes": 100,
                "source_type": "ledger_pack",
            },
        ],
        run_id="r1",
        as_of_date="2025-01-15",
    )


def test_validate_manifest_files_schema_missing_key() -> None:
    """Missing required key raises ValueError (ASCII-only message, run_id + as_of_date)."""
    with pytest.raises(ValueError) as exc_info:
        _validate_manifest_files_schema(
            [
                {
                    "path": "x.json",
                    "sha256": "a",
                    # missing size_bytes
                    "source_type": "evidence_index",
                },
            ],
            run_id="r1",
            as_of_date="2025-01-15",
        )
    msg = exc_info.value.args[0]
    assert "run_id" in msg or "r1" in msg
    assert "as_of_date" in msg or "2025-01-15" in msg
    assert "size_bytes" in msg or "missing" in msg


def test_validate_manifest_files_schema_wrong_type_path() -> None:
    """path must be str."""
    with pytest.raises(ValueError) as exc_info:
        _validate_manifest_files_schema(
            [
                {
                    "path": 123,
                    "sha256": "a",
                    "size_bytes": 0,
                    "source_type": "pack_manifest",
                },
            ],
            run_id="r2",
            as_of_date="2025-01-16",
        )
    assert "path" in exc_info.value.args[0]


def test_validate_manifest_files_schema_wrong_type_size_bytes() -> None:
    """size_bytes must be int."""
    with pytest.raises(ValueError) as exc_info:
        _validate_manifest_files_schema(
            [
                {
                    "path": "x",
                    "sha256": "a",
                    "size_bytes": "not_an_int",
                    "source_type": "ledger_pack",
                },
            ],
            run_id="r3",
            as_of_date="2025-01-17",
        )
    assert "size_bytes" in exc_info.value.args[0]


def test_validate_manifest_files_schema_entry_not_dict() -> None:
    """files[] entry must be a dict."""
    with pytest.raises(ValueError) as exc_info:
        _validate_manifest_files_schema(
            ["not a dict"],  # type: ignore[list-item]
            run_id="r4",
            as_of_date="2025-01-18",
        )
    assert "dict" in exc_info.value.args[0]
