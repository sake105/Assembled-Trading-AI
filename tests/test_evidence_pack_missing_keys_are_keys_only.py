"""Evidence pack: required_missing and optional_missing must only contain allowed keys (no paths)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.assembled_core.accounting.evidence_pack import (
    build_evidence_pack,
    collect_evidence_files,
)


def test_evidence_pack_optional_missing_invalid_path_raises(tmp_path: Path) -> None:
    """Injecting missing_optional with a path (e.g. C:\\abs\\path) raises ValueError with ASCII message."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "keys_only"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"

    # Real collection would have files; we need to get past "no files" and "missing required" checks.
    # So we need a collection that has files, no missing_required, but missing_optional with invalid entry.
    ledger_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("dummy", encoding="utf-8")

    evidence_dir = output_dir / f"evidence_{run_id}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_index_path = evidence_dir / f"evidence_{date_str}.json"
    evidence_index_path.write_text(
        '{"paths":{"ledger_pack_path":"ledger_run/ledger_events.parquet"}}',
        encoding="utf-8",
    )

    def fake_collect(*args: object, **kwargs: object) -> dict:
        real = collect_evidence_files(*args, **kwargs)
        # Inject invalid entry: path instead of key
        real["missing_optional"] = ["C:\\abs\\path"]
        return real

    with patch(
        "src.assembled_core.accounting.evidence_pack.collect_evidence_files",
        side_effect=fake_collect,
    ):
        with pytest.raises(ValueError) as exc_info:
            build_evidence_pack(
                output_dir=output_dir,
                run_id=run_id,
                as_of_date=as_of_date,
                include_optional=True,
                strict=False,
            )
    msg = str(exc_info.value)
    assert (
        msg.encode("ascii", errors="ignore").decode("ascii") == msg
    ), "ValueError must be ASCII-only"
    assert "optional_missing" in msg or "allowed keys" in msg or "invalid" in msg


def test_evidence_pack_required_missing_invalid_key_raises(tmp_path: Path) -> None:
    """Injecting required_missing with non-allowed key raises ValueError."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = "req_keys"
    as_of_date = "2025-01-15"
    date_str = "2025-01-15"
    ledger_path = output_dir / "ledger_run" / "ledger_events.parquet"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text("dummy", encoding="utf-8")
    evidence_dir = output_dir / f"evidence_{run_id}"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_index_path = evidence_dir / f"evidence_{date_str}.json"
    evidence_index_path.write_text(
        '{"paths":{"ledger_pack_path":"ledger_run/ledger_events.parquet"}}',
        encoding="utf-8",
    )

    def fake_collect(*args: object, **kwargs: object) -> dict:
        real = collect_evidence_files(*args, **kwargs)
        real["missing_required"] = ["some_invalid_key"]
        return real

    with patch(
        "src.assembled_core.accounting.evidence_pack.collect_evidence_files",
        side_effect=fake_collect,
    ):
        with pytest.raises(ValueError) as exc_info:
            build_evidence_pack(
                output_dir=output_dir,
                run_id=run_id,
                as_of_date=as_of_date,
                include_optional=True,
                strict=False,
            )
    msg = str(exc_info.value)
    assert msg.encode("ascii", errors="ignore").decode("ascii") == msg
    assert "required_missing" in msg or "allowed keys" in msg or "invalid" in msg
