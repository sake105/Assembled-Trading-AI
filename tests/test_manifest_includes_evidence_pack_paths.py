"""Tests that orchestrator manifest includes evidence pack paths deterministically.

This test exercises the low-level manifest helpers without running the full pipeline.
"""

from __future__ import annotations

from pathlib import Path

import json

from src.assembled_core.pipeline.orchestrator import (
    _manifest_path_str,
    _write_manifest_json,
)


def test_manifest_includes_evidence_pack_paths_and_is_deterministic(
    tmp_path: Path,
) -> None:
    """Manifest includes evidence_* fields with POSIX-relative paths and is byte-deterministic."""
    base = tmp_path

    # Create dummy evidence files under base/output-style structure
    evidence_index_file = base / "evidence_run_1d" / "evidence_2025-01-15.json"
    evidence_pack_file = base / "evidence_run_1d" / "pack_2025-01-15.zip"
    evidence_pack_manifest_file = (
        base / "evidence_run_1d" / "pack_manifest_2025-01-15.json"
    )

    for p in [evidence_index_file, evidence_pack_file, evidence_pack_manifest_file]:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy", encoding="utf-8")

    # Build manifest dictionary using _manifest_path_str for all evidence fields
    manifest_path = base / "run_manifest_1d.json"
    manifest = {
        "schema_version": 1,
        "freq": "1d",
        "start_capital": 10000.0,
        "evidence_index_path": _manifest_path_str(evidence_index_file, base_dir=base),
        "evidence_pack_path": _manifest_path_str(evidence_pack_file, base_dir=base),
        "evidence_pack_manifest_path": _manifest_path_str(
            evidence_pack_manifest_file,
            base_dir=base,
        ),
        # Fixed timestamps to keep this unit test deterministic
        "timestamps": {
            "started": "2025-01-15T00:00:00",
            "finished": "2025-01-15T00:00:01",
        },
        "failure": False,
    }

    # Write manifest twice and ensure bytes are identical
    _write_manifest_json(manifest_path, manifest)
    bytes_1 = manifest_path.read_bytes()

    _write_manifest_json(manifest_path, manifest)
    bytes_2 = manifest_path.read_bytes()

    assert bytes_1 == bytes_2, "Manifest bytes should be identical for same input"

    # Load manifest and verify evidence_* fields and POSIX-relative paths
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))

    for key in (
        "evidence_index_path",
        "evidence_pack_path",
        "evidence_pack_manifest_path",
    ):
        assert key in loaded
        value = loaded[key]
        # Paths should be relative to base and use POSIX slashes
        assert isinstance(value, str)
        assert "\\" not in value, (
            f"Manifest path should not contain backslashes: {value}"
        )
        assert not Path(value).is_absolute(), (
            f"Manifest path should be relative: {value}"
        )
