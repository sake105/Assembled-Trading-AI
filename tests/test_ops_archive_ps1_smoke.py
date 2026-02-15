"""Smoke test: ops_archive_pack.ps1 exists and defines the expected function (no PowerShell execution)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS_ARCHIVE_PS1 = ROOT / "scripts" / "dev" / "ops_archive_pack.ps1"
FUNCTION_NAME = "New-OpsEvidenceArchive"


def test_ops_archive_ps1_file_exists() -> None:
    """scripts/dev/ops_archive_pack.ps1 exists."""
    assert OPS_ARCHIVE_PS1.exists()
    assert OPS_ARCHIVE_PS1.is_file()


def test_ops_archive_ps1_contains_function_name() -> None:
    """Script contains the function name New-OpsEvidenceArchive."""
    content = OPS_ARCHIVE_PS1.read_text(encoding="utf-8")
    assert FUNCTION_NAME in content


def test_ops_archive_ps1_contains_exit_codes_and_convertfrom_json() -> None:
    """Script uses exit 2, 3, 4 and ConvertFrom-Json for robust JSON handling."""
    content = OPS_ARCHIVE_PS1.read_text(encoding="utf-8")
    assert "exit 2" in content
    assert "exit 3" in content
    assert "exit 4" in content
    assert "ConvertFrom-Json" in content
