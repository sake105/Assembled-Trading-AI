"""Contract smoke test for ops_archive_pack.ps1: file exists, function signature, params, script calls (no PowerShell execution)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS_ARCHIVE_PS1 = ROOT / "scripts" / "dev" / "ops_archive_pack.ps1"


def test_ops_archive_ps1_file_exists() -> None:
    """scripts/dev/ops_archive_pack.ps1 exists."""
    assert OPS_ARCHIVE_PS1.exists()
    assert OPS_ARCHIVE_PS1.is_file()


def test_ops_archive_ps1_contains_new_ops_evidence_archive() -> None:
    """Script contains function signature New-OpsEvidenceArchive."""
    content = OPS_ARCHIVE_PS1.read_text(encoding="utf-8")
    assert "New-OpsEvidenceArchive" in content


def test_ops_archive_ps1_contains_parameter_names() -> None:
    """Script contains parameter names RunId, AsOfDate, OutputDir, ArchiveDir."""
    content = OPS_ARCHIVE_PS1.read_text(encoding="utf-8")
    assert "RunId" in content
    assert "AsOfDate" in content
    assert "OutputDir" in content
    assert "ArchiveDir" in content


def test_ops_archive_ps1_calls_export_evidence_pack() -> None:
    """Script calls export_evidence_pack.py."""
    content = OPS_ARCHIVE_PS1.read_text(encoding="utf-8")
    assert "export_evidence_pack.py" in content


def test_ops_archive_ps1_calls_verify_evidence_pack() -> None:
    """Script calls verify_evidence_pack.py."""
    content = OPS_ARCHIVE_PS1.read_text(encoding="utf-8")
    assert "verify_evidence_pack.py" in content


def test_ops_archive_ps1_verify_json_out_param_and_filenames() -> None:
    """Script has -VerifyJsonOut and writes export_ / verify_ JSON filenames when set."""
    content = OPS_ARCHIVE_PS1.read_text(encoding="utf-8")
    assert "VerifyJsonOut" in content
    assert "export_" in content
    assert "verify_" in content
