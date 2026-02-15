"""Smoke tests for scripts/dev/tag_release.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_tag_release_help_exits_zero() -> None:
    """python scripts/dev/tag_release.py --help exits 0."""
    script = ROOT / "scripts" / "dev" / "tag_release.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0
    assert "tag" in result.stdout.lower()
    assert "--tag" in result.stdout


def test_tag_release_dry_run_prints_commands() -> None:
    """--dry-run with --tag prints expected git commands (no tag created)."""
    script = ROOT / "scripts" / "dev" / "tag_release.py"
    result = subprocess.run(
        [sys.executable, str(script), "--tag", "v0.1.0", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0
    assert "dry-run" in result.stdout
    assert "git tag" in result.stdout
    assert "v0.1.0" in result.stdout
    assert "OK: tag_created=v0.1.0" in result.stdout


def test_tag_release_dry_run_allows_any_tag_form() -> None:
    """--dry-run exits 0 for any tag form (version mismatch not enforced in dry-run)."""
    script = ROOT / "scripts" / "dev" / "tag_release.py"
    result = subprocess.run(
        [sys.executable, str(script), "--tag", "v9.9.9", "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0
    assert "OK: tag_created=v9.9.9" in result.stdout


def test_tag_release_script_contains_version_check() -> None:
    """Script contains version match check (tag must match __version__)."""
    script = ROOT / "scripts" / "dev" / "tag_release.py"
    content = script.read_text(encoding="utf-8")
    assert "tag_version_matches_package" in content or "__version__" in content
