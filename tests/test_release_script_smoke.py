"""Smoke tests for scripts/dev/release_sprint13.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_release_sprint13_help_exits_zero() -> None:
    """python scripts/dev/release_sprint13.py --help exits 0."""
    script = ROOT / "scripts" / "dev" / "release_sprint13.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0
    assert "release" in result.stdout.lower() or "sprint" in result.stdout.lower()


def test_release_sprint13_dry_run_prints_commands() -> None:
    """--dry-run prints commands only and exits 0."""
    script = ROOT / "scripts" / "dev" / "release_sprint13.py"
    result = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert result.returncode == 0
    assert "dry-run" in result.stdout or "run_checks" in result.stdout
    assert "release_sprint13" in result.stdout
    assert "evidence_pack" in result.stdout
