"""Test CLI batch_backtest alias functionality."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cli_batch_backtest_help() -> None:
    """Test that batch_backtest subcommand shows help."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "cli.py"), "batch_backtest", "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"Expected exit code 0, got {result.returncode}. stderr: {result.stderr}"
    assert "batch_backtest" in result.stdout.lower() or "batch" in result.stdout.lower()
    assert "--config-file" in result.stdout


def test_batch_backtest_script_help() -> None:
    """Test that batch_backtest.py script shows help."""
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "batch_backtest.py"), "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, f"Expected exit code 0, got {result.returncode}. stderr: {result.stderr}"
    assert "--config-file" in result.stdout

