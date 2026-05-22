"""Smoke and Help Tests for Paper Track CLI.

Tests the CLI interface for paper track operations (if available).
Since paper track may be invoked through core functions or CLI, we test both paths.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.advanced


@pytest.mark.advanced
def test_paper_track_module_importable():
    """Test that paper track module can be imported."""
    from src.assembled_core.paper.paper_track import (
        PaperTrackConfig,
        PaperTrackState,
        PaperTrackDayResult,
    )

    # Just verify imports work
    assert PaperTrackConfig is not None
    assert PaperTrackState is not None
    assert PaperTrackDayResult is not None


@pytest.mark.advanced
def test_paper_track_health_check_cli_help():
    """Test that health check CLI supports paper track arguments."""
    cli_path = Path(__file__).parent.parent / "scripts" / "cli.py"

    cmd = [
        sys.executable,
        str(cli_path),
        "check_health",
        "--help",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    assert result.returncode == 0, (
        f"Help should exit with code 0. stderr: {result.stderr}"
    )
    assert "paper-track" in result.stdout.lower(), (
        "Help should mention paper-track arguments"
    )


@pytest.mark.advanced
def test_paper_track_health_check_cli_smoke(tmp_path: Path):
    """Smoke test: check_health with paper track options should not crash."""
    cli_path = Path(__file__).parent.parent / "scripts" / "cli.py"
    backtests_root = tmp_path / "backtests"
    backtests_root.mkdir(parents=True)
    paper_track_root = tmp_path / "paper_track"
    paper_track_root.mkdir(parents=True)
    output_dir = tmp_path / "health"

    cmd = [
        sys.executable,
        str(cli_path),
        "check_health",
        "--backtests-root",
        str(backtests_root),
        "--paper-track-root",
        str(paper_track_root),
        "--skip-paper-track-if-missing",
        "--output-dir",
        str(output_dir),
        "--format",
        "json",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    # Should not crash (exit code 0, 1, or 2 is acceptable)
    assert result.returncode in [0, 1, 2], (
        f"check_health with paper-track should exit with 0/1/2, got {result.returncode}. "
        f"stderr: {result.stderr}"
    )

    # Should create output files
    assert (output_dir / "health_summary.json").exists(), (
        "health_summary.json should be created"
    )
    assert (output_dir / "health_summary.md").exists(), (
        "health_summary.md should be created"
    )


@pytest.mark.advanced
def test_check_health_script_paper_track_args(tmp_path: Path):
    """Test that standalone check_health.py script supports paper track arguments."""
    script_path = Path(__file__).parent.parent / "scripts" / "check_health.py"
    backtests_root = tmp_path / "backtests"
    backtests_root.mkdir(parents=True)
    paper_track_root = tmp_path / "paper_track"
    paper_track_root.mkdir(parents=True)
    output_dir = tmp_path / "health"

    cmd = [
        sys.executable,
        str(script_path),
        "--backtests-root",
        str(backtests_root),
        "--paper-track-root",
        str(paper_track_root),
        "--skip-paper-track-if-missing",
        "--output-dir",
        str(output_dir),
        "--format",
        "json",
    ]

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=Path(__file__).parent.parent
    )

    # Should not crash
    assert result.returncode in [0, 1, 2], (
        f"check_health.py should exit with 0/1/2, got {result.returncode}. "
        f"stderr: {result.stderr}"
    )


@pytest.mark.advanced
def test_paper_track_core_functions_callable():
    """Test that core paper track functions are callable (smoke test)."""
    from src.assembled_core.paper.paper_track import (
        _filter_prices_for_date,
        _compute_position_value,
    )
    import pandas as pd

    # Test _filter_prices_for_date
    prices = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="D", tz="UTC"),
            "symbol": ["AAPL"] * 5,
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        }
    )

    as_of = pd.Timestamp("2025-01-03", tz="UTC")
    filtered = _filter_prices_for_date(prices, as_of)

    assert not filtered.empty, "Filtered prices should not be empty"
    assert len(filtered) == 1, "Should have one row (last available <= as_of)"
    assert filtered["timestamp"].iloc[0] <= as_of, (
        "Filtered timestamp should be <= as_of"
    )

    # Test _compute_position_value
    positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "qty": [10.0, 20.0],
        }
    )

    prices_for_value = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"],
            "close": [100.0, 200.0],
        }
    )

    value = _compute_position_value(positions, prices_for_value)

    assert value == 5000.0, f"Expected value 5000.0 (10*100 + 20*200), got {value}"
