# tests/test_full_system_run_smoke.py
"""Smoke test: run_full_system_backtests.py in temp dir with synthetic mode.

Asserts metrics_summary.json and SYSTEM_RUN_REPORT.md are produced.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dev" / "run_full_system_backtests.py"


@pytest.mark.smoke
def test_full_system_run_produces_artifacts(tmp_path: Path) -> None:
    """Run full system backtests with synthetic data; assert metrics and report exist."""
    if not SCRIPT.exists():
        pytest.skip("run_full_system_backtests.py not found")
    out_root = tmp_path / "system_run"
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--output-root",
        str(out_root),
        "--synthetic-only",
        "--freq",
        "1d",
        "--skip-sweep",
    ]
    result = subprocess.run(
        cmd,
        cwd=str(ROOT),
        timeout=420,
        capture_output=True,
        text=True,
    )
    assert (
        result.returncode == 0
    ), f"Script failed: {result.returncode}\nstdout: {result.stdout[-2000:]}\nstderr: {result.stderr[-2000:]}"
    assert (
        out_root / "SYSTEM_RUN_REPORT.md"
    ).exists(), "SYSTEM_RUN_REPORT.md not produced"
    runs = out_root / "runs"
    assert runs.is_dir(), "runs/ dir not produced"
    metrics_found = False
    for run_dir in runs.iterdir():
        if run_dir.is_dir():
            m = run_dir / "metrics_summary.json"
            if m.exists():
                metrics_found = True
                break
    assert metrics_found, "No metrics_summary.json found under runs/<run_id>/"
