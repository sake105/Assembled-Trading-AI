"""Regression guard for scripts/ack_halt.py (2026-07-02).

The halt-ack CLI fires a ``halt_cleared`` alert via
``from src.assembled_core.ops.alerting import AlertManager``. When run as a
script, ``sys.path[0]`` is scripts/ (not the repo root), so that import failed
with ``ModuleNotFoundError: No module named 'src'`` and the alert silently
never fired (the flag was still cleared). The fix adds the repo root to
sys.path at module load. This test reproduces the real entry-point condition
via a subprocess with an isolated cwd so no real output/ops state is touched.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.fast

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ack_halt.py"


def test_ack_halt_clears_flag_without_src_import_error(tmp_path):
    """Running the script clears the flag, writes the ledger, and does NOT
    raise ModuleNotFoundError: No module named 'src' (the fixed bug)."""
    ops = tmp_path / "output" / "ops"
    ops.mkdir(parents=True)
    flag = ops / "halt_ack_required.json"
    flag.write_text(
        json.dumps(
            {"ts_utc": "2026-05-22T19:59:27+00:00", "reason": "test", "source": "test"}
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reason=regression_2026-07-02_ack_halt_src_import",
            "--actor=pytest",
        ],
        cwd=tmp_path,  # isolate the relative output/ops paths into tmp
        capture_output=True,
        text=True,
        timeout=60,
    )

    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "No module named 'src'" not in combined, combined
    assert not flag.exists(), "halt flag should have been cleared"
    ledger = ops / "halt_ack_ledger.jsonl"
    assert ledger.exists(), "ack ledger entry should have been written"
    entry = json.loads(ledger.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert entry["actor"] == "pytest"
    assert entry["reason"] == "regression_2026-07-02_ack_halt_src_import"


def test_ack_halt_noop_when_no_flag(tmp_path):
    """No flag present -> exit 0, nothing to clear, no ledger written."""
    (tmp_path / "output" / "ops").mkdir(parents=True)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--reason=regression_noop_2026-07-02_check",
            "--actor=pytest",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not (tmp_path / "output" / "ops" / "halt_ack_ledger.jsonl").exists()
