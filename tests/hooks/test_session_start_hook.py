"""Integration test for the SessionStart hook."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / ".claude" / "hooks" / "session_start_load_errors.py"


def test_hook_outputs_top10_entries_as_additional_context(tmp_path, monkeypatch):
    # Write a minimal errors-log fixture
    log_path = tmp_path / "CLAUDE_CODING_ERRORS.md"
    log_path.write_text(
        "# Claude Coding Errors\n\n"
        "## E-001 — Test pattern\n"
        "**Datum:** 2026-05-05\n"
        "**Kategorie:** pandas-pitfall\n"
        "**Was passierte:** something broke\n"
        "**Warum falsch:** wrong assumption\n"
        "**Wie vermeiden:** do X instead\n"
        "**Erkannt in:** src/foo.py\n"
        "**Referenzen:** memory/x.md\n",
        encoding="utf-8",
    )

    env = {"CLAUDE_HOOKS_ERRORS_LOG_PATH": str(log_path), "PYTHONIOENCODING": "utf-8"}
    # Hook receives minimal SessionStart event
    hook_input = json.dumps({"session_id": "test", "transcript_path": ""})
    result = subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=hook_input,
        capture_output=True,
        text=True,
        env={**env},
    )

    assert result.returncode == 0, f"stderr: {result.stderr}"
    payload = json.loads(result.stdout)
    out = payload["hookSpecificOutput"]
    assert out["hookEventName"] == "SessionStart"
    assert "E-001" in out["additionalContext"]
    assert "do X instead" in out["additionalContext"]


def test_hook_with_missing_log_outputs_empty_section(tmp_path):
    env = {
        "CLAUDE_HOOKS_ERRORS_LOG_PATH": str(tmp_path / "nope.md"),
        "PYTHONIOENCODING": "utf-8",
    }
    hook_input = json.dumps({"session_id": "test", "transcript_path": ""})
    result = subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=hook_input,
        capture_output=True,
        text=True,
        env={**env},
    )

    assert result.returncode == 0
    payload = json.loads(result.stdout)
    # Empty log → still valid output, just empty entries section
    assert "additionalContext" in payload["hookSpecificOutput"]
    assert (
        "Top 10" in payload["hookSpecificOutput"]["additionalContext"]
        or "keine Einträge" in payload["hookSpecificOutput"]["additionalContext"]
    )
