"""Tests for transcript_parser: extract edited paths from Claude Code transcript JSONL."""

from __future__ import annotations

import sys
from pathlib import Path

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

from hook_utils.transcript_parser import edited_paths_in_last_turn  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "transcript_with_edits.jsonl"
REPO_ROOT_FAKE = Path("F:/Python_Projekt/Aktiengeruest")


def test_extracts_edit_and_write_paths_relative_to_repo_root():
    paths = edited_paths_in_last_turn(FIXTURE, repo_root=REPO_ROOT_FAKE)
    assert "src/assembled_core/execution/router.py" in paths
    assert "docs/foo.md" in paths


def test_missing_transcript_returns_empty(tmp_path):
    assert (
        edited_paths_in_last_turn(tmp_path / "nope.jsonl", repo_root=REPO_ROOT_FAKE)
        == []
    )


def test_only_returns_paths_from_last_assistant_turn(tmp_path):
    """If a user message follows the last assistant edits, last 'turn' is the
    contiguous trailing assistant messages."""
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Edit","input":{"file_path":"F:/Python_Projekt/Aktiengeruest/src/old.py","old_string":"a","new_string":"b"}}]},"uuid":"a-old"}\n'
        '{"type":"user","message":{"role":"user","content":"new request"},"uuid":"u-new"}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Edit","input":{"file_path":"F:/Python_Projekt/Aktiengeruest/src/new.py","old_string":"a","new_string":"b"}}]},"uuid":"a-new"}\n',
        encoding="utf-8",
    )
    paths = edited_paths_in_last_turn(p, repo_root=REPO_ROOT_FAKE)
    assert "src/new.py" in paths
    assert "src/old.py" not in paths
