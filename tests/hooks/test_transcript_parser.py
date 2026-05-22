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


def test_null_message_does_not_crash(tmp_path):
    """F-senior-1 regression: type=user with message=null must not crash the parser.

    Pre-fix the parser called .get("content") on None → AttributeError → hook
    exited non-zero → Claude Code fail-opened → silent bypass via crash path.
    Now: null message is treated as not-a-real-user-message (not a turn boundary).
    """
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"type":"user","message":{"role":"user","content":"real request"},"uuid":"u-real"}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Edit","input":{"file_path":"F:/Python_Projekt/Aktiengeruest/scripts/x.py","old_string":"a","new_string":"b"}}]},"uuid":"a-edit"}\n'
        '{"type":"user","message":null,"uuid":"u-null"}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"uuid":"a-final"}\n',
        encoding="utf-8",
    )
    paths = edited_paths_in_last_turn(p, repo_root=REPO_ROOT_FAKE)
    assert "scripts/x.py" in paths


def test_whitespace_only_user_content_not_a_turn_boundary(tmp_path):
    """F-senior-3: whitespace-only string content is not a real user turn."""
    p = tmp_path / "t.jsonl"
    p.write_text(
        '{"type":"user","message":{"role":"user","content":"real request"},"uuid":"u-real"}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Edit","input":{"file_path":"F:/Python_Projekt/Aktiengeruest/scripts/y.py","old_string":"a","new_string":"b"}}]},"uuid":"a-edit"}\n'
        '{"type":"user","message":{"role":"user","content":"   "},"uuid":"u-ws"}\n'
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"uuid":"a-final"}\n',
        encoding="utf-8",
    )
    paths = edited_paths_in_last_turn(p, repo_root=REPO_ROOT_FAKE)
    assert "scripts/y.py" in paths


def test_tool_result_user_wrappers_do_not_terminate_trailing_turn(tmp_path):
    """Regression: Claude Code wraps tool results as type=user entries whose
    content is a list of tool_result blocks. Those must NOT count as turn
    boundaries — otherwise the parser stops at the first tool result coming
    back after an Edit/Write and never sees the edit.

    Pre-fix bug: parser stopped at the user(tool_result) entry, so trailing
    edits were invisible to the Stop hook → review chain never blocked despite
    real edits in protected paths.
    """
    p = tmp_path / "t.jsonl"
    p.write_text(
        # Real user message (turn boundary)
        '{"type":"user","message":{"role":"user","content":"fix scripts/foo.py"},"uuid":"u-real"}\n'
        # Assistant makes an Edit
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_use","name":"Edit","input":{"file_path":"F:/Python_Projekt/Aktiengeruest/scripts/foo.py","old_string":"a","new_string":"b"}}]},"uuid":"a-edit"}\n'
        # Tool result wrapped as type=user (NOT a real user message)
        '{"type":"user","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]},"uuid":"u-tr"}\n'
        # Assistant final text response
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"done"}]},"uuid":"a-final"}\n'
        # Attachment entries (unknown type — must be skipped)
        '{"type":"attachment","content":""}\n',
        encoding="utf-8",
    )
    paths = edited_paths_in_last_turn(p, repo_root=REPO_ROOT_FAKE)
    assert "scripts/foo.py" in paths, (
        f"Edit was hidden by tool_result wrapper bug. Got: {paths}"
    )
