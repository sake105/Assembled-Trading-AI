"""Tests for review_marker: per-turn marker preventing re-trigger of review chain."""

from __future__ import annotations

import sys
from pathlib import Path

HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

from hook_utils.review_marker import (  # noqa: E402
    turn_id_from_transcript,
    has_review_marker,
    write_review_marker,
)


def test_turn_id_is_stable_for_same_transcript(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text('{"type":"assistant","uuid":"abc"}\n', encoding="utf-8")
    id1 = turn_id_from_transcript(t)
    id2 = turn_id_from_transcript(t)
    assert id1 == id2
    assert id1  # non-empty


def test_turn_id_changes_when_transcript_grows(tmp_path):
    t = tmp_path / "t.jsonl"
    t.write_text('{"type":"assistant","uuid":"abc"}\n', encoding="utf-8")
    id1 = turn_id_from_transcript(t)
    t.write_text(
        '{"type":"assistant","uuid":"abc"}\n{"type":"assistant","uuid":"def"}\n',
        encoding="utf-8",
    )
    id2 = turn_id_from_transcript(t)
    assert id1 != id2


def test_has_marker_false_initially(tmp_path):
    state_dir = tmp_path / "state"
    assert has_review_marker("turn-123", state_dir) is False


def test_write_then_has_marker_true(tmp_path):
    state_dir = tmp_path / "state"
    write_review_marker("turn-456", state_dir)
    assert has_review_marker("turn-456", state_dir) is True


def test_different_turns_independent(tmp_path):
    state_dir = tmp_path / "state"
    write_review_marker("turn-A", state_dir)
    assert has_review_marker("turn-A", state_dir) is True
    assert has_review_marker("turn-B", state_dir) is False
