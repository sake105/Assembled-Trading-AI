"""Integration tests for the Stop hook entry point."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_PATH = REPO_ROOT / ".claude" / "hooks" / "stop_review_chain.py"


def _run_hook(
    stdin_payload: dict, env_overrides: dict | None = None
) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        **(env_overrides or {}),
    }
    return subprocess.run(
        [sys.executable, str(HOOK_PATH)],
        input=json.dumps(stdin_payload),
        capture_output=True,
        text=True,
        env=env,
    )


def test_hook_allows_stop_when_no_transcript(tmp_path):
    """If transcript doesn't exist, hook can't classify → fail-open, allow stop."""
    res = _run_hook(
        {
            "session_id": "test",
            "transcript_path": str(tmp_path / "nope.jsonl"),
            "stop_hook_active": False,
        }
    )
    assert res.returncode == 0
    # No block decision → empty stdout or {"decision": "approve"}
    if res.stdout.strip():
        payload = json.loads(res.stdout)
        assert payload.get("decision") != "block"


def test_hook_allows_stop_when_no_protected_edits(tmp_path):
    transcript = tmp_path / "t.jsonl"
    # Last assistant turn only edits docs/ → not protected
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {
                                "file_path": str(REPO_ROOT / "docs" / "foo.md"),
                                "old_string": "a",
                                "new_string": "b",
                            },
                        }
                    ],
                },
                "uuid": "a1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    res = _run_hook(
        {
            "session_id": "test",
            "transcript_path": str(transcript),
            "stop_hook_active": False,
        },
        env_overrides={"CLAUDE_HOOKS_STATE_DIR": str(tmp_path / "state")},
    )
    assert res.returncode == 0
    if res.stdout.strip():
        payload = json.loads(res.stdout)
        assert payload.get("decision") != "block"


def test_hook_blocks_stop_when_protected_edit_and_no_marker(tmp_path):
    transcript = tmp_path / "t.jsonl"
    # Last assistant turn edits src/ → protected
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {
                                "file_path": str(
                                    REPO_ROOT
                                    / "src"
                                    / "assembled_core"
                                    / "execution"
                                    / "router.py"
                                ),
                                "old_string": "a",
                                "new_string": "b",
                            },
                        }
                    ],
                },
                "uuid": "a1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    res = _run_hook(
        {
            "session_id": "test",
            "transcript_path": str(transcript),
            "stop_hook_active": False,
        },
        env_overrides={"CLAUDE_HOOKS_STATE_DIR": str(tmp_path / "state")},
    )
    assert res.returncode == 0
    payload = json.loads(res.stdout)
    assert payload["decision"] == "block"
    assert "REVIEW-CHAIN-REQUIRED" in payload["reason"]
    assert "risk-execution-reviewer" in payload["reason"]
    assert "senior-code-reviewer" in payload["reason"]
    assert "task-completion-auditor" in payload["reason"]


def test_hook_allows_stop_when_marker_already_written(tmp_path):
    """If review chain has already run for this turn, marker exists, hook allows stop."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {
                                "file_path": str(REPO_ROOT / "src" / "foo.py"),
                                "old_string": "a",
                                "new_string": "b",
                            },
                        }
                    ],
                },
                "uuid": "a1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    # First call: should block
    res1 = _run_hook(
        {
            "session_id": "test",
            "transcript_path": str(transcript),
            "stop_hook_active": False,
        },
        env_overrides={"CLAUDE_HOOKS_STATE_DIR": str(state_dir)},
    )
    payload1 = json.loads(res1.stdout)
    assert payload1["decision"] == "block"
    # Write marker manually (simulating main agent finishing review)
    write_marker_script = (
        "import sys; sys.path.insert(0, r'"
        + str(REPO_ROOT / ".claude" / "hooks")
        + "');"
        "from hook_utils.review_marker import turn_id_from_transcript, write_review_marker;"
        "from pathlib import Path;"
        "tid = turn_id_from_transcript(Path(r'" + str(transcript) + "'));"
        "write_review_marker(tid, Path(r'" + str(state_dir) + "'));"
    )
    subprocess.run([sys.executable, "-c", write_marker_script], check=True)

    # Second call: should allow
    res2 = _run_hook(
        {
            "session_id": "test",
            "transcript_path": str(transcript),
            "stop_hook_active": False,
        },
        env_overrides={"CLAUDE_HOOKS_STATE_DIR": str(state_dir)},
    )
    assert res2.returncode == 0
    if res2.stdout.strip():
        payload2 = json.loads(res2.stdout)
        assert payload2.get("decision") != "block"


def test_hook_respects_stop_hook_active_to_avoid_infinite_loop(tmp_path):
    """If stop_hook_active=true, never re-block — Claude Code is already in a hook loop."""
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Edit",
                            "input": {
                                "file_path": str(REPO_ROOT / "src" / "foo.py"),
                                "old_string": "a",
                                "new_string": "b",
                            },
                        }
                    ],
                },
                "uuid": "a1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    res = _run_hook(
        {
            "session_id": "test",
            "transcript_path": str(transcript),
            "stop_hook_active": True,  # already in a loop
        },
        env_overrides={"CLAUDE_HOOKS_STATE_DIR": str(tmp_path / "state")},
    )
    assert res.returncode == 0
    if res.stdout.strip():
        payload = json.loads(res.stdout)
        assert payload.get("decision") != "block"
