"""Parse Claude Code transcript JSONL to extract edited file paths.

Claude Code writes each turn as one JSONL line. Each line is a JSON object with
fields like `type` (user|assistant), `message.content` (list of content blocks),
etc.

We care about tool_use blocks with name in {Edit, Write, NotebookEdit} and an
`input.file_path` field. We return paths from the *trailing* assistant turn
(contiguous trailing assistant messages — stops at the most recent user message).
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import List

EDITING_TOOLS = {"Edit", "Write", "NotebookEdit", "MultiEdit"}


def _is_real_user_message(obj: dict) -> bool:
    """True if this is genuine user input, False if it's a tool_result wrapper.

    Claude Code wraps tool results as `type=user` entries whose `message.content`
    is a list of `tool_result` blocks. Those must not count as turn boundaries —
    otherwise the parser stops at the first tool result coming back to the
    assistant and never sees the Edit/Write that preceded it.

    A real user message either has a string content or a content list that
    contains at least one `text` block. Pure-tool_result wrappers do not.
    """
    # F-senior-1: `message=null` (vs. missing key) reaches .get on None → AttributeError.
    # The hook would then exit non-zero, Claude Code treats hook failure as fail-open,
    # same silent-bypass outcome as the original bug via a different mechanism.
    message = obj.get("message") or {}
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        # F-senior-3: whitespace-only string is not a real user turn (production
        # transcripts never emit them; defensive against synthetic injections).
        return bool(content.strip())
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") == "text" for b in content)


def _rel_to_repo(file_path: str, repo_root: Path) -> str | None:
    """Return path relative to repo_root in posix form, or None if outside repo."""
    try:
        p = Path(file_path)
        rel = p.resolve().relative_to(repo_root.resolve())
        return str(PurePosixPath(rel))
    except (ValueError, OSError):
        # Fall back to string slicing if the file doesn't exist (e.g., tests)
        norm = file_path.replace("\\", "/")
        root_norm = str(PurePosixPath(repo_root.as_posix()))
        if norm.startswith(root_norm + "/"):
            return norm[len(root_norm) + 1 :]
        if norm.startswith(root_norm.rstrip("/") + "/"):
            return norm[len(root_norm.rstrip("/")) + 1 :]
        return None


def edited_paths_in_last_turn(transcript_path: Path, repo_root: Path) -> List[str]:
    """Return list of repo-relative paths edited in the trailing assistant turn.

    'Trailing turn' = contiguous trailing assistant messages, back to last user message.
    """
    if not transcript_path.exists():
        return []

    lines = transcript_path.read_text(encoding="utf-8").splitlines()
    # Walk from end backwards, collecting assistant entries until we hit a user
    # entry or the start of the file.
    trailing: List[dict] = []
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = obj.get("type")
        if t == "user" and _is_real_user_message(obj):
            break
        if t == "assistant":
            trailing.append(obj)

    paths: List[str] = []
    for obj in reversed(trailing):  # restore chronological order
        content = obj.get("message", {}).get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            if block.get("name") not in EDITING_TOOLS:
                continue
            fp = block.get("input", {}).get("file_path")
            if not fp:
                continue
            rel = _rel_to_repo(fp, repo_root)
            if rel and rel not in paths:
                paths.append(rel)
    return paths
