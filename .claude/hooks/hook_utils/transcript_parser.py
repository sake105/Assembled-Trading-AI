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
        if t == "user":
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
