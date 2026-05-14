#!/usr/bin/env python3
"""SessionStart hook: load Top-10 Claude coding anti-patterns into context.

Reads docs/CLAUDE_CODING_ERRORS.md, parses entries, returns the 10 most recent
as additionalContext via Claude Code's SessionStart hook output schema.

Env override: CLAUDE_HOOKS_ERRORS_LOG_PATH points to an alternative log path
(used in tests).

Output schema (Claude Code hooks):
    {"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "..."}}
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make hook_utils importable when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hook_utils.errors_log import parse_errors_log, top_n_entries  # noqa: E402


def _resolve_log_path() -> Path:
    override = os.environ.get("CLAUDE_HOOKS_ERRORS_LOG_PATH")
    if override:
        return Path(override)
    # Default: docs/CLAUDE_CODING_ERRORS.md relative to repo root.
    # Repo root is two levels up from this file (.claude/hooks/).
    return Path(__file__).resolve().parents[2] / "docs" / "CLAUDE_CODING_ERRORS.md"


def _format_entries(entries: list[dict]) -> str:
    if not entries:
        return (
            "# Claude Coding Errors — Top 10\n"
            "(noch keine Einträge — Anti-Pattern-Register ist leer)\n"
        )
    lines = ["# Claude Coding Errors — Top 10 (vermeide diese Muster!)"]
    for e in entries:
        lines.append(
            f"- **{e['id']}** ({e.get('datum', '?')}, {e.get('kategorie', '?')}): "
            f"{e.get('title', '')} — *Wie vermeiden:* {e.get('how_to_avoid', '?')}"
        )
    lines.append(
        "\n*Volle Details in docs/CLAUDE_CODING_ERRORS.md. "
        "Wenn ein neuer Anti-Pattern erkannt wird → senior-code-reviewer schlägt Eintrag vor.*"
    )
    return "\n".join(lines)


def main() -> int:
    # Drain stdin (Claude Code provides JSON event), we don't need its fields
    try:
        sys.stdin.read()
    except Exception:
        pass

    log_path = _resolve_log_path()
    entries = parse_errors_log(log_path)
    top10 = top_n_entries(entries, n=10)
    context = _format_entries(top10)

    payload = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": context,
        }
    }
    sys.stdout.write(json.dumps(payload))
    return 0


if __name__ == "__main__":
    sys.exit(main())
