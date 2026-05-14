"""Parse docs/CLAUDE_CODING_ERRORS.md into structured entries.

Schema per entry (matches spec §4.3):

    {
        "id": "E-001",
        "title": "...",
        "datum": "2026-05-05",
        "kategorie": "pandas-pitfall",
        "what_happened": "...",
        "why_wrong": "...",
        "how_to_avoid": "...",
        "detected_in": "...",
        "references": "...",
    }
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

ENTRY_HEADING_RE = re.compile(r"^##\s+(E-\d+)\s+—\s+(.+?)\s*$")
FIELD_RE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")

FIELD_MAP = {
    "Datum": "datum",
    "Kategorie": "kategorie",
    "Was passierte": "what_happened",
    "Warum falsch": "why_wrong",
    "Wie vermeiden": "how_to_avoid",
    "Erkannt in": "detected_in",
    "Referenzen": "references",
}


def parse_errors_log(path: Path) -> List[Dict[str, str]]:
    """Parse the errors-log markdown file. Return list of entries in file order."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    entries: List[Dict[str, str]] = []
    current: Dict[str, str] | None = None

    for line in lines:
        m = ENTRY_HEADING_RE.match(line)
        if m:
            if current is not None:
                entries.append(current)
            current = {"id": m.group(1), "title": m.group(2).strip()}
            continue
        if current is None:
            continue
        fm = FIELD_RE.match(line)
        if fm:
            label = fm.group(1).strip()
            value = fm.group(2).strip()
            key = FIELD_MAP.get(label)
            if key:
                current[key] = value

    if current is not None:
        entries.append(current)

    return entries


def top_n_entries(entries: List[Dict[str, str]], n: int = 10) -> List[Dict[str, str]]:
    """Return up to n entries sorted by 'datum' descending (newest first)."""

    def _key(e: Dict[str, str]) -> str:
        return e.get("datum", "0000-00-00")

    return sorted(entries, key=_key, reverse=True)[:n]
