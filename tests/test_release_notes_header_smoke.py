"""Smoke test: RELEASE_NOTES_SPRINT13.md has version header in first 5 lines (ASCII-only)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE_NOTES = ROOT / "docs" / "RELEASE_NOTES_SPRINT13.md"


def test_release_notes_file_exists() -> None:
    """Release notes file exists."""
    assert RELEASE_NOTES.exists()
    assert RELEASE_NOTES.is_file()


def test_release_notes_header_contains_release_and_tag() -> None:
    """First 5 lines contain 'Release:' and 'Tag:' (exact format). Header lines ASCII-only."""
    content = RELEASE_NOTES.read_text(encoding="utf-8")
    lines = content.strip().splitlines()
    first_lines = lines[:5] if len(lines) >= 5 else lines
    combined = "\n".join(first_lines)
    assert "Release:" in combined, "Header must contain 'Release:' in first 5 lines"
    assert "Tag:" in combined, "Header must contain 'Tag:' in first 5 lines"
    # Header area must be ASCII (version line is machine-readable)
    for line in first_lines:
        assert line.encode("ascii", errors="ignore").decode("ascii") == line, "Header lines must be ASCII-only"
