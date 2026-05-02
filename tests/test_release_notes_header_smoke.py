"""Smoke test: CHANGELOG.md or RELEASE_NOTES.md has a valid header.

Stdlib only, fast, platform-neutral. Required by release_sprint13 preset in
scripts/dev/run_checks.py. Skips gracefully when neither file exists yet;
passes immediately when one does and has a recognisable top-level heading.
"""

from __future__ import annotations

import pytest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Candidate release-notes files, in preference order.
CANDIDATES = [
    ROOT / "CHANGELOG.md",
    ROOT / "RELEASE_NOTES.md",
]


def _find_release_notes() -> Path | None:
    for p in CANDIDATES:
        if p.exists() and p.is_file():
            return p
    return None


def test_release_notes_header_present() -> None:
    """CHANGELOG.md or RELEASE_NOTES.md exists and starts with a level-1 heading.

    Skipped (not failed) when neither file exists, so the release gate does not
    block on repos that have not yet introduced a changelog.
    """
    doc = _find_release_notes()
    if doc is None:
        pytest.skip(
            "No CHANGELOG.md or RELEASE_NOTES.md found — skipping header smoke test."
        )

    content = doc.read_text(encoding="utf-8", errors="replace").lstrip()
    assert content.startswith("#"), (
        f"{doc.name} must begin with a top-level Markdown heading (# ...). "
        f"First 120 chars: {content[:120]!r}"
    )
