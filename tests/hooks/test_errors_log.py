"""Tests for the errors-log parser used by the SessionStart hook."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path


# Make .claude/hooks importable for tests
HOOKS_DIR = Path(__file__).resolve().parents[2] / ".claude" / "hooks"
sys.path.insert(0, str(HOOKS_DIR))

from hook_utils.errors_log import parse_errors_log, top_n_entries  # noqa: E402


def _write_log(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "CLAUDE_CODING_ERRORS.md"
    p.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return p


def test_parse_empty_file_returns_empty_list(tmp_path):
    p = _write_log(tmp_path, "# Claude Coding Errors\n\nNo entries yet.\n")
    assert parse_errors_log(p) == []


def test_parse_single_entry(tmp_path):
    body = """
    # Claude Coding Errors

    ## E-001 — pandas .where(Series) row-index alignment bug
    **Datum:** 2026-05-05
    **Kategorie:** pandas-pitfall
    **Was passierte:** z-score broke.
    **Warum falsch:** alignment.
    **Wie vermeiden:** use .values.
    **Erkannt in:** src/foo.py
    **Referenzen:** memory/...md
    """
    p = _write_log(tmp_path, body)
    entries = parse_errors_log(p)
    assert len(entries) == 1
    e = entries[0]
    assert e["id"] == "E-001"
    assert e["title"] == "pandas .where(Series) row-index alignment bug"
    assert e["datum"] == "2026-05-05"
    assert e["kategorie"] == "pandas-pitfall"
    assert "use .values" in e["how_to_avoid"]


def test_parse_multiple_entries_in_order(tmp_path):
    body = """
    # Claude Coding Errors

    ## E-001 — First
    **Datum:** 2026-05-01
    **Kategorie:** other
    **Was passierte:** a
    **Warum falsch:** b
    **Wie vermeiden:** c
    **Erkannt in:** x
    **Referenzen:** y

    ## E-002 — Second
    **Datum:** 2026-05-02
    **Kategorie:** other
    **Was passierte:** d
    **Warum falsch:** e
    **Wie vermeiden:** f
    **Erkannt in:** z
    **Referenzen:** w
    """
    p = _write_log(tmp_path, body)
    entries = parse_errors_log(p)
    assert [e["id"] for e in entries] == ["E-001", "E-002"]


def test_top_n_returns_most_recent_by_date(tmp_path):
    body = """
    # Claude Coding Errors

    ## E-001 — Old
    **Datum:** 2026-01-01
    **Kategorie:** other
    **Was passierte:** a
    **Warum falsch:** b
    **Wie vermeiden:** c
    **Erkannt in:** x
    **Referenzen:** y

    ## E-002 — New
    **Datum:** 2026-05-01
    **Kategorie:** other
    **Was passierte:** d
    **Warum falsch:** e
    **Wie vermeiden:** f
    **Erkannt in:** z
    **Referenzen:** w

    ## E-003 — Middle
    **Datum:** 2026-03-01
    **Kategorie:** other
    **Was passierte:** g
    **Warum falsch:** h
    **Wie vermeiden:** i
    **Erkannt in:** q
    **Referenzen:** r
    """
    p = _write_log(tmp_path, body)
    entries = parse_errors_log(p)
    top2 = top_n_entries(entries, n=2)
    assert [e["id"] for e in top2] == ["E-002", "E-003"]


def test_top_n_handles_fewer_entries_than_n(tmp_path):
    body = """
    # Claude Coding Errors

    ## E-001 — Solo
    **Datum:** 2026-01-01
    **Kategorie:** other
    **Was passierte:** a
    **Warum falsch:** b
    **Wie vermeiden:** c
    **Erkannt in:** x
    **Referenzen:** y
    """
    p = _write_log(tmp_path, body)
    entries = parse_errors_log(p)
    assert len(top_n_entries(entries, n=10)) == 1


def test_missing_file_returns_empty(tmp_path):
    assert parse_errors_log(tmp_path / "does-not-exist.md") == []
