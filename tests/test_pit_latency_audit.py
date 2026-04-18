"""P0 A5 — CI audit: every ``filter_events_pit`` call passes ``latency_days``.

This test was authored for System-Check Deep Run v2 finding A5 (2026-04-18).

Intent
------
The previous default of ``latency_days=0`` in
``src/assembled_core/data/altdata/contract.py::filter_events_pit`` silently
assumed every caller's data source publishes with zero additional latency
beyond ``disclosure_date``. The fix makes ``latency_days`` a required
parameter; this test is the CI guard that prevents a regression where a
new caller forgets to pass it.

What this test does
-------------------
* Parses every ``.py`` file in ``src/`` and ``tests/`` with ``ast``.
* Collects every ``Call`` whose function name is ``filter_events_pit``.
* Fails if any such call does not supply ``latency_days`` — either as a
  keyword argument or as the third positional argument.
* Skips worktree mirrors under ``.claude/worktrees/``.

This is a structural test. It does not import or run the callers; it only
checks their source. A call like ``filter_events_pit(events, as_of)`` is
rejected; ``filter_events_pit(events, as_of, latency_days=0)`` and
``filter_events_pit(events, as_of, 0)`` are both accepted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.phase_zero

REPO_ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [REPO_ROOT / "src", REPO_ROOT / "tests"]
EXCLUDE_FRAGMENTS = (".claude\\worktrees", ".claude/worktrees", "__pycache__")
# This very file documents the contract by name in its module docstring; do
# not audit its own string occurrences.
SELF_PATH = Path(__file__).resolve()


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            posix = path.as_posix()
            if any(frag in posix for frag in EXCLUDE_FRAGMENTS):
                continue
            if path.resolve() == SELF_PATH:
                continue
            files.append(path)
    return files


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _call_has_latency_days(node: ast.Call) -> bool:
    # Keyword form: filter_events_pit(..., latency_days=X)
    for kw in node.keywords:
        if kw.arg == "latency_days":
            return True
        if kw.arg is None:
            # **kwargs spread — treat as opaque and accept; this is an escape
            # hatch but using **kwargs here is already a code smell and
            # acceptable for the audit's purpose.
            return True
    # Positional form: filter_events_pit(events, as_of, latency_days_value)
    if len(node.args) >= 3:
        return True
    return False


def _collect_pytest_raises_lines(tree: ast.AST) -> set[int]:
    """Line ranges covered by ``with pytest.raises(...)`` blocks.

    Calls inside such blocks are intentionally invalid — the test asserts
    the raised exception — and must not be flagged by the audit.
    """
    covered: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.With, ast.AsyncWith)):
            continue
        for item in node.items:
            expr = item.context_expr
            if not isinstance(expr, ast.Call):
                continue
            func = expr.func
            name = None
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name != "raises":
                continue
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None) or start
            if start is None:
                continue
            for lineno in range(start, (end or start) + 1):
                covered.add(lineno)
    return covered


def test_every_filter_events_pit_call_passes_latency_days() -> None:
    offenders: list[tuple[str, int]] = []
    for path in _iter_python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            # Stub / broken collection files exist in this repo; skip them
            # rather than masking the audit behind a parse error.
            continue
        raises_lines = _collect_pytest_raises_lines(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if _call_name(node) != "filter_events_pit":
                continue
            if node.lineno in raises_lines:
                continue
            if not _call_has_latency_days(node):
                rel = path.relative_to(REPO_ROOT).as_posix()
                offenders.append((rel, node.lineno))

    assert not offenders, (
        "filter_events_pit() must be called with an explicit latency_days "
        "argument (P0 A5, Deep Run v2, 2026-04-18). Offending call sites:\n"
        + "\n".join(f"  {rel}:{line}" for rel, line in offenders)
    )
