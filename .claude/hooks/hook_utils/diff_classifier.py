"""Classify a set of edited file paths into a diff-kind.

Per spec §4.2.1:
- skip: no protected paths → review chain doesn't run at all.
- docs-only: only documentation files (excluding governance) → Stage 2+3 skipped.
- test-only: only tests/** files → Stage 1 = test-runner, Stage 2+3 run.
- full: any mix that includes src/, scripts/, workflows, or governance → full chain.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable

from .path_classifier import _norm, is_protected_path


def _is_docs_only_path(path: str) -> bool:
    p = _norm(path)
    if p == "CLAUDE.md":
        return False
    if p.startswith(".claude/rules/"):
        return False
    if p.startswith("docs/") and p.endswith(".md"):
        return True
    if p.endswith(".md"):
        return True
    return False


def _is_test_only_path(path: str) -> bool:
    p = _norm(path)
    return p.startswith("tests/") and p.endswith(".py")


def classify_diff(paths: Iterable[str]) -> Dict[str, Any]:
    """Classify the diff. Returns {kind, run_full_chain, protected_paths}."""
    paths = list(paths)
    protected = [p for p in paths if is_protected_path(p)]

    if not paths:
        return {"kind": "skip", "run_full_chain": False, "protected_paths": []}

    if not protected:
        # Mark test-only / docs-only specially even when not "protected",
        # because tests/** and docs/** don't appear in protected list but
        # are common edit targets we care about.
        if all(_is_test_only_path(p) for p in paths):
            return {"kind": "test-only", "run_full_chain": True, "protected_paths": []}
        if all(_is_docs_only_path(p) for p in paths):
            return {"kind": "docs-only", "run_full_chain": False, "protected_paths": []}
        return {"kind": "skip", "run_full_chain": False, "protected_paths": []}

    # We have protected paths.
    # If ALL paths are docs-only (which means no protected since docs/ isn't protected),
    # we've already returned above. So 'protected non-empty' implies code/governance change.
    return {"kind": "full", "run_full_chain": True, "protected_paths": protected}
