"""Classify edited file paths for the Stop-hook review chain.

Two questions:
1. is_protected_path(path) → bool: should the review chain run at all?
2. specialists_for_paths(paths) → set[str]: which Stage-1 specialists?

Path families (per spec §4.2.1):
- src/**                       → test-runner (always)
- src/assembled_core/{execution,risk,pipeline,accounting,portfolio,paper}/**
                               → also risk-execution-reviewer
- scripts/**                   → test-runner
- .github/workflows/**         → ci-debugger
- CLAUDE.md, .claude/rules/**  → docs-governance-sync
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Iterable, Set

PROTECTED_PREFIXES = (
    "src/",
    "scripts/",
    ".github/workflows/",
    ".claude/rules/",
    # F-senior-2 (review_chain_disclosure §20.8, umgesetzt 2026-07-22
    # GESAMTBEWERTUNG P8): the enforcement layer itself must be
    # chain-protected — an edit to the hooks could silently disable the
    # chain that reviews edits.
    ".claude/hooks/",
)

SENSITIVE_ZONES = (
    "src/assembled_core/execution/",
    "src/assembled_core/risk/",
    "src/assembled_core/pipeline/",
    "src/assembled_core/accounting/",
    "src/assembled_core/portfolio/",
    "src/assembled_core/paper/",
)


def _norm(path: str) -> str:
    """Normalize path to posix-style relative path."""
    return str(PurePosixPath(path.replace("\\", "/")))


def is_protected_path(path: str) -> bool:
    """Return True iff editing this path should trigger the review chain."""
    p = _norm(path)
    if p == "CLAUDE.md":
        return True
    return any(p.startswith(prefix) for prefix in PROTECTED_PREFIXES)


def specialists_for_paths(paths: Iterable[str]) -> Set[str]:
    """Return the set of specialist subagent names that should run Stage 1.

    Empty set means no protected paths → review chain should not run at all.
    """
    specs: Set[str] = set()
    for raw in paths:
        if not is_protected_path(raw):
            continue
        p = _norm(raw)

        # test-runner: any code change in src/ or scripts/
        if p.startswith("src/") or p.startswith("scripts/"):
            specs.add("test-runner")

        # risk-execution-reviewer: sensitive zones
        if any(p.startswith(zone) for zone in SENSITIVE_ZONES):
            specs.add("risk-execution-reviewer")

        # ci-debugger: workflow changes
        if p.startswith(".github/workflows/"):
            specs.add("ci-debugger")

        # docs-governance-sync: CLAUDE.md and rules
        if p == "CLAUDE.md" or p.startswith(".claude/rules/"):
            specs.add("docs-governance-sync")

    return specs
