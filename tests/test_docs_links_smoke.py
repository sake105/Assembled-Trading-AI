"""Smoke test: required docs exist and (optional) referenced workflow files exist.

Stdlib only, fast, platform-neutral. Catches dead links before release.
"""

from __future__ import annotations

import re
from pathlib import Path

# Repo root
ROOT = Path(__file__).resolve().parents[1]


# Required docs (relative to repo root)
REQUIRED_DOCS = [
    "docs/OPS_EVIDENCE_GOLDEN_PATH.md",
    "docs/EVIDENCE_PACK.md",
    "docs/PROJECT_STRUCTURE.md",
]

# Pattern to find .github/workflows/... references in markdown (links or plain paths)
WORKFLOW_REF_PATTERN = re.compile(r"\.github/workflows/([a-zA-Z0-9_.-]+\.(?:yml|yaml))")


def test_required_docs_exist() -> None:
    """Required doc files exist (relative paths from repo root)."""
    for rel in REQUIRED_DOCS:
        path = ROOT / rel
        assert path.exists(), f"Required doc missing: {rel}"
        assert path.is_file(), f"Required doc is not a file: {rel}"


def test_readme_links_ops_evidence_golden_path() -> None:
    """README must reference the canonical ops archive workflow doc (docs/OPS_EVIDENCE_GOLDEN_PATH.md)."""
    readme = ROOT / "README.md"
    assert readme.exists(), "README.md must exist"
    content = readme.read_text(encoding="utf-8")
    assert "docs/OPS_EVIDENCE_GOLDEN_PATH.md" in content, (
        "README must contain docs/OPS_EVIDENCE_GOLDEN_PATH.md (canonical ops workflow)"
    )


def test_referenced_workflows_exist() -> None:
    """Scan required docs for .github/workflows/... references; referenced YAMLs must exist."""
    seen: set[str] = set()
    for rel in REQUIRED_DOCS:
        path = ROOT / rel
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for match in WORKFLOW_REF_PATTERN.finditer(text):
            wf_name = match.group(1)
            if wf_name in seen:
                continue
            seen.add(wf_name)
            wf_path = ROOT / ".github" / "workflows" / wf_name
            assert wf_path.exists(), (
                f"Doc references workflow that does not exist: .github/workflows/{wf_name} (referenced from {rel})"
            )
            assert wf_path.is_file(), (
                f"Workflow path is not a file: .github/workflows/{wf_name}"
            )
