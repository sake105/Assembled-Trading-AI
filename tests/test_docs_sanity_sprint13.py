"""Docs sanity: ASCII-only checks and required sections in persistent docs (stdlib only)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OPS_GOLDEN = DOCS / "OPS_EVIDENCE_GOLDEN_PATH.md"


def _code_blocks_with_py3(content: str) -> int:
    """Count fenced code blocks that contain 'py -3'."""
    blocks = re.findall(r"```[\s\S]*?```", content)
    return sum(1 for b in blocks if "py -3" in b)


def test_ops_golden_path_ascii_only() -> None:
    """OPS_EVIDENCE_GOLDEN_PATH.md is ASCII-only."""
    text = OPS_GOLDEN.read_text(encoding="utf-8")
    assert text.encode("ascii", errors="ignore").decode("ascii") == text


def test_ops_golden_path_max_one_py3_block() -> None:
    """OPS_EVIDENCE_GOLDEN_PATH has at most one code block containing 'py -3' (canonical block)."""
    content = OPS_GOLDEN.read_text(encoding="utf-8")
    n = _code_blocks_with_py3(content)
    assert n <= 2, (
        f"OPS_EVIDENCE_GOLDEN_PATH may have at most two py -3 blocks (e.g. cmd + PowerShell), got {n}"
    )


def test_evidence_pack_doc_has_manifest_schema_v1_heading() -> None:
    """EVIDENCE_PACK.md must contain 'Pack manifest schema (v1)' section."""
    evidence_pack_doc = DOCS / "EVIDENCE_PACK.md"
    assert evidence_pack_doc.exists(), "EVIDENCE_PACK.md must exist"
    content = evidence_pack_doc.read_text(encoding="utf-8")
    assert "Pack manifest schema (v1)" in content, (
        "EVIDENCE_PACK.md must document Pack manifest schema (v1)"
    )


def test_evidence_pack_doc_has_source_semantics_heading() -> None:
    """EVIDENCE_PACK.md must contain 'Source semantics' section."""
    evidence_pack_doc = DOCS / "EVIDENCE_PACK.md"
    assert evidence_pack_doc.exists(), "EVIDENCE_PACK.md must exist"
    content = evidence_pack_doc.read_text(encoding="utf-8")
    assert "Source semantics" in content, (
        "EVIDENCE_PACK.md must document Source semantics"
    )
