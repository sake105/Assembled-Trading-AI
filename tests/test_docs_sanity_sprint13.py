"""Docs sanity for Sprint 13: ASCII-only, no duplicate py -3 blocks, single Verification section (stdlib only)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RELEASE_NOTES = DOCS / "RELEASE_NOTES_SPRINT13.md"
MERGE_GATE = DOCS / "MERGE_GATE_SPRINT13.md"
OPS_GOLDEN = DOCS / "OPS_EVIDENCE_GOLDEN_PATH.md"


def _code_blocks_with_py3(content: str) -> int:
    """Count fenced code blocks that contain 'py -3'."""
    count = 0
    in_block = False
    for line in content.splitlines():
        if line.strip().startswith("```"):
            if in_block:
                in_block = False
            else:
                in_block = True
                block_content: list[str] = []
            continue
        if in_block:
            block_content.append(line)
    # Simpler: find all ```...``` blocks and check for py -3
    blocks = re.findall(r"```[\s\S]*?```", content)
    for b in blocks:
        if "py -3" in b:
            count += 1
    return count


def _count_section(content: str, section_title: str) -> int:
    """Count markdown section headers that exactly match section_title (## Title)."""
    pattern = re.escape("## " + section_title)
    return len(re.findall(r"^" + pattern + r"\s*$", content, re.MULTILINE))


def test_release_notes_ascii_only() -> None:
    """RELEASE_NOTES_SPRINT13.md is ASCII-only."""
    text = RELEASE_NOTES.read_text(encoding="utf-8")
    assert text.encode("ascii", errors="ignore").decode("ascii") == text


def test_merge_gate_ascii_only() -> None:
    """MERGE_GATE_SPRINT13.md is ASCII-only."""
    text = MERGE_GATE.read_text(encoding="utf-8")
    assert text.encode("ascii", errors="ignore").decode("ascii") == text


def test_ops_golden_path_ascii_only() -> None:
    """OPS_EVIDENCE_GOLDEN_PATH.md is ASCII-only."""
    text = OPS_GOLDEN.read_text(encoding="utf-8")
    assert text.encode("ascii", errors="ignore").decode("ascii") == text


def test_merge_gate_exactly_one_py3_block() -> None:
    """MERGE_GATE has exactly one code block containing 'py -3' (single primary command)."""
    content = MERGE_GATE.read_text(encoding="utf-8")
    n = _code_blocks_with_py3(content)
    assert n == 1, f"MERGE_GATE should have exactly one py -3 command block, got {n}"


def test_ops_golden_path_max_one_py3_block() -> None:
    """OPS_EVIDENCE_GOLDEN_PATH has at most one code block containing 'py -3' (canonical block)."""
    content = OPS_GOLDEN.read_text(encoding="utf-8")
    n = _code_blocks_with_py3(content)
    assert (
        n <= 2
    ), f"OPS_EVIDENCE_GOLDEN_PATH may have at most two py -3 blocks (e.g. cmd + PowerShell), got {n}"


def test_release_notes_single_verification_windows_section() -> None:
    """RELEASE_NOTES has exactly one 'Verification (Windows)' section."""
    content = RELEASE_NOTES.read_text(encoding="utf-8")
    n = _count_section(content, "Verification (Windows)")
    assert (
        n == 1
    ), f"RELEASE_NOTES should contain exactly one 'Verification (Windows)' section, got {n}"


def test_evidence_pack_doc_has_manifest_schema_v1_heading() -> None:
    """EVIDENCE_PACK.md must contain 'Pack manifest schema (v1)' section."""
    evidence_pack_doc = DOCS / "EVIDENCE_PACK.md"
    assert evidence_pack_doc.exists(), "EVIDENCE_PACK.md must exist"
    content = evidence_pack_doc.read_text(encoding="utf-8")
    assert (
        "Pack manifest schema (v1)" in content
    ), "EVIDENCE_PACK.md must document Pack manifest schema (v1)"


def test_evidence_pack_doc_has_source_semantics_heading() -> None:
    """EVIDENCE_PACK.md must contain 'Source semantics' section."""
    evidence_pack_doc = DOCS / "EVIDENCE_PACK.md"
    assert evidence_pack_doc.exists(), "EVIDENCE_PACK.md must exist"
    content = evidence_pack_doc.read_text(encoding="utf-8")
    assert (
        "Source semantics" in content
    ), "EVIDENCE_PACK.md must document Source semantics"
