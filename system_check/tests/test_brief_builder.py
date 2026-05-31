"""Unit tests for brief_builder.

These tests are deterministic and do not require network or API access.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from system_check.runner.brief_builder import (
    BriefSources,
    brief_hash,
    build_brief,
    load_sources,
)


def _minimal_claude_md() -> str:
    return (
        "# CLAUDE.md\n\n"
        "## Projekt\n\n"
        "Das Projekt soll ein robustes, nachvollziehbares, modular "
        "erweiterbares Trading-System werden.\n\n"
        "Weitere Details hier irrelevant.\n\n"
        "## Plan ≠ Implementierung\n\n"
        "Irrelevant.\n\n"
        "## Sensible Zonen (Tabu ohne Auftrag)\n\n"
        "Diese Pfade sind hart geschützt:\n\n"
        "- `src/assembled_core/execution/`\n"
        "- `src/assembled_core/risk/`\n"
        "- `src/assembled_core/accounting/`\n"
        "- `src/assembled_core/pipeline/`\n"
        "- `src/assembled_core/paper/`\n"
        "- `.github/workflows/`\n\n"
        "## Datenrealismus\n\n"
        "Irrelevant.\n"
    )


def _minimal_architecture_md() -> str:
    return (
        "# Architecture Backend\n\n"
        "## Schichtenlogik, Datenfluss & Backtest-Grundsatz\n\n"
        "### Bevorzugte Schichtenlogik (vormals §5.1)\n\n"
        "Bevorzugte Systemrichtung:\n\n"
        "`data` → `features` → `signals` → `portfolio` → `execution` → "
        "`pipeline`\n\n"
        "Flankierend: `qa`, `reports`.\n"
    )


def _minimal_memory_md() -> str:
    return (
        "# Memory\n\n"
        "| File | Topic | Date |\n"
        "|---|---|---|\n"
        "| [m1.md](./m1.md) | First | 2026-04-01 |\n"
        "| [m2.md](./m2.md) | Second | 2026-04-10 |\n"
        "| [m3.md](./m3.md) | Third | 2026-04-17 |\n"
    )


def _minimal_roadmap_md() -> str:
    return (
        "# ROADMAP_STATE\n\n"
        "## Active\n"
        "- M20: pending\n"
        "- M21: implemented\n"
        "- CRITICAL: .env rotation still open\n"
        "Normal line without keyword.\n"
    )


def test_build_brief_deterministic() -> None:
    src = BriefSources(
        claude_md=_minimal_claude_md(),
        roadmap_state_md=_minimal_roadmap_md(),
        memory_index_md=_minimal_memory_md(),
        recent_git_log="abc123 feat: test\ndef456 fix: bug",
        top_level_dirs=["scripts", "src", "tests"],
    )
    a = build_brief(src)
    b = build_brief(src)
    assert a == b
    assert brief_hash(a) == brief_hash(b)
    assert len(brief_hash(a)) == 16


def test_build_brief_contains_expected_sections() -> None:
    src = BriefSources(
        claude_md=_minimal_claude_md(),
        roadmap_state_md=_minimal_roadmap_md(),
        memory_index_md=_minimal_memory_md(),
        recent_git_log="abc123 feat: test",
        top_level_dirs=["scripts", "src", "tests"],
    )
    brief = build_brief(src)

    for section in (
        "# Project Brief — Assembled-Trading-AI",
        "## Mission",
        "## Architecture Map",
        "## Current State — Recent Completions",
        "## Known Open Items & Risks",
        "## Recent Git Log",
        "## Sensitive Zones",
        "## Review Framing — Two Lenses",
    ):
        assert section in brief, f"missing section: {section}"


def test_build_brief_tolerates_missing_sources() -> None:
    """Empty sources must not crash — every section has a fallback."""
    src = BriefSources()
    brief = build_brief(src)
    assert "Project Brief" in brief
    # Each fallback mentions the missing input.
    assert "unavailable" in brief.lower() or "missing" in brief.lower()


def test_build_brief_pulls_open_items() -> None:
    src = BriefSources(
        roadmap_state_md=(
            "Normal line.\n"
            "- M99: pending investigation\n"
            "- SECURITY: CRITICAL .env rotation\n"
            "Another normal line.\n"
        ),
    )
    brief = build_brief(src)
    assert "pending" in brief
    assert "CRITICAL" in brief


def test_build_brief_extracts_architecture_hint() -> None:
    src = BriefSources(
        claude_md=_minimal_claude_md(),
        architecture_md=_minimal_architecture_md(),
        top_level_dirs=["scripts", "src", "tests", "docs", "system_check"],
    )
    brief = build_brief(src)
    assert "`scripts`" in brief
    assert "`src/assembled_core`" in brief
    # Pipeline direction (migrated to ARCHITECTURE_BACKEND.md) should surface.
    assert "Preferred pipeline direction" in brief
    assert "data" in brief and "execution" in brief


def test_sensitive_zones_lists_all_protected_paths() -> None:
    """The extracted zone section must surface all six hard-protected paths."""
    src = BriefSources(claude_md=_minimal_claude_md())
    brief = build_brief(src)
    zone_section = brief.split("## Sensitive Zones", 1)[1]
    for path in (
        "src/assembled_core/execution/",
        "src/assembled_core/risk/",
        "src/assembled_core/accounting/",
        "src/assembled_core/pipeline/",
        "src/assembled_core/paper/",
        ".github/workflows/",
    ):
        assert path in zone_section, f"missing protected zone: {path}"


def test_sensitive_zones_fallback_complete_when_claude_md_missing() -> None:
    """E-034 guard: the no-CLAUDE.md fallback must list all six zones.

    The prior fallback returned only 2-of-6 paths, hiding four protected
    zones from every tournament agent.
    """
    src = BriefSources()  # no CLAUDE.md at all
    brief = build_brief(src)
    zone_section = brief.split("## Sensitive Zones", 1)[1]
    for path in (
        "src/assembled_core/execution/",
        "src/assembled_core/risk/",
        "src/assembled_core/accounting/",
        "src/assembled_core/pipeline/",
        "src/assembled_core/paper/",
        ".github/workflows/",
    ):
        assert path in zone_section, f"fallback missing protected zone: {path}"


def test_build_brief_token_budget_sanity() -> None:
    """Brief should fit the budget: < ~3000 tokens even on a full repo.

    We use word-count as a cheap proxy (roughly 1.3 tokens per word).
    """
    src = BriefSources(
        claude_md="x " * 5000,
        roadmap_state_md="pending\n" * 200,
        memory_index_md=_minimal_memory_md(),
        recent_git_log="\n".join(f"abc{i} line" for i in range(50)),
        top_level_dirs=[f"dir{i}" for i in range(30)],
    )
    brief = build_brief(src)
    # Even under adversarial input the builder must cap the brief length.
    # Soft assertion: we expect ≲ 4000 words.
    assert len(brief.split()) < 4000


def test_load_sources_runs_on_real_repo(tmp_path: Path) -> None:
    """load_sources must not raise even if the project root lacks files."""
    src = load_sources(project_root=tmp_path)
    assert src.claude_md == ""
    assert src.roadmap_state_md == ""
    # top_level_dirs should be empty or at least a list.
    assert isinstance(src.top_level_dirs, list)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
