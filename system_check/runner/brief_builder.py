"""Builds a compressed project brief (<=2500 tokens) for all tournament agents.

The brief is a single Markdown document that each agent sees verbatim so the
discussion is reproducible. The content is deterministic for a fixed source
state — a snapshot hash is included in the run manifest.

Design goals:
* Do not ship the full CLAUDE.md / ROADMAP_STATE.md (~7k words) to every
  agent; instead distil the operative truth into ~1.5-2k words.
* Extract only sections actually useful for reviewing the project
  (mission, architecture map, current open items, recent commits, sensitive
  zones, last known test headline).
* Be robust to missing source files — every section has a fallback so the
  tournament can still run on a partial checkout.
"""

from __future__ import annotations

import hashlib
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Sources
# -------------------------------------------------------------------------

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Relative to project root
CLAUDE_MD = Path("CLAUDE.md")
ROADMAP_STATE_MD = Path("docs/roadmap/ROADMAP_STATE.md")
# Memory lives under the user's profile, not the repo — best-effort only.
MEMORY_INDEX_REL = Path("memory/MEMORY.md")


@dataclass
class BriefSources:
    """Container for the raw inputs fed to :func:`build_brief`.

    All fields default to empty strings so the builder can run on a partial
    set; every downstream section is individually resilient.
    """

    claude_md: str = ""
    roadmap_state_md: str = ""
    memory_index_md: str = ""
    recent_git_log: str = ""
    top_level_dirs: list[str] = field(default_factory=list)


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


def load_sources(project_root: Path | None = None) -> BriefSources:
    """Read brief sources from the repository on disk.

    Safe when files are missing — returns empty strings for unavailable
    sections.
    """
    root = project_root or DEFAULT_PROJECT_ROOT
    return BriefSources(
        claude_md=_read_text(root / CLAUDE_MD),
        roadmap_state_md=_read_text(root / ROADMAP_STATE_MD),
        memory_index_md=_read_text(root / MEMORY_INDEX_REL),
        recent_git_log=_read_git_log(root, n=10),
        top_level_dirs=_list_top_level_dirs(root),
    )


def build_brief(sources: BriefSources) -> str:
    """Assemble the Markdown project brief.

    Output is deterministic: identical ``BriefSources`` ⇒ identical brief.
    """
    sections: list[str] = []
    sections.append("# Project Brief — Assembled-Trading-AI")
    sections.append(
        "This brief is the shared context for every tournament agent. "
        "It is a compressed snapshot of the project at review time."
    )

    sections.append("\n## Mission")
    sections.append(_extract_mission(sources.claude_md))

    sections.append("\n## Architecture Map")
    sections.append(
        _extract_architecture_map(sources.claude_md, sources.top_level_dirs)
    )

    sections.append("\n## Current State — Recent Completions")
    sections.append(_extract_recent_memory(sources.memory_index_md))

    sections.append("\n## Known Open Items & Risks")
    sections.append(_extract_open_items(sources.roadmap_state_md))

    sections.append("\n## Recent Git Log (last 10 commits)")
    sections.append(_format_git_log(sources.recent_git_log))

    sections.append("\n## Sensitive Zones")
    sections.append(_extract_sensitive_zones(sources.claude_md))

    sections.append("\n## Review Framing — Two Lenses")
    sections.append(
        "Every agent must work with BOTH lenses:\n\n"
        "1. **Retrospective**: What is weak, incorrect, or under-tested today?\n"
        "2. **Forward**: What would meaningfully strengthen the system in the "
        "next 6–18 months? Concrete feature, architecture, strategy or data "
        "proposals — not generic advice."
    )

    return "\n".join(sections).rstrip() + "\n"


def brief_hash(brief: str) -> str:
    """Stable 16-char sha256 hex prefix of the brief content."""
    return hashlib.sha256(brief.encode("utf-8")).hexdigest()[:16]


# -------------------------------------------------------------------------
# Section extractors
# -------------------------------------------------------------------------


def _extract_mission(claude_md: str) -> str:
    """Pull a 2-3 sentence mission statement from CLAUDE.md §1.3."""
    if not claude_md:
        return "_Mission statement unavailable — CLAUDE.md missing._"

    m = re.search(
        r"### 1\.3 Kernmission.*?\n(.*?)(?=\n###|\Z)", claude_md, re.DOTALL
    )
    if not m:
        return (
            "Modular Python backend for research, backtests, paper/simulation "
            "and risk-steered trading, evolving into an institutional-grade "
            "quantitative system."
        )
    body = m.group(1)
    # Compress to first meaningful paragraph, max ~400 chars.
    para = _first_meaningful_paragraph(body)
    if len(para) > 400:
        para = para[:380].rsplit(" ", 1)[0] + " …"
    return para


def _extract_architecture_map(
    claude_md: str, top_level_dirs: list[str]
) -> str:
    """Render a compact module map.

    Uses CLAUDE.md §5 (Architekturübersicht) when available, enriched with the
    detected top-level directories. Falls back to directory listing alone.
    """
    module_roles: dict[str, str] = {
        "src/assembled_core": "Core trading backend (signals, portfolio, execution, risk, accounting)",
        "scripts": "CLI entry points / runners (run_daily, run_backtest, run_api, ...)",
        "tests": "Unit + regression + phase tests (phase10/11/12/13 markers)",
        "config": "YAML policy, factor bundles, cost tiers, regime overlays",
        "configs": "Legacy / environment-specific config variants",
        "docs": "Roadmap, runbooks, governance, audit reports",
        "memory": "Memory index (project-level claude-mem)",
        ".github/workflows": "CI pipelines (Ubuntu + Windows)",
        "data": "Raw / processed market and intel data (gitignored)",
        "output": "Evidence packs, ledgers, accounting reports (gitignored)",
        "experiments": "Research notebooks and ad-hoc runs",
        "notebooks": "Jupyter analysis, prototype signals",
        "system_check": "This meta-review tool (adversarial agent tournament)",
    }

    lines: list[str] = []
    detected: set[str] = set(top_level_dirs)

    # Known module pointers first (deterministic order).
    for key, role in module_roles.items():
        first = key.split("/")[0]
        if first in detected or Path(first).exists():
            lines.append(f"- `{key}` — {role}")

    # Note any other top-level dirs we don't have an annotation for.
    unannotated = sorted(
        d
        for d in detected
        if d not in {k.split("/")[0] for k in module_roles}
        and not d.startswith(".")
    )
    if unannotated:
        lines.append(
            "- Other top-level folders: "
            + ", ".join(f"`{d}`" for d in unannotated)
        )

    # Pipeline direction from CLAUDE.md §5.1 if present.
    m = re.search(r"5\.1 Bevorzugte Schichtenlogik.*?\n(.*?)(?=\n###|\Z)",
                  claude_md, re.DOTALL)
    if m:
        layers = re.findall(r"[*-]\s+`?(\w+)`?", m.group(1))
        if layers:
            lines.append(
                "\n**Preferred pipeline direction**: "
                + " → ".join(layers[:8])
            )

    return "\n".join(lines) if lines else "_No architecture map detectable._"


def _extract_recent_memory(memory_md: str) -> str:
    """Extract the last N rows of MEMORY.md's index table."""
    if not memory_md:
        return "_Memory index unavailable._"

    # Find table rows that look like `| [title](file.md) | topic | date |`
    rows = re.findall(
        r"^\|\s*\[[^\]]+\]\([^)]+\)\s*\|[^|]+\|[^|]+\|\s*$",
        memory_md,
        re.MULTILINE,
    )
    if not rows:
        return "_No memory entries found in expected table format._"

    # Keep last 6 by appearance order (MEMORY.md appends newest at bottom).
    tail = rows[-6:]
    header = "| Entry | Topic | Date |\n|---|---|---|\n"
    return header + "\n".join(tail)


def _extract_open_items(roadmap_md: str) -> str:
    """Pull bullet / table lines hinting at open work."""
    if not roadmap_md:
        return "_ROADMAP_STATE.md unavailable._"

    keywords = (
        "pending", "offen", "CRITICAL", "open", "TODO", "not yet",
        "not implemented", "skip", "defer",
    )
    collected: list[str] = []
    for line in roadmap_md.splitlines():
        lower = line.lower()
        if any(k.lower() in lower for k in keywords) and len(line.strip()) > 5:
            if len(collected) < 20:
                collected.append(line.rstrip())

    if not collected:
        return "_No explicit open items detected — treat with healthy skepticism._"
    return "\n".join(collected)


def _format_git_log(git_log: str) -> str:
    if not git_log:
        return "_Git log unavailable._"
    return "```\n" + git_log.strip() + "\n```"


def _extract_sensitive_zones(claude_md: str) -> str:
    if not claude_md:
        return (
            "- `src/assembled_core/execution/*`\n"
            "- `src/assembled_core/risk/*`\n"
            "- `src/assembled_core/pipeline/*`\n"
            "- `src/assembled_core/accounting/*`\n"
            "- `.github/workflows/*`"
        )
    m = re.search(
        r"6\.1 Besonders sensible Kernbereiche.*?\n(.*?)(?=\n###|\n##|\Z)",
        claude_md,
        re.DOTALL,
    )
    if not m:
        return "- `src/assembled_core/execution/*`\n- `src/assembled_core/risk/*`"
    body = m.group(1).strip()
    # Keep only list lines, max 12.
    lines = [line for line in body.splitlines() if line.strip().startswith(("*", "-"))]
    return "\n".join(lines[:12]) or body[:400]


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError) as exc:
        logger.debug("brief: source unavailable %s (%s)", path, exc)
        return ""


def _read_git_log(root: Path, n: int = 10) -> str:
    """Return `git log -n --oneline` for the given root, or empty on failure."""
    try:
        res = subprocess.run(
            ["git", "-C", str(root), "log", f"-{n}", "--oneline"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if res.returncode == 0:
            return res.stdout
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.debug("brief: git log unavailable (%s)", exc)
    return ""


def _list_top_level_dirs(root: Path) -> list[str]:
    try:
        return sorted(
            p.name
            for p in root.iterdir()
            if p.is_dir() and not p.name.startswith((".git",))
        )
    except OSError:
        return []


def _first_meaningful_paragraph(text: str) -> str:
    for block in re.split(r"\n\s*\n", text.strip()):
        stripped = block.strip()
        if len(stripped) > 30 and not stripped.startswith(("#", "---", "|")):
            return re.sub(r"\s+", " ", stripped)
    return text.strip()[:400]
