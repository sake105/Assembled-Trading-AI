"""Parser for the judge's synthesis output.

The judge is instructed (in :mod:`tournament`) to emit two Markdown sections
with fenced JSON blocks for findings and recommendations. This module
extracts those blocks into structured objects so :mod:`report` can build a
clean ``recommendations.json`` / ``scoreboard.json`` / ``report.md``.

The parser is intentionally tolerant: if the judge produces slightly
off-format output, we degrade gracefully (drop the malformed entry) rather
than raise.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Data containers
# -------------------------------------------------------------------------


@dataclass
class Finding:
    id: str
    title: str
    severity: str
    category: str
    description: str
    proposed_mitigation: str = ""
    affected_modules: list[str] = field(default_factory=list)
    evidence_from_transcript: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "severity": self.severity,
            "category": self.category,
            "description": self.description,
            "proposed_mitigation": self.proposed_mitigation,
            "affected_modules": self.affected_modules,
            "evidence_from_transcript": self.evidence_from_transcript,
        }


@dataclass
class Recommendation:
    id: str
    title: str
    timeframe: str        # quick-win | medium | strategic
    impact: str           # high | medium | low
    effort: str           # low | medium | high
    affected_modules: list[str] = field(default_factory=list)
    rationale: str = ""
    main_risk: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "timeframe": self.timeframe,
            "impact": self.impact,
            "effort": self.effort,
            "affected_modules": self.affected_modules,
            "rationale": self.rationale,
            "main_risk": self.main_risk,
        }


@dataclass
class DefenderScore:
    defender_id: str
    score: int
    reasoning: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "defender_id": self.defender_id,
            "score": self.score,
            "reasoning": self.reasoning,
        }


@dataclass
class JudgeOutput:
    raw: str
    section_a_md: str
    section_b_md: str
    findings: list[Finding]
    recommendations: list[Recommendation]
    defender_scores: list[DefenderScore]
    blindspots: list[str]
    convergence: list[str]
    dissonance: list[str]


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


def parse_judge_output(markdown: str) -> JudgeOutput:
    """Parse the judge Markdown into structured objects."""
    if not markdown:
        return _empty_output(markdown)

    section_a, section_b = _split_sections(markdown)

    findings = _parse_json_blocks(section_a, Finding, required=("id", "title", "severity", "category", "description"))
    recommendations = _parse_json_blocks(section_b, Recommendation, required=("id", "title", "timeframe", "impact", "effort"))

    defender_scores = _parse_defender_scoreboard(section_a)
    blindspots = _parse_bullet_list(section_a, heading="Blindspot")
    convergence = _parse_bullet_list(section_b, heading="Convergence")
    dissonance = _parse_bullet_list(section_b, heading="Dissonance")

    return JudgeOutput(
        raw=markdown,
        section_a_md=section_a,
        section_b_md=section_b,
        findings=findings,
        recommendations=recommendations,
        defender_scores=defender_scores,
        blindspots=blindspots,
        convergence=convergence,
        dissonance=dissonance,
    )


# -------------------------------------------------------------------------
# Section split
# -------------------------------------------------------------------------


_SECTION_A_PATTERN = re.compile(
    r"(?:^|\n)##\s+Section\s+A[^\n]*\n(.*?)(?=\n##\s+Section\s+B|\Z)",
    re.DOTALL | re.IGNORECASE,
)
_SECTION_B_PATTERN = re.compile(
    r"(?:^|\n)##\s+Section\s+B[^\n]*\n(.*)",
    re.DOTALL | re.IGNORECASE,
)


def _split_sections(md: str) -> tuple[str, str]:
    a = _SECTION_A_PATTERN.search(md)
    b = _SECTION_B_PATTERN.search(md)
    section_a = a.group(1).strip() if a else ""
    section_b = b.group(1).strip() if b else ""
    # If the judge omitted the heading but the document looks coherent, treat
    # everything as section A (section B will be empty, triggering a warning
    # downstream).
    if not section_a and not section_b:
        section_a = md.strip()
    return section_a, section_b


# -------------------------------------------------------------------------
# JSON block parsing
# -------------------------------------------------------------------------


_FENCED_JSON_PATTERN = re.compile(
    r"```(?:json)?\s*\n(.*?)\n```",
    re.DOTALL,
)


def _parse_json_blocks(
    section: str,
    cls: Any,
    *,
    required: tuple[str, ...],
) -> list[Any]:
    """Extract fenced JSON blocks and coerce them to *cls*."""
    items: list[Any] = []
    for m in _FENCED_JSON_PATTERN.finditer(section):
        raw = m.group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("judge: discarded malformed JSON block (%d chars)", len(raw))
            continue
        if isinstance(payload, list):
            for obj in payload:
                parsed = _coerce(obj, cls, required)
                if parsed is not None:
                    items.append(parsed)
        elif isinstance(payload, dict):
            parsed = _coerce(payload, cls, required)
            if parsed is not None:
                items.append(parsed)
    return items


def _coerce(
    obj: dict[str, Any], cls: Any, required: tuple[str, ...],
) -> Any | None:
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in required):
        return None
    try:
        kwargs = {k: obj.get(k) for k in _fields_for(cls) if k in obj}
        if "affected_modules" in kwargs and isinstance(kwargs["affected_modules"], str):
            kwargs["affected_modules"] = [kwargs["affected_modules"]]
        return cls(**kwargs)
    except (TypeError, ValueError) as exc:
        logger.debug("judge: coerce failed for %s: %s", cls.__name__, exc)
        return None


def _fields_for(cls: Any) -> list[str]:
    return list(cls.__dataclass_fields__.keys())


# -------------------------------------------------------------------------
# Defender scoreboard + bullet lists
# -------------------------------------------------------------------------


_SCOREBOARD_ROW_PATTERN = re.compile(
    r"^\|\s*(D\d+)\s*\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*$",
    re.MULTILINE,
)


def _parse_defender_scoreboard(section: str) -> list[DefenderScore]:
    scores: list[DefenderScore] = []
    for m in _SCOREBOARD_ROW_PATTERN.finditer(section):
        try:
            scores.append(DefenderScore(
                defender_id=m.group(1).strip(),
                score=int(m.group(2)),
                reasoning=m.group(3).strip(),
            ))
        except ValueError:
            continue
    return scores


def _parse_bullet_list(section: str, *, heading: str) -> list[str]:
    """Extract bullets under a heading or label containing *heading*."""
    # Accept any line that looks like `- ...` or `* ...` under a line that
    # mentions the heading word.
    pattern = re.compile(
        rf"(?i){heading}[^\n]*\n((?:\s*[-*]\s+[^\n]+\n?)+)"
    )
    m = pattern.search(section)
    if not m:
        return []
    items = re.findall(r"^\s*[-*]\s+(.+)$", m.group(1), re.MULTILINE)
    return [i.strip() for i in items if i.strip()]


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _empty_output(markdown: str) -> JudgeOutput:
    return JudgeOutput(
        raw=markdown,
        section_a_md="",
        section_b_md="",
        findings=[],
        recommendations=[],
        defender_scores=[],
        blindspots=[],
        convergence=[],
        dissonance=[],
    )
