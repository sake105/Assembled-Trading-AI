"""Report writer — turns a :class:`TournamentResult` into durable artefacts.

Outputs, all under ``run_dir``:

* ``report.md``             — human-readable two-section report
* ``recommendations.json``  — machine-readable expansion proposals
* ``scoreboard.json``       — defender scores + attack metadata
* ``config_snapshot.yaml``  — YAML snapshot of the resolved config
* ``manifest.json``         — run metadata (git_sha, tokens, cost, hashes)
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from system_check.runner.judge import JudgeOutput, parse_judge_output
from system_check.runner.tournament import TournamentResult

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


def write_run_artifacts(
    result: TournamentResult,
    *,
    judge_output: JudgeOutput | None = None,
) -> dict[str, Path]:
    """Materialise all per-run files. Returns a dict of {name: path}."""
    run_dir = result.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    judge_output = judge_output or parse_judge_output(result.judge_content)

    report_path = run_dir / "report.md"
    report_path.write_text(_render_report_md(result, judge_output), encoding="utf-8")

    recs_path = run_dir / "recommendations.json"
    recs_path.write_text(
        json.dumps(
            [r.as_dict() for r in judge_output.recommendations],
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    scoreboard_path = run_dir / "scoreboard.json"
    scoreboard_path.write_text(
        json.dumps(
            {
                "defender_scores": [s.as_dict() for s in judge_output.defender_scores],
                "blindspots": judge_output.blindspots,
                "convergence": judge_output.convergence,
                "dissonance": judge_output.dissonance,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    config_path = run_dir / "config_snapshot.yaml"
    config_path.write_text(
        yaml.safe_dump(result.config_snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    manifest = {
        "run_id": result.run_id,
        "started_at_utc": result.started_at_utc,
        "finished_at_utc": result.finished_at_utc,
        "git_sha": _git_sha(),
        "brief_hash": result.brief_hash,
        "total_input_tokens": result.total_input_tokens,
        "total_output_tokens": result.total_output_tokens,
        "cost_estimate_usd": result.cost_estimate_usd,
        "turns": len(result.turns),
        "findings": len(judge_output.findings),
        "recommendations": len(judge_output.recommendations),
        "errors": result.errors,
        "artifacts": {
            "brief": "brief.md",
            "transcript": "transcript.jsonl",
            "report": "report.md",
            "recommendations": "recommendations.json",
            "scoreboard": "scoreboard.json",
            "config_snapshot": "config_snapshot.yaml",
        },
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    return {
        "brief": run_dir / "brief.md",
        "transcript": run_dir / "transcript.jsonl",
        "report": report_path,
        "recommendations": recs_path,
        "scoreboard": scoreboard_path,
        "config_snapshot": config_path,
        "manifest": manifest_path,
    }


# -------------------------------------------------------------------------
# Markdown rendering
# -------------------------------------------------------------------------


def _render_report_md(result: TournamentResult, j: JudgeOutput) -> str:
    parts: list[str] = []
    parts.append(f"# System-Check Report — {result.run_id}")
    parts.append(
        f"**Run finished**: `{result.finished_at_utc}`  \n"
        f"**Brief hash**: `{result.brief_hash}`  \n"
        f"**Turns**: {len(result.turns)}  \n"
        f"**Tokens**: {result.total_input_tokens} in / "
        f"{result.total_output_tokens} out  \n"
        f"**Est. cost (USD)**: ${result.cost_estimate_usd:.4f}"
    )
    if result.errors:
        parts.append("\n> ⚠ Errors were recorded during the run:")
        for e in result.errors:
            parts.append(f"> - {e}")

    parts.append("\n## Section A — Current Weaknesses & Findings\n")
    if j.findings:
        parts.append(_render_findings_md(j))
    else:
        parts.append(
            "_Judge did not emit a machine-readable findings block._\n\n"
            "Raw section A is preserved below for manual review.\n\n"
            f"<details><summary>Raw Section A</summary>\n\n{j.section_a_md}\n\n</details>"
        )
    if j.defender_scores:
        parts.append("\n### Defender Scoreboard\n")
        parts.append("| Defender | Score | Reasoning |")
        parts.append("|---|---|---|")
        for s in j.defender_scores:
            parts.append(f"| {s.defender_id} | {s.score} | {s.reasoning} |")
    if j.blindspots:
        parts.append("\n### Blindspots (unanswered attacks)\n")
        for b in j.blindspots:
            parts.append(f"- {b}")

    parts.append("\n## Section B — Strategic Improvement & Expansion\n")
    if j.recommendations:
        parts.append(_render_recommendations_md(j))
    else:
        parts.append(
            "_Judge did not emit a machine-readable recommendations block._\n\n"
            f"<details><summary>Raw Section B</summary>\n\n{j.section_b_md}\n\n</details>"
        )
    if j.convergence:
        parts.append("\n### Convergence Cluster\n")
        for c in j.convergence:
            parts.append(f"- {c}")
    if j.dissonance:
        parts.append("\n### Dissonance Cluster\n")
        for c in j.dissonance:
            parts.append(f"- {c}")

    parts.append("\n---\n")
    parts.append(
        "_Generated by the System-Check Tournament. "
        "Raw transcript in `transcript.jsonl`, machine-readable "
        "recommendations in `recommendations.json`._"
    )
    return "\n\n".join(parts) + "\n"


def _render_findings_md(j: JudgeOutput) -> str:
    lines: list[str] = []
    for f in j.findings:
        modules = ", ".join(f.affected_modules) if f.affected_modules else "—"
        lines.append(
            f"### [{f.severity.upper()}] {f.title}  \n"
            f"- **ID**: {f.id}  \n"
            f"- **Category**: {f.category}  \n"
            f"- **Affected modules**: {modules}  \n"
            f"- **Description**: {f.description}  \n"
            f"- **Proposed mitigation**: {f.proposed_mitigation or '—'}  \n"
            f"- **Evidence**: {f.evidence_from_transcript or '—'}\n"
        )
    return "\n".join(lines)


def _render_recommendations_md(j: JudgeOutput) -> str:
    # Priority matrix first, then detail cards.
    priority_rows = ["| Title | Impact | Effort | Timeframe |",
                     "|---|---|---|---|"]
    for r in j.recommendations:
        priority_rows.append(
            f"| {r.title} | {r.impact} | {r.effort} | {r.timeframe} |"
        )

    detail: list[str] = []
    for r in j.recommendations:
        modules = ", ".join(r.affected_modules) if r.affected_modules else "—"
        detail.append(
            f"### [{r.timeframe.upper()}] {r.title}  \n"
            f"- **ID**: {r.id}  \n"
            f"- **Impact**: {r.impact} · **Effort**: {r.effort}  \n"
            f"- **Affected modules**: {modules}  \n"
            f"- **Rationale**: {r.rationale or '—'}  \n"
            f"- **Main risk**: {r.main_risk or '—'}\n"
        )

    return (
        "\n".join(priority_rows)
        + "\n\n"
        + "\n".join(detail)
    )


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _git_sha(root: Path | None = None) -> str:
    try:
        args = ["git", "rev-parse", "HEAD"]
        if root:
            args = ["git", "-C", str(root)] + args[1:]
        res = subprocess.run(args, capture_output=True, text=True, timeout=5, check=False)
        if res.returncode == 0:
            return res.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        logger.debug("report: git_sha unavailable (%s)", exc)
    return "unknown"


# Ensure tournament's asdict() use is available when the writer is imported
# independently of the orchestrator.
_ = asdict
