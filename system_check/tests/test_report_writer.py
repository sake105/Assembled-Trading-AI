"""Tests for judge parsing + report writer."""

from __future__ import annotations

import json
from pathlib import Path

from system_check.runner.judge import parse_judge_output
from system_check.runner.report import write_run_artifacts
from system_check.runner.tournament import TournamentResult, Turn


SAMPLE_JUDGE_MD = """## Section A — Current Weaknesses & Findings

```json
{
  "id": "F1",
  "title": "Overfitting risk",
  "severity": "high",
  "category": "quant_research",
  "description": "Factor weights may be overfit.",
  "proposed_mitigation": "Introduce walk-forward validation",
  "affected_modules": ["signals", "features"],
  "evidence_from_transcript": "See C5 attack"
}
```

```json
{
  "id": "F2",
  "title": "Missing secret rotation",
  "severity": "critical",
  "category": "security",
  "description": ".env history contains keys",
  "proposed_mitigation": "Rotate keys + history rewrite",
  "affected_modules": [".env"],
  "evidence_from_transcript": "C18"
}
```

### Defender Scoreboard

| Defender | Score | Reasoning |
|---|---|---|
| D1 | 4 | strong rebuttal |
| D2 | 3 | honest on gaps |
| D3 | 5 | excellent evidence |

### Blindspots
- C14 received no direct answer
- C24 was ignored

## Section B — Strategic Improvement & Expansion Recommendations

```json
{
  "id": "R1",
  "title": "Walk-forward validation harness",
  "timeframe": "medium",
  "impact": "high",
  "effort": "medium",
  "affected_modules": ["qa", "signals"],
  "rationale": "Reduces overfitting risk",
  "main_risk": "Longer CI time"
}
```

```json
{
  "id": "R2",
  "title": "Pre-commit secret scanner",
  "timeframe": "quick-win",
  "impact": "high",
  "effort": "low",
  "affected_modules": [".github/workflows"],
  "rationale": "Prevents future leaks",
  "main_risk": "False positives"
}
```

### Convergence cluster
- Multiple critics flagged test hygiene
- Factor decay raised by 3 critics

### Dissonance cluster
- Priority of borrow-cost modelling disputed
"""


def test_parse_judge_output_findings() -> None:
    j = parse_judge_output(SAMPLE_JUDGE_MD)
    assert len(j.findings) == 2
    assert j.findings[0].id == "F1"
    assert j.findings[0].severity == "high"
    assert j.findings[1].severity == "critical"
    assert "features" in j.findings[0].affected_modules


def test_parse_judge_output_recommendations() -> None:
    j = parse_judge_output(SAMPLE_JUDGE_MD)
    assert len(j.recommendations) == 2
    assert {r.id for r in j.recommendations} == {"R1", "R2"}
    assert j.recommendations[0].timeframe == "medium"
    assert j.recommendations[1].timeframe == "quick-win"


def test_parse_judge_output_scoreboard_and_clusters() -> None:
    j = parse_judge_output(SAMPLE_JUDGE_MD)
    assert len(j.defender_scores) == 3
    assert j.defender_scores[0].defender_id == "D1"
    assert j.defender_scores[0].score == 4
    assert any("test hygiene" in c.lower() for c in j.convergence)
    assert any("borrow" in c.lower() for c in j.dissonance)
    assert any("C14" in b for b in j.blindspots)


def test_parse_judge_output_tolerates_malformed() -> None:
    bad = "## Section A\n\n```json\n{not valid json\n```\n## Section B\n"
    j = parse_judge_output(bad)
    assert j.findings == []
    assert j.recommendations == []


def _make_result(tmp_path: Path, judge_md: str) -> TournamentResult:
    run_dir = tmp_path / "run1"
    run_dir.mkdir(parents=True)
    turns = [
        Turn(round=1, agent_id="C1", agent_name="Critic",
             role="critic", model="haiku", prompt_tokens=100,
             completion_tokens=50, content="ATTACK: ..."),
        Turn(round=2, agent_id="D1", agent_name="Defender",
             role="defender", model="sonnet", prompt_tokens=300,
             completion_tokens=200, content="REPLY..."),
        Turn(round=4, agent_id="JUDGE", agent_name="Judge",
             role="judge", model="sonnet", prompt_tokens=500,
             completion_tokens=400, content=judge_md),
    ]
    return TournamentResult(
        run_id="run1",
        run_dir=run_dir,
        brief="# brief\n",
        brief_hash="deadbeefdeadbeef",
        turns=turns,
        started_at_utc="2026-04-18T00:00:00+00:00",
        finished_at_utc="2026-04-18T00:30:00+00:00",
        config_snapshot={"models": {"critic": "haiku"}},
        judge_content=judge_md,
        total_input_tokens=900,
        total_output_tokens=650,
        cost_estimate_usd=0.0089,
    )


def test_write_run_artifacts_creates_all_files(tmp_path: Path) -> None:
    result = _make_result(tmp_path, SAMPLE_JUDGE_MD)
    out = write_run_artifacts(result)
    for key in (
        "report", "recommendations", "scoreboard", "config_snapshot", "manifest",
    ):
        assert out[key].exists(), key

    manifest = json.loads((tmp_path / "run1" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["findings"] == 2
    assert manifest["recommendations"] == 2
    assert manifest["brief_hash"] == "deadbeefdeadbeef"
    assert manifest["total_input_tokens"] == 900

    recs = json.loads((tmp_path / "run1" / "recommendations.json").read_text(encoding="utf-8"))
    assert len(recs) == 2
    assert recs[0]["id"] in {"R1", "R2"}

    scoreboard = json.loads((tmp_path / "run1" / "scoreboard.json").read_text(encoding="utf-8"))
    assert len(scoreboard["defender_scores"]) == 3
    assert scoreboard["blindspots"]
    assert scoreboard["convergence"]
    assert scoreboard["dissonance"]

    report_md = (tmp_path / "run1" / "report.md").read_text(encoding="utf-8")
    assert "Section A" in report_md
    assert "Section B" in report_md
    assert "Overfitting" in report_md
    assert "Walk-forward" in report_md
