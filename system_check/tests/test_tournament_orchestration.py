"""End-to-end orchestration test with a stubbed Claude client.

Verifies that :func:`run_tournament` drives all four rounds, writes a
transcript JSONL in real time, and aggregates totals correctly.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

import pytest

from system_check.runner.claude_client import CallResult, ClaudeClient
from system_check.runner.tournament import run_tournament


PROJECT_CONFIG: dict[str, Any] = {
    "models": {
        "defender": "claude-sonnet-4-6",
        "critic": "claude-haiku-4-5",
        "judge": "claude-sonnet-4-6",
        "router": "claude-haiku-4-5",
    },
    "tokens": {
        "critic_attack_max_tokens": 100,
        "defender_rebuttal_max_tokens": 100,
        "critic_counter_max_tokens": 100,
        "judge_synthesis_max_tokens": 500,
        "router_max_tokens": 50,
    },
    "rounds": {
        "counter_rebuttal_top_n": 3,
        "max_parallel_requests": 4,
        "per_call_timeout_seconds": 10,
    },
    "retry": {
        "max_attempts": 1,
        "initial_backoff_seconds": 0,
        "backoff_multiplier": 1,
        "retry_on_status": [429, 500],
    },
    "judge": {
        "min_findings_full": 10,
        "min_findings_reduced": 3,
        "min_recommendations_full": 8,
        "min_recommendations_reduced": 3,
    },
    "cost_rates_per_mtok": {
        "sonnet_input": 3.0,
        "sonnet_output": 15.0,
        "haiku_input": 0.25,
        "haiku_output": 1.25,
    },
    "safety": {"require_api_key": False, "redact_env_in_logs": True},
}


class _StubClient(ClaudeClient):
    """Returns deterministic content keyed by a tag in the user prompt."""

    def __init__(self) -> None:
        super().__init__()
        self.call_count = 0

    async def call(self, **kwargs: Any) -> CallResult:  # type: ignore[override]
        self.call_count += 1
        up = kwargs.get("user_prompt", "")
        model = kwargs["model"]
        # Router always emits a JSON array.
        if "Select up to 10 critic ids" in up:
            return CallResult(
                content='["C1","C2"]', model=model,
                prompt_tokens=50, completion_tokens=10,
            )
        if "adversarial tournament" in up and "Section A" in up:
            return CallResult(
                content=_sample_judge_output(), model=model,
                prompt_tokens=500, completion_tokens=200,
            )
        if "You are acting as" in up and "CHALLENGE" in up:
            return CallResult(
                content="ATTACK: x\nCHALLENGE_1: a\nCHALLENGE_2: b\n"
                        "FORWARD_PROPOSAL: medium | t | d",
                model=model, prompt_tokens=100, completion_tokens=50,
            )
        if "REPLY to" in up or "Respond using this exact structure" in up:
            return CallResult(
                content="CLUSTER_SUMMARY: ok\n\n### REPLY to C1\nVERDICT: "
                        "partial_concession\nARGUMENT: see tests\n"
                        "IF_CONCEDED_PRIORITY: Q2\nFORWARD_NOTE: nil",
                model=model, prompt_tokens=200, completion_tokens=100,
            )
        # Counter-rebuttal
        return CallResult(
            content="COUNTER: partially unanswered\nSTILL_OPEN: partial\n"
                    "FORWARD_INSIST: yes",
            model=model, prompt_tokens=100, completion_tokens=50,
        )


def _sample_judge_output() -> str:
    finding_block = json.dumps({
        "id": "F1",
        "title": "Test finding",
        "severity": "high",
        "category": "testing",
        "description": "something is weak",
        "proposed_mitigation": "add coverage",
        "affected_modules": ["system_check"],
        "evidence_from_transcript": "see C1 attack",
    })
    rec_block = json.dumps({
        "id": "R1",
        "title": "Ship property tests",
        "timeframe": "medium",
        "impact": "high",
        "effort": "low",
        "affected_modules": ["tests/"],
        "rationale": "catch edge cases",
        "main_risk": "flake",
    })
    return (
        "## Section A — Current Weaknesses & Findings\n\n"
        f"```json\n{finding_block}\n```\n\n"
        "### Defender Scoreboard\n\n"
        "| Defender | Score | Reasoning |\n"
        "|---|---|---|\n"
        "| D1 | 4 | strong arguments |\n"
        "| D2 | 3 | mixed |\n\n"
        "### Blindspot list\n"
        "- C3 did not receive a direct response\n\n"
        "## Section B — Strategic Improvement & Expansion Recommendations\n\n"
        f"```json\n{rec_block}\n```\n\n"
        "### Convergence cluster\n"
        "- Test hygiene recurred across 5 critics\n\n"
        "### Dissonance cluster\n"
        "- Priority of secret rotation disputed\n"
    )


@pytest.fixture
def personas_dir(tmp_path: Path) -> Path:
    (tmp_path / "defenders.yaml").write_text(
        yaml.safe_dump({"defenders": [
            {"id": f"D{i}", "name": f"Def{i}", "model_key": "defender",
             "cluster_affinity": ["testing"],
             "system_prompt": f"you are defender {i}"}
            for i in range(1, 4)
        ]}),
        encoding="utf-8",
    )
    (tmp_path / "critics.yaml").write_text(
        yaml.safe_dump({"critics": [
            {"id": f"C{i}", "name": f"Crit{i}", "model_key": "critic",
             "cluster": "testing",
             "system_prompt": f"you are critic {i}"}
            for i in range(1, 5)
        ]}),
        encoding="utf-8",
    )
    return tmp_path


def test_tournament_runs_four_rounds(tmp_path: Path, personas_dir: Path) -> None:
    run_dir = tmp_path / "run"
    client = _StubClient()

    result = asyncio.run(run_tournament(
        project_root=tmp_path,
        run_dir=run_dir,
        config=PROJECT_CONFIG,
        defenders_path=personas_dir / "defenders.yaml",
        critics_path=personas_dir / "critics.yaml",
        client=client,
        full_scale=False,
    ))

    assert run_dir.exists()
    # Brief, transcript must be on disk.
    assert (run_dir / "brief.md").exists()
    transcript = (run_dir / "transcript.jsonl").read_text(encoding="utf-8")
    assert transcript.strip(), "transcript should have turns"

    # Round coverage: at least critics + defenders + judge + router.
    rounds_seen = {t.round for t in result.turns}
    assert {1, 2, 4}.issubset(rounds_seen)

    # Judge content is non-empty.
    assert result.judge_content
    assert result.total_input_tokens > 0
    assert result.total_output_tokens > 0
    assert result.cost_estimate_usd > 0


def test_tournament_truncates_personas(tmp_path: Path, personas_dir: Path) -> None:
    run_dir = tmp_path / "run"
    client = _StubClient()
    result = asyncio.run(run_tournament(
        project_root=tmp_path,
        run_dir=run_dir,
        config=PROJECT_CONFIG,
        defenders_path=personas_dir / "defenders.yaml",
        critics_path=personas_dir / "critics.yaml",
        client=client,
        max_defenders=2,
        max_critics=2,
        full_scale=False,
    ))
    r1_critics = [t for t in result.turns if t.round == 1 and t.role == "critic"]
    r2_defenders = [t for t in result.turns if t.round == 2 and t.role == "defender"]
    assert len(r1_critics) == 2
    assert len(r2_defenders) == 2
