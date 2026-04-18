"""Four-round adversarial tournament orchestrator.

Public entry point: :func:`run_tournament`.

Round overview:

1. Critic opening attacks      (N critics, parallel)
2. Defender rebuttals          (M defenders, parallel; each sees all attacks)
3. Critic counter-rebuttals    (top-k unresolved critics, parallel)
4. Judge synthesis             (single call)

The orchestrator streams every agent turn into a JSONL transcript as it
happens so a partial run is still useful if something aborts mid-way.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from system_check.runner.brief_builder import brief_hash, build_brief, load_sources
from system_check.runner.claude_client import (
    CallResult,
    ClaudeClient,
    ClaudeClientConfig,
    RetryConfig,
)

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# Types
# -------------------------------------------------------------------------


@dataclass
class Persona:
    id: str
    name: str
    model_key: str
    system_prompt: str
    cluster: str | None = None
    cluster_affinity: list[str] = field(default_factory=list)


@dataclass
class Turn:
    """One agent turn — recorded in the transcript."""

    round: int
    agent_id: str
    agent_name: str
    role: str              # "critic" | "defender" | "judge" | "router"
    model: str
    prompt_tokens: int
    completion_tokens: int
    content: str
    attempts: int = 1
    error: str | None = None

    def as_jsonl(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class TournamentResult:
    run_id: str
    run_dir: Path
    brief: str
    brief_hash: str
    turns: list[Turn]
    started_at_utc: str
    finished_at_utc: str
    config_snapshot: dict[str, Any]
    judge_content: str = ""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    cost_estimate_usd: float = 0.0
    errors: list[str] = field(default_factory=list)


# -------------------------------------------------------------------------
# Persona loading
# -------------------------------------------------------------------------


def load_personas(
    defenders_path: Path,
    critics_path: Path,
    *,
    max_defenders: int | None = None,
    max_critics: int | None = None,
) -> tuple[list[Persona], list[Persona]]:
    """Load persona YAMLs and apply the per-role caps."""
    d_raw = yaml.safe_load(defenders_path.read_text(encoding="utf-8"))
    c_raw = yaml.safe_load(critics_path.read_text(encoding="utf-8"))

    defenders = [_as_persona(x) for x in (d_raw or {}).get("defenders", [])]
    critics = [_as_persona(x) for x in (c_raw or {}).get("critics", [])]

    if max_defenders is not None:
        defenders = defenders[:max_defenders]
    if max_critics is not None:
        critics = critics[:max_critics]

    return defenders, critics


def _as_persona(obj: dict[str, Any]) -> Persona:
    return Persona(
        id=str(obj["id"]),
        name=str(obj.get("name", obj["id"])),
        model_key=str(obj.get("model_key", "critic")),
        system_prompt=str(obj.get("system_prompt", "")),
        cluster=obj.get("cluster"),
        cluster_affinity=list(obj.get("cluster_affinity", [])) or [],
    )


# -------------------------------------------------------------------------
# Prompt factories
# -------------------------------------------------------------------------


def _critic_prompt_round1(brief: str, critic: Persona) -> str:
    return (
        f"{brief}\n\n"
        f"You are acting as {critic.name} (critic id={critic.id}, "
        f"cluster={critic.cluster}). Using ONLY the project brief above, produce:\n\n"
        "1. One sharp primary attack — specific, testable, names concrete modules "
        "or invariants. No generic platitudes.\n"
        "2. Two concrete 'show-me' challenges the defender must answer with "
        "evidence (file paths, test names, metric thresholds).\n"
        "3. One constructive forward proposal: what should the project build in the "
        "next 6–18 months to close this gap? Specific feature, data source, test, "
        "architecture step — not generic advice.\n\n"
        "Format exactly:\n"
        "ATTACK: <primary attack>\n"
        "CHALLENGE_1: <show-me challenge>\n"
        "CHALLENGE_2: <show-me challenge>\n"
        "FORWARD_PROPOSAL: <timeframe> | <title> | <one-sentence description>\n"
    )


def _defender_prompt_round2(
    brief: str, defender: Persona, attacks: list[Turn]
) -> str:
    bullet_list = "\n".join(
        f"- [{t.agent_id} / {t.agent_name}]\n{t.content.strip()}\n"
        for t in attacks if t.content
    )
    cluster_note = (
        f"Your cluster affinity: {', '.join(defender.cluster_affinity)}. "
        "Select the 3 attacks most relevant to your affinity, or the 3 hardest "
        "overall if none match cleanly."
    )
    return (
        f"{brief}\n\n"
        f"You are acting as {defender.name} (defender id={defender.id}). "
        f"{cluster_note}\n\n"
        "Full list of critic attacks:\n\n"
        f"{bullet_list}\n\n"
        "Respond using this exact structure:\n\n"
        "CLUSTER_SUMMARY: <one paragraph: how does your cluster look overall?>\n\n"
        "For each of your three chosen attacks:\n"
        "### REPLY to <critic_id>\n"
        "VERDICT: <factual_rebuttal | partial_concession | full_concession>\n"
        "ARGUMENT: <2–4 sentences, point to file paths / tests / commits when "
        "possible; acknowledge honestly when evidence is missing>\n"
        "IF_CONCEDED_PRIORITY: <now | Q2 | Q4 | later | n/a>\n"
        "FORWARD_NOTE: <one sentence forward-looking improvement on this topic>\n"
    )


def _critic_counter_prompt_round3(
    brief: str,
    critic: Persona,
    original_attack: Turn,
    defender_reply_excerpt: str,
) -> str:
    return (
        f"{brief}\n\n"
        f"You are {critic.name} (critic id={critic.id}). Your original attack was:\n\n"
        f"{original_attack.content.strip()}\n\n"
        "A defender responded with the following excerpt:\n\n"
        f"{defender_reply_excerpt.strip()}\n\n"
        "Counter-rebuttal rules:\n"
        "- If the defender has actually answered your challenge, say so and stop.\n"
        "- If not, point precisely at what is still missing (file path, test, metric).\n"
        "- Max 4 sentences for the rebuttal itself.\n"
        "- One additional line: FORWARD_INSIST: <y/n> — whether your forward "
        "proposal still stands; if no, briefly say why.\n\n"
        "Format:\n"
        "COUNTER: <rebuttal>\n"
        "STILL_OPEN: <yes | partial | no>\n"
        "FORWARD_INSIST: <yes | no + short reason>\n"
    )


def _router_prompt_round3(
    brief_sample: str, defender_turns: list[Turn], critic_turns: list[Turn],
) -> str:
    """Pick the critics whose attacks were not convincingly resolved."""
    attacks_block = "\n\n".join(
        f"[{c.agent_id}] {c.content.strip()[:600]}" for c in critic_turns
    )
    defender_block = "\n\n".join(
        f"[{d.agent_id}] {d.content.strip()[:1500]}" for d in defender_turns
    )
    return (
        f"Project brief preview:\n{brief_sample}\n\n"
        f"Critic attacks:\n{attacks_block}\n\n"
        f"Defender replies:\n{defender_block}\n\n"
        "Select up to 10 critic ids whose attack was NOT convincingly resolved "
        "by the defenders. Return ONLY a JSON array of critic ids, "
        "e.g. [\"C1\", \"C6\", \"C13\"]. No prose."
    )


def _judge_prompt_round4(
    brief: str, all_turns: list[Turn], min_findings: int, min_recs: int,
) -> str:
    transcript = "\n\n".join(
        f"--- round={t.round} role={t.role} id={t.agent_id} name={t.agent_name} ---\n"
        f"{t.content.strip()}"
        for t in all_turns if t.content
    )
    return (
        f"{brief}\n\n"
        "Full transcript of the adversarial tournament follows. Read it, then "
        "produce a structured report with TWO parallel sections.\n\n"
        f"{transcript}\n\n"
        "Output format — emit exactly these two sections, in Markdown:\n\n"
        "## Section A — Current Weaknesses & Findings\n\n"
        f"Top findings (at least {min_findings}). For each, emit a fenced block "
        "containing a JSON object with keys: id, title, severity "
        "(critical|high|medium|low), category, description, proposed_mitigation, "
        "affected_modules, evidence_from_transcript.\n\n"
        "After the findings, include:\n"
        "- Defender scoreboard: a Markdown table with columns "
        "(defender_id, score 0-5, reasoning in ≤ 30 words).\n"
        "- Blindspot list: bullet list of critic ids whose attack received no "
        "direct defender response.\n\n"
        "## Section B — Strategic Improvement & Expansion Recommendations\n\n"
        f"Top expansion opportunities (at least {min_recs}). For each, emit a "
        "fenced JSON block with keys: id, title, timeframe "
        "(quick-win|medium|strategic), impact (high|medium|low), effort "
        "(low|medium|high), affected_modules, rationale, main_risk.\n\n"
        "After the recommendations, include:\n"
        "- Prioritisation matrix: Markdown table with columns "
        "(title, impact, effort, bucket).\n"
        "- Convergence cluster: bullet list of themes raised by multiple critics.\n"
        "- Dissonance cluster: bullet list of topics where critics and defenders "
        "disagreed about priority.\n\n"
        "Hard rules:\n"
        "- No invented file paths: stay at module-level naming unless the "
        "transcript cites a concrete path.\n"
        "- Severity `critical` must be justified by clear evidence in the "
        "transcript; do not overuse it.\n"
        "- Do not soften findings to be polite — this report is for the user who "
        "owns the project."
    )


# -------------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------------


async def run_tournament(
    *,
    project_root: Path,
    run_dir: Path,
    config: dict[str, Any],
    defenders_path: Path,
    critics_path: Path,
    client: ClaudeClient | None = None,
    max_defenders: int | None = None,
    max_critics: int | None = None,
    full_scale: bool = True,
) -> TournamentResult:
    """Execute the 4-round tournament and persist artefacts."""

    run_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = run_dir / "transcript.jsonl"
    # Truncate any pre-existing file (run-dirs are per-run and fresh, but we
    # are defensive).
    transcript_path.write_text("", encoding="utf-8")

    sources = load_sources(project_root=project_root)
    brief_md = build_brief(sources)
    bhash = brief_hash(brief_md)
    (run_dir / "brief.md").write_text(brief_md, encoding="utf-8")

    defenders, critics = load_personas(
        defenders_path, critics_path,
        max_defenders=max_defenders, max_critics=max_critics,
    )
    if not defenders or not critics:
        raise RuntimeError("No defenders or critics loaded — check persona YAMLs.")

    if client is None:
        client = ClaudeClient(ClaudeClientConfig(
            retry=RetryConfig(
                max_attempts=config["retry"]["max_attempts"],
                initial_backoff_seconds=config["retry"]["initial_backoff_seconds"],
                backoff_multiplier=config["retry"]["backoff_multiplier"],
                retry_on_status=tuple(config["retry"]["retry_on_status"]),
                per_call_timeout_seconds=config["rounds"]["per_call_timeout_seconds"],
            ),
        ))

    models = config["models"]
    tok = config["tokens"]
    max_par = int(config["rounds"]["max_parallel_requests"])
    counter_top_n = int(config["rounds"]["counter_rebuttal_top_n"])

    turns: list[Turn] = []
    errors: list[str] = []
    started_at = _utc_now_iso()

    def _append_turn(turn: Turn) -> None:
        turns.append(turn)
        with transcript_path.open("a", encoding="utf-8") as fh:
            fh.write(turn.as_jsonl() + "\n")

    # ---------------- Round 1: critic attacks ----------------
    logger.info("[tournament] round 1 — %s critics", len(critics))
    r1_jobs = [
        dict(
            model=models[c.model_key],
            system_prompt=c.system_prompt,
            user_prompt=_critic_prompt_round1(brief_md, c),
            max_tokens=tok["critic_attack_max_tokens"],
            temperature=0.8,
        )
        for c in critics
    ]
    r1_results = await client.call_many(r1_jobs, max_parallel=max_par)
    r1_turns = _results_to_turns(
        round_=1, role="critic", personas=critics, results=r1_results,
    )
    for t in r1_turns:
        _append_turn(t)

    # ---------------- Round 2: defender rebuttals ----------------
    logger.info("[tournament] round 2 — %s defenders", len(defenders))
    r2_jobs = [
        dict(
            model=models[d.model_key],
            system_prompt=d.system_prompt,
            user_prompt=_defender_prompt_round2(brief_md, d, r1_turns),
            max_tokens=tok["defender_rebuttal_max_tokens"],
            temperature=0.4,
        )
        for d in defenders
    ]
    r2_results = await client.call_many(r2_jobs, max_parallel=max_par)
    r2_turns = _results_to_turns(
        round_=2, role="defender", personas=defenders, results=r2_results,
    )
    for t in r2_turns:
        _append_turn(t)

    # ---------------- Round 3: counter-rebuttals ----------------
    selected_ids = await _select_counter_critics(
        client=client,
        config=config,
        brief_sample=brief_md[:1500],
        r1_turns=r1_turns,
        r2_turns=r2_turns,
        cap=counter_top_n,
    )
    # Router turn recorded for audit.
    if selected_ids is not None:
        _append_turn(Turn(
            round=3, agent_id="ROUTER", agent_name="counter-rebuttal router",
            role="router", model=models["router"],
            prompt_tokens=0, completion_tokens=0,
            content="selected=" + ",".join(selected_ids),
        ))

    critics_by_id = {c.id: c for c in critics}
    turns_by_critic = {t.agent_id: t for t in r1_turns}
    defender_reply_lookup = "\n\n".join(t.content for t in r2_turns)

    r3_personas = [critics_by_id[cid] for cid in (selected_ids or []) if cid in critics_by_id]
    r3_jobs = [
        dict(
            model=models[c.model_key],
            system_prompt=c.system_prompt,
            user_prompt=_critic_counter_prompt_round3(
                brief_md, c, turns_by_critic[c.id], defender_reply_lookup,
            ),
            max_tokens=tok["critic_counter_max_tokens"],
            temperature=0.7,
        )
        for c in r3_personas if c.id in turns_by_critic
    ]
    if r3_jobs:
        r3_results = await client.call_many(r3_jobs, max_parallel=max_par)
        r3_turns = _results_to_turns(
            round_=3, role="critic", personas=r3_personas, results=r3_results,
        )
        for t in r3_turns:
            _append_turn(t)

    # ---------------- Round 4: judge synthesis ----------------
    min_findings = (
        config["judge"]["min_findings_full"]
        if full_scale
        else config["judge"]["min_findings_reduced"]
    )
    min_recs = (
        config["judge"]["min_recommendations_full"]
        if full_scale
        else config["judge"]["min_recommendations_reduced"]
    )

    judge_job = dict(
        model=models["judge"],
        system_prompt=(
            "You are an impartial senior reviewer synthesising an adversarial "
            "tournament about a quantitative trading project. Be rigorous, "
            "evidence-driven, and brutally honest. Do not flatter the project."
        ),
        user_prompt=_judge_prompt_round4(brief_md, turns, min_findings, min_recs),
        max_tokens=tok["judge_synthesis_max_tokens"],
        temperature=0.2,
    )
    judge_result = await client.call(**judge_job)
    judge_turn = Turn(
        round=4, agent_id="JUDGE", agent_name="Synthesis Judge",
        role="judge", model=judge_result.model,
        prompt_tokens=judge_result.prompt_tokens,
        completion_tokens=judge_result.completion_tokens,
        content=judge_result.content,
        attempts=judge_result.attempts,
        error=judge_result.error,
    )
    _append_turn(judge_turn)
    if judge_result.error:
        errors.append(f"judge error: {judge_result.error}")

    # ---------------- Totals ----------------
    total_in = sum(t.prompt_tokens for t in turns)
    total_out = sum(t.completion_tokens for t in turns)
    cost = _estimate_cost(config, turns)

    finished_at = _utc_now_iso()

    return TournamentResult(
        run_id=run_dir.name,
        run_dir=run_dir,
        brief=brief_md,
        brief_hash=bhash,
        turns=turns,
        started_at_utc=started_at,
        finished_at_utc=finished_at,
        config_snapshot=config,
        judge_content=judge_result.content,
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        cost_estimate_usd=cost,
        errors=errors,
    )


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------


def _results_to_turns(
    *, round_: int, role: str, personas: list[Persona], results: list[CallResult],
) -> list[Turn]:
    turns: list[Turn] = []
    for p, r in zip(personas, results, strict=True):
        turns.append(Turn(
            round=round_, agent_id=p.id, agent_name=p.name, role=role,
            model=r.model, prompt_tokens=r.prompt_tokens,
            completion_tokens=r.completion_tokens, content=r.content,
            attempts=r.attempts, error=r.error,
        ))
    return turns


async def _select_counter_critics(
    *,
    client: ClaudeClient,
    config: dict[str, Any],
    brief_sample: str,
    r1_turns: list[Turn],
    r2_turns: list[Turn],
    cap: int,
) -> list[str] | None:
    """Ask the router which critics were not resolved; fall back on heuristics."""
    jobs = dict(
        model=config["models"]["router"],
        system_prompt=(
            "You are a neutral dispatcher. Your only job is to select unresolved "
            "critic attacks for a counter-rebuttal round. Output strict JSON."
        ),
        user_prompt=_router_prompt_round3(brief_sample, r2_turns, r1_turns),
        max_tokens=config["tokens"]["router_max_tokens"],
        temperature=0.0,
    )
    try:
        result = await client.call(**jobs)
    except Exception as exc:  # pragma: no cover
        logger.warning("[tournament] router error — using heuristic: %s", exc)
        return _heuristic_counter_selection(r1_turns, cap)

    if not result.ok or not result.content:
        return _heuristic_counter_selection(r1_turns, cap)

    try:
        payload = result.content
        start = payload.find("[")
        end = payload.rfind("]")
        if start == -1 or end == -1:
            return _heuristic_counter_selection(r1_turns, cap)
        ids = json.loads(payload[start:end + 1])
        if not isinstance(ids, list):
            return _heuristic_counter_selection(r1_turns, cap)
        valid_ids = {t.agent_id for t in r1_turns}
        selected = [str(x) for x in ids if str(x) in valid_ids][:cap]
        return selected
    except Exception:
        return _heuristic_counter_selection(r1_turns, cap)


def _heuristic_counter_selection(r1_turns: list[Turn], cap: int) -> list[str]:
    """Fallback: pick the first *cap* critic ids — better than nothing."""
    return [t.agent_id for t in r1_turns[:cap]]


def _estimate_cost(config: dict[str, Any], turns: list[Turn]) -> float:
    rates = config["cost_rates_per_mtok"]
    total_cost = 0.0
    for t in turns:
        model = t.model.lower()
        if "sonnet" in model:
            in_rate = rates["sonnet_input"]
            out_rate = rates["sonnet_output"]
        else:
            in_rate = rates["haiku_input"]
            out_rate = rates["haiku_output"]
        total_cost += t.prompt_tokens / 1_000_000.0 * in_rate
        total_cost += t.completion_tokens / 1_000_000.0 * out_rate
    return round(total_cost, 4)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
