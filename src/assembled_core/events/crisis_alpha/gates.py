"""Activation and deactivation gate checks for Crisis-Alpha v1.

Gates are pure functions that take a CrisisAlphaContext and return a
boolean with a reason string.  They are used by the state machine and
the pipeline to decide whether transitions are allowed.

Guard hierarchy (highest priority first):
    1. Health gate       — ERROR health blocks ALL activation
    2. Social-only guard — social-only geo cannot activate
    3. Evidence gate     — min qualifying trigger count required
    4. Source gate       — min distinct sources required
    5. Market stress     — market stress must be confirmed for ACTIVE
"""

from __future__ import annotations

from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext


def check_health_gate(ctx: CrisisAlphaContext) -> tuple[bool, str]:
    """Return (ok, reason). Blocks activation if health is ERROR (health_ok=False).

    Note: health_ok=True means GREEN or DEGRADED (degraded allows monitoring,
    but severity-capped triggers are acceptable for WATCH).
    ERROR health (health_ok=False) must block WATCH→ACTIVE.
    """
    if not ctx.health_ok:
        return False, "health gate: news/intel health is ERROR — activation blocked"
    return True, "health gate: OK"


def check_social_only_guard(ctx: CrisisAlphaContext) -> tuple[bool, str]:
    """Return (ok, reason). Blocks activation if the signal is social-media-only.

    Social-only signals are unreliable and subject to manipulation.
    At least one Tier-A news or disclosure source must contribute.
    """
    if ctx.social_only:
        return (
            False,
            "social-only guard: geo signal comes from social media only — activation blocked",
        )
    return True, "social-only guard: OK"


def check_evidence_gate(
    ctx: CrisisAlphaContext,
    min_trigger_count: int = 1,
) -> tuple[bool, str]:
    """Return (ok, reason). Requires a minimum number of qualifying news triggers.

    Args:
        ctx: CrisisAlphaContext with news_trigger_items list.
        min_trigger_count: Minimum number of triggers with severity >= 1.
    """
    qualifying = [t for t in ctx.news_trigger_items if int(t.get("severity", 0)) >= 1]
    count = len(qualifying)
    if count < min_trigger_count:
        return (
            False,
            f"evidence gate: {count} qualifying trigger(s) < min {min_trigger_count}",
        )
    return True, f"evidence gate: OK ({count} qualifying triggers)"


def check_source_gate(
    ctx: CrisisAlphaContext,
    min_sources: int = 2,
) -> tuple[bool, str]:
    """Return (ok, reason). Requires geo signal from at least min_sources distinct sources.

    Args:
        ctx: CrisisAlphaContext with geo_sources count.
        min_sources: Minimum distinct sources (default 2 — guards against single-source spike).
    """
    if ctx.geo_sources < min_sources:
        return (
            False,
            f"source gate: {ctx.geo_sources} source(s) < min {min_sources}",
        )
    return True, f"source gate: OK ({ctx.geo_sources} sources)"


def check_market_stress_gate(ctx: CrisisAlphaContext) -> tuple[bool, str]:
    """Return (ok, reason). Requires market stress confirmation for ACTIVE state.

    Market stress confirmation is required to avoid false activations during
    geo-event noise without corresponding market reaction.
    """
    if not ctx.market_stress_ok:
        return (
            False,
            "market stress gate: market_stress_ok=False — WATCH→ACTIVE blocked",
        )
    return True, "market stress gate: OK"


def check_daily_loss_gate(ctx: CrisisAlphaContext) -> tuple[bool, str]:
    """Return (ok, reason). Returns False if daily loss limit is breached.

    When daily loss is breached, the state machine should move to PAUSE.
    This gate is separate from the state machine for testability.
    """
    if ctx.daily_loss_breached():
        return (
            False,
            f"daily loss gate: pnl={ctx.daily_pnl:.4f} >= limit={ctx.daily_loss_limit:.4f}",
        )
    return True, "daily loss gate: OK"


def check_evidence_grade_gate_from_ctx(
    ctx: "CrisisAlphaContext",
    require_for_active: str = "B",
) -> tuple[bool, str]:
    """Gate check using evidence grade stored in context.

    Args:
        ctx: CrisisAlphaContext -- reads ctx.evidence_grade if present.
        require_for_active: Minimum grade required for ACTIVE.

    Returns:
        (ok, reason).
    """
    from src.assembled_core.events.evidence_engine import (
        EvidenceGrade,
        check_evidence_grade_gate,
    )

    grade_str = getattr(ctx, "evidence_grade", None)
    if grade_str is None:
        return False, "evidence grade gate: DENIED (no grade set — default-deny)"

    try:
        grade = EvidenceGrade(grade_str)
    except ValueError:
        return (
            False,
            f"evidence grade gate: DENIED (unknown grade {grade_str!r} — default-deny)",
        )

    return check_evidence_grade_gate(grade, require_for_active=require_for_active)


def run_all_activation_gates(
    ctx: CrisisAlphaContext,
    *,
    min_trigger_count: int = 1,
    min_sources: int = 2,
    require_evidence_grade: str | None = None,
) -> tuple[bool, list[str]]:
    """Run all activation gates and return (all_ok, list_of_reasons).

    Gates are checked in priority order.  All must pass for activation to proceed.

    Args:
        ctx: CrisisAlphaContext.
        min_trigger_count: Min qualifying triggers for evidence gate.
        min_sources: Min distinct sources for source gate.
        require_evidence_grade: If set, also run evidence grade gate with this
            minimum grade (e.g. "B").  None skips the evidence grade gate.

    Returns:
        (True, reasons) if all pass, (False, [first_failing_reason]) otherwise.
    """
    checks = [
        check_health_gate(ctx),
        check_social_only_guard(ctx),
        check_evidence_gate(ctx, min_trigger_count),
        check_source_gate(ctx, min_sources),
        check_market_stress_gate(ctx),
        check_daily_loss_gate(ctx),
    ]
    if require_evidence_grade is not None:
        checks.append(
            check_evidence_grade_gate_from_ctx(
                ctx, require_for_active=require_evidence_grade
            )
        )

    reasons = []
    for ok, reason in checks:
        reasons.append(reason)
        if not ok:
            return False, reasons  # fail-fast: return on first failure

    return True, reasons
