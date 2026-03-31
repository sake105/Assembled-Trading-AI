"""Crisis-Alpha v1 pipeline — M5.

Orchestrates the full crisis-alpha evaluation cycle:
    1. Load persisted crisis state.
    2. Compute next state (deterministic transition).
    3. Run activation gates.
    4. If ACTIVE: generate entry targets.
    5. Check exit rules for open positions.
    6. Check deactivation triggers.
    7. Persist updated state.
    8. Emit structured result dict.

The pipeline is deliberately thin.  Order submission is NOT done here —
the result contains intent data (target_weights, positions_to_exit) that
the caller (worker script or trading cycle hook) uses to generate and
submit orders via the existing execution layer.

This keeps the pipeline testable with fixtures and avoids live I/O side
effects inside the core logic.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.assembled_core.events.crisis_alpha.context import CrisisAlphaContext
from src.assembled_core.events.crisis_alpha.entry import generate_crisis_entry
from src.assembled_core.events.crisis_alpha.exit_rules import (
    check_deactivation_triggers,
    get_positions_to_exit,
)
from src.assembled_core.events.crisis_alpha.gates import run_all_activation_gates
from src.assembled_core.events.crisis_alpha.state_machine import (
    CrisisStateRecord,
    compute_next_crisis_state,
    load_crisis_state,
    save_crisis_state,
)

logger = logging.getLogger(__name__)


def run_crisis_alpha_pipeline(
    ctx: CrisisAlphaContext,
    policy: dict | None = None,
    *,
    state_path: Path | str | None = None,
    prices: dict[str, float] | None = None,
    reset_pause: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run one crisis-alpha evaluation cycle.

    Args:
        ctx: Current CrisisAlphaContext (all pipeline inputs).
        policy: Policy dict (from crisis_alpha.yaml). None → use defaults.
        state_path: Path to the JSON state persistence file. None → default path.
        prices: Optional {symbol: current_price} for exit/break-even checks.
        reset_pause: If True, allows PAUSE→WATCH transition (manual override).
        dry_run: If True, state is computed but NOT persisted.

    Returns:
        Dict with keys:
            state:              str — new crisis state
            state_record:       CrisisStateRecord — full state record
            previous_state:     str — state before this run
            gates_ok:           bool — True if all activation gates passed
            gate_reasons:       list[str] — gate evaluation results
            target_weights:     dict[str,float] — entry targets (empty if not ACTIVE)
            entry_reasons:      list[str] — entry signal audit log
            positions_to_exit:  list[tuple[dict,str]] — (position, reason) pairs
            should_flatten_all: bool — True if full portfolio flatten required
            flatten_reason:     str — reason for flatten (or "")
            errors:             list[str] — non-fatal errors/warnings
    """
    policy = policy or {}
    now_utc = ctx.timestamp_utc
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)

    errors: list[str] = []

    # --- Step 1: Load persisted state ---
    prev: CrisisStateRecord = load_crisis_state(state_path)
    previous_state = prev.state
    logger.info(
        "[CRISIS_PIPELINE] start | prev_state=%s geo_score=%.2f health_ok=%s",
        previous_state,
        ctx.geo_score,
        ctx.health_ok,
    )

    # --- Step 2: Compute next state ---
    new_record = compute_next_crisis_state(
        ctx, policy, now_utc, prev, reset=reset_pause
    )
    current_state = new_record.state

    # --- Step 3: Run activation gates (for audit log, not to block transitions) ---
    min_trigger_count = int(
        (policy.get("crisis_alpha") or {})
        .get("hysteresis", {})
        .get("min_trigger_count", 1)
    )
    min_sources = int(
        (policy.get("crisis_alpha") or {}).get("hysteresis", {}).get("min_sources", 2)
    )
    gates_ok, gate_reasons = run_all_activation_gates(
        ctx,
        min_trigger_count=min_trigger_count,
        min_sources=min_sources,
    )

    # --- Step 4: Entry targets (only when ACTIVE) ---
    target_weights: dict[str, float] = {}
    entry_reasons: list[str] = []
    if current_state == "ACTIVE":
        target_weights, entry_reasons = generate_crisis_entry(ctx, policy)
    else:
        entry_reasons = [f"no entry: state={current_state}"]

    # --- Step 5: Exit rules for open positions ---
    positions_to_exit = get_positions_to_exit(
        ctx.open_positions, now_utc, policy, prices
    )

    # --- Step 6: Deactivation trigger check ---
    should_flatten_all, flatten_reason = check_deactivation_triggers(ctx, current_state)

    if should_flatten_all and ctx.open_positions:
        logger.warning(
            "[CRISIS_PIPELINE] FLATTEN ALL triggered: %s | %d open positions",
            flatten_reason,
            len(ctx.open_positions),
        )

    # --- Step 7: Persist state ---
    if not dry_run:
        try:
            save_crisis_state(new_record, state_path)
        except Exception as exc:
            err = f"state persistence failed: {exc}"
            errors.append(err)
            logger.error("[CRISIS_PIPELINE] %s", err)
    else:
        logger.info("[CRISIS_PIPELINE] dry_run=True — state NOT persisted")

    # --- Step 8: Emit result ---
    logger.info(
        "[CRISIS_PIPELINE] done | state=%s→%s | gates_ok=%s | "
        "targets=%d | exits=%d | flatten=%s",
        previous_state,
        current_state,
        gates_ok,
        len(target_weights),
        len(positions_to_exit),
        should_flatten_all,
    )

    return {
        "state": current_state,
        "state_record": new_record,
        "previous_state": previous_state,
        "gates_ok": gates_ok,
        "gate_reasons": gate_reasons,
        "target_weights": target_weights,
        "entry_reasons": entry_reasons,
        "positions_to_exit": positions_to_exit,
        "should_flatten_all": should_flatten_all,
        "flatten_reason": flatten_reason,
        "errors": errors,
    }
