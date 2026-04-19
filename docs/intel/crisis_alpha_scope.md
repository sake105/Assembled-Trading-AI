# Crisis Alpha — Scope and Migration Notes

**Status:** v0 deprecated, v1 active (2026-04-19)

## Two Implementations

### v0 — `src/assembled_core/intel/crisis_alpha_worker.py`

- Original state machine: `CrisisAlphaState`, `update_crisis_state()`
- Entry point: `scripts/run_crisis_alpha_worker.py`
- Status: **DEPRECATED** — kept only to avoid breaking the script
- Do not add features here

### v1 — `src/assembled_core/events/crisis_alpha/`

- Pure functions, pydantic models, 63 tests
- `CrisisAlphaContext`, `EvidenceGrade`, `check_evidence_grade_gate()`
- `gates.py` wired into `risk/disclosures_confirm.py`
- Policy-gated: `policy.intel.crisis_alpha.enabled: false` (shadow-only by default)

## Migration Path

When `policy.intel.crisis_alpha.enabled` is flipped to `true` (T4.1):

1. `trading_cycle.py` calls `run_crisis_alpha_pipeline(ctx)` after signal generation
2. Result is logged (shadow-only in step 1)
3. Step 2: result influences orders (flag-gated)
4. Step 3: `PAUSE` state triggers pre-trade kill-switch (T4.3)

At that point, `run_crisis_alpha_worker.py` becomes redundant and can be sunset.

## Why Two Implementations Exist

The v1 was written as a clean-room rewrite for the Phase 8 Intel upgrade.
The v0 was never formally deprecated because no task explicitly scheduled it.
This gap is addressed by goofy-questing-crystal T3.6.
