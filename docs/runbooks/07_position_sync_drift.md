# Runbook 07: Position Sync Drift

**Severity:** critical
**ETA to resolution:** 30–90 min
**On-call contact:** trading-ops
**Component:** `src/assembled_core/accounting/reconciliation.py`, `ops/paper_ledger.py`, `execution/broker_execution.py`, `execution/broker_adapter.py`

## Symptoms

- `reconciliation_report.json` shows non-empty `diffs` list with quantity or cash mismatches above the configured tolerance.
- `reconcile_ledger_vs_broker` raises when `fail_fast=True`.
- `paper_ledger.ledger_state.json` reports different positions than the broker's `get_positions()` response.
- `run_kpis.json` equity differs from the broker account equity by more than a few cents.
- Audit log shows `RECONCILE_FAILED` or equivalent entry.

## Immediate Actions (first 5 min)

1. **Stop the scheduler immediately.** No further orders may be generated while the ledger is out of sync.
2. Snapshot both sides of the truth:
   - `cp output/runs/_paper_ledger/ledger_state.json output/runs/_paper_ledger/ledger_state.backup_drift_$(date +%s).json`
   - Save the broker's `get_positions()` + `get_account()` response to `output/runs/_incident/broker_snapshot_$(date +%s).json`.
3. Capture the last `reconciliation_report.json` in the same incident directory.
4. If the global kill switch is not already active, activate it manually:
   - `python -c "from src.assembled_core.execution.kill_switch import activate_kill_switch; activate_kill_switch(throttle_pct=100.0, reason='position_sync_drift')"`

## Diagnosis

1. Classify the drift:
   - **Quantity mismatch on one symbol:** most likely a missed fill, partial fill, or fill emitted but not written to the ledger.
   - **Quantity mismatch on several symbols of the same recent order batch:** likely a crash between `execute_via_broker` and `apply_fills_to_ledger`.
   - **Cash drift only:** commission / fee mismatch, currency conversion, or a ledger write that used the wrong avg price.
   - **Equity drift with flat positions:** almost always a cost or dividend accounting issue.
2. Cross-reference the broker's fill history:
   - Pull the broker `list_orders` for the last 24 hours and compare against `trade_journal.json`.
   - Missing fills in the journal but present at the broker confirm the ledger is the lagging side.
3. Check the run log for exceptions between `execute_via_broker` and `append_trade_journal_entries`:
   - A `KeyboardInterrupt`, OOM, or un-handled broker-adapter exception at that point leaves the ledger half-updated.
4. Verify reconciliation tolerances in the current policy:
   - If the drift is below the tolerance the reconciler should have accepted it silently. If it did not, the tolerance was tightened recently — check policy history.

## Resolution

### A) Missing fill on the ledger side

1. Fetch the authoritative fill from the broker order history.
2. Build a manual patch entry matching the existing `apply_fills_to_ledger` schema and run it through the normal function — **do not hand-edit `ledger_state.json`**.
3. Re-run reconciliation; confirm diffs are empty.

### B) Ghost position on the ledger (ledger has it, broker does not)

1. Verify the broker order was never submitted OR was canceled upstream.
2. If canceled, apply a compensating ledger adjustment via the same function that closes positions. Again, no direct file edits.
3. Re-run reconciliation.

### C) Cash-only drift within expected fee variance

1. If the drift is below a sane fee tolerance (e.g. 1–5 bps of notional), bump the `cash_tol` in the reconciliation config for this run only and document it.
2. Do **not** permanently loosen the tolerance without review — that defeats the guardrail.

### D) Unexplained drift

1. If the root cause cannot be identified within the ETA, keep the kill switch active and escalate.
2. Do not resume trading on a ledger of unknown quality.

## Post-Incident

- File a post-mortem in `docs/post_mortems/YYYY-MM-DD_position_sync_drift.md`.
- Update `KNOWN_ISSUES.md` with the incident and the exact resolution used.
- If the drift was caused by a crash between `execute_via_broker` and `apply_fills_to_ledger`, add an idempotency or journal-first pattern on that path so a crash cannot repeat the drift.
- If the drift was caused by a tolerance mismatch, do **not** loosen the tolerance; tighten the writer instead.
- Manual reset of the kill switch requires a second pair of eyes; the safety invariant is in `CLAUDE.md`.
