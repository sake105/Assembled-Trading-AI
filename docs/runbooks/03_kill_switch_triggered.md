# Runbook 03: Kill Switch Triggered

**Severity:** critical
**ETA to resolution:** 15–60 min (plus human approval for recovery)
**On-call contact:** trading-ops
**Component:** `src/assembled_core/execution/kill_switch.py`

## Symptoms

- `output/runs/_kill_switch/state.json` shows `active: true`.
- `kill_switch_audit.jsonl` has recent activation entry.
- Cycle logs contain `[CRITICAL] KILL SWITCH ACTIVATED`.
- Orders filtered out at Phase 18; `orders_placed == 0` with non-empty targets.
- Alert emitted with severity `critical`.

## Immediate Actions (first 5 min)

1. **Do NOT deactivate immediately.** A kill-switch trigger is a safety signal, not a nuisance.
2. Read the activation entry:
   - `tail -n 1 output/runs/_kill_switch/kill_switch_audit.jsonl | python -m json.tool`
   - Note `reason`, `throttle_pct`, `trigger_source`, `timestamp`.
3. Snapshot current ledger + last run folder.
4. Notify stakeholders (Telegram / email channel) with reason + audit entry.

## Diagnosis

1. Classify the activation source:
   - `reason` contains "drawdown" → see Runbook 04 (drawdown limit hit).
   - `reason` contains "broker" / "api" → see Runbook 01.
   - `reason` contains "reconcile" / "ledger" → see Runbook 02.
   - `reason` contains "circuit_breaker" / "flash_crash" → see Runbook 10.
   - `reason` contains "pit" / "data" → see Runbook 05 / 06.
   - `reason` contains "manual" → an operator triggered it; find who and why.
2. Inspect upstream logs for the cycle that triggered activation:
   - `output/runs/<run_id>/run_kpis.json`
   - `output/runs/<run_id>/trading_cycle.log`
3. Determine whether the root cause persists:
   - Is the underlying condition (drawdown, outage, drift) still present?
   - Has market state changed since trigger?

## Resolution

**Do not auto-recover.** Kill-switch recovery always requires human approval.

Steps:
1. Follow the appropriate sub-runbook for the root cause.
2. Once root cause is resolved and verified, document in an incident report:
   - What triggered it
   - What was fixed
   - What validation was performed
   - Who approves re-enablement
3. Run a single dry-run cycle: `python scripts/run_live_paper.py once --dry-run`.
4. Verify dry-run outputs are sensible (target_weights reasonable, no stale symbols, no risk-state warnings).
5. Deactivate only after human sign-off:
   - `python scripts/kill_switch_cli.py deactivate --reason "post-incident recovery, approved by <name>"`
6. Run ONE real cycle in foreground. Monitor live.
7. Verify trade_journal + reconcile_report before restoring scheduler.

## Post-Incident

- Append audit entry to `KNOWN_ISSUES.md`.
- Full post-mortem in `docs/post_mortems/YYYY-MM-DD_kill_switch_*.md`, including:
  - Trigger cause
  - Time to detect
  - Time to resolve
  - Mitigation actions
  - Prevention recommendations
- If the trigger was a false positive, tighten the trigger threshold or add a guard; open a tracked issue.
- If the trigger was correct, keep thresholds and improve upstream prevention.
