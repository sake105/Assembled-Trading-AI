INC-0002 — Pending Intents Accumulation (Pilot v1, 2026-05-05/06)
=================================================================

## Kontext

- System: Paper pilot v1 (30-day Alpaca paper-trading pilot)
- Strategy: `run_live_paper.py` + `multifactor_long_short` via AlpacaAdapter
- Environment: paper-api.alpaca.markets (paper trading, not live capital)
- Branch: main, commit range 2026-05-05 to 2026-05-06
- Files: `output/pilot/pilot_manifest_v1_aborted_2026-05-06.json`, `output/pilot/pilot_verdict_20260506.json`

## Symptom

Each restart of `run_live_paper.py` printed:

```
[WARNING] [run_live_paper] 25 pending order intents from prior crash — re...
```

The count grew over successive restarts (9 → 15 → 25 → 25). By day 3 and 4 of
pilot v1, 25 ghost intents were reported every restart. The verdict was NO-GO
because `days_run = 4` (failed minimum 14-day criterion), not because of crashes
(0 crash days).

## Impact

- 25 stale intents persisted across 4 pilot days.
- `n_orders_detected` was consistently 1 per day despite potentially needing more,
  suggesting the intent backlog blocked fresh order generation or the signals were
  effectively zero.
- Pilot was aborted at 4 days; the 30-day GO/NO-GO criteria could not be evaluated.
- No real capital at risk (paper trading). No financial loss.

## Detection / Signal

- Detected immediately on each restart via log warning in `run_live_paper.py` startup.
- Visible in `output/pilot/pilot_manifest_v1_aborted_2026-05-06.json` →
  `days[*].output_snippet` across all 4 days.
- Reported but not automatically resolved: `position_sync.py` cleared them manually
  in commit d5630b6 (cash_tol 0.01→1.0 fix, 21 tests pass).

## Root Cause

**Technical cause:** The intent state file (`output/runs/_paper_ledger/intent_state.json`
or equivalent) accumulated intent records for orders that were submitted but whose fills
were never confirmed back into the state machine before the process crashed or restarted.
Each restart re-detected the same unconfirmed intents rather than discarding them after a
configurable TTL.

Two contributing sub-causes:
1. **No TTL on pending intents.** If a submitted order was filled (or cancelled) by
   Alpaca but the local state was not updated before crash, the intent remained in
   the pending set indefinitely.
2. **No stale-order cancellation at startup.** On restart, open orders from a
   prior session were not cancelled — so new cycles could collide with stale orders,
   and intent accounting became inconsistent.
3. **cash_tol too tight (0.01).** The reconciliation in `position_sync.py` used a
   $0.01 cash tolerance; floating-point rounding caused ghost discrepancies to
   accumulate instead of resolving (fixed in d5630b6: cash_tol → 1.0).

**Process cause:** No pre-flight check existed to detect and clear stale broker orders
on restart before the main trading cycle ran.

## Fix

Implemented in two steps:

1. **d5630b6 (2026-05-06):** `position_sync.py` cash_tol 0.01 → 1.0. 21 tests pass.
   Cleared the 25 ghost intents in KNOWN_ISSUES §0.1.

2. **Items 68 + 80 (2026-05-07, this commit):** `scripts/run_paper_pilot.py`:
   - `cancel_all_stale_orders(older_than_minutes=5)` — cancels any open broker order
     older than 5 minutes at startup, before the trading cycle runs.
   - `check_state_recovery()` — loads disk intent-state, fetches broker positions,
     logs all discrepancies (symbols only-on-disk or only-at-broker) as WARN.
   - Both run inside `run_startup_checks()`, called at the top of `cmd_run_day()`.

Code pointers:
- `scripts/run_paper_pilot.py`: `cancel_all_stale_orders()`, `check_state_recovery()`,
  `run_startup_checks()` (added 2026-05-07)
- `src/assembled_core/ops/position_sync.py`: cash_tol fix (d5630b6)

## Tests

- Item 42 test: `tests/test_margin_call_handler.py` (separate incident, same session)
- `run_startup_checks()` is a best-effort runtime function; unit tests for it would
  require broker mock. Marked as TODO: add integration smoke test for startup checks.

## Prevention (Guardrails)

- Startup: `cancel_all_stale_orders(older_than_minutes=5)` runs before every
  daily cycle in `run_paper_pilot.py --run-day`.
- Startup: `check_state_recovery()` logs disk-vs-broker discrepancies as WARN.
- Monitoring: `output/pilot/pilot_manifest.json` → `days[].output_snippet` now
  captures first 500 chars of each day's log for quick triage.
- cash_tol in reconciliation is now 1.0 (not 0.01).

## Follow-ups (Backlog Items)

- `[ ]` Add broker-adapter mock and write unit test for `cancel_all_stale_orders`.
- `[ ]` Add explicit intent TTL (e.g. 24h) to the intent state machine — intents
  older than TTL should be auto-discarded on restart regardless of fill status.
- `[ ]` Add Prometheus/alerting counter for `stale_intents_at_startup` metric.
- `[ ]` On each successful fill, ensure intent is removed from pending set atomically
  (avoid crash between order submission and state update).
