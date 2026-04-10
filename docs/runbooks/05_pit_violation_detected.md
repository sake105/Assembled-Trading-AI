# Runbook 05: PIT Violation Detected

**Severity:** high
**ETA to resolution:** 1–4 h (plus research revalidation)
**On-call contact:** trading-ops + research
**Component:** `src/assembled_core/data/pit_guard.py`, `data/freshness_monitor.py`, `qa/qa_gates.py`

## Symptoms

- `pit_guard_audit.jsonl` contains new `violation` entries.
- `check_leakage` QA gate fails.
- A feature references data from timestamp > `as_of`.
- Backtest results show implausibly high Sharpe / low DD vs prior runs.
- Feature-value distribution shift detected by drift monitor.
- New feature added without PIT-safe timestamp handling.

## Immediate Actions (first 5 min)

1. Engage kill-switch if live trading is affected:
   - `python scripts/kill_switch_cli.py activate --throttle 100 --reason "PIT violation"`.
2. Snapshot PIT audit log:
   - `cp output/runs/_pit_audit/pit_guard_audit.jsonl output/runs/_incidents/`
3. Identify which feature(s) violated:
   - Inspect the most recent violation entries: `feature`, `as_of`, `offending_timestamp`, `delta`.
4. Tag the affected runs / models as quarantined (do not trust results).

## Diagnosis

1. Classify the violation source:
   - **Feature compute uses raw data later than as_of** → code bug in feature builder. Go to 2a.
   - **Data file has newer rows than expected** → ingestion wrote forward-dated data. Go to 2b.
   - **Lookahead via merge/join** → join key aligned wrong. Go to 2c.
   - **External API delivers stamped data with wrong timestamp** → data provider issue. Go to 2d.
2. Targeted checks:
   - **2a** Review the feature function; check windowing, shift(), `<` vs `<=` in time filters.
   - **2b** Check recent ingest logs for the affected symbol + feature; look for backfills that overwrote current data.
   - **2c** Inspect merge keys; verify `as_of_left <= as_of_right` semantics.
   - **2d** Contact data provider or add stricter provider-side freshness check.
3. Quantify impact:
   - Which runs are affected? (date range)
   - Which models were trained on contaminated data?
   - Which backtest KPIs are now unreliable?

## Resolution

**Path A — Feature bug:**
1. Fix the feature computation with explicit as_of filter.
2. Add PIT unit test for the specific violation pattern.
3. Run `pytest tests/data/test_pit_guard.py tests/features/test_<module>.py`.
4. Rebuild feature cache from scratch for the affected symbols + date range.
5. Re-run backtest and compare to prior contaminated run.

**Path B — Ingestion bug:**
1. Restore last known clean data snapshot.
2. Re-ingest from authoritative source with strict as_of enforcement.
3. Verify `pit_guard` passes on rebuilt data.

**Path C — Join / merge bug:**
1. Fix merge semantics.
2. Add property test for merge invariant: `result.timestamp <= max(left.ts, right.ts)`.

**Path D — Provider issue:**
1. Raise with data vendor.
2. Add defensive guard in ingestion layer to reject or delay forward-stamped rows.

## Post-Incident

- Any model trained on contaminated data must be retrained or quarantined.
- Any backtest using the affected feature must be re-run.
- Add regression test that replays the violation and asserts detection.
- Post-mortem in `docs/post_mortems/YYYY-MM-DD_pit_violation_*.md`.
- Update `KNOWN_ISSUES.md`.
- Review all features for similar patterns using `check_leakage` QA gate.
- Do not deactivate kill-switch until clean full backtest run validates the fix.
