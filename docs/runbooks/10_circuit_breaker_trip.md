# Runbook 10: Circuit Breaker Trip

**Severity:** high
**ETA to resolution:** 30–90 min
**On-call contact:** trading-ops
**Component:** `src/assembled_core/risk/circuit_breaker.py` (`CircuitBreaker`, `VolCircuitBreaker`), `execution/kill_switch.py`, `pipeline/trading_cycle.py` (phase 5.5)

## Symptoms

- Log line `[CircuitBreaker] TRIPPED: x.x% drop in ...` or `[VolCircuitBreaker] TRIPPED: short/long vol ratio=...`.
- `run_kpis.json` has the cycle end with empty targets + `reason: circuit_breaker`.
- Kill switch is active with reason `flash_crash` / `vol_spike` / similar.
- No new orders generated for the current cycle despite non-empty inputs.
- `get_state()` of the circuit breaker shows `is_tripped: true` with a recent `tripped_at`.

## Immediate Actions (first 5 min)

1. **Do not reset the breaker.** The breaker tripped on purpose. Treat this as "system did the right thing" until proven otherwise.
2. Confirm whether the trip is **real** (true price event) or a **false positive** (data artefact, stale prices, bad tick):
   - Cross-check the drop against a second source (e.g. a public quote page) for the trigger symbol / SPY / VIX.
   - If the trip was on realised vol ratio, check whether the short-window return set contains any outlier that looks like a bad tick.
3. Snapshot the run artefacts for the tripped cycle into `output/runs/_incident/cb_trip_$(date +%s)/`.

## Diagnosis

### Real event

1. What was the trigger magnitude?
   - Rolling-window drop: check `window_minutes` and the `drop_threshold_pct` in policy.
   - Vol-ratio spike: check `short_window`, `long_window`, `ratio_threshold`.
2. Is the broader market confirming (SPY drop, VIX spike, major news)?
3. Is the account in a state where resuming is safe?
   - Positions currently held: mark-to-market against the new price level.
   - Drawdown gate already engaged? (see runbook 04)
   - Liquidity of open positions acceptable?

### False positive

1. Data:
   - Bad tick from the primary source (e.g. a split-unadjusted print).
   - Stale price used as the reference high.
   - Wrong symbol mapped to SPY / index reference.
2. Logic:
   - Window just elapsed and a legitimate old high dominated the ratio.
   - `window_minutes` too small for the bar cadence you are using.

## Resolution

### Real event, safe to resume

1. Wait out the cooldown (`cooldown_minutes`) or set it explicitly on the next cycle.
2. Resume the scheduler with a **single dry-run cycle** first:
   - `python scripts/run_live_paper.py once --dry-run`
3. Inspect the resulting targets. If they match expectations, resume normal scheduling.
4. Optionally switch the next few sessions to a more conservative sizing profile.

### Real event, not safe to resume

1. Keep the kill switch active.
2. Escalate to the on-call contact; resuming trading after a large market event needs a human.
3. Document the decision and the criterion that would allow resumption.

### False positive

1. Identify the bad data point or stale reference explicitly.
2. Fix the upstream: re-ingest the affected day, correct the symbol mapping, tighten the freshness gate.
3. **Do not loosen the threshold** to silence the false positive — the threshold is a policy, not a nuisance.
4. After the fix, reset the breaker via `reset()` and resume with a dry-run cycle.

## Post-Incident

- Write a post-mortem in `docs/post_mortems/YYYY-MM-DD_circuit_breaker_trip.md` covering:
  - cause (real vs false positive)
  - exact trigger values (drop%, vol ratio)
  - whether a resumption happened same-day
  - any policy change that was made
- If the trip was a false positive caused by data, add a regression test in `tests/test_circuit_breaker.py` (or equivalent) with the exact input vector so the same false positive can never re-trip.
- If the trip was a real event and the cool-off rule needs a revision, formalize it in `configs/policy.yaml` under the `circuit_breaker` section and note the decision in an ADR.
- Verify that the alert sinks (runbook 01 style) actually notified the right people within the expected latency. If not, that is a separate gap.
