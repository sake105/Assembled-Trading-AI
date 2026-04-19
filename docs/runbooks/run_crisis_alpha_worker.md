# Runbook: run_crisis_alpha_worker.py

**Script:** `scripts/run_crisis_alpha_worker.py`  
**Status:** Uses `intel/crisis_alpha_worker.py` (v0, deprecated)

> **Note:** This worker uses the v0 crisis_alpha implementation. The v1 lives in
> `events/crisis_alpha/` and is wired into `trading_cycle.py` (shadow-only, T4.1).
> Once T4.1 Step 3 is promoted to production, this worker becomes redundant.
> See `docs/intel/crisis_alpha_scope.md`.

---

## Start

```bash
python scripts/run_crisis_alpha_worker.py
python scripts/run_crisis_alpha_worker.py --dry-run
```

## Kill Switch

The standalone worker does not currently check `policy.intel.kill_switch`.  
To halt: kill the process manually.

## Log Patterns

| Pattern | Meaning |
|---------|---------|
| `[OK] crisis_alpha cycle` | Successful v0 cycle |
| `[ERROR]` | Check traceback for v0 state machine issues |

## Outputs

Writes state to `output/ops/crisis_alpha_state.json` (v0 path).  
The v1 state is at `output/state/crisis_state.json`.

## Relationship to v1 (T4.1)

| | v0 (this worker) | v1 (trading_cycle.py) |
|-|------------------|-----------------------|
| Entry point | run_crisis_alpha_worker.py | trading_cycle.py (after signal gen) |
| Policy gate | None | `intel.crisis_alpha.enabled` |
| Shadow mode | No | Yes (`shadow_only: true`) |
| Tests | Limited | 63 tests in events/crisis_alpha/ |
| State file | output/ops/crisis_alpha_state.json | output/state/crisis_state.json |

## Deprecation Plan

1. T4.1 Step 3 promoted (non-shadow orders) → v1 fully active
2. Verify v0 outputs are no longer consumed by any pipeline
3. Archive `scripts/run_crisis_alpha_worker.py` + `intel/crisis_alpha_worker.py`
