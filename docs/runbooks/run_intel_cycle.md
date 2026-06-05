# Runbook: run_intel_cycle.py

**Script:** `scripts/run_intel_cycle.py`  
**Purpose:** Fetch GDELT events every 15 minutes, update crisis state, write geo-trigger artifacts.

> ⚠️ **PIT note (crisis_state.json):** this script writes `data/intel/crisis_state.json` — the SAME
> path `_load_intel` reads for `ctx.news_geo` / `ctx.crisis_state_intel`. The snapshot is a single
> "latest" state with NO point-in-time index. It is intentionally injected ONLY in live/EOD; in a
> backtest (`as_of` set) it is NOT injected and the intel flag is DEGRADED (since commit `83f3c2c8`),
> because `CrisisState` carries no availability timestamp to filter on. So running this intel cycle
> and THEN backtesting is PIT-safe now — but do not hand-feed a `crisis_state.json` into a backtest
> expecting historical-accurate crisis state; it would be today's snapshot.

---

## Start

```bash
python scripts/run_intel_cycle.py              # run once
python scripts/run_intel_cycle.py --loop       # continuous (15-min intervals)
python scripts/run_intel_cycle.py --dry-run    # fetch but do not write artifacts
python scripts/run_intel_cycle.py --loop --interval 300  # custom interval (seconds)
```

## Kill Switch

Set `policy.intel.kill_switch.enabled: true` — worker exits before first cycle with `[SKIP] kill_switch_active`.

## Log Patterns

| Pattern | Meaning |
|---------|---------|
| `[START] Intel loop` | Loop mode started |
| `[SKIP] kill_switch_active` | Policy kill switch active |
| `[OK] Cycle complete in X.Xs` | Successful cycle |
| `[WARN] Could not load dependency graph` | Graph file missing — propagation disabled |
| `[WARN] Could not load crisis_state.json` | State file corrupt — fresh state used |
| `[ERROR] Cycle failed` | Exception in cycle — loop continues |
| `[OK] Artifacts written` | Artifacts saved successfully |

## Outputs

```
data/intel/
  triggers_latest.json      — geo triggers (schema: news.triggers.v1)
  crisis_state.json         — current crisis mode + risk posture
  dependency_signal.json    — beneficiaries/losers (if geo_score >= 1)
  intel_health.json         — component freshness

data/intel/state/
  gdelt_state.json          — GDELT fetch state (last seen event IDs)
  dedupe_index.json         — in-memory dedupe index snapshot
```

## Trigger Snapshot Archival (T6.1)

Use `TriggerSnapshotStore` to archive `triggers_latest.json` per run_id before it's overwritten:

```python
from src.assembled_core.intel.trigger_snapshot_store import TriggerSnapshotStore
store = TriggerSnapshotStore("output/intel/snapshots")
store.archive("gdelt", run_id, Path("data/intel/triggers_latest.json"))
```

## Troubleshoot

**`crisis_state.json` corrupt:** Delete the file; worker creates a fresh NORMAL state.

**GDELT fetch errors:** Transient — GDELT API has rate limits. Loop mode retries next interval.

**Dependency graph missing:** `configs/dependency_graph.yaml` not found → shock propagation disabled. Geo triggers still work.

**Crisis mode stuck:** Check `crisis_state.json` → `mode` field. If stuck in CRISIS: verify `geo_score` in latest cycle; if conditions cleared, state should transition to COOLDOWN after cooldown period.
