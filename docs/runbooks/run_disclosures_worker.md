# Runbook: run_disclosures_worker.py

**Script:** `scripts/run_disclosures_worker.py`  
**Purpose:** Fetch EDGAR Form 4 + House PTR disclosures, normalize, score, emit triggers.

---

## Start

```bash
python scripts/run_disclosures_worker.py
python scripts/run_disclosures_worker.py --cadence daily
python scripts/run_disclosures_worker.py --no-lock
```

## Kill Switch

Set `policy.intel.kill_switch.enabled: true` in `configs/policy.yaml`.
Worker exits with `[SKIP] kill_switch_active`.

## Lock Recovery

Lock path: `output/intel/disclosures/cache/.disclosures_worker.lock`  
Delete manually if stale (no TTL auto-release on this worker — see T1.3 for news worker).

## Log Patterns

| Pattern | Meaning |
|---------|---------|
| `[SKIP] kill_switch_active` | Kill switch enabled |
| `[SKIP] disclosures_worker already running` | Lock held |
| `[WARN] using stale cache` | HTTP fetch failed, serving cached data |
| `[OK] disclosures pipeline done` | Success |

## Stale Cache Behaviour

When EDGAR or House PTR HTTP fetch fails:
- Returns cached data with `is_stale: True, cached_from_ts: <timestamp>`
- Logs `[WARN] using stale cache (from=<ts>)`
- Downstream consumers should check `is_stale` and log accordingly

## Outputs

```
output/intel/disclosures/
  triggers_latest.json      — scored disclosure triggers
  health.json               — pipeline health
  cache/
    edgar_cache.json        — EDGAR response cache
    house_ptr_cache.json    — House PTR response cache
```

## Troubleshoot

**Stale data:** EDGAR/House PTR HTTP unreachable. Check network; stale cache serves last known data.

**No triggers:** Normal if no qualifying disclosures (Tier-A, severity ≥ 1) in window.

**House PTR empty:** `house_ptr.index_url` may be unconfigured (N4). Check `configs/disclosures/sources.yaml`.
