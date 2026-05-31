# Runbook 06: Data Feed Stalled

**Severity:** high
**ETA to resolution:** 15–45 min
**On-call contact:** data-ops
**Component:** `src/assembled_core/data/freshness_monitor.py`, `data/prices_ingest.py`, `data/universe.py`, the price-source adapters under `data/sources/`

## Symptoms

- `freshness_monitor.build_cache_freshness_monitor().check_all()` returns a non-empty alert list (entries with `status` `stale` or `unknown`).
- `trading_cycle` Phase 6 (as_of filter) emits `[WARN] stale_prices` or drops symbols.
- `run_kpis.json` shows `universe_size` dropping from day to day while the watchlist is unchanged.
- `prices_ingest` raises `ValueError: missing required columns` or returns empty frames.
- Last `timestamp` in the prices parquet/csv is older than the previous trading day for multiple symbols.

## Immediate Actions (first 5 min)

1. Stop the scheduler to avoid firing the cycle on stale data.
2. Snapshot the current state:
   - `cp -r data/prices data/prices.snapshot_$(date +%s)` (or the platform equivalent).
3. Identify the affected caches:
   - Run `python -c "import json; from src.assembled_core.data.freshness_monitor import build_cache_freshness_monitor; print(json.dumps(build_cache_freshness_monitor().check_all(), indent=2))"` — each alert names the cache, its `status` (`stale` vs `unknown`), `age_hours`, and the on-disk `path`.
4. Check the source adapter's last successful fetch timestamp:
   - Look for `data/sources/<source>_last_success.json` or equivalent state file.
5. Verify the upstream provider has not had an outage (Polygon, AlphaVantage, Alpaca data, whoever owns the feed).

## Diagnosis

1. Is the stall on **all** symbols or a subset?
   - **All symbols:** likely the adapter / network / credential problem. Skip to step 3.
   - **Subset:** likely a per-symbol upstream issue (delisting, halted, ticker change). Check corporate actions.
2. Is the stall on the **latest** day only or on an older window?
   - **Latest only:** upstream may not have published yet; verify with a manual API call.
   - **Older window:** persisted data loss; restore from the snapshot directory or re-fetch.
3. Adapter health:
   - Check `output/logs/assembled.log` for the adapter name and the most recent exception.
   - Confirm the adapter's credentials still work: make one direct GET against the upstream API with the same token.
4. Corporate actions:
   - If stale symbols coincide with recent splits, mergers, or delistings, that is the root cause — the feed did not drop, the ticker did.

## Resolution

### A) Single-adapter failure, upstream healthy

1. Re-run the ingest job for the stale window:
   - `python scripts/run_eod_pipeline.py --as-of <date> --refresh-prices`
2. Verify freshness:
   - `build_cache_freshness_monitor().check_all()` returns an empty list (all caches within budget).
3. Re-enable the scheduler.

### B) Upstream outage

1. Switch to a fallback source if one is wired in the adapter registry.
2. If no fallback exists, pause the scheduler and wait for upstream recovery.
3. Document the pause in the run log and open a tracking note in `KNOWN_ISSUES.md`.

### C) Corporate-action-driven stall

1. Update the universe / watchlist to reflect the ticker change.
2. For delistings, the symbol kill-switch (`execution/symbol_kill_switch.py`) may be the right tool to stop further order generation.
3. Re-run the cycle in dry-run and confirm only the affected names are excluded.

## Post-Incident

- Add an entry to `KNOWN_ISSUES.md` with the affected symbols, window, and root cause.
- If the adapter has no freshness probe yet, add one — silent staleness must fail loud.
- Write a post-mortem in `docs/post_mortems/YYYY-MM-DD_data_feed_stall.md` if the stall affected live trades or if it blocked a cycle for more than one session.
- Consider whether `freshness_monitor`'s threshold should be tightened; a stall that took hours to detect is a monitoring gap, not just a data gap.
