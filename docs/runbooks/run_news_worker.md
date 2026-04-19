# Runbook: run_news_worker.py

**Script:** `scripts/run_news_worker.py`  
**Purpose:** Fetch, normalize, deduplicate, cluster, and score news events.

---

## Start

```bash
python scripts/run_news_worker.py                     # hourly cadence (default)
python scripts/run_news_worker.py --cadence daily     # daily cadence
python scripts/run_news_worker.py --no-lock           # skip file lock (testing only)
```

## Kill Switch

Set `policy.intel.kill_switch.enabled: true` in `configs/policy.yaml`.
The worker reads the flag at startup and exits with `[SKIP] kill_switch_active`.

To halt immediately: kill the process; clear the lockfile manually if needed.

## Lock Recovery

If the worker crashed and left a stale lock:
```
output/intel/news/cache/.news_worker.lock
```

The lock auto-releases after 2 hours (TTL) if the PID is dead. To force:
```bash
rm output/intel/news/cache/.news_worker.lock
```

## Log Patterns

| Pattern | Meaning |
|---------|---------|
| `[START] news_worker cadence=...` | Worker started |
| `[SKIP] kill_switch_active` | Kill switch enabled in policy |
| `[SKIP] news_worker already running` | Another instance holds the lock |
| `[WARN] stale lock released` | Auto-released stale lock |
| `[OK] news_worker done` | Successful completion |
| `[WARN] dedupe_store prune failed` | SQLite prune issue — non-fatal |
| `[ERROR] news_worker` | Fatal error — check traceback |

## Outputs

```
output/intel/news/
  news_health.json          — pipeline health + trigger summary
  events_latest.json        — most recent scored events
  cache/
    dedupe.db               — SQLite WAL deduplication store
    .news_worker.lock       — PID lockfile
```

## Troubleshoot

**Worker never starts:** Check lockfile PID. If stale, delete the lockfile.

**`database is locked`:** SQLite WAL should prevent this. Check if two processes are writing simultaneously; ensure WAL mode is active (`PRAGMA journal_mode;`).

**No events output:** Check `news_health.json.status`. If `degraded`, a source fetcher failed. Check logs for `[WARN]` lines.

**Vacuum:** Runs automatically on the 1st of each month. Force with internal API or restart with `--cadence daily`.
