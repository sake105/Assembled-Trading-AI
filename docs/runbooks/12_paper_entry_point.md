# 12 — Paper Trading Entry Point (authoritative)

This runbook is the **single authoritative reference** for how a paper
trading cycle is invoked. Past incidents (notably the 2026-04-10 to
2026-04-17 stall) were partly enabled by ambiguity about which script
actually drives the paper cycle. This document resolves that.

---

## 1. Authoritative entry point

```
python scripts/run_live_paper.py once
```

This is the only supported command for a **single paper-trading cycle**
in both CI and local contexts. All other runners are either wrappers,
specific sub-tools, or legacy.

Supporting subcommands of the same script:

- `python scripts/run_live_paper.py reconcile` — position reconcile only
- `python scripts/run_live_paper.py rebuild-ledger` — emergency ledger rebuild from broker

**Not** authoritative:

- `scripts/paper_trading_scheduler.py` — long-running local daemon;
  now a **fallback** to GH Actions, not the primary driver.
- `scripts/run_eod_pipeline.py` / `scripts/batch_backtest.py` — research
  / backtest tools, not paper entry points.
- `scripts/daily_paper_trading.bat` — Windows convenience wrapper.

---

## 2. Primary scheduler: GitHub Actions

`.github/workflows/paper-trading-ci.yml` is the primary scheduler as of
2026-04-17.

- Two cron lines cover EDT and EST:
  - `30 19 * * 1-5` (EDT 15:30 ET)
  - `30 20 * * 1-5` (EST 15:30 ET)
- The ET time-gate step aborts with exit 78 outside `[15:20, 15:59]` ET
  or on weekends, so exactly one of the two crons actually runs any
  given weekday.
- `concurrency: { group: paper-trading, cancel-in-progress: false }`
  prevents overlap if the local fallback daemon is also running.

Secrets required (set via `gh secret set`):

- `ALPACA_API_KEY`
- `ALPACA_API_SECRET` (matches the env var read by `AlpacaAdapter` and
  the workflow YAML — NOT `ALPACA_SECRET_KEY`)
- `DISCORD_WEBHOOK` (optional; for failure alerts)

### Manual trigger

```
gh workflow run paper-trading-ci.yml
```

This bypasses the ET-gate only if invoked via `workflow_dispatch` inside
the 15:20–15:59 ET window. Outside the window the ET-gate step still
exits 78.

### Expected artifacts per run

- `output/ops/` — scheduler heartbeat, last_run_date, halt flag if any
- `output/paper/` — paper engine equity, ledger, manifest
- `output/executions/` — intent store, real Alpaca fill records
- `output/runs/` — run manifests
- `output/ops/alpaca_eod_<date>.json` — EOD balance snapshot (A4)

---

## 3. Local fallback daemon

`scripts/paper_trading_scheduler.py` remains installed as a **fallback**:

- Start: `python scripts/paper_trading_scheduler.py`
- Writes `output/ops/scheduler_heartbeat.json` every loop iteration.
- Calls the same authoritative entry point (`run_live_paper.py once`)
  internally — no divergent code path.

The local daemon must not be treated as authoritative. If GH Actions
ran today (see `output/ops/last_run_date.txt`) the local daemon's
`_already_ran_today` guard will skip; this is intended.

---

## 4. Health monitoring

Two independent stall detectors run against this entry point:

1. `scripts/check_scheduler_health.py` — heartbeat staleness monitor.
   Reads the heartbeat the deployed pilot actually writes
   (`output/state/heartbeat.json`, via `_tc_execution`). Exit 1 when
   stale, exit 2 when missing/unparseable, exit 0 when fresh. Wired in
   production (OPS-03) as an **independent** Windows Task
   (`AssembledTradingAI-HealthCheck`, register via
   `scripts/ops/register_health_check_task.ps1`) running the wrapper
   `scripts/check_scheduler_health.bat` Mon–Fri 22:30 local — ~1h after
   the 21:30 pilot window — with `--ignore-market-hours --stale-minutes
   1080 --notify`. Because the pilot writes the heartbeat once per day, a
   healthy day's heartbeat is minutes old at check time while any stall
   (no run today) leaves yesterday's heartbeat ≥ ~24h old. The task is
   deliberately separate from the pilot task so that if the pilot stops
   firing (the 2026-04-10 silent-stall mode) the watchdog still runs and
   alerts via Discord (`DISCORD_WEBHOOK`) with SMTP email fallback.
2. `scripts/snapshot_alpaca_balance.py` — runs after each paper cycle
   and writes `output/ops/alpaca_eod_<date>.json`. A cash delta >
   $50 vs. local state triggers a warning log line (not a gate — the
   reconcile-halt policy in E0.4 is the gate).

Absence of a recent `alpaca_eod_<date>.json` is itself a stall signal
independent of the heartbeat.

---

## 5. Halt acknowledgement

If the reconcile-halt policy (E0.4) engages, `output/ops/halt_ack_required.json`
is written and the next paper-trading-ci run's halt-ack gate aborts
with exit 1.

Clearing the flag is manual and must be audited:

```
python scripts/ack_halt.py --reason="reviewed_YYYY-MM-DD_<topic>"
```

Each clear writes an entry to `output/ops/halt_ack_ledger.jsonl` with
the prior flag payload embedded, the actor, and an ISO UTC timestamp.

Never remove `halt_ack_required.json` manually — always use the CLI so
the ledger is preserved.

---

## 6. Do / Don't

### Do

- Run `python scripts/run_live_paper.py once` (or the GH Actions workflow)
  as the single paper-cycle entry.
- Treat `output/ops/last_run_date.txt` as the single source of truth for
  "did today's cycle run".
- Use `scripts/ack_halt.py` to clear halt flags.

### Don't

- Don't invoke engine internals directly from ad-hoc scripts for paper
  trading — the entry point is responsible for run-manifest, reconcile,
  and halt-flag semantics.
- Don't manually delete `output/ops/halt_ack_required.json`.
- Don't silence the heartbeat monitor when investigating a stall —
  capture the stale heartbeat file first.

---

## 7. Change history

- 2026-04-17: Runbook created. GH Actions promoted to primary scheduler
  following 7-day stall. Local daemon demoted to fallback.
