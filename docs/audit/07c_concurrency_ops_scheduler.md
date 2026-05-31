# 07c — Concurrency / State-Durability / Scheduler / Automation / Alerting Audit

**Round 3, read-only reliability/ops audit.** Static source analysis only, **NOT CI-confirmed**.
Items whose truth depends on runtime/deployment state are marked **UNSURE (execution-dependent)**.
Every finding cites file:line with an evidence quote. Findings prefixed `OPS-`.

Scope: concurrent-write/state-race, atomicity/durability, scheduler/automation, Dead-Man's-Switch,
heartbeat/liveness, alerting sinks, clock/time, resource exhaustion.

---

## Verdict (one line)

**Not robust for unattended running.** The advertised safety net (Dead-Man's-Switch + heartbeat
staleness monitor + multi-channel alerting) is largely **disconnected from the actual production
path**: the DMS daemon is undeployed, two divergent heartbeat files exist of which the production
(GitHub Actions) path writes *neither*, the staleness detector is only ever run against a synthetic
heartbeat, and the kill-switch state file has no locking despite being writable by multiple
processes. The 2026-04-10 7-day silent stall postmortem documents this class of failure as already
having happened once.

---

## Severity legend

- **KRITISCH** — can cause silent trading stop, undetected state corruption, or a safety mechanism that does not fire.
- **HOCH** — real defect with operational impact, workaround exists or trigger is narrow.
- **MITTEL** — correctness/robustness gap, degraded but not silent.
- **NIEDRIG** — hygiene / latent risk / doc-vs-code drift.

---

## KRITISCH

### OPS-01 — Dead-Man's-Switch is dead code in production (undeployed, no scheduler wiring)
**Severity: KRITISCH** — verified at source + deployment-artifact search.

The DMS auto-flatten-on-stale-heartbeat is fully implemented (`src/assembled_core/ops/dead_man_switch.py`)
and has a daemon entry (`scripts/dms_daemon.py`), but **nothing launches the daemon**. A repo-wide
search for the daemon in any `.bat`/`.ps1`/workflow found only the daemon file itself and its test:

- `scripts/dms_daemon.py:44` `from src.assembled_core.ops.dead_man_switch import dms_monitor_loop`
- The Task-Scheduler registration script `scripts/ops/register_paper_pilot_task.ps1:29` registers
  `scripts\daily_paper_trading.bat` — which (read in full) runs `refresh_daily_cache`, `prewarm_price_cache`,
  and `run_paper_pilot.py --run-day`, and **never references `dms_daemon`**.
- No GitHub workflow runs the daemon (`.github/workflows/*` search: no `dms_daemon` hit).

Confirms MEMORY.md ("DMS daemon not yet wired into Task Scheduler (86468b0c)"). The auto-flat-on-stale
code path is **unreachable in production** — it is a spec-grade safety feature presented as
implemented. (CLAUDE.md: "Ein Stub ist keine Integration.")

### OPS-02 — Two divergent heartbeat files; the production path writes neither one the DMS reads
**Severity: KRITISCH** — verified at source.

There are **two independent heartbeat systems with different paths and different schemas**:

1. **DMS heartbeat** — `src/assembled_core/ops/heartbeat.py:33`
   `_DEFAULT_HEARTBEAT_PATH = Path("output") / "state" / "heartbeat.json"`, field `timestamp`.
   Written only by the trading-cycle-v2 pipeline:
   `src/assembled_core/pipeline/_tc_execution.py:526` `_hb_path = ctx.output_dir / "state" / "heartbeat.json"`.
2. **Scheduler heartbeat** — `scripts/paper_trading_scheduler.py:36`
   `HEARTBEAT_PATH = ROOT / "output" / "ops" / "scheduler_heartbeat.json"`, field `timestamp_utc`.

`dms_daemon.py:106` calls `dms_monitor_loop(policy, stop_event=stop_event)` **without** a
`heartbeat_path` → defaults to `output/state/heartbeat.json` (system 1). But:
- The staleness monitor `scripts/check_scheduler_health.py:41` reads `output/ops/scheduler_heartbeat.json` (system 2).
- The production entry `run_paper_pilot.py`/`daily_paper_trading.bat` writes **neither** directly; the
  DMS heartbeat (system 1) is updated only as a side-effect of `_tc_execution.py` running once per daily cycle.

Net effect: even if the DMS daemon *were* deployed (it is not — OPS-01), its 900 s default
`timeout_seconds` (`dead_man_switch.py:46`) would fire **every single day** because the heartbeat it
watches is refreshed at most once per ~24 h, not every 15 min. The two systems and the production
writer are mutually inconsistent. (`scheduler_heartbeat.json` writers, repo-wide: only
`paper_trading_scheduler.py`, plus the drill and docs — confirmed by grep.)

### OPS-03 — Staleness detector never runs against a real production heartbeat
**Severity: KRITISCH** — verified at source.

`check_scheduler_health.py` (the only stale-heartbeat detector) is invoked in exactly one place:
the synthetic drill `scripts/run_alert_drill.py:88-95`, which first **writes its own fake stale
heartbeat** (`_write_stale_heartbeat`, line 60) and then checks it. No scheduled job runs the detector
against the live heartbeat:

- `.github/workflows/*` grep for `check_scheduler_health`/`liveness_check`: **no production cron hit**;
  only `fail-drill.yml` runs `run_alert_drill.py` (the synthetic path).
- The production cron `paper-trading-ci.yml` has only a self-scoped `if: failure()` Discord step
  (`paper-trading-ci.yml:185-194`). If the **cron itself stops firing** (the exact 2026-04-10 stall
  mode — process/trigger gone), there is no failure event, so **no alert is produced at all**.

The drill therefore proves the detector *logic* works but does **not** prove the production system is
monitored. This is precisely the gap called out in the postmortem
(`docs/incidents/2026-04-10_paper_stall_postmortem.md:65-71`, five-whys #3/#4: "there is no staleness
alert … the heartbeat was added as pre-infrastructure for a monitor that was never built").

### OPS-04 — Kill-switch state file has NO locking — concurrent-write race + audit-chain TOCTOU
**Severity: KRITISCH** — verified at source.

`src/assembled_core/execution/kill_switch.py` reads and writes `kill_switch_state.json` and appends to
the hash-chained `kill_switch_audit.jsonl` with **no file lock and no in-process lock** (grep for
`filelock`/`FileLock`/`threading.Lock`/`.lock` in the file: **0 occurrences**).

- `_write_state` (line 83) does a correct atomic tmp+fsync+replace+dir-fsync, but two concurrent
  writers still last-writer-wins on the whole state blob.
- The audit chain is computed read-modify-write **without holding any lock**:
  `_append_audit` (line 168) calls `_last_audit_hash(p)` (line 142, reads the last line) then appends.
  If the DMS (`auto_flatten_on_stale` → `activate_kill_switch`, `dead_man_switch.py:147`) and the
  runner's drawdown check (`check_drawdown_kill_switch` → `activate_kill_switch`, `kill_switch.py:472`)
  race, two appends can read the **same** `prev_hash`, producing a forked/broken chain that
  `verify_audit_chain` (line 200) will then report as tampered. This is a **TOCTOU on a
  safety-critical, integrity-claiming log**.

By contrast the paper ledger *does* lock (`paper_ledger.py:152-170`, `FileLock(..., timeout=10)`),
so the absence here is an inconsistency, not a design choice. In single-process operation the race is
latent; with the DMS daemon + runner both live (the intended design) it becomes reachable.

---

## HOCH

### OPS-05 — `_alert_health_worker` critical alerts reach NO external sink (blind operator)
**Severity: HOCH** — verified at source.

There are **three separate AlertManager implementations** with different capabilities:

- `ops/alerting.py` `AlertManager.fire` → telegram / email / log (real external channels, lines 103-158).
- `ops/alert_manager.py` `AlertManager.alert` → **console log + JSON file only** (lines 66-101). No
  telegram, no email, no webhook.
- `ops/alert_failover.py` `send_with_failover` → Discord → email failover (lines 75-110).

The daily scheduler's alert worker uses the **console/JSON-only** one:
`src/assembled_core/ops/daily_scheduler.py:738` `from src.assembled_core.ops.alert_manager import AlertManager`,
then raises `mgr.alert("CRITICAL", "kill_switch", …)` (line 750) and `mgr.flush_to_json()` (line 831).
So a kill-switch-engaged / reconciliation-failure CRITICAL produced by the daily cycle is **only
written to a JSON file under `output/alerts/` and logged** — it never reaches telegram/email/Discord.
An operator who is not tailing logs or polling that directory is blind. The "blind operator" failure
mode the task brief warns about is realized here for the daily-scheduler alert path.

(Note: the kill-switch's *own* `activate_kill_switch` does call the real sink —
`kill_switch.py:277` `AlertManager().fire("kill_switch_activated", …)` — but wraps it in a
`try/except … logger.debug` that **swallows dispatch failure to DEBUG level**, lines 274-281, so a
broken telegram/email channel is invisible at INFO.)

### OPS-06 — No mutex across scheduler instances; `LOCK_PATH` defined but never used
**Severity: HOCH** — verified at source.

`scripts/paper_trading_scheduler.py:38` defines `LOCK_PATH = ROOT / "output" / "ops" / ".paper_trading_lock"`.
Grep for `LOCK_PATH` in that file: **1 occurrence (the definition only)** — it is never acquired,
checked, or written. Nothing prevents two scheduler daemons (e.g. a leftover one plus a restart) from
both reaching the daily window and each launching `run_live_paper.py once`
(`paper_trading_scheduler.py:174`). The `_already_ran_today` marker (line 71) is the only guard and it
is a non-locked read-then-write, so two near-simultaneous instances can both pass the check before
either writes `last_run_date.txt`. The news/disclosures workers, by contrast, *do* use real locks
(`scripts/run_news_worker.py:219`, `scripts/run_disclosures_worker.py:156`), so this is an omission.

The GitHub Actions path has its own `concurrency: group: paper-trading` guard
(`paper-trading-ci.yml:21-23`), but that does **not** coordinate with a locally-running daemon — the
workflow header itself acknowledges "if the local daemon is also running" (line 11), relying solely on
engine-level intent idempotency, which is outside this audit's verified scope. **UNSURE
(execution-dependent)** whether intent-store idempotency fully closes the double-submit window.

### OPS-07 — Reconciliation mismatch is warning-only in the local daemon path
**Severity: HOCH** — verified at source + postmortem evidence.

`run_live_paper.py` *does* implement halt-on-mismatch with an ack gate
(`_write_halt_flag`, lines 50-54; trip at 613-636; `halt_on_mismatch` default `True`, line 44), and
`paper-trading-ci.yml:75-84` enforces the halt-ack pre-check. **However**, the GitHub-Actions path runs
`run_paper_pilot.py --run-day` (`paper-trading-ci.yml:110`), and the **daily reconcile workflow**
runs `run_reconcile_worker.py --dry-run || true` (`daily-paper-reconcile.yml:41`) — the `--dry-run`
plus `|| true` means a reconciliation break in *that* workflow is swallowed and the job stays green.
The postmortem documents a real $412.54 reconciliation mismatch that "reported … exit_code=0 …
reconcile=OK" and did not halt (`2026-04-10_paper_stall_postmortem.md:30-38`). The halt path exists in
`run_live_paper` but the audited reconcile workflow neutralizes its own exit code.

### OPS-08 — Soft-timeout only checks at discrete stage boundaries; a blocked call still gets hard-killed
**Severity: HOCH** — verified at source.

`run_live_paper.py` arms a soft-timeout Timer (`_arm_soft_timeout`, line 458) that flips
`_SOFT_TIMEOUT_TRIPPED` and writes the halt flag before the Task-Scheduler `ExecutionTimeLimit`
hard-kill (PT15M in `register_paper_pilot_task.ps1:80`, PT30M per the code comment line 462 / workflow
`timeout-minutes: 30`). But the trip is only *observed* at three explicit checkpoints
(`_check_soft_timeout` at lines 538, 564, 579). If a blocking network/order call hangs **between**
checkpoints (e.g. inside `run_paper_daily_one`), the Timer fires and writes the halt flag, but the
main thread stays blocked and is still hard-killed mid-operation — the exact "stale pending intent on
2026-05-19 (mid-submission kill)" the soft-timeout was meant to prevent (comment lines 462-465). The
mitigation is partial. Also note the ExecutionTimeLimit value **disagrees** between the registration
script (15 min, `register_paper_pilot_task.ps1:80`) and the code's own assumption (`soft_timeout_s`
default 1500 s = 25 min, line 528) — under the 15-min OS limit the 25-min soft-timeout **never fires
before the hard kill**. **Severity HOCH** due to this misconfiguration window.

---

## MITTEL

### OPS-09 — Paper ledger save: backup rotation by copy, then atomic write — crash window + non-atomic multi-file update
**Severity: MITTEL** — verified at source. (Extends R2-18.)

`paper_ledger.py:save_ledger_state` (line 138) holds the file lock and does:
`_rotate_backups(p)` (line 162, three `shutil.copy2` ops, lines 113-135) **then** `tmp.write_text`
+ `tmp.replace(p)` (lines 163-165). Two issues:
1. **No fsync** on the tmp file or directory before/after `replace` (R2-18 confirmed) — a power loss
   after `replace` returns but before the page-cache flush can lose the just-written state on some
   filesystems. (Contrast `kill_switch._write_state` lines 101-122 which *does* fsync — inconsistent
   durability guarantees across two state files.)
2. The rotation copies (`.1/.2/.3`) and the main write are **not a single atomic unit**: a crash
   between `_rotate_backups` and `tmp.replace` leaves `.1` already overwritten with the *current*
   (about-to-be-superseded) state while the main file is unchanged — recoverable, but the "3
   generations" invariant is momentarily violated. The lock prevents concurrent corruption but not
   crash-window inconsistency.

### OPS-10 — No fill-level dedupe in `apply_fills_to_ledger`; idempotency relies on caller
**Severity: MITTEL** — verified at source. (Confirms R2-19.)

`paper_ledger.apply_fills_to_ledger` (line 241) iterates `fills` and mutates cash/positions with **no
fill-id / dedupe key** — replaying the same fills list (e.g. a retried cycle after a partial write)
double-applies them. The equity *curve* is deduped by date (`append_equity_curve_deduped`, line 361),
but the cash/position ledger is not. Idempotency is entirely delegated to whoever builds the `fills`
list and to the engine's intent-store (out of scope here). R2-19 confirmed at source.

### OPS-11 — DMS escalation logs "kill switch is already active" even in shadow mode (false assurance)
**Severity: MITTEL** — verified at source.

In `dead_man_switch.py` the outer-failure escalation path (lines 274-298) calls
`auto_flatten_on_stale(...)` then unconditionally logs
`"[DMS-CRITICAL] DMS is now in degraded mode … kill switch is already active."` (lines 295-297).
But `auto_flatten_on_stale` in `flatten_mode == "shadow"` (line 134) **returns without activating the
kill switch** (lines 134-143). So under `flatten_mode: shadow` the escalation log asserts the kill
switch is active when it is not — a false-assurance log that would mislead an operator reading the
audit trail during an incident. The policy default is `market` (line 48), so this bites only when
shadow mode is configured, but the log is unconditional.

### OPS-12 — DMS daemon SIGTERM handler does not fire on Windows forced kill; relies on "stateless" claim
**Severity: MITTEL** — verified at source.

`dms_daemon.py:101-103` registers SIGTERM/SIGINT but the inline comment (line 101) admits
"Windows note: SIGTERM handler … may not fire on forced kill (taskkill /F). Stateless design means no
data loss." The loop *is* largely stateless, but on `taskkill /F` the daemon dies without writing any
"DMS stopped/last-seen" marker, so there is **no way to distinguish a cleanly-stopped DMS from a
crashed one** — and (per OPS-01) nothing monitors the DMS's own liveness anyway. A watchdog that
itself silently dies is the classic second-order failure.

### OPS-13 — Two distinct paper ledgers (JSON vs SQLite) — divergent reconcile sources
**Severity: MITTEL** — verified at source.

The live runner persists ledger state as JSON at
`output/runs/_paper_ledger/ledger_state.json` (`run_live_paper.py:592`, via `paper_ledger.save_ledger_state`).
The daily-scheduler reconcile worker instead reads a **SQLite** ledger:
`daily_scheduler.py:293-298` `db_path = Path(output_dir) / "paper_ledger.db"` via `LedgerStore`. These
are two different stores with no synchronization shown in this surface; a reconcile run against the
SQLite DB tells you nothing about the JSON state the runner actually mutates. **UNSURE
(execution-dependent)** whether some upstream step keeps them in sync — not visible in the audited
files. At minimum this is a "zweite Wahrheit" (CLAUDE.md Rule 50) for ledger state.

---

## NIEDRIG

### OPS-14 — `daily_scheduler.schedule_loop` uses wall-clock `time.sleep(interval_hours*3600)` with no drift correction
**Severity: NIEDRIG** — verified at source.

`daily_scheduler.py:1089` `time.sleep(interval_hours * 3600)` between cycles. Cumulative drift over
many iterations and no catch-up if the machine sleeps/hibernates; acceptable for a coarse loop but not
wall-clock-anchored. The per-worker durations correctly use `time.monotonic()` (e.g. line 33), so the
measurement side is fine; only the *cadence* side is wall-clock-naive.

### OPS-15 — `clock_drift` is a good helper but is not wired into any ops/state path
**Severity: NIEDRIG** — verified at source + grep.

`utils/clock_drift.py` is a clean stdlib NTP drift detector (`measure_drift_seconds`, line 43;
`drift_status`, line 95) and is UTC-aware. However it appears to be a library with no caller in the
scheduler/heartbeat/DMS path (no import found in the audited ops files). The DMS/heartbeat staleness
math (`heartbeat.heartbeat_age_seconds`, line 99) trusts the local wall clock; a silently-skewed local
clock would mis-age the heartbeat with no drift guard applied. Helper exists; integration does not.

### OPS-16 — `daily_scheduler` subprocess workers swallow non-zero return codes into an error string only
**Severity: NIEDRIG** — verified at source.

`_news_fetch_worker` (line 92) runs three subprocesses and collects `rc != 0` into a joined
`error_msg` string (lines 117-128) but the overall `run_daily_cycle` continues to the next worker
regardless (`daily_scheduler.py:1000-1021`). A failed news/sentiment refresh degrades silently to an
"error" WorkerResult that does not stop the cycle or alert externally (alerting gap → OPS-05). This is
log-visible, so NIEDRIG, but it is a "warn-and-continue" pattern on a data-freshness dependency.

---

## POSITIVE confirmations (things done right)

- **PC-01** — `heartbeat.write_heartbeat` (`heartbeat.py:78-80`) uses tmp+`os.replace` atomic write and
  ages from the in-file `timestamp` (line 116), not mtime — survives file copies, tz-safe.
- **PC-02** — `kill_switch._write_state` (`kill_switch.py:101-122`) does full durability: data fsync +
  atomic `os.replace` + directory fsync, with graceful Windows fallback. Return value is checked by
  callers (`activate_kill_switch` lines 282-296, with CRITICAL log on failure).
- **PC-03** — Kill-switch deactivation is token-gated (`deactivate_kill_switch`, lines 314-344) with
  `hmac.compare_digest` and fail-closed + audit-before-raise. Activation is ungated (correct: you want
  the safety stop to be frictionless).
- **PC-04** — Audit log is hash-chained and self-verifiable (`_append_audit` / `verify_audit_chain`,
  lines 168-235) and fsync'd — good *intent*, undermined only by the missing lock (OPS-04).
- **PC-05** — `paper_trading_scheduler._mark_today_done` (line 83) and `_write_heartbeat` (line 108)
  both use atomic tmp+replace and only advance `last_run_date` on `exit_code == 0` (lines 212-224),
  correctly fixing the "crashed cycle skips retry" (E3) mode.
- **PC-06** — `dms_monitor_loop` never swallows flatten errors silently: failures are logged and
  retried next interval, with an escalation after N consecutive `check_liveness` failures
  (`dead_man_switch.py:270-298`). The design intent is sound (the gap is deployment, OPS-01).
- **PC-07** — `run_live_paper` soft-timeout writes a halt-ack flag that blocks the *next* run's
  preflight (`paper-trading-ci.yml:75-84`) — a real "fail-forward-safe" mechanism (the limit is the
  checkpoint granularity, OPS-08).
- **PC-08** — Alert failover (`alert_failover.py:75-110`) tries Discord then email and logs an ERROR
  when *all* channels fail (line 106) instead of swallowing — correct loud-fail behavior. The weekly
  `fail-drill.yml` keeps the path warm.
- **PC-09** — `register_paper_pilot_task.ps1:62-68` correctly uses `powershell.exe` (UTF-16) over
  `cmd.exe` to survive the umlaut in the repo path `Aktiengerüst` — the codepage bug from the
  2026-05-15 session is fixed at the registration layer.

---

## Cross-cutting summary for unattended running

| Concern | State | Key finding |
|---|---|---|
| Auto-flat safety net (DMS) | **Not deployed** | OPS-01, OPS-02 |
| Liveness monitoring (real path) | **Not wired** | OPS-02, OPS-03 |
| Kill-switch concurrency | **Unsafe (no lock)** | OPS-04 |
| External alerting from daily cycle | **Blind (JSON/log only)** | OPS-05 |
| Multi-instance scheduler guard | **Unused `LOCK_PATH`** | OPS-06 |
| Reconcile-halt in CI reconcile job | **Neutralized (`--dry-run \|\| true`)** | OPS-07 |
| Hard-kill protection | **Partial + misconfigured limit** | OPS-08 |
| Ledger durability | **fsync gap (JSON)** | OPS-09, R2-18 |
| Fill idempotency | **Caller-dependent** | OPS-10, R2-19 |

The atomic-write and durability primitives are individually well-built (PC-01..PC-08); the systemic
failure is **integration and deployment**: the safety mechanisms are not connected to the path that
actually trades, and the one documented real-world incident (7-day silent stall) is the materialized
form of exactly this gap.

---

*Method: Grep/Glob to locate, Read to verify. All findings cite file:line. Static-only, not CI-run.
Deployment-state findings (OPS-01/02/03/06 production wiring, OPS-13 ledger sync) are evidenced by
absence-of-reference grep and are marked UNSURE where runtime could differ.*
