# Incident: 7-Day Silent Paper-Trading Stall (2026-04-10 → 2026-04-17)

**Status:** Post-mortem, evidence-based
**Severity:** P0 — Operational blindness, trading stopped for 5 business days without alert
**Authoring finding:** System-Check Deep Run v2, A4 (2026-04-18)
**Related findings:** A2 (signal-decay-cron undeployed), A13 (no synthetic fail-drill)

---

## 1. What happened

- Last successful paper-trading cycle: `2026-04-10 21:39:18 UTC` (= 15:39 ET)
  (evidence: `output/ops/scheduler.log` tail; `output/ops/last_run_date.txt = "2026-04-10"`)
- Heartbeat last "alive" timestamp: `2026-04-12T16:32:41Z` (a Sunday — not a market cycle,
  only the ambient process pulse; evidence: `output/ops/scheduler_heartbeat.json`).
- Scheduler log terminates at `2026-04-12 18:00:01 UTC` with weekend-skip messages; no
  further entries through the detection date (`2026-04-17`).
- No alert, no artifact, no kill-switch fire, no PR, no ticket, no heartbeat-staleness
  page-out. User noticed manually on `2026-04-17`.
- Business days lost: `2026-04-13, -14, -15, -16, -17` = **5 trading days silent**.

## 2. Concurrent reconciliation mismatch (2026-04-10 cycle)

The last successful cycle wrote two separate reconciliation warnings (evidence:
`output/ops/scheduler.log`):

```
[WARNING] [position_sync] MISMATCH — cash_diff=412.54, 8 position diffs,
                          6 missing_in_ledger, 0 missing_in_broker
[WARNING] [run_live_paper] POST-EXECUTION MISMATCH: Reconciliation FAILED:
  Cash mismatch: diff=412.537574 (ledger=39969.037574, broker=39556.500000);
  Position qty mismatches: 8 symbol(s): ['ABBV','CVX','JNJ','JPM','LLY','NFLX','WMT','XOM'];
  Missing in ledger: 6 symbol(s): ['COST','KO','MCD','MRK','PEP','V']
```

Despite that, the cycle reported `BROKER cycle complete — exit_code=0 reconcile=OK` and
`=== Daily cycle COMPLETE (success) ===`. The reconciliation subsystem was warning-only —
confirming P0 finding O2 from the Ultra-Plan (reconciliation-mismatch does not halt).

Causal relationship to the stall is **not proven**; the stall started on the next trading
day (`2026-04-13`). Two hypotheses remain open (see §5).

## 3. Timeline (UTC)

| Time | Event |
|------|-------|
| 2026-04-09 18:30:06 | Scheduler started, execution time 15:30 ET, 300s check-interval |
| 2026-04-10 21:30:32 | First daily cycle (intra-day retry) — success, exit 0 |
| 2026-04-10 21:36:58 | Same cycle observed MISMATCH warning |
| 2026-04-10 21:39:18 | Second cycle — success, exit 0, MISMATCH $412.54 |
| 2026-04-11–04-12 | Weekend — scheduler skips, heartbeat pulses |
| 2026-04-12 18:00:01 | Last log entry — weekend skip |
| 2026-04-13–04-17 | **No log entries, no cycles, no alerts** |
| 2026-04-17 | User notices stall by manual inspection |
| 2026-04-18 | Ultra-Plan v3 drafted; System-Check Deep Run v2 flags A4 |

## 4. Five whys

1. **Why did paper-trading stop?**
   Because the local scheduler process exited between 2026-04-12 18:00 UTC and the
   2026-04-13 open; no new process replaced it.
2. **Why did nothing restart it?**
   There is no watchdog / supervisor / service-unit configured for
   `scripts/paper_trading_scheduler.py`. It runs as a user-launched foreground process.
3. **Why didn't anyone notice?**
   Because there is no staleness alert. `check_scheduler_health.py` does not exist;
   `scheduler_heartbeat.json` is written but not consumed by any external monitor.
4. **Why is a heartbeat written if nobody reads it?**
   Because the heartbeat was added as pre-infrastructure for a monitor that was never
   built (CLAUDE.md §2.1: *spec is not implementation*). The plan was A2-A4 of
   Ultra-Plan v3 Part A — not yet executed at the time of the stall.
5. **Why wasn't any of this surfaced by CI?**
   Because CI does not exercise the scheduler path. No synthetic fail-drill exists
   (see A13). The only scheduler evidence is the local `scheduler.log` tail.

## 5. Open hypotheses for the process exit

Neither has been confirmed. Both are listed for future investigation.

- **H1 (operational):** The machine was shut down, rebooted, or the user-launched
  foreground shell was closed at some point on 2026-04-12. Standard user behavior for a
  PC running an ad-hoc Python daemon.
- **H2 (cascading reconcile-panic):** The 2026-04-10 $412.54 mismatch may have
  corrupted local state (ledger/position snapshot) such that the next non-weekend cycle
  would have raised inside the pipeline and silently exited. The log tail does not show
  any 2026-04-13 entries at all, which argues more strongly for H1 than H2.

Evidence needed to distinguish:

- OS event log for `2026-04-12` / `2026-04-13` (Windows Event Viewer, System log).
- `python.exe` PID `111832` (last heartbeat PID) — whether it is still running or when
  it exited. Given the date, this evidence may no longer be recoverable.

## 6. Preventive actions (linked to Ultra-Plan Part A + Deep Run v2 findings)

| ID | Action | Effort | Blocks |
|----|--------|--------|--------|
| A1 (gitleaks + rotation) | Harden secret hygiene — unrelated but required first | 5h | — |
| A2 (signal-decay-cron) | Move trading cadence off local scheduler to GitHub Actions | 2h | — |
| A4 (this postmortem) | Evidence-based record | 2h | **Done** |
| O1 (staleness alert) | `scripts/check_scheduler_health.py` + 5-min external cron | 1d | Detect stall |
| O2 (reconcile-halt)  | `policy.reconciliation.halt_on_mismatch: true` + ack-gate | 0.5d | Contain cascade |
| O10 (alpaca EOD snapshot) | Daily `snapshot_alpaca_balance.py` artifact | 0.5d | Second stall detector |
| A13 (synthetic fail-drill) | GitHub-Actions-triggered alert-path test, weekly | 2d | Verify alert works |

## 7. Explicitly out of scope for this postmortem

- Root cause of the `$412.54` cash diff. That is a separate reconciliation-forensics
  task. A backup of `output/state/` at the time of the stall was not taken.
- Whether the stall cost alpha. Without orders, the 5 lost business days are a no-op on
  the paper P&L, not a drawdown.
- Git history rewrite of `.env`. Separate (see A1 incident doc).

## 8. Decision log

1. The Ultra-Plan v3 Part A sequence (GitHub-Actions primary scheduler, heartbeat
   monitor, reconcile-halt) is the correct structural fix. It was planned before the
   stall; the stall validates its priority.
2. Until Part A is deployed, **the local scheduler should be treated as unreliable**.
   Continuing to rely on a user-launched foreground process is not acceptable for a
   system that claims production-readiness.
3. Reconciliation must be moved from warning-only to halt-on-mismatch (Ultra-Plan E0.4)
   before the next D-phase flag-flip (Correlation-Guard, Zombie-Killer, etc.). A
   shadow-enabled module that writes to state while reconcile is warning-only is a
   silent corruption vector.
4. This postmortem is evidence, not ceremony. Marking A4 "done" does not reduce risk —
   the linked preventive actions (O1, O2, A13) must be executed.

## 9. Verification checklist for "stall-safe" state

The system is **not** stall-safe until all of the following are true:

- [ ] GitHub-Actions `paper-trading-ci.yml` runs on cron and is the primary path.
- [ ] `check_scheduler_health.py` cron fires an alert when heartbeat > 10 min stale in market hours.
- [ ] `policy.reconciliation.halt_on_mismatch = true` with ack-gate file.
- [ ] `snapshot_alpaca_balance.py` runs post-cycle, diffs against local state, alerts on > $50 drift.
- [ ] Weekly synthetic fail-drill verifies the alert path end-to-end.
- [ ] A similar stall (scheduler exit on Friday evening, 5-day gap) triggers an alert within 15 min of the first missed market-hour heartbeat.
