# Operator Runbook

> Audit C3-088 — what the operator does **daily**, what they do **on
> alert**, and what they do **in emergency**. References the safety
> hooks introduced through the Wave 1–7 audit sweep.
>
> This is a runbook, not a strategy doc. It assumes the system is
> already deployed and tested. For setup → see `docs/OPERATIONS_BACKEND.md`.

---

## 1. Daily checklist (start of trading day)

Before any market interaction:

1. **Kill-switch state** — `GET /api/v1/kill-switch/state` MUST be
   `engaged=false`. If engaged, identify activator + reason in the
   audit log and decide:
   - planned (drill / off-hours): deactivate before market open.
   - unexpected: investigate before deactivating.
2. **Readiness** — `GET /ready` MUST return 200. A 503 lists which
   sub-check failed:
   - `kill_switch: false` → kill-switch state file unreadable.
   - `disk_quota: false` → output/ dir at ≥90% usage; rotate logs first.
3. **Audit chain integrity** — run
   ```bash
   python -c "from src.assembled_core.execution.kill_switch import verify_audit_chain; print(verify_audit_chain())"
   ```
   Expect `(True, n)`. Anything else = halt, escalate (FMEA row 2).
4. **Data source health** —
   `python scripts/check_data_sources_health.py`. Exit 0 = green;
   non-zero = at least one critical provider down.
5. **Clock drift** —
   ```python
   from src.assembled_core.utils.clock_drift import drift_status
   print(drift_status())
   ```
   Expect `status="ok"`. `warn`/`fail` = investigate NTP daemon (FMEA row 6).
6. **Drift / freshness** — `GET /api/v1/monitoring/drift_status?freq=1d`.
   200 with `overall_severity ∈ {NONE,MODERATE}` is acceptable; SEVERE
   means promote-blocking.
7. **Reconciliation audit tail** —
   ```bash
   tail -n 5 output/ops/reconciliation_audit.jsonl
   ```
   Last entry should be `severity=ok`. `warn`/`fail` → see §3.

If any of the above fails, **do not start trading**. Resolve, document
in the journal, then re-check.

## 2. End-of-day checklist

1. Verify the day's reconciliation tail is clean.
2. Trade summary: `GET /api/v1/oms/blotter` — reconcile with broker
   statement (manual today, automated future C3-054).
3. Generate journal entry per `journal/README.md` if any decision
   was made today.
4. If a position closed, copy `docs/POST_TRADE_REVIEW_TEMPLATE.md`
   into `journal/<date>-<symbol>.md` and fill the decision-side **as
   you remember it at entry time** (no hindsight).
5. Verify audit-log fsync surfaces are non-empty:
   ```bash
   wc -l output/ops/*.jsonl
   ```

## 3. Alert response (by severity)

Alerts arrive via Telegram / Email per
`configs/alerting.yaml`. Severity routing is defined there; the
response below is keyed by **rule name**.

### `kill_switch_activated`
- Read audit tail to identify activator.
- If activator was `drawdown_check`: respect the verdict, do not
  override without explicit risk-officer approval.
- If activator was `system` or unknown: investigate logs (FMEA row 1).

### `reconciliation_warn` / `reconciliation_fail`
- Pull the latest `output/ops/reconciliation_audit.jsonl` entry.
- For `warn`: investigate today's fills vs broker; usually a timing
  difference that resolves at next reconciliation tick.
- For `fail` repeated 3× consecutively (planned auto-escalation
  C4-031 long-tail): expect kill-switch to engage automatically.
  If it does not, do it manually:
  ```bash
  curl -X POST -H "X-API-Key: $ASSEMBLED_API_KEY" \
       "http://localhost:8000/api/v1/kill-switch/activate?throttle_pct=0&reason=recon-3-strikes&actor=operator"
  ```

### `circuit_breaker_tripped`
- The breaker stays tripped for `cooldown_minutes` (default 30 min).
- During cooldown: no manual override unless a senior reviewer co-signs.
- After cooldown: `vcb.reset()` only after the underlying volatility
  has normalised.

### `fill_rate_low`
- Indicates broker rejecting orders or our pre-trade filtering is too
  tight. Check the recent `output/ops/api_audit.jsonl` and
  pre_trade summary outputs.

## 4. Emergencies

### Overnight loss > 5%
1. Engage kill-switch at throttle=0 immediately (above curl).
2. Read the journal entry for the closing day to understand the
   exposure.
3. Re-run `python scripts/run_backtest_strategy.py` on the day's
   universe to test whether the loss matches expectation.
4. Do NOT trade until a written post-mortem is on file.

### Broker connection lost > 30 min
1. /ready should already be 503 (broker check, FMEA row 3).
2. Engage kill-switch at throttle=0.
3. Wait for broker to recover; do NOT switch broker on the fly —
   that needs a deliberate plan, not a panic move.

### Disk full
1. /ready returns 503 with `disk_quota: false`.
2. Rotate logs:
   ```bash
   python -c "from src.assembled_core.ops.log_rotation import rotate; rotate()"
   ```
3. If still full, manually move oldest `output/ops/*.jsonl.gz` to
   off-site cold storage and remove from live host (subject to
   `docs/AUDIT_LOG_RETENTION.md` 7y rule).

### Kill-switch state file corrupt
1. /ready returns 503 with `kill_switch: false`.
2. Inspect `output/ops/kill_switch_state.json`. If valid JSON but
   stale (no `engaged` key etc.):
   ```bash
   echo '{"engaged": true, "throttle_pct": 0.0, "reason": "recovery", "actor": "operator"}' \
     > output/ops/kill_switch_state.json
   ```
   This puts the system into a safe state immediately. Then
   investigate via the audit chain.

## 5. Weekly hygiene

- Sunday evening: run `python scripts/check_data_sources_health.py`
  and `verify_audit_chain()` — file findings as a journal entry.
- Verify off-site backup of `output/ops/*.jsonl` is current
  (`docs/AUDIT_LOG_RETENTION.md`).
- Re-score `docs/FMEA.md` top-5 if any incident occurred this week.

## 6. Quarterly

- DR drill: simulate the four FMEA top-5 scenarios.
- Re-run all snapshot tests with `--snapshot-update` and inspect
  diffs — anything moving without an intentional code change is a
  regression flag.

## 7. Annual

- RTS-6 self-assessment refresh (`docs/RTS6_SELF_ASSESSMENT.md`) if
  gewerblich.
- Rotate every API key at every provider (audit C3-010), even when
  no leak has been detected.
- Refresh insurance / banking / regulatory contacts.
