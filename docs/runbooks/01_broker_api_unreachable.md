# Runbook 01: Broker API Unreachable

**Severity:** critical
**ETA to resolution:** 15–60 min
**On-call contact:** trading-ops
**Component:** `src/assembled_core/execution/broker_adapter.py`, `execution/broker_execution.py`

## Symptoms

- `execute_via_broker` raises `BrokerAPIError`, `ConnectionError`, `ReadTimeout`, or HTTP 5xx from Alpaca.
- Repeated `[WARN] broker_adapter: retry N/M` entries in `output/logs/assembled.log`.
- `run_kpis.json` shows `orders_placed: 0` while `orders_generated > 0`.
- Trade journal empty for current run while targets are non-empty.
- `reconciliation_report.json` shows unchanged broker_positions across cycles despite generated orders.

## Immediate Actions (first 5 min)

1. Stop the scheduler to prevent cycle pile-up:
   - `taskkill /F /IM python.exe` (Windows) targeted on scheduler PID only, OR disable the cron/Task Scheduler entry.
2. Check Alpaca status page: https://status.alpaca.markets
3. Verify network reachability: `curl -s -o /dev/null -w "%{http_code}" https://paper-api.alpaca.markets/v2/account`
4. Confirm kill-switch state:
   - `cat output/runs/_kill_switch/state.json` — if `active: true`, note reason.
5. Snapshot current ledger:
   - `cp output/runs/_paper_ledger/ledger_state.json output/runs/_paper_ledger/ledger_state.backup_$(date +%s).json`

## Diagnosis

1. Inspect last run log:
   - Tail `output/logs/assembled.log` for `broker_adapter` / `broker_execution` errors.
   - Look for HTTP status code + response body.
2. Classify the error:
   - **401 / 403** → credentials issue (expired or rotated key). Go to step 3a.
   - **429** → rate limit. Go to step 3b.
   - **5xx / timeout / connection refused** → upstream outage. Go to step 3c.
   - **SSL / cert** → TLS chain or clock skew. Go to step 3d.
3. Targeted checks:
   - **3a** Verify `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` in environment (do NOT echo). Test via `scripts/check_broker_credentials.py` if available; otherwise curl with masked output.
   - **3b** Inspect recent retry frequency. Reduce `broker.max_requests_per_min` in `configs/app.yaml` and document change.
   - **3c** Wait + verify status page. If Alpaca confirms outage, engage kill-switch (step 4).
   - **3d** Check system clock (`w32tm /query /status`), CA bundle, corporate proxy.
4. Determine open orders at broker:
   - `python scripts/alpaca_list_open_orders.py` (if present) — compare with `output/runs/<latest>/orders.csv`.

## Resolution

**Path A — Credentials issue:**
1. Rotate keys at Alpaca dashboard.
2. Update secrets store / `.env` (never commit).
3. Restart scheduler.
4. Run single cycle dry-run: `python scripts/run_live_paper.py once --dry-run`.

**Path B — Rate limiting:**
1. Lower request rate in config.
2. Restart scheduler.
3. Monitor `[WARN] rate_limit` for 30 min.

**Path C — Upstream outage:**
1. Engage full kill-switch: `python scripts/kill_switch_cli.py activate --throttle 100 --reason "broker outage"`.
2. Wait for status page to clear.
3. Before re-enabling: reconcile positions (Runbook 02).
4. Deactivate kill-switch manually; require human confirmation.

**Path D — TLS / clock:**
1. Resync system clock.
2. Reinstall CA bundle / certifi.
3. Retry.

## Post-Incident

- Append entry to `KNOWN_ISSUES.md` with commit SHA + symptoms + root cause.
- Write post-mortem `docs/post_mortems/YYYY-MM-DD_broker_outage.md` if duration > 30 min.
- If recurring pattern, open issue to implement multi-broker fallback (out of current scope).
- Verify next scheduled cycle completes end-to-end before closing incident.
