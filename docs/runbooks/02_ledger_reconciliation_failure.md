# Runbook 02: Ledger Reconciliation Failure

**Severity:** high
**ETA to resolution:** 30–90 min
**On-call contact:** trading-ops
**Component:** `src/assembled_core/accounting/reconciliation.py`, `ops/paper_ledger.py`

## Symptoms

- `reconciliation_report.json` shows non-empty `mismatches` block.
- `reconcile_ledger_vs_broker` raises `ReconciliationError` / `fail_fast=True` path triggered.
- Per-symbol qty drift > 0 or cash drift > `cash_tol` (default 1e-8).
- `equity` in ledger diverges from broker account value by > 1bp.
- Run halts in Phase 18 or prior to order generation.

## Immediate Actions (first 5 min)

1. **Freeze trading** — activate kill-switch:
   - `python scripts/kill_switch_cli.py activate --throttle 100 --reason "ledger reconcile mismatch"`.
2. Snapshot current state (do not overwrite):
   - `cp output/runs/_paper_ledger/ledger_state.json output/runs/_paper_ledger/ledger_state.incident_$(date +%s).json`
   - `cp output/runs/<latest>/reconciliation_report.json output/runs/_incidents/`
3. Record current broker-side positions:
   - `python scripts/alpaca_list_positions.py > output/runs/_incidents/broker_positions_$(date +%s).json`

## Diagnosis

1. Open `reconciliation_report.json` and inspect:
   - `mismatches[].symbol` — which symbols diverge?
   - `mismatches[].ledger_qty` vs `mismatches[].broker_qty`
   - `cash_delta`, `equity_delta`
2. Classify the drift:
   - **Single symbol, small qty (1-2 shares)** → likely missed partial fill or rounding. Go to 3a.
   - **Single symbol, large qty** → likely corporate action (split, symbol change) not applied to ledger. Go to 3b.
   - **Multiple symbols, all directions** → likely ledger corruption or bad merge. Go to 3c.
   - **Cash only** → missed commission, dividend, or fees. Go to 3d.
3. Targeted checks:
   - **3a** Inspect trade journal for the symbol: `grep SYMBOL output/runs/<latest>/trade_journal.jsonl`. Compare filled qty vs broker fills from Alpaca activity feed.
   - **3b** Check `corporate_actions` module logs; verify splits/dividends for the date range.
   - **3c** Inspect git history for concurrent `ledger_state.json` writes. Check for multiple scheduler processes.
   - **3d** Sum commission + fees from recent trade_journal; compare to broker account ledger.

## Resolution

**Path A — Missed partial fill:**
1. Manually patch `ledger_state.json` with the exact delta.
2. Re-run reconciliation dry-run: `python scripts/reconcile_dry_run.py`.
3. Append manual adjustment entry to `trade_journal.jsonl` with `reason: "manual_reconcile_patch"`.

**Path B — Corporate action gap:**
1. Pull CA history from data provider.
2. Apply split/dividend to ledger entry manually.
3. Update `data/corporate_actions.py` cache to prevent recurrence.

**Path C — Ledger corruption / double writer:**
1. Identify latest clean `ledger_state.backup_*.json`.
2. Replay fills from `trade_journal.jsonl` forward using `scripts/replay_ledger.py` (if available) or manually.
3. Ensure only one scheduler instance is running.

**Path D — Cash drift:**
1. Compute expected commission from trade_journal using policy rate.
2. Patch cash in ledger.
3. If systematic, verify `commission_bps` in `configs/app.yaml` matches broker tariff.

## Post-Incident

- Leave kill-switch ON until manual human approval + one successful clean reconcile cycle.
- Tighten `cash_tol` / `qty_tol` if drift was silently accumulating.
- Add regression test fixture to `tests/accounting/` with the incident drift pattern.
- Post-mortem in `docs/post_mortems/YYYY-MM-DD_reconcile_*.md`.
- Update `KNOWN_ISSUES.md`.
