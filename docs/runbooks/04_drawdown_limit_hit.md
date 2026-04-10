# Runbook 04: Drawdown Limit Hit

**Severity:** critical
**ETA to resolution:** 30 min – several hours (plus human approval)
**On-call contact:** trading-ops + portfolio-owner
**Component:** `src/assembled_core/risk/state_machine.py`, `risk/risk_controls.py`, `execution/kill_switch.py`

## Symptoms

- Equity drops below drawdown threshold: soft (−8%), hard (−12%), kill (−18%).
- `risk_state.json` shows `mode: "de_risk"` or `"halt"`.
- Kill-switch activated with `reason` containing "drawdown".
- `run_kpis.json` shows `current_drawdown` exceeding policy caps.
- Target weights scaled down or zeroed in Phase 18.

## Immediate Actions (first 5 min)

1. Verify the drawdown is **real**, not a pricing glitch:
   - Pull broker account equity directly: `python scripts/alpaca_account.py`
   - Compare to `ledger_state.json` equity.
   - Large divergence → Runbook 02 (reconcile).
2. If drawdown is real, leave kill-switch ON.
3. Snapshot state:
   - Copy ledger + last 10 run folders to `output/runs/_incidents/`.
4. Alert portfolio owner — DD events require human business decision.

## Diagnosis

1. Compute drawdown breakdown:
   - Over what window did the DD accumulate? (1 day, 1 week, 1 month)
   - Which positions contributed most? Sort trade_journal losses descending.
2. Identify drivers:
   - **Concentrated single-name loss** → idiosyncratic. Check for earnings surprise, halt, delisting.
   - **Broad market drop** → systemic. Verify SPY / sector ETF drawdowns.
   - **Strategy degradation** → compare signal hit-rate recent vs historical.
   - **Data / model bug** → check recent deploys, feature drift, PIT audit.
3. Verify risk controls were not bypassed:
   - `pre_trade_risk_filter` logs showing expected scale-down?
   - Were position caps respected?
   - Did the correlation guard + exposure caps trigger?

## Resolution

**Decision points require portfolio-owner approval.**

Options:

**Option A — Pause & reassess (soft DD, −8% to −12%):**
1. Keep throttle at 50–80%.
2. Reduce new position entries for N trading days.
3. Continue managing existing positions.
4. Review strategy signals daily.

**Option B — De-risk to cash (hard DD, −12% to −18%):**
1. Close losing positions systematically (market orders or TWAP).
2. Hold cash until diagnostic is complete.
3. Require full post-mortem before re-enabling.

**Option C — Full halt (kill DD, ≥ −18%):**
1. Close ALL positions via manual orders.
2. Halt strategy indefinitely.
3. Conduct full model review, backtest review, and data audit.
4. Require written approval from portfolio owner to resume.

## Post-Incident

- Post-mortem MANDATORY in `docs/post_mortems/YYYY-MM-DD_drawdown_*.md`:
  - Timeline of the drawdown
  - Positions + sector exposure at peak
  - Root cause analysis (single factor vs multi-factor)
  - Policy review (were DD thresholds appropriate?)
  - Was the kill-switch latency appropriate?
- Update `KNOWN_ISSUES.md`.
- If strategy degradation: trigger walk-forward revalidation before re-enabling.
- If data issue: trigger data audit (Runbook 05 / 06).
- Verify `drawdown_policy` thresholds in `configs/policy.yaml`.
- Consider adding regression fixtures for the exact DD path.
