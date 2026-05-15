# Pilot R6 Followup Notes — 2026-05-15

After Round 6 audit closed with PASS, the first paper-pilot Day-3 run
(R6-followup) surfaced three operational items. This file documents
findings and remediations.

---

## 1. ✅ `alpaca_adapter` import path (FIXED in `5102865`)

**Finding:** `scripts/run_paper_pilot.py` imported from
`src.assembled_core.execution.alpaca_adapter` — module does not exist.
The class lives in `broker_adapter.py`.

**Effect (pre-fix):** Startup safety net was effectively broken. Both
`cancel_all_stale_orders` (Item 80) and `check_state_recovery` (Item 68)
failed silently as "non-fatal warnings". Stale broker orders weren't
cleaned, and disk-vs-broker state divergence wasn't detected.

**Fix:** 2 occurrences updated to `broker_adapter`. Smoke-test confirms
imports + AlpacaAdapter instantiation works.

**Status:** CLOSED.

---

## 2. ✅ 101 pending ORDER_SUBMIT intents (DRAINED, root cause noted)

**Finding:** Pilot startup logs warning `101 pending order intents from
prior crash — reconcile manually before proceeding`.

**Investigation:**
- All 101 records: MSFT BUY 5 shares
- All `broker_order_id` empty
- Date distribution: 2026-05-06 (3), 2026-05-07 (13), 2026-05-08 (20),
  2026-05-09 (26), 2026-05-10 (33), 2026-05-12 (5), 2026-05-15 (1)
- **301 MSFT-BUY-5 ORDER_SUBMITs total**, only 200 ORDER_COMPLETE pairs
- COMPLETE status distribution: cancelled_stale (172), cancelled (25),
  submit_failed (3), **filled (0)**

**Root cause:** Not a crash recovery issue. The strategy targets MSFT
BUY 5 shares **every cycle**, but the order NEVER actually fills. Most
get marked `cancelled_stale` by reconciliation; 101 fell through that
path and stayed pending.

**Broker cross-check:** `AlpacaAdapter.get_open_orders()` reports 0 open
orders. All 101 pending intents have NO matching broker-side state.

**Action taken:**
- Created `scripts/ops/reconcile_pending_intents.py` — one-shot drain
  tool with dry-run + broker cross-check.
- Ran `--apply` → 101 ORDER_COMPLETE records appended with
  `status="cancelled_stale_reconciliation"`.
- Verification: 0 pending intents remaining.

**Open question (NOT fixed in R6-followup, requires triage):**
Why does MSFT BUY 5 get targeted every cycle but never fill? Most
likely explanations:
- Buying-power constraint hits MSFT specifically (it's at ~$400/share,
  larger notional than the 9 currently-held positions)
- Strategy adds MSFT to top-N pick but post-trade weight check rejects
- Sanity-check halt (some MSFT-specific rule firing repeatedly)
- Cycle-notional-cap eats MSFT after other orders consumed budget

**Recommendation:** Tail next pilot day's logs for MSFT-specific
"rejected" / "blocked" / "halted" messages, OR temporarily exclude MSFT
from the universe to confirm the spam stops.

**Status:** Pending intents DRAINED (operational); strategy-level
root cause OPEN (separate triage).

---

## 3. 🔍 Logger file-truncation (UNDER INVESTIGATION)

**Finding:** `logs/live_paper_*.log` files are truncated after the
yfinance data-fetch step (~15 lines), but the run completes successfully
(rc=0) with state updates landing on disk. Trading cycle output (signal
gen, risk filter, fill simulation, order submission) appears only in
stdout (captured by the pilot subprocess), not in the dedicated log file.

**Implication:** Operator-facing logs are incomplete. Cannot verify
post-fact whether audit-fixed paths (qty-sync, pre_trade fail-closed,
NewsRAG preserve, etc.) actually fired without examining the pilot
manifest's `output_snippet` (which is also truncated to 500 chars).

**Status:** Not yet fixed — separate troubleshooting required.
Hypothesis: a logging handler in the trading cycle is configured to
stderr/stdout-only instead of attaching to the file handler from
`logging_config.py`.

---

## Pilot State After R6-Followups

- Day 3/30 OK (equity $99,236 broker / $99,314 ledger, +1.4% over 3.5 weeks)
- 0 pending intents
- 9 positions, leverage 1.114x (within 1.20 cap)
- 0 broker open orders
- Startup safety checks now functional (broker_adapter import fixed)

**Ready for Day 4 (next market-day call to `--run-day`).**
