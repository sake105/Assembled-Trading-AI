# Pilot Operations Playbook

> **⚠️ TEILWEISE VERALTET (Stand-Hinweis 2026-07-23):** Dieses Playbook ist auf dem Stand
> 2026-05-07. Seit dem Pilot-Relaunch am **2026-07-02** gilt statt der hier ggf. genannten
> **−8 %**-Drawdown-Regel eine **−10 %-SOFT-Halt**-Regel (neue Baseline 87.874,90 USD;
> `configs/app.yaml` `start_capital`+`dd_stop_pct`, Resume via `ack_halt`, kein
> OPERATOR_KILL_TOKEN nötig). Zusätzlich existieren seit `6a4fd712` (2026-07-22)
> `drawdown_policy.levels` in `configs/policy.yaml` (soft −10 % / hard −15 % / kill −20 %).
> Bei Widerspruch gilt der Code-/Config-Stand, nicht dieses Dokument.

**Audience:** Solo operator running 30-day Alpaca paper pilot  
**Updated:** 2026-05-07 (Deprecation-Banner 2026-07-23)  
**Purpose:** Pre-written decision tree for the 7 most likely failure modes.
Read this BEFORE starting a trading session, not during an incident.

---

## Quick Reference

| Symptom | Jump to |
|---------|---------|
| Rolling Sharpe < 0.5 after day 14 | [Mode 1](#mode-1-sharpe-degradation) |
| Alpaca API errors / outage | [Mode 2](#mode-2-broker-api-outage) |
| Process crash / restart | [Mode 3](#mode-3-system-crash-and-restart) |
| yfinance data gap | [Mode 4](#mode-4-data-feed-gap) |
| Too many EDCL orders per day | [Mode 5](#mode-5-edcl-hyperactivity) |
| Drawdown approaching -8% hard stop | [Mode 6](#mode-6-drawdown-approaching-hard-stop) |
| Margin call / low equity | [Mode 7](#mode-7-margin-call) |

Hard-stop criteria (pre-committed, do not change during pilot):
- Max drawdown: **-8%**
- Min Sharpe after day 14: **0.5**
- Max consecutive loss days: **7**

---

## Mode 1: Sharpe Degradation

**Symptom:** Rolling 7d Sharpe drops below 0.5 after pilot day 14.
Check: `python scripts/daily_pilot_review.py`

**Diagnose:**
1. Check if it's market-wide (`SPY 7d return` on any finance site).
2. Check recent signal quality: `output/diagnostics/signal_health_YYYY-MM-DD.json`
3. Check if regime changed: `output/pilot/daily_review_*.md` last 5 days.
4. Check crash days: `output/pilot/pilot_manifest.json` → `days[].crashed`

**Decision tree:**
- Market down >5% this week AND all long positions → expected, monitor
- Signal health shows `LOW_IC` alert for > 3 factors → factor degradation, reduce position size 50% in `policy.yaml` (`max_position_size_pct`)
- Sharpe < 0.5 for 5+ consecutive days → hard stop per manifest criteria → `python scripts/run_live_paper.py halt`
- Crash days > 2 this week → check logs, fix crash before continuing

**Escalation:** If Sharpe < 0 for 7 consecutive days → halt pilot, review before resuming.

---

## Mode 2: Broker API Outage

**Symptom:** `run_live_paper.py` exits with Alpaca connection errors. Status page: https://status.alpaca.markets

**Diagnose:**
1. Check Alpaca status page for ongoing incidents.
2. Check log: `logs/live_paper_*.log` — last 50 lines.
3. Run: `python -c "from src.assembled_core.execution.alpaca_adapter import AlpacaAdapter; AlpacaAdapter().get_account()"`

**Decision tree:**
- Outage confirmed on status page → wait, retry after outage resolved. No action needed. Paper positions are safe.
- API key issue (401) → rotate key in `.env`, re-export, restart. Keys may have been regenerated.
- Rate limit (429) → reduce run frequency, add sleep between retries
- Partial outage (orders work, data doesn't) → run with `--skip-data-refresh` flag if available, or skip today's cycle

**Recovery:** After outage resolves, run once with `python scripts/run_live_paper.py --command once` to catch up. Check reconcile output.

---

## Mode 3: System Crash and Restart

**Symptom:** Process died unexpectedly. Check `output/pilot/pilot_manifest.json` → `days[].crashed`

**Diagnose:**
1. Find last log: `ls -t logs/live_paper_*.log | head -1`
2. Check exit code in manifest.
3. Look for open orders: Alpaca dashboard → Orders tab.
4. Check for "25 pending order intents" warning in new log — this means stale orders detected.

**Decision tree:**
- Crash during data fetch → data issue, usually safe to restart. Orders not submitted.
- Crash during order submission → check Alpaca dashboard for partial fills. Cancel open orders manually if needed.
- Crash during reconcile → positions are correct, just accounting state uncertain. Run `python scripts/run_live_paper.py --command reconcile` first.
- Repeated crashes same file/line → fix before restarting

**Recovery:** Restart triggers automatic stale-order cancellation (5+ minute old orders get bulk-cancelled on preflight). Verify on Alpaca dashboard after restart.

---

## Mode 4: Data Feed Gap

**Symptom:** `[DATA-FRESHNESS] Stale data detected` in logs, or signals all zero, or very few trades.

**Diagnose:**
1. Check: `python -c "import yfinance as yf; df = yf.download('AAPL', period='2d'); print(df.tail(3))"`
2. Check panel file: `python -c "import pandas as pd; df=pd.read_parquet('output/prices_panel.parquet'); print(df.tail(3))"`
3. Check data freshness: timestamp of `output/prices_panel.parquet` file.

**Decision tree:**
- yfinance returns empty → yfinance outage (happens ~1-2x/year). Options:
  - Wait (usually resolves within hours)
  - Skip today's trading cycle (paper pilot — no harm)
  - Use yesterday's panel with explicit staleness warning
- Panel file old but yfinance works → re-run `python scripts/run_eod_pipeline.py`
- Data gap for specific symbols only → those symbols get zero signals, position sizes zero. Expected behavior.
- Gap > 2 trading days → investigate before trading

**Policy:** Better to skip a day than to trade on stale data. Paper pilot allows this.

---

## Mode 5: EDCL Hyperactivity

**Symptom:** Daily order count approaching 50 limit (`max_daily_orders` in manifest). EDCL firing on every symbol.

**Diagnose:**
1. Check: `output/reports/trades_1d.csv` — how many trades per day last 5 days?
2. Check conviction distribution: are most convictions above 0.85 threshold?
3. Check news pipeline: is GDELT/NewsAPI returning many high-confidence events?

**Decision tree:**
- Conviction threshold too low → increase in `policy.yaml`: `conviction_threshold: 0.90`
- Max daily orders being hit → reduce: `configs/paper_track/multifactor_long_short.yaml` → `max_daily_orders: 30`
- EDCL `edcl_multiplier` too aggressive → reduce from 2.0x to 1.5x
- News pipeline returning garbage → check `scripts/fetch_news_*.py` logs

**Quick fix:** Edit `policy.yaml`, set `edcl.enabled: false` temporarily to isolate whether EDCL is the cause.

---

## Mode 6: Drawdown Approaching Hard Stop

**Symptom:** Current drawdown approaching -8% hard stop. `daily_pilot_review.py` shows < 2pp room.

**Diagnose:**
1. `python scripts/daily_pilot_review.py` — check current drawdown and room to hard stop.
2. Is it market-wide (SPY also down) or strategy-specific?
3. Are shorts covering losses from longs?

**Decision tree:**
- Drawdown at -6% to -8% (2pp room) → reduce position sizes 50%: `policy.yaml` → `max_position_size_pct: 0.025` (from 0.05)
- Drawdown at -8% → **HARD STOP TRIGGERED**
  1. Run: `python scripts/run_live_paper.py halt` (or set kill_switch in policy.yaml)
  2. Log event in `output/pilot/pilot_manifest.json`
  3. Send alert (Discord webhook if configured)
  4. Do NOT restart until drawdown reviewed
  5. Wait for equity recovery before resuming (minimum 24h, ideally 48h)
- Market crash (SPY -5%+ in one day) → consider pausing regardless of drawdown level

**Hard stop is pre-committed. The rule: if you are thinking about overriding it, you are in the Disposition Effect. Follow the rule.**

---

## Mode 7: Margin Call

**Symptom:** Equity drops below Alpaca margin requirement. Alert from `check_margin_requirements()` in logs.

**Diagnose:**
1. Check account equity on Alpaca dashboard.
2. Check `logs/live_paper_*.log` for `[MARGIN]` warnings.
3. This is paper trading — Alpaca paper accounts have margin but no real money at risk.

**Decision tree:**
- Paper account (current pilot) → Alpaca will not actually liquidate. But treat as if real:
  1. Reduce all position targets to 50%: `policy.yaml` → `max_gross_exposure: 0.80` (from 1.20)
  2. Disable leverage: `leverage_allowed: false`
  3. Monitor for 48h before increasing exposure
- Margin call handler fires automatically (50% position reduction per `risk/margin_call_handler.py`)
  - Check logs for `[MARGIN-CALL-HANDLER]` to confirm it ran
  - Verify positions reduced on Alpaca dashboard

**For live trading (future):** A margin call means real money at risk of forced liquidation. Rule: close all positions manually, move to cash, review before re-entering.

---

## Checklist: Starting Each Trading Day

Before running `run_live_paper.py`:
- [ ] Check `output/pilot/daily_review_YYYY-MM-DD.md` (yesterday's review exists)
- [ ] Check current drawdown is > -6% (2pp above hard stop)
- [ ] Check Alpaca status page (green)
- [ ] Verify `.env` has valid ALPACA_API_KEY and ALPACA_API_SECRET
- [ ] Note today's market regime (SPY pre-market direction)

After running:
- [ ] Check log for `[WARN]` and `[ERROR]` lines
- [ ] Check reconcile status in log
- [ ] Check order count (< 50 per day)
- [ ] Note if any new symbols entered/exited

---

## Hard-Stop-Kriterien Pre-Commitment (Item 120)

These criteria are **pre-committed** before pilot start. They must not be changed
during the pilot run — doing so defeats the purpose of the commitment device.

| Criterion | Value | Action if triggered |
|-----------|-------|---------------------|
| Max drawdown | **-8%** | Halt immediately via `run_live_paper.py halt` |
| Min Sharpe after day 14 | **0.5** | Review and halt if below for 3 consecutive days |
| Max consecutive loss days | **7** | Halt and move to cash |
| Kill-switch action | `halt_trading_send_alert` | Discord alert + trading stopped |

**Source of truth:** `output/pilot/pilot_manifest.json` → `hard_stop_criteria`

**Rule:** If you are considering overriding a hard stop, you are in the Disposition Effect.
Follow the rule. The pilot can be restarted after review; real capital cannot be un-lost.

**Verification command:**
```bash
python -c "import json; m=json.load(open('output/pilot/pilot_manifest.json')); print(m['hard_stop_criteria'])"
```

---

## Emergency Contacts

- Alpaca status: https://status.alpaca.markets
- Alpaca support: support@alpaca.markets
- Kill switch: edit `configs/policy.yaml` → `kill_switch: true`
- Hard stop action: `python scripts/run_live_paper.py halt`

---

*This playbook covers paper pilot operation only. Update before going live with real capital.*
