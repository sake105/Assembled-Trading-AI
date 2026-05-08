# OPERATING.md — Assembled-Trading-AI Pilot v2 Operations Guide

Audience: solo operator running Pilot v2 on Windows 11.
For architecture detail: `docs/` tree. For runbooks: `docs/runbooks/`.
---

## 5-Min Quickstart

```powershell
# 1. Activate venv
.venv\Scripts\Activate.ps1

# 2. Run pre-flight and confirm startup
.\scripts\start_pilot_v2.ps1

# 3. Start one daily cycle (after script confirms GO)
python scripts/run_paper_pilot.py --run-day

# 4. Verify cycle ran
python scripts/run_paper_pilot.py --status
```

If `start_pilot_v2.ps1` shows any `[FAIL]` lines, resolve them before proceeding.
The script checks: ENV vars, policy.yaml exists, smoke tests pass, and asks for confirmation.

---

## Daily Operations Checklist

Run each trading morning before market open (09:00 ET latest):

```powershell
# Morning review: equity curve, drawdown, signal health
python scripts/daily_pilot_review.py

# Run one cycle (fires around 09:25 ET — set a scheduler task for production)
python scripts/run_paper_pilot.py --run-day
```

Things to check in the output:

- **Equity**: should be monotonically above hard-stop floor (-8% from peak)
- **Drawdown**: `output/state/risk_state.json` → `current_drawdown_pct`
- **Position count**: expect 5–20 active positions
- **Signal health**: `output/diagnostics/signal_health_*.json` → `ic_rolling` should not be zero for all symbols
- **Regime**: logged as `regime=bull|sideways|bear|crisis` — bear/crisis reduces exposure automatically
- **Logs**: `logs/paper_pilot_*.log` — scan for `[ERROR]` or `[WARN]` lines

---

## Weekly Review Checklist

Every Friday after close:

- [ ] `python scripts/run_paper_pilot.py --evaluate-only` — check NO-GO indicators
- [ ] `output/pilot/pilot_v2_manifest.json` → verify `day_count`, `pnl_pct`, running Sharpe
- [ ] `output/diagnostics/` — signal drift? IC near zero for 5+ days?
- [ ] Logs — repeated data-freshness or stale-feature warnings?
- [ ] Model age: no deployed model older than 14 days (`ml.model_max_age_days` in policy.yaml)
- [ ] OOS Sharpe < 0.5 after day 14 → consider NO-GO

---

## Troubleshooting — Top 5

### 1. Alpaca auth failure (401 or connection refused)

Symptoms: `[FAIL] Alpaca auth` in start_pilot_v2.ps1, or `HTTP 401` in logs.

```powershell
# Verify key is present
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('ALPACA_API_KEY','MISSING')[:8])"

# Test endpoint directly
python -c "
import os; from dotenv import load_dotenv; load_dotenv()
import requests
r = requests.get(os.getenv('ALPACA_BASE_URL','https://paper-api.alpaca.markets') + '/v2/account',
    headers={'APCA-API-KEY-ID': os.getenv('ALPACA_API_KEY',''), 'APCA-API-SECRET-KEY': os.getenv('ALPACA_SECRET_KEY','')})
print(r.status_code, r.text[:200])
"
```

Fix: rotate key at https://app.alpaca.markets → Paper → API Keys → Regenerate. Update `.env`.
If connection refused: check https://status.alpaca.markets — may be a platform outage.

### 2. Price data stale (panel too old)

Symptoms: `[WARN] Data freshness` — panel is N days old. Signal scores all identical or zero.

```powershell
# Check panel date
python -c "import pandas as pd; df=pd.read_parquet('output/prices_panel.parquet'); print(df.index.get_level_values('date').max())"

# Refresh prices (run before market open)
python scripts/fetch_prices.py
```

If panel path differs (check `configs/policy.yaml` → `multifactor_signal.bundle_path`), the bundle yaml will show the correct data path.

### 3. All signals zero (data freshness / factor issue)

Symptoms: cycle runs but generates 0 orders. `signal_health_*.json` → all IC values 0.

1. Check `output/diagnostics/signal_health_*.json` — which factors are zero?
2. Check `output/prices_panel.parquet` has recent data (see issue 2 above)
3. Check `conviction_threshold` in `configs/policy.yaml` — if above 0.90, lower to 0.85
4. Check `enforce_market_hours` (global or per-strategy override in policy.yaml `execution_policy`) — outside market hours blocks orders
5. Scan logs for `[WARN]` data-freshness lines: `Select-String -Path logs\*.log -Pattern "stale|freshness|zero"`

### 4. Memory growing / log rotation needed

Symptoms: process RAM grows over hours; disk fills with log files in `logs/`.

```powershell
# Check log size
Get-ChildItem logs\ | Sort-Object Length -Descending | Select-Object Name, @{N='MB';E={[int]($_.Length/1MB)}} | head

# Rotate (compress + archive older than 7d)
Get-ChildItem logs\ -Filter "*.log" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) } |
    ForEach-Object { Compress-Archive $_.FullName "$($_.FullName).gz"; Remove-Item $_.FullName }
```

Each run creates a new log file (`logging_policy.audit_required: true` in policy.yaml). The runner exits after each cycle by design — no persistent daemon, so RAM is released automatically. The issue is log file retention. Add a weekly Windows Task Scheduler job to archive old logs.

### 5. System crash recovery (stale order cleanup)

Symptoms: pilot crashed mid-cycle; Alpaca may hold open orders.

```powershell
# Cancel stale orders: Alpaca dashboard → Paper → Orders → Cancel All
# https://app.alpaca.markets/paper/dashboard

# Re-run reconciliation, then verify
python scripts/run_paper_pilot.py --reconcile
python scripts/run_paper_pilot.py --status

# Re-run pre-flight before next cycle
.\scripts\start_pilot_v2.ps1
```

---

## Emergency Procedures

### Kill Switch — immediate halt

```powershell
# Halt all trading NOW
python scripts/run_live_paper.py halt

# OR set env var (if runner is in a loop)
$env:ASSEMBLED_KILL_SWITCH = "1"
```

Confirm halt: check `output/state/risk_state.json` → `state` should be `PAUSE`.

### Hard-Stop Triggered (drawdown >= -8%)

The system auto-fires a 50% position reduction at -8% drawdown (`drawdown_policy.hard: 0.15`).
At -20% (`drawdown_policy.kill: 0.20`) the kill switch fires automatically.

If auto-halt did NOT fire (bug):
1. Run `python scripts/run_live_paper.py halt` manually
2. Check `output/state/risk_state.json` for current drawdown value
3. Do NOT restart until drawdown source is identified

### Margin Call (paper account)

Automatic handler: `src/assembled_core/risk/margin_call_handler.py` — fires 50% reduction.

Manual steps if handler failed:
1. Reduce `max_gross_exposure` to 0.80 in `configs/policy.yaml`
2. Set `leverage_allowed: false`
3. Restart: `python scripts/run_paper_pilot.py --run-day`
4. Wait 48h before restoring exposure

### Full System Reset (last resort)

```powershell
# Archive current state
$stamp = Get-Date -Format "yyyyMMdd_HHmm"
Copy-Item output\ "archive\pilot_v2_$stamp\" -Recurse

# Clear live state (preserves history)
Remove-Item output\state\risk_state.json -ErrorAction SilentlyContinue
Remove-Item output\paper\paper_state.json -ErrorAction SilentlyContinue

# Re-init and restart
.\scripts\start_pilot_v2.ps1
```

---

## Key Files

`configs/policy.yaml` — policy config | `output/state/risk_state.json` — risk state
`output/pilot/pilot_v2_manifest.json` — manifest + hard-stop criteria
`output/diagnostics/signal_health_*.json` — signal diagnostics
`logs/paper_pilot_*.log` — daily run logs
