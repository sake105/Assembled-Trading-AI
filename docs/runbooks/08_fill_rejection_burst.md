# Runbook 08: Fill Rejection Burst

**Severity:** high
**ETA to resolution:** 15–45 min
**On-call contact:** trading-ops
**Component:** `src/assembled_core/execution/broker_adapter.py`, `execution/pre_trade_checks.py`, `execution/risk_controls.py`, `execution/fat_finger_guard.py`

## Symptoms

- A run has an unusually large ratio of `rejected` vs `accepted` orders in `run_kpis.json`.
- Trade journal rows for this run are dominated by `status: rejected`.
- Broker returns `422` / `400` / `insufficient_buying_power` / `wash_trade` / `halted` errors for many symbols.
- Alerts from `alert_manager` of level WARNING / CRITICAL tagged with `source: broker_execution`.
- `pre_trade_checks` reasons list contains many `risk_controls:` or `fat_finger:` entries.

## Immediate Actions (first 5 min)

1. Classify the burst: **broker-side rejects** vs **local pre-trade rejects**. Look at whether the rejection strings come from `broker_execution` (broker-side) or from `pre_trade_checks` / `risk_controls` / `fat_finger_guard` (local).
2. If the rejections are **local**, the system is doing its job — the concern is whether the policy is too tight or whether the upstream sizing layer is producing bad orders.
3. If the rejections are **broker-side**, there is a real external problem. Stop the scheduler until root cause is found.
4. Snapshot the trade journal and orders frame for the affected run into `output/runs/_incident/`.

## Diagnosis

### Broker-side rejects

1. **`insufficient_buying_power`**: the ledger's view of cash disagrees with the broker. This is a position-sync-drift symptom — switch to runbook 07.
2. **`wash_trade` / `pattern_day_trader`**: the account has crossed a regulatory rule. Pause trading and review account classification.
3. **`halted` / `security_not_tradable`**: the symbol is halted or delisted. Add it to the per-symbol kill switch (`execution/symbol_kill_switch.block_symbol`) until the halt is lifted.
4. **Generic 5xx bursts**: broker outage — switch to runbook 01.
5. **`qty_too_large` / `notional_too_large`**: the broker's own fat-finger guard triggered. Tighten the local `fat_finger_guard` policy so these never leave our system.

### Local pre-trade rejects

1. **Many `risk_controls: drawdown_exposure` rejections**: the drawdown exposure cap is doing its job. If real equity is close to a DD level, that is expected — do not widen the cap just to push orders through.
2. **Many `fat_finger: notional` rejections**: the sizing step produced an oversized order. Root cause is upstream (sizing or a bad recent price). Check `target_positions` for a degenerate row.
3. **Many `correlation_guard` rejections**: correlation clusters hit the cap. Usually this is correct; if not, it is a covariance data problem, not a guard problem.
4. **Many `symbol_kill_switch`** rejections: a previously blocked symbol is still on the target list. Either unblock it (if the block is stale) or remove it from the universe upstream.

## Resolution

### Broker-side bursts

1. For halted / delisted names: use `block_symbol(sym, reason='halted')` and re-run with the filter enabled.
2. For `insufficient_buying_power`: stop, fix position sync drift per runbook 07, then resume.
3. For generic 5xx: stop, follow runbook 01.

### Local bursts

1. For legitimate guards firing, do nothing to the guard — the guard is the last honest line. Instead fix the upstream reason:
   - A bad price in the panel (`data/prices` has a zero / stale value).
   - An un-normalized target weight.
   - A stale kill-switch entry.
2. For guards that are too tight, widen the config value **only** after a written justification. The widening must be reverted after the incident unless a follow-up change is agreed.

## Post-Incident

- Write a one-page note in `docs/post_mortems/YYYY-MM-DD_fill_rejection_burst.md` with:
  - class (broker vs local)
  - top-3 distinct rejection reasons
  - whether any guard was (temporarily) widened
- If a guard was widened, open a tracking item to revert or formalize the change.
- If the burst was broker-side, check whether the system continued to generate orders during the burst — if yes, the cycle needs a circuit breaker on the error rate.
