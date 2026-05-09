# System Check Tournament — Iteration 4 Report
**Run:** 20260509_180000_7b1d6f1
**Date:** 2026-05-09
**Critics:** 25 domains | **Judge:** Sonnet 4.6 (inline)
**Note:** ANTHROPIC_API_KEY absent — tournament run inline by judge agent

---

## Executive Summary

Iteration 4 audited 10 fresh files: `accounting/ledger.py` (modified), `ml/model_registry.py`
(new), `risk/vol_targeting.py`, `config/policy_loader.py`, `risk/var_methods.py`,
`pipeline/trading_cycle_v2.py` (comment audit), `pipeline/_tc_risk.py` (PAUSE enforcement),
`portfolio/hrp_sizing.py`, `portfolio/covariance.py`, and `ml/purged_cv.py`.
6 carryover items are confirmed still open. 4 new confirmed bugs found. Most important new
findings: (1) `register_model()` in `model_registry.py` uses `.read_bytes()` — same OOM path
as carryover item #5 — extending the scope of that fix; (2) `hrp_sizing.py` and
`covariance.py` both use deprecated `pct_change()` without `fill_method=None`.

**Overall Grade: C+**
The CRITICAL kill-switch auth gap is unresolved (carryover from Iter 1). The geo-risk PAUSE
state is computed and persisted but never read to block orders — a structural enforcement gap
that has now been confirmed across 3 iterations. The system continues hardening at the edges
but two systemic safety holes remain open.

---

## Prioritized Fix Backlog — Top 10

| Rank | Sev | File : Line | Fix | Est |
|------|-----|-------------|-----|-----|
| 1 | CRITICAL | `api/app.py:70-97` | Kill-switch POST endpoints accept throttle_pct/reason/actor as unauthenticated URL query params. Any network-accessible process can halt trading. Add API key Depends() and move body params to Pydantic request model. **Carryover Iter 1.** | 2.0h |
| 2 | HIGH | `pipeline/_tc_risk.py` (all) + `trading_cycle_v2.py:100` | `ctx.risk_state` is set in `ingest_data()` (state machine PAUSE/ACTIVE/etc.) but `check_risk()` never reads it to block orders. Docstring at line 100 says "read by check_risk" — false. PAUSE state from geo-risk is silently ignored. Add PAUSE guard at top of `check_risk()` that clears `result.orders_filtered` when `ctx.risk_state.get("state") == "PAUSE"`. **Carryover Iter 2.** | 1.5h |
| 3 | HIGH | `api/app.py:48-62` | `/ready` returns HTTP 200 when `checks["kill_switch"]=False`. Load-balancers/k8s readiness probes require 503 on degraded. Return `JSONResponse(status_code=503)` when any check fails. **Carryover Iter 3.** | 0.5h |
| 4 | HIGH | `qa/backtest_engine.py:785` | `_px_cache.get(sym, 0.0)` returns 0.0 for missing prices. Long position with missing price computes `qty * 0.0 = 0`, silently understating equity. Use last known price or warn+exclude. **Carryover Iter 2.** | 0.5h |
| 5 | HIGH | `ml/model_registry.py:89` and `:139` | Both `verify_model_hash()` (line 89) and `register_model()` (line 139) use `path.read_bytes()` loading entire model into RAM. The streaming `_verify_model_file_hash()` exists but is only used in `ModelRegistry.load_deployed()`. Replace both `.read_bytes()` calls with the streaming version. **Carryover Iter 3; line 139 is new scope.** | 0.5h |
| 6 | HIGH | `ml/model_registry.py:68-69` | `if not registry: return True` silently passes all hash checks when `registry.json` is empty or corrupt. Log WARN; raise in strict mode. **Carryover Iter 3.** | 1.0h |
| 7 | HIGH | `ml/purged_cv.py:29+68` | `embargo_pct` stored as `self.embargo_pct` but never used in `split()`. Actual embargo is always `max(self.label_horizon, 1)` days regardless of the pct arg. Fix: `embargo_days = max(int(embargo_pct * fold_size), self.label_horizon, 1)`. **Carryover Iter 3.** | 1.0h |
| 8 | MEDIUM | `risk/vol_targeting.py:52` | Default `max_scale=1.5` allows 50% over-allocation in no-leverage mode (implicit leverage). When `policy.scope.leverage_allowed=false`, cap max_scale at 1.0. **Carryover Iter 2.** | 0.5h |
| 9 | MEDIUM | `config/policy_loader.py:57-65` | `validate_policy_consistency()` (checks max_short<=max_gross, drawdown threshold ordering, etc.) exists in `policy_schema.py:119` but is never called from `load_policy()`. Only `validate_policy()` is called. Add the call after `validate_policy()`. **Carryover Iter 2.** | 0.5h |
| 10 | MEDIUM | `risk/var_methods.py:80-81` | `_z_from_alpha()` silently clamps `alpha > 0.999` to `_Z_TABLE[0.999]=3.0902`. No warning. Callers with alpha=0.9999 receive wrong z-value. Log a warning when alpha exceeds table bounds. **Carryover Iter 2.** | 0.5h |

**Est. total: ~8.5h**

---

## Notable Items Outside Top 10

| # | Sev | File : Line | Finding | Est |
|---|-----|-------------|---------|-----|
| 11 | MEDIUM | `portfolio/hrp_sizing.py:105` | `pct_change()` without `fill_method=None`. Deprecated in pandas 2.2 — silently ffills NaN gaps. Replace with `pct_change(fill_method=None)`. **New.** | 0.25h |
| 12 | MEDIUM | `portfolio/covariance.py:32` | Same `pct_change()` deprecation issue. This feeds the covariance matrix used in production position sizing. **New.** | 0.25h |
| 13 | MEDIUM | `accounting/ledger.py:246-250` and `:422-426` | Both `events_from_orders()` and `events_from_trades()` generate `event_id` using `row_index=idx` (original DataFrame index), then re-sort the output. The comment "before final event_id generation if needed" implies IDs should be generated post-sort, but they are generated pre-sort. IDs are still unique (timestamp+qty+price differ), but `row_index` loses meaning as a tiebreaker. Low collision risk; clarify the invariant or move ID generation after sort. **New.** | 0.5h |
| 14 | LOW | `ml/model_registry.py:263` (`ModelRegistry.register()`) | `hashlib.sha256(model_path.read_bytes())` in the class `register()` method is a second independent `.read_bytes()` site. A 500 MB model is loaded into RAM twice during class-based registration. Combine with finding #5 fix. **New.** | 0.5h |
| 15 | LOW | `accounting/ledger.py:463` | `generate_dividend_events()` does not skip `div_per_share == 0.0`, generating zero-valued DIVIDEND events that clutter the ledger. Guard: `if not div_per_share: continue`. **New.** | 0.25h |

---

## Carryover Backlog Status

| # | Sev | Status | File | Note |
|---|-----|--------|------|------|
| C1 | CRITICAL | **Still open** | `api/app.py:70-97` | Kill-switch POST auth — confirmed unchanged |
| C2 | HIGH | **Still open** | `pipeline/_tc_risk.py` | geo-risk PAUSE never enforced — `check_risk()` never reads `ctx.risk_state`, confirmed by Grep |
| C3 | HIGH | **Still open** | `qa/backtest_engine.py:785` | `_px_cache.get(sym, 0.0)` confirmed at line 785 |
| C4 | MEDIUM | **Still open** | `risk/vol_targeting.py:52` | `max_scale=1.5` default unchanged |
| C5 | MEDIUM | **Still open** | `config/policy_loader.py` | `validate_policy_consistency()` never called from `load_policy()` |
| C6 | MEDIUM | **Still open** | `pipeline/trading_cycle_v2.py:100` | Misleading comment confirmed; `_tc_risk.py` has zero references to `risk_state` |
| C7 | MEDIUM | **Still open** | `risk/var_methods.py:80-81` | Silent clamp confirmed |
| C8 | MEDIUM | **Narrowed** | `pipeline/_tc_features.py` | `_tc_features.py` already uses `fill_method=None` throughout; open sites are `hrp_sizing.py:105` and `covariance.py:32` (new findings #11/#12) |

---

## False Positives Confirmed This Iteration

- **`ledger.py` sort with None price**: pandas coerces None to NaN float, sorts last. No crash.
- **`vol_targeting.py` pct_change**: line 150 already uses `fill_method=None`. Correct.
- **`policy_loader.py` bare except**: wraps only the warning log, not validation. Acceptable.
- **`purged_cv.py` 4x min_samples**: conservative but not incorrect. Not a bug.
- **`ModelRegistry.rollback()` status bypass**: intentional for rollback to archived versions.
- **`model_registry.py` module-level `_registry_cache`**: intentional. Each process has its own cache; not a bug for single-process deployment.
- **`ledger.py` cash_delta signs**: BUY `-(fill_qty * fill_price + costs)` and SELL `+(fill_qty * fill_price - costs)` are correct.
- **`risk/state_machine.py` PAUSE transitions**: state machine logic is correct; bug is downstream non-enforcement in `_tc_risk.py`.

---

## Domains with No New Confirmed Bugs

`pipeline/_tc_features.py` (all pct_change calls correct),
`risk/var_methods.py` (Cornish-Fisher, MC, component VaR paths sound),
`ml/model_registry.py:ModelRegistry.load_deployed()` (streaming hash verification correct),
`accounting/ledger.py` (dividend sign logic, cash_delta formula, margin equity calculation clean),
`risk/state_machine.py` (PAUSE transition logic itself correct)

---

*Judge: claude-sonnet-4-6 — 2026-05-09T18:00Z*
