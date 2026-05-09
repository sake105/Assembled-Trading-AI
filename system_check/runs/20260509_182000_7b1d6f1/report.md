# System Check Tournament — Iteration 4 Report
**Run:** 20260509_182000_7b1d6f1  
**Date:** 2026-05-09  
**Critics:** 25 domains | **Judge:** Sonnet 4.6 (inline)  
**Note:** ANTHROPIC_API_KEY absent — tournament run inline by judge agent

---

## Executive Summary

Iteration 4 audited domains not covered in previous iterations: `accounting/ledger.py`,
`ml/model_registry.py`, `risk/vol_targeting.py`, `risk/var_methods.py`, and confirmed
all carryover items. Two new bugs were found: (1) `verify_model_hash()` return value is
silently ignored in `meta_model.py` — hash mismatch loads corrupted models anyway; (2)
`_registry_cache` in `model_registry.py` has no mtime-based expiry (unlike `load_policy()`).
The five carryover items most impactful for execution correctness remain: kill-switch
auth (Critical), geo-risk PAUSE gap (High), MTM 0.0 (High), vol_targeting max_scale (Medium),
and `var_methods` silent clamp (Medium). The misleading comment in `trading_cycle_v2.py:100`
is now confirmed — `check_risk()` reads regime from `ctx.risk_state` indirectly via
`_tc_signals.py:653`, not directly in `_tc_risk.py`.

**Overall Grade: C+** (carryover criticals still open; two new medium findings)

---

## Prioritized Fix Backlog — Top 10

| Rank | Sev | File : Line | Fix | Est |
|------|-----|-------------|-----|-----|
| 1 | **Critical** | `api/app.py:70-104` | Kill-switch POST endpoints have no auth — add API-key or IP guard | 1-2 h |
| 2 | **High** | `risk/state_machine.py` + `execution/risk_controls.py` | Geo-risk PAUSE never blocks orders — add PAUSE enforcement in `apply_risk_controls()` | 1 h |
| 3 | **High** | `qa/backtest_engine.py:785` | `_px_cache.get(sym, 0.0)` silently values delisted/missing positions at $0 — use last known price | 30 min |
| 4 | **Medium** | `ml/meta_model.py:494-499` | `verify_model_hash(path)` return value is ignored in `load_meta_model()` — a `False` (hash mismatch) never prevents load; fix: check return or pass `strict=True` | 10 min |
| 5 | **Medium** | `risk/vol_targeting.py:52` | `max_scale=1.5` allows 50% exposure amplification in no-leverage mode — cap at 1.0 when policy `leverage_allowed=False` | 20 min |
| 6 | **Medium** | `risk/var_methods.py:80-81` | `_z_from_alpha` silently clamps `alpha > keys[-1]` to table max instead of raising — silent precision loss at very high confidence levels | 15 min |
| 7 | **Medium** | `pipeline/trading_cycle_v2.py:100` | Comment "read by check_risk" is wrong — `ctx.risk_state` is read by `_tc_signals.py:653` for regime, not by `_tc_risk.check_risk()` — remove/fix misleading comment | 5 min |
| 8 | **Medium** | `pipeline/_tc_features.py` (7+ sites) | `pct_change(fill_method=None)` breaks on pandas 3.x — replace with `pct_change()` (no kwarg) after ffill step | 1 h |
| 9 | **Low** | `ml/model_registry.py` | `_registry_cache` never expires on file-mtime change (unlike `load_policy()`) — add mtime check to `_load_registry()` | 20 min |
| 10 | **Low** | `accounting/ledger.py:203` | Unknown order side silently produces `qty=0.0` with no warning — add `logger.warning()` for unrecognized side | 5 min |

**Est. total: ~5 h. Items 7 and 10 are ≤5 min — fix first. Item 4 is 10 min.**

---

## Notable Items Outside Top 10

- **`ml/model_registry.py` filename collision**: `verify_model_hash()` looks up by `path.name`
  (filename only, not full path) — two models with the same filename in different directories
  will share the same registry entry. Low risk in practice but fragile.
- **`accounting/ledger.py` sort after event_id**: `events_from_orders()` computes `event_id`
  (which includes `row_index=idx`) before sorting at lines 246-250. The sort reorders rows
  but doesn't recompute IDs — so the final DataFrame row order doesn't match the event_id
  ordering. Not a correctness bug (IDs remain unique) but makes auditing confusing.
- **`risk/var_methods.py` alpha mirror branch**: line 79 mirrors small alphas as
  `-_z_from_alpha(1.0 - alpha) if alpha < 0.5 else ...` — for `alpha` values just below
  `keys[0]` but above 0.5, the `else` branch returns `_Z_TABLE[keys[0]]`, which is the
  minimum table value. This is correct but undocumented.
- **`fill_method=None` sites count**: 7 confirmed sites in `src/` using `pct_change(fill_method=None)`.
  Not yet broken on pandas 2.3.3, but will raise `TypeError` on pandas 3.x.

---

## False Positives Confirmed This Iteration

- **`model_registry.py` `register_model()` cache invalidation**: `_registry_cache = None`
  is set after write — subsequent `verify_model_hash()` calls will reload from disk.
  Not a bug in the same-process single-registry scenario.
- **`ledger.py` event_id uniqueness**: `row_index=idx` (DataFrame index) ensures uniqueness
  for same-timestamp same-symbol same-price orders. Not a collision bug for normal DataFrames.
- **`model_registry.py` `promote_to_deployed()` double-transition**: when promoting an
  already-deployed version, the loop first sets `status="archived"` then `status="deployed"`.
  Net result is correct.

---

*Judge: claude-sonnet-4-6 — 2026-05-09T18:20Z*
