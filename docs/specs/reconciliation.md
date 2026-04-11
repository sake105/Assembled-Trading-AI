# Spec: `accounting/reconciliation.py`

## Purpose

Daily reconciliation between the internal ledger state and the
broker snapshot. Every class of drift — cash, quantity, symbols
only in one side — must be flagged. Floating-point noise must not
trigger a false positive.

## Public API

- `reconcile_ledger_vs_broker(ledger_positions_df, ledger_cash,
  broker_positions_df, broker_cash, *, cash_tol=1e-8, qty_tol=1e-6,
  fail_fast=True) -> dict`
- `reconcile_daily_pnl(...)` — ledger-side PnL sanity check.

## Returned report

```
{
  "ok": bool,
  "cash_match": bool,
  "cash_diff": float,                 # ledger_cash - broker_cash
  "position_diffs_df": DataFrame,     # only drifted rows
  "missing_in_ledger": list[str],     # in broker but not ledger
  "missing_in_broker": list[str],     # in ledger but not broker
  "message": str,
}
```

## Invariants

- Drift within `cash_tol` / `qty_tol` is treated as zero. Tolerances
  are `1e-8` and `1e-6` respectively (tightened in Phase 2 of the
  due-diligence pass).
- `fail_fast=True` raises `ValueError` on any drift. The default is
  fail-fast; the non-raising path is opt-in for callers that want to
  inspect the report (e.g. the chaos tests).
- All four drift classes are evaluated every call. The reconciler
  does not short-circuit on the first mismatch.

## Error handling

- Schema mismatch on either DataFrame → raise immediately. A
  reconciler that interprets a missing `qty` column as "zero" would
  silently accept drift.
- Duplicate symbols on either side → raise. Reconciliation assumes
  a unique symbol key.

## Test strategy

- Clean snapshot reconciles.
- Single-class drift (cash, qty, missing-in-ledger, missing-in-broker)
  each flagged in isolation.
- Multi-class drift flags all classes simultaneously.
- Within-tolerance drift is ignored.
- `fail_fast=True` raises on drift.

All covered by `tests/test_chaos_reconcile_drift.py`.

## Known limits

- The reconciler is stateless. A sequence of small daily drifts is
  not detected as a trend; that lives in a separate monitoring
  layer.
