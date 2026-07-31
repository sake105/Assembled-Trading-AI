# Regression Test Map

**Purpose.** Every real bug or hardening finding that landed in this
repo should have a test that re-raises if the bug comes back. This
document is the living index of those mappings. It is not a bug
tracker and not a roadmap — it is a contract between known-broken
past behaviour and the test(s) that guard against re-entry.

**Scope.** Findings are pulled from `CHANGELOG_DUE_DILIGENCE.md` and
from the Sprint 4 chaos-test session (Plan C21). Enhancements and
future work live in `KNOWN_ISSUES.md`, not here.

**Rules.**

- One row per finding, not one row per test.
- `guard_test` names the *primary* regression test. Supporting tests
  can be listed in the notes.
- `status` is one of:
  - `guarded` — regression test exists and is running in the default
    lane
  - `xfail` — regression test exists but is currently marked xfail
    because the bug is real and unfixed; fix is a follow-up
  - `gap` — no regression test yet; needs writing before the finding
    can be considered closed

---

## Phase 1 — Pre-backtest hardening

| ID | Finding | File | Fix summary | guard_test | status |
|----|---------|------|-------------|------------|--------|
| P1-1 | `broker_adapter` silently returned 0.0 on missing price lookup | `execution/broker_adapter.py` | Raise `PriceLookupError` instead of returning 0.0 | `tests/test_broker_adapter_price_lookup.py` | guarded |
| P1-2 | 28 source files had `except Exception: pass` silent-pass blocks that swallowed real failures | 28 files under `src/assembled_core/` | Replaced with explicit logging + re-raise or typed handling | `tests/test_no_silent_pass_policy.py` | guarded |
| P1-3 | Drawdown exposure caps were configured but not enforced in `pre_trade_risk_filter` | `execution/risk_controls.py` | Cap enforcement wired into the pre-trade filter | `tests/test_risk_controls_drawdown_caps.py` | guarded |
| P1-4 | Commission default was 0 bps (unrealistic for US equities) | `execution/transaction_costs.py` | Default raised to 1 bps | `tests/test_transaction_costs_defaults.py` | guarded |

## Phase 2 — Pre-paper hardening

| ID | Finding | File | Fix summary | guard_test | status |
|----|---------|------|-------------|------------|--------|
| P2-1 | Kill switch had no persistent state, no audit, no fractional throttle | `execution/kill_switch.py` | Persistent JSON state, JSONL audit trail, three-tier throttle | `tests/test_kill_switch_persistent_state.py` | guarded |
| P2-3 | Reconciliation `cash_tol` was 1e-6 (too loose) and `fail_fast` was False by default | `accounting/reconciliation.py` | `cash_tol=1e-8`, `fail_fast=True` | `tests/test_chaos_reconcile_drift.py` | guarded |
| P2-4 | `prices_ingest` fabricated synthetic OHLC columns when missing | `data/prices_ingest.py` | Raise `ValueError` on missing OHLC columns instead | `tests/test_prices_ingest_missing_ohlc.py` | guarded |
| P2-5 | PIT guard had no audit log for warn-mode violations | `data/pit_guard.py` | JSONL audit trail in `output/pit_audit/` | `tests/test_pit_guard_audit_log.py` | guarded |

## Phase 4 — Architecture hardening

| ID | Finding | File | Fix summary | guard_test | status |
|----|---------|------|-------------|------------|--------|
| P4-2 | Look-ahead / leakage was not gated at QA boundary | `qa/qa_gates.py` | `check_leakage()` added — **NOT mandatory** (label corrected 2026-08-01): since 2026-08-01 part of `evaluate_all_gates`, but fail-open without `feature_df` and no production caller supplies one → visible, not enforcing | `tests/test_qa_gates_leakage.py` | partial (visibility only) |
| P4-3 | Stale features were not detected → old data silently drove fresh decisions | `data/freshness_monitor.py` | `detect_stale_features()` added | `tests/test_freshness_monitor_stale.py` | guarded |
| P4-4 | Flash-crash circuit breaker missing | `risk/circuit_breaker.py` (new) | Rolling-window detector class + volatility variant | `tests/test_circuit_breaker.py`, `tests/test_vol_circuit_breaker.py` | guarded |
| P4-5 | Orders had no lifecycle state tracking → duplicate-fill risk | `execution/order_lifecycle.py` (new) | State machine: pending → submitted → filled/cancelled/rejected | `tests/test_order_lifecycle_state_machine.py` | guarded |

## Sprint 4 chaos findings

| ID | Finding | File | Fix summary | guard_test | status |
|----|---------|------|-------------|------------|--------|
| C21-1 | Concurrent block/unblock on `symbol_kill_switch` leaves the state file well-formed but the final symbol set is subset-only (not strictly consistent) | `execution/symbol_kill_switch.py` | Documented; convergence invariant locked in | `tests/test_chaos_kill_switch_race.py::test_concurrent_block_unblock_converges` | guarded |
| C21-2 | `symbol_kill_switch` uses unlocked JSON read-modify-write. A `block(TARGET)` can be overwritten by a racing `block(NOISE)` before the query sees the effect. Not a regression — module is designed for single-writer use. Fix requires file locking (portalocker / fcntl). | `execution/symbol_kill_switch.py` | Not yet fixed. Follow-up in Sprint 4 tail. | `tests/test_chaos_kill_switch_race.py::test_block_then_query_is_consistent` | xfail |
| C21-3 | Reconciler must flag every class of broker drift (cash, qty, missing-in-ledger, missing-in-broker) and must ignore drift within tolerance | `accounting/reconciliation.py` | Behaviour locked in | `tests/test_chaos_reconcile_drift.py` | guarded |

---

## How to add a row

1. The bug must be a real past or present finding, not a roadmap item.
2. Write the test first. If the test can pass without the fix, it is
   not a regression guard.
3. Add the row with `status: gap` only if a fix exists but no test
   has been written yet. The goal is zero `gap` rows.
4. If a bug is found and the fix is deferred, add the row with
   `status: xfail` and a test marked `@pytest.mark.xfail(strict=False)`
   with a reason that cites the fix requirement.
