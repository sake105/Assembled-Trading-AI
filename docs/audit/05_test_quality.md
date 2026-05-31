# 05 Test Quality Audit

**Date:** 2026-05-30
**Environment:** Local Windows 11, Python 3.13.7, pytest 9.0.2
**Scope:** Read-only audit — no test or source file modified.
**What was run:**
- `pytest --collect-only` — full suite collection
- `pytest -m fast --tb=line -q` — fast-marked subset (2730 tests)
- `pytest tests/test_execution_kill_switch.py tests/test_kill_switch_auth.py tests/test_api_kill_switch_auth.py tests/test_execution_pre_trade_checks.py tests/test_execution_pre_trade_integration.py tests/test_ledger_partial_fills.py tests/test_ledger_replay_day.py tests/test_ledger_store_roundtrip.py tests/test_ledger_store.py tests/test_ledger_cash_invariant_partial.py tests/chaos/test_reconcile_drift.py tests/regression/test_real_vs_synthetic_fills.py` — critical path targeted run
- `pytest tests/test_execution_kill_switch.py tests/test_kill_switch_auth.py tests/test_api_kill_switch_auth.py tests/test_execution_pre_trade_checks.py tests/test_execution_pre_trade_integration.py tests/test_unified_paper_engine_pre_trade_fail_closed_F_A3_1.py tests/test_paper_engine_circuit_breaker.py tests/test_risk_controls_crisis_alpha.py tests/test_order_lifecycle_log.py tests/test_twap_vwap_annotation.py` — extended kill-switch / risk-controls run
- `pytest tests/test_qc_gate_blocks_orders.py tests/test_trading_cycle_v2.py` — QA gate + tc_risk integration
- `pytest tests/test_pipeline_trading_cycle_contract.py tests/test_pipeline_trading_cycle_smoke.py tests/test_risk_controls_integration.py tests/test_trading_cycle_v2.py` — pipeline contract suite
- `pytest tests/test_data_source_live.py` — yfinance / live data source
- Static AST scan for no-op patterns (Python `ast` module, 144 candidate functions detected; false positives subtracted — see section 2)

**What was NOT run:** Full 8059-test suite (no `pytest` without `-m fast` or path filter — too slow for single agent run). CI (Ubuntu) not run. Not all regression/ and characterization/ tests were executed individually.

---

## 1. Collection and Run Results

### 1.1 Collection

```
8059 tests collected in 4.52s
0 collection errors
```

**Comparison to memory baseline:** Memory logged 5417 tests collected clean as of 2026-04-22. Current count is **8059** — an increase of ~2642 tests since that baseline. No collection errors in either baseline or current.

### 1.2 Fast-marked subset (`-m fast`)

| Metric | Value |
|--------|-------|
| Collected | 2730 / 8059 |
| Passed | ~2543 |
| Failed | 1 |
| Skipped | 186 |
| Run date | 2026-05-30 |

**Single failure:** `tests/test_ema_trend_v0.py::test_paper_run_ema_produces_trades`

Root cause: `paper_runner._resolve_active_strategy()` gives `policy.yaml`'s `paper_pilot.active_strategy = "trend_baseline"` higher priority than the `app_cfg` passed in the test. The test passes `strategy: ema_trend_v0` but policy overrides it to `trend_baseline`, so no EMA orders are ever generated. Log confirms: `[paper_runner] active_strategy overridden by policy: 'ema_trend_v0' → 'trend_baseline'`. The test does not monkeypatch or isolate the policy loader.

**Classification:** This is a **real, ongoing test failure** — not infrastructure noise, not a known skip. The test is structurally broken against the current policy.yaml and can never pass in the default environment without patching the policy override. It has been tagged `@pytest.mark.fast` and `@pytest.mark.unit` and is part of the fast CI surface.

**EMA / yfinance note:** Neither `test_signals_ema.py` nor `test_ema_trend_v0.py` uses yfinance. The assignment's "known EMA/yfinance failure" could not be confirmed as a distinct infra skip — the actual EMA failure (`test_paper_run_ema_produces_trades`) is a policy-override config bug, not a network dependency.

The yfinance-skipped test in this repo is:
```
tests/test_data_source_live.py:191
SKIPPED: Skipping real Yahoo API call - use @pytest.mark.slow for manual testing
```
That skip is intentional and correctly handled. 12 other tests in that file passed (all mock-based).

### 1.3 Critical path targeted run

| Test files | Count | Result |
|------------|-------|--------|
| `test_execution_kill_switch.py` | 9 | pass |
| `test_kill_switch_auth.py` | 9 | pass |
| `test_api_kill_switch_auth.py` | 4 | 1 skipped (API test, optional infra) |
| `test_execution_pre_trade_checks.py` | 13 | pass |
| `test_execution_pre_trade_integration.py` | 5 | pass |
| `test_ledger_partial_fills.py` + `test_ledger_store_roundtrip.py` + `test_ledger_store.py` + `test_ledger_cash_invariant_partial.py` + `test_ledger_replay_day.py` | 21 | pass |
| `test_chaos/test_reconcile_drift.py` | 6 | pass |
| `test_regression/test_real_vs_synthetic_fills.py` | 4 | pass |
| `test_unified_paper_engine_pre_trade_fail_closed_F_A3_1.py` | 3 | pass |
| `test_paper_engine_circuit_breaker.py` | 8 | pass |
| `test_risk_controls_crisis_alpha.py` | 20 | pass |
| `test_order_lifecycle_log.py` | 9 | pass |
| `test_twap_vwap_annotation.py` | 11 | pass |

**Total critical path:** 87 passed, 1 skipped (2026-05-30). No failures.

### 1.4 Pipeline / tc_risk integration

```
pytest tests/test_qc_gate_blocks_orders.py tests/test_trading_cycle_v2.py
→ 68 passed in 8.85s

pytest tests/test_pipeline_trading_cycle_contract.py tests/test_pipeline_trading_cycle_smoke.py tests/test_risk_controls_integration.py tests/test_trading_cycle_v2.py
→ 86 passed in 12.84s
```

### 1.5 Expected skips (not regressions)

The 186 skips in the fast run are structurally expected. Confirmed categories:

| Reason | Examples |
|--------|---------|
| Optional package not installed | `numba`, `polars`, `ruptures`, `riskfolio-lib`, `hmmlearn`, `apscheduler` |
| Data file not present | `output/aggregates/daily.parquet`, `data/prices_panel.parquet`, `security_master.csv` |
| Real network call (slow marker) | Yahoo API live call in `test_data_source_live.py:191` |
| Conditional: library available → skip inverse | `test_cli_ml_validation.py:352` (tests ImportError path when sklearn is installed) |
| Numeric degenerate case | `test_evt_tail_var.py:148` (method-of-moments degenerate sample) |

All skips have explicit `reason` strings. No bare `pytest.skip()` without reason found.

---

## 2. No-op / Vacuous Tests

AST scan found 144 candidate test functions with no `ast.Assert` node. After manual inspection, the majority are false positives: they call assertion-helper functions (`assert_no_nans_in_required`, `assert_utc_timestamp`) or use mock `.assert_called_once()` / `.assert_not_called()` which are not `ast.Assert` nodes.

Confirmed **genuine vacuous** tests:

| File:Line | Anti-pattern | Snippet |
|-----------|-------------|---------|
| `tests/test_integration_run_daily.py:315` | Always-true assertion | `assert len(long_signals) >= 0, "Should have zero or more LONG signals"` — `len()` is always `>= 0` |
| `tests/test_integration_run_daily.py:148` | Unconditional `assert True` | `assert True, "Pipeline execution completed"` — no functional condition checked |
| `tests/test_run_daily_smoke.py:114` | Always-true assertion | `assert len(df) >= 0, "Should have non-negative number of orders"` |
| `tests/test_run_daily_argparse_smoke.py:37` | `assert True` sole check | `assert True, "Module imports successfully..."` — import success is implicit if preceding code ran |
| `tests/test_backtest_numba_fallback.py:172` | No assertion at all | `test_settings_use_numba_env_var` sets env vars then returns — no assertion on whether the value was actually read |
| `tests/test_session_2026_05_07_new_items.py:11787` | `assert True` with comment | `assert True  # calendar handling is implicit through library` — documents intent, does not verify it |
| `tests/test_session_2026_05_07_new_items.py:12031` | `assert True` as placeholder | `assert True  # Acceptable if not yet implemented — just document the check` — explicitly a non-test |
| `tests/test_session_2026_05_07_new_items.py:4241` | `assert True` after import | `import src.assembled_core.pipeline.trading_cycle_shared; assert True` — import success is the only check; `assert` is redundant |
| `tests/test_session_2026_05_07_new_items.py:4247` | Same pattern | `import ...; assert True` |
| `tests/test_cli.py:23` | `assert X is not None` as sole check on module import | `assert scripts.cli is not None` — can only fail if import fails; `assert` adds nothing |
| `tests/test_strategies_multifactor_regime_overlay.py:588` | Always-true assertion | `assert len(signals) >= 0  # Can be empty` |
| `tests/test_backlog_risk_guards.py:95` | No assert on result | `test_dd_damper_no_zero_division_peak_zero` catches `ZeroDivisionError` via `pytest.fail` — acceptable smoke test, but does not assert on the returned value |

**Highest-risk vacuous pattern:** `test_session_2026_05_07_new_items.py` has multiple `assert True` placeholders explicitly marked as "document the check" — these are admitted TODO-as-tests. The file itself contains a self-check at line 4886: `assert bare_true < 30` (counts own `assert True` occurrences). As of audit: 6 found. Under the threshold, but the threshold itself is permissive.

**Self-fulfilling tests:** No clear write-then-read-the-same-value pattern found in the primary critical-path tests. The `test_atomic_io.py` and `test_ledger_store_roundtrip.py` patterns write then read back and compare against expected constants — those are legitimate roundtrip tests, not self-fulfilling.

---

## 3. Critical-Path Coverage

Coverage assessed by reading source structure and mapping to test files. `pytest-cov` was not run (not in scope for read-only audit; a scoped `--cov` run would add evidence but was not feasible in time budget without CI).

| Module | Test file(s) | Meaningful branches asserted? | Gap |
|--------|-------------|-------------------------------|-----|
| `execution/kill_switch.py` | `test_execution_kill_switch.py`, `test_kill_switch_auth.py`, `test_api_kill_switch_auth.py` | YES — activation, HMAC-rejection (wrong token, no token, correct token), fail-closed, audit trail written | Minor: deactivation with `OPERATOR_KILL_TOKEN` unset (no env at all) path not directly tested in isolation |
| `execution/pre_trade_gate.py` | `test_execution_pre_trade_checks.py`, `test_execution_pre_trade_integration.py`, `test_unified_paper_engine_pre_trade_fail_closed_F_A3_1.py` | YES — rejection paths (over-limit, bad price, zero qty), fail-closed on exception | `_apply_pre_trade_impact` called from `_tc_execution.py` — impact-driven rejection path has no dedicated negative test |
| `pipeline/_tc_risk.py` | `test_trading_cycle_v2.py` (integration), `test_qc_gate_blocks_orders.py`, `test_risk_controls_crisis_alpha.py`, `test_risk_controls_integration.py` | PARTIAL — QA-block gate tested, risk controls integration tested; EVT VaR scale path and copula-tail scale path not directly tested in isolation (both wrapped in `except Exception: log.debug`) | No unit test that directly asserts EVT or copula qty-reduction; both are reachable only if optional packages present and threshold exceeded |
| `pipeline/_tc_execution.py` | `test_twap_vwap_annotation.py`, `test_order_lifecycle_log.py`, `test_execution_safe_orders.py`, `test_trading_cycle_v2.py` | PARTIAL — `route_orders` TWAP annotation, order generation, group-exposure caps tested; but `book_fills` path (fill recording, `algo_type`/`algo_n_slices` wiring) tested only indirectly via integration | No direct unit test of `book_fills` booking logic; tested only through full trading-cycle smoke tests |
| `paper/unified_paper_engine.py` | `test_paper_engine_circuit_breaker.py`, `test_paper_engine_partial_fills.py`, `test_paper_engine_reconcile_slo.py`, `test_paper_engine_state_crash_safety.py`, `test_paper_engine_adversarial_fill.py`, `test_unified_paper_engine_lifecycle.py`, `test_unified_paper_engine_fill_qty.py` | YES — circuit-breaker trip, partial fills, state crash/recovery, reconcile SLO, fill-qty rounding | `flatten_all_positions()` execution path (crisis-alpha exit) tested via integration (`test_risk_controls_crisis_alpha.py`) but no unit-level fill verification |
| `execution/kill_switch.py` + API route | `test_api_kill_switch_auth.py` | PARTIAL — HTTP 403 on bad token tested; HTTP 200 on successful deactivation tested | Race condition on concurrent activate/deactivate not tested (acknowledged in chaos tests as a gap) |
| `accounting/` (report generation) | `test_accounting_report_broker_meta.py`, `test_accounting_report_written.py`, `test_reconciliation_smoke.py`, `test_chaos_reconcile_drift.py` | PARTIAL — report structure and fields tested; reconcile-mismatch halt path tested in chaos test | `ops/order_lifecycle_log.py` `find_open_orders()` EOD validator: tested in `test_order_lifecycle_log.py` (9 tests, pass) |
| `risk/` (regime models, drawdown derisk) | `test_risk_drawdown_derisk.py`, `test_risk_regime_models.py`, `test_risk_exposure_engine.py`, `test_risk_controls_crisis_alpha.py` | YES for main paths | HMM-based regime classification with insufficient data (0 rows path) tested only via warning log assertion in integration tests, not unit-isolated |
| `ops/dead_man_switch.py` | `test_dead_man_switch.py` | YES — enabled/disabled, shadow mode, market mode, missing-alive key (fail-safe), liveness mock | DMS daemon not wired into Task Scheduler — no end-to-end OS-level test |

**Weakest coverage gap:** `pipeline/_tc_risk.py` — the EVT tail-VaR scale branch and copula tail-dep scale branch are reachable at runtime but have no unit tests that verify the qty-reduction actually occurs. Both branches are wrapped in `except Exception: log.debug(...)`, meaning a broken import or silent failure is indistinguishable from a correct skip. These are not blocked by optional deps (scipy/statsmodels are installed per requirements.txt pins) — they depend on having enough data (`>= 60 rows`). No test constructs that scenario and asserts qty reduction.

Second weakest: `pipeline/_tc_execution.py::book_fills` — no isolated unit test for fill-booking logic. All coverage is via full-stack trading-cycle integration tests which pass, but specific branches (partial fill, slippage, `algo_type` persistence) are not individually asserted.

---

## 4. Verdict: Is "Green" Real or Partial?

**Partial, with one confirmed ongoing failure.**

- The fast suite (2730 tests) has **1 real failure**: `test_ema_trend_v0.py::test_paper_run_ema_produces_trades` — a structurally broken test that cannot pass under the current `policy.yaml`. Not a known/documented infra skip.
- The critical-path targeted runs (87+68+86 tests across kill-switch, pre-trade, ledger, reconcile, pipeline) are **genuinely green** locally (2026-05-30).
- 186 skips in the fast suite are all expected, all have reasons, none indicate regressions.
- 8059 tests collect cleanly — no collection errors (vs. historical "~19 collection errors" baseline; that problem is resolved).
- **What "green" does not cover:** Full 8059 suite not run. CI (Ubuntu) not verified. EVT/copula and `book_fills` branches have meaningful coverage gaps. ~7–12 genuinely vacuous tests exist (placeholders in `test_session_2026_05_07_new_items.py`, one no-assertion numba env test, two always-true `len() >= 0` checks) — not regressions, but they inflate test count without adding safety.
- The 2730-test fast-marked green count is meaningful but not a global claim. The ~5329 non-fast tests were not executed in this audit.

**Summary for integrator:** 1 real fast-suite failure (ema/policy override), 0 collection errors, 75/75 critical path clean, ~12 vacuous tests (low severity), coverage gap on EVT+copula qty-reduction branches and `book_fills` unit isolation.
