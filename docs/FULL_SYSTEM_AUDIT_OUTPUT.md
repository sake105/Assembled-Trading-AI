# Full-System Deep Audit — Output (Strict Format)

**Date:** 2025-01-23  
**Constraints:** No concept/domain/business logic changes; no new dependencies; minimal, backward-compatible; ASCII-only CLI errors; relative POSIX paths in outputs.

---

## 1) Issues List (Prioritized)

### P0 — Blocking / data integrity

*None identified* **only after** the 15 collection errors are classified (see **Optional-deps matrix** below). Gate (release_sprint13 + evidence_pack) passes; core invariants hold. One collection failure is a **real import bug** (test_data_factor_store → P1-3); the other 14 are **missing optional dependencies** (scipy, fastapi, sklearn). Until the matrix is documented and the real bug is fixed or skipped, "P0: none" should be read as "no P0 in the gate subset".

### P1 — High value, low risk (fixed in this audit)

| ID | File | Function / location | Repro |
|----|------|----------------------|--------|
| P1-1 | tests/test_accounting_report_broker_meta.py | test_accounting_report_csv_fixed_schema_and_broker_columns, expected_columns list (line ~146) | `py -3 -m pytest tests/test_accounting_report_broker_meta.py::test_accounting_report_csv_fixed_schema_and_broker_columns -v` |
| P1-2 | src/assembled_core/accounting/ledger_integration.py | build_ledger_from_trades return dict + params to write_accounting_* (paths as str) | On Windows: `py -3 -m pytest tests/test_ops_evidence_pack_e2e.py::test_ops_evidence_pack_e2e -v` → assert `"\\" not in reconcile_rel` fails |
| P1-3 | tests/test_data_factor_store.py | imports `load_price_panel` from `src.assembled_core.data.factor_store` | `py -3 -m pytest tests/test_data_factor_store.py --collect-only` → ImportError: cannot import name 'load_price_panel' |
| P1-4 | scripts/data/* (6 files) | compileall fails (IndentationError/SyntaxError) | `py -3 -m compileall -q scripts` → 6 errors in scripts/data |

### P2 — Improvements / consistency

| ID | File | Function / location | Repro |
|----|------|----------------------|--------|
| P2-1 | src/assembled_core/accounting/reconciliation_report.py | JSON write (open "w" + json.dump) | N/A — no test failure; audit of writers |
| P2-2 | src/assembled_core/accounting/accounting_report.py | JSON write (open "w" + json.dump) | N/A |
| P2-3 | .github/workflows/ci.yml | No artifact upload on failure | Trigger a failing test on Ubuntu run |
| P2-4 | .github/workflows/backend-ci.yml | No artifact upload on failure | Same |
| P2-5 | tests (15 files) | Collection errors (ImportError / optional deps) | `py -3 -m pytest -q` → 15 errors during collection |
| P2-6 | Repo-wide | ruff 68 errors (F401, F841, E741, F811, etc.) | `py -3 -m ruff check .` |
| P2-7 | scripts/data/*, scripts/validate_altdata_snapshot.py | compileall fails (syntax/indent) | `py -3 -m compileall -q scripts` |
| P2-8 | tests/test_backtest_write_broker_snapshot_smoke.py | Depends on exchange_calendars | `py -3 -m pytest tests/test_backtest_write_broker_snapshot_smoke.py -v` (without optional dep) |

---

## 3) Optional-deps matrix (collection errors → handling plan)

| Failing test module | Missing dependency / cause | Recommended handling |
|---------------------|----------------------------|------------------------|
| test_alt_delay_pit_safety | **scipy** (via robustness.py top-level import) | **Skip marker:** e.g. `@pytest.mark.requires_scipy` and add to pytest.ini; skip when scipy not installed. Or install scipy in CI for "full" run. |
| test_deflated_sharpe_basic | scipy | Same as above. |
| test_multiple_testing_warning | scipy | Same. |
| test_robustness_crisis_windows | scipy | Same. |
| test_robustness_heatmap_schema | scipy | Same. |
| test_robustness_pack_smoke | scipy | Same. |
| test_robustness_param_sweep_deterministic | scipy | Same. |
| test_robustness_sensitivity_deterministic | scipy | Same. |
| test_api_monitoring | **fastapi** (TestClient) | **Skip marker** or optional extra `[api]`; document in README. |
| test_api_oms | fastapi | Same. |
| test_api_paper_trading | fastapi | Same. |
| test_api_smoke | fastapi | Same. |
| test_portfolio_report_freq_specific | fastapi | Same. |
| test_ml_explainability | **sklearn** | **Skip marker** or optional extra `[ml]`. |
| **test_data_factor_store** | **Real bug:** cannot import name `load_price_panel` from `factor_store` (symbol lives in panel_store as `load_price_panel_parquet`) | **Fix:** Update test to import from correct module (e.g. `panel_store.load_price_panel_parquet`) or add alias in factor_store; or mark skip until API is restored. |

**Which tests do we skip officially?** Those that require scipy, fastapi, or sklearn when those are not installed. Recommendation: add pytest markers (e.g. `requires_scipy`, `requires_fastapi`, `requires_sklearn`) and in default runs use `-m "not requires_scipy and not requires_fastapi and not requires_sklearn"` so full run completes without optional deps; or install optional deps in CI for "full" matrix.

**Extras (optional deps):** Document in README or pyproject.toml optional extras, e.g. `[dev,scipy,api,ml]` so `pip install -e ".[scipy,api]"` enables robustness + API tests.

---

## 2) Per-Issue: Risk, Minimal Fix, Proof

### P1-1 — Accounting report CSV schema drift (test expected_columns)

- **Risk:** Test locked old CSV schema; code added `schema_version` column. Test fails in CI/PR; schema contract unclear.
- **Minimal fix:** Add `"schema_version"` to `expected_columns` in test_accounting_report_broker_meta.py (order matches accounting_report.write_accounting_report_csv fixed_columns).
- **Proof:**  
  `py -3 -m pytest tests/test_accounting_report_broker_meta.py::test_accounting_report_csv_fixed_schema_and_broker_columns -v` → PASS.

### P1-2 — Windows backslash in ledger result paths (POSIX invariant)

- **Risk:** build_ledger_from_trades returned paths as str(Path) → Windows backslashes. Evidence index / accounting JSON / tests expect relative POSIX; breaks test_ops_evidence_pack_e2e and any consumer of the return dict.
- **Minimal fix:** (1) Pass POSIX paths into write_accounting_report_*: use `ledger_base.relative_to(output_dir).as_posix()` and `reconcile_report_path.as_posix() if reconcile_report_path else None`. (2) In return dict, normalize all path fields to POSIX (helper _posix_path(p) and .as_posix() for Paths).
- **Proof:**  
  `py -3 -m pytest tests/test_ops_evidence_pack_e2e.py::test_ops_evidence_pack_e2e -v` → PASS on Windows.

### P2-1, P2-2 — Reconcile / accounting report JSON not atomic

- **Risk:** Partial write on crash; downstream readers may see invalid JSON.
- **Minimal fix:** Use temp file in same dir + replace (same pattern as evidence_index / orchestrator manifest). No behavior change; only write pattern.
- **Proof:** Existing report tests; optional: assert file exists and is valid JSON after write.

### P2-3, P2-4 — CI: no logs on failure (Ubuntu)

- **Risk:** Hard to debug failures in ci.yml and backend-ci.yml.
- **Minimal fix:** Add step "Upload logs on failure" with `if: failure()` (or outcome == 'failure'), capture pytest/ruff output to a file, upload-artifact that file. No behavior change to test/lint steps.
- **Proof:** Manually trigger a failing run and confirm artifact present.

### P2-5 — 15 tests fail collection

- **Risk:** Full `pytest` never completes; coverage unclear; some failures are optional deps (e.g. sklearn, scipy, exchange_calendars).
- **Minimal fix:** Document which tests need which optional deps; or add pytest markers and exclude by default; or fix imports to skip gracefully. No product logic change.
- **Proof:** `py -3 -m pytest -q` after markers/excludes or after installing optional deps.

### P2-6 — Ruff 68 errors repo-wide

- **Risk:** Noise; gate only runs ruff on preset paths (clean). Broader adoption of ruff may block on these.
- **Minimal fix:** Fix or exclude file-by-file; or run ruff only on preset paths in CI. No product logic change.
- **Proof:** `py -3 -m ruff check <paths>`.

### P2-7 — scripts/data and validate_altdata_snapshot compile fail

- **Risk:** compileall fails on scripts/; already excluded from gate (run_checks uses preset paths). scripts/data and some scripts are known-broken.
- **Minimal fix:** Fix syntax/indent in scripts/data/* and validate_altdata_snapshot.py, or formally exclude from compileall in CI. No product logic change.
- **Proof:** `py -3 -m compileall -q scripts` (or exclude list).

### P2-8 — test_backtest_write_broker_snapshot_smoke requires exchange_calendars

- **Risk:** broker_snapshot preset fails on machines without exchange_calendars (CI/env).
- **Minimal fix:** Mark test as requiring exchange_calendars and skip if not installed; or add exchange_calendars to CI deps for that preset. No product logic change.
- **Proof:** Run preset with and without exchange_calendars.

### P1-3 — test_data_factor_store real import bug

- **Risk:** Test imports `load_price_panel` from `factor_store`; that symbol does not exist there (it is `load_price_panel_parquet` in `panel_store`). Full pytest collection fails; any run that includes this file fails.
- **Minimal fix:** Update test to import from the module that actually exports the function (e.g. `from src.assembled_core.data.panel_store import load_price_panel_parquet` and use it, or add `load_price_panel` alias in factor_store if API is intended). No new deps.
- **Proof:** After fix, `py -3 -m pytest tests/test_data_factor_store.py --collect-only` → no error.

### P1-4 — compileall fails on scripts/data (hard quality gate)

- **Risk:** `py -3 -m compileall -q scripts` fails on 6 files in scripts/data (IndentationError, SyntaxError). This is a **system rift**: either code must compile or exclusions must be explicit and documented.
- **Minimal fix (choose one):**  
  **(A)** Fix syntax in scripts/data (io_utils.py, rate_limit.py, pull_alpha_vantage_intraday.py, pull_coingecko_ohlc.py, pull_ecb_fx.py, pull_stooq_eod.py) so compileall passes; or  
  **(B)** Explicitly exclude scripts/data from compileall and document: in README or docs/DEVELOPMENT.md state "scripts/data is auxiliary tooling; not part of core compile gate. Gate uses `compileall src scripts` with exclude list: scripts/data, scripts/tools, 00_seed_demo_data.py." Ensure ci.yml and run_checks presets use the same exclude list (ci.yml already skips scripts/tools and 00_seed_demo_data.py; add scripts/data to skip list).
- **Proof:** Run `py -3 -m compileall -q src scripts` with documented exclusions and no errors; or fix files and run full compileall scripts.

---

## 4) Workflow Optimization

### run_checks presets (coverage vs speed)

- **Current:** release_sprint13 = small fast set; evidence_pack = evidence + schema + smoke; ops_evidence = lighter (skip-compile, skip-ruff); broker_snapshot / accounting = broader.
- **Suggestions:**  
  - Add a “smoke-only” preset (e.g. CLI smoke + schema_stable only, no deterministic_bytes) for very fast feedback.  
  - Document in MERGE_GATE or run_checks which presets are “full” vs “smoke” and which optional deps they need (e.g. exchange_calendars for broker_snapshot).  
  - Keep release_sprint13 as the single merge gate; avoid duplicating long command blocks.

### CI artifact / log improvements

- **Windows workflows:** Already capture logs and upload on failure (or always) for release-gate, evidence-pack, ops-evidence, accounting. Keep using `py -3` for project scripts.
- **Ubuntu workflows (ci.yml, backend-ci.yml):** Add a single step to capture pytest and ruff output to a file and upload as artifact on failure (e.g. `ci-logs`). No change to run commands or pass/fail semantics.

### Docs: single source of truth

- **Evidence Pack:** EVIDENCE_PACK.md is the schema/doc source; verify JSON table already includes missing_entries_count and paths_not_in_zip_entries_count (from previous audit). Keep one place for each schema (evidence index, pack manifest, verify JSON, export JSON).
- **Merge gate:** MERGE_GATE_SPRINT13.md has one primary command; reference OPS_EVIDENCE_GOLDEN_PATH and RELEASE_NOTES for details. Avoid copy-paste of long blocks elsewhere.

### Ruff policy (68 repo-wide errors)

Choose one and document in README or docs:

- **Option A — "No new warnings" gate:** In CI, run ruff only on paths that are currently clean (e.g. preset paths in run_checks). New or touched files must pass ruff; existing 68 are grandfathered until fixed. No new ruff errors in diff.
- **Option B — "Fix only in touched files":** Ruff runs on full repo but CI fails only if touched files introduce new issues; baseline = current 68. (Requires tooling to diff touched files.)
- **Option C — Baseline ignore file:** Add a ruff exclude/ignore list or per-file noqa baseline so `ruff check .` passes; new code may not add to the baseline. No new dependencies.

Recommendation: **Option A** (current gate behavior) and document "Ruff is run on preset paths for merge gate; full-repo ruff is for gradual cleanup."

### compileall gating

- **Current:** `py -3 -m compileall -q src` passes. `py -3 -m compileall -q scripts` fails on scripts/data (6 files) and possibly validate_altdata_snapshot (SyntaxWarning). tests: not run in isolation; collection errors block full run.
- **Required:** Either (1) fix scripts/data syntax so compileall passes, or (2) document exclusions: "Core compile gate: src + scripts excluding scripts/data, scripts/tools, 00_seed_demo_data.py" and ensure ci.yml / run_checks use that exclude list. See P1-4.

---

## 5) CI: PR-ready mini patch plan (capture logs + upload on failure)

**Goal:** Every key workflow that runs tests or lint must capture output to a file and upload an artifact on failure (no behavior change to pass/fail).

### ci.yml (.github/workflows/ci.yml)

- **File:** `.github/workflows/ci.yml`
- **Step to add after "Run fast tests" (and optionally after ruff / py_compile):**  
  Give the "Run fast tests" step an `id`, e.g. `id: pytest`. Then add a new step:

```yaml
      - name: Upload logs on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: ci-lint-test-logs
          path: ci_log.txt
          if-no-files-found: ignore
```

- **Condition:** Capture stdout/stderr to `ci_log.txt` in the same job. So: in "Run ruff" and "Run fast tests" steps, append output to a file, e.g.  
  `run: ruff check ... 2>&1 | tee -a ci_log.txt` and `run: pytest ... 2>&1 | tee -a ci_log.txt` (Ubuntu has tee). Then the upload step above will have a file to upload when the job fails.
- **Concrete patch:**  
  1. In "Run ruff" step: add `2>&1 | tee -a ci_log.txt` (and `|| true` if you do not want ruff failure to stop the job before pytest runs; else let it fail and ci_log.txt still has ruff output).  
  2. In "Run fast tests" step: add `id: pytest`, and change run to `pytest ... 2>&1 | tee -a ci_log.txt`.  
  3. Add the "Upload logs on failure" step with `path: ci_log.txt`, `if: failure()`.

### backend-ci.yml (.github/workflows/backend-ci.yml)

- **File:** `.github/workflows/backend-ci.yml`
- **Same idea:** Give "Run backend core tests" step `id: backend_tests`. Capture pytest output to `backend_log.txt` (e.g. `pytest ... 2>&1 | tee -a backend_log.txt`). Add step:

```yaml
      - name: Upload logs on failure
        if: failure()
        uses: actions/upload-artifact@v4
        with:
          name: backend-ci-logs
          path: backend_log.txt
          if-no-files-found: ignore
```

- **Condition:** Only when the test step fails; artifact name distinct from ci.yml.

No other workflow changes; no change to pass/fail semantics.

### CI workflow diagnostics coverage (all workflows)

| Workflow | Log capture | Artifact on failure | py -3 for project scripts |
|----------|-------------|---------------------|----------------------------|
| release-gate-ci.yml | Yes (release_gate_log.txt) | Yes (always) | Yes |
| evidence-pack-ci.yml | Yes (run_checks_log.txt) | Yes (on failure) | Yes |
| ops-evidence-ci.yml | Yes (ops_evidence_log.txt + tail) | Yes (always) | Yes |
| accounting-ci.yml | Yes (run_checks_log.txt) | Yes (on failure) | Yes |
| **ci.yml** | No | **No** | python (Ubuntu) |
| **backend-ci.yml** | No | **No** | python (Ubuntu) |
| repo-health.yml | — | Not checked | — |
| nightly-sync.yml | — | Not checked | — |
| nightly-runall.yml | — | Not checked | — |

Gaps: **ci.yml** and **backend-ci.yml** do not upload logs on failure; apply the PR-ready patch above.

---

## 6) Changes Made in This Audit

### Diff summary

1. **tests/test_accounting_report_broker_meta.py**  
   - Added `"schema_version"` to `expected_columns` in `test_accounting_report_csv_fixed_schema_and_broker_columns` so test matches accounting_report CSV schema (schema_version column).

2. **src/assembled_core/accounting/ledger_integration.py**  
   - Pass POSIX paths into write_accounting_report_csv and write_accounting_report_json: `ledger_pack_path=ledger_base.relative_to(output_dir).as_posix()`, `reconcile_report_path=reconcile_report_path.as_posix() if reconcile_report_path else None` (both call sites).  
   - In return dict of build_ledger_from_trades: added helper `_posix_path(p)` and used it (and .as_posix() where applicable) for ledger_pack_path, reconcile_report_path, accounting_report_path, broker_snapshot_path, evidence_index_path so all path strings are POSIX on all platforms.

### Exact tests run (full commands)

```text
py -3 -m pytest tests/test_accounting_report_broker_meta.py::test_accounting_report_csv_fixed_schema_and_broker_columns -v --tb=short
py -3 -m pytest tests/test_ops_evidence_pack_e2e.py::test_ops_evidence_pack_e2e -q --tb=short
py -3 -m pytest tests/test_accounting_report_broker_meta.py tests/test_ops_golden_path_evidence_pack_e2e.py tests/test_run_daily_manifest_smoke.py tests/test_run_daily_write_evidence_pack_smoke.py -q --tb=short
py -3 scripts/dev/release_sprint13.py
```

All of the above passed after the changes.

---

## Phase 0 — Risk Map & Invariant List (Concise)

### Repo inventory (abbreviated)

- **src/assembled_core:** accounting (evidence_index, evidence_pack, ledger_integration, reconciliation_report, accounting_report, broker_snapshot_*), pipeline (orchestrator, backtest, portfolio), qa, execution, features, config, risk, reports.  
- **scripts:** verify_evidence_pack, export_evidence_pack, import_broker_snapshot, run_daily, run_eod_pipeline, run_backtest_strategy; dev/run_checks, release_sprint13.  
- **tests:** Evidence/accounting, pipeline/orchestrator, broker snapshot, QA/backtest, CLI/docs — mix of fast smoke vs heavier determinism (e.g. test_evidence_pack_deterministic_bytes).  
- **.github/workflows:** release-gate-ci (blocking on main), evidence-pack-ci, ops-evidence-ci, accounting-ci (Windows, py -3, logs on failure); ci.yml, backend-ci (Ubuntu, no artifact on failure); repo-health, nightly-*.

### Core invariants (must not break)

- Evidence index: PATHS_KEYS always present (None allowed); sort_keys=True, indent=2, newline; atomic write.  
- Pack manifest v1: schema_version, run_id, as_of_date, source/source_path, zip_entries, files[], required_missing/optional_missing (keys only), zip_compression, tool_version; deterministic; atomic write.  
- Verify JSON: schema_version, ok, error_code, counts (including missing_entries_count, paths_not_in_zip_entries_count), details; ASCII-only paths.  
- Export JSON: ok, error_code, pack_path_resolved, tool_version, pack_manifest_schema_version, etc.; stdout pure JSON unless --text/--print-pack-path.  
- Paths: relative POSIX in all manifests and CLI outputs; no backslashes, no "..", no absolute.  
- Determinism: where promised (pack, evidence index, verify/export JSON), stable bytes, fixed timestamps/compression/order.  
- Broker snapshot require semantics and reconciliation correctness as implemented and tested.

### High-risk areas

- I/O: non-atomic JSON writers (reconcile/accounting reports).  
- Paths: Windows str(Path) → backslashes (fixed in ledger_integration in this audit).  
- Schema evolution: CSV/JSON columns or keys added without test/doc update (fixed for accounting CSV in this audit).  
- Optional deps: exchange_calendars, sklearn, scipy → collection or runtime failures if missing.

---

## Phase 1 — Run Everything (Summary)

| Check | Command | Result |
|-------|---------|--------|
| Pytest (full) | py -3 -m pytest -q | 15 collection errors then stop; 1 runnable test failure (accounting CSV columns) before fixes. |
| Pytest (gate subset) | release_sprint13 + evidence_pack | PASS after fixes. |
| Ruff | py -3 -m ruff check . | 68 errors (repo-wide). Preset paths: PASS. |
| compileall | py -3 -m compileall -q src scripts tests | src OK; scripts fails (scripts/data/*, validate_altdata_snapshot). |
| release_sprint13 | py -3 scripts/dev/release_sprint13.py | PASS. |
| evidence_pack | py -3 scripts/dev/run_checks.py --preset evidence_pack | PASS. |
| ops_evidence | py -3 scripts/dev/run_checks.py --preset ops_evidence --skip-compile --skip-ruff | Not re-run in this session. |
| broker_snapshot | py -3 scripts/dev/run_checks.py --preset broker_snapshot | FAIL: 2 tests (exchange_calendars missing; test_ops_evidence_pack_e2e backslash — fixed). |
| accounting | py -3 scripts/dev/run_checks.py --preset accounting | FAIL: py_compile permission denied on one dir (flaky); 1 test fail (test_ops_evidence_pack_e2e backslash — fixed). |

Classification: accounting CSV drift = schema/test drift (deterministic). test_ops_evidence_pack_e2e = Windows path (backslash). broker_snapshot preset = optional dep (exchange_calendars). Collection errors = optional deps / env.

---

## Phase 2–5 (Summary)

- **Phase 2 (E2E):** test_ops_golden_path_evidence_pack_e2e and test_ops_evidence_pack_e2e exercised; test_run_daily_manifest_smoke and test_run_daily_write_evidence_pack_smoke run. Flow 1 (ops golden path) passes after POSIX fix. Flows 2–4 partially covered by existing tests; full manual runs recommended.
- **Phase 3 (Determinism/atomicity):** Evidence index, pack manifest, orchestrator manifest, broker snapshot JSON: deterministic + atomic. Reconcile/accounting JSON: deterministic, not atomic (P2). ZIP: deterministic and path-sanitized. Errors ASCII-only where audited.
- **Phase 4 (Schema/docs):** Evidence index, pack manifest, verify/export JSON aligned with tests and docs (EVIDENCE_PACK.md updated in previous audit). Accounting CSV schema fixed in test (P1-1). tool_version from package metadata.
- **Phase 5 (CI):** Windows workflows use py -3 and upload logs on failure. ci.yml and backend-ci do not upload artifacts on failure (P2-3, P2-4).
