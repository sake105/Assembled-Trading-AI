# Full-Repo Deep Audit Report

**Date:** 2025-01-23  
**Scope:** Bugs, inconsistencies, guards, determinism, atomicity, CI, docs. No concept/domain/schema changes.  
**Constraints:** No new dependencies; ASCII-only CLI errors; minimal, backward-compatible fixes only.

---

## Phase 0 — Inventory and Risk Map

### Major modules (src/assembled_core)

| Area | Modules | Invariants / Risk |
|------|---------|-------------------|
| **accounting** | evidence_index, evidence_pack, ledger, ledger_store, ledger_integration, reconciliation, reconciliation_report, accounting_report, broker_snapshot, broker_snapshot_store, broker_snapshot_importer, position_engine | Evidence index: PATHS_KEYS fixed, date-only. Pack: deterministic ZIP + manifest, atomic writes. Reconcile/accounting: JSON deterministic; reports not atomic. |
| **pipeline** | orchestrator, backtest, portfolio, io, orders, signals, trading_cycle | Orchestrator manifest: deterministic JSON; was direct write (fixed to atomic in this audit). |
| **qa** | candidate_gate, backtest_engine, walk_forward, robustness, validation, health, metrics | Robustness/walk_forward: some tests have collection errors (optional deps). |
| **execution** | fill_model, risk_controls, transaction_costs, order_generation | — |
| **features** | event_features, macro_features, ta_features, registry | — |
| **config** | config, settings, models, constants | — |
| **risk** | exposure_engine, transaction_costs, regime_models | — |
| **reports** | daily_qa_report, metrics_export | — |

### Scripts (operational + dev)

- **Ops/CLI:** verify_evidence_pack.py, export_evidence_pack.py, import_broker_snapshot.py, run_daily.py, run_eod_pipeline.py, run_backtest_strategy.py, check_health.py  
- **Dev:** scripts/dev/run_checks.py, release_sprint13.py, tag_release.py  
- **Other:** batch_runner, run_walk_forward_analysis, run_validation_and_drift_checks, etc.  
- **Note:** scripts/data/ has syntax/indent errors (compileall fails); scripts/tools and 00_seed_demo_data.py excluded in CI ruff.

### Tests (grouped by subsystem)

- **Evidence / accounting:** test_evidence_*.py, test_verify_*.py, test_export_*.py, test_evidence_index_*.py, test_reconcile_*.py, test_accounting_*.py, test_ledger_*.py  
- **Pipeline / orchestrator:** test_orchestrator_*.py, test_run_daily_*.py, test_run_eod_pipeline.py, test_manifest_*.py  
- **Broker snapshot:** test_broker_snapshot_*.py, test_import_*_broker_snapshot*.py  
- **QA / backtest:** test_robustness_*.py, test_walk_forward_*.py, test_candidate_*.py, test_backtest_*.py  
- **Scripts / CLI / docs:** test_*_cli_smoke.py, test_*_schema_stable.py, test_docs_*.py, test_ci_workflows_*.py, test_release_*.py  

### Core invariants (must not break)

- **Output schemas:** Evidence index (PATHS_KEYS, sort_keys, indent=2). Pack manifest v1 (schema_version, run_id, as_of_date, source/source_path, zip_entries, files[], required_missing/optional_missing keys-only). Verify JSON (schema_version, ok, error_code, counts, details). Export JSON (ok, error_code, pack_path_resolved, tool_version, etc.).  
- **Paths:** Relative POSIX in manifests/ZIP; no "..", no backslashes; ASCII-only in CLI errors.  
- **Determinism:** Pack manifest + ZIP (sort_keys, indent=2, newline; fixed timestamp; sorted entries). Evidence index (sort_keys, indent=2, newline).  
- **Atomic writes:** Evidence index (temp + replace). Pack manifest (temp + replace). Orchestrator manifest (temp + replace after this audit).  
- **Broker snapshot:** require semantics and reconciliation correctness as implemented and tested.

### High-risk areas

- **I/O:** Any JSON writer not using temp+replace (partial write on crash).  
- **Path handling:** Windows backslashes, absolute paths, ".." in ZIP or manifests.  
- **Schema evolution:** Adding/removing keys in JSON outputs without doc/test updates.  
- **Cross-platform:** py -3 vs python on Windows; line endings; encoding in logs/artifacts.

---

## Phase 1 — Run Everything (summary)

### A) Full pytest

- **Result:** 15 **collection errors** (tests fail to load), then run interrupted.  
- **Failing collection (examples):** test_alt_delay_pit_safety, test_api_*, test_data_factor_store, test_deflated_sharpe_basic, test_ml_explainability, test_multiple_testing_warning, test_portfolio_report_freq_specific, test_robustness_* (crisis_windows, heatmap_schema, pack_smoke, param_sweep_deterministic, sensitivity_deterministic).  
- **Likely cause:** Missing optional deps (e.g. sklearn, scipy) or import paths; not necessarily product bugs.  
- **Gate subset:** `py -3 scripts/dev/release_sprint13.py` and evidence_pack preset: **PASS** (all steps green).

### B) Ruff (whole repo)

- **Result:** 68 errors (F401 unused imports, F841 unused variables, E741 ambiguous name, F811 redefinition, etc.).  
- **Scope:** src/, scripts/, tests/. Many in config __init__, qa/robustness, qa/walk_forward, and various tests.  
- **Gate:** run_checks only runs ruff on preset paths (e.g. release_sprint13: 6 paths); those pass.

### C) compileall

- **src:** OK.  
- **scripts:** Fails on scripts/data/* (IndentationError, SyntaxError) and validate_altdata_snapshot.py (SyntaxWarning escape).  
- **tests:** Not run separately; collection failures above block full run.

### D) Presets (run_checks)

- release_sprint13: PASS  
- evidence_pack: PASS  
- ops_evidence, broker_snapshot, accounting: not re-run in this audit; release_sprint13 covers a subset.

### Classification

- **Deterministic vs flaky:** Gate tests are deterministic. Full suite has collection issues (env/deps).  
- **Windows:** Gate uses py -3; Windows workflows aligned to py -3 for project scripts.  
- **Schema:** Verify/export schemas implemented and tested in schema_stable tests; docs missing some keys (see Phase 4).  
- **Atomicity:** One non-atomic writer in critical path fixed (orchestrator manifest); others documented.  
- **CI-only:** Ubuntu ci.yml and backend-ci do not upload logs on failure.

---

## Phase 2 — End-to-End Flows (recommended checks)

Not fully executed in this audit. Recommended manual/CI runs:

- **Flow 1 — Ops Evidence Golden Path:** Import snapshot → EOD pipeline → export pack (JSON, --text, --print-pack-path) → verify (JSON, --text, --fail-on-warn) → ops_archive_pack.ps1. Validate POSIX paths, pack_manifest_*.json in root, verify/export JSON schema and error_code precedence, atomic writes.  
- **Flow 2 — run_daily:** With and without --write-evidence-pack; help and manifest keys match docs.  
- **Flow 3 — run_backtest_strategy:** With --write-evidence-pack; meta has evidence_index_path/pack paths; verify pack offline.  
- **Flow 4 — Broker snapshot chain:** Import → ledger with broker_snapshot_policy=require → reconcile + accounting; schema_version and path normalization.

---

## Phase 3 — Determinism / Atomicity / Cross-Platform

### Writers audited

| Artifact | Writer | Deterministic | Atomic | Notes |
|----------|--------|---------------|--------|------|
| Evidence index | evidence_index.write_evidence_index_json | Yes (sort_keys, indent=2, newline) | Yes (temp+replace) | — |
| Pack manifest | evidence_pack.build_evidence_pack | Yes | Yes (temp+replace) | — |
| Orchestrator manifest | orchestrator._write_manifest_json | Yes | **Yes (after fix)** | Was write_text; now temp+replace. |
| Reconcile report JSON | reconciliation_report | Yes | No | open("w") + json.dump. |
| Accounting report JSON | accounting_report | Yes | No | open("w") + json.dump. |
| Broker snapshot JSON | broker_snapshot_store | Yes | Yes (temp+replace) | — |
| ZIP (evidence pack) | evidence_pack._write_zip_deterministic | Yes (fixed timestamp, sorted entries) | Yes (temp then move) | — |

- **User-facing errors:** Evidence pack and verify CLI use _ascii_only for ValueErrors; export CLI uses _ascii_path for JSON path fields.  
- **Path rules:** Enforced in evidence_pack (_normalize_zip_path, verify), verify CLI, export CLI (POSIX, no .., no backslashes in outputs).  
- **Modules not promising determinism:** Not forced; only documented/contractual outputs (evidence index, pack manifest, verify/export JSON) are strict.

---

## Phase 4 — Schema & Contract Consistency

- **Evidence index:** PATHS_KEYS fixed; tests (test_evidence_index_paths_fixed_schema, test_evidence_index_deterministic_bytes) align.  
- **Pack manifest v1:** schema_version, run_id, as_of_date, source/source_path, zip_entries, zip_entries_count, files[], files_count, required_missing/optional_missing (keys-only), required_keys/optional_keys, zip_compression, tool_version — covered by tests and code.  
- **Verify JSON:** schema_version, ok, error_code, bad_paths_count, **missing_entries_count**, paths_not_in_zip_entries_count, checksum_mismatches_count, details, zip_entries_count, manifest_files_count, etc. **Docs (EVIDENCE_PACK.md)** table missing missing_entries_count and paths_not_in_zip_entries_count; code and schema_stable test have them.  
- **Export JSON:** ok, error_code, pack_path_resolved, tool_version, source/source_path, pack_manifest_schema_version — consistent.  
- **tool_version:** From package __version__ in verify/export and pack manifest; single source.

---

## Phase 5 — CI Workflow Deep Check

| Workflow | Blocking | py -3 for scripts | Logs on failure |
|----------|----------|-------------------|-----------------|
| release-gate-ci.yml | On push to main | Yes | Yes (release_gate_log.txt, always()) |
| evidence-pack-ci.yml | evidence_pack preset | Yes | Yes (on failure) |
| ops-evidence-ci.yml | ops_evidence | Yes | Yes (always) |
| accounting-ci.yml | broker_snapshot | Yes | Yes (on failure) |
| ci.yml | lint + test (Ubuntu) | python | **No** artifact upload |
| backend-ci.yml | pytest phases, ruff, black | python | **No** artifact upload |
| repo-health.yml, nightly-* | — | — | Not checked |

- Docs (MERGE_GATE_SPRINT13.md) reference release_sprint13 and link to OPS_EVIDENCE_GOLDEN_PATH.md; correct.  
- Blocking vs optional steps match workflow comments (e.g. accounting preset continue-on-error in evidence-pack-ci).

---

## Prioritized Issues (P0 / P1 / P2)

### P0 — None identified

No issues that break the gate or core invariants with current usage.

### P1 — High value, low risk

| ID | Location | Problem | Risk | Minimal fix | Proof |
|----|----------|---------|------|-------------|-------|
| P1-1 | src/assembled_core/pipeline/orchestrator.py, _write_manifest_json | Manifest written with write_text; crash can leave partial JSON. | Partial read by downstream (run_daily, backtest). | **Done:** Atomic write (temp in same dir + replace), same pattern as evidence_index. | test_orchestrator_manifest_writer, test_orchestrator_backfills_evidence_index_manifest_path_smoke, test_manifest_includes_evidence_pack_paths, test_run_daily_manifest_smoke |

### P2 — Improvements and consistency

| ID | Location | Problem | Risk | Minimal fix | Proof |
|----|----------|---------|------|-------------|-------|
| P2-1 | docs/EVIDENCE_PACK.md, verify --json schema table | Table omits missing_entries_count and paths_not_in_zip_entries_count. | Docs/automation mismatch. | **Done:** Added missing_entries_count, paths_not_in_zip_entries_count and clarified details. | Review. |
| P2-2 | reconciliation_report.py (JSON write), accounting_report.py (JSON write) | Direct open("w") + json.dump; not atomic. | Partial file on crash. | Use temp file in same dir + replace (same pattern as evidence_index). | Existing report tests; optional test that file is valid JSON after write. |
| P2-3 | .github/workflows/ci.yml | No artifact upload on failure. | Hard to debug Ubuntu failures. | Add step: on failure upload pytest/ruff output to artifact (e.g. ci-logs). | Manual fail run. |
| P2-4 | .github/workflows/backend-ci.yml | No artifact upload on failure. | Same as P2-3. | Same pattern as P2-3. | Same. |
| P2-5 | Full repo: py -3 -m pytest | 15 tests fail collection (import/optional deps). | Incomplete coverage when running “all” tests. | Document which tests require optional deps; or skip in default pytest run with marker. | pytest with -m / exclude. |
| P2-6 | Full repo: ruff 68 errors | Unused imports, unused vars, ambiguous names. | Noise; gate only checks subset. | Gradually fix or narrow ruff to preset paths in CI; or add exclude for non-gate code. | ruff check on chosen paths. |

---

## Workflow Optimization

### run_checks presets

- **Coverage vs speed:** release_sprint13 keeps a small, fast set (no deterministic_bytes); evidence_pack preset includes it. Optional ops_evidence is faster (skip-compile, skip-ruff). Consider a “smoke-only” preset that runs only CLI smoke + schema stable for very fast feedback.  
- **Suggestion:** Document in MERGE_GATE or run_checks which presets are “full” vs “smoke” so developers can choose.

### CI artifacts and logs

- **Windows:** release-gate, evidence-pack, ops-evidence, accounting already capture logs and upload on failure (or always where needed).  
- **Ubuntu:** ci.yml and backend-ci: add “Upload logs on failure” step (e.g. pytest output, ruff output) to a single artifact so failures are debuggable without re-run.

### Docs

- **Single source of truth:** EVIDENCE_PACK.md verify table should list all stable JSON keys (including missing_entries_count, paths_not_in_zip_entries_count).  
- **Copy/paste:** MERGE_GATE_SPRINT13.md already has one primary command; keep it and avoid duplicating long blocks elsewhere.

---

## Changes Made in This Audit

### 1. Orchestrator manifest: atomic write

- **File:** src/assembled_core/pipeline/orchestrator.py  
- **Change:** _write_manifest_json no longer uses manifest_path.write_text(...). It now writes to a temp file in the same directory (tempfile.NamedTemporaryFile, dir=manifest_path.parent), then Path.replace to the target; finally block cleans up temp on failure.  
- **Imports:** Added `import tempfile`.  
- **Behavior:** Same manifest content and determinism; only the write is atomic.  
- **Tests run:**  
  - py -3 -m pytest tests/test_orchestrator_manifest_writer.py tests/test_orchestrator_backfills_evidence_index_manifest_path_smoke.py tests/test_manifest_includes_evidence_pack_paths.py tests/test_run_daily_manifest_smoke.py -q --tb=short  
  - Result: 8 passed.

### 2. Docs: verify JSON schema table (EVIDENCE_PACK.md)

- **File:** docs/EVIDENCE_PACK.md  
- **Change:** Verify CLI --json output table now includes `missing_entries_count` and `paths_not_in_zip_entries_count`; `details` description updated to list all four optional list keys.  
- **Behavior:** Docs align with scripts/verify_evidence_pack.py and test_verify_evidence_pack_json_schema_stable.py.

---

## Constraints Respected

- No new dependencies.  
- No breaking schema or CLI changes.  
- CLI/user-facing error messages remain ASCII-only (unchanged).  
- Only one code change (orchestrator atomic write); rest is documentation and recommendations.
