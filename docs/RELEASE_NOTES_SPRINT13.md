# Release Notes - Sprint 13 (Accounting & Ops Evidence)

Release: Sprint 13
Tag: vX.Y.Z

Short, shareable summary for Slack / Notion / PR description. Full details: **docs/LEDGER_RECONCILIATION.md**, **docs/EVIDENCE_PACK.md**, **docs/PROJECT_STRUCTURE.md**.

---

## Summary

- **Ledger & reconciliation**: Event storage, position engine, reconciliation reports (CSV/JSON/MD), accounting reports. All artifacts use `schema_version: 1`.
- **Broker snapshot**: Import from external JSON/CSV, policies `ignore` / `prefer` / `require`. Standalone CLI `import_broker_snapshot.py`; EOD/Backtest/Daily pipelines support broker snapshot controls.
- **Evidence Index**: Central JSON per run/date linking snapshot, ledger, reconcile, accounting, optional manifest. Single entry point for Ops.
- **Evidence Pack**: Deterministic ZIP + pack manifest; export from index (or manifest fallback). Verify CLI offline: `--zip path --json` with stable `error_code`. **tool_version** in Evidence Index, Pack manifest, and Verify/Export JSON is derived from the installed package version (pyproject.toml).
- **CI**: Evidence Pack CI (blocking); Ops Evidence CI (optional, logs on fail); Accounting CI (broker_snapshot blocking, accounting optional).

---

## New Ops Evidence Artifacts

| Artifact | Location / CLI | Purpose |
|----------|----------------|---------|
| Evidence Index | `output/evidence_<run_id>/evidence_<YYYY-MM-DD>.json` | Links all artifacts for a run/date |
| Evidence Pack ZIP | `output/evidence_<run_id>/pack_<YYYY-MM-DD>.zip` | Portable archive of evidence files |
| Pack Manifest | Inside ZIP + `output/evidence_<run_id>/pack_manifest_<YYYY-MM-DD>.json` | Schema, checksums, file list |
| Verify CLI | `scripts/verify_evidence_pack.py --zip <path> [--json]` | Offline validation (manifest, checksums, paths) |

Details and schemas: **docs/EVIDENCE_PACK.md**.

---

## Determinism Guarantees

- **ZIP**: Fixed timestamps (default 1980-01-01 00:00:00), entries sorted lexicographically (POSIX paths).
- **JSON**: Pack manifest and verify `--json` output use `sort_keys=True`, `indent=2`, trailing newline. Two runs on same inputs yield identical bytes where specified.
- **Paths**: Relative POSIX only; no backslashes, no absolute paths in manifests.

See **docs/EVIDENCE_PACK.md** (Determinism Rules) and **docs/LEDGER_RECONCILIATION.md** (Determinism).

---

## CI Coverage

- **Evidence Pack CI (Windows)** - [.github/workflows/evidence-pack-ci.yml](.github/workflows/evidence-pack-ci.yml) - Blocking: `evidence_pack`. Optional: broker_snapshot, accounting. Logs artifact on failure.
- **Ops Evidence CI (Windows)** - [.github/workflows/ops-evidence-ci.yml](.github/workflows/ops-evidence-ci.yml) - Runs only `ops_evidence` preset; logs artifact on failure.
- **Accounting CI (Windows)** - [.github/workflows/accounting-ci.yml](.github/workflows/accounting-ci.yml) - Blocking: `broker_snapshot`. Optional: `accounting`.

Preset table: **docs/PROJECT_STRUCTURE.md** (run_checks.py). Full Ops path (import -> require -> pack -> verify -> archive): **docs/OPS_EVIDENCE_GOLDEN_PATH.md**.

---

## Troubleshooting Pointers

- **verify_evidence_pack.py --json `error_code`**  
  `""` = ok. Else: `MISSING_MANIFEST`, `BAD_PATHS`, `CHECKSUM_MISMATCH`, `UNSUPPORTED_SCHEMA`, `FILE_NOT_FOUND`, `UNEXPECTED_ERROR`. Schema and keys: **docs/EVIDENCE_PACK.md** (Verify Evidence Pack --json output schema).

- **Export strict mode**  
  `build_evidence_pack(..., strict=True)` or export CLI `--strict`: fails if optional files are missing. Default: warn and continue. **docs/EVIDENCE_PACK.md** (CLI Usage, Error Handling).

- **Broker snapshot required but not found**  
  Policy `require` with no snapshot for run_id/date -> `ValueError`. Fix: import snapshot first or use `prefer`/`ignore`. **docs/LEDGER_RECONCILIATION.md** (Troubleshooting: Broker snapshot required).

- **Presets and CI**  
  Which preset is blocking vs optional, and how to run locally: **docs/PROJECT_STRUCTURE.md** (run_checks.py Presets).

---

## Verification (Windows)

Single verify block (ASCII, py -3). Copy-paste ready.

```powershell
py -3 scripts/dev/run_checks.py --preset evidence_pack
py -3 scripts/dev/run_checks.py --preset ops_evidence --skip-compile --skip-ruff
```

More commands (export, verify ZIP, import, full golden path): **docs/OPS_EVIDENCE_GOLDEN_PATH.md**.
