# ROADMAP STATUS - Sprint 13 (Accounting & Ledger System)

**Zweck:** Status-Übersicht für Sprint 13 - Accounting/Ledger/Reconciliation System

**Erstellt:** 2025-01-23  
**Basis:** Implementierung aus Sprint 13 (L1-L5, Broker Snapshot, Evidence Index, Schema Versioning)

---

## Sprint 13 Status: ✅ DONE

### Done (H-L): Core Ledger & Reconciliation

| Feature | Status | Evidence |
|---------|--------|----------|
| **H) Ledger Events & Storage** | ✅ done | `src/assembled_core/accounting/ledger.py` |
| | | `src/assembled_core/accounting/ledger_store.py` |
| | | `tests/test_ledger_*.py` |
| **I) Position Engine** | ✅ done | `src/assembled_core/accounting/position_engine.py` |
| | | Average cost basis, realized/unrealized PnL |
| **J) Reconciliation Engine** | ✅ done | `src/assembled_core/accounting/reconciliation.py` |
| | | `src/assembled_core/accounting/reconciliation_report.py` |
| | | `tests/test_reconciliation_*.py` |
| **K) Ledger Integration** | ✅ done | `src/assembled_core/accounting/ledger_integration.py` |
| | | Integration in EOD/Backtest/Daily pipelines |
| **L) Reconciliation Reports** | ✅ done | CSV, JSON, Markdown formats |
| | | Broker meta fields, fixed CSV schema |

### Done (M-P): Broker Snapshot & Ops Features

| Feature | Status | Evidence |
|---------|--------|----------|
| **M) Broker Snapshot System** | ✅ done | `src/assembled_core/accounting/broker_snapshot.py` |
| | | `src/assembled_core/accounting/broker_snapshot_store.py` |
| | | Normalization, atomic writes, deterministic storage |
| **N) Broker Snapshot Import** | ✅ done | `src/assembled_core/accounting/broker_snapshot_importer.py` |
| | | `scripts/import_broker_snapshot.py` (standalone CLI) |
| | | Robust parsing (strings, thousands separators, duplicates) |
| **O) Broker Snapshot Policy** | ✅ done | `ignore`, `prefer`, `require` policies |
| | | Integration in EOD/Backtest/Daily pipelines |
| | | `tests/test_broker_snapshot_policy_*.py` |
| **P) Evidence Index** | ✅ done | `src/assembled_core/accounting/evidence_index.py` |
| | | Central JSON linking all artifacts |
| | | `tests/test_evidence_index_written.py` |

### Additional Features (Beyond Original Scope)

| Feature | Status | Evidence |
|---------|--------|----------|
| **Schema Versioning** | ✅ done | All artifacts include `schema_version: 1` |
| | | Broker snapshots, reconciliation reports, accounting reports, manifest, evidence index |
| | | `tests/test_schema_versioning_smoke.py` |
| **Accounting Reports** | ✅ done | `src/assembled_core/accounting/accounting_report.py` |
| | | CSV, JSON formats with broker meta cross-references |
| | | `tests/test_accounting_report_*.py` |
| **Candidate Gate Integration** | ✅ done | `src/assembled_core/qa/candidate_gate.py` |
| | | Combined robustness + reconciliation gates |
| | | `tests/test_candidate_gate_reconciliation.py` |
| **Daily Manifest** | ✅ done | `scripts/run_daily.py` optional manifest |
| | | Aligned with orchestrator manifest schema |
| | | `tests/test_run_daily_manifest_smoke.py` |
| **Ops-Safe CLI Tools** | ✅ done | `scripts/import_broker_snapshot.py` |
| | | Exit codes, ASCII-only errors, strict validation |
| | | `tests/test_import_broker_snapshot_cli_smoke.py` |
| **CI Integration** | ✅ done | `.github/workflows/accounting-ci.yml` |
| | | Windows CI with `run_checks.py` presets |
| | | `--preset broker_snapshot`, `--preset accounting` |

---

## Key Artifacts

### Core Modules
- `src/assembled_core/accounting/ledger.py` - Ledger event generation
- `src/assembled_core/accounting/ledger_store.py` - Parquet storage
- `src/assembled_core/accounting/position_engine.py` - Position tracking, PnL
- `src/assembled_core/accounting/reconciliation.py` - Reconciliation logic
- `src/assembled_core/accounting/reconciliation_report.py` - Report writers
- `src/assembled_core/accounting/accounting_report.py` - Accounting reports
- `src/assembled_core/accounting/broker_snapshot.py` - Snapshot normalization
- `src/assembled_core/accounting/broker_snapshot_store.py` - Snapshot storage
- `src/assembled_core/accounting/broker_snapshot_importer.py` - External import
- `src/assembled_core/accounting/evidence_index.py` - Evidence index writer
- `src/assembled_core/accounting/ledger_integration.py` - Pipeline integration

### CLI Tools
- `scripts/import_broker_snapshot.py` - Standalone snapshot importer
- `scripts/run_eod_pipeline.py` - EOD pipeline with broker snapshot controls
- `scripts/run_backtest_strategy.py` - Backtest with broker snapshot controls
- `scripts/run_daily.py` - Daily run with broker snapshot controls

### Tests
- `tests/test_ledger_*.py` - Ledger event tests
- `tests/test_reconciliation_*.py` - Reconciliation tests
- `tests/test_broker_snapshot_*.py` - Broker snapshot tests
- `tests/test_accounting_report_*.py` - Accounting report tests
- `tests/test_evidence_index_*.py` - Evidence index tests
- `tests/test_schema_versioning_*.py` - Schema versioning tests
- `tests/test_ops_evidence_pack_e2e.py` - E2E ops workflow test

### Documentation
- `docs/LEDGER_RECONCILIATION.md` - Complete system documentation
- `docs/PROJECT_STRUCTURE.md` - Golden Path (Ops) workflows
- `.github/workflows/accounting-ci.yml` - CI workflow

---

## Determinism & Reproducibility

All components follow strict determinism rules:
- **Event IDs**: SHA256 hash of canonical fields (stable across runs)
- **Float Formatting**: `Decimal.quantize()` for canonical representation
- **Sorting**: Stable `mergesort` algorithm
- **JSON Serialization**: `sort_keys=True`, `indent=2`, trailing newline
- **Atomic Writes**: Temp file → rename/move (Windows-safe)
- **Path Normalization**: Relative POSIX paths in manifests

---

## Ops-Ready Features

### Evidence Index
- Central JSON file linking all artifacts per run/date
- Location: `output/evidence_<run_id>/evidence_<YYYY-MM-DD>.json`
- Links: snapshot, ledger, reconcile, accounting, manifest

### Schema Versioning
- All artifacts include `schema_version: 1`
- Enables future schema evolution
- Backward compatible (defaults to `1` if missing)

### Broker Snapshot Policy
- `ignore`: Never use snapshots (always paper view)
- `prefer`: Use snapshot if available, fallback to paper view
- `require`: Snapshot must exist (fail-fast if missing)

### Troubleshooting
- Clear error messages with expected paths
- Namespace mismatch detection
- Reconciliation failure diagnostics
- Import failure handling

---

## Regression Checks

**Windows CI:**
```bash
python scripts/dev/run_checks.py --preset broker_snapshot
python scripts/dev/run_checks.py --preset accounting
```

**Manual (Windows):**
```powershell
py -3 -m py_compile src/assembled_core/accounting/*.py
py -3 -m ruff check src/assembled_core/accounting/
py -3 -m pytest -q tests/test_ledger*.py tests/test_reconciliation*.py tests/test_broker_snapshot*.py
```

---

## Next Steps (Future Enhancements)

1. **Schema Migration**: When schema version 2 is needed, provide migration helpers
2. **Broker API Integration**: Direct broker API snapshot fetching
3. **Real-time Reconciliation**: Continuous reconciliation during trading
4. **Advanced Reporting**: BI/ETL integration for accounting reports
5. **Audit Trail**: Enhanced event logging for compliance

---

**Status:** ✅ Sprint 13 Complete - All core features implemented, tested, and documented.
