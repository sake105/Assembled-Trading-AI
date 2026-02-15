# ROADMAP STATUS - Sprint 13 - DONE

**Purpose:** Final status for Sprint 13 - Accounting/Ledger/Evidence Pack. All items done; no remaining work tracked here.

**References:** docs/LEDGER_RECONCILIATION.md | docs/EVIDENCE_PACK.md | docs/PROJECT_STRUCTURE.md | docs/RELEASE_NOTES_SPRINT13.md

---

## Definition of Done (checked)

- [x] **evidence_pack preset green** — `py -3 scripts/dev/run_checks.py --preset evidence_pack`
- [x] **ops_evidence preset green** — `py -3 scripts/dev/run_checks.py --preset ops_evidence --skip-compile --skip-ruff`
- [x] **evidence-pack CI exists (blocking)** — `.github/workflows/evidence-pack-ci.yml`; `evidence_pack` preset must pass.
- [x] **ops-evidence CI exists (non-blocking by separation, but red on failure)** — `.github/workflows/ops-evidence-ci.yml`; runs `ops_evidence` only; logs artifact on failure.
- [x] **verify JSON schema stable** — `verify_evidence_pack.py --json` stable keys; schema in docs/EVIDENCE_PACK.md; tests: test_verify_evidence_pack_json_schema_stable.py, test_verify_evidence_pack_cli_smoke.py.

---

## Sprint 13 Scope (Done)

### Core Ledger & Reconciliation (H-L)

| Feature | Evidence |
|---------|----------|
| Ledger events & storage | ledger.py, ledger_store.py, test_ledger_*.py |
| Position engine | position_engine.py |
| Reconciliation engine | reconciliation.py, reconciliation_report.py, test_reconciliation_*.py |
| Ledger integration | ledger_integration.py (EOD/Backtest/Daily) |
| Reconciliation reports | CSV, JSON, Markdown; broker meta, fixed schema |

### Broker Snapshot & Ops (M-P)

| Feature | Evidence |
|---------|----------|
| Broker snapshot system | broker_snapshot.py, broker_snapshot_store.py |
| Broker snapshot import | broker_snapshot_importer.py, scripts/import_broker_snapshot.py |
| Broker snapshot policy | ignore / prefer / require; test_broker_snapshot_policy_*.py |
| Evidence Index | evidence_index.py, test_evidence_index_written.py |

### Additional (Done)

Schema versioning (schema_version: 1), accounting reports, candidate gate, daily manifest, ops-safe CLIs. CI: accounting-ci.yml, evidence-pack-ci.yml, ops-evidence-ci.yml.

---

## Key files changed (max 10)

- `src/assembled_core/accounting/ledger.py`
- `src/assembled_core/accounting/ledger_store.py`
- `src/assembled_core/accounting/reconciliation.py`
- `src/assembled_core/accounting/broker_snapshot_importer.py`
- `src/assembled_core/accounting/evidence_index.py`
- `src/assembled_core/accounting/evidence_pack.py`
- `scripts/import_broker_snapshot.py`
- `scripts/verify_evidence_pack.py`
- `scripts/dev/run_checks.py`
- `.github/workflows/evidence-pack-ci.yml`

---

## CI inventory

- **Evidence Pack CI (Windows)** — [.github/workflows/evidence-pack-ci.yml](.github/workflows/evidence-pack-ci.yml) — Blocking: evidence_pack. Optional: broker_snapshot, accounting.
- **Ops Evidence CI (Windows)** — [.github/workflows/ops-evidence-ci.yml](.github/workflows/ops-evidence-ci.yml) — ops_evidence only; red on failure; logs artifact.
- **Accounting CI (Windows)** — [.github/workflows/accounting-ci.yml](.github/workflows/accounting-ci.yml) — Blocking: broker_snapshot. Optional: accounting.

Preset table: docs/PROJECT_STRUCTURE.md (run_checks.py).

---

## Copy/paste (Windows)

Single block; exactly 2 commands.

```powershell
py -3 scripts/dev/run_checks.py --preset evidence_pack
py -3 scripts/dev/run_checks.py --preset ops_evidence --skip-compile --skip-ruff
```

---

## Determinism & links

Determinism: event IDs (SHA256), float formatting (Decimal), stable sort, JSON sort_keys+indent+newline, atomic writes, POSIX paths. See docs/EVIDENCE_PACK.md, docs/LEDGER_RECONCILIATION.md.

**Status:** Sprint 13 DONE. Definition of Done met; key files and CI documented; one copy/paste block above.
