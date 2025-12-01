# Legacy Overview - Assembled Trading AI

**Letzte Aktualisierung:** 2025-01-15

**Status**: Work in Progress - Diese Dokumentation wird nach und nach ergänzt.

## Ziel

Dieses Dokument dient als Inventur der "alten Welt" - historische PowerShell-Jobs, Task-Scheduler-Tasks, alte Python-Skripte und andere Legacy-Komponenten, die vor der neuen Core-Architektur (Phase 4+) verwendet wurden.

## PowerShell-Jobs / Task Scheduler

### Bekannte Jobs (aus Code-Analyse)

| Job-Name | Startet Skript(e) | Aufgabe | Status |
|----------|-------------------|---------|--------|
| **TODO: EOD-Daily-Job** | `scripts/run_eod_pipeline.py` (oder Legacy-Version) | Täglicher EOD-Lauf (Execute → Backtest → Portfolio → QA) | ⚠️ In Migration |
| **TODO: Backtest-Job** | `scripts/run_backtest_strategy.py` (oder Legacy-Version) | Einmaliger Backtest-Run | ⚠️ In Migration |
| **TODO: Intraday-Pull-Job** | `scripts/live/pull_intraday.ps1` | Intraday-Daten-Pull (AlphaVantage) | ⚠️ In Migration |
| **TODO: Resample-Job** | `scripts/50_resample_intraday.ps1` | Resampling 1m → 5m | ⚠️ In Migration |
| **TODO: QC-Job** | `scripts/51_qc_intraday_gaps.ps1` | Quality-Check für Intraday-Daten | ⚠️ In Migration |

### Unbekannte Jobs (Platzhalter)

| Job-Name | Startet Skript(e) | Aufgabe | Status |
|----------|-------------------|---------|--------|
| **TODO: [Job-Name]** | `TODO: [Skript-Pfad]` | TODO: [Beschreibung] | ❓ Unbekannt |
| **TODO: [Job-Name]** | `TODO: [Skript-Pfad]` | TODO: [Beschreibung] | ❓ Unbekannt |

**Hinweis**: Bitte ergänzen Sie diese Tabelle mit weiteren bekannten Jobs aus Ihrem Task-Scheduler.

---

## Alte Python-Skripte

### Bekannte Legacy-Skripte (aus Code-Analyse)

| Skript | Zweck | Status | Ersetzt durch |
|--------|-------|--------|---------------|
| `scripts/sprint9_backtest.py` | Sprint-9-Backtest | ⚠️ Legacy | `scripts/run_backtest_strategy.py` |
| `scripts/sprint9_execute.py` | Sprint-9-Execute | ⚠️ Legacy | `scripts/run_eod_pipeline.py` |
| `scripts/sprint10_portfolio.py` | Sprint-10-Portfolio | ⚠️ Legacy | `scripts/run_eod_pipeline.py` |
| `scripts/run_daily.py` | Täglicher Run (Legacy) | ⚠️ Legacy | `scripts/run_eod_pipeline.py` |
| `scripts/sprint8_cost_model.ps1` | Cost-Model-Tests | ⚠️ Legacy | `src/assembled_core/costs.py` |
| `scripts/sprint9_dashboard.ps1` | Dashboard-Generierung | ⚠️ Legacy | TODO: Phase 5/6 |
| `scripts/sprint9_cost_grid.ps1` | Cost-Grid-Parameter-Sweep | ⚠️ Legacy | TODO: Phase 5/6 |
| `scripts/sprint10_param_sweep.ps1` | Parameter-Sweep | ⚠️ Legacy | TODO: Phase 5/6 |
| `scripts/sprint10_portfolio_simple.ps1` | Einfache Portfolio-Simulation | ⚠️ Legacy | `scripts/run_eod_pipeline.py` |
| `scripts/run_sprint8_rehydrate.ps1` | Rehydrate (Legacy) | ⚠️ Legacy | TODO: Phase 5/6 |
| `scripts/run_all_sprint10.ps1` | Sprint-10-All-in-One | ⚠️ Legacy | `scripts/run_eod_pipeline.py` |

### Daten-Ingest-Skripte (Legacy)

| Skript | Zweck | Status | Ersetzt durch |
|--------|-------|--------|---------------|
| `scripts/data/pull_stooq_eod.py` | Stooq EOD-Pull | ⚠️ Legacy | `src/assembled_core/data/prices_ingest.py` |
| `scripts/data/pull_alpha_vantage_intraday.py` | AlphaVantage Intraday-Pull | ⚠️ Legacy | `scripts/live/pull_intraday.py` |
| `scripts/data/pull_coingecko_ohlc.py` | CoinGecko OHLC-Pull | ⚠️ Legacy | TODO: Phase 5/6 |
| `scripts/data/pull_ecb_fx.py` | ECB FX-Pull | ⚠️ Legacy | TODO: Phase 5/6 |
| `scripts/31_assemble_intraday.ps1` | Intraday-Assembly | ⚠️ Legacy | TODO: Phase 5/6 |
| `scripts/50_resample_intraday.ps1` | Resampling 1m → 5m | ⚠️ Legacy | TODO: Phase 5/6 |
| `scripts/51_qc_intraday_gaps.ps1` | QC für Intraday-Gaps | ⚠️ Legacy | `src/assembled_core/qa/health.py` |
| `scripts/52_make_acceptance_intraday_sprint7.ps1` | Acceptance-Tests | ⚠️ Legacy | `tests/test_*` |

### Feature-Engineering-Skripte (Legacy)

| Skript | Zweck | Status | Ersetzt durch |
|--------|-------|--------|---------------|
| `scripts/features/build_daily_features.py` | Daily-Features-Build | ⚠️ Legacy | `src/assembled_core/features/ta_features.py` |

### Development-Skripte (Legacy)

| Skript | Zweck | Status | Ersetzt durch |
|--------|-------|--------|---------------|
| `scripts/dev/create_sample_eod_data.py` | Sample-EOD-Daten | ⚠️ Legacy | `scripts/00_seed_demo_data.py` |
| `scripts/dev/resample_1m_to_5m.py` | Resampling-Dev | ⚠️ Legacy | TODO: Phase 5/6 |
| `scripts/dev/quick_daily_backtest.py` | Quick-Backtest | ⚠️ Legacy | `scripts/run_backtest_strategy.py` |
| `scripts/dev/fix_resample_5min.py` | Fix-Resample | ⚠️ Legacy | TODO: Phase 5/6 |

---

## PowerShell-Skripte (Legacy)

### Bekannte Legacy-PS-Skripte

| Skript | Zweck | Status | Ersetzt durch |
|--------|-------|--------|---------------|
| `scripts/run_live_pipeline.ps1` | Live-Pipeline | ⚠️ Legacy | `scripts/run_eod_pipeline.py` |
| `scripts/run_all_sprint10.ps1` | Sprint-10-All-in-One | ⚠️ Legacy | `scripts/run_eod_pipeline.py` |
| `scripts/live/pull_intraday.ps1` | Intraday-Pull (PS) | ⚠️ Legacy | `scripts/live/pull_intraday.py` |

### Tools (Legacy, aber noch in Verwendung)

| Skript | Zweck | Status |
|--------|-------|--------|
| `scripts/tools/package_artifacts.ps1` | Package-Artefakte | ✅ In Verwendung |
| `scripts/tools/notify_discord.ps1` | Discord-Notifications | ✅ In Verwendung |
| `scripts/tools/git_sync.ps1` | Git-Sync | ✅ In Verwendung |
| `scripts/tools/fix_indent.ps1` | Fix-Indentation | ✅ In Verwendung |
| `scripts/tools/fix_all_project.ps1` | Fix-All-Project | ✅ In Verwendung |
| `scripts/tools/convert_heredocs.ps1` | Convert-Heredocs | ✅ In Verwendung |
| `scripts/tools/activate_python.ps1` | Activate-Python | ✅ In Verwendung |
| `scripts/ps/ps_py_utils.ps1` | PS-Python-Utils | ✅ In Verwendung |
| `scripts/ps/fix_heredocs.ps1` | Fix-Heredocs | ✅ In Verwendung |

---

## Batch-Dateien / Sonstiges

### Bekannte Batch-Dateien

| Datei | Zweck | Status |
|-------|-------|--------|
| **TODO: [Batch-Datei]** | TODO: [Beschreibung] | ❓ Unbekannt |

**Hinweis**: Bitte ergänzen Sie diese Liste mit bekannten Batch-Dateien.

---

## Legacy-Ordner

| Ordner | Zweck | Status |
|--------|-------|--------|
| `legacy/` | Legacy-Skripte (archiviert) | 📦 Archiviert |
| `archive/` | Archivierte Dateien | 📦 Archiviert |
| `backup/` | Backup-Dateien | 📦 Backup |

---

## Migrations-Status

### ✅ Abgeschlossen (Phase 4)

- Backtest-Engine → `src/assembled_core/qa/backtest_engine.py`
- QA-Metriken → `src/assembled_core/qa/metrics.py`
- QA-Gates → `src/assembled_core/qa/qa_gates.py`
- TA-Features → `src/assembled_core/features/ta_features.py`
- EOD-Pipeline → `scripts/run_eod_pipeline.py`
- Strategy-Backtest → `scripts/run_backtest_strategy.py`
- Phase-4-Tests → `scripts/run_phase4_tests.ps1` / `pytest -m phase4`

### ⚠️ In Migration (Phase 5)

- Intraday-Pull → TODO: Phase 5
- Resampling → TODO: Phase 5
- QC-Gaps → TODO: Phase 5

### ❓ Unbekannt / Zu klären

- Dashboard-Generierung
- Parameter-Sweep
- Cost-Grid
- Rehydrate

---

## Hinweise

1. **Work in Progress**: Diese Dokumentation wird nach und nach ergänzt, wenn weitere Legacy-Komponenten identifiziert werden.

2. **Platzhalter**: Alle Einträge mit "TODO:" sind Platzhalter und müssen vom Benutzer ausgefüllt werden.

3. **Status-Legende**:
   - ✅ In Verwendung
   - ⚠️ Legacy (wird ersetzt)
   - 📦 Archiviert
   - ❓ Unbekannt

4. **Ergänzungen**: Bitte ergänzen Sie diese Dokumentation mit weiteren bekannten Legacy-Komponenten aus Ihrem System.

