# Audit 01 — Dead Paths & Ballast

- **Date:** 2026-05-30
- **Agent:** AGENT 1 of 5 (read-only system audit)
- **Scope:** Static import/call-graph analysis of `src/assembled_core/` and `scripts/`. Identifies modules with no reachable productive import path (cli, paper pipeline, API, scripts runners), orphaned scripts, stub/dummy functions, and stale docs.
- **Method:** Grep-based import tracing from entry points (`scripts/run_*.py`, `scripts/cli.py`, `scripts/commands/`, `src/assembled_core/pipeline/`, `src/assembled_core/api/`). Dynamic imports (lazy `from X import Y` inside functions) were traced individually. Registry/plugin patterns checked by name search. Archive/ directory excluded from analysis (explicitly archived).
- **Caution:** 512 Python files in `src/`. Transitive reachability was checked one level deep from known entry points. A module is marked TOT only where zero productive-path imports were found. UNSURE is used where reachability is ambiguous or only test-side.

---

## 1. Main Findings Table

| Datei / Funktion | Erreichbar von wo? | Beleg (file:line) | Verdikt |
|---|---|---|---|
| `src/assembled_core/ml/logic_tensor_network.py` | Kein Produktivimport gefunden — nur eigene Datei + kein `from` in `src/` oder `scripts/` | Grep: `LogicTensorNetwork\|logic_tensor_network` → nur `ml/logic_tensor_network.py` selbst | TOT (Stub) |
| `src/assembled_core/ml/temporal_fusion_transformer.py` | Kein Produktivimport gefunden — `TFTForecaster\|temporal_fusion_transformer` kommt nur in der eigenen Datei vor | Grep vollständig negativ für `src/` und `scripts/` | TOT (Stub) |
| `src/assembled_core/domain/trading/__init__.py` | Nur eigene Datei. Inhalt: Docstring „Empty during Month-1 skeleton phase". Kein Code. Kein Import aus `src/`. | `domain/trading/__init__.py`:6 — „Empty during the Month-1 skeleton phase" | TOT (Leeres Skeleton) |
| `src/assembled_core/domain/risk/__init__.py` | Kein Import aus `src/` oder `scripts/`. Nur Docstring-Zeile. | `domain/risk/__init__.py`:1 — 1-Zeile Docstring, kein Code | TOT (Leeres Skeleton) |
| `src/assembled_core/domain/accounting/__init__.py` | Wie oben | `domain/accounting/__init__.py`:1 | TOT (Leeres Skeleton) |
| `src/assembled_core/domain/research/__init__.py` | Wie oben | `domain/research/__init__.py` (nur Docstring) | TOT (Leeres Skeleton) |
| `src/assembled_core/domain/operations/__init__.py` | Wie oben | `domain/operations/__init__.py` (nur Docstring) | TOT (Leeres Skeleton) |
| `src/assembled_core/adapters/inbound/__init__.py` | Kein Import. Inhalt: ein Docstring „Driving (inbound) adapters". | `adapters/inbound/__init__.py`:1 | TOT (Leeres Skeleton) |
| `src/assembled_core/bootstrap/container.py` + `bootstrap/__init__.py` | Nur von `application/use_cases/record_kill_switch_trip.py` importiert, welches selbst nur von Tests referenziert wird — kein Produktionspfad | Grep `bootstrap\.container\|build_production_container` → nur 4 Dateien, davon 3 Tests/selbst | UNSURE (nur Test-Pfad) |
| `src/assembled_core/application/use_cases/record_kill_switch_trip.py` | Nur von Tests referenziert. Kein Import aus `src/` (außer eigenem) oder `scripts/`. | Grep `record_kill_switch_trip` → 2 Dateien: eigene + `tests/test_hexagonal_skeleton.py` | UNSURE (nur Test-Pfad) |
| `src/assembled_core/certify/` (alle 4 Dateien) | Kein Import aus `src/` oder `scripts/`. Nur Tests (`test_certify.py`) + intern. | Grep `certify\|ReproducibilityCertificate` → kein Produktiv-Script | UNSURE (nur Test-Pfad) |
| `src/assembled_core/compliance/elster.py` | Kein Import außerhalb von `compliance/__init__.py` und Tests | Grep `elster` → nur Tests + eigene Datei | UNSURE (nur Test-Pfad) |
| `src/assembled_core/compliance/` (ganzes Paket) | Kein `from src.assembled_core.compliance` in `src/` oder `scripts/` (außer eigenem `__init__`) | Grep vollständig negativ für Produktivpfade | UNSURE (nur Test-Pfad) |
| `src/assembled_core/attribution/` (ganzes Paket) | Kein Import aus `src/` außerhalb der eigenen Dateien. Nicht in `scripts/`. Nur Tests. | Grep `assembled_core\.attribution` in `src/` → nur interne Querverweise | UNSURE (nur Test-Pfad) |
| `src/assembled_core/experiments/batch_config.py` | Kein Import aus `src/` (`__init__` selbst-referenz). Scripts: kein Treffer. Nur Tests. | Grep `assembled_core\.experiments` in `scripts/` → 0 Treffer | UNSURE (nur Test-Pfad) |
| `src/assembled_core/strategy/` (config, hyperparameter, experiment_tracker) | Kein Import aus `src/` außer eigenem `experiment_tracker`. Scripts: kein Treffer. Nur Tests. | Grep `assembled_core\.strategy` → 4 Dateien, alle Tests oder eigene | UNSURE (nur Test-Pfad) |
| `src/assembled_core/data/shipping/__init__.py` | Nur leere `__all__=[]`. Kein Code. Nicht aus `src/` oder `scripts/` referenziert (nur `test_import_all_core_modules.py`). | `data/shipping/__init__.py`:5 — `__all__: list[str] = []` | TOT (Leeres Paket) |
| `src/assembled_core/data/streaming/__init__.py` | Wie oben — leere `__all__` | `data/streaming/__init__.py`:5 — `__all__: list[str] = []` | TOT (Leeres Paket) |
| `src/assembled_core/data/sources/stooq_source.py` | Kein Import in `src/` oder `scripts/`. Nur `test_free_stack_modules.py`. | Grep `stooq_source` → 1 Treffer: nur Test | UNSURE (nur Test-Pfad) |
| `src/assembled_core/data/sources/weather_source.py` | Kein Import in `src/` außer eigenem. Nicht in `scripts/`. | Grep `weather_source` in `src/` → nur `altdata_wikipedia_features.py` — letztere selbst ohne Produktiv-Import | UNSURE (nur Test-Pfad, 2-fach isoliert) |
| `src/assembled_core/data/sources/wikipedia_views_source.py` | Nur von `features/altdata_wikipedia_features.py` importiert, welches selbst kein Produktiv-Import hat | Grep `altdata_wikipedia_features` in `src/` → 0 Produktiv-Importe | UNSURE (isoliert) |
| `src/assembled_core/data/panel_store.py` | Kein Import aus `src/` oder `scripts/` (Produktivpfad). Nur 2 Tests. | Grep `panel_store\b` → nur `test_data_factor_store.py` + `test_ml_wave2.py` | UNSURE (nur Test-Pfad) |
| `src/assembled_core/data/data_versioning.py` | Wie `panel_store.py` — nur Testimporte, keine Produktionspfade | Grep `data_versioning` → nur 2 Test-Dateien | UNSURE (nur Test-Pfad) |
| `src/assembled_core/signals/lppls_crash.py` | Kein Import aus `src/` (Grep `lppls_crash` in `src/` → 0). Kein Scripts-Import. Nur Tests. | Grep vollständig: `tests/test_signals_lppls_validation.py` + `test_competitive_analysis_impl.py` | UNSURE (nur Test-Pfad) |
| `src/assembled_core/signals/cross_asset_carry_v2.py` | Kein Import irgendwo in `src/` oder `scripts/`. Tests: `test_competitive_analysis_impl.py` | Grep `cross_asset_carry_v2` in `src/` → 0 Treffer; in ganzer Basis → 2 Test-only | UNSURE (nur Test-Pfad) |
| `src/assembled_core/pipeline/dispatcher.py` | Kein Import aus Produktivpfad. Nur `tests/test_non_paid_modules.py` | Grep `pipeline\.dispatcher` → 1 Treffer: Test | UNSURE (nur Test-Pfad) |
| `src/assembled_core/features/chart_pattern_matrix.py` | Kein direkter Import aus `src/` (keine `from src.assembled_core.features.chart_pattern_matrix`). `composite_score.py` hat eigene `chart_pattern_score()` Stub (returns 0.0) ohne Importnutzung. | Grep `chart_pattern_matrix` in `src/` → 0 Treffer | TOT |
| `src/assembled_core/features/altdata_bls_features.py` | Kein Import in `src/` oder `scripts/`. Nur `test_altdata_feature_builders.py`. | Grep `altdata_bls_features` → 1 Test-Datei | UNSURE (nur Test-Pfad) |
| `src/assembled_core/features/altdata_finra_features.py` | Wie oben | Grep `altdata_finra_features` → 1 Test-Datei | UNSURE (nur Test-Pfad) |
| `src/assembled_core/features/altdata_wikipedia_features.py` | Kein Import in `src/` Produktivpfad. | Grep `altdata_wikipedia_features` in `src/` → 0 | UNSURE (nur Test-Pfad) |

---

## 2. Stub / Dummy Functions

Vollständige Stubs: `fit()` und `predict()` werfen immer `NotImplementedError`, kein Produktivpfad existiert.

| Datei | Funktion(en) | Zeilen | Evidenz |
|---|---|---|---|
| `src/assembled_core/ml/logic_tensor_network.py` | `LogicTensorNetwork.fit()`, `LogicTensorNetwork.predict()` | 98–122 | `raise NotImplementedError("LTN fit: full implementation pending ltn setup")` — beide Zweige (mit und ohne ltn) werfen. Docstring: „stub". |
| `src/assembled_core/ml/temporal_fusion_transformer.py` | `TFTForecaster.fit()`, `TFTForecaster.predict()`, `tft_forecast()` | 96–159 | `raise NotImplementedError("TFT fit: full implementation pending pytorch-forecasting setup")`. Docstring explizit: „fit() and predict() raise NotImplementedError regardless of whether pytorch_forecasting is installed". |
| `src/assembled_core/ml/gnn_signal.py` | `GNNSignalModel.fit()` | 131–157 | `raise NotImplementedError("GNNSignalModel.fit() requires torch-geometric")` + `raise NotImplementedError("Full GNN training not yet implemented")`. `predict()` gibt in stub-mode Zero-Scores zurück (nicht NotImplementedError), also nur fit() ist DUMMY. |
| `src/assembled_core/signals/composite_score.py` | `chart_pattern_score()` | 251–256 | `return 0.0` mit Kommentar „Placeholder: returns 0 until ML model is trained. Phase 3: replace with stumpy Matrix-Profile." — Hardcoded fake. |
| `src/assembled_core/domain/trading/__init__.py` | gesamtes Modul | 1–7 | Nur Docstring „Empty during the Month-1 skeleton phase" — kein ausführbarer Code. |
| `src/assembled_core/domain/risk/__init__.py` | gesamtes Modul | 1 | Nur Docstring-Zeile. |
| `src/assembled_core/domain/accounting/__init__.py` | gesamtes Modul | 1 | Nur Docstring-Zeile. |
| `src/assembled_core/domain/research/__init__.py` | gesamtes Modul | 1 | Nur Docstring-Zeile. |
| `src/assembled_core/domain/operations/__init__.py` | gesamtes Modul | 1 | Nur Docstring-Zeile. |

---

## 3. Orphaned Scripts

Scripts, die kein Produktivsystem referenziert und die nicht als reguläre Entry-Points fungieren:

| Script | Befund | Verdikt |
|---|---|---|
| `scripts/dev/tmp_script.py` | Inhalt: `print("hello")` — 2 Zeilen. Temporäres Wegwerf-Skript. | TOT (wegwerfen) |
| `scripts/dev/tmp_check.py` | 5-Zeilen Equity-Curve-Debug-Check. Temporär. | TOT (wegwerfen) |
| `scripts/dev/tmp_peek_ec.py` | 4-Zeilen Equity-Curve-Read. Temporär. | TOT (wegwerfen) |
| `scripts/_crisis_alpha_pit_verify.py` | Untracked (git status). Steht isoliert, kein anderes Script referenziert es. Ist ein eigenständiges PIT-Verifikationsskript (direkt runnable). | BEHALTEN (standalone research tool) |
| `scripts/_crisis_alpha_backtest_compare.py` | Untracked (git status). Standalone-Backtest-Vergleich. Referenced in `docs/cleanup/01c_cleanup_ergebnis.md`. | BEHALTEN (standalone research tool) |
| `scripts/_crisis_alpha_replay.py` | Untracked. Standalone-Replay-Skript. Referenced in `docs/cleanup/`. | BEHALTEN (standalone research tool) |
| `scripts/backtest_news_alpha.py` | Untracked (git status). Standalone-Backtest für news_alpha Modul. Docstring: „Run: python scripts/backtest_news_alpha.py". | BEHALTEN (standalone research entry point) |

Alle anderen `scripts/_oos_wf_*.py` und `scripts/_cpcv_tb_leakage_check.py` haben korrespondierende `docs/results/*.md`-Artefakte — sie sind standalone OOS-Analyseskripte, kein produktiver Ballast.

---

## 4. Stale / Superseded Markdown Docs

Nur eindeutige Fälle gelistet; konservative Schwelle (Datum + inhaltliche Evidenz).

| Datei | Befund | Verdikt |
|---|---|---|
| `docs/refactor_plan_sprint9.md` | Leer (1 Zeile). Sprint 9 ist abgeschlossen. | TOT (leer + veraltet) |
| `docs/cost_model_refactor_summary.md` | Leer. Refactor ist abgeschlossen. | TOT (leer + veraltet) |
| `docs/pipeline_refactor_step1_summary.md` | Datum fehlt. Beschreibt Sprint-9-Extraktion, die längst abgeschlossen ist (pipeline/ existiert produktiv). | UNSURE (veraltet, aber nicht leer) |
| `docs/ema_config_refactor_summary.md` | Beschreibt Sprint-9-EMA-Config-Refactor (abgeschlossen, `ema_config.py` produktiv). Kein Datum sichtbar. | UNSURE (veraltet, aber nicht leer) |
| `docs/LEGACY_OVERVIEW.md` | Datum: 2025-01-15, Status „Work in Progress". Über 17 Monate alt. | UNSURE (möglicherweise veraltet) |
| `docs/NEXT_STEPS_RECOMMENDATIONS.md` | Datum: 2025-12-09. Inhalt beschreibt Download-Status aus 2025 — durch spätere Sessions längst überholt. | UNSURE (veraltet) |
| `docs/OPTIMIZATION_AND_IMPROVEMENTS.md` | Datum: 2025-12-22. Drei weitere OPTIMIZATION_*.md-Dateien gleiches Datum. Optimierungen als „abgeschlossen" markiert, aber Repo hat sich seit 2025-12 massiv verändert. | UNSURE (möglicherweise veraltet) |
| `docs/OPTIMIZATION_IMPLEMENTATION_STATUS.md` | Wie oben — 2025-12-22. | UNSURE |
| `docs/OPTIMIZATION_IMPLEMENTATION_SUMMARY.md` | Wie oben — 2025-12-22. | UNSURE |
| `docs/FURTHER_IMPROVEMENTS_SUGGESTIONS.md` | Wie oben — 2025-12-22. | UNSURE |
| `docs/E2E_TEST_STABILIZATION.md` | Datum: 2025-12-23. Beschreibt Test-Stabilisierung, seitdem viele Testzyklen durchgelaufen. | UNSURE (veraltet) |
| `docs/PIPELINE_INTEGRATION_TEST_RESULTS.md` | Datum: 2025-12-09. Beschreibt Integrationstests aus 2025. | UNSURE (veraltet) |
| `docs/CURSOR_NEXT_10.md` | Cursor-Prompt-Datei aus Sprint-9-Ära. Cursor-Workflow wurde durch Claude Code ersetzt. | UNSURE (veraltet) |

---

## 5. Cross-Check gegen vorhandene Artefakte

### `docs/audit/trading_cycle_dead_imports.csv`

Dieses Artefakt (Stand 2026-04-26) listet **nur 4 Einträge** — alle als `OK` markiert. Es ist kein Dead-Import-Report, sondern ein Coverage-Lauf-Artefakt der Datei `trading_cycle_shared.py`. **Nicht nützlich als Dead-Code-Quelle.**

### `docs/audit/trading_cycle_coverage_2026-04-26.txt`

Coverage-Run-Output mit vielen Skips (`s`) und einer CoverageWarning: `Module src/assembled_core/pipeline/trading_cycle was never imported`. Das bedeutet: `pipeline/trading_cycle.py` (der Thin-Alias-Shim) wurde beim Coverage-Run nicht geladen. **Befund hier ist konsistent**: `trading_cycle.py` ist nur ein Shim für Patch-Targets, kein direkter Produktionseinstieg.

Die Coverage-Datei gibt keine Information über Dead-Module außerhalb des Trading-Cycle — sie ist zu eng gefasst, um als generelle Dead-Code-Basis zu dienen.

---

## 6. Analyse-Grenzen

1. **Dynamische Imports**: Mehrere Module werden via `from X import Y` innerhalb von `try/except`-Blöcken oder `if`-Guards geladen (z.B. `gnn_signal`, `cboe_source`, `tick_store`). Diese wurden einzeln nachverfolgt, könnten aber bei unbekannten Policy-Konfigurationspfaden aktiviert werden.
2. **Entry-Point-Registrierung**: `pyproject.toml` nicht vollständig auf Entry-Points geprüft. Bekannte Entry-Points (`assembled-run-daily`, `assembled-cli`) wurden berücksichtigt.
3. **Hexagonales Skeleton** (`domain/`, `adapters/inbound/`, `bootstrap/`, `ports/`, `application/`): Diese Module sind by-design leer oder test-only — sie sind Teil eines Migration-Plans (audit C-001..C-007), nicht produktiver Ballast im klassischen Sinn. Als UNSURE/TOT markiert, nicht als Fehler.
4. **`scripts/dev/`-Ordner**: Enthält sowohl legitime Dev-Tools als auch temporäre Debug-Scripts. Nur eindeutige Temporärskripte als TOT markiert.

---

**Hinweis: Nichts wurde gelöscht oder verändert. Dieser Report ist ausschließlich eine Auflistung.**
