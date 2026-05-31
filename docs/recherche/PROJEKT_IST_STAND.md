# PROJEKT_IST_STAND.md
Stand: 2026-05-27 | Branch: main | Commit: a7e01689

---

## 1. Verzeichnisbaum (3 Ebenen, ohne .git / data / __pycache__ / .venv / .hypothesis/constants)

```
.
+-- .claude/
|   +-- .review_markers/        (74 .done-Dateien + 2 JSON — Review-Chain-Artefakte)
|   +-- agents/
|   |   +-- ci-debugger.md
|   |   +-- docs-governance-sync.md
|   |   +-- memory-tracker.md
|   |   +-- risk-execution-reviewer.md
|   |   +-- senior-code-reviewer.md
|   |   +-- task-completion-auditor.md
|   |   L-- test-runner.md
|   +-- hooks/
|   |   +-- hook_utils/
|   |   |   +-- __init__.py
|   |   |   +-- diff_classifier.py
|   |   |   +-- errors_log.py
|   |   |   +-- path_classifier.py
|   |   |   +-- review_marker.py
|   |   |   L-- transcript_parser.py
|   |   +-- __init__.py
|   |   +-- session_start_load_errors.py
|   |   L-- stop_review_chain.py
|   +-- rules/
|   |   +-- 10-core-operating-rules.md
|   |   +-- 20-security-and-secrets.md
|   |   +-- 30-risk-execution-safeguards.md
|   |   +-- 40-testing-and-ci.md
|   |   +-- 50-architecture-boundaries.md
|   |   +-- 60-git-and-change-management.md
|   |   +-- 70-memory-context-and-token-discipline.md
|   |   +-- 80-logging-and-output-standards.md
|   |   +-- 85-response-style.md
|   |   +-- 90-subagents-hooks-and-automation.md
|   |   +-- 95-token-efficiency.md
|   |   L-- README.md
|   +-- worktrees/              (4 aktive Worktrees: agent-a7b606cb, a8530ae9, ac275289, aff57adf)
|   +-- .review_skip_log.jsonl
|   +-- scheduled_tasks.lock
|   +-- settings.json
|   L-- settings.local.json
+-- .cursor/
|   +-- commands/               (01pull.md .. 06costgrid.md)
|   L-- rules/                  (01-backend-overview.md, 02-backend-guidelines.md, rule1.mdc)
+-- .dvc/
|   +-- .gitignore
|   +-- README.md
|   L-- config.example
+-- .github/
|   +-- ISSUE_TEMPLATE/
|   |   L-- review_feedback.md
|   L-- workflows/
|       +-- accounting-ci.yml
|       +-- backend-ci.yml
|       +-- ci.yml
|       +-- daily-diagnostics.yml
|       +-- daily-paper-reconcile.yml
|       +-- disclosures-worker-ci.yml
|       +-- earnings-calendar-refresh.yml
|       +-- evidence-pack-ci.yml
|       +-- fail-drill.yml
|       +-- news-worker-ci.yml
|       +-- nightly-runall.yml
|       +-- nightly-sync.yml
|       +-- ops-evidence-ci.yml
|       +-- paper-trading-ci.yml
|       +-- prewarm-factor-store.yml
|       +-- release-gate-ci.yml
|       +-- repo-health.yml
|       +-- secrets-scan.yml
|       +-- signal-decay-update.yml
|       +-- weekly-drills.yml
|       L-- weekly-research.yml
+-- autonome_weiterarbeit/
|   +-- wichtig/                (3 compass_artifact-Markdown-Dateien)
|   +-- AUDIT_SWEEP_2026-05-12.md
|   L-- (weitere Markdown-Dateien)
+-- configs/
|   +-- app.yaml
|   +-- policy.yaml
|   +-- policy_no_leverage.yaml
|   L-- (weitere YAML-Dateien)
+-- docs/
|   +-- architecture/
|   |   L-- system_map/         (index.html, system_map_data.js, system_map_overrides.yaml, assets/)
|   +-- incidents/
|   |   L-- 2026-04-18_env_exposure.md
|   +-- recherche/
|   |   L-- PROJEKT_IST_STAND.md  (diese Datei)
|   +-- superpowers/
|   |   +-- specs/              (Review-Chain-Design, diverse Spec-Dokumente)
|   |   L-- (weitere Docs)
|   +-- CLAUDE_CODING_ERRORS.md
|   +-- CONTRACTS.md
|   +-- KNOWN_ISSUES_ARCHIVE.md
|   +-- MERGE_GATE_SPRINT13.md
|   +-- OPERATING.md
|   +-- RELEASE_NOTES_SPRINT13.md
|   L-- (weitere Markdown-Dateien)
+-- experiments/
|   L-- (Experimentordner mit batch-Backtest-Configs und Artefakten)
+-- logs/
|   L-- (CLI-Run-Logs, *.log)
+-- output/
|   +-- runs/                   (Paper-Run-Verzeichnisse, datiert)
|   +-- macro_gpr.parquet
|   L-- (weitere Parquet-/CSV-Artefakte)
+-- scripts/
|   +-- architecture/
|   |   +-- diff_system_map.py
|   |   +-- download_vendors.py
|   |   +-- generate_system_map.py
|   |   L-- validate_system_map.py
|   +-- calibration/
|   |   +-- paper_vs_backtest_divergence.py
|   |   L-- (weitere Kalibrations-Scripts)
|   +-- commands/
|   |   +-- __init__.py
|   |   +-- backtest.py
|   |   +-- info.py
|   |   +-- ml.py
|   |   +-- news.py
|   |   +-- ops.py
|   |   +-- paper.py
|   |   +-- reports.py
|   |   L-- run_daily.py
|   +-- comparison/
|   |   L-- paper_trade_v1_v2.py
|   +-- data/                   (Download-Scripts: earnings_calendar.py, gdelt_daily.py, etc.)
|   +-- dev/
|   |   +-- release_sprint13.py
|   |   L-- run_checks.py
|   +-- tools/                  (Analyse-Hilfsskripte)
|   +-- _crisis_alpha_backtest_compare.py  (untracked)
|   +-- _crisis_alpha_pit_verify.py        (untracked)
|   +-- _crisis_alpha_replay.py            (untracked)
|   +-- backtest_news_alpha.py             (untracked)
|   +-- batch_backtest.py
|   +-- cli.py                  (Haupt-Entry-Point)
|   +-- paper_trading_scheduler.py
|   +-- run_api.py
|   +-- run_backtest_strategy.py
|   +-- run_daily.py
|   +-- run_eod_pipeline.py
|   L-- (weitere Scripts)
+-- src/
|   L-- assembled_core/
|       +-- accounting/         (broker_snapshot, currency, ledger, position_sync, reconcile, ...)
|       +-- api/
|       |   +-- routers/        (diagnostics, paper_trading, reports, ...)
|       |   +-- __init__.py
|       |   +-- auth.py
|       |   +-- main.py
|       |   L-- models.py
|       +-- config/             (constants, loader, policy, secrets_loader, ...)
|       +-- data/
|       |   +-- altdata/        (acled, earnings_calendar, gdelt, polymarket, ...)
|       |   +-- macro/          (fred, gdelt_loader, gpr, macro_factors, ...)
|       |   +-- corporate_actions.py
|       |   +-- feature_store.py
|       |   +-- freshness_monitor.py
|       |   +-- prices_ingest.py
|       |   L-- universe.py
|       +-- events/
|       |   +-- news_alpha/     (asset_router, exit_rules, models, pipeline, signal_generator)
|       |   +-- replayer.py
|       |   L-- (weitere Event-Module)
|       +-- execution/
|       |   +-- paper/          (paper-spezifische Execution-Module)
|       |   +-- kill_switch.py
|       |   +-- order_lifecycle.py
|       |   +-- order_management.py
|       |   +-- unified_paper_engine.py
|       |   L-- (weitere Execution-Module)
|       +-- features/
|       |   +-- event_features.py
|       |   +-- mean_reversion_factors.py
|       |   +-- residual_momentum.py
|       |   +-- ta_features.py
|       |   L-- (weitere Feature-Module)
|       +-- intel/
|       |   +-- news_alerts.py
|       |   +-- source_registry.py
|       |   +-- health_monitor.py
|       |   L-- (weitere Intel-Module)
|       +-- ml/
|       |   +-- model_registry.py
|       |   L-- (HMM, meta_model, retraining_scheduler, ...)
|       +-- ops/
|       |   +-- dead_man_switch.py
|       |   +-- dms_daemon.py
|       |   +-- metrics_exporter.py
|       |   +-- paper_ledger.py
|       |   +-- paper_runner.py
|       |   +-- paper_summary.py
|       |   +-- scheduler.py
|       |   L-- (weitere Ops-Module)
|       +-- paper/
|       |   +-- deadzone_rebalance.py
|       |   +-- georisk_gate.py
|       |   +-- intel_context.py
|       |   +-- intel_runner.py
|       |   +-- paper_track.py
|       |   +-- ranking_hysteresis.py
|       |   +-- rebalance_filter.py
|       |   +-- strategy_adapters.py
|       |   L-- __init__.py
|       +-- pipeline/
|       |   +-- _shared_eod.py
|       |   +-- _tc_features.py
|       |   +-- _tc_signals.py
|       |   +-- _tc_sizing.py
|       |   +-- orchestrator.py
|       |   +-- trading_cycle_shared.py
|       |   +-- trading_cycle_v2.py
|       |   L-- (weitere Pipeline-Module)
|       +-- portfolio/          (constraints, optimizer, risk_parity, sizing, ...)
|       +-- qa/
|       |   +-- backtest_engine.py
|       |   +-- dataset_builder.py
|       |   +-- deflated_sharpe.py
|       |   +-- evidence_pack.py
|       |   +-- experiment_tracking.py
|       |   +-- factor_decay_reporter.py
|       |   +-- metrics.py
|       |   +-- point_in_time_checks.py
|       |   L-- (weitere QA-Module)
|       +-- reports/            (risk_report, tca_report, ...)
|       +-- risk/               (risk_controls, risk_state, vol_circuit_breaker, ...)
|       +-- signals/
|       |   +-- base.py
|       |   +-- meta_model.py
|       |   +-- pairs_trading.py
|       |   +-- recession_probability.py
|       |   +-- registry.py
|       |   L-- (weitere Signal-Module)
|       +-- strategies/
|       |   +-- stat_arb/       (Submodul-Verzeichnis)
|       |   +-- base.py
|       |   +-- ema_trend_v0.py
|       |   +-- ic_decay_weights.py
|       |   +-- multifactor_long_short.py
|       |   +-- multifactor_v1.py
|       |   +-- multifactor_v2.py
|       |   +-- multifactor_v2_constants.py
|       |   +-- pairs_trading_v1.py
|       |   +-- pead_strategy.py
|       |   +-- signal_decay_gate.py
|       |   L-- trend_baseline.py
|       +-- utils/              (clock_drift, random_state, reproducibility, retry, ...)
|       +-- logging_config.py
|       L-- __init__.py
+-- tests/
|   +-- hooks/                  (Hook-Tests: test_errors_log.py, test_transcript_parser.py, ...)
|   +-- (679 test_*.py Dateien flach im tests/-Ordner)
|   L-- conftest.py
+-- CLAUDE.md
+-- KNOWN_ISSUES.md
+-- LICENSE
+-- pyproject.toml
+-- README.md
+-- requirements.lock           (89 Zeilen, transitive Freeze)
+-- requirements.txt
+-- ROADMAP_STATE.md
L-- _screener_run.log
```

---

## 2. Branches mit letztem Commit-Datum

| Branch | Letztes Commit |
|--------|---------------|
| `main` | 2026-05-27 |
| `ERWEITERUNG` | 2026-05-12 |
| `feat/edcl` | 2026-05-02 |
| `sprint1/blocker-core-safety` | 2026-04-10 |
| `sprint2/algo-phase1-exec-quality` | 2026-04-12 |
| `worktree-agent-a0a9c2c0` | 2026-03-31 |
| `worktree-agent-a2f080f2` | 2026-03-31 |
| `worktree-agent-a700e54f` | 2026-03-31 |
| `worktree-agent-ae6ea25d` | 2026-03-31 |
| `worktree-agent-a73bbad9` | 2026-04-26 |
| `worktree-agent-a7b606cb13c522e1d` | 2026-05-26 |
| `worktree-agent-a8530ae9d7d5595f6` | 2026-05-26 |
| `worktree-agent-ac275289d3bf5b9ed` | 2026-05-26 |
| `worktree-agent-aff57adf12c4aadd1` | 2026-05-26 |
| `origin/ERWEITERUNG` | 2026-05-12 |
| `origin` (HEAD → main) | 2026-05-26 |
| `origin/main` | 2026-05-26 |

---

## 3. Inhalt der Kerndateien

### README.md (Kurzform)
- Projektname: **Assembled Trading AI — Backend**
- CI-Badges: backend-ci, release-gate-ci, Python 3.11+, ruff, Proprietary License
- Beschreibung: Modulares Trading-Core-System für Daten-Ingest, Signal-Generierung, Backtesting, Portfolio-Simulation
- Quickstart: `pip install -e ".[dev,ml,ml-boost,ml-tune,ml-explain,ml-nlp,ml-hmm,ml-online,scipy,perf,intermarket,historical-data,system_check,all]"`
- Dependency-Hierarchie: `pyproject.toml` (Ranges, lokal) → `requirements.txt` (Pins, CI) → `requirements.lock` (Freeze)
- Primary Merge-Gate: `py -3 scripts/dev/release_sprint13.py`
- CI Blocking: evidence-pack-ci (Windows), accounting-ci (Windows), ops-evidence-ci (Windows)

### pyproject.toml (Auszug)
```toml
[project]
name = "assembled-trading-core"
version = "0.0.1"
requires-python = ">=3.10"
license = {text = "Proprietary"}

# Runtime dependencies (Ranges):
pandas>=2.0.0, numpy>=1.24.0, pyarrow>=10.0.0, fastparquet>=2023.1.0,
yfinance>=0.2.40, alpaca-py>=0.30.0, polygon-api-client>=1.12.0,
fredapi>=0.5.0, edgartools>=2.0.0, fastapi>=0.100.0, uvicorn>=0.23.0,
pydantic>=2.0.0, arch>=6.0.0, feedparser>=6.0.0, ...

[project.scripts]
assembled-cli = "scripts.cli:main"
assembled-run-backtest = "scripts.run_backtest_strategy:main"
assembled-run-daily = "scripts.run_eod_pipeline:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q --strict-markers -m 'not external' --tb=short"
markers = [fast, integration, regression, smoke, slow, unit, external,
           characterization, unwired_code, advanced, requires_scipy,
           requires_fastapi, requires_sklearn, chaos, property, stress,
           phase_zero, phase_speed, phase_realism, phase_depth]
```

### requirements.txt (gepinnte CI-Versionen, Auszug)
```
pandas==2.2.3
numpy==2.2.6
pyarrow==21.0.0
fastparquet==2024.11.0
yfinance==0.2.54
alpaca-py==0.38.0
polygon-api-client==1.14.4
fredapi==0.5.2
edgartools==2.26.1
fastapi==0.122.0
pydantic==2.12.5
pydantic-settings==2.12.0
hypothesis==6.152.4
joblib==1.5.2
anthropic==0.96.0
PyYAML==6.0.3
pandera==0.31.1
pandas-market-calendars==4.6.1
arch==8.0.0
statsmodels==0.14.6
```

### requirements.lock
Existiert. 89 Zeilen. Vollständiger transitive Freeze (via `pip freeze | grep -v "^-e "`).

---

## 4. Registrierte Strategien

Das System verwendet zwei Registry-Mechanismen:

**A. StrategyRegistry (src/assembled_core/strategies/base.py)**
Klassen-/Funktions-basiertes Plugin-System mit `@StrategyRegistry.register(name)`.

| Strategie-Name (Laufzeit) | Datei | Anmerkung |
|--------------------------|-------|-----------|
| `trend_baseline` | `src/assembled_core/strategies/trend_baseline.py` | MA-Crossover-Trend-Follower. Primäre Paper-Strategie seit §9.6(b) Phase-2. OOS 2025-01..05: CAGR ~43%, Sharpe ~1.44. |
| `multifactor_v2` | `src/assembled_core/strategies/multifactor_v2.py` | 34-Faktor-Strategie, VERSION="multifactor_v2.4.0", Regime-Conditional Weights, Meta-Model. Aktuell schlechter als trend_baseline. |
| `multifactor_v1` | `src/assembled_core/strategies/multifactor_v1.py` | Ältere Multi-Faktor-Implementierung, von v2 genutzt. |
| `ema_trend` | `src/assembled_core/strategies/ema_trend_v0.py` | EMA-Trend v0, `name = "ema_trend"`. In PRICE_ONLY_SHADOW_WHITELIST. |
| `pairs_trading_v1` | `src/assembled_core/strategies/pairs_trading_v1.py` | `name = "pairs_trading_v1"`. |
| `multifactor_long_short` | `src/assembled_core/strategies/multifactor_long_short.py` | Long-Short-Variante, in `paper_track.py` Literal-Liste. |
| `pead_strategy` | `src/assembled_core/strategies/pead_strategy.py` | PEAD (Post-Earnings-Announcement-Drift). |
| `event_insider_shipping` | `src/assembled_core/qa/dataset_builder.py` | Nur in dataset_builder referenziert, kein eigenes strategies/-Modul. |
| stat_arb | `src/assembled_core/strategies/stat_arb/` | Unterverzeichnis, genaue Klassen nicht aufgelöst. |

**B. SignalRegistry (src/assembled_core/signals/registry.py)**
Plugin-System via Python Entry-Points, Gruppe `ata.signals`. Aktuell keine externen Signal-Packages in `pyproject.toml` registriert → `load_all()` lädt 0 Signals aus Entry-Points.

---

## 5. CLI-Befehle (scripts/cli.py)

```
python scripts/cli.py <subcommand> [args]
```

| Subcommand | Beschreibung |
|-----------|-------------|
| `info` | Projektinformationen anzeigen |
| `run_daily` | Tägliche EOD-Pipeline ausführen (execute, backtest, portfolio, QA) |
| `build_ml_dataset` | ML-fähigen Datensatz aus Backtest-Ergebnissen erstellen |
| `train_meta_model` | Meta-Model für Setup-Erfolgsvorhersage trainieren |
| `factor_report` | Faktor-Analyse-Report für Universe und Datumsbereich |
| `analyze_factors` | Umfassende Faktoranalyse (IC + Portfolio-Evaluation) |
| `ml_validate_factors` | ML-Validierung auf Faktor-Panels |
| `ml_model_zoo` | Mehrere ML-Modelle auf Faktor-Panels vergleichen |
| `run_backtest` | Strategie-Backtest ausführen |
| `batch_backtest` | Batch-Backtests aus Config-Datei ausführen (Blessed Entry Point) |
| `batch_run` | Batch-Backtests mit Resume-Support (MVP) |
| `leaderboard` | Beste Runs aus Batch-Backtest-Ergebnissen ranken und anzeigen |
| `run_news_pipeline` | NEWS v1 Pipeline (fetch → normalize → dedupe → health → emit) |
| `run_disclosures_pipeline` | Disclosures Pipeline (House PTR / SEC EDGAR stubs) |
| `run_news` | Alias für `run_news_pipeline` |
| `risk_report` | Risiko-Report aus Backtest-Ergebnissen generieren |
| `tca_report` | Transaction Cost Analysis (TCA) Report generieren |
| `run_paper_daily` | Paper/Shadow-Trading-Zyklus einmal ausführen + KPI-Artefakte schreiben |
| `run_paper_range` | Paper/Shadow-Zyklus über Datumsbereich ausführen + Summary schreiben |
| `run_paper_experiment` | A/B-Paper-Experiment mit Policy-Overrides ausführen |
| `compare_paper_experiments` | Zwei Paper-Experiment-Summaries vergleichen (A vs B) |
| `summarize_intel_activity` | Intel-Aktivitäts-Summary für ein Experiment erstellen |
| `inspect_eod_range` | EOD-Preis-Coverage prüfen, empfohlenen Experiment-Start/End ausgeben |
| `check_health` | Backend Health-Status prüfen (read-only, Operations Monitoring) |
| `paper_track` | Paper Track für einzelnen Tag oder Datumsbereich ausführen |
| `walk_forward` | Walk-Forward-Analyse (OOS-Validierung, Research-Tool) |
| `run_phase4_tests` | Phase-4-Test-Suite ausführen |

---

## 6. pytest --collect-only (Zusammenfassung)

Ausgeführt: `python -m pytest --collect-only -q` (lokal, 2026-05-27)

- **Test-Dateien:** 679 (`tests/test_*.py` + `tests/hooks/test_*.py`)
- **Gesammelte Test-Items:** Gesamtzahl nicht ablesbar wegen PowerShell-Output-Truncation bei >1.8 MB. Letzter bekannter Stand (2026-05-21): ~5417 Tests ohne Collection-Errors.
- **Collection-Errors:** 0 (kein ERROR in stderr erkennbar)
- **Hinweis:** pytest-Marker `not external` ist aktiv (`addopts`). Tests mit `@pytest.mark.external` werden standardmäßig übersprungen.

Letzte Zeilen des `--collect-only`-Outputs (Auswahl):
```
tests/test_wave18_helpers.py: 31
tests/test_wave19_helpers.py: 19
tests/test_wave20_wiring.py: 14
tests/test_twap_vwap_annotation.py: 11
tests/test_algo_type_wiring.py: 3
```

---

## 7. PaperPilot-Status

### Windows Task Scheduler

| Task-Name | Nächste Laufzeit | Status |
|-----------|-----------------|--------|
| `AssembledTradingAI-PaperPilot` | 27.05.2026 21:30:00 | Bereit |
| `AssembledTradingAI_PaperEngine` | Nicht zutreffend | Bereit |

- **Frequenz:** Täglich 21:30 Uhr (Windows Task Scheduler, lokal, nur interaktiv)
- **Letzter erfolgreicher Run:** Mai 2026 (exakter Stand: laut früheren Session-Analysen lief der Run am 23.05.2026 nach dem Rate-Limit-Fix erfolgreich)

### Zugehörige Dateien

| Datei | Rolle |
|-------|-------|
| `src/assembled_core/paper/paper_track.py` | Kern-Orchestrator Paper Track (Stateful, PIT-safe) |
| `src/assembled_core/ops/paper_runner.py` | Paper Runner für EOD-Pipeline |
| `src/assembled_core/ops/paper_summary.py` | Summary-Generierung nach Paper-Run |
| `src/assembled_core/ops/paper_ledger.py` | Ledger-Persistenz für Paper-Positionen |
| `src/assembled_core/execution/unified_paper_engine.py` | Unified Paper Engine (Fill-Simulation) |
| `src/assembled_core/paper/strategy_adapters.py` | Strategie-Adapter für Paper Track |
| `src/assembled_core/paper/paper_track.py::PaperTrackConfig` | Config: `strategy_type: Literal["trend_baseline", "multifactor_long_short"]` |
| `scripts/commands/paper.py` | CLI-Bindings (run_paper_daily, run_paper_range, ...) |
| `scripts/paper_trading_scheduler.py` | Scheduler-Script |
| `scripts/commands/ops.py::paper_track_subcommand` | `paper_track`-CLI-Handler |

### `paper_track` aktuell aktiv?

`paper_track` ist via Task Scheduler `AssembledTradingAI-PaperPilot` täglich um 21:30 aktiv.
Aktueller Alpaca-Paper-Broker-Stand (letzte bekannte Abfrage): Equity ~$92.141 (Stand 2026-05-20).
Letzte Rate-Limit-Fixes (yfinance 429 → YFinanceRateLimitError + Alpaca Fallback): Commit `aa032441` (2026-05-23).

---

## 8. §9-Pre-PaperPilot-Checkliste

**Nicht gefunden.**

Es existiert keine eigenständige Datei mit dem Namen "§9-Pre-PaperPilot-Checkliste" oder ähnlichem im Repository. `KNOWN_ISSUES.md` enthält einen Abschnitt §9 (bzgl. bekannter Probleme), aber keine dedizierte Pre-PaperPilot-Checkliste. Geprüfte Dateien: alle `.md`-Dateien im Repo-Root und in `docs/`.

---

## 9. Offene TODO/FIXME/XXX-Marker in src/

Ergebnis von `grep -rn "TODO|FIXME|XXX" src/ --include="*.py"`:

**3 Treffer** (Stand 2026-05-27):

| Datei | Zeile | Inhalt |
|-------|-------|--------|
| `src/assembled_core/config/secrets_loader.py` | 17 | `# export ATA_STAGING_ALPACA_API_KEY=xxx` — Kommentar-Beispiel für Env-Var, kein echter TODO |
| `src/assembled_core/pipeline/orchestrator.py` | 1431 | `# TODO: wire to post-signal-computation step when factor panel is available.` — Offener Wiring-Punkt für Factor-Decay-Monitor (bekannt aus 9467b0ae) |
| `src/assembled_core/ops/halt_cache.py` | 1 (Docstring) | `"""60s-refresh halt-symbol cache (audit C5-091, closes _tc_sizing TODO line 1715).` — Referenz auf erledigten TODO, kein neuer offener Punkt |

**Fazit:** 1 echter offener TODO (`orchestrator.py:1431` — factor decay monitor Wiring, bewusst zurückgestellt bis post-signal-computation Schritt verfügbar ist, dokumentiert in Memory als DMS/Factor-Decay-Session).
