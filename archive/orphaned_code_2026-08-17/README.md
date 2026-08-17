# Archivierter Waisen-Code — Tranche 1+2 (2026-08-17, Audit-Plan 6.4)

Quelle: Nutzungsaudit `docs/DATEN_UND_NUTZUNGSAUDIT.md` §3, Kategorie (a)
„totes Erbe". Aufnahme-Kriterium fuer Tranche 1: NULL Referenzen **in *.py/*.yaml unter
src/, scripts/, configs/, .github/** UND keine Bindung an Sammel-Testdateien
(dedizierte Tests wurden MIT-archiviert, s. tests/). KORREKTUR 2026-08-17
(Stage-2-Review, E-167): der urspruengliche Scope deckte research/**.ipynb und
Doku-Dateien NICHT ab — ein Notebook-Import von breakout_signal
(research/trend/trend_baseline_experiments.ipynb) wurde uebersehen und
nachtraeglich auf den Archivpfad umgebogen; ~40 Doku-Zeiger laufen als
Sammel-Follow-up mit Tranche 2. Archiviert statt geloescht
(Projektkonvention).

| Archiviert | Herkunft | dedizierte Tests |
|---|---|---|
| signals/breakout_signal.py | src/assembled_core/signals/ | keine |
| ml/temporal_fusion_transformer.py | src/assembled_core/ml/ (NotImplementedError-Stub) | keine |
| ml/logic_tensor_network.py | src/assembled_core/ml/ (Stub) | keine |
| ops/mlflow_tracking.py | src/assembled_core/ops/ | keine |
| ops/incident_tracker.py | src/assembled_core/ops/ | tests/test_incident_tracker.py |
| ops/rejection_collector.py | src/assembled_core/ops/ | tests/test_rejection_collector.py |
| strategies/pead_strategy.py | src/assembled_core/strategies/ (kein Dispatch-Name, nicht in __init__) | tests/test_pead_strategy.py |
| sources/worldbank_source.py | src/assembled_core/data/sources/ (nur __init__-Reexport, mit entfernt) | keine |
| streaming__init__.py | src/assembled_core/data/streaming/ (leeres Paket) | keine |

Zusaetzlich entfernt (leere Verzeichnisse, nur __pycache__):
`strategies/stat_arb/`, `data/streaming/`.

## Tranche 2 (2026-08-17, gleiche Session)

Die 16 Module, die an Sammel-Testdateien mit harten Imports hingen (s. o.),
wurden nach Testdatei-Chirurgie archiviert. Referenz-Sweep-Scope je Kandidat
(vor dem Move einzeln verifiziert): repo-weit ueber *.py, *.ipynb, *.md,
*.bat, *.ps1, *.yaml/yml, *.json, *.toml inkl. research/ und docs/;
ausgenommen archive/, __pycache__, .mypy_cache, *.egg-info (generiert),
docs/architecture/system_map (generiert), .git. Fuer newsapi_source lief der
Sweep via git grep (nur getrackte Dateien) + Einzelpruefung der 3 zum
Zeitpunkt untrackten Dateien; fuer die uebrigen 15 als Dateisystem-Grep.
Ergebnis fuer alle 16: NULL Referenzen in aktivem Code (src/, scripts/,
configs/, .github/) — nur Tests + Doku. Notebooks (*.ipynb): 0 Treffer.
Entfernte Testabschnitte tragen Inline-Marker
(`ENTFERNT 2026-08-17: testete <modul> ...`).

| Archiviert | Herkunft | Test-Behandlung |
|---|---|---|
| signals/cross_asset_carry_v2.py | src/assembled_core/signals/ | test_competitive_analysis_impl.py: Klasse TestCrossAssetCarryV2 entfernt |
| signals/analyst_revisions.py | src/assembled_core/signals/ | test_free_stack_modules.py: 2 Tests entfernt |
| signals/lppls_crash.py | src/assembled_core/signals/ | test_competitive_analysis_impl.py: Klasse TestLPPLSCrashDetector entfernt; dedizierte Datei tests/test_signals_lppls_validation.py mit-archiviert (s. tests/) |
| signals/recession_probability.py | src/assembled_core/signals/ | test_free_stack_modules.py: 2 Tests entfernt |
| signals/regime_conditional_ensemble.py | src/assembled_core/signals/ | test_wave19_helpers.py: 4 Tests entfernt |
| signals/sentiment_panel.py | src/assembled_core/signals/ | test_free_stack_modules.py: 1 Test entfernt |
| signals/tail_risk_vvix.py | src/assembled_core/signals/ | test_competitive_analysis_impl.py: Klasse TestVVIXTailRiskSignal entfernt |
| intel/polymarket_loader.py | src/assembled_core/intel/ | test_session_2026_05_07_new_items.py: Klassen TestPolymarketLoaderF821 + TestF821Cleared entfernt, 1 Methode aus TestNetworkTimeoutsComprehensive |
| intel/feedback_loops.py | src/assembled_core/intel/ | test_ml_wave2_batch2.py: Klasse TestFeedbackLoopTracker entfernt |
| intel/structural_cycles.py | src/assembled_core/intel/ | test_ml_wave2.py: Klasse TestStructuralCycles entfernt |
| intel/wild_card_detector.py | src/assembled_core/intel/ | test_ml_wave2.py: Klasse TestWildCardDetector entfernt |
| ops/alert_failover.py | src/assembled_core/ops/ | test_session_2026_05_07_new_items.py: Klasse TestAlertFailover entfernt |
| ops/calibration_tracker.py | src/assembled_core/ops/ | test_session_2026_05_07_new_items.py: Klasse TestCalibrationTracker + 4 Einzelmethoden aus Mixed-Klassen entfernt |
| ops/shap_explainer.py | src/assembled_core/ops/ | test_free_stack_modules.py: 2 Tests; test_session_2026_05_07_new_items.py: Klassen TestSHAPExplainerModule + TestFeatureImportanceMonitoringB entfernt. (TestFeatureImportanceMonitoring referenziert assembled_core.ml.shap_explainer — nie existiert, skippt unveraendert; unberuehrt gelassen) |
| ops/slippage_collector.py | src/assembled_core/ops/ | test_session_2026_05_07_new_items.py: Klasse TestSlippageCollector + 1 Methode aus TestSpreadCaptureTracking entfernt; dedizierte Datei tests/test_slippage_collector.py mit-archiviert (s. tests/) |
| sources/newsapi_source.py | src/assembled_core/data/sources/ (Reexport `fetch_news_headlines` aus __init__ mit entfernt) | test_session_2026_05_07_new_items.py: Klasse TestNewsAPIRateLimit entfernt, 1 Methode aus TestNetworkTimeoutsComprehensive, Dateiliste in TestOsPathPathlibMix getrimmt |

## Nachtrag 6.5 (2026-08-17): daily_scheduler-Trio

Audit-Plan 6.5, Empfehlung "streichen" umgesetzt — fertige Scheduler-Kette
ohne Launcher (keine Task-Registrierung in .bat/.ps1/.yml, Doppelstruktur
zum Pilot-Bat):

| Archiviert | Herkunft | Test-/Zeiger-Behandlung |
|---|---|---|
| ops/daily_scheduler.py | src/assembled_core/ops/ (Reexporte DailyScheduler/WorkerResult/run_daily_cycle/build_cycle_summary aus __init__ mit entfernt — 0 Paket-Importe repo-weit) | test_ops_correctness.py: A12-Block (3 Tests + _RecordingFire/_patch_fire) und A35-Test entfernt; lebender reconcile-Alert-Pfad (accounting/reconciliation.py:205) bleibt durch test_batchB2_accounting_failclosed.py abgedeckt |
| scripts/run_daily_scheduler.py | scripts/ | SCRIPTS_INDEX.md-Zeile auf ARCHIVIERT gestellt |
| tests/test_daily_scheduler.py | tests/ | dedizierte Datei mit-archiviert |

## Tranche 3 (2026-08-17): Ketten-Kandidaten aufgeloest

Frische Referenz-Analyse (Explore-Agent, very thorough) der Restliste aus
dem Audit ("(a) totes Erbe", DATEN_UND_NUTZUNGSAUDIT.md:175-183) +
unabhaengiger Kreuzcheck im Hauptkontext (alle Restreferenzen = reine
Kommentare, per Stichprobe verifiziert):

| Archiviert | Herkunft | Test-/Zeiger-Behandlung |
|---|---|---|
| portfolio/quantum_portfolio.py | src/assembled_core/portfolio/ | 0 Referenzen — keine Chirurgie |
| research/qc_client.py | research/mandat/ (QuantConnect REST, 0 Importer) | keine |
| intel/ic_loop.py | src/assembled_core/intel/ | dedizierte Datei tests/test_ic_loop.py mit-archiviert (s. tests/) |
| intel/pit_store.py | src/assembled_core/intel/ | test_atomic_io.py: Methode test_pit_store_uses_atomic_write entfernt |
| portfolio/dro_portfolio.py | src/assembled_core/portfolio/ | dedizierte Datei tests/test_portfolio_dro.py mit-archiviert; test_batch14_portfolio_qa_honesty.py: 3 dro-Funktionen entfernt (optimizers-Tests bleiben) |
| sources/alphavantage_source.py | src/assembled_core/data/sources/ (Reexport aus __init__ entfernt) | test_session_2026_05_07_new_items.py: test_alphavantage_source_has_timeout entfernt |
| sources/edgar_source.py | src/assembled_core/data/sources/ (Reexport aus __init__ entfernt; NICHT der lebende Form-4-Ingest data/edgar_form4_ingest.py!) | test_session_2026_05_07_new_items.py: test_edgar_source_has_timeout + Klasse TestEDGARRateLimiting entfernt, Listeneintrag in test_new_modules_prefer_pathlib getrimmt; TestEDGARThrottling bewusst belassen (rglob("*edgar*") trifft den lebenden Ingest); diagnostics.py-Eintrag auf status=archived |

NICHT archivierbar (lebende Referenz): intel/crisis_alpha_worker.py —
scripts/run_intel_cycle.py:39,635 importiert und ruft es (run_intel_cycle
selbst lebendig). Stale-Docstring im Modul (behauptet
run_crisis_alpha_worker.py als Importer — falsch) = Follow-up.

DENY-BLOCKIERT (execution/risk/pipeline — nur benannt, nichts empfohlen):
ml/gnn_signal.py (Importer pipeline/_tc_signals.py:593 ist live),
risk/wash_sale_guard.py, risk/barra_risk_model.py.

Damit ist Audit-Plan 6.4 ABGESCHLOSSEN: 32 Module ueber 3 Tranchen
(9+16+7), dazu aus 6.5 das daily_scheduler-Modul + sein Script-Runner,
sowie 8 dedizierte Testdateien (3 Tranche 1, 2 Tranche 2, 2 Tranche 3,
1 aus 6.5) = 42 archivierte .py gesamt; Rest ist lebendig oder deny-gated.
