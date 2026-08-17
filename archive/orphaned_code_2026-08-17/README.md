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

Offen aus dem urspruenglichen Audit: 9 weitere Kandidaten hatten
doch Referenzen (teils Waisen-Ketten wie crisis_alpha_worker <-
health_monitor) — Ketten-Analyse noetig, NICHT Teil von Tranche 2.
Details: Dry-Run-Protokoll der Session 2026-08-17.
