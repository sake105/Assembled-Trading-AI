# Archivierter Waisen-Code — Tranche 1 (2026-08-17, Audit-Plan 6.4)

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

TRANCHE 2 (bewusst NICHT hier — dokumentiert offen): 16 weitere referenzfreie
Module haengen an Sammel-Testdateien mit HARTEN Imports
(test_free_stack_modules, test_competitive_analysis_impl,
test_session_2026_05_07_new_items, test_ml_wave2*, test_wave19_helpers) —
Archivierung erfordert Testdatei-Chirurgie. 9 weitere Kandidaten hatten
doch Referenzen (teils Waisen-Ketten wie crisis_alpha_worker <-
health_monitor) — Ketten-Analyse noetig. Details: Dry-Run-Protokoll der
Session 2026-08-17.
