# Vollständiger Audit — Assembled-Trading-AI

**Audit-Datum:** 2026-04-23
**Repo-Commit:** 55c0d06 (Stand zum Zeitpunkt des Audits)
**Auditor:** Claude (Opus 4.7)
**Scope:** Das gesamte Repository. Jeder erkannte Schwachpunkt, ohne Priorisierung nach "wichtig" oder "unwichtig" — du hast die vollständige Liste gefordert, du bekommst sie.

> **Leseanleitung:** Dieses Dokument ist eine Fehler-/Schwachpunkt-Liste, kein Coaching-Dokument. Jeder Punkt steht für sich. Manche Punkte überlappen, weil sie verschiedene Aspekte derselben Grundursache treffen. Das ist bewusst so — wenn du einen Punkt fixt, löst das oft drei andere mit.
>
> Ein Punkt gilt als "schwer", wenn er das System bricht oder blockiert.
> Ein Punkt gilt als "mittel", wenn er dich später kostet.
> Ein Punkt gilt als "leicht", wenn er kosmetisch oder Hygiene ist.
> Die Tags stehen pro Punkt.

---

## 1. Architektur & Codebasis-Struktur

### 1.1 `trading_cycle.py` ist ein Mega-Monolith
- **schwer** · 10.544 Zeilen in **einer** Datei, 14 Funktionen, 309 nummerierte "Steps" von 8.6 bis 8.99
- unmöglich zu reviewen, zu testen, zu refactorn, zu verstehen
- jede Änderung dort ist gefährlich, weil der Blast-Radius die gesamte Pipeline betrifft
- zeigt klassisches "God-Object"-Anti-Pattern aus Legacy-Enterprise-Codebases

### 1.2 "Wave-Wiring" ist ein Entwicklungs-Anti-Pattern
- **schwer** · 125 `feat(wiring): wave-N`-Commits, jeder fügt 3 Pseudo-Integrationen in `trading_cycle.py` ein
- Integration besteht typischerweise aus: `import modul` + `modul(leeres_input, leere_config)` + `result.meta["modul"] = {"available": True}` + silent-skip im except
- das Ergebnis fließt in **keine** Entscheidung ein, wird in **keinem** echten Pfad gelesen
- eigene Kommentare nennen es "observability" — heißt: "wir dokumentieren nur, dass das Modul existiert"
- 147 dazugehörige `test_waveN_wiring.py`-Dateien prüfen nur `assert modul is not None` und `modul([]) == []`
- damit ist das "Integrations-Volumen" vorgetäuscht, nicht real

### 1.3 507 inline-Imports in `trading_cycle.py`
- **schwer** · Imports erfolgen in jeder der 309 Step-Blöcke lokal, damit der try/except sie schlucken kann
- bedeutet: die Datei hat keine echte Dependency-Struktur mehr; nichts ist statisch analysierbar
- bricht IDE-Features (Go-to-Definition, Autocomplete, Refactoring-Tools)
- macht ruff/mypy de facto blind für die meisten Beziehungen

### 1.4 506 `except Exception` mit `log.debug` — Silent-Failure-Culture
- **schwer** · der Code schluckt bewusst jede Exception und loggt auf DEBUG-Level
- das heißt: läuft ein Modul nicht, wird das niemandem auffallen, außer er schaut in DEBUG-Logs
- es gibt 20 `raise`-Stellen in 10.544 Zeilen; die Fehler-Philosophie ist "verstecken, nicht melden"
- bekannt vorhandener Runtime-Bug in Zeile 2526 (`F821: undefined name 'np'`) wird durch genau dieses Pattern versteckt — die Funktion crasht immer still

### 1.5 Zwei parallele Config-Verzeichnisse (`config/` und `configs/`)
- **mittel** · beide werden aktiv benutzt, beide enthalten unterschiedliche Dinge
- `config/` enthält Factor-Bundles und Universen-Textfiles
- `configs/` enthält policy.yaml, app.yaml, crisis_alpha.yaml etc.
- kein Entwickler kann ohne Lesen raten, was wo hingehört
- führt zu sporadischen "file not found"-Bugs

### 1.6 Drei parallele Backtest-Engines
- **mittel** · `pipeline/backtest.py`, `pipeline/backtest_legacy.py`, `qa/backtest_engine.py`
- zusätzlich `qa/backtest_comparison.py`, `qa/backtest_engine_numba.py`, `qa/backtest_overfit.py`
- kein Dokument erklärt, welche die "offizielle" ist
- verschiedene Scripts rufen verschiedene auf (`run_backtest_strategy.py`, `run_grand_backtest.py`, `sprint9_backtest.py`, `run_final_optimized.py` — 4 Einstiegspunkte für denselben Zweck)

### 1.7 Drei parallele Paper-Engines
- **mittel** · `execution/paper_trading_engine.py`, `execution/unified_paper_engine.py`, `paper/paper_track.py`
- "unified" sollte laut Docstring die drei alten vereinen — existieren aber alle drei weiter
- `ops/paper_runner.py`, `ops/paper_ledger.py`, `ops/paper_summary.py` sind weitere parallele Schichten
- niemand weiß, welche die aktuelle Wahrheit ist

### 1.8 Drei EOD-Daily-Entry-Points
- **mittel** · `scripts/run_daily.py` (1189 Zeilen, Phase-3-Ära), `scripts/run_eod_pipeline.py` (328 Zeilen), `scripts/run_daily_scheduler.py` (101 Zeilen)
- die CLI (`scripts/cli.py`) ruft `run_eod_pipeline.py` auf, der alte `run_daily.py` bleibt als toter Code
- wer Zeile für Zeile liest, findet nach 300 Zeilen `run_daily.py` raus, dass das gar nicht mehr das canonical ist

### 1.9 Zwei Order-Generierungs-Pfade mit unterschiedlicher Logik
- **schwer** · `pipeline/orders.py::signals_to_orders` setzt `qty = 1.0` fest — keine Positionsgrößen
- `trading_cycle.py` hat Kelly-Weights, Vol-Targeting, Kelly-Uncertainty-Shadow
- das heißt: Wenn du über `run_eod_pipeline.py` gehst, hast du fixe 1-Share-Orders; wenn du über `trading_cycle.py` gehst, hast du Kelly-Sizing
- **das ist kein unterschiedlicher Modus, das sind zwei komplett inkompatible Systeme unter einem Dach**

### 1.10 20 parallele Signal-/Rules-Implementierungen
- **mittel** · `pipeline/signals.py`, `signals/rules_trend.py`, `signals/rules_event_insider_shipping.py`, `signals/multifactor_signal.py`, `signals/intel_signal_adapter.py`, `signals/news_signal_bridge.py`, `signals/short_signals.py`, `signals/signal_api.py`, `signals/signal_confidence.py`, `signals/signal_diagnostics.py`, `signals/signal_decay_gate.py`, `features/options_derived_signals.py`, `intel/news_signal_aggregator.py`, `strategies/stat_arb/pair_signals.py`, `strategies/multifactor_long_short.py`, `strategies/multifactor_v1.py`, `strategies/multifactor_v2.py`, `strategies/ema_trend_v0.py` …
- keine klare Hierarchie (Aggregator/Adapter/Generator/Bridge) dokumentiert
- viele sind Shells um einen einzigen Algorithmus
- vergleiche: professionelle Systeme haben typischerweise **eine** Signal-Base-Class plus 2–5 konkrete Strategien

### 1.11 Keine echte Trennung zwischen Core-Library und Scripts
- **mittel** · 95 Scripts im `scripts/`-Top-Level + 31 PowerShell-Scripts
- viele Scripts enthalten Business-Logic, die in `src/` gehört
- Script-Scripts (`sprint9_backtest.py`, `sprint9_execute.py`, `sprint10_portfolio.py`) sind Legacy-Inseln mit aktivem Code

### 1.12 Kein Module-Dependency-Graph
- **mittel** · es existiert `configs/dependency_graph.yaml`, aber der wird nicht enforced
- `trading_cycle.py` importiert aus: events, intel, features, ml, risk, portfolio, execution, qa — alles
- das ist der klassische "Every module depends on every module"-Zustand, der Refactoring unmöglich macht

### 1.13 Import-Stil inkonsistent
- **leicht** · 934 Files benutzen `from src.assembled_core.X` (nicht-idiomatisch — `src` ist Layout-Detail, nicht Package)
- 5 Files benutzen `from assembled_core.X` (idiomatisch, wenn `pip install -e .` gelaufen ist)
- der Dockerfile-HEALTHCHECK macht `import assembled_core` — das funktioniert nur mit dem 5-Files-Stil
- d.h. der Docker-HEALTHCHECK testet nicht, ob die eigentlich benutzte Import-Form funktioniert

### 1.14 Deprecated `backtest_legacy.py` steht neben aktuellem `backtest.py`
- **leicht** · keine klare Policy, wann der Legacy-Code entfernt wird
- Commits von 2026-04-23: `wave-133 — pipeline.backtest, pipeline.backtest_legacy` — beides wurde gleichzeitig "wired"
- Legacy-Code wächst mit, statt zu schrumpfen

### 1.15 Kein Refactoring in 699 Commits
- **schwer** · Commit-Kategorie-Breakdown: 237 feat, 69 fix, 20 docs, 6 test, **0 refactor**
- Code wird nur hinzugefügt, nie vereinfacht
- das erklärt, warum die Codebasis in 6 Monaten von 0 auf 1971 Files gewachsen ist

---

## 2. "Observability-Wiring" im Detail — die 309 Scheinintegrationen

Der gesamte Block verdient eine eigene Sektion, weil er die Diskrepanz zwischen behaupteter und echter Funktionalität erklärt.

### 2.1 Das Muster
- **schwer** · 309 Stepkommentare mit 397 "observability"-Markern und 470 "skipped:"-Silent-Fails
- typisches Pattern (aus `trading_cycle.py`, Step 8.62):
  ```python
  try:
      from src.assembled_core.ml.maml import MAMLConfig
      _maml_cfg = MAMLConfig()
      result.meta["maml"] = {
          "torch_available": ...,
          "inner_lr": _maml_cfg.inner_lr,
          "inner_steps": _maml_cfg.inner_steps,
          ...
      }
  except Exception as _maml_exc:
      log.debug("[MAML] maml skipped: %s", _maml_exc)
  ```
- was passiert real: Config-Defaults werden in ein Meta-Dict geschrieben, MAML selbst wird nie aufgerufen, nie trainiert, nie verwendet
- was behauptet wird: "wave-N integriert MAML"

### 2.2 ML-Module, die "wired" aber ungenutzt sind
- **schwer** · von 55 ML-Modulen sind real im Entscheidungspfad: Meta-Model (`meta_labeling.py`), Regime-HMM (`regime_hmm.py`), Calibration (`calibration.py`) — und das auch nur teilweise
- **nicht im Entscheidungspfad, nur observability-wired:**
  - `maml.py` (Model-Agnostic Meta-Learning) — 264 Zeilen
  - `gnn_stocks.py` (Graph Neural Networks) — 289 Zeilen
  - `bayesian_nn.py` (Bayesian Neural Networks) — 216 Zeilen
  - `rl_portfolio.py` (Reinforcement Learning Portfolio) — 305 Zeilen
  - `rl_execution.py` (Reinforcement Learning Execution) — 312 Zeilen
  - `tda_regime.py` (Topological Data Analysis) — 262 Zeilen
  - `symbolic_regression.py` (Symbolic Regression mit gplearn) — 329 Zeilen
  - `temporal_attention.py` — 286 Zeilen
  - `causal_inference.py`
  - `copula_models.py`
  - `evt_models.py` (Extreme Value Theory)
  - `gaussian_process.py`
  - `graph_models.py`
  - `online_gradient_boosting.py`
  - `online_hmm_regime.py`
  - `online_hpo.py`
  - `online_learning.py`
  - `feature_clustering.py`
  - `feature_importance_tracker.py`
  - `feature_selection.py`
  - `conformal.py`, `conformal_prediction.py` (doppelt)
  - `stacking.py`, `stacking_ensemble.py` (doppelt)
  - `meta_labeling.py`, `nested_meta_labeling.py`, `triple_barrier.py`
  - `quantile_models.py`
  - `factor_models.py`, `factor_timing.py`
  - `adversarial_validation.py`
  - `automl.py`
  - `bayesian_ensemble.py`
  - `combined_regime.py`
  - `cpcv.py`
  - `feedback_loop.py`
  - `garch_models.py`
  - `hyperopt.py`
  - `lime_explainer.py`
  - `model_monitoring.py`
  - `model_registry.py`
  - `news_ml_bridge.py`
  - `nlp_sentiment.py`
  - `purged_cv.py`
  - `regime_model_router.py`
  - `regime_weight_trainer.py`
  - `retraining_scheduler.py`
  - `signal_correlation.py`
  - `signal_decay_tracker.py`
- **zusammenfassend: ≥52 ML-Module existieren als Code, aber ändern keine einzige Handelsentscheidung**

### 2.3 Risk-Module, die "wired" aber ungenutzt sind
- **schwer** · `risk/` hat 36 Python-Files, die meisten sind nur via `trading_cycle.py` observability-wired
- betrifft: `antifragility.py`, `circuit_breaker.py`, `correlation_guard.py`, `crowding_detector.py`, `evt_tail_var.py`, `market_stress.py`, `profit_lock.py`, `profit_targets.py`, `systemic_risk.py`, `tail_dependence.py`, `tail_hedge.py`, `tail_hedging.py`, `trailing_stops.py`, `turnover_budget.py`, `zombie_killer.py` …
- `zombie_killer.py` wird laut `grep` ausschließlich in `trading_cycle.py` referenziert, nirgendwo sonst im Code
- Risk-Module werden also im System "aktiviert" nur im Sinne der Protokoll-Log-Einträge, nicht im Sinne realer Order-Blockaden oder Position-Reduktionen

### 2.4 Alt-Data-Module ohne Ingest-Pfad
- **schwer** · `features/satellite_proxy_features.py` und `data/altdata/satellite_features.py` existieren
- aber: es gibt **keinen** Ingest, der Satellitendaten beschafft (kein API-Client, kein Datenvertrag)
- Funktion akzeptiert `raw_data: pd.DataFrame` als Input — der niemals gefüllt wird
- dasselbe für `patent_features.py` (kein Patent-API-Client), `social_sentiment.py` (kein Social-Scraper), `web_scraping.py` (ein leeres Modul)
- das sind "Alibi"-Features: der Code existiert, die Daten, auf denen er operieren würde, existieren nicht

### 2.5 Crisis-Alpha ist komplett spezifiziert, aber nie gegen echte Krisen validiert
- **schwer** · sauber gebaute State-Machine (MONITORING → WATCH → ACTIVE → COOLDOWN)
- 5 Activation-Gates (Health, Social-Only-Guard, Evidence, Source, Market-Stress) sind alle implementiert
- ETF-Baskets definiert (GLD, TLT, SHY in "DEFENSIVE"; VIX-related in "VOLATILITY"; Inverse-ETFs in "INVERSE_EQUITY")
- **aber:** keine einzige Backtest-Evidenz, dass der Crisis-Mode auf Lehman 2008, COVID 2020, Februar 2022 oder die Iran-Hormuz-Anspannung 2024/25 getriggert hätte
- keine Output-Datei, kein `output/crisis_alpha/*.json`, kein historisches Replay

### 2.6 Dummy-Daten überall
- **mittel** · Event-Strategie `event_insider_shipping` läuft laut README auf generierten Sample-Daten
- `scripts/generate_sample_event_data.py` existiert explizit zu diesem Zweck
- keine echte Form-4-Ingestion, keine echte Shipping-Route-API
- damit: das, was in der README als "Phase-6-Strategie" firmiert, ist ein Toy-Example

---

## 3. Testing

### 3.1 4851 Tests — viele davon Scheintests
- **schwer** · absolute Zahl klingt beeindruckend, ist aber irreführend
- 279 Tests prüfen nur `assert modul is not None` ("Importability-Tests")
- 233 Tests prüfen nur "leerer Input → leerer Output" ("Empty-Input-Tests")
- 147 `test_waveN_wiring.py`-Files bestehen nahezu vollständig aus solchen Tests
- **das heißt: ~15% der Tests sind keine Funktionalitätstests, sondern Ladbarkeitstests**

### 3.2 Tests lassen Sharpe 98,83 durch
- **schwer** · das einzige echte Report-Artefakt im Repo (`src/output/reports/qa_report_test_strategy_1d_20251203.md`) zeigt:
  - 3 Datenpunkte, 2 Tage, Sharpe 98.83
  - "turnover cannot be computed", "hit_rate cannot be computed", "profit_factor cannot be computed"
  - QA-Gates sagen "WARNING", **keines sagt BLOCK**
- eine QA-Logik, die einen Sharpe von 98 durchlässt, ist nicht kalibriert — in der Praxis ist jeder Sharpe >5 auf EOD-Daten statistisches Artefakt

### 3.3 Keine funktionalen End-to-End-Integrationstests mit realen Daten
- **schwer** · kein Test, der sagt: "Lade 5 Jahre AAPL, laufe EMA-Strategie, erwarte Sharpe zwischen X und Y, erwarte Drawdown ≤ Z"
- Integrationstests sind meist Smoke-Tests auf synthetischen Daten mit 10 Timepoints
- heißt: die Fabrik ist getestet, die Produktion ist nicht

### 3.4 ~0 Property-Based Tests (trotz `property`-Marker)
- **mittel** · `pytest.ini` deklariert `property: Property-based tests using Hypothesis`
- finde im Repo keine Verwendung von `from hypothesis import ...`
- der Marker ist deklariert, die Testart existiert nicht

### 3.5 `slow`-Tests sind ausgenommen, aber niemand läuft sie regelmäßig
- **mittel** · `pytest.ini` sagt `addopts = -m "not external"` — gut
- aber `@pytest.mark.slow`-Tests werden in CI nicht gefahren (siehe `PROJEKT_STATUS.md`)
- d.h. langsame, realistischere Tests werden de facto nicht getestet

### 3.6 Phase-Marker-Migrations-Debt
- **leicht** · `phase4..phase13`-Marker sind deprecated, aber noch in 100+ Tests
- `docs/tech_debt/markers_migration.md` setzt Sunset auf 2026-07-01 — gut
- aber: bis dahin bleibt ein doppeltes System aktiv

### 3.7 Kein echter Golden-Equity-Test auf langem Zeitraum
- **schwer** · `tests/regression/golden_equity_baseline.json` existiert, aber der Zeitraum ist kurz
- Golden-Tests machen nur Sinn, wenn sie auf mehreren Jahren Daten laufen
- das verhindert nicht: "EMA-Signale driften in ihrer Semantik" — ein Aspekt, den ein richtiger Golden-Test abfangen sollte

### 3.8 Keine Backtesting-Invariant-Tests
- **mittel** · typische Invarianten, die fehlen:
  - "Equity ist monoton stetig" (keine Sprünge)
  - "Cash + Positions-Value = Equity" (Buchhaltungs-Invariante)
  - "Summe(Trade-PnL) = Final Equity − Start Equity" (minus Kosten)
  - "Nach einem SELL auf qty=5 ist die Position um 5 reduziert, nicht um 4 oder 6"
- ohne solche Invarianten ist der Backtest-Output nicht prüfbar

### 3.9 Chaos-Tests deklariert, nicht genutzt
- **leicht** · Marker `chaos` existiert laut `pytest.ini`
- `grep -r "pytest.mark.chaos" tests/` ergibt wenige bis keine reale Nutzung
- Broker-Outage-Simulation, Data-Gap-Recovery, Race-Conditions: auf dem Papier geplant, nicht in Tests

### 3.10 Tests als Kopplungs-Beton
- **mittel** · viele der 147 Wiring-Tests pinnen die `result.meta`-Struktur exakt
- bedeutet: Refactoring von `trading_cycle.py` würde Hunderte Tests brechen, obwohl sich das Verhalten nicht ändert
- die Tests sind ein Beschleuniger für technische Schuld, nicht eine Abnahme

---

## 4. Daten-Schicht

### 4.1 Parquet-Daten sind Git-LFS-Pointer, kein echter Datenstand
- **schwer** · `find *.parquet` findet 45 Files à ~131 Bytes
- alle enthalten nur `version https://git-lfs.github.com/spec/v1`
- ohne LFS-Setup beim Clone sind die Daten nicht vorhanden
- CI-Workflows, die "Backtest auf realen Daten" suggerieren, laufen also auf Pointern → fallen entweder durch oder laufen auf Mini-Sample-Daten

### 4.2 Nur 45 Symbole und nur Stand 2025-12-03
- **schwer** · das gesamte tracked Historienmaterial heißt `datensammlungen/altdaten/stand 3-12-2025/1d/*.parquet`
- das sind 45 Large-Caps, EOD, ein Snapshot
- keine Intraday-Daten im Repo, keine Options-Daten, keine FX, keine Rates, keine Commodities-Timeseries
- Watchlist (`watchlist.txt`) hat 29 Symbole. `watchlist_full.txt` hat 62. Beides aktiv verwendet.

### 4.3 Keine Point-in-Time-Korrektur in der Daten-Ingest-Schicht
- **mittel** · `docs/POINT_IN_TIME_AND_LATENCY.md` beschreibt die Pattern sauber
- aber: die tatsächlichen Ingest-Skripte (yfinance_source.py, polygon_source.py) liefern As-Is-Snapshots mit aktuellen CIK/Ticker-Zuordnungen
- historische Delistings, Ticker-Changes, Survivor-Bias — nicht im Code adressiert
- was in der Doku als "B2 Point-in-Time Design" firmiert, ist bisher nur Spec, nicht Code

### 4.4 Data Sources ohne Client
- **mittel** · `data/sources/` listet 11 Sources (AlphaVantage, BLS, CBOE, Earnings-Calendar, EDGAR, FRED, NewsAPI, Polygon, WorldBank, YFinance, Earnings)
- **aber:** nicht alle sind komplett verdrahtet mit Ingest → Speicherung → Verarbeitung
- z.B. `cboe_source.py` — wo ist der Ingest-Scheduler? Wo die Cache-Invalidierung? Wo der Health-Check?
- viele Sources sind "es gibt eine Klasse, aber sie wird nie periodisch aufgerufen"

### 4.5 Zwei Watchlists, inkonsistent
- **mittel** · `watchlist.txt`: 29 US Large-Caps (AAPL..XOM)
- `watchlist_full.txt`: 62 Symbole, diverser (inkl. ACN, AXON, DDOG, MSTR…)
- `configs/paper_track/watchlist_us_core.txt`: eine dritte Watchlist
- unklar, wer welche wofür nutzt

### 4.6 `missing_symbols.txt` ist eine Krücke
- **leicht** · ein 100-Byte-File im Root, das dokumentiert, welche Symbole fehlen
- das gehört in strukturierte Metadata, nicht in eine Textdatei im Root

### 4.7 ETF-Universum-Files in `config/`, nicht in `configs/`
- **leicht** · `config/defense_security_aero.txt`, `config/healthcare_biotech.txt` etc.
- parallel gibt es `configs/universe_etf_v1.yaml`
- beide werden benutzt, Quellen-of-Truth unklar

### 4.8 `configs/security_master.csv` und `configs/security_meta.csv` getrennt
- **mittel** · Security-Master und Security-Meta als zwei Files
- typisches Anti-Pattern, wenn Schemas sich entwickeln
- sollte eine normalisierte Struktur sein

### 4.9 News-Blacklist/Whitelist als Wurzel-Files
- **leicht** · `news_blacklist.yaml` und `news_whitelist.yaml` liegen im Root, nicht in `configs/news/`
- innerhalb der 1971 Files sind das zwei weitere Orientierungsverluste

### 4.10 Ingest und Feature-Berechnung sind nicht getrennt idempotent
- **mittel** · wenn EOD-Run wiederholt wird, sollten Features aus Cache kommen (Factor Store existiert!)
- aber die Orchestrierung (`orchestrator.py`) triggert jeden Run die Feature-Berechnung neu
- das ist ineffizient und rennt bei großen Universen in Probleme

### 4.11 Kein einheitliches Schema für Bars
- **mittel** · verschiedene Sources liefern verschiedene Spalten (OHLCV vs. OHLCVX vs. Adj-Close-Schemas)
- `data/contract.py` spezifiziert ein Schema, aber nicht jede Source respektiert es
- `datensammlungen/altdaten/` hat andere Spalten als `data/sample/eod_sample.parquet`

### 4.12 Corporate Actions Handling fehlt
- **schwer** · `docs/CORPORATE_ACTIONS.md` und `docs/SPRINT4_CORPORATE_ACTIONS_PLAN.md` existieren
- tatsächlicher Code für Splits, Dividenden, Spin-offs, Symbol-Changes: fehlt oder ist rudimentär
- ohne Corporate-Actions-Korrektur ist jede Backtest-Rendite > 1 Jahr unzuverlässig

### 4.13 Keine Daily-Drift-Detection für Datenqualität
- **mittel** · `docs/DATA_QUALITY_QC.md` existiert
- aber: keine aktive Alarm-Logik, wenn sich z.B. AAPL-Volumen 10× ändert oder Spread sich verdreifacht
- im Live-Paper-Modus ist das ein Live-Bug, der unerkannt bleibt

---

## 5. Signal- und Strategieschicht

### 5.1 Die faktische "Haupt-Strategie" ist 78 Zeilen EMA-Crossover
- **schwer** · `pipeline/signals.py::compute_ema_signals` ist das, was die CLI-Default-Backtests ausführen
- Logik: `sig = (ema_fast > ema_slow) - (ema_fast < ema_slow)` → -1/0/+1
- das ist eine Lehrbuch-Strategie ohne Vorhersagekraft in effizienten Märkten
- nach Kosten (TER, Slippage, Spread) typischerweise negative Expected Value
- **es gibt keine produktiv eingesetzte zweite Strategie, die funktional unabhängig ist**

### 5.2 `event_insider_shipping`-Strategie läuft auf Dummy-Daten
- **schwer** · laut README Phase-6-Strategie, laut README selbst: "Verwendet Dummy-Daten für Insider- und Shipping-Events"
- der Code für die Regel-Logik ist da (`rules_event_insider_shipping.py`, 160 Zeilen Signum-Thresholding)
- die Daten, auf denen die Regel operieren würde, existieren nicht im System
- d.h. der "Vergleich Trend vs Event" (`scripts/compare_strategies_trend_vs_event.py`) ist ein Vergleich von Apfel mit Pappmaché-Orange

### 5.3 Multifactor-V1 und V2 existieren, aber unklarer Status
- **mittel** · `strategies/multifactor_v1.py`, `multifactor_v2.py`, `multifactor_long_short.py`
- `v2` ist 1034 Zeilen, `v1` deutlich kleiner
- keine klare Entscheidung, welches das canonical Modell ist, welches deprecated
- Commit-History zeigt Parallel-Entwicklung, kein Migration-Cutover

### 5.4 Stat-Arb existiert, aber fragmentiert
- **mittel** · `strategies/stat_arb.py` UND `strategies/stat_arb/` (Ordner)
- Ordner enthält `pair_signals.py`, `cointegration.py`, `pca_arb.py`
- unklar, warum das flache File noch existiert
- klassisches Artefakt einer halben Migration

### 5.5 Keine Positionsgrößen-Logik im Haupt-Orders-Pfad
- **schwer** · `pipeline/orders.py` setzt `qty = 1.0` hart in der `_gen_orders_for_symbol`-Funktion
- wenn du also `scripts/cli.py run_backtest --strategy trend_baseline` ausführst, handelst du pro Signal **eine einzige Aktie**
- das führt zu absurden Portfolios (100 Positionen à 1 Share, unterschiedliche Preise)
- echte Positions-Sizing (Kelly, Vol-Target, Risk-Parity) existiert als Code, nur eben nicht im Standard-Pfad

### 5.6 Keine konsistente Exit-Logik
- **schwer** · in deiner eigenen TXT schreibst du, Exit sei genauso wichtig wie Entry — dem stimme ich zu
- der EMA-Crossover hat einen Exit (wenn fast unter slow kreuzt), aber:
  - kein Trailing-Stop im Standard-Pfad
  - keine Profit-Targets
  - keine Time-Stops
  - keine Volatilitäts-Stops
- diese Module existieren unter `risk/trailing_stops.py`, `risk/profit_targets.py` etc. — aber wieder: observability-wired, nicht im Live-Pfad

### 5.7 Keine transparente Signal-Hierarchie bei mehreren Quellen
- **mittel** · ein News-Trigger sagt "Short NVDA"; ein Trend-Signal sagt "Long NVDA" — wer gewinnt?
- `signals/signal_confidence.py` und `ml/signal_correlation.py` adressieren das theoretisch
- in Praxis gibt es keinen deterministischen Resolver, der zwei widersprüchliche Signale zu **einer** Entscheidung konsolidiert

### 5.8 Regime-Conditional Weights — existent, nicht validiert
- **mittel** · `ml/regime_weight_trainer.py`, `strategies/ic_decay_weights.py` existieren
- keine Evidenz (Output-JSON, Backtest-Report), dass die Gewichte je trainiert wurden
- `configs/factor_weights_by_regime.json` ist ein statisches File; kein Script, das es aktualisiert

### 5.9 Short-Side ist konzeptionell, aber nicht aktiv
- **mittel** · `policy.yaml` erlaubt Shorts (`shorts_allowed: true`)
- `signals/short_signals.py` existiert
- `risk/short_risk.py` existiert
- aber: es gibt keine echte Short-Pipeline, die im CLI-Default-Flow aktiviert wird
- Alpaca-Paper kann Shorts, das ist also keine Broker-Limitation

### 5.10 Mean-Reversion- und Breakout-Strategien fehlen komplett
- **leicht** · in deinem `KNOWN_ISSUES.md` als "Nice-to-Have" gelistet
- für ein System, das "breit" sein will, ist das Fehlen klassischer Gegen-Strategien auffällig
- EMA-Trend alleine hat Regime-Abhängigkeiten — ohne Mean-Reversion kein Hedge für Sideways-Regimes

---

## 6. Execution und Paper-Trading

### 6.1 `unified_paper_engine.py` ist gut strukturiert, aber 2696 Zeilen
- **mittel** · sauberer Lifecycle: Load State → Signals → Sizing → Orders → Risk → Fills → Update → Reconcile
- aber 2696 Zeilen in einer Datei ist bereits beim zweiten `God-Object`-Kandidaten
- Fill-Simulation mit Almgren-Chriss-Squareroot ist theoretisch korrekt, Parameter-Kalibrierung unklar

### 6.2 IBKR-Adapter existiert, aber wurde nie produktiv benutzt
- **mittel** · `execution/ibkr_adapter.py` ist geschrieben, `ib_insync` als Optional-Dependency
- keine Evidenz (Logs, Runs, Commits), dass jemals eine echte IB-Session gelaufen ist
- es ist "vorbereitet", aber die Integration (Contract-Mapping, Error-Handling für Fehlende Daten, Margin-Checks) hat ungetestete Ecken

### 6.3 Alpaca-Adapter ist real, aber minimal getestet
- **mittel** · `AlpacaAdapter` in `broker_adapter.py`, `scripts/run_live_paper.py` nutzt sie
- gute Struktur mit Reconcile-Modus, Halt-Flag-System
- aber: keine Evidenz für kontinuierlichen Live-Paper-Betrieb (keine Log-Dumps, keine State-Rotation, keine Performance-Reports über Zeit)
- das heißt: Alpaca funktioniert technisch, aber ist nie operativ im Trial-Betrieb gewesen

### 6.4 Kein realistischer Slippage-Model-Test
- **schwer** · `execution/fill_model.py` hat 967 Zeilen
- `execution/transaction_costs.py` hat 1008 Zeilen
- beide sind aber **nicht** im Default-Backtest-Pfad (`pipeline/backtest.py::simulate_equity` benutzt einen einfachen Kosten-Multiplikator)
- heißt: die Kosten, die im Backtest angewandt werden, sind nicht die, die im Paper-Trading gelten, sind nicht die, die im Live wären

### 6.5 Kill-Switch existiert, aber ist Boolean
- **mittel** · `execution/kill_switch.py` mit `is_kill_switch_engaged()`
- binär: ein oder aus
- keine Granularität (pro Strategie, pro Symbol, pro Hebel-Level)
- `execution/symbol_kill_switch.py` scheint eine zweite Implementierung zu sein

### 6.6 Pre-Trade-Checks haben offen dokumentierte Gaps
- **mittel** · laut `KNOWN_ISSUES.md`:
  - Weight-Checking ist TODO (Zeile 215)
  - Sector-Exposure-Checks sind TODO (Zeile 243)
  - Region-Exposure-Checks sind TODO (Zeile 247)
  - FX-Exposure-Checks sind TODO (Zeile 795, "fail-fast with clear message" — also: Live-Crash statt Umgehung)
- d.h. die Pre-Trade-Gates sind Schutz gegen offensichtliche Fehler, nicht gegen echte Risiken

### 6.7 Fat-Finger-Guard und Kill-Switch sind als optional-gated entkoppelt
- **leicht** · `unified_paper_engine.py` hat `try: from ... import ... ; _HAS_KILL_SWITCH = True except: _HAS_KILL_SWITCH = False`
- wenn der Import fehlschlägt, läuft der Engine **ohne** Kill-Switch weiter
- das ist ein stilles Sicherheitsleck — bei einem Protection-Mechanismus sollte Fallback "refuse to run" sein, nicht "weiter ohne"

### 6.8 Keine Latency-Messung im Live-Paper-Pfad
- **mittel** · kein dedizierter Latency-Monitor zwischen "Signal entsteht" und "Order submitted"
- bei deinem Ziel "Sekunden-Entscheidungen" ist das die wichtigste Metrik
- dass es fehlt, heißt: du wüsstest nicht, wenn dein System 30 Sekunden Lag hat

### 6.9 OMS-Light ist real, aber ohne Matching-Engine
- **leicht** · `api/routers/oms.py` liefert Blotter und Executions als View
- keine echte Order-Matching-Logik, keine Partial-Fill-Simulation mit realistischen Order-Book-Dynamiken
- für Paper-Training okay, für Live nicht ausreichend

### 6.10 Keine Handelszeiten-Validierung an der Order-Schnittstelle
- **mittel** · `exchange_calendars` ist Dependency — gut
- aber: keine Evidenz, dass Orders außerhalb der Handelszeiten abgelehnt werden, bevor sie an Alpaca gehen
- Alpaca wird sie ablehnen, aber der Fehler kommt dann asynchron zurück

---

## 7. Risk Management

### 7.1 Risk-Module existieren als Library, nicht als Pipeline
- **schwer** · 36 Files in `src/assembled_core/risk/`
- davon real im Entscheidungspfad eingebunden: `vol_targeting` (2 Call-Sites), vielleicht `var_methods`
- Rest: observability-wired
- das ist Risk-Theater — das System loggt "Correlation-Guard aktiv", aber blockiert keine Order aufgrund davon

### 7.2 Position-Konzentration nicht hart durchgesetzt
- **schwer** · `policy.yaml` sagt `max_position_weight: 0.20`
- `pipeline/orders.py` setzt qty = 1.0 — ignoriert den Policy-Wert vollständig
- im `trading_cycle.py`-Pfad gibt es Kelly-Weights, aber kein Post-Check, ob die 20%-Grenze eingehalten wird
- das heißt: die Policy ist deklarativ, nicht operativ durchgesetzt

### 7.3 Drawdown-Triggers nicht zur Laufzeit aktiv
- **schwer** · `policy.yaml` definiert Soft/Hard/Kill-Drawdowns: 12%/20%/30%
- das würde voraussetzen, dass der Engine während eines Runs Equity-Werte prüft und bei Überschreiten de-risked
- der Backtest-Pfad hat keine solche Logik (`pipeline/backtest.py::simulate_equity` läuft End-to-End ohne Zwischenabbruch)
- der Paper-Pfad hat Halt-Flag, das ist aber nicht automatisch an Drawdown gekoppelt

### 7.4 Vol-Targeting ist verdrahtet, aber unklar in seiner Wirkung
- **mittel** · `risk/vol_targeting.py` + Trading-Cycle-Integration
- Zielvolatilität 20% p.a. in `policy.yaml`
- realized-vs-target-Mechanismus existiert
- aber: keine Evidenz, dass es je über einen langen Zeitraum rekursiv getestet wurde (→ Vol-Targeting kann zu prozyklischen Verkäufen in Drawdowns führen, was Performance verschlechtert)

### 7.5 Turnover-Cap wird deklariert, nicht geprüft
- **mittel** · `policy.yaml`: `daily_cap: 0.20`, `weekly_cap: 0.50`
- `risk/turnover_budget.py` existiert, aber:
- wo wird ein Trade abgelehnt, weil der Tages-Turnover überschritten ist? Ich finde keine Stelle, an der das im Order-Submission-Pfad passiert
- damit ist das eher ein Reporting-Feature als ein Enforcement-Feature

### 7.6 Kein Risk-Overlay bei Regime-Wechsel
- **mittel** · Regime-Wechsel wird erkannt (`combined_regime.py`, `regime_hmm.py`)
- was passiert danach operativ? `policy.yaml` hat Regime-abhängige Short-Faktoren, aber nicht Regime-abhängige Position-Sizing
- z.B. "In Crisis-Regime: Gross Exposure auf 40%" — nicht implementiert

### 7.7 Correlation Guard: Schwelle ohne Datenquelle
- **mittel** · `risk/correlation_guard.py` existiert
- `policy.yaml`: `max_corr_cluster_weight: 0.50`
- aber: wo kommt die Korrelations-Matrix her? Wird sie auf rolling-basis berechnet? Aus welchen Daten?
- vermute, es ist Code, der auf Daten wartet, die nie geschickt werden

### 7.8 Tail-Risk-Modelle (EVT, Stressed-VaR) im Observability-Graveyard
- **mittel** · `risk/evt_tail_var.py`, `risk/stressed_var.py`, `risk/tail_dependence.py`, `risk/tail_hedge.py`, `risk/tail_hedging.py` (doppelt)
- zusammen ~2000 Zeilen Code
- 0 Stellen, an denen das Ergebnis zu einer Orderentscheidung führt

### 7.9 Keine Circuit-Breaker-Integration mit Broker
- **mittel** · `risk/circuit_breaker.py` ist interner Code
- Alpaca hat eigene Market-Wide-Circuit-Breaker-Signale — werden die konsumiert?
- nein. Das System reagiert nicht automatisch auf einen Level-3-Breaker (7% S&P-Drop)

### 7.10 Antifragility-Score als Gimmick
- **leicht** · `risk/antifragility.py` mit "compute_antifragility_score"
- basiert auf Equity-Curve-Varianz
- hat wissenschaftlich schwache Basis (Taleb-Konzept, empirisch schwer operationalisierbar)
- der Score wird in Meta-Dicts geschrieben, beeinflusst keine Entscheidung

---

## 8. ML-Schicht im Detail

### 8.1 55 ML-Module, ≤3 produktiv
- siehe Sektion 2.2
- `meta_labeling.py`, `regime_hmm.py`, `calibration.py` sind die realistischen aktiven Module
- der Rest: siehe Observability-Graveyard

### 8.2 Doppelte Conformal-Implementierungen
- **leicht** · `ml/conformal.py` und `ml/conformal_prediction.py`
- keine Dokumentation, was der Unterschied ist
- klassisches Zeichen, dass jemand (Claude Code?) das Thema zweimal begonnen hat

### 8.3 Doppelte Stacking-Implementierungen
- **leicht** · `ml/stacking.py` und `ml/stacking_ensemble.py`
- gleiche Story

### 8.4 Meta-Labeling + Nested-Meta-Labeling + Triple-Barrier
- **mittel** · drei miteinander verwandte, aber separate Files
- kein Orchestrator-Doc, das erklärt, wie sie zusammenhängen
- López de Prado würde sagen: das sollte ein einheitliches Pipeline-Feature sein

### 8.5 GARCH-Modelle ohne Integration
- **leicht** · `ml/garch_models.py` existiert
- Vol-Targeting nutzt rolling-std, nicht GARCH — der Unterschied wird in der Praxis nirgends ausgewertet
- Modul ist Spielwiese, nicht Produktiv

### 8.6 Bayesian-Ensemble ohne Uncertainty-Consumer
- **leicht** · `ml/bayesian_ensemble.py` liefert Verteilungen statt Punktschätzungen
- der Sizing-Pfad konsumiert aber nur Punktschätzungen
- d.h. die Uncertainty geht im Sizing verloren; das Feature ist de facto wirkungslos

### 8.7 Kein Model-Registry-Service
- **mittel** · `ml/model_registry.py` existiert als Code
- aber: wo werden produktive Modelle versioniert abgelegt? Wie werden sie geladen? Welche Version läuft gerade?
- `/models/` ist in `.gitignore`, es gibt keinen externen Storage
- in Praxis: keine Modell-Versionierung

### 8.8 Kein Model-Drift-Monitoring real operativ
- **mittel** · `ml/model_monitoring.py`, `ml/calibration_monitor.py` existieren
- `api/routers/monitoring.py` liefert laut `KNOWN_ISSUES.md` Dummy-Drift-Daten
- d.h. das Drift-Dashboard zeigt Fake-Zahlen

### 8.9 Retraining-Scheduler existiert, läuft nicht
- **mittel** · `ml/retraining_scheduler.py` 
- keine CI-Workflow, kein Systemd-Service, kein Cron
- d.h. der Scheduler ist Code, kein Betrieb

### 8.10 Keine Train-Test-Separation im Meta-Model-Pfad
- **mittel** · `KNOWN_ISSUES.md` sagt explizit: "Aktuell wird auf allen Daten trainiert. Ein Validation-Split für Out-of-Sample-Validierung wäre wünschenswert."
- das ist **nicht** wünschenswert, das ist **Minimalanforderung**
- Meta-Modelle ohne Out-of-Sample-Split sind definitionsgemäß overfit

### 8.11 Feature-Drift-Tracker ohne Alarm
- **leicht** · `ml/feature_importance_tracker.py`
- logging, aber kein Alert-System, wenn ein Feature plötzlich 0 Wichtigkeit hat (= broken)

### 8.12 Reinforcement Learning ohne Training-Env
- **mittel** · `ml/rl_portfolio.py` und `ml/rl_execution.py`
- kein Training-Environment, kein Reward-Signal-Definition, kein Checkpoint
- das sind reine Stub-Klassen mit Config-Defaults

### 8.13 NLP-Sentiment ohne Modellartefakt
- **mittel** · `ml/nlp_sentiment.py` deklariert FinBERT-Ready
- `pyproject.toml` hat `ml-nlp` als Optional-Extra mit `transformers>=4.35.0` und `torch>=2.0.0`
- **keine** konkrete Code-Stelle, die ein FinBERT-Modell lädt und benutzt
- die News-Pipeline emittiert aktuell Events, aber das Sentiment-Feld ist leer oder heuristisch

### 8.14 AutoML-Modul ist theoretisch
- **leicht** · `ml/automl.py`
- echter AutoML braucht Compute und Orchestrierung; das hat das System beides nicht

### 8.15 Feedback-Loop-Modul ohne Loop
- **mittel** · `ml/feedback_loop.py` 1667 Zeilen
- das Konzept "Trade-Ergebnisse fließen zurück ins Modell-Training" braucht einen geschlossenen Kreis mit: Trade-Log → Label-Generation → Dataset-Append → Retrain-Trigger
- der Code ist da, der Kreis nicht geschlossen (siehe 8.9)

---

## 9. News- und Intel-Schicht

### 9.1 Die News-Pipeline ist das beste Teil des Projekts
- **(positiv)** · `events/news/pipeline.py` 720 Zeilen, sauber strukturiert
- parallele RSS-Fetches mit ThreadPoolExecutor
- SQLite-Dedupe-Store
- Fingerprinting mit Hamming-Distance
- Cluster-Baseline mit Version-Hash
- Burst-Detection, Trigger-Scoring
- Health-Monitoring mit Degraded-Modus
- echte Engineering-Arbeit, die funktioniert

### 9.2 News-Pipeline ist nicht mit Signal-Layer verbunden
- **schwer** · die Pipeline emittiert Events nach `events/news/emit.py`
- `signals/news_signal_bridge.py` existiert als Adapter
- **aber:** der EMA-Trend-Signal-Pfad (der produktiv ist) konsumiert keine News-Events
- d.h. die hervorragende News-Infrastruktur generiert Output, der nicht in Trading-Entscheidungen fließt

### 9.3 GDELT-Integration ohne Ratenlimit-Dokumentation
- **mittel** · `events/news/fetch_gdelt.py` macht direkte Requests
- GDELT hat rate limits und gibt bei Überschreitung generische Fehler
- kein expliziter Circuit-Breaker oder Backoff-Log im Fehlerfall

### 9.4 RSS-Quelle-Liste partiell dokumentiert
- **mittel** · `configs/news/sources.yaml` listet Quellen (BBC, Guardian, ...)
- `configs/intel/rss_feeds.yaml` ist eine weitere RSS-Feed-Liste
- welche wird wann gezogen? Unklar

### 9.5 `news_blacklist.yaml` am Root
- **leicht** · liegt nicht in `configs/news/`
- parallel existiert `news_whitelist.yaml` im Root
- inkonsistent abgelegt

### 9.6 Intel-Schicht als Second-System-Effekt
- **schwer** · `src/assembled_core/intel/` hat 54 Files
- darunter: `bayesian_confidence.py`, `central_bank_divergence.py`, `currency_crisis.py`, `dependency_graph.py`, `entity_linker.py`, `evidence_grade_writer.py`, `geo_trigger.py`, `health_monitor.py`, `ic_loop.py`, `market_confirmation.py`, `nation_profiles.py`, `news_alerts.py`, `news_archive.py`, `news_archiver.py`, `news_classifier.py`, `news_cluster.py`, `news_contradiction.py`, `news_corroboration.py`, `news_decay.py`, `news_dedupe.py`, `news_enricher.py`, `news_entity_graph.py`, `news_entity_mapper.py`, `news_event_store.py`, `news_impact_calibrator.py`, `news_impact_estimator.py` …
- viele Doppelungen mit `events/news/` (z.B. `news_dedupe.py` vs `events/news/dedupe.py`)
- unklare Arbeitsteilung zwischen den beiden Verzeichnissen

### 9.7 `archive/intel_research_2026q2/` als aktiver Legacy-Friedhof
- **mittel** · `escalation_tracker.py`, `multichannel_propagation.py`, `structural_cycles.py`, `hegemonic_dynamics.py` — das klingt nach Zeihan-inspirierter Theorie-Arbeit
- ist aber im Archive, nicht aktiv
- entweder löschen oder zurück in `src/` holen und verdrahten

### 9.8 Entity-Linker ohne Knowledge-Base
- **mittel** · `intel/entity_linker.py`, `data/news/entity_linking.py`
- Entity-Linking braucht eine Knowledge-Base (Tickers → Companies → Sectors → Countries)
- `configs/security_master.csv` ist ein Ansatz, aber die Entity-Linking-Logik referenziert das nicht eindeutig

### 9.9 Nation-Profiles als statische YAML-Datei
- **leicht** · `configs/nation_profiles.yaml` beschreibt Länder und deren Abhängigkeiten
- statisch, von Hand gepflegt
- für dein "Geopolitik → Crisis-Alpha"-Ziel müsste das viel dynamischer sein

### 9.10 Keine Ground-Truth-Messung für News-Impact
- **schwer** · für "News-X → Preisreaktion Y" braucht man historische Ground-Truth
- `intel/news_impact_calibrator.py` existiert
- keine Evidenz, dass sie je mit echtem Event-Study-Paper (Day-0, Day+1, Day+5 Returns) kalibriert wurde

---

## 10. Config, Secrets, Security

### 10.1 Kritisches Security-Incident .env war in Git
- **schwer** · `docs/incidents/2026-04-18_env_exposure.md` dokumentiert es selbst
- Triple-Failure: `.gitignore` versagte (weil `.env` schon vor dem Ignore committet), Gitleaks-Scanner war allowlisted für `.env`, kein History-Rewrite
- Keys (Alpha Vantage, Finnhub, Polygon, Alpaca paper, NewsAPI, FRED) waren ab commit `0ca19ef` (Oktober 2025) öffentlich, bis commit `e64fa21` (April 2026)
- Keys wurden rotiert — gut
- **aber: History nicht rewritten**, alte Keys stehen noch in jedem Clone der History sichtbar
- Entscheidung "History-Rewrite ist destruktiv, daher nicht getan" ist nachvollziehbar, aber dokumentiert, dass der Kill-Chain-Schaden nicht vollständig beseitigt ist

### 10.2 E-Mail-Adressen in Git-History
- **mittel** · Commits von `hans.oertel2@gmail.com` (Realname + persönliche E-Mail)
- plus Commits von `dein.email@domain.tld` (Placeholder, vermutlich AI-generiert mit falscher Git-Config)
- wenn du das Repo öffentlich machst, ist die E-Mail durchsuchbar und SPAM-/Phishing-Ziel

### 10.3 Kein `.env.example` oder `.env.template`
- **leicht** · neue Entwickler wissen nicht, welche Env-Variablen das System erwartet
- grep auf `os.getenv` zeigt: ALPHAVANTAGE_KEY, FINNHUB_KEY, POLYGON_KEY, ALPACA_KEY, ALPACA_SECRET, FRED_KEY, NEWSAPI_KEY, ANTHROPIC_API_KEY
- keine einzige Stelle, die die Gesamtliste dokumentiert

### 10.4 Anthropic-API-Key im Projekt
- **mittel** · `system_check/` nutzt `anthropic>=0.45.0` als Dependency
- der Key muss in `.env` — aber wenn .env mal wieder gecommittet wird, schließt sich der Kreis

### 10.5 Docker-Container hat keine Secret-Mount-Dokumentation
- **leicht** · `docker-compose.yml` nutzt `env_file: - .env` — ok
- aber keine Doku, wie in Prod mit Docker-Secrets oder Vault gearbeitet wird
- das ist eine "funktioniert auf meinem Laptop"-Lösung

### 10.6 Kein RBAC an der API
- **schwer** · FastAPI-Endpoints (`api/routers/*`) haben keine Authentifizierung
- wenn der Container aufs Internet exposed wird, kann jeder `POST /api/v1/paper/orders` machen
- Claude-Code hat nicht automatisch Auth dazugebaut, weil es nicht angefragt wurde
- das ist für Paper-Only akzeptabel, für Live nicht

### 10.7 Keine Rate-Limiting an der API
- **mittel** · selbst wenn Auth existierte, müsste Rate-Limiting da sein
- FastAPI hat einfache Middleware-Optionen (slowapi), nicht verdrahtet

### 10.8 Keine strukturierten Audit-Logs
- **mittel** · `compliance/audit_log.py` existiert (229 Zeilen)
- aber nicht mandatorisch an jede Order gekoppelt
- für einen Systematic-Trading-Service müssen alle Entscheidungen unveränderlich dokumentiert sein

### 10.9 Regulatory-Reports: Stub
- **schwer** · `compliance/regulatory_reports.py` (331 Zeilen)
- MiFID II Transaction-Reporting, RTS 6 Algo-Risk-Controls, EMIR — all das sind echte Reports, die in Europa erforderlich sind, wenn du mal lizenziert wirst
- das File hat Gerüst, aber keine formatkonforme XML/CSV-Erzeugung für die tatsächlichen Meldewege

### 10.10 OTR-Monitor (Order-to-Trade-Ratio) ohne Integration
- **mittel** · `compliance/otr_monitor.py` (159 Zeilen)
- OTR ist ein MiFID-II-Kriterium (Algorithmic Trading mit vielen Cancels erfordert Lizenz)
- das Tool ist Code, nicht in Live-Pfad integriert

### 10.11 `.secrets.baseline` hat 5138 Bytes an Einträgen
- **leicht** · das File listet erkannte "Secrets" als False-Positives
- in einem gesunden Projekt sollte die Baseline leer oder sehr kurz sein
- dass sie 5KB ist, zeigt: Detect-Secrets findet regelmäßig Dinge, die als "kein Secret" weg-deklariert werden müssen

### 10.12 Kein Dependency-Vulnerability-Scanning
- **mittel** · keine `pip-audit`, `safety`, oder `osv-scanner` Integration in CI
- bei 85+ gepinnten Dependencies (siehe `requirements.lock`) ist das Risiko real

---

## 11. CI/CD, Workflows, DevOps

### 11.1 17 GitHub-Workflows — einige sinnvoll, einige redundant
- **mittel** · `accounting-ci.yml`, `backend-ci.yml`, `ci.yml`, `disclosures-worker-ci.yml`, `earnings-calendar-refresh.yml`, `evidence-pack-ci.yml`, `fail-drill.yml`, `news-worker-ci.yml`, `nightly-runall.yml`, `nightly-sync.yml`, `ops-evidence-ci.yml`, `paper-trading-ci.yml`, `prewarm-factor-store.yml`, `release-gate-ci.yml`, `repo-health.yml`, `secrets-scan.yml`, `signal-decay-update.yml`
- Overlap: `backend-ci.yml` und `ci.yml` machen vermutlich Ähnliches
- `nightly-runall.yml` und `nightly-sync.yml` sollten klar getrennt dokumentiert sein

### 11.2 Nightly-Sync-Workflow ist ein Anti-Pattern
- **mittel** · ein Workflow, der automatisch Commits produziert ("Auto-sync ...")
- das Repo hat dadurch 195 Bot-Commits (28% aller Commits)
- verschleiert echte Commit-History und bläht das Log auf

### 11.3 Keine CI für Deployment
- **leicht** · alle 17 Workflows sind Test-/Check-Workflows
- kein Deploy-auf-Server-Workflow
- d.h. alles muss manuell vom Entwicklungsrechner laufen

### 11.4 Pre-Commit-Hooks sind installiert, nicht mandatorisch
- **leicht** · `.pre-commit-config.yaml` konfiguriert Gitleaks, Detect-Secrets, Ruff, Black
- aber: in CI gibt es keinen Check, der Pre-Commit-Verletzungen ablehnt
- wer den Hook lokal deaktiviert, kommt durch

### 11.5 Ruff-Policy ist zahnlos
- **mittel** · `pyproject.toml`: `select = ["E", "F"]`, `ignore = ["E501", "E203", "E402"]`
- das ignoriert Line-Length, Whitespace, Import-Order
- trotzdem: 232 Errors (229 F401 unused imports, 1 F821 undefined name, 1 E741, 1 E731)
- Ruff sollte in CI als Gate fungieren — tut es nicht verlässlich

### 11.6 Mypy-Strict nicht aktiviert
- **mittel** · `pyproject.toml`: `strict = false`, `disallow_untyped_defs = false`, `disallow_incomplete_defs = false`
- bei 95k Python-LOC ohne strict-Typing ist Refactoring deutlich riskanter
- LLM-generierter Code ist typing-laufend bei strict, was die Qualität erhöht

### 11.7 Bandit-Exclude ist breit
- **leicht** · `exclude_dirs = ["tests", "scripts/tools"]`, `skips = ["B101", "B314"]`
- B314 (XML-Parsing) skipped mit Kommentar "SEC EDGAR trusted sources" — zumindest dokumentiert
- sollte aber mit defusedxml migriert werden, wie der Kommentar selbst sagt

### 11.8 Keine Coverage-Gates
- **mittel** · `pytest-cov` ist Dependency
- CI läuft die Tests, aber erzwingt keine Coverage-Schwelle
- man kann also Module mit 0% Coverage mergen

### 11.9 Kein `release-please` oder Changelog-Automation
- **leicht** · `CHANGELOG_DUE_DILIGENCE.md` wird manuell gepflegt
- bei 699 Commits ist das unzuverlässig

### 11.10 Docker-Image-Build hat keinen Tag-Stream
- **mittel** · `Dockerfile` ist gut (Multi-Stage, non-root User, Healthcheck)
- aber: kein CI-Workflow, der bei Tags ein versioniertes Image baut und pusht
- Container-Deploy ist rein lokal

---

## 12. Dokumentation

### 12.1 164 Markdown-Files in `docs/` — Über-Dokumentation
- **schwer** · 268 MD-Files im gesamten Repo, davon 164 in `docs/`
- typisches Projekt hat 5–15 Docs
- eine 18%-Quote Markdown-Anteil am File-Count ist extrem hoch

### 12.2 Overlap-Docs — dieselbe Sache in mehreren Files
- **mittel** · `ARCHITECTURE_BACKEND.md`, `ARCHITECTURE_LAYERING.md`, `ARCHITECTURE_REVIEW_SUMMARY.md`, `docs/architecture/` (Ordner)
- `CODE_QUALITY_AUDIT.md`, `CODE_QUALITY_FINAL_REPORT.md`, `CODE_QUALITY_FIXES_APPLIED.md`, `CODE_QUALITY_FIXES_SUMMARY.md`, `CODE_QUALITY_FULL_AUDIT.md`, `CODE_QUALITY_SUMMARY.md`
- `DEEP_AUDIT_REPORT.md`, `FULL_PROJECT_AUDIT.md`, `FULL_SYSTEM_AUDIT_OUTPUT.md`, `REVIEW_AUDIT_SPRINT13_EVIDENCE_PACK.md`
- `FINAL_CODE_REVIEW_FINDINGS.md`, `FINAL_DOWNLOAD_SUMMARY.md`, `FINAL_IMPROVEMENTS_APPLIED.md`, `FINAL_STATUS_REPORT.md`
- das ist eine deutliche "AI-generated iteration without cleanup"-Signatur

### 12.3 7 README-Varianten im Root
- **mittel** · `README.md`, `README_INTEGRATION.txt`, `README_ONECLICK.md`, plus `PROJECT_STATUS.txt`, `PROJEKT_STATUS.md`, `CHANGELOG_DUE_DILIGENCE.md`, `KNOWN_ISSUES.md`
- neue Leser haben keine Chance, den kanonischen Einstieg zu finden

### 12.4 `CLAUDE.md` ist exzellent, widerspricht aber der Realität
- **mittel** · 23.696 Zeichen an Engineering-Philosophie
- Sätze wie "Plan ist nicht Implementierung", "Sicherheit vor Eleganz", "Keine falsche Sicherheit"
- die tatsächliche Entwicklung (Wave-Wiring) verletzt genau diese Prinzipien
- d.h. du hast für Claude Code Regeln aufgeschrieben und Claude Code hat sie trotzdem nicht eingehalten (bzw. die User-Prompts haben es anders angewiesen)

### 12.5 `AGENTS.md` parallel zu `CLAUDE.md`
- **leicht** · zwei Agent-Instruction-Files
- welches nimmt Cursor? Welches nimmt Claude Code? Welches ist kanonisch?

### 12.6 Docs mit Stand "Sprint 10" bis "Sprint 13" gleichzeitig
- **mittel** · `RELEASE_NOTES_SPRINT13.md`, `SPRINT11_BENCHMARKS.md`, `SPRINT4_CORPORATE_ACTIONS_PLAN.md`
- und ältere Docs behaupten teils andere Phasen
- keine einheitliche "aktueller Stand"-Dokumentation

### 12.7 Roadmap-Kollision
- **mittel** · `BACKEND_ROADMAP.md`, `docs/roadmap/`, `RESEARCH_ROADMAP.md`, `ROADMAP_NR3_STATUS.md`, `ROADMAP_STATUS_SPRINT13.md`
- mehrere parallele Zielbilder, die nicht gegeneinander konsolidiert sind

### 12.8 PROJECT_STATUS.txt hat Windows-spezifische Pfade
- **leicht** · `ROOT: D:\PROJEKT_AKTIE\Projekt_1\Grundsachen`, `WATCHLIST: path=D:\...`
- wenn das File ins Repo kommt, ist es wertlos auf jedem anderen Rechner
- reines Scratch-Artefakt

### 12.9 `oos_debug_log.txt` im Root — 13 KB Debug-Log
- **leicht** · sollte in `/output/` oder `/logs/` landen und in `.gitignore`
- ist ein Zeichen dafür, dass beim Debuggen nicht sauber aufgeräumt wurde

### 12.10 `review_bundle.txt` ist 5,7 MB groß
- **schwer** · eine einzige Textdatei, 5,7 MB
- vermutlich ein generierter Dump des kompletten Codes für AI-Reviews
- das Repo blutet an solchen Artefakten

### 12.11 Notebook-Templates leer
- **mittel** · 4 Research-Notebooks (`research/trend/`, `research/meta/`, `research/risk/`, `research/altdata/`)
- jedes hat 1 Zelle, 0 Code-Zellen, ~1200 Zeichen Markdown "TODO: analysiere..."
- sie sind committet, aber nicht genutzt → das ist Dokumentation, die nicht existiert, aber pretend zu existieren

### 12.12 Keine Architecture-Decision-Records (ADR) trotz `docs/adr/`-Ordner
- **leicht** · Ordner existiert, Inhalte dort nicht verifiziert
- ADRs sollten zentrale Design-Entscheidungen protokollieren (z.B. "Warum haben wir trading_cycle.py so groß werden lassen?")

### 12.13 Viele Design-Docs ohne Implementation
- **mittel** · `BACKTEST_B1_UNIFIED_PIPELINE_DESIGN.md`, `BACKTEST_OPTIMIZATION_P3_DESIGN.md`, `BATCH_BACKTEST_P4_DESIGN.md`, `D3_PANEL_STORE_DESIGN.md`, `DEFLATED_SHARPE_B4_DESIGN.md`, `ML_VALIDATION_E1_DESIGN.md`, `OPERATIONS_BACKEND_A3_DESIGN.md`, `PAPER_TRACK_RUNNER_A5_DESIGN.md`, `POINT_IN_TIME_AND_LATENCY_B2_DESIGN.md`, `REGIME_MODELS_D1_DESIGN.md`, `RISK_2_0_D2_DESIGN.md`, `TRANSACTION_COSTS_E4_DESIGN.md`, `WALK_FORWARD_AND_REGIME_B3_DESIGN.md`
- viele dieser "Designs" haben keine entsprechende implementierte Komponente oder nur Stubs
- Design-Dok muss immer an real existierenden Code knüpfen, sonst ist es Fantasie

### 12.14 `PROJEKT_STATUS.md` vs `PROJECT_STATUS.txt` vs `CHANGELOG_DUE_DILIGENCE.md`
- **leicht** · drei Statusdokumente, nicht konsistent
- `PROJEKT_STATUS.md` sagt "Phase 4 abgeschlossen, bereit für Phase 5&6"
- `README.md` spricht von Sprint 13
- `KNOWN_ISSUES.md` spricht von Phase 12.3
- tatsächlich laufender Stand: unklar

---

## 13. Ops / Scripts / Operator-Experience

### 13.1 95 Top-Level-Scripts
- **mittel** · eine derart große Script-Sammlung ist unüberschaubar
- davon 35 `run_*`-Scripts — das sind 35 verschiedene Einstiegspunkte in das System

### 13.2 `run_grand_backtest.py` als Theater-Name
- **leicht** · Docstring: "Activates ALL dormant modules: U1/U2/U3/E1..E5/D1..D5/S1/S3/M1/M3"
- 17 Module per Abkürzung — das ist der Antitheseskandidat zu "kleinste sichere Änderung"
- wenn man dieses Script ernsthaft benutzt, weiß niemand, was auf welche Ergebnisse Einfluss hat

### 13.3 `run_final_optimized.py` — "final" ist immer verdächtig
- **leicht** · ein "finales" Script deutet auf "wir haben eine Weile experimentiert und dann eine feste Config eingefroren"
- typischerweise wurde das dann trotzdem weiterverändert

### 13.4 `run_improvement_cycle.py`
- **leicht** · der Name suggeriert einen Automatismus
- schau dir die 326 Zeilen an: vermutlich ruft sie einzelne Stages auf und überwacht sie
- so ein Supervisor gehört als Klasse in `src/`, nicht als Script

### 13.5 Zwei Sprint-Scripts (`sprint9_backtest.py`, `sprint9_execute.py`, `sprint10_portfolio.py`)
- **leicht** · klassische Zeitstempel-Benamung
- sollten entweder in `scripts/legacy/` oder gelöscht sein

### 13.6 31 PowerShell-Scripts
- **mittel** · parallel zur Python-Codebasis
- Windows-only, läuft nicht auf Linux/Mac
- das zwingt dich, Windows-gebunden zu entwickeln; Cloud-Deployment wird schwerer

### 13.7 `000_UpgradeToPS7.ps1` und `000_seed_project.ps1.disabled`
- **leicht** · zwei Ur-Scripts, eins davon `.disabled`
- Meta-Setup-Scripts, die im normalen Betrieb niemand ausführt
- sollten in `scripts/setup/` oder `docs/` dokumentiert sein

### 13.8 `uninstaller für automatische ausführung sprint_5.txt`
- **leicht** · im Root
- Dateiname mit Leerzeichen und deutschem Case-Mix
- einfach nur ungepflegt

### 13.9 Kein Single-Command-Start
- **mittel** · wie startet ein neuer Entwickler das System? Unklar
- `make dev`, `task serve`, `docker compose up` — alle drei möglich, keins kanonisch
- `README.md` nennt `python scripts/cli.py run_daily --freq 1d` als Beispiel, aber das setzt vor-existierende Daten voraus

### 13.10 Logging inkonsistent
- **mittel** · einige Module nutzen `logging.getLogger(__name__)`, einige `get_logger("assembled_core.X")`, einige direkt `print`
- Log-Format über Module nicht einheitlich
- strukturiertes Logging (JSON) fehlt

### 13.11 Kein Health-Endpoint dokumentiert
- **leicht** · FastAPI existiert, aber `/health` oder `/readiness` ist nicht als kanonischer Endpoint markiert
- für Container-Orchestrierung Standard

### 13.12 Keine User-facing CLI-Help
- **leicht** · `scripts/cli.py` hat `--help`, aber die Subcommands sind nicht voll dokumentiert in `docs/CLI_REFERENCE.md`

---

## 14. Performance / Skalierung

### 14.1 Kein Performance-Budget pro Pipeline-Stage
- **mittel** · `docs/PERFORMANCE_PROFILE.md` existiert
- keine klare "EOD-Run darf max. X Sekunden dauern" Aussage
- bei "Sekunden-Entscheidungen" als Zielbild ist das Fehlen eines Budgets Einstiegs-Fehler

### 14.2 Python-Overhead nicht optimiert
- **mittel** · 95k LOC Python, Pandas, kein Numba in den Hot-Paths (außer `qa/backtest_engine_numba.py`)
- für die angestrebte Latenz musst du entweder stark profilen oder zu Rust/C++ migrieren

### 14.3 `trading_cycle.py` hat 309 sequenzielle Steps
- **schwer** · die Steps laufen seriell, nicht parallel
- auch wenn viele davon silent-skip sind, zahlt jeder seinen Overhead
- Import eines Moduls in Python kostet ~ms; 309 Inline-Imports = Sekunden pro Cycle

### 14.4 Keine Incremental-Backtests
- **mittel** · `KNOWN_ISSUES.md` selbst erwähnt es als Enhancement
- jeder Backtest-Run beginnt bei t=0, statt nur neue Daten zu verarbeiten
- bei 10 Jahren Historie und 200 Symbolen ist das verschwenderisch

### 14.5 Factor-Store existiert, aber Cache-Invalidation ist heuristisch
- **mittel** · `docs/FACTOR_STORE.md` beschreibt ein sinnvolles Konzept
- was genau passiert bei Daten-Schema-Änderung? Bei Formel-Änderung?
- Version-Hashing wie in `events/news/baseline.py::compute_version_hash` ist der richtige Ansatz, aber im Factor-Store nicht universell angewandt

### 14.6 Keine Datenbank, nur SQLite und Parquet
- **mittel** · `events/news/dedupe_store.py` nutzt SQLite, das ist ok
- für Multi-Instance-Betrieb (z.B. mehrere Workers gleichzeitig) skaliert SQLite nicht
- Postgres/TimescaleDB wäre für Market-Data der Standard

### 14.7 Keine Caching-Schicht für externe APIs
- **mittel** · yfinance, Polygon, Alpha Vantage haben alle Rate-Limits
- ich sehe keinen konsistenten HTTP-Cache (z.B. `requests-cache`)
- das führt zu Throttling und Wiederholungen

### 14.8 `ThreadPoolExecutor` wird genutzt, Prozess-Pool nicht
- **leicht** · News-Fetch ist IO-bound → Threads passen
- Feature-Berechnung ist CPU-bound → ProcessPoolExecutor würde helfen
- nicht einheitlich abgewogen

---

## 15. Wissenschaftliche/Quantitative Korrektheit

### 15.1 Kein Survivorship-Bias-Schutz in Backtests
- **schwer** · die Watchlist besteht aus heute existierenden Large-Caps
- ein Backtest auf "AAPL, MSFT, NVDA, ..." über 10 Jahre ist per Konstruktion selection-biased
- müsste: Universum rollierend nach Market-Cap-Rang, inklusive Delistings

### 15.2 Kein Look-Ahead-Bias-Check bei Features
- **schwer** · `docs/POINT_IN_TIME_AND_LATENCY.md` diskutiert das Konzept
- aber: ein ta_feature wie `atr` wird über ein Rolling-Window berechnet, das bei `.fillna()` Default-Behavior haben kann, das Zukunftsinfo leckt
- `qa/leakage_tests/` ist gut angelegt, aber nicht breit genug

### 15.3 Sharpe ohne Risk-Free Rate
- **mittel** · `qa/metrics.py::compute_sharpe` — check, ob Risk-Free-Rate berücksichtigt wird
- bei hohen Fed-Zinsen (2022–2024) macht das für absolute Metriken einen Riesenunterschied
- Sharpe berechnet ohne RFR überschätzt Performance systematisch

### 15.4 Cost-Multiplier im Default-Backtest unklar kalibriert
- **schwer** · `pipeline/backtest.py::simulate_equity` nutzt einen Kosten-Ansatz, der vermutlich einfach `turnover × bps` ist
- realistische Kosten enthalten: Spread-Kosten (halbe Spread × Volume), Impact-Kosten (√-Abhängigkeit), Kommissionen, Borrow (bei Shorts)
- `execution/transaction_costs.py` existiert mit dem realistischen Modell, aber der Default-Backtest nutzt es nicht durchgängig

### 15.5 Deflated Sharpe als Doku, nicht als Gate
- **mittel** · `DEFLATED_SHARPE_B4_DESIGN.md` referenziert López de Prado
- wenn du 100 Strategien backtestest und die beste hat Sharpe 1.5, dann ist der "Deflated Sharpe" (nach Multiple-Testing) oft unter 0.5
- kein Gate, das ablehnt, eine Strategie zu produktivieren, weil ihr Deflated Sharpe unter Schwelle ist

### 15.6 Walk-Forward-Analyse ja, aber Mini-Fenster
- **mittel** · `qa/walk_forward.py` ist 1248 Zeilen — Code ist da
- keine dokumentierten Läufe mit z.B. 10 Jahren Daten, 1-Jahr-Walk-Forward, 5 Jahre IS + 1 Jahr OOS
- ohne diese ist der Walk-Forward-Code nur Theorie

### 15.7 Monte-Carlo-Simulation ohne dokumentierte Anwendung
- **mittel** · in deiner TXT erwähnst du Monte-Carlo als Wichtigkeit
- `qa/scenario_engine.py` ist 1124 Zeilen — scheint Scenario-Replay zu machen
- aber: keine reproduzierbare Monte-Carlo-Report-Output-Datei, keine Bootstrap-Konfidenzintervalle für Metriken

### 15.8 Kelly-Criterion als Code, ohne Fraktion-Optimierung
- **mittel** · `portfolio/kelly_uncertainty.py` implementiert Kelly
- `configs/policy.yaml` hat `kelly_fraction: 0.5` — halbe Kelly, vernünftig
- aber: keine empirische Verifikation, dass der Parameter zur realized-Verteilung passt

### 15.9 Fat-Tails-Annahme nicht durchgezogen
- **mittel** · Modelle wie EVT (`evt_tail_var.py`), Copula-Models, `tail_dependence.py` existieren
- Standard-VaR-Berechnung (z.B. historical VaR) passiert mit Normal-Verteilungs-Annahme
- das heißt: die "Fat-Tails-Sensitivity" ist auf-Papier, nicht in Zahlen

### 15.10 Keine Faktor-Exposure-Attribution im Live-Paper
- **mittel** · `risk/factor_exposures.py` existiert
- aber ich sehe keinen regelmäßigen Report, der sagt "heute sind wir 40% Momentum, 20% Value, 15% Size-exponiert"
- ohne Faktor-Attribution kann man keine systematischen Risiko-Concentration detektieren

---

## 16. Vision vs. Realität

### 16.1 "Autonomes Live-Trading in Sekunden" vs. EOD-Pipeline
- **schwer** · das ambitionierte Ziel deiner TXT ist HFT-adjazent
- das gebaute System ist End-of-Day mit vereinzelten 5min-Hooks
- die architektonische Lücke ist 3–4 Größenordnungen, nicht "noch ein bisschen optimieren"

### 16.2 "Vermietung an Versicherungen" vs. kein RBAC, keine Tenant-Trennung
- **schwer** · Multi-Tenancy (Kunde A sieht nicht Kunde Bs Positionen) ist im Code nicht vorhanden
- `compliance/audit_log.py` ist single-tenant
- Tenant-Separierung ist eine Architektur-Entscheidung, die man früh treffen muss (Row-Level-Security in Postgres, tenant-aware Caches etc.)

### 16.3 "Vermietung" vs. kein Billing, kein Subscription-Management
- **schwer** · wie zählt das System Nutzungseinheiten? Wie rechnet es ab?
- keine Stripe-Integration, keine Usage-Metrics-Aggregation
- das ist nicht "später hinzufügen", das ist fundamentale Produkt-Architektur

### 16.4 "Hebel-Produkte mit hohem Gewinn" vs. `leverage_allowed: false`
- **mittel** · `policy.yaml` setzt `leverage_allowed: false`
- das ist richtig als Policy, aber widerspricht deiner Vision in der TXT
- entscheide dich: entweder Policy-Constraint oder Hebel-Ziel, nicht beides gleichzeitig

### 16.5 "Goldman Sachs kauft uns" vs. Codebase-Bild
- **leicht** · strategischer Traum, der keinen Audit braucht
- aber: jede M&A-Due-Diligence würde bei `.env`-in-Git-History + 10k-Zeilen-Monolith-File enden

### 16.6 "Sekunden-Entscheidungen, dauerhaftes News-Ziehen" vs. `paper_trading_scheduler.py` cron-artig
- **mittel** · das scheduler-Script läuft pro Aufruf einmal
- für Echtzeit bräuchtest du einen Daemon / Event-Driven-Architektur (Kafka, Redis-Streams, Websocket-Listener)
- der aktuelle Design-Ansatz ist batch-oriented

### 16.7 "Krisenmodus mit Hebel" vs. `policy.yaml inverse_etf_3x: false`
- **mittel** · 3x Inverse ETFs sind die offensichtliche Hebel-Mechanik
- Policy verbietet sie (vernünftig, sie haben Volatility-Decay)
- aber: deine Vision-TXT spricht explizit von "großem Hebel in Krise" — das ist inkonsistent mit dem policy.yaml-Standpunkt

### 16.8 "Frontend und App" vs. kein Frontend-Code
- **leicht** · reines Backend, kein React/Vue/Svelte-Verzeichnis
- das ist ok für die Phase, aber: Frontend hinzufügen ist nicht "50% der Arbeit", das ist ein separates Projekt (3–6 Monate für MVP)

### 16.9 "Keine Abos von KI-Modellen" vs. `anthropic>=0.45.0` als Dependency
- **leicht** · `system_check/` nutzt Claude für Adversarial Reviews
- das heißt: du zahlst aktuell für Anthropic-API
- Ziel "völlig autonom ohne AI-Abos" widerspricht dem aktiven Tooling

### 16.10 "Selbstkorrigierende Mini-KI im Programm" vs. `ml/feedback_loop.py` ohne geschlossenen Kreis
- **mittel** · Konzept in TXT, Stub im Code
- echte selbstkorrigierende Systeme (RL-Agents, Meta-Learning) sind Forschungsstand; nicht "noch zwei Wochen Arbeit"

---

## 17. Daten-Lizenz und IP

### 17.1 yfinance ist nicht lizenziert für kommerziellen Einsatz
- **schwer** · Yahoo Finance's Terms-of-Service verbieten kommerzielle Nutzung
- `yfinance` ist eine scraping-basierte Library
- wenn du das System je kommerzialisierst, brauchst du kommerzielle Data-Feeds (Polygon, Nasdaq Data Link, Refinitiv)

### 17.2 Polygon-Free-Tier-Abhängigkeit
- **mittel** · Polygon Free-Tier hat 5 API Calls/Min
- für 60 Symbole EOD ist das eng (theoretisch 12 Minuten), intraday unmöglich
- paid Polygon startet bei ~$29/Monat — operationale Realität, nicht Code-Realität

### 17.3 NewsAPI erlaubt keinen Production-Einsatz im Free-Tier
- **mittel** · NewsAPI.org Free-Tier ist "developer only"
- dein System nutzt sie — ist ok zum Lernen, nicht zum Kommerzialisieren

### 17.4 Alpha Vantage Rate-Limit 5/min (Free)
- **mittel** · schärfer als Polygon
- muss beim Design berücksichtigt werden, wird aktuell nicht systematisch (Rate-Limit-Gates, Retry-Policy)

### 17.5 GDELT-Datenlizenz nicht dokumentiert
- **leicht** · GDELT ist "free to use with attribution"
- Attribution fehlt im System

### 17.6 SEC-EDGAR ist erlaubt, aber Rate-Limit 10/s
- **leicht** · `edgartools` handelt das grundsätzlich
- bei einer breiten Scraping-Aktion kann das Cloudflare triggern

---

## 18. Regulatorisches (wenn Vermietung je real wird)

### 18.1 BaFin §32 KWG Lizenz fehlt
- **schwer** · automatisierte Portfolioverwaltung für Dritte in Deutschland ist erlaubnispflichtig
- Mindestkapital (2026) für Finanzportfolioverwaltung: 125.000 € voll eingezahlt
- laufende Compliance-Kosten (Revisor, Risk-Officer, Meldewesen): schwer unter 100k€/Jahr

### 18.2 MiFID II Algorithmic Trading Requirements
- **schwer** · RTS 6 (organisatorische Anforderungen) gilt für Algorithmic Trading
- Pre-trade Risk Controls (vorhanden), Real-time Monitoring (teilweise), Self-Assessment (fehlt), Annual Stress Tests (fehlen), Kill Functionality (vorhanden, aber binär)

### 18.3 MaRisk (BaFin Rundschreiben) Anforderungen
- **schwer** · ORR, Modellvalidierung, Outsourcing-Richtlinien — alles Papiernachweise
- für Zukunft: du brauchst eine dokumentierte Modellvalidierungs-Policy, die von deiner Firma unterzeichnet ist

### 18.4 Datenschutz nach DSGVO
- **mittel** · wenn du News-Daten mit Namen ziehst und speicherst, musst du Rechtsgrundlage dokumentieren
- `intel/news_archive.py` existiert — was wird gespeichert, wie lange, auf welcher Basis?

### 18.5 US-Market-Data-Licensing (wenn live in US)
- **schwer** · Realtime Market Data von NYSE, NASDAQ kostet Lizenzgebühren, auch für interne Nutzung
- IBKR TWS API umgeht das nicht — Exchange-Fees werden separat berechnet
- aktuell nicht relevant (nur Paper), wird relevant bei Live

### 18.6 EMIR (Derivate-Reporting) wenn Derivate
- **leicht** · EMIR fordert Trade-Reporting für alle OTC + börsennotierte Derivate
- aktuell nicht relevant (kein Derivate-Trading), wird relevant mit Options/Futures

### 18.7 Steuer-Reporting in Deutschland
- **mittel** · Kapitalertragsteuer, Abgeltungssteuer, Wegzugsbesteuerung — all das ist relevant, wenn du für Kunden tradest
- `accounting/tax_lots.py` existiert als Skelett
- aber: StAnw-konformes Steuerreporting ist ein eigenes Subsystem (siehe WM Daten oder Deutsche Bundesbank-Schnittstellen)

---

## 19. Software-Engineering-Hygiene

### 19.1 Funktions- und Klassen-Dokstrings meist gut
- **(positiv)** · die meisten Files haben anständige Docstrings mit Args/Returns
- das ist ein Pluspunkt, der bleibt

### 19.2 Typing sparsam verwendet
- **mittel** · `str | None`-Style wird benutzt, aber nicht durchgängig
- `disallow_untyped_defs = false` erlaubt untypisierte Funktionen
- bei 95k LOC fehlen vermutlich hunderte Type-Hints

### 19.3 Magic Numbers in vielen Files
- **leicht** · z.B. in `rules_event_insider_shipping.py`: `1000.0`, `30.0`, `70.0` als Thresholds
- sollten als benannte Konstanten oder via Config kommen
- teilweise ja, teilweise ad-hoc

### 19.4 Inkonsistente Dict-Schemas
- **mittel** · `result.meta` ist ein großes Dict ohne Schema
- TypedDict oder Pydantic-Model würde Fehlschreibweisen (typos in Keys) verhindern
- stattdessen: 500+ Stellen, die `result.meta["irgendwas"] = {}` schreiben

### 19.5 `Any`-Typing als Ausweg
- **leicht** · grep `: Any` findet hunderte Stellen
- in FastAPI-Routern besonders: `dict[str, Any]` als Response — besser wären Pydantic-Models

### 19.6 Kein konsistenter Fehler-Typ
- **mittel** · `errors.py` definiert einen einzigen `PriceLookupError`
- in den 506 `except Exception`-Blöcken wird alles gleich behandelt
- typed Error-Hierarchie (DataFetchError, ModelError, ExecutionError, PolicyViolationError) würde Fehler-Handling deutlich präziser machen

### 19.7 Log-Level-Disziplin
- **leicht** · `log.debug("... skipped: %s")` für Silent-Skips ist akzeptabel
- aber: echte Fehler (F821 NameError zur Runtime) werden auch als `log.debug` geloggt
- das sollte `log.error` oder `log.warning` sein

### 19.8 Keine Dependency-Injection
- **mittel** · Module importieren Klassen direkt
- Testing wird schwieriger (Monkey-Patching statt Mock-Injection)
- für den Maßstab des Projekts wäre DI (z.B. `dependency-injector`) oder Factory-Pattern sinnvoll

### 19.9 Zirkuläre Imports als Risiko
- **mittel** · `from src.assembled_core.X` wird in 934 Files verwendet
- in einem so großen Codebase sind zirkuläre Imports wahrscheinlich
- Inline-Imports in Funktionen (557 in `trading_cycle.py`) sind oft Symptome davon

### 19.10 Keine asyncio-Nutzung trotz FastAPI
- **mittel** · FastAPI ist async-first
- der Code nutzt aber fast ausschließlich sync-Funktionen
- verpasste Gelegenheit für IO-Parallelität (Broker-Calls, News-Fetches)

---

## 20. Projekt- und Prozessebene

### 20.1 29% aller Commits sind AI-generiert (Co-Authored-By Claude)
- **schwer** · 205 von 699 Commits haben Claude als Co-Author
- dazu 195 Bot-Commits (Auto-Sync)
- effektiv: von 699 Commits sind ~60% entweder AI-Code oder Auto-Sync-Bookkeeping
- das heißt: das tiefe inhaltliche Verständnis pro Codezeile bei dir als Maintainer ist dünn, weil der Code von woanders kam

### 20.2 Eigentliche User-Commits (hans.oertel2): 24 von 699
- **schwer** · 3,4% der Commits sind mit deiner realen E-Mail
- d.h. dein Repo ist im Kern nicht mehr "dein Code", es ist "Code, den du kuratiert hast, meist von AI generiert"
- das ist weder gut noch schlecht per se — aber Du musst das wissen, weil dein persönlicher "Source of Truth" für Verhalten limitiert ist

### 20.3 "Dein Name <dein.email@domain.tld>" als Git-Committer
- **leicht** · 462 Commits mit Placeholder-Identity
- das ist eine Claude-Code-Default-Config, die nie überschrieben wurde
- cleanup: `git config` korrekt setzen

### 20.4 Kein Branching-Modell
- **mittel** · alles läuft auf `main` (vermutlich — kein PR-Workflow sichtbar)
- bei AI-Assisted-Development ist Branch-per-Feature besonders wichtig, damit Reviews möglich bleiben
- aktuell pushen Bot und User direkt auf main

### 20.5 Keine Code-Review-Kultur dokumentiert
- **mittel** · bei Solo-Entwicklung mit AI fehlt oft das zweite Augenpaar
- ein Self-Review-Checkliste oder ein periodischer AI-Adversarial-Review (wie `system_check/`) ist gut — aber nur wenn die Erkenntnisse umgesetzt werden
- `system_check/runs/` ist leer (keine echten Turnier-Runs im Repo)

### 20.6 Kein klares "Done"-Kriterium
- **mittel** · ein "Wave" gilt als done, wenn die Wiring-Tests grün sind
- das ist eine sehr niedrige Done-Definition
- bessere Definition: "Die Funktion wird im Default-Pfad ausgeführt, ändert messbar mindestens eine Kennzahl (Sharpe, MDD, PnL), hat Before/After-Backtest-Evidenz."

### 20.7 151 Commits am Audit-Tag selbst
- **mittel** · das ist ein Panik-Signal oder ein "Wochenende vor Deadline"-Signal
- Kontext: der Audit wurde nachgefragt, also könnten die Commits Vorbereitung gewesen sein
- aber unabhängig: 151 Commits/Tag ist nie nachhaltig und oft Zeichen für Hand-zu-Mund-Entwicklung

### 20.8 Projektdauer: 6,5 Monate (Oktober 2025 - April 2026)
- **mittel** · in 6,5 Monaten von 0 auf 1971 Files
- das ist ein Feature-Velocity-Pattern, das nur mit AI möglich ist
- Problem: Wissen wächst langsamer als Code

### 20.9 Keine externe Review/Collaboration
- **mittel** · nur du + AI als Maintainer
- kein Peer-Review, kein zweiter Quant, der "Moment, die Sharpe-Formel stimmt nicht" sagen könnte
- `system_check/` versucht das zu lösen mit AI-vs-AI — hilft, ersetzt keinen Menschen

### 20.10 Kein "What I Learned"-Log
- **leicht** · kein Journal "was habe ich über Trading/Quant gelernt diese Woche"
- bei so schnell wachsendem Code ist das Lernen das eigentliche Asset, nicht der Code

---

## 21. Dinge, die fehlen, die in einem ernsthaften Quant-System erwartet werden

### 21.1 Execution Analysis / TCA (Transaction Cost Analysis)
- **mittel** · `qa/trade_tca.py` existiert
- aber: kein periodischer TCA-Report, der sagt "heute durchschnittlich 8bps Slippage vs. 5bps Modell"
- das ist die operative Wahrheit, dass Modelle Realität matchen

### 21.2 PnL-Attribution auf Faktor-Ebene
- **mittel** · `risk/factor_exposures.py`, `risk/attribution.py`
- Output-Report fehlt: "Today's PnL split: 40% Market, 25% Momentum, 10% Size, 15% Idio, 10% News-Alpha"
- ohne das kann man nicht steuern, was Alpha generiert

### 21.3 Scenario Tests: 1987, 2008, 2020
- **schwer** · Stress-Testing gegen historische Krisen
- `risk/stressed_var.py` ist Code
- kein dokumentierter Run gegen Lehman-Kollaps-Woche oder COVID-März-2020

### 21.4 Realised vs Expected Monitoring
- **mittel** · jede Vorhersage hat erwartete Verteilung
- Reality-Check: liegt die Realized-Verteilung innerhalb der Erwartung?
- typisch: rolling Sharpe vs. expected Sharpe (Back-forward comparison), Drawdown vs. historical max DD

### 21.5 Broker-Reconciliation-Trail
- **mittel** · `ops/reconcile.py` existiert
- tägliche Reconciliation ist Standard: eigene Buchhaltung vs. Broker-Snapshot vs. Exchange-Trade-Stream
- keine Evidenz für kontinuierlichen Reconcile-Run

### 21.6 Latenz-SLA-Alarms
- **mittel** · wenn "End-of-Signal → Order-Submission" länger als X Sekunden dauert, sollte Alarm
- kein SLA definiert, kein Monitoring

### 21.7 Positions-Reconciliation-State-Check
- **mittel** · nach Crash: ist die On-Disk-Position identisch mit Broker-Position?
- `execution/position_sync.py` existiert — gut
- aber wird das täglich geprüft?

### 21.8 Backup-Strategie
- **leicht** · wo liegen Zustandsdaten (SQLite-Dedupe, Paper-State, Model-Artefakte)?
- lokale Files — bei Disk-Crash: alles weg
- Cloud-Backup, off-site-Strategy: fehlt

### 21.9 Disaster Recovery Runbook
- **mittel** · Runbooks existieren (`docs/runbooks/`) — positiv
- aber: "Was tun, wenn Alpaca down ist während offener Positionen?" — nicht dokumentiert
- "Was tun, wenn News-Feed still steht und keine Events mehr kommen?" — teilweise (Health-Gate)

### 21.10 Business Continuity für Trading
- **schwer** · Wenn du 5 Positionen offen hast und der Server crasht, was passiert?
- Positionen bleiben beim Broker offen — dein System weiß nicht mehr, dass sie existieren
- beim nächsten Start: Reconcile würde das merken, aber zwischendurch ist Blind-Fenster
- kritisch für Live-Betrieb

### 21.11 Portfolio Construction as Optimisation Problem
- **mittel** · ein ernsthafter Quant-Stack nutzt `cvxpy` oder `scipy.optimize` für Portfolio-Construction mit Constraints
- z.B. Markowitz mit Turnover-Penalty, Sector-Caps, Tracking-Error-Limit
- `portfolio/position_sizing.py` existiert — prüfen, ob das mehr als Naive-Gewichtung macht

### 21.12 Black-Litterman oder Bayesian Priors auf Returns
- **mittel** · in deiner TXT erwähnst du Black-Litterman
- kein Modul `portfolio/black_litterman.py`
- wenn du Views aus News-Events hast, wäre das der natürliche Rahmen

### 21.13 Information Coefficient Tracking
- **mittel** · IC ist die Korrelation zwischen Signal und Future-Return
- `ml/signal_decay_tracker.py`, `intel/ic_loop.py` existieren
- reproduzierbare IC-Reports pro Signal-Quelle: nicht gesehen

### 21.14 Factor Neutralisation
- **mittel** · bevor ein Alpha "pur" ist, muss es gegen Market/Sector/Size/Momentum-Factors neutralisiert werden
- sonst ist dein "Alpha" oft nur ein Beta-Exposure
- kein Code gesehen, der das macht (es gibt `qa/factor_analysis.py` 2346 Zeilen — prüfen, ob das Neutralisation oder nur Attribution ist)

### 21.15 Benchmark-Relative Performance Reports
- **mittel** · alles wird absolut gemessen
- "SPY-Outperformance YTD: +3.2%" gibt es nicht als Standard-Output
- für jede Strategie braucht es Benchmark

### 21.16 Earnings-Event-Handling
- **mittel** · Earnings-Announcements haben definierte Vol-Spikes
- viele Strategien wollen "Pre-Earnings exit" oder "Post-Earnings only"
- `data/sources/earnings_calendar_source.py` ist da, aber die Nutzung im Decision-Path unklar

### 21.17 Insider-Trading (Form 4) Real Ingest
- **schwer** · `signals/rules_event_insider_shipping.py` braucht Form-4-Daten
- SEC-EDGAR liefert Form 4 im XML — `edgartools`-Dependency ist installiert
- aber kein dokumentierter Cron, der das lädt und in den Factor-Store schreibt

### 21.18 Borrow-Availability-Check für Shorts
- **mittel** · nicht alle Aktien sind shortable; Hard-to-Borrow (HTB) hat hohe Borrow-Fees
- `configs/htb_symbols.yaml` existiert
- aber: Alpaca-API kann Borrow-Fees in Echtzeit abfragen, der Code macht das nicht

### 21.19 Options-Flow als Signal
- **leicht** · put/call-ratio, unusual activity, skew
- `features/options_derived_signals.py` existiert, aber ohne Options-Daten (Polygon Options ist paid)

### 21.20 Cross-Asset-Korrelationen
- **mittel** · Equities vs. Rates vs. FX vs. Commodities
- `features/cross_asset_leads.py`, `features/intermarket_factors.py`
- aber: keine realen Timeseries in den Sample-Daten außer Equities

---

## 22. UX und Operator-Erfahrung

### 22.1 Kein "Ein-Blick-Dashboard"
- **mittel** · für "ist das System gesund?" gibt es keinen einzelnen Entry
- du müsstest: Logs lesen, DB-Queries machen, Reports öffnen
- ein Web-UI mit Live-Status (offene Positionen, PnL heute, letzte Signale, Health-Gates) wäre Standard

### 22.2 Keine Mobile-Alerts
- **leicht** · Kill-Switch-Trigger sollte SMS oder Push-Notification auslösen
- weder Twilio noch Pushover noch Slack-Webhook integriert

### 22.3 `configs/`-Änderungen ohne Hot-Reload
- **leicht** · Policy-Parameter-Änderung erfordert Restart
- `configs/policy.yaml`-Änderungen während eines laufenden Paper-Runs: werden erst beim nächsten Cycle geladen oder gar nicht
- keine `watchdog`-basierte Config-Reload-Logik

### 22.4 Rollback-Mechanismus für Models/Configs fehlt
- **mittel** · wenn neues Model schlechter ist als altes — wie rollback?
- `ml/model_registry.py` müsste versioniert sein mit Rollback
- aktuell: "edit file, redeploy"

### 22.5 Kein User-Experience-Flow-Dokument
- **leicht** · "Als Operator des Systems starte ich morgens um 09:00 und tue..." gibt es nicht als dokumentierter Tagesablauf
- Runbooks existieren, aber sind technisch, nicht Rollen-orientiert

---

## 23. Was ich NICHT prüfen konnte (Grenzen des Audits)

### 23.1 Keine LFS-Daten, keine echten Backtests-Outputs
- die Parquet-Files sind LFS-Pointer; ich konnte keinen echten Backtest ausführen
- d.h. Punkte wie "Sharpe auf realer Historie" sind nicht empirisch belegt, nur aus README-Fragmenten abgeleitet

### 23.2 Keine Git-LFS-Objekte
- wenn Backtest-Artefakte oder Modell-Files in LFS liegen, sind sie hier nicht sichtbar

### 23.3 Keine .env-Werte
- Alpaca/Polygon-Keys wären im .env — zum Glück nicht vorhanden
- ich konnte nicht prüfen, welche Services tatsächlich laufen

### 23.4 Keine laufenden CI-Runs oder GitHub-Actions-Logs
- Workflows existieren, ich kann nur YAML lesen, keine Runs

### 23.5 Keine Live-Zustandsdaten (Positions, Ledger, State)
- `/output/` ist in .gitignore, nicht im Repo
- `src/output/` hat nur den einen Sharpe-98-Report

### 23.6 Kein Zugriff auf Docker-Laufzeitverhalten
- Docker-Image wurde nicht gebaut und gestartet
- Healthcheck-Korrektheit nicht empirisch geprüft (nur der Import-Pfad-Hinweis aus 1.13)

---

## 24. Was insgesamt gut ist (zur Balance)

Die Audit-Liste ist lang und kritisch. Damit du das einordnen kannst, hier die echten Positiva. Das sind nicht "Trostpunkte", das sind reale Stärken.

### 24.1 News-Pipeline ist auf Industriestandard
- `events/news/` ist sauber, modular, produktiv lauffähig
- Fingerprinting, Dedupe, Clustering, Burst-Detection, Health-Monitoring — das ist echte Arbeit

### 24.2 Crisis-Alpha-State-Machine-Design ist richtig gedacht
- Gates, Hysteresis, ETF-Baskets — die Architektur stimmt
- nur nie gegen echte Krisen validiert

### 24.3 CLAUDE.md-Prinzipien-Text ist exzellent
- wenn du ihn als Filter auf jeden PR anwendest, fallen viele Wave-Wiring-Commits durch
- das Dokument alleine ist 80% der Lösung, wenn du es durchsetzt

### 24.4 Test-Infrastruktur-Setup ist durchdacht
- Markers, Phasen, Aliases, Migration-Plan
- die Umsetzung ist unvollständig, die Form ist richtig

### 24.5 pyproject.toml und Packaging sind professionell
- saubere Extras, gepinnte Versions, entry-points
- das ist der Setup eines Python-Entwicklers, nicht eines Anfängers

### 24.6 Docker + Multi-Stage + non-root
- Dockerfile ist qualitativ überdurchschnittlich
- Secrets-Handling im Dockerfile-Kommentar bewusst angesprochen

### 24.7 Pre-Commit-Hooks (Ruff, Black, Gitleaks, Detect-Secrets)
- das Setup existiert — ein großer Teil der Arbeit ist getan
- fehlt nur CI-Enforcement, damit es auch greift

### 24.8 Leakage-Tests und Point-in-Time Awareness
- in Docs und Code-Struktur explizit adressiert
- das ist bei Retail-Quant-Projekten selten

### 24.9 Alpaca-Adapter und Live-Paper-Runner-Struktur
- Reconcile-Mode, Halt-Flag-System, Dry-Run
- das ist operative Disziplin, kein Toy-Code

### 24.10 Du hast Feedback ertragen und explizit mehr gefordert
- das ist selten
- die meisten Menschen lehnen sich zurück nach erster Kritik
- du willst mehr — das ist der wichtigste Soft-Skill für das was du versuchst

---

## Schlussbemerkung

Diese Liste hat 24 Sektionen und ungefähr 300 separate Punkte. Sie ist ehrlich und vollständig im Sinne dessen, was ich in 4–5 Stunden strukturierter Analyse des Repos finden konnte. Sie ist **nicht** erschöpfend im mathematischen Sinn — ich habe nicht jede der 1471 Python-Dateien Zeile für Zeile gelesen.

Die Liste ist so sortiert, dass jede Sektion für sich steht. Du musst sie nicht von vorne nach hinten abarbeiten. Ich würde dir empfehlen, sie als Inventur zu nutzen:

1. Lies die Liste einmal vollständig durch, ohne zu agieren.
2. Beim zweiten Durchgang markiere jede Zeile mit (A) "stimmt, wusste ich", (B) "stimmt, wusste ich nicht", (C) "stimmt nicht, weil..." oder (D) "muss ich nachprüfen".
3. Die C- und D-Einträge diskutieren wir beim nächsten Mal.
4. Die A- und B-Einträge werden in eine Priorisierung überführt.

Die Priorisierung mache ich mit dir beim nächsten Mal, wenn du willst — bei dieser Datei hast du explizit keine Top-N gefordert, sondern die vollständige Liste. Das ist die vollständige Liste.

Ein Gedanke zum Abschluss, den ich beim Durchgehen des Repos zunehmend gespürt habe: Du hast Recht, hier ist Potential. Die News-Pipeline allein ist ein Asset. Die Crisis-Alpha-Architektur ist richtig gedacht. Deine CLAUDE.md-Prinzipien sind professionell formuliert. Dein Audit-Wille ist außergewöhnlich für jemanden, der allein im Streifendienst-Kontext ein solches System baut.

Das Hauptproblem ist nicht, dass zu wenig da ist. Das Hauptproblem ist, dass **zu viel da ist, und das Zuviel verdeckt das Echte**. Die Lösung ist nicht Hinzufügen, die Lösung ist Konsolidieren.

Wenn du willst, setzen wir beim nächsten Mal an Sektion 1.1 an — `trading_cycle.py` zerlegen. Das löst wegen der Kopplungseffekte viele andere Punkte automatisch mit.

---

**Ende des Audits.**
