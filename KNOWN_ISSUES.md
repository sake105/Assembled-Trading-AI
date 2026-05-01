# Known Issues & Open Topics

**Letzte Aktualisierung:** 2026-04-30

Dieses Dokument listet bekannte offene Punkte, technische Schulden und geplante Erweiterungen im Backend von Assembled Trading AI.

---

## 0. Bekannte Datenqualitäts-Risiken (AUDIT A10)

### 0.1 Survivorship-Bias: PIT-Universe nicht aktiviert

**Schwere:** AKUT für Backtests mit Mid-Cap / historischem Universum  
**Entdeckt:** 2026-04-26 (Audit A10)  
**Status:** API implementiert, aber KEIN produktiver Aufrufer — statische watchlist.txt in Betrieb

**Problem:** `data/universe.py::get_universe_members_pit(as_of)` existiert mit vollständiger PIT-Logik.  
Kein Script oder Pipeline-Schritt ruft sie produktiv auf. Stattdessen kommt das Universum aus der  
statischen `watchlist.txt`, die nur *aktuelle* Symbole enthält.

**Konsequenz:** Backtests 2015–2024 mit heutiger Watchlist kennen Symbole, die damals noch nicht
handelbar waren (TSLA vor 2010 als Penny-Stock; Zwischenzeit-Delistings usw.).
Erwarteter Bias: +1–2% p.a. bei US Large-Caps, **+5–10% p.a.** bei Mid-Caps.

**Voraussetzung für echten Fix:** Historische Mitgliedschaftsdaten beschaffen:
- Kommerziell: Sharadar (SFACT), Norgate, FactSet
- Open: S&P-Zusammensetzungs-CSVs via Wikipedia-Scraper (unvollständig)

**Datei:** `src/assembled_core/data/universe.py:144` (Funktion existiert, unverkabelt)  
**Tracking:** autonome weiterarbeit/AUDIT_2026-04-26_FINDINGS_AND_REMEDIATION_v2.md#a10

---

## 1. Funktionale Open Points

### 1.1 Labeling-Schemata (ML)

- [x] **[DONE 2026-04-30]** `binary_outperformance` und `multi_class` Labeling vollständig implementiert  
  **Datei:** `src/assembled_core/qa/labeling.py` (Zeilen 574–591)

- [ ] **[open]** HMM Regime Multipliers — hand-tuned, nicht kalibriert  
  **Datei:** `configs/policy.yaml` (`hmm_regime_overlay.multipliers`)  
  **Beschreibung:** bull=1.15, sideways=1.00, bear=0.75, crisis=0.40 sind Schätzwerte.
  Grid-Search auf OOS 2020–2024 ausstehend. Risiko: Whipsaw bei häufigen Regime-Switches.
  Siehe §4.3 für Aktivierungskriterien.

### 1.2 Trade-Level-Metriken

- [x] **[DONE 2026-04-30]** `hit_rate`, `profit_factor`, `avg_win`, `avg_loss` implementiert  
  **Datei:** `src/assembled_core/qa/metrics.py` (`compute_hit_rate_and_profit_factor`, Zeile 1030)

### 1.3 Pre-Trade-Checks

- [x] **[DONE 2026-04-30]** Weight/Sector/Region-Exposure-Checks implementiert  
  **Datei:** `src/assembled_core/execution/pre_trade_checks.py` — `_ptc_check_max_weight()` aktiv

### 1.4 Monitoring-API

- [x] **[DONE 2026-04-30]** Drift-Persistierung implementiert  
  **Datei:** `src/assembled_core/qa/drift_detection.py` — `save_drift_results()` schreibt `output/drift_analysis_{freq}.parquet`; API liest daraus

### 1.5 Backtest: Monatlicher Rebalance-Modus — BEHOBEN (3478948)

- [x] **[FIXED 2026-05-02]** Zwei kombinierte Bugs ließen ~6/63 monatliche Rebalance-Dates leer:
  1. `_is_rebalance_date()` prüfte `timestamp.day == 1` (Kalender-Tag) statt ersten Handelstag.
     Fix: Month-boundary-Erkennung aus der tatsächlichen Timestamp-Serie der Preisdaten.
  2. `backtest_use_snapshot` triggerte nur bei `--rebalance monthly`, nicht `--rebalance-freq M`.
     Fix: `rebalance_freq in ("M", "W")` löst jetzt ebenfalls Snapshot-Modus aus.
  **Betroffene Monate:** Jun/Sep/Nov 2025, Jan/Feb/Mar 2026 (1. auf Wochenende/Feiertag).  
  **Commit:** `3478948` — 113 Tests bestanden.

### 1.6 Live-Trading-Mode

- [ ] **[enhancement]** Live-Trading-Mode (Environment.LIVE)  
  **Datei:** `src/assembled_core/config/settings.py` (Zeile ~28)  
  **Beschreibung:** Live-Trading-Mode ist als Kommentar markiert ("Future: Live trading mode (not yet implemented)").

---

## 2. Technische Schulden

### 2.1 Legacy-Migration

- [ ] **[tech-debt]** Legacy-Skripte migrieren/bereinigen  
  **Dateien:** `docs/LEGACY_OVERVIEW.md`, `docs/LEGACY_TO_CORE_MAPPING.md`  
  **Beschreibung:** Viele Legacy-Skripte (z.B. `sprint9_dashboard.ps1`, `sprint9_cost_grid.ps1`, `sprint10_param_sweep.ps1`) sind noch vorhanden, aber nicht in die neue Core-Architektur migriert. Status: "TODO: Phase 5/6".

- [ ] **[tech-debt]** Intraday-Resampling in Core-Architektur integrieren  
  **Datei:** `docs/LEGACY_TO_CORE_MAPPING.md` (Zeile ~23)  
  **Beschreibung:** Resampling 1m → 5m ist in Legacy-Skripten vorhanden (`scripts/50_resample_intraday.ps1`), aber noch nicht als Core-Modul (`src/assembled_core/data/resample.py`) implementiert.

### 2.2 Meta-Model-Training

- [x] **[DONE 2026-04-30]** Chronologischer OOS-Validation-Split implementiert  
  **Datei:** `src/assembled_core/signals/meta_model.py` — `train_meta_model()` evaluiert auf letzten `test_size`-Anteil, loggt OOS-Accuracy, trainiert Final-Modell auf allen Daten

### 2.3 API-Models

- [ ] **[tech-debt]** API-Models-Dokumentation aktualisieren  
  **Datei:** `src/assembled_core/api/models.py` (Zeile ~2)  
  **Beschreibung:** Docstring erwähnt "future implementation", aber Models sind bereits implementiert.

### 2.4 Security & Secrets (deferred)

- [ ] **[security]** Secrets / .env Hardening & Secret-Scanning in CI  
  **Beschreibung:** Aktuell existiert kein konsistenter Hardening- und Scan-Prozess für Secrets/.env-Dateien.  
  TODO (deferred):  
  - Secret-Management-Konzept definieren (Trennung Code/Config/Secrets).  
  - Secret-Scanning in CI aktivieren (z.B. GitHub-Scanner, Pre-Commit-Hooks).  
  - Richtlinien zur Ablage/Rotation von Credentials dokumentieren.  
  **Hinweis:** Umsetzung erfolgt in einem späteren Sprint; in M0-B wird lediglich diese TODO-Notiz gepflegt.

---

## 3. Performance & Skalierung

### 3.1 Backtest-Performance

- [ ] **[enhancement]** Parallelisierung von Backtests  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.4)  
  **Beschreibung:** Für größere Datensätze oder Parameter-Sweeps wäre Parallelisierung (Multi-Processing) wünschenswert.

- [ ] **[enhancement]** Caching von Features  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.4)  
  **Beschreibung:** Feature-Berechnungen könnten gecacht werden, um wiederholte Berechnungen zu vermeiden.

### 3.2 Daten-Ingest

- [ ] **[enhancement]** Incremental-Backtests  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.4)  
  **Beschreibung:** Nur neue Daten verarbeiten, statt vollständigen Backtest neu zu starten.

---

## 4. Nice-to-Haves

### 4.1 Erweiterte Strategien

- [ ] **[enhancement]** Mean-Reversion-Strategien  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.1)  
  **Beschreibung:** RSI-basierte Mean-Reversion, Bollinger-Band-Mean-Reversion, Pairs-Trading.

- [ ] **[enhancement]** Breakout-Strategien  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.1)  
  **Beschreibung:** Bollinger-Band-Breakouts, Support/Resistance-Breakouts.

- [ ] **[enhancement]** Multi-Timeframe-Trend  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.1)  
  **Beschreibung:** Kombination von 1d- und 5min-Trend-Signalen.

### 4.2 Erweiterte Alt-Daten

- [ ] **[enhancement]** Congress-Trading-Daten  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.2)  
  **Beschreibung:** Congress-Member-Trades als Feature integrieren.

- [ ] **[enhancement]** News-Sentiment-Scoring  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.2)  
  **Beschreibung:** FinBERT oder ähnliches für News-Sentiment verwenden.

- [ ] **[enhancement]** Makro-Daten  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.2)  
  **Beschreibung:** Economic-Indicators (CPI, Unemployment), Fed-Announcements.

### 4.3 ML-Experimente

#### ML-Aktivierungskriterien (Stand 2026-05-01)

Alle drei ML-Schichten sind policy-gated und per Default **disabled**. Die folgenden
Schwellen müssen erfüllt sein, bevor eine Schicht aktiviert wird:

**Schicht 1 — HMM Regime Overlay** (`hmm_regime_overlay.enabled`)
- Artefakt: `models/regime_hmm_4state_spy.joblib` (retrained 2026-05-02: 3-state, diag cov, StandardScaler)
- Multipliers (bull/sideways/bear/crisis): aktuell hand-tuned — **nicht kalibriert**
- Aktivierungsschwelle: Sharpe-Differenz >= 0.0 (nicht-negativ), MDD-Zunahme <= 1.5pp
- A/B-Backtest 2023-01-01 bis 2026-04-15 (monatliches Rebalancing):

  | Metrik  | Baseline | HMM v3 | Delta    |
  |---------|----------|--------|----------|
  | CAGR    | 18.68%   | 20.09% | +1.41pp  |
  | Sharpe  | 1.678    | 1.673  | -0.005   |
  | MDD     | -13.30%  | -14.22%| -0.92pp  |
  | PF      | 1.749    | 1.819  | +3.9%    |

- **Befund HMM v3:** Sharpe-Delta = -0.005 < 0 → Aktivierungskriterium **nicht erfüllt**.
  CAGR-Gewinn (+1.41pp) kommt weitgehend aus Bull-Regime-Leverage (1.15x), kein echter Regime-Edge.
  Bugs behoben (2026-05-02): 4-state-Modell hatte degenerate Konvergenz (3 States bei extremen Means);
  `parents[4]` → `parents[3]` path bug verhinderte Modell-Loading. Beide fixes committed.
- Kalibrierung ausstehend: Bull-Multiplier auf 1.0 setzen (nur bear/crisis als echte Regime-Signale
  nutzen); Grid-Search auf Multiplier-Werte {0.5, 0.6, 0.75, 0.85} × {bear, crisis} auf OOS 2020–2024.

**Schicht 2 — Meta-Model Filter** (`meta_model.enabled` / policy `use_meta_model`)
- Artefakt: `models/meta_model_lgbm_v4.joblib` (aktuell kanonisch, v4 = cs-rank target)
- OOS AUC v1–v4: alle zwischen 0.50 und 0.51 — **near-random mit TA-Features alleine**
- Aktivierungsschwelle: OOS AUC ≥ 0.55 **und** Bootstrap-p-Value < 0.05 (5000 Iterationen)
- Voraussetzung: News/Earnings/Macro-Features im Feature-Set (TA-Features alleine
  reichen nicht — durch 4 Iterationen empirisch belegt).
- Nächster Schritt: `events/news/` Pipeline als Feature-Source integrieren.

**Schicht 3 — Conformal Quantile Sizing** (`conformal.enabled`)
- Artefakt: `models/conformal_position_v2.joblib` (v2 = q05/q95, 87% Coverage)
- OOS Coverage: 87.0 % (Ziel 85–92 %) — Modell deployment-fähig, Feature-Alignment offen
- A/B-Backtest 2021-01-04→2026-04-10 (Schicht 3 only, Schichten 1+2 disabled, monatliches Rebalancing):

  | Metrik       | Baseline  | Conformal  | Delta    |
  |--------------|-----------|------------|----------|
  | CAGR         | 12.04%    | 4.16%      | -7.88pp  |
  | Sharpe       | 1.726     | 1.528      | -0.198   |
  | MDD          | -10.74%   | -3.49%     | +7.25pp  |
  | Trades       | 840       | 840        | 0        |

- **Befund:** Overlay funktioniert (MDD -7.3pp). Return-Einbruch erklärt durch Feature-Gap:
  v2-Modell wurde auf Kurznamen trainiert (`rsi_14`, `vol_20d`, …), Panel liefert Präfixnamen
  (`ta_rsi_14_v1`, `rv_20`, …). 7/13 Features konnten gemappt werden; 6/13 werden zero-gefüllt
  → Intervalbreiten systematisch zu weit → Multiplikatoren klemmen bei 0.25 (Minimum).
- **Bug-Fixes committed (e10348e):** `parents[4]` → `parents[3]` (falscher Repo-Root-Pfad),
  `target_weight` + `target_qty` beide skaliert (vorher nur `target_pct` geprüft → kein Effekt).
- **v3 (panel-native names, runtime-median anchor) — A/B-Backtest 2023-01-03→2026-04-15
  (post-Rebalance-Fix, commit 3478948, monatliches Rebalancing, Kosten an):**

  | Metrik   | Baseline  | Conformal v3 | Delta     |
  |----------|-----------|--------------|-----------|
  | CAGR     | 19.19%    | 15.20%       | -3.99pp   |
  | Sharpe   | 2.988     | 2.813        | -0.175    |
  | MDD      | -3.39%    | -3.32%       | +0.07pp   |
  | PF       | 1.7739    | 1.5873       | -10.5%    |
  | Trades   | 438       | 438          | 0         |

- **Befund v3:** Feature-Gap behoben (alle 13 Features resolven). Runtime-Median-Anchor
  reduziert Verteilungsshift (Train-Median 0.183 vs. Test-Median 0.306). Dennoch:
  Sharpe-Schaden (-0.175) überschreitet Schwelle (-0.1), MDD-Verbesserung (+0.07pp)
  unterschreitet Mindest-Reduktion (1.0pp). **K4-Aktivierungskriterium nicht erfüllt.**
- **Aktivierungsschwelle:** *nicht erfüllt.* Nächste Option: breitere Alpha-Features
  (Fundamentaldaten, News-Sentiment) trainieren, die MDD-relevante Risiko-Signale
  vom CAGR-generierenden Alpha trennen.

#### Offene Verbesserungen

- [ ] **[enhancement]** Feature-Selection-Pipeline  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.3)  
  **Beschreibung:** Automatische Feature-Selection für Meta-Modelle (Univariate, RFE, L1-Regularization).

- [ ] **[enhancement]** SHAP-Explainability  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.3)  
  **Beschreibung:** SHAP-Values für Meta-Modelle berechnen und visualisieren.

- [ ] **[enhancement]** Walk-Forward-Analyse-Tool  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.4)  
  **Beschreibung:** Walk-Forward-Analyse-Tool für robuste Validierung.

### 4.4 Visualisierung

- [ ] **[enhancement]** Erweiterte Reports  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.4)  
  **Beschreibung:** Strategy-Comparison-Reports, Regime-Analysis-Reports, Feature-Importance-Reports.

- [ ] **[enhancement]** Bessere Visualisierung  
  **Dokumentation:** `docs/RESEARCH_ROADMAP.md` (Sektion 3.4)  
  **Beschreibung:** Equity-Curve-Plots mit Drawdowns, Trade-Distribution-Plots, Feature-Correlation-Matrix.

---

## 5. Dokumentation & Review

### 5.1 Research-Notebooks

- [ ] **[enhancement]** Research-Notebook-Templates ausfüllen  
  **Dateien:** `research/trend/trend_baseline_experiments.ipynb`, `research/meta/meta_model_calibration.ipynb`, etc.  
  **Beschreibung:** Notebook-Templates enthalten TODOs und müssen mit konkreten Experimenten gefüllt werden.

### 5.2 Legacy-Dokumentation

- [ ] **[tech-debt]** Legacy-Mapping vervollständigen  
  **Datei:** `docs/LEGACY_TO_CORE_MAPPING.md`  
  **Beschreibung:** Viele Einträge sind noch als "TODO" markiert und müssen ausgefüllt werden.

---

## Priorisierung

**Hoch (sollte bald angegangen werden):**
- Trade-Level-Metriken (Position-Tracking)
- `binary_outperformance` Labeling
- Legacy-Migration (wenn Legacy-Skripte noch aktiv genutzt werden)

**Mittel (wichtig, aber nicht kritisch):**
- Pre-Trade-Checks (Weight, Sector, Region)
- Persistierte Drift-Analyse
- Validation-Split für Meta-Modelle

**Niedrig (Nice-to-Have):**
- Erweiterte Strategien (Mean-Reversion, Breakout)
- Erweiterte Alt-Daten (Congress, News-Sentiment)
- Performance-Optimierungen (Parallelisierung, Caching)

---

**Hinweis:** Dieses Dokument wird regelmäßig aktualisiert. Neue Issues sollten hier eingetragen werden, bevor sie in GitHub Issues erstellt werden.


---

## 5. trading_cycle.py Migration Audit (2026-04-26)

**Status:** Phase 0 abgeschlossen — Phase 3+4 bereit

### Audit-Ergebnis

| Metrik | Wert |
|---|---|
| Zeilen trading_cycle.py | 9.141 |
| Imports gesamt (intern) | 516 |
| OK (Datei existiert) | 331 |
| Dead total | 185 |
| davon ARCHIVED | 151 |
| davon MISSING | 34 |
| in try/except (safe to delete) | 184 |
| außerhalb try/except | 1 (`news_triggers_loader`, bereits archiviert) |

**Coverage-Befund:** `trading_cycle.py` wird in der gesamten phase12-Suite **nie importiert** — 100% toter Code im Testlauf.

### Phase 2 — Implementiert als neue Funktionen in trading_cycle_v2 (2026-04-28)

Entgegen dem früheren Audit-Befund (reine Dummy-Blöcke) wurden die drei Phase-2-Funktionen als
echte Implementierungen in `trading_cycle_v2.py` neu geschrieben:

- **Phase 2a** (`_apply_evidence_gate`): Filtert News-Signale nach Evidence-Grade (T1/T2/T3 Quellen → grade A/B/C/D); policy-key: `evidence_gate.enabled + require_grade`. 8 Tests in `test_evidence_gate_v2.py`.
- **Phase 2b** (`_compute_news_triggers`): Verarbeitet News-Events → actionable Trigger-DataFrame; Pipeline: simhash-Dedupe → TF-IDF-Clustering → Burst-Bonus → Tier-Scoring. 9 Tests in `test_news_triggers_pipeline.py`.
- **Phase 2c** (IC-Weights via `news_ml_bridge.get_event_type_ic_weights`): **Nicht migriert** — Modul liegt in `archive/observability_graveyard_2026q2/ml/news_ml_bridge.py`. Bewusstes Backlog-Item: braucht historische IC-Daten für sinnvolle Kalibrierung.

### Aktueller Stand (2026-04-28)

- `trading_cycle.py`: 62 Zeilen — Phasen 3+4 sind effektiv abgeschlossen.
- `trading_cycle_v2.py`: Primärer Pfad; enthält alle aktiven Phasen + Phase 2a+2b.

### Phase 4 — Archiviert (2026-04-29)

- `src/assembled_core/pipeline/trading_cycle.py` (62-Zeilen-Shim) nach `archive/pipeline_legacy_2026q2/trading_cycle.py` verschoben (`git mv`).
- Import-Sweep bestätigt: 0 verbleibende `trading_cycle`-Imports außerhalb der kanonischen Module.
- `filterwarnings = ["error::DeprecationWarning:src.assembled_core.pipeline.*"]` aktiv in `pyproject.toml`.
- `trading_cycle_v2.py` → Umbenennung zu `trading_cycle.py` bewusst verschoben: 24 Stellen importieren `trading_cycle_v2`; Rename bringt keinen Funktionsgewinn und erzeugt großen Diff.

**Abgeschlossen:** Alle 4 Migrationsphasen für `trading_cycle.py → trading_cycle_v2.py` sind done.

---

## 6. Strategische Deferred Items (2026-04-29)

Bewusst vertagt. Hier tracken für spätere Planung.

### 6.1 PIT-Universe Wiring (vollständig)

**Status:** `get_universe_members_pit(as_of)` implementiert, aber kein produktiver Aufrufer (Details: §0.1)  
**Action:** PIT-Funktion in Pipeline-/Backtest-Universe-Selektion verdrahten  
**Prerequisite:** Historische Mitgliedschaftsdaten (Sharadar, Norgate, oder Open-S&P-CSVs)

### 6.2 regime_analysis.py — TODOs

- [x] **[DONE 2026-04-30]** Alle 6 TODOs implementiert  
  **Datei:** `src/assembled_core/risk/regime_analysis.py`  
  - `win_rate` = Anteil positiver Renditetage im Regime  
  - `avg_trade_duration` = BUY/SELL-Pairing per Symbol  
  - `avg_profit_per_trade` = BUY/SELL-P&L per Round-Trip  
  - `factor_ic_mean` = Correlation factor→next-period-return per Regime  
  - `compute_regime_transitions()` = vollständige Transitionsmatrix mit Wahrscheinlichkeiten

### 6.3 Dead-Module-Audit

**Status:** Import-basierter Grep (2026-04-30) zeigt 311 "unreferenced" Module — fast alle false positives.  
API-Routers (FastAPI dynamic registration), Feature-Module (config-driven), Data-Sources (factory) erscheinen als "unreferenced", sind aber aktiv.  
**Echter Handlungsbedarf:** Klein — kein breiter Audit nötig. Bei konkretem Verdacht einzelne Module prüfen.

### 6.4 trading_cycle_v2.py Package-Split

**Datei:** `src/assembled_core/pipeline/trading_cycle_v2.py`  
**Stand:** ~2673 LOC (2026-04-29); wächst mit jeder Wiring-Welle weiter  
**Action:** In 5–6 fokussierte Module aufteilen (z.B. steps, risk, execution, features, signals)  
**Risiko bei weiterem Aufschub:** Datei überschreitet Wartbarkeitsschwelle; Diffs werden unlesbar
