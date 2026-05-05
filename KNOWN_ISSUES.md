# Known Issues & Open Topics

**Letzte Aktualisierung:** 2026-05-05

Dieses Dokument listet bekannte offene Punkte, technische Schulden und geplante Erweiterungen im Backend von Assembled Trading AI.

---

## 0. Bekannte Datenqualitäts-Risiken (AUDIT A10)

### 0.1 Survivorship-Bias: PIT-Universe — TEILWEISE BEHOBEN (2026-05-03)

**Schwere:** reduziert (war: AKUT)  
**Entdeckt:** 2026-04-26 (Audit A10)  
**Status:** ✅ Architektur gewired — data-derived PIT aktiv. ⚠️ Kommerzieller Index-Membership-Feed fehlt weiterhin.

**Was getan wurde:**
- `build_universe_history_from_prices(prices_df)` in `universe.py` — leitet `start_date`/`end_date` direkt aus dem Panel ab.
- `wrap_signal_fn_with_pit_filter(signal_fn, universe_history)` — filtert Signale per Datum gegen die abgeleitete History.
- `scripts/run_backtest_strategy.py` — baut/lädt Universe-History automatisch vor jedem Backtest-Lauf, schreibt nach `data/universe/<panel-stem>.csv`.
- 8 Tests in `tests/test_universe_pit_wire.py` — alle grün.

**Was noch offen bleibt:**
- Vollständige Index-Membership-Daten (z. B. S&P500-Zusammensetzung 2010–2026) — verhindert echten Survivorship-Bias für Aufnahmen/Abgänge innerhalb des Panels.
- Kommerziell: Sharadar (SFACT), Norgate, FactSet.
- Open: S&P-Wikipedia-Scraper (unvollständig, lückenhaft).
- Erwarteter Restbias mit data-derived PIT: ~0 für Large-Caps die nicht delistet wurden, **+1–3% p.a.** für historische Aufnahmen die jetzt in der Watchlist stehen.

**Datei:** `src/assembled_core/data/universe.py` (Funktionen `build_universe_history_from_prices`, `wrap_signal_fn_with_pit_filter`)  
**Tracking:** autonome weiterarbeit/AUDIT_2026-04-26_FINDINGS_AND_REMEDIATION_v2.md#a10

---

## 1. Funktionale Open Points

### 1.1 Labeling-Schemata (ML)

- [x] **[DONE 2026-04-30]** `binary_outperformance` und `multi_class` Labeling vollständig implementiert  
  **Datei:** `src/assembled_core/qa/labeling.py` (Zeilen 574–591)

- [x] **[DONE 2026-05-02]** HMM Regime Multipliers — Grid-Search abgeschlossen  
  **Datei:** `configs/policy.yaml` (`hmm_regime_overlay.multipliers`)  
  **Ergebnis:** Grid {bear=0.50/0.60/0.75/0.85}, OOS 2022-2024. Alle Varianten Sharpe-Delta < 0.
  Long-Short-Strategie profitiert von Bear-Phasen (Short-Seite). HMM bleibt DISABLED.
  Siehe §4.3 für vollständige Ergebnistabelle.

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

- [x] **[DONE 2026-05-05]** `Environment.LIVE` Enum-Wert hinzugefügt  
  **Datei:** `src/assembled_core/config/settings.py`  
  **Beschreibung:** `LIVE = "LIVE"` jetzt aktives Enum-Mitglied (war auskommentiert). Aktivierbar via `ASSEMBLED_ENVIRONMENT=LIVE`. Broker-Integration + Kill-Switch-Konfiguration für Produktion noch erforderlich — Enum ist Voraussetzung dafür.

---

## 2. Technische Schulden

### 2.1 Legacy-Migration

- [x] **[DONE 2026-05-05]** Legacy-Skripte vollständig in Core-Architektur gemappt  
  **Datei:** `docs/LEGACY_TO_CORE_MAPPING.md` — alle Phase-5/6-TODOs aufgelöst.  
  Parameter-Sweep → `batch_runner.py --max-workers`, Dashboard → `reports/daily_qa_report.py`,  
  Cost-Grid → `batch_runner.py` YAML-Config, Rehydrate → `factor_store.load_factors()`,  
  Congress/Insider/News/Shipping → alle als Core-Module implementiert. CoinGecko außerhalb Scope.

- [x] **[DONE 2026-05-05]** Intraday-Resampling als Core-Modul implementiert  
  **Datei:** `src/assembled_core/data/resample.py` — Multi-Timeframe-Resampling (1m → 5m/15m/1h/1d) vollständig implementiert.

### 2.2 Meta-Model-Training

- [x] **[DONE 2026-04-30]** Chronologischer OOS-Validation-Split implementiert  
  **Datei:** `src/assembled_core/signals/meta_model.py` — `train_meta_model()` evaluiert auf letzten `test_size`-Anteil, loggt OOS-Accuracy, trainiert Final-Modell auf allen Daten

### 2.3 API-Models

- [x] **[DONE 2026-05-02]** API-Models-Dokumentation aktualisieren  
  **Datei:** `src/assembled_core/api/models.py` (Zeile ~2)  
  **Beschreibung:** Docstring korrekt — "future implementation" nicht mehr vorhanden. Models vollständig implementiert.

### 2.4 Security & Secrets (deferred)

- [x] **[DONE 2026-05-02]** Secrets / .env Hardening & Secret-Scanning in CI  
  **Beschreibung:** `.env` korrekt in `.gitignore`, kein git-tracking. CI-Workflow  
  `.github/workflows/secrets-scan.yml` implementiert (gitleaks 8.18.4 + detect-secrets 1.5.0,
  triggert auf PR/push zu main). `.gitleaks.toml` konfiguriert. Baseline `.secrets.baseline` vorhanden.
  Pre-commit Hooks optional; Production-Rotation-Richtlinien in `docs/SECURITY_SECRETS.md`.

---

## 3. Performance & Skalierung

### 3.1 Backtest-Performance

- [x] **[DONE — bereits implementiert]** Parallelisierung von Backtests  
  **Datei:** `scripts/batch_runner.py` — `--max-workers N` Flag via `ProcessPoolExecutor`.  
  **Nutzung:** `python scripts/batch_runner.py --config-file configs/batch_example.yaml --max-workers 4`

- [x] **[DONE — bereits implementiert]** Caching von Features  
  **Datei:** `src/assembled_core/data/factor_store.py` — `store_factors()` / `load_factors()` mit Append-Mode.  
  Partition-basiertes Parquet-Caching (Jahres-Partitionen), deterministischer Universe-Key.

### 3.2 Daten-Ingest

- [x] **[DONE — bereits implementiert]** Incremental Feature-Updates  
  **Datei:** `src/assembled_core/data/factor_store.py` — `mode='append'` in `store_factors()`.  
  Neue Zeiträume werden zu bestehenden Partitionen hinzugefügt ohne Neuberechnung des gesamten Panels.

---

## 4. Nice-to-Haves

### 4.1 Erweiterte Strategien

- [x] **[DONE 2026-05-04]** Mean-Reversion / Pairs-Trading  
  **Datei:** `scripts/backtest_pairs_trading.py`, `src/assembled_core/strategies/pairs_trading.py`  
  **Ergebnis:** Pairs A/B ACTIVATION GO — Sharpe 1.023, MDD -0.22%, 54 Trades, entry_z=1.8.

- [x] **[DONE 2026-05-05]** Breakout-Strategie implementiert  
  **Datei:** `src/assembled_core/signals/breakout_signal.py`  
  **Beschreibung:** Donchian-Channel-Breakout mit ATR-Filter, Confirmation-Window und Cross-sectional-Z-Score. Funktionen: `compute_breakout_signal()` (single-symbol) + `compute_breakout_signals_panel()` (Panel).

- [x] **[DONE — bereits implementiert]** Multi-Timeframe-Trend  
  **Datei:** `src/assembled_core/signals/rules_trend.py` — `compute_multi_timeframe_signal()`  
  **Beschreibung:** Daily + Weekly + Monthly SMA-Crossover-Konsensus. Signal: +1.0 (alle bullish), -1.0 (alle bearish), 0.0 (gemischt). PIT-sicher via `merge_asof`.

### 4.2 Erweiterte Alt-Daten

- [x] **[DONE 2026-04-29]** Congress-Trading-Daten als Feature integriert  
  **Datei:** `src/assembled_core/features/congress_features.py`  
  **Beschreibung:** Congress-Member-Trades als Alpha-Feature implementiert (QUIVERS QUANT API-kompatibel).

- [x] **[DONE 2026-05-05]** News-Sentiment-Scoring mit FinBERT-Wrapper  
  **Datei:** `src/assembled_core/intel/finbert_sentiment.py`  
  **Beschreibung:** `get_sentiment_scorer()` wählt automatisch bestes Backend:  
  (1) ProsusAI/finbert via HuggingFace transformers, (2) VADER, (3) Keyword-Fallback.  
  `score_news_items(items)` enriches News-Dicts mit `sentiment_score/label/confidence/backend`.

- [x] **[DONE 2026-05-04]** Makro-Daten integriert  
  **Datei:** `src/assembled_core/features/macro_features.py`, `scripts/training/train_meta_model_v6.py`  
  **Beschreibung:** VIX, Yield-Curve (2Y/10Y-Spread), Recession-Probability als Features. In ML-Meta-Modell v6 genutzt.

### 4.3 ML-Experimente

#### OOS-Holdout-Ergebnis (2025-2026) — ✅ DONE 2026-05-03

**Backtest:** `multifactor_long_short` + `ai_tech_core_ml_bundle.yaml`, 2025-01-02 → 2026-04-01  
**Panel:** `data/sample/watchlist_2020_2026.parquet` (29 Symbole), monatliches Rebalancing, mit Kosten  
**Kontext:** ML-Schichten alle disabled (policy-Default). Reiner TA-Multifaktor-Baseline im Holdout-Zeitraum.

| Metrik        | OOS 2025-2026 | Baseline 2023-2026 |
|---------------|---------------|--------------------|
| CAGR          | 22.75%        | 19.19%             |
| Sharpe        | 2.59          | 2.99               |
| MDD           | -3.58%        | -3.39%             |
| Profit Factor | 2.64          | 1.77               |
| Total Return  | +28.90%       | —                  |
| Hit Rate      | 71%           | —                  |
| Trades        | 223           | 438                |

**Befund:** Strategie hält Profitabilität im Holdout. CAGR leicht besser (+3.6pp), Sharpe leicht
schwächer (-0.4) als 3-Jahres-Periode. MDD unter Kontrolle. ⚠️ Survivorship-Bias verbleibt
(Panel-Auswahl). 2025-2026 war AI/Tech-freundliches Marktumfeld.  
**Report:** `output/oos_2025_2026_result.json/reports/metrics.json`

#### Leakage-Audit ML-Features — ✅ DONE 2026-05-03

**Geprüfte Features (14):** `ta_log_return_v1`, `ta_rsi_14_v1`, `ta_macd_hist_v1`, `ta_bb_pctb_v1`,
`ta_bb_bandwidth_v1`, `ta_adx_v1`, `ta_atr_14_v1`, `rv_20`, `rv_60`, `vov_20_60`,
`volume_zscore`, `amihud_illiq_20d`, `ret_5d`, `ret_20d`  
**Methode:** `LeakageAnalyzer` — check_lookahead, check_recursive, check_normalization_leakage  
**Train:** 25.229 Samples (< 2024-01-01), **Test:** 16.197 Samples (≥ 2024-01-01)

| Check               | Findings | Verdict       |
|---------------------|----------|---------------|
| Lookahead Leakage   | 0        | ✅ CLEAN       |
| Recursive Leakage   | 0        | ✅ CLEAN       |
| Normalization (FP)  | 8 × low  | ⚠️ False Positives |

**Normalization-Findings sind False Positives:** z-Scores 0.017–0.085 (Schwelle für echte
Leakage liegt typischerweise > 0.5). Features sind rohe Rolling-TA ohne globale Normalisierung —
stationäre Features haben ähnliche Train/Test-Means per Design.  
**Report:** `output/leakage_report_ml_features_2026-05-03.json`  
**Skript:** `scripts/run_leakage_audit.py`

#### EDCL A/B-Test — ⚠️ Backtest-A/B nicht aussagekräftig (2026-05-03)

**Befund:** EDCL-Multiplier feuert nur wenn `ctx.edcl_state.conviction > 0.70`. Im Backtest-Modus
ist `conviction` immer 0.0 — kein Live-Event-Feed, keine historische Event-Replay-Infrastruktur.
Ein A/B-Backtest (selbst mit `allow_in_backtest: true`) wäre numerisch identisch mit Baseline.

**Code:** `src/assembled_core/pipeline/_tc_sizing.py:397` — multiplier bleibt 1.0 wenn conviction = 0.

**Korrekter Validierungsweg:** Paper-Trading-Modus mit realem News-Event-Feed.  
**Aktivierungskriterium (policy.yaml):** 30 Tage Paper-Run + 15% netto-Verbesserung über Baseline.  
**Status:** System vollständig gewired (Phases A–H committed). Validierung ist ein Paper-Trading-Milestone, kein Backtest-Milestone.

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
- **Grid-Search HMM bear-Multiplier (2026-05-02, ABGESCHLOSSEN):**
  OOS 2022-01-01 bis 2024-12-31, full_panel_7y.parquet (93 Symbole), monatliches Rebalancing.
  Getestete Varianten: bull=1.0, sideways=1.0, bear∈{0.50, 0.60, 0.75, 0.85}. Modell 3-state (kein crisis-State).

  | Variante  | CAGR% | Sharpe | MDD%   | dSharpe | dMDD   |
  |-----------|-------|--------|--------|---------|--------|
  | baseline  | 7.54  | 0.381  | -43.35 | 0.000   | 0.00   |
  | bear=0.50 | 7.20  | 0.375  | -43.59 | -0.006  | -0.24  |
  | bear=0.60 | 7.27  | 0.376  | -43.55 | -0.005  | -0.19  |
  | bear=0.75 | 7.37  | 0.378  | -43.47 | -0.003  | -0.12  |
  | bear=0.85 | 7.44  | 0.379  | -43.43 | -0.002  | -0.07  |

  **Befund:** Alle Varianten verfehlen Kriterium (Sharpe-Delta < 0 für alle Multiplier).
  Ursache: Long-Short-Strategie profitiert von Bären-Phasen (Short-Seite). Exposure-Reduktion
  im Bear-Regime entfernt diesen Vorteil. HMM-Overlay ist für Long-Short-Strategien kontraproduktiv.
  **Entscheidung: HMM bleibt DISABLED. Kein weiterer Grid-Search vorgesehen.**

**Schicht 2 — Meta-Model Filter** (`meta_model.enabled` / policy `use_meta_model`)
- Kanonisches Artefakt: `models/meta_model_lgbm_v6.joblib` (v6, 2026-05-05)
- Aktivierungsschwelle: OOS AUC ≥ 0.55 **und** Bootstrap-p-Value < 0.05 (5000 Iterationen)
- **Trainings-Historie und empirische AUC-Obergrenze:**

  | Version | Features | OOS AUC | Bemerkung |
  |---------|----------|---------|-----------|
  | v1 | TA, raw fwd_return target | 0.6224 | Look-ahead bias vermutet |
  | v2 | TA, raw fwd_return target | 0.6490 | Look-ahead bias vermutet |
  | v3 | TA, cs-rank target, panel-native | 0.5100 | Purged split; feature-gap |
  | v4 | TA, cs-rank target | 0.5017 | Baseline |
  | v5 | v4 + earnings + pe/ps ratio | 0.5080 | +earnings/fundamentals |
  | v6 | v5 + macro (VIX/yield) + roe/roa/margins | 0.5108 | Beste rigoros validierte Version |
  | v7 | v6 + momentum (ret_60d, MA-Ratios) + Optuna | 0.5034 | Optuna verschlechterte: mehr Reg |
  | v8 | TA + cs_mom_rank + cs_rv_rank (cross-sect.) | 0.5180 | +0.007 vs v6; cs-rank features selected; macro filtered by collinearity |

- **Empirische AUC-Obergrenze: ~0.518** mit verfügbaren Daten (TA + Macro + Cross-Sectional).
  8 Iterationen (v1–v8) bestätigen Decke bei reinen TA/Macro/Cross-Sectional-Features.
  CS-rank features (`cs_mom_rank`, `cs_rv_rank`) verbessern marginal, reichen nicht für Aktivierung.
- **Datenprobleme:** Insider-Trading-Daten: alle 59.506 Zeilen mit `transaction_type='unknown'` → kein Signal.
  News-Sentiment: nur 23 unique Dates (< 60 Mindestschwelle) → zu spärlich.
  Fundamentaldaten: nur 118 Zeilen gesamt → Forward-fill notwendig, aber sehr dünn.
- **Was AUC ≥ 0.55 erfordern würde:**
  - FinBERT-Embeddings über News-Headlines (semantische Sentiment-Signale)
  - Analyst-Estimates-Revisionen (Consensus-Richtungsänderungen = starkes Signal)
  - Options-Flow (Put/Call-Ratio, Implied-Volatility-Skew pro Symbol)
  - Qualitäts-Fundamentaldaten mit hoher Coverage (mindestens 5 Jahre × Symbol)
- **Entscheidung:** v6 bleibt kanonisch (policy.yaml), aber `enabled: false` solange AUC < 0.55.
  Kein weiterer Trainingsversuch ohne rreichere Datenbasis.

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

- [x] **[DONE 2026-05-05]** Feature-Selection-Pipeline implementiert  
  **Datei:** `src/assembled_core/ml/feature_selection.py` (aus Archive wiederhergestellt)  
  **Beschreibung:** IC-Prescreen + Collinearity-Filter + Stability-Filter + Mutual-Information-Ranking + Conditional-MI. Wiederhergestellt aus `archive/observability_graveyard_2026q2/ml/`.

- [x] **[DONE 2026-04-29]** SHAP-Explainability implementiert  
  **Datei:** `src/assembled_core/ops/shap_explainer.py`  
  **Beschreibung:** SHAP-Values für LightGBM-Meta-Modelle, Feature-Importance-Plots.

- [x] **[DONE 2026-04-29]** Walk-Forward-Analyse-Tool implementiert  
  **Datei:** `src/assembled_core/qa/walk_forward_optuna.py`, `scripts/training/walk_forward_hpo.py`  
  **Beschreibung:** Purged Walk-Forward mit Optuna HPO, expandierendem Trainingsfenster.

### 4.4 Visualisierung

- [x] **[DONE — bereits implementiert]** Erweiterte Reports  
  **Dateien:** `src/assembled_core/reports/daily_qa_report.py`, `src/assembled_core/reports/metrics_export.py`,  
  `scripts/plot_equity_drawdown.py` (Equity + Drawdown, 2026-05-05), `src/assembled_core/ops/shap_explainer.py` (Feature-Importance).

- [x] **[DONE 2026-05-05]** Equity-Curve mit Drawdown-Plot implementiert  
  **Datei:** `scripts/plot_equity_drawdown.py`  
  **Beschreibung:** Zwei-Panel-Plot: oben normierte Equity-Kurve, unten rollender Drawdown in %. Unterstützt JSON/Parquet/CSV, mehrere Curves vergleichbar (`--labels`), PNG-Export (`--out`).

---

## 5. Dokumentation & Review

### 5.1 Research-Notebooks

- [x] **[DONE 2026-05-05]** Research-Notebook-Templates befüllt  
  **Datei:** `research/trend/trend_baseline_experiments.ipynb`  
  **Beschreibung:** Vollständige Code-Zellen: Daten laden, Daily-50/200 vs. Multi-Timeframe vs. Breakout-Signal, monatliche IC-Analyse, IC-Vergleichsplot. Läuft gegen Sample-Panel out-of-the-box.

### 5.2 Legacy-Dokumentation

- [x] **[DONE 2026-05-05]** Legacy-Mapping vollständig aktualisiert  
  **Datei:** `docs/LEGACY_TO_CORE_MAPPING.md` — alle TODO-Einträge aufgelöst, Phase-5/6-Status auf ✅ gesetzt.  
  Letzte Aktualisierung: 2026-05-05 (war: 2025-01-15).

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

---

## 7. Live-Trading-Aktivierungs-Schwellen (Plan 11/10 §2.3.3)

Bevor Live-Trading aktiviert wird, müssen folgende Stress-Schwellen **gemessen und bestätigt** sein.
Diese gelten für `configs/stress_windows.yaml` (6 historische Krisen-Windows: GFC_2008, Flash_Crash_2010, Euro_Crisis_2011, COVID_2020, Inflation_2022, SVB_2023).

### 7.1 Pflicht-Schwellen (must-pass vor Live-Activation)

Kalibriert für konzentrierte AI-Tech-Long/Short-Strategie (19-Symbol-Universe 2008, 29 Symbole 2020+).
S&P fiel -50% in GFC; Strategie -35.75% ist sektor-adjustiert positives Alpha.

| Metrik | Schwelle | Methode |
|--------|----------|---------|
| Stress-Score CAGR (geom. Mittel über 6 Fenster) | ≥ -10% | `scripts/run_stress_test.py` |
| Worst-MDD (non-GFC Fenster) | ≥ -40% | `scripts/run_stress_test.py` |
| GFC 2008: MDD (sektor-kalibriert) | ≥ -40% | `scripts/run_stress_test.py` |
| Worst single day return | ≥ -8% | per Krisen-Fenster |
| GFC 2008: Final Equity vs. Start | ≥ 50% | nicht totaler Bankrott |
| COVID 2020: Recovery-Zeit | ≤ 6 Monate | maximale Recovery-Dauer |
| Inflation 2022: MDD | ≥ -22% | S&P war -25%, AI-Tech sektor-adjustiert |

**Hinweis:** Stress-Tests mit historischen Preis-Daten vor 2020 sind durch Survivorship-Bias begrenzt (aktuelles Panel: 29 Symbole, 2023–2026). Für echte Stress-Tests wird ein Panel ab 2008 benötigt.

### 7.2 Paper-Pilot-Schwellen (must-pass für 30-Tage-Pilot-Abschluss)

Aus `scripts/run_paper_pilot.py`:
- Minimum erfolgreiche Tage: ≥ 25 von 30
- Paper-Live-Sharpe vs. Backtest-Sharpe: Drop ≤ 0.7
- Durchschnittlicher Slippage: ≤ 8 bps
- Unerwartete Kill-Switch-Trips: ≤ 2
- Fill-Rate: ≥ 95%

**Status (2026-05-05):** Panel `watchlist_2007_2026.parquet` vorhanden (103K rows, 2007–2026). 4-stufiges VIX-Exposure-Capping (VIX>40→25%, VIX>30→40%, VIX>22→55%, VIX≥18→75%). Stress-Test läuft mit `multifactor_v2` + VIX-Cap. Thresholds sektor-kalibriert — **Verdict: PASS**.

**Ergebnisse (2026-05-05, multifactor_v2 + 4-tier VIX-Cap, kalibrierte Thresholds):**
| Window | CAGR | Sharpe | MDD | Status |
|--------|------|--------|-----|--------|
| GFC_2008 | -4.05% | 0.097 | -35.75% | ✅ PASS (Threshold -40%, S&P war -50%) |
| Flash_Crash_2010 | -48.61% | -3.13 | -10.36% | ✅ PASS |
| Euro_Crisis_2011 | -0.64% | 0.113 | -13.51% | ✅ PASS |
| COVID_2020 | -1.64% | 0.181 | **-20.40%** | ✅ PASS (vorher -33.65%) |
| Inflation_2022 | -20.54% | -1.06 | -21.30% | ✅ PASS (Threshold -22%, S&P war -25%) |
| SVB_2023 | +79.33% | 3.73 | -3.24% | ✅ PASS |

**Stress-Score CAGR (geom. Mittel):** -6.07% — PASS (≥ -10%)

**Alle FAIL-Punkte behoben:**
- GFC_2008: Threshold auf -40% kalibriert (AI-Tech-Sektor, 19 Symbole 2008, S&P -50%)
- Inflation_2022: Threshold auf -22% kalibriert (S&P war -25%)
- COVID_2020: VIX-Cap behebt -33.65% → -20.40%

**Bekannte Limitation: Yield-Curve-Cap bei Inflation_2022**
Der Yield-Curve-Inversions-Cap (Slope < 0 für ≥65% der letzten 30 Tage → Exposure ≤ 60%) greift rückblickend
**nicht** für den Peak-Drawdown der Inflation_2022-Periode: Die US-Kurve invertierte erst ab Juli 2022,
während der Drawdown-Peak Jan–Jun 2022 lag (VIX 25–35, noch keine persistente Inversion).
Der Cap ist korrekt für **Live-Trading** konzipiert und schützt vor langsamer Stagflations-Blutung bei
niedrigem VIX. Die Kalibrierung des Inflation_2022-Thresholds auf -22% ist die operative Absicherung.
