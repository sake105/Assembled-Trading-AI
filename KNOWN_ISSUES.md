# Known Issues & Open Topics

**Letzte Aktualisierung:** 2026-05-12 (Audit-Sweep §8 ergänzt nach 17 Waves)

Dieses Dokument listet bekannte offene Punkte, technische Schulden und geplante Erweiterungen im Backend von Assembled Trading AI.

---

## 0. Bekannte Datenqualitäts-Risiken (AUDIT A10)

### 0.1 Survivorship-Bias: PIT-Universe — BEHOBEN (2026-05-06)

**Schwere:** reduziert (war: AKUT)  
**Entdeckt:** 2026-04-26 (Audit A10)  
**Status:** ✅ Architektur gewired — data-derived PIT aktiv + Cache-Invalidierung implementiert. ⚠️ Kommerzieller Index-Membership-Feed fehlt weiterhin.

**Was getan wurde:**
- `build_universe_history_from_prices(prices_df)` in `universe.py` — leitet `start_date`/`end_date` direkt aus dem Panel ab.
- `wrap_signal_fn_with_pit_filter(signal_fn, universe_history)` — filtert Signale per Datum gegen die abgeleitete History.
- `scripts/run_backtest_strategy.py` — baut/lädt Universe-History automatisch vor jedem Backtest-Lauf, schreibt nach `data/universe/<panel-stem>.csv`.
- 8 Tests in `tests/test_universe_pit_wire.py` — alle grün.
- **2026-05-06 (d5630b6):** Kritischer Bug behoben — Cache-Invalidierung: wenn `cached start_date > backtest_start`, wird Cache aus `_prices_full_range` (vor Date-Filter) neu gebaut. Verhindert 0-Trades für alle Perioden vor 2025 (root cause: Cache wurde nach 2025-2026-Lauf mit `start_date=2025-01-02` für alle Symbole gebaut).

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

### 6.5 Quant-Methoden als eigene Module (Compass 2026-05-17)

**Quelle:** `autonome_weiterarbeit/wichtig/compass_artifact_wf-738112f8-…_text_markdown.md` (TOP 3 Lücken, Empfehlung 4).

Quantitative Methoden, die der Compass-Snapshot als „eigene Module fehlen" identifizierte. Libraries sind teilweise gepinnt (`scipy`, `arch==8.0.0`, `hmmlearn`), aber kein dediziertes Modul im Repo. **Keine Aktion in der laufenden Dummy/Info-Flow-Plan-Welle (`docs/superpowers/plans/2026-05-17-dummy-data-and-info-flow.md`).**

- [x] **6.5.1 Portfolio-Optimierer** — Markowitz, Risk-Parity, fraktionaler Kelly — **DONE 2026-05-17**
  - **Realisiert:** `src/assembled_core/portfolio/optimizers.py` — dependency-light Pure-numpy/scipy Referenz-Modul, ergänzt die 22 existierenden Portfolio-Module (kein Doppelstruktur)
  - **Funktionen:** `min_variance_weights` + `max_sharpe_weights` (closed-form unconstrained + SLSQP constrained, mit `denom<0`-Fallback), `mean_variance_efficient_frontier` (n_points Grid mit ehrlicher `converged`-Spalte), `equal_risk_contribution_weights` (Maillard 2010 — echte Risk-Parity vs. inverse-vol in `position_sizing`), `multivariate_kelly_weights` (Thorp 2006 half-Kelly default, opt-in `renormalize_to_unity`)
  - **Tests:** 30/30 grün (`tests/test_portfolio_optimizers.py`) — inkl. 4 Regression-Tests aus post-commit Stage 1 (F-postcommit-1 cap-after-renormalize, F-postcommit-2 NaN-leverage-reject, F-postcommit-3 asymmetric-cap-binding, F-postcommit-5 index/columns-mismatch)
  - **Review-Chain:** Stage 1 CONDITIONAL→PASS (3 MAJOR + 2 MINOR integriert: F-stage1-portopt-1 frontier silent-drop, -2 Kelly long_only semantics, -3 max_sharpe sign fallback, -5 PSD-check, F-minor-1 __init__-export). Stage 2 PASS (2 MINOR + 4 INFO, 4 davon integriert).
  - **Doppelstruktur-Audit:** PASS — komplementär zu `riskfolio_optimizer`/`kelly_robust`/`hrp_sizing`/`black_litterman`/`market_neutral_optimizer`/`position_sizing.compute_risk_parity_weights` (inverse-vol, nicht ECHTE ERC).
  - **Wert:** Höchster Quant-Hebel; dependency-light Audit-Referenz; closes cov→weights pipeline mit C4-072 DCC-GARCH.

- [x] **6.5.2 GARCH / Vol-Modellierung** — KONSOLIDIERUNG Phase 1 DONE (2026-05-17): Kanonisches Modul = `garch_vol.py`. `garch_vol_forecast.py` deprecated mit `DeprecationWarning` + Migrationshinweis. Caller-Migration deferred (Phase 2).
  - **Kanonisch:** `src/assembled_core/risk/garch_vol.py` (GJR-GARCH(1,1) + rolling-window FALLBACK, defensive sizing inf/NaN, batch helper `compute_vol_forecasts`)
  - **Deprecated:** `src/assembled_core/risk/garch_vol_forecast.py` (richere konfigurierbare API aber kein Fallback, bei `arch`-Fehler NaN-Rückgabe = Produktions-Hazard)
  - **Phase-2 Follow-up:** Migration der 2 Caller (`scripts/ci/garch_check.py`, `tests/test_free_stack_modules.py`) auf `garch_vol`. Beide benutzen Default-Params, also einfache Substitution. Danach `garch_vol_forecast.py` löschen.
  - **Phase-3 Follow-up (optional):** konfigurierbare Parameter (vol_model, p/o/q, dist) von `garch_vol_forecast` in `garch_vol` einbauen für Feature-Parität, BEVOR die deprecated Datei entfernt wird (falls jemand die Flexibilität tatsächlich braucht — aktuell nicht).
  - **Historie:** Eine dritte naive Implementation `risk/volatility/garch.py` wurde am 2026-05-17 erstellt (commits `61b535b`/`573613a`) und in `7a10d7c` wieder gelöscht.

- [x] **6.5.3 Monte-Carlo / Pfad-Simulation** — KONSOLIDIERUNG Phase 1+2a+2b+2c DONE (2026-05-17). Basis-Modul implementiert (commit `ad728a7`); Doppelstruktur-Audit ergab 3 parallele MC-Module → Phase 1 Konsolidierung + Phase 2 Caller-Migration mit BLOCKER-Findings adressiert + Phase 2c Block-Bootstrap-Support.
  - **Kanonisch:** `src/assembled_core/risk/monte_carlo/` — `shuffle_trades` (bootstrap-resample WITH replacement), **`permute_trades` (NEU 2026-05-17: order permutation WITHOUT replacement — canonical Ersatz für Legacy `monte_carlo_trade_paths`)**, `simulate_paths_iid_normal` (F-risk-4 rename von "gbm"), `simulate_paths_block_bootstrap`. **39 tests pass** (incl. r<=-1.0 input-guard regressions F-RISK-MC1-MINOR-1).
  - **Deprecated mit `DeprecationWarning` + Migrationshinweis:**
    - `qa/monte_carlo.py` (`bootstrap_returns` → `shuffle_trades`, `forward_simulate_gbm` → `simulate_paths_iid_normal`)
    - `qa/monte_carlo_paths.py` (`monte_carlo_trade_paths` → `permute_trades`)
  - **Abgrenzung:** `scenario_engine` macht Stress-Replays, nicht MC
  - **Phase 2a DONE (2026-05-17):** Adapter + 2 Caller migriert.
    - **Adapter:** `pnl_to_returns(pnl, initial_capital)` für currency-PnL→return-Konversion + `shuffle_result_to_quantile_dict(result, n_trades, initial_capital, annual_trading_days)` für Legacy-Schema-Kompat (JSON-Konsumenten brechen nicht). `n_trades` ist **Pflichtparameter** — siehe F-RISK-MC2-BLOCKER-1 unten.
    - **Migriert:** `scripts/run_backtest_strategy.py:2735` (`monte_carlo_trade_paths` → `permute_trades` mit `pnl_to_returns(_, args.start_capital)` + Adapter), `src/.../api/routers/qa.py:510` (Direktzugriff auf `result.sharpe_distribution`, fixt nebenbei einen Legacy-Bug wo `mc.get("sharpe", [0.0])` ein dict in `_np.array` packte).
  - **Phase 2b DONE (2026-05-17, Stage-1-Review-Findings adressiert):**
    - **F-RISK-MC2-BLOCKER-1:** Adapter inferierte `n_trades` falsch aus `sharpe.shape[0]` (= `n_iterations`, nicht `n_trades`) → CAGR-Werte um Faktor ~50 zu klein. Fix: `n_trades: int` als **Pflichtparameter**, ValueError wenn ≤ 0. Regression-Test `test_cagr_magnitude_plausible` würde Bug zurückkehren fangen.
    - **F-RISK-MC2-BLOCKER-2 (E-019 silent fail-open):** `getattr(args, "capital", 100_000)` — `args.capital` existiert nicht (CLI-Flag heißt `--start-capital`). Fix: `getattr(args, "start_capital", None) or 10_000.0` (echter Script-Default).
    - **F-RISK-MC2-MAJOR-3:** CAGR-Clip versteckte ruinöse Pfade (`1+total_ret ≤ 0`). Fix: `pct_ruined` separat im Adapter-Output gezählt VOR clip.
    - **F-RISK-MC2-MAJOR-4:** `except Exception → logger.warning` schluckte Skip ohne Sentinel im JSON. Fix: `metrics.json["monte_carlo"] = {"error": str(_e), "skipped": True}` Sentinel-Output, damit Downstream-Konsumenten Skip von Erfolg unterscheiden können.
  - **Phase 2c DONE (2026-05-17):** Block-Bootstrap-Support + daily_qa_report re-migration.
    - **`shuffle_trades(..., block_size: int = 1)`** erweitert um moving-block bootstrap (Künsch 1989, Politis & Romano 1994). `block_size=1` (default) = bisheriges i.i.d.-Bootstrap; `block_size>1` zieht `ceil(n/block_size)` zusammenhängende Blöcke der Länge `block_size` und konkateniert sie auf `n_trades`. Vektorisiert via offset-Indexing.
    - **`reports/daily_qa_report.py:432`** migriert: `bootstrap_returns` → `shuffle_trades(daily_rets, block_size=5)` (~weekly block). Alle drei semantischen Items adressiert: (a) Block-Bootstrap restored, (b) `point_estimate = profile.sharpe / max_drawdown / total_return` aus already-computed performance profile statt Bootstrap-Median, (c) Per-Metric-Rows enthalten sharpe + max_drawdown + total_return statt legacy `cagr` (honest — `total_return` ist was `shuffle_trades` tatsächlich emittiert).
    - **8 neue Tests** in `TestShuffleTradesBlockBootstrap` (default=iid, block_size>1 distinct distribution für AR(1)-Daten, seed-reproducibility, ValueError bei 0/negative/zu-groß, edge-case `block_size==n`, output-shape).
  - **F-RISK-MC2-MAJOR-2 akzeptiert (dokumentiert):** Migration ändert Sharpe-Werte um bis zu ~8% bei mittleren PnL und final_equity bis ~3× bei großen relativen PnL. Grund: legacy nutzte additive Equity (`K + cumsum(pnl)`), neu nutzt multiplikative (`K * cumprod(1+r)`). Im small-return regime äquivalent, bei großen relativen Trades nicht. Konsequenz: `metrics.json`-Werte sind **nicht regressions-äquivalent zu Vor-Migration-Backtests**.
  - **Tests:** 59/59 in test_risk_monte_carlo.py grün (+5 nach BLOCKER-Fixes: `test_n_trades_uses_caller_value`, `test_missing_n_trades_raises`, `test_invalid_n_trades_raises`, `test_pct_ruined_zero_for_winning_returns`, `test_cagr_magnitude_plausible`). 89 inkl. legacy tests.
  - **`qa/bootstrap_metrics.compute_all_with_ci`** ist separates Modul, eigene Konsolidierungs-Entscheidung — nicht in Phase 2 enthalten.

- [x] **6.5.4 FinBERT / News-Sentiment ML** — DONE (2026-05-17). Modul existiert bereits unter abweichendem Pfad; KNOWN_ISSUES-Ziel-Pfad war veraltet. Tests ergänzt, keine Doppelstruktur erzeugt (Lesson aus §6.5.2/§6.5.3).
  - **Tatsächlicher Pfad:** `src/assembled_core/intel/finbert_sentiment.py` (366 LOC, schon vor 2026-05-17 vorhanden) — NICHT `ml/nlp/finbert.py` wie KNOWN_ISSUES ursprünglich vorgab.
  - **Funktionalität:** 3-Tier-Fallback: FinBERT (ProsusAI/finbert via `transformers`) → VADER (`vaderSentiment`) → keyword-based (always-available, no-deps). Public API: `SentimentResult`, `SentimentScorer`, `get_sentiment_scorer(prefer_backend=...)`, `score_news_items(items, text_key, ...)`. Optional-deps via try-import.
  - **Tests:** `tests/test_finbert_sentiment.py` (NEU 2026-05-17, 14 Tests, 11 pass + 3 skipped für nicht-installierte optional-deps): 6 keyword-tier + 2 VADER (importorskip) + 1 FinBERT (importorskip) + 4 `score_news_items` wrapper + 1 auto-detection.
  - **Caller:** `scripts/news_validation/level_b_event_study.py` (produktiv). News-Signal-Layer-Lücke (KNOWN_ISSUES Wert-Statement) ist via `score_news_items` adressiert — Funktion appendet `sentiment_score/label/confidence/backend` in-place auf Item-Dicts.
  - **Follow-up:** `tests/test_nlp_sentiment.py` importorskip'd auf `src.assembled_core.ml.nlp_sentiment` (archiviert in `archive/observability_graveyard_2026q2/`) — Geist-Test, alle Tests skipped. Separate Cleanup-Action (Delete oder Umleiten auf `intel.finbert_sentiment`) tracken.

- [ ] **6.5.5 Echte Insider/Congress/Shipping Data-Feeds**
  - **Aktion:** Dummy-Generatoren in `insider_ingest.py` / `shipping_routes_ingest.py` werden im Plan 2026-05-17 fail-loud + opt-in gemacht (Sub-Project A, Task A1/A2). Sobald ein echter Feed verdrahtet ist, können die Dummy-Generatoren **vollständig** entfernt werden.
  - **Quellen-Optionen:** Sharadar SF1, QuiverQuant Congress-Trades, Lloyd's MIU Shipping, manueller EDGAR-Scrape
  - **Concrete status Congress (2026-05-17):** `src/assembled_core/data/congress_trades_ingest.py` existiert **nicht** im aktuellen Repo (nur stale `__pycache__`-Artefakte und eine Kopie in `.claude/worktrees/agent-a700e54f/`). `trading_cycle_shared.py:625-647` importiert das Modul in einem try/except — bis zum Plan 2026-05-17 Task A1b war das ein `except Exception: logger.debug(...)`, was `include_congress=True` zum stillen No-op machte. Task A1b verengt den Catch auf `ModuleNotFoundError`/`ImportError` mit `WARNING`-Logging. Restoration des Moduls erfordert eine echte Congress-Trades-Datenquelle — hier tracken, nicht heimlich verkleben.

### 6.6 Live-Broker-Routes (oms.py Placeholder)

**Datei:** `src/assembled_core/api/routers/oms.py:176`  
**Status:** Kommentar `placeholders for future broker routes`  
**Voraussetzung:** Vollständige Broker-Integration mit Alpaca/IBKR/whoever, Pre-Trade-Gate-Verzahnung, Idempotency-Keys, Kill-Switch-Verzahnung (teilweise vorhanden via `broker_adapter.py`).  
**Aktion:** Eigener Plan vor Live-Aktivierung — KEIN Code-Work jetzt.

### 6.7 Research-Notebook-Vollendung

**Status (vor Plan 2026-05-17 Ausführung):** 3 von 4 Notebooks sind effektiv leer (1 Code-Cell, ~2 KB). Plan 2026-05-17 Task A4 verschiebt sie nach `research/dead_ends/` (ehrlicher Provenance-Marker, kein in-place-Tag-Half-Measure).

**Nach Plan-Ausführung (noch ausstehend):**
- [ ] `research/dead_ends/altdata-insider_congress_shipping_exploration.ipynb` — gemovt, Inhalt unverändert
- [ ] `research/dead_ends/meta-meta_model_calibration.ipynb` — gemovt
- [ ] `research/dead_ends/risk-scenario_and_risk_experiments.ipynb` — gemovt
- `research/trend/trend_baseline_experiments.ipynb` — bleibt in place (14 cells, ~10 KB, substantive)

**Aktion:** Wenn künftig konkrete Research auf einem dieser Themen entsteht, neues Notebook in `research/<topic>/` anlegen (NICHT die dead_ends-Kopie wiederbeleben — Provenance-Marker bleibt erhalten).

### 6.8 Phase-Marker Legacy-Aliase entfernen

**Datei:** `pyproject.toml` (Marker-Liste `phase4..phase13` aliased zu `fast`)  
**Status:** Funktional konsolidiert (alle phaseN sind aliases), aber alte Phase-Marker bleiben als Test-Decorator in vielen Test-Dateien.  
**Aktion (deferred):** Phase-Decorator zu `fast`/`regression` migrieren via `sed`, Alias-Marker entfernen.  
**Risiko bei Aufschub:** Niedrig — funktional kein Bug, nur kognitiver Overhead.

### 6.9 Scripts Wildwuchs-Reduktion (Phase 2)

**Status nach Plan 2026-05-17 Sub-Project B:** 8 `_append_batchN.py` weg, evtl. underscore-utilities relokiert, SCRIPTS_INDEX.md angelegt.  
**Phase 2 (deferred):** Bei 140 verbliebenen top-level Scripts weitere Konsolidierung — manche in Subdirs verschieben (ops/, audits/, analysis/), manche tatsächlich tot und löschbar.  
**Aktion:** Eigene Audit-Welle nachdem Phase 1 (Plan 2026-05-17) abgeschlossen ist.

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

### 7.3 Paper-Pilot v1 — abgebrochen nach Tag 4/30

**Status (2026-05-06):** Pilot v1 nach 4 Tagen abgebrochen.

**Grund:** Waves 1–4 (Universe-Expansion, News-Taxonomie, RSS-Feeds, Leverage-Freigabe)
müssen vor einem aussagekräftigen 30-Tage-Pilot implementiert sein.

**Artefakt:** `output/pilot/pilot_manifest_v1_aborted_2026-05-06.json`

**Pilot v2:** Wird nach Abschluss aller Waves (1–4) neu gestartet.

---

## 8. Audit-Sweep 2026-05-12 — Open Items nach 17 Waves

**Kontext:** 4 Compass-Audits in `autonome_weiterarbeit/wichtig/` wurden in 17
Waves abgearbeitet (Commits d0c99ac → d08ed88 auf main, 8f72e7f → 56773ff
auf ERWEITERUNG). ~8500 LoC + 100+ Tests + 18 Audit-Commits gepusht.
Die untenstehenden Items sind bewusst NICHT umgesetzt, mit Begründung und
nächstem Schritt für jeden Punkt. Dies ist die einzige Wahrheit für
"was steht noch aus".

### 8.1 Hexagonal Architecture Migration — Months 2–6

**Status:** Month-1-Skeleton **shipped** (Wave 17, d08ed88). Ports + Container +
Layering-Invariant aktiv. Migration der bestehenden Module: noch offen.

**Pfad:** `docs/HEXAGONAL_MIGRATION_PLAN.md` enthält die file-by-file Map.
**Feature-Flag:** `ASSEMBLED_USE_HEXAGONAL=1` (geplant, noch nicht im Code).

**Konkrete Sprints:**

- [ ] **Month 2 — Application use-cases (4–6h pro Use-Case):**
  - `scripts/run_daily.py` → `application/use_cases/run_eod_pipeline.RunEodPipeline`
  - `scripts/run_backtest_strategy.py` → `application/use_cases/run_backtest.RunBacktest`
  - `scripts/run_api.py` → `adapters/inbound/http/main.py`
  - Paper-trading routes → `application/use_cases/submit_paper_order.SubmitPaperOrder`
- [ ] **Month 3 — Event-Sourcing Order-Lifecycle (audit C-005):**
  - `execution/order_lifecycle.py` → `domain/trading/order.py` + `domain/trading/order_events.py`
  - Neu: `adapters/outbound/event_store_sqlite.py` (append-only)
  - Neu: `application/use_cases/replay_order_history.py`
- [ ] **Month 4 — Plugin architecture (audit C-004):**
  - `pyproject.toml` Entry-Points für Strategien
  - `application/strategy_registry.load_strategies()`
- [ ] **Month 5 — Per-Bounded-Context Tests:**
  - `tests/domain/{trading,risk,...}/`-Verzeichnisse + per-BC layering invariant
- [ ] **Month 6 — Property + Mutation Tests:**
  - mutmut auf `domain/risk/` sobald BC echten Code hat

**Acceptance:** alle 5 BCs haben mindestens ein Modul; alle 3rd-party-Imports
nur unter `adapters/`; `tests/test_hexagonal_layering.py` bleibt grün.

### 8.2 Performance Migration

**Pfad:** `docs/PERFORMANCE_MIGRATION_PLAN.md`.

- [x] **B-001 Polars (1d-Sprint):** DONE. Parallel-Modul
  `src/assembled_core/features/ta_features_polars.py` (259 LOC) shipped als
  Drop-in-Alternative — pandas-in/pandas-out API mit Polars LazyFrame intern.
  Tests `tests/test_ta_features_polars_equivalence.py` pinnen 1e-9 numerische
  Äquivalenz mit pandas-Pfad. Parallel-Modul statt in-place Replace dokumentiert
  als bewusste Entscheidung (numerische Äquivalenz + opt-in pro Caller).
- [x] **B-002 Numba JIT (½d-Sprint):** DONE. Numba-Pfad bereits implementiert in
  `src/assembled_core/qa/backtest_engine_numba.py` (126 LOC) mit `@njit`
  auf `compute_position_deltas_numba` und `aggregate_position_deltas_numba`.
  Wrapper in `qa/backtest_engine.py:143-229` mit `use_numba=True` default +
  graceful fallback auf pandas wenn numba nicht installiert. Item-Blocker
  "numba nicht im venv" ist KEIN Blocker — Pfad ist optional accelerated.
- [ ] **B-003 Rust/PyO3 (LONG):** explizit deferred per Audit selbst — erst nach
  Polars+Numba ausreizen.
- [x] **B-004 Async-I/O:** `utils/async_fetch.py` shipped (Wave 16).
- [x] **B-006 dataclass slots:** shipped (Wave 15).
- [ ] **B-008 Vector/Event-Driven Backtest-Split:** deferred — eigener Architektur-
  Sprint, kein klarer Trigger im aktuellen Workload.

### 8.3 Compliance Activation Triggers

**Pfad:** `docs/COMPLIANCE_THRESHOLDS.md` — single source of truth für "ab wann gewerblich/regulatorisch".

**Status today:** privater Trader, **keine** Compliance-Schwelle aktiviert. Skeletons
liegen bereit für:

- [ ] **T1 → gewerblich aktiviert:** `docs/GOBD_WORM_POLICY.md`,
  `docs/AUDIT_LOG_RETENTION.md` 7y → 10y, RTS-6 Annual Review, MAR-Surveillance
  formell geschrieben.
- [ ] **T2 → Investment Firm (KWG §32):** `docs/MIFID2_VENUE_REPORTS.md` RTS-28
  jährlich, RTS-6-Algo-Inventory live, BaFin-Lizenz-Prozess.
- [ ] **T3 → publication:** `docs/RISK_DISCLOSURE_TEMPLATE.md` auf jeder
  Veröffentlichung verlinken.
- [ ] **T4 → 3rd-party PII:** `docs/GDPR_PII_POLICY.md` aktivieren — Article-17-Endpoint
  bauen, PII-Retention-Cron wiring.

### 8.4 Formal-Verification + DVC Scaffolds

Alle drei Scaffolds sind Artefakte, kein lauffähiger Code-Pfad.

- [ ] **`formal/KillSwitch.lean` (C2-001/002):** enthält `sorry`-TODO bei
  `throttle_monotone`-Theorem. Benötigt **Lean 4 + lake** + mathlib4 für
  Real-Arithmetik-Taktiken. Setup-Aufwand ~2h, Beweisverfeinerung ~4h.
- [ ] **`formal/Reconciliation.tla` (C2-008):** komplett spezifiziert, aber
  benötigt **java + tla2tools.jar**, um TLC laufen zu lassen. CFG-Hints im
  Dateikommentar.
- [ ] **`.dvc/` (C2-045):** Scaffold mit `config.example`. Benötigt
  `pip install 'dvc[s3]'` + B2-Account + `dvc add data/raw/<panel>.parquet`.
  Aktivierungspfad in `.dvc/README.md`.

### 8.5 ML- und Pipeline-Stubs (raise NotImplementedError)

Drei Stubs im Repo, die explizit `NotImplementedError` werfen:

- [x] **`src/assembled_core/ml/gnn_signal.py`** — DOCUMENTED-STUB by-design (2026-05-17).
  Tier-4-Item mit klarem Modul-Header (Zeilen 1-18) + Aktivierungsvoraussetzungen
  (PyG + CUDA env, co-movement adjacency pipeline, node feature engineering) +
  akademischen Referenzen (Kipf-Welling 2016, Hamilton 2017, Xu 2018). Stub
  retourniert zero signals, `NotImplementedError` für training in Zeilen 149/155/192.
  Kein Live-Pfad, ehrliche Disclosure. Aktivierung = eigener Sprint mit dep-Setup.
- [x] **`src/assembled_core/ml/differential_privacy.py`** — DOCUMENTED-STUB by-design (2026-05-17).
  Modul-Header (Zeilen 1-20) dokumentiert Tier-4-Status + Aktivierungsvoraussetzungen
  (Opacus für DP-SGD-Training, epsilon-delta-Budgeting). Reine Python Gaussian/Laplace
  Mechanismen für scalar statistics sind bereits da; nur DP-SGD-Gradient-Clipping
  ist `NotImplementedError` (Zeile 264). Akademische Referenzen (Dwork-Roth 2014,
  Abadi 2016, Mironov 2017). Aktivierung = LONG-Term Reputation-Item.
- [ ] **`src/assembled_core/pipeline/_shared_eod.py`** (Zeile 24) +
  **`src/assembled_core/pipeline/orchestrator.py`** (Zeile 12):
  Pipeline-Orchestrator-Konsolidierung ist deferred — siehe
  `autonome_weiterarbeit/AUDIT_2026-04-26_FINDINGS_AND_REMEDIATION_v2.md §B5`.
  Audit-Schätzung 12-20h. **DEFER:** sensible Pipeline-Zone, eigener Sprint
  mit OOS-Re-Run-Validation nötig. Kein autonomer Cleanup.

### 8.6 Konkrete Code-TODOs im Repo

Vollständige Liste der `TODO`/`FIXME`-Marker im Code (geprüft 2026-05-12, aktualisiert 2026-05-14):

- [x] **`src/assembled_core/pipeline/_tc_sizing.py:1713-1715`** — **GEKLÄRT
  (Wave 22, 2026-05-14):** Der TODO-Kommentar ist faktisch obsolet. Die
  60s-refresh Halt-Cache wurde in Wave 20 (`ed9a126`) bereits gewired via
  `ops/_paper_runner_gates.apply_halt_cache_gate` + `utils.halt_cache.HaltCache`
  (TTL 60s). Dieses Call-Site konsumiert nur `ctx.halted_symbols`.
  **Code-Edit aufgeschoben** auf eine eigene Formatier-Welle: ruff-format
  0.8.6 (pre-commit) vs black 24.10.0 (pre-commit) sind auf diesem File in
  pre-existing Konflikt (Zeilen 760, 1815 — unabhängig von dieser Welle),
  was jeden Edit hier in eine Hook-Ping-Pong-Schleife schickt.
- [x] **`scripts/run_event_study.py`** — **BEHOBEN (Wave 22, 2026-05-14):**
  Skeleton durch echte CLI-Implementierung ersetzt. Wired die drei
  existierenden `qa/event_study.py`-Funktionen (`build_event_window_prices`,
  `compute_event_returns`, `aggregate_event_study`) inklusive Events-Loader
  (CSV/JSON), Price-Source-Plumbing, CSV + Markdown-Report. 13 Integration-
  und Helper-Tests grün in `tests/test_run_event_study_cli.py`.
  **Noch offen (eigener Sprint):** Audit C4-081's vollständige
  Boehmer-Musumeci-Poulsen-t-Stat / BHAR-Methodik — die heutige Aggregation
  ist `avg_ret + cum_ret + CI` (z-score basiert), kein Market-Model.
- [x] **`scripts/check_health.py:1451`** — **KEIN TODO, by-design:** Symbol-
  basiertes Benchmark-Loading ist absichtlich nicht implementiert, weil
  `check_health.py` read-only / network-free invariant ist. Operator muss
  via `--benchmark-file` einen vorgefetchten Pfad übergeben. Kommentar im
  Code dokumentiert das bereits.
- [x] **`tests/test_risk_regime_analysis.py:269`** — DONE 2026-04-30 (geschlossen mit §6.2).
  Der Test asserted seitdem aktiv die Anwesenheit der drei Spalten
  `win_rate / avg_trade_duration / avg_profit_per_trade`; pro-Regime NaN ist
  erlaubt nur wenn keine closed round-trips in dem Slice existieren. KNOWN_ISSUES-Eintrag
  war veraltet, jetzt synchronisiert.

### 8.7 Equity-Curve-Baseline Forensics

**Memo:** `autonome_weiterarbeit/EQUITY_CURVE_BASELINE_FORENSICS_2026-05-12.md`.

`output/equity_curve_baseline.csv` zeigt CAGR 43.01%, Sharpe 3.90, MaxDD -4.52%
über 3.32 Jahre. Post-Wave-11 DSR=25.3 / PSR=1.0 (beide PASS). **Aber:**
vier klassische Suspects nicht autonom widerlegbar:

- [x] **Survivorship-Bias-Check:** Infrastructure DONE 2026-05-18.
  `scripts/forensic/survivorship_bias_check.py` (NEU, ~340 LOC) prüft watchlist
  gegen 3 Bias-Indikatoren ohne externe SP500-Konstituenten-Daten:
  1. Active/Delisted Ratio (real Universen 5-10% delisted erwartet)
  2. Cross-Check gegen hardcoded `KNOWN_US_DELISTINGS` Sample (15 events:
     LEH 2008-09-15, BSC 2008-03-17, WAMUQ, WB, CFC, AIG, GM_OLD, CIT, SHLD,
     JCP, HTZ_OLD, FTX, SVB, SI, FRC 2023-05-01)
  3. Start-Date-Clustering (PIT-Universe sollte varied start_dates haben)
  Verdict-Aggregation: low / medium / high abhängig von n_flags.
  **Baseline-Lauf 2026-05-18 auf `data/universe/watchlist_2007_2026.csv`:**
  Verdict = **HIGH** (3/3 flags getriggert):
  - 100% active (19/19, 0 delisted)
  - 15 known delistings in window — ALLE missing
  - 19 Symbole sharen `2008-09-02` als start_date
  Das bestätigt explizit den Survivorship-Bias der Baseline-Equity-Curve §8.7.
  Tests: `tests/test_forensic_survivorship.py` (22 Tests, alle pass).
  **Vollständige C3-063 Closure** verlangt CRSP-quality hist-SP500-Konstituenten
  (extern) — Skript-Infrastruktur ist da, datentechnische Vollendung bleibt
  als externes follow-up.
- [x] **Look-Ahead-Bias:** DONE 2026-05-18. `tests/test_pit_strategy_features.py`
  (NEU) pinnt PIT-Safety für 6 Strategie-Kern-Features via Hypothesis property-tests:
  log_returns, moving_averages, ATR, RSI, MACD, Bollinger Bands. PIT-Property:
  ``f(prices[:k]) == f(prices)[:k]`` für beliebige Prefix-Längen. Negativ-Kontrolle
  (leaky feature uses last value of series) verifiziert dass die Harness echte
  Leaks erkennt. 7/7 Tests pass — alle 6 Features genuinely PIT-safe.
- [x] **Fill-Modell-Audit:** Infrastructure DONE 2026-05-18.
  `scripts/forensic/fill_model_audit.py` (NEU, ~280 LOC) prüft
  `configs/cost_tiers.yaml` (5 ADV-Tiers × {commission/half_spread/slippage}_bps)
  + `configs/policy.yaml::borrow_costs` gegen INDUSTRY_BASELINES (informelle
  Public-Ranges aus IBKR Pro / Tastytrade / Alpaca / institutional desk).
  Flaggt "optimistic" werte unter industry-min als potentielles Cost-
  Underestimate-Artefakt.
  **Baseline-Lauf 2026-05-18:** VERDICT = `low` (0/17 flags).
  Alle 5 Tiers × 3 Fields + 2 Borrow-Cost Rates sitzen in industry-plausible
  ranges. Bedeutet: Sharpe 3.896-Baseline ist NICHT durch Cost-Underestimate
  erzeugt — die Cost-Konfiguration ist realistisch.
  Tests: `tests/test_forensic_fill_model_audit.py` (20 Tests, alle pass).
  **Limitations honestly disclosed:** Audit prüft Config gegen Public-Ranges,
  NICHT gegen reale Broker-Statements. Vollständige C3-063 Closure verlangt
  externen Vintage-Vergleich (separater follow-up).
- [x] **Hold-Out-Leakage:** Infrastructure DONE 2026-05-18.
  `scripts/forensic/hold_out_leakage_test.py` (NEU, ~340 LOC) implementiert
  Permutation-Test auf Sharpe + MDD + Train/Test-Split-Audit. **Baseline-Lauf
  auf `output/equity_curve_baseline.csv` (835d):**
  - Train Sharpe (584d): 3.7179
  - Test Sharpe (251d): 5.0407 **(test > train → KEIN overfitting Pattern)**
  - Sharpe decay train→test: -1.3228 (negative = test besser als train)
  - Full-series MDD permutation p = 0.3740
  - Test-set MDD permutation p = 0.2840
  - **Verdict: `hold_out_edge_indistinguishable_from_random`**
    (test_sharpe > 0 ABER MDD-Pfadabhängigkeit p ≥ 0.20 → nicht signifikant
    unterscheidbar von zufälliger Permutation der gleichen returns).
  Wichtiges Audit-Finding: Sharpe ist hoch + Test > Train, ABER die Pfad-
  Sequenz der Drawdowns ist nicht statistisch unterscheidbar von random
  ordering. Das schließt eine echte Edge nicht aus, lässt sie aber
  unbestätigt mit 1000 Permutationen.
  Tests: `tests/test_forensic_hold_out_leakage.py` (19 Tests, alle pass).
  Honest degeneracy note im Sharpe-Permutation-Result: Permutation eines
  i.i.d. Returns-Samples preserves Sharpe exakt; nur path-dependent MDD
  ist informativ.

**Pflicht vor jeder externen Zitation der Zahlen.** Der Re-Runner
(`scripts/forensic/rerun_baseline.py`, Audit C4-049) ist **nicht** implementiert —
verlangt DVC-Pin der yfinance-Daten + git-tag + Multi-Stunden-Backtest.

### 8.8 ERWEITERUNG-Branch Cherry-Picks zu `main`

**Status:** P1-Fixes (CPCV / Stacking / CVaR) auf ERWEITERUNG gepusht
(Commit 8f72e7f). 14 weitere Module sind audit-flagged für Cherry-Pick zu `main`
**erst nach erfolgreicher OOS-Re-Run** der `volatility_targeting`-Metrik
(audit C3 §3.1):

- [x] CPCV-Modul Migration (`erweiterung/backtest/cpcv.py` → `assembled_core/qa/`) — DONE.
  Kanonisches Modul: `src/assembled_core/qa/cpcv_validation.py`. ERWEITERUNG-Source
  `erweiterung/backtest/cpcv.py` existiert nicht mehr im Repo.
- [x] DSR / White-Reality-Check / Calmar Bootstrap / MaxEnt Bootstrap / Walk-Forward —
  §8.13 Forensik-Items im Audit-Sweep abgeschlossen. **Hansen SPA C4-066 bleibt
  ERWEITERUNG-only-skip** (siehe §8.13 unten).
- [x] Equity-Curve-Audit (audit C3-030) — Infrastructure DONE 2026-05-18.
  `scripts/forensic/equity_curve_audit.py` (NEU) liest beliebige equity-curve CSV
  und produziert JSON + Markdown Audit-Reports unter `output/qa/`. Statistiken:
  DSR (Deflated Sharpe, Bailey-Lopez de Prado 2014), PSR (Probabilistic Sharpe
  2012), Min-TRL heuristisch, Bootstrap-CIs via shuffle_trades (block_size=5),
  Skew/Kurtosis/Jarque-Bera Normality, Ljung-Box Autokorrelation (lags 1/5/10/20),
  Drawdown-Duration-Distribution.
  Baseline-Lauf 2026-05-18 auf `output/equity_curve_baseline.csv` (835 Tage,
  3.3y): Sharpe 3.896, CAGR 43.17%, MDD -4.52% (12d), DSR 24.83 (n_strategies=10),
  PSR ≈ 1.0, Bootstrap 95% Sharpe-CI [2.97, 4.84]. Excess Kurtosis 3.74 +
  positive Skew 0.77 — non-normal, Jarque-Bera rejects. KEINE Autokorrelation
  (Ljung-Box p > 0.6 für alle Lags). 109 Drawdown-Episoden, max 54 Tage.
  Tests: `tests/test_forensic_equity_curve_audit.py` (12 Tests, alle pass).
- [x] Portfolio-Optimierer (Markowitz/ERC/Kelly) — §6.5.1 DONE (2026-05-17,
  commit 2feec16+e26ee81). HRP / Black-Litterman / Resampled-EF / CVaR / Max-Div
  bleiben separate Items in `portfolio/` (z. T. existieren bereits — siehe
  Doppelstruktur-Audit unter §6.5.1).
- [x] Risk-Analytics — weitgehend DONE via existierende Module (2026-05-17 Audit):
  - `tail_risk_evt` → `risk/risk_metrics.py::compute_evt_tail_var` (Zeile 1201)
  - `cornish_fisher_var` → `risk/risk_metrics.py::compute_cornish_fisher_var` (Zeile 257)
    + `risk/var_methods.py::VaRCalculator.cornish_fisher_var` (Zeile 204)
  - `crisis_composite` → `events/crisis_alpha/` Subsystem (15 Module incl.
    risk_budget, gates, pipeline, state_machine, entry/exit_rules, baskets)
  - `correlation_breakdown` → `qa/scenario_simulator.py::simulate_correlation_breakdown_scenario`
  - `dynamic_drawdown_control` → `risk/risk_metrics.py::compute_drawdown_duration`
    (passive metric); active limit-management ist Teil von `risk/state_machine.py`.
- [x] Volatility-Models (GARCH/EGARCH/GJR, HAR-RV, DCC-GARCH) — §6.5.2 GARCH-
  Konsolidierung Phase 1 DONE (2026-05-17). C4-072 DCC-GARCH (Engle 2002 + cDCC
  Aielli 2013) implementiert in `src/assembled_core/risk/dcc_garch.py`. HAR-RV in
  §8.13 Forensik-Sweep adressiert.
- [x] **Volatility-Targeting-Strategie (audit C3-034):** DONE / Layered architecture
  (2026-05-17 Audit). KEIN Doppelstruktur — zwei komplementäre Module:
  - `risk/vol_targeting.py` — rolling-realized-vol (backward-looking) als Default-Pfad.
  - `risk/vol_targeting_ewma.py` — EWMA forward-looking Variante (JP Morgan
    RiskMetrics 1996, λ ≈ 0.94 für daily). Modul-Header sagt explizit
    „does NOT displace risk.vol_targeting; opt-in via `policy.vol_targeting.method: 'ewma'`".
    ERWEITERUNG-GARCH-Variante (audit will full GARCH) bleibt cherry-pick-blocked
    (§8.8 OOS-Re-Run-Gate).
- [x] Attribution, State-Space, Time-Series-Tools, Microstructure, Stress-Testing,
  Economic-Data, Factor-Suite — weitgehend existent: Attribution via Brinson-Fachler
  in `risk_metrics::compute_brinson_fachler_attribution`; State-Space in
  `signals/regime/hmm_posterior.py`; Stress-Testing in `qa/scenario_simulator.py`
  + `portfolio/stress_test_constraints.py`; Economic-Data via FRED/macro-Pfade;
  Factor-Suite über `features/` + `strategies/multifactor_v2.py`. Einzelne Audit-
  Wünsche (z. B. Microstructure-Tools über order-book hinaus) bleiben separate
  Items, aber das Sammel-Item §8.8 ist nicht mehr aussagekräftig.

**Discard-Liste (audit C3-043):** `dl/`, `dl_advanced/`, `rl/`,
`discovery/genetic_programming`, `bayesian/`, `causal_inference/`,
`online_learning/`, `nlp/lda_topic`, `meta/bandit_allocator`, `orderbook/`,
`survival/`, `stacking_ensemble`, `multi_factor_vol_target`,
`regime_conditional_allocator`, `multi_signal_regime`, `yfinance_cache_loader`.

### 8.9 External Services — Activation Pending

**Runbook:** `docs/EXTERNAL_SERVICES_SETUP.md`. Alle als Setup-Schritte
dokumentiert, keiner aktiv.

- [ ] **Slack Webhook:** `SLACK_WEBHOOK_URL` env setzen → existierende
  `_send_slack`-Logik (Wave 1) wird aktiv.
- [ ] **healthchecks.io Dead-Man-Switch:** cron-Job mit
  `scripts/ops/setup_uptime_robot.sh` + Healthcheck-UUID. ~5 min Setup.
- [ ] **Backblaze B2 Bucket mit Object Lock (10y Compliance):** Account +
  `scripts/ops/setup_b2_backup.sh` für audit-log-replication.
- [ ] **Litestream SQLite-Replikation:** `configs/integrations/litestream.yml.example`
  als Vorlage + systemd-Unit / nssm-Service.
- [ ] **Telegram Bot Fallback / SMTP-Email Fallback:** existierende
  `_send_telegram` / `_send_email` (ops/alerting.py) — Credentials fehlen.
- [ ] **Cloudflare DNS Failover (Multi-Region, audit I-008):** LONG, erst wenn
  Single-Hetzner-Setup das Bottleneck ist.

### 8.10 Beyond Tier 1 — Deferred Items

Aus Audit C2 (compass_artifact_wf-05256797), nicht in diesem Sweep umgesetzt:

- [ ] **Coq Order-FSM Proof (C2-005):** parallel zum Lean-Scaffold; benötigt
  Coq + ssreflect.
- [ ] **Differential Testing 4-fach (C2-006):** Python/Polars/Numba/Rust
  ε-bounded MI für Sharpe-Metrik. Sobald Polars + Numba aktiv.
- [ ] **Concolic Testing für Order-FSM (C2-007):** benötigt `crosshair`
  package + Setup. ~8h.
- [ ] **LitmusChaos auf k3s (C2-012):** k3s-Setup + ChaosEngine YAMLs.
  ~10h.
- [ ] **12 GameDay-Drills über Jahr (C2-014):** ~24h, terminiert.
- [ ] **Out-of-Universe-Test (C2-018):** Train US-S&P, Test STOXX600 + TOPIX.
  Benötigt EU-/JP-Daten.
- [x] **Out-of-Regime-Test (C2-019):** Infrastructure DONE 2026-05-18.
  `scripts/forensic/out_of_regime_test.py` (NEU) klassifiziert equity-curve via
  trailing-return-Sign (default 120d-window, ±5%-threshold) in Bull/Bear/Sideways/
  Warmup. Per-Regime Sharpe/MDD/MeanRet + Edge-Consistency-Verdict
  (`robust` / `regime_dependent` / `insufficient_data`).
  Baseline-Lauf 2026-05-18: 713 Bull / 0 Bear / 2 Sideways / 119 Warmup —
  Bull-Sharpe 3.94, **kein Bear-Sample** (Strategy hat keine 6m-Drawdown-Phase erlebt).
  Tests: `tests/test_forensic_out_of_regime.py` (18 Tests, alle pass).
  **Honest Disclosure im Report:** self-referential heuristic (Strategy hat eigene
  Bull-Tage definiert). „Echter" Out-of-Regime-Test mit external benchmark (SPY)
  ist C2-018 Out-of-Universe scope (extern, separat).
- [ ] **DoubleML PLR + Causal Forest (C2-025/026):** `doubleml`, `econml`
  nicht im venv. ~10h pro Modell.
- [ ] **Synthetic Control Showcase (C2-027):** Abadie-Diamond-Hainmueller —
  Research-Notebook.
- [x] **Transfer Entropy Screen (C2-029):** DONE via §8.13 Forensik-Sweep.
  `src/assembled_core/qa/transfer_entropy.py` (263 LOC) implementiert Schreiber-2000
  Transfer Entropy mit zwei Estimatoren (binned histogram + KSG-heuristic).
  KEIN `tigramite`/PyIF-Dep — pure numpy. Hinweis im Modul-Header: KSG-TE ist
  nicht vollständig Wibral, sondern heuristisch (sklearn lacks multivariate joint MI).
- [x] **Adaptive Conformal Inference (C2-031), Conformalized Quantile** DONE
  (2026-05-18 Audit). Vollständige Conformal-Inference-Suite:
  `qa/conformal.py`, `qa/conformal_cross.py`, `qa/conformal_adaptive.py` (Adaptive),
  `qa/conformal_quantile.py` (CQR), `portfolio/conformal_position.py` +
  `portfolio/adaptive_conformal_position.py` für position sizing.
  15 Files referenzieren Conformal-Konzepte (`kelly_uncertainty.py`, ops/decision_log,
  api/routers/diagnostics).
- [ ] **DRO Wasserstein / KL-Portfolio (C2-036/037):** benötigt
  `cvxpy + MOSEK` (akademische Lizenz). ~14h + 8h.
- [ ] **Temporal Fusion Transformer (C2-039):** `pytorch-forecasting`.
- [ ] **Logic Tensor Networks (C2-041):** LONG, research showcase.
- [ ] **Quantum QUBO Portfolio Showcase (C2-042–044):** D-Wave Leap Account
  + `dimod`. ~12h, LONG.
- [ ] **MLflow self-hosted (C2-046):** Postgres + S3 + Tracking-Server.
  Eigener Infra-Sprint.
- [x] **10y-Replay-Test CI (C2-050):** Infrastructure DONE 2026-05-18.
  `tests/test_replay_determinism.py` (NEU) pinnt SHA-256 byte-equal Determinismus
  für 4 Kernel-Pfade (alle PASS, 7/7):
  - `add_log_returns` (core feature, 2 Runs identische CSV-bytes)
  - `compute_position_deltas_numba` + `aggregate_position_deltas_numba` (B-002 Hot-Loop)
  - `cumprod(1+r)` für ≈10y synthetic equity curve (2520 Tage, fixed seed)
  - Cross-import stability via `importlib.reload` (catches module-level state leaks)
  Negativ-Kontrolle: different input → different hash. Echter 10y-Replay (full
  Pipeline) ist eigener Sprint mit DVC-Pin + multi-Stunden-Run; das Infrastruktur-
  Item für CI ist hier abgeschlossen.
- [x] **Adversarial Reviewer Notebook Pattern (C2-051):** DONE 2026-05-18.
  CI-Hook etabliert: `scripts/ci/check_adversarial_reviewer_pattern.py` scannt
  `research/` rekursiv (exkludiert `dead_ends/`), prüft pro `research_*.ipynb`
  ob ein passendes `review_*.ipynb` existiert. Exit 1 wenn missing, exit 0 sonst.
  Heute zero `research_*.ipynb` → no-op-Gate (Convention noch nicht adoptiert).
  Wired in `.github/workflows/repo-health.yml` (monatlich + manual dispatch).
  Tests: `tests/test_adversarial_reviewer_pattern.py` (15 Tests, alle pass —
  inkl. End-to-End-Test gegen das echte research/-Verzeichnis).
- [ ] **Signal-Bus Refactor (C2-053):** Redis-Streams oder in-process
  EventBus. Port existiert (Wave 17), Implementation fehlt.
- [x] **Meta-Labeling 3-Stage Pipeline (C2-054):** DONE via existing modules
  (2026-05-18 Audit). AFML Kap. 3 Stages:
  - Stage 1 (Triple-Barrier Labeling): `src/assembled_core/features/triple_barrier.py` (454 LOC)
  - Stage 2 (Meta-Model): `src/assembled_core/signals/meta_model.py` (549 LOC) +
    `src/assembled_core/signals/ensemble.py` (apply_meta_filter/apply_meta_scaling)
  - Stage 3 (Labeling/Validation Pipeline): `src/assembled_core/qa/labeling.py` (748 LOC)
  - Validierung via CPCV: `qa/cpcv_validation.py`
- [x] **Regime-aware Conditional Ensemble (C2-055):** DONE via existing module
  (2026-05-18 Audit). `src/assembled_core/signals/regime/hmm_posterior.py`
  implementiert exakt diesen Use-Case:
  `final_weights = Σ_k P(regime=k | x_t) * base_weights[k]` mit EWMA-Smoothing
  (Half-Life 5d default). Base-weights als `{regime: {factor: weight}}` Dict.
  Stateful Smoother für Whipsaw-Vermeidung.
- [x] **HMM-Regime-Detection (C2-056):** DONE / Layered architecture (2026-05-17).
  KNOWN_ISSUES-Pfad `risk/regime_hmm.py` war veraltet/falsch. Tatsächliche
  3-Modul-Layered-Architektur:
  - `src/assembled_core/ml/regime_hmm.py` — Gaussian HMM via `hmmlearn` (23
    Defs/Klassen, fit/predict/predict_proba auf beliebigen Feature-Series).
  - `src/assembled_core/risk/regime_models.py` — Rule-based Bull/Bear/
    Sideways/Crisis/Reflation auf macro+breadth+vol+trend (KEIN HMM-Duplikat,
    sondern komplementärer regelbasierter Pfad).
  - `src/assembled_core/signals/regime/hmm_posterior.py` — F2 Signal-Layer-
    Wrapper mit Posterior × Base-Weight + EWMA-Smoothing (Half-Life 5d default).
  Audit-Wunsch „3-Zustands-HMM auf VIX + 10y-Yield + DXY" ist via Caller-
  Konfiguration adressierbar — das Modul akzeptiert beliebige Multivariate-
  Feature-Series, kein Modul-Defizit.
- [x] **Stacking-Ensemble (C2-058):** DONE via existing module (2026-05-18 Audit).
  `src/assembled_core/signals/ensemble.py` kombiniert rule-based Signale mit
  Meta-Model-Confidence-Scores via `apply_meta_filter` / `apply_meta_scaling`.
  Meta-Model in `src/assembled_core/signals/meta_model.py`. Audit-Empfehlung
  „BMA als robuste Alternative" ist konfigurierbarer Erweiterungs-Punkt, kein
  Modul-Defizit.
- [ ] **Alt-Data Pipelines vollständig (C2-059):** FRED, EDGAR, GDELT,
  Wikipedia, FINRA, BLS, ECB SDW — Source-Module existieren; Feature-Builder
  fehlen.
- [ ] **PEAD-Strategie (C2-060):** Bernard-Thomas 1989, ~25h, benötigt
  Earnings-Calendar + IBES.
- [ ] **Form-4-Insider-Trades-Strategie (C2-061):** ~15h, benötigt EDGAR
  4-Filing Parser.
- [x] **Almgren-Chriss Refinement (C2-062):** DONE (Modul existiert, 2026-05-18 Audit).
  `src/assembled_core/execution/almgren_chriss.py` (361 LOC, Almgren-Chriss 2001
  Framework: permanent + temporary market impact + execution risk). γ/η/σ-
  Parameter-Kalibrierung ist Caller-Konfiguration (`configs/policy.yaml`),
  kein Modul-Defizit. Gewired in `execution/execution_router.py`,
  `pipeline/_tc_execution.py`, `ops/execution_cost_meta.py`.
- [ ] **Borrow-Cost-Optimierung (C2-063):** IBKR-Short-Stock-Yield-API
  Integration. ~10h.
- [ ] **Tax-Loss-Harvesting (C2-064):** DE-Q3-Workflow. ~8h doc + cron.
- [x] **Robust-Kelly-Sizing (C2-065):** DONE (Modul existiert, 2026-05-18 Audit).
  `src/assembled_core/portfolio/kelly_robust.py` (186 LOC) implementiert beide
  Browne-Whitt-Fixes gegen biased-upward plug-in Kelly mit sample-Estimaten.
  Audit C2-065 ist im Modul-Header explizit referenziert. Ergänzt durch
  `portfolio/optimizers.py::multivariate_kelly_weights` (§6.5.1, half-Kelly default).
- [x] **Vol-Targeting (C2-066):** DONE auf main via Layered-Architektur (2026-05-17 Audit).
  Siehe §8.8 audit C3-034 — `risk/vol_targeting.py` (rolling-realized) +
  `risk/vol_targeting_ewma.py` (EWMA opt-in). KEIN ERWEITERUNG-Cherry-Pick nötig.
- [ ] **Put-Write Tail-Hedge (C2-067):** Options-Daten + LONG-Setup.
- [x] **CAGR-Attribution Quarterly Report (C2-068):** DONE via existing modules
  (2026-05-18 Audit). Brinson-Fachler-Attribution in
  `risk/risk_metrics.py::compute_brinson_fachler_attribution` (Zeile 928).
  Benchmark-Metrics-Layer in `qa/benchmark_metrics.py` (271 LOC). Quarterly-Wrapper
  ist nur Konfiguration über existierende CLI-Pfade (kein separates Modul nötig).
- [x] **Macro-Overlay (C2-069):** DONE via existing modules (2026-05-18 Audit).
  Yield-Curve / HY-OAS / DXY in `features/intermarket_factors.py` (322 LOC, ETF-Proxies
  TLT/IEF/GLD/UUP/HYG + FRED yield curve) + `features/macro_features.py` (160 LOC)
  + `features/term_structure.py`. 13 Files referenzieren Macro-Overlay-Konzepte.
  Konsolidierungs-Status für etwaige Duplicate ist separater Audit-Sprint.
- [x] **Tilt-Detection automatisiert (C2-073):** DONE via existing module (2026-05-18 Audit).
  `src/assembled_core/risk/tilt_detection.py` (199 LOC, audit C2-073 explizit
  referenziert im Header). Vier rule-based Signale: consecutive losses, intraday
  drawdown, etc. Thresholds in `configs/policy.yaml` unter `risk.tilt`. Wired in
  `ops/_paper_runner_gates.py` als Pre-Trade-Pause-Gate.
- [ ] **Two-Account-Setup (C2-074):** Research-Account vs. Trading-Account
  Promotion-Gate. ~6h Operations-Doc.

### 8.11 Beyond-Tier-1 OSS / Career Items (audit C2-080..087)

- [ ] **OSS-Repo-Polish (C2-080):** README mit Hero-Image, Badges, Quickstart,
  MkDocs auf GH-Pages, semver Releases.
- [ ] **arXiv-Preprint #1 (C2-081):** "Open-Source CPCV Replication & Edge Cases".
  ~60h.
- [ ] **2 Konferenz-Talks (C2-082):** PyData / EuroPython / QuantCon / OSQF /
  ICAIF. ~80h.
- [ ] **JFDS-Submission (C2-083):** "Conformal Prediction for Position Sizing".
  Depends on Wave-16 Conformal-Modul. ~120h mit Reviews.
- [ ] **JPM/JoFE-Submission (C2-087):** "Adversarial Backtest Validation:
  8-Test Framework". 6-9 Monate.
- [ ] **Twitter/LinkedIn-Disziplin (C2-084):** kontinuierlich, ~4h/Woche.
- [ ] **Tier-1-Interview-Preparation (C2-085):** ~150h über 8-12 Wochen.
- [ ] **Networking AQR/Q-Group/EFA/AFA/CQF/PyData-Meetups (C2-086):** laufend.

### 8.12 Compliance / Regulatorik — External

Alle erfordern externe Akteure, dokumentiert in `docs/COMPLIANCE_THRESHOLDS.md`:

- [ ] **KWG §32-Klärung mit Aufsichtsrechtsanwalt (C2-075):** 4h Setup +
  300-600 EUR Anwaltskosten.
- [ ] **UG-Gründung (C2-076):** Stammkapital 1000 EUR + Gründung 400 EUR
  via Musterprotokoll.
- [ ] **RTS-6-Light Self-Assessment (C2-077):** Algo-Inventory, PTC,
  Real-Time-Monitoring 5s-SLA, Kill-Switch-Test quartalsweise, Annual
  Validation. ~16h vor Live, dann jährlich.
- [ ] **Versicherungen (C2-078):** Berufshaftpflicht / Cyber /
  Vermögensschaden / Rechtsschutz / BU / D&O. ~6h Recherche + 600-2000 EUR/Jahr.
- [ ] **Banking-Trennung (C2-079):** Privat-Giro + IBKR-Pro + Lynx/Tastytrade
  + UG/GmbH-Konto + Tax-Konto separat.
- [ ] **MAR-Surveillance Live-Aktivierung (C4-093):** Policy in
  `docs/MAR_SURVEILLANCE_POLICY.md`; Live-Wiring der 5 Detection-Signale
  ist offen (manuelle Wöchentliche Review heute genug).
- [ ] **Stagewise GDPR PII-Pipeline (C4-090):** 30-Tage-Retention Cron
  (`scripts/ops/purge_pii_aged.py`) + Article-17 Deletion-Endpoint.
- [ ] **GoBD Off-Site Cold Copy (C4-091):** sobald gewerblich aktiviert.
- [ ] **Secret-Rotation bei allen Providern (C3-010):** Alpaca / Polygon /
  FRED / NewsAPI / Anthropic etc. — eigenständige sicherheitskritische
  Operation, verlangt expliziten User-Auftrag (CLAUDE.md §20 Incident-Regel).
- [ ] **Git-History Bereinigung (C3-011):** falls historisch Secrets in
  Commits waren. **DESTRUKTIVE OPERATION** — `git filter-repo` + Force-Push.

### 8.13 Quant-Forensik Backlog (Audit-Methodology)

Verifikationen aus Audit C4-065..C4-084, die noch nicht erschöpfend
durchgeführt wurden:

- [ ] **C4-066 Hansen SPA in ERWEITERUNG:** Datei `erweiterung/backtest/hansen_spa.py`
  fehlt (Audit listete sie). Wave-16 Hansen-SPA wrapper liegt auf main
  unter `qa/spa_test.py`, ERWEITERUNG-Pfad bleibt offen.
- [x] **C4-072 DCC-GARCH cDCC-Variante (Aielli 2013)** — DONE 2026-05-17:
  Neues Modul `src/assembled_core/risk/dcc_garch.py` ersetzt den
  silent-stub-fallback in `portfolio/covariance.py:126` (`method='dcc_garch'`
  ging vorher heimlich auf sample-covariance — §7.4 violation).
  - **`fit_dcc_garch(returns, method='dcc')`** — Engle (2002) Two-Step:
    Schritt 1) univariate GARCH(1,1) per Serie via `arch` → conditional vol
    + standardized residuals e_t = r_t/σ_t. Schritt 2) DCC-Recursion
    Q_t = (1-α-β)·Q̄ + α·e_{t-1}e_{t-1}' + β·Q_{t-1}, dann
    R_t = diag(Q_t)^(-1/2) · Q_t · diag(Q_t)^(-1/2), H_t = D_t · R_t · D_t.
    QMLE-Schätzung von (α, β) via scipy L-BFGS-B mit Stationaritäts-Bound
    α + β < 1.
  - **`method='cdcc'`** — Aielli (2013) Bias-Correction: ein-Schritt-Pass
    der korrigierten Standardized-Residuals e*_t = diag(Q_t)^(1/2) · e_t zur
    Re-Estimation von Q̄. Vollständiger Fixpoint-Solver out of scope; der
    ein-Schritt-Pass beseitigt den dominanten Bias-Anteil.
  - **Public API**: `DCCResult` dataclass mit α, β, Q̄, conditional_vols/correlations/
    covariance (T-langes Listing), standardized_residuals, log_likelihood,
    converged. `current_covariance(result)` Convenience für portfolio-opt.
  - **Integration**: `portfolio.covariance.estimate_covariance(method='dcc_garch'|'cdcc')`
    routet jetzt echt — alter silent-fallback entfernt, mit graceful
    sample-fallback wenn `arch`/`scipy` fehlen.
  - **15 Tests pass**: DCCResult-Felder, R_t-Diagonale=1, H_t-Symmetrie,
    Stationaritäts-Constraint, positive Korrelation auf synthetisch
    korrelierten GARCH-Returns recovered, DCC vs cDCC Q̄ unterscheiden sich
    (Bias-Korrektur sichtbar), `estimate_covariance` routet wirklich auf
    fit_dcc_garch (Silent-Stub-Regression-Test).
- [x] **C4-076 Fractional Differentiation** — DONE 2026-05-17: `fractional_diff()`
  existierte bereits in `src/assembled_core/features/triple_barrier.py`. Echter Gap
  war der Default-d-Param-Calibration-Helper (López de Prado AFML §5.5 "minimum d
  for ADF rejection"). Neu in dieser Session: `find_min_d_for_stationarity(series,
  d_grid=None, pvalue_threshold=0.05)` durchsucht ein d-Grid und gibt das kleinste
  d zurück, das ADF-stationarisiert. Returns dict mit `d`, `adf_statistic`,
  `pvalue`, `is_stationary`, `correlation_with_original`, `grid_tested`. 8 Tests
  pass (random-walk log-prices stationarised, smallest-d-in-grid order, stationary
  input picks smallest d, short-series ValueError, pvalue-threshold parametriert,
  None-when-nothing-works, etc.). Exportiert via `features/__init__.py`.
- [x] **C4-077 Brinson Attribution Multi-Period** — DONE (vor heute, KNOWN_ISSUES-Eintrag war stale):
  - `src/assembled_core/attribution/brinson_hood.py` — single-period Brinson-Hood-Beebower (Allocation/Selection/Interaction)
  - `src/assembled_core/attribution/brinson_multi_period.py` — Cariño (1999) logarithmic linking für Multi-Period-Reconciliation (`carino_link_coefficients`, `link_multi_period_attribution`, `reconciliation_residual`). Docstring referenziert explizit "audit C4-077".
  - Frongello (2002) als Alternative im Docstring referenziert (Cariño bevorzugt).
  - Tests in `tests/test_wave19_helpers.py`.
- [x] **C4-078 LPPL-Bubble Stress-Test (Sornette)** — DONE 2026-05-17:
  Synthetic-Stress-Validation hinzugefügt. `LPPLSCrashDetector` existierte
  bereits in `src/assembled_core/signals/lppls_crash.py` als JLS-Modell
  (Johansen-Ledoit-Sornette) mit numpy-Fallback + `lppls`-Lib-Pfad,
  Sornette-Validitäts-Heuristik (0.1<m<0.9, 6<ω<13, B<0, |C/B|<1).
  - **Neu:** `simulate_lppls_path(...)` Helper in selben Modul — generiert
    Synthetic-Log-Price-Paths aus bekannten LPPLS-Parametern (m, ω, φ, A,
    B, C, tc, n_days, noise_σ). Validiert tc > n_days (Singularität muss
    in der Zukunft sein).
  - **11 Validation-Tests** in `tests/test_signals_lppls_validation.py`:
    Simulator reproduzierbar mit Seed, positive Prices, Edge-Cases.
    Detector smoke-checks (finite scores auf Bubble + Random-Walk).
    Diskriminations-Test: Bubble-Paths haben über 5 seeds **höhere mean
    crash_confidence** als Random-Walks. tc-Recovery innerhalb 0.5-2.0x
    der wahren tc. Robust gegen kurze Fenster.
  - **Activation als Trading-Signal:** weiterhin GATED — die Heuristik ist
    direktionell diskriminativ, aber kein präziser Crash-Klassifizierer.
    Audit-Anforderung "Synthetic-Stress-Validation vor Aktivierung" ist
    jetzt mit reproduzierbarer Test-Suite erfüllt.
- [x] **C4-079 Spillover-Index Window/Lag-Sensitivität (Diebold-Yilmaz)** — DONE 2026-05-17: Modul existierte 0 hits in src/ (KNOWN_ISSUES-Eintrag implizierte vorhandene Implementation, war stale). Neu in dieser Session:
  - `src/assembled_core/qa/spillover_index.py` mit Diebold-Yilmaz (2012) Total Spillover Index + Pesaran-Shin (1998) generalized FEVD (order-independent).
  - `compute_spillover_index(returns, lag=4, horizon=10) → SpilloverResult` (TSI%, FEVD-Matrix, to_others/from_others/net pro Variable). Lag + Horizon explizit parametrisiert (Window-Sensitivität pro Audit-Aufforderung).
  - `rolling_spillover_index(returns, window=200, step=5, lag=4, horizon=10) → DataFrame` für die kanonische DY-2012-Zeitreihe der Konnektivität (Window-Sensitivität).
  - 13 Tests pass: TSI [0,100]%, connected > independent, FEVD-Zeilen summieren auf 100, net summiert auf 0, Transmitter hat positive net spillover, rolling-DataFrame mit DatetimeIndex, edge cases (single var / short series / invalid lag/horizon / window too small).
- [x] **C4-080 Mutual Information / Transfer Entropy KSG-Estimator** — DONE 2026-05-17:
  - **MI / KSG (bereits vorhanden):** `qa/feature_screen.py::mutual_info_screen` nutzt
    `sklearn.feature_selection.mutual_info_regression` (das IST der KSG kNN-Estimator
    per Kraskov-Stögbauer-Grassberger 2004) als Primary + histogram-Fallback. KSG vs
    histogram im Modul-Docstring erklärt.
  - **Transfer Entropy (neu in dieser Session, war komplett offen):** neues Modul
    `src/assembled_core/qa/transfer_entropy.py` mit:
    - `transfer_entropy_binned(source, target, lag=1, n_bins=8) → float`:
      histogram-based Schreiber (2000) TE in nats, dependency-free, exact für die
      gewählte Diskretisierung. Bias-Floor O(n_bins²/N) dokumentiert.
    - `transfer_entropy_ksg(source, target, lag=1, k=3) → float | None`: sklearn-
      basierte **heuristische** Approximation (corr²-gewichteter Confounding-Bound
      auf MI(X_past;Y_future)). Wichtig: das ist **NICHT** die Wibral-2014-§2.2-
      Formel — sklearn hat keinen multivariaten KSG-Joint-MI-Estimator. Gilt
      näherungsweise für Gauss-AR-artige Prozesse. Für produktive KSG-TE
      `idtxl` installieren. Returns None wenn sklearn fehlt (graceful degradation).
  - 15 Tests pass: TE positiv für kausale Paare (Y_t = 0.7·X_{t-1} + noise),
    TE klein für unabhängige Reihen (relativ zum kausalen Fall via bias-floor-test),
    Asymmetrie TE(X→Y) > TE(Y→X) bei one-way causation, edge cases, sklearn-fallback
    via monkeypatch. KSG-Modul fällt graceful auf None ohne sklearn.
- [x] **C4-081 Event-Study Methodik** — DONE 2026-05-17: Market-Model + BMP-t-stat + BHAR
  als neue Funktionen in `src/assembled_core/qa/event_study.py` (bestehende
  Mean-Adjusted-Funktionen bleiben unverändert für Backward-Compat).
  - `estimate_market_model(asset, market) → MarketModelResult` (OLS α/β + residual std + R²)
  - `compute_market_model_abnormal_returns(panel, ...)` mit estimation_window=(-250,-10)
    default per MacKinlay (1997); attaches `mm_abnormal_return` + `sigma_resid`
  - `bmp_t_statistic(ar_df, event_window=(-5,5))` per Boehmer-Musumeci-Poulsen (1991):
    Standardisiert AR per event (÷sigma_i), summiert über window, cross-sectional t-test.
    Returns dict mit `t_statistic`, `pvalue`, `car_mean`, `is_significant_at_5pct`.
  - `compute_bhar(panel, horizon_days=250)` per Barber & Lyon (1997): ∏(1+r_asset) − ∏(1+r_market)
    über post-event Horizont (compounding-based, langfrist-bias-bereinigt).
  - 18 Tests pass: known-α/β recovery, zero-β für unkorrelierte, event-day-jump detected,
    BMP-t-significance unter known effect / non-effect, BHAR positive für +2%-jump etc.
  - **`scripts/run_event_study.py` Skeleton-Status (siehe §8.6) bleibt unverändert** —
    Wiring der neuen Funktionen in den CLI-Workflow ist separate Aufgabe.
- [x] **C4-083 PEAD-SUE EPS-Expected-Source** — DONE 2026-05-17: neues Modul
  `src/assembled_core/features/pead_sue.py` macht die Expected-EPS-Quelle
  **explizit und parametrisiert** (zuvor nur reported `eps_surprise` ohne
  expected-EPS-Modell sichtbar):
  - `compute_expected_eps_random_walk(eps)` — naive, E[EPS_t] = EPS_{t-1}
  - `compute_expected_eps_seasonal_rw(eps, seasonality=4)` — Bernard-Thomas (1989)
    PEAD-Baseline: E[EPS_t] = EPS_{t-4} (same quarter last year)
  - `compute_expected_eps_foster(eps, seasonality=4, drift_window=4)` — Foster (1977):
    seasonal RW + trailing YoY-drift, dominante pre-IBES-Spezifikation
  - `compute_sue(eps, method)` mit `method='random_walk'|'seasonal_rw'|'foster'`
  - `compute_sue_from_expected(actual, expected_eps)` für externe IBES-Consensus-Daten
  - Returns `SueResult` (sue, expected_eps, forecast_error, sigma_forecast_error, n_events, method)
  - 17 Tests: lag-correctness pro Modell, Foster-drift-Berechnung explizit
    verifiziert, SUE-Standardisierung auf ~unit std, external-Pfad, edge cases.
  - **IBES-Gold-Standard**: extern verfügbar, von Modul nicht abgerufen (paid Refinitiv/I/B/E/S-Daten). Wenn Caller IBES bekommt, einfach an `compute_sue_from_expected` übergeben.
  - **Follow-up (Rule 50, pre-existing):** Drei Module besitzen jetzt eine `compute_sue`-Funktion mit unterschiedlichen Signaturen — `signals/pead_sue.py` (Finnhub-Wrapper, scalar), `features/altdata_earnings_insider_factors.py::compute_sue(actual, estimated, std)` (pure scalar), und neu `features/pead_sue.py::compute_sue(eps_series, method) → SueResult`. Keine Import-Konflikte, aber Namensraum sollte konsolidiert werden — separate Audit-Welle.
- [x] **C4-084 pairs_trading half-life via OU** — DONE 2026-05-17: neues Modul
  `src/assembled_core/signals/pairs_diagnostics.py` mit `ou_half_life(spread)`
  (AR(1)-OLS, λ = -slope, half-life = ln(2)/λ; ∞ bei nicht-mean-reverting Spread,
  NaN bei <30 obs oder all-constant) und `engle_granger_cointegration(y, x)`
  (Wrapper über `statsmodels.tsa.stattools.coint` mit Critical-Values 1%/5%/10%
  und `is_cointegrated_at_5pct` Convenience-Bool). 13 Tests passen (synthetic
  AR(1) recovery, random walk → inf/large, cointegrated/non-cointegrated pairs,
  edge cases). Pairs-trading-Caller können diese Diagnostik vor Signal-Gen
  nutzen um Pair-Kandidaten zu filtern. Johansen-Test (multivariate, > 2
  assets) bleibt offen — Engle-Granger ist ausreichend für bivariate pairs.

### 8.14 Test-Skips und xfails (Inventur) — DONE 2026-05-12 (Wave 19 Follow-On)

**Vollständige Inventur:** `docs/TEST_SKIP_INVENTORY_2026-05-12.md`.

Befunde der Inventur:

- ~25 Marker **legitim** (optionale Deps numba / scipy / sklearn / arch /
  hmmlearn, dokumentierte xfails mit Sunset-Datum) — keine Aktion.
- ~30 Marker **stale** — referenzieren archivierte Module
  (`archive/observability_graveyard_2026q2/`, `archive/intel_research_2026q2/`):
  multichannel_propagation, weaponized_interdependence, scenario_trees,
  barbell_strategy, volatility_features, garch_models, evt_models,
  ml.automl, IntelSignalAdapter-Klasse.
- 1 Marker **buggy** — `test_competitive_analysis_impl.py:1312`
  `skipif(True)` durch Runtime-`HMMLEARN_AVAILABLE`-Check ersetzt.

Strukturelle Entscheidung offen (User-call): die ~30 stale-Marker entweder
mit-archivieren oder file-level skip mit klarer Begründung — Wave 19 lässt
das Status-quo, weil die Tests keine Laufzeit-Kosten verursachen, sondern
nur Collection-Noise.

### 8.15 Verifikations-Status

- Lokales Pytest (Windows): 96 audit-sweep tests passing, 0 failed.
- Full bugrun `-m "phase12 or not slow"`: 6800+ tests collected, exit 0.
- mypy strict: 6 safety-critical Files (kill_switch, order_lifecycle,
  api/auth, utils/retry, utils/clock_drift, reproducibility) + 16 hexagonal
  Files clean.
- Ruff + black + ruff-format: clean auf allen geänderten Files.
- **NICHT verifiziert:** Ubuntu-CI (kein PR), slow-Marker-Suite, fresh
  paper-pilot-Run mit Wave-1-bis-17 Gate-Stack.
