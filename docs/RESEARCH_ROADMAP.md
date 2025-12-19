# Research Roadmap – Assembled Trading AI

**Letzte Aktualisierung:** 2025-01-15  
**Status:** Phase 12.1 & 12.2 – Research-Prozess, Roadmap & Experiment-Tracking (fertig)

---

## Bestandsaufnahme: Aktueller Stand

**Bestehende Strategien & Layer:**
- **Trend Baseline**: EMA-basierte Trend-Strategie (`signals/rules_trend.py`)
  - Fast/Slow Moving Average Crossover
  - Konfigurierbare MA-Windows (z.B. 20/50, 10/30)
  - Volume-Filter optional
- **Event Insider Shipping**: Event-basierte Strategie (`signals/rules_event_insider_shipping.py`)
  - Insider-Trading-Signale (net_buy_20d)
  - Shipping-Congestion-Signale (congestion_score_7d)
  - Kombiniert Insider + Shipping für LONG/SHORT-Signale
- **ML-Meta-Layer** (Phase 7):
  - Labeling: `qa/labeling.py` (binary_absolute, binary_outperformance, multi_class)
  - ML-Dataset-Builder: `qa/dataset_builder.py`
  - Meta-Modelle: `signals/meta_model.py` (GradientBoosting, RandomForest)
  - Ensemble-Layer: `signals/ensemble.py` (Filter-Modus, Scaling-Modus)
- **Risk-Engine** (Phase 8):
  - Risk-Metriken: `qa/risk_metrics.py` (VaR, CVaR, Volatility, Drawdown)
  - Scenario-Engine: `qa/scenario_engine.py` (Stress-Tests, Regime-Shifts)
  - Shipping-Risk: `qa/shipping_risk.py` (Sektor-spezifische Risiken)
- **QA & Governance** (Phase 9):
  - QA-Gates: `qa/qa_gates.py` (Performance-Thresholds, Trade-Counts)
  - Model-Validation: `qa/validation.py` (Performance-Validation, Overfitting-Checks)
  - Drift-Detection: `qa/drift_detection.py` (Feature-Drift, Label-Drift, Performance-Drift)
  - Model-Inventory: `docs/MODEL_INVENTORY.md`
- **Paper-Trading & OMS** (Phase 10):
  - Paper-Trading-Engine: `execution/paper_trading_engine.py`
  - Pre-Trade-Checks: `execution/pre_trade_checks.py`
  - Kill-Switch: `execution/kill_switch.py`
  - OMS-Light: `api/routers/oms.py` (Blotter, Executions, Routing)

**Bestehende Research-Tools:**
- Backtest-Engine: `qa/backtest_engine.py` (Portfolio-Level-Backtests mit Kostenmodellen)
- QA-Reports: `reports/daily_qa_report.py` (Performance-Metriken, Sharpe, Drawdown, Trades)
- ML-Dataset-Export: CLI `build_ml_dataset` (Features + Labels für ML-Experimente)
- Meta-Model-Training: CLI `train_meta_model` (Confidence-Score-Prediction)
- Monitoring-API: `api/routers/monitoring.py` (QA-Status, Risk-Status, Drift-Status)

**Bisherige Research-Aktivitäten:**
- Backtests für Trend-Baseline und Event-Strategien
- QA-Reports für Performance-Validierung
- ML-Dataset-Building für Meta-Model-Training
- Meta-Model-Experimente (GradientBoosting vs. RandomForest)
- Ensemble-Layer-Tests (Filter vs. Scaling)

---

## 1. Zielbild

**Research** in Assembled Trading AI bedeutet:

- **Systematische Exploration** neuer Trading-Ideen, Strategien und Datenquellen
- **Kontinuierliche Verbesserung** bestehender Strategien durch Parameter-Optimierung, Feature-Engineering und Meta-Learning
- **Rigorous Testing** neuer Ansätze mit Backtests, Walk-Forward-Analysen und Out-of-Sample-Validierung
- **Dokumentation & Tracking** aller Experimente für Reproduzierbarkeit und Lernen

**Bezug auf Phase 12 der Backend-Roadmap:**
Phase 12 ("God-Level" Research & Evolution) zielt darauf ab, das System zu einem **kontinuierlich lernenden System** zu machen, das:
- Neue Strategien systematisch evaluiert
- Bestehende Strategien kontinuierlich verbessert
- Neue Datenquellen und Faktoren integriert
- ML-Experimente strukturiert durchführt
- Research-Ergebnisse nachvollziehbar dokumentiert

---

## 2. Aktueller Stand

### 2.1 Implementierte Strategien & Layer

**Trend-Strategien:**
- ✅ Trend Baseline (EMA-Crossover)
- ⏳ Mean-Reversion (geplant)
- ⏳ Breakout-Strategien (geplant)
- ⏳ Multi-Timeframe-Trend (geplant)

**Event-Strategien:**
- ✅ Event Insider Shipping (Insider + Shipping)
- ✅ Earnings & Insider Alt-Data Factors (Phase B1) - Earnings Surprise & Insider Activity Faktoren verfügbar
- ✅ News & Macro Alt-Data Factors (Phase B2) - News Sentiment & Macro Regime Faktoren verfügbar
- ⏳ Congress-Trading (geplant)

**ML-Layer:**
- ✅ Meta-Modelle (Confidence-Score-Prediction)
- ✅ Ensemble-Layer (Filter/Scaling)
- ⏳ Feature-Selection (geplant)
- ⏳ Explainability (SHAP, Feature-Importance) (geplant)

**Risk & QA:**
- ✅ Risk-Metriken (VaR, CVaR, Volatility)
- ✅ Scenario-Engine (Stress-Tests)
- ✅ QA-Gates (Performance-Thresholds)
- ✅ Drift-Detection (Feature/Label/Performance)
- ✅ Model-Validation (Overfitting-Checks)

**Infrastruktur:**
- ✅ Backtest-Engine (Portfolio-Level mit Kosten)
- ✅ QA-Reports (Performance-Analysen)
- ✅ ML-Dataset-Export (Features + Labels)
- ✅ Paper-Trading-API (Simulation)
- ✅ OMS-Light (Blotter, Executions)
- ⏳ Walk-Forward-Analyse (geplant)
- ⏳ Experiment-Tracking (geplant, Sprint 12.2)

### 2.2 Verfügbare Tools für Research

**Backtesting:**
- Portfolio-Level-Backtests mit konfigurierbaren Kostenmodellen
- Support für 1d und 5min Frequenzen
- Equity-Curve-Generierung mit täglichen Returns
- Trade-Level-Analysen

**ML-Experimente:**
- ML-Dataset-Building aus Backtest-Ergebnissen
- Meta-Model-Training (GradientBoosting, RandomForest)
- Ensemble-Layer für Signal-Filterung/Scaling
- Calibration-Plots für Modell-Validierung

**Reports & Monitoring:**
- QA-Reports mit Performance-Metriken (Sharpe, Drawdown, CAGR, Trades)
- Monitoring-API für QA-Status, Risk-Status, Drift-Status
- Portfolio-Reports mit Kosten-Analysen

---

## 3. Fokus 3–6 Monate

### 3.1 Neue Ideen & Strategien

**Trend-Varianten:**
- Multi-Timeframe-Trend (z.B. 1d + 5min Kombination)
- Adaptive Moving Averages (z.B. KAMA, TEMA)
- Trend-Strength-Indikatoren (ADX, Trend-Filter)
- Breakout-Strategien (z.B. Bollinger-Band-Breakouts, Support/Resistance)

**Mean-Reversion:**
- RSI-basierte Mean-Reversion (Oversold/Overbought)
- Bollinger-Band-Mean-Reversion
- Pairs-Trading (Korrelation-basiert)

**Intraday-Varianten:**
- 5min-spezifische Strategien (z.B. Opening-Range-Breakouts)
- Volume-Profile-Strategien
- Time-of-Day-Patterns (z.B. End-of-Day-Momentum)

**Options-/Vol-Strategien:**
- Implied-Volatility-Rankings
- Volatility-Spread-Strategien
- Options-Flow-Analysen (falls Daten verfügbar)

### 3.2 Neue Alt-Daten & Faktoren

**Alt-Data-Faktoren (Phase B1 - ✅ Completed):**
- **Earnings Surprise Faktoren**: `earnings_eps_surprise_last`, `earnings_revenue_surprise_last`, `post_earnings_drift_return_{window_days}d`
- **Insider Activity Faktoren**: `insider_net_notional_{lookback_days}d`, `insider_buy_count_{lookback_days}d`, `insider_sell_count_{lookback_days}d`, `insider_buy_sell_ratio_{lookback_days}d`
- **Integration mit Phase C1/C2**: Alt-Data-Faktoren können mit `analyze_factors --factor-set core+alt` evaluiert werden
- **Integration mit Phase C3**: Events können für Event-Studies verwendet werden (`research/events/event_study_template_core.py`)
- **Dokumentation**: Siehe `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md` (Phase B1) und `docs/WORKFLOWS_FACTOR_ANALYSIS.md`

**Alt-Data-Faktoren 2.0 (Phase B2 - ✅ Completed):**
- **News Sentiment Faktoren**: `news_sentiment_mean_{lookback_days}d`, `news_sentiment_trend_{lookback_days}d`, `news_sentiment_shock_flag`, `news_sentiment_volume_{lookback_days}d`
- **Macro Regime Faktoren**: `macro_growth_regime`, `macro_inflation_regime`, `macro_risk_aversion_proxy`
- **Integration mit Phase C1/C2**: News/Macro-Faktoren können mit `analyze_factors --factor-set core+alt_news` oder `core+alt_full` evaluiert werden
- **Integration mit Phase D**: Macro-Regime-Faktoren werden für Regime-Modelle verwendet
- **Dokumentation**: Siehe `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md` (Phase B2) und `docs/WORKFLOWS_FACTOR_ANALYSIS.md`

**Erweiterte Insider-Daten (Zukunft - B2):**
- Insider-Transaction-Types (Buy/Sell, Open-Market, etc.)
- Insider-Cluster-Analysen (mehrere Insiders gleichzeitig)
- Insider-Historie (wie oft hat dieser Insider erfolgreich getradet?)

**Congress-Trading:**
- Congress-Member-Trades (STOCK Act Daten)
- Timing-Analysen (Trades vor wichtigen Entscheidungen)
- Sector-Exposure-Analysen

**News & Sentiment:**
- ✅ News-Sentiment-Faktoren (Phase B2) - `news_sentiment_mean_{lookback_days}d`, `news_sentiment_trend_{lookback_days}d`, `news_sentiment_shock_flag`
- News-Volume-Spikes (kann über `news_sentiment_volume_{lookback_days}d` analysiert werden)
- Earnings-Announcements (Pre/Post-Earnings-Drifts) ✅ Implementiert in Phase B1

**Makro-Daten:**
- ✅ Macro-Regime-Faktoren (Phase B2) - `macro_growth_regime`, `macro_inflation_regime`, `macro_risk_aversion_proxy`
- Economic-Indicators (z.B. CPI, Unemployment) - Verfügbar über `fetch_macro_series()` und `build_macro_regime_factors()`
- Fed-Announcements (FOMC-Meetings) - Kann über Macro-Series integriert werden
- Sector-Rotation-Indikatoren (geplant)

**Vol-Daten:**
- VIX-Term-Structure
- Sector-Vol-Rankings
- Realized-Vol vs. Implied-Vol

### 3.3 ML-Experimente

**Verbesserte Labeling-Schemata:**
- Binary-Outperformance (vs. SPY, vs. Sector)
- Multi-Class-Labeling (Strong Buy, Buy, Hold, Sell, Strong Sell)
- Time-Weighted-Labels (frühe Exits vs. späte Exits)
- Risk-Adjusted-Labels (Sharpe-basiert statt Return-basiert)

**Weitere Meta-Model-Typen:**
- Neural Networks (z.B. einfache MLPs)
- XGBoost/LightGBM (falls verfügbar)
- Ensemble von Meta-Modellen (Stacking, Voting)

**Feature-Selection:**
- Univariate-Feature-Selection (z.B. Mutual-Information)
- Recursive-Feature-Elimination (RFE)
- L1-Regularization (Lasso)

**Explainability:**
- SHAP-Values für Meta-Modelle
- Feature-Importance-Plots
- Partial-Dependence-Plots

**Tests:**
- "Meta-Layer vs. rein regelbasiert" unter verschiedenen Märkten/Regimes
- Walk-Forward-Analysen für Meta-Modelle
- Out-of-Sample-Validierung mit Time-Series-Splits

### 3.4 Infrastruktur & Tooling

**Schnelleres Backtesting:**
- Parallelisierung von Backtests (Multi-Processing)
- Caching von Features (wiederverwendbare Feature-Berechnungen)
- Incremental-Backtests (nur neue Daten verarbeiten)

**Mehr Reports:**
- Strategy-Comparison-Reports (mehrere Strategien nebeneinander)
- Regime-Analysis-Reports (Performance in verschiedenen Markt-Regimes)
- Feature-Importance-Reports (welche Features tragen am meisten bei?)

**Bessere Visualisierung:**
- Equity-Curve-Plots mit Drawdowns
- Trade-Distribution-Plots (Win-Rate, Avg-Win/Loss)
- Feature-Correlation-Matrix
- Calibration-Plots für Meta-Modelle

**Experiment-Tracking:**
- Strukturierte Experiment-Logs (Hypothese, Setup, Ergebnisse)
- Versionierung von Experimenten (Git-Tags, Experiment-IDs)
- Vergleich von Experimenten (A/B-Testing-Style)

---

## 4. Konkrete Research-Tasks (Backlog)

| ID | Titel | Kategorie | Priorität | Beschreibung |
|---|---|---|---|---|
| R-001 | Mean-Reversion-Strategie (RSI) | Strategie | High | RSI-basierte Mean-Reversion-Strategie implementieren und backtesten |
| R-002 | Multi-Timeframe-Trend | Strategie | High | Kombination von 1d- und 5min-Trend-Signalen |
| R-003 | Congress-Trading-Features | Daten | High | Congress-Member-Trades als Feature integrieren |
| R-004 | News-Sentiment-Scoring | Daten | Medium | FinBERT oder ähnliches für News-Sentiment verwenden |
| R-005 | Binary-Outperformance-Labeling | ML | High | Outperformance-Labeling vs. SPY/Sector implementieren |
| R-006 | SHAP-Explainability | ML | Medium | SHAP-Values für Meta-Modelle berechnen und visualisieren |
| R-007 | Walk-Forward-Analyse | Infra | ✅ Completed (B3) | Walk-Forward-Analyse-Tool implementiert (Phase B3: Walk-Forward & Regime Analysis) |
| R-008 | Strategy-Comparison-Reports | Infra | Medium | Reports für Vergleich mehrerer Strategien |
| R-009 | Feature-Selection-Pipeline | ML | Medium | Automatische Feature-Selection für Meta-Modelle |
| R-010 | Regime-Analysis | Risk | ✅ Completed (B3) | Markt-Regime-Erkennung und Regime-spezifische Performance implementiert (Phase B3) |
| R-011 | Adaptive-Position-Sizing | Portfolio | Low | Kelly-Criterion oder ähnliches für Position-Sizing |
| R-012 | Options-Vol-Strategien | Strategie | Low | Implied-Volatility-basierte Strategien (falls Daten verfügbar) |
| R-013 | Pairs-Trading | Strategie | Low | Korrelations-basierte Pairs-Trading-Strategie |
| R-014 | Earnings-Events | Daten | ✅ Completed (B1) | Earnings-Announcements als Feature/Event integriert (Phase B1: Alt-Data Factors) |
| R-015 | XGBoost-Meta-Model | ML | Low | XGBoost als Alternative zu GradientBoosting testen |
| R-016 | Deflated-Sharpe-Integration | Infra | ✅ Completed (B4) | Deflated Sharpe Ratio für Multiple-Testing-Adjustierung implementiert (Phase B4) |

---

## 5. Arbeitsweise

### 5.1 Research-Experiment-Workflow

**1. Hypothese formulieren:**
- Klare Frage: "Verbessert RSI-Mean-Reversion die Sharpe-Ratio gegenüber Trend-Baseline?"
- Erwartetes Ergebnis: z.B. "RSI-Strategie sollte in Seitwärts-Märkten besser performen"

**2. Setup definieren:**
- Daten: Welche Symbole, welcher Zeitraum?
- Strategie-Parameter: z.B. RSI-Perioden, Oversold/Overbought-Thresholds
- Backtest-Konfiguration: Start-Capital, Kosten-Modell, Frequenz

**3. Experiment durchführen:**
- Code in `research/` Ordner (Notebook oder Script)
- Backtest ausführen
- Ergebnisse sammeln (Equity-Curve, Metriken, Trades)

**4. Auswertung:**
- Performance-Metriken analysieren (Sharpe, Drawdown, Win-Rate)
- Vergleich mit Baseline-Strategie
- Visualisierung (Equity-Curves, Trade-Distribution)

**5. Dokumentation:**
- Ergebnisse in Notebook/Script dokumentieren
- Fazit: War die Hypothese korrekt? Was haben wir gelernt?
- Nächste Schritte: Was sollte als nächstes getestet werden?

### 5.2 Experiment-Struktur (Notebook/Script)

Ein typisches Research-Experiment sollte folgende Abschnitte haben:

```markdown
# Experiment: [Titel]

## Hypothese
[Was testen wir? Was erwarten wir?]

## Setup
- Daten: [Symbole, Zeitraum]
- Strategie: [Parameter, Konfiguration]
- Backtest: [Start-Capital, Kosten, Frequenz]

## Methode
[Code für Backtest, Feature-Berechnung, etc.]

## Ergebnisse
[Performance-Metriken, Visualisierungen]

## Fazit
[War die Hypothese korrekt? Was haben wir gelernt?]

## Nächste Schritte
[Was sollte als nächstes getestet werden?]
```

### 5.3 Experiment-Tracking (Sprint 12.2, ✅ fertig)

**Status:** ✅ Implementiert

Ein leichtgewichtiges Experiment-Tracking-System wurde in Sprint 12.2 implementiert (`src/assembled_core/qa/experiment_tracking.py`).

**Speicherort:**
- Runs werden in `experiments/` gespeichert (konfigurierbar via `settings.experiments_dir`)
- Jeder Run erhält einen eindeutigen Ordner: `experiments/{run_id}/`

**Struktur pro Run:**
- `run.json`: Metadaten (run_id, name, config, tags, status, created_at, finished_at)
- `metrics.csv`: Zeitreihen-Metriken (step, timestamp, metric_name, metric_value)
- `artifacts/`: Kopien wichtiger Dateien (Reports, Plots, Model-Files)

**Verwendung:**
- **Backtests:** `python scripts/cli.py run_backtest --track-experiment --experiment-name "trend_ma20_50" --experiment-tags "trend,baseline"`
- **Meta-Model-Training:** `python scripts/cli.py train_meta_model --track-experiment --experiment-name "meta_gb_v1" --experiment-tags "meta_model,gradient_boosting"`

**Auswertung:**
- Runs auflisten: `tracker.list_runs(tags=["trend"])`
- Metriken laden: `tracker.get_run_metrics(run)` → DataFrame
- Run-Ordner direkt durchsuchen: `experiments/{run_id}/`

**Vorteile:**
- Keine externen Services nötig (nur lokale Dateien)
- Einfache Integration in bestehende CLI-Commands
- Reproduzierbar und nachvollziehbar
- Einfache Auswertung mit pandas

**Verweis:** Siehe `docs/BACKEND_ROADMAP.md` Phase 12, Sprint 12.2

---

## 6. Advanced Analytics & Factor Labs

For a comprehensive roadmap on advanced factor development, technical analysis indicators, alternative data integration, and factor research tools, see:

- **[Advanced Analytics & Factor Labs Roadmap](ADVANCED_ANALYTICS_FACTOR_LABS.md)** - Extended roadmap for Phase A–E (Technical Analysis Factors, Alt-Data Factors 2.0, Factor Analysis, Regime Models, ML Validation & Explainability)

The Factor Labs roadmap extends the research capabilities described in this document with systematic tools for:
- Technical analysis factor library development
- Enhanced alternative data integration (congressional trading, news sentiment)
- Factor evaluation and selection frameworks
- Regime-aware risk modeling
- Model explainability and transaction cost analysis

---

## 7. Verknüpfungen

- **Backend-Roadmap**: `docs/BACKEND_ROADMAP.md` (Phase 12)
- **Advanced Analytics & Factor Labs**: `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md` (Phases A–E)
- **Architektur**: `docs/ARCHITECTURE_BACKEND.md`
- **Model-Inventory**: `docs/MODEL_INVENTORY.md`
- **Research-Ordner**: `research/README.md`

---

**Nächste Schritte:**
1. ✅ Research-Roadmap erstellt (Sprint 12.1)
2. ⏳ Experiment-Tracking implementieren (Sprint 12.2)
3. ⏳ Erste Research-Experimente durchführen (Sprint 12.3+)

