# Model Inventory & Governance

## Übersicht

Dieses Model Inventory dient der systematischen Dokumentation und Governance aller Modelle, Strategien und kritischer Komponenten im Assembled Trading AI System. Im Stil von Bank- und Hedgefonds-Standards werden hier Modelle kategorisiert, verantwortlich zugeordnet und ihre Risikorelevanz klassifiziert.

**Zweck:**
- Transparenz über alle verwendeten Modelle/Strategien
- Klare Zuordnung von Verantwortlichkeiten
- Risikoklassifikation (Tier-System)
- Nachverfolgbarkeit von Validierungen und Änderungen
- Compliance und Audit-Freundlichkeit

---

## Model Inventory Tabelle

| model_id | type | description | tier | owner | last_validation | status |
|----------|------|-------------|------|-------|-----------------|--------|
| trend_baseline | STRATEGY | EMA-basierte Trend-Following-Strategie (Fast/Slow Moving Average Crossover). Generiert BUY/SELL-Signale basierend auf technischer Analyse. | TIER_1 | Hans | n/a | active |
| event_insider_shipping | STRATEGY | Event-basierte Strategie kombiniert Insider-Trading-Daten (Net-Buy-Flows) und Shipping-Congestion-Daten. Generiert Signale basierend auf fundamentalen/Event-Signalen. | TIER_2 | Hans | n/a | pilot |
| qa_metrics | QA_METRICS | Zentrale Performance-Metriken-Berechnung (Sharpe, Sortino, CAGR, Max Drawdown, VaR, etc.) aus Equity-Kurven und Trade-Daten. | TIER_1 | Hans | n/a | active |
| qa_gates | QA_METRICS | QA-Gates evaluieren Performance-Metriken und bestimmen OK/WARNING/BLOCK Status für Backtests/Portfolios. Strukturierte Qualitätsprüfungen vor Deployment. | TIER_1 | Hans | n/a | active |
| risk_metrics | RISK | Portfolio-Risiko-Metriken (VaR 95%, Expected Shortfall 95%, Volatilität, Max Drawdown). Historische Risikoberechnung aus Equity-Kurven. | TIER_1 | Hans | n/a | active |
| scenario_engine | SCENARIO | Szenario-Engine für Stress-Tests (Equity Crash, Volatility Spike, Shipping Blockade). Simuliert adverse Marktbedingungen auf Preis- und Equity-Daten. | TIER_2 | Hans | n/a | active |
| shipping_risk | RISK | Shipping-Exposure- und Systemic-Risk-Analyse. Berechnet Portfolio-gewichtete Shipping-Congestion-Metriken und generiert Risk-Flags (LOW/MEDIUM/HIGH). | TIER_2 | Hans | n/a | active |
| labeling | ML_TOOLING | Trade- und Equity-Curve-Labeling für ML-Datasets. Berechnet Labels (0/1) basierend auf P&L-Thresholds oder Forward-Returns über definierte Horizonte. | TIER_3 | Hans | n/a | active |
| dataset_builder | ML_TOOLING | ML-Dataset-Builder verheiratet Features (TA, Insider, Shipping, Congress, News) mit Trade-Labels. Exportiert strukturierte Parquet-Datasets für Modell-Training. | TIER_3 | Hans | n/a | active |
| backtest_engine | PIPELINE | Portfolio-Level-Backtest-Engine. Orchestriert Feature-Computation, Signal-Generation, Position-Sizing, Order-Generation und Equity-Simulation. | TIER_1 | Hans | n/a | active |
| walk_forward | PIPELINE | Walk-Forward-Analyse für Time-Series-Cross-Validation. Splittet Daten in Train/Test-Windows und führt wiederholte Backtests durch. | TIER_2 | Hans | n/a | active |

---

## Tier-Klassifikation

### TIER_1: Produktiv relevante Kernmodelle

- Direkt in Trading-Entscheidungen involviert
- Erfordert strenge Validierung und Monitoring
- Änderungen müssen dokumentiert und reviewed werden
- **Beispiele:** `trend_baseline`, `qa_metrics`, `qa_gates`, `risk_metrics`, `backtest_engine`

### TIER_2: Unterstützende Tools & Risiko-Analysen

- Wird für Analyse und Risikobewertung verwendet
- Nicht direkt in Trade-Execution involviert
- Moderate Validierungsanforderungen
- **Beispiele:** `event_insider_shipping`, `scenario_engine`, `shipping_risk`, `walk_forward`

### TIER_3: Research & Experimental

- Entwicklung und Experimentation
- Noch nicht produktiv eingesetzt
- Minimal Validierungsanforderungen
- **Beispiele:** `labeling`, `dataset_builder`

---

## Status-Legende

- **active**: In produktivem Einsatz oder regelmäßiger Verwendung
- **pilot**: In Testphase, begrenzter Einsatz
- **retired**: Nicht mehr aktiv verwendet (historische Dokumentation)

---

## Validierung

**last_validation:** Datum der letzten formalen Validierung (YYYY-MM-DD) oder "n/a" wenn noch nicht validiert.

**Zukünftige Validierungsaktivitäten:**
- Regelmäßige Backtesting auf Out-of-Sample-Daten
- Performance-Monitoring vs. historische Benchmarks
- Sensitivitäts-Analysen (Parameter-Robustheit)
- Scenario-Testing unter verschiedenen Marktbedingungen

---

## Weitere Informationen

- **Model Cards:** Detaillierte Dokumentation pro Modell in `docs/models/`
- **Change History:** Änderungen werden in Model Cards dokumentiert
- **Governance:** Regelmäßige Reviews und Updates des Inventories

