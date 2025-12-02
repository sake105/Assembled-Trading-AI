# Phase 6: Event-basierte Features und Strategien

**Letzte Aktualisierung:** 2025-01-15

## Überblick

Phase 6 erweitert das Trading-System um **Event-basierte Features** aus verschiedenen Datenquellen:
- Insider-Trading-Daten (SEC Form 4 Filings)
- Congressional-Trading-Daten (STOCK Act Filings)
- Shipping-Route-Daten (Hafen-Congestion, Schiffsbewegungen)
- News-Sentiment-Daten (Finanznachrichten, Pressemitteilungen)

Diese Features ermöglichen **alternative Strategien**, die nicht nur auf Preis-/Volumen-Daten basieren, sondern auch externe Events und Sentiment-Daten nutzen.

---

## Daten-Ingest-Module

### 1. Insider-Trading (`assembled_core.data.insider_ingest`)

**Zweck**: Laden und Normalisierung von Insider-Trading-Daten.

**Funktionen**:
- `load_insider_sample(path=None)`: Lädt Insider-Daten aus Datei oder generiert Dummy-Daten
- `normalize_insider(df)`: Normalisiert Insider-Daten zu Standardformat

**Datenformat**:
- `timestamp`: UTC-Timestamp des Trades
- `symbol`: Aktien-Symbol
- `trades_count`: Anzahl der Trades
- `net_shares`: Netto-Aktien gekauft/verkauft (positiv = Kauf, negativ = Verkauf)
- `role`: Insider-Rolle (z.B. "CEO", "CFO", "Director")

**Aktueller Status**: ✅ Skeleton implementiert, verwendet Dummy-Daten

**Zukünftige Integration**: SEC Form 4 Filings, InsiderMonkey API

---

### 2. Congressional-Trading (`assembled_core.data.congress_trades_ingest`)

**Zweck**: Laden und Normalisierung von Congressional-Trading-Daten.

**Funktionen**:
- `load_congress_sample(path=None)`: Lädt Congress-Daten aus Datei oder generiert Dummy-Daten
- `normalize_congress(df)`: Normalisiert Congress-Daten zu Standardformat

**Datenformat**:
- `timestamp`: UTC-Timestamp des Trades
- `symbol`: Aktien-Symbol
- `politician`: Name des Politikers
- `party`: Partei (z.B. "Democrat", "Republican")
- `amount`: Transaktionsbetrag (positiv = Kauf, negativ = Verkauf)

**Aktueller Status**: ✅ Skeleton implementiert, verwendet Dummy-Daten

**Zukünftige Integration**: House Stock Watcher, Senate Financial Disclosures, Quiver Quantitative API

---

### 3. Shipping-Routes (`assembled_core.data.shipping_routes_ingest`)

**Zweck**: Laden und Normalisierung von Shipping-Route-Daten.

**Funktionen**:
- `load_shipping_sample(path=None)`: Lädt Shipping-Daten aus Datei oder generiert Dummy-Daten
- `normalize_shipping(df)`: Normalisiert Shipping-Daten zu Standardformat

**Datenformat**:
- `timestamp`: UTC-Timestamp des Events
- `route_id`: Eindeutige Route-Identifikation
- `port_from`: Ursprungs-Hafen
- `port_to`: Ziel-Hafen
- `symbol`: Verknüpftes Aktien-Symbol (z.B. für Supply-Chain-Impact)
- `ships`: Anzahl der Schiffe auf dieser Route
- `congestion_score`: Congestion-Score (0-100, höher = mehr Stau)

**Aktueller Status**: ✅ Skeleton implementiert, verwendet Dummy-Daten

**Zukünftige Integration**: MarineTraffic API, Port Authority APIs

---

### 4. News & Sentiment (`assembled_core.data.news_ingest`)

**Zweck**: Laden und Normalisierung von News- und Sentiment-Daten.

**Funktionen**:
- `load_news_sample(path=None)`: Lädt News-Daten aus Datei oder generiert Dummy-Daten
- `normalize_news(df)`: Normalisiert News-Daten zu Standardformat

**Datenformat**:
- `timestamp`: UTC-Timestamp der News
- `symbol`: Verknüpftes Aktien-Symbol
- `headline`: News-Headline
- `sentiment_score`: Sentiment-Score (-1.0 bis 1.0, höher = positiver)
- `source`: News-Quelle (z.B. "Reuters", "Bloomberg")

**Aktueller Status**: ✅ Skeleton implementiert, verwendet Dummy-Daten

**Zukünftige Integration**: NewsAPI, Alpha Vantage News, Social Media APIs

---

## Feature-Module

### 1. Insider-Features (`assembled_core.features.insider_features`)

**Funktion**: `add_insider_features(prices, events)`

**Berechnete Features**:
- `insider_net_buy_20d`: Netto-Aktien gekauft in letzten 20 Tagen
- `insider_trade_count_20d`: Anzahl Insider-Trades in letzten 20 Tagen
- `insider_net_buy_60d`: Netto-Aktien gekauft in letzten 60 Tagen
- `insider_trade_count_60d`: Anzahl Insider-Trades in letzten 60 Tagen

**Verwendung**: Aggregiert Insider-Events über Lookback-Fenster und merged sie mit Preis-Daten.

---

### 2. Congressional-Features (`assembled_core.features.congress_features`)

**Funktion**: `add_congress_features(prices, events)`

**Berechnete Features**:
- `congress_trade_count_60d`: Anzahl Congress-Trades in letzten 60 Tagen
- `congress_total_amount_60d`: Gesamter Transaktionsbetrag in letzten 60 Tagen
- `congress_trade_count_90d`: Anzahl Congress-Trades in letzten 90 Tagen
- `congress_total_amount_90d`: Gesamter Transaktionsbetrag in letzten 90 Tagen

---

### 3. Shipping-Features (`assembled_core.features.shipping_features`)

**Funktion**: `add_shipping_features(prices, events)`

**Berechnete Features**:
- `shipping_congestion_score`: Aktueller Congestion-Score
- `shipping_ships_count`: Aktuelle Anzahl Schiffe
- `shipping_congestion_score_7d`: 7-Tage-Durchschnitt des Congestion-Scores
- `shipping_ships_count_7d`: 7-Tage-Summe der Schiffe

---

### 4. News-Features (`assembled_core.features.news_features`)

**Funktion**: `add_news_features(prices, events)`

**Berechnete Features**:
- `news_sentiment_7d`: 7-Tage-Durchschnitt des Sentiment-Scores
- `news_count_7d`: Anzahl News in letzten 7 Tagen
- `news_sentiment_30d`: 30-Tage-Durchschnitt des Sentiment-Scores
- `news_count_30d`: Anzahl News in letzten 30 Tagen

---

## Event-Strategien

### 1. Insider + Shipping Strategy (`event_insider_shipping`)

**Modul**: `assembled_core.signals.rules_event_insider_shipping`

**Funktion**: `generate_event_signals(prices_with_features, ...)`

**Strategie-Logik** (Proof-of-Concept):

```
LONG (+1):  insider_net_buy_20d > threshold AND shipping_congestion_score_7d < low_threshold
SHORT (-1): insider_net_buy_20d < threshold AND shipping_congestion_score_7d > high_threshold
FLAT (0):   Sonst
```

**Parameter**:
- `insider_weight`: Gewicht für Insider-Signale (default: 1.0)
- `shipping_weight`: Gewicht für Shipping-Signale (default: 1.0)
- `insider_net_buy_threshold`: Minimum Net-Buy für LONG-Signal (default: 1000.0)
- `insider_net_sell_threshold`: Maximum Net-Buy (negativ = Sell) für SHORT-Signal (default: -1000.0)
- `shipping_congestion_low_threshold`: Maximum Congestion für LONG-Signal (default: 30.0)
- `shipping_congestion_high_threshold`: Minimum Congestion für SHORT-Signal (default: 70.0)

**Aktueller Status**: ✅ Implementiert und getestet

**Verwendung**:
```bash
# Via CLI
python scripts/cli.py run_backtest --freq 1d --strategy event_insider_shipping --universe watchlist.txt

# Direkt via Script
python scripts/run_backtest_strategy.py --freq 1d --strategy event_insider_shipping --price-file data/sample/eod_sample.parquet
```

---

## Sprint 6.3 – Vergleich Trend vs Event

### Überblick

Das Vergleichs-Tool (`scripts/compare_strategies_trend_vs_event.py`) ermöglicht es, die **Trend-Baseline-Strategie** und die **Event-Insider-Shipping-Strategie** direkt auf denselben Preisdaten zu vergleichen.

### Vergleichte Strategien

1. **Trend Baseline** (`trend_baseline`):
   - EMA-basierte Trend-Strategie (Fast/Slow Moving Average Crossover)
   - Standard-Strategie für Preis-/Volumen-basierte Signale

2. **Event Insider Shipping** (`event_insider_shipping`):
   - Event-basierte Strategie mit Insider-Trading und Shipping-Congestion-Daten
   - Nutzt Phase-6-Features für alternative Signal-Generierung

### Vergleichs-Kennzahlen

Die Markdown-Tabelle (`comparison_summary.md`) enthält:

**Performance-Metriken:**
- Total Return (Gesamtrendite)
- CAGR (Compound Annual Growth Rate)
- Final PF (Final Performance Factor)
- End Equity (Endkapital)

**Risk-Metriken:**
- Sharpe Ratio (Risiko-adjustierte Rendite)
- Sortino Ratio (Downside-Risk-adjustierte Rendite)
- Max Drawdown (Maximaler Verlust)
- Volatility (Volatilität)

**Trade-Metriken:**
- Total Trades (Anzahl Trades)
- Hit Rate (Gewinnrate)
- Profit Factor (Gewinn-Faktor)
- Turnover (Portfolio-Umschlag)

### Ausführung

**Voraussetzung**: Sample-Event-Daten generieren (falls noch nicht vorhanden):

```bash
python scripts/generate_sample_event_data.py
```

**Vergleichs-Run starten:**

```bash
# Standard-Vergleich
python scripts/compare_strategies_trend_vs_event.py --freq 1d --price-file data/sample/eod_sample.parquet

# Ohne Transaktionskosten
python scripts/compare_strategies_trend_vs_event.py --freq 1d --price-file data/sample/eod_sample.parquet --no-costs

# Mit angepasstem Startkapital
python scripts/compare_strategies_trend_vs_event.py --freq 1d --price-file data/sample/eod_sample.parquet --start-capital 50000
```

**Output-Verzeichnis**: `output/strategy_compare/trend_vs_event/`

**Erstellte Dateien:**
- `comparison_summary.md`: Formatierte Markdown-Tabelle mit allen Metriken
- `comparison_summary.csv`: CSV-Datei für weitere Analysen (Excel, Python, etc.)

### Interpretation

Das Vergleichs-Tool hilft dabei:
- **Performance-Unterschiede** zwischen Preis-basierten und Event-basierten Strategien zu identifizieren
- **Risiko-Profile** zu vergleichen (Sharpe, Drawdown, Volatility)
- **Trade-Charakteristika** zu analysieren (Anzahl Trades, Hit Rate, Turnover)
- **Strategie-Auswahl** zu optimieren basierend auf historischen Daten

---

## Ausführung

### CLI-Beispiele

```bash
# Event-Strategie mit Standard-Parametern
python scripts/cli.py run_backtest --freq 1d --strategy event_insider_shipping --universe watchlist.txt --start-capital 10000

# Mit QA-Report
python scripts/cli.py run_backtest --freq 1d --strategy event_insider_shipping --universe watchlist.txt --generate-report

# Mit expliziter Preis-Datei
python scripts/cli.py run_backtest --freq 1d --strategy event_insider_shipping --price-file data/sample/eod_sample.parquet

# Ohne Transaktionskosten
python scripts/cli.py run_backtest --freq 1d --strategy event_insider_shipping --no-costs
```

### Integration in Backtest-Engine

Die Event-Strategie wird automatisch integriert:

1. **Laden der Event-Daten**: `load_insider_sample()`, `load_shipping_sample()`
2. **Feature-Berechnung**: `add_insider_features()`, `add_shipping_features()`
3. **Signal-Generierung**: `generate_event_signals()`
4. **Backtest**: Standard Backtest-Engine mit Position-Sizing und Kostenmodell

---

## Tests

**Phase-6-Test-Suite**:

```bash
# Alle Phase-6-Tests
pytest -m phase6

# Spezifische Tests
pytest tests/test_features_events_phase6.py
pytest tests/test_signals_event_phase6.py
pytest tests/test_run_backtest_strategy.py::test_run_backtest_event_insider_shipping
```

**Erwartete Ausgabe**: ~13 Tests in < 2 Sekunden

**Test-Bereiche**:
- Ingest-Module (Dummy-Daten-Generierung)
- Feature-Module (Feature-Berechnung)
- Signal-Generierung (Event-Strategien)
- Backtest-Integration (End-to-End)

---

## Future Work

### Kurzfristig

- [ ] **Echte Datenquellen anbinden**:
  - SEC EDGAR API für Insider-Daten
  - House Stock Watcher / Quiver Quantitative für Congress-Daten
  - MarineTraffic API für Shipping-Daten
  - NewsAPI / Alpha Vantage für News-Daten

- [ ] **Erweiterte Features**:
  - Time-weighted Features (recente Events wichtiger)
  - Cross-Symbol-Korrelationen
  - Role-weighted Insider-Signale (CEO vs. Director)
  - Route-spezifische Shipping-Features (China-Routen vs. EU-Routen)

- [ ] **Erweiterte Strategien**:
  - Multi-Event-Strategien (Kombination aller 4 Event-Quellen)
  - Machine-Learning-basierte Signal-Generierung
  - News-Sentiment-Strategien
  - Congress-Trading-Strategien

### Mittelfristig

- [ ] **Performance-Optimierung**:
  - Caching von Event-Daten
  - Incrementelle Feature-Updates
  - Parallele Feature-Berechnung

- [ ] **Validierung**:
  - Historical Backtesting mit echten Event-Daten
  - Walk-Forward-Analyse für Event-Strategien
  - Out-of-Sample-Testing

- [ ] **Dokumentation**:
  - Strategy-Performance-Reports
  - Feature-Importance-Analysen
  - Best-Practice-Guides

### Langfristig

- [ ] **Live-Integration**:
  - Real-time Event-Feeds
  - Automatische Feature-Updates
  - Live-Strategy-Monitoring

- [ ] **Erweiterte Datenquellen**:
  - Social Media Sentiment (Twitter, Reddit)
  - Satellite-Imagery (Parking-Lot-Daten, etc.)
  - Economic Indicators

---

## Dokumentation

**Weitere Details**:
- **Backend-Architektur**: `docs/ARCHITECTURE_BACKEND.md`
- **CLI-Referenz**: `docs/CLI_REFERENCE.md`
- **Testing-Commands**: `docs/TESTING_COMMANDS.md`

**Code-Referenzen**:
- Ingest-Module: `src/assembled_core/data/*_ingest.py`
- Feature-Module: `src/assembled_core/features/*_features.py`
- Signal-Module: `src/assembled_core/signals/rules_event_insider_shipping.py`
- Tests: `tests/test_features_events_phase6.py`, `tests/test_signals_event_phase6.py`

