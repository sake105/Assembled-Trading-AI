# Workflows – Multi-Factor Long/Short Strategy

**Last Updated:** 2025-12-10  
**Status:** Active Workflow for Factor-Based Trading Strategies

---

## Overview

**Ziel:** Dieser Workflow beschreibt, wie eine Multi-Faktor Long/Short-Strategie auf Basis von Factor-Bundles und lokalen Alt-Daten-Snapshots entwickelt und getestet wird.

Die Strategie nutzt:
- **Factor Bundles**: Vordefinierte Kombinationen von Faktoren mit Gewichten und Verarbeitungsoptionen
- **Multi-Factor Scores**: Gewichtete, normalisierte Faktor-Kombinationen
- **Quantil-basierte Auswahl**: Top/Bottom-Quantile für Long/Short-Positionen
- **Rebalancing**: Konfigurierbare Rebalancing-Frequenzen (täglich, wöchentlich, monatlich)

**Wichtig:** Diese Strategie basiert ausschließlich auf **lokalen Alt-Daten-Snapshots**. Es werden **keine Live-API-Calls** verwendet. Alle Preise und Faktoren stammen aus dem lokalen Datensatz (`ASSEMBLED_LOCAL_DATA_ROOT`).

---

## Data Setup

### Voraussetzungen

**1. Lokaler Alt-Daten-Snapshot:**

Die Strategie benötigt einen vollständigen lokalen Alt-Daten-Snapshot mit:
- Preisdaten (EOD: 1d, Intraday: 1min)
- Alt-Data Events (Earnings, Insider, News, Macro) - optional je nach Factor-Bundle

**Umgebungsvariable setzen:**
```powershell
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"
$env:ASSEMBLED_DATA_SOURCE = "local"
```

**2. Download/Validierung von Alt-Daten:**

Falls noch nicht vorhanden, müssen Alt-Daten heruntergeladen werden. Siehe:
- `scripts/download_historical_snapshot.py` - Historische Daten-Download
- `docs/WORKFLOWS_FACTOR_ANALYSIS.md` - Alt-Daten-Validierung und Smoketests

**3. Universe-Datei:**

Erstelle oder verwende eine Universe-Datei mit Symbol-Liste, z.B.:
- `config/macro_world_etfs_tickers.txt` - Breite ETF-Liste
- `config/universe_ai_tech_tickers.txt` - AI/Tech-Unternehmen

---

## Factor-Layer

### Zusammenhang mit Factor-Analyse

Die Multi-Factor-Strategie baut auf den Ergebnissen der Factor-Analyse auf:

**Phase A/B1/B2 (Factor-Generierung):**
- **Phase A1**: Core TA/Price Factors (Multi-Horizon Returns, Trend Strength, Reversal)
- **Phase A2**: Volatility & Liquidity Factors (RV, Vol-of-Vol, Turnover)
- **Phase B1**: Alt-Data Earnings & Insider Factors
- **Phase B2**: Alt-Data News & Macro Factors

**Phase C1/C2 (Factor-Evaluation):**
- **Phase C1**: IC/IR-basierte Evaluation (Information Coefficient, Information Ratio)
- **Phase C2**: Portfolio-Returns & Deflated Sharpe Evaluation

**Verweise:**
- **Factor Rankings**: `docs/FACTOR_RANKING_OVERVIEW.md` - Überblick über Top-Faktoren
- **Factor Bundles**: `config/factor_bundles/*.yaml` - Vordefinierte Factor-Bundle-Konfigurationen
- **Factor Analysis Workflows**: `docs/WORKFLOWS_FACTOR_ANALYSIS.md` - Detaillierte Factor-Analyse-Workflows

### Verfügbare Factor Bundles

**1. Macro World ETFs Core Bundle** (`config/factor_bundles/macro_world_etfs_core_bundle.yaml`):
- Fokus: Core TA/Price + Volatility/Liquidity Faktoren
- Universe: `macro_world_etfs`
- Factor-Set: `core+vol_liquidity`
- Faktoren: Momentum, Trend Strength, Realized Volatility

**2. AI Tech Core + Alt Bundle** (`config/factor_bundles/ai_tech_core_alt_bundle.yaml`):
- Fokus: Core TA/Price + Alt-Data (Earnings, Insider, News)
- Universe: `universe_ai_tech`
- Factor-Set: `core+alt_full`
- Faktoren: Momentum, Trend Strength, Earnings Surprise, Insider Activity, News Sentiment

**Neue Bundles erstellen:**

Factor Bundles sind YAML-Dateien in `config/factor_bundles/`. Siehe `docs/FACTOR_RANKING_OVERVIEW.md` Abschnitt "Configured Factor Bundles" für Details zur Struktur.

---

## Strategy Workflow

### Schritt 1: Factor-Analyse (Optional, aber empfohlen)

Bevor du die Strategie testest, solltest du die Faktoren evaluieren:

**Quick Smoke Test:**
```powershell
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --data-source local `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --factor-set core+vol_liquidity `
  --horizon-days 20
```

**Ausgabe:**
- `output/factor_analysis/factor_analysis_*.csv` - IC- und Portfolio-Statistiken
- `output/factor_analysis/factor_analysis_*_report.md` - Detaillierter Report

Siehe `docs/WORKFLOWS_FACTOR_ANALYSIS.md` für vollständige Factor-Analyse-Workflows.

### Schritt 2: Factor-Rankings konsolidieren (Optional)

Falls du Rankings über mehrere Universes konsolidieren möchtest:

```powershell
python scripts/summarize_factor_rankings.py
```

**Ausgabe:**
- `output/factor_analysis/factor_ranking_overview.csv` - Konsolidierte Rankings
- `output/factor_analysis/FACTOR_RANKING_BY_UNIVERSE.md` - Rankings je Universe

Siehe `docs/FACTOR_RANKING_OVERVIEW.md` für Interpretation der Rankings.

### Schritt 3: Factor-Bundle konfigurieren

Wähle oder erstelle ein Factor-Bundle:

**Option A: Bestehendes Bundle verwenden:**
```powershell
# Beispiel: Macro World ETFs Core Bundle
$bundlePath = "config/factor_bundles/macro_world_etfs_core_bundle.yaml"
```

**Option B: Neues Bundle erstellen:**

1. Erstelle eine neue YAML-Datei in `config/factor_bundles/`
2. Definiere Universe, Factor-Set, Faktoren mit Gewichten und Richtungen
3. Konfiguriere Verarbeitungsoptionen (Winsorizing, Z-Scoring)

Siehe `docs/FACTOR_RANKING_OVERVIEW.md` Abschnitt "Configured Factor Bundles" für Beispiel-Struktur.

### Schritt 4: Backtest der Strategie

**Hinweis:** Die Backtest-Engine verwendet jetzt optimierte, vektorisierte Operationen und optionale Numba-Beschleunigung (P3), was zu deutlich schnelleren Backtest-Laufzeiten führt. Die Optimierungen sind vollständig transparent – alle bestehenden Workflows und Scripts nutzen automatisch die optimierte Engine. Siehe [Backtest Optimization P3 Design](BACKTEST_OPTIMIZATION_P3_DESIGN.md) für Details.

**Basis-Backtest:**

```powershell
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

python scripts/cli.py run_backtest `
  --freq 1d `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --data-source local `
  --strategy multifactor_long_short `
  --bundle-path config/factor_bundles/macro_world_etfs_core_bundle.yaml `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --generate-report
```

**Erweiterte Optionen:**

```powershell
python scripts/cli.py run_backtest `
  --freq 1d `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --data-source local `
  --strategy multifactor_long_short `
  --bundle-path config/factor_bundles/macro_world_etfs_core_bundle.yaml `
  --top-quantile 0.2 `
  --bottom-quantile 0.2 `
  --rebalance-freq M `
  --max-gross-exposure 1.0 `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --start-capital 100000 `
  --with-costs `
  --generate-report
```

**Parameter-Erklärung:**
- `--bundle-path`: Pfad zum Factor-Bundle YAML-File
- `--top-quantile`: Top-Quantil-Schwelle für Long-Positionen (default: 0.2 = top 20%)
- `--bottom-quantile`: Bottom-Quantil-Schwelle für Short-Positionen (default: 0.2 = bottom 20%)
- `--rebalance-freq`: Rebalancing-Frequenz (`D` = täglich, `W` = wöchentlich, `M` = monatlich)
- `--max-gross-exposure`: Maximales Gross-Exposure (Long + Short) als Bruchteil des Kapitals (default: 1.0)

**Ausgabe:**
- `output/performance_report_1d.md` - Performance-Report mit Metriken
- `output/equity_curve_1d.csv` - Equity-Kurve
- `output/trades_1d.csv` - Trade-Liste (falls `--with-costs`)

---

## Interpretation

### Strategie-Logik

**1. Multi-Factor Score Berechnung:**
- Faktoren werden aus Preisen (und optional Alt-Data) berechnet
- **Point-in-Time Safety:** Alt-Data-Faktoren (Earnings, Insider, News) respektieren `disclosure_date` - Events werden nur verwendet, wenn sie zum Backtest-Datum bereits bekannt waren (siehe [Point-in-Time and Latency](POINT_IN_TIME_AND_LATENCY.md))
- Faktoren werden optional winsorisiert (Clipping von Extremwerten)
- Faktoren werden cross-sectional z-gescored (pro Timestamp über alle Symbole)
- Negativ-Richtung-Faktoren werden invertiert
- Gewichteter Multi-Factor Score: `mf_score = sum(weight_i * z_i)`

**2. Quantil-basierte Auswahl:**
- **Long-Positionen**: Symbole im Top-Quantil (z.B. top 20%) des mf_score
- **Short-Positionen**: Symbole im Bottom-Quantil (z.B. bottom 20%) des mf_score

**3. Position Sizing:**
- Equal-Weighting innerhalb der Long-Seite
- Equal-Weighting innerhalb der Short-Seite
- Netto-Exposure ≈ 0 (Long ≈ Short)
- Gross-Exposure begrenzt durch `max_gross_exposure`

**4. Rebalancing:**
- Rebalancing nur zu konfigurierten Zeitpunkten:
  - `D`: Täglich
  - `W`: Wöchentlich (Montag)
  - `M`: Monatlich (Monatsanfang)

### Typische Diagnosen

**Problem: Zu wenige Symbole im Top/Bottom-Quantil**

**Symptom:**
- Wenige oder keine Signale generiert
- Strategie hat sehr geringe Positionsgrößen

**Mögliche Ursachen:**
- Zu kleine Universe (zu wenige Symbole insgesamt)
- Zu strenges Quantil (z.B. top 10% bei nur 10 Symbolen)
- Faktoren zu stark korreliert (alle Symbole haben ähnliche Scores)

**Lösungen:**
- Größere Universe verwenden
- Quantile erhöhen (z.B. top 30% statt top 20%)
- Factor-Bundle anpassen (mehr Diversifikation)

**Problem: Extreme Drawdowns**

**Symptom:**
- Sehr negative Sharpe Ratio oder Sortino Ratio
- Lange Perioden mit negativen Returns

**Mögliche Ursachen:**
- Faktoren funktionieren nicht in allen Marktphasen
- Zu häufiges Rebalancing (Transaktionskosten)
- Factor-Bundle nicht robust genug (z.B. zu stark auf Momentum)

**Lösungen:**
- Robustere Factor-Bundles testen (z.B. mehr Diversifikation)
- Selteneres Rebalancing (z.B. monatlich statt täglich)
- Walk-Forward-Analyse durchführen (siehe `docs/WORKFLOWS_BACKTEST_AND_ENSEMBLE.md`)

**Problem: Sehr hohe Turnover-Rate**

**Symptom:**
- Viele Trades pro Periode
- Hohe Transaktionskosten

**Mögliche Ursachen:**
- Zu häufiges Rebalancing
- Factor-Scores ändern sich zu stark zwischen Perioden

**Lösungen:**
- Selteneres Rebalancing
- Factor-Bundle mit stabileren Faktoren (z.B. längerfristiges Momentum statt kurzfristiges)
- **Transaction Cost Analysis:** Führe einen TCA-Report aus, um den tatsächlichen Kosten-Einfluss zu quantifizieren (siehe [Risk Metrics & Attribution Workflows](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md) – TCA Section)

**Problem: Faktoren nicht verfügbar**

**Symptom:**
- Fehler beim Laden des Factor-Bundles
- Warnung: "Factor 'X' not found in DataFrame"

**Mögliche Ursachen:**
- Factor-Set passt nicht zum Bundle (z.B. Bundle fordert Alt-Data, aber Factor-Set enthält nur Core)
- Alt-Data nicht heruntergeladen

**Lösungen:**
- Sicherstellen, dass der richtige Factor-Set verwendet wird
- Alt-Data herunterladen (siehe Data Setup)

### Performance-Metriken interpretieren

**Wichtige Metriken:**
- **Sharpe Ratio**: Risiko-adjustierte Rendite (> 1.0 ist gut)
- **Sortino Ratio**: Downside-Risiko-adjustierte Rendite (> 1.0 ist gut)
- **Max Drawdown**: Maximaler Peak-to-Trough-Verlust (< 20% ist akzeptabel)
- **Win Rate**: Anteil profitabler Trades (> 50% ist gut)
- **Annualized Return**: Annualisierte Rendite (abhängig von Risiko)

**Vergleich mit Factor-Analyse:**
- Vergleiche Strategie-Performance mit Factor-Analyse (Phase C2)
- Wenn Strategie schlechter als einzelne Faktoren: Position-Sizing oder Rebalancing anpassen
- Wenn Strategie besser: Good! Robustere Factor-Kombination

**Transaction Cost Impact prüfen:**
Nach jedem Backtest sollte ein **Transaction Cost Analysis (TCA) Report** generiert werden, um zu prüfen:
- **Kosten vs. Brutto-PnL:** Wie viel der Brutto-Returns gehen durch Kosten verloren?
- **Net-Performance:** Ist die Strategie nach Kosten noch attraktiv (Net Sharpe > 1.0)?
- **Cost Ratio:** Costs / Gross Return sollte < 30% sein

**TCA-Report generieren:**
```powershell
python scripts/cli.py tca_report --backtest-dir output/backtests/experiment_123/
```

Siehe auch: [Risk Metrics & Attribution Workflows – Transaction Cost Analysis](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md#transaction-cost-analysis-e4)

**Batch-Backtests:**

Für systematische Strategie-Vergleiche (z.B. Core vs. Core+ML vs. ML-only, verschiedene Rebalancing-Frequenzen oder Universes) empfiehlt sich der Batch-Runner:

```powershell
python scripts/cli.py batch_backtest `
  --config-file configs/batch_backtests/ai_tech_core_vs_mlalpha.yaml
```

Details zum Batch-Config-Format und zu Best Practices: [Batch Backtests & Parallelization Workflow](WORKFLOWS_BATCH_BACKTESTS_AND_PARALLELIZATION.md)

---

## Nächste Schritte

**1. Robustheitstests:**
- Walk-Forward-Analyse (siehe `docs/WORKFLOWS_BACKTEST_AND_ENSEMBLE.md`)
- Verschiedene Universes testen
- Verschiedene Zeiträume testen

**2. Factor-Bundle-Optimierung:**
- Verschiedene Factor-Kombinationen testen
- Gewichte anpassen basierend auf IC-IR und DSR
- Neue Faktoren hinzufügen

**3. ML-Integration (Phase E):**
- Meta-Models für Signal-Filterung (siehe `docs/WORKFLOWS_ML_AND_EXPERIMENTS.md`)
- ML-basierte Position-Sizing

**4. Live-Trading (Zukunft):**
- Strategie für Live-Trading vorbereiten (Phase F)
- Execution-Layer integrieren
- Risk-Management erweitern

---

## Verweise

**Verwandte Workflows:**
- [Factor Analysis Workflows](WORKFLOWS_FACTOR_ANALYSIS.md) - Factor-Evaluation und Rankings
- [Backtest & Ensemble Workflows](WORKFLOWS_BACKTEST_AND_ENSEMBLE.md) - Erweiterte Backtest-Funktionen
- [ML Meta-Models & Experiments](WORKFLOWS_ML_AND_EXPERIMENTS.md) - ML-Integration

**Verwandte Dokumentation:**
- [Factor Ranking Overview](FACTOR_RANKING_OVERVIEW.md) - Top-Faktoren und Factor-Bundles
- [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) - Factor-Research-Roadmap (inkl. Research Playbook R1)

**Code-Referenzen:**
- `src/assembled_core/strategies/multifactor_long_short.py` - Strategie-Implementierung
- `src/assembled_core/signals/multifactor_signal.py` - Multi-Factor-Signal-Generierung
- `src/assembled_core/config/factor_bundles.py` - Factor-Bundle-Konfiguration

