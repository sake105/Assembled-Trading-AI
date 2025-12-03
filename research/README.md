# Research Directory – Assembled Trading AI

**Zweck:** Dieser Ordner enthält Research-Experimente, Notebooks und Skripte für die systematische Exploration neuer Trading-Ideen, Strategien und Datenquellen.

---

## Struktur

```
research/
├── README.md                          # Diese Datei
├── trend/                             # Trend-Strategie-Experimente
│   └── trend_baseline_experiments.ipynb
├── meta/                              # ML-Meta-Layer-Experimente
│   └── meta_model_calibration.ipynb
├── altdata/                           # Alternative-Daten-Experimente
│   └── insider_congress_shipping_exploration.ipynb
├── risk/                              # Risk-Engine-Experimente
│   └── scenario_and_risk_experiments.ipynb
└── [weitere Kategorien nach Bedarf]
```

---

## Experiment-Struktur

Ein typisches Research-Experiment sollte folgende Abschnitte haben:

### 1. Hypothese
- **Was testen wir?** Klare Frage formulieren
- **Was erwarten wir?** Erwartetes Ergebnis beschreiben

**Beispiel:**
> "Hypothese: RSI-Mean-Reversion-Strategie sollte in Seitwärts-Märkten besser performen als Trend-Baseline."

### 2. Setup
- **Daten:** Welche Symbole, welcher Zeitraum?
- **Strategie:** Parameter, Konfiguration
- **Backtest:** Start-Capital, Kosten-Modell, Frequenz

**Beispiel:**
```python
# Setup
symbols = ["AAPL", "MSFT", "GOOGL"]
start_date = "2024-01-01"
end_date = "2024-12-31"
freq = "1d"
start_capital = 10000.0
rsi_period = 14
oversold_threshold = 30
overbought_threshold = 70
```

### 3. Methode
- Code für Backtest, Feature-Berechnung, etc.
- Wiederverwendbare Funktionen aus `src/assembled_core/` nutzen

**Beispiel:**
```python
# Methode
from src.assembled_core.data.prices_ingest import load_eod_prices
from src.assembled_core.features.ta_features import add_rsi
from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

# Load data
prices = load_eod_prices(...)

# Add features
prices_with_features = add_rsi(prices, window=rsi_period)

# Generate signals
signals = generate_rsi_signals(prices_with_features, oversold_threshold, overbought_threshold)

# Run backtest
result = run_portfolio_backtest(...)
```

### 4. Ergebnisse
- Performance-Metriken (Sharpe, Drawdown, Win-Rate, etc.)
- Visualisierungen (Equity-Curves, Trade-Distribution)
- Vergleich mit Baseline-Strategie

**Beispiel:**
```python
# Ergebnisse
print(f"Sharpe Ratio: {result.metrics['sharpe']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown_pct']:.2f}%")
print(f"Total Trades: {result.metrics['trades']}")

# Visualize
import matplotlib.pyplot as plt
plt.plot(result.equity['timestamp'], result.equity['equity'])
plt.title("Equity Curve")
plt.show()
```

### 5. Fazit
- **War die Hypothese korrekt?** Ja/Nein mit Begründung
- **Was haben wir gelernt?** Key Insights
- **Limitationen:** Was war nicht ideal? Was könnte verbessert werden?

**Beispiel:**
> "Fazit: Die Hypothese war teilweise korrekt. RSI-Strategie performte besser in Seitwärts-Märkten, aber schlechter in starken Trend-Märkten. Nächster Schritt: Kombination aus Trend- und Mean-Reversion-Signalen testen."

### 6. Nächste Schritte
- Was sollte als nächstes getestet werden?
- Welche Parameter sollten variiert werden?
- Welche neuen Features könnten helfen?

---

## Best Practices

### Code-Organisation
- **Wiederverwendung:** Nutze Funktionen aus `src/assembled_core/` statt Code zu duplizieren
- **Modularität:** Teile Experimente in wiederverwendbare Funktionen auf
- **Dokumentation:** Kommentiere nicht-trivialen Code

### Daten-Management
- **Lokale Daten:** Nutze `data/sample/` für Experimente mit Sample-Daten
- **Reproduzierbarkeit:** Fixiere Random-Seeds für ML-Experimente
- **Versionierung:** Dokumentiere, welche Daten-Version verwendet wurde

### Experiment-Tracking
- **Git-Commit:** Committe Experimente mit aussagekräftigen Commit-Messages
- **Notebook-Outputs:** Speichere wichtige Visualisierungen als PNG/PDF
- **Metadaten:** Dokumentiere Parameter, Daten-Quellen, Timestamps

### Performance-Analyse
- **Vergleich:** Vergleiche immer mit Baseline-Strategien
- **Statistische Signifikanz:** Berücksichtige Sample-Size und Overfitting
- **Robustheit:** Teste auf verschiedenen Zeiträumen und Universen

---

## Experiment-Tracking Integration

**So verknüpfst du ein Notebook mit einem Experiment-Run:**

### 1. Run-ID im Notebook referenzieren

Wenn du einen Backtest oder Meta-Model-Training mit `--track-experiment` ausführst, erhältst du eine Run-ID (z.B. `20250115_143022_abc12345`). Diese kannst du in deinem Notebook verwenden:

```python
# Beispiel: Metriken aus einem Experiment-Run laden
from src.assembled_core.qa.experiment_tracking import ExperimentTracker
from src.assembled_core.config.settings import get_settings

settings = get_settings()
tracker = ExperimentTracker(settings.experiments_dir)

# Run-ID aus CLI-Output oder manuell setzen
run_id = "20250115_143022_abc12345"

# Run finden
runs = tracker.list_runs()
run = next((r for r in runs if r.run_id == run_id), None)

if run:
    # Metriken laden
    metrics_df = tracker.get_run_metrics(run)
    print(metrics_df[metrics_df["metric_name"] == "sharpe"])
    
    # Config anzeigen
    print(f"Config: {run.config}")
    print(f"Tags: {run.tags}")
```

### 2. Experiment-Run direkt aus Notebook starten

Du kannst auch direkt aus einem Notebook einen Experiment-Run starten:

```python
from src.assembled_core.qa.experiment_tracking import ExperimentTracker
from src.assembled_core.config.settings import get_settings

settings = get_settings()
tracker = ExperimentTracker(settings.experiments_dir)

# Start run
run = tracker.start_run(
    name="notebook_experiment_ma20_50",
    config={"ma_fast": 20, "ma_slow": 50, "freq": "1d"},
    tags=["notebook", "trend", "ma20_50"]
)

# ... führe Backtest/Experiment durch ...

# Log metrics
tracker.log_metrics(run, {"sharpe": 1.23, "max_drawdown": -0.15})

# Log artifacts (z.B. Plot)
tracker.log_artifact(run, "equity_curve.png", "equity_curve.png")

# Finish run
tracker.finish_run(run, status="finished")

print(f"Experiment Run-ID: {run.run_id}")
print(f"Run Directory: {settings.experiments_dir / run.run_id}")
```

### 3. Runs durchsuchen und vergleichen

```python
# Alle Runs mit bestimmten Tags auflisten
trend_runs = tracker.list_runs(tags=["trend"])

# Metriken mehrerer Runs vergleichen
for run in trend_runs[:5]:  # Top 5
    metrics_df = tracker.get_run_metrics(run)
    sharpe = metrics_df[metrics_df["metric_name"] == "sharpe"]["metric_value"].iloc[-1] if not metrics_df.empty else None
    print(f"{run.name}: Sharpe = {sharpe}")
```

---

## Verknüpfungen

- **Research-Roadmap**: `docs/RESEARCH_ROADMAP.md` (Übersicht, Backlog, Fokus)
- **Backend-Roadmap**: `docs/BACKEND_ROADMAP.md` (Phase 12)
- **Architektur**: `docs/ARCHITECTURE_BACKEND.md` (Verfügbare Module)
- **CLI-Referenz**: `docs/CLI_REFERENCE.md` (CLI-Befehle für Backtests, etc.)

---

## Nächste Schritte

1. **Erste Experimente starten:** Siehe `docs/RESEARCH_ROADMAP.md` für konkrete Research-Tasks
2. **Experiment-Tracking nutzen:** Verwende `--track-experiment` bei Backtests und Meta-Model-Training
3. **Dokumentation:** Dokumentiere alle Experimente nach obiger Struktur

---

**Hinweis:** Dieser Ordner ist für Research-Experimente gedacht. Code, der produktionsreif ist, sollte in `src/assembled_core/` verschoben werden.

