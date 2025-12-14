# Risk Metrics & Attribution Workflows (D2)

## Overview

Dieses Dokument beschreibt den Workflow für erweiterte Risk-Analysen und Performance-Attribution von Backtest-Strategien. Diese Funktionalität ist Teil von **Phase D: Regime Models & Risk 2.0** im [Advanced Analytics Factor Labs Plan](ADVANCED_ANALYTICS_FACTOR_LABS.md).

### Was der Risk-Workflow leistet

Der Risk-Workflow bietet **Deep-Dive-Analysen** über das Risiko-Profil von Strategien:

1. **Erweiterte Risk-Metriken**: 
   - Sharpe, Sortino, Volatilität, Max Drawdown, Calmar Ratio
   - Tail-Risiken: Skewness, Kurtosis, Value at Risk (VaR), Expected Shortfall (ES)

2. **Exposure-Analyse**:
   - Gross/Net Exposure über Zeit
   - Portfolio-Konzentration (HHI - Herfindahl-Hirschman Index)
   - Anzahl der Positionen über Zeit

3. **Regime-Attribution**:
   - Risiko-Metriken segmentiert nach identifizierten Markt-Regimes (Bull, Bear, Crisis, Neutral)
   - Identifikation von Regime-spezifischen Schwächen

4. **Faktor-Gruppen-Attribution**:
   - Performance-Beitrag nach Faktor-Kategorien (Trend, Vol/Liq, Earnings, Insider, News/Macro)
   - Korrelation zwischen Portfolio-Returns und Faktor-Exposures

### Integration in die Factor Labs Roadmap

- **D1 (Completed)**: Regime-Detection → liefert `regime_state_df` für Regime-Attribution
- **D2 (In Progress)**: Risk 2.0 & Attribution → dieser Workflow
- **D3 (Future)**: Adaptive Factor Selection → nutzt Risk-Attribution für dynamische Faktor-Gewichtung
- **D4 (Future)**: Regime-Aware Risk Models → nutzt Regime-Metriken für dynamisches Risk-Management

---

## Inputs & Voraussetzungen

### Erforderlich

- **Backtest-Outputs**: Ergebnisse von `run_backtest_strategy.py` oder ähnlichen Backtest-Skripten
  - `equity_curve.csv` oder `equity_curve.parquet` (mit Spalten: `timestamp`, `equity`)
  - `positions.csv` oder `positions.parquet` (mit Spalten: `timestamp`, `symbol`, `weight`)
  - Optional: `trades.csv` oder `trades.parquet` (für erweiterte Analysen)

### Optional (für erweiterte Attribution)

- **Regime-State-DataFrame** (aus D1):
  - Format: `timestamp`, `regime_label` (z.B. "bull", "bear", "neutral", "crisis")
  - Kann aus `build_regime_state()` (siehe [Regime Models Workflow](WORKFLOWS_REGIME_MODELS_AND_RISK.md)) erzeugt werden
  - Typischer Pfad: `output/regimes/<universe>_regime_state.parquet`

- **Factor-Panel-DataFrame** (aus C1/C2/B1/B2):
  - Format: Panel mit `timestamp`, `symbol`, `factor_*` Spalten
  - Kann aus `analyze_factors` oder Research-Skripten erzeugt werden
  - Typischer Pfad: `output/factor_analysis/<experiment_id>_factors.parquet`

### Wichtiger Hinweis: Nur lokale Daten

**Alle Daten werden aus lokalen Dateien geladen:**
- ✅ Lokale Backtest-Outputs (CSV/Parquet)
- ✅ Lokale Alt-Daten-Snapshots (Parquet-Dateien)
- ❌ **Keine Live-Preis-APIs** (Yahoo Finance, Twelve Data, etc.)
- ❌ **Keine Online-Fetches** im Risk-Workflow selbst

Der Risk-Workflow ist vollständig **offline** und arbeitet nur mit bereits vorhandenen Backtest-Ergebnissen und lokalen Daten-Snapshots.

---

## Workflow-Beispiele

### 1. Basic Risk Report

Generiert einen grundlegenden Risk-Report mit allen erweiterten Metriken und Exposure-Analyse.

```powershell
python scripts/cli.py risk_report `
  --backtest-dir output/backtests/experiment_123
```

**Outputs:**
- `risk_summary.csv`: Alle globalen Risk-Metriken (Sharpe, Sortino, Vol, MaxDD, Skew, Kurtosis, VaR, ES)
- `exposure_timeseries.csv`: Gross/Net Exposure, HHI, Anzahl Positionen über Zeit
- `risk_report.md`: Markdown-Report mit Übersicht und Interpretation

**Beispiel-Output (risk_summary.csv):**
```
mean_return_annualized,vol_annualized,sharpe,sortino,max_drawdown,calmar,skew,kurtosis,var_95,cvar_95,n_periods
0.1250,0.1800,0.6944,0.8500,-15.50,0.8065,-0.25,2.10,-0.0250,-0.0320,252
```

---

### 2. Risk Report mit Regime-Attribution

Segmentiert Risk-Metriken nach identifizierten Markt-Regimes.

```powershell
python scripts/cli.py risk_report `
  --backtest-dir output/backtests/experiment_123 `
  --regime-file output/regimes/ai_tech_regime_state.parquet
```

**Zusätzliche Outputs:**
- `risk_by_regime.csv`: Risk-Metriken pro Regime (Sharpe, Vol, MaxDD, Total Return pro Regime)

**Interpretation:**
- Vergleich Sharpe/Volatilität zwischen Regimes
- Identifikation von Regimes, in denen die Strategie unterdurchschnittlich performt
- Beispiel: "Sharpe in Bull-Regime: 1.2, Sharpe in Bear-Regime: -0.3 → Strategie leidet in Bear-Phasen"

**Wie man `regime_state.parquet` erzeugt:**
Siehe [Regime Models Workflow](WORKFLOWS_REGIME_MODELS_AND_RISK.md) für Details. Kurzfassung:

```python
from src.assembled_core.risk.regime_models import build_regime_state

regime_state_df = build_regime_state(
    prices=price_panel_df,
    macro_factors=macro_df,
    breadth_df=breadth_df,
    vol_df=vol_df,
    config=RegimeStateConfig(...)
)
regime_state_df.to_parquet("output/regimes/my_universe_regime_state.parquet")
```

---

### 3. Risk Report mit Regime + Faktor-Gruppen-Attribution

Vollständige Attribution: Regime-segmentierte Risiken + Faktor-Gruppen-Performance.

**Schritt 1: Factor-Panel erzeugen (falls noch nicht vorhanden)**

Option A: Aus Factor-Analyse-Workflow (C1/C2):
```powershell
python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/universe_ai_tech_tickers.txt `
  --start-date 2020-01-01 `
  --end-date 2025-12-03 `
  --factor-set core+vol_liquidity `
  --horizon-days 20 `
  --output-dir output/factor_analysis/ai_tech_core_vol
```

Die Factor-Analyse erzeugt typischerweise ein `*_factors.parquet` File (Panel-Format).

Option B: Manuell aus Research-Skript:
```python
# Beispiel: Faktoren aus Feature-Engineering extrahieren
from src.assembled_core.features.price_features import add_price_features
from src.assembled_core.features.volatility_features import add_volatility_features

factor_panel_df = prices_df.copy()
factor_panel_df = add_price_features(factor_panel_df)
factor_panel_df = add_volatility_features(factor_panel_df)
# ... weitere Faktoren

# Nur Faktor-Spalten + timestamp, symbol behalten
factor_cols = [col for col in factor_panel_df.columns if col.startswith(("returns_", "trend_", "rv_", "earnings_", "insider_", "news_", "macro_"))]
factor_panel_df = factor_panel_df[["timestamp", "symbol"] + factor_cols]
factor_panel_df.to_parquet("output/factor_analysis/my_factors.parquet")
```

**Schritt 2: Risk Report mit beiden Optionen generieren**

```powershell
python scripts/cli.py risk_report `
  --backtest-dir output/backtests/experiment_123 `
  --regime-file output/regimes/ai_tech_regime_state.parquet `
  --factor-panel-file output/factor_analysis/ai_tech_core_vol_factors.parquet
```

**Zusätzliche Outputs:**
- `risk_by_factor_group.csv`: Attribution pro Faktor-Gruppe (Korrelation, Avg Exposure, Perioden)

**Faktor-Gruppen (Default):**
- **Trend**: `returns_12m`, `trend_strength_50`, `trend_strength_200`
- **Vol/Liq**: `rv_20`, `vov_20_60`, `turnover_20d`
- **Earnings**: `earnings_eps_surprise_last`, `post_earnings_drift_20d`
- **Insider**: `insider_net_notional_60d`, `insider_buy_ratio_60d`
- **News/Macro**: `news_sentiment_trend_20d`, `macro_growth_regime`

**Beispiel-Output (risk_by_factor_group.csv):**
```
factor_group,factors,correlation_with_returns,avg_exposure,n_periods
Trend,"returns_12m,trend_strength_50",0.65,0.45,250
Vol/Liq,"rv_20,vov_20_60",-0.20,0.30,250
Earnings,"earnings_eps_surprise_last",0.15,0.10,250
```

---

## Interpretation & Best Practices

### Risk-Metriken im Kontext verstehen

#### Sharpe vs. Sortino

- **Sharpe Ratio**: Risiko-adjustierte Returns (berücksichtigt Gesamt-Volatilität)
  - > 1.0: Gut, > 2.0: Sehr gut
  - Kann bei hoher Upside-Volatilität niedrig sein (bestraft auch gute Schwankungen)

- **Sortino Ratio**: Downside-Risk-adjusted (bestraft nur negative Volatilität)
  - > 1.5: Gut
  - **Interpretation**: Wenn Sortino >> Sharpe → Strategie hat häufige positive Ausreißer (gutes Zeichen)

#### Max Drawdown & Calmar

- **Max Drawdown**: Größter Peak-to-Trough-Verlust (in %)
  - < -10%: Niedriges Risiko
  - -10% bis -20%: Moderate Risiko
  - > -20%: Hohes Risiko

- **Calmar Ratio**: CAGR / |Max Drawdown|
  - > 1.0: Gut (CAGR kompensiert MaxDD)
  - **Interpretation**: Hoher Calmar = gute Erholung nach Drawdowns

#### Tail-Risiken: Skewness, Kurtosis, VaR, ES

- **Skewness**:
  - < 0: Left-skewed → häufige große Verluste (schlecht)
  - ≈ 0: Symmetrisch
  - > 0: Right-skewed → häufige große Gewinne (gut)

- **Kurtosis**:
  - > 3: Fat tails → häufige extreme Events (höheres Risiko)
  - **Interpretation**: Hohe Kurtosis + negative Skewness = gefährliche Kombination

- **VaR (95%)**: Wert, der in den schlechtesten 5% der Fälle unterschritten wird
- **ES/CVaR (95%)**: Durchschnittlicher Verlust in den schlechtesten 5% der Fälle
  - **Interpretation**: ES > VaR → Tail-Risiko ist größer als der VaR suggeriert

### Regime-Attribution interpretieren

**Beispiel-Szenario:**
```
Regime | Periods | Sharpe | Volatility | MaxDD  | Total Return
-------|---------|--------|------------|--------|--------------
bull   | 150     | 1.20   | 0.15       | -8.50  | 45.2%
bear   | 80      | -0.30  | 0.25       | -18.20 | -12.5%
neutral| 22      | 0.50   | 0.12       | -3.00  | 2.1%
```

**Interpretation:**
- Strategie performt **sehr gut in Bull-Phasen** (Sharpe 1.2, hoher Total Return)
- Strategie **verliert in Bear-Phasen** (negativer Sharpe, großer Drawdown)
- **Action Item**: Regime-Risk-Map anpassen → niedrigere Exposure in Bear-Regime

### Faktor-Attribution interpretieren

**Beispiel-Szenario:**
```
Factor Group      | Correlation | Avg Exposure | Interpretation
------------------|-------------|--------------|------------------
Trend             | 0.65        | 0.45         | Stark positiv → Trend ist Haupttreiber
Vol/Liq           | -0.20       | 0.30         | Schwach negativ → Vol-Faktoren reduzieren Returns
Earnings          | 0.15        | 0.10         | Schwach positiv → geringer Beitrag
Insider           | 0.05        | 0.08         | Fast neutral → wenig Einfluss
News/Macro        | -0.10       | 0.07         | Schwach negativ → News-Faktoren schaden
```

**Interpretation:**
- **Zu starke Abhängigkeit von Trend**: Correlation 0.65 → Strategie ist sehr trend-abhängig
- **Vol-Faktoren schaden**: Negative Correlation → Portfolio profitiert nicht von Vol-Faktoren
- **Action Items**:
  - Diversifikation: Andere Faktor-Gruppen stärker gewichten
  - Factor-Bundle überarbeiten: Vol-Gewichte reduzieren, Trend-Gewichte ggf. reduzieren

### Exposure-Analyse (HHI Concentration)

**HHI (Herfindahl-Hirschman Index):**
- 0.0: Perfekte Diversifikation (alle Positionen gleich gewichtet)
- 1.0: Maximale Konzentration (eine einzige Position)

**Interpretation:**
- **HHI > 0.3**: Hohe Konzentration → höheres Idiosyncratic Risk
- **HHI < 0.1**: Gute Diversifikation
- **Action Item**: Wenn HHI zu hoch → mehr Positionen oder gleichmäßigere Gewichtung

**Gross vs. Net Exposure:**
- **Gross Exposure**: Summe der absoluten Gewichte (Long + Short)
  - > 1.5: Sehr hohe Leverage
  - 1.0-1.5: Moderate Leverage
  - < 1.0: Niedrige/Negative Leverage

- **Net Exposure**: Summe der Gewichte (Long - Short)
  - > 0.5: Stark Long-bias
  - -0.5 bis 0.5: Relativ balanced
  - < -0.5: Stark Short-bias

### Findings zurück in Strategie-Parametrisierung übertragen

#### 1. Regime-Risk-Map anpassen

Wenn Regime-Attribution zeigt, dass Strategie in bestimmten Regimes leidet:

```python
# Vorher: Zu aggressive Exposure in allen Regimes
regime_risk_map = {
    "bull": {"max_gross_exposure": 1.2, "target_net_exposure": 0.6},
    "bear": {"max_gross_exposure": 1.0, "target_net_exposure": 0.0},  # ❌ Zu aggressiv
    "crisis": {"max_gross_exposure": 0.8, "target_net_exposure": -0.2},
}

# Nachher: Konservativer in Bear-Regime
regime_risk_map = {
    "bull": {"max_gross_exposure": 1.2, "target_net_exposure": 0.6},
    "bear": {"max_gross_exposure": 0.5, "target_net_exposure": 0.0},  # ✅ Reduziert
    "crisis": {"max_gross_exposure": 0.3, "target_net_exposure": 0.0},  # ✅ Sehr defensiv
}
```

#### 2. Factor-Bundle-Gewichtung anpassen

Wenn Faktor-Attribution zeigt, dass bestimmte Gruppen schaden oder zu dominant sind:

```yaml
# Factor Bundle: Vorher
factors:
  - name: returns_12m
    weight: 0.6  # ❌ Zu dominant
  - name: rv_20
    weight: 0.4

# Factor Bundle: Nachher (nach Attribution)
factors:
  - name: returns_12m
    weight: 0.4  # ✅ Reduziert
  - name: rv_20
    weight: 0.3  # ✅ Reduziert
  - name: earnings_eps_surprise_last
    weight: 0.3  # ✅ Neu hinzugefügt (wenn Correlation positiv)
```

#### 3. Position Sizing anpassen

Wenn Exposure-Analyse zeigt, dass Portfolio zu konzentriert ist:

```python
# Vorher: Top-N-Strategie mit starker Konzentration
config = MultiFactorStrategyConfig(
    max_gross_exposure=1.2,
    top_n_long=10,  # ❌ Zu wenige Positionen → hohe HHI
    top_n_short=10,
)

# Nachher: Mehr Positionen für Diversifikation
config = MultiFactorStrategyConfig(
    max_gross_exposure=1.0,  # ✅ Geringere Leverage
    top_n_long=20,  # ✅ Mehr Positionen
    top_n_short=20,
)
```

---

## Beispiel-Workflow: End-to-End

### Komplettes Beispiel: Multi-Factor-Strategie mit Regime-Aware Risk

```powershell
# Schritt 1: Backtest mit Multi-Factor-Strategie + Regime-Overlay
python scripts/cli.py run_backtest `
  --freq 1d `
  --strategy multifactor_long_short `
  --bundle-path config/factor_bundles/macro_world_etfs_core_bundle.yaml `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --data-source local `
  --use-regime-overlay `
  --regime-config-file config/regime/risk_overlay_default.yaml `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --generate-report

# Schritt 2: Regime-State erzeugen (falls noch nicht vorhanden)
# (Siehe WORKFLOWS_REGIME_MODELS_AND_RISK.md)

# Schritt 3: Factor-Analyse (falls Faktor-Panel benötigt)
python scripts/cli.py analyze_factors `
  --freq 1d `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --start-date 2010-01-01 `
  --end-date 2025-12-03 `
  --factor-set core+vol_liquidity `
  --output-dir output/factor_analysis/etfs_core_vol

# Schritt 4: Risk Report generieren
python scripts/cli.py risk_report `
  --backtest-dir output/backtests/<experiment_id> `
  --regime-file output/regimes/macro_world_etfs_regime_state.parquet `
  --factor-panel-file output/factor_analysis/etfs_core_vol/<experiment_id>_factors.parquet

# Schritt 5: Findings interpretieren und Strategie-Parameter anpassen
# (Siehe Interpretation & Best Practices oben)
```

---

## Troubleshooting

### "Could not find equity curve file"

**Ursache**: Backtest-Verzeichnis enthält keine `equity_curve.csv` oder `equity_curve.parquet`.

**Lösung**:
- Prüfen, ob Backtest erfolgreich durchgelaufen ist
- Prüfen, ob Datei in anderem Format vorliegt (z.B. `portfolio_equity_1d.csv`)
- Script unterstützt verschiedene Dateinamen-Patterns, aber nicht alle möglichen

### "No overlapping timestamps between returns and regime_state_df"

**Ursache**: Timestamps in Returns und Regime-State stimmen nicht überein.

**Lösung**:
- Prüfen, ob beide DataFrames denselben Zeitraum abdecken
- Prüfen, ob Timestamps im gleichen Format vorliegen (UTC, etc.)
- Prüfen, ob Regime-State für dasselbe Universe erzeugt wurde

### "Missing expected columns in factor_panel_df"

**Ursache**: Factor-Panel hat nicht die erwartete Struktur (`timestamp`, `symbol`, `factor_*`).

**Lösung**:
- Prüfen, ob Factor-Panel im Panel-Format vorliegt (nicht Wide-Format)
- Prüfen, ob `timestamp` und `symbol` Spalten vorhanden sind
- Prüfen, ob Faktor-Spalten die erwarteten Namen haben (siehe Default-Faktor-Gruppen)

### Risk-Metriken sind None

**Ursache**: Zu wenige Datenpunkte oder unzureichende Datenqualität.

**Lösung**:
- Prüfen, ob Equity-Kurve genügend Datenpunkte hat (mindestens 5-10)
- Prüfen, ob Returns sinnvoll sind (nicht alle NaN)
- Prüfen, ob Regime-Attribution genügend Perioden pro Regime hat

---

## Weitere Ressourcen

- [Risk 2.0 & Attribution D2 Design Document](RISK_2_0_D2_DESIGN.md): Detaillierte Design-Spezifikation
- [Regime Models Workflow](WORKFLOWS_REGIME_MODELS_AND_RISK.md): Wie man Regime-State erzeugt
- [Advanced Analytics Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md): Gesamte Roadmap
- [Factor Ranking Overview](FACTOR_RANKING_OVERVIEW.md): Wie Factor-Rankings mit Risk-Attribution kombiniert werden können

