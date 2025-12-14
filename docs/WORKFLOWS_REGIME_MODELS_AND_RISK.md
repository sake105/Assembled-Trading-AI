# Regime Models & Risk Overlay Workflow

**Phase D1** – Advanced Analytics & Factor Labs

## Overview

**Ziel:** Identifikation von Marktregimes (bull, bear, sideways, crisis, reflation) und adaptive Risikosteuerung für Strategien basierend auf den identifizierten Regimes.

Dieser Workflow ermöglicht:
- **Regime Detection**: Automatische Klassifikation von Marktregimes basierend auf Macro-Faktoren, Market Breadth und Volatilität
- **Risk Overlay**: Dynamische Anpassung von Exposure und Netto-Positionen je nach identifiziertem Regime
- **Factor Evaluation**: Analyse der Faktor-Performance getrennt nach Regime

**Datenbasis:** Alle Analysen basieren auf **lokalen Alt-Daten-Snapshots** (`ASSEMBLED_LOCAL_DATA_ROOT`) und gespeicherten Alt-Data-Files. Es werden **keine Live-Preis-APIs** verwendet.

**Siehe auch:**
- [Regime Models D1 Design Document](REGIME_MODELS_D1_DESIGN.md) – Detailliertes Design und Data Contracts
- [Advanced Analytics & Factor Labs](../ADVANCED_ANALYTICS_FACTOR_LABS.md) – Gesamt-Roadmap

---

## Inputs

### 1. Preise

Preise werden über **LocalParquetPriceDataSource** aus dem Alt-Daten-Snapshot geladen:

```python
from src.assembled_core.data.data_source import LocalParquetPriceDataSource

data_source = LocalParquetPriceDataSource(
    data_root=os.environ.get("ASSEMBLED_LOCAL_DATA_ROOT")
)
prices = data_source.load_prices(
    freq="1d",
    symbols=["AAPL", "MSFT", "GOOGL", ...],
    start_date="2010-01-01",
    end_date="2025-12-03",
)
```

**Format:** Panel-Format mit Spalten `timestamp`, `symbol`, `close` (und optional `open`, `high`, `low`, `volume`).

### 2. Faktoren

**Trend / Vol / Breadth (Phase A):**
- Market Breadth: `fraction_above_ma_50`, `risk_on_off_score`
- Volatilität: `rv_20`, `vov_20_60`
- Trend Strength: `trend_strength_50`, `trend_strength_200`

**Macro-Regime (Phase B2):**
- `macro_growth_regime` (+1 = Expansion, -1 = Recession)
- `macro_inflation_regime` (+1 = High Inflation, -1 = Low/Deflation)
- `macro_risk_aversion_proxy` (+1 = Risk-Off, -1 = Risk-On)

Diese Faktoren werden aus gespeicherten Alt-Data-Files geladen (keine Live-APIs).

---

## Workflow

### 1. Regime-State berechnen

Das Regime-Modul identifiziert tägliche Regime-Labels basierend auf den verfügbaren Inputs.

**Beispiel:**

```python
from src.assembled_core.risk.regime_models import (
    RegimeStateConfig,
    build_regime_state,
)

# Optional: Custom Config
config = RegimeStateConfig(
    trend_ma_windows=(50, 200),
    vol_window=20,
    vov_window=60,
    breadth_ma_window=50,
    combine_macro_and_market=True,
)

# Build regime state
regime_state_df = build_regime_state(
    prices=prices_df,
    macro_factors=macro_factors_df,  # Optional: kann None sein
    breadth_df=breadth_df,  # Optional
    vol_df=vol_df,  # Optional
    config=config,  # Optional: verwendet Default wenn None
)

# Output: DataFrame mit Spalten:
# - timestamp
# - regime_label: "bull", "bear", "sideways", "crisis", "reflation", "neutral"
# - regime_trend_score: float (-1.0 bis +1.0)
# - regime_macro_score: float (-1.0 bis +1.0)
# - regime_risk_score: float (-1.0 bis +1.0)
# - regime_confidence: float (0.0 bis 1.0)

print(regime_state_df.head())
print(regime_state_df["regime_label"].value_counts())
```

**Regime-Klassifikations-Logik:**

- **`crisis`**: `regime_risk_score < -0.8` (extreme Volatilität)
- **`bear`**: `regime_trend_score < -0.5` UND `regime_risk_score < 0`
- **`bull`**: `regime_trend_score > +0.5` UND `regime_risk_score > 0`
- **`reflation`**: `regime_macro_score > 0.3` UND `inflation_regime > 0`
- **`sideways`**: Moderate Scores in allen Dimensionen
- **`neutral`**: Default

**Hinweis:** Fehlende Inputs (z.B. keine Macro-Faktoren verfügbar) werden übersprungen. Das Regime-Modul funktioniert auch mit teilweise fehlenden Daten.

### 2. Regime-Statistiken

Analysiere Regime-Übergänge und typische Regime-Dauern.

**Beispiel:**

```python
from src.assembled_core.risk.regime_models import compute_regime_transition_stats

transition_stats = compute_regime_transition_stats(regime_state_df)

# Output: DataFrame mit Spalten:
# - from_regime: Quell-Regime
# - to_regime: Ziel-Regime
# - count: Anzahl Übergänge
# - avg_duration_days: Durchschnittliche Dauer des Quell-Regimes
# - transition_probability: Wahrscheinlichkeit des Übergangs

print(transition_stats)

# Beispiel-Ausgabe:
#   from_regime  to_regime  count  avg_duration_days  transition_probability
# 0        bull    neutral      8             25.5                  0.40
# 1        bull       bear      6             25.5                  0.30
# 2       bear    neutral      5             18.2                  0.35
# ...
```

**Interpretation:**
- **Transition Probabilities** summieren sich pro Quell-Regime zu ≈ 1.0
- **Average Duration** zeigt typische Regime-Dauern (z.B. Bull-Phasen dauern durchschnittlich 25 Tage)

### 3. Factor-Performance nach Regime

Evaluiere, welche Faktoren in welchen Regimes am besten funktionieren.

**Beispiel:**

```python
from src.assembled_core.risk.regime_models import evaluate_factor_ic_by_regime
from src.assembled_core.qa.factor_analysis import compute_rolling_ic

# Zuerst: IC über Zeit berechnen (z.B. aus Phase C1)
ic_df = compute_rolling_ic(
    factors_df=factor_df,
    forward_returns=forward_returns_df,
    ic_method="pearson",
)

# Dann: IC nach Regime segmentieren
ic_by_regime = evaluate_factor_ic_by_regime(
    ic_df=ic_df,
    regime_state_df=regime_state_df,
    ic_col_suffix="_ic",
)

# Output: DataFrame mit Spalten:
# - factor: Faktor-Name
# - regime: Regime-Label
# - mean_ic: Durchschnittlicher IC in diesem Regime
# - std_ic: Standardabweichung des IC
# - ic_ir: IC-Information-Ratio (mean_ic / std_ic)
# - hit_ratio: Anteil Perioden mit positivem IC
# - n_observations: Anzahl Beobachtungen

print(ic_by_regime[ic_by_regime["factor"] == "returns_12m"])

# Beispiel-Ausgabe:
#     factor regime  mean_ic  std_ic    ic_ir  hit_ratio  n_observations
# 0  returns_12m   bull    0.12   0.08    1.50       0.65             125
# 1  returns_12m   bear   -0.08   0.10   -0.80       0.40              45
# 2  returns_12m  crisis  -0.15   0.12   -1.25       0.30              20
```

**Interpretation:**
- **IC-IR > 1.0**: Faktor funktioniert gut in diesem Regime
- **Negative IC-IR**: Faktor funktioniert schlecht oder invers
- **Hit Ratio > 0.6**: Faktor liefert konsistent positive ICs

### 4. Regime-Overlay in Strategien

Integriere Regime-basierte Risikosteuerung in die Multi-Factor-Strategie.

#### 4.1 Regime-Risk-Map konfigurieren

Definiere, wie Exposure und Netto-Positionen je Regime angepasst werden sollen.

**Beispiel-YAML (`config/regime/risk_overlay_default.yaml`):**

```yaml
regime_risk_map:
  bull:
    max_gross_exposure: 1.2
    target_net_exposure: 0.6
  neutral:
    max_gross_exposure: 1.0
    target_net_exposure: 0.2
  sideways:
    max_gross_exposure: 0.8
    target_net_exposure: 0.0
  bear:
    max_gross_exposure: 0.6
    target_net_exposure: 0.0
  crisis:
    max_gross_exposure: 0.3
    target_net_exposure: 0.0
  reflation:
    max_gross_exposure: 1.1
    target_net_exposure: 0.3

regime_config:  # Optional
  trend_ma_windows: [50, 200]
  vol_window: 20
  vov_window: 60
  breadth_ma_window: 50
  combine_macro_and_market: true
```

**Parameter:**
- **`max_gross_exposure`**: Maximale Brutto-Exposure (Long + Short) als Anteil des Kapitals
- **`target_net_exposure`**: Ziel-Netto-Exposure (Long - Short) als Anteil des Kapitals
  - Positive Werte = Net-Long (z.B. in Bull-Märkten)
  - Negative Werte = Net-Short (z.B. in Bear-Märkten)
  - 0.0 = Market-Neutral

#### 4.2 Backtest mit Regime-Overlay

**Beispiel-Command:**

```powershell
$env:ASSEMBLED_LOCAL_DATA_ROOT = "F:\Python_Projekt\Aktiengerüst\datensammlungen\altdaten\stand 3-12-2025"

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
```

**Alternativ ohne Config-File (verwendet Default-Risk-Map):**

```powershell
python scripts/cli.py run_backtest `
  --freq 1d `
  --strategy multifactor_long_short `
  --bundle-path config/factor_bundles/macro_world_etfs_core_bundle.yaml `
  --symbols-file config/macro_world_etfs_tickers.txt `
  --data-source local `
  --use-regime-overlay `
  --start-date 2010-01-01 `
  --end-date 2025-12-03
```

**Was passiert:**
1. **Regime-Detection**: Strategie berechnet tägliche Regime-Labels aus Preisen, Faktoren und Market Breadth
2. **Signale**: Multi-Faktor-Scores werden wie gewohnt berechnet
3. **Position-Sizing**: Bei jedem Rebalance-Datum wird das aktuelle Regime identifiziert und die entsprechenden Risikoparameter aus der Risk-Map geladen
4. **Exposure-Anpassung**:
   - Long-Seite: `(max_gross_exposure / 2 + target_net_exposure / 2)`
   - Short-Seite: `(max_gross_exposure / 2 - target_net_exposure / 2)`

**Logging:**
Pro Rebalance-Datum wird geloggt:
- Aktuelles Regime
- Verwendetes `max_gross_exposure` und `target_net_exposure`
- Anzahl Long/Short-Symbole
- Tatsächliche Gross/Net-Exposure

---

## Interpretation

### Typische Regime-Sequenzen

**Normale Marktzyklen:**
1. **Bull** → Positive Trends, niedrige Volatilität, Risk-On
2. **Neutral / Sideways** → Moderate Trends, moderate Volatilität
3. **Bear** → Negative Trends, erhöhte Volatilität, Risk-Off
4. **Crisis** → Extreme Volatilität, starke Risk-Aversion (selten, kurze Dauer)
5. **Recovery / Reflation** → Erholung, moderate Expansion

**Hinweis:** Regime-Übergänge sind nicht deterministisch. Statistiken helfen, typische Muster zu identifizieren.

### Exposure-Verhalten in verschiedenen Regimes

**Bull-Regime:**
- **Hohe Gross-Exposure** (z.B. 1.2x)
- **Net-Long** (z.B. +0.6x)
- → Strategie nutzt Aufwärtstrends voll aus

**Bear-Regime:**
- **Niedrige Gross-Exposure** (z.B. 0.6x)
- **Market-Neutral** (z.B. 0.0x)
- → Strategie reduziert Risiko, vermeidet Net-Long

**Crisis-Regime:**
- **Sehr niedrige Gross-Exposure** (z.B. 0.3x)
- **Defensiv** (0.0x Net-Exposure)
- → Strategie minimiert Risiko, nur essentielle Positionen

**Neutral/Sideways:**
- **Moderate Exposure** (z.B. 0.8–1.0x)
- **Leicht Long oder Neutral**
- → Strategie agiert vorsichtig, wartet auf klare Trends

### Problemdiagnose

**Problem: Strategie verliert in Crisis-Regime**

**Ursachen:**
1. **Regime-Detection zu spät**: Regime wird erst nach Beginn der Crisis erkannt
   - **Lösung**: Kürzere Windows für Volatilität/Market Breadth, frühere Warnsignale
2. **Exposure zu hoch**: `max_gross_exposure` für Crisis-Regime ist zu hoch
   - **Lösung**: Reduziere `max_gross_exposure` in Risk-Map (z.B. von 0.3 auf 0.2)
3. **Faktoren funktionieren nicht**: Faktoren, die in Bull-Regimes funktionieren, versagen in Crisis
   - **Lösung**: Verwende `evaluate_factor_ic_by_regime()` um regim-spezifische Faktoren zu identifizieren

**Problem: Regime-Übergänge zu häufig**

**Ursachen:**
1. **Noise in Inputs**: Market Breadth oder Volatilität schwanken stark
   - **Lösung**: Verwende längere Moving-Average-Windows (`breadth_ma_window`, `vol_window`)
2. **Sensitive Thresholds**: Regime-Klassifikations-Schwellenwerte sind zu sensitiv
   - **Lösung**: Passe Regime-Klassifikations-Logik an (in `build_regime_state()`)

**Problem: Regime-Labels stimmen nicht mit erwarteten Marktphasen überein**

**Ursachen:**
1. **Fehlende Macro-Faktoren**: Regime-Detection basiert nur auf Price-Daten
   - **Lösung**: Stelle sicher, dass Macro-Faktoren verfügbar sind (aus Alt-Data-Snapshot)
2. **Universe zu klein**: Market Breadth ist nicht aussagekräftig bei wenigen Symbolen
   - **Lösung**: Verwende größeres Universe oder prüfe `fraction_above_ma_50`-Werte

### Best Practices

1. **Start mit Default-Risk-Map**: Verwende die Standard-Konfiguration, bevor du eigene Parameter optimierst
2. **Analyse Regime-Statistiken**: Prüfe `compute_regime_transition_stats()` um typische Regime-Muster zu verstehen
3. **Factor-Performance nach Regime**: Identifiziere regim-spezifische Faktoren mit `evaluate_factor_ic_by_regime()`
4. **Backtest über verschiedene Regimes**: Stelle sicher, dass dein Backtest mehrere Regime-Zyklen umfasst (z.B. 2010–2025)
5. **Monitoring**: Logge Regime-Labels und Exposure-Parameter während des Backtests für spätere Analyse

---

## Weiterführende Dokumentation

- [Regime Models D1 Design Document](REGIME_MODELS_D1_DESIGN.md) – Detailliertes Design, Data Contracts, geplante Funktionen
- [Multi-Factor Strategy Workflow](WORKFLOWS_STRATEGIES_MULTIFACTOR.md) – Basis-Workflow für Multi-Factor-Strategien
- [Factor Analysis Workflow](WORKFLOWS_FACTOR_ANALYSIS.md) – IC-Berechnung und Factor-Rankings
- [Advanced Analytics & Factor Labs](../ADVANCED_ANALYTICS_FACTOR_LABS.md) – Gesamt-Roadmap Phase D

