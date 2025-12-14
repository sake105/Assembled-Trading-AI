# Regime Models & Risk Overlay – Design Document (D1)

**Last Updated:** 2025-12-10  
**Status:** Design Phase  
**Phase:** D1 – Regime Models & Risk Overlay (Advanced Analytics & Factor Labs)

---

## Overview

**Ziel:** Entwicklung eines Regime-Erkennungssystems, das Marktregime identifiziert und als Risk-Overlay für Strategien genutzt werden kann. Das System kombiniert vorhandene Inputs (Macro-Faktoren, Market Breadth, Volatilität) zu täglichen Regime-Labels und ermöglicht regime-adaptive Risikosteuerung.

**Grundprinzip:**
- **Keine neuen APIs**: Nutzt ausschließlich lokale Alt-Daten-Snapshots und bereits vorhandene Faktoren
- **Regime-Labels**: Tägliche Klassifikation in diskrete Regimes (z.B. `bull`, `bear`, `sideways`, `crisis`, `reflation`)
- **Risk Overlay**: Regime → Risikoparameter-Mapping für adaptive Strategien

**Integration:**
- Multi-Factor-Strategie: Exposure-Steuerung basierend auf Regime
- Event-Studien: CAAR-Analyse nach Regime
- Faktor-Analyse: IC/Sharpe nach Regime evaluieren

---

## Scope D1

### Verfügbare Inputs

Das Regime-Modell nutzt bereits vorhandene Faktoren und Indikatoren:

**1. Macro-Faktoren** (aus `altdata_news_macro_factors.py`):
- `macro_growth_regime`: Growth regime indicator (+1 = expansion, -1 = recession, 0 = neutral)
- `macro_inflation_regime`: Inflation regime indicator (+1 = high inflation, -1 = low/deflation, 0 = neutral)
- `macro_risk_aversion_proxy`: Risk-on/risk-off indicator

**2. Market Breadth** (aus `market_breadth.py`):
- `fraction_above_ma_{window}`: Fraction of symbols above moving average (z.B. `fraction_above_ma_50`, `fraction_above_ma_200`)
- `ad_line`: Advance/Decline Line (cumulative net advances)
- `risk_on_off_score`: Risk-on/risk-off score (-1 = risk-off, +1 = risk-on)

**3. Volatilität** (aus `ta_liquidity_vol_factors.py`):
- `rv_20`: Realized Volatility (20-day window, annualized)
- `vov_20_60`: Volatility of Volatility (rolling std of RV over 60-day window)

**4. Trend-Indikatoren** (optional, aus `ta_factors_core.py`):
- `trend_strength_50`, `trend_strength_200`: Trend strength indicators (können für Trend-Regime verwendet werden)

### Regime-Labels (Target Output)

**Diskrete Regime-Kategorien:**

| Regime Label | Beschreibung | Beispiel-Perioden |
|--------------|--------------|-------------------|
| `bull` | Starker Aufwärtstrend, hohe Breadth, niedrige Volatilität | 2017-2018, 2020-2021 |
| `bear` | Abwärtstrend, niedrige Breadth, hohe Volatilität | 2008-2009, 2022 |
| `sideways` | Seitwärtsmarkt, moderate Breadth, moderate Volatilität | 2015-2016 |
| `crisis` | Extrem hohe Volatilität, panikartige Verkäufe | März 2020, September 2008 |
| `reflation` | Expansion bei steigender Inflation, hohe Growth + Inflation | 2021-2022 (teilweise) |

**Hinweis:** Die genaue Anzahl und Definition der Regimes kann während der Implementierung verfeinert werden. Die obige Liste dient als Ausgangspunkt.

---

## Data Contracts

### Input DataFrames

**1. `prices_df`** (Standard Panel-Format):
```python
prices_df: pd.DataFrame
    - timestamp: pd.Timestamp (UTC-aware)
    - symbol: str
    - close: float
    - Optional: open, high, low, volume
```

**2. `macro_factors_df`** (Panel-Format mit Macro-Faktoren):
```python
macro_factors_df: pd.DataFrame
    - timestamp: pd.Timestamp (UTC-aware)
    - symbol: str (kann ignoriert werden, da Macro-Faktoren market-wide sind)
    - macro_growth_regime: float (+1, -1, 0, oder NaN)
    - macro_inflation_regime: float (+1, -1, 0, oder NaN)
    - macro_risk_aversion_proxy: float (oder NaN)
```

**3. `breadth_df`** (Time-Series-Format, eine Zeile pro Timestamp):
```python
breadth_df: pd.DataFrame
    - timestamp: pd.Timestamp (UTC-aware)
    - fraction_above_ma_50: float (0.0 bis 1.0)
    - fraction_above_ma_200: float (0.0 bis 1.0, optional)
    - ad_line: float (cumulative)
    - ad_line_normalized: float (optional)
    - risk_on_off_score: float (-1.0 bis +1.0)
```

**4. `vol_df`** (Panel-Format mit Volatilitäts-Faktoren):
```python
vol_df: pd.DataFrame
    - timestamp: pd.Timestamp (UTC-aware)
    - symbol: str
    - rv_20: float (annualized realized volatility)
    - vov_20_60: float (volatility of volatility, optional)
```

### Output DataFrame

**`regime_state_df`** (Time-Series-Format, eine Zeile pro Timestamp):

```python
regime_state_df: pd.DataFrame
    - timestamp: pd.Timestamp (UTC-aware)
    - regime_label: str (z.B. "bull", "bear", "sideways", "crisis", "reflation")
    - regime_trend_score: float (-1.0 bis +1.0)
        # Score für Trend-Stärke und Richtung
        # +1.0 = starker Bull-Markt, -1.0 = starker Bear-Markt, 0.0 = neutral
    - regime_macro_score: float (-1.0 bis +1.0)
        # Kombinierter Macro-Score aus Growth + Inflation
        # Höhere Werte = expansivere Macro-Umgebung
    - regime_risk_score: float (-1.0 bis +1.0)
        # Score für Risiko-Umgebung
        # +1.0 = risk-on, niedrige Volatilität, -1.0 = risk-off, hohe Volatilität
    - regime_confidence: float (0.0 bis 1.0, optional)
        # Konfidenz des Regime-Labels
        # 1.0 = sehr hohe Konfidenz, 0.0 = unsicher
```

**Beispiel:**
```
timestamp              regime_label  regime_trend_score  regime_macro_score  regime_risk_score  regime_confidence
2020-01-15 00:00:00+00:00  bull             0.75              0.50               0.80              0.85
2020-03-15 00:00:00+00:00  crisis          -0.90             -0.30              -0.95              0.95
2020-06-15 00:00:00+00:00  bull             0.60              0.40               0.70              0.75
```

---

## Geplante Funktionen (Interfaces)

### 1. `build_regime_state()`

**Hauptfunktion zur Berechnung der Regime-Labels.**

```python
def build_regime_state(
    prices: pd.DataFrame,
    macro_factors: pd.DataFrame | None = None,
    breadth_df: pd.DataFrame | None = None,
    vol_df: pd.DataFrame | None = None,
    trend_factors: pd.DataFrame | None = None,
    timestamp_col: str = "timestamp",
    group_col: str = "symbol",
    method: Literal["rule_based", "threshold", "ml"] = "rule_based",
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    Build regime state DataFrame from various input factors.
    
    Combines macro factors, market breadth, volatility, and trend indicators
    to produce daily regime labels and sub-scores.
    
    Args:
        prices: DataFrame with price data (panel format)
        macro_factors: Optional DataFrame with macro regime factors
                      (must have timestamp, macro_growth_regime, macro_inflation_regime, macro_risk_aversion_proxy)
        breadth_df: Optional DataFrame with market breadth indicators
                    (must have timestamp, fraction_above_ma_50, ad_line, risk_on_off_score)
        vol_df: Optional DataFrame with volatility factors
                (must have timestamp, rv_20)
        trend_factors: Optional DataFrame with trend strength factors
                      (must have timestamp, trend_strength_50, trend_strength_200)
        timestamp_col: Name of timestamp column (default: "timestamp")
        group_col: Name of symbol column (default: "symbol", unused if inputs are time-series)
        method: Regime classification method (default: "rule_based")
                - "rule_based": Rule-based classification using thresholds
                - "threshold": Similar to rule_based, but with configurable thresholds
                - "ml": Machine learning-based classification (future extension)
        config: Optional configuration dictionary for method-specific parameters
                (e.g., thresholds for rule-based classification)
    
    Returns:
        regime_state_df: DataFrame with columns:
            - timestamp: Timestamp (UTC)
            - regime_label: Regime label (str)
            - regime_trend_score: Trend score (-1.0 to +1.0)
            - regime_macro_score: Macro score (-1.0 to +1.0)
            - regime_risk_score: Risk score (-1.0 to +1.0)
            - regime_confidence: Confidence score (0.0 to 1.0, optional)
        
        One row per timestamp, sorted by timestamp.
    
    Raises:
        ValueError: If required columns are missing or inputs are invalid
        KeyError: If timestamp_col not found in inputs
    """
    pass
```

**Regime-Klassifikations-Logik (Rule-Based, Beispiel):**

1. **Regime Risk Score** (aus Volatilität und Risk-On/Off):
   - Hohe `rv_20` + niedrige `risk_on_off_score` → `regime_risk_score` = -1.0 (Crisis)
   - Niedrige `rv_20` + hohe `risk_on_off_score` → `regime_risk_score` = +1.0 (Risk-On)

2. **Regime Trend Score** (aus Market Breadth und Trend Strength):
   - Hohe `fraction_above_ma_50` + positive `trend_strength_200` → `regime_trend_score` = +1.0 (Bull)
   - Niedrige `fraction_above_ma_50` + negative `trend_strength_200` → `regime_trend_score` = -1.0 (Bear)

3. **Regime Macro Score** (aus Macro-Faktoren):
   - Kombination von `macro_growth_regime` und `macro_inflation_regime`
   - Expansion + moderate Inflation → `regime_macro_score` = +0.5 (Reflation)
   - Recession + Deflation → `regime_macro_score` = -1.0 (Crisis)

4. **Final Regime Label** (Kombination aller Scores):
   - `crisis`: `regime_risk_score` < -0.8 (extreme Volatilität)
   - `bear`: `regime_trend_score` < -0.5 UND `regime_risk_score` < 0
   - `bull`: `regime_trend_score` > +0.5 UND `regime_risk_score` > 0
   - `reflation`: `regime_macro_score` > +0.3 UND `macro_inflation_regime` > 0
   - `sideways`: Sonstiges (moderate Scores in alle Dimensionen)

### 2. `compute_regime_transition_stats()`

**Statistiken über Regime-Übergänge.**

```python
def compute_regime_transition_stats(
    regime_state_df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    regime_col: str = "regime_label",
) -> pd.DataFrame:
    """
    Compute statistics about regime transitions.
    
    Analyzes how often regimes transition to other regimes and typical
    duration of each regime.
    
    Args:
        regime_state_df: DataFrame from build_regime_state()
        timestamp_col: Name of timestamp column (default: "timestamp")
        regime_col: Name of regime label column (default: "regime_label")
    
    Returns:
        DataFrame with regime transition statistics:
            - from_regime: Source regime label
            - to_regime: Target regime label
            - count: Number of transitions
            - avg_duration_days: Average duration of source regime (before transition)
            - transition_probability: Probability of transitioning from source to target
    
    Example:
        from_regime  to_regime   count  avg_duration_days  transition_probability
        bull         bear          12    45.2               0.15
        bull         sideways       8    52.1               0.10
        bear         bull          15    38.5               0.25
    """
    pass
```

### 3. `evaluate_factor_by_regime()`

**Factor-Evaluation nach Regime (z.B. IC nach Regime).**

```python
def evaluate_factor_by_regime(
    factor_ic_df: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    factor_cols: str | list[str] | None = None,
    timestamp_col: str = "timestamp",
    ic_col_suffix: str = "_ic",
    regime_col: str = "regime_label",
) -> pd.DataFrame:
    """
    Evaluate factor effectiveness (IC) by regime.
    
    Computes IC statistics (mean IC, IC-IR, hit ratio) separately for each regime.
    This allows identification of factors that work well in specific regimes.
    
    Args:
        factor_ic_df: DataFrame with IC time-series (from compute_ic or compute_rank_ic)
                     Must have timestamp_col and IC columns (e.g., "returns_12m_ic")
        regime_state_df: DataFrame from build_regime_state()
                         Must have timestamp_col and regime_col
        factor_cols: Optional list of factor column names (if None, auto-detect IC columns)
                     IC columns are identified by suffix (default: "_ic")
        timestamp_col: Name of timestamp column (default: "timestamp")
        ic_col_suffix: Suffix for IC columns (default: "_ic")
        regime_col: Name of regime label column (default: "regime_label")
    
    Returns:
        DataFrame with IC statistics per factor and regime:
            - factor: Factor name
            - regime: Regime label
            - mean_ic: Mean IC in this regime
            - std_ic: Std of IC in this regime
            - ic_ir: IC-IR (mean_ic / std_ic) in this regime
            - hit_ratio: Percentage of periods with positive IC in this regime
            - n_observations: Number of observations (timestamps) in this regime
    
    Example:
        factor            regime    mean_ic  ic_ir  hit_ratio  n_observations
        returns_12m       bull      0.15     1.20   0.65       250
        returns_12m       bear      0.08     0.50   0.55       180
        returns_12m       crisis   -0.05     0.20   0.45        30
        trend_strength_200 bull    0.12     1.00   0.60       250
        trend_strength_200 bear   -0.10     0.80   0.40       180
    """
    pass
```

---

## Risk Overlay Konzept

**Ziel:** Mapping von Regime → Risikoparameter für adaptive Strategien.

### Risk Overlay Configuration

**Struktur:** Dictionary oder YAML-Datei mit Regime → Risikoparameter-Mapping.

**Beispiel-Konfiguration:**

```yaml
# config/risk_overlay_default.yaml
regime_risk_config:
  bull:
    max_gross_exposure: 1.0      # 100% gross exposure (50% long + 50% short)
    target_net_exposure: 0.1     # Slightly net long
    max_single_position_weight: 0.05  # Max 5% per position
    min_position_weight: 0.01    # Min 1% per position
    allow_trading: true
  
  bear:
    max_gross_exposure: 0.8      # Reduced exposure
    target_net_exposure: -0.2    # Net short
    max_single_position_weight: 0.04
    min_position_weight: 0.01
    allow_trading: true
  
  sideways:
    max_gross_exposure: 0.6      # Low exposure
    target_net_exposure: 0.0     # Market neutral
    max_single_position_weight: 0.03
    min_position_weight: 0.01
    allow_trading: true
  
  crisis:
    max_gross_exposure: 0.2      # Minimal exposure
    target_net_exposure: -0.1    # Slightly net short
    max_single_position_weight: 0.02
    min_position_weight: 0.005
    allow_trading: true          # Still allow trading, but very conservative
  
  reflation:
    max_gross_exposure: 0.9
    target_net_exposure: 0.2     # Net long (inflation hedge)
    max_single_position_weight: 0.05
    min_position_weight: 0.01
    allow_trading: true
```

**Tabellarische Übersicht:**

| Regime   | Max Gross Exposure | Target Net Exposure | Max Single Position | Allow Trading |
|----------|-------------------|---------------------|---------------------|---------------|
| `bull`   | 1.0 (100%)        | +0.1 (net long)     | 5%                  | Yes           |
| `bear`   | 0.8 (80%)         | -0.2 (net short)    | 4%                  | Yes           |
| `sideways` | 0.6 (60%)       | 0.0 (neutral)       | 3%                  | Yes           |
| `crisis` | 0.2 (20%)         | -0.1 (slightly short) | 2%                | Yes           |
| `reflation` | 0.9 (90%)      | +0.2 (net long)     | 5%                  | Yes           |

**Optional: "No Trading" Regime:**

Falls gewünscht, kann ein Regime auch `allow_trading: false` haben, was bedeutet, dass die Strategie in diesem Regime keine neuen Positionen eingeht (bestehende Positionen können aufgelöst werden).

### Risk Overlay Funktion

```python
def apply_risk_overlay(
    target_positions: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    current_timestamp: pd.Timestamp,
    risk_config: dict[str, dict[str, Any]],
    timestamp_col: str = "timestamp",
    regime_col: str = "regime_label",
) -> pd.DataFrame:
    """
    Apply risk overlay to target positions based on current regime.
    
    Adjusts target positions according to regime-specific risk limits:
    - Scales positions to max_gross_exposure
    - Adjusts net exposure to target_net_exposure
    - Enforces max_single_position_weight limits
    - Optionally blocks trading if allow_trading=False
    
    Args:
        target_positions: DataFrame with target positions (columns: symbol, target_weight, target_qty)
        regime_state_df: DataFrame from build_regime_state()
        current_timestamp: Current timestamp to look up regime
        risk_config: Dictionary mapping regime_label -> risk parameters
                     (see example YAML structure above)
        timestamp_col: Name of timestamp column (default: "timestamp")
        regime_col: Name of regime label column (default: "regime_label")
    
    Returns:
        Adjusted target_positions DataFrame with columns:
            - symbol: Symbol
            - target_weight: Adjusted target weight
            - target_qty: Adjusted target quantity
            - original_weight: Original target weight (for reference)
            - regime: Current regime label (for reference)
    
    Raises:
        KeyError: If regime not found in risk_config
        ValueError: If current_timestamp not found in regime_state_df
    """
    pass
```

---

## Integration

### 1. Multi-Factor-Strategie (Risk Overlay)

**Ziel:** Regime-adaptive Exposure-Steuerung in der Multi-Factor Long/Short-Strategie.

**Integration-Punkte:**
- **Pre-Trade Risk Check**: Vor Position-Sizing prüfen, welches Regime aktuell ist
- **Position Scaling**: Anpassen der Position-Größen basierend auf `max_gross_exposure` und `max_single_position_weight`
- **Net Exposure Adjustment**: Anpassen der Long/Short-Balance basierend auf `target_net_exposure`
- **Hedging** (optional, z.B. D1.3+): In `crisis`-Regime können zusätzliche Hedges eingebaut werden

**Beispiel-Workflow:**
```python
# 1. Build regime state (einmal pro Tag)
regime_state = build_regime_state(prices, macro_factors, breadth_df, vol_df)

# 2. Generate multi-factor signals (wie bisher)
signals = generate_multifactor_long_short_signals(prices, factors, config)

# 3. Compute target positions (wie bisher)
target_positions = compute_multifactor_long_short_positions(signals, capital, config)

# 4. Apply risk overlay (NEU)
current_regime = regime_state[regime_state["timestamp"] == current_timestamp]["regime_label"].iloc[0]
adjusted_positions = apply_risk_overlay(
    target_positions,
    regime_state,
    current_timestamp,
    risk_config,
)
```

### 2. Event-Studien (CAAR nach Regime)

**Ziel:** Abnormal Returns nach Events getrennt nach Regime analysieren.

**Integration:**
- Event-Study-Framework erweitern um Regime-Information
- CAAR (Cumulative Abnormal Return) getrennt nach Regime berechnen
- Beispiel: "Wie reagieren Earnings-Events in Bull- vs. Bear-Märkten?"

**Beispiel-Output:**
```
Event Type    Regime   CAAR_3d   CAAR_5d   CAAR_10d  Significance
earnings      bull     0.02      0.03      0.04      Yes
earnings      bear    -0.01      0.00      0.01      No
insider_buy   bull     0.015     0.025     0.035     Yes
insider_buy   bear     0.005     0.010     0.015     Yes (weaker)
```

### 3. Faktor-Analyse (IC/Sharpe nach Regime)

**Ziel:** Faktor-Performance nach Regime evaluieren.

**Integration:**
- `evaluate_factor_by_regime()` nutzen, um IC-Statistiken nach Regime zu berechnen
- Portfolio-Returns (Sharpe, DSR) nach Regime aufteilen
- Beispiel: "Funktioniert Momentum besser in Bull-Märkten?"

**Beispiel-Output:**
```
Factor            Regime    IC-IR    Sharpe   Deflated Sharpe
returns_12m       bull      1.20     2.50     1.80
returns_12m       bear      0.50     0.80     0.40
trend_strength_200 bull    1.00     2.20     1.50
trend_strength_200 bear   -0.30     0.20    -0.10
```

---

## Umsetzungsschritte D1

### D1.1: Core Regime-Modul + Data Contracts

**Ziel:** Basis-Implementierung von `build_regime_state()` mit Rule-Based-Klassifikation.

**Aufgaben:**
- [ ] Erstelle `src/assembled_core/regime/regime_detection.py`
- [ ] Implementiere `build_regime_state()` mit Rule-Based-Methode
- [ ] Definiere Regime-Labels und Klassifikations-Logik
- [ ] Implementiere Sub-Scores (`regime_trend_score`, `regime_macro_score`, `regime_risk_score`)
- [ ] Optional: Implementiere `regime_confidence`
- [ ] Erstelle Tests (`tests/test_regime_detection.py`)

**Akzeptanzkriterien:**
- Funktion läuft mit verfügbaren Inputs (macro_factors, breadth_df, vol_df)
- Regime-Labels sind plausibel (manuelle Prüfung an bekannten Perioden, z.B. März 2020 = crisis)
- Output-Format entspricht Data Contract (`regime_state_df`)

### D1.2: Regime-Auswertung vs. Faktoren

**Ziel:** Factor-Evaluation nach Regime ermöglichen.

**Aufgaben:**
- [ ] Implementiere `evaluate_factor_by_regime()` in `regime_detection.py`
- [ ] Implementiere `compute_regime_transition_stats()` in `regime_detection.py`
- [ ] Erweitere `scripts/cli.py analyze_factors` um `--regime-analysis` Flag (optional)
- [ ] Erstelle Tests für Regime-basierte Factor-Evaluation

**Akzeptanzkriterien:**
- IC-Statistiken können nach Regime aufgeteilt werden
- Regime-Übergangs-Statistiken sind korrekt
- CLI-Integration funktioniert

### D1.3: Integration in Multi-Factor-Strategie als Risk-Overlay

**Ziel:** Risk Overlay in Multi-Factor-Strategie integrieren.

**Aufgaben:**
- [ ] Erstelle `src/assembled_core/regime/risk_overlay.py` mit `apply_risk_overlay()`
- [ ] Definiere Standard Risk-Overlay-Config (YAML oder Python-Dict)
- [ ] Integriere Risk Overlay in `scripts/run_backtest_strategy.py` für `multifactor_long_short`
- [ ] Erweitere `MultiFactorStrategyConfig` um `risk_overlay_config` Parameter
- [ ] Erstelle Tests für Risk-Overlay-Logik

**Akzeptanzkriterien:**
- Risk Overlay kann in Backtest-Strategie aktiviert werden
- Position-Größen werden korrekt skaliert basierend auf Regime
- Net Exposure wird korrekt angepasst
- Strategie-Performance kann mit/ohne Risk Overlay verglichen werden

### D1.4: Tests + Dokumentation

**Ziel:** Vollständige Testabdeckung und Dokumentation.

**Aufgaben:**
- [ ] Erstelle umfassende Tests für alle Funktionen
- [ ] Dokumentiere Regime-Klassifikations-Logik
- [ ] Erstelle Beispiel-Workflows in `docs/WORKFLOWS_REGIME_DETECTION.md` (optional, oder in bestehendes Dokument integrieren)
- [ ] Update `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md` mit D1-Status

**Akzeptanzkriterien:**
- Alle Tests bestehen
- Dokumentation ist vollständig und verständlich
- Beispiel-Workflows sind verfügbar

---

## Technische Anforderungen

**Dependencies:**
- Keine neuen externen Dependencies (nutzt nur pandas, numpy)
- Optional: scikit-learn für ML-basierte Klassifikation (D1.1+ Extension)

**Performance:**
- `build_regime_state()` sollte schnell sein (< 1 Sekunde für 10 Jahre tägliche Daten)
- Regime-State kann gecacht werden (einmal pro Tag berechnen)

**Data Quality:**
- Robuste Behandlung von fehlenden Daten (NaN in Inputs)
- Forward-Fill von Regime-Labels, falls Lücken in Inputs (z.B. fehlende Macro-Daten)

---

## Offene Fragen / Erweiterungen

**D1.1+ Extensions:**
- ML-basierte Regime-Klassifikation (z.B. Hidden Markov Model, k-means Clustering)
- Regime-Probabilistic Scores (statt diskreter Labels)
- Multi-Timeframe-Regimes (z.B. kurzfristiges vs. langfristiges Regime)

**D1.3+ Extensions:**
- Dynamisches Hedging basierend auf Regime
- Regime-spezifische Factor-Bundles (verschiedene Faktoren in verschiedenen Regimes)
- Portfolio-Optimierung mit Regime-Constraints

**Integration mit anderen Phasen:**
- Phase D2: Adaptive Factor Selection (wähle Faktoren basierend auf Regime)
- Phase D3: Regime-Aware Risk Models (VAR, CVaR nach Regime)

---

## Verweise

**Verwandte Dokumentation:**
- [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) - Gesamter Factor-Labs-Roadmap
- [Workflows – Factor Analysis](WORKFLOWS_FACTOR_ANALYSIS.md) - Factor-Evaluation-Workflows
- [Workflows – Multi-Factor Long/Short Strategy](WORKFLOWS_STRATEGIES_MULTIFACTOR.md) - Multi-Factor-Strategie
- [Workflows – Event Studies](WORKFLOWS_EVENT_STUDIES.md) - Event-Study-Workflows

**Code-Referenzen:**
- `src/assembled_core/features/market_breadth.py` - Market Breadth Indikatoren
- `src/assembled_core/features/altdata_news_macro_factors.py` - Macro-Faktoren
- `src/assembled_core/features/ta_liquidity_vol_factors.py` - Volatilitäts-Faktoren
- `src/assembled_core/qa/factor_analysis.py` - Factor-Analyse-Module
- `src/assembled_core/strategies/multifactor_long_short.py` - Multi-Factor-Strategie

