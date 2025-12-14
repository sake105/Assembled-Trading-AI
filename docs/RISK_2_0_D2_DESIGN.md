# Risk 2.0 & Attribution Design (Sprint D2)

**Phase D2** – Advanced Analytics & Factor Labs

**Status:** 📋 Design Phase  
**Last Updated:** 2025-12-13

---

## Overview

**Ziel:** Erweiterte Risk-Analyse und Performance-Attribution für Backtests mit Segmentierung nach Regime, Faktor-Gruppen und Universes.

Dieses Design-Dokument beschreibt die geplante Erweiterung des Risk-Moduls (Phase D2) für:
- **Erweiterte Risk-Metriken**: Volatilität, Sharpe, Sortino, Max Drawdown, Calmar, Skew, Kurtosis, Tail-Risiken (VaR/ES)
- **Exposure-Analyse**: Gross/Net Exposure, Konzentration (HHI), Turnover über Zeit
- **Risiko-Segmentierung**: Nach Regime, Faktor-Gruppen, Universes/Factor-Sets
- **Performance-Attribution**: Einfache Attribution nach Faktor-Gruppen

**Bezug zu bestehenden Modulen:**
- Baut auf `qa/metrics.py` auf (PerformanceMetrics, Sharpe, Sortino, etc.)
- Nutzt `qa/risk_metrics.py` (VaR/ES)
- Integriert `risk/regime_models.py` (Regime-Detection aus D1)
- Erweitert `qa/backtest_engine.py` (BacktestResult)

**Datenbasis:** Alle Analysen basieren auf **lokalen Alt-Daten-Snapshots** und Backtest-Outputs. Keine Live-APIs.

---

## Scope D2

### 1. Erweiterte Risk-Metriken für Backtests

**Basis-Metriken (bereits in `qa/metrics.py` vorhanden):**
- Volatilität (annualisiert)
- Sharpe Ratio, Sortino Ratio
- Max Drawdown, Calmar Ratio
- VaR (95%)

**Neu hinzuzufügende Metriken:**
- **Skewness**: Verteilungssymmetrie der Returns (negative = linksschief, häufiger große Verluste)
- **Kurtosis**: "Tail-Heaviness" der Returns (hohe Kurtosis = häufige Extreme)
- **Expected Shortfall (ES)**: Conditional VaR (durchschnittlicher Verlust in Worst-Case-Szenarien)
- **Win Rate / Hit Rate**: Bereits vorhanden in `qa/metrics.py`, aber als Teil des Risk-Reports
- **Tail Ratio**: Verhältnis von 95. Perzentil zu 5. Perzentil der Returns

**Berechnungsmethode:**
- **Skewness/Kurtosis**: Standard-Pandas-Funktionen (`pd.Series.skew()`, `pd.Series.kurtosis()`)
- **ES**: Erweiterung von `qa/risk_metrics.py::compute_portfolio_risk_metrics()` (bereits teilweise vorhanden)
- **Tail Ratio**: `np.percentile(returns, 95) / abs(np.percentile(returns, 5))`

### 2. Exposure-Zeitreihen

**Metriken pro Timestamp:**
- **Gross Exposure**: Summe der absoluten Positionen (Long + Short)
- **Net Exposure**: Summe der Positionen (Long - Short)
- **Number of Positions**: Anzahl nicht-null Positionen
- **HHI Concentration**: Herfindahl-Hirschman Index (Summe der quadrierten Gewichte)
- **Turnover**: Optional, wenn Trades verfügbar (Notional gehandelt / Average Equity)

**Berechnung aus Positions-DataFrame:**
```python
# Input: positions_df mit Spalten: timestamp, symbol, weight (oder qty)
# Output: DataFrame mit timestamp und exposure-Metriken
```

### 3. Risiko nach Regime

**Segmentierung:**
- Sharpe, Volatilität, Max Drawdown, Calmar, Win Rate pro identifiziertem Regime
- Anzahl Perioden (Tage) pro Regime
- Performance-Metriken (Total Return, CAGR) pro Regime

**Input:**
- `equity_curve_df`: Equity-Zeitreihe mit Returns
- `regime_state_df`: Regime-Labels pro Tag (aus D1)

**Output:**
- DataFrame mit einer Zeile pro Regime
- Spalten: `regime`, `n_periods`, `sharpe`, `volatility`, `max_drawdown`, `total_return`, etc.

### 4. Risiko nach Faktor-Gruppen

**Einfache Performance-Attribution:**
- Gruppierung von Faktoren in Kategorien:
  - **Trend**: `returns_12m`, `trend_strength_50`, `trend_strength_200`
  - **Vol/Liq**: `rv_20`, `vov_20_60`, `turnover_20d`
  - **Earnings**: `earnings_eps_surprise_last`, `post_earnings_drift_20d`
  - **Insider**: `insider_net_notional_60d`, `insider_buy_ratio_60d`
  - **News/Macro**: `news_sentiment_trend_20d`, `macro_growth_regime`
- **Attribution-Methode**: Korrelations-basiert oder Exposure-basiert
  - **Korrelations-Methode**: Korrelation zwischen Portfolio-Returns und Faktor-Returns
  - **Exposure-Methode**: Gewichtete Summe der Faktor-Exposures (falls Positionen mit Faktor-Werten verknüpft werden können)

**Input:**
- `returns`: Portfolio-Returns (Zeitreihe)
- `factor_panel_df`: Faktor-Werte pro Symbol und Timestamp
- `positions_df`: Positionen pro Symbol und Timestamp
- `factor_groups`: Dictionary mapping Gruppe → Liste von Faktor-Namen

**Output:**
- DataFrame mit einer Zeile pro Faktor-Gruppe
- Spalten: `factor_group`, `correlation_with_returns`, `avg_exposure` (falls berechenbar), `contribution_to_returns` (approximiert)

### 5. Risiko nach Universen / Factor-Sets

**Vergleich über verschiedene Konfigurationen:**
- Metriken-Vergleich bei verschiedenen Universes (z.B. `macro_world_etfs` vs. `universe_ai_tech`)
- Metriken-Vergleich bei verschiedenen Factor-Sets (z.B. `core` vs. `core+alt_full`)
- Wird primär über separate Backtest-Runs realisiert (nicht innerhalb eines Backtests)

**Hinweis:** Diese Segmentierung wird nicht direkt im Risk-Report implementiert, sondern durch Vergleich mehrerer Reports erreicht.

---

## Data Contracts

### 1. Input DataFrames

#### `equity_curve_df`
**Format:** Time-Series (eine Zeile pro Timestamp)

**Spalten:**
- `timestamp`: `pd.Timestamp` (UTC-aware)
- `equity`: `float` (Portfolio-Equity zum Timestamp)
- `returns`: `float` (optional, tägliche Returns; wird berechnet falls nicht vorhanden)

**Beispiel:**
```
timestamp              equity     returns
2020-01-01 00:00:00+00:00  10000.0      NaN
2020-01-02 00:00:00+00:00  10100.0    0.0100
2020-01-03 00:00:00+00:00  10050.0   -0.0049
...
```

#### `positions_df`
**Format:** Panel-Format (mehrere Zeilen pro Timestamp)

**Spalten:**
- `timestamp`: `pd.Timestamp` (UTC-aware)
- `symbol`: `str` (Symbol-Name)
- `weight`: `float` (Position-Gewicht als Anteil des Kapitals; kann negativ für Short sein)
  - ODER `qty`: `float` (Position-Quantity; wird in Weight umgerechnet falls notwendig)

**Beispiel:**
```
timestamp              symbol  weight
2020-01-01 00:00:00+00:00  AAPL     0.10
2020-01-01 00:00:00+00:00  MSFT     0.15
2020-01-01 00:00:00+00:00  GOOGL   -0.05  (Short)
...
```

#### `trades_df` (optional)
**Format:** Panel-Format (mehrere Zeilen pro Timestamp)

**Spalten:**
- `timestamp`: `pd.Timestamp` (UTC-aware)
- `symbol`: `str`
- `side`: `str` ("BUY" oder "SELL")
- `qty`: `float` (absolute Quantity)
- `price`: `float` (Ausführungspreis)

**Verwendung:** Für Turnover-Berechnung (optional)

#### `regime_state_df`
**Format:** Time-Series (eine Zeile pro Timestamp)

**Spalten:**
- `timestamp`: `pd.Timestamp` (UTC-aware)
- `regime_label`: `str` (Regime-Label: "bull", "bear", "sideways", "crisis", "reflation", "neutral")
- Weitere Spalten: `regime_trend_score`, `regime_macro_score`, `regime_risk_score`, `regime_confidence` (optional)

**Siehe:** `docs/REGIME_MODELS_D1_DESIGN.md` für vollständige Spezifikation

#### `factor_panel_df` (optional, für Faktor-Attribution)
**Format:** Panel-Format (mehrere Zeilen pro Timestamp)

**Spalten:**
- `timestamp`: `pd.Timestamp` (UTC-aware)
- `symbol`: `str`
- `factor_*`: `float` (Faktor-Werte, z.B. `returns_12m`, `trend_strength_50`, etc.)

**Beispiel:**
```
timestamp              symbol  returns_12m  trend_strength_50  rv_20
2020-01-01 00:00:00+00:00  AAPL        0.15              0.80   0.25
2020-01-01 00:00:00+00:00  MSFT        0.12              0.75   0.20
...
```

### 2. Output DataFrames

#### `exposure_timeseries_df`
**Format:** Time-Series (eine Zeile pro Timestamp)

**Spalten:**
- `timestamp`: `pd.Timestamp` (UTC-aware)
- `gross_exposure`: `float` (Summe der absoluten Gewichte)
- `net_exposure`: `float` (Summe der Gewichte, kann negativ sein)
- `n_positions`: `int` (Anzahl nicht-null Positionen)
- `hhi_concentration`: `float` (Herfindahl-Hirschman Index, 0.0-1.0)
- `turnover`: `float` (optional, annualisiert, falls Trades verfügbar)

#### `risk_by_regime_df`
**Format:** Time-Series (eine Zeile pro Regime)

**Spalten:**
- `regime`: `str` (Regime-Label)
- `n_periods`: `int` (Anzahl Perioden in diesem Regime)
- `sharpe_ratio`: `float | None`
- `sortino_ratio`: `float | None`
- `volatility`: `float | None` (annualisiert)
- `max_drawdown_pct`: `float | None`
- `calmar_ratio`: `float | None`
- `total_return`: `float`
- `cagr`: `float | None`
- `win_rate`: `float | None` (falls Trades verfügbar)

#### `risk_by_factor_group_df`
**Format:** Time-Series (eine Zeile pro Faktor-Gruppe)

**Spalten:**
- `factor_group`: `str` (z.B. "Trend", "Vol/Liq", "Earnings")
- `factors`: `list[str]` (Liste der Faktoren in dieser Gruppe)
- `correlation_with_returns`: `float | None` (Korrelation zwischen Faktor-Returns und Portfolio-Returns)
- `avg_exposure`: `float | None` (Durchschnittliche Exposure zu Faktoren dieser Gruppe, falls berechenbar)
- `estimated_contribution`: `float | None` (Approximierter Beitrag zu Portfolio-Returns)

---

## Geplante Funktionen (Interfaces)

### Modul: `src/assembled_core/risk/risk_metrics_advanced.py` (neu)

**Hinweis:** Die Basis-Risk-Metriken sind bereits in `qa/metrics.py` und `qa/risk_metrics.py` vorhanden. Dieses Modul erweitert sie um:
- Skewness/Kurtosis
- Exposure-Zeitreihen
- Regime-Segmentierung
- Faktor-Attribution

```python
from dataclasses import dataclass
from typing import Literal
import pandas as pd

@dataclass
class AdvancedRiskMetrics:
    """Erweiterte Risk-Metriken (erweitert PerformanceMetrics aus qa.metrics)."""
    # Basis-Metriken (aus PerformanceMetrics)
    sharpe_ratio: float | None
    sortino_ratio: float | None
    volatility: float | None
    max_drawdown_pct: float | None
    calmar_ratio: float | None
    var_95: float | None
    
    # Neue Metriken
    skewness: float | None
    kurtosis: float | None
    expected_shortfall_95: float | None
    tail_ratio: float | None
    
    # Metadata
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    n_periods: int


def compute_basic_risk_metrics(
    returns: pd.Series,
    equity: pd.Series | None = None,
    freq: Literal["1d", "5min"] = "1d",
    risk_free_rate: float = 0.0,
) -> dict[str, float | None]:
    """
    Berechnet erweiterte Risk-Metriken aus Returns.
    
    Args:
        returns: Zeitreihe der täglichen Returns
        equity: Optional, Equity-Zeitreihe (für VaR/ES in absoluten Werten)
        freq: Trading-Frequenz für Annualisierung
        risk_free_rate: Risk-free Rate (annualisiert)
    
    Returns:
        Dictionary mit Metriken:
        - sharpe_ratio, sortino_ratio, volatility (annualisiert)
        - max_drawdown_pct, calmar_ratio
        - var_95, expected_shortfall_95
        - skewness, kurtosis, tail_ratio
    
    Hinweis: Nutzt intern qa.metrics für Basis-Metriken.
    """
    pass


def compute_exposure_timeseries(
    positions: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    equity: pd.DataFrame | None = None,
    timestamp_col: str = "timestamp",
    weight_col: str = "weight",
    freq: Literal["1d", "5min"] = "1d",
) -> pd.DataFrame:
    """
    Berechnet Exposure-Zeitreihen aus Positions-DataFrame.
    
    Args:
        positions: DataFrame mit Spalten: timestamp, symbol, weight (oder qty)
        trades: Optional, für Turnover-Berechnung
        equity: Optional, für Turnover-Berechnung (umgerechnet auf Anteil des Kapitals)
        timestamp_col: Name der Timestamp-Spalte
        weight_col: Name der Weight-Spalte ("weight" oder "qty")
        freq: Trading-Frequenz für Annualisierung
    
    Returns:
        DataFrame mit Spalten:
        - timestamp
        - gross_exposure: Summe(abs(weight))
        - net_exposure: Summe(weight)
        - n_positions: Anzahl nicht-null Positionen
        - hhi_concentration: HHI (Summe(weight^2))
        - turnover: Optional, falls trades und equity verfügbar
    
    Raises:
        ValueError: Wenn required Spalten fehlen
    """
    pass


def compute_risk_by_regime(
    equity_curve: pd.DataFrame,
    regime_state_df: pd.DataFrame,
    trades: pd.DataFrame | None = None,
    timestamp_col: str = "timestamp",
    regime_col: str = "regime_label",
    freq: Literal["1d", "5min"] = "1d",
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Berechnet Risk-Metriken pro Regime.
    
    Args:
        equity_curve: DataFrame mit Spalten: timestamp, equity (und optional returns)
        regime_state_df: DataFrame mit Spalten: timestamp, regime_label
        trades: Optional, für Win-Rate-Berechnung
        timestamp_col: Name der Timestamp-Spalte
        regime_col: Name der Regime-Spalte
        freq: Trading-Frequenz für Annualisierung
        risk_free_rate: Risk-free Rate (annualisiert)
    
    Returns:
        DataFrame mit einer Zeile pro Regime:
        - regime: Regime-Label
        - n_periods: Anzahl Perioden
        - sharpe_ratio, sortino_ratio, volatility, max_drawdown_pct, calmar_ratio
        - total_return, cagr
        - win_rate: Optional, falls trades verfügbar
    
    Raises:
        ValueError: Wenn required Spalten fehlen oder keine Overlaps zwischen equity und regime
    """
    pass


def compute_risk_by_factor_group(
    returns: pd.Series,
    factor_panel_df: pd.DataFrame,
    positions_df: pd.DataFrame | None = None,
    factor_groups: dict[str, list[str]] | None = None,
    timestamp_col: str = "timestamp",
    symbol_col: str = "symbol",
    method: Literal["correlation", "exposure"] = "correlation",
) -> pd.DataFrame:
    """
    Berechnet Performance-Attribution nach Faktor-Gruppen.
    
    Args:
        returns: Portfolio-Returns (Zeitreihe, Index = timestamp)
        factor_panel_df: DataFrame mit Spalten: timestamp, symbol, factor_*
        positions_df: Optional, für Exposure-basierte Attribution
        factor_groups: Dictionary mapping Gruppe → Liste von Faktor-Namen
                      Default: {
                          "Trend": ["returns_12m", "trend_strength_50", "trend_strength_200"],
                          "Vol/Liq": ["rv_20", "vov_20_60", "turnover_20d"],
                          "Earnings": ["earnings_eps_surprise_last", "post_earnings_drift_20d"],
                          "Insider": ["insider_net_notional_60d", "insider_buy_ratio_60d"],
                          "News/Macro": ["news_sentiment_trend_20d", "macro_growth_regime"],
                      }
        timestamp_col: Name der Timestamp-Spalte
        symbol_col: Name der Symbol-Spalte
        method: "correlation" (Korrelation zwischen Faktor- und Portfolio-Returns) oder
                "exposure" (gewichtete Summe der Faktor-Exposures)
    
    Returns:
        DataFrame mit einer Zeile pro Faktor-Gruppe:
        - factor_group: Gruppen-Name
        - factors: Liste der Faktoren
        - correlation_with_returns: Korrelation (bei method="correlation")
        - avg_exposure: Durchschnittliche Exposure (bei method="exposure")
        - estimated_contribution: Approximierter Beitrag zu Returns
    
    Raises:
        ValueError: Wenn required Spalten fehlen
        NotImplementedError: Falls method="exposure" aber positions_df nicht verfügbar
    """
    pass
```

---

## Risk Report Konzept

### Struktur

Ein **Risk Report** ist ein zentrales Objekt/Dictionary, das alle Risk-Analysen sammelt:

```python
@dataclass
class RiskReport:
    """Komplettes Risk-Report für einen Backtest."""
    # Global Metrics (aus qa.metrics.PerformanceMetrics + AdvancedRiskMetrics)
    global_metrics: dict[str, float | None]
    
    # Exposure-Zeitreihen
    exposure_timeseries: pd.DataFrame
    
    # Risk nach Regime
    risk_by_regime: pd.DataFrame
    
    # Risk nach Faktor-Gruppen
    risk_by_factor_group: pd.DataFrame | None
    
    # Metadata
    backtest_id: str | None
    strategy: str | None
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    freq: str
    universe: str | None
    factor_set: str | None
```

### Output-Formate

#### 1. Markdown-Report (`*_risk_report.md`)

**Struktur:**
```markdown
# Risk Report

## Overview
- Strategy: multifactor_long_short
- Period: 2020-01-01 to 2025-12-03
- Frequency: 1d

## Global Risk Metrics
- Sharpe Ratio: 1.25
- Sortino Ratio: 1.45
- Volatility: 15.2%
- Max Drawdown: -18.5%
- Calmar Ratio: 0.67
- Skewness: -0.3
- Kurtosis: 2.8
- VaR (95%): -2.1%
- ES (95%): -3.2%

## Exposure Analysis
- Average Gross Exposure: 1.05x
- Average Net Exposure: 0.12x
- Average Number of Positions: 25
- Average HHI Concentration: 0.08

## Risk by Regime
| Regime | Periods | Sharpe | Volatility | MaxDD | Total Return |
|--------|---------|--------|------------|-------|--------------|
| bull   | 1250    | 1.45   | 12.5%      | -10%  | +45.2%       |
| bear   | 450     | 0.85   | 18.2%      | -25%  | -8.5%        |
| crisis | 50      | -0.5   | 32.5%      | -35%  | -15.2%       |

## Risk by Factor Group
| Factor Group | Correlation | Avg Exposure | Contribution |
|--------------|-------------|--------------|--------------|
| Trend        | 0.45        | 0.35         | +12.5%       |
| Vol/Liq      | -0.15       | 0.20         | -2.1%        |
| Earnings     | 0.25        | 0.15         | +5.8%        |
```

#### 2. CSV-Outputs

**`*_risk_summary.csv`:**
- Eine Zeile mit globalen Metriken
- Spalten: `sharpe_ratio`, `sortino_ratio`, `volatility`, `max_drawdown_pct`, etc.

**`*_risk_by_regime.csv`:**
- Eine Zeile pro Regime
- Spalten: `regime`, `n_periods`, `sharpe_ratio`, `volatility`, etc.

**`*_risk_by_factor_group.csv`:**
- Eine Zeile pro Faktor-Gruppe
- Spalten: `factor_group`, `factors`, `correlation_with_returns`, etc.

**`*_exposure_timeseries.csv`:**
- Eine Zeile pro Timestamp
- Spalten: `timestamp`, `gross_exposure`, `net_exposure`, `n_positions`, `hhi_concentration`, `turnover`

### Generierung

**Funktion:**
```python
def generate_risk_report(
    equity_curve: pd.DataFrame,
    positions_df: pd.DataFrame | None = None,
    trades_df: pd.DataFrame | None = None,
    regime_state_df: pd.DataFrame | None = None,
    factor_panel_df: pd.DataFrame | None = None,
    factor_groups: dict[str, list[str]] | None = None,
    start_capital: float = 10000.0,
    freq: Literal["1d", "5min"] = "1d",
    risk_free_rate: float = 0.0,
    output_dir: Path | str | None = None,
    backtest_id: str | None = None,
    strategy: str | None = None,
    universe: str | None = None,
    factor_set: str | None = None,
) -> tuple[RiskReport, Path]:
    """
    Generiert vollständiges Risk-Report aus Backtest-Ergebnissen.
    
    Args:
        equity_curve: Equity-Zeitreihe
        positions_df: Optional, für Exposure-Analyse
        trades_df: Optional, für Turnover und Win-Rate
        regime_state_df: Optional, für Regime-Segmentierung
        factor_panel_df: Optional, für Faktor-Attribution
        factor_groups: Optional, für Faktor-Attribution
        start_capital: Start-Kapital
        freq: Trading-Frequenz
        risk_free_rate: Risk-free Rate
        output_dir: Output-Verzeichnis (default: settings.output_dir)
        backtest_id: Optional, Backtest-ID für Report-Namen
        strategy: Optional, Strategie-Name
        universe: Optional, Universe-Name
        factor_set: Optional, Factor-Set-Name
    
    Returns:
        Tuple von (RiskReport-Objekt, Pfad zu Markdown-Report)
    
    Side Effects:
        Schreibt Markdown-Report und CSV-Dateien in output_dir
    """
    pass
```

---

## Integration

### 1. Post-Processing in Backtests

**Option A: Automatisch nach `run_backtest_strategy.py`**

Erweitere `scripts/run_backtest_strategy.py`:
- Neues CLI-Argument: `--with-risk-report`
- Nach Backtest-Abschluss: Rufe `generate_risk_report()` auf
- Schreibe Reports in Output-Verzeichnis

**Beispiel-Command:**
```powershell
python scripts/cli.py run_backtest `
  --freq 1d `
  --strategy multifactor_long_short `
  --bundle-path config/factor_bundles/macro_world_etfs_core_bundle.yaml `
  --with-risk-report `
  --start-date 2020-01-01 `
  --end-date 2025-12-03
```

**Option B: Separates Script für nachträgliche Analyse**

Neues Script: `scripts/generate_risk_report.py`

**CLI-Interface:**
```powershell
python scripts/generate_risk_report.py `
  --equity-file output/backtest/equity_curve_5min.csv `
  --positions-file output/backtest/positions_5min.csv `
  --trades-file output/backtest/trades_5min.csv `
  --regime-state-file output/regime/regime_state_5min.parquet `
  --factor-panel-file output/factor_analysis/factors_5min.parquet `
  --output-dir output/risk_reports
```

**Vorteil Option B:**
- Flexibler: Kann auch für bestehende Backtest-Ergebnisse verwendet werden
- Unabhängig: Risk-Report-Generierung ist optional und kann später nachgeholt werden

### 2. Integration mit BacktestResult

**Erweiterung von `qa/backtest_engine.py::BacktestResult`:**
- Optionales Feld: `risk_report: RiskReport | None`
- Wird nur gefüllt, wenn `generate_risk_report=True` in `run_portfolio_backtest()`

**Hinweis:** Um Rückwärtskompatibilität zu erhalten, ist `risk_report` optional und wird standardmäßig auf `None` gesetzt.

### 3. Regime-State-Berechnung

**Falls `regime_state_df` nicht verfügbar:**
- Risk-Report kann optional selbst Regime-State berechnen
- Nutzt `build_regime_state()` aus `risk/regime_models.py`
- Benötigt: `prices_df`, optional `macro_factors_df`, `breadth_df`, `vol_df`

**Funktion:**
```python
def generate_risk_report_with_regime_detection(
    equity_curve: pd.DataFrame,
    prices_df: pd.DataFrame,
    positions_df: pd.DataFrame | None = None,
    trades_df: pd.DataFrame | None = None,
    factor_panel_df: pd.DataFrame | None = None,
    # ... weitere Parameter wie oben
) -> tuple[RiskReport, Path]:
    """
    Wie generate_risk_report(), aber berechnet Regime-State automatisch aus Preisen.
    """
    pass
```

---

## Umsetzungsschritte

### D2.1: Risk-Metrics-Core-Modul

**Aufgaben:**
1. Erstelle `src/assembled_core/risk/risk_metrics_advanced.py`
2. Implementiere `compute_basic_risk_metrics()`:
   - Nutze `qa/metrics.py` für Basis-Metriken (Sharpe, Sortino, etc.)
   - Füge Skewness, Kurtosis, Tail Ratio hinzu
   - Erweitere `qa/risk_metrics.py` um ES (Expected Shortfall)
3. Implementiere `compute_exposure_timeseries()`:
   - Berechne Gross/Net Exposure pro Timestamp
   - Berechne HHI Concentration
   - Optional: Turnover aus Trades
4. Tests: `tests/test_risk_metrics_advanced.py`

**Schätzung:** 2-3 Tage

### D2.2: Risk-by-Regime & Risk-by-Factor-Group

**Aufgaben:**
1. Implementiere `compute_risk_by_regime()`:
   - Merge Equity-Curve mit Regime-State
   - Berechne Metriken pro Regime-Segment
   - Nutze `compute_basic_risk_metrics()` intern
2. Implementiere `compute_risk_by_factor_group()`:
   - Korrelations-Methode: Berechne Korrelation zwischen Faktor-Returns und Portfolio-Returns
   - Exposure-Methode: Optional, falls Positions-Daten verfügbar
3. Tests: Erweitere `tests/test_risk_metrics_advanced.py`

**Schätzung:** 2-3 Tage

### D2.3: Risk-Report-Generierung

**Aufgaben:**
1. Implementiere `RiskReport` Dataclass
2. Implementiere `generate_risk_report()`:
   - Orchestriert alle Berechnungen
   - Generiert Markdown-Report
   - Schreibt CSV-Dateien
3. Erstelle `scripts/generate_risk_report.py`:
   - CLI-Interface für nachträgliche Analyse
   - Lädt DataFrames aus Files
   - Ruft `generate_risk_report()` auf
4. Tests: `tests/test_risk_report.py`

**Schätzung:** 2-3 Tage

### D2.4: CLI-Integration & Workflows

**Aufgaben:**
1. Erweitere `scripts/run_backtest_strategy.py`:
   - Neues Argument: `--with-risk-report`
   - Nach Backtest: Rufe `generate_risk_report()` auf
   - Falls Regime-State nicht verfügbar: Optional automatische Berechnung
2. Erweitere `docs/WORKFLOWS_REGIME_MODELS_AND_RISK.md`:
   - Abschnitt "Risk Reports" hinzufügen
   - Beispiele für Risk-Report-Interpretation
3. Aktualisiere `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md`:
   - Status D2 auf "In Progress" setzen
4. Tests: Integrationstests für CLI-Flags

**Schätzung:** 1-2 Tage

**Gesamt-Schätzung:** 7-11 Tage (ca. 2 Wochen)

---

## Abhängigkeiten

**Voraussetzungen:**
- ✅ D1 (Regime Models): `build_regime_state()`, `regime_state_df` Format
- ✅ `qa/metrics.py`: PerformanceMetrics, Basis-Metriken
- ✅ `qa/risk_metrics.py`: VaR/ES (teilweise vorhanden)
- ✅ `qa/backtest_engine.py`: BacktestResult mit equity, trades, positions

**Keine neuen externen Dependencies:**
- Nutzt nur pandas, numpy (bereits vorhanden)
- Keine neuen APIs oder Datenquellen

---

## Offene Fragen / Future Enhancements

1. **Faktor-Attribution-Methoden:**
   - Aktuell: Einfache Korrelations-/Exposure-Methode
   - Zukünftig: Brinson-Attribution, Performance-Attribution mit Regression

2. **Regime-Detection:**
   - Aktuell: Nutzt Regime-State aus D1
   - Zukünftig: Alternative Regime-Detection-Methoden (ML-basiert)

3. **Visualisierungen:**
   - Aktuell: Nur Tabellen (Markdown/CSV)
   - Zukünftig: Plots (Equity-Curve, Exposure über Zeit, Risk-by-Regime-Bar-Charts)

4. **Benchmark-Vergleich:**
   - Aktuell: Absolute Metriken
   - Zukünftig: Relative Metriken vs. Benchmark (z.B. S&P 500)

---

## Referenzen

- [Regime Models D1 Design](REGIME_MODELS_D1_DESIGN.md) – Regime-Detection-Design
- [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) – Gesamt-Roadmap Phase D
- [Workflows – Regime Models & Risk Overlay](WORKFLOWS_REGIME_MODELS_AND_RISK.md) – Regime-Workflow
- `src/assembled_core/qa/metrics.py` – Basis-Performance-Metriken
- `src/assembled_core/qa/risk_metrics.py` – Basis-Risk-Metriken (VaR/ES)

