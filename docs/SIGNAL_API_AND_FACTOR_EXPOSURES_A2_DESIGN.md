# Signal API & Factor Exposures (A2)

**Phase A2** - Advanced Analytics & Factor Labs

**Status:** ✅ Completed  
**Last Updated:** 2025-01-XX

---

## 1. Overview & Goals

**Ziel:** Einheitliche Signal-API zur Standardisierung von Strategy-Signals und Position-Weightings, sowie Factor-Exposure-Analyse zur Attribution von Strategie-Returns auf Faktor-Returns.

**Signal API:**
Die "Signal API" im Kontext dieses Projekts standardisiert die Reprasentation von Trading-Signals und Position-Weightings. Ein SignalFrame enthalt:
- **Strategy Signals**: Signal-Werte (Scores) pro Symbol und Zeitpunkt
- **Position Weights**: Optionale Target-Weights fur Position-Sizing
- **Metadata**: Strategy-Name, Frequenz, Universe, Point-in-Time-Information (as_of)

Die Signal-API ermoglicht:
- Konsistente Signal-Reprasentation uber verschiedene Strategien hinweg
- Point-in-Time-Safety-Checks (keine Look-Ahead-Violations)
- Normalisierung und Validierung von Signals
- Integration mit Backtest-Engine und Risk-Reports

**Factor-Exposure-Analyse:**
Die Factor-Exposure-Analyse regressiert Strategie-Returns gegen Factor-Returns, um zu verstehen, welche Faktoren die Strategie-Performance treiben. Dies ist eine Form der Performance-Attribution:
- **Rolling Regression**: Betas pro Faktor uber Zeit (z.B. 60-Tage-Rolling-Window)
- **Summary Statistics**: Langfristige Durchschnitts-Betas, Signifikanz, R²
- **Integration in Risk-Reports**: Optionaler Abschnitt "Factor Exposures" mit Summary-Tabelle

**Integration mit bestehenden Phasen:**
- **P1-P4 (Backtest Optimization)**: Signal-API nutzt optimierte Backtest-Engine, Batch-Runner kann SignalFrames verarbeiten
- **B1-B4 (Backtest & Validation)**: Signal-API integriert mit BacktestResult, Deflated Sharpe kann auf Factor-Exposures angewendet werden
- **B2 (Point-in-Time Safety)**: Signal-API nutzt PIT-Checks aus `qa/point_in_time_checks.py` fur Validation
- **E4 (Risk/TCA)**: Factor-Exposures werden in Risk-Reports integriert, TCA kann nach Faktor-Gruppen segmentiert werden

**Datenbasis:** Alle Analysen basieren auf **lokalen Backtest-Outputs** und Factor-Returns. Keine Live-APIs.

---

## 2. Data Contracts

### 2.1 SignalFrame Contract

**Index:** `timestamp` (pd.DatetimeIndex, UTC-aware)

**Spalten:**
- `symbol`: str - Symbol-Name (z.B. "AAPL", "MSFT")
- `signal_value`: float - Signal-Score (normalisiert oder raw, je nach Konfiguration)
- `weight_target`: float (optional) - Target-Weight fur Position-Sizing (0.0 bis 1.0)
- `side`: str (optional) - "LONG", "SHORT", "FLAT" (falls vorhanden)
- `as_of`: pd.Timestamp (optional) - Point-in-Time-Timestamp (wann wurde Signal erzeugt?)
- `source`: str (optional) - Signal-Quelle (z.B. "trend_baseline", "ml_alpha", "multifactor")

**Anforderungen:**
- **Keine Look-Ahead-Violations**: `as_of >= timestamp` fur alle Zeilen (falls `as_of` vorhanden)
- **Universe-Konsistenz**: Alle Symbole mussen im definierten Universe enthalten sein
- **Keine Duplikate**: Keine (timestamp, symbol) Duplikate
- **Sortierung**: Nach `timestamp` aufsteigend sortiert

**Beispiel:**
```
timestamp              symbol  signal_value  weight_target  side   as_of                source
2025-01-15 00:00:00+00:00  AAPL        0.75           0.10  LONG  2025-01-15 00:00:00+00:00  trend_baseline
2025-01-15 00:00:00+00:00  MSFT        0.50           0.05  LONG  2025-01-15 00:00:00+00:00  trend_baseline
2025-01-16 00:00:00+00:00  AAPL        0.80           0.12  LONG  2025-01-16 00:00:00+00:00  trend_baseline
```

**Verweise auf existierende Strukturen:**
- Ahnlich zu `BacktestResult.signals` (timestamp, symbol, direction, score)
- Erweitert um `weight_target` und `as_of` fur PIT-Safety
- Kompatibel mit `execution.order_generation.generate_orders_from_signals()`

### 2.2 Factor Returns Contract

**Index:** `timestamp` (pd.DatetimeIndex, UTC-aware)

**Spalten:** Eine Spalte pro Faktor (z.B. `value`, `quality`, `momentum`, `ml_alpha`, `market`, `size`, `rv_20`, `trend_strength_200`)

**Format:**
- Returns in "per period"-Form (z.B. daily returns fur freq="1d")
- Annualisierung erfolgt via `freq` Parameter in der Regression
- NaN-Werte sind erlaubt (werden in Regression ignoriert)

**Beispiel:**
```
timestamp              value  quality  momentum  ml_alpha  market  size
2025-01-15 00:00:00+00:00  0.001   0.002     -0.001    0.003    0.001  0.000
2025-01-16 00:00:00+00:00  0.002   0.001      0.002     0.001    0.002  0.001
```

**Quellen fur Factor Returns:**
- **Factor Store (P2)**: Geladene Factor-Panels konnen zu Factor-Returns aggregiert werden
- **Factor Analysis (C1/C2)**: Long/Short-Portfolio-Returns aus `summarize_factor_portfolios()`
- **ML Alpha**: ML-Model-Predictions konnen als Factor-Returns verwendet werden
- **Market Factors**: Market-Returns (SPY), Size-Factor (SMB), etc.

**Verweise auf existierende Strukturen:**
- Factor Store: `src/assembled_core/data/factor_store.py`
- Factor Analysis: `src/assembled_core/qa/factor_analysis.py::summarize_factor_portfolios()`

### 2.3 Strategy Returns Contract

**Format:** `pd.Series` oder `pd.DataFrame` mit Strategy-Returns pro Periode

**Index:** `timestamp` (pd.DatetimeIndex, UTC-aware)

**Werte:** Returns in "per period"-Form (z.B. daily returns fur freq="1d")

**Quellen:**
- **BacktestResult.equity**: `daily_return` Spalte aus Equity-Curve
- **Portfolio Simulation**: Returns aus `pipeline.portfolio.simulate_with_costs()`
- **Risk Metrics**: Returns aus `risk/risk_metrics.py::compute_portfolio_risk_metrics()`

**Beispiel:**
```
timestamp
2025-01-15 00:00:00+00:00    0.001
2025-01-16 00:00:00+00:00    0.002
2025-01-17 00:00:00+00:00   -0.001
```

**Verweise auf existierende Strukturen:**
- `BacktestResult.equity["daily_return"]`
- `risk/risk_metrics.py` berechnet Returns aus Equity-Curves

---

## 3. Signal API Design

### 3.1 Dataclass: SignalMetadata

```python
@dataclass
class SignalMetadata:
    """Metadata for a SignalFrame.
    
    Attributes:
        strategy_name: Name of the strategy (e.g., "trend_baseline", "ml_alpha")
        freq: Trading frequency ("1d" or "5min")
        universe_name: Name of the universe (e.g., "ai_tech", "macro_world_etfs")
        as_of: Point-in-time timestamp (when were signals generated?)
        horizon_days: Optional forward-looking horizon for signal evaluation (default: None)
        notes: Optional notes or description
    """
    strategy_name: str
    freq: Literal["1d", "5min"]
    universe_name: str
    as_of: pd.Timestamp
    horizon_days: int | None = None
    notes: str | None = None
```

### 3.2 Funktion: normalize_signals

```python
def normalize_signals(
    df: pd.DataFrame,
    method: Literal["zscore", "rank", "none"] = "zscore",
    clip: float | None = None,
    signal_col: str = "signal_value",
) -> pd.DataFrame:
    """Normalize signal values in a SignalFrame.
    
    Args:
        df: DataFrame with signal values (must contain signal_col)
        method: Normalization method:
            - "zscore": Z-score normalization (mean=0, std=1) per timestamp
            - "rank": Rank normalization (0 to 1) per timestamp
            - "none": No normalization (return as-is)
        clip: Optional clipping value (e.g., 3.0 for zscore = clip to [-3, 3])
        signal_col: Name of signal column (default: "signal_value")
    
    Returns:
        DataFrame with normalized signal values (same structure as input)
    
    Note:
        Normalization is performed per timestamp (cross-sectional) to ensure
        signals are comparable across symbols at each time point.
    """
```

### 3.3 Funktion: make_signal_frame

```python
def make_signal_frame(
    prices: pd.DataFrame,
    raw_scores: pd.DataFrame | pd.Series,
    meta: SignalMetadata,
    normalize: bool = True,
    normalize_method: Literal["zscore", "rank", "none"] = "zscore",
    add_weights: bool = False,
) -> pd.DataFrame:
    """Create a standardized SignalFrame from raw scores.
    
    Args:
        prices: Price DataFrame with columns: timestamp, symbol, close
        raw_scores: Raw signal scores (DataFrame with timestamp, symbol, score OR Series with MultiIndex)
        meta: SignalMetadata with strategy information
        normalize: If True, normalize signals (default: True)
        normalize_method: Normalization method (default: "zscore")
        add_weights: If True, compute target weights from signals (default: False)
    
    Returns:
        SignalFrame DataFrame with columns: timestamp, symbol, signal_value, (optional: weight_target, side, as_of, source)
    
    Raises:
        ValueError: If raw_scores format is invalid or missing required columns
    """
```

### 3.4 Funktion: validate_signal_frame

```python
def validate_signal_frame(
    signal_df: pd.DataFrame,
    as_of: pd.Timestamp | None = None,
    strict: bool = False,
    universe: set[str] | None = None,
) -> None:
    """Validate a SignalFrame for correctness and PIT-safety.
    
    Args:
        signal_df: SignalFrame to validate
        as_of: Optional maximum allowed timestamp (for PIT-check)
        strict: If True, raise ValueError on violations. If False, log warnings.
        universe: Optional set of allowed symbols (for universe consistency check)
    
    Raises:
        ValueError: If strict=True and validation fails:
            - Missing required columns (timestamp, symbol, signal_value)
            - Duplicate (timestamp, symbol) pairs
            - PIT violation (as_of < timestamp)
            - Universe violation (symbol not in universe)
    
    Checks:
        1. Formale Checks: Index-Typ, Spalten, Duplikate
        2. PIT-Checks: as_of >= timestamp (via point_in_time_checks.check_features_pit_safe)
        3. Universe-Checks: Alle Symbole in universe (falls universe angegeben)
    """
```

**Integration mit PIT-Checks:**
- Nutzt `qa/point_in_time_checks.py::check_features_pit_safe()` fur PIT-Validation
- SignalFrame wird als "features DataFrame" behandelt (timestamp, symbol, signal_value)

---

## 4. Factor-Exposure API Design

### 4.1 Dataclass: FactorExposureConfig

```python
@dataclass
class FactorExposureConfig:
    """Configuration for factor exposure analysis.
    
    Attributes:
        freq: Trading frequency ("1d" or "5min") for annualization
        window_size: Rolling window size in periods (default: 60)
        min_obs: Minimum observations required for regression (default: 30)
        mode: Window mode - "rolling" or "expanding" (default: "rolling")
        add_constant: If True, add intercept term to regression (default: True)
        standardize_factors: If True, standardize factor returns before regression (default: False)
        regression_method: Regression method - "ols" or "ridge" (default: "ols")
        shrinkage_lambda: Ridge shrinkage parameter (only used if regression_method="ridge", default: 0.1)
        min_r2_for_report: Minimum R² to include in summary report (default: 0.01)
    """
    freq: Literal["1d", "5min"] = "1d"
    window_size: int = 60
    min_obs: int = 30
    mode: Literal["rolling", "expanding"] = "rolling"
    add_constant: bool = True
    standardize_factors: bool = False
    regression_method: Literal["ols", "ridge"] = "ols"
    shrinkage_lambda: float = 0.1
    min_r2_for_report: float = 0.01
```

### 4.2 Kernfunktion: compute_factor_exposures

```python
def compute_factor_exposures(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    config: FactorExposureConfig,
) -> pd.DataFrame:
    """Compute rolling factor exposures (betas) for strategy returns.
    
    Args:
        strategy_returns: Strategy returns time-series (index = timestamp)
        factor_returns: Factor returns DataFrame (index = timestamp, columns = factor names)
        config: FactorExposureConfig with regression parameters
    
    Returns:
        DataFrame with columns:
        - timestamp: Window end timestamp
        - factor_*: Beta for each factor (one column per factor)
        - intercept: Intercept term (if add_constant=True)
        - r2: R-squared of regression
        - residual_vol: Residual volatility (annualized)
        - n_obs: Number of observations in window
        - t_stat_*: t-statistic for each factor (one column per factor)
    
    Note:
        - Rolling regression: For each timestamp, regress strategy_returns[t-window:t] on factor_returns[t-window:t]
        - Expanding regression: For each timestamp, regress strategy_returns[:t] on factor_returns[:t]
        - Missing values (NaN) in factor_returns are dropped before regression
        - Returns NaN for windows with insufficient observations (< min_obs)
    """
```

**Implementation Details:**
- Nutzt `statsmodels.api.OLS` oder `sklearn.linear_model.Ridge` fur Regression
- Rolling Window: `pd.rolling()` oder manuelle Window-Generierung
- Annualisierung: `residual_vol * sqrt(periods_per_year)` basierend auf `freq`

### 4.3 Funktion: summarize_factor_exposures

```python
def summarize_factor_exposures(
    exposures_df: pd.DataFrame,
    config: FactorExposureConfig,
) -> pd.DataFrame:
    """Summarize factor exposures over time.
    
    Args:
        exposures_df: Output from compute_factor_exposures()
        config: FactorExposureConfig (for min_r2_for_report)
    
    Returns:
        DataFrame with one row per factor:
        - factor: Factor name
        - mean_beta: Mean beta over all windows
        - std_beta: Standard deviation of beta over time
        - mean_t_stat: Mean absolute t-statistic
        - pct_significant: Percentage of windows with |t-stat| > 2.0
        - mean_r2: Mean R² across all windows
        - n_windows: Number of windows with valid regression
        - n_windows_total: Total number of windows
    
    Note:
        - Only includes factors with mean_r2 >= min_r2_for_report
        - Sorted by mean_beta (absolute value, descending)
    """
```

### 4.4 Optional: Static Regression Helper

```python
def compute_static_factor_exposures(
    strategy_returns: pd.Series,
    factor_returns: pd.DataFrame,
    add_constant: bool = True,
    regression_method: Literal["ols", "ridge"] = "ols",
    shrinkage_lambda: float = 0.1,
) -> dict[str, float | None]:
    """Compute static factor exposures over entire period (single regression).
    
    Args:
        strategy_returns: Strategy returns time-series
        factor_returns: Factor returns DataFrame
        add_constant: If True, add intercept (default: True)
        regression_method: "ols" or "ridge" (default: "ols")
        shrinkage_lambda: Ridge shrinkage (default: 0.1)
    
    Returns:
        Dictionary with:
        - betas: Dict mapping factor name to beta
        - intercept: Intercept term (if add_constant=True)
        - r2: R-squared
        - residual_vol: Residual volatility (annualized)
        - n_obs: Number of observations
        - t_stats: Dict mapping factor name to t-statistic
    """
```

---

## 5. Integration Points

### 5.1 Risk-Report Integration

**Erweiterung von `scripts/generate_risk_report.py`:**

- **Optionaler Abschnitt "Factor Exposures"**: 
  - Summary-Tabelle mit mean_beta, mean_t_stat, pct_significant, mean_r2 pro Faktor
  - CSV-Export: `output/risk_reports/<backtest_id>/factor_exposures.csv`
  - Markdown-Report: Neuer Abschnitt im Risk-Report mit Summary-Tabelle

**CLI-Integration:**
```bash
python scripts/cli.py risk_report \
  --backtest-dir output/backtests/experiment_123/ \
  --factor-returns-file output/factor_returns/market_factors.parquet \
  --compute-factor-exposures
```

**Konfiguration:**
- Factor-Returns-File: Parquet-Datei mit Factor-Returns (timestamp, factor_*)
- Config: FactorExposureConfig (Defaults oder via CLI-Parameter)

### 5.2 Research-Playbook Integration

**Erweiterung von `research/playbooks/ai_tech_multifactor_mlalpha_regime_playbook.py`:**

- **Factor-Exposure-Vergleich**: 
  - Compute Factor-Exposures fur Core-only, Core+ML, ML-only Bundles
  - Vergleichstabelle: Mean Betas pro Bundle und Faktor
  - Identifikation: Welche Faktoren treiben welche Bundle-Performance?

**Workflow:**
1. Run Backtests fur alle Bundles (bereits vorhanden)
2. Load Factor-Returns (aus Factor Store oder Factor Analysis)
3. Compute Factor-Exposures fur jedes Bundle
4. Generate Comparison Table (Mean Betas pro Bundle)
5. Add to Research Summary Report

### 5.3 Walk-Forward / Regime Integration (Future Work)

**Nicht in A2 umgesetzt, aber Design vorbereitet:**

- **Regime-spezifische Factor-Exposures**: 
  - Compute Factor-Exposures pro Regime (Bull, Bear, Sideways, etc.)
  - Vergleich: Welche Faktoren sind in welchem Regime dominant?

- **Walk-Forward Factor-Exposures**: 
  - Compute Factor-Exposures pro Walk-Forward-Split
  - Stabilitats-Analyse: Bleiben Factor-Exposures uber Zeit stabil?

**Integration:**
- Nutzt `qa/walk_forward.py::WalkForwardResult` fur Split-Definition
- Nutzt `risk/regime_analysis.py` fur Regime-Klassifikation

---

## 6. Implementation Plan (A2.1-A2.4)

### A2.1: Design & Skeleton ✅ (Completed)

**Status:** Completed

**Tasks:**
- [x] Design-Dokument erstellen
- [x] Neue Module anlegen:
  - `src/assembled_core/signals/signal_api.py` (SignalFrame, normalize_signals, make_signal_frame, validate_signal_frame)
  - `src/assembled_core/risk/factor_exposures.py` (FactorExposureConfig, compute_factor_exposures, summarize_factor_exposures)

**Deliverables:**
- Design-Dokument (dieses Dokument)
- Module-Skeletons mit Docstrings und Type Hints

### A2.2: Signal-API-Implementierung + Tests ✅ (Completed)

**Status:** Completed

**Tasks:**
- [x] Implementiere `SignalMetadata` Dataclass
- [x] Implementiere `normalize_signals()` (zscore, rank, none)
- [x] Implementiere `make_signal_frame()` (von raw scores zu SignalFrame)
- [x] Implementiere `validate_signal_frame()` (formale Checks + PIT-Checks)
- [x] Unit-Tests: `tests/test_signals_signal_api.py`
  - Test normalize_signals (alle Methoden)
  - Test make_signal_frame (verschiedene Input-Formate)
  - Test validate_signal_frame (PIT-Violations, Duplikate, Universe-Checks)

**Deliverables:**
- Funktionsfahige Signal-API
- Unit-Tests (alle grun)
- Module: `src/assembled_core/signals/signal_api.py`
- Tests: `tests/test_signals_signal_api.py`

### A2.3: Factor-Exposure-Implementierung + Tests ✅ (Completed)

**Status:** Completed

**Tasks:**
- [x] Implementiere `FactorExposureConfig` Dataclass
- [x] Implementiere `compute_factor_exposures()` (rolling OLS/Ridge)
- [x] Implementiere `summarize_factor_exposures()` (Summary-Statistiken)
- [x] Unit-Tests: `tests/test_risk_factor_exposures.py`
  - Test compute_factor_exposures (rolling, expanding, OLS, Ridge)
  - Test summarize_factor_exposures (Summary-Statistiken)
  - Test Edge Cases (insufficient observations, NaN handling, multicollinearity)

**Deliverables:**
- Funktionsfahige Factor-Exposure-API
- Unit-Tests (alle grun)
- Module: `src/assembled_core/risk/factor_exposures.py`
- Tests: `tests/test_risk_factor_exposures.py`

### A2.4: Integration in Risk-Report und Research-Workflows + Doku-Update ✅ (Completed)

**Status:** Completed

**Tasks:**
- [x] Erweitere `scripts/generate_risk_report.py`:
  - Optionaler Parameter `--enable-factor-exposures`, `--factor-returns-file`, `--factor-exposures-window`
  - Load Factor-Returns (aus CSV/Parquet File)
  - Compute Factor-Exposures
  - Add Summary-Tabelle zu Markdown-Report
  - Export CSV: `factor_exposures_detail.csv`, `factor_exposures_summary.csv`
- [x] CLI-Integration in `scripts/cli.py`:
  - Neue Argumente im `risk_report` Subcommand
  - Argumente werden an `generate_risk_report()` weitergegeben
- [x] Integration-Tests: `tests/test_cli_risk_report_factor_exposures.py`
  - Test dass CSV-Dateien erzeugt werden
  - Test dass Markdown-Report erweitert wird
  - Test Backward Compatibility (ohne Flag werden keine Dateien erzeugt)
- [x] Update Dokumentation:
  - `docs/WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md`: Factor-Exposure-Abschnitt
  - `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md`: A2 als Completed markieren
  - `README.md`: Hinweis auf Factor-Exposures

**Deliverables:**
- Risk-Report mit Factor-Exposures (optional)
- CLI-Integration
- Aktualisierte Dokumentation
- Tests: `tests/test_cli_risk_report_factor_exposures.py`

**Future Work (nicht in A2):**
- Regime-spezifische Factor-Exposures
- Walk-Forward Factor-Exposures
- TCA nach Faktor-Gruppen segmentiert

---

## 7. Implementation Status (A2.1-A2.4)

**Status:** ✅ All tasks completed (A2.1-A2.4)

### Implemented Modules

**Signal API:**
- `src/assembled_core/signals/signal_api.py`
  - `SignalMetadata` dataclass
  - `normalize_signals()` function (zscore, rank, none methods)
  - `make_signal_frame()` function
  - `validate_signal_frame()` function (with PIT-checks)

**Factor Exposures:**
- `src/assembled_core/risk/factor_exposures.py`
  - `FactorExposureConfig` dataclass
  - `compute_factor_exposures()` function (rolling/expanding OLS/Ridge regression)
  - `summarize_factor_exposures()` function (aggregation over time)

**CLI Integration:**
- `scripts/generate_risk_report.py` extended with:
  - `--enable-factor-exposures` flag
  - `--factor-returns-file PATH` argument
  - `--factor-exposures-window INT` argument (default: 252)
- `scripts/cli.py` extended with new arguments for `risk_report` subcommand

### Test Files

- `tests/test_signals_signal_api.py` (10 tests, all passing)
- `tests/test_risk_factor_exposures.py` (9 tests, all passing)
- `tests/test_cli_risk_report_factor_exposures.py` (3 tests, all passing)

### Output Files

When `--enable-factor-exposures` is set:
- `factor_exposures_detail.csv`: Time-series of factor betas, intercept, R², residual_vol, n_obs
- `factor_exposures_summary.csv`: Aggregated statistics (mean_beta, std_beta, mean_r2, etc.) per factor
- `risk_report.md`: Extended with "Factor Exposures" section containing top factors table and summary

---

## 7. Risks & Limitations

### 7.1 Performance

**Problem:** Rolling Regression uber viele Zeitpunkte kann langsam sein.

**Mitigation:**
- Nutze vektorisierte Pandas-Operationen wo moglich
- Optionale Numba-Beschleunigung fur kritische Loops (Future Work)
- Caching von Factor-Returns (via Factor Store P2)

### 7.2 Multicollinearity

**Problem:** Faktoren konnen hoch korreliert sein (z.B. value und quality), was zu instabilen Betas fuhrt.

**Mitigation:**
- Ridge-Regression als Alternative zu OLS (reduziert Overfitting)
- Standardisierung von Faktoren (standardize_factors=True) kann helfen
- Warnung in Summary-Report bei hoher Faktor-Korrelation

### 7.3 Korrelierte Factors

**Problem:** Wenn Faktoren korreliert sind, ist Attribution mehrdeutig (welcher Faktor treibt Performance?).

**Mitigation:**
- Dokumentiere Korrelationen in Summary-Report
- Empfehlung: Nutze orthogonalisierte Faktoren (Future Work)

### 7.4 Overfitting

**Problem:** Rolling Regression mit vielen Faktoren kann zu Overfitting fuhren (hohe R², aber instabile Betas).

**Mitigation:**
- Ridge-Regression mit Shrinkage (reduziert Overfitting)
- Minimum R² Threshold (min_r2_for_report) filtert schwache Fits
- Cross-Validation uber Walk-Forward-Splits (Future Work)

### 7.5 PIT-Konformitat

**Problem:** Factor-Returns selbst mussen PIT-sicher sein (keine Look-Ahead-Violations).

**Mitigation:**
- Factor-Returns mussen aus PIT-sicheren Quellen stammen (Factor Store, Factor Analysis)
- Validation: `validate_signal_frame()` kann auch auf Factor-Returns angewendet werden (Future Work)
- Dokumentation: Klare Anforderungen an Factor-Returns-Quellen

---

## 8. Success Criteria

### 8.1 Einheitlicher Signal-Contract

**Kriterium:** SignalFrame ist der Standard fur alle Strategy-Signals.

**Messung:**
- Alle neuen Strategien nutzen `make_signal_frame()` zur Signal-Erzeugung
- Bestehende Strategien konnen optional migriert werden (nicht zwingend in A2)

**Tests:**
- Unit-Tests fur SignalFrame-Erzeugung und Validation
- Integration-Test: SignalFrame -> Backtest-Engine funktioniert

### 8.2 Factor-Exposure-API funktioniert

**Kriterium:** Factor-Exposure-API kann fur mindestens eine beispielhafte Strategie verwendet werden.

**Messung:**
- `compute_factor_exposures()` funktioniert mit BacktestResult.equity["daily_return"]
- `summarize_factor_exposures()` erzeugt sinnvolle Summary-Statistiken
- Integration in Risk-Report funktioniert (optionaler Abschnitt)

**Tests:**
- Unit-Tests fur Factor-Exposure-Berechnung
- Integration-Test: BacktestResult -> Factor-Exposures -> Summary
- Smoke-Test: Risk-Report mit Factor-Exposures generiert

### 8.3 Unit-Tests und Integrationstests grun

**Kriterium:** Alle Tests sind grun.

**Messung:**
- `pytest tests/test_signals_signal_api.py` - alle Tests grun
- `pytest tests/test_qa_factor_exposures.py` - alle Tests grun
- Integration-Tests (SignalFrame -> Backtest, Strategy Returns -> Exposures) - alle grun

**Tests:**
- Mindestens 80% Code-Coverage fur neue Module
- Edge Cases abgedeckt (NaN handling, insufficient observations, PIT violations)

---

## 9. References

**Design Documents:**
- [Backtest B1 Design](BACKTEST_B1_UNIFIED_PIPELINE_DESIGN.md) - Backtest-Engine-Integration
- [Point-in-Time B2 Design](POINT_IN_TIME_AND_LATENCY_B2_DESIGN.md) - PIT-Safety
- [Risk 2.0 D2 Design](RISK_2_0_D2_DESIGN.md) - Risk-Report-Integration
- [Factor Store P2 Design](FACTOR_STORE_P2_DESIGN.md) - Factor-Returns-Quellen

**Code References:**
- `src/assembled_core/qa/backtest_engine.py` - BacktestResult, Signal-Format
- `src/assembled_core/qa/point_in_time_checks.py` - PIT-Validation
- `src/assembled_core/qa/factor_analysis.py` - Factor-Returns aus Portfolio-Summaries
- `src/assembled_core/data/factor_store.py` - Factor-Returns aus Factor Store
- `scripts/generate_risk_report.py` - Risk-Report-Generierung

**Workflows:**
- [Risk Metrics & Attribution Workflows](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md) - Risk-Report-Workflow
- [Research Playbooks](ADVANCED_ANALYTICS_FACTOR_LABS.md) - Research-Playbook-Integration

