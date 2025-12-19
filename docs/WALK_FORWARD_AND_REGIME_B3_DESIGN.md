# Walk-Forward Analysis & Regime-Based Evaluation (B3)

**Phase B3** – Advanced Analytics & Factor Labs

**Status:** ✅ Completed  
**Last Updated:** 2025-01-XX

---

## 1. Overview & Goals

**Ziel:** Systematische Out-of-Sample-Validierung von Strategien durch Walk-Forward-Analyse und Evaluation der Strategie-Performance in verschiedenen Marktregimen.

**Kernfunktionen:**

1. **Walk-Forward-Analyse:**
   - Zeitfenster-basierte Train/Test-Splits (Rolling oder Expanding Window)
   - Out-of-Sample-Validierung für jede Split-Periode
   - Aggregation der Ergebnisse über alle Splits
   - Identifikation von Overfitting und zeitlicher Stabilität

2. **Regime-basierte Analyse:**
   - Performance-Evaluation nach Marktregimen (Bull, Bear, Sideways, Crisis, Reflation)
   - Identifikation regime-spezifischer Stärken und Schwächen
   - Regime-Transition-Analyse (Wie verhält sich die Strategie bei Regime-Wechseln?)
   - Faktor-Performance nach Regime

**Bezug zu bestehenden Modulen:**

- Baut auf `qa/backtest_engine.py` auf (BacktestResult, run_portfolio_backtest)
- Nutzt `risk/regime_models.py` (D1) für Regime-Detection
- Erweitert `risk/regime_analysis.py` (B3) für erweiterte Regime-Analyse
- Reuse von Zeitreihen-CV-Logik aus `ml/factor_models.py` (_split_time_series Pattern)
- Integration mit `qa/risk_metrics.py` und `scripts/generate_risk_report.py`

**Datenbasis:** Alle Analysen basieren auf **lokalen Backtest-Outputs** und Preis-Daten. Keine Live-APIs.

---

## 2. Data Contracts & Inputs

### 2.1 Walk-Forward-Analyse

**Inputs:**

- **Preis-Panel**: DataFrame mit Spalten: `timestamp`, `symbol`, `close` (und optional OHLCV)
- **Strategie-Konfiguration**: 
  - Signal-Funktion (callable)
  - Position-Sizing-Funktion (callable)
  - Weitere Parameter (start_capital, costs, rebalancing_freq, etc.)
- **Walk-Forward-Config**: 
  - Train/Test-Fenster-Größen
  - Rolling vs. Expanding Window
  - Step-Size (wie weit rückt das Fenster vor?)
  - Minimum Train-Periods

**Outputs:**

- **WalkForwardResult**: 
  - Liste von Split-Ergebnissen (eine pro Train/Test-Periode)
  - Aggregierte Metriken (Mittelwert, Std, Min/Max über alle Splits)
  - Equity-Kurven pro Split (optional)
  - Summary-Report (Markdown/CSV)

### 2.2 Regime-Analyse

**Inputs:**

- **Equity-Kurve**: DataFrame mit Spalten: `timestamp`, `equity`, `daily_return`
- **Regime-State**: DataFrame mit Spalten: `timestamp`, `regime_label` (aus `regime_models.py` oder `regime_analysis.py`)
- **Optional:**
  - Trades DataFrame (für Trade-Level-Metriken)
  - Factor-Panel (für Faktor-Performance nach Regime)
  - Benchmark-Returns (für Relative-Performance)

**Outputs:**

- **Regime-Performance-Report**: 
  - Metriken pro Regime (Sharpe, Sortino, Max DD, Win Rate, etc.)
  - Anzahl Perioden pro Regime
  - Relative Performance vs. Benchmark (optional)
  - Faktor-IC nach Regime (optional)

---

## 3. Walk-Forward-API (geplant)

### 3.1 Dataclasses

```python
@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward backtest analysis.
    
    Attributes:
        train_size_days: Training window size in days (None = expanding window)
        test_size_days: Test window size in days
        step_size_days: Step size for rolling window (default: test_size_days)
        min_train_periods: Minimum number of periods required for training (default: 252)
        min_test_periods: Minimum number of periods required for testing (default: 63)
        overlap_allowed: Whether test windows can overlap (default: False)
        include_costs: Whether to include transaction costs in simulation (default: True)
        start_capital: Starting capital for each split (default: 10000.0)
    """
    train_size_days: int | None  # None = expanding window
    test_size_days: int
    step_size_days: int | None = None  # Default: test_size_days
    min_train_periods: int = 252  # ~1 year for daily data
    min_test_periods: int = 63    # ~3 months for daily data
    overlap_allowed: bool = False
    include_costs: bool = True
    start_capital: float = 10000.0


@dataclass
class WalkForwardWindow:
    """Single walk-forward window (train/test split).
    
    Attributes:
        train_start: Start date of training period
        train_end: End date of training period (exclusive)
        test_start: Start date of test period
        test_end: End date of test period (exclusive)
        split_index: Index of this split (0-based)
    """
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    split_index: int


@dataclass
class WalkForwardWindowResult:
    """Results for a single walk-forward window.
    
    Attributes:
        window: WalkForwardWindow configuration
        backtest_result: BacktestResult from test period
        train_periods: Number of periods in training window
        test_periods: Number of periods in test window
        status: "success" or "failed"
        error_message: Error message if status == "failed"
    """
    window: WalkForwardWindow
    backtest_result: BacktestResult | None
    train_periods: int
    test_periods: int
    status: Literal["success", "failed"] = "success"
    error_message: str | None = None


@dataclass
class WalkForwardResult:
    """Aggregated results from walk-forward analysis.
    
    Attributes:
        config: WalkForwardConfig used for analysis
        window_results: List of WalkForwardWindowResult (one per split)
        aggregated_metrics: Dictionary with aggregated metrics across all splits:
            - mean_sharpe, std_sharpe, min_sharpe, max_sharpe
            - mean_return, std_return
            - mean_max_dd, std_max_dd
            - n_splits, n_successful_splits, n_failed_splits
        summary_df: DataFrame with one row per split (summary of metrics)
    """
    config: WalkForwardConfig
    window_results: list[WalkForwardWindowResult]
    aggregated_metrics: dict[str, float]
    summary_df: pd.DataFrame
```

### 3.2 Core Functions

```python
def generate_walk_forward_splits(
    timestamps: pd.Series,
    config: WalkForwardConfig,
) -> list[WalkForwardWindow]:
    """Generate walk-forward train/test splits from timestamps.
    
    Args:
        timestamps: Series of timestamps (must be sorted, UTC-aware)
        config: WalkForwardConfig
    
    Returns:
        List of WalkForwardWindow objects (one per split)
    
    Raises:
        ValueError: If insufficient data for splits or invalid config
    """


def run_walk_forward_backtest(
    prices: pd.DataFrame,
    signal_fn: Callable,
    position_sizing_fn: Callable,
    config: WalkForwardConfig,
    timestamp_col: str = "timestamp",
    group_col: str = "symbol",
    price_col: str = "close",
) -> WalkForwardResult:
    """Run walk-forward backtest analysis.
    
    For each train/test split:
    1. Filter prices to train period (for signal/position sizing training, if needed)
    2. Filter prices to test period
    3. Run backtest on test period
    4. Collect metrics
    
    Args:
        prices: Price panel DataFrame
        signal_fn: Signal generation function (callable)
        position_sizing_fn: Position sizing function (callable)
        config: WalkForwardConfig
        timestamp_col: Name of timestamp column (default: "timestamp")
        group_col: Name of symbol column (default: "symbol")
        price_col: Name of price column (default: "close")
    
    Returns:
        WalkForwardResult with all split results and aggregated metrics
    
    Note:
        The signal_fn and position_sizing_fn are called per split.
        If these functions need training data, they should accept a train_prices
        parameter or be wrapped in a factory that creates trained functions.
    """
```

---

## 4. Regime-Modell-Design

### 4.1 Basis-Regime

**Regime-Kategorien (aus `regime_models.py` und `regime_analysis.py`):**

- **Bull**: Positive Trend (Preis > MA) AND moderate/low volatility
- **Bear**: Negative Trend (Preis < MA) AND elevated volatility
- **Sideways**: Moderate trend AND moderate volatility
- **Crisis**: Deep drawdown (< -20%) OR very high volatility (> 30% annualized)
- **Reflation**: Positive trend after crisis (Recovery phase)
- **Neutral**: Ambiguous signals (default classification)

### 4.2 Regime-Klassifikation

**Basis-Kennzahlen (aus `regime_analysis.py`):**

- **Realized Volatility**: Rolling window (default: 20 Tage)
- **Trend**: Price vs. Moving Average (default: 200-Tage MA)
- **Drawdown**: Running maximum drawdown from recent peak
- **Volatility of Volatility (VoV)**: Optional, für regime refinement

**Klassifikationsregeln:**

1. **Crisis**: Drawdown < threshold (-20%) OR volatility > high_threshold (30%)
2. **Bear**: Trend < 0 (price < MA) AND volatility > low_threshold (15%)
3. **Bull**: Trend > 0 (price > MA) AND volatility < high_threshold (30%)
4. **Sideways**: Moderate trend AND moderate volatility
5. **Reflation**: Positive trend AND recovering from crisis (drawdown improving)
6. **Neutral**: Default (ambiguous)

### 4.3 API-Design (in `regime_analysis.py`)

**Bereits vorhanden (Skeleton):**

- `classify_regimes_from_index(index_returns, config)` -> pd.Series
- `summarize_metrics_by_regime(equity, regimes, trades, factor_panel)` -> pd.DataFrame
- `compute_regime_transitions(regime_state)` -> pd.DataFrame

**Ergänzungen:**

- Integration mit Walk-Forward-Ergebnissen
- Regime-Performance-Vergleich zwischen Splits
- Faktor-IC nach Regime (bereits in `regime_models.py::evaluate_factor_ic_by_regime`)

---

## 5. Integration mit bestehenden Reports

### 5.1 Walk-Forward in Risk-Reports

**Erweiterung von `scripts/generate_risk_report.py`:**

- Optional: Walk-Forward-Summary-Tabelle
  - Zeile pro Split: Split-Index, Train-Period, Test-Period, Sharpe, Return, Max DD
  - Aggregierte Metriken: Mean/Std/Min/Max über alle Splits
- Optional: Walk-Forward-Equity-Kurven-Visualisierung (wenn matplotlib verfügbar)

### 5.2 Regime-Performance in Risk-Reports

**Bereits vorhanden:**

- `compute_risk_by_regime()` in `risk_metrics.py` (D2)
- Integration in `generate_risk_report.py` (optional `--regime-file`)

**Erweiterungen:**

- Walk-Forward + Regime: Performance pro Regime für jeden Split
- Regime-Transition-Performance: Wie verhält sich die Strategie bei Regime-Wechseln?
- Faktor-IC nach Regime: Welche Faktoren funktionieren in welchen Regimen?

---

## 6. Implementation Plan B3.1–B3.3

### B3.1: Walk-Forward-Core-Modul

**Tasks:**

1. Implementiere `WalkForwardConfig`, `WalkForwardWindow`, `WalkForwardWindowResult`, `WalkForwardResult` dataclasses
2. Implementiere `generate_walk_forward_splits()`:
   - Rolling Window: Fixed-size training window
   - Expanding Window: Growing training window
   - Step-Size-Handling (overlap prevention)
   - Validation (min_train_periods, min_test_periods)
3. Implementiere `run_walk_forward_backtest()`:
   - Iteration über Splits
   - Preis-Filterung pro Split
   - Backtest-Aufruf pro Split (via `run_portfolio_backtest`)
   - Metriken-Aggregation
   - Error-Handling (failed splits)

**Module:** `src/assembled_core/qa/walk_forward.py`

**Tests:** `tests/test_qa_walk_forward.py`

### B3.2: Regime-Analyse-Modul (Erweiterung)

**Tasks:**

1. Implementiere `classify_regimes_from_index()` (Skeleton vorhanden)
2. Implementiere `summarize_metrics_by_regime()` (Skeleton vorhanden)
   - Erweitere um Trade-Level-Metriken
   - Erweitere um Faktor-Performance (IC nach Regime)
3. Implementiere `compute_regime_transitions()` (Skeleton vorhanden)
4. Integration mit Walk-Forward: Regime-Performance pro Split

**Module:** `src/assembled_core/risk/regime_analysis.py` (bereits vorhanden, TODOs entfernen)

**Tests:** `tests/test_risk_regime_analysis.py`

### B3.3: Integration in Reports + Tests + Doku

**Tasks:**

1. CLI-Subcommand: `python scripts/cli.py walk_forward_backtest ...`
2. Erweiterung von `generate_risk_report.py`: Walk-Forward-Summary (optional)
3. Workflow-Dokumentation: `docs/WORKFLOWS_WALK_FORWARD_AND_REGIME.md`
4. Integration-Tests: Walk-Forward + Regime-Analyse kombiniert

---

## 7. Risks & Limitations

### 7.1 Walk-Forward

**Performance:**
- Lange Laufzeiten bei vielen Splits (z.B. tägliches Rolling Window über 10 Jahre)
- Lösung: Optionale Parallelisierung (nutze P4 Batch-Runner als Basis)

**Data-Snooping:**
- Walk-Forward verringert, aber eliminiert nicht komplett Data-Snooping
- Lösung: Dokumentation von Best Practices (keine Parameter-Optimierung auf Test-Daten)

**Komplexe Konfiguration:**
- Viele Parameter (train_size, test_size, step_size, etc.)
- Lösung: Sensible Defaults, umfassende Dokumentation

### 7.2 Regime-Analyse

**Regime-Definition:**
- Vereinfachte Regeln können Marktphasen falsch klassifizieren
- Lösung: Konfigurierbare Thresholds, Option für externe Regime-Labels

**Data-Quality:**
- Regime-Klassifikation abhängig von Index-Daten-Qualität
- Lösung: Robustes Error-Handling, Fallback auf einfache Regime

**Overlapping Regimes:**
- Einige Perioden können mehrere Regime-Kriterien erfüllen
- Lösung: Prioritäts-Regeln (z.B. Crisis > Bear > Bull > Sideways)

---

## 8. References

- **Existing Modules:**
  - `src/assembled_core/qa/backtest_engine.py`: Core backtest engine
  - `src/assembled_core/risk/regime_models.py`: Basic regime detection (D1)
  - `src/assembled_core/risk/regime_analysis.py`: Extended regime analysis (B3 skeleton)
  - `src/assembled_core/ml/factor_models.py`: Time-series CV pattern (_split_time_series)
  - `src/assembled_core/qa/risk_metrics.py`: Risk metrics computation (D2)
  - `scripts/generate_risk_report.py`: Risk report generation

- **Design Documents:**
  - [Backtest B1 Unified Pipeline Design](BACKTEST_B1_UNIFIED_PIPELINE_DESIGN.md)
  - [Risk 2.0 & Attribution D2 Design](RISK_2_0_D2_DESIGN.md)
  - [Advanced Analytics & Factor Labs Roadmap](ADVANCED_ANALYTICS_FACTOR_LABS.md)

---

## 9. Success Criteria

- **Walk-Forward:**
  - ✅ Kann 5-10 Splits über 5-10 Jahre Daten generieren
  - ✅ Aggregierte Metriken sind konsistent mit Einzel-Backtests
  - ✅ Failed Splits werden graceful gehandhabt (Logging, Weiterlaufen)

- **Regime-Analyse:**
  - ✅ Klassifikation funktioniert auf Standard-Index-Returns (SPY, QQQ, etc.)
  - ✅ Performance-Metriken pro Regime sind konsistent mit `compute_risk_by_regime()`
  - ⏳ Regime-Transitions können analysiert werden (TODO: `compute_regime_transitions`)

- **Integration:**
  - ⏳ Walk-Forward + Regime-Analyse können kombiniert werden (B3.3)
  - ⏳ Reports enthalten Walk-Forward- und Regime-Summaries (B3.3)
  - ⏳ CLI-Tools sind benutzerfreundlich (B3.3)

---

## 10. Implementation Status (B3.1-B3.2)

### B3.1: Walk-Forward-Core ✅ (Completed)

**Implemented:**
- `generate_walk_forward_splits()`: Generates train/test splits for expanding and rolling windows
- `run_walk_forward_backtest()`: Executes walk-forward analysis with configurable `backtest_fn`
- `make_engine_backtest_fn()`: Helper to create `backtest_fn` from portfolio engine
- Comprehensive validation and error handling

**Testing:**
- 7 tests in `tests/test_qa_walk_forward.py` (all passing)
- Tests cover: expanding/rolling splits, metrics aggregation, error handling, max_splits

**Usage:**
```python
from src.assembled_core.qa.walk_forward import (
    WalkForwardConfig,
    run_walk_forward_backtest,
    make_engine_backtest_fn,
)

config = WalkForwardConfig(
    start_date=pd.Timestamp("2020-01-01", tz="UTC"),
    end_date=pd.Timestamp("2023-12-31", tz="UTC"),
    train_window_days=252,
    test_window_days=63,
    mode="rolling",
    step_size_days=63,
)

backtest_fn = make_engine_backtest_fn(
    prices=prices_df,
    signal_fn=signal_fn,
    position_sizing_fn=position_sizing_fn,
)

result = run_walk_forward_backtest(config=config, backtest_fn=backtest_fn)
```

### B3.2: Regime-Analyse ✅ (Completed)

**Implemented:**
- `classify_regimes_from_index()`: Classifies regimes from index returns using rules-based approach
  - Computes rolling volatility, trend (price vs MA), and drawdown
  - Applies priority-based classification: Crisis > Bear > Bull > Sideways > Reflation > Neutral
  - Handles warm-up periods gracefully
- `summarize_metrics_by_regime()`: Aggregates performance metrics by regime
  - Basic metrics: Sharpe, Sortino, Volatility, Max Drawdown, CAGR, Total Return
  - Trade-level metrics: n_trades (win_rate, avg_trade_duration, avg_profit_per_trade are TODO)
  - Factor IC metrics: factor_ic_mean (TODO)
- `summarize_factor_ic_by_regime()`: Helper for IC analysis by regime
  - Computes mean, std, count, IR per regime
  - Can be used with ML validation or factor ranking results

**Testing:**
- 9 tests in `tests/test_risk_regime_analysis.py` (all passing)
- Tests cover: regime classification with synthetic phases, metrics summarization, IC analysis

**Usage:**
```python
from src.assembled_core.risk.regime_analysis import (
    RegimeConfig,
    classify_regimes_from_index,
    summarize_metrics_by_regime,
    summarize_factor_ic_by_regime,
)

# Classify regimes
config = RegimeConfig(
    vol_window=20,
    trend_ma_window=200,
    drawdown_threshold=-0.20,
    vol_threshold_high=0.30,
    vol_threshold_low=0.15,
)

regimes = classify_regimes_from_index(index_returns, config)

# Summarize metrics
metrics_df = summarize_metrics_by_regime(
    equity=equity_series,
    regimes=regimes,
    trades=trades_df,  # Optional
    freq="1d",
)
```

**Implementation Notes:**
- Regime classification uses iterative approach (not fully vectorized) because Reflation requires sequential logic (checking for recent crisis)
- Classification thresholds are configurable via `RegimeConfig`
- Metrics summarization reuses `compute_basic_risk_metrics()` from `risk_metrics.py`
- CAGR and Sortino use `freq` parameter (not `periods_per_year`) to match existing API

**Known Limitations / TODOs:**
- `compute_regime_transitions()` is still a stub (returns empty DataFrame)
- Trade-level metrics (win_rate, avg_trade_duration, avg_profit_per_trade) require position tracking (TODO)
- Factor IC by regime requires factor_panel integration (TODO)

### B3.3: Integration ✅ (Completed)

**Implemented:**
- CLI-Subcommand: `walk_forward` in `scripts/cli.py`
- Walk-Forward-Runner: `scripts/run_walk_forward_analysis.py` (research tool)
- Risk-Report-Erweiterung: Regime-Analyse mit Benchmark/Index in `scripts/generate_risk_report.py`
  - `--benchmark-symbol`, `--benchmark-file`, `--enable-regime-analysis` Flags
  - Erweiterte Regime-Metriken in `risk_report.md` ("Performance by Regime" Sektion)
  - Automatische Regime-Klassifikation aus Benchmark-Returns
- Integration-Tests: `tests/test_cli_walk_forward_analysis.py`, `tests/test_cli_risk_report_regime.py`

**Usage:**
```bash
# Walk-Forward Analysis
python scripts/cli.py walk_forward \
  --freq 1d \
  --strategy trend_baseline \
  --universe config/watchlist.txt \
  --start-date 2020-01-01 \
  --end-date 2023-12-31 \
  --train-window 252 \
  --test-window 63 \
  --mode rolling

# Risk Report with Regime Analysis
python scripts/cli.py risk_report \
  --backtest-dir output/backtests/experiment_123/ \
  --benchmark-symbol SPY \
  --enable-regime-analysis
```
