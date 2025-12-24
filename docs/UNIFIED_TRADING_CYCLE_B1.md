# Unified Trading Cycle (B1) - Design & Plan

**Last Updated:** 2025-01-XX  
**Status:** Design Phase  
**Related:** [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md), [Backend Architecture](ARCHITECTURE_BACKEND.md)

---

## 1. Ziel

**Ziel:** Eliminierung von Code-Duplikation durch eine einheitliche Orchestrator-Schnittstelle (`TradingContext` + `run_trading_cycle(ctx)`) für die gemeinsamen Steps:

1. **Prices Loading** (Daten-Ingest)
2. **Features Building** (TA-Features, Factor-Store-Integration)
3. **Signals Generation** (Trend, Event, Multi-Factor)
4. **Position Sizing** (Target-Positions-Berechnung)
5. **Order Generation** (Orders aus Targets)
6. **Risk Controls** (Pre-Trade-Checks, Kill-Switch)
7. **Outputs** (SAFE-CSV, Equity-Curves, Reports)

**Nicht-Ziel (Non-Goals):**

- Keine Änderung der finanziellen Logik (Signals, Sizing, Costs bleiben unverändert)
- Keine Konsolidierung der unterschiedlichen Output-Formate (EOD: SAFE-CSV, Backtest: Equity-Curves, Paper-Track: State-Persistence)
- Keine Vereinheitlichung der State-Management-Strategien (EOD: stateless, Backtest: temporal equity tracking, Paper-Track: persistent state)

---

## 2. Gefundene Duplikationen

### 2.1 Prices Loading

**Duplikation:**
- `scripts/run_daily.py`: `load_eod_prices()`, `filter_prices_for_date()`, `validate_universe_vs_data()`
- `scripts/run_backtest_strategy.py`: `load_eod_prices()`, `load_eod_prices_for_universe()`
- `src/assembled_core/qa/backtest_engine.py`: Preise werden in `run_portfolio_backtest()` als Parameter übergeben (keine direkte Duplikation, aber ähnliche Filter-Logik)
- `src/assembled_core/paper/paper_track.py`: `load_eod_prices_for_universe()`, `_filter_prices_for_date()`

**Gemeinsamkeiten:**
- Universe-Validierung (Symbole vs. verfügbare Daten)
- Date-Filtering (last_available, exact)
- Timezone-Handling (UTC)

### 2.2 Features Building

**Duplikation:**
- `scripts/run_daily.py`: `add_all_features()` oder `build_or_load_factors()` (mit Factor-Store-Integration)
- `scripts/run_backtest_strategy.py`: Features werden indirekt in Signal-Funktionen erstellt (z.B. `create_event_insider_shipping_signal_fn()` ruft `add_insider_features()` auf)
- `src/assembled_core/qa/backtest_engine.py`: `add_all_features()` in `run_portfolio_backtest()` (optional, wenn `compute_features=True`)
- `src/assembled_core/paper/paper_track.py`: `add_all_features()` in `_run_paper_track_day()`

**Gemeinsamkeiten:**
- TA-Features (`add_all_features()` mit `ma_windows`, `atr_window`, `rsi_window`)
- Factor-Store-Integration (optional, via `build_or_load_factors()`)
- Feature-Building-Parameter (ma_fast, ma_slow, etc.)

### 2.3 Signals Generation

**Duplikation:**
- `scripts/run_daily.py`: `generate_trend_signals_from_prices()` direkt aufgerufen
- `scripts/run_backtest_strategy.py`: Signal-Funktionen werden als Callables erstellt (`create_trend_baseline_signal_fn()`, `create_event_insider_shipping_signal_fn()`, `create_multifactor_long_short_signal_fn()`), dann in `run_portfolio_backtest()` aufgerufen
- `src/assembled_core/qa/backtest_engine.py`: Signal-Funktion als Parameter (`signal_fn: Callable`) wird pro Timestamp aufgerufen
- `src/assembled_core/paper/paper_track.py`: `generate_signals_and_targets_for_day()` (Wrapper über Strategy-Adapters)

**Gemeinsamkeiten:**
- Signal-Funktionen nehmen `prices_df: pd.DataFrame` und geben `signals_df: pd.DataFrame` zurück
- Signal-Format: `timestamp`, `symbol`, `direction` (LONG/SHORT/FLAT), `score`

### 2.4 Position Sizing

**Duplikation:**
- `scripts/run_daily.py`: `compute_target_positions_from_trend_signals()` direkt aufgerufen
- `scripts/run_backtest_strategy.py`: Position-Sizing-Funktionen als Callables (`create_position_sizing_fn()`, `create_event_position_sizing_fn()`, `create_multifactor_long_short_position_sizing_fn()`), dann in `run_portfolio_backtest()` aufgerufen
- `src/assembled_core/qa/backtest_engine.py`: Position-Sizing-Funktion als Parameter (`position_sizing_fn: Callable`) wird pro Timestamp aufgerufen
- `src/assembled_core/paper/paper_track.py`: `generate_signals_and_targets_for_day()` (integriert in Strategy-Adapters)

**Gemeinsamkeiten:**
- Position-Sizing-Funktionen nehmen `signals_df: pd.DataFrame`, `capital: float` und geben `target_positions_df: pd.DataFrame` zurück
- Target-Position-Format: `symbol`, `target_weight`, `target_qty`

### 2.5 Order Generation

**Duplikation:**
- `scripts/run_daily.py`: `generate_orders_from_signals()` direkt aufgerufen
- `scripts/run_backtest_strategy.py`: Orders werden in `run_portfolio_backtest()` via `generate_orders_from_targets()` erstellt
- `src/assembled_core/qa/backtest_engine.py`: `generate_orders_from_targets()` in `_process_rebalancing_timestamp()`
- `src/assembled_core/paper/paper_track.py`: `generate_orders_from_targets()` in `_run_paper_track_day()`

**Gemeinsamkeiten:**
- Order-Generierung aus Targets (current_positions vs. target_positions)
- Order-Format: `timestamp`, `symbol`, `side` (BUY/SELL), `qty`, `price`

### 2.6 Risk Controls

**Duplikation:**
- `scripts/run_daily.py`: `filter_orders_with_risk_controls()` (Pre-Trade-Checks, Kill-Switch)
- `scripts/run_backtest_strategy.py`: Keine explizite Risk-Control-Integration (kann später hinzugefügt werden)
- `src/assembled_core/qa/backtest_engine.py`: Keine explizite Risk-Control-Integration
- `src/assembled_core/paper/paper_track.py`: Keine explizite Risk-Control-Integration (kann später hinzugefügt werden)

**Gemeinsamkeiten:**
- Pre-Trade-Checks (max position size, max gross exposure, etc.)
- Kill-Switch (global kill switch file check)

### 2.7 Outputs

**Duplikation:**
- `scripts/run_daily.py`: `write_safe_orders_csv()` (SAFE-Bridge-CSV)
- `scripts/run_backtest_strategy.py`: Equity-Curves (`equity_curve_{freq}.csv`), Reports (`performance_report_{freq}.md`)
- `src/assembled_core/qa/backtest_engine.py`: Equity-Curve in `BacktestResult.equity`, optional Trades/Signals/Targets
- `src/assembled_core/paper/paper_track.py`: State-Persistence (`state.json`), Daily-Summary (`daily_summary.json`), Equity-Curve (`equity_curve.csv`)

**Gemeinsamkeiten:**
- Output-Verzeichnis-Struktur (`output/`, `output/paper_track/{strategy}/`, etc.)
- Logging (gleiche Log-Struktur)

---

## 3. API-Skizze

### 3.1 TradingContext

```python
@dataclass
class TradingContext:
    """Unified context for trading cycle execution.
    
    Attributes:
        # Input data
        prices: DataFrame with columns: timestamp, symbol, close, ... (OHLCV)
        as_of: pd.Timestamp | None - Point-in-time cutoff (PIT-safe filtering)
        universe: list[str] | None - Universe symbols (optional, for validation)
        
        # Feature building
        feature_config: FeatureConfig | None - Configuration for feature building
        use_factor_store: bool - Enable factor store caching
        factor_store_root: Path | None - Factor store root directory
        factor_group: str - Factor group name (e.g., "core_ta")
        
        # Signal generation
        signal_fn: Callable[[pd.DataFrame], pd.DataFrame] - Signal function
        signal_config: dict[str, Any] - Signal-specific config (e.g., ma_fast, ma_slow)
        
        # Position sizing
        position_sizing_fn: Callable[[pd.DataFrame, float], pd.DataFrame] - Position sizing function
        capital: float - Total capital for position sizing
        
        # Order generation
        current_positions: pd.DataFrame | None - Current positions (columns: symbol, qty)
        order_timestamp: pd.Timestamp - Timestamp for orders
        
        # Risk controls
        enable_risk_controls: bool - Enable risk controls (pre-trade checks, kill switch)
        risk_config: dict[str, Any] - Risk control configuration
        
        # Outputs
        output_dir: Path - Output directory
        output_format: Literal["safe_csv", "equity_curve", "state"] - Output format
        write_outputs: bool - Whether to write output files
        
        # Metadata
        run_id: str | None - Run identifier (for logging/tracking)
        strategy_name: str | None - Strategy name
    """
    
    # Input data
    prices: pd.DataFrame
    as_of: pd.Timestamp | None = None
    universe: list[str] | None = None
    
    # Feature building
    feature_config: FeatureConfig | None = None
    use_factor_store: bool = False
    factor_store_root: Path | None = None
    factor_group: str = "core_ta"
    
    # Signal generation
    signal_fn: Callable[[pd.DataFrame], pd.DataFrame]
    signal_config: dict[str, Any] = field(default_factory=dict)
    
    # Position sizing
    position_sizing_fn: Callable[[pd.DataFrame, float], pd.DataFrame]
    capital: float = 10000.0
    
    # Order generation
    current_positions: pd.DataFrame | None = None
    order_timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())
    
    # Risk controls
    enable_risk_controls: bool = True
    risk_config: dict[str, Any] = field(default_factory=dict)
    
    # Outputs
    output_dir: Path = Path("output")
    output_format: Literal["safe_csv", "equity_curve", "state"] = "safe_csv"
    write_outputs: bool = True
    
    # Metadata
    run_id: str | None = None
    strategy_name: str | None = None
```

### 3.2 TradingCycleResult

```python
@dataclass
class TradingCycleResult:
    """Result of unified trading cycle execution.
    
    Attributes:
        # Intermediate results
        prices_filtered: pd.DataFrame - Prices after filtering (as_of, universe)
        prices_with_features: pd.DataFrame - Prices with computed features
        signals: pd.DataFrame - Generated signals (columns: timestamp, symbol, direction, score)
        target_positions: pd.DataFrame - Target positions (columns: symbol, target_weight, target_qty)
        orders: pd.DataFrame - Generated orders (columns: timestamp, symbol, side, qty, price)
        orders_filtered: pd.DataFrame - Orders after risk controls
        
        # Metadata
        run_id: str | None - Run identifier
        timestamp: pd.Timestamp - Execution timestamp
        status: Literal["success", "error"] - Execution status
        error_message: str | None - Error message if status == "error"
        
        # Output paths (if written)
        output_paths: dict[str, Path] - Dictionary of output file paths (e.g., {"safe_csv": Path(...)})
    """
    
    # Intermediate results
    prices_filtered: pd.DataFrame
    prices_with_features: pd.DataFrame
    signals: pd.DataFrame
    target_positions: pd.DataFrame
    orders: pd.DataFrame
    orders_filtered: pd.DataFrame
    
    # Metadata
    run_id: str | None = None
    timestamp: pd.Timestamp = field(default_factory=lambda: pd.Timestamp.utcnow())
    status: Literal["success", "error"] = "success"
    error_message: str | None = None
    
    # Output paths
    output_paths: dict[str, Path] = field(default_factory=dict)
```

### 3.3 run_trading_cycle

```python
def run_trading_cycle(ctx: TradingContext) -> TradingCycleResult:
    """Execute unified trading cycle.
    
    Steps:
    1. Filter prices (as_of, universe validation)
    2. Build features (TA features, factor store integration)
    3. Generate signals (via signal_fn)
    4. Compute target positions (via position_sizing_fn)
    5. Generate orders (current_positions vs. target_positions)
    6. Apply risk controls (pre-trade checks, kill switch)
    7. Write outputs (SAFE-CSV, equity curve, state, etc.)
    
    Args:
        ctx: TradingContext with all configuration and data
        
    Returns:
        TradingCycleResult with intermediate results and outputs
        
    Raises:
        ValueError: If required context fields are missing or invalid
        FileNotFoundError: If required input files are missing
    """
    # Step 1: Filter prices
    prices_filtered = _filter_prices(ctx.prices, ctx.as_of, ctx.universe)
    
    # Step 2: Build features
    prices_with_features = _build_features(prices_filtered, ctx)
    
    # Step 3: Generate signals
    signals = ctx.signal_fn(prices_with_features)
    
    # Step 4: Compute target positions
    target_positions = ctx.position_sizing_fn(signals, ctx.capital)
    
    # Step 5: Generate orders
    orders = _generate_orders(target_positions, ctx.current_positions, ctx.order_timestamp)
    
    # Step 6: Apply risk controls
    orders_filtered = _apply_risk_controls(orders, ctx) if ctx.enable_risk_controls else orders
    
    # Step 7: Write outputs
    output_paths = _write_outputs(orders_filtered, ctx) if ctx.write_outputs else {}
    
    return TradingCycleResult(
        prices_filtered=prices_filtered,
        prices_with_features=prices_with_features,
        signals=signals,
        target_positions=target_positions,
        orders=orders,
        orders_filtered=orders_filtered,
        run_id=ctx.run_id,
        output_paths=output_paths,
    )
```

---

## 4. Mapping: Bisherige Steps -> Neue API

### 4.1 run_daily.py

**Aktuell:**
```python
# Step 1: Load prices
prices = load_eod_prices(...)

# Step 2: Filter prices (date, universe)
prices_filtered = filter_prices_for_date(prices, target_date, mode="last_available")

# Step 3: Build features
prices_with_features = add_all_features(prices_filtered, ...)  # oder build_or_load_factors(...)

# Step 4: Generate signals
signals = generate_trend_signals_from_prices(prices_with_features, ...)

# Step 5: Compute target positions
targets = compute_target_positions_from_trend_signals(signals, ...)

# Step 6: Generate orders
orders = generate_orders_from_signals(signals, ...)

# Step 7: Apply risk controls
filtered_orders = filter_orders_with_risk_controls(orders, ...)

# Step 8: Write SAFE-CSV
safe_path = write_safe_orders_csv(filtered_orders, ...)
```

**Mit Unified API:**
```python
ctx = TradingContext(
    prices=prices,
    as_of=target_date,
    universe=universe_symbols,
    signal_fn=lambda df: generate_trend_signals_from_prices(df, ma_fast=ma_fast, ma_slow=ma_slow),
    position_sizing_fn=lambda sig, cap: compute_target_positions_from_trend_signals(sig, total_capital=cap, top_n=top_n, min_score=min_score),
    capital=total_capital,
    current_positions=None,  # EOD: keine aktuellen Positionen
    order_timestamp=target_date,
    enable_risk_controls=not disable_pre_trade_checks,
    output_dir=output_dir,
    output_format="safe_csv",
    use_factor_store=use_factor_store,
    factor_store_root=factor_store_root,
    factor_group=factor_group,
)

result = run_trading_cycle(ctx)
safe_path = result.output_paths["safe_csv"]
```

### 4.2 run_backtest_strategy.py

**Aktuell:**
```python
# Prices werden bereits in run_portfolio_backtest() als Parameter übergeben
# Features werden optional in run_portfolio_backtest() gebaut (compute_features=True)
# Signals werden pro Timestamp via signal_fn() generiert
# Position sizing wird pro Timestamp via position_sizing_fn() berechnet
# Orders werden in _process_rebalancing_timestamp() generiert
```

**Mit Unified API:**
- `run_portfolio_backtest()` kann `run_trading_cycle()` pro Timestamp aufrufen
- Oder: Pre-compute features einmal, dann `run_trading_cycle()` pro Timestamp (nur Signals -> Orders)

### 4.3 paper_track.py

**Aktuell:**
```python
# In _run_paper_track_day():
# 1. Load prices
prices = load_eod_prices_for_universe(...)

# 2. Filter prices
prices_filtered = _filter_prices_for_date(prices, as_of)

# 3. Build features
prices_with_features = add_all_features(prices_filtered, ...)

# 4. Generate signals + targets (via strategy adapters)
signals, targets = generate_signals_and_targets_for_day(prices_with_features, ...)

# 5. Generate orders
orders = generate_orders_from_targets(targets, current_positions, ...)

# 6. Simulate fills, update positions, etc. (paper-track-spezifisch)
```

**Mit Unified API:**
```python
ctx = TradingContext(
    prices=prices,
    as_of=as_of,
    universe=universe_symbols,
    signal_fn=lambda df: generate_signals_and_targets_for_day(df, ...)[0],  # Extract signals
    position_sizing_fn=lambda sig, cap: generate_signals_and_targets_for_day(prices_with_features, ...)[1],  # Extract targets
    capital=state.equity,  # Paper track: current equity
    current_positions=state.positions,
    order_timestamp=as_of,
    enable_risk_controls=False,  # Paper track: risk controls optional
    output_dir=output_dir,
    output_format="state",  # Paper track: state persistence
)

result = run_trading_cycle(ctx)
# Dann: Paper-track-spezifische Fill-Simulation, State-Update, etc.
```

---

## 5. Migrationsplan

### Schritt 1: Neue Pipeline-API implementieren

**Datei:** `src/assembled_core/pipeline/trading_cycle.py` (neu)

**Implementierung:**
- `TradingContext` Dataclass
- `TradingCycleResult` Dataclass
- `run_trading_cycle(ctx: TradingContext) -> TradingCycleResult`
- Helper-Funktionen:
  - `_filter_prices()` (konsolidiert `filter_prices_for_date()`, `_filter_prices_for_date()`, etc.)
  - `_build_features()` (konsolidiert `add_all_features()`, `build_or_load_factors()`)
  - `_generate_orders()` (konsolidiert `generate_orders_from_signals()`, `generate_orders_from_targets()`)
  - `_apply_risk_controls()` (Wrapper über `filter_orders_with_risk_controls()`)
  - `_write_outputs()` (konsolidiert `write_safe_orders_csv()`, Equity-Curve-Writing, etc.)

**Tests:**
- `tests/test_pipeline_trading_cycle.py` (Unit-Tests für jede Helper-Funktion)
- Integration-Test: Vergleich mit bestehender `run_daily_eod()` Logik

### Schritt 2: run_daily.py umstellen

**Änderungen:**
- `run_daily_eod()` refaktorisieren, um `run_trading_cycle()` aufzurufen
- Alte Step-Funktionen als deprecated markieren (oder intern für Backward-Compat)
- CLI-Interface bleibt unverändert (keine Breaking Changes)

**Tests:**
- Bestehende Tests für `run_daily.py` müssen weiterhin grün sein
- Smoke-Test: Vergleich Output (SAFE-CSV) vor/nach Migration

### Schritt 3: paper_track.py optional umstellen

**Änderungen:**
- `_run_paper_track_day()` refaktorisieren, um `run_trading_cycle()` zu nutzen
- Paper-Track-spezifische Logik (Fill-Simulation, State-Update) bleibt unverändert
- Optional: Flag `--use-unified-cycle` für schrittweise Migration

**Tests:**
- Bestehende Paper-Track-Tests müssen weiterhin grün sein
- Smoke-Test: Vergleich State/Equity vor/nach Migration

**Optional:**
- `run_backtest_strategy.py` kann später umgestellt werden (weniger kritisch, da bereits abstrahiert)

---

## 6. Risiken & Mitigation

### Risiko 1: Breaking Changes

**Mitigation:**
- Schrittweise Migration (jeder Schritt isoliert testbar)
- Backward-Compat-Layer (alte Funktionen bleiben verfügbar)
- Deprecation-Warnings für alte APIs

### Risiko 2: Performance-Regression

**Mitigation:**
- Benchmarks vor/nach Migration
- Profiling (P3-Infrastruktur nutzen)
- Optional: Caching in `_build_features()` (Factor-Store bereits vorhanden)

### Risiko 3: Feature-Loss

**Mitigation:**
- Vollständige Feature-Matrix (alle bisherigen Optionen in `TradingContext`)
- Integration-Tests garantieren gleiche Outputs
- Dokumentation: Migration-Guide mit Beispielen

---

## 7. Success Criteria

**Funktional:**
- `run_trading_cycle()` deckt alle gemeinsamen Steps ab
- `run_daily.py` nutzt neue API (Schritt 2)
- `paper_track.py` kann neue API nutzen (Schritt 3, optional)
- Alle bestehenden Tests bleiben grün

**Nicht-funktional:**
- Keine Performance-Regression (< 5% Overhead akzeptabel)
- Code-Reduktion: ~30-40% weniger Duplikation in Step-Orchestrierung
- Wartbarkeit: Neue Trading-Strategien können `run_trading_cycle()` direkt nutzen

**Testing:**
- Unit-Tests für `run_trading_cycle()` und Helper-Funktionen
- Integration-Tests: Vergleich Outputs vor/nach Migration
- Regression-Tests: Bestehende Tests bleiben grün

---

## 8. Out of Scope (für B1)

- **State-Management-Vereinheitlichung:** EOD (stateless), Backtest (temporal), Paper-Track (persistent) bleiben unterschiedlich
- **Output-Format-Vereinheitlichung:** SAFE-CSV, Equity-Curves, State-Persistence bleiben unterschiedlich
- **Backtest-Engine-Umstellung:** `run_backtest_strategy.py` kann später umgestellt werden (weniger kritisch)
- **Fill-Simulation:** Paper-Track-spezifische Fill-Simulation bleibt außerhalb `run_trading_cycle()`

---

## 9. References

- [Backend Architecture](ARCHITECTURE_BACKEND.md) - Overview der Backend-Module
- [Advanced Analytics & Factor Labs](ADVANCED_ANALYTICS_FACTOR_LABS.md) - B1 Phase
- `scripts/run_daily.py` - EOD-MVP Runner
- `scripts/run_backtest_strategy.py` - Backtest Runner
- `src/assembled_core/qa/backtest_engine.py` - Portfolio-Level Backtest Engine
- `src/assembled_core/paper/paper_track.py` - Paper Track Runner

