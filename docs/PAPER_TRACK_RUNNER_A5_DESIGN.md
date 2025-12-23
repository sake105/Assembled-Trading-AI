# Paper-Track Runner (A5) - Design

**Phase:** Advanced Analytics & Factor Labs - Track A (Anwendung & Produktisierung)  
**Status:** Design Phase  
**Last Updated:** 2025-01-XX

---

## 1. Overview & Scope

**Ziel:** Ein "EOD-like" Paper-Track-Runner, der taglich lauft und Strategien im Paper-Track simuliert (ohne echte Orders, read-only bzgl. Datenquellen).

**Scope:**

- **Paper-only:** Keine echten Orders werden an Broker/APIs gesendet
- **Read-only bzgl. Datenquellen:** Nur lokale Daten lesen (Parquet/CSV), keine Netzwerk-Calls
- **Stateful:** Persistiert Portfolio-State (Positions, Cash) zwischen Runs
- **Daily Flow:** Jeden Tag: State laden → Signale berechnen (as_of) → Position-Sizing → Simulierte Fills → State speichern → Artefakte schreiben

**Bezug zu bestehenden Modulen:**

- Nutzt `run_daily.py` Flow als Basis (Load prices → Features → Signals → Positions → Orders)
- Erweitert um State-Management (Positions, Cash persistieren)
- Erweitert um Fill-Simulation (Orders in Positions umwandeln mit Kosten)
- Integriert mit Risk-Reports (optional), Health-Checks (A3), PIT-Guards (B2)

**Abgrenzung:**

- **Backtest (`run_portfolio_backtest`):** Stateless, lauft uber komplette Historie
- **Paper-Track-Runner:** Stateful, lauft taglich, simuliert kontinuierlichen Betrieb
- **EOD-Pipeline (`run_daily.py`):** Generiert nur Orders, keine State-Persistenz, keine Fill-Simulation

---

## 2. Config Contract (YAML/JSON)

### 2.1 Config-Schema

```yaml
# Example: configs/paper_track/strategy_core_ai_tech.yaml

strategy_name: "core_ai_tech"
description: "Core AI/Tech factors strategy in paper track"

# Strategy configuration
strategy:
  type: "multifactor_long_short"  # or "trend_baseline", etc.
  bundle_path: "config/factor_bundles/ai_tech_core_bundle.yaml"  # if multifactor
  # Strategy-specific parameters
  ma_fast: 20  # if trend_baseline
  ma_slow: 50  # if trend_baseline

# Universe
universe:
  file: "config/universe_ai_tech_tickers.txt"
  # or: symbols: ["AAPL", "MSFT", "GOOGL", ...]

# Trading parameters
trading:
  freq: "1d"  # "1d" or "5min"
  rebalance_freq: "M"  # "D", "W", "M" (daily, weekly, monthly)
  max_gross_exposure: 1.0  # Maximum gross exposure (100%)

# Cost model
costs:
  commission_bps: 0.5  # Commission in basis points
  spread_w: 0.25  # Spread weight (multiplier for bid/ask spread)
  impact_w: 0.5  # Market impact weight

# Portfolio
portfolio:
  seed_capital: 100000.0  # Starting capital
  # or: start_capital (alias for seed_capital)

# Output
output:
  root: "output/paper_track"  # Base output directory
  strategy_dir: "strategy_core_ai_tech"  # Subdirectory for this strategy

# Optional: Benchmark for comparison
benchmark:
  symbol: "SPY"  # or None
  file: null  # Path to benchmark returns file (CSV/Parquet)

# Optional: Thresholds for alerts
thresholds:
  min_sharpe: 0.5  # Minimum acceptable Sharpe Ratio
  max_drawdown_min: -0.25  # Critical threshold for Max Drawdown
  max_drawdown_max: -0.05  # Warning threshold for Max Drawdown
  max_turnover: 0.5  # Maximum acceptable daily turnover

# Optional: Integration flags
integration:
  enable_risk_report: true  # Generate risk reports (weekly/monthly)
  enable_health_check: true  # Enable health checks (A3)
  enable_pit_checks: true  # Enable PIT-safety checks (B2)
  enable_regime_analysis: false  # Enable regime analysis in risk reports (B3)
  enable_factor_exposures: false  # Enable factor exposures in risk reports (A2)
  enable_deflated_sharpe: false  # Calculate deflated Sharpe (B4, requires n_tests)
```

### 2.2 Config-Validierung

- **Required Fields:** `strategy_name`, `strategy.type`, `universe` (file or symbols), `trading.freq`, `portfolio.seed_capital`
- **Optional Fields:** Alle anderen Felder haben Defaults
- **Validation:** 
  - `freq` muss in `SUPPORTED_FREQS` sein
  - `seed_capital` > 0
  - `max_gross_exposure` > 0
  - `commission_bps`, `spread_w`, `impact_w` >= 0

---

## 3. State Model

### 3.1 State-Struktur

Das Paper-Track-System persistiert State zwischen Runs:

**State-Datei:** `{output_root}/{strategy_dir}/state/state.json` oder `state.parquet`

**State-Fields:**

```python
@dataclass
class PaperTrackState:
    """State of paper track portfolio.
    
    Attributes:
        strategy_name: Name of the strategy (from config)
        last_run_date: Last date when paper track was executed (pd.Timestamp, UTC)
        version: Version of state format (for migration/compatibility)
        
        # Portfolio state
        positions: DataFrame with columns: symbol, qty (positive = long, negative = short)
        cash: Current cash balance (float)
        equity: Current portfolio equity (cash + mark-to-market positions)
        
        # Optional: Open orders (if not yet filled)
        open_orders: Optional[DataFrame] with columns: timestamp, symbol, side, qty, price, status
        
        # Metadata
        created_at: Timestamp when state was first created
        updated_at: Timestamp when state was last updated
        total_trades: Total number of trades executed since start
        total_pnl: Cumulative PnL (equity - seed_capital)
    """
    
    strategy_name: str
    last_run_date: pd.Timestamp | None
    version: str = "1.0"
    
    # Portfolio
    positions: pd.DataFrame  # columns: symbol, qty
    cash: float
    equity: float
    
    # Optional
    open_orders: pd.DataFrame | None = None
    
    # Metadata
    created_at: pd.Timestamp
    updated_at: pd.Timestamp
    total_trades: int = 0
    total_pnl: float = 0.0
```

**State-Persistenz:**

- **Format:** JSON (human-readable) oder Parquet (efficient) - konfigurierbar
- **Location:** `{output_root}/{strategy_dir}/state/`
- **Backup:** Vor jedem Update: Backup von `state.json` → `state.json.backup`
- **Atomic Write:** Write to temp file, then rename (to avoid corruption on crash)

### 3.2 State-Loading & Initialization

**Erster Run (kein State vorhanden):**

- Erstelle neues State mit `positions = empty`, `cash = seed_capital`, `equity = seed_capital`
- `last_run_date = None`
- `created_at = now()`

**Nachfolgende Runs:**

- Lade State aus Datei
- Validierung: `strategy_name` muss mit Config ubereinstimmen
- Validierung: `version` muss kompatibel sein (bei Incompatibilitat: Migration oder Error)
- `last_run_date` wird verwendet, um zu prufen, ob Run bereits fur diesen Tag erfolgt ist

### 3.3 State-Update (nach Fill-Simulation)

Nach erfolgreicher Fill-Simulation:

- `positions`: Aktualisiert aus Orders (via `_update_positions_vectorized()`)
- `cash`: Aktualisiert durch Fill-Kosten (Order-Preis + Commission + Spread + Impact)
- `equity`: Neu berechnet als `cash + sum(positions * current_prices)`
- `last_run_date`: Aktualisiert auf aktuelles Run-Datum
- `updated_at`: Aktualisiert auf `now()`
- `total_trades`: Inkrementiert um Anzahl gefullter Orders
- `total_pnl`: Neu berechnet als `equity - seed_capital`

---

## 4. Daily Flow

### 4.1 Flow-Ubersicht

```
1. Load Config
   ↓
2. Load State (or initialize if first run)
   ↓
3. Check: Already run for today? (if yes, skip or error)
   ↓
4. Load Prices (as_of = today, last available <= today)
   ↓
5. Compute Features (PIT-safe: only use data <= as_of)
   ↓
6. Generate Signals (as_of = today)
   ↓
7. Compute Target Positions (based on signals + current equity)
   ↓
8. Generate Orders (difference: target_positions - current_positions)
   ↓
9. Simulate Fills (apply costs, update cash)
   ↓
10. Update State (positions, cash, equity, metadata)
   ↓
11. Write State (atomic write with backup)
   ↓
12. Write Daily Artifacts (equity snapshot, positions, orders/trades, summary)
   ↓
13. Optional: Generate Risk-Report (if weekly/monthly trigger)
   ↓
14. Optional: Run Health-Check (A3 integration)
```

### 4.2 Schritt-fur-Schritt

#### Step 1: Load Config

- Lade Config-Datei (YAML/JSON)
- Validiere Required Fields
- Setze Defaults fur Optional Fields
- Erstelle `PaperTrackConfig` Dataclass

#### Step 2: Load State (or Initialize)

- Prufe, ob State-Datei existiert
- **Wenn nicht vorhanden:** Erstelle neues State (siehe 3.2)
- **Wenn vorhanden:** Lade State, validiere `strategy_name` und `version`

#### Step 3: Check Run-Freshness

- Wenn `last_run_date` == heute: Skip (oder Error, wenn `--force` nicht gesetzt)
- Wenn `last_run_date` < heute: Continue
- Wenn `last_run_date` > heute: Error (State ist in der Zukunft, moglicherweise Clock-Problem)

#### Step 4: Load Prices

- Nutze `load_eod_prices_for_universe()` oder `load_eod_prices()`
- Filter: Nur Daten <= `as_of` (heute)
- Pro Symbol: Letzte verfugbare Preis-Date <= `as_of`
- PIT-Safety: Keine Daten aus der Zukunft

#### Step 5: Compute Features

- Nutze `add_all_features()` oder Strategy-spezifische Feature-Funktion
- **PIT-Guard (optional):** Wenn `enable_pit_checks=true`, rufe `check_features_pit_safe(features, as_of=today, strict=True)`
- Features mussen Timestamps <= `as_of` haben

#### Step 6: Generate Signals

- Strategy-abhangig:
  - **multifactor_long_short:** Nutze `compute_multifactor_long_short_signals()` mit Bundle
  - **trend_baseline:** Nutze `generate_trend_signals_from_prices()`
- Signals haben Timestamp = `as_of` (heute)

#### Step 7: Compute Target Positions

- Nutze `compute_target_positions_from_trend_signals()` oder multifactor Position-Sizing
- **Input:** Signals, `total_capital = current_equity` (nicht seed_capital!)
- **Output:** DataFrame mit `symbol`, `target_weight`, `target_qty`

#### Step 8: Generate Orders

- Nutze `generate_orders_from_targets()` oder `generate_orders_from_signals()`
- **Input:** Target Positions, Current Positions (aus State), Prices (fur Fill-Preise)
- **Output:** DataFrame mit `timestamp`, `symbol`, `side`, `qty`, `price`
- Orders reprasentieren die Differenz zwischen Target und Current Positions

#### Step 9: Simulate Fills

- **Fill-Logik:**
  - BUY Order: Cash reduziert um `qty * price * (1 + spread_w + impact_w) + commission_bps * notional`
  - SELL Order: Cash erhoht um `qty * price * (1 - spread_w - impact_w) - commission_bps * notional`
  - Orders werden sofort gefullt (keine Teil-Fills, keine Slippage-Modellierung zunachst)
- **Position-Update:**
  - Nutze `_update_positions_vectorized()` aus `backtest_engine.py`
  - Input: Current Positions (aus State), Orders (gefullt)
  - Output: Updated Positions
- **Fill-Status:** Alle Orders werden als "FILLED" markiert

#### Step 10: Update State

- **Positions:** Updated Positions aus Step 9
- **Cash:** Neu berechnet aus vorherigem Cash + Fill-Deltas
- **Equity:** Neu berechnet als `cash + sum(positions * current_prices)` (Mark-to-Market)
- **Metadata:** `last_run_date`, `updated_at`, `total_trades`, `total_pnl`

#### Step 11: Write State

- **Backup:** Kopiere `state.json` → `state.json.backup` (falls vorhanden)
- **Atomic Write:** Schreibe zu `state.json.tmp`, dann rename zu `state.json`
- **Format:** JSON (human-readable) oder Parquet (efficient)

#### Step 12: Write Daily Artifacts

**Output-Verzeichnis:** `{output_root}/{strategy_dir}/runs/{YYYYMMDD}/`

**Dateien:**

- `equity_snapshot.json`: Aktueller Equity-Wert, Timestamp
- `positions.csv`: Aktuelle Positions (symbol, qty)
- `orders_today.csv`: Heute generierte/gefullte Orders (timestamp, symbol, side, qty, price, fill_price, costs)
- `trades_today.csv`: Gleiche wie `orders_today.csv` (fur Kompatibilitat mit Backtest-Format)
- `daily_summary.json`: Tages-Metriken (equity, cash, pnl, trades_count, turnover, etc.)
- `daily_summary.md`: Human-readable Tages-Zusammenfassung

#### Step 13: Optional Risk-Report

- **Trigger:** Weekly (z.B. jeden Freitag) oder Monthly (z.B. am letzten Tag des Monats)
- **Input:** Equity-Curve (aus allen bisherigen Runs), Positions, Trades
- **Tool:** Nutze `scripts/generate_risk_report.py` oder programmatische API
- **Output:** `risk_report.md`, `risk_summary.csv`, etc. in `{output_root}/{strategy_dir}/risk_reports/`

#### Step 14: Optional Health-Check

- **Tool:** Nutze `scripts/check_health.py` oder programmatische API (A3)
- **Input:** Equity-Curve, Positions, Risk-Reports (falls vorhanden)
- **Output:** `health_summary.json`, `health_summary.md` in `{output_root}/{strategy_dir}/health/`
- **Action:** Bei CRITICAL: Log Alert, optional: Pause Paper-Track (State bleibt erhalten)

---

## 5. Output-Struktur

### 5.1 Verzeichnisstruktur

```
{output_root}/{strategy_dir}/
├── config.yaml  # Copy of config file (for reproducibility)
├── state/
│   ├── state.json  # Current state
│   └── state.json.backup  # Backup (previous state)
├── runs/
│   ├── 20250115/  # Daily run outputs
│   │   ├── equity_snapshot.json
│   │   ├── positions.csv
│   │   ├── orders_today.csv
│   │   ├── trades_today.csv
│   │   ├── daily_summary.json
│   │   └── daily_summary.md
│   ├── 20250116/
│   │   └── ...
│   └── ...
├── equity_curve.csv  # Aggregated equity curve (all runs)
├── positions_history.csv  # Historical positions (optional, can be large)
├── trades_all.csv  # Aggregated trades (all runs)
├── risk_reports/  # Weekly/monthly risk reports (optional)
│   ├── risk_report_20250115.md
│   ├── risk_summary_20250115.csv
│   └── ...
└── health/  # Health check outputs (optional)
    ├── health_summary_20250115.json
    └── health_summary_20250115.md
```

### 5.2 Equity-Snapshot Format

**File:** `runs/{YYYYMMDD}/equity_snapshot.json`

```json
{
  "timestamp": "2025-01-15T00:00:00Z",
  "equity": 102345.67,
  "cash": 23456.78,
  "positions_value": 78888.89,
  "seed_capital": 100000.0,
  "total_pnl": 2345.67,
  "total_return_pct": 2.35
}
```

### 5.3 Daily Summary Format

**File:** `runs/{YYYYMMDD}/daily_summary.json`

```json
{
  "date": "2025-01-15",
  "equity": 102345.67,
  "cash": 23456.78,
  "positions_value": 78888.89,
  "daily_return_pct": 0.23,
  "daily_pnl": 234.56,
  "trades_count": 5,
  "buy_count": 3,
  "sell_count": 2,
  "turnover": 0.15,
  "sharpe_daily": null,
  "max_drawdown": -0.05,
  "positions_count": 8
}
```

**File:** `runs/{YYYYMMDD}/daily_summary.md`

```markdown
# Daily Summary - 2025-01-15

## Portfolio
- Equity: $102,345.67
- Cash: $23,456.78
- Positions Value: $78,888.89
- Daily Return: +0.23%
- Daily PnL: +$234.56

## Trading
- Trades: 5 (3 BUY, 2 SELL)
- Turnover: 15.0%
- Positions: 8

## Performance
- Total Return: +2.35%
- Max Drawdown: -5.0%
```

### 5.4 Aggregated Equity Curve

**File:** `equity_curve.csv`

**Spalten:** `timestamp`, `equity`, `cash`, `positions_value`, `daily_return`, `cumulative_return`

**Aktualisierung:** Nach jedem Run wird eine neue Zeile angehangt (oder Datei neu generiert aus allen Runs)

---

## 6. Integration Points

### 6.1 Risk-Report Integration (Optional)

**Trigger:** Weekly oder Monthly (konfigurierbar)

**Inputs:**

- Equity-Curve: Aggregiert aus allen Runs (`equity_curve.csv`)
- Positions: Aktuelle Positions (aus State oder `positions_history.csv`)
- Trades: Aggregiert aus allen Runs (`trades_all.csv`)

**Tool:**

- Nutze `scripts/generate_risk_report.py` programmatisch oder via Subprocess
- Flags: `--enable-regime-analysis`, `--enable-factor-exposures` (aus Config)

**Outputs:**

- `risk_report_{date}.md`
- `risk_summary_{date}.csv`
- `exposure_timeseries.csv`
- `risk_by_regime.csv` (wenn Regime-Analyse aktiviert)

### 6.2 Health-Check Integration (A3, Optional)

**Trigger:** Daily (nach State-Update)

**Inputs:**

- Equity-Curve: Aggregiert (`equity_curve.csv`)
- Risk-Reports: Neueste Risk-Reports (falls vorhanden)
- State: Aktueller State (fur Timestamp-Validierung)

**Tool:**

- Nutze `scripts/check_health.py` programmatisch oder via Subprocess
- Thresholds: Aus Config (`thresholds.min_sharpe`, `thresholds.max_drawdown_min`, etc.)

**Outputs:**

- `health_summary_{date}.json`
- `health_summary_{date}.md`

**Action bei CRITICAL:**

- Log Alert (ERROR-Level)
- Optional: Pause Paper-Track (State bleibt erhalten, aber kein weiterer Run bis manuell fortgesetzt)
- Dokumentation in `health/alerts.log`

### 6.3 PIT-Guards Integration (B2, Optional)

**Integration-Punkte:**

1. **Features (Step 5):** 
   - Nach Feature-Computation: `check_features_pit_safe(features, as_of=today, strict=enable_pit_checks)`
   - Bei Violation: Error (strict=True) oder Warning (strict=False)

2. **Signals (Step 6):**
   - Nach Signal-Generation: PIT-Check auf Signal-Timestamps
   - Bei Violation: Error

**Config-Flag:** `integration.enable_pit_checks` (default: true)

### 6.4 Regime-Analyse Integration (B3, Optional)

**Trigger:** In Risk-Reports (wenn `enable_regime_analysis=true`)

**Inputs:**

- Equity-Curve: Aggregiert
- Regime-State: Aus `scripts/generate_risk_report.py` oder separate Regime-Datei

**Outputs:**

- `risk_by_regime.csv`: Performance nach Regime (Bull/Bear/Crisis)
- Erweitertes `risk_report.md` mit Regime-Abschnitt

### 6.5 Deflated Sharpe Integration (B4, Optional)

**Trigger:** In Risk-Reports oder separate Metrik-Berechnung (wenn `enable_deflated_sharpe=true`)

**Inputs:**

- Returns: Aus Equity-Curve berechnet
- `n_tests`: Anzahl getesteter Strategien/Varianten (aus Config oder manuell gesetzt)

**Outputs:**

- `deflated_sharpe` in `risk_summary.csv`
- Erwahnt in `risk_report.md`

---

## 7. API Design

### 7.1 Dataclasses

```python
@dataclass
class PaperTrackConfig:
    """Configuration for paper track runner.
    
    Attributes:
        strategy_name: Name of the strategy
        strategy: Strategy configuration (type, bundle_path, parameters)
        universe: Universe definition (file or symbols list)
        trading: Trading parameters (freq, rebalance_freq, max_gross_exposure)
        costs: Cost model (commission_bps, spread_w, impact_w)
        portfolio: Portfolio parameters (seed_capital)
        output: Output configuration (root, strategy_dir)
        benchmark: Optional benchmark configuration
        thresholds: Optional thresholds for alerts
        integration: Optional integration flags
    """
    
    strategy_name: str
    strategy: dict[str, Any]
    universe: dict[str, Any]
    trading: dict[str, Any]
    costs: dict[str, float]
    portfolio: dict[str, float]
    output: dict[str, str]
    benchmark: dict[str, Any] | None = None
    thresholds: dict[str, float] | None = None
    integration: dict[str, bool] | None = None


@dataclass
class PaperTrackState:
    """State of paper track portfolio (siehe 3.1)."""
    
    strategy_name: str
    last_run_date: pd.Timestamp | None
    version: str = "1.0"
    positions: pd.DataFrame
    cash: float
    equity: float
    open_orders: pd.DataFrame | None = None
    created_at: pd.Timestamp
    updated_at: pd.Timestamp
    total_trades: int = 0
    total_pnl: float = 0.0


@dataclass
class PaperTrackRunResult:
    """Result of a single paper track run.
    
    Attributes:
        date: Run date
        config: PaperTrackConfig used
        state_before: State before run
        state_after: State after run
        orders: Orders generated and filled
        trades: Trades executed (same as orders, for compatibility)
        daily_metrics: Daily metrics (equity, cash, pnl, trades_count, etc.)
        status: "success" or "error"
        error_message: Error message if status == "error"
    """
    
    date: pd.Timestamp
    config: PaperTrackConfig
    state_before: PaperTrackState
    state_after: PaperTrackState
    orders: pd.DataFrame
    trades: pd.DataFrame
    daily_metrics: dict[str, float | int]
    status: Literal["success", "error"] = "success"
    error_message: str | None = None
```

### 7.2 Core Functions

```python
def load_paper_track_config(config_path: Path) -> PaperTrackConfig:
    """Load paper track configuration from YAML/JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        PaperTrackConfig instance
        
    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid
    """


def load_paper_track_state(state_dir: Path, strategy_name: str) -> PaperTrackState | None:
    """Load paper track state from file.
    
    Args:
        state_dir: Directory containing state file
        strategy_name: Expected strategy name (for validation)
        
    Returns:
        PaperTrackState if file exists, None otherwise
        
    Raises:
        ValueError: If state is invalid or strategy_name mismatch
    """


def save_paper_track_state(state: PaperTrackState, state_dir: Path) -> None:
    """Save paper track state to file (atomic write with backup).
    
    Args:
        state: PaperTrackState to save
        state_dir: Directory to save state file
        
    Raises:
        IOError: If write fails
    """


def simulate_order_fills(
    orders: pd.DataFrame,
    current_cash: float,
    commission_bps: float,
    spread_w: float,
    impact_w: float,
) -> tuple[pd.DataFrame, float]:
    """Simulate order fills and update cash.
    
    Args:
        orders: DataFrame with columns: timestamp, symbol, side, qty, price
        current_cash: Current cash balance
        commission_bps: Commission in basis points
        spread_w: Spread weight
        impact_w: Impact weight
        
    Returns:
        Tuple of (filled_orders DataFrame with fill_price and costs columns, new_cash balance)
    """


def run_paper_track_daily(
    config: PaperTrackConfig,
    run_date: pd.Timestamp | None = None,
    force: bool = False,
) -> PaperTrackRunResult:
    """Run paper track for a single day.
    
    Args:
        config: PaperTrackConfig
        run_date: Date to run for (default: today)
        force: If True, allow re-running for the same date (default: False)
        
    Returns:
        PaperTrackRunResult
        
    Raises:
        ValueError: If already run for date (and force=False)
        FileNotFoundError: If required data files not found
        PointInTimeViolationError: If PIT-checks enabled and violation detected
    """
```

### 7.3 Module-Struktur

**Neues Modul:** `src/assembled_core/paper_track/paper_track_runner.py`

**Exports:**

- `PaperTrackConfig`
- `PaperTrackState`
- `PaperTrackRunResult`
- `load_paper_track_config()`
- `load_paper_track_state()`
- `save_paper_track_state()`
- `simulate_order_fills()`
- `run_paper_track_daily()`

---

## 8. CLI Design

### 8.1 Standalone Script

**Neues Script:** `scripts/run_paper_track.py`

**CLI-Parameter:**

```bash
python scripts/run_paper_track.py \
    --config configs/paper_track/strategy_core_ai_tech.yaml \
    --date 2025-01-15 \
    --force
```

**Argumente:**

- `--config`: Pfad zur Config-Datei (required)
- `--date`: Datum fur Run (YYYY-MM-DD, default: heute)
- `--force`: Erlaube Re-Run fur denselben Tag (default: False)
- `--skip-risk-report`: Uberspringe Risk-Report-Generierung (auch wenn Trigger aktiv)
- `--skip-health-check`: Uberspringe Health-Check (auch wenn aktiviert)

### 8.2 CLI Integration

**Integration in `scripts/cli.py` als Subcommand `paper_track`:**

```bash
python scripts/cli.py paper_track \
    --config configs/paper_track/strategy_core_ai_tech.yaml \
    --date 2025-01-15
```

**Subcommand-Handler:**

```python
def paper_track_subcommand(args: argparse.Namespace) -> int:
    """Run paper track daily execution subcommand."""
    from scripts.run_paper_track import run_paper_track_from_args
    
    try:
        return run_paper_track_from_args(args)
    except Exception as e:
        logger.error(f"Paper track failed: {e}", exc_info=True)
        return 1
```

---

## 9. Tests

### 9.1 Unit-Tests

**Neue Test-Datei:** `tests/test_paper_track_runner.py`

**Tests (mindestens 10, `@pytest.mark.advanced`):**

1. `test_load_paper_track_config_valid()`: Lade valide Config
2. `test_load_paper_track_config_invalid()`: Lade invalide Config → Error
3. `test_load_paper_track_state_first_run()`: Erster Run → Neues State
4. `test_load_paper_track_state_existing()`: Lade existierendes State
5. `test_save_paper_track_state_atomic()`: Atomisches Schreiben mit Backup
6. `test_simulate_order_fills_buy()`: Fill-Simulation fur BUY-Order
7. `test_simulate_order_fills_sell()`: Fill-Simulation fur SELL-Order
8. `test_simulate_order_fills_costs()`: Kosten werden korrekt berechnet
9. `test_run_paper_track_daily_first_run()`: Erster Run (kein State)
10. `test_run_paper_track_daily_subsequent_run()`: Nachfolgender Run (State vorhanden)
11. `test_run_paper_track_daily_already_run()`: Re-Run ohne `--force` → Error
12. `test_run_paper_track_daily_pit_violation()`: PIT-Violation → Error (wenn enabled)

### 9.2 Mini E2E-Test (Synthetisch)

**Test:** `test_paper_track_e2e_synthetic()`

**Setup:**

- Erstelle synthetische Preis-Daten (10 Symbole, 20 Tage)
- Erstelle minimale Config (Trend-Strategie)
- Fuhre Paper-Track fur 5 aufeinanderfolgende Tage aus

**Assertions:**

- State wird korrekt persistiert zwischen Runs
- Equity-Kurve wachst/stagniert konsistent
- Positions werden korrekt aktualisiert
- Cash wird korrekt aktualisiert (inkl. Kosten)
- Daily Artifacts werden geschrieben
- Equity-Curve-Aggregation funktioniert

### 9.3 Determinismus (B1)

**Test:** `test_paper_track_deterministic()`

**Setup:**

- Fester Seed fur alle Zufallsoperationen
- Gleiche Config, gleiche Preis-Daten
- Fuhre Paper-Track zweimal aus (separate Prozesse)

**Assertions:**

- Equity-Werte sind identisch (bit-exact)
- Positions sind identisch
- Orders sind identisch
- State-Dateien sind identisch (nach Normalisierung von Timestamps)

---

## 10. Implementation Plan

### 10.1 A5.1: Design & Config + State-Model

**Tasks:**

- [ ] Design-Dokument `docs/PAPER_TRACK_RUNNER_A5_DESIGN.md` erstellen (✅ Completed)
- [ ] `PaperTrackConfig` Dataclass implementieren
- [ ] `PaperTrackState` Dataclass implementieren
- [ ] Config-Loading (`load_paper_track_config()`)
- [ ] State-Loading/Saving (`load_paper_track_state()`, `save_paper_track_state()`)
- [ ] Tests fur Config und State

**Deliverables:**

- Config-Loading funktionsfahig
- State-Persistenz funktionsfahig
- Tests grun

### 10.2 A5.2: Daily Flow + Fill-Simulation

**Tasks:**

- [ ] `simulate_order_fills()` implementieren
- [ ] `run_paper_track_daily()` implementieren (ohne Integration)
- [ ] Daily Artifacts schreiben (equity_snapshot, positions, orders, summary)
- [ ] Equity-Curve-Aggregation
- [ ] Tests fur Daily Flow

**Deliverables:**

- Paper-Track lauft fur einen Tag
- State wird aktualisiert
- Artefakte werden geschrieben
- Tests grun

### 10.3 A5.3: Integration (Risk-Report, Health-Check, PIT)

**Tasks:**

- [ ] Risk-Report-Integration (weekly/monthly trigger)
- [ ] Health-Check-Integration (A3)
- [ ] PIT-Guards-Integration (B2)
- [ ] Regime-Analyse-Integration (B3, optional)
- [ ] Deflated Sharpe-Integration (B4, optional)
- [ ] Tests fur Integrationen

**Deliverables:**

- Optionale Integrationen funktionsfahig
- Tests grun

### 10.4 A5.4: CLI + Tests + Docs

**Tasks:**

- [ ] CLI-Script `scripts/run_paper_track.py`
- [ ] CLI-Integration in `scripts/cli.py`
- [ ] Umfassende Tests (Unit + E2E + Determinismus)
- [ ] Dokumentation:
  - Aktualisiere `docs/PAPER_TRACK_PLAYBOOK.md` (Verweis auf Runner)
  - Aktualisiere `docs/ADVANCED_ANALYTICS_FACTOR_LABS.md` (A5 als Completed)
  - Aktualisiere `README.md` (Quick Link)

**Deliverables:**

- CLI funktionsfahig
- Vollstandige Test-Suite
- Dokumentation aktualisiert

---

## 11. Risks & Limitations

### 11.1 State-Corruption

**Problem:** State-Datei kann bei Crash wahrend Write korrupt werden.

**Mitigation:**

- Atomic Write (temp file → rename)
- Backup vor jedem Write
- Validierung beim Laden (bei Korruption: Restore aus Backup)

### 11.2 Clock-Skew / Date-Probleme

**Problem:** System-Uhr kann falsch sein, oder Run wird mit falschem Datum aufgerufen.

**Mitigation:**

- Validierung: `run_date` muss >= `last_run_date` sein
- Validierung: `run_date` sollte nicht > heute sein (Warning)
- UTC-Timestamps uberall

### 11.3 Preis-Daten fehlen

**Problem:** Fur Run-Datum sind keine Preis-Daten verfugbar.

**Mitigation:**

- Verwende "last available" Preis (wie in `run_daily.py`)
- Wenn kein Preis verfugbar: Skip Symbol oder Error (konfigurierbar)
- Dokumentation: Paper-Track erfordert kontinuierliche Datenverfugbarkeit

### 11.4 Performance (bei vielen Runs)

**Problem:** Equity-Curve-Aggregation kann langsam werden bei vielen Runs.

**Mitigation:**

- Incremental Update (append nur neue Zeile statt Neu-Generierung)
- Optional: Parquet-Format statt CSV (schneller)
- Optional: Caching von aggregierten Daten

### 11.5 Determinismus (B1)

**Problem:** Floating-Point-Rundungsfehler konnen zu nicht-deterministischen Ergebnissen fuhren.

**Mitigation:**

- Fester Seed fur alle Zufallsoperationen
- Gleiche Reihenfolge von Operationen
- Tests fur Determinismus (siehe 9.3)

---

## 12. Success Criteria

### 12.1 Daily Flow funktioniert

**Kriterium:** Paper-Track lauft taglich ohne Fehler.

**Messung:**

- `python scripts/cli.py paper_track --config <config>` lauft erfolgreich
- State wird korrekt aktualisiert
- Daily Artifacts werden geschrieben
- Equity-Curve wird aktualisiert

### 12.2 State-Persistenz funktioniert

**Kriterium:** State wird korrekt zwischen Runs persistiert.

**Messung:**

- State-Datei existiert nach erstem Run
- Nach zweitem Run: Positions und Cash sind konsistent mit ersten Run
- Backup wird erstellt vor jedem Update

### 12.3 Integrationen funktionieren

**Kriterium:** Risk-Reports und Health-Checks werden korrekt generiert.

**Messung:**

- Risk-Reports werden bei Weekly/Monthly-Trigger generiert
- Health-Checks werden taglich ausgefuhrt (wenn enabled)
- PIT-Checks funktionieren (wenn enabled)

### 12.4 Tests grun

**Kriterium:** Alle Tests sind grun.

**Messung:**

- `pytest tests/test_paper_track_runner.py` - alle Tests grun
- E2E-Test lauft erfolgreich
- Determinismus-Test bestatigt bit-exact Reproduzierbarkeit

---

## 13. References

**Design Documents:**

- [Paper Track Playbook](PAPER_TRACK_PLAYBOOK.md) - Prozess-Definition fur Backtest → Paper → Live
- [Operations & Monitoring (A3)](OPERATIONS_BACKEND_A3_DESIGN.md) - Health-Check-Design
- [Point-in-Time & Latency (B2)](POINT_IN_TIME_AND_LATENCY.md) - PIT-Safety-Design
- [Walk-Forward & Regime Analysis (B3)](WALK_FORWARD_AND_REGIME_B3_DESIGN.md) - Regime-Analyse-Design
- [Deflated Sharpe (B4)](DEFLATED_SHARPE_B4_DESIGN.md) - Deflated Sharpe-Design

**Code References:**

- `scripts/run_daily.py` - EOD-MVP Runner (Basis fur Daily Flow)
- `src/assembled_core/qa/backtest_engine.py` - Backtest-Engine (Position-Update, Equity-Simulation)
- `src/assembled_core/pipeline/portfolio.py` - Portfolio-Simulation mit Kosten
- `src/assembled_core/execution/order_generation.py` - Order-Generierung
- `src/assembled_core/qa/point_in_time_checks.py` - PIT-Safety-Checks
- `scripts/generate_risk_report.py` - Risk-Report-Generierung
- `scripts/check_health.py` - Health-Check-Script (A3)

**Workflows:**

- [EOD Pipeline Workflows](WORKFLOWS_EOD_AND_QA.md) - EOD-Pipeline-Workflows
- [Risk Metrics & Attribution Workflows](WORKFLOWS_RISK_METRICS_AND_ATTRIBUTION.md) - Risk-Report-Workflows
- [Operations Backend Runbook](OPERATIONS_BACKEND.md) - Operations-Runbook

---

