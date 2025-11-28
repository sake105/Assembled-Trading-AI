# Backend Modules - Assembled Trading AI

## Übersicht

Dieses Dokument beschreibt alle Kernmodule in `src/assembled_core/` mit ihren Funktionen, Abhängigkeiten und Verwendungszwecken.

**Package-Struktur:**
```
src/assembled_core/
├── __init__.py          # Package-Initialisierung
├── config.py            # Zentrale Konfiguration
├── costs.py             # Cost-Model-Konfiguration
├── ema_config.py        # EMA-Parameter-Konfiguration
├── pipeline/            # Core Trading-Pipeline
├── api/                 # FastAPI Backend
├── qa/                  # QA/Health-Checks
├── data/                # Data Ingestion (Skelett)
├── features/            # Technical Analysis Features (Skelett)
├── signals/             # Signal Generation Framework (Skelett)
├── portfolio/           # Portfolio Management (Skelett)
├── execution/           # Order Execution (Skelett)
└── reports/             # Report Generation (Skelett)
```

---

## Top-Level Module

### `config.py`

**Zweck:** Zentrale Konfiguration für das gesamte Backend.

**Konstanten:**
- `OUTPUT_DIR` - Pfad zum `output/` Verzeichnis
- `SUPPORTED_FREQS` - Unterstützte Frequenzen `("1d", "5min")`

**Funktionen:**
- `get_output_path(*parts: str) -> Path` - Pfad innerhalb OUTPUT_DIR erstellen
- `get_base_dir() -> Path` - Repository-Root-Verzeichnis

**Verwendung:**
```python
from src.assembled_core.config import OUTPUT_DIR, SUPPORTED_FREQS, get_output_path

price_file = get_output_path("aggregates", "5min.parquet")
```

**Abhängigkeiten:** Keine (nur `pathlib`)

---

### `costs.py`

**Zweck:** Cost-Model-Konfiguration für Portfolio-Simulation.

**Dataclass:**
- `CostModel` - `commission_bps`, `spread_w`, `impact_w`

**Funktionen:**
- `get_default_cost_model() -> CostModel` - Standard-Kosten-Parameter

**Verwendung:**
```python
from src.assembled_core.costs import get_default_cost_model

costs = get_default_cost_model()
# Default: commission_bps=0.0, spread_w=0.25, impact_w=0.5
```

**Abhängigkeiten:** Keine (nur `dataclasses`)

---

### `ema_config.py`

**Zweck:** EMA-Parameter-Konfiguration für Signal-Generierung.

**Dataclass:**
- `EmaConfig` - `fast`, `slow` (EMA-Perioden)

**Funktionen:**
- `get_default_ema_config(freq: str) -> EmaConfig` - Frequenz-spezifische Defaults

**Verwendung:**
```python
from src.assembled_core.ema_config import get_default_ema_config

ema = get_default_ema_config("1d")
# Default für 1d: fast=20, slow=60
# Default für 5min: fast=10, slow=30
```

**Abhängigkeiten:** Keine (nur `dataclasses`)

---

## Pipeline Module (`pipeline/`)

### `pipeline.io`

**Zweck:** Daten-I/O für Preise und Orders.

**Funktionen:**
- `load_prices(freq, price_file=None, output_dir=None) -> pd.DataFrame`
- `load_prices_with_fallback(freq, output_dir=None) -> pd.DataFrame`
- `load_orders(freq, output_dir=None, strict=True) -> pd.DataFrame`
- `write_orders(orders, freq, output_dir=None) -> Path`

**Verwendung:**
```python
from src.assembled_core.pipeline.io import load_prices, load_orders

prices = load_prices("1d")
orders = load_orders("1d", strict=True)
```

**Abhängigkeiten:** `pandas`, `pyarrow` (Parquet)

---

### `pipeline.signals`

**Zweck:** Trading-Signal-Generierung (EMA-Crossover).

**Funktionen:**
- `compute_ema_signals(prices, fast, slow) -> pd.DataFrame`

**Output:** DataFrame mit Spalten: `timestamp`, `symbol`, `sig` (-1/0/+1), `price`

**Verwendung:**
```python
from src.assembled_core.pipeline.signals import compute_ema_signals

signals = compute_ema_signals(prices, fast=20, slow=60)
```

**Abhängigkeiten:** `pandas`, `numpy`

---

### `pipeline.orders`

**Zweck:** Order-Generierung aus Signalen.

**Funktionen:**
- `signals_to_orders(signals) -> pd.DataFrame`
- `write_orders(orders, freq, output_dir=None) -> Path`

**Output:** DataFrame mit Spalten: `timestamp`, `symbol`, `side` (BUY/SELL), `qty`, `price`

**Verwendung:**
```python
from src.assembled_core.pipeline.orders import signals_to_orders, write_orders

orders = signals_to_orders(signals)
orders_path = write_orders(orders, "1d")
```

**Abhängigkeiten:** `pandas`

---

### `pipeline.backtest`

**Zweck:** Backtest-Simulation ohne Transaktionskosten.

**Funktionen:**
- `simulate_equity(prices, orders, start_capital) -> pd.DataFrame`
- `compute_metrics(equity) -> dict`
- `write_backtest_report(equity, metrics, freq, output_dir=None) -> tuple[Path, Path]`

**Output:**
- Equity-Kurve: DataFrame mit `timestamp`, `equity`
- Metriken: `final_pf`, `sharpe`, `rows`, `first`, `last`

**Verwendung:**
```python
from src.assembled_core.pipeline.backtest import simulate_equity, compute_metrics

equity = simulate_equity(prices, orders, start_capital=10000.0)
metrics = compute_metrics(equity)
```

**Abhängigkeiten:** `pandas`, `numpy`

---

### `pipeline.portfolio`

**Zweck:** Portfolio-Simulation mit Transaktionskosten.

**Funktionen:**
- `simulate_with_costs(orders, start_capital, commission_bps, spread_w, impact_w, freq) -> tuple[pd.DataFrame, dict]`
- `write_portfolio_report(equity, metrics, freq, output_dir=None) -> tuple[Path, Path]`

**Kostenmodell:**
- Commission (Basis-Punkte)
- Spread (Bid/Ask-Spread)
- Market Impact (Preisabschlag)

**Output:**
- Equity-Kurve: DataFrame mit `timestamp`, `equity`
- Metriken: `final_pf`, `sharpe`, `trades`

**Verwendung:**
```python
from src.assembled_core.pipeline.portfolio import simulate_with_costs

equity, metrics = simulate_with_costs(
    orders, start_capital=10000.0,
    commission_bps=0.0, spread_w=0.25, impact_w=0.5, freq="1d"
)
```

**Abhängigkeiten:** `pandas`, `numpy`

---

### `pipeline.orchestrator`

**Zweck:** EOD-Pipeline-Orchestrierung.

**Funktionen:**
- `run_execute_step(freq, output_dir=None, price_file=None) -> tuple[Path, pd.DataFrame]`
- `run_backtest_step(freq, start_capital, output_dir=None, price_file=None) -> tuple[Path, Path]`
- `run_portfolio_step(freq, start_capital, commission_bps=None, spread_w=None, impact_w=None, output_dir=None) -> tuple[Path, Path]`
- `run_eod_pipeline(freq, start_capital=10000.0, skip_backtest=False, skip_portfolio=False, skip_qa=False, ...) -> dict`

**Verwendung:**
```python
from src.assembled_core.pipeline.orchestrator import run_eod_pipeline

manifest = run_eod_pipeline(freq="1d", start_capital=10000.0)
```

**Abhängigkeiten:** Alle Pipeline-Module, `qa.health`

---

## API Module (`api/`)

### `api.app`

**Zweck:** FastAPI Application Factory.

**Funktionen:**
- `create_app() -> FastAPI`

**Verwendung:**
```python
from src.assembled_core.api.app import create_app

app = create_app()
```

**Abhängigkeiten:** `fastapi`

---

### `api.models`

**Zweck:** Pydantic-Modelle für API-Requests/Responses.

**Modelle:**
- `Signal`, `SignalType`, `SignalsResponse`
- `OrderPreview`, `OrdersResponse`
- `PortfolioSnapshot`
- `EquityPoint`, `EquityCurveResponse`
- `RiskMetrics`
- `QaStatus`, `QaCheck`, `QaStatusEnum`
- `Frequency` (Enum)

**Verwendung:**
```python
from src.assembled_core.api.models import Signal, PortfolioSnapshot

signal = Signal(timestamp=..., symbol="AAPL", signal_type=SignalType.BUY, ...)
```

**Abhängigkeiten:** `pydantic`

---

### `api.routers.*`

**Zweck:** FastAPI-Router für verschiedene Endpoints.

**Routers:**
- `signals.py` - `GET /api/v1/signals/{freq}`, `GET /api/v1/signals/{freq}/latest`
- `orders.py` - `GET /api/v1/orders/{freq}`
- `portfolio.py` - `GET /api/v1/portfolio/{freq}/current`, `GET /api/v1/portfolio/{freq}/equity-curve`
- `performance.py` - `GET /api/v1/performance/{freq}/backtest-curve`, `GET /api/v1/performance/{freq}/metrics`
- `risk.py` - `GET /api/v1/risk/{freq}/summary`
- `qa.py` - `GET /api/v1/qa/status`

**Abhängigkeiten:** `fastapi`, `pipeline.*`, `qa.health`

---

## QA Module (`qa/`)

### `qa.health`

**Zweck:** QA/Health-Checks für Pipeline-Outputs.

**Dataclass:**
- `QaCheckResult` - `name`, `status`, `message`, `details`

**Funktionen:**
- `check_prices(freq, output_dir=None) -> QaCheckResult`
- `check_orders(freq, output_dir=None) -> QaCheckResult`
- `check_portfolio(freq, output_dir=None) -> QaCheckResult`
- `aggregate_qa_status(freq, output_dir=None) -> dict`

**Verwendung:**
```python
from src.assembled_core.qa.health import aggregate_qa_status

qa_result = aggregate_qa_status("1d")
# Returns: {"overall_status": "ok", "checks": [...], ...}
```

**Abhängigkeiten:** `pandas`

---

### `qa.backtest_engine`

**Zweck:** Portfolio-Level-Backtest-Engine für flexible Strategie-Tests.

**Dataclass:**
- `BacktestResult` - `equity`, `metrics`, `trades`, `signals`, `target_positions`

**Funktionen:**
- `run_portfolio_backtest(prices, signal_fn, position_sizing_fn, start_capital=10000.0, ...) -> BacktestResult`

**Eingaben:**
- `prices`: DataFrame mit OHLCV-Daten (timestamp, symbol, close, ...)
- `signal_fn`: Callable[[pd.DataFrame], pd.DataFrame] - Custom Signal-Funktion
- `position_sizing_fn`: Callable[[pd.DataFrame, float], pd.DataFrame] - Custom Position-Sizing-Funktion
- `start_capital`: Startkapital (default: 10000.0)
- Kosten: `commission_bps`, `spread_w`, `impact_w` oder `cost_model: CostModel`
- Flags: `include_costs`, `include_trades`, `include_signals`, `include_targets`
- `compute_features`: Ob TA-Features berechnet werden sollen (default: True)
- `feature_config`: Konfiguration für Feature-Computation

**Ausgaben:**
- `BacktestResult` mit:
  - `equity`: DataFrame (date, timestamp, equity, daily_return)
  - `metrics`: dict (final_pf, sharpe, trades, ...)
  - `trades`: Optional DataFrame (alle Trades, wenn `include_trades=True`)
  - `signals`: Optional DataFrame (alle Signale, wenn `include_signals=True`)
  - `target_positions`: Optional DataFrame (Zielpositionen, wenn `include_targets=True`)

**Workflow:**
1. Feature-Computation (optional, via `features.ta_features.add_all_features`)
2. Signal-Generierung (via `signal_fn`)
3. Position-Sizing (via `position_sizing_fn`, gruppiert nach timestamp)
4. Order-Generierung (via `execution.order_generation.generate_orders_from_targets`)
5. Equity-Simulation:
   - Mit Kosten: `pipeline.portfolio.simulate_with_costs`
   - Ohne Kosten: `pipeline.backtest.simulate_equity`
6. Equity-Enhancement (date, daily_return hinzufügen)
7. Performance-Metriken (via `pipeline.backtest.compute_metrics`)

**Verwendung:**
```python
from src.assembled_core.qa.backtest_engine import run_portfolio_backtest
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices
from src.assembled_core.portfolio.position_sizing import compute_target_positions

def signal_fn(prices_df):
    return generate_trend_signals_from_prices(prices_df, ma_fast=20, ma_slow=50)

def sizing_fn(signals_df, capital):
    return compute_target_positions(signals_df, total_capital=capital, equal_weight=True)

result = run_portfolio_backtest(
    prices=prices,
    signal_fn=signal_fn,
    position_sizing_fn=sizing_fn,
    start_capital=10000.0,
    include_costs=True,
    include_trades=True
)

print(f"Final PF: {result.metrics['final_pf']:.4f}")
print(f"Sharpe: {result.metrics['sharpe']:.4f}")
print(f"Trades: {result.metrics['trades']}")
```

**Abhängigkeiten:**
- `features.ta_features` (für Feature-Computation)
- `execution.order_generation` (für Order-Generierung)
- `pipeline.backtest` (für kostenfreie Simulation)
- `pipeline.portfolio` (für kostenbewusste Simulation)
- `costs.CostModel` (für Kosten-Parameter)

**Vorteile:**
- Flexibel: Custom Signal- und Sizing-Funktionen als Callables
- Komposabel: Nutzt bestehende Module (keine Duplikation)
- Vollständig: Equity + Metriken + optionale Details (Trades, Signale, Targets)
- Testbar: Kann mit synthetischen Daten getestet werden

---

### `qa.metrics`

**Zweck:** Zentrale Berechnung von Performance- und Risk-Metriken.

**Dataclass:**
- `PerformanceMetrics` - Umfassende Metriken-Dataclass mit Returns, Risk-Adjusted Returns, Risk Metrics, Trade Metrics, Metadata

**Funktionen:**
- `compute_all_metrics(equity, trades=None, start_capital=10000.0, freq="1d", risk_free_rate=0.0) -> PerformanceMetrics`
- `compute_equity_metrics(equity, start_capital, freq, risk_free_rate) -> PerformanceMetrics`
- `compute_trade_metrics(trades, equity, start_capital, freq) -> PerformanceMetrics`
- `compute_sharpe_ratio(returns, freq, risk_free_rate=0.0) -> float | None`
- `compute_sortino_ratio(returns, freq, risk_free_rate=0.0) -> float | None`
- `compute_drawdown(equity) -> tuple[pd.Series, float, float]`
- `compute_cagr(equity, start_capital, freq) -> float | None`
- `compute_turnover(trades, equity, start_capital, freq) -> float | None`

**Metriken:**
- **Returns:** Final PF, Total Return, CAGR
- **Risk-Adjusted:** Sharpe Ratio, Sortino Ratio, Calmar Ratio
- **Risk:** Max Drawdown, Current Drawdown, Volatility, VaR (95%)
- **Trade Metrics:** Hit Rate, Profit Factor, Avg Win/Loss, Turnover

**Verwendung:**
```python
from src.assembled_core.qa.metrics import compute_all_metrics

metrics = compute_all_metrics(
    equity=equity_df,
    trades=trades_df,  # Optional
    start_capital=10000.0,
    freq="1d",
    risk_free_rate=0.0
)

print(f"Sharpe: {metrics.sharpe_ratio:.4f}")
print(f"CAGR: {metrics.cagr:.2%}")
print(f"MaxDD: {metrics.max_drawdown_pct:.2f}%")
```

**Abhängigkeiten:** `pandas`, `numpy`

**Integration:**
- Wird von `qa.backtest_engine` genutzt (via `pipeline.backtest.compute_metrics` als Fallback)
- Wird von `qa.walk_forward` genutzt (pro Fenster IS/OOS-Metriken)
- Wird von `qa.qa_gates` genutzt (als Input für Gate-Evaluierung)
- Wird von `reports.daily_qa_report` genutzt (für Report-Generierung)

---

### `qa.qa_gates`

**Zweck:** Qualitäts-Gates zur automatisierten Strategie-Evaluierung.

**Enum:**
- `QAResult` - `OK`, `WARNING`, `BLOCK`

**Dataclasses:**
- `QAGateResult` - `gate_name`, `result`, `reason`, `details`
- `QAGatesSummary` - `overall_result`, `passed_gates`, `warning_gates`, `blocked_gates`, `gate_results`

**Funktionen:**
- `evaluate_all_gates(metrics, config=None) -> QAGatesSummary`
- `check_sharpe_ratio(metrics, min_sharpe=1.0, warning_sharpe=0.5) -> QAGateResult`
- `check_max_drawdown(metrics, max_dd_pct_limit=-20.0, warning_dd_pct=-15.0) -> QAGateResult`
- `check_turnover(metrics, max_turnover=10.0, warning_turnover=5.0) -> QAGateResult`
- `check_cagr(metrics, min_cagr=0.05, warning_cagr=0.0) -> QAGateResult`
- `check_volatility(metrics, max_volatility=0.30, warning_volatility=0.25) -> QAGateResult`
- `check_hit_rate(metrics, min_hit_rate=0.50, warning_hit_rate=0.40) -> QAGateResult`
- `check_profit_factor(metrics, min_profit_factor=1.5, warning_profit_factor=1.2) -> QAGateResult`

**Gate-Logik:**
- **OK:** Metrik erfüllt Minimum-Schwellenwert
- **WARNING:** Metrik zwischen Warning- und Minimum-Schwellenwert
- **BLOCK:** Metrik unter Warning-Schwellenwert (oder kritischer Fehler)

**Overall Status:**
- **OK:** Alle Gates OK
- **WARNING:** Mindestens ein Gate WARNING, keine BLOCK
- **BLOCK:** Mindestens ein Gate BLOCK

**Verwendung:**
```python
from src.assembled_core.qa.qa_gates import evaluate_all_gates
from src.assembled_core.qa.metrics import compute_all_metrics

metrics = compute_all_metrics(equity_df, trades_df, start_capital=10000.0, freq="1d")
gate_result = evaluate_all_gates(metrics)

print(f"Overall Status: {gate_result.overall_result.value}")
print(f"Passed: {gate_result.passed_gates}, Warnings: {gate_result.warning_gates}, Blocked: {gate_result.blocked_gates}")
```

**Abhängigkeiten:** `qa.metrics`

**Integration:**
- Wird von `pipeline.orchestrator` genutzt (in `run_eod_pipeline`, Step 5c)
- Wird von `reports.daily_qa_report` genutzt (für Report-Generierung)
- Wird von `api.routers.qa` genutzt (für API-Endpoint `/api/v1/qa/status`)

---

### `qa.walk_forward`

**Zweck:** Walk-Forward-Analyse für robuste Strategie-Validierung.

**Dataclasses:**
- `WalkForwardConfig` - `train_size`, `test_size`, `step_size`, `window_type`, `embargo_periods`, `purge_periods`, etc.
- `WalkForwardWindow` - `train_start`, `train_end`, `test_start`, `test_end`, `train_data`, `test_data`, `window_index`
- `WalkForwardWindowResult` - `window_index`, `train_start`, `train_end`, `test_start`, `test_end`, `is_metrics`, `oos_metrics`, `backtest_result`
- `WalkForwardResult` - `config`, `windows`, `window_results`, `summary_metrics`

**Funktionen:**
- `run_walk_forward_backtest(prices, signal_fn, position_sizing_fn, config, ...) -> WalkForwardResult`
- `_split_time_series(data, config, freq) -> list[WalkForwardWindow]`
- `_aggregate_walk_forward_metrics(window_results) -> dict`

**Workflow:**
1. Split Zeitreihe in Train/Test-Fenster (rolling oder expanding)
2. Pro Fenster:
   - Backtest auf Test-Fenster (OOS - Out-of-Sample)
   - OOS-Metriken berechnen (via `qa.metrics.compute_all_metrics`)
   - Optional: Backtest auf Train-Fenster (IS - In-Sample)
   - Optional: IS-Metriken berechnen
3. Aggregation über alle Fenster (Mean, Std, Win Rate)

**Summary Metrics:**
- **IS (In-Sample):** `is_mean_final_pf`, `is_mean_sharpe`, `is_mean_cagr`
- **OOS (Out-of-Sample):** `oos_mean_final_pf`, `oos_std_final_pf`, `oos_mean_sharpe`, `oos_std_sharpe`, `oos_mean_cagr`, `oos_std_cagr`, `oos_mean_max_drawdown_pct`, `oos_win_rate`
- **Totals:** `total_periods`, `total_trades`, `num_windows`

**Verwendung:**
```python
from src.assembled_core.qa.walk_forward import WalkForwardConfig, run_walk_forward_backtest
from src.assembled_core.signals.rules_trend import generate_trend_signals_from_prices
from src.assembled_core.portfolio.position_sizing import compute_target_positions

config = WalkForwardConfig(
    train_size=252,  # 1 Jahr
    test_size=63,    # 1 Quartal
    step_size=21,    # 1 Monat
    window_type="rolling"
)

result = run_walk_forward_backtest(
    prices=price_df,
    signal_fn=generate_trend_signals_from_prices,
    position_sizing_fn=compute_target_positions,
    config=config,
    start_capital=10000.0,
    freq="1d"
)

print(f"OOS Mean Sharpe: {result.summary_metrics['oos_mean_sharpe']:.2f}")
print(f"OOS Win Rate: {result.summary_metrics['oos_win_rate']:.1%}")
```

**Abhängigkeiten:**
- `qa.backtest_engine` (für Backtest pro Fenster)
- `qa.metrics` (für Metriken-Berechnung pro Fenster)

**Vorteile:**
- Robustheit: Vermeidet Overfitting durch Out-of-Sample-Testing
- Datenleck-Vermeidung: Embargo & Purging zwischen Train/Test
- Flexibel: Rolling & Expanding Windows
- Detailliert: IS/OOS-Metriken pro Fenster + Aggregation

---

### `reports.daily_qa_report`

**Zweck:** Generierung von QA-Reports in Markdown-Format.

**Funktionen:**
- `generate_qa_report(metrics, gate_result, strategy_name, freq, ...) -> Path`
- `generate_qa_report_from_files(freq, strategy_name, equity_file=None, ...) -> Path`
- `_build_report_content(...) -> str`

**Report-Inhalt:**
1. **Header:** Strategy-Name, Frequenz, Generierungszeit
2. **Performance Metrics:** Returns, Risk-Adjusted Returns, Risk Metrics, Trade Metrics, Period Information
3. **QA Gates:** Overall Status (mit Emoji), Gate Counts, Gate Details (Tabelle)
4. **Equity Curve:** Link/Pfad zur Equity-Kurve CSV, Visualisierungs-Hinweis
5. **Data Status:** Data Start/End Date, Range, Frequency
6. **Configuration:** Optionale Config-Info (EMA-Params, Cost Model, etc.)

**Output:**
- Dateiname: `qa_report_{strategy}_{freq}_{date}.md`
- Speicherort: `output/reports/`

**Verwendung:**
```python
from src.assembled_core.reports.daily_qa_report import generate_qa_report_from_files

report_path = generate_qa_report_from_files(
    freq="1d",
    strategy_name="ema_trend",
    config_info={"ema_fast": 20, "ema_slow": 60, "commission_bps": 0.5}
)
# → output/reports/qa_report_ema_trend_1d_20250115.md
```

**Abhängigkeiten:**
- `qa.metrics` (für Metriken-Berechnung, wenn `generate_qa_report_from_files` genutzt wird)
- `qa.qa_gates` (für Gate-Evaluierung, wenn `generate_qa_report_from_files` genutzt wird)

**Integration:**
- Kann direkt mit `PerformanceMetrics` + `QAGatesSummary` genutzt werden
- Oder via Convenience-Funktion `generate_qa_report_from_files` (lädt Equity/Trades, berechnet Metriken, evaluiert Gates, generiert Report)

---

## Zukünftige Module (Skelett vorhanden)

### `data/`

**Zweck:** Data Ingestion (Multi-Source-Support).

**Platzhalter:**
- `data/prices_ingest.py` - EOD-Preis- und Volumendaten laden, validieren

**Zukünftige Integration:**
- Nutzt `pipeline.io.load_prices` für Basis-I/O
- Erweitert um Multi-Source-Support (Yahoo, Alpha Vantage, lokale Dateien)

---

### `features/`

**Zweck:** Technical Analysis Features.

**Implementiert:**
- `features/ta_features.py` - Technische Indikatoren
  - `add_log_returns()`: Logarithmische Returns
  - `add_moving_averages()`: Simple Moving Averages (SMA) für mehrere Fenster
  - `add_atr()`: Average True Range (Volatilitätsindikator)
  - `add_rsi()`: Relative Strength Index (Momentum-Indikator)
  - `add_all_features()`: Convenience-Funktion für alle Features

**Zukünftige Integration:**
- Erweitert um weitere TA-Indikatoren (MACD, Bollinger Bands, etc.)
- ML-Feature-Engineering-Pipeline

---

### `signals/`

**Zweck:** Signal-Generation-Framework.

**Implementiert:**
- `signals/rules_trend.py` - Trend-Following-Signale
  - `generate_trend_signals()`: Generiert LONG/FLAT Signale basierend auf MA-Crossover
  - `generate_trend_signals_from_prices()`: Convenience-Funktion für direkte Signal-Generierung
  - Signal-Logik: LONG wenn ma_fast > ma_slow AND Volumen über Schwellenwert
  - Score: Signal-Stärke (0.0 bis 1.0) basierend auf MA-Spread und Volumen

**Zukünftige Integration:**
- Erweitert um weitere Trend-Regeln (EMA-Crossover, etc.)
- Mean-Reversion-Strategien

---

### `portfolio/`

**Zweck:** Portfolio-Management.

**Implementiert:**
- `portfolio/position_sizing.py` - Position-Sizing-Strategien
  - `compute_target_positions()`: Berechnet Zielpositionen aus Signalen
  - `compute_target_positions_from_trend_signals()`: Convenience-Funktion für Trend-Signale
  - Strategien: Equal Weight (1/N) oder Score-basiert
  - Top-N Selektion: Wählt beste Signale nach Score
  - Output: DataFrame mit symbol, target_weight, target_qty

**Zukünftige Integration:**
- Erweitert um weitere Sizing-Strategien (Kelly Criterion, Risk Parity, etc.)
- Integration mit Risk-Management

---

### `execution/`

**Zweck:** Order-Execution-Logik.

**Implementiert:**
- `execution/order_generation.py` - Orders aus Signalen/Positionen generieren
  - `generate_orders_from_targets()`: Vergleicht aktuelle vs. Zielpositionen
  - `generate_orders_from_signals()`: Convenience: Signale → Orders
  - Output: DataFrame mit timestamp, symbol, side, qty, price
- `execution/safe_bridge.py` - SAFE-Bridge Order-Dateien
  - `write_safe_orders_csv()`: Erstellt SAFE-kompatible CSV-Dateien
  - Format: `orders_YYYYMMDD.csv` mit Spalten: Ticker, Side, Quantity, PriceType, Comment
  - Human-in-the-Loop: Alle Orders müssen manuell geprüft werden

**Zukünftige Integration:**
- Erweitert um verschiedene Order-Typen (Market, Limit, Stop-Loss)
- Integration mit Risk-Management und Position-Limits

---

### `reports/`

**Zweck:** Report-Generierung.

**Platzhalter:**
- `reports/daily_qa_report.py` - Tägliche QA-Reports

**Zukünftige Integration:**
- Nutzt `qa.health.aggregate_qa_status` für Health-Checks
- Kombiniert Pipeline-Metriken

---

## Weiterführende Dokumentation

- [Backend Architecture](ARCHITECTURE_BACKEND.md) - Gesamtarchitektur
- [Backend API](backend_api.md) - FastAPI-Endpoints
- [Backend Core](backend_core.md) - Konfiguration & Testing
- [EOD Pipeline](eod_pipeline.md) - Pipeline-Orchestrierung

