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

**Platzhalter:**
- `features/ta_features.py` - Technische Indikatoren (EMA, SMA, RSI, MACD, etc.)

**Zukünftige Integration:**
- Nutzt `pipeline.signals.compute_ema_signals` als Basis
- Erweitert um weitere TA-Indikatoren

---

### `signals/`

**Zweck:** Signal-Generation-Framework.

**Platzhalter:**
- `signals/rules_trend.py` - Trend-Following-Signale

**Zukünftige Integration:**
- Nutzt `pipeline.signals.compute_ema_signals` als Basis
- Erweitert um weitere Signal-Regeln

---

### `portfolio/`

**Zweck:** Portfolio-Management.

**Platzhalter:**
- `portfolio/position_sizing.py` - Position-Sizing-Strategien

**Zukünftige Integration:**
- Nutzt `pipeline.portfolio.simulate_with_costs` für Backtesting
- Erweitert um Position-Sizing-Logik

---

### `execution/`

**Zweck:** Order-Execution-Logik.

**Platzhalter:**
- `execution/order_generation.py` - Orders aus Signalen generieren

**Zukünftige Integration:**
- Nutzt `pipeline.orders.signals_to_orders` als Basis
- Erweitert um Position-Sizing und Risk-Management

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

