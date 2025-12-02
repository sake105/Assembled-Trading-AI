# Phase 10 – Paper-Trading, Live-Bridge & OMS-Light

## Übersicht

Phase 10 führt Pre-Trade-Kontrollen, Kill-Switch-Mechanismen und eine Paper-Trading-Bridge ein. Ziel ist es, einen sicheren, kontrollierten Weg von Backtest-Signalen zu Live-Order-Generierung zu schaffen, bevor ein vollständiges Order Management System (OMS) implementiert wird.

**Komponenten:**
- **Pre-Trade-Kontrollen**: Risikolimits und Validierungen vor Order-Ausführung
- **Kill-Switch**: Notfall-Mechanismus zum sofortigen Blockieren aller Orders
- **Paper-Trading-API** (Sprint 10.2): API-Endpoints für Paper-Trading-Orders
- **OMS-Light** (Sprint 10.3): Minimales Order Management System für Live-Trading

---

## Sprint 10.1 – Pre-Trade-Kontrollen & Kill-Switch

### Ziele

Sprint 10.1 implementiert eine robuste Risiko-Kontrollebene, die alle generierten Orders vor der Ausführung validiert. Dies stellt sicher, dass:

- Position-Größen innerhalb definierter Limits bleiben
- Gesamt-Exposure nicht überschritten wird
- QA-Gates respektiert werden
- Notfall-Blockierung aller Orders möglich ist

### Neue Module

#### 1. `src/assembled_core/execution/pre_trade_checks.py`

**Zweck:** Validiert Orders gegen Risikolimits und QA-Status.

**Hauptkomponenten:**
- `PreTradeConfig` (Dataclass): Konfiguration für Pre-Trade-Limits
  - `max_notional_per_symbol`: Maximales Notional pro Symbol
  - `max_weight_per_symbol`: Maximales Gewicht pro Symbol
  - `max_gross_exposure`: Maximale Gesamt-Exposure
  - `max_sector_exposure`: Optional, Sektor-spezifische Limits
  - `max_region_exposure`: Optional, Region-spezifische Limits
- `PreTradeCheckResult` (Dataclass): Ergebnis der Pre-Trade-Validierung
  - `is_ok`: True wenn alle Checks bestehen
  - `blocked_reasons`: Liste von Gründen für Blockierung
  - `filtered_orders`: Gefilterte Orders (nur die, die passieren)
  - `summary`: Optional, zusätzliche Metriken
- `run_pre_trade_checks()`: Hauptfunktion für Pre-Trade-Validierung

**Wichtigste Checks:**
1. **Max Notional per Symbol**: Prüft, ob einzelne Orders das maximale Notional pro Symbol überschreiten
2. **Max Weight per Symbol**: Prüft, ob einzelne Orders das maximale Gewicht pro Symbol überschreiten
3. **Max Gross Exposure**: Prüft, ob die resultierende Gesamt-Exposure (Summe aller Long + Short Positions) das Limit überschreitet
4. **QA Block**: Prüft, ob `qa_status.can_trade_today == False` → blockiert alle Orders mit Reason "QA_BLOCK"
5. **VolTarget-Platzhalter**: Vorbereitet für zukünftige Volatility-Targeting-Checks

**Beispiel:**
```python
from src.assembled_core.execution.pre_trade_checks import (
    PreTradeConfig,
    run_pre_trade_checks
)
import pandas as pd

orders = pd.DataFrame({
    "symbol": ["AAPL", "GOOGL"],
    "side": ["BUY", "BUY"],
    "qty": [100, 50],
    "price": [150.0, 2500.0]
})

config = PreTradeConfig(
    max_notional_per_symbol=50000.0,
    max_gross_exposure=100000.0
)

result, filtered = run_pre_trade_checks(
    orders,
    portfolio=None,
    qa_status=None,
    config=config
)

if not result.is_ok:
    print(f"Orders blocked: {result.blocked_reasons}")
```

#### 2. `src/assembled_core/execution/kill_switch.py`

**Zweck:** Notfall-Mechanismus zum sofortigen Blockieren aller Orders.

**Hauptkomponenten:**
- `is_kill_switch_engaged()`: Prüft, ob Kill-Switch aktiv ist (via Environment Variable)
- `guard_orders_with_kill_switch()`: Blockiert alle Orders, wenn Kill-Switch aktiv ist

**Kill-Switch-Mechanismus:**
- **Environment Variable**: `ASSEMBLED_KILL_SWITCH`
- **Aktivierung**: Setze `ASSEMBLED_KILL_SWITCH=1` (oder "true", "yes", "on")
- **Verhalten**: Wenn aktiv, werden **alle** Orders blockiert, unabhängig von Pre-Trade-Checks
- **Logging**: Klare Warnung wird geloggt, wenn Kill-Switch aktiv ist

**Beispiel:**
```python
import os
from src.assembled_core.execution.kill_switch import (
    is_kill_switch_engaged,
    guard_orders_with_kill_switch
)

# Setze Kill-Switch (z.B. in Production bei Notfall)
os.environ["ASSEMBLED_KILL_SWITCH"] = "1"

# Prüfe Status
if is_kill_switch_engaged():
    print("Kill switch is active - all orders will be blocked")

# Wende Kill-Switch auf Orders an
orders = pd.DataFrame({...})  # Ihre Orders
filtered = guard_orders_with_kill_switch(orders)
# filtered ist jetzt leer, wenn Kill-Switch aktiv ist
```

#### 3. `src/assembled_core/execution/risk_controls.py`

**Zweck:** Zentrale Integration von Pre-Trade-Checks und Kill-Switch.

**Hauptkomponenten:**
- `RiskControlResult` (Dataclass): Aggregiertes Ergebnis aller Risk Controls
- `filter_orders_with_risk_controls()`: Zentrale Funktion, die beide Checks kombiniert

**Flow:**
1. Pre-Trade-Checks werden zuerst angewendet
2. Kill-Switch wird danach angewendet (hat höchste Priorität)
3. Gefilterte Orders werden zurückgegeben
4. Detailliertes Logging für geblockte Orders

**Beispiel:**
```python
from src.assembled_core.execution.risk_controls import filter_orders_with_risk_controls

filtered_orders, result = filter_orders_with_risk_controls(
    orders,
    portfolio=None,
    qa_status=None,
    pre_trade_config=config,
    enable_pre_trade_checks=True,
    enable_kill_switch=True
)

print(f"Orders before: {result.total_orders_before}")
print(f"Orders after: {result.total_orders_after}")
print(f"Kill switch engaged: {result.kill_switch_engaged}")
```

### Integration in Order-Flow

**Typischer Flow:**
```
Backtest/Signals
    ↓
Position Sizing (compute_target_positions)
    ↓
Order Generation (generate_orders_from_targets)
    ↓
Pre-Trade-Checks (run_pre_trade_checks)
    ↓
Kill-Switch (guard_orders_with_kill_switch)
    ↓
SAFE-CSV / API Output
```

**Integration in `scripts/run_daily.py`:**
- Risk Controls werden direkt nach Order-Generierung angewendet
- CLI-Argumente:
  - `--disable-pre-trade-checks`: Deaktiviert Pre-Trade-Checks (nur für Debug/Backtest)
  - `--ignore-kill-switch`: Ignoriert Kill-Switch (nur für Offline-Backtests/Dev)
- **Default**: Beide Checks sind aktiv (sicher)

**Beispiel-Usage:**
```bash
# Standard (mit Risk Controls)
python scripts/run_daily.py --date 2024-01-15

# Ohne Pre-Trade-Checks (nur für Debug)
python scripts/run_daily.py --date 2024-01-15 --disable-pre-trade-checks

# Ohne Kill-Switch (nur für Offline-Backtest)
python scripts/run_daily.py --date 2024-01-15 --ignore-kill-switch
```

### Tests

**Test-Dateien:**
- `tests/test_execution_pre_trade_checks.py` (13 Tests)
- `tests/test_execution_kill_switch.py` (8 Tests)
- `tests/test_execution_pre_trade_integration.py` (5 Tests)

**Marker:** `@pytest.mark.phase10`

**Testlauf:**
```bash
# Nur Phase 10
pytest -m "phase10" -q

# Alle Phasen inkl. Phase 10
pytest -m "phase4 or phase6 or phase7 or phase8 or phase9 or phase10" -W error
```

---

## Sprint 10.2 – Paper-Trading-API

**Status:** ✅ Fertig

### Ziele

Sprint 10.2 implementiert eine vollständige REST-API für Paper-Trading mit in-memory Engine. Die API ermöglicht:
- Order-Submission über REST-Endpoints
- Order-Status-Tracking (FILLED, REJECTED)
- Position-Tracking und Portfolio-Simulation
- Integration mit Pre-Trade-Checks und Kill-Switch (gleiche Risk Controls wie reguläre Pipeline)

### Neue Module

#### 1. `src/assembled_core/execution/paper_trading_engine.py`

**Zweck:** In-memory Paper-Trading-Engine für Order-Simulation.

**Hauptkomponenten:**
- `PaperOrder` (Dataclass): Order-Repräsentation mit Status-Tracking
- `PaperPosition` (Dataclass): Position-Repräsentation
- `PaperTradingEngine` (Klasse): Zentrale Engine für Order-Verwaltung
  - `submit_orders()`: Orders einreichen und sofort ausführen (FILLED)
  - `list_orders()`: Orders auflisten (mit Limit)
  - `get_positions()`: Aktuelle Positionen abrufen
  - `reset()`: Engine zurücksetzen (für Tests)

**Features:**
- Alle Orders werden sofort als "FILLED" markiert
- Positionen werden automatisch aggregiert (BUY = +, SELL = -)
- Kein File-I/O, kein Netzwerk – alles in Memory
- Thread-safe für Multi-User-Szenarien (über FastAPI-App-State)

#### 2. `src/assembled_core/api/routers/paper_trading.py`

**Zweck:** FastAPI-Router für Paper-Trading-Endpoints.

**Endpoints:**

##### POST `/api/v1/paper/orders`
- **Zweck:** Orders einreichen
- **Body:** `PaperOrderRequest` (einzelne Order) oder `list[PaperOrderRequest]`
- **Response:** `list[PaperOrderResponse]`
- **Status:**
  - `FILLED`: Order hat Risk Controls bestanden und wurde ausgeführt
  - `REJECTED`: Order wurde blockiert (Reason enthält Details)

##### GET `/api/v1/paper/orders`
- **Zweck:** Orders auflisten
- **Query-Parameter:** `limit: int | None = 50`
- **Response:** `list[PaperOrderResponse]` (neueste zuerst)

##### GET `/api/v1/paper/positions`
- **Zweck:** Aktuelle Positionen abrufen
- **Response:** `list[PaperPosition]` (nur non-zero Positionen)

##### POST `/api/v1/paper/reset`
- **Zweck:** Engine zurücksetzen (für Tests/Dev)
- **Response:** `PaperResetResponse`

**Risk Controls Integration:**
- Standard: Risk Controls aktiv (`_ENABLE_RISK_CONTROLS = True`)
- Pre-Trade-Checks werden automatisch angewendet
- Kill-Switch wird geprüft (höchste Priorität)
- Portfolio-Snapshot wird automatisch aus aktuellen Positionen erstellt

### Pydantic-Modelle

Erweitert in `src/assembled_core/api/models.py`:

- `PaperOrderRequest`: Request-Modell
  - `symbol: str`
  - `side: OrderSide` (BUY/SELL)
  - `quantity: float` (> 0)
  - `price: float | None` (optional)
  - `client_order_id: str | None` (optional)
- `PaperOrderResponse`: Response-Modell
  - `order_id: str`
  - `symbol, side, quantity, price`
  - `status: Literal["NEW", "FILLED", "REJECTED"]`
  - `reason: str | None`
- `PaperPosition`: Position-Modell
  - `symbol: str`
  - `quantity: float` (positiv = long, negativ = short)
- `PaperResetResponse`: Reset-Bestätigung

### Beispiel-Requests

#### Order-Submission (POST /api/v1/paper/orders)

**Request:**
```json
[
  {
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 10.0,
    "price": 150.0,
    "client_order_id": "client-order-123"
  },
  {
    "symbol": "GOOGL",
    "side": "SELL",
    "quantity": 5.0,
    "price": 2500.0
  }
]
```

**Response (200 OK):**
```json
[
  {
    "order_id": "uuid-abc-123",
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 10.0,
    "price": 150.0,
    "status": "FILLED",
    "reason": null,
    "client_order_id": "client-order-123"
  },
  {
    "order_id": "uuid-def-456",
    "symbol": "GOOGL",
    "side": "SELL",
    "quantity": 5.0,
    "price": 2500.0,
    "status": "FILLED",
    "reason": null,
    "client_order_id": null
  }
]
```

**Response bei Kill-Switch aktiv:**
```json
[
  {
    "order_id": "uuid-abc-123",
    "symbol": "AAPL",
    "side": "BUY",
    "quantity": 10.0,
    "price": 150.0,
    "status": "REJECTED",
    "reason": "KILL_SWITCH: Kill switch is engaged",
    "client_order_id": "client-order-123"
  }
]
```

#### Positions-Abfrage (GET /api/v1/paper/positions)

**Response (200 OK):**
```json
[
  {
    "symbol": "AAPL",
    "quantity": 6.0
  },
  {
    "symbol": "GOOGL",
    "quantity": -5.0
  }
]
```

### Python-Client-Beispiel

```python
import requests

BASE_URL = "http://localhost:8000/api/v1/paper"

# Reset Engine
response = requests.post(f"{BASE_URL}/reset")
print(response.json())  # {"status": "ok"}

# Submit Orders
orders = [
    {
        "symbol": "AAPL",
        "side": "BUY",
        "quantity": 10.0,
        "price": 150.0
    },
    {
        "symbol": "AAPL",
        "side": "SELL",
        "quantity": 4.0,
        "price": 151.0
    }
]
response = requests.post(f"{BASE_URL}/orders", json=orders)
filled_orders = response.json()

# Check Positions
response = requests.get(f"{BASE_URL}/positions")
positions = response.json()
# [{"symbol": "AAPL", "quantity": 6.0}]
```

### Integration mit Risk Controls

Die Paper-Trading-API nutzt **exakt die gleichen Risk Controls** wie die reguläre Order-Pipeline:

1. **Pre-Trade-Checks:**
   - Position-Größen-Limits (`max_notional_per_symbol`)
   - Gross Exposure Limits (`max_gross_exposure`)
   - Portfolio-Snapshot wird automatisch aus aktuellen Positionen erstellt

2. **Kill-Switch:**
   - Wird automatisch geprüft (höchste Priorität)
   - Wenn aktiv (`ASSEMBLED_KILL_SWITCH=1`), werden alle Orders blockiert

3. **Order-Flow:**
   ```
   Order-Request (PaperOrderRequest)
       ↓
   Risk Controls (Pre-Trade + Kill-Switch)
       ↓
   Gefilterte Orders
       ↓
   Engine-Submission (nur FILLED Orders)
       ↓
   Position-Update
   ```

### Tests

**Test-Datei:** `tests/test_api_paper_trading.py` (10 Tests, alle mit `@pytest.mark.phase10`)

**Test-Kategorien:**
- Basic API-Tests (Order-Submission, Listing, Positions)
- Risk-Control-Integration:
  - Kill-Switch-Tests
  - Pre-Trade-Limits-Tests
  - Position-Update-Verification (nur FILLED Orders)

**Testlauf:**
```bash
# Nur Paper-Trading-API-Tests
pytest -m "phase10" tests/test_api_paper_trading.py -q

# Alle Phase-10-Tests (inkl. Pre-Trade & Kill-Switch)
pytest -m "phase10" -q
```

---

## Sprint 10.3 – OMS-Light (Platzhalter)

**Status:** Geplant

**Ziele:**
- Minimales Order Management System
- Order-Status-Tracking (PENDING, FILLED, REJECTED, CANCELLED)
- Order-History und Audit-Log
- Integration mit Live-Broker-APIs (später)

**Geplante Komponenten:**
- `src/assembled_core/oms/order_manager.py`: Zentrale Order-Verwaltung
- `src/assembled_core/oms/order_status.py`: Order-Status-Tracking
- `src/assembled_core/oms/audit_log.py`: Audit-Log für Compliance

---

## Nächste Schritte

1. **Sprint 10.2**: Paper-Trading-API implementieren
2. **Sprint 10.3**: OMS-Light aufbauen
3. **Integration**: Pre-Trade-Checks in alle Order-Flows integrieren (auch `run_eod_pipeline.py`)
4. **Monitoring**: Metriken zu geblockten Orders tracken
5. **Configuration**: Pre-Trade-Config aus YAML/JSON laden

---

## Referenzen

- **Pre-Trade-Checks Module**: `src/assembled_core/execution/pre_trade_checks.py`
- **Kill-Switch Module**: `src/assembled_core/execution/kill_switch.py`
- **Risk Controls Integration**: `src/assembled_core/execution/risk_controls.py`
- **Pre-Trade-Checks Tests**: `tests/test_execution_pre_trade_checks.py`
- **Kill-Switch Tests**: `tests/test_execution_kill_switch.py`
- **Integration Tests**: `tests/test_execution_pre_trade_integration.py`

