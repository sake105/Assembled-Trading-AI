# Data Contracts - Assembled Trading AI

**Status:** Verbindlich (muss bei Code-Änderungen befolgt werden)
**Letzte Aktualisierung:** 2025-01-04

## Zweck

Dieses Dokument definiert die **I/O-Verträge** (Data Contracts) für alle kritischen Datenstrukturen in der Trading-Pipeline. Diese Contracts sind **verbindlich** und müssen von allen Modulen eingehalten werden, um Kompatibilität und Korrektheit zu gewährleisten.

**Wichtig:** Alle Contracts sind durch Tests abgesichert (`tests/test_contracts_core_io.py`).

---

## 1. Preise/Panel Contract

### Zweck
Rohdaten für Preis-Panels (1d oder 5min aggregiert).

### Required Columns

| Spalte | Typ | Beschreibung | Constraints |
|--------|-----|--------------|-------------|
| `timestamp` | `pd.Timestamp` (UTC, tz-aware) | Zeitstempel | Muss UTC sein, keine NaNs |
| `symbol` | `string` | Ticker-Symbol | Keine NaNs |
| `close` | `float64` | Schlusskurs | Keine NaNs, > 0 |

### Optional Columns

| Spalte | Typ | Beschreibung |
|--------|-----|--------------|
| `open` | `float64` | Eröffnungskurs |
| `high` | `float64` | Höchstkurs |
| `low` | `float64` | Tiefstkurs |
| `volume` | `float64` | Handelsvolumen |

### Sortierung
- **Primär:** `symbol` (aufsteigend)
- **Sekundär:** `timestamp` (aufsteigend)

### TZ-Policy
- **Intern:** Alle Timestamps sind **UTC** (timezone-aware).
- **Extern:** Bei Ingestion werden Zeitzonen auf UTC normalisiert.
- **Ausgabe:** Alle Dateien enthalten UTC-Timestamps.

### Beispiel

```python
import pandas as pd

# Gültiges Preis-Panel
prices = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
    "symbol": ["AAPL"] * 10,
    "close": [150.0, 151.0, 152.0, 151.5, 153.0, 152.5, 154.0, 153.5, 155.0, 154.5],
})

# Validierung
assert prices["timestamp"].dt.tz is not None  # UTC-aware
assert prices["timestamp"].dt.tz.zone == "UTC"
assert prices["close"].isna().sum() == 0  # Keine NaNs
assert (prices["close"] > 0).all()  # Positive Preise
```

### Dateien
- **Input:** `data/raw/1min/*.parquet` (Rohdaten)
- **Output:** `output/aggregates/5min.parquet`, `output/aggregates/daily.parquet`

### Validierungsfunktionen
- `src.assembled_core.utils.dataframe.coerce_price_types()`: Typ-Konvertierung
- `src.assembled_core.pipeline.io.load_prices()`: Laden mit Validierung
- `src.assembled_core.qa.health.check_prices()`: Health-Check

---

## 2. Signal Contract

### Zweck
Trading-Signale aus Strategien (z.B. EMA-Crossover).

### Required Columns

| Spalte | Typ | Beschreibung | Constraints |
|--------|-----|--------------|-------------|
| `timestamp` | `pd.Timestamp` (UTC, tz-aware) | Signal-Zeitstempel | Muss UTC sein, keine NaNs |
| `symbol` | `string` | Ticker-Symbol | Keine NaNs |
| `sig` | `int8` | Signal-Wert | -1 (SELL), 0 (neutral), +1 (BUY) |
| `price` | `float64` | Preis zum Signal-Zeitpunkt | Keine NaNs, > 0 |

### Signal-Werte

| Wert | Bedeutung |
|------|-----------|
| `-1` | SELL-Signal |
| `0` | Neutral (kein Signal) |
| `+1` | BUY-Signal |

### Sortierung
- **Primär:** `symbol` (aufsteigend)
- **Sekundär:** `timestamp` (aufsteigend)

### TZ-Policy
- **Intern:** Alle Timestamps sind **UTC** (timezone-aware).

### Beispiel

```python
import pandas as pd
import numpy as np

# Gültiges Signal-Panel
signals = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
    "symbol": ["AAPL"] * 5,
    "sig": np.array([0, 1, 1, -1, 0], dtype=np.int8),  # neutral, BUY, BUY, SELL, neutral
    "price": [150.0, 151.0, 152.0, 151.5, 153.0],
})

# Validierung
assert signals["timestamp"].dt.tz is not None  # UTC-aware
assert signals["sig"].dtype == np.int8
assert signals["sig"].isin([-1, 0, 1]).all()  # Nur erlaubte Werte
assert signals["price"].isna().sum() == 0  # Keine NaNs
```

### Dateien
- **Output:** `output/signals_{freq}.parquet` (optional, aktuell nicht persistiert)

### Validierungsfunktionen
- `src.assembled_core.pipeline.signals.compute_ema_signals()`: Signal-Generierung
- `src.assembled_core.pipeline.orders.signals_to_orders()`: Konvertierung zu Orders

---

## 3. Target Positions Contract

### Zweck
Ziel-Positionen aus Portfolio-Sizing (vor Order-Generierung).

### Required Columns

| Spalte | Typ | Beschreibung | Constraints |
|--------|-----|--------------|-------------|
| `symbol` | `string` | Ticker-Symbol | Keine NaNs |
| `target_qty` | `float64` | Ziel-Quantität | Kann negativ sein (Short-Positionen) |

### Optional Columns

| Spalte | Typ | Beschreibung |
|--------|-----|--------------|
| `target_weight` | `float64` | Ziel-Gewicht (0.0 - 1.0) |
| `qty` | `float64` | Alias für `target_qty` (für Kompatibilität) |

### Sortierung
- **Primär:** `symbol` (aufsteigend)

### Beispiel

```python
import pandas as pd

# Gültiges Target-Positions-Panel
targets = pd.DataFrame({
    "symbol": ["AAPL", "GOOGL", "MSFT"],
    "target_qty": [100.0, 50.0, -25.0],  # Long, Long, Short
    "target_weight": [0.4, 0.4, -0.2],  # Optional
})

# Validierung
assert targets["symbol"].isna().sum() == 0  # Keine NaNs
assert targets["target_qty"].dtype == np.float64
```

### Dateien
- **Intern:** Wird nicht persistiert (nur im Memory während Pipeline-Lauf)

### Validierungsfunktionen
- `src.assembled_core.portfolio.position_sizing.compute_target_positions()`: Position-Sizing
- `src.assembled_core.execution.order_generation.generate_orders_from_targets()`: Order-Generierung

---

## 4. Orders Contract

### Zweck
Ausführbare Orders (vor Risk-Controls).

### Required Columns

| Spalte | Typ | Beschreibung | Constraints |
|--------|-----|--------------|-------------|
| `timestamp` | `pd.Timestamp` (UTC, tz-aware) | Order-Zeitstempel | Muss UTC sein, keine NaNs |
| `symbol` | `string` | Ticker-Symbol | Keine NaNs |
| `side` | `string` | Order-Seite | "BUY" oder "SELL" (uppercase) |
| `qty` | `float64` | Quantität | Immer positiv (> 0), keine NaNs |
| `price` | `float64` | Order-Preis | > 0, keine NaNs |

### Side-Werte

| Wert | Bedeutung |
|------|-----------|
| `"BUY"` | Kauf-Order |
| `"SELL"` | Verkaufs-Order |

### Sortierung
- **Primär:** `timestamp` (aufsteigend)
- **Sekundär:** `symbol` (aufsteigend)

### TZ-Policy
- **Intern:** Alle Timestamps sind **UTC** (timezone-aware).

### Beispiel

```python
import pandas as pd

# Gültiges Orders-Panel
orders = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
    "symbol": ["AAPL", "GOOGL", "AAPL"],
    "side": ["BUY", "BUY", "SELL"],
    "qty": [100.0, 50.0, 50.0],  # Immer positiv
    "price": [150.0, 2500.0, 151.0],
})

# Validierung
assert orders["timestamp"].dt.tz is not None  # UTC-aware
assert orders["side"].isin(["BUY", "SELL"]).all()  # Nur erlaubte Werte
assert (orders["qty"] > 0).all()  # Immer positiv
assert orders["qty"].isna().sum() == 0  # Keine NaNs
assert orders["price"].isna().sum() == 0  # Keine NaNs
```

### Dateien
- **Output:** `output/orders_{freq}.csv` (CSV-Format)

### Validierungsfunktionen
- `src.assembled_core.pipeline.orders.signals_to_orders()`: Konvertierung von Signalen
- `src.assembled_core.pipeline.orders.write_orders()`: Schreiben mit Validierung
- `src.assembled_core.pipeline.io.load_orders()`: Laden mit Validierung
- `src.assembled_core.qa.health.check_orders()`: Health-Check

---

## 5. Risk Gate Contract

### Zweck
Risiko-Kontrollen für Orders (Pre-Trade-Checks + Kill-Switch).

### Input

| Parameter | Typ | Beschreibung | Constraints |
|-----------|-----|--------------|-------------|
| `orders` | `pd.DataFrame` | Orders (siehe Orders Contract) | Muss Orders Contract erfüllen |
| `portfolio` | `pd.DataFrame \| None` | Aktuelles Portfolio (optional) | - |
| `qa_status` | `QAGatesSummary \| None` | QA-Gates-Status (optional) | - |
| `risk_summary` | `dict[str, Any] \| None` | Risk-Summary (optional) | - |
| `pre_trade_config` | `PreTradeConfig \| None` | Pre-Trade-Config (optional) | - |

### Output

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `filtered_orders` | `pd.DataFrame` | Gefilterte Orders (siehe Orders Contract) | Orders, die alle Risk-Controls passiert haben |
| `pre_trade_result` | `PreTradeCheckResult \| None` | Pre-Trade-Check-Ergebnis | Enthält `blocked_reasons` (list of strings) |
| `kill_switch_engaged` | `bool` | Kill-Switch aktiviert | `True` wenn Kill-Switch alle Orders blockiert hat |
| `total_orders_before` | `int` | Anzahl Orders vor Filterung | - |
| `total_orders_after` | `int` | Anzahl Orders nach Filterung | - |

### Reason Fields

| Feld | Typ | Beschreibung |
|------|-----|--------------|
| `pre_trade_result.blocked_reasons` | `list[str]` | Liste von Blockierungs-Gründen | Jeder String ist ein menschenlesbarer Grund (z.B. "Position size limit exceeded", "QA_BLOCK gate failed") |

### Beispiel

```python
import pandas as pd
from src.assembled_core.execution.risk_controls import filter_orders_with_risk_controls

# Gültige Orders
orders = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=2, freq="1d", tz="UTC"),
    "symbol": ["AAPL", "GOOGL"],
    "side": ["BUY", "BUY"],
    "qty": [100.0, 50.0],
    "price": [150.0, 2500.0],
})

# Risk-Controls anwenden
filtered, result = filter_orders_with_risk_controls(
    orders,
    enable_pre_trade_checks=True,
    enable_kill_switch=True,
)

# Validierung
assert isinstance(filtered, pd.DataFrame)
assert len(filtered) <= len(orders)  # Kann nicht mehr Orders haben
assert result.total_orders_before == len(orders)
assert result.total_orders_after == len(filtered)

# Reason-Felder prüfen
if result.pre_trade_result is not None:
    if not result.pre_trade_result.is_ok:
        print(f"Blocked reasons: {result.pre_trade_result.blocked_reasons}")
```

### Validierungsfunktionen
- `src.assembled_core.execution.risk_controls.filter_orders_with_risk_controls()`: Hauptfunktion
- `src.assembled_core.execution.pre_trade_checks.run_pre_trade_checks()`: Pre-Trade-Checks
- `src.assembled_core.execution.kill_switch.guard_orders_with_kill_switch()`: Kill-Switch

---

## 5.5 Trades Contract (Post-Simulator)

### Zweck
Resultat-Schema nach `simulate_with_costs(...)` — die cash-gated, fill-modellierte Repräsentation der eingegangenen Orders. Dient als `BacktestResult.trades` und als Input für Ledger/Reconcile/Cost-Reporting.

### Required Columns

| Spalte | Typ | Beschreibung | Constraints |
|--------|-----|--------------|-------------|
| `timestamp` | `datetime64[ns, UTC]` | Bar-Timestamp (inherited from orders) | UTC-aware, nicht NaN |
| `symbol` | `string` | Ticker (inherited from orders) | nicht NaN, nicht leer |
| `side` | `string` | `"BUY"` oder `"SELL"` (inherited) | enum |
| `qty` | `float64` | Requested quantity (inherited from orders) | ≥ 0 unter aktueller BUY/SELL-Konvention; `simulate_with_costs` arbeitet defensiv mit `abs(qty)` |
| `price` | `float64` | Request price / decision price | unverändert vom orders-input |
| `status` | `string` | Fill-Status nach Pipeline | enum: `filled`/`partial`/`rejected` |
| `fill_qty` | `float64` | Tatsächlich gefillte Quantity (≤ `abs(qty)`) | 0 bei `rejected` |
| `fill_price` | `float64` | Ausführungspreis (kann von `price` abweichen bei limit-/partial-fills) | NaN bei `rejected` |
| `remaining_qty` | `float64` | `abs(qty) - fill_qty` | 0 bei `filled`, `qty` bei `rejected` |
| `reject_reason` | `string` | Reason-Code bei `status=rejected`, sonst `""` | enum (s.u.) |
| `notional` | `float64` | `abs(fill_qty) * abs(fill_price)` (legacy column, kein cash_delta-Input mehr) | ≥ 0, Skalar |
| `commission_cash` | `float64` | Provisions-Kosten (≥ 0) | 0 bei commission_bps=0 |
| `spread_cash` | `float64` | Spread-Kosten (≥ 0) | 0 bei spread_w=0 |
| `slippage_cash` | `float64` | Slippage/Impact-Kosten (≥ 0) | 0 bei impact_w=0 |
| `total_cost_cash` | `float64` | Summe der drei Cost-Komponenten | = `commission_cash + spread_cash + slippage_cash` |

### Optional / Internal-Scratch-Spalten

Die folgenden Spalten werden von `simulate_with_costs` an `trades_df` angehängt, sind aber nicht Teil des stabilen Contracts — Konsumenten **sollen sich nicht auf ihre Präsenz oder Semantik verlassen**:

| Spalte | Typ | Beschreibung | Stabilität |
|--------|-----|--------------|-----------|
| `cash_delta` | `float64` | Signierter Cash-Effekt der Order (BUY=negativ, SELL=positiv, inkl. costs) | internal scratch — wird in `apply_cash_gate` als kumulativer Constraint genutzt |
| `sign` | `float64` | `+1` für BUY, `-1` für SELL | internal scratch — convenience für portfolio.py |
| `order_type` | `string` | `"market"` (default) oder `"limit"` | Optional input column; `apply_limit_order_fills` aktiv nur bei `"limit"` |
| `limit_price` | `float64` | Limit-Preis für `order_type="limit"` | Optional input column; NaN → `reject_reason="LIMIT_PRICE_INVALID"` |

### Status-Enum (Werte aus `apply_fill_model_pipeline`, lowercase)

| Wert | Bedeutung |
|------|-----------|
| `"filled"` | Order voll gefüllt (`fill_qty == abs(qty)`) |
| `"partial"` | Teil-Fill (`0 < fill_qty < abs(qty)`), z. B. cash-gated oder partial-fill-policy |
| `"rejected"` | Vollständig abgewiesen (`fill_qty = 0`); Grund in `reject_reason` |

### Reject-Reason-Enum (Spalte `reject_reason`, nur bei `status=rejected`)

Producer-verified Werte aus `src/assembled_core/execution/fill_model.py`:

| Wert | Bedeutung | Quelle (fill_model.py) |
|------|-----------|------------------------|
| `"INSUFFICIENT_CASH"` | Cash-Gate Reject | `apply_cash_gate` (line ~121) |
| `"NOT_TRADING_DAY"` | Kalender-Reject (Feiertag/Wochenende) | `apply_session_gate` (line ~444) |
| `"OUTSIDE_SESSION"` | Außerhalb der Handelszeiten | `apply_session_gate` (line ~484) |
| `"SESSION_CHECK_FAILED"` | Calendar-Lookup raised exception (unknown date / out-of-range) | `apply_session_gate` fail-safe (line ~501) |
| `"LIMIT_NOT_REACHED"` | Limit-Preis nicht erreicht im Bar-Range | `apply_limit_order_fills` (line ~798) |
| `"LIMIT_PRICE_INVALID"` | `order_type="limit"` aber `limit_price` ist NaN/None | `apply_limit_order_fills` (line ~795) |
| `"QC_FAIL_MIN_FILL_QTY"` | Partial-Fill-Quantity unter `min_fill_qty` | `apply_partial_fills` (line ~649) |
| `"UNKNOWN"` | Kein spezifischer Grund gesetzt (Fallback) | `_ensure_reject_reason_filled` |

**Reservierte Konstanten (nicht aktuell emittiert):** `MIN_QTY` (`REJECT_MIN_QTY`), `MIN_NOTIONAL` (`REJECT_MIN_NOTIONAL`), `RISK_LIMIT` (`REJECT_RISK_LIMIT`), `MAX_POSITION` (`REJECT_MAX_POSITION`) sind als Module-Level-Konstanten in `fill_model.py` line 57-62 definiert, aber **kein Producer in `simulate_with_costs` schreibt sie**. Sie sind reserviert für künftige risk-/paper-/OMS-Layer; Konsumenten dürfen sie als gültige enum-Werte erwarten, sollten aber nicht voraussetzen, dass sie aktuell auftreten.

### Cash-Gate-Semantik (audit §9.6(d), 2026-05-19)

`simulate_with_costs` enforced eine kumulative Cash-Schranke über `available_cash` und `apply_fill_model_pipeline`. Orders die nach Cost-Abzug zu Cash<0 führen würden, werden auf `status=rejected` (mit `reject_reason="INSUFFICIENT_CASH"`) bzw. `status=partial` gesetzt; `fill_qty` reflektiert die tatsächlich ausführbare Menge. Equity- und Trades-Repräsentation sind seither **konsistent** — beide kommen aus demselben Simulator (`simulate_with_costs`), auch unter `include_costs=False` (dann commission_bps=spread_w=impact_w=0). Im Default-Pfad (`partial_fill_model=None`, `start_capital>0`) ist `apply_fill_model_pipeline` row-preserving — rejected-Rows verbleiben mit `fill_qty=0`, und die Invariante `len(trades_df) == len(orders_df)` gilt. Bekannte Ausnahmen: (i) `partial_fill_model` non-None UND `qty==0`-Rows im Input — `apply_partial_fills` droppt diese explizit (fill_model.py:596). (ii) `start_capital <= 0` — `apply_cash_gate` early-returned das raw orders_df ohne schema-enrichment (siehe Edge-Case unten).

**Edge-Case (empty-orders shortcut):** Wenn `orders_df` empty ist, gibt `simulate_with_costs` (portfolio.py:55-74) ein minimal-Schema trades_df mit nur den 5 Basis-Spalten zurück (`timestamp`, `symbol`, `side`, `qty`, `price`) — ohne `status`/`fill_qty`/`reject_reason`/cost-Spalten. Konsumenten müssen über Schema-Check (`"status" in trades_df.columns`) auf dieses Edge-Schema reagieren; row-count ist dann trivial 0==0. Siehe `qa/backtest_engine.py:1570-1583` als Referenz-Konsument.

### Beispiel

```python
import pandas as pd

# Gültiges Trades-Panel (Output von simulate_with_costs)
trades = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=3, freq="1d", tz="UTC"),
    "symbol":   ["AAPL", "GOOGL", "AAPL"],
    "side":     ["BUY",  "BUY",   "SELL"],
    "qty":      [100.0,  50.0,    50.0],
    "price":    [150.0,  2500.0,  151.0],
    "status":   ["filled", "rejected", "filled"],
    "fill_qty": [100.0,  0.0,     50.0],
    "reject_reason": ["", "INSUFFICIENT_CASH", ""],
    "commission_cash": [0.0, 0.0, 0.0],
    "spread_cash":     [0.0, 0.0, 0.0],
    "slippage_cash":   [0.0, 0.0, 0.0],
    "total_cost_cash": [0.0, 0.0, 0.0],
})

# Validierung
assert "status" in trades.columns and "fill_qty" in trades.columns
assert trades["status"].isin(["filled", "partial", "rejected"]).all()
assert (trades["fill_qty"] <= trades["qty"]).all()
assert (trades["fill_qty"] >= 0).all()
# Bei status=rejected muss reject_reason gesetzt sein (non-empty)
rejected_mask = trades["status"] == "rejected"
assert (trades.loc[rejected_mask, "reject_reason"].str.len() > 0).all()
```

### Validierungsfunktionen
- `src.assembled_core.pipeline.portfolio.simulate_with_costs()`: Producer (canonical)
- `src.assembled_core.execution.fill_model_pipeline.apply_fill_model_pipeline()`: Status/fill_qty-Setter
- `src.assembled_core.execution.cost_columns.add_cost_columns_to_trades()`: Cost-Column-Anreicherung

---

## 6. Equity Curve Contract

### Zweck
Portfolio-Equity-Kurve (für Backtesting und Reporting).

### Required Columns

| Spalte | Typ | Beschreibung | Constraints |
|--------|-----|--------------|-------------|
| `timestamp` | `pd.Timestamp` (UTC, tz-aware) | Zeitstempel | Muss UTC sein, keine NaNs |
| `equity` | `float64` | Portfolio-Equity | > 0, keine NaNs |

### Sortierung
- **Primär:** `timestamp` (aufsteigend)

### TZ-Policy
- **Intern:** Alle Timestamps sind **UTC** (timezone-aware).

### Beispiel

```python
import pandas as pd

# Gültige Equity-Kurve
equity = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=10, freq="1d", tz="UTC"),
    "equity": [10000.0, 10100.0, 10200.0, 10150.0, 10300.0, 10250.0, 10400.0, 10350.0, 10500.0, 10450.0],
})

# Validierung
assert equity["timestamp"].dt.tz is not None  # UTC-aware
assert (equity["equity"] > 0).all()  # Positive Equity
assert equity["equity"].isna().sum() == 0  # Keine NaNs
```

### Dateien
- **Output:** `output/equity_curve_{freq}.csv`, `output/portfolio_equity_{freq}.csv`

### Validierungsfunktionen
- `src.assembled_core.pipeline.portfolio.simulate_with_costs()`: Portfolio-Simulation (canonical; auch für no-costs via commission_bps=spread_w=impact_w=0)
- `src.assembled_core.pipeline.backtest.simulate_equity()`: **DEPRECATED** (audit §9.6(d), 2026-05-19) — kein cash-constraint check, retained nur für parity-tests gegen `backtest_legacy`

---

## 7. Allgemeine Regeln

### TZ-Policy (Zeitzonen-Politik)

**Grundsatz:** Alle Timestamps sind **intern UTC** (timezone-aware).

1. **Ingestion:** Bei Datenimport werden Zeitzonen auf UTC normalisiert.
2. **Verarbeitung:** Alle Pipeline-Schritte arbeiten mit UTC-Timestamps.
3. **Ausgabe:** Alle Dateien enthalten UTC-Timestamps.
4. **API:** API-Endpoints akzeptieren und liefern UTC-Timestamps (ISO-8601 Format mit 'Z' Suffix).

**Ausnahmen:** Keine. UTC ist die einzige erlaubte Zeitzone.

### NaN-Policy

**Grundsatz:** Keine stillen NaNs in Pflichtfeldern.

1. **Validierung:** Alle Contract-Validierungen prüfen auf NaNs in Pflichtfeldern.
2. **Bereinigung:** NaNs werden entweder entfernt (`dropna()`) oder explizit behandelt (z.B. `fillna(0.0)`).
3. **Fehlerbehandlung:** Wenn NaNs in Pflichtfeldern gefunden werden, wird ein `ValueError` oder `KeyError` geworfen.

### Typ-Konvertierung

**Grundsatz:** Explizite Typ-Konvertierung bei Ingestion.

1. **Timestamp:** `pd.to_datetime(..., utc=True, errors="coerce")`
2. **Numeric:** `pd.to_numeric(..., errors="coerce").astype("float64")`
3. **String:** `.astype("string")` oder `.astype(str).str.strip()`

### Sortierung

**Grundsatz:** Alle DataFrames sind nach definierten Spalten sortiert.

1. **Preise:** `symbol`, dann `timestamp`
2. **Signale:** `symbol`, dann `timestamp`
3. **Orders:** `timestamp`, dann `symbol`
4. **Equity:** `timestamp`

---

## 8. Contract-Tests

Alle Contracts sind durch Tests abgesichert in `tests/test_contracts_core_io.py`:

- ✅ Required Columns existieren
- ✅ Datatypes plausibel (timestamp tz-aware, numeric types)
- ✅ Keine stillen NaNs in Pflichtfeldern
- ✅ Sortierung korrekt
- ✅ TZ-Policy eingehalten (UTC-only)

**Ausführen:**
```bash
pytest tests/test_contracts_core_io.py -v
```

---

## 9. Änderungen an Contracts

**Wichtig:** Contracts sind **verbindlich** und sollten nicht ohne triftigen Grund geändert werden.

**Prozess für Änderungen:**
1. Änderung in `docs/CONTRACTS.md` dokumentieren
2. Contract-Tests aktualisieren
3. Alle betroffenen Module anpassen
4. Integrationstests ausführen
5. Dokumentation aktualisieren

---

## 10. Referenzen

- `src/assembled_core/utils/dataframe.py`: DataFrame-Utilities
- `src/assembled_core/pipeline/io.py`: I/O-Funktionen
- `src/assembled_core/qa/health.py`: Health-Checks
- `tests/test_contracts_core_io.py`: Contract-Tests
