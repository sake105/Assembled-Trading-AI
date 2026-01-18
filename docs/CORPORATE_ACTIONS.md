# Corporate Actions Handling - Assembled Trading AI

**Status:** Verbindlich (Sprint 4 / C1)  
**Letzte Aktualisierung:** 2025-01-04

---

## Zweck

Dieses Dokument definiert die Policy fuer Corporate Actions (Splits, Dividenden) im Trading-System. Wichtigste Regel: **Research-adjusted vs Trading-unadjusted trennen**.

---

## Policy

### 1. Research-adjusted vs Trading-unadjusted

**Prinzip:**
- **Research-adjusted prices** (`close_research`): Split-adjusted fuer Returns/Features
- **Trading-unadjusted prices** (`close`): Unadjusted fuer Order-Execution

**Warum?**
- Splits erzeugen keine echten Returns (nur Preis-Anpassung)
- Research-adjusted Preise verhindern "fake crashes" in Returns (z.B. -50% bei 2:1 Split)
- Trading muss mit tatsaechlichen Preisen arbeiten (unadjusted)

**Beispiel (2:1 Split):**
```
Tag 1: close = 200.0 (vor Split)
Tag 2: close = 100.0 (nach Split, tatsaechlicher Trading-Preis)
Tag 3: close = 102.0

Research-adjusted (close_research):
Tag 1: close_research = 100.0 (200.0 * 0.5, adjusted backward)
Tag 2: close_research = 100.0 (100.0, unveraendert)
Tag 3: close_research = 102.0 (102.0, unveraendert)

Returns (mit close_research):
Tag 1->2: 0% (kein fake crash)
Tag 2->3: +2% (korrekte Return)

Returns (mit close, falsch):
Tag 1->2: -50% (fake crash!)
Tag 2->3: +2%
```

### 2. Dividenden als Cashflow

**Prinzip:**
- Dividenden werden **nicht** in Preise adjustiert
- Dividenden sind **Cashflow-Events** (Ledger-ready)
- `compute_dividend_cashflows()` generiert Cashflow-Events fuer Positionen

**Warum?**
- Dividenden sind tatsaechliche Cashflows (nicht nur Preis-Anpassung)
- Portfolio-Simulation muss Dividenden als Cashflow behandeln
- Ledger-System (spaeter) braucht Cashflow-Events

**Beispiel:**
```
Position: 100 shares AAPL
Dividend: $0.25 per share
Cashflow Event:
  timestamp: 2024-01-15 (ex-date)
  symbol: AAPL
  cashflow_type: DIVIDEND
  amount: 25.0 (100 * 0.25)
```

---

## Datenstrukturen

### CorporateAction (Data Model)

**Felder:**
- `symbol`: str (z.B. "AAPL")
- `action_type`: Literal["SPLIT", "DIVIDEND"]
- `effective_date`: pd.Timestamp (UTC session close)
- `split_ratio`: float | None (fuer SPLIT, z.B. 2.0 fuer 2:1)
- `dividend_cash`: float | None (fuer DIVIDEND, cash per share)

**Beispiel (Split):**
```python
action = CorporateAction(
    symbol="AAPL",
    action_type="SPLIT",
    effective_date=pd.Timestamp("2024-01-15", tz="UTC"),
    split_ratio=2.0,  # 2:1 split
)
```

**Beispiel (Dividend):**
```python
action = CorporateAction(
    symbol="AAPL",
    action_type="DIVIDEND",
    effective_date=pd.Timestamp("2024-01-15", tz="UTC"),
    dividend_cash=0.25,  # $0.25 per share
)
```

### DataFrame Schema

**Corporate Actions (Input):**
- `symbol`: str
- `action_type`: str ("SPLIT" oder "DIVIDEND")
- `effective_date`: pd.Timestamp (UTC)
- `split_ratio`: float | None (fuer SPLIT)
- `dividend_cash`: float | None (fuer DIVIDEND)

**Prices (Output nach apply_splits_for_research_prices):**
- Alle originalen Spalten (timestamp, symbol, close, ...)
- `close_research`: float (split-adjusted, fuer Research)

**Cashflows (Output von compute_dividend_cashflows):**
- `timestamp`: pd.Timestamp (UTC, ex-date)
- `symbol`: str
- `cashflow_type`: str ("DIVIDEND")
- `amount`: float (total cashflow = dividend_cash * qty)

---

## Funktionen

### apply_splits_for_research_prices()

**Zweck:** Erstellt split-adjusted Preise fuer Research (Returns/Features).

**Input:**
- `prices`: DataFrame mit timestamp, symbol, close, ...
- `actions`: DataFrame mit SPLIT actions

**Output:**
- DataFrame mit zusaetzlicher Spalte `close_research` (split-adjusted)

**Regel:**
- Alle Preise **vor** Split-Datum werden mit `1 / split_ratio` multipliziert
- Preise **nach** Split-Datum bleiben unveraendert
- `close` bleibt unveraendert (fuer Trading)

**Beispiel:**
```python
from src.assembled_core.data.corporate_actions import apply_splits_for_research_prices

prices = pd.DataFrame({
    "timestamp": pd.date_range("2024-01-01", periods=5, freq="1d", tz="UTC"),
    "symbol": ["AAPL"] * 5,
    "close": [200.0, 205.0, 100.0, 102.0, 104.0],  # 2:1 split on day 3
})

actions = pd.DataFrame({
    "symbol": ["AAPL"],
    "action_type": ["SPLIT"],
    "effective_date": [pd.Timestamp("2024-01-03", tz="UTC")],
    "split_ratio": [2.0],
})

result = apply_splits_for_research_prices(prices, actions)
# result["close_research"] = [100.0, 102.5, 100.0, 102.0, 104.0]
# (Tag 1-2 adjusted by 0.5, Tag 3-5 unchanged)
```

### compute_dividend_cashflows()

**Zweck:** Berechnet Dividenden-Cashflows fuer Positionen (Ledger-ready).

**Input:**
- `positions`: DataFrame mit symbol, qty (und optional timestamp)
- `actions`: DataFrame mit DIVIDEND actions
- `as_of`: Optional cutoff timestamp

**Output:**
- DataFrame mit Cashflow-Events (timestamp, symbol, cashflow_type, amount)

**Regel:**
- Nur Dividenden fuer Symbole in Positionen
- `amount = dividend_cash * qty`
- Gefiltert nach `as_of` (falls angegeben)

**Beispiel:**
```python
from src.assembled_core.data.corporate_actions import compute_dividend_cashflows

positions = pd.DataFrame({
    "symbol": ["AAPL", "MSFT"],
    "qty": [100.0, 50.0],
})

actions = pd.DataFrame({
    "symbol": ["AAPL"],
    "action_type": ["DIVIDEND"],
    "effective_date": [pd.Timestamp("2024-01-15", tz="UTC")],
    "dividend_cash": [0.25],
})

cashflows = compute_dividend_cashflows(positions, actions)
# cashflows["amount"] = [25.0]  # 100 shares * $0.25
```

---

## Integration

### Features/Signals

**Regel:**
- Features (MA, RSI, Returns) sollen `close_research` nutzen (falls vorhanden)
- Fallback: `close` (wenn `close_research` nicht vorhanden)

**Beispiel:**
```python
# In ta_features.py oder signals.py
if "close_research" in prices.columns:
    price_col = "close_research"  # Use research-adjusted
else:
    price_col = "close"  # Fallback to unadjusted
```

### Execution/Orders

**Regel:**
- Orders/Execution nutzen weiterhin `close` (trading-unadjusted)
- `close_research` wird **nicht** fuer Order-Preise verwendet

---

## Tests

### Split Test: "Kein fake crash"

**Szenario:**
- 2:1 Split am Tag 3
- Preise: [200.0, 205.0, 100.0, 102.0, 104.0]

**Erwartung:**
- `close_research`: [100.0, 102.5, 100.0, 102.0, 104.0]
- Returns (mit close_research): [0%, +2%, +2%] (kein -50% crash)

### Dividend Test: "Cashflow event"

**Szenario:**
- Position: 100 shares AAPL
- Dividend: $0.25 per share am 2024-01-15

**Erwartung:**
- Cashflow Event: timestamp=2024-01-15, amount=25.0
- Kein "mystery return" in Preisen

---

## Dateispeicherung (Zukunft)

**Geplant (nicht in C1):**
- Corporate Actions Store: `data/corporate_actions/corporate_actions.parquet`
- Format: Parquet (spaltenorientiert, schnell)

**C1:**
- Funktionen arbeiten mit DataFrames (kein Store/Load in C1)

---

## Roadmap

**Sprint 4 / C1 (aktuell):**
- ✅ Split-Adjustment fuer Research-Preise
- ✅ Dividend-Cashflow-Berechnung
- ✅ Integration in Features/Signals (optional)

**Sprint 5+ (Zukunft):**
- Corporate Actions Store/Load
- Integration in Factor Store
- Corporate Actions aus externen Quellen (Yahoo, Alpha Vantage)

---

## ASCII-Only Policy

Dieses Dokument ist ASCII-only (keine Umlaute). Verwende:
- ae statt ä
- oe statt ö
- ue statt ü
- ss statt ß
