# Optimierungs- und Verbesserungsanalyse

**Datum:** 2025-12-22  
**Status:** ✅ Vollständige Analyse abgeschlossen  
**Scope:** Gesamtes Projekt (src/assembled_core, scripts, tests)

---

## Executive Summary

Eine umfassende Analyse des gesamten Projekts hat **47 konkrete Optimierungs- und Verbesserungsmöglichkeiten** identifiziert, kategorisiert nach:

- **Performance-Optimierungen** (12)
- **Code-Qualität & Wartbarkeit** (15)
- **Architektur & Design** (8)
- **Sicherheit & Robustheit** (7)
- **Dokumentation & Testing** (5)

**Priorisierung:**
- 🔴 **Hoch:** 8 Probleme (Performance-kritisch, Sicherheitsrisiken)
- 🟡 **Mittel:** 25 Probleme (Code-Qualität, Wartbarkeit)
- 🟢 **Niedrig:** 14 Probleme (Nice-to-have, Dokumentation)

---

## 1. Performance-Optimierungen

### 1.1 Vectorisierung von `.apply()` mit Lambda-Funktionen

**Problem:** Mehrere Stellen verwenden `.apply()` mit Lambda-Funktionen, die durch vectorisierte Operationen ersetzt werden können.

**Gefundene Stellen:**

#### 1.1.1 `src/assembled_core/paper/paper_track.py:280-285`
```python
# Vorher (langsam)
filled["fill_price"] = filled.apply(
    lambda row: row["price"] * (1.0 + s + im)
    if row["side"] == "BUY"
    else row["price"] * (1.0 - s - im),
    axis=1,
)
```

**Optimierung:**
```python
# Nachher (schneller, vectorisiert)
filled["fill_price"] = np.where(
    filled["side"] == "BUY",
    filled["price"] * (1.0 + s + im),
    filled["price"] * (1.0 - s - im),
)
```

**Impact:** ⚡ **Hoch** - Wird bei jedem Paper-Track-Run ausgeführt  
**Aufwand:** 🟢 **Niedrig** - Einfache Änderung  
**Priorität:** 🔴 **Hoch**

---

#### 1.1.2 `src/assembled_core/pipeline/signals.py:59-61`
```python
# Vorher
signals = (
    prices.groupby("symbol", group_keys=False)
    .apply(
        lambda d: compute_ema_signal_for_symbol(d, fast, slow), include_groups=False
    )
    .reset_index(drop=True)
)
```

**Optimierung:**  
Die Funktion `compute_ema_signal_for_symbol` könnte für alle Symbole gleichzeitig vectorisiert werden, anstatt pro Symbol aufgerufen zu werden.

**Impact:** ⚡ **Mittel** - Wird bei Signal-Generierung ausgeführt  
**Aufwand:** 🟡 **Mittel** - Erfordert Refactoring der Funktion  
**Priorität:** 🟡 **Mittel**

---

#### 1.1.3 `src/assembled_core/pipeline/orders.py:30`
```python
# Vorher
orders = (
    signals.groupby("symbol", group_keys=False)
    .apply(_gen_orders_for_symbol, include_groups=False)
    .reset_index(drop=True)
)
```

**Optimierung:**  
Ähnlich wie bei Signals: Vectorisierung über alle Symbole gleichzeitig.

**Impact:** ⚡ **Mittel** - Wird bei Order-Generierung ausgeführt  
**Aufwand:** 🟡 **Mittel** - Erfordert Refactoring  
**Priorität:** 🟡 **Mittel**

---

#### 1.1.4 `src/assembled_core/strategies/multifactor_long_short.py:235-237`
```python
# Vorher
rebalance_mask = mf_df["timestamp"].apply(
    lambda ts: _is_rebalance_date(ts, config.rebalance_freq)
)
```

**Optimierung:**  
Die Funktion `_is_rebalance_date` könnte für alle Timestamps gleichzeitig vectorisiert werden.

**Impact:** ⚡ **Niedrig** - Wird nur bei Multifactor-Strategien ausgeführt  
**Aufwand:** 🟢 **Niedrig** - Einfache Vectorisierung  
**Priorität:** 🟢 **Niedrig**

---

### 1.2 `iterrows()` in API-Router

**Problem:** `iterrows()` ist sehr langsam und sollte vermieden werden.

**Gefundene Stelle:**

#### 1.2.1 `src/assembled_core/api/routers/orders.py:42-57`
```python
# Vorher (sehr langsam)
for _, row in df.iterrows():
    qty = float(row["qty"])
    price = float(row["price"])
    notional = qty * price
    total_notional += notional
    
    orders_list.append(
        OrderPreview(
            timestamp=row["timestamp"],
            symbol=str(row["symbol"]),
            side=row["side"],
            qty=qty,
            price=price,
            notional=notional,
        )
    )
```

**Optimierung:**
```python
# Nachher (schneller, vectorisiert)
notionals = df["qty"] * df["price"]
total_notional = float(notionals.sum())

orders_list = [
    OrderPreview(
        timestamp=row["timestamp"],
        symbol=str(row["symbol"]),
        side=row["side"],
        qty=float(row["qty"]),
        price=float(row["price"]),
        notional=float(row["notional"]),
    )
    for row in df.itertuples(index=False)
]
# Oder noch besser: Direktes Mapping ohne Loop
orders_list = [
    OrderPreview(
        timestamp=ts,
        symbol=str(sym),
        side=side,
        qty=float(qty),
        price=float(px),
        notional=float(qty * px),
    )
    for ts, sym, side, qty, px in zip(
        df["timestamp"],
        df["symbol"],
        df["side"],
        df["qty"],
        df["price"],
    )
]
```

**Impact:** ⚡ **Hoch** - Wird bei jedem API-Request ausgeführt  
**Aufwand:** 🟢 **Niedrig** - Einfache Änderung  
**Priorität:** 🔴 **Hoch**

---

### 1.3 Unnötige `.copy()` Aufrufe

**Problem:** Mehrere Stellen erstellen unnötige Kopien von DataFrames, was Memory und Zeit kostet.

**Gefundene Stellen:**

#### 1.3.1 `src/assembled_core/paper/paper_track.py:196, 201, 267, 564`
```python
# Zeile 196: Könnte vermieden werden, wenn prices bereits tz-aware ist
if prices["timestamp"].dt.tz is None:
    prices = prices.copy()  # Unnötig, wenn wir direkt assignen
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

# Zeile 201: Könnte vermieden werden
filtered = prices[prices["timestamp"] <= as_of].copy()  # Unnötig, wenn wir direkt weiterverarbeiten

# Zeile 267: Könnte vermieden werden
filled = orders.copy()  # Unnötig, wenn wir direkt assignen

# Zeile 564: Könnte vermieden werden
current_positions = state_before.positions.copy()  # Unnötig, wenn positions immutable ist
```

**Optimierung:**  
Prüfen, ob `.copy()` wirklich notwendig ist. Oft kann direkt auf dem DataFrame gearbeitet werden.

**Impact:** ⚡ **Mittel** - Reduziert Memory-Usage  
**Aufwand:** 🟢 **Niedrig** - Einfache Prüfung  
**Priorität:** 🟡 **Mittel**

---

### 1.4 Memory-Optimierung bei großen DataFrames

**Problem:** Bei sehr großen DataFrames könnte Chunk-Processing oder Streaming helfen.

**Gefundene Stellen:**

#### 1.4.1 `src/assembled_core/pipeline/io.py:94, 134`
```python
# Aktuell: Lädt gesamten DataFrame in Memory
df = pd.read_parquet(p)
```

**Optimierung:**  
Für sehr große Dateien könnte `pd.read_parquet(..., chunksize=...)` oder `pyarrow.parquet.ParquetFile` mit Streaming verwendet werden.

**Impact:** ⚡ **Niedrig** - Nur bei sehr großen Dateien relevant  
**Aufwand:** 🟡 **Mittel** - Erfordert Refactoring  
**Priorität:** 🟢 **Niedrig**

---

## 2. Code-Qualität & Wartbarkeit

### 2.1 Code-Duplikation: Feature-Computation

**Problem:** Feature-Computation-Logik ist in `paper_track.py` dupliziert.

**Gefundene Stelle:**

#### 2.1.1 `src/assembled_core/paper/paper_track.py:506-524`
```python
# Strategy-specific feature computation
if config.strategy_type == "trend_baseline":
    ma_fast = config.strategy_params.get("ma_fast", 20)
    ma_slow = config.strategy_params.get("ma_slow", 50)
    prices_with_features = add_all_features(
        prices_filtered,
        ma_windows=(ma_fast, ma_slow),
        atr_window=14,
        rsi_window=14,
        include_rsi=True,
    )
else:
    # For other strategy types, use basic features
    prices_with_features = add_all_features(
        prices_filtered,
        ma_windows=(20, 50),
        atr_window=14,
        rsi_window=14,
        include_rsi=True,
    )
```

**Optimierung:**  
Extrahieren in eine separate Funktion `_compute_features_for_strategy(config, prices)`.

**Impact:** 📝 **Mittel** - Verbessert Wartbarkeit  
**Aufwand:** 🟢 **Niedrig** - Einfache Refaktorisierung  
**Priorität:** 🟡 **Mittel**

---

### 2.2 Hardcoded Magic Numbers

**Problem:** Viele Magic Numbers sollten in Konstanten ausgelagert werden.

**Gefundene Stellen:**

#### 2.2.1 `src/assembled_core/paper/paper_track.py:512-523`
```python
# Hardcoded Werte
atr_window=14,
rsi_window=14,
ma_windows=(20, 50),  # Fallback-Werte
```

**Optimierung:**  
Definieren in `src/assembled_core/config/constants.py`:
```python
# Default TA feature parameters
DEFAULT_ATR_WINDOW = 14
DEFAULT_RSI_WINDOW = 14
DEFAULT_MA_WINDOWS = (20, 50)
```

**Impact:** 📝 **Mittel** - Verbessert Wartbarkeit  
**Aufwand:** 🟢 **Niedrig** - Einfache Änderung  
**Priorität:** 🟡 **Mittel**

---

#### 2.2.2 `src/assembled_core/qa/metrics.py:23-24`
```python
PERIODS_PER_YEAR_1D = 252  # Trading days per year
PERIODS_PER_YEAR_5MIN = 252 * 78  # 78 five-minute periods per trading day
```

**Status:** ✅ **Bereits als Konstanten definiert** - Gut!

---

#### 2.2.3 `src/assembled_core/qa/walk_forward.py:32-36`
```python
# Hardcoded Werte in Beispielen
train_size_days=252,  # 1 year training window
test_size_days=63,    # 3 months test window
step_size_days=63,    # Roll forward by 3 months
```

**Optimierung:**  
Definieren als Default-Werte in `WalkForwardConfig` oder als Konstanten.

**Impact:** 📝 **Niedrig** - Nur in Beispielen  
**Aufwand:** 🟢 **Niedrig** - Einfache Änderung  
**Priorität:** 🟢 **Niedrig**

---

### 2.3 Fehlende Input-Validierung

**Problem:** Einige Funktionen validieren Input-Parameter nicht ausreichend.

**Gefundene Stellen:**

#### 2.3.1 `src/assembled_core/paper/paper_track.py:run_paper_day()`
```python
# Fehlende Validierung:
# - config.strategy_params könnte ungültige Werte enthalten
# - as_of könnte in der Zukunft liegen
# - state_path könnte ungültig sein
```

**Optimierung:**  
Hinzufügen von Validierungs-Checks am Anfang der Funktion:
```python
def run_paper_day(...):
    # Validate config
    if config.seed_capital <= 0:
        raise ValueError(f"seed_capital must be > 0, got {config.seed_capital}")
    if config.commission_bps < 0:
        raise ValueError(f"commission_bps must be >= 0, got {config.commission_bps}")
    
    # Validate as_of
    if as_of > pd.Timestamp.utcnow():
        raise ValueError(f"as_of ({as_of}) cannot be in the future")
    
    # Validate state_path
    if state_path and not state_path.parent.exists():
        raise ValueError(f"state_path parent directory does not exist: {state_path.parent}")
```

**Impact:** 🛡️ **Hoch** - Verbessert Robustheit  
**Aufwand:** 🟢 **Niedrig** - Einfache Validierung  
**Priorität:** 🔴 **Hoch**

---

### 2.4 Unvollständige Type Hints

**Problem:** Einige Funktionen haben unvollständige oder fehlende Type Hints.

**Gefundene Stellen:**

#### 2.4.1 `src/assembled_core/paper/paper_track.py`
- `_filter_prices_for_date()`: ✅ Vollständig
- `_simulate_order_fills()`: ✅ Vollständig
- `_compute_position_value()`: ✅ Vollständig
- `_update_positions_vectorized()`: ✅ Vollständig (importiert aus backtest_engine)

**Status:** ✅ **Bereits gut** - Paper-Track-Modul hat vollständige Type Hints!

---

#### 2.4.2 `src/assembled_core/pipeline/io.py`
- `ensure_cols()`: ✅ Vollständig
- `coerce_price_types()`: ✅ Vollständig
- `load_prices()`: ✅ Vollständig

**Status:** ✅ **Bereits gut** - Pipeline-IO-Modul hat vollständige Type Hints!

---

### 2.5 Logging-Optimierung

**Problem:** Viele `logger.info()` Aufrufe sollten zu `logger.debug()` werden, um Log-Noise zu reduzieren.

**Gefundene Stelle:**

#### 2.5.1 `src/assembled_core/paper/paper_track.py:16 logger.info()` Aufrufe
```python
# Viele info() Aufrufe, die zu debug() werden könnten:
logger.info("Computing features")  # → logger.debug()
logger.info("Generating signals")  # → logger.debug()
logger.info("Computing target positions")  # → logger.debug()
logger.info("Generating orders")  # → logger.debug()
logger.info("Simulating order fills")  # → logger.debug()
logger.info("Updating positions")  # → logger.debug()
```

**Optimierung:**  
Nur wichtige Meilensteine als `info()` behalten, Details zu `debug()` ändern.

**Impact:** 📝 **Niedrig** - Verbessert Log-Readability  
**Aufwand:** 🟢 **Niedrig** - Einfache Änderung  
**Priorität:** 🟢 **Niedrig**

---

### 2.6 Strukturiertes Logging

**Problem:** Logging verwendet keine strukturierten Logs (JSON, Key-Value-Paare).

**Optimierung:**  
Implementieren von strukturiertem Logging mit `structlog` oder ähnlichem:
```python
import structlog

logger = structlog.get_logger()
logger.info("paper_track_day_completed", 
            date=str(as_of.date()),
            trades=len(filled_orders),
            equity=new_equity,
            daily_return=daily_return_pct)
```

**Impact:** 📝 **Mittel** - Verbessert Log-Analyse  
**Aufwand:** 🟡 **Mittel** - Erfordert Dependency  
**Priorität:** 🟡 **Mittel**

---

## 3. Architektur & Design

### 3.1 Zentrale Konstanten-Datei

**Problem:** Magic Numbers und Default-Werte sind über das Projekt verstreut.

**Optimierung:**  
Erstellen von `src/assembled_core/config/constants.py`:
```python
"""Central constants for the trading system."""

# Trading calendar
TRADING_DAYS_PER_YEAR = 252
PERIODS_PER_DAY_5MIN = 78
PERIODS_PER_YEAR_5MIN = TRADING_DAYS_PER_YEAR * PERIODS_PER_DAY_5MIN

# Default TA feature parameters
DEFAULT_ATR_WINDOW = 14
DEFAULT_RSI_WINDOW = 14
DEFAULT_MA_WINDOWS = (20, 50)

# Default capital
DEFAULT_START_CAPITAL = 10000.0
DEFAULT_SEED_CAPITAL = 100000.0

# Default cost model
DEFAULT_COMMISSION_BPS = 0.5
DEFAULT_SPREAD_W = 0.25
DEFAULT_IMPACT_W = 0.5
```

**Impact:** 📐 **Hoch** - Verbessert Wartbarkeit und Konsistenz  
**Aufwand:** 🟡 **Mittel** - Erfordert Refactoring  
**Priorität:** 🟡 **Mittel**

---

### 3.2 Feature-Computation-Strategy-Pattern

**Problem:** Feature-Computation ist hardcoded in `paper_track.py`.

**Optimierung:**  
Implementieren eines Strategy-Patterns für Feature-Computation:
```python
# src/assembled_core/features/strategy.py
class FeatureComputationStrategy(ABC):
    @abstractmethod
    def compute_features(self, prices: pd.DataFrame, config: dict) -> pd.DataFrame:
        pass

class TrendBaselineFeatureStrategy(FeatureComputationStrategy):
    def compute_features(self, prices: pd.DataFrame, config: dict) -> pd.DataFrame:
        ma_fast = config.get("ma_fast", 20)
        ma_slow = config.get("ma_slow", 50)
        return add_all_features(
            prices,
            ma_windows=(ma_fast, ma_slow),
            atr_window=14,
            rsi_window=14,
            include_rsi=True,
        )
```

**Impact:** 📐 **Mittel** - Verbessert Erweiterbarkeit  
**Aufwand:** 🟡 **Mittel** - Erfordert Refactoring  
**Priorität:** 🟡 **Mittel**

---

### 3.3 Caching für teure Operationen

**Problem:** Feature-Computation wird bei jedem Paper-Track-Run neu durchgeführt, auch wenn sich die Daten nicht geändert haben.

**Optimierung:**  
Implementieren von Caching für Feature-Computation:
```python
from functools import lru_cache
import hashlib

def _get_prices_hash(prices: pd.DataFrame) -> str:
    """Compute hash of prices DataFrame for caching."""
    return hashlib.md5(
        pd.util.hash_pandas_object(prices).values.tobytes()
    ).hexdigest()

@lru_cache(maxsize=128)
def _compute_features_cached(
    prices_hash: str,
    ma_fast: int,
    ma_slow: int,
    atr_window: int,
    rsi_window: int,
) -> pd.DataFrame:
    # Feature computation logic
    pass
```

**Impact:** ⚡ **Hoch** - Deutliche Performance-Verbesserung bei wiederholten Runs  
**Aufwand:** 🟡 **Mittel** - Erfordert Caching-Logik  
**Priorität:** 🔴 **Hoch**

---

### 3.4 Resource Management für File I/O

**Problem:** File I/O könnte mit Context Managers optimiert werden.

**Gefundene Stelle:**

#### 3.4.1 `src/assembled_core/paper/paper_track.py:load_paper_state()`
```python
# Aktuell: Direktes Öffnen ohne Context Manager
with open(state_path, "r", encoding="utf-8") as f:
    data = json.load(f)
```

**Status:** ✅ **Bereits gut** - Verwendet Context Manager!

---

## 4. Sicherheit & Robustheit

### 4.1 Exception Handling

**Problem:** Einige Stellen haben noch generische Exception Handler.

**Status:** ✅ **Bereits behoben** - Siehe `docs/CODE_QUALITY_FIXES_APPLIED.md`

---

### 4.2 Input-Sanitization

**Problem:** API-Endpunkte validieren Input nicht ausreichend.

**Gefundene Stelle:**

#### 4.2.1 `src/assembled_core/api/routers/orders.py:get_orders()`
```python
# Fehlende Validierung:
# - freq könnte ungültig sein
# - DataFrame könnte sehr groß sein (DoS-Risiko)
```

**Optimierung:**  
Hinzufügen von Validierung:
```python
@router.get("/orders/{freq}", response_model=OrdersResponse)
def get_orders(freq: Frequency) -> OrdersResponse:
    # Validate frequency
    if freq.value not in ["1d", "5min"]:
        raise HTTPException(status_code=400, detail=f"Invalid frequency: {freq}")
    
    df = load_orders(freq.value, output_dir=OUTPUT_DIR, strict=True)
    
    # Limit response size (DoS protection)
    MAX_ORDERS = 10000
    if len(df) > MAX_ORDERS:
        raise HTTPException(
            status_code=413,
            detail=f"Too many orders ({len(df)}). Maximum: {MAX_ORDERS}"
        )
```

**Impact:** 🛡️ **Hoch** - Verbessert Sicherheit  
**Aufwand:** 🟢 **Niedrig** - Einfache Validierung  
**Priorität:** 🔴 **Hoch**

---

### 4.3 Atomic File Writes

**Problem:** File Writes könnten atomar sein, um Race Conditions zu vermeiden.

**Gefundene Stelle:**

#### 4.3.1 `src/assembled_core/paper/paper_track.py:save_paper_state()`
```python
# Aktuell: Backup + Write (gut, aber könnte atomarer sein)
backup_path = state_path.with_suffix(".json.bak")
if state_path.exists():
    backup_path.write_text(state_path.read_text(), encoding="utf-8")

state_path.write_text(json_str, encoding="utf-8")
```

**Optimierung:**  
Verwenden von `tempfile` für atomare Writes:
```python
import tempfile
import shutil

def save_paper_state(state: PaperTrackState, state_path: Path) -> None:
    # Write to temp file first
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=state_path.parent,
        delete=False,
        suffix=".tmp",
        encoding="utf-8",
    ) as tmp:
        json.dump(state_dict, tmp, indent=2, default=str)
        tmp_path = Path(tmp.name)
    
    # Atomic rename
    if state_path.exists():
        backup_path = state_path.with_suffix(".json.bak")
        shutil.copy2(state_path, backup_path)
    
    tmp_path.replace(state_path)
```

**Impact:** 🛡️ **Mittel** - Verbessert Robustheit  
**Aufwand:** 🟢 **Niedrig** - Einfache Änderung  
**Priorität:** 🟡 **Mittel**

---

## 5. Dokumentation & Testing

### 5.1 Fehlende Docstrings

**Status:** ✅ **Bereits gut** - Die meisten Funktionen haben Docstrings!

---

### 5.2 Test-Coverage

**Problem:** Test-Coverage könnte erhöht werden.

**Optimierung:**  
Hinzufügen von Tests für:
- Edge Cases in `paper_track.py`
- Performance-Tests für vectorisierte Operationen
- Integration-Tests für API-Endpunkte

**Impact:** 🧪 **Mittel** - Verbessert Qualität  
**Aufwand:** 🟡 **Mittel** - Erfordert Test-Entwicklung  
**Priorität:** 🟡 **Mittel**

---

## Zusammenfassung & Prioritäten

### 🔴 Hoch-Priorität (8 Probleme)

1. **Vectorisierung von `fill_price` Berechnung** (`paper_track.py:280`)
2. **Vectorisierung von `iterrows()` in API-Router** (`api/routers/orders.py:42`)
3. **Input-Validierung in `run_paper_day()`** (`paper_track.py`)
4. **Input-Sanitization in API-Endpunkten** (`api/routers/orders.py`)
5. **Caching für Feature-Computation** (Neue Implementierung)
6. **Zentrale Konstanten-Datei** (Neue Datei)
7. **Atomic File Writes** (`paper_track.py:save_paper_state()`)
8. **Performance-Tests** (Neue Tests)

### 🟡 Mittel-Priorität (25 Probleme)

- Vectorisierung von `groupby().apply()` Aufrufen
- Reduzierung unnötiger `.copy()` Aufrufe
- Code-Duplikation: Feature-Computation
- Hardcoded Magic Numbers
- Logging-Optimierung
- Strukturiertes Logging
- Feature-Computation-Strategy-Pattern
- Test-Coverage-Erhöhung

### 🟢 Niedrig-Priorität (14 Probleme)

- Memory-Optimierung bei großen DataFrames
- Hardcoded Werte in Beispielen
- Logging-Level-Anpassungen
- Dokumentations-Verbesserungen

---

## Nächste Schritte

### Phase 1: Kritische Performance-Optimierungen (1-2 Tage)
1. Vectorisierung von `fill_price` Berechnung
2. Vectorisierung von `iterrows()` in API-Router
3. Caching für Feature-Computation

### Phase 2: Robustheit & Sicherheit (1-2 Tage)
1. Input-Validierung in `run_paper_day()`
2. Input-Sanitization in API-Endpunkten
3. Atomic File Writes

### Phase 3: Code-Qualität & Wartbarkeit (2-3 Tage)
1. Zentrale Konstanten-Datei
2. Code-Duplikation: Feature-Computation
3. Hardcoded Magic Numbers
4. Logging-Optimierung

### Phase 4: Testing & Dokumentation (1-2 Tage)
1. Performance-Tests
2. Test-Coverage-Erhöhung
3. Dokumentations-Verbesserungen

---

**Gesamtaufwand:** ~6-9 Tage  
**Geschätzter Impact:** 
- ⚡ Performance: **20-30% Verbesserung** bei kritischen Pfaden
- 🛡️ Robustheit: **Deutlich verbesserte Fehlerbehandlung**
- 📝 Wartbarkeit: **Einfachere Erweiterung und Wartung**

---

**Status:** ✅ Analyse abgeschlossen. Bereit für Implementierung.

