# Sprint 4: Corporate Actions & Universe as-of - Implementierungsplan

**Status:** Plan (noch nicht implementiert)  
**Erstellt:** 2025-01-04  
**Basis:** Roadmap NR3, Sprint 4 Anforderungen

---

## 1. Scan-Ergebnisse

### 1.1 Bestehende Module

**Corporate Actions:**
- ❌ Kein `corporate_actions.py` Modul vorhanden
- ❌ Kein `universe.py` Modul vorhanden
- ⚠️ `adj_close` wird in `scripts/features/build_daily_features.py` verwendet (Legacy-Script)
- ❌ Keine Split/Dividend-Datenstrukturen
- ❌ Keine Delist-Tracking

**Universe:**
- ✅ `universe: list[str] | None` Parameter in `TradingContext` (statisch)
- ✅ `validate_universe_vs_data()` in `scripts/run_daily.py` (filtert nach Symbolen)
- ✅ `default_universe` in `Settings` (statisch)
- ❌ Keine zeitabhängige Universe-Membership (kein `as_of` Filter)
- ❌ Keine Universe-History-Tracking

### 1.2 Hotspots: Wo wird `close` genutzt?

**Features (Returns/MA):**
1. `src/assembled_core/features/ta_features.py`:
   - `add_log_returns(price_col="close")` (default)
   - `add_moving_averages(price_col="close")` (default)
   - `add_rsi(price_col="close")` (default)

2. `src/assembled_core/pipeline/signals.py`:
   - `compute_ema_signal_for_symbol()`: nutzt `d["close"]` direkt

3. `scripts/features/build_daily_features.py` (Legacy):
   - Nutzt `adj_close` (aber nicht in Core-Pipeline integriert)

**Portfolio/Execution:**
4. `src/assembled_core/pipeline/portfolio.py`:
   - `simulate_with_costs()`: nutzt `orders["price"]` (nicht direkt `close`, aber Preis-basiert)

5. `src/assembled_core/execution/order_generation.py`:
   - `generate_orders_from_targets_fast()`: nutzt `prices_latest` (Preis-basiert)

**Zusammenfassung:**
- Features nutzen `close` (nicht `adj_close`)
- Keine Corporate-Actions-Adjustierung in Core-Pipeline
- Legacy-Script nutzt `adj_close`, aber nicht integriert

---

## 2. Minimal-Policy (Roadmap-konform)

### 2.1 Corporate Actions: Research-adjusted vs Trading-unadjusted

**Prinzip:**
- **Research-adjusted**: Preise sind split/dividend-adjusted für Returns/Features (z.B. `close_adj`)
- **Trading-unadjusted**: Preise sind unadjusted für Order-Execution (z.B. `close`)
- **Dividenden**: Werden als Cashflow behandelt (nicht in Preisen adjustiert)

**Datenstrukturen:**
- `CorporateAction` (dataclass): `symbol`, `timestamp`, `action_type` ("split", "dividend", "delist"), `ratio` (für Splits), `amount` (für Dividenden), `ex_date`
- `CorporateActionsStore`: Speichert/Lädt Corporate Actions (Parquet/CSV)

**Adjustment-Logik:**
- `adjust_prices_for_research(prices: pd.DataFrame, corporate_actions: pd.DataFrame) -> pd.DataFrame`:
  - Fügt `close_adj` Spalte hinzu (split-adjusted, dividend-adjusted für Returns)
  - Behält `close` unverändert (für Trading)
- `compute_dividend_cashflow(corporate_actions: pd.DataFrame, positions: pd.DataFrame) -> pd.DataFrame`:
  - Berechnet Dividenden-Cashflow pro Position

### 2.2 Universe as-of: Zeitabhängige Membership

**Prinzip:**
- Universe-Membership ist zeitabhängig (Symbol kann zu/aus Universe hinzugefügt/entfernt werden)
- `as_of` Filter: Welche Symbole sind am `as_of` Datum im Universe?

**Datenstrukturen:**
- `UniverseMembership` (dataclass): `symbol`, `added_date`, `removed_date | None`
- `UniverseStore`: Speichert/Lädt Universe-History (Parquet/CSV)

**Filter-Logik:**
- `filter_universe_as_of(universe_history: pd.DataFrame, as_of: pd.Timestamp) -> list[str]`:
  - Gibt Liste der Symbole zurück, die am `as_of` Datum im Universe sind
  - Regel: `added_date <= as_of` AND (`removed_date is None` OR `removed_date > as_of`)

---

## 3. Implementierungsplan

### 3.1 Module-Struktur

**Neue Module:**
1. `src/assembled_core/data/corporate_actions.py`
   - `CorporateAction` (dataclass)
   - `CorporateActionsStore` (Klasse)
   - `adjust_prices_for_research()`
   - `compute_dividend_cashflow()`

2. `src/assembled_core/data/universe.py`
   - `UniverseMembership` (dataclass)
   - `UniverseStore` (Klasse)
   - `filter_universe_as_of()`
   - `load_universe_history()`

**Erweiterte Module:**
3. `src/assembled_core/pipeline/trading_cycle.py`
   - `TradingContext.universe` bleibt `list[str] | None` (statisch, backward compatible)
   - Neue Funktion: `_resolve_universe_as_of(universe_history: pd.DataFrame | None, as_of: pd.Timestamp | None) -> list[str] | None`
   - Integration: Wenn `universe_history` vorhanden, filtere mit `as_of`

4. `src/assembled_core/features/ta_features.py`
   - Erweitere `add_log_returns()`, `add_moving_averages()`, `add_rsi()`:
   - Neuer Parameter: `use_adjusted: bool = False`
   - Wenn `use_adjusted=True`: nutze `close_adj` statt `close` (falls vorhanden)

5. `src/assembled_core/pipeline/signals.py`
   - Erweitere `compute_ema_signal_for_symbol()`:
   - Neuer Parameter: `use_adjusted: bool = False`
   - Wenn `use_adjusted=True`: nutze `close_adj` statt `close` (falls vorhanden)

### 3.2 Funktionen (Details)

**Corporate Actions:**

```python
# src/assembled_core/data/corporate_actions.py

@dataclass
class CorporateAction:
    symbol: str
    timestamp: pd.Timestamp  # Ex-date (UTC)
    action_type: Literal["split", "dividend", "delist"]
    ratio: float | None = None  # Für Splits (z.B. 2.0 für 2:1 Split)
    amount: float | None = None  # Für Dividenden (pro Share)
    ex_date: pd.Timestamp | None = None  # Falls unterschiedlich von timestamp

class CorporateActionsStore:
    def load_corporate_actions(
        self, symbols: list[str] | None = None,
        start_date: pd.Timestamp | None = None,
        end_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame
    
    def store_corporate_actions(
        self, actions: pd.DataFrame, path: Path | None = None
    ) -> Path

def adjust_prices_for_research(
    prices: pd.DataFrame,
    corporate_actions: pd.DataFrame,
) -> pd.DataFrame:
    """Fügt close_adj Spalte hinzu (split-adjusted, dividend-adjusted für Returns).
    
    Behält close unverändert (für Trading).
    """
    
def compute_dividend_cashflow(
    corporate_actions: pd.DataFrame,
    positions: pd.DataFrame,  # symbol, qty, timestamp
) -> pd.DataFrame:
    """Berechnet Dividenden-Cashflow pro Position."""
```

**Universe:**

```python
# src/assembled_core/data/universe.py

@dataclass
class UniverseMembership:
    symbol: str
    added_date: pd.Timestamp  # UTC
    removed_date: pd.Timestamp | None = None  # None = noch aktiv

class UniverseStore:
    def load_universe_history(
        self, universe_name: str = "default",
    ) -> pd.DataFrame
    
    def store_universe_history(
        self, history: pd.DataFrame, universe_name: str = "default",
    ) -> Path

def filter_universe_as_of(
    universe_history: pd.DataFrame,
    as_of: pd.Timestamp,
) -> list[str]:
    """Gibt Liste der Symbole zurück, die am as_of Datum im Universe sind.
    
    Regel: added_date <= as_of AND (removed_date is None OR removed_date > as_of)
    """
```

### 3.3 Integration-Punkte

**Daily Ingest:**
- `scripts/run_daily.py`: Lädt Corporate Actions, adjustiert Preise für Research
- `scripts/run_daily.py`: Lädt Universe-History, filtert mit `as_of`

**Backtest:**
- `scripts/run_backtest_strategy.py`: Lädt Corporate Actions, adjustiert Preise für Research
- `scripts/run_backtest_strategy.py`: Lädt Universe-History, filtert pro Timestamp mit `as_of`

**Trading Cycle:**
- `src/assembled_core/pipeline/trading_cycle.py`: 
  - Wenn `corporate_actions` vorhanden: `adjust_prices_for_research()` aufrufen
  - Wenn `universe_history` vorhanden: `filter_universe_as_of()` aufrufen

**Features:**
- `src/assembled_core/features/ta_features.py`: 
  - Wenn `close_adj` vorhanden: `use_adjusted=True` als Default (optional)
  - Backward compatible: `use_adjusted=False` bleibt Default

### 3.4 Tests

**Corporate Actions:**
- `tests/test_corporate_actions.py`:
  - `test_split_adjustment()`: 2:1 Split → `close_adj` halbiert vor Split
  - `test_dividend_adjustment()`: Dividend → `close_adj` adjusted (für Returns)
  - `test_dividend_cashflow()`: Dividenden-Cashflow pro Position
  - `test_delist_handling()`: Delist → Symbol aus Universe entfernen
  - `test_store_load_roundtrip()`: Corporate Actions Store/Load

**Universe:**
- `tests/test_universe_as_of.py`:
  - `test_filter_universe_as_of()`: Symbole am `as_of` Datum
  - `test_universe_added_removed()`: Symbol hinzugefügt/entfernt
  - `test_universe_active()`: Symbol noch aktiv (removed_date=None)
  - `test_store_load_roundtrip()`: Universe-History Store/Load

**Integration:**
- `tests/test_trading_cycle_corporate_actions.py`:
  - `test_trading_cycle_uses_adjusted_prices()`: Features nutzen `close_adj`
  - `test_trading_cycle_uses_unadjusted_prices()`: Orders nutzen `close`
  - `test_universe_as_of_filtering()`: Universe wird mit `as_of` gefiltert

---

## 4. Datenstrukturen (Schema)

### 4.1 Corporate Actions (Parquet/CSV)

**Spalten:**
- `symbol`: str
- `timestamp`: pd.Timestamp (UTC, ex-date)
- `action_type`: str ("split", "dividend", "delist")
- `ratio`: float | None (für Splits)
- `amount`: float | None (für Dividenden, pro Share)
- `ex_date`: pd.Timestamp | None (falls unterschiedlich)

**Pfad:**
- `data/corporate_actions/corporate_actions.parquet` (oder CSV)

### 4.2 Universe History (Parquet/CSV)

**Spalten:**
- `symbol`: str
- `added_date`: pd.Timestamp (UTC)
- `removed_date`: pd.Timestamp | None (None = noch aktiv)

**Pfad:**
- `data/universe/{universe_name}_history.parquet` (oder CSV)
- Default: `data/universe/default_history.parquet`

---

## 5. Backward Compatibility

**Wichtig:**
- `close` bleibt unverändert (für Trading)
- `close_adj` ist optional (nur wenn Corporate Actions vorhanden)
- `universe: list[str] | None` bleibt statisch (backward compatible)
- `use_adjusted=False` bleibt Default in Features/Signals

**Migration:**
- Bestehende Scripts funktionieren ohne Änderung
- Neue Features (Corporate Actions, Universe as-of) sind opt-in

---

## 6. Nächste Schritte (nicht in diesem Sprint)

**Sprint 5 (Factor Store Integration):**
- Corporate Actions in Factor Store integrieren
- Universe-History in Factor Store integrieren

**Sprint 6+ (Erweiterte Features):**
- Corporate Actions aus Yahoo Finance/Alpha Vantage fetchen
- Universe-History aus externen Quellen (z.B. S&P 500 History)

---

## 7. Dokumentation

**Neue Docs:**
- `docs/CORPORATE_ACTIONS.md`: Policy, Datenstrukturen, Beispiele
- `docs/UNIVERSE_AS_OF.md`: Policy, Datenstrukturen, Beispiele

**Erweiterte Docs:**
- `docs/CONTRACTS.md`: Erweitere um `close_adj` Spalte (optional)
- `docs/ARCHITECTURE_LAYERING.md`: Erweitere um Corporate Actions/Universe Layer

---

## Status

**Plan erstellt:** ✅  
**Implementierung:** ❌ (noch nicht gestartet)  
**Tests:** ❌ (noch nicht geschrieben)
