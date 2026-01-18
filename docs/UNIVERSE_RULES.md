# Universe Rules - Assembled Trading AI

**Status:** Verbindlich (Sprint 4 / C2)  
**Letzte Aktualisierung:** 2025-01-04

---

## Zweck

Dieses Dokument definiert die Policy fuer Universe Management (zeitabhaengige Membership) im Trading-System. Wichtigste Regel: **Universe-Membership ist zeitabhaengig (as_of deterministisch)**.

---

## Policy

### 1. Zeitabhaengige Membership

**Prinzip:**
- Universe-Membership ist zeitabhaengig (Symbole koennen hinzugefuegt/entfernt werden)
- `as_of` Filter: Welche Symbole sind am `as_of` Datum im Universe?
- Deterministisch: Gleiche Eingabe -> gleiche Ausgabe (UTC-normalisiert)

**Membership-Regel:**
- Symbol ist im Universe am `as_of` wenn:
  - `start_date <= as_of` AND (`end_date is None` OR `end_date > as_of`)
  - `end_date` ist EXCLUSIVE (Symbol ist NICHT im Universe am `end_date`)

**Beispiel:**
```
Symbol: AAPL
start_date: 2024-01-01
end_date: 2024-06-30 (exclusive)

as_of = 2023-12-31: AAPL NICHT im Universe (start_date > as_of)
as_of = 2024-01-01: AAPL im Universe (start_date <= as_of, end_date > as_of)
as_of = 2024-06-29: AAPL im Universe (start_date <= as_of, end_date > as_of)
as_of = 2024-06-30: AAPL NICHT im Universe (end_date = as_of, EXCLUSIVE)
as_of = 2024-07-01: AAPL NICHT im Universe (end_date < as_of)
```

### 2. end_date Semantik: EXCLUSIVE

**Wichtig:**
- `end_date` ist EXCLUSIVE (Symbol ist NICHT im Universe am `end_date`)
- Wenn `end_date = 2024-06-30`, dann ist Symbol am 2024-06-30 NICHT mehr im Universe
- Symbol ist im Universe bis (aber nicht einschliesslich) `end_date`

**Warum EXCLUSIVE?**
- Klare Semantik: "Symbol wurde am end_date entfernt" bedeutet "nicht mehr im Universe ab end_date"
- Konsistent mit typischen Membership-Systemen (z.B. S&P 500 Index Changes)

**Beispiel:**
```
Symbol: MSFT
start_date: 2024-01-01
end_date: 2024-06-30 (exclusive)

as_of = 2024-06-29: MSFT im Universe
as_of = 2024-06-30: MSFT NICHT im Universe (end_date exclusive)
as_of = 2024-07-01: MSFT NICHT im Universe
```

### 3. Aktive Symbole (end_date = None)

**Prinzip:**
- Wenn `end_date = None`, dann ist Symbol noch aktiv im Universe
- Symbol bleibt im Universe ab `start_date` bis unbegrenzt

**Beispiel:**
```
Symbol: GOOGL
start_date: 2024-01-01
end_date: None (noch aktiv)

as_of = 2024-01-01: GOOGL im Universe
as_of = 2024-12-31: GOOGL im Universe (noch aktiv)
as_of = 2025-12-31: GOOGL im Universe (noch aktiv)
```

---

## Datenstrukturen

### Universe History (DataFrame)

**Spalten:**
- `symbol`: str (z.B. "AAPL")
- `start_date`: pd.Timestamp (UTC, wann Symbol hinzugefuegt wurde)
- `end_date`: pd.Timestamp | None (UTC, wann Symbol entfernt wurde, EXCLUSIVE)
  - Wenn None: Symbol ist noch aktiv
  - Wenn gesetzt: Symbol wurde am end_date entfernt (nicht mehr im Universe ab end_date)

**Beispiel:**
```python
history = pd.DataFrame({
    "symbol": ["AAPL", "MSFT", "GOOGL"],
    "start_date": [
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
    ],
    "end_date": [
        None,  # AAPL noch aktiv
        pd.Timestamp("2024-06-30", tz="UTC"),  # MSFT entfernt am 2024-06-30 (exclusive)
        None,  # GOOGL noch aktiv
    ],
})
```

### Dateispeicherung

**Pfad:**
- `data/universe/{universe_name}_history.parquet` (bevorzugt)
- `data/universe/{universe_name}_history.csv` (Fallback)
- `data/universe/{universe_name}_history.json` (Fallback)

**Format-Praeferenz:**
1. Parquet (spaltenorientiert, schnell, komprimiert)
2. CSV (human-readable, einfach zu editieren)
3. JSON (flexibel, aber langsamer)

---

## Funktionen

### get_universe_members()

**Zweck:** Gibt Liste der Symbole zurueck, die am `as_of` Datum im Universe sind.

**Input:**
- `as_of`: pd.Timestamp (UTC, Zeitpunkt fuer Membership-Aufloesung)
- `universe_name`: str (Name des Universes, default: "default")
- `root`: Path | None (Root-Verzeichnis, default: repo root / "data")

**Output:**
- `list[str]`: Liste der Symbole (uppercase, sortiert)

**Regel:**
- `start_date <= as_of` AND (`end_date is None` OR `end_date > as_of`)
- `end_date` ist EXCLUSIVE

**Beispiel:**
```python
from src.assembled_core.data.universe import get_universe_members

as_of = pd.Timestamp("2024-03-15", tz="UTC")
members = get_universe_members(as_of, universe_name="sp500")
# members = ["AAPL", "GOOGL", "MSFT", ...]  # Nur aktive Symbole
```

### load_universe_history()

**Zweck:** Laedt Universe-Membership-History aus Datei.

**Input:**
- `universe_name`: str (Name des Universes, default: "default")
- `root`: Path | None (Root-Verzeichnis, default: repo root / "data")

**Output:**
- `pd.DataFrame`: DataFrame mit columns: symbol, start_date, end_date

**Format-Support:**
- Parquet (bevorzugt)
- CSV (Fallback)
- JSON (Fallback)

### store_universe_history()

**Zweck:** Speichert Universe-Membership-History in Datei.

**Input:**
- `history`: pd.DataFrame (columns: symbol, start_date, end_date)
- `universe_name`: str (Name des Universes, default: "default")
- `root`: Path | None (Root-Verzeichnis, default: repo root / "data")
- `format`: str ("parquet", "csv", oder "json", default: "parquet")

**Output:**
- `Path`: Pfad zur geschriebenen Datei

---

## Integration

### Daily/Backtest Entry Points

**Regel:**
- Entry Points (scripts/run_daily.py, scripts/run_backtest_strategy.py) koennen Universe via `as_of` aufloesen
- Keine Provider-Calls (nur lokale Dateien)
- Deterministisch: Gleiche Eingabe -> gleiche Ausgabe

**Beispiel:**
```python
from src.assembled_core.data.universe import get_universe_members

# In run_daily.py oder run_backtest_strategy.py
as_of = pd.Timestamp("2024-01-15", tz="UTC")
universe_symbols = get_universe_members(as_of, universe_name="sp500")

# Verwende universe_symbols fuer Preis-Laden, Signal-Generierung, etc.
```

---

## Tests

### Universe Change Day Test

**Szenario:**
- Symbol AAPL: start_date=2024-01-01, end_date=2024-06-30 (exclusive)
- Symbol MSFT: start_date=2024-01-01, end_date=None (noch aktiv)
- Symbol GOOGL: start_date=2024-07-01, end_date=None (noch aktiv)

**Erwartung:**
- `as_of = 2024-06-29`: [AAPL, MSFT] (AAPL noch aktiv, GOOGL noch nicht hinzugefuegt)
- `as_of = 2024-06-30`: [MSFT] (AAPL entfernt am 2024-06-30, GOOGL noch nicht hinzugefuegt)
- `as_of = 2024-07-01`: [MSFT, GOOGL] (GOOGL hinzugefuegt am 2024-07-01)

---

## Deterministismus

### UTC-Normalisierung

**Regel:**
- Alle Timestamps sind UTC-aware
- Naive Timestamps werden als UTC interpretiert
- Konsistente Sortierung: symbol, dann start_date

### Reproduzierbarkeit

**Regel:**
- Gleiche Eingabe (as_of, universe_name) -> gleiche Ausgabe (Symbol-Liste)
- Sortierung: Alphabetisch (uppercase)
- Keine zufaelligen Elemente

---

## ASCII-Only Policy

Dieses Dokument ist ASCII-only (keine Umlaute). Verwende:
- ae statt ä
- oe statt ö
- ue statt ü
- ss statt ß

---

## Roadmap

**Sprint 4 / C2 (aktuell):**
- ✅ Universe-Membership-History (CSV/Parquet/JSON)
- ✅ get_universe_members() mit as_of Filter
- ✅ end_date EXCLUSIVE Semantik
- ✅ Deterministisch, UTC-normalisiert

**Sprint 5+ (Zukunft):**
- Integration in Factor Store
- Universe-History aus externen Quellen (z.B. S&P 500 History)
- Automatische Universe-Updates
