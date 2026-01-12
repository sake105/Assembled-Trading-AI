# Time and Calendar Policy - Assembled Trading AI

**Status:** Verbindlich (muss bei Code-Aenderungen befolgt werden)
**Letzte Aktualisierung:** 2025-01-04

## Zweck

Dieses Dokument definiert die **verbindliche Zeit- und Kalender-Policy** fuer das Assembled Trading AI Projekt. Es legt fest, wie Timestamps normalisiert werden, insbesondere fuer EOD (End-of-Day) Runs.

---

## TZ-Policy: UTC intern

**Regel:** Alle Timestamps sind intern in UTC.

- Alle `timestamp` Spalten in DataFrames sind UTC (timezone-aware)
- Alle `as_of` Parameter sind UTC (timezone-aware)
- Alle Logs und Reports verwenden UTC
- Keine lokalen Zeitzonen in internen Datenstrukturen

**Begruendung:** UTC ist eindeutig, vermeidet DST-Probleme, und ist Standard in Finanzdaten.

---

## as_of Semantics

### 1d (Daily) Frequency

**Regel:** `as_of` ist der **Session Close Timestamp (UTC)**, nicht "irgendein Timestamp am Tag".

**Session Close:**
- NYSE Session Close: 16:00 ET (Eastern Time)
- Winter (EST, UTC-5): 16:00 ET = **21:00 UTC**
- Summer (EDT, UTC-4): 16:00 ET = **20:00 UTC**

**Normalisierung:**
- `as_of` muss auf Session Close normalisiert werden (via `normalize_as_of_to_session_close()`)
- Entry Points (daily/backtest) normalisieren `as_of` automatisch fuer `freq=="1d"`
- Hard Gate: Wenn `freq=="1d"` und `as_of` nicht Session Close ist, wird `ValueError` geworfen

**Beispiel:**
```python
from src.assembled_core.data.calendar import normalize_as_of_to_session_close

# Eingabe: Datum oder Timestamp
as_of = normalize_as_of_to_session_close("2024-02-01")
# Ergebnis: pd.Timestamp("2024-02-01 21:00:00+00:00")  # Winter: 21:00 UTC

as_of = normalize_as_of_to_session_close("2024-06-01")
# Ergebnis: pd.Timestamp("2024-06-01 20:00:00+00:00")  # Summer: 20:00 UTC
```

### Intraday (5min) Frequency

**Regel:** `as_of` muss Bar-Ende sein (UTC), plus optional "in-session" Check.

**Bar-Ende:**
- `as_of` sollte auf 5-Minuten-Grenzen liegen (z.B. 20:00, 20:05, 20:10 UTC)
- Optional: Pruefen ob `as_of` innerhalb einer Trading-Session liegt

**Beispiel:**
```python
# 5min: as_of sollte Bar-Ende sein
as_of = pd.Timestamp("2024-02-01 20:05:00", tz="UTC")  # OK: Bar-Ende
as_of = pd.Timestamp("2024-02-01 20:03:00", tz="UTC")  # Warnung: nicht Bar-Ende
```

---

## Beispiele: Berlin Local -> UTC -> NYSE Close

### Winter (EST, UTC-5)

**Szenario:** Berlin (CET, UTC+1) -> UTC -> NYSE Close

1. Berlin Local: 2024-02-01 22:00 CET
2. UTC: 2024-02-01 21:00 UTC
3. NYSE Close: 2024-02-01 21:00 UTC (16:00 ET = 21:00 UTC im Winter)

**Code:**
```python
from datetime import datetime
import pytz

# Berlin Local
berlin_tz = pytz.timezone("Europe/Berlin")
berlin_time = datetime(2024, 2, 1, 22, 0)
berlin_aware = berlin_tz.localize(berlin_time)

# Convert to UTC
utc_time = berlin_aware.astimezone(pytz.UTC)
# utc_time = datetime(2024, 2, 1, 21, 0, tzinfo=UTC)

# Normalize to NYSE Close
from src.assembled_core.data.calendar import normalize_as_of_to_session_close
as_of = normalize_as_of_to_session_close(utc_time.date())
# as_of = pd.Timestamp("2024-02-01 21:00:00+00:00")  # Session Close
```

### Summer (EDT, UTC-4)

**Szenario:** Berlin (CEST, UTC+2) -> UTC -> NYSE Close

1. Berlin Local: 2024-06-01 22:00 CEST
2. UTC: 2024-06-01 20:00 UTC
3. NYSE Close: 2024-06-01 20:00 UTC (16:00 ET = 20:00 UTC im Sommer)

**Code:**
```python
# Berlin Local (Summer)
berlin_time = datetime(2024, 6, 1, 22, 0)
berlin_aware = berlin_tz.localize(berlin_time)

# Convert to UTC
utc_time = berlin_aware.astimezone(pytz.UTC)
# utc_time = datetime(2024, 6, 1, 20, 0, tzinfo=UTC)

# Normalize to NYSE Close
as_of = normalize_as_of_to_session_close(utc_time.date())
# as_of = pd.Timestamp("2024-06-01 20:00:00+00:00")  # Session Close
```

---

## API: src/assembled_core/data/calendar.py

### get_nyse_calendar()

Get NYSE calendar singleton (cached).

```python
from src.assembled_core.data.calendar import get_nyse_calendar

cal = get_nyse_calendar()
```

### is_trading_day(date_or_ts)

Check if a date is a trading day.

```python
from src.assembled_core.data.calendar import is_trading_day
from datetime import date

# Weekend
assert not is_trading_day(date(2024, 1, 6))  # Saturday

# Holiday
assert not is_trading_day(date(2024, 12, 25))  # Christmas

# Regular trading day
assert is_trading_day(date(2024, 1, 2))  # Tuesday
```

### session_close_utc(session_date)

Get session close timestamp in UTC.

```python
from src.assembled_core.data.calendar import session_close_utc
from datetime import date

# Winter: 21:00 UTC
close = session_close_utc(date(2024, 2, 1))
assert close.hour == 21
assert close.tz.zone == "UTC"

# Summer: 20:00 UTC
close = session_close_utc(date(2024, 6, 1))
assert close.hour == 20
```

### normalize_as_of_to_session_close(as_of, *, exchange="XNYS")

Normalize as_of to session close (UTC).

```python
from src.assembled_core.data.calendar import normalize_as_of_to_session_close

# Date string
as_of = normalize_as_of_to_session_close("2024-02-01")

# Timestamp (extracts date, normalizes to close)
as_of = normalize_as_of_to_session_close(pd.Timestamp("2024-02-01 12:00", tz="UTC"))
# Result: pd.Timestamp("2024-02-01 21:00:00+00:00")
```

### trading_sessions(start, end)

Get trading sessions between start and end dates.

```python
from src.assembled_core.data.calendar import trading_sessions
from datetime import date

sessions = trading_sessions(date(2024, 1, 1), date(2024, 1, 31))
# Returns: pd.DatetimeIndex with trading session dates
```

---

## Integration in Entry Points

### run_daily.py

```python
from src.assembled_core.data.calendar import normalize_as_of_to_session_close

# In parse_target_date or main():
target_date = parse_target_date(date_str)
if freq == "1d":
    as_of = normalize_as_of_to_session_close(target_date)
else:
    as_of = target_date  # Intraday: use as-is
```

### run_backtest_strategy.py

```python
from src.assembled_core.data.calendar import normalize_as_of_to_session_close

# In run_backtest_from_args():
if freq == "1d":
    # Normalize start_date and end_date to session closes
    start_ts = normalize_as_of_to_session_close(start_date)
    end_ts = normalize_as_of_to_session_close(end_date)
else:
    # Intraday: use as-is
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC")
```

### run_trading_cycle (Hard Gate)

```python
from src.assembled_core.data.calendar import session_close_utc

# In run_trading_cycle(), after ctx.as_of is set:
if ctx.freq == "1d" and ctx.as_of is not None:
    # Check if as_of is session close
    expected_close = session_close_utc(ctx.as_of.date())
    if ctx.as_of != expected_close:
        raise ValueError(
            f"as_of ({ctx.as_of}) must be session close ({expected_close}) for freq=1d"
        )
```

---

## DST-Check (kritisch)

**Winter (EST, UTC-5):**
- Session Close: 16:00 ET = **21:00 UTC**
- Beispiel: 2024-02-01 (Februar, Standard Time)

**Summer (EDT, UTC-4):**
- Session Close: 16:00 ET = **20:00 UTC**
- Beispiel: 2024-06-01 (Juni, Daylight Time)

**Tests muessen beide Faeelle abdecken:**
- Test mit Februar-Datum (Winter)
- Test mit Juni-Datum (Summer)
- Verifizieren: `session_close_utc().hour` ist 21 (Winter) oder 20 (Summer)

---

## Holidays

**Beispiele fuer NYSE Holidays:**
- New Year's Day: 2024-01-01
- Christmas: 2024-12-25
- Independence Day: 2024-07-04
- Thanksgiving: 2024-11-28

**Tests muessen Holidays abdecken:**
- `is_trading_day(holiday)` sollte `False` sein
- `session_close_utc(holiday)` sollte `ValueError` werfen

---

## Referenzen

- `src/assembled_core/data/calendar.py` - Calendar Utilities
- `docs/ARCHITECTURE_LAYERING.md` - Layer-Architektur
- `docs/PROJECT_STRUCTURE.md` - Repo-Struktur

---

## Aenderungen an der Policy

**Wichtig:** Die TZ-Policy ist **verbindlich** und sollte nicht ohne triftigen Grund geaendert werden.

**Prozess fuer Aenderungen:**
1. Aenderung in `docs/TIME_AND_CALENDAR.md` dokumentieren
2. Alle betroffenen Module anpassen
3. Tests aktualisieren (DST + Holidays)
4. Integrationstests ausfuehren
