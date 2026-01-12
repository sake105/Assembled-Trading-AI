# Data Quality Control (QC) - Assembled Trading AI

**Status:** Verbindlich (muss bei Code-Aenderungen befolgt werden)
**Letzte Aktualisierung:** 2025-01-04

## Zweck

Dieses Dokument definiert die **Data Quality Control (QC) Policy** fuer das Assembled Trading AI Projekt.

**Wichtig:** QC ist ein **Data Hygiene Gate**, nicht Strategy Evaluation. QC prueft die Qualitaet der Eingabedaten (Preise/Panels), nicht die Performance einer Strategie.

---

## QC-Modul: src/assembled_core/qa/data_qc.py

### Zweck

Das QC-Modul bietet deterministische, side-effect-freie Qualitaetspruefungen fuer Preis-Panels.

**Key Functions:**
- `run_price_panel_qc()`: Fuehrt alle QC-Checks auf einem Preis-Panel aus
- `qc_report_to_dict()`: Konvertiert QcReport zu Dictionary
- `write_qc_report_json()`: Schreibt QC-Report in JSON-Datei
- `write_qc_summary_md()`: Schreibt QC-Summary in Markdown-Datei (ASCII-only)

---

## Check-Liste + Default Thresholds

### 1. Invalid Prices (FAIL)

**Check:** `invalid_prices`

**Beschreibung:** Prueft auf ungueltige Preise (close <= 0 oder NaN in required Feldern).

**Severity:** FAIL (immer)

**Details:**
- NaN in required columns (`timestamp`, `symbol`, `close`) -> FAIL
- `close <= 0` -> FAIL

**Threshold:** Keine (immer FAIL)

### 2. Duplicate Rows (FAIL)

**Check:** `duplicate_rows`

**Beschreibung:** Prueft auf doppelte Zeilen (identische `symbol,timestamp` Kombination).

**Severity:** FAIL (immer)

**Details:**
- Anzahl der Duplikate pro `symbol,timestamp` Kombination

**Threshold:** Keine (immer FAIL)

### 3. Missing Sessions (WARN/FAIL)

**Check:** `missing_sessions`

**Beschreibung:** Prueft auf fehlende Trading-Sessions (nur `freq=="1d"`).

**Severity:**
- WARN: Wenn >5% der Trading-Tage fehlen
- FAIL: Wenn >20% der Trading-Tage fehlen

**Details:**
- Anzahl fehlender Sessions
- Prozentsatz fehlender Sessions
- Erwartete vs. tatsaechliche Anzahl

**Thresholds:**
- `missing_sessions_warn_pct`: 5.0 (default)
- `missing_sessions_fail_pct`: 20.0 (default)

**Hinweis:** Nutzt Calendar-Modul (`is_trading_day`, `trading_sessions`) fuer erwartete Tage.

### 4. Stale Prices (WARN)

**Check:** `stale_price`

**Beschreibung:** Prueft auf "stale" Preise (Preis unveraendert fuer >= N Sessions).

**Severity:** WARN

**Details:**
- Anzahl Sessions mit unveraendertem Preis
- Preis-Wert
- Start- und End-Timestamp

**Threshold:**
- `stale_price_sessions`: 3 (default)

### 5. Outlier Returns (WARN/FAIL)

**Check:** `outlier_return`

**Beschreibung:** Prueft auf Ausreisser-Renditen (abs(daily_return) >= threshold).

**Severity:**
- WARN: Wenn abs(return) >= 20%
- FAIL: Wenn abs(return) >= 30%

**Details:**
- Return-Wert (absolut und relativ)
- Verwendeter Threshold
- Preis und vorheriger Preis

**Thresholds:**
- `outlier_return_warn`: 0.20 (20%, default)
- `outlier_return_fail`: 0.30 (30%, default)

### 6. Zero Volume Anomalies (WARN/FAIL)

**Check:** `zero_volume`

**Beschreibung:** Prueft auf Zero-Volume-Anomalien (volume==0 an Trading-Tagen).

**Severity:**
- WARN: Wenn >10% der Trading-Tage Zero-Volume haben
- FAIL: Wenn >50% der Trading-Tage Zero-Volume haben

**Details:**
- Anzahl Zero-Volume-Tage
- Prozentsatz Zero-Volume-Tage
- Gesamtanzahl Tage

**Thresholds:**
- `zero_volume_warn_pct`: 10.0 (default)
- `zero_volume_fail_pct`: 50.0 (default)

**Hinweis:** Nur wenn `volume` Spalte vorhanden ist.

---

## Semantik "WARN vs FAIL"

### FAIL

**Bedeutung:** Kritischer Fehler, der die Datenqualitaet fundamental beeintraechtigt.

**Beispiele:**
- Negative oder Null-Preise
- Doppelte Zeilen
- Sehr viele fehlende Sessions (>20%)
- Sehr hohe Returns (>30%)

**Konsequenz:** `report.ok = False`

### WARN

**Bedeutung:** Potenzielle Qualitaetsprobleme, die ueberprueft werden sollten.

**Beispiele:**
- Einige fehlende Sessions (5-20%)
- Stale Preise (>=3 Sessions unveraendert)
- Moderate Returns (20-30%)
- Einige Zero-Volume-Tage (10-50%)

**Konsequenz:** `report.ok = True` (wenn keine FAIL-Issues)

---

## Determinismus + TZ Policy

### Determinismus

**Regel:** Issues werden stabil sortiert:
1. Severity (FAIL vor WARN)
2. Check-Name (alphabetisch)
3. Symbol (alphabetisch, None -> leerer String)
4. Timestamp (chronologisch, None -> min Timestamp)

**Begruendung:** Reproduzierbare Reports, einfache Vergleiche.

### TZ Policy (UTC)

**Regel:** Alle Timestamps im Report sind UTC (timezone-aware).

**Normalisierung:**
- Naive Timestamps werden als UTC interpretiert
- Timezone-aware Timestamps werden zu UTC konvertiert
- `created_at_utc` ist immer UTC

**Begruendung:** Konsistenz mit restlicher Pipeline (siehe `docs/TIME_AND_CALENDAR.md`).

---

## Beispiel: Entry Points Integration (D2)

**Hinweis:** Dies ist ein Beispiel fuer zukuenftige Integration (D2), nicht Teil von D1.

### run_daily.py

```python
from src.assembled_core.qa.data_qc import run_price_panel_qc, write_qc_report_json

# Nach dem Laden der Preise:
prices = load_eod_prices(...)

# QC ausfuehren
qc_report = run_price_panel_qc(
    prices=prices,
    freq="1d",
    calendar="NYSE",
    as_of=target_timestamp,
)

# Report schreiben
write_qc_report_json(
    qc_report,
    path=output_dir / "qc_report.json"
)

# Bei FAIL: Warnung oder Abbruch
if not qc_report.ok:
    logger.warning(f"QC failed: {qc_report.summary['fail_count']} FAIL issues")
    # Optional: sys.exit(1) fuer hard gate
```

### run_backtest_strategy.py

```python
# Nach dem Laden der Preise:
prices = load_eod_prices(...)

# QC ausfuehren
qc_report = run_price_panel_qc(
    prices=prices,
    freq=args.freq,
    calendar="NYSE",
)

# Report schreiben
write_qc_report_json(
    qc_report,
    path=output_dir / "qc_report.json"
)

# Bei FAIL: Warnung (Backtest kann trotzdem laufen, aber mit Warnung)
if not qc_report.ok:
    logger.warning(f"QC failed: {qc_report.summary['fail_count']} FAIL issues")
```

---

## API-Referenz

### run_price_panel_qc()

```python
def run_price_panel_qc(
    prices: pd.DataFrame,
    *,
    freq: str,
    calendar: str = "NYSE",
    as_of: pd.Timestamp | None = None,
    thresholds: dict[str, Any] | None = None,
) -> QcReport:
    """Run quality control checks on a price panel.

    Args:
        prices: Price DataFrame with columns: timestamp, symbol, close, ... (volume optional)
        freq: Frequency string ("1d" or "5min")
        calendar: Calendar name (default: "NYSE", currently only NYSE supported)
        as_of: Optional cutoff timestamp for PIT-safe checks
        thresholds: Optional threshold dictionary (overrides defaults)

    Returns:
        QcReport with all issues found
    """
```

### QcReport

```python
@dataclass
class QcReport:
    ok: bool  # True if no FAIL issues, False otherwise
    summary: dict[str, Any]  # Summary statistics
    issues: list[QcIssue]  # List of QC issues (sorted deterministically)
    created_at_utc: pd.Timestamp  # Report creation timestamp (UTC)
```

### QcIssue

```python
@dataclass
class QcIssue:
    check: str  # Check name (e.g., "missing_sessions", "negative_price")
    severity: Literal["WARN", "FAIL"]  # Severity level
    symbol: str | None  # Symbol affected (None if panel-wide)
    timestamp: pd.Timestamp | None  # Timestamp affected (None if symbol-wide or panel-wide)
    message: str  # Human-readable message
    details: dict[str, Any]  # Additional details (JSON-serializable)
```

---

## QC as QA Gate (Sprint 3 / D2)

**Zweck:** QC-Ergebnisse werden als QA-Gate verwendet, um Trading zu blockieren wenn Datenqualitaet kritisch ist.

### Semantik

**FAIL -> Block Trading -> Orders Empty:**
- Wenn QC-Report `ok=False` (FAIL-Issues vorhanden), wird `qa_block_trading=True` gesetzt
- Trading Cycle blockt Orders (setzt auf leeres DataFrame)
- Backtest laeuft deterministisch mit 0 Trades

**WARN -> Proceed:**
- Wenn QC-Report nur WARN-Issues hat, wird Trading nicht blockiert
- Backtest/Pipeline laeuft normal weiter

### Integration

**Entry Points (scripts/):**
- `scripts/run_daily.py`: QC nach Preis-Laden, Gate-Flags in TradingContext
- `scripts/run_backtest_strategy.py`: QC nach Preis-Laden, Gate-Flags in TradingContext Template

**Pipeline (src/assembled_core/pipeline/trading_cycle.py):**
- Gate-Hook: Nach Order-Generation, vor Risk-Controls
- Wenn `ctx.qa_block_trading=True`: Orders auf empty setzen
- Reason in `result.meta["qa_block_reason"]` speichern

**Layering:**
- Pipeline importiert kein qa (Layer-Regel)
- Entry Points (scripts/) duerfen qa importieren
- Gate-Flags werden extern gesetzt, intern geprueft

### Logging / Output

**QC Report:**
- `output/qc_report.json` (deterministic JSON)
- Optional: `output/qc_summary.md` (ASCII-only Markdown)

**Gate-Reason:**
- In `TradingCycleResult.meta["qa_block_reason"]`
- In Logs als WARNING

**Beispiel:**
```python
# In scripts/run_daily.py:
qc_report = run_price_panel_qc(prices, freq="1d", calendar="NYSE")
write_qc_report_json(qc_report, output_dir / "qc_report.json")

if not qc_report.ok:
    qa_block_trading = True
    qa_block_reason = f"DATA_QC_FAIL: {qc_report.summary['fail_count']} FAIL issues"

ctx = TradingContext(..., qa_block_trading=qa_block_trading, qa_block_reason=qa_block_reason)
result = run_trading_cycle(ctx)

# result.orders ist leer wenn qa_block_trading=True
# result.meta["qa_block_reason"] enthaelt den Grund
```

---

## Referenzen

- `src/assembled_core/qa/data_qc.py` - QC-Modul
- `docs/TIME_AND_CALENDAR.md` - TZ-Policy
- `docs/CONTRACTS.md` - Data Contracts
- `src/assembled_core/data/calendar.py` - Calendar Utilities

---

## Aenderungen an der Policy

**Wichtig:** Die QC-Policy ist **verbindlich** und sollte nicht ohne triftigen Grund geaendert werden.

**Prozess fuer Aenderungen:**
1. Aenderung in `docs/DATA_QUALITY_QC.md` dokumentieren
2. QC-Modul anpassen
3. Tests aktualisieren
4. Integrationstests ausfuehren
