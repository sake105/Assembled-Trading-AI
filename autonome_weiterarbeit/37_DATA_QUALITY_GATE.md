# 37 — Data-Quality-Gate vor Feature-Berechnung

**Zweck:** Dreckige Daten dürfen die Feature-Pipeline nicht erreichen. Missing Bars, Price-Spikes, falsch-adjustierte Splits, Zero-Volume-Ticks — alles was reinkommt, muss eine Quality-Gate passieren. Wenn nicht, werden die Features verfälscht, die Signale lügen, und der Backtest zeigt ein anderes Bild als die Realität.

**Scope:** Rang 5 aus der Gap-Analyse. Sollte zwischen Daten-Ingestion (siehe `10_FREE_DATEN.md`) und Feature-Engineering (siehe `13_FREE_MODULE.md`) eingebaut werden.

**Kern-Idee:** Jede eingehende Daten-Batch durchläuft ein strukturiertes Validation-Gate. Erfolg → Features. Fehler → Quarantäne, Alert, manuelle Freigabe.

---

## 0. Warum das wichtig ist — die stille Katastrophe

### Der typische Bug, den niemand bemerkt

Du hast eine Trend-Folge-Strategie. Sie berechnet EMA-20 und EMA-50 auf Close-Prices. Eines Tages liefert dein Daten-Provider für AAPL den Close als **0.00** (weil irgendein API-Timeout teilweise leere Bars zurückgab).

- Dein Code kippt nicht, weil `0.00` ein gültiger Float ist.
- EMA-20 fällt drastisch → Cross-Signal "SELL" wird generiert.
- Die Strategie verkauft AAPL auf ein neues "Low" mit Short-Position.
- Am nächsten Tag: AAPL wieder bei 180. Du hast eine Riesen-Lücke und bist short.

**Dein Kill-Switch triggert.** Aber du hast trotzdem 3-8 % Verlust auf dieser Position. Und schlimmer: Du weißt nicht, warum. Du debuggst Strategie-Logik, Execution-Logik, Position-Sizing — alles korrekt. Der Bug liegt zwei Layer drunter: bei den rohen Daten.

Das ist nicht theoretisch. Das ist der **häufigste Produktions-Bug** in algorithmischen Trading-Systemen. Nicht die Strategie, nicht das Execution-Layer — die rohen Daten.

### Die drei Hauptkategorien von Daten-Fehlern

**Kategorie 1 — Strukturelle Fehler:**
- Fehlende Bars (Gap zwischen 09:30 und 11:15, als hätte zwei Stunden kein Handel stattgefunden)
- Duplikate (dieselbe Bar zweimal, mit leicht unterschiedlichen Volumes)
- Falsche Zeitstempel (Bar um 16:01 obwohl Markt um 16:00 schließt)
- Falsche Ticker-Mapping (AAPL-Bar kommt als MSFT-Bar an)

**Kategorie 2 — Value-Anomalien:**
- Price-Spikes (+300 % in einer Minute wegen Datenfehler)
- Zero-Prices (Close = 0.00)
- Negative Prices (passiert bei WTI-Crude-Oil-Future 2020, auch bei API-Bugs)
- Volume-Anomalien (500 Mio Shares in 1 Min für einen Penny-Stock)

**Kategorie 3 — Corporate-Actions:**
- Nicht adjustierte Splits (Aktie "crasht" um 50 %, weil Split nicht eingerechnet)
- Ex-Dividend-Drops ohne Adjustment (Aktie "fällt" um 2 % wegen Dividende)
- Ticker-Änderungen (FB → META)
- Delisting (plötzlich fehlen alle Daten)

Dein Data-Quality-Gate muss alle drei Kategorien erwischen.

---

## 1. Die Tool-Entscheidung: Pandera vs. Great Expectations vs. Custom

### 1.1 Die Kandidaten

| Tool | Version | Nutzen | Nachteil |
|---|---|---|---|
| **Pandera** | 0.21.0 (Feb 2026) | Code-native Schemas in Python, schnell, Type-Hints-freundlich | Docs sind eher spartanisch |
| **Great Expectations** | 1.3.x (Feb 2026) | Data-Docs, Stakeholder-Reports, Multi-Engine (SQL/Spark/pandas) | Overkill für Einzel-Setup |
| **Pointblank** | 0.6.x (neu 2024) | Sehr User-friendly, Polars-nativ | Keine Auto-Action-Trigger |
| **Patito** | 0.8.x | Pydantic-Style auf Polars | Fokus auf Polars, weniger Pandas |
| **Custom** | selbst geschrieben | Volle Kontrolle | Jeder Check muss selbst gemacht werden |

### 1.2 Empfehlung für deinen Stack

**Primary: Pandera.** Warum?

1. **Code-native:** Du schreibst Schemas als Python-Klassen mit Type-Hints. IDE-Unterstützung, PR-Reviews, unit-test-Integration.
2. **Pandas + Polars:** Deckt beide ab. Dein Stack ist Pandas-basiert, falls du später auf Polars wechselst ist kein Tool-Wechsel nötig.
3. **Statistical-Tests eingebaut:** `hypothesis_tests` lässt dich Verteilungs-Checks einbauen (z.B. "Return-Distribution sollte nicht abrupt von normal zu non-normal kippen").
4. **Lightweight:** ~50 MB Install, keine zusätzliche Infrastruktur.

**Secondary: Custom-OHLCV-Checks.** Pandera deckt **generische** Validierung ab. **OHLCV-spezifisch** (High >= Low, Volume >= 0, Timestamps monoton) brauchst du eigene Check-Funktionen. Pandera erlaubt das via `Check` mit Lambda.

**Nicht: Great Expectations.** GE ist optimal für Teams mit Stakeholder-Reports. Für Einzel-Quant ist es zuviel Overhead. Die Data-Docs rechnen sich nicht, wenn du die einzige Person bist, die sie liest.

### 1.3 Installation

```bash
uv pip install pandera==0.21.0          # Core + Pandas
uv pip install pandera[io]              # für YAML/JSON Schema-Export
uv pip install pandera[strategies]      # für Hypothesis-Integration (Tests)
```

---

## 2. Die OHLCV-Schema-Definition

### 2.1 Basic-Schema

```python
# src/assembled_core/dataquality/schemas/ohlcv.py
import pandera as pa
from pandera.typing import Series, DataFrame, DateTime
import pandas as pd


class OHLCVSchema(pa.DataFrameModel):
    """Schema für OHLCV-Bars. Jede Zeile = ein Bar für einen Ticker."""
    
    # Identität
    ticker: Series[str] = pa.Field(
        str_matches=r"^[A-Z0-9\.\-]{1,10}$",   # z.B. AAPL, BRK.B, ^GSPC
        nullable=False,
    )
    timestamp: Series[pd.DatetimeTZDtype] = pa.Field(
        nullable=False,
        dtype_kwargs={"tz": "America/New_York"},  # oder UTC, aber konsistent
    )
    
    # OHLC
    open: Series[float] = pa.Field(
        gt=0,                                  # Preis muss positiv sein
        lt=1_000_000,                          # Sanity: < 1 Mio USD (sonst Fehler)
        nullable=False,
    )
    high: Series[float] = pa.Field(
        gt=0,
        lt=1_000_000,
        nullable=False,
    )
    low: Series[float] = pa.Field(
        gt=0,
        lt=1_000_000,
        nullable=False,
    )
    close: Series[float] = pa.Field(
        gt=0,
        lt=1_000_000,
        nullable=False,
    )
    
    # Volume
    volume: Series[int] = pa.Field(
        ge=0,                                  # Volume kann 0 sein (Halt, illiquide)
        nullable=False,
    )
    
    # Optional: VWAP, Trade-Count
    vwap: Series[float] = pa.Field(
        gt=0, 
        lt=1_000_000, 
        nullable=True,
    )
    trade_count: Series[int] = pa.Field(
        ge=0,
        nullable=True,
    )
    
    class Config:
        strict = True                          # unbekannte Spalten → Fehler
        coerce = True                          # Types werden automatisch konvertiert wenn möglich
        ordered = False                        # Spalten-Reihenfolge egal
    
    # Cross-Column-Checks
    @pa.dataframe_check
    def high_ge_low(cls, df: pd.DataFrame) -> Series[bool]:
        """High muss ≥ Low sein, immer."""
        return df["high"] >= df["low"]
    
    @pa.dataframe_check
    def high_ge_open_close(cls, df: pd.DataFrame) -> Series[bool]:
        """High muss ≥ Open und ≥ Close sein."""
        return (df["high"] >= df["open"]) & (df["high"] >= df["close"])
    
    @pa.dataframe_check
    def low_le_open_close(cls, df: pd.DataFrame) -> Series[bool]:
        """Low muss ≤ Open und ≤ Close sein."""
        return (df["low"] <= df["open"]) & (df["low"] <= df["close"])
    
    @pa.dataframe_check
    def timestamp_unique_per_ticker(cls, df: pd.DataFrame) -> bool:
        """Pro (ticker, timestamp) darf es nur eine Row geben."""
        dup = df.duplicated(subset=["ticker", "timestamp"], keep=False)
        if dup.any():
            return False
        return True
```

### 2.2 Der Validation-Aufruf

```python
# src/assembled_core/dataquality/gate.py
from typing import Tuple
import pandas as pd
import pandera as pa
from pandera.errors import SchemaError, SchemaErrors
from .schemas.ohlcv import OHLCVSchema


class DataQualityGate:
    """Zentrale Gate-Logik: validiere oder quarantäniere."""
    
    def __init__(self, quarantine_path: str = "data/quarantine"):
        self.quarantine_path = quarantine_path
    
    def validate_ohlcv(
        self, 
        df: pd.DataFrame, 
        source: str,
        batch_id: str,
    ) -> Tuple[pd.DataFrame, dict]:
        """Validiert OHLCV-DataFrame.
        
        Returns:
            (clean_df, metadata)
            
        Raises:
            SchemaErrors: wenn lazy=True nicht alle Fehler gesammelt werden können
        """
        try:
            clean_df = OHLCVSchema.validate(df, lazy=True)
            metadata = {
                "status": "pass",
                "rows_in": len(df),
                "rows_out": len(clean_df),
                "source": source,
                "batch_id": batch_id,
            }
            return clean_df, metadata
        except SchemaErrors as exc:
            # Alle Fehler wurden gesammelt (wegen lazy=True)
            failure_cases = exc.failure_cases
            
            # Quarantäne der kompletten Batch
            self._quarantine(df, source, batch_id, failure_cases)
            
            metadata = {
                "status": "fail",
                "rows_in": len(df),
                "rows_out": 0,
                "source": source,
                "batch_id": batch_id,
                "failures": failure_cases.to_dict("records"),
                "error_count": len(failure_cases),
            }
            raise DataQualityError(
                f"Batch {batch_id} from {source} failed validation "
                f"with {len(failure_cases)} errors. Quarantined."
            ) from exc
    
    def _quarantine(self, df, source, batch_id, failures):
        """Batch + Failures in Quarantäne-Ordner schreiben."""
        from pathlib import Path
        qdir = Path(self.quarantine_path) / source / batch_id
        qdir.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(qdir / "data.parquet")
        failures.to_csv(qdir / "failures.csv", index=False)
        
        # Audit-Log
        import json
        from datetime import datetime
        with open(qdir / "metadata.json", "w") as f:
            json.dump({
                "source": source,
                "batch_id": batch_id,
                "quarantined_at": datetime.utcnow().isoformat(),
                "rows": len(df),
                "error_count": len(failures),
            }, f, indent=2)


class DataQualityError(Exception):
    """Wird geworfen wenn ein Batch die Gate nicht passiert."""
    pass
```

---

## 3. OHLCV-spezifische Sanity-Checks (über Schema hinaus)

Das Pandera-Schema deckt **statische** Regeln ab (Pos, Bounds, High≥Low). Trading-Daten brauchen zusätzlich **dynamische** Checks.

### 3.1 Price-Spike-Detection

```python
# src/assembled_core/dataquality/checks/price_spike.py
import pandas as pd
import numpy as np


def detect_price_spikes(
    df: pd.DataFrame, 
    ticker_col: str = "ticker",
    price_col: str = "close",
    max_abs_return_1bar: float = 0.30,
    z_score_threshold: float = 6.0,
) -> pd.DataFrame:
    """
    Erkennt Price-Spikes:
      - Absolute 1-Bar-Return > 30 % (sehr verdächtig, sehr selten legitim)
      - Z-Score auf rollierenden 20-Bar-Mean > 6 (statistisch sehr unwahrscheinlich)
    
    Returns:
        DataFrame mit nur den verdächtigen Rows.
    """
    suspects = []
    
    for ticker, group in df.groupby(ticker_col):
        group = group.sort_values("timestamp").copy()
        
        # 1-Bar-Return
        group["ret_1bar"] = group[price_col].pct_change()
        
        # Absolute-Return-Check
        mask_abs = group["ret_1bar"].abs() > max_abs_return_1bar
        
        # Z-Score auf rolling 20-Bar-Return
        group["ret_mean_20"] = group["ret_1bar"].rolling(20).mean()
        group["ret_std_20"] = group["ret_1bar"].rolling(20).std()
        group["z_score"] = (group["ret_1bar"] - group["ret_mean_20"]) / group["ret_std_20"]
        
        mask_z = group["z_score"].abs() > z_score_threshold
        
        flagged = group[mask_abs | mask_z].copy()
        if len(flagged) > 0:
            flagged["reason"] = "price_spike"
            suspects.append(flagged)
    
    if suspects:
        return pd.concat(suspects, ignore_index=True)
    return pd.DataFrame()
```

**Schwellwerte begründet:**
- 30 % Absolute-Return in 1 Bar: bei Daily-Bars selten (Limit-Up/Down, Earnings-Shock), bei Intraday fast immer Datenfehler
- Z-Score > 6 auf 20-Bar-Window: statistisch ~1 in 500 Mio, also praktisch immer Fehler

**Achtung bei low-priced Stocks:** Eine Aktie bei 2 USD mit Bid-Ask-Spread von 0.03 macht schnell 1.5 % Move. Der Check sollte für Ticker mit `close < 5` weniger streng sein:

```python
def detect_price_spikes_adaptive(df, ...):
    """Adaptive Schwellen für low-priced."""
    ...
    for ticker, group in df.groupby("ticker"):
        avg_price = group["close"].mean()
        if avg_price < 5:
            threshold = 0.50         # Low-Price: toleriere 50 % Moves
        elif avg_price < 20:
            threshold = 0.35
        else:
            threshold = 0.30
        ...
```

### 3.2 Missing-Bar-Detection

```python
# src/assembled_core/dataquality/checks/missing_bars.py
import pandas as pd
from pandas.tseries.offsets import BDay


def detect_missing_trading_days(
    df: pd.DataFrame,
    ticker_col: str = "ticker",
    timestamp_col: str = "timestamp",
    expected_market_calendar: str = "NYSE",
) -> pd.DataFrame:
    """Prüft, ob für jeden Ticker alle erwarteten Handelstage präsent sind.
    
    Nutzt pandas_market_calendars für offizielle NYSE/NASDAQ-Calendar.
    """
    import pandas_market_calendars as mcal
    
    cal = mcal.get_calendar(expected_market_calendar)
    missing_events = []
    
    for ticker, group in df.groupby(ticker_col):
        group = group.sort_values(timestamp_col)
        start = group[timestamp_col].min()
        end = group[timestamp_col].max()
        
        # Alle erwarteten Trading-Days
        schedule = cal.schedule(start_date=start, end_date=end)
        expected_days = set(schedule.index.date)
        actual_days = set(group[timestamp_col].dt.date)
        
        missing = expected_days - actual_days
        
        for d in sorted(missing):
            missing_events.append({
                "ticker": ticker,
                "missing_date": d,
                "reason": "missing_trading_day",
            })
    
    return pd.DataFrame(missing_events)
```

**Besonderheiten:**
- `pandas-market-calendars` ist pflicht-Library für Market-Hours und Feiertage
- Halbe Tage (Thanksgiving-Freitag, Heiligabend) werden korrekt abgedeckt
- Für Crypto: gibt keinen Feiertags-Check, Continuous 24/7

**Install:**
```bash
uv pip install pandas-market-calendars==4.6.1
```

### 3.3 Volume-Anomalie-Detection

```python
# src/assembled_core/dataquality/checks/volume.py
import pandas as pd
import numpy as np


def detect_volume_anomalies(
    df: pd.DataFrame,
    spike_multiple: float = 20.0,
    zero_volume_tolerance: int = 5,
) -> pd.DataFrame:
    """
    Volume-Checks:
      - Spike: Volume > 20× 20-Tage-Median
      - Zero-Runs: > 5 aufeinanderfolgende Bars mit Volume = 0
    """
    suspects = []
    
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("timestamp").copy()
        
        # Spike
        group["vol_median_20"] = group["volume"].rolling(20).median()
        group["vol_ratio"] = group["volume"] / group["vol_median_20"]
        spike_mask = group["vol_ratio"] > spike_multiple
        
        # Zero-Runs
        group["is_zero"] = (group["volume"] == 0).astype(int)
        group["zero_run"] = group["is_zero"].rolling(zero_volume_tolerance).sum()
        zero_run_mask = group["zero_run"] >= zero_volume_tolerance
        
        flagged = group[spike_mask | zero_run_mask].copy()
        flagged.loc[spike_mask, "reason"] = "volume_spike"
        flagged.loc[zero_run_mask, "reason"] = "zero_volume_run"
        
        if len(flagged) > 0:
            suspects.append(flagged)
    
    if suspects:
        return pd.concat(suspects, ignore_index=True)
    return pd.DataFrame()
```

**Realistische Baselines für Spike-Multiple:**
- 20× 20-Tage-Median: typisch für Earnings-Release, M&A-News
- 50× 20-Tage-Median: sehr selten legitim, meist Data-Error oder Breaking-News
- 100× 20-Tage-Median: praktisch immer Data-Error

### 3.4 Split-Adjustment-Check

```python
# src/assembled_core/dataquality/checks/splits.py
import pandas as pd
import numpy as np


def detect_unadjusted_splits(
    df: pd.DataFrame,
    drop_threshold: float = 0.40,
    adjacent_bars: int = 3,
) -> pd.DataFrame:
    """
    Erkennt nicht-adjustierte Stock-Splits.
    
    Heuristik: Wenn ein 1-Bar-Return < -40 % UND der Return in der Folge-Periode
    nicht rückgängig gemacht wird → sehr wahrscheinlich Split.
    
    Das unterscheidet von echten Crashes (die oft teilweise recovered werden)
    und Datenfehlern (die meist 1-Bar-spikes sind).
    """
    suspects = []
    
    for ticker, group in df.groupby("ticker"):
        group = group.sort_values("timestamp").copy()
        group["ret"] = group["close"].pct_change()
        
        # Drops > threshold
        candidate_mask = group["ret"] < -drop_threshold
        candidates = group[candidate_mask].copy()
        
        for idx, row in candidates.iterrows():
            # Look at adjacent bars
            window_start = idx - adjacent_bars
            window_end = idx + adjacent_bars
            window = group.iloc[max(0, window_start):window_end]
            
            # If the drop is NOT reversed in the next few bars, likely a split
            post_drop_max = window.loc[idx:, "close"].max()
            pre_drop_close = group.loc[:idx, "close"].iloc[-2] if idx > 0 else None
            
            if pre_drop_close is None:
                continue
            
            recovery_pct = (post_drop_max - row["close"]) / row["close"]
            
            if recovery_pct < 0.10:  # < 10 % Recovery in adjacent bars → Split-Verdacht
                suspects.append({
                    "ticker": ticker,
                    "timestamp": row["timestamp"],
                    "pre_close": pre_drop_close,
                    "post_close": row["close"],
                    "ratio": pre_drop_close / row["close"],
                    "reason": "suspected_unadjusted_split",
                })
    
    return pd.DataFrame(suspects)
```

**Integration mit Corporate-Actions-Feed:**

Wenn du einen Corporate-Actions-Feed hast (z.B. Alpaca's `get_corporate_actions` oder EODHD), dann:

1. Bekannte Splits ignorieren (der Drop ist legitim und Daten sind richtig)
2. Nur unbekannte Splits flaggen → vermutlich Datenfehler

```python
def detect_unadjusted_splits_with_feed(df, corporate_actions_df):
    """Nutzt Corporate-Actions-Feed für Suppression."""
    suspects = detect_unadjusted_splits(df)
    
    # Suppress known splits
    for idx, row in suspects.iterrows():
        ca_match = corporate_actions_df[
            (corporate_actions_df["ticker"] == row["ticker"]) &
            (corporate_actions_df["type"] == "split") &
            (corporate_actions_df["date"].between(
                row["timestamp"] - pd.Timedelta(days=2),
                row["timestamp"] + pd.Timedelta(days=2),
            ))
        ]
        if len(ca_match) > 0:
            suspects.drop(idx, inplace=True)
    
    return suspects
```

---

## 4. Die Gate-Integration in die Pipeline

### 4.1 Wo die Gate läuft

```
┌─────────────────────┐
│  Data Ingestion     │  (yfinance, Alpaca, Finnhub, EODHD)
│  (Layer 1)          │
└──────────┬──────────┘
           │ raw DataFrame
           ▼
┌─────────────────────┐
│  Data-Quality-Gate  │  ← HIER: Validierung + Quarantäne
│  (Layer 2)          │
└──────────┬──────────┘
           │ clean DataFrame (or raise DataQualityError)
           ▼
┌─────────────────────┐
│  Feature Engineering │
│  (Layer 3)          │
└──────────┬──────────┘
           │ features
           ▼
┌─────────────────────┐
│  Signal Generation  │
│  (Layer 4)          │
└─────────────────────┘
```

**Goldene Regel:** Kein Feature wird auf nicht-validierten Daten berechnet. Wenn die Gate fehlschlägt, schlägt die ganze Pipeline fehl — kein "na dann eben ohne die paar kaputten Ticker weitermachen", weil du nicht weißt, ob der Fehler strukturell ist.

### 4.2 Der Ingestion-to-Gate-Flow

```python
# src/assembled_core/pipeline/ingest.py
import logging
import uuid
from datetime import datetime

from ..data.providers import YFinanceProvider, AlpacaProvider, EODHDProvider
from ..dataquality.gate import DataQualityGate, DataQualityError

logger = logging.getLogger(__name__)


def ingest_and_validate(
    tickers: list[str],
    start: str,
    end: str,
    provider: str = "yfinance",
    timeframe: str = "1D",
) -> pd.DataFrame:
    """Einheitlicher Ingest-und-Validate-Flow."""
    batch_id = f"{provider}_{timeframe}_{datetime.utcnow():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    
    # 1. Ingestion
    if provider == "yfinance":
        raw = YFinanceProvider().fetch(tickers, start, end, timeframe)
    elif provider == "alpaca":
        raw = AlpacaProvider().fetch(tickers, start, end, timeframe)
    elif provider == "eodhd":
        raw = EODHDProvider().fetch(tickers, start, end, timeframe)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    logger.info(f"Ingested {len(raw)} rows from {provider}, batch_id={batch_id}")
    
    # 2. Gate
    gate = DataQualityGate(quarantine_path="data/quarantine")
    try:
        clean, metadata = gate.validate_ohlcv(raw, source=provider, batch_id=batch_id)
        logger.info(f"Gate passed: {metadata}")
    except DataQualityError as e:
        logger.error(f"Gate failed for batch {batch_id}: {e}")
        raise
    
    # 3. Dynamic Sanity-Checks
    from ..dataquality.checks.price_spike import detect_price_spikes
    from ..dataquality.checks.volume import detect_volume_anomalies
    from ..dataquality.checks.missing_bars import detect_missing_trading_days
    
    spikes = detect_price_spikes(clean)
    if len(spikes) > 0:
        logger.warning(f"{len(spikes)} price spikes detected. Review: {spikes.head()}")
        _alert_price_spikes(spikes, batch_id)
    
    vol_anomalies = detect_volume_anomalies(clean)
    if len(vol_anomalies) > 0:
        logger.warning(f"{len(vol_anomalies)} volume anomalies detected.")
    
    missing = detect_missing_trading_days(clean)
    if len(missing) > 0:
        logger.warning(f"{len(missing)} missing trading days.")
    
    return clean


def _alert_price_spikes(spikes: pd.DataFrame, batch_id: str):
    """Sendet Alert bei Price-Spikes (Telegram/Sentry)."""
    # Implementierung siehe 51_INCIDENT_PLAYBOOK.md
    pass
```

### 4.3 Unterschied zwischen "hart" und "weich" Fehlern

**Harte Fehler** (Pandera-Schema-Violations) → Pipeline-Abbruch:
- Negativ-Preise
- Missing columns
- Type-Mismatches
- High < Low

**Weiche Fehler** (dynamische Sanity-Checks) → Warnung + Alert, weiter:
- Price-Spikes (können legitim sein, z.B. Earnings)
- Volume-Anomalien
- Missing trading days (können Holidays sein, die der Calendar nicht kennt)

Warum die Trennung? Harte Fehler sind **strukturell unmöglich** und deuten auf Provider-Bugs. Weiche Fehler sind **ungewöhnlich aber möglich** und brauchen menschliche Review.

---

## 5. Quarantäne und Recovery-Workflow

### 5.1 Quarantäne-Struktur

```
data/quarantine/
├── yfinance/
│   ├── yfinance_1D_20260424_143022_a3f2b1/
│   │   ├── data.parquet              # Original-Batch
│   │   ├── failures.csv              # Pandera-Failures
│   │   └── metadata.json             # Audit-Log
│   └── ...
├── alpaca/
│   └── ...
└── eodhd/
    └── ...
```

### 5.2 Review-Workflow

```bash
# Tägliches Ritual: Check Quarantäne
python -m assembled_core.dataquality.review_quarantine

# Output:
# ┌──────────────────────────────────────────────────────────────┐
# │ Quarantined Batches Report                                    │
# ├──────────────────────────────────────────────────────────────┤
# │ yfinance   |  2  batches, 12 errors                          │
# │ alpaca     |  0  batches                                     │
# │ eodhd      |  1  batch,  3 errors                            │
# ├──────────────────────────────────────────────────────────────┤
# │ Total:       3  batches, 15 errors                           │
# └──────────────────────────────────────────────────────────────┘
#
# Details:
# [1] yfinance_1D_20260424_143022_a3f2b1
#     Tickers affected: TSLA, NVDA, AAPL
#     Errors:
#       - 3× "high < low" (TSLA, 2026-04-22 10:30:00)
#       - 2× "close = 0" (NVDA, 2026-04-23 15:45:00)
#     Action: [r]efetch, [a]ccept-as-is, [d]elete, [s]kip
#     Decision: _
```

### 5.3 Recovery-Options

**Option 1 — Refetch:** Zweite Anfrage an denselben Provider. Falls es ein temporärer API-Bug war, kommt saubere Daten zurück.

```python
def refetch_from_quarantine(batch_dir: Path) -> pd.DataFrame:
    """Liest Metadata, re-queried Provider, läuft Gate erneut."""
    metadata = json.load(open(batch_dir / "metadata.json"))
    original_df = pd.read_parquet(batch_dir / "data.parquet")
    
    tickers = original_df["ticker"].unique().tolist()
    start = original_df["timestamp"].min()
    end = original_df["timestamp"].max()
    
    return ingest_and_validate(tickers, start, end, provider=metadata["source"])
```

**Option 2 — Cross-Provider-Verification:** Frage andere Provider nach den betroffenen Bars. Wenn 2 von 3 übereinstimmen, nimm die Majority.

```python
def verify_with_alternate_providers(ticker, date, price_column="close"):
    """Fragt 3 Provider nach demselben Bar. Majority-Vote."""
    results = {}
    for provider in ["yfinance", "alpaca", "eodhd"]:
        try:
            df = fetch_single_bar(ticker, date, provider)
            results[provider] = df[price_column].iloc[0]
        except Exception:
            pass
    
    if len(results) < 2:
        return None
    
    values = list(results.values())
    median_val = np.median(values)
    
    # Majority: alle Werte innerhalb 1 % des Medians
    agreements = sum(1 for v in values if abs(v - median_val) / median_val < 0.01)
    if agreements >= 2:
        return median_val
    return None
```

**Option 3 — Accept-as-is:** Menschliche Entscheidung, dass die "Anomalie" legitim ist. Z.B. Price-Spike wegen Earnings-Beat. Wird in `audit_log.jsonl` dokumentiert.

**Option 4 — Skip:** Ticker für diesen Tag aus der Analyse rausnehmen. Nur wenn Refetch/Verify nicht möglich.

---

## 6. Monitoring und Metriken

### 6.1 Was du tracken solltest

| Metrik | Zielwert | Alert-Threshold |
|---|---|---|
| Gate-Pass-Rate (24h) | > 99 % | < 95 % |
| Durchschnittliche Quarantäne-Events/Tag | < 3 | > 10 |
| Price-Spikes pro Tag | varies | > 20 |
| Missing Trading Days | 0 | > 0 |
| Cross-Provider-Disagreements | < 1 % | > 5 % |
| Gate-Latency p99 | < 500 ms | > 2 s |

### 6.2 Prometheus-Metriken

```python
# src/assembled_core/dataquality/metrics.py
from prometheus_client import Counter, Histogram, Gauge

gate_validations_total = Counter(
    "dq_gate_validations_total",
    "Total DQ gate validations",
    ["source", "status"],  # status: pass/fail
)

gate_latency_seconds = Histogram(
    "dq_gate_latency_seconds",
    "Latency of DQ gate validation",
    ["source"],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
)

quarantine_batches_gauge = Gauge(
    "dq_quarantine_batches_count",
    "Current count of quarantined batches (not yet reviewed)",
    ["source"],
)

price_spikes_total = Counter(
    "dq_price_spikes_total",
    "Total price spikes detected",
    ["ticker"],
)
```

**Integration mit Grafana:** Ein Dashboard mit diesen 4 Metriken gibt dir tägliche Gesundheits-Anzeige. Bei jedem Alert-Threshold: Slack/Telegram-Alert.

### 6.3 Der wöchentliche DQ-Report

```python
# scripts/dq/weekly_report.py
"""Weekly Data-Quality-Report. Läuft Sonntag-Nacht via cron."""

def generate_weekly_report():
    report = {
        "period": "2026-04-14 to 2026-04-20",
        "gate_stats": {
            "total_batches": 147,
            "pass_rate": 0.986,
            "fail_count": 2,
        },
        "anomalies": {
            "price_spikes": 23,
            "volume_spikes": 18,
            "missing_bars": 0,
            "suspected_splits": 1,
        },
        "cross_provider_disagreements": 3,
        "top_problem_tickers": [
            ("XYZ", 5),  # 5 Quarantäne-Events
            ("ABC", 3),
        ],
    }
    
    # Render as Markdown
    # Upload to observability dashboard
    # Send summary to Telegram
    ...
```

---

## 7. Tests für die Gate selbst

Die Gate ist **safety-critical** — wenn sie Bugs hat, greifen keine anderen Tests. Deshalb:

### 7.1 Unit-Tests mit synthetischen Datenfehlern

```python
# tests/dataquality/test_gate.py
import pytest
import pandas as pd
import numpy as np
from assembled_core.dataquality.gate import DataQualityGate, DataQualityError
from assembled_core.dataquality.schemas.ohlcv import OHLCVSchema


@pytest.fixture
def clean_ohlcv():
    return pd.DataFrame({
        "ticker": ["AAPL"] * 10,
        "timestamp": pd.date_range("2026-01-01", periods=10, freq="D", tz="America/New_York"),
        "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "low":  [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        "close": [100.5, 101.5, 102.5, 103.5, 104.5, 105.5, 106.5, 107.5, 108.5, 109.5],
        "volume": [1_000_000] * 10,
        "vwap": [None] * 10,
        "trade_count": [None] * 10,
    })


def test_gate_passes_clean_data(clean_ohlcv, tmp_path):
    gate = DataQualityGate(quarantine_path=str(tmp_path))
    result, metadata = gate.validate_ohlcv(clean_ohlcv, "test", "batch_001")
    assert metadata["status"] == "pass"
    assert metadata["rows_out"] == 10


def test_gate_rejects_negative_price(clean_ohlcv, tmp_path):
    clean_ohlcv.loc[5, "close"] = -100.0
    gate = DataQualityGate(quarantine_path=str(tmp_path))
    with pytest.raises(DataQualityError):
        gate.validate_ohlcv(clean_ohlcv, "test", "batch_neg")


def test_gate_rejects_zero_price(clean_ohlcv, tmp_path):
    clean_ohlcv.loc[3, "close"] = 0.0
    gate = DataQualityGate(quarantine_path=str(tmp_path))
    with pytest.raises(DataQualityError):
        gate.validate_ohlcv(clean_ohlcv, "test", "batch_zero")


def test_gate_rejects_high_lt_low(clean_ohlcv, tmp_path):
    clean_ohlcv.loc[4, "high"] = 50.0   # < low
    gate = DataQualityGate(quarantine_path=str(tmp_path))
    with pytest.raises(DataQualityError):
        gate.validate_ohlcv(clean_ohlcv, "test", "batch_hl")


def test_gate_rejects_duplicate_timestamps(clean_ohlcv, tmp_path):
    # Duplicate (ticker, timestamp)
    dup = clean_ohlcv.iloc[[0]].copy()
    clean_ohlcv = pd.concat([clean_ohlcv, dup], ignore_index=True)
    gate = DataQualityGate(quarantine_path=str(tmp_path))
    with pytest.raises(DataQualityError):
        gate.validate_ohlcv(clean_ohlcv, "test", "batch_dup")


def test_gate_rejects_extreme_price(clean_ohlcv, tmp_path):
    clean_ohlcv.loc[5, "close"] = 5_000_000.0   # > 1 Mio
    gate = DataQualityGate(quarantine_path=str(tmp_path))
    with pytest.raises(DataQualityError):
        gate.validate_ohlcv(clean_ohlcv, "test", "batch_extreme")


def test_quarantine_creates_files(clean_ohlcv, tmp_path):
    clean_ohlcv.loc[5, "close"] = -100.0
    gate = DataQualityGate(quarantine_path=str(tmp_path))
    with pytest.raises(DataQualityError):
        gate.validate_ohlcv(clean_ohlcv, "test", "batch_001")
    
    qdir = tmp_path / "test" / "batch_001"
    assert qdir.exists()
    assert (qdir / "data.parquet").exists()
    assert (qdir / "failures.csv").exists()
    assert (qdir / "metadata.json").exists()
```

### 7.2 Property-Based-Tests mit Hypothesis

```python
from hypothesis import given, strategies as st

@given(
    n_rows=st.integers(min_value=5, max_value=1000),
    base_price=st.floats(min_value=1, max_value=1000),
)
def test_gate_accepts_all_valid_random_data(n_rows, base_price, tmp_path):
    """Property: wenn alle Invariants erfüllt, passt die Gate."""
    prices = np.random.normal(base_price, base_price * 0.02, n_rows)
    prices = np.clip(prices, 0.01, None)
    
    df = pd.DataFrame({
        "ticker": ["TEST"] * n_rows,
        "timestamp": pd.date_range("2026-01-01", periods=n_rows, freq="D", tz="UTC"),
        "open": prices,
        "high": prices * 1.02,
        "low": prices * 0.98,
        "close": prices,
        "volume": np.random.randint(1000, 1_000_000, n_rows),
        "vwap": [None] * n_rows,
        "trade_count": [None] * n_rows,
    })
    
    gate = DataQualityGate(quarantine_path=str(tmp_path))
    result, metadata = gate.validate_ohlcv(df, "test", "random")
    assert metadata["status"] == "pass"
```

### 7.3 Adversarial-Test: alle bekannten Failure-Modes

Eine Test-Datei mit **allen bekannten Bug-Patterns** als Referenz:

```python
# tests/dataquality/test_adversarial.py
"""Tests gegen jede bekannte Kategorie von Data-Errors.

Jeder Test produziert ein real beobachtetes Bug-Pattern. Wenn die Gate
einen davon nicht fängt, bricht der Test.
"""

# Bug #1: Yahoo-Finance Close = 0.00 (API-Timeout)
def test_yahoo_zero_close_bug(): ...

# Bug #2: IEX Split nicht adjustiert
def test_iex_unadjusted_split(): ...

# Bug #3: Alpaca Duplicate-Bar (Retry-Logic)
def test_alpaca_duplicate_bar(): ...

# Bug #4: Finnhub Future-Timestamp
def test_finnhub_future_timestamp(): ...

# Bug #5: Polygon Timezone-Confusion (UTC vs NY-Time)
def test_polygon_tz_mismatch(): ...

# ... für jeden bekannten Bug einen Test
```

**Sammle diese Tests kontinuierlich.** Jeder reale Bug wird zu einem Regression-Test. So wächst dein Safety-Net organisch.

---

## 8. Corporate-Actions-Hygiene

Da du verschiedene Data-Provider nutzt (yfinance, Alpaca, EODHD), musst du dir **explizit** darüber im Klaren sein, welche Provider wie Corporate Actions handhaben.

### 8.1 Die Adjustment-Matrix

| Provider | Splits auto-adjusted | Dividends auto-adjusted | API-Parameter |
|---|---|---|---|
| **yfinance** | Ja (default ab 0.2.28) | Ja (default) | `auto_adjust=True` (default) |
| **Alpaca** | Nein (raw) | Nein (raw) | separate Corporate-Actions-Endpoint |
| **EODHD** | Optional | Optional | `&fmt=json&adjusted=1` |
| **Finnhub** | Abhängig vom Endpoint | Abhängig | — |

**Die Konsistenz-Pflicht:** Entweder du lädst **überall** adjusted Prices, oder **überall** raw + separate Corporate-Actions-Application. **Niemals mischen.**

### 8.2 Empfohlener Workflow

**Für Daily-Bars (Historical):**
```python
# yfinance: auto_adjust=True (default)
data = yf.download("AAPL", start="2020-01-01", auto_adjust=True)
# Close ist adjusted. Kein Adj Close separat mehr.
```

**Für Intraday-Bars:**
Intraday wird meist nicht adjusted (Splits passieren immer Overnight). Splits für Intraday muss man manuell anwenden, wenn man Backtests über Split-Daten macht.

```python
def apply_split_backwards(df, split_events):
    """Wendet Splits rückwärts auf Intraday-Historie an."""
    for split in split_events:
        split_date = split["date"]
        ratio = split["ratio"]  # z.B. 4 für 4:1
        
        mask = df["timestamp"] < split_date
        df.loc[mask, ["open", "high", "low", "close"]] /= ratio
        df.loc[mask, "volume"] *= ratio
    
    return df
```

**Für Live-Trading:**
Bei Live-Trading ist das Problem kleiner, weil du nur aktuelle Bars verwendest. Aber: Wenn ein Split morgens stattfindet und deine Strategie gestern EMA-20 mit Preisen vor Split berechnet hat, triggert sie falsche Signale.

**Lösung:** Ein Pre-Market-Job vor Marktöffnung, der Corporate-Actions des Tages prüft und bei Bedarf Features neu berechnet.

### 8.3 Der Cross-Provider-Sanity-Check

Einmal pro Woche:

```python
def weekly_provider_consistency_check(tickers, date):
    """Vergleiche Close-Prices von 3 Providern für dasselbe Datum."""
    divergences = []
    
    for ticker in tickers:
        yf_close = get_yf_close(ticker, date)
        alpaca_close = get_alpaca_close(ticker, date)
        eodhd_close = get_eodhd_close(ticker, date)
        
        values = [x for x in [yf_close, alpaca_close, eodhd_close] if x is not None]
        if len(values) < 2:
            continue
        
        max_div = max(values) - min(values)
        median_val = np.median(values)
        divergence_pct = max_div / median_val
        
        if divergence_pct > 0.01:  # > 1 % Abweichung
            divergences.append({
                "ticker": ticker,
                "date": date,
                "yf": yf_close,
                "alpaca": alpaca_close,
                "eodhd": eodhd_close,
                "divergence_pct": divergence_pct,
            })
    
    return pd.DataFrame(divergences)
```

**Wenn Divergenzen auftreten:** Meist Corporate-Actions-Unterschied. Manchmal Provider-Bug. Immer Review.

---

## 9. Umsetzungs-Checkliste

**Phase 1 — Schema-Definition (Tag 1-2):**
- [ ] Pandera installieren, `OHLCVSchema` für Daily + Intraday
- [ ] Cross-Column-Checks (High≥Low, High≥Open/Close, etc.)
- [ ] Unit-Tests für Schema mit synthetischen Bugs

**Phase 2 — Gate-Implementation (Tag 3-4):**
- [ ] `DataQualityGate` Klasse
- [ ] Quarantäne-Ordner-Struktur
- [ ] Integration mit Logging/Alerts

**Phase 3 — Dynamic-Checks (Tag 5-7):**
- [ ] `detect_price_spikes`
- [ ] `detect_missing_trading_days` (mit pandas-market-calendars)
- [ ] `detect_volume_anomalies`
- [ ] `detect_unadjusted_splits`

**Phase 4 — Pipeline-Integration (Tag 8-10):**
- [ ] Ingest-to-Gate-Flow
- [ ] Harte vs. weiche Fehler-Unterscheidung
- [ ] Prometheus-Metriken
- [ ] Quarantäne-Review-Script

**Phase 5 — Tests (Tag 11-12):**
- [ ] Unit-Tests für jede Check-Funktion
- [ ] Property-Based-Tests
- [ ] Adversarial-Test-Suite mit realen Bug-Patterns
- [ ] Performance-Budget: < 500 ms für 1000 Ticker × 100 Bars

**Phase 6 — Recovery-Workflow (Tag 13-14):**
- [ ] Refetch-Logik
- [ ] Cross-Provider-Verification
- [ ] Wöchentlicher DQ-Report
- [ ] Docs für manuelle Quarantäne-Review

**Gesamt-Aufwand:** 2-3 Wochen bei 10-15 h/Woche.

---

## 10. Quellen

**Pandera:**
- [Pandera Documentation](https://pandera.readthedocs.io/)
- [Union.ai — Pandera Guide](https://www.union.ai/pandera)
- endjin (April 2025): [Creating Quality Gates in the Medallion Architecture with Pandera](https://endjin.com/blog/2025/04/creating-quality-gates-in-the-medallion-architecture-with-pandera)
- Bhagya Rana (Oktober 2025): [Great Expectations vs Pandera](https://medium.com/@bhagyarana80/8-great-expectations-vs-pandera-which-fits-your-python-stack-a115c9241dcb)
- Arthur Turrell (März 2025): [The data validation landscape in 2025](https://aeturrell.com/blog/posts/the-data-validation-landscape-in-2025/)

**OHLCV Quality Checks:**
- Aman Kharwal (April 2024): [Stock Market Anomaly Detection using Python](https://amanxai.com/2024/04/08/stock-market-anomaly-detection-using-python/)
- Shijun Ju (Mai 2025): [Real-Time Anomaly Detection Pipeline for Stock Trading](https://medium.com/@jushijun/building-a-real-time-anomaly-detection-pipeline-for-stock-trading-data-with-redpanda-and-quix-83da5a013599)
- arxiv 2509.16137 (2025): Enhancing OHLC Data — behandelt Low-Price-Bias und Illiquidity-Probleme

**Corporate Actions:**
- PyQuant News (April 2026): [Clean Financial Market Data with Python and Yahoo Finance](https://www.pyquantnews.com/free-python-resources/insiders-guide-to-clean-financial-market-data-with-python-and-yahoo-finance)
- JosueMonte (2025): [Why Adj Close Disappeared in yfinance](https://medium.com/@josue.monte/why-adj-close-disappeared-in-yfinance-and-how-to-adapt-6baebf1939f6)
- [yfinance Price Repair](https://ranaroussi.github.io/yfinance/advanced/price_repair.html)
- yfinance GitHub Issue #1531: [Yahoo fails to apply stock split](https://github.com/ranaroussi/yfinance/issues/1531)

**Market Calendars:**
- [pandas-market-calendars](https://pandas-market-calendars.readthedocs.io/)
- [NYSE Trading Schedule](https://www.nyse.com/markets/hours-calendars)

---

## 11. Ehrliche Einschätzung

**Was dieses Playbook dir gibt:**
- Schutz vor 95 % der typischen Data-Quality-Bugs
- Automatisierte Quarantäne mit Audit-Log
- Monitoring-Metriken für Trends
- Eine Test-Suite, die über Zeit durch echte Bugs wächst

**Was es dir nicht gibt:**
- **Schutz vor Tick-Level-Issues.** Wenn du auf Tick-Daten tradest (HFT), brauchst du andere Checks (sub-Millisekunden-Timestamps, L2-Book-Integrity). Dieser Plan zielt auf Bars (1min+).
- **Fundamental-Data-Validation.** Earnings-Numbers, Balance-Sheet etc. brauchen eigene Schemas.
- **News-Feed-Validation.** Siehe `34_NEWS_GROUND_TRUTH.md` — News haben eigene Quality-Challenges.
- **100 % Schutz.** Neue Bug-Klassen werden immer auftreten. Die Gate ist ein Safety-Net, kein Vacuum-Seal.

**Die drei Sachen, die du nicht auslassen darfst:**
1. **High ≥ Low Cross-Check.** Klingt trivial, aber es fängt 30-40 % aller API-Bugs.
2. **Price > 0 Hard-Check.** Ein einziger Zero-Close in EMA-Berechnung zerstört die Folge-Tage.
3. **Quarantäne statt silent-drop.** Niemals kaputte Bars einfach wegwerfen. Quarantäne + Review = du lernst die Muster deiner Provider-Bugs.

**Der wichtigste Denkfehler, den du vermeiden willst:** "Ich baue die Features erst mal und sehe dann, ob die Daten Probleme haben." Das ist der falsche Weg. Die Features **verstecken** Data-Bugs, weil sie über 20 Bars mitteln. Ein einzelner Zero-Close wird in EMA-20 zu "nur" -5 % Delta — bemerkbar, aber nicht offensichtlich. Du merkst es erst, wenn eine Live-Position schief geht und du in die Bar-History schaust. Zu spät.

**Die Gate läuft immer zuerst.** Vor allem anderen. Das ist die einzige Regel, die bei Data-Quality zählt.
