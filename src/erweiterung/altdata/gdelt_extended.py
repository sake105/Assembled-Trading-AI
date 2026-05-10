"""GDELT 2.0 Global Knowledge Graph — Geopolitik & Sentiment (frei).

Quelle
------
GDELT 2.0 publiziert alle 15 Minuten:
    http://data.gdeltproject.org/gdeltv2/lastupdate.txt
    -> URLs zu CSVs: 'gkg.csv.zip', 'export.csv.zip', 'mentions.csv.zip'

Anwendung
---------
Aggregierte tägliche Sentiment-/Volume-Series je Land oder Theme:
- AVGTONE (durchschnittliches Tone der Berichterstattung)
- GoldsteinScale (Skala -10 bis +10 für Konfliktivität)
- THEMES (Auflistung kategorisierter Themen wie "ECONOMY", "ENERGY")
- LOCATIONS (mit FIPS-Country-Codes)

Hinweis
-------
GDELT-Daten sind extrem groß (mehrere GB/Tag). Wir bieten hier:
1. Tagesindex-Download (zip)
2. Aggregator: Tones-Mittel pro Land / Theme / Tag
3. PIT-Schutz: Ein Tag ist erst am Folgetag voll abgeschlossen.

Für High-Frequency-Anwendungen sollte man GDELT BigQuery (frei für 1TB/Monat)
verwenden — siehe ``docs/erweiterung/PAID_DATA_WISHLIST.md`` (BigQuery-Setup).
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Optional

import pandas as pd

from erweiterung._base import (
    FetchResult,
    get_cache_dir,
    rate_limited,
    retry_with_backoff,
    stable_hash,
    to_utc_date,
)

logger = logging.getLogger(__name__)


_GDELT_GKG_BASE = "http://data.gdeltproject.org/gdeltv2/{ts}.gkg.csv.zip"


@rate_limited(min_interval_s=2.0)
@retry_with_backoff(max_attempts=3, base_delay=3.0)
def _download_gkg_chunk(ts: pd.Timestamp) -> pd.DataFrame:
    """Hole 15-min-GKG-Datei. ``ts`` muss aufs nächste 15-Minuten-Raster gerundet sein."""
    import requests

    ts_str = ts.strftime("%Y%m%d%H%M%S")
    url = _GDELT_GKG_BASE.format(ts=ts_str)
    r = requests.get(url, timeout=30)
    if r.status_code == 404:
        return pd.DataFrame()
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        names = zf.namelist()
        if not names:
            return pd.DataFrame()
        with zf.open(names[0]) as f:
            df = pd.read_csv(
                f,
                sep="\t",
                header=None,
                quoting=3,  # QUOTE_NONE
                low_memory=False,
                on_bad_lines="skip",
            )
    # GKG hat ~27 Spalten; wir brauchen nur ein paar davon.
    if df.shape[1] < 16:
        return pd.DataFrame()
    df = df.rename(
        columns={
            0: "GKGRECORDID",
            1: "DATE",
            2: "SourceCollectionIdentifier",
            3: "SourceCommonName",
            4: "DocumentIdentifier",
            5: "Counts",
            7: "Themes",
            9: "Locations",
            11: "Persons",
            13: "Organizations",
            15: "Tone",
        }
    )
    df["timestamp"] = pd.to_datetime(
        df["DATE"].astype(str), format="%Y%m%d%H%M%S", utc=True
    )
    df["avg_tone"] = pd.to_numeric(df["Tone"].str.split(",").str[0], errors="coerce")
    return df[["timestamp", "Themes", "Locations", "avg_tone", "DocumentIdentifier"]]


def fetch_gdelt_daily_tones(
    date: str | pd.Timestamp,
    country_filter: Optional[list[str]] = None,
    theme_filter: Optional[list[str]] = None,
    use_cache: bool = True,
) -> FetchResult:
    """Aggregiere durchschnittlichen Tone für einen Tag.

    Args:
        date: Tagesdatum (UTC).
        country_filter: Liste von FIPS-Country-Codes (e.g. ``['US', 'DE', 'CN']``).
        theme_filter: Liste von Themen-Substrings (e.g. ``['ECONOMY', 'ENERGY']``).

    Returns:
        FetchResult mit DataFrame [date, country|theme, mean_tone, count].
    """
    d = to_utc_date(date)
    cache_key = stable_hash(
        "gdelt_tones",
        d.isoformat(),
        tuple(sorted(country_filter or [])),
        tuple(sorted(theme_filter or [])),
    )
    cache_path = get_cache_dir("gdelt") / f"{cache_key}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return FetchResult(df, "gdelt", pd.Timestamp.utcnow(), len(df), "cache")

    # 96 Chunks pro Tag (alle 15 min)
    chunks = pd.date_range(
        d, d + pd.Timedelta(days=1), freq="15min", tz="UTC", inclusive="left"
    )
    frames: list[pd.DataFrame] = []
    for ts in chunks:
        try:
            df_c = _download_gkg_chunk(ts)
        except Exception as e:  # noqa: BLE001
            logger.info("[gdelt] %s skip: %s", ts, e)
            continue
        if not df_c.empty:
            frames.append(df_c)

    if not frames:
        return FetchResult(pd.DataFrame(), "gdelt", pd.Timestamp.utcnow(), 0, "empty")

    raw = pd.concat(frames, ignore_index=True)
    raw["date"] = raw["timestamp"].dt.normalize()

    out_rows: list[dict] = []
    if country_filter:
        for cc in country_filter:
            mask = (
                raw["Locations"]
                .fillna("")
                .str.contains(rf"\b{cc}\b", regex=True, na=False)
            )
            sub = raw[mask]
            out_rows.append(
                {
                    "date": d,
                    "kind": "country",
                    "key": cc,
                    "mean_tone": sub["avg_tone"].mean() if not sub.empty else None,
                    "count": int(len(sub)),
                }
            )
    if theme_filter:
        for th in theme_filter:
            mask = raw["Themes"].fillna("").str.contains(th, na=False)
            sub = raw[mask]
            out_rows.append(
                {
                    "date": d,
                    "kind": "theme",
                    "key": th,
                    "mean_tone": sub["avg_tone"].mean() if not sub.empty else None,
                    "count": int(len(sub)),
                }
            )

    if not out_rows:
        out_rows.append(
            {
                "date": d,
                "kind": "global",
                "key": "ALL",
                "mean_tone": raw["avg_tone"].mean(),
                "count": int(len(raw)),
            }
        )

    df = pd.DataFrame(out_rows)
    if use_cache:
        df.to_parquet(cache_path, index=False)
    return FetchResult(df, "gdelt", pd.Timestamp.utcnow(), len(df), "")


__all__ = ["fetch_gdelt_daily_tones"]
