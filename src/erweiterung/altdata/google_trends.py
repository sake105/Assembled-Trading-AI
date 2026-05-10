"""Google Trends — Search-Attention Signal.

Quelle
------
Inoffizielle pytrends-Bibliothek (https://github.com/GeneralMills/pytrends).
**Wichtig:** Google rate-limitet aggressiv und ändert das Format häufig. Daher:
- konservatives Rate-Limit
- mehrere Retry-Stufen
- Cache forciert
- bei Ausfall: leeres FetchResult statt Crash

Hintergrund: Preis & Da (2014, *Journal of Finance*) zeigten, dass Google-Suchen
("Search Volume Index", SVI) Aktienrenditen kurzfristig prognostizieren — ein
viel zitierter Befund.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from erweiterung._base import (
    FetchResult,
    get_cache_dir,
    retry_with_backoff,
    stable_hash,
    to_utc_date,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrendsConfig:
    """Konfiguration für Google-Trends-Abruf."""

    geo: str = "US"
    timeframe_days: int = 365  # max ~5 Jahre für daily
    chunk_size: int = 5  # Google erlaubt max 5 keywords pro Anfrage
    sleep_between_chunks: float = 5.0


@retry_with_backoff(max_attempts=2, base_delay=10.0, max_delay=60.0)
def _fetch_one_chunk(keywords: list[str], timeframe: str, geo: str) -> pd.DataFrame:
    """Hole eine Charge von max. 5 Keywords."""
    try:
        from pytrends.request import TrendReq  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "pytrends not installed; install via `pip install pytrends`"
        ) from e

    pytrends = TrendReq(hl="en-US", tz=0, retries=2, backoff_factor=0.3)
    pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo=geo)
    df = pytrends.interest_over_time()
    if df is None or df.empty:
        return pd.DataFrame()
    if "isPartial" in df.columns:
        df = df[~df["isPartial"]]
        df = df.drop(columns=["isPartial"])
    return df


def fetch_google_trends(
    keywords: Sequence[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    config: Optional[TrendsConfig] = None,
    use_cache: bool = True,
) -> FetchResult:
    """Hole Google-Trends-SVI für eine Liste von Keywords.

    Args:
        keywords: Suchbegriffe (z. B. Firmennamen oder Ticker).
        start, end: Datumsgrenzen.
        config: TrendsConfig (default).
        use_cache: Disk-Cache.

    Returns:
        FetchResult mit DataFrame [date, keyword, svi].
    """
    config = config or TrendsConfig()
    start_ts = to_utc_date(start)
    end_ts = to_utc_date(end)

    cache_key = stable_hash(
        "google_trends",
        tuple(sorted(keywords)),
        str(start_ts),
        str(end_ts),
        config.geo,
    )
    cache_path = get_cache_dir("google_trends") / f"{cache_key}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return FetchResult(
            df=df,
            source="google_trends",
            as_of=pd.Timestamp.utcnow(),
            rows=len(df),
            notes="cache",
        )

    timeframe = f"{start_ts.strftime('%Y-%m-%d')} {end_ts.strftime('%Y-%m-%d')}"
    chunks = [
        list(keywords)[i : i + config.chunk_size]
        for i in range(0, len(keywords), config.chunk_size)
    ]

    out_frames: list[pd.DataFrame] = []
    for i, chunk in enumerate(chunks):
        try:
            df_chunk = _fetch_one_chunk(chunk, timeframe, config.geo)
        except Exception as e:  # noqa: BLE001
            logger.warning("[gtrends] chunk %d/%d failed: %s", i + 1, len(chunks), e)
            continue
        if df_chunk.empty:
            continue
        long = (
            df_chunk.reset_index()
            .melt(id_vars=["date"], var_name="keyword", value_name="svi")
            .dropna()
        )
        out_frames.append(long)
        if i + 1 < len(chunks):
            time.sleep(config.sleep_between_chunks)

    if not out_frames:
        df = pd.DataFrame(columns=["date", "keyword", "svi"])
    else:
        df = pd.concat(out_frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.sort_values(["keyword", "date"]).reset_index(drop=True)

    if use_cache and not df.empty:
        df.to_parquet(cache_path, index=False)

    return FetchResult(
        df=df,
        source="google_trends",
        as_of=pd.Timestamp.utcnow(),
        rows=len(df),
        notes="",
    )


def trends_zscore(
    df: pd.DataFrame, lookback: int = 30, shift_days: int = 1
) -> pd.DataFrame:
    """SVI -> rolling z-score je Keyword (PIT-shift)."""
    if df.empty:
        return df.assign(svi_z=pd.Series(dtype=float))
    out = df.copy().sort_values(["keyword", "date"])
    grp = out.groupby("keyword", group_keys=False)
    out["svi_pit"] = grp["svi"].shift(shift_days)
    out["svi_mean"] = grp["svi_pit"].transform(
        lambda s: s.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
    )
    out["svi_std"] = grp["svi_pit"].transform(
        lambda s: s.rolling(lookback, min_periods=max(5, lookback // 3)).std()
    )
    out["svi_z"] = (out["svi_pit"] - out["svi_mean"]) / out["svi_std"]
    return out


__all__ = ["TrendsConfig", "fetch_google_trends", "trends_zscore"]
