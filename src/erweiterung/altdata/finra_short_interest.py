"""FINRA Short Interest — bi-weekly Daten (frei, offizielle Veröffentlichung).

Quelle
------
FINRA veröffentlicht Short-Interest-Daten zwei Mal pro Monat:
    https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data

API: https://api.finra.org/data/group/otcMarket/name/regShoDaily (frei, ohne Key,
auch öffentliche CSV-Listings unter https://cdn.finra.org/equity/regsho/daily/).

PIT-Hinweis
-----------
FINRA-Daily-Reports enthalten "ShortVolume / TotalVolume" je Symbol je Tag.
Veröffentlichung: ~T+1, also frühestens am Folgetag verwendbar.

Short-Interest (das halbmonatlich gemeldete Aggregat) hat Settlement-Verzögerung
von ~10 Werktagen — siehe ``settlement_lag_days`` Argument.
"""

from __future__ import annotations

import io
import logging

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

_DAILY_URL = "https://cdn.finra.org/equity/regsho/daily/CNMSshvol{date}.txt"


@rate_limited(min_interval_s=0.5)
@retry_with_backoff(max_attempts=3, base_delay=2.0)
def _fetch_daily_file(date: pd.Timestamp) -> pd.DataFrame:
    """Hole eine Tagesdatei (kein Wochenende; Feiertage 404)."""
    import requests

    url = _DAILY_URL.format(date=date.strftime("%Y%m%d"))
    r = requests.get(url, headers={"User-Agent": "AssembledTradingAI/0.1"}, timeout=20)
    if r.status_code == 404:
        return pd.DataFrame()
    r.raise_for_status()
    text = r.text
    # Letzte Zeile ist Trailer "File-Trailer..."; ignorieren.
    df = pd.read_csv(io.StringIO(text), sep="|", skipfooter=1, engine="python")
    if df.empty:
        return df
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d", utc=True)
    return df


def fetch_finra_daily_short(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    use_cache: bool = True,
) -> FetchResult:
    """Hole tägliche FINRA Reg-SHO-Daten für einen Zeitraum.

    Returns:
        FetchResult mit DataFrame [date, symbol, market, short_volume, total_volume,
        short_ratio (pit-shifted)].
    """
    start_ts = to_utc_date(start)
    end_ts = to_utc_date(end)
    cache_key = stable_hash("finra_daily", str(start_ts), str(end_ts))
    cache_path = get_cache_dir("finra") / f"{cache_key}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return FetchResult(df, "finra_short", pd.Timestamp.utcnow(), len(df), "cache")

    rng = pd.bdate_range(start_ts, end_ts, tz="UTC")
    frames: list[pd.DataFrame] = []
    for d in rng:
        try:
            df_d = _fetch_daily_file(d)
        except Exception as e:  # noqa: BLE001
            logger.info("[finra] %s skip: %s", d.date(), e)
            continue
        if not df_d.empty:
            frames.append(df_d)

    if not frames:
        df = pd.DataFrame()
    else:
        df = pd.concat(frames, ignore_index=True)
        df = df.rename(
            columns={
                "Date": "date",
                "Symbol": "symbol",
                "Market": "market",
                "ShortVolume": "short_volume",
                "ShortExemptVolume": "short_exempt_volume",
                "TotalVolume": "total_volume",
            }
        )
        df["short_ratio"] = (
            (df["short_volume"] / df["total_volume"])
            .replace([float("inf"), float("-inf")], 0.0)
            .fillna(0.0)
        )
        df = df.sort_values(["symbol", "date"])

    if use_cache and not df.empty:
        df.to_parquet(cache_path, index=False)
    return FetchResult(df, "finra_short", pd.Timestamp.utcnow(), len(df), "")


def short_pressure_signal(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Z-Score des Short-Ratios als "Short Pressure"-Indikator.

    Ein hoher Wert bedeutet, dass der heutige Short-Ratio überdurchschnittlich
    hoch ist — historisch ein bärisches kurzfristiges Signal mit Mean-Reversion-
    Tendenz auf 1-2-Wochen-Horizont (siehe Boehmer/Jones/Zhang 2008).
    """
    if df.empty:
        return df.assign(short_pressure=pd.Series(dtype=float))
    out = df.sort_values(["symbol", "date"]).copy()
    grp = out.groupby("symbol", group_keys=False)
    # PIT: heute publizierter Wert ist (D+1) tatsächlich verfügbar
    out["short_ratio_pit"] = grp["short_ratio"].shift(1)
    out["mean"] = grp["short_ratio_pit"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback // 2).mean()
    )
    out["std"] = grp["short_ratio_pit"].transform(
        lambda s: s.rolling(lookback, min_periods=lookback // 2).std()
    )
    out["short_pressure"] = (out["short_ratio_pit"] - out["mean"]) / out["std"]
    return out


__all__ = ["fetch_finra_daily_short", "short_pressure_signal"]
