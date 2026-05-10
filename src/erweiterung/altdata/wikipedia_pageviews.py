"""Wikipedia Pageviews — Aufmerksamkeitssignal für Aktien-Symbole.

Hintergrund
-----------
Moat, Curme, Avakian, Stanley, Stanley & Preis (2013, *Scientific Reports*) zeigten,
dass Wikipedia-Pageviews für Finanztitel mit Marktbewegungen korrelieren — ein
"Attention"-Signal, das in akademischer Literatur mehrfach repliziert wurde.

Quelle
------
Offizielle Wikimedia REST API:
    https://wikimedia.org/api/rest_v1/metrics/pageviews/...

Kostenlos, kein API-Key. Public-Cache, sehr stabil. Tagesgranularität ist
typischerweise mit ~24h Verzögerung verfügbar — entsprechend PIT-versetzt.

PIT-Hinweis
-----------
Die API liefert Pageviews **am Tag X**. Für PIT-Backtests muss der Wert frühestens
am Tag X+1 (00:00 UTC) als Feature einfließen — siehe ``shift=1`` im Wrapper unten.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

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

_BASE_URL = (
    "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
    "en.wikipedia.org/all-access/all-agents/"
    "{article}/daily/{start}/{end}"
)


@dataclass(frozen=True)
class WikiArticleMap:
    """Mapping Symbol -> Wikipedia-Artikeltitel.

    Für die meisten US-Tickers ist der Firmenname (mit Underscore) korrekt:
    "AAPL" -> "Apple_Inc."
    "MSFT" -> "Microsoft"
    """

    mapping: dict[str, str]

    def article_for(self, symbol: str) -> Optional[str]:
        return self.mapping.get(symbol.upper())


DEFAULT_MAP = WikiArticleMap(
    mapping={
        "AAPL": "Apple_Inc.",
        "MSFT": "Microsoft",
        "GOOGL": "Alphabet_Inc.",
        "GOOG": "Alphabet_Inc.",
        "AMZN": "Amazon_(company)",
        "NVDA": "Nvidia",
        "META": "Meta_Platforms",
        "TSLA": "Tesla,_Inc.",
        "JPM": "JPMorgan_Chase",
        "BRK.B": "Berkshire_Hathaway",
        "V": "Visa_Inc.",
        "MA": "Mastercard",
        "JNJ": "Johnson_%26_Johnson",
        "WMT": "Walmart",
        "PG": "Procter_%26_Gamble",
        "UNH": "UnitedHealth_Group",
        "HD": "Home_Depot",
        "BAC": "Bank_of_America",
        "XOM": "ExxonMobil",
        "CVX": "Chevron_Corporation",
        "PFE": "Pfizer",
        "KO": "The_Coca-Cola_Company",
        "PEP": "PepsiCo",
        "DIS": "The_Walt_Disney_Company",
        "NFLX": "Netflix",
        "AMD": "AMD",
        "INTC": "Intel",
        "CSCO": "Cisco",
        "ORCL": "Oracle_Corporation",
        "CRM": "Salesforce",
        "ADBE": "Adobe_Inc.",
        "PYPL": "PayPal",
        "BA": "Boeing",
        "GE": "General_Electric",
        "F": "Ford_Motor_Company",
        "GM": "General_Motors",
        "T": "AT%26T",
        "VZ": "Verizon",
    }
)


@rate_limited(min_interval_s=0.2)  # konservativ; API erlaubt mehr
@retry_with_backoff(max_attempts=3, base_delay=1.5)
def _fetch_one(article: str, start: str, end: str) -> list[dict]:
    """Hole Pageviews für **einen** Artikel.

    Args:
        article: URL-encoded Wikipedia-Artikeltitel.
        start, end: YYYYMMDD-Strings.

    Returns:
        Liste aus Dicts ``{article, granularity, timestamp, access, agent, views}``.
    """
    import requests  # lazy import — Modul auch ohne Netzwerk importierbar

    url = _BASE_URL.format(article=article, start=start, end=end)
    headers = {
        "User-Agent": (
            "AssembledTradingAI-Erweiterung/0.1 "
            "(research; contact: hans.oertel2@gmail.com)"
        ),
        "Accept": "application/json",
    }
    r = requests.get(url, headers=headers, timeout=15)
    if r.status_code == 404:
        # Article unbekannt -> kein Fehler, leeres Ergebnis
        logger.info("[wiki] 404 for article=%s (no pageviews)", article)
        return []
    r.raise_for_status()
    payload = r.json()
    return payload.get("items", [])


def fetch_wikipedia_pageviews(
    symbols: Sequence[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    article_map: Optional[WikiArticleMap] = None,
    use_cache: bool = True,
) -> FetchResult:
    """Hole Wikipedia-Pageviews für eine Symbol-Liste.

    Args:
        symbols: Tickerliste, z. B. ``['AAPL', 'MSFT']``.
        start, end: Datumsgrenzen (inklusive).
        article_map: Optional. Default DEFAULT_MAP.
        use_cache: Disk-Cache aktivieren (Parquet).

    Returns:
        FetchResult mit DataFrame ``[date, symbol, article, views, log_views]``.
    """
    article_map = article_map or DEFAULT_MAP
    start_ts = to_utc_date(start)
    end_ts = to_utc_date(end)
    if end_ts < start_ts:
        raise ValueError("end must be >= start")

    cache_key = stable_hash(
        "wikipedia", tuple(sorted(symbols)), str(start_ts), str(end_ts)
    )
    cache_path = get_cache_dir("wikipedia") / f"{cache_key}.parquet"
    if use_cache and cache_path.exists():
        logger.debug("[wiki] cache hit: %s", cache_path)
        df = pd.read_parquet(cache_path)
        return FetchResult(
            df=df,
            source="wikipedia",
            as_of=pd.Timestamp.utcnow(),
            rows=len(df),
            notes="cache",
        )

    rows: list[dict] = []
    missing: list[str] = []
    s_str = start_ts.strftime("%Y%m%d")
    e_str = end_ts.strftime("%Y%m%d")
    for sym in symbols:
        article = article_map.article_for(sym)
        if not article:
            missing.append(sym)
            continue
        try:
            items = _fetch_one(article, s_str, e_str)
        except Exception as e:  # noqa: BLE001 — boundary
            logger.warning("[wiki] fetch failed for %s: %s", sym, e)
            continue
        for it in items:
            ts = it.get("timestamp", "")
            if len(ts) < 8:
                continue
            d = pd.Timestamp(f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}", tz="UTC")
            rows.append(
                {
                    "date": d,
                    "symbol": sym.upper(),
                    "article": article,
                    "views": int(it.get("views", 0)),
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df["log_views"] = (df["views"].astype(float) + 1.0).pipe(_safe_log)
        df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    if use_cache and not df.empty:
        df.to_parquet(cache_path, index=False)

    notes = f"missing_articles={len(missing)}" if missing else ""
    return FetchResult(
        df=df,
        source="wikipedia",
        as_of=pd.Timestamp.utcnow(),
        rows=len(df),
        notes=notes,
    )


def _safe_log(s: pd.Series) -> pd.Series:
    import numpy as np

    return pd.Series(np.log(s.clip(lower=1e-9)), index=s.index)


def attention_score(
    pageviews_df: pd.DataFrame,
    lookback: int = 30,
    shift_days: int = 1,
) -> pd.DataFrame:
    """Berechne Aufmerksamkeitsscore = log(views_t / mean(views over lookback)).

    Args:
        pageviews_df: Output von ``fetch_wikipedia_pageviews``.
        lookback: Vergleichsfenster in Tagen.
        shift_days: PIT-Schutz; Pageviews von Tag X werden frühestens an X+shift_days nutzbar.

    Returns:
        DataFrame mit Spalte ``attention_score`` (per symbol/date).
    """
    if pageviews_df.empty:
        return pageviews_df.assign(attention_score=pd.Series(dtype=float))

    df = pageviews_df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["symbol", "date"])

    grouped = df.groupby("symbol", group_keys=False)
    df["log_views_pit"] = grouped["log_views"].shift(shift_days)
    df["log_views_mean"] = grouped["log_views_pit"].transform(
        lambda s: s.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
    )
    df["log_views_std"] = grouped["log_views_pit"].transform(
        lambda s: s.rolling(lookback, min_periods=max(5, lookback // 3)).std()
    )
    df["attention_score"] = (df["log_views_pit"] - df["log_views_mean"]) / df[
        "log_views_std"
    ]
    return df


__all__ = [
    "WikiArticleMap",
    "DEFAULT_MAP",
    "fetch_wikipedia_pageviews",
    "attention_score",
]
