"""SEC EDGAR Filings — Form 4, 8-K, 13D/G, 10-K/Q (frei, real-time).

Quelle
------
Offizielle SEC EDGAR-API (frei, JSON):
    https://www.sec.gov/edgar/sec-api-documentation
    https://data.sec.gov/submissions/CIK{cik}.json

Fokus
-----
- Form 4 (insider transactions): Tag-genaue Insider-Trades pro CIK.
- 8-K (current report): Material events.
- 13D / 13G (beneficial ownership): Großaktionärsmeldungen.
- 10-K / 10-Q: Periodische Disclosures.

PIT-Hinweis
-----------
EDGAR-Filings haben ``filingDate`` (= Tag, an dem öffentlich zugänglich) und
``effectiveDate`` (= materielle Wirkung, oft Tage früher). Für PIT-Backtests
ist **ausschließlich filingDate** der relevante Zeitpunkt.

API-Etikette
------------
SEC verlangt ``User-Agent`` mit Kontaktdaten und einen Rate-Limit von 10 r/s.
Wir gehen konservativ mit ``rate_limited(0.15)`` (~6.5 r/s).
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from erweiterung._base import (
    FetchResult,
    get_cache_dir,
    rate_limited,
    retry_with_backoff,
    stable_hash,
)

logger = logging.getLogger(__name__)

_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _user_agent() -> str:
    return (
        "AssembledTradingAI-Erweiterung/0.1 "
        "(research; contact: hans.oertel2@gmail.com)"
    )


@rate_limited(min_interval_s=0.15)
@retry_with_backoff(max_attempts=3, base_delay=2.0)
def _http_get_json(url: str) -> dict:
    import requests

    r = requests.get(url, headers={"User-Agent": _user_agent()}, timeout=20)
    r.raise_for_status()
    return r.json()


def get_ticker_cik_map(use_cache: bool = True) -> dict[str, str]:
    """Lade Ticker -> 10-stelligen CIK von SEC.

    Returns:
        Dict ``{TICKER: '0000320193', ...}``.
    """
    cache_path = get_cache_dir("sec_edgar") / "ticker_cik_map.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return dict(zip(df["ticker"], df["cik"]))
    payload = _http_get_json(_TICKERS_URL)
    rows = []
    # payload ist {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    for _, v in payload.items():
        cik_padded = f"{int(v['cik_str']):010d}"
        rows.append(
            {"ticker": v["ticker"].upper(), "cik": cik_padded, "title": v["title"]}
        )
    df = pd.DataFrame(rows)
    if use_cache:
        df.to_parquet(cache_path, index=False)
    return dict(zip(df["ticker"], df["cik"]))


def fetch_recent_filings(
    ticker: str,
    forms: tuple[str, ...] = ("4", "8-K", "13D", "13G", "10-K", "10-Q"),
    max_per_form: int = 200,
    use_cache: bool = True,
) -> FetchResult:
    """Hole jüngste Filings einer Firma (filtered by form types).

    Returns:
        FetchResult mit DataFrame
        ``[ticker, cik, form, filing_date, accession, primary_doc, report_date, items]``.
    """
    ticker = ticker.upper()
    cache_key = stable_hash("sec_filings", ticker, tuple(sorted(forms)), max_per_form)
    cache_path = get_cache_dir("sec_edgar/filings") / f"{cache_key}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return FetchResult(
            df=df,
            source="sec_edgar",
            as_of=pd.Timestamp.utcnow(),
            rows=len(df),
            notes="cache",
        )
    cik_map = get_ticker_cik_map(use_cache=use_cache)
    cik = cik_map.get(ticker)
    if not cik:
        logger.info("[edgar] no CIK for ticker=%s", ticker)
        return FetchResult(
            pd.DataFrame(), "sec_edgar", pd.Timestamp.utcnow(), 0, "no_cik"
        )

    payload = _http_get_json(_SUBMISSIONS_URL.format(cik=cik))
    recent = payload.get("filings", {}).get("recent", {})
    if not recent:
        return FetchResult(
            pd.DataFrame(), "sec_edgar", pd.Timestamp.utcnow(), 0, "no_recent"
        )

    df = pd.DataFrame(
        {
            "form": recent.get("form", []),
            "filing_date": recent.get("filingDate", []),
            "report_date": recent.get("reportDate", []),
            "accession": recent.get("accessionNumber", []),
            "primary_doc": recent.get("primaryDocument", []),
            "items": recent.get("items", []),
        }
    )
    df["ticker"] = ticker
    df["cik"] = cik
    df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce", utc=True)
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce", utc=True)
    df = df[df["form"].isin(forms)].copy()
    df = df.sort_values("filing_date", ascending=False).head(max_per_form * len(forms))

    if use_cache and not df.empty:
        df.to_parquet(cache_path, index=False)
    return FetchResult(
        df=df, source="sec_edgar", as_of=pd.Timestamp.utcnow(), rows=len(df), notes=""
    )


def filings_to_event_features(
    df: pd.DataFrame, lookback_days: int = 30
) -> pd.DataFrame:
    """Konvertiere Filings -> Event-Features pro (date, ticker).

    Output-Spalten
    --------------
    - ``count_form_X`` für jede Form-Klasse (Anzahl Filings in lookback)
    - ``has_8k_recent`` (Boolean)
    - ``days_since_last_4`` (Insider-Aktivität)
    """
    if df.empty:
        return df

    df = df.dropna(subset=["filing_date"]).copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"], utc=True).dt.normalize()
    if df.empty:
        return pd.DataFrame()

    rng = pd.date_range(
        df["filing_date"].min(), df["filing_date"].max(), freq="D", tz="UTC"
    )
    out_rows: list[dict] = []
    for tkr, gdf in df.groupby("ticker"):
        for form in gdf["form"].unique():
            f_dates = sorted(gdf.loc[gdf["form"] == form, "filing_date"].unique())
            if not f_dates:
                continue
        # für jedes Datum in rng, zähle Filings je Form in [date - lookback, date]
        # einfache O(N*M) — bei großen rngs lieber vektorisiert; hier akzeptabel
        for d in rng:
            window = gdf[
                (gdf["filing_date"] <= d)
                & (gdf["filing_date"] > d - pd.Timedelta(days=lookback_days))
            ]
            row = {"date": d, "ticker": tkr}
            for form, count in window["form"].value_counts().items():
                row[f"count_form_{form}"] = int(count)
            row["has_8k_recent"] = (window["form"] == "8-K").any()
            last4 = window.loc[window["form"] == "4", "filing_date"]
            row["days_since_last_4"] = (
                (d - last4.max()).days if not last4.empty else None
            )
            out_rows.append(row)

    return pd.DataFrame(out_rows).fillna(0)


# ---- Form-4 (insider transactions) detail-fetch ---------------------------

_FORM4_URL = (
    "https://www.sec.gov/Archives/edgar/data/{cik_int}/"
    "{accession_clean}/{primary_doc}"
)


def parse_form4_xml(xml_text: str) -> dict:
    """Parse Form-4 XML grob — Insider-Trader, Code (P/S/A), Anzahl, Preis.

    *Beste-Effort*-Parser: SEC-Form-4-XMLs haben mehrere Schemas; bei Parser-Fehler
    wird ``{}`` zurückgegeben. Vollständige Implementierung: siehe `edgartools`.
    """
    out: dict = {}
    if not xml_text:
        return out
    # pragmatischer Regex-Parser für die häufigsten Felder
    name = re.search(r"<rptOwnerName>([^<]+)</rptOwnerName>", xml_text)
    if name:
        out["owner"] = name.group(1).strip()
    code = re.search(r"<transactionCode>\s*([^<\s]+)", xml_text)
    if code:
        out["transaction_code"] = code.group(1).strip()
    shares = re.search(r"<transactionShares>\s*<value>([0-9.,]+)", xml_text)
    if shares:
        out["shares"] = float(shares.group(1).replace(",", ""))
    price = re.search(r"<transactionPricePerShare>\s*<value>([0-9.,]+)", xml_text)
    if price:
        out["price"] = float(price.group(1).replace(",", ""))
    is_director = re.search(r"<isDirector>\s*([01])\s*</isDirector>", xml_text)
    if is_director:
        out["is_director"] = is_director.group(1) == "1"
    is_officer = re.search(r"<isOfficer>\s*([01])\s*</isOfficer>", xml_text)
    if is_officer:
        out["is_officer"] = is_officer.group(1) == "1"
    is_ten_percent = re.search(
        r"<isTenPercentOwner>\s*([01])\s*</isTenPercentOwner>", xml_text
    )
    if is_ten_percent:
        out["is_ten_percent"] = is_ten_percent.group(1) == "1"
    return out


__all__ = [
    "get_ticker_cik_map",
    "fetch_recent_filings",
    "filings_to_event_features",
    "parse_form4_xml",
]
