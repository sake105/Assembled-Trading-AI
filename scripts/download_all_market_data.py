"""
Download all free market data: macro (FRED), fundamentals, earnings,
dividends, news sentiment, and SEC insider trading.

Saves to:
  output/macro.parquet              — FRED macro time series
  output/fundamentals.parquet       — yfinance fundamental snapshots
  output/events_earnings.parquet    — earnings events (eps_actual/estimate)
  output/dividends.parquet          — dividend history
  output/news_sentiment_daily.parquet — daily news sentiment per symbol
  output/insider_trading.parquet    — SEC Form 4 insider transactions
"""

from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

OUTPUT = Path("output")


def _stamp(df: "pd.DataFrame") -> "pd.DataFrame":
    """Add _fetched_at column (ISO-8601 UTC) so factor_store can detect stale data."""
    df = df.copy()
    df["_fetched_at"] = datetime.now(timezone.utc).isoformat()
    return df


OUTPUT.mkdir(exist_ok=True)

UNIVERSE_FILE = Path("configs/universes/universe_ai_tech_tickers.txt")
QUALITY_TICKERS = [
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "BLK",
    "SCHW",
    "AXP",
    "CB",
    "TRV",
    "JNJ",
    "UNH",
    "LLY",
    "ABBV",
    "MRK",
    "PFE",
    "TMO",
    "ABT",
    "BMY",
    "AMGN",
    "PG",
    "KO",
    "PEP",
    "WMT",
    "COST",
    "MCD",
    "NKE",
    "SBUX",
    "CL",
    "GIS",
    "AMZN",
    "HD",
    "LOW",
    "TGT",
    "BKNG",
    "MAR",
    "DG",
    "ORLY",
    "AZO",
    "ROST",
    "CAT",
    "HON",
    "UPS",
    "RTX",
    "BA",
    "GE",
    "MMM",
    "DE",
    "EMR",
    "ETN",
    "XOM",
    "CVX",
    "COP",
    "EOG",
    "SLB",
    "VLO",
    "MPC",
    "PSX",
    "OXY",
    "LIN",
    "APD",
    "ECL",
    "SHW",
    "NEM",
    "FCX",
    "ALB",
    "NUE",
    "VMC",
    "MLM",
    "NEE",
    "DUK",
    "SO",
    "AEP",
    "EXC",
    "XEL",
    "ES",
    "ED",
    "FE",
    "AMT",
    "PLD",
    "CCI",
    "EQIX",
    "SPG",
    "PSA",
    "O",
    "WELL",
    "AVB",
    "EQR",
    "GOOGL",
    "META",
    "NFLX",
    "DIS",
    "CMCSA",
    "VZ",
    "T",
    "TMUS",
    "CHTR",
    "OMC",
]

START_DATE = "2018-01-01"
END_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")


def load_universe() -> list[str]:
    tickers: list[str] = []
    if UNIVERSE_FILE.exists():
        for line in UNIVERSE_FILE.read_text().splitlines():
            t = line.split("#")[0].strip()
            if t:
                tickers.append(t)
    # Combine AI-tech + quality universe, deduplicate
    all_tickers = list(dict.fromkeys(tickers + QUALITY_TICKERS))
    # Filter out non-US/non-yfinance tickers
    all_tickers = [t for t in all_tickers if "." not in t]
    log.info("[UNIVERSE] %d unique tickers", len(all_tickers))
    return all_tickers


# ---------------------------------------------------------------------------
# 1. FRED Macro Data (free CSV endpoint, no API key required)
# ---------------------------------------------------------------------------

FRED_SERIES = {
    "DGS10": "treasury_10y",  # 10-year Treasury yield
    "DGS2": "treasury_2y",  # 2-year Treasury yield
    "T10Y2Y": "yield_curve_spread",  # 10y-2y spread (recession indicator)
    "VIXCLS": "vix",  # CBOE VIX
    "BAMLH0A0HYM2": "hy_spread",  # High-yield credit spread (OAS)
    "DPCREDIT": "fed_discount_rate",  # Federal discount rate
    "FEDFUNDS": "fed_funds_rate",  # Effective fed funds rate
    "CPIAUCSL": "cpi_yoy",  # CPI all-urban (SA)
    "UNRATE": "unemployment_rate",  # Unemployment rate
    "ICSA": "initial_claims",  # Initial jobless claims
    "INDPRO": "industrial_prod",  # Industrial production index
    "M2SL": "m2_money_supply",  # M2 money supply
    "DTWEXBGS": "usd_index",  # USD broad index (trade-weighted)
    "DCOILWTICO": "wti_crude_oil",  # WTI crude oil price
    "GOLDAMGBD228NLBM": "gold_price",  # Gold price (London fixing)
}

FRED_BASE = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="


def download_fred_macro() -> pd.DataFrame:
    log.info("[MACRO] Downloading %d FRED series...", len(FRED_SERIES))
    frames: list[pd.DataFrame] = []
    for series_id, label in FRED_SERIES.items():
        try:
            url = f"{FRED_BASE}{series_id}"
            df = pd.read_csv(url)
            # FRED uses 'observation_date' as date column, value column = series_id
            date_col = next(
                (c for c in df.columns if "date" in c.lower()), df.columns[0]
            )
            val_col = next((c for c in df.columns if c != date_col), df.columns[1])
            df = df.rename(columns={date_col: "date", val_col: label})
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df[df[label] != "."].copy()
            df[label] = pd.to_numeric(df[label], errors="coerce")
            df = df.set_index("date")[[label]]
            frames.append(df)
            log.info("  [OK] %s (%s): %d obs", series_id, label, len(df))
            time.sleep(0.3)
        except Exception as exc:
            log.warning("  [WARN] %s failed: %s", series_id, exc)

    if not frames:
        log.warning("[MACRO] No FRED data downloaded")
        return pd.DataFrame()

    macro = pd.concat(frames, axis=1).sort_index()
    # Transform raw CPIAUCSL index level → true YoY % rate (pct_change over 12
    # monthly periods). Must be done BEFORE ffill so pct_change operates on the
    # sparse monthly observations, not daily-repeated values.
    if "cpi_yoy" in macro.columns:
        cpi_notnull_idx = macro["cpi_yoy"].dropna().index
        macro.loc[cpi_notnull_idx, "cpi_yoy"] = (
            macro.loc[cpi_notnull_idx, "cpi_yoy"].pct_change(12) * 100
        )
    macro = macro.reset_index().rename(columns={"date": "timestamp"})
    # Forward-fill daily (macro releases are infrequent)
    macro = macro.set_index("timestamp").ffill().reset_index()
    _stamp(macro).to_parquet(OUTPUT / "macro.parquet", index=False)
    log.info("[MACRO] Saved: %d rows, %d series", len(macro), len(FRED_SERIES))
    return macro


# ---------------------------------------------------------------------------
# 2. Fundamentals snapshot from yfinance
# ---------------------------------------------------------------------------

FUNDAMENTAL_FIELDS = [
    "symbol",
    "timestamp",
    "market_cap",
    "enterprise_value",
    "pe_ratio",
    "forward_pe",
    "pb_ratio",
    "ps_ratio",
    "peg_ratio",
    "ev_ebitda",
    "eps_trailing",
    "eps_forward",
    "revenue_ttm",
    "gross_margins",
    "operating_margins",
    "profit_margins",
    "roe",
    "roa",
    "debt_to_equity",
    "current_ratio",
    "quick_ratio",
    "dividend_yield",
    "payout_ratio",
    "beta",
    "float_shares",
    "short_ratio",
    "52w_high",
    "52w_low",
]

INFO_FIELD_MAP = {
    "marketCap": "market_cap",
    "enterpriseValue": "enterprise_value",
    "trailingPE": "pe_ratio",
    "forwardPE": "forward_pe",
    "priceToBook": "pb_ratio",
    "priceToSalesTrailing12Months": "ps_ratio",
    "pegRatio": "peg_ratio",
    "enterpriseToEbitda": "ev_ebitda",
    "trailingEps": "eps_trailing",
    "forwardEps": "eps_forward",
    "totalRevenue": "revenue_ttm",
    "grossMargins": "gross_margins",
    "operatingMargins": "operating_margins",
    "profitMargins": "profit_margins",
    "returnOnEquity": "roe",
    "returnOnAssets": "roa",
    "debtToEquity": "debt_to_equity",
    "currentRatio": "current_ratio",
    "quickRatio": "quick_ratio",
    "dividendYield": "dividend_yield",
    "payoutRatio": "payout_ratio",
    "beta": "beta",
    "floatShares": "float_shares",
    "shortRatio": "short_ratio",
    "fiftyTwoWeekHigh": "52w_high",
    "fiftyTwoWeekLow": "52w_low",
}


def download_fundamentals(tickers: list[str]) -> pd.DataFrame:
    log.info("[FUNDAMENTALS] Downloading snapshots for %d symbols...", len(tickers))
    rows: list[dict] = []
    now = pd.Timestamp.now(tz="UTC")
    for i, sym in enumerate(tickers):
        try:
            info = yf.Ticker(sym).info
            row: dict = {"symbol": sym, "timestamp": now}
            for yf_key, our_key in INFO_FIELD_MAP.items():
                row[our_key] = info.get(yf_key)
            rows.append(row)
            if (i + 1) % 20 == 0:
                log.info("  %d/%d done", i + 1, len(tickers))
            time.sleep(0.2)
        except Exception as exc:
            log.warning("  [WARN] %s: %s", sym, exc)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    _stamp(df).to_parquet(OUTPUT / "fundamentals.parquet", index=False)
    log.info("[FUNDAMENTALS] Saved: %d symbols", len(df))
    return df


# ---------------------------------------------------------------------------
# 3. Earnings history from yfinance
# ---------------------------------------------------------------------------


def download_earnings(tickers: list[str]) -> pd.DataFrame:
    log.info("[EARNINGS] Downloading earnings history for %d symbols...", len(tickers))
    rows: list[dict] = []
    for sym in tickers:
        try:
            ticker = yf.Ticker(sym)
            # Quarterly earnings
            qe = ticker.quarterly_earnings
            if qe is not None and not qe.empty:
                for date, row_data in qe.iterrows():
                    ts = (
                        pd.Timestamp(date, tz="UTC")
                        if not isinstance(date, pd.Timestamp)
                        else date
                    )
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    eps_act = row_data.get("Earnings")
                    eps_est = row_data.get("Estimate")
                    revenue = row_data.get("Revenue")
                    rows.append(
                        {
                            "timestamp": ts,
                            "symbol": sym,
                            "event_type": "earnings",
                            "event_id": f"{sym}_earnings_{ts.date()}",
                            "event_date": ts,
                            "disclosure_date": ts,
                            "eps_actual": eps_act,
                            "eps_estimate": eps_est,
                            "eps_surprise_pct": (
                                (eps_act - eps_est) / abs(eps_est) * 100
                                if eps_est and eps_act and eps_est != 0
                                else None
                            ),
                            "revenue_actual": revenue,
                            "revenue_estimate": None,
                        }
                    )
            time.sleep(0.15)
        except Exception as exc:
            log.warning("  [WARN] %s earnings: %s", sym, exc)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    _stamp(df).to_parquet(OUTPUT / "events_earnings.parquet", index=False)
    log.info(
        "[EARNINGS] Saved: %d events across %d symbols", len(df), df["symbol"].nunique()
    )
    return df


# ---------------------------------------------------------------------------
# 4. Dividend history
# ---------------------------------------------------------------------------


def download_dividends(tickers: list[str]) -> pd.DataFrame:
    log.info("[DIVIDENDS] Downloading dividend history for %d symbols...", len(tickers))
    rows: list[dict] = []
    for sym in tickers:
        try:
            divs = yf.Ticker(sym).dividends
            if divs is not None and not divs.empty:
                for date, amount in divs.items():
                    ts = pd.Timestamp(date)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize("UTC")
                    rows.append(
                        {
                            "timestamp": ts,
                            "symbol": sym,
                            "dividend_amount": float(amount),
                            "event_type": "dividend",
                        }
                    )
            time.sleep(0.1)
        except Exception as exc:
            log.warning("  [WARN] %s dividends: %s", sym, exc)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (
        df[df["timestamp"] >= START_DATE]
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )
    _stamp(df).to_parquet(OUTPUT / "dividends.parquet", index=False)
    log.info("[DIVIDENDS] Saved: %d dividend events", len(df))
    return df


# ---------------------------------------------------------------------------
# 5. News headlines + simple sentiment (yfinance)
# ---------------------------------------------------------------------------

SENTIMENT_KEYWORDS = {
    "positive": [
        "beat",
        "record",
        "surge",
        "growth",
        "profit",
        "strong",
        "expand",
        "upgrade",
        "outperform",
        "bull",
        "rally",
        "gain",
        "rise",
        "soar",
        "boost",
        "win",
        "success",
        "breakthrough",
        "partnership",
        "contract",
    ],
    "negative": [
        "miss",
        "loss",
        "fall",
        "decline",
        "cut",
        "layoff",
        "downgrade",
        "underperform",
        "bear",
        "drop",
        "plunge",
        "crash",
        "risk",
        "warning",
        "investigation",
        "lawsuit",
        "fine",
        "recall",
        "halt",
        "bankruptcy",
    ],
}


def _score_headline(title: str) -> float:
    """Simple keyword sentiment score in [-1, +1]."""
    title_lower = title.lower()
    pos = sum(1 for w in SENTIMENT_KEYWORDS["positive"] if w in title_lower)
    neg = sum(1 for w in SENTIMENT_KEYWORDS["negative"] if w in title_lower)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


def download_news_sentiment(tickers: list[str]) -> pd.DataFrame:
    log.info("[NEWS] Downloading news headlines for %d symbols...", len(tickers))
    rows: list[dict] = []
    for sym in tickers:
        try:
            news_items = yf.Ticker(sym).news
            if not news_items:
                continue
            for item in news_items:
                pub_ts = item.get("providerPublishTime") or item.get("publishedAt")
                if pub_ts is None:
                    continue
                ts = pd.Timestamp(pub_ts, unit="s", tz="UTC")
                title = item.get("title", "")
                score = _score_headline(title)
                rows.append(
                    {
                        "timestamp": ts,
                        "symbol": sym,
                        "title": title[:200],
                        "source": item.get("publisher", ""),
                        "url": item.get("link", "")[:300],
                        "sentiment_score": score,
                        "sentiment_volume": 1,
                    }
                )
            time.sleep(0.1)
        except Exception as exc:
            log.warning("  [WARN] %s news: %s", sym, exc)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Save raw news
    _stamp(df).to_parquet(OUTPUT / "news_raw.parquet", index=False)

    # Build daily sentiment aggregate per symbol
    df["date"] = df["timestamp"].dt.normalize()
    daily = (
        df.groupby(["date", "symbol"])
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            sentiment_volume=("sentiment_volume", "sum"),
            headline_count=("title", "count"),
        )
        .reset_index()
        .rename(columns={"date": "timestamp"})
    )
    daily["timestamp"] = pd.to_datetime(daily["timestamp"], utc=True)
    _stamp(daily).to_parquet(OUTPUT / "news_sentiment_daily.parquet", index=False)

    log.info(
        "[NEWS] Saved: %d raw articles, %d daily sentiment rows across %d symbols",
        len(df),
        len(daily),
        daily["symbol"].nunique(),
    )
    return daily


# ---------------------------------------------------------------------------
# 6. SEC EDGAR Insider Trading (Form 4) — free REST API
# ---------------------------------------------------------------------------

EDGAR_HEADERS = {"User-Agent": "AssembledTradingAI research@example.com"}
EDGAR_COMPANY_URL = "https://data.sec.gov/submissions/CIK{:010d}.json"
EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index?q=%22form+4%22&dateRange=custom&startdt={start}&enddt={end}&hits.hits.total.value=true&hits.hits._source.period_of_report=true&hits.hits._source.entity_name=true"
EDGAR_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"


def _get_cik_map() -> dict[str, str]:
    """Return {ticker: CIK} from SEC EDGAR company list."""
    try:
        r = requests.get(EDGAR_TICKERS_URL, headers=EDGAR_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        return {v["ticker"]: str(v["cik_str"]).zfill(10) for v in data.values()}
    except Exception as exc:
        log.warning("[INSIDER] CIK map failed: %s", exc)
        return {}


def _fetch_form4_for_cik(cik: str, sym: str) -> list[dict]:
    """Fetch recent Form 4 filings for a CIK."""
    rows: list[dict] = []
    try:
        url = EDGAR_COMPANY_URL.format(int(cik))
        r = requests.get(url, headers=EDGAR_HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        filings = data.get("filings", {}).get("recent", {})
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])

        for form, date_str, acc in zip(forms, dates, accessions):
            if form != "4":
                continue
            ts = pd.Timestamp(date_str, tz="UTC")
            rows.append(
                {
                    "timestamp": ts,
                    "symbol": sym,
                    "cik": cik,
                    "accession": acc,
                    "form_type": "4",
                    "filing_date": ts,
                    # transaction details need deeper parsing — use filing date as proxy
                    "transaction_type": "unknown",
                    "shares": None,
                    "price": None,
                }
            )
    except Exception as exc:
        log.debug("[INSIDER] CIK %s (%s): %s", cik, sym, exc)
    return rows


def download_insider_trading(tickers: list[str]) -> pd.DataFrame:
    log.info("[INSIDER] Downloading SEC Form 4 for %d symbols...", len(tickers))
    cik_map = _get_cik_map()
    rows: list[dict] = []
    found = 0
    for sym in tickers:
        cik = cik_map.get(sym)
        if not cik:
            continue
        sym_rows = _fetch_form4_for_cik(cik, sym)
        rows.extend(sym_rows)
        found += 1
        time.sleep(0.12)  # EDGAR rate limit: ~8 req/sec

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = (
        df[df["timestamp"] >= START_DATE]
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )
    _stamp(df).to_parquet(OUTPUT / "insider_trading.parquet", index=False)
    log.info(
        "[INSIDER] Saved: %d filings across %d symbols (%d found CIK)",
        len(df),
        df["symbol"].nunique(),
        found,
    )
    return df


# ---------------------------------------------------------------------------
# 7. Macro ETF proxies via yfinance (VIX, bond yields, etc.)
# ---------------------------------------------------------------------------

MACRO_ETF_PROXIES = {
    "^VIX": "vix_close",
    "^TNX": "tnx_10y_yield",
    "^TYX": "tyx_30y_yield",
    "^IRX": "irx_3m_yield",
    "^GSPC": "sp500_close",
    "^NDX": "nasdaq100_close",
    "^RUT": "russell2000_close",
    "DX-Y.NYB": "dollar_index",
    "GC=F": "gold_futures",
    "CL=F": "crude_oil_futures",
    "ZB=F": "treasury_bond_futures",
}


def download_macro_etf_proxies() -> pd.DataFrame:
    log.info("[MACRO-ETF] Downloading %d macro proxies...", len(MACRO_ETF_PROXIES))
    tickers_list = list(MACRO_ETF_PROXIES.keys())
    raw = yf.download(
        tickers_list, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False
    )
    rows: list[dict] = []
    if isinstance(raw.columns, pd.MultiIndex):
        close = (
            raw["Close"]
            if "Close" in raw.columns.get_level_values(0)
            else raw.xs("Close", axis=1, level=0)
        )
        for ts, row in close.iterrows():
            r: dict = {
                "timestamp": (
                    pd.Timestamp(ts).tz_localize("UTC") if ts.tzinfo is None else ts
                )
            }
            for ticker, label in MACRO_ETF_PROXIES.items():
                val = row.get(ticker)
                r[label] = float(val) if val is not None and pd.notna(val) else None
            rows.append(r)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Merge with FRED macro if available
    fred_path = OUTPUT / "macro.parquet"
    if fred_path.exists():
        fred = pd.read_parquet(fred_path)
        fred["timestamp"] = pd.to_datetime(fred["timestamp"], utc=True)
        fred["timestamp"] = fred["timestamp"].dt.normalize()
        df["timestamp"] = df["timestamp"].dt.normalize()
        df = df.merge(fred, on="timestamp", how="outer").sort_values("timestamp")
        df = df.ffill().reset_index(drop=True)

    _stamp(df).to_parquet(OUTPUT / "macro.parquet", index=False)
    log.info(
        "[MACRO-ETF] Saved: %d rows with %d macro columns", len(df), len(df.columns) - 1
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tickers = load_universe()

    log.info("=" * 60)
    log.info("Step 1/6: FRED Macro Data")
    log.info("=" * 60)
    download_fred_macro()

    log.info("=" * 60)
    log.info("Step 2/6: Macro ETF Proxies (VIX, yields, indices)")
    log.info("=" * 60)
    download_macro_etf_proxies()

    log.info("=" * 60)
    log.info("Step 3/6: Fundamental Snapshots (%d symbols)", len(tickers))
    log.info("=" * 60)
    download_fundamentals(tickers)

    log.info("=" * 60)
    log.info("Step 4/6: Earnings History")
    log.info("=" * 60)
    download_earnings(tickers)

    log.info("=" * 60)
    log.info("Step 5/6: Dividend History")
    log.info("=" * 60)
    download_dividends(tickers)

    log.info("=" * 60)
    log.info("Step 6/6: News Headlines + Sentiment")
    log.info("=" * 60)
    download_news_sentiment(tickers)

    # Insider trading is slow — run last
    # log.info("Step 7: SEC EDGAR Insider Trading")
    # download_insider_trading(tickers)

    log.info("=" * 60)
    log.info("ALL DONE. Output files:")
    for f in sorted(OUTPUT.glob("*.parquet")):
        size_kb = f.stat().st_size // 1024
        log.info("  %s (%d KB)", f.name, size_kb)
    log.info("=" * 60)
