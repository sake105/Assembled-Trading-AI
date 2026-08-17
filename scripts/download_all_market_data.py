"""
Download all free market data: macro (FRED), fundamentals, earnings,
dividends, news sentiment, and SEC insider trading.

Saves to:
  output/macro.parquet              — FRED macro time series
  output/fundamentals.parquet       — yfinance fundamental snapshots
  output/events_earnings.parquet    — earnings events (eps_actual/estimate)
  output/dividends.parquet          — dividend history
  output/news_sentiment_daily.parquet — daily news sentiment per symbol
  output/insider_form4.parquet      — SEC Form 4 insider transactions (real parser)
"""

from __future__ import annotations

import logging
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
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
    # UMGESTELLT 2026-08-17: Ticker.quarterly_earnings ist in yfinance
    # deprecated und liefert STILL None fuer jedes Symbol — der alte Pfad
    # produzierte seit dem yfinance-Upgrade 0 Zeilen ohne eine einzige
    # Warnung (kein Rate-Limit; per 3-Symbol-Probe verifiziert). Ersatz:
    # get_earnings_dates() (EPS Estimate/Reported + Surprise). SPALTEN-Schema
    # unveraendert; Revenue liefert der neue Endpoint nicht -> None.
    # DATUMS-SEMANTIK GEAENDERT (F-senior-5): event_date/disclosure_date sind
    # jetzt der ANKUENDIGUNGS-Zeitstempel in UTC (After-Close ET faellt auf
    # den Folge-UTC-Tag), nicht mehr das Fiskalperioden-Ende — event_ids sind
    # zu Altzeilen derselben Datei NICHT vergleichbar. PIT-Richtung sicher
    # (spaeter, nie frueher).
    log.info("[EARNINGS] Downloading earnings history for %d symbols...", len(tickers))
    rows: list[dict] = []
    n_fail = 0
    for sym in tickers:
        try:
            ticker = yf.Ticker(sym)
            ed = ticker.get_earnings_dates(limit=12)
            if ed is not None and not ed.empty:
                # PIT: nur berichtete Quartale (EPS vorhanden) — zukuenftige
                # Termine haben NaN-Reported und waeren Zukunfts-Events.
                rep_col = next((c for c in ed.columns if "Reported" in str(c)), None)
                est_col = next((c for c in ed.columns if "Estimate" in str(c)), None)
                for date, row_data in ed.iterrows():
                    eps_act = row_data.get(rep_col) if rep_col else None
                    if eps_act is None or pd.isna(eps_act):
                        continue
                    ts = pd.Timestamp(date)
                    ts = (
                        ts.tz_localize("UTC")
                        if ts.tzinfo is None
                        else ts.tz_convert("UTC")
                    )
                    eps_est = row_data.get(est_col) if est_col else None
                    if eps_est is not None and pd.isna(eps_est):
                        eps_est = None
                    rows.append(
                        {
                            "timestamp": ts,
                            "symbol": sym,
                            "event_type": "earnings",
                            "event_id": f"{sym}_earnings_{ts.date()}",
                            "event_date": ts,
                            "disclosure_date": ts,
                            "eps_actual": float(eps_act),
                            "eps_estimate": (
                                float(eps_est) if eps_est is not None else None
                            ),
                            "eps_surprise_pct": (
                                (eps_act - eps_est) / abs(eps_est) * 100
                                if eps_est and eps_act and eps_est != 0
                                else None
                            ),
                            "revenue_actual": None,
                            "revenue_estimate": None,
                        }
                    )
            time.sleep(0.15)
        except Exception as exc:
            n_fail += 1
            log.warning("  [WARN] %s earnings: %s", sym, exc)

    if not rows:
        # E-176-Lektion: leer darf nicht still aussehen wie Erfolg.
        log.error(
            "[EARNINGS] 0 rows from %d symbols (%d hard failures) — "
            "NOT overwriting existing parquet",
            len(tickers),
            n_fail,
        )
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    _stamp(df).to_parquet(OUTPUT / "events_earnings.parquet", index=False)
    log.info(
        "[EARNINGS] Saved: %d events across %d symbols", len(df), df["symbol"].nunique()
    )
    # F-senior-3 (E-180): Datei-mtime ist nicht Daten-Frische — der
    # freshness_monitor urteilt per mtime und wuerde einen frisch
    # geschriebenen, inhaltlich alten Payload gruen melden. Deshalb den
    # Ereignis-Horizont IMMER loggen und bei grossem Abstand laut warnen
    # (2026-08-17 real: Reported-EPS endete vendor-seitig ~14 Monate zurueck).
    horizon = df["event_date"].max()
    age_days = (pd.Timestamp.now(tz="UTC") - horizon).days
    if age_days > 120:
        log.warning(
            "[EARNINGS] newest reported event is %s (%d days old) — vendor "
            "cutoff or filter issue; file mtime will still look fresh",
            horizon,
            age_days,
        )
    else:
        log.info(
            "[EARNINGS] newest reported event: %s (%d days old)", horizon, age_days
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

# The legacy per-CIK stub here hardcoded transaction_type="unknown" (shares/price
# unparsed) and wrote output/insider_trading.parquet. RETIRED 2026-06-09: it now
# delegates to the real EDGAR Form 4 parser (edgar_form4_ingest), which classifies
# the transactionCode (P/S/unknown) and writes output/insider_form4.parquet with a
# PIT-correct available_at = SGML ACCEPTANCE-DATETIME, using a SEC-compliant
# declared User-Agent (SEC_USER_AGENT / settings).


def download_insider_trading(tickers: list[str]) -> pd.DataFrame:
    """Download + parse REAL SEC Form 4 insider trades for ``tickers``.

    Delegates to :func:`edgar_form4_ingest.ingest_form4_for_symbols` (per-CIK
    submissions enumeration + real ownership-XML parse). Writes
    ``output/insider_form4.parquet`` — NOT the legacy all-'unknown'
    ``insider_trading.parquet``. For deeper history, use the date-based
    ``edgar_form4_ingest.ingest_form4`` or pass a larger ``lookback_days``.
    """
    from src.assembled_core.data.edgar_form4_ingest import ingest_form4_for_symbols

    log.info(
        "[INSIDER] Parsing SEC Form 4 (real classifier) for %d symbols...",
        len(tickers),
    )
    return ingest_form4_for_symbols(tickers, out_path=OUTPUT / "insider_form4.parquet")


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
