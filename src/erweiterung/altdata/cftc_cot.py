"""CFTC Commitments of Traders — Positionierung großer Marktteilnehmer (frei).

Quelle
------
CFTC veröffentlicht jeden Freitag die Commitments-of-Traders-Reports:
    https://publicreporting.cftc.gov/api/views/?accessType=DOWNLOAD

Frei zugänglich via Socrata Open Data API. Kein Key erforderlich.

Inhalt
------
- Disaggregated Futures-Only Reports (commodities, FX, indices)
- Traders by category: Producer/Merchant, Swap Dealers, Managed Money,
  Other Reportables, Non-Reportable
- Long & Short Open Interest

Anwendung
---------
COT als Sentimentsignal für Risk-On/Risk-Off:
- Managed-Money-Long-Bias in Equity-Index-Futures = bullish
- Net-Short-Bias = potenziell zu pessimistisch (Mean-Reversion-Signal)
- Extremwerte (Net-Position relativ zu eigenem Histogramm) sind die wertvollsten
  Indikatoren — siehe Briese (2008), *The Commitments of Traders Bible*.

PIT-Hinweis
-----------
Reports enthalten Daten von Dienstag, werden Freitag nach Markschluss
veröffentlicht. -> ``shift_days >= 1`` (Montag-Open ist erste verwendbare Zeit).
"""

from __future__ import annotations

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

# Socrata-API: Disaggregated Futures-Only (DFO) historisch
_DFO_API = "https://publicreporting.cftc.gov/resource/72hh-3qpy.json"


@rate_limited(min_interval_s=1.0)
@retry_with_backoff(max_attempts=3, base_delay=2.0)
def _fetch_socrata(
    contract_market_code: str | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
    limit: int = 50000,
) -> pd.DataFrame:
    """Generischer SoQL-Query."""
    import requests

    where_clauses = [
        f"report_date_as_yyyy_mm_dd >= '{start.strftime('%Y-%m-%d')}'",
        f"report_date_as_yyyy_mm_dd <= '{end.strftime('%Y-%m-%d')}'",
    ]
    if contract_market_code:
        where_clauses.append(f"cftc_contract_market_code = '{contract_market_code}'")
    params = {
        "$where": " AND ".join(where_clauses),
        "$limit": limit,
        "$order": "report_date_as_yyyy_mm_dd ASC",
    }
    r = requests.get(_DFO_API, params=params, timeout=30)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def fetch_cot_disaggregated(
    contract_market_code: str | None = None,
    start: str | pd.Timestamp = "2010-01-01",
    end: str | pd.Timestamp | None = None,
    use_cache: bool = True,
) -> FetchResult:
    """Hole CFTC Disaggregated COT-Daten.

    Args:
        contract_market_code: z. B. ``'13874A'`` für E-Mini S&P 500.
            Wenn ``None``: alle (sehr groß!).
        start, end: Datumsbereich.
        use_cache: Disk-Cache.

    Wichtige Codes (Auszug):
        - 13874A : E-Mini S&P 500
        - 209742 : E-Mini Nasdaq-100
        - 020601 : Crude Oil (WTI)
        - 088691 : Gold
        - 098662 : 10Y Treasury Note
        - 132741 : VIX Futures
        - 098745 : 30-Year Treasury Bond
    """
    end_ts = to_utc_date(end or pd.Timestamp.utcnow())
    start_ts = to_utc_date(start)
    cache_key = stable_hash("cot_dfo", contract_market_code, str(start_ts), str(end_ts))
    cache_path = get_cache_dir("cftc") / f"{cache_key}.parquet"
    if use_cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
        return FetchResult(df, "cftc_cot", pd.Timestamp.utcnow(), len(df), "cache")

    df = _fetch_socrata(contract_market_code, start_ts, end_ts)
    if df.empty:
        return FetchResult(df, "cftc_cot", pd.Timestamp.utcnow(), 0, "empty")

    keep_cols = [
        "report_date_as_yyyy_mm_dd",
        "market_and_exchange_names",
        "cftc_contract_market_code",
        "open_interest_all",
        "prod_merc_positions_long_all",
        "prod_merc_positions_short_all",
        "swap_positions_long_all",
        "swap__positions_short_all",
        "m_money_positions_long_all",
        "m_money_positions_short_all",
        "other_rept_positions_long",
        "other_rept_positions_short",
        "nonrept_positions_long_all",
        "nonrept_positions_short_all",
    ]
    avail = [c for c in keep_cols if c in df.columns]
    df = df[avail].copy()
    df = df.rename(
        columns={
            "report_date_as_yyyy_mm_dd": "date",
            "market_and_exchange_names": "market",
            "cftc_contract_market_code": "code",
        }
    )
    df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
    for c in df.columns:
        if c not in ("date", "market", "code"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if use_cache:
        df.to_parquet(cache_path, index=False)
    return FetchResult(df, "cftc_cot", pd.Timestamp.utcnow(), len(df), "")


def cot_net_position_zscore(
    df: pd.DataFrame,
    category: str = "m_money",
    lookback_weeks: int = 52,
) -> pd.DataFrame:
    """Net-Position-z-Score (rolling) je Markt.

    Args:
        df: Output von ``fetch_cot_disaggregated``.
        category: ``'m_money'`` (Managed Money) | ``'prod_merc'`` (Hedger) |
            ``'swap'`` | ``'other_rept'`` | ``'nonrept'``.
        lookback_weeks: Rolling-Fenster.

    Returns:
        DataFrame mit ``net_pos`` und ``net_pos_z`` je (date, market).

    Interpretation
    --------------
    - Hoher z-Score von Managed-Money-Net = Trendfolger sind extrem long ->
      potenzieller Reversal-Indikator.
    - Hoher z-Score von Hedgern (Producer/Merchant) auf Long-Seite = ungewöhnlich;
      i. d. R. Frühindikator für höhere Spot-Preise.
    """
    if df.empty:
        return df

    long_col_map = {
        "m_money": "m_money_positions_long_all",
        "prod_merc": "prod_merc_positions_long_all",
        "swap": "swap_positions_long_all",
        "other_rept": "other_rept_positions_long",
        "nonrept": "nonrept_positions_long_all",
    }
    short_col_map = {
        "m_money": "m_money_positions_short_all",
        "prod_merc": "prod_merc_positions_short_all",
        "swap": "swap__positions_short_all",
        "other_rept": "other_rept_positions_short",
        "nonrept": "nonrept_positions_short_all",
    }
    if category not in long_col_map:
        raise ValueError(f"unknown category: {category}")
    long_c = long_col_map[category]
    short_c = short_col_map[category]
    if long_c not in df.columns or short_c not in df.columns:
        raise ValueError("required columns missing in df")

    out = df[["date", "market", long_c, short_c]].copy()
    out = out.sort_values(["market", "date"])
    out["net_pos"] = out[long_c] - out[short_c]
    grp = out.groupby("market", group_keys=False)
    out["net_pos_pit"] = grp["net_pos"].shift(1)
    out["mean"] = grp["net_pos_pit"].transform(
        lambda s: s.rolling(lookback_weeks, min_periods=lookback_weeks // 2).mean()
    )
    out["std"] = grp["net_pos_pit"].transform(
        lambda s: s.rolling(lookback_weeks, min_periods=lookback_weeks // 2).std()
    )
    out["net_pos_z"] = (out["net_pos_pit"] - out["mean"]) / out["std"]
    return out


__all__ = ["fetch_cot_disaggregated", "cot_net_position_zscore"]
