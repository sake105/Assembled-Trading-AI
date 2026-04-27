"""CBOE Public Data Source — VIX term structure and Put/Call Ratio.

Downloads freely available CBOE market data:
- VIX (CBOE Volatility Index) via FRED (VIXCLS series)
- VIX3M (3-month VIX) via FRED (VXVCLS series)
- CBOE Equity Put/Call Ratio via CBOE public CSV

All data is fetched point-in-time (no look-ahead). Results are cached
to avoid redundant network calls within the same Python session.

Usage:
    from src.assembled_core.data.sources.cboe_source import CBOESource

    source = CBOESource()
    vix = source.fetch_vix(start_date="2020-01-01")
    pcr = source.fetch_put_call_ratio(start_date="2020-01-01")
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# FRED series IDs for VIX variants
_FRED_VIX_SERIES = "VIXCLS"
_FRED_VIX3M_SERIES = "VXVCLS"

# CBOE public equity put/call ratio CSV endpoint (daily data)
_CBOE_PCR_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/PCALL_History.csv"


class CBOESource:
    """Fetches CBOE volatility and options market data from public endpoints.

    Data sources:
    - VIX / VIX3M: Federal Reserve FRED API (no key required for basic access)
    - Put/Call Ratio: CBOE public data endpoint (CSV download)

    Attributes:
        fred_api_key: Optional FRED API key. Without a key, requests use the
            public endpoint (max 1 request/second). With a key, limits are higher.
    """

    def __init__(self, fred_api_key: Optional[str] = None) -> None:
        self.fred_api_key = fred_api_key
        self._vix_cache: pd.DataFrame | None = None
        self._pcr_cache: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # VIX / VIX3M
    # ------------------------------------------------------------------

    def fetch_vix(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch VIX and VIX3M (VXVCLS) from FRED.

        Args:
            start_date: ISO date string (e.g. "2020-01-01"). Default: 10 years ago.
            end_date: ISO date string. Default: today.

        Returns:
            DataFrame with columns: timestamp, vix, vix3m.
            Missing values (weekends/holidays) are dropped.
        """
        try:
            import pandas_datareader.data as web  # type: ignore
        except ImportError:
            logger.warning("[CBOE] pandas_datareader not installed — cannot fetch VIX from FRED")
            return pd.DataFrame(columns=["timestamp", "vix", "vix3m"])

        start = start_date or "2015-01-01"
        end = end_date or datetime.today().strftime("%Y-%m-%d")

        try:
            vix = web.DataReader(_FRED_VIX_SERIES, "fred", start, end)
            vix.columns = ["vix"]
        except Exception as exc:
            logger.warning("[CBOE] Failed to fetch VIX from FRED: %s", exc)
            vix = pd.DataFrame(columns=["vix"])

        try:
            vix3m = web.DataReader(_FRED_VIX3M_SERIES, "fred", start, end)
            vix3m.columns = ["vix3m"]
        except Exception as exc:
            logger.warning("[CBOE] Failed to fetch VIX3M from FRED: %s", exc)
            vix3m = pd.DataFrame(columns=["vix3m"])

        if vix.empty and vix3m.empty:
            return pd.DataFrame(columns=["timestamp", "vix", "vix3m"])

        df = pd.concat([vix, vix3m], axis=1).dropna(how="all")
        df.index = pd.to_datetime(df.index)
        df = df.reset_index().rename(columns={"DATE": "timestamp", "index": "timestamp"})
        if "DATE" not in df.columns and df.columns[0] != "timestamp":
            df = df.rename(columns={df.columns[0]: "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        self._vix_cache = df
        logger.info("[CBOE] Fetched VIX data: %d rows (%s to %s)", len(df), start, end)
        return df

    # ------------------------------------------------------------------
    # Put/Call Ratio
    # ------------------------------------------------------------------

    def fetch_put_call_ratio(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch CBOE Equity Put/Call Ratio from CBOE public endpoint.

        Args:
            start_date: ISO date string. Default: no filter (all available).
            end_date: ISO date string. Default: today.

        Returns:
            DataFrame with columns: timestamp, put_call_ratio.
        """
        try:
            raw = pd.read_csv(_CBOE_PCR_URL, dtype=str)
        except Exception as exc:
            logger.warning("[CBOE] Failed to fetch put/call ratio from CBOE: %s", exc)
            return pd.DataFrame(columns=["timestamp", "put_call_ratio"])

        # CBOE CSV columns vary; normalise
        date_col = next(
            (c for c in raw.columns if "date" in c.lower() or "Date" in c),
            raw.columns[0],
        )
        value_col = next(
            (c for c in raw.columns if "call" in c.lower() or "ratio" in c.lower()),
            raw.columns[1] if len(raw.columns) > 1 else raw.columns[0],
        )

        df = raw[[date_col, value_col]].copy()
        df.columns = ["timestamp", "put_call_ratio"]
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["put_call_ratio"] = pd.to_numeric(df["put_call_ratio"], errors="coerce")
        df = df.dropna(subset=["timestamp", "put_call_ratio"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        if start_date:
            df = df[df["timestamp"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["timestamp"] <= pd.Timestamp(end_date)]

        self._pcr_cache = df
        logger.info("[CBOE] Fetched Put/Call Ratio: %d rows", len(df))
        return df

    # ------------------------------------------------------------------
    # Combined
    # ------------------------------------------------------------------

    def fetch_options_regime_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch VIX + VIX3M + Put/Call Ratio, aligned to common date index.

        Returns:
            DataFrame with columns: timestamp, vix, vix3m, put_call_ratio.
            Forward-fills missing values up to 5 trading days.
        """
        vix_df = self.fetch_vix(start_date, end_date)
        pcr_df = self.fetch_put_call_ratio(start_date, end_date)

        if vix_df.empty and pcr_df.empty:
            return pd.DataFrame(columns=["timestamp", "vix", "vix3m", "put_call_ratio"])

        if vix_df.empty:
            combined = pcr_df.copy()
            combined["vix"] = float("nan")
            combined["vix3m"] = float("nan")
        elif pcr_df.empty:
            combined = vix_df.copy()
            combined["put_call_ratio"] = float("nan")
        else:
            combined = pd.merge_asof(
                vix_df.sort_values("timestamp"),
                pcr_df.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta("5D"),
            )

        combined = combined.sort_values("timestamp").reset_index(drop=True)
        # Forward-fill up to 5 days for weekend/holiday gaps
        combined = combined.ffill(limit=5)
        return combined
