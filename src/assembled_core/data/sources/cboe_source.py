"""CBOE Public Data Source — VIX term structure and Put/Call Ratio.

Downloads freely available CBOE market data:
- VIX (CBOE Volatility Index) via yfinance (^VIX ticker)
- VIX3M (3-month VIX) via yfinance (^VIX3M ticker)
- CBOE Equity Put/Call Ratio via CBOE public CSV (CDN-gated; may be unavailable)

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

# yfinance tickers for VIX variants (replaces FRED / pandas_datareader)
_YF_VIX = "^VIX"
_YF_VIX3M = "^VIX3M"

# CBOE public equity put/call ratio CSV endpoint.
# Note: CBOE CDN (cdn.cboe.com) blocks programmatic access via Cloudflare
# bot-management. Fetches return 403 even with browser User-Agent headers.
# PCR data is treated as optional; absence degrades regime detection slightly.
_CBOE_PCR_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/PCALL_History.csv"
)


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
        """Fetch VIX and VIX3M via yfinance (^VIX, ^VIX3M tickers).

        Args:
            start_date: ISO date string (e.g. "2020-01-01"). Default: 2015-01-01.
            end_date: ISO date string. Default: today.

        Returns:
            DataFrame with columns: timestamp, vix, vix3m.
        """
        try:
            import yfinance as yf  # type: ignore
        except ImportError:
            logger.warning("[CBOE] yfinance not installed — cannot fetch VIX")
            return pd.DataFrame(columns=["timestamp", "vix", "vix3m"])

        start = start_date or "2015-01-01"
        end = end_date or datetime.today().strftime("%Y-%m-%d")

        try:
            raw = yf.download(
                [_YF_VIX, _YF_VIX3M],
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
            )
            if raw.empty:
                return pd.DataFrame(columns=["timestamp", "vix", "vix3m"])

            # Multi-index: (price_type, ticker) → extract Close
            closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            closes = closes.reset_index()
            # Rename Date index and ticker columns
            col0 = closes.columns[0]
            if str(col0).lower() in ("date", "datetime", "index"):
                closes = closes.rename(columns={col0: "timestamp"})
            else:
                closes.insert(0, "timestamp", closes.pop(col0))
            closes["timestamp"] = pd.to_datetime(closes["timestamp"])

            rename_map: dict = {}
            for col in closes.columns:
                s = str(col)
                if "VIX3M" in s.upper():
                    rename_map[col] = "vix3m"
                elif "VIX" in s.upper() and "vix3m" not in rename_map.values():
                    rename_map[col] = "vix"
            closes = closes.rename(columns=rename_map)

            for col in ("vix", "vix3m"):
                if col not in closes.columns:
                    closes[col] = float("nan")

            df = closes[["timestamp", "vix", "vix3m"]].dropna(
                how="all", subset=["vix", "vix3m"]
            )
            df = df.sort_values("timestamp").reset_index(drop=True)
            self._vix_cache = df
            logger.info(
                "[CBOE] Fetched VIX data via yfinance: %d rows (%s to %s)",
                len(df),
                start,
                end,
            )
            return df
        except Exception as exc:
            logger.warning("[CBOE] Failed to fetch VIX via yfinance: %s", exc)
            return pd.DataFrame(columns=["timestamp", "vix", "vix3m"])

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
            import requests as _req

            resp = _req.get(
                _CBOE_PCR_URL,
                timeout=15,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/csv,text/plain,*/*",
                    "Referer": "https://www.cboe.com/",
                },
            )
            if resp.status_code != 200:
                # CBOE CDN (Cloudflare-gated) blocks programmatic access.
                logger.debug(
                    "[CBOE] PCR endpoint returned HTTP %d — data unavailable",
                    resp.status_code,
                )
                return pd.DataFrame(columns=["timestamp", "put_call_ratio"])
            import io as _io

            raw = pd.read_csv(_io.StringIO(resp.text), dtype=str)
        except Exception as exc:
            logger.debug("[CBOE] PCR fetch failed: %s", exc)
            return pd.DataFrame(columns=["timestamp", "put_call_ratio"])

        # CBOE CSV columns vary; normalise
        if len(raw.columns) == 0:
            logger.warning("[CBOE] CSV returned no columns")
            return pd.DataFrame(columns=["timestamp", "put_call_ratio"])
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
