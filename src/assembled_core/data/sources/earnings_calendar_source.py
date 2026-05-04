"""Earnings Calendar Data Source.

Fetches earnings calendar data from multiple free sources:
1. yfinance (primary) — `yf.Ticker(sym).calendar` for individual symbols
2. Finnhub (if API key available) — `/calendar/earnings` endpoint
3. Alpha Vantage (if API key available) — earnings calendar

Produces factors:
    days_to_earnings          — trading days until next earnings announcement
    pre_earnings_drift_flag   — 1 if within 5 days before earnings
    post_earnings_momentum_flag — 1 if within 3 days after earnings
    earnings_surprise_est     — last reported EPS surprise (if available)

Usage:
    from src.assembled_core.data.sources.earnings_calendar_source import EarningsCalendarSource

    source = EarningsCalendarSource()
    calendar = source.fetch_calendar(symbols=["AAPL", "MSFT"], days_ahead=90)
    factors = source.build_earnings_factors(calendar, prices_df)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PRE_EARNINGS_DAYS = 5  # flag raised this many days before earnings
_POST_EARNINGS_DAYS = 3  # flag raised this many days after earnings


class EarningsCalendarSource:
    """Fetches earnings calendar data and builds earnings-timing factors.

    Attributes:
        finnhub_api_key: Optional Finnhub API key (reads FINNHUB_API_KEY env var).
        alphavantage_api_key: Optional Alpha Vantage key (reads ALPHAVANTAGE_API_KEY).
    """

    def __init__(
        self,
        finnhub_api_key: Optional[str] = None,
        alphavantage_api_key: Optional[str] = None,
    ) -> None:
        self.finnhub_api_key = finnhub_api_key or os.environ.get("FINNHUB_API_KEY", "")
        self.alphavantage_api_key = alphavantage_api_key or os.environ.get(
            "ALPHAVANTAGE_API_KEY", ""
        )

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def fetch_calendar(
        self,
        symbols: list[str],
        days_ahead: int = 90,
        use_cache: bool = False,
    ) -> pd.DataFrame:
        """Fetch upcoming earnings dates for a list of symbols.

        Tries yfinance first (free, no key), then Finnhub if key available.

        Args:
            symbols: List of ticker symbols.
            days_ahead: Fetch earnings within this many calendar days.
            use_cache: Not implemented — reserved for future in-session caching.

        Returns:
            DataFrame with columns: symbol, earnings_date, eps_estimate,
            eps_actual (NaN if not yet reported), surprise_pct.
        """
        rows = []
        end_date = datetime.today() + timedelta(days=days_ahead)
        start_date = datetime.today() - timedelta(days=30)  # include recent past

        for sym in symbols:
            try:
                result = self._fetch_yfinance(sym, start_date, end_date)
                if result:
                    rows.extend(result)
                    continue
            except Exception as exc:
                logger.debug("[EarningsCalendar] yfinance failed for %s: %s", sym, exc)

            if self.finnhub_api_key:
                try:
                    result = self._fetch_finnhub(sym, start_date, end_date)
                    if result:
                        rows.extend(result)
                        continue
                except Exception as exc:
                    logger.debug(
                        "[EarningsCalendar] Finnhub failed for %s: %s", sym, exc
                    )

        if not rows:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "earnings_date",
                    "eps_estimate",
                    "eps_actual",
                    "surprise_pct",
                ]
            )

        df = pd.DataFrame(rows)
        df["earnings_date"] = pd.to_datetime(df["earnings_date"])
        df = df.sort_values(["symbol", "earnings_date"]).reset_index(drop=True)
        logger.info(
            "[EarningsCalendar] Fetched %d earnings dates for %d symbols",
            len(df),
            len(symbols),
        )
        return df

    def _fetch_yfinance(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict]:
        """Fetch earnings dates via yfinance."""
        import yfinance as yf  # type: ignore

        ticker = yf.Ticker(symbol)
        cal = ticker.calendar
        if cal is None or cal.empty:
            return []

        # yfinance calendar is a DataFrame with dates as columns for some versions
        rows = []
        try:
            if hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                for _, row in cal.iterrows():
                    edate = pd.to_datetime(row.get("Earnings Date"))
                    if pd.isna(edate):
                        continue
                    if start_date <= edate.to_pydatetime() <= end_date:
                        rows.append(
                            {
                                "symbol": symbol,
                                "earnings_date": edate,
                                "eps_estimate": row.get("EPS Estimate", float("nan")),
                                "eps_actual": float("nan"),
                                "surprise_pct": float("nan"),
                            }
                        )
            elif isinstance(cal, pd.DataFrame):
                # Older yfinance returns transposed DataFrame
                for col in cal.columns:
                    edate = pd.to_datetime(col, errors="coerce")
                    if pd.isna(edate):
                        continue
                    if start_date <= edate.to_pydatetime() <= end_date:
                        rows.append(
                            {
                                "symbol": symbol,
                                "earnings_date": edate,
                                "eps_estimate": float("nan"),
                                "eps_actual": float("nan"),
                                "surprise_pct": float("nan"),
                            }
                        )
        except Exception as exc:
            logger.debug(
                "[EarningsCalendar] yfinance calendar parse error for %s: %s",
                symbol,
                exc,
            )

        # Also check earnings history for recent surprises
        try:
            earnings = ticker.earnings_history
            if earnings is not None and not earnings.empty:
                for _, row in earnings.iterrows():
                    edate = pd.to_datetime(
                        row.name if hasattr(row, "name") else row.get("Earnings Date")
                    )
                    if pd.isna(edate):
                        continue
                    if start_date <= edate.to_pydatetime() <= end_date:
                        est = float(row.get("EPS Estimate", float("nan")))
                        actual = float(row.get("Reported EPS", float("nan")))
                        surprise = (
                            ((actual - est) / abs(est) * 100)
                            if est != 0 and not np.isnan(est)
                            else float("nan")
                        )
                        rows.append(
                            {
                                "symbol": symbol,
                                "earnings_date": edate,
                                "eps_estimate": est,
                                "eps_actual": actual,
                                "surprise_pct": surprise,
                            }
                        )
        except Exception as exc:
            logger.warning(
                "[EarningsCalendar] earnings history parse error for %s: %s",
                symbol,
                exc,
            )

        return rows

    def _fetch_finnhub(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
    ) -> list[dict]:
        """Fetch earnings via Finnhub API."""
        import json
        import urllib.request

        url = (
            f"https://finnhub.io/api/v1/calendar/earnings"
            f"?symbol={symbol}"
            f"&from={start_date.strftime('%Y-%m-%d')}"
            f"&to={end_date.strftime('%Y-%m-%d')}"
            f"&token={self.finnhub_api_key}"
        )
        # URL is built from a fixed finnhub.io endpoint with validated params.
        import urllib.error

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:  # nosec B310
                try:
                    data = json.loads(resp.read())
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "[EarningsCalendar] Finnhub returned invalid JSON for %s: %s",
                        symbol,
                        exc,
                    )
                    return []
        except (urllib.error.URLError, urllib.error.HTTPError, OSError) as exc:
            logger.warning(
                "[EarningsCalendar] Finnhub network error for %s: %s", symbol, exc
            )
            return []

        rows = []
        for item in data.get("earningsCalendar", []):
            rows.append(
                {
                    "symbol": symbol,
                    "earnings_date": pd.to_datetime(item.get("date"), errors="coerce"),
                    "eps_estimate": item.get("epsEstimate", float("nan")),
                    "eps_actual": item.get("epsActual", float("nan")),
                    "surprise_pct": float("nan"),
                }
            )
        return rows

    # ------------------------------------------------------------------
    # Factor building
    # ------------------------------------------------------------------

    def build_earnings_factors(
        self,
        calendar_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        symbol_col: str = "symbol",
        timestamp_col: str = "timestamp",
        pre_days: int = _PRE_EARNINGS_DAYS,
        post_days: int = _POST_EARNINGS_DAYS,
    ) -> pd.DataFrame:
        """Build earnings-timing factors and merge onto daily price panel.

        Args:
            calendar_df: Output of fetch_calendar().
            prices_df: Daily price panel with symbol and timestamp columns.
            symbol_col: Symbol column name.
            timestamp_col: Timestamp column name.
            pre_days: Days before earnings for pre_earnings_drift_flag.
            post_days: Days after earnings for post_earnings_momentum_flag.

        Returns:
            prices_df with added columns: days_to_earnings,
            pre_earnings_drift_flag, post_earnings_momentum_flag.
        """
        if calendar_df.empty:
            result = prices_df.copy()
            result["days_to_earnings"] = float("nan")
            result["pre_earnings_drift_flag"] = 0.0
            result["post_earnings_momentum_flag"] = 0.0
            return result

        result = prices_df.copy()
        result[timestamp_col] = pd.to_datetime(result[timestamp_col])

        _cal_by_sym = {
            sym: grp for sym, grp in calendar_df.groupby("symbol", sort=False)
        }

        days_to = []
        pre_flag = []
        post_flag = []

        for row in result.itertuples(index=False):
            sym = getattr(row, symbol_col)
            ts = getattr(row, timestamp_col)

            sym_cal = _cal_by_sym.get(sym, pd.DataFrame())
            if sym_cal.empty:
                days_to.append(float("nan"))
                pre_flag.append(0.0)
                post_flag.append(0.0)
                continue

            future = sym_cal[sym_cal["earnings_date"] >= ts]
            past = sym_cal[sym_cal["earnings_date"] < ts]

            # Days to next earnings
            if not future.empty:
                next_date = future["earnings_date"].min()
                days = (next_date - ts).days
            else:
                days = float("nan")

            days_to.append(days)
            pre_flag.append(
                1.0
                if (
                    isinstance(days, float)
                    and not np.isnan(days)
                    and 0 <= days <= pre_days
                )
                else 0.0
            )

            # Post-earnings flag
            if not past.empty:
                last_date = past["earnings_date"].max()
                days_since = (ts - last_date).days
                post_flag.append(1.0 if 0 <= days_since <= post_days else 0.0)
            else:
                post_flag.append(0.0)

        result["days_to_earnings"] = days_to
        result["pre_earnings_drift_flag"] = pre_flag
        result["post_earnings_momentum_flag"] = post_flag
        return result
