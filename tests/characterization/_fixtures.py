"""Synthetic fixture generators for characterization tests.

All generators are deterministic given the same seed.
Generates OHLCV data with realistic price dynamics for a fixed set of tickers
and date ranges, without requiring any external data files.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(
    tickers: list[str],
    start: str,
    end: str,
    freq: str = "1d",
    initial_price: float = 100.0,
    annual_vol: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV data.

    Returns a MultiIndex DataFrame with (Date, ticker) or a long-format
    DataFrame with columns [Date, ticker, Open, High, Low, Close, Volume].
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq=freq, tz="UTC")
    if len(dates) == 0:
        raise ValueError(f"Empty date range {start!r}→{end!r} with freq={freq!r}")

    dt = 1 / 252 if freq in ("1d", "D") else 1 / (252 * 390)
    daily_vol = annual_vol * np.sqrt(dt)

    rows = []
    for ticker in tickers:
        import hashlib

        _h = int(hashlib.sha1(ticker.encode()).hexdigest(), 16)
        price = initial_price * (1 + 0.01 * (_h % 10))
        for ts in dates:
            ret = rng.normal(0.0002, daily_vol)
            close = max(price * (1 + ret), 0.01)
            spread = close * 0.002
            high = close + abs(rng.normal(0, spread))
            low = close - abs(rng.normal(0, spread))
            open_ = low + rng.random() * (high - low)
            volume = int(rng.lognormal(15, 1))
            rows.append(
                {
                    "Date": ts,
                    "ticker": ticker,
                    "Open": round(open_, 4),
                    "High": round(high, 4),
                    "Low": round(low, 4),
                    "Close": round(close, 4),
                    "Volume": volume,
                }
            )
            price = close

    return pd.DataFrame(rows)


def make_crisis_scenario(
    name: str,
    tickers: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic crisis scenario with known characteristics.

    Parameters
    ----------
    name : "gfc_2008" | "covid_2020" | "rates_2022" | "calm_2017"
    """
    scenarios = {
        "gfc_2008": ("2008-09-01", "2009-03-31", 0.60, -0.001),
        "covid_2020": ("2020-02-01", "2020-05-31", 0.80, -0.0008),
        "rates_2022": ("2022-01-01", "2022-12-31", 0.30, -0.0004),
        "calm_2017": ("2017-01-01", "2017-12-31", 0.10, 0.0005),
    }
    if name not in scenarios:
        raise ValueError(f"Unknown scenario {name!r}. Choose from {list(scenarios)}")
    start, end, vol, drift = scenarios[name]

    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="1d", tz="UTC")
    rows = []
    for ticker in tickers:
        price = 100.0
        for ts in dates:
            ret = rng.normal(drift, vol * np.sqrt(1 / 252))
            close = max(price * (1 + ret), 0.01)
            high = close * (1 + abs(rng.normal(0, 0.003)))
            low = close * (1 - abs(rng.normal(0, 0.003)))
            open_ = low + rng.random() * (high - low)
            volume = int(rng.lognormal(15, 0.5))
            rows.append(
                {
                    "Date": ts,
                    "ticker": ticker,
                    "Open": round(open_, 4),
                    "High": round(high, 4),
                    "Low": round(low, 4),
                    "Close": round(close, 4),
                    "Volume": volume,
                }
            )
            price = close
    return pd.DataFrame(rows)
