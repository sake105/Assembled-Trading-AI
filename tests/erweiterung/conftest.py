"""Shared fixtures for erweiterung tests.

Bevorzugt echte Daten (yfinance_long-Cache, sample-panels) gegenüber
synthetic noise. Synthetic-Fallback nur für Edge-Cases / Unit-Tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Isoliere Disk-Cache pro Test."""
    monkeypatch.setenv("ERWEITERUNG_CACHE_DIR", str(tmp_path / "cache"))


# ============================================================================
# REAL-DATA fixtures (preferred)
# ============================================================================


def _load_xa_long_returns() -> pd.DataFrame | None:
    cache_dir = Path("data/cache/yfinance_long")
    if not cache_dir.exists():
        return None
    symbols = [
        "SPY",
        "QQQ",
        "IWM",
        "EFA",
        "EEM",
        "AGG",
        "TLT",
        "HYG",
        "GLD",
        "SLV",
        "DBC",
    ]
    frames = []
    for sym in symbols:
        p = cache_dir / f"{sym}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).reset_index()
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        return None
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    wide_close = panel.pivot_table(
        index="date", columns="symbol", values="close"
    ).sort_index()
    return wide_close.pct_change().dropna()


def _load_eq_long_returns() -> pd.DataFrame | None:
    src = Path("data/sample/watchlist_2007_2026.parquet")
    if not src.exists():
        return None
    df = pd.read_parquet(src)
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["return"] = df.groupby("symbol")["close"].pct_change()
    return (
        df.pivot_table(index="date", columns="symbol", values="return")
        .sort_index()
        .fillna(0)
    )


@pytest.fixture(scope="session")
def real_xa_returns() -> pd.DataFrame:
    """11-ETF Cross-Asset wide returns (real, ~19 years)."""
    data = _load_xa_long_returns()
    if data is None or data.empty:
        pytest.skip("Real cross-asset data missing (data/cache/yfinance_long)")
    return data


@pytest.fixture(scope="session")
def real_eq_returns_wide() -> pd.DataFrame:
    """22 Mega-Caps wide returns (real, ~19 years)."""
    data = _load_eq_long_returns()
    if data is None or data.empty:
        pytest.skip("Real equity data missing")
    return data


@pytest.fixture(scope="session")
def real_vix() -> pd.Series:
    """VIX close series (real, since 2018)."""
    p = Path("output/macro.parquet")
    if not p.exists():
        pytest.skip("macro.parquet missing")
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    if "vix_close" not in df.columns:
        pytest.skip("VIX not in macro panel")
    return df["vix_close"].dropna()


@pytest.fixture(scope="session")
def real_news_panel() -> pd.DataFrame:
    """News-sentiment-fused panel (real, ~5 months sparse)."""
    p = Path("output/news_sentiment_fused.parquet")
    if not p.exists():
        pytest.skip("news_sentiment_fused.parquet missing")
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


# ============================================================================
# SYNTHETIC fixtures (fallback / edge-cases)
# ============================================================================


@pytest.fixture
def synthetic_returns():
    """Reproduzierbare Returns für 5 Symbole, 500 Tage (Edge-Case-Tests)."""
    rng = np.random.default_rng(42)
    n_days = 500
    n_sym = 5
    dates = pd.date_range("2022-01-01", periods=n_days, freq="B", tz="UTC")
    sym_names = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    factor = rng.normal(0.0005, 0.012, n_days)
    idio = rng.normal(0, 0.015, (n_days, n_sym))
    returns = factor[:, None] * np.array([1.0, 0.9, 0.7, -0.3, 0.5]) + idio
    df = pd.DataFrame(returns, index=dates, columns=sym_names)
    return df


@pytest.fixture
def synthetic_prices(synthetic_returns):
    """Cumulative price levels."""
    return (1 + synthetic_returns).cumprod() * 100


@pytest.fixture
def synthetic_panel(synthetic_returns):
    """Long-format panel."""
    rows = []
    for sym in synthetic_returns.columns:
        for d, r in synthetic_returns[sym].items():
            rows.append({"date": d, "symbol": sym, "return": float(r)})
    return pd.DataFrame(rows)
