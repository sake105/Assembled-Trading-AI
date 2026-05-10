"""Yahoo Finance Options Chains — IV, Put/Call-Ratio, Skew (frei via yfinance).

Quelle
------
yfinance scrapt Options-Chains von Yahoo Finance — frei, aber Schema-Änderungen
möglich. Für Forschung & Backtests akzeptabel; für Live-Trading sollten
robustere Quellen genutzt werden (siehe ``docs/erweiterung/PAID_DATA_WISHLIST.md``).

Inhalt
------
- Strike, Bid/Ask, Last
- Volume, Open Interest
- Implied Volatility
- Greek-Schätzungen (z. T. fehlend)

Ableitbare Faktoren
-------------------
1. **Put/Call-Ratio**: Hoher PC-Ratio = bärisches Sentiment, Mean-Reversion-Signal.
2. **IV-Skew**: 25Δ-Put-IV − 25Δ-Call-IV.  Hohe positive Skew = Crash-Furcht.
3. **IV-Term-Structure**: Front-IV / Back-IV.  Inversion = Stress.
4. **Open-Interest-Konzentration**: Strikes mit hohem OI = "Magnet"-Levels.

PIT
---
yfinance liefert nur den **aktuellen** Snapshot (nicht historisch).  Für
historische IV/Skew benötigt man kommerzielle Daten (OptionMetrics, ORATS).
Für Live-/Walk-Forward-Anwendung jedoch ausreichend.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from erweiterung._base import (
    FetchResult,
    get_cache_dir,
    rate_limited,
    retry_with_backoff,
    stable_hash,
)

logger = logging.getLogger(__name__)


@rate_limited(min_interval_s=0.5)
@retry_with_backoff(max_attempts=3, base_delay=2.0)
def _yfinance_options(symbol: str) -> tuple[float, list[dict]]:
    """Hole alle Options-Chains für ``symbol``. Liefert (spot, list_of_chain_dicts)."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance required") from e

    tk = yf.Ticker(symbol)
    expirations = list(tk.options or [])
    spot = (
        float(tk.info.get("regularMarketPrice", float("nan")))
        if tk.info
        else float("nan")
    )
    if not np.isfinite(spot):
        # fallback via history
        hist = tk.history(period="5d", auto_adjust=False)
        if not hist.empty:
            spot = float(hist["Close"].iloc[-1])
    out: list[dict] = []
    for exp in expirations:
        try:
            chain = tk.option_chain(exp)
        except Exception as e:  # noqa: BLE001
            logger.info("[yopt] %s %s skip: %s", symbol, exp, e)
            continue
        out.append(
            {
                "expiration": exp,
                "calls": chain.calls.assign(side="C", expiration=exp),
                "puts": chain.puts.assign(side="P", expiration=exp),
            }
        )
    return spot, out


def fetch_options_snapshot(
    symbol: str, use_cache: bool = True, cache_ttl_hours: int = 4
) -> FetchResult:
    """Snapshot aller Options-Chains für ``symbol``.

    Returns:
        FetchResult mit DataFrame der zusammengeführten Calls/Puts +
        Notes mit ``spot=...``.
    """
    cache_key = stable_hash("yopt", symbol.upper())
    cache_path = get_cache_dir("yahoo_options") / f"{cache_key}.parquet"
    spot_path = get_cache_dir("yahoo_options") / f"{cache_key}.spot.txt"
    if use_cache and cache_path.exists() and spot_path.exists():
        age_h = (pd.Timestamp.utcnow().timestamp() - cache_path.stat().st_mtime) / 3600
        if age_h < cache_ttl_hours:
            df = pd.read_parquet(cache_path)
            spot = float(spot_path.read_text())
            return FetchResult(
                df,
                "yahoo_options",
                pd.Timestamp.utcnow(),
                len(df),
                f"spot={spot};cache",
            )

    spot, chains = _yfinance_options(symbol)
    if not chains:
        return FetchResult(
            pd.DataFrame(), "yahoo_options", pd.Timestamp.utcnow(), 0, "no_chains"
        )

    frames = []
    for c in chains:
        for side_df in (c["calls"], c["puts"]):
            if side_df is None or side_df.empty:
                continue
            frames.append(side_df)
    if not frames:
        return FetchResult(
            pd.DataFrame(), "yahoo_options", pd.Timestamp.utcnow(), 0, "empty"
        )
    df = pd.concat(frames, ignore_index=True)
    df["expiration"] = pd.to_datetime(df["expiration"], utc=True)
    df["symbol"] = symbol.upper()
    if use_cache:
        df.to_parquet(cache_path, index=False)
        spot_path.write_text(str(spot))
    return FetchResult(
        df, "yahoo_options", pd.Timestamp.utcnow(), len(df), f"spot={spot}"
    )


def put_call_ratio(df: pd.DataFrame) -> dict:
    """Volume- und OI-basierte Put/Call-Ratios."""
    if df.empty:
        return {"pc_volume": np.nan, "pc_oi": np.nan, "pc_volume_near": np.nan}
    calls = df[df["side"] == "C"]
    puts = df[df["side"] == "P"]
    cv = calls["volume"].fillna(0).sum()
    pv = puts["volume"].fillna(0).sum()
    co = calls["openInterest"].fillna(0).sum()
    po = puts["openInterest"].fillna(0).sum()
    pc_v = pv / cv if cv > 0 else np.nan
    pc_o = po / co if co > 0 else np.nan

    # Near-the-money (front-expiration)
    near_exp = df["expiration"].min()
    near = df[df["expiration"] == near_exp]
    near_calls = near[near["side"] == "C"]["volume"].fillna(0).sum()
    near_puts = near[near["side"] == "P"]["volume"].fillna(0).sum()
    pc_near = near_puts / near_calls if near_calls > 0 else np.nan

    return {"pc_volume": pc_v, "pc_oi": pc_o, "pc_volume_near": pc_near}


def iv_skew_25delta(df: pd.DataFrame, spot: float) -> dict:
    """IV-Skew = IV(25Δ-Put) − IV(25Δ-Call) für die Front-Expiration.

    Approximation: 25Δ-Put liegt bei ~95% des Spots, 25Δ-Call bei ~105%.

    Returns:
        Dict mit ``skew_25d``, ``front_atm_iv``, ``term_slope``.
    """
    if df.empty or not np.isfinite(spot):
        return {"skew_25d": np.nan, "front_atm_iv": np.nan, "term_slope": np.nan}

    df = df.copy()
    df["distance_to_spot"] = (df["strike"] - spot) / spot
    front_exp = df["expiration"].min()
    back_exp = df["expiration"].max()
    front = df[df["expiration"] == front_exp]
    back = df[df["expiration"] == back_exp] if back_exp != front_exp else pd.DataFrame()

    # 25Δ-Put: Strike ≈ 0.95 × spot
    put_target = 0.95
    call_target = 1.05
    front_puts = front[front["side"] == "P"].copy()
    front_calls = front[front["side"] == "C"].copy()
    if front_puts.empty or front_calls.empty:
        return {"skew_25d": np.nan, "front_atm_iv": np.nan, "term_slope": np.nan}

    front_puts["dist"] = (front_puts["strike"] / spot - put_target).abs()
    front_calls["dist"] = (front_calls["strike"] / spot - call_target).abs()
    iv_put = front_puts.sort_values("dist").iloc[0].get("impliedVolatility", np.nan)
    iv_call = front_calls.sort_values("dist").iloc[0].get("impliedVolatility", np.nan)
    skew = (
        float(iv_put) - float(iv_call)
        if pd.notna(iv_put) and pd.notna(iv_call)
        else np.nan
    )

    # ATM-IV front
    atm_front = front.iloc[(front["strike"] - spot).abs().argsort()[:5]]
    front_atm_iv = (
        float(atm_front["impliedVolatility"].mean()) if not atm_front.empty else np.nan
    )

    term_slope = np.nan
    if not back.empty:
        atm_back = back.iloc[(back["strike"] - spot).abs().argsort()[:5]]
        if not atm_back.empty:
            back_atm = float(atm_back["impliedVolatility"].mean())
            if pd.notna(back_atm) and pd.notna(front_atm_iv) and back_atm > 0:
                term_slope = (front_atm_iv / back_atm) - 1.0  # >0 = inverted (stress)
    return {
        "skew_25d": skew,
        "front_atm_iv": front_atm_iv,
        "term_slope": term_slope,
    }


def options_factor_panel(symbols: list[str]) -> pd.DataFrame:
    """Snapshot-Panel für eine Symbol-Liste.

    Returns:
        DataFrame mit ``[as_of, symbol, spot, pc_volume, pc_oi, skew_25d,
        front_atm_iv, term_slope]``.
    """
    rows = []
    for sym in symbols:
        try:
            res = fetch_options_snapshot(sym, use_cache=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("[yopt] %s skip: %s", sym, e)
            continue
        if res.df.empty:
            continue
        spot_str = res.notes.split("spot=")[-1].split(";")[0]
        try:
            spot = float(spot_str)
        except ValueError:
            spot = np.nan
        pcr = put_call_ratio(res.df)
        sk = iv_skew_25delta(res.df, spot)
        rows.append(
            {
                "as_of": pd.Timestamp.utcnow(),
                "symbol": sym.upper(),
                "spot": spot,
                **pcr,
                **sk,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "fetch_options_snapshot",
    "put_call_ratio",
    "iv_skew_25delta",
    "options_factor_panel",
]
