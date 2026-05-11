"""EMA-Trend Cross-Section-Strategie (portiert aus Mainline ema_trend_v0).

Idee
----
Aus dem Mainline ``src/assembled_core/strategies/ema_trend_v0.py``:
Bench-Strategie "EMA20/EMA60". Long-Signal wenn EMA-fast > EMA-slow, Score
als normalized EMA-Spread.

Erweiterung-Variante: Cross-Section Long-Top-Quintile by EMA-Spread.

Test-Erwartung: orthogonaler Faktor zu Mom-12/1 — beide sind Trend, aber
unterschiedliche Time-Horizons (EMA-Spread = kurz-mittel, Mom-12/1 = mittel-lang).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EMATrendConfig:
    ema_fast: int = 20
    ema_slow: int = 60
    quantile_long: float = 0.2
    quantile_short: float = 0.2
    long_only: bool = True
    rebalance_freq: str = "M"  # "M" = monthly, "W" = weekly, "D" = daily


def compute_ema_spread(
    prices: pd.DataFrame, config: EMATrendConfig | None = None
) -> pd.DataFrame:
    """Berechne normalized EMA-Spread pro Symbol.

    Args:
        prices: DataFrame [date, symbol, close] (long format).
        config: EMATrendConfig.

    Returns:
        DataFrame [date, symbol, ema_spread].
    """
    cfg = config or EMATrendConfig()
    if prices.empty:
        return pd.DataFrame()
    out = prices.copy().sort_values(["symbol", "date"]).reset_index(drop=True)
    out["ema_fast"] = out.groupby("symbol")["close"].transform(
        lambda s: s.ewm(span=cfg.ema_fast, adjust=False).mean()
    )
    out["ema_slow"] = out.groupby("symbol")["close"].transform(
        lambda s: s.ewm(span=cfg.ema_slow, adjust=False).mean()
    )
    out["ema_spread"] = (out["ema_fast"] - out["ema_slow"]) / out["ema_slow"].replace(
        0, np.nan
    )
    return out[["date", "symbol", "ema_spread"]]


def cross_section_ema_signal(
    prices: pd.DataFrame, config: EMATrendConfig | None = None
) -> pd.DataFrame:
    """Long-Top-Quantile, Short-Bottom-Quantile (oder Long-Only) by EMA-Spread.

    Args:
        prices: DataFrame [date, symbol, close, return].
        config: EMATrendConfig.

    Returns:
        DataFrame [date, symbol, ema_spread, position, return].
    """
    cfg = config or EMATrendConfig()
    if prices.empty:
        return pd.DataFrame()
    sp = compute_ema_spread(prices, cfg)
    merged = prices.merge(sp, on=["date", "symbol"], how="left")
    merged = merged.sort_values(["symbol", "date"]).reset_index(drop=True)
    # t-1 lag
    merged["sig_lag"] = merged.groupby("symbol")["ema_spread"].shift(1)
    # Cross-section rank
    merged["sig_pct"] = merged.groupby("date")["sig_lag"].rank(pct=True)
    merged["position"] = 0.0
    long_mask = merged["sig_pct"] >= 1 - cfg.quantile_long
    short_mask = merged["sig_pct"] <= cfg.quantile_short
    merged.loc[long_mask, "position"] = 1.0
    if not cfg.long_only:
        merged.loc[short_mask, "position"] = -1.0
    # Equal-weight innerhalb der Sides
    n_long = merged.groupby("date")["position"].transform(lambda s: (s > 0).sum())
    n_short = merged.groupby("date")["position"].transform(lambda s: (s < 0).sum())
    merged.loc[long_mask, "position"] = 1.0 / n_long[long_mask]
    if not cfg.long_only:
        merged.loc[short_mask, "position"] = -1.0 / n_short[short_mask]
    return merged[["date", "symbol", "ema_spread", "position", "return"]]


def backtest_ema_trend(
    prices: pd.DataFrame, config: EMATrendConfig | None = None
) -> pd.Series:
    """Backtest der EMA-Trend-Strategie. Returns Tagesreturn-Series."""
    sig = cross_section_ema_signal(prices, config)
    if sig.empty:
        return pd.Series(dtype=float)
    sig["pnl"] = sig["position"] * sig["return"]
    daily = sig.groupby("date")["pnl"].sum()
    return daily


__all__ = [
    "EMATrendConfig",
    "compute_ema_spread",
    "cross_section_ema_signal",
    "backtest_ema_trend",
]
