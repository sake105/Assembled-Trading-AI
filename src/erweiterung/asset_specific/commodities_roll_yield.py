"""Commodities Futures-Curve-Shape + Roll-Yield.

Theorie
-------
Commodity-Futures haben Maturity-Curve:
- **Contango**: longer futures higher than spot (positive carry cost).
- **Backwardation**: longer futures LOWER than spot (negative cost = positive
  roll-yield für long).

**Roll-Yield** = (F_{near} − F_{far}) / F_{near} × (365 / days_between).
Positive roll-yield = backwardation = supplier shortage = commodity-trend strong.

Reference
---------
- Gorton, G. & Rouwenhorst, K. (2006). Facts and Fantasies about Commodity
  Futures. *FAJ* 62.
- Erb, C. & Harvey, C. (2006). Tactical and Strategic Value of Commodity
  Futures. *FAJ* 62.

Anwendung
---------
- **Carry-Strategy**: long backwardated commodities, short contango.
- **Convenience-Yield**: positive roll-yield = supplier signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def roll_yield(
    near_price: pd.Series,
    far_price: pd.Series,
    days_between: int = 30,
) -> pd.Series:
    """Compute annualized roll-yield.

    Args:
        near_price, far_price: aligned price-series.
        days_between: business days between contract expiries.

    Returns:
        Series of annualized roll-yields (positive = backwardation, profitable
        for long).
    """
    df = pd.concat([near_price, far_price], axis=1).dropna()
    df.columns = ["near", "far"]
    if days_between <= 0:
        return pd.Series(dtype=float)
    raw = (df["near"] - df["far"]) / df["near"]
    return raw * (365.0 / days_between)


def curve_steepness(contracts: pd.DataFrame, expiry_days: list[int]) -> pd.Series:
    """Slope of the futures curve.

    Args:
        contracts: DataFrame [date, c1, c2, c3, ...] for futures at increasing maturities.
        expiry_days: days-to-expiry per column.

    Returns:
        Series of slopes (rate of change per day).
    """
    df = contracts.copy()
    out = pd.Series(np.nan, index=df.index)
    for d, row in df.iterrows():
        vals = row.values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 2:
            continue
        x = np.array(expiry_days)[valid]
        y = vals[valid]
        if len(x) < 2 or x.std() == 0:
            continue
        slope, *_ = np.polyfit(x, y, 1)
        out.loc[d] = slope
    return out


def backwardation_cross_section(
    roll_yields: pd.DataFrame, n_long: int = 4, n_short: int = 4
) -> pd.DataFrame:
    """Cross-section: long top-N backwardated, short top-N contangoed.

    Args:
        roll_yields: DataFrame [date, commodity, roll_yield].
        n_long, n_short: legs.

    Returns:
        Positions per (date, commodity).
    """
    df = roll_yields.copy()
    df["rank"] = df.groupby("date")["roll_yield"].rank(ascending=False, method="first")
    n_total = df.groupby("date")["rank"].transform("max")
    df["position"] = 0.0
    df.loc[df["rank"] <= n_long, "position"] = 1.0 / n_long
    df.loc[df["rank"] > n_total - n_short, "position"] = -1.0 / n_short
    return df[["date", "commodity", "position", "roll_yield"]]


def momentum_in_commodity_curve(
    near_prices: pd.DataFrame, lookback: int = 252, skip: int = 21
) -> pd.DataFrame:
    """12-1 Momentum on near-contract commodity prices.

    Args:
        near_prices: DataFrame [date, commodity, near_price].

    Returns:
        DataFrame [date, commodity, momentum_score].
    """
    df = near_prices.copy().sort_values(["commodity", "date"])
    df["momentum"] = (
        df.groupby("commodity")["near_price"]
        .apply(lambda s: s.shift(skip) / s.shift(lookback) - 1)
        .reset_index(0, drop=True)
    )
    return df[["date", "commodity", "momentum"]]


def combined_carry_momentum(
    roll_yields: pd.DataFrame,
    momentum: pd.DataFrame,
    carry_weight: float = 0.5,
) -> pd.DataFrame:
    """Combined signal: weighted average of carry + momentum.

    Asness/Moskowitz/Pedersen (2013) "Value and Momentum Everywhere".
    """
    merged = roll_yields.merge(momentum, on=["date", "commodity"], how="inner")

    def _z(s):
        mu = s.mean()
        sd = s.std()
        return (s - mu) / sd if sd > 0 else s * 0

    merged["carry_z"] = merged.groupby("date")["roll_yield"].transform(_z)
    merged["mom_z"] = merged.groupby("date")["momentum"].transform(_z)
    merged["combined"] = (
        carry_weight * merged["carry_z"] + (1 - carry_weight) * merged["mom_z"]
    )
    return merged[["date", "commodity", "combined", "carry_z", "mom_z"]]


__all__ = [
    "roll_yield",
    "curve_steepness",
    "backwardation_cross_section",
    "momentum_in_commodity_curve",
    "combined_carry_momentum",
]
