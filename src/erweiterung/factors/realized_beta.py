"""Realized Beta + Beta-Decomposition + Beta-Hedging.

Theorie
-------
Klassisches β: COV(r_asset, r_market) / VAR(r_market) — geschätzt over window.

Realized Beta (Andersen/Bollerslev/Diebold/Vega 2003) nutzt intraday-realized
covariance für präzisere Schätzung. Wir implementieren daily-realized-Beta
basierend auf overlapping daily returns.

Beta-Decomposition
------------------
β_total = β_systematic + β_specific  (latter aus residual-orthogonalization)

Beta-Hedging
------------
hedge_size = β × asset_notional / market_notional

Anwendung
---------
- Market-neutral portfolio construction
- Beta-stabile Returns für long-short strategies
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_beta(
    asset_returns: pd.Series, market_returns: pd.Series, window: int = 60
) -> pd.Series:
    """Rolling β via OLS — analytical."""
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    df.columns = ["a", "m"]
    cov = df["a"].rolling(window).cov(df["m"])
    var = df["m"].rolling(window).var()
    return (cov / var.replace(0, np.nan)).rename("beta")


def realized_beta_daily(
    asset_returns: pd.Series, market_returns: pd.Series, window: int = 22
) -> pd.Series:
    """Realized Beta as sum-product / sum-square over rolling-window."""
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    df.columns = ["a", "m"]
    cov_sum = (df["a"] * df["m"]).rolling(window).sum()
    var_sum = (df["m"] ** 2).rolling(window).sum()
    return (cov_sum / var_sum.replace(0, np.nan)).rename("realized_beta")


def beta_components(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
) -> pd.DataFrame:
    """Decompose β into upside-β + downside-β.

    Returns:
        DataFrame [date, beta, upside_beta, downside_beta].
    """
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    df.columns = ["a", "m"]
    df["up"] = (df["m"] > 0).astype(int)
    df["down"] = (df["m"] < 0).astype(int)

    def _beta(sub: pd.DataFrame) -> float:
        if len(sub) < 5 or sub["m"].var() == 0:
            return float("nan")
        return float(sub[["a", "m"]].cov().iat[0, 1] / sub["m"].var())

    out_rows = []
    for end in range(window, len(df) + 1):
        sub = df.iloc[end - window : end]
        out_rows.append(
            {
                "date": df.index[end - 1],
                "beta": _beta(sub),
                "upside_beta": (
                    _beta(sub[sub["up"] == 1]) if sub["up"].sum() >= 5 else np.nan
                ),
                "downside_beta": (
                    _beta(sub[sub["down"] == 1]) if sub["down"].sum() >= 5 else np.nan
                ),
            }
        )
    return pd.DataFrame(out_rows).set_index("date")


def beta_hedge_size(
    asset_notional: float, beta: float, market_notional_per_unit: float = 1.0
) -> float:
    """Hedge-Size in Units des Markts.

    Args:
        asset_notional: dollar value of asset position.
        beta: asset's β vs market.
        market_notional_per_unit: $ per 1 unit of market proxy (e.g. $ per ES future contract).

    Returns:
        Number of market-units to short for β-hedge.
    """
    if market_notional_per_unit == 0:
        return 0.0
    return -beta * asset_notional / market_notional_per_unit


def beta_neutralize_portfolio(
    weights: pd.Series, betas: pd.Series, target_beta: float = 0.0
) -> tuple[pd.Series, float]:
    """Compute portfolio-beta and required market-hedge to reach target.

    Args:
        weights: portfolio weights per asset.
        betas: β je asset (gleiche Indizes wie weights).
        target_beta: ziel-β des Portfolios (default = 0 = market-neutral).

    Returns:
        Tuple ``(weights, market_hedge_notional)``:
        - weights bleibt unverändert.
        - market_hedge_notional = Δβ × Σ|weights| (so dass nach short von
          ``hedge`` Markteinheiten das resultierende Portfolio-β = target).
          Negativ ⇒ short der Market-Position.
        Der Caller integriert diese Hedge-Position selbst in seine Execution.
    """
    portfolio_beta = float((weights * betas).sum())
    market_hedge_notional = -(portfolio_beta - target_beta)
    return weights.copy(), float(market_hedge_notional)


__all__ = [
    "rolling_beta",
    "realized_beta_daily",
    "beta_components",
    "beta_hedge_size",
    "beta_neutralize_portfolio",
]
