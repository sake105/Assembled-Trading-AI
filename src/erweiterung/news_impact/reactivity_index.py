"""News-Reactivity-Index — wie stark reagiert ein Asset auf News?

Theorie
-------
Different stocks have different news-elasticity:
- Small-caps + low-coverage: extrem reaktiv (jede news bewegt ±5%).
- Mega-caps: gedämpft (news ist eingepreist).
- Firms mit hohem Retail-Ownership: reaktiver.

Empirisches Modell:
    |r_{t+1}| = α + β · |sentiment_t| · I[news_t] + ε

mit I[news_t] = 1 wenn news arrived an Tag t. β = Reactivity-Coefficient.

Anwendung
---------
- Risk-Management: erwartete Volatility-Spikes bei bekanntem News-Flow.
- Stock-Selection: hochreaktive Stocks für News-getriebene Strategies.
- Position-Sizing: skaliere Trade-Size invers zu Reactivity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ReactivityFit:
    alpha: float  # baseline |return|
    beta: float  # additional |return| per unit |sentiment| on news days
    r_squared: float
    n_obs: int


def fit_reactivity(
    news_df: pd.DataFrame,
    returns_panel: pd.DataFrame,
    symbol: str,
    sentiment_col: str = "sentiment",
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> ReactivityFit:
    """Fit Reactivity-Index per Symbol.

    Args:
        news_df: News for this and other symbols.
        returns_panel: long-format returns.
        symbol: the asset to fit.

    Returns:
        ReactivityFit.
    """
    panel_sym = returns_panel[returns_panel[symbol_col] == symbol]
    if panel_sym.empty:
        raise ValueError(f"no returns for {symbol}")
    # build daily series
    ret_series = panel_sym.set_index(date_col)["return"]
    ret_series.index = pd.to_datetime(ret_series.index, utc=True)

    news_sym = news_df[news_df[symbol_col] == symbol].copy()
    news_sym[date_col] = pd.to_datetime(news_sym[date_col], utc=True)
    # Aggregate sentiment per day (some days may have multiple news)
    abs_sent = news_sym.groupby(date_col)[sentiment_col].apply(
        lambda x: float(np.abs(x).mean())
    )
    has_news = pd.Series(1.0, index=abs_sent.index)

    # Align on returns next-day
    abs_ret_next = ret_series.abs().shift(-1)
    df = pd.DataFrame(
        {
            "abs_ret_next": abs_ret_next,
            "abs_sent_news_day": abs_sent.reindex(ret_series.index, fill_value=0),
            "has_news": has_news.reindex(ret_series.index, fill_value=0),
        }
    ).dropna()
    if len(df) < 30:
        raise ValueError("not enough overlap")

    # OLS: |r_{t+1}| = α + β · |sent_t| · has_news_t
    X = np.column_stack(
        [np.ones(len(df)), df["abs_sent_news_day"].values * df["has_news"].values]
    )
    y = df["abs_ret_next"].values
    beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta_full
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return ReactivityFit(
        alpha=float(beta_full[0]),
        beta=float(beta_full[1]),
        r_squared=r2,
        n_obs=len(df),
    )


def reactivity_panel(
    news_df: pd.DataFrame,
    returns_panel: pd.DataFrame,
    min_news: int = 20,
) -> pd.DataFrame:
    """Reactivity-Fit for every symbol with sufficient news count.

    Returns:
        DataFrame [symbol, alpha, beta, r_squared, n_obs] sorted by β desc.
    """
    symbols = news_df["symbol"].value_counts()
    eligible = symbols[symbols >= min_news].index.tolist()
    rows = []
    for sym in eligible:
        try:
            fit = fit_reactivity(news_df, returns_panel, sym)
            rows.append(
                {
                    "symbol": sym,
                    "alpha": fit.alpha,
                    "beta": fit.beta,
                    "r_squared": fit.r_squared,
                    "n_obs": fit.n_obs,
                }
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
    return (
        pd.DataFrame(rows).sort_values("beta", ascending=False).reset_index(drop=True)
    )


def reactivity_position_scaling(
    base_position: pd.Series, reactivities: pd.Series, scale_target_mean: float = 1.0
) -> pd.Series:
    """Scale positions inversely to reactivity (lower reactivity → larger).

    Args:
        base_position: pd.Series indexed by symbol.
        reactivities: β-coefficients per symbol.
        scale_target_mean: target average scaling factor.

    Returns:
        Adjusted positions.
    """
    common = base_position.index.intersection(reactivities.index)
    pos = base_position.loc[common]
    reac = reactivities.loc[common].clip(lower=1e-6)
    scaling = 1.0 / reac
    scaling = scaling * scale_target_mean / scaling.mean()
    return pos * scaling


__all__ = [
    "ReactivityFit",
    "fit_reactivity",
    "reactivity_panel",
    "reactivity_position_scaling",
]
