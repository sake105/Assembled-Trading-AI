"""News-Impact Decay Model (Pasini/Veronese 2014; Tetlock 2007).

Theorie
-------
News-Shocks haben **exponentially decaying** Impact auf Asset-Returns:

    r_{t+h} | news_t = α + β · sentiment_t · exp(-λ h) + ε_{t+h}

mit λ = Decay-Rate. Half-Life = ln(2)/λ.

Empirisch
---------
Tetlock (2007): Pessimismus in Dow-Jones-News prognostiziert kurzfristige
SP500-Reverals mit Half-Life ~3-5 Tage. Negative news decay faster than positive.

Anwendung
---------
- Sizing: aktuelle News-Impact-Erwartung × Position-Size.
- Strategy: trade contrarian gegen extreme Sentiment-Spikes mit Decay-Profile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DecayFit:
    alpha: float
    beta: float
    decay_lambda: float
    half_life_days: float
    r_squared: float
    n_obs: int


def fit_news_decay_model(
    news_df: pd.DataFrame,
    returns_panel: pd.DataFrame,
    sentiment_col: str = "sentiment",
    date_col: str = "date",
    symbol_col: str = "symbol",
    horizons: tuple[int, ...] = (1, 2, 3, 5, 7, 10, 14, 21),
) -> DecayFit:
    """Fit exponential-decay impact model on news → forward-returns.

    Algorithmus
    -----------
    1. Für jeden Horizon h: compute cumulative-return r_{t→t+h} per (symbol, news_date).
    2. Stack across alle (h, symbols, dates).
    3. OLS: r_h = α + β · s · exp(-λ h) + ε.
       Da λ in exp non-linear, grid-search λ + OLS auf (α, β).

    Args:
        news_df: DataFrame mit [date, symbol, sentiment]. sentiment ∈ [-1, 1].
        returns_panel: DataFrame [date, symbol, return].
        sentiment_col, date_col, symbol_col: column names.
        horizons: forward-day-horizons.

    Returns:
        DecayFit mit β, λ, half-life.
    """
    if news_df.empty or returns_panel.empty:
        raise ValueError("empty input")
    # Build cumulative return panel
    pivot_ret = returns_panel.pivot_table(
        index=date_col, columns=symbol_col, values="return"
    ).sort_index()

    rows = []
    for _, row in news_df.iterrows():
        d = pd.Timestamp(row[date_col])
        sym = row[symbol_col]
        try:
            s = float(row[sentiment_col])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(s):
            continue  # skip NaN/inf sentiments
        if sym not in pivot_ret.columns or d not in pivot_ret.index:
            continue
        d_idx = pivot_ret.index.get_loc(d)
        for h in horizons:
            if d_idx + h >= len(pivot_ret):
                continue
            cum = pivot_ret[sym].iloc[d_idx + 1 : d_idx + 1 + h].fillna(0).sum()
            cum_f = float(cum)
            if not np.isfinite(cum_f):
                continue
            rows.append({"h": h, "s": s, "cum_ret": cum_f})
    if not rows:
        raise ValueError("no matched (news, return) pairs")
    df = pd.DataFrame(rows)

    # Grid search λ
    best_r2 = -np.inf
    best_fit = None
    for lam in np.linspace(0.01, 2.0, 50):
        decayed_s = df["s"] * np.exp(-lam * df["h"])
        X = np.column_stack([np.ones(len(df)), decayed_s.values])
        y = df["cum_ret"].values
        try:
            beta_full, *_ = np.linalg.lstsq(X, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        pred = X @ beta_full
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        if r2 > best_r2:
            best_r2 = r2
            best_fit = (float(beta_full[0]), float(beta_full[1]), float(lam))

    if best_fit is None:
        raise RuntimeError("decay fit failed")
    alpha, beta, lam = best_fit
    half_life = np.log(2) / lam if lam > 0 else float("inf")
    return DecayFit(
        alpha=alpha,
        beta=beta,
        decay_lambda=lam,
        half_life_days=half_life,
        r_squared=best_r2,
        n_obs=len(df),
    )


def expected_impact(fit: DecayFit, sentiment: float, h: int) -> float:
    """Expected return-impact at horizon h given current sentiment."""
    return fit.alpha + fit.beta * sentiment * np.exp(-fit.decay_lambda * h)


def cumulative_news_impact_signal(
    news_df: pd.DataFrame,
    fit: DecayFit,
    horizon_max: int = 21,
    sentiment_col: str = "sentiment",
    date_col: str = "date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """For each (symbol, date), compute current accumulated news-impact-signal
    based on past news within decay-window.

    Returns:
        DataFrame [date, symbol, news_impact_signal].
    """
    if news_df.empty:
        return pd.DataFrame()
    df = news_df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df = df.sort_values([symbol_col, date_col])
    out_rows = []
    for sym, g in df.groupby(symbol_col):
        dates = g[date_col].values
        sents = g[sentiment_col].values
        for t in range(len(dates)):
            d = dates[t]
            signal = 0.0
            # Sum decayed sentiment of all past news within horizon
            for past in range(t + 1):
                age_days = (d - dates[past]).astype("timedelta64[D]").astype(float)
                if age_days > horizon_max or age_days < 0:
                    continue
                signal += sents[past] * np.exp(-fit.decay_lambda * age_days)
            out_rows.append(
                {date_col: d, symbol_col: sym, "news_impact_signal": fit.beta * signal}
            )
    return pd.DataFrame(out_rows)


__all__ = [
    "DecayFit",
    "fit_news_decay_model",
    "expected_impact",
    "cumulative_news_impact_signal",
]
