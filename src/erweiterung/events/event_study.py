"""Event-Study Framework (MacKinlay 1997).

Methodik
--------
1. **Estimation Window**: T_pre Tage vor dem Event (typ. -250 bis -30).
   Schätze Marktmodell: r_{i,t} = α_i + β_i·r_{m,t} + ε_{i,t}.
2. **Event Window**: ± Tage um Event (z. B. -5 bis +20).
3. **Abnormal Return** AR_{i,t} = r_{i,t} − (α̂_i + β̂_i·r_{m,t}).
4. **Cumulative AR** CAR_i(T1, T2) = Σ_{t=T1}^{T2} AR_{i,t}.
5. **Average AAR**: mean across events.
6. **Test-Statistic**: t = AAR_t / (σ_AAR / √N) (Patell 1976 standardisiert).

Erweiterung BHAR
----------------
Buy-and-Hold-Abnormal-Return: ∏(1+r_{i,t}) − ∏(1+r_{m,t}). Genauer für lange
Horizonte als CAR.

Anwendung
---------
- Earnings-Drift-Validation
- Insider-Trading-Cluster-Reaction
- 8-K-Material-Event-Reaction
- M&A-Announcement-Returns
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class EventStudyResult:
    aar: pd.Series  # Average Abnormal Returns per event-relative day
    caar: pd.Series  # Cumulative AAR
    n_events: int
    t_stats: pd.Series  # t-stat per relative day


def market_model_alpha_beta(
    asset_returns: pd.Series, market_returns: pd.Series
) -> tuple[float, float, float]:
    """OLS: r_a = α + β·r_m + ε. Return (α, β, σ_ε)."""
    df = pd.concat([asset_returns, market_returns], axis=1).dropna()
    df.columns = ["a", "m"]
    if len(df) < 20:
        return (0.0, 1.0, float("nan"))
    X = np.column_stack([np.ones(len(df)), df["m"].values])
    y = df["a"].values
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    alpha, slope = float(beta[0]), float(beta[1])
    eps = y - X @ beta
    sigma = float(np.std(eps, ddof=2))
    return (alpha, slope, sigma)


def run_event_study(
    events: pd.DataFrame,
    asset_panel: pd.DataFrame,
    market_returns: pd.Series,
    estimation_window: tuple[int, int] = (-250, -30),
    event_window: tuple[int, int] = (-5, 20),
) -> EventStudyResult:
    """Compute event-study aggregating across all events.

    Args:
        events: DataFrame with [symbol, event_date].
        asset_panel: DataFrame [date, symbol, return].
        market_returns: market return series indexed by date.
        estimation_window: (start_offset, end_offset) in trading days vs event.
        event_window: (start_offset, end_offset) in trading days vs event.

    Returns:
        EventStudyResult with AAR, CAAR, t-stats.
    """
    if events.empty:
        return EventStudyResult(
            aar=pd.Series(dtype=float),
            caar=pd.Series(dtype=float),
            n_events=0,
            t_stats=pd.Series(dtype=float),
        )

    # build per-symbol date-indexed returns
    pivot = asset_panel.pivot_table(index="date", columns="symbol", values="return")
    market = pd.Series(market_returns)

    # collect AR-matrix: rows = events, cols = relative days
    rel_days = list(range(event_window[0], event_window[1] + 1))
    ar_matrix = np.full((len(events), len(rel_days)), np.nan)

    for ev_idx, row in enumerate(events.itertuples(index=False)):
        sym = getattr(row, "symbol", None)
        d = getattr(row, "event_date", None)
        if sym is None or pd.isna(d) or sym not in pivot.columns:
            continue
        d = pd.Timestamp(d)
        # find positional indices
        all_dates = pivot.index.sort_values()
        if d not in all_dates:
            # snap to next available
            future = all_dates[all_dates >= d]
            if future.empty:
                continue
            d = future[0]
        pos = all_dates.get_loc(d)
        # estimation window
        est_start = pos + estimation_window[0]
        est_end = pos + estimation_window[1]
        if est_start < 0 or est_end >= len(all_dates):
            continue
        est_dates = all_dates[est_start : est_end + 1]
        a_returns = pivot.loc[est_dates, sym]
        m_returns = market.reindex(est_dates)
        alpha, beta, _sigma = market_model_alpha_beta(a_returns, m_returns)

        # event window
        for j, rd in enumerate(rel_days):
            ev_pos = pos + rd
            if ev_pos < 0 or ev_pos >= len(all_dates):
                continue
            ev_date = all_dates[ev_pos]
            r_a = pivot.at[ev_date, sym] if ev_date in pivot.index else np.nan
            r_m = market.get(ev_date, np.nan)
            if pd.notna(r_a) and pd.notna(r_m):
                ar = r_a - (alpha + beta * r_m)
                ar_matrix[ev_idx, j] = ar

    # AAR + CAAR + t-stat
    aar = pd.Series(np.nanmean(ar_matrix, axis=0), index=rel_days, name="aar")
    aar_std = pd.Series(np.nanstd(ar_matrix, axis=0, ddof=1), index=rel_days)
    n_obs = pd.Series((~np.isnan(ar_matrix)).sum(axis=0), index=rel_days)
    se = aar_std / n_obs.pow(0.5).replace(0, np.nan)
    t_stats = aar / se
    caar = aar.cumsum()

    return EventStudyResult(
        aar=aar, caar=caar, n_events=int(n_obs.max()), t_stats=t_stats
    )


def buy_and_hold_abnormal_return(
    asset_returns: pd.Series, market_returns: pd.Series
) -> float:
    """BHAR = ∏(1+r_a) − ∏(1+r_m). Long-horizon."""
    a = pd.Series(asset_returns).dropna()
    m = pd.Series(market_returns).reindex(a.index).fillna(0)
    if a.empty:
        return float("nan")
    return float(((1 + a).prod() - 1) - ((1 + m).prod() - 1))


__all__ = [
    "EventStudyResult",
    "market_model_alpha_beta",
    "run_event_study",
    "buy_and_hold_abnormal_return",
]
