"""Historical-Event-Replay: GFC, COVID, Energy-Crisis, Volmageddon.

Idee
----
Apply the strategy's signal-logic to a historical crisis window and report
how the strategy would have performed.

Standard-Events
---------------
- **GFC (2008-09-15 → 2009-03-09)**: Lehman bankruptcy + March 2009 trough.
  Spy: -55 %, VIX peak 89.
- **Flash Crash (2010-05-06)**: 1-day, -9 % intraday recovered.
- **EU Crisis (2011-08 → 2011-10)**: SPY -19 %.
- **China-Devaluation (2015-08)**: SPY -12 %.
- **Volmageddon (2018-02-05)**: VIX +116 %, SPY -4 %.
- **COVID (2020-02-19 → 2020-03-23)**: SPY -34 % in 23 days.
- **Inflation/Energy (2022-01-04 → 2022-10-12)**: SPY -25 %.
- **SVB-Banking (2023-03-08 → 2023-03-15)**: brief banking stress.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CrisisWindow:
    name: str
    start: str
    end: str
    description: str


STANDARD_CRISES = [
    CrisisWindow("GFC_2008", "2008-09-15", "2009-03-09", "Lehman → March 2009 trough"),
    CrisisWindow("FlashCrash_2010", "2010-05-06", "2010-05-07", "1-day flash crash"),
    CrisisWindow(
        "EU_Crisis_2011", "2011-08-01", "2011-10-04", "European debt + US debt ceiling"
    ),
    CrisisWindow("China_2015", "2015-08-17", "2015-09-29", "Yuan devaluation"),
    CrisisWindow("Brexit_2016", "2016-06-23", "2016-07-08", "UK leave vote"),
    CrisisWindow(
        "Volmageddon_2018", "2018-02-02", "2018-02-09", "VIX spike, XIV implosion"
    ),
    CrisisWindow("Q4_2018_selloff", "2018-10-03", "2018-12-24", "Powell pivot needed"),
    CrisisWindow("COVID_2020", "2020-02-19", "2020-03-23", "Pandemic crash"),
    CrisisWindow("Inflation_2022", "2022-01-04", "2022-10-12", "Inflation/rate hikes"),
    CrisisWindow("SVB_2023", "2023-03-08", "2023-03-31", "Banking stress"),
]


def replay_window(strategy_returns: pd.Series, crisis: CrisisWindow) -> dict:
    """Slice strategy returns to crisis window and compute summary stats."""
    s = pd.Series(strategy_returns).copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index)
    if s.index.tz is None:
        s.index = s.index.tz_localize("UTC")
    start = pd.Timestamp(crisis.start, tz="UTC")
    end = pd.Timestamp(crisis.end, tz="UTC")
    sub = s.loc[start:end]
    if sub.empty:
        return {"crisis": crisis.name, "n_obs": 0, "error": "no data in window"}

    eq = (1 + sub).cumprod()
    cum_ret = float(eq.iloc[-1] - 1)
    cummax = eq.cummax()
    dd = eq / cummax - 1
    max_dd = float(dd.min())
    daily_std = float(sub.std(ddof=0))
    return {
        "crisis": crisis.name,
        "start": crisis.start,
        "end": crisis.end,
        "n_obs": int(len(sub)),
        "cumulative_return": cum_ret,
        "max_drawdown": max_dd,
        "worst_day": float(sub.min()),
        "best_day": float(sub.max()),
        "volatility_daily": daily_std,
    }


def replay_all_crises(
    strategy_returns: pd.Series, crises: list[CrisisWindow] | None = None
) -> pd.DataFrame:
    """Replay strategy across ALL standard crisis windows."""
    crises = crises or STANDARD_CRISES
    rows = [replay_window(strategy_returns, c) for c in crises]
    return pd.DataFrame(rows)


def stress_score(replay_df: pd.DataFrame) -> dict:
    """Aggregate stress metrics across all crises."""
    df = replay_df.dropna(subset=["max_drawdown"])
    if df.empty:
        return {"error": "no replay results"}
    return {
        "worst_drawdown": float(df["max_drawdown"].min()),
        "mean_drawdown": float(df["max_drawdown"].mean()),
        "worst_crisis": df.loc[df["max_drawdown"].idxmin(), "crisis"],
        "n_crises_with_negative_return": int((df["cumulative_return"] < 0).sum()),
        "n_crises_evaluated": len(df),
    }


__all__ = [
    "CrisisWindow",
    "STANDARD_CRISES",
    "replay_window",
    "replay_all_crises",
    "stress_score",
]
