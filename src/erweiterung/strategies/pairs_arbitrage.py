"""Pairs Statistical Arbitrage — Engle-Granger + Z-Score Trading.

Algorithmus
-----------
1. **Formation**: Engle-Granger-Test auf cointegrated pairs.
2. **Hedge-Ratio**: β aus OLS p_i ~ p_j.
3. **Spread**: s_t = p_i_t − β p_j_t.
4. **Trading**:
   - Z = (s - μ_s) / σ_s (rolling).
   - Open long-spread if Z < -entry_z, short if Z > +entry_z.
   - Close if |Z| < exit_z.
   - Stop-loss if |Z| > stop_z.

Reference
---------
- Gatev, Goetzmann & Rouwenhorst (2006). Pairs Trading. *RFS* 19.
- Vidyamurthy, G. (2004). *Pairs Trading*. Wiley.

Notes
-----
Diese Implementation ist Forschungs-orientiert; production-Variante mit
Kalman-basierten Hedge-Ratios siehe ``state_space.kalman_pairs_hedge_ratio``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PairsTrade:
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp | None
    direction: int  # +1 long spread, -1 short spread
    entry_z: float
    exit_z: float | None
    pnl: float


def cointegration_engle_granger(
    y: pd.Series, x: pd.Series, p_threshold: float = 0.05
) -> dict:
    """Engle-Granger 2-step Cointegration test.

    Returns:
        dict mit ``beta`` (hedge ratio), ``adf_p`` (residual stationarity), ``cointegrated``.
    """
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 30:
        return {"error": "too few obs"}
    df.columns = ["y", "x"]
    X = np.column_stack([np.ones(len(df)), df["x"].values])
    beta_all, *_ = np.linalg.lstsq(X, df["y"].values, rcond=None)
    intercept, beta = float(beta_all[0]), float(beta_all[1])
    spread = df["y"] - beta * df["x"]
    # ADF on spread
    try:
        from statsmodels.tsa.stattools import adfuller  # type: ignore

        adf_p = float(adfuller(spread.values, regression="c", autolag="AIC")[1])
    except ImportError:
        # Crude fallback: variance ratio < 1 indicates stationarity
        d = spread.diff().dropna()
        if d.std() == 0:
            adf_p = 1.0
        else:
            vr = ((d + d.shift(1)).dropna()).var() / (2 * d.var())
            adf_p = 0.04 if vr < 0.7 else (0.1 if vr < 0.85 else 0.5)
    return {
        "beta": beta,
        "intercept": intercept,
        "spread_mean": float(spread.mean()),
        "spread_std": float(spread.std(ddof=0)),
        "adf_p": adf_p,
        "cointegrated": adf_p < p_threshold,
        "n_obs": len(df),
    }


def trade_pair(
    y: pd.Series,
    x: pd.Series,
    rolling_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> tuple[pd.DataFrame, list[PairsTrade]]:
    """Trade a pair via rolling Z-Score-Strategy.

    Args:
        y, x: aligned price-Series.
        rolling_window: window for spread mean+std.
        entry_z, exit_z, stop_z: thresholds.

    Returns:
        (positions_df, trades_list).
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    # Fit hedge ratio once on first half (training)
    half = len(df) // 2
    train = df.iloc[:half]
    res = cointegration_engle_granger(train["y"], train["x"])
    if "beta" not in res:
        return pd.DataFrame(), []
    beta = res["beta"]

    spread = df["y"] - beta * df["x"]
    mean = spread.rolling(rolling_window, min_periods=rolling_window // 2).mean()
    std = spread.rolling(rolling_window, min_periods=rolling_window // 2).std()
    z = (spread - mean) / std

    position = pd.Series(0, index=df.index, dtype=int)
    state = 0
    entry_idx = None
    trades: list[PairsTrade] = []
    for d, z_val in z.items():
        if not np.isfinite(z_val):
            continue
        if state == 0:
            if z_val < -entry_z:
                state = 1  # long spread
                entry_idx = d
            elif z_val > entry_z:
                state = -1  # short spread
                entry_idx = d
        elif state == 1:
            if z_val > -exit_z or abs(z_val) > stop_z:
                pnl = float(spread.loc[d] - spread.loc[entry_idx])
                trades.append(
                    PairsTrade(
                        entry_date=entry_idx,
                        exit_date=d,
                        direction=1,
                        entry_z=float(z.loc[entry_idx]),
                        exit_z=float(z_val),
                        pnl=pnl,
                    )
                )
                state = 0
                entry_idx = None
        elif state == -1:
            if z_val < exit_z or abs(z_val) > stop_z:
                pnl = float(-(spread.loc[d] - spread.loc[entry_idx]))
                trades.append(
                    PairsTrade(
                        entry_date=entry_idx,
                        exit_date=d,
                        direction=-1,
                        entry_z=float(z.loc[entry_idx]),
                        exit_z=float(z_val),
                        pnl=pnl,
                    )
                )
                state = 0
                entry_idx = None
        position.loc[d] = state

    out_df = pd.DataFrame(
        {"spread": spread, "z": z, "position": position},
    )
    return out_df, trades


def aggregate_pair_pnl(trades: list[PairsTrade]) -> dict:
    """Summary metrics for a pairs trading strategy."""
    if not trades:
        return {"n_trades": 0}
    pnls = np.array([t.pnl for t in trades])
    durations = []
    for t in trades:
        if t.exit_date is None:
            continue
        diff = t.exit_date - t.entry_date
        # diff is Timedelta for DatetimeIndex, int for RangeIndex
        if hasattr(diff, "days"):
            durations.append(int(diff.days))
        else:
            durations.append(int(diff))
    durations_arr = np.array(durations) if durations else np.array([])
    return {
        "n_trades": len(trades),
        "total_pnl": float(pnls.sum()),
        "mean_pnl": float(pnls.mean()),
        "win_rate": float((pnls > 0).mean()),
        "mean_duration": float(durations_arr.mean()) if len(durations_arr) > 0 else 0,
    }


__all__ = [
    "PairsTrade",
    "cointegration_engle_granger",
    "trade_pair",
    "aggregate_pair_pnl",
]
