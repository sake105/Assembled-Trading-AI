"""Strategie-Templates: konfigurierbare Pre-built-Bausteine.

Templates
---------
- ``trend_following``  : Cross-section 12-1 momentum + simple MA filter
- ``low_vol_strategy`` : Long bottom-quintile rolling-vol
- ``carry_quality``    : Quality-Carry Combo (Asness/Frazzini)
- ``vol_premium``      : Long IV<RV (variance-risk-premium harvesting)
- ``regime_switching`` : Trend if regime=trending else mean-rev

Each function returns a `pd.Series` of strategy returns indexed by date.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    quantile: float = 0.2
    transaction_cost_bps: float = 5.0
    long_only: bool = True
    rebalance_every_n_days: int = 5


def _equal_weight_long_short(
    signals: pd.DataFrame,
    signal_col: str,
    return_col: str = "return",
    quantile: float = 0.2,
    long_only: bool = True,
    long_high: bool = True,
) -> pd.DataFrame:
    """Build positions from cross-sectional signal."""
    out = signals.copy().sort_values(["symbol", "date"])
    out["sig_lag"] = out.groupby("symbol")[signal_col].shift(1)
    out["pct"] = out.groupby("date")["sig_lag"].rank(pct=True)
    out["position"] = 0.0
    if long_high:
        out.loc[out["pct"] >= 1 - quantile, "position"] = +1.0
        if not long_only:
            out.loc[out["pct"] <= quantile, "position"] = -1.0
    else:
        out.loc[out["pct"] >= 1 - quantile, "position"] = -1.0 if not long_only else 0.0
        out.loc[out["pct"] <= quantile, "position"] = +1.0
    n_long = out.groupby("date")["position"].transform(lambda s: (s > 0).sum())
    n_short = out.groupby("date")["position"].transform(lambda s: (s < 0).sum())
    long_mask = out["position"] > 0
    short_mask = out["position"] < 0
    out.loc[long_mask, "position"] = 1.0 / n_long[long_mask]
    if not long_only:
        out.loc[short_mask, "position"] = -1.0 / n_short[short_mask]
    out["pnl"] = out["position"] * out[return_col]
    return out


def _aggregate_pnl(positions: pd.DataFrame, tc_bps: float = 5.0) -> pd.Series:
    daily = positions.groupby("date").agg(
        pnl=("pnl", "sum"),
        gross=("position", lambda s: s.abs().sum()),
    )
    daily["turnover"] = daily["gross"].diff().abs().fillna(0)
    daily["pnl_after_tc"] = daily["pnl"] - tc_bps / 10000 * daily["turnover"]
    return daily["pnl_after_tc"]


def trend_following(
    panel: pd.DataFrame,
    config: StrategyConfig | None = None,
    momentum_col: str = "momentum_12_1",
) -> pd.Series:
    """12-1 cross-section trend-following template."""
    cfg = config or StrategyConfig(long_only=True, quantile=0.2)
    pos = _equal_weight_long_short(
        panel, momentum_col, "return", quantile=cfg.quantile, long_only=cfg.long_only
    )
    return _aggregate_pnl(pos, cfg.transaction_cost_bps)


def low_vol_strategy(
    panel: pd.DataFrame,
    config: StrategyConfig | None = None,
    vol_col: str = "rolling_vol_60",
) -> pd.Series:
    """Long bottom-quintile rolling-volatility (low-vol anomaly)."""
    cfg = config or StrategyConfig(long_only=True, quantile=0.2)
    pos = _equal_weight_long_short(
        panel,
        vol_col,
        "return",
        quantile=cfg.quantile,
        long_only=cfg.long_only,
        long_high=False,
    )
    return _aggregate_pnl(pos, cfg.transaction_cost_bps)


def vol_premium_strategy(
    panel: pd.DataFrame,
    config: StrategyConfig | None = None,
    vrp_col: str = "vrp",
) -> pd.Series:
    """Variance-Risk-Premium harvesting: long high-VRP names."""
    cfg = config or StrategyConfig(long_only=True, quantile=0.3)
    pos = _equal_weight_long_short(
        panel, vrp_col, "return", quantile=cfg.quantile, long_only=cfg.long_only
    )
    return _aggregate_pnl(pos, cfg.transaction_cost_bps)


def regime_switching_strategy(
    panel: pd.DataFrame,
    regime: pd.Series,
    trending_signal: str = "momentum_12_1",
    range_signal: str = "residual_reversal",
    config: StrategyConfig | None = None,
) -> pd.Series:
    """Switch between trend & mean-rev based on regime.

    regime ∈ {0=low-vol-trend, 1=low-vol-range, 2=high-vol-trend, 3=crisis}.
    """
    cfg = config or StrategyConfig(long_only=True, quantile=0.2)
    out = panel.copy()
    out = out.merge(
        regime.rename("regime"), left_on="date", right_index=True, how="left"
    )
    out["composite_signal"] = np.nan
    out.loc[out["regime"].isin([0, 2]), "composite_signal"] = out.loc[
        out["regime"].isin([0, 2]), trending_signal
    ]
    out.loc[out["regime"].isin([1, 3]), "composite_signal"] = out.loc[
        out["regime"].isin([1, 3]), range_signal
    ]
    out["composite_signal"] = out["composite_signal"].fillna(0)
    pos = _equal_weight_long_short(
        out,
        "composite_signal",
        "return",
        quantile=cfg.quantile,
        long_only=cfg.long_only,
    )
    return _aggregate_pnl(pos, cfg.transaction_cost_bps)


__all__ = [
    "StrategyConfig",
    "trend_following",
    "low_vol_strategy",
    "vol_premium_strategy",
    "regime_switching_strategy",
]
