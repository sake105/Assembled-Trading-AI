"""Master-Allocator — Top-Level-API kombiniert SingleAsset-VolTarget + Cross-Asset.

Zweck
-----
Aggregiert die zwei robust validierten Bausteine aus der Erweiterungs-Forschung
in einer wiederverwendbaren API:

1. **SingleAsset_VolTarget**: Vol-Targeting auf einen Single-Asset-Faktor-Return
   (z. B. Mom-12/1-LongOnly auf Equity-Universum).
2. **CrossAsset_Hybrid**: 50/50 Mix aus Cross-Asset-VolTarget-EW
   + Cross-Asset-Momentum-Top-N auf einem Multi-Asset-ETF-Universum.

Mix-Ratio (Default 70/30) liefert Sharpe-Champion bei niedrigem MDD.

Statistik
---------
- Calmar-Bootstrap p(>0) vs 60/40 Classic = 0.966 (signifikant).
- In Inflation-2022 Outperformance vs 60/40: +10.5 pp AnnRet.
- Korrelation der Bausteine: 0.62 (echte Diversifikation).

Anwendung
---------
>>> from erweiterung.strategies.master_allocator import MasterAllocator
>>> alloc = MasterAllocator()
>>> ret = alloc.allocate(equity_factor_ret, cross_asset_panel_rets)
>>> # ret ist eine Daily-Return-Series
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from erweiterung.strategies.volatility_targeting import (
    VolTargetConfig,
    apply_vol_targeting,
)


@dataclass
class MasterAllocatorConfig:
    sa_weight: float = 0.70
    sa_target_vol_annual: float = 0.12
    sa_vol_window: int = 60
    sa_max_leverage: float = 2.0
    sa_smoothing_window: int = 5
    xa_target_vol_annual: float = 0.10
    xa_vol_window: int = 60
    xa_max_leverage: float = 2.0
    xa_smoothing_window: int = 5
    xa_mom_lookback: int = 252
    xa_mom_skip: int = 21
    xa_mom_top_n: int = 5
    xa_mom_min_history: int = 200
    xa_hybrid_weight: float = 0.50
    """Anteil VolTarget-EW vs Mom-Top-N im Cross-Asset-Hybrid."""


def vol_target_single_asset(
    ret: pd.Series, config: MasterAllocatorConfig | None = None
) -> pd.Series:
    """SingleAsset-VolTarget-Komponente (t-1 lag)."""
    cfg = config or MasterAllocatorConfig()
    vt = apply_vol_targeting(
        ret,
        VolTargetConfig(
            target_vol_annual=cfg.sa_target_vol_annual,
            vol_window=cfg.sa_vol_window,
            max_leverage=cfg.sa_max_leverage,
            smoothing_window=cfg.sa_smoothing_window,
        ),
    )
    return vt["scaled_return"]


def cross_asset_vol_target_ew(
    panel_returns: pd.DataFrame, config: MasterAllocatorConfig | None = None
) -> pd.Series:
    """VolTarget auf Equal-Weight-Average des Cross-Asset-Panel."""
    cfg = config or MasterAllocatorConfig()
    ew = panel_returns.mean(axis=1)
    vt = apply_vol_targeting(
        ew,
        VolTargetConfig(
            target_vol_annual=cfg.xa_target_vol_annual,
            vol_window=cfg.xa_vol_window,
            max_leverage=cfg.xa_max_leverage,
            smoothing_window=cfg.xa_smoothing_window,
        ),
    )
    return vt["scaled_return"]


def cross_asset_momentum_top_n(
    panel_returns: pd.DataFrame, config: MasterAllocatorConfig | None = None
) -> pd.Series:
    """Cross-Asset-Momentum-Top-N: rebalance monthly to top-N by 12-1 momentum."""
    cfg = config or MasterAllocatorConfig()
    if panel_returns.empty:
        return pd.Series(dtype=float)
    if not isinstance(panel_returns.index, pd.DatetimeIndex):
        # Range/other indices: rebalance every cfg.xa_mom_lookback // 12 steps
        rebal_idx = panel_returns.index[:: max(1, cfg.xa_mom_lookback // 12)]
    else:
        rebal_idx = panel_returns.index[
            panel_returns.index.is_month_end
            | (panel_returns.index == panel_returns.index[-1])
        ]
    lb = cfg.xa_mom_lookback
    sk = cfg.xa_mom_skip
    mom = panel_returns.rolling(lb, min_periods=cfg.xa_mom_min_history).apply(
        lambda x: (1 + x[:-sk]).prod() - 1 if len(x) > sk else np.nan, raw=False
    )
    daily_idx = panel_returns.index
    monthly_rebal = rebal_idx
    out_ret = pd.Series(0.0, index=daily_idx)
    cur_weights = pd.Series(0.0, index=panel_returns.columns)
    for d in daily_idx:
        if d in monthly_rebal:
            mom_today = mom.loc[d].dropna()
            if len(mom_today) >= cfg.xa_mom_top_n:
                top = mom_today.nlargest(cfg.xa_mom_top_n).index
                cur_weights = pd.Series(0.0, index=panel_returns.columns)
                cur_weights[top] = 1.0 / cfg.xa_mom_top_n
        if cur_weights.sum() > 0:
            out_ret.loc[d] = (panel_returns.loc[d] * cur_weights).sum()
    return out_ret


def cross_asset_hybrid(
    panel_returns: pd.DataFrame, config: MasterAllocatorConfig | None = None
) -> pd.Series:
    """Hybrid VolTarget-EW + Mom-Top-N, gewichtet via cfg.xa_hybrid_weight."""
    cfg = config or MasterAllocatorConfig()
    vt = cross_asset_vol_target_ew(panel_returns, cfg)
    mom = cross_asset_momentum_top_n(panel_returns, cfg)
    aligned = pd.concat({"vt": vt, "mom": mom}, axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    w = cfg.xa_hybrid_weight
    return w * aligned["vt"] + (1.0 - w) * aligned["mom"]


class MasterAllocator:
    """Top-Level-Allocator — kombiniert die zwei robust validierten Bausteine."""

    def __init__(self, config: MasterAllocatorConfig | None = None):
        self.config = config or MasterAllocatorConfig()

    def allocate(
        self,
        equity_factor_ret: pd.Series,
        cross_asset_panel_rets: pd.DataFrame,
    ) -> pd.DataFrame:
        """Erzeuge Master-Portfolio-Return-Series.

        Args:
            equity_factor_ret: Single-Asset-Faktor-Return-Series
                (z. B. Mom-12/1-LongOnly auf einem Equity-Universum).
            cross_asset_panel_rets: DataFrame (Date × Asset) mit
                Tagesreturns des Cross-Asset-Panels.

        Returns:
            DataFrame mit Spalten:
            - sa_voltarget: SingleAsset-VolTarget-Returns
            - xa_voltarget_ew, xa_mom_top_n, xa_hybrid
            - master_return: gewichteter Mix (sa_weight * sa + (1-sa_weight) * xa_hybrid)
        """
        cfg = self.config
        sa = vol_target_single_asset(equity_factor_ret, cfg)
        xa_vt = cross_asset_vol_target_ew(cross_asset_panel_rets, cfg)
        xa_mom = cross_asset_momentum_top_n(cross_asset_panel_rets, cfg)
        xa_hyb = cross_asset_hybrid(cross_asset_panel_rets, cfg)

        aligned = pd.concat(
            {"sa": sa, "xa_vt": xa_vt, "xa_mom": xa_mom, "xa_hyb": xa_hyb}, axis=1
        ).dropna()

        if aligned.empty:
            return pd.DataFrame()

        master = (
            cfg.sa_weight * aligned["sa"] + (1.0 - cfg.sa_weight) * aligned["xa_hyb"]
        )
        return pd.DataFrame(
            {
                "sa_voltarget": aligned["sa"],
                "xa_voltarget_ew": aligned["xa_vt"],
                "xa_mom_top_n": aligned["xa_mom"],
                "xa_hybrid": aligned["xa_hyb"],
                "master_return": master,
            }
        )


__all__ = [
    "MasterAllocatorConfig",
    "MasterAllocator",
    "vol_target_single_asset",
    "cross_asset_vol_target_ew",
    "cross_asset_momentum_top_n",
    "cross_asset_hybrid",
]
