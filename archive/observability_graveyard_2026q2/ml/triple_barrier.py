"""Triple-Barrier Labeling (Lopez de Prado, AIFML Chapter 3).

Dynamisch skalierte Labels statt Fixed-Horizon fwd_return:
- Upper Barrier: Profit-Take (+k × σ_daily)
- Lower Barrier: Stop-Loss (-k × σ_daily)
- Vertical Barrier: Zeit-Limit (T Tage)

Label:
    +1 wenn Upper zuerst getroffen (profitable Richtung)
    -1 wenn Lower zuerst getroffen (loss)
     0 wenn Vertical zuerst (time-out, kein signifikanter Move)

Warum besser als fwd_return_Nd:
- respektiert Volatilität (hoch-vol Aktien brauchen größere Bewegung)
- unterscheidet zwischen "großer Move in 3d" und "schleichender Move in 20d"
- Meta-Label-Trainingsgrundlage für Lopez de Prado Workflow

PIT-Invariante:
- Labels werden mit forward-looking prices gebildet → nur für Training verwenden
- Letzte `horizon_days` Zeilen pro Symbol können keinen finalen Label erhalten → NaN
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_daily_volatility(
    prices: pd.Series,
    lookback: int = 20,
    min_periods: int = 5,
) -> pd.Series:
    """Daily EWMA-Volatilität für Barrier-Skalierung.

    EWMA statt rolling std: reagiert schneller auf Regime-Wechsel.
    """
    returns = prices.pct_change()
    vol = returns.ewm(span=lookback, min_periods=min_periods).std()
    return vol


def apply_triple_barrier(
    prices: pd.Series,
    volatility: pd.Series,
    horizon_days: int = 5,
    upper_mult: float = 2.0,
    lower_mult: float = 2.0,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """Triple-Barrier auf Einzel-Asset-Preisserie anwenden.

    Args:
        prices: Preisserie (sortiert nach Timestamp)
        volatility: daily σ (gleiche Länge wie prices)
        horizon_days: Vertical-Barrier in Handelstagen (Default: 5)
        upper_mult: Profit-Take als Multiplikator der σ (Default: 2.0)
        lower_mult: Stop-Loss als Multiplikator der σ (Default: 2.0)
        min_ret: Minimum-Return um Label als relevant zu klassifizieren

    Returns:
        DataFrame mit Spalten:
            t1: Index des Barrier-Treffers (Vertical-Limit wenn keine Barriere)
            label: -1 / 0 / +1
            ret: realisierter Return zwischen t0 und t1
            barrier_type: "UPPER" / "LOWER" / "VERTICAL"

    PIT-Invariante: Letzte horizon_days Zeilen haben NaN in t1/label (kein vollständiger Forward-Window).
    """
    n = len(prices)
    labels = np.full(n, np.nan)
    t1_values = np.full(n, -1, dtype=np.int64)
    rets = np.full(n, np.nan)
    barrier_types = np.full(n, "", dtype=object)

    prices_vals = prices.values
    vol_vals = volatility.values

    for i in range(n - horizon_days):
        p0 = prices_vals[i]
        sigma = vol_vals[i]
        if np.isnan(sigma) or sigma <= 0:
            continue

        up_thresh = p0 * (1.0 + upper_mult * sigma)
        dn_thresh = p0 * (1.0 - lower_mult * sigma)

        # Scan horizon
        hit_up = -1
        hit_dn = -1
        for j in range(i + 1, min(i + horizon_days + 1, n)):
            p = prices_vals[j]
            if hit_up < 0 and p >= up_thresh:
                hit_up = j
            if hit_dn < 0 and p <= dn_thresh:
                hit_dn = j
            if hit_up >= 0 and hit_dn >= 0:
                break

        # Resolve: whichever came first
        if hit_up >= 0 and (hit_dn < 0 or hit_up < hit_dn):
            t_hit = hit_up
            bt = "UPPER"
            label = 1
        elif hit_dn >= 0 and (hit_up < 0 or hit_dn < hit_up):
            t_hit = hit_dn
            bt = "LOWER"
            label = -1
        else:
            t_hit = min(i + horizon_days, n - 1)
            bt = "VERTICAL"
            realized_ret = (prices_vals[t_hit] - p0) / p0
            if abs(realized_ret) < min_ret:
                label = 0
            else:
                label = int(np.sign(realized_ret))

        t1_values[i] = t_hit
        labels[i] = label
        rets[i] = (prices_vals[t_hit] - p0) / p0
        barrier_types[i] = bt

    return pd.DataFrame({
        "t1": t1_values,
        "label": labels,
        "ret": rets,
        "barrier_type": barrier_types,
    }, index=prices.index)


def build_triple_barrier_labels(
    panel_df: pd.DataFrame,
    price_col: str = "close",
    symbol_col: str = "symbol",
    timestamp_col: str = "timestamp",
    horizon_days: int = 5,
    upper_mult: float = 2.0,
    lower_mult: float = 2.0,
    vol_lookback: int = 20,
) -> pd.DataFrame:
    """Triple-Barrier-Labels für komplettes Panel (mehrere Symbole).

    Fügt Spalten hinzu:
        tb_label_{h}d: -1/0/+1
        tb_ret_{h}d: realisierter Return bis Barrier
        tb_barrier_{h}d: UPPER/LOWER/VERTICAL

    Returns:
        panel_df mit zusätzlichen Spalten.
    """
    result = panel_df.sort_values([symbol_col, timestamp_col]).copy()
    suffix = f"{horizon_days}d"

    all_labels: list[pd.DataFrame] = []
    for sym, grp in result.groupby(symbol_col, sort=False):
        if len(grp) < vol_lookback + horizon_days:
            continue
        prices = grp[price_col]
        vol = compute_daily_volatility(prices, lookback=vol_lookback)
        tb = apply_triple_barrier(
            prices=prices,
            volatility=vol,
            horizon_days=horizon_days,
            upper_mult=upper_mult,
            lower_mult=lower_mult,
        )
        tb["_symbol"] = sym
        tb["_orig_index"] = grp.index
        all_labels.append(tb)

    if not all_labels:
        return result

    combined = pd.concat(all_labels)
    combined = combined.set_index("_orig_index")

    result[f"tb_label_{suffix}"] = combined["label"]
    result[f"tb_ret_{suffix}"] = combined["ret"]
    result[f"tb_barrier_{suffix}"] = combined["barrier_type"]

    # PIT-Sanity: last horizon_days rows per symbol should be NaN
    logger.info(
        "[TripleBarrier] h=%dd, %d Labels gesetzt, %d Zeilen NaN (PIT)",
        horizon_days,
        int(result[f"tb_label_{suffix}"].notna().sum()),
        int(result[f"tb_label_{suffix}"].isna().sum()),
    )
    return result


__all__ = [
    "compute_daily_volatility",
    "apply_triple_barrier",
    "build_triple_barrier_labels",
]
