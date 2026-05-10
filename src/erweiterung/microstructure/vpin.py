"""VPIN — Volume-Synchronized Probability of Informed Trading.

Easley, López de Prado, O'Hara (2012, *Review of Financial Studies*).

Idee
----
Order-Flow-Toxicity treibt Liquidity-Stress.  VPIN misst Imbalance zwischen
Buy- und Sell-Volume in **gleichgroßen Volume-Buckets** (statt Time-Buckets):

    VPIN = mean over n buckets of |V_buy − V_sell| / V_total

Hohe VPIN-Werte sind Vorlaufindikator für Volatilitäts-Spikes (Flash-Crash 2010).

Bulk-Volume-Classification (BVC)
--------------------------------
Da Tick-Direction oft nicht verfügbar ist, klassifiziert man Volume per Bucket
basierend auf Preis-Veränderung & Vola:
    V_buy = V_total × Φ((Δp / σ_p))
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bulk_volume_classify(
    prices: pd.Series, volumes: pd.Series, std_window: int = 50
) -> tuple[pd.Series, pd.Series]:
    """Bulk-Volume-Classification: schätze V_buy und V_sell."""
    p = pd.Series(prices)
    v = pd.Series(volumes)
    dp = p.diff()
    sigma = dp.rolling(std_window, min_periods=std_window // 2).std()
    z = (dp / sigma).fillna(0)
    # Φ(z) approximation (normal CDF)
    cdf = 0.5 * (1 + np.tanh(z * np.sqrt(2 / np.pi)))  # tanh approx of erf
    v_buy = v * cdf
    v_sell = v * (1 - cdf)
    return v_buy.fillna(0), v_sell.fillna(0)


def compute_vpin(
    prices: pd.Series,
    volumes: pd.Series,
    bucket_size: float | None = None,
    n_buckets: int = 50,
    std_window: int = 50,
) -> pd.Series:
    """Compute VPIN-Series.

    Args:
        prices, volumes: indexed by time.
        bucket_size: V per bucket. Default = mean(V) × 5.
        n_buckets: number of buckets in window for VPIN computation.
        std_window: window for return-std estimation.

    Returns:
        Series ``vpin`` indexed by bucket-end-time.
    """
    p = pd.Series(prices).dropna()
    v = pd.Series(volumes).reindex(p.index).fillna(0)
    if bucket_size is None:
        bucket_size = float(v.mean() * 5.0)
    if bucket_size <= 0:
        return pd.Series(dtype=float)

    v_buy, v_sell = bulk_volume_classify(p, v, std_window=std_window)

    # Build buckets by accumulating volume until bucket_size reached
    cum_v = 0.0
    cum_buy = 0.0
    cum_sell = 0.0
    bucket_imbalances: list[tuple[pd.Timestamp, float]] = []
    for t in p.index:
        cum_v += v.loc[t]
        cum_buy += v_buy.loc[t]
        cum_sell += v_sell.loc[t]
        if cum_v >= bucket_size:
            imb = abs(cum_buy - cum_sell) / cum_v if cum_v > 0 else 0
            bucket_imbalances.append((t, imb))
            cum_v = 0.0
            cum_buy = 0.0
            cum_sell = 0.0

    if not bucket_imbalances:
        return pd.Series(dtype=float)

    df = pd.DataFrame(bucket_imbalances, columns=["time", "imb"]).set_index("time")
    df["vpin"] = df["imb"].rolling(n_buckets, min_periods=max(5, n_buckets // 4)).mean()
    return df["vpin"].dropna()


__all__ = ["bulk_volume_classify", "compute_vpin"]
