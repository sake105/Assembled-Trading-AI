"""Volume-Synchronized Probability of Informed Trading (M26 Task 26.4).

VPIN estimates the probability that trading is driven by informed traders.
High VPIN → high toxicity → adverse selection risk → widen spreads / reduce size.

Reference:
    Easley, Lopez de Prado & O'Hara (2012) "Flow Toxicity and Liquidity"
    Abad & Yague (2012) VPIN validation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VPINResult:
    """VPIN computation result."""
    vpin_series: pd.Series       # VPIN values over time
    avg_vpin: float              # Average VPIN
    max_vpin: float              # Peak VPIN
    current_vpin: float          # Most recent VPIN
    n_buckets: int               # Number of volume buckets used
    alert_threshold: float       # VPIN threshold for toxicity alert
    is_toxic: bool               # Whether current VPIN exceeds threshold


def classify_volume_bulk(
    prices: pd.Series,
    volumes: pd.Series,
    method: str = "tick",
) -> tuple[pd.Series, pd.Series]:
    """Classify trade volume as buy or sell initiated.

    Args:
        prices: Trade prices or close prices.
        volumes: Trade volumes.
        method: "tick" (tick rule) or "bvc" (Bulk Volume Classification).

    Returns:
        (buy_volume, sell_volume) series.
    """
    if method == "bvc":
        # Bulk Volume Classification (Easley et al. 2012)
        # Uses price changes normalized by volatility
        returns = prices.pct_change().fillna(0)
        sigma = returns.rolling(20).std().fillna(returns.std())
        sigma = sigma.replace(0, 1e-8)
        z = returns / sigma

        # CDF approximation
        buy_pct = _normal_cdf(z)
        sell_pct = 1 - buy_pct
    else:
        # Tick rule: price up → buy, price down → sell
        price_change = prices.diff().fillna(0)
        buy_pct = pd.Series(0.5, index=prices.index)
        buy_pct[price_change > 0] = 1.0
        buy_pct[price_change < 0] = 0.0
        sell_pct = 1 - buy_pct

    buy_volume = volumes * buy_pct
    sell_volume = volumes * sell_pct

    return buy_volume, sell_volume


def _normal_cdf(z: pd.Series) -> pd.Series:
    """Approximate standard normal CDF."""
    try:
        from scipy.stats import norm
        return pd.Series(norm.cdf(z.values), index=z.index)
    except ImportError:
        # Rational approximation of normal CDF
        x = z.values
        t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
        d = 0.3989422804 * np.exp(-x * x / 2.0)
        p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
        result = np.where(x > 0, 1.0 - p, p)
        return pd.Series(result, index=z.index)


def compute_vpin(
    prices: pd.Series,
    volumes: pd.Series,
    bucket_size: float | None = None,
    n_buckets_window: int = 50,
    method: str = "bvc",
) -> VPINResult:
    """Compute VPIN (Volume-Synchronized Probability of Informed Trading).

    Algorithm:
    1. Classify each bar's volume as buy/sell initiated
    2. Aggregate into equal-volume buckets
    3. Compute |V_buy - V_sell| / V_bucket for each bucket
    4. VPIN = rolling average of bucket order imbalances

    Args:
        prices: Price series (close prices or trade prices).
        volumes: Volume series.
        bucket_size: Volume per bucket. Defaults to avg_daily_volume / 50.
        n_buckets_window: Number of buckets for rolling VPIN average.
        method: Volume classification method ("bvc" or "tick").

    Returns:
        VPINResult with VPIN time series and metrics.
    """
    if len(prices) < 20:
        return VPINResult(
            vpin_series=pd.Series(dtype=float),
            avg_vpin=0.0, max_vpin=0.0, current_vpin=0.0,
            n_buckets=0, alert_threshold=0.7, is_toxic=False,
        )

    # Classify volume
    buy_vol, sell_vol = classify_volume_bulk(prices, volumes, method)

    # Determine bucket size
    if bucket_size is None:
        avg_daily_vol = volumes.mean()
        bucket_size = max(avg_daily_vol / 50, 1)

    # Build volume buckets
    buckets = []
    current_buy = 0.0
    current_sell = 0.0
    current_vol = 0.0
    bucket_dates = []

    for i in range(len(prices)):
        current_buy += float(buy_vol.iloc[i])
        current_sell += float(sell_vol.iloc[i])
        current_vol += float(volumes.iloc[i])

        while current_vol >= bucket_size:
            # Fill this bucket
            overflow = current_vol - bucket_size
            frac = bucket_size / max(current_vol, 1e-8)

            b_buy = current_buy * frac
            b_sell = current_sell * frac

            # Order imbalance for this bucket
            imbalance = abs(b_buy - b_sell) / bucket_size
            buckets.append(imbalance)
            bucket_dates.append(prices.index[i])

            # Carry over overflow
            current_buy -= b_buy
            current_sell -= b_sell
            current_vol = overflow

    if len(buckets) < n_buckets_window:
        n_buckets_window = max(len(buckets), 1)

    # Compute VPIN as rolling average of bucket imbalances
    bucket_series = pd.Series(buckets, index=bucket_dates[:len(buckets)])
    vpin = bucket_series.rolling(n_buckets_window, min_periods=1).mean()

    # Map back to daily frequency
    vpin_daily = vpin.groupby(vpin.index).last()

    avg_vpin = float(vpin.mean()) if len(vpin) > 0 else 0.0
    max_vpin = float(vpin.max()) if len(vpin) > 0 else 0.0
    current = float(vpin.iloc[-1]) if len(vpin) > 0 else 0.0

    # Alert threshold: 90th percentile or 0.7, whichever is lower
    threshold = min(float(vpin.quantile(0.9)) if len(vpin) > 10 else 0.7, 0.9)

    logger.info("[VPIN] Computed over %d buckets: avg=%.3f, current=%.3f, threshold=%.3f",
                len(buckets), avg_vpin, current, threshold)

    return VPINResult(
        vpin_series=vpin_daily,
        avg_vpin=round(avg_vpin, 4),
        max_vpin=round(max_vpin, 4),
        current_vpin=round(current, 4),
        n_buckets=len(buckets),
        alert_threshold=round(threshold, 4),
        is_toxic=current > threshold,
    )


def compute_vpin_panel(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    n_buckets_window: int = 50,
) -> pd.DataFrame:
    """Compute VPIN for multiple stocks.

    Args:
        prices: (T, N) price DataFrame.
        volumes: (T, N) volume DataFrame.
        n_buckets_window: Buckets for rolling average.

    Returns:
        DataFrame with VPIN per stock.
    """
    vpin_dict = {}
    for col in prices.columns:
        if col not in volumes.columns:
            continue
        result = compute_vpin(prices[col], volumes[col], n_buckets_window=n_buckets_window)
        if not result.vpin_series.empty:
            vpin_dict[col] = result.vpin_series

    if not vpin_dict:
        return pd.DataFrame()

    return pd.DataFrame(vpin_dict).reindex(prices.index).ffill().fillna(0.0)


__all__ = [
    "VPINResult",
    "classify_volume_bulk",
    "compute_vpin",
    "compute_vpin_panel",
]
