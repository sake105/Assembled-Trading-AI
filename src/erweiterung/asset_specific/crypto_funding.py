"""Crypto Perpetual-Funding-Rate Signals.

Theorie
-------
Perpetual-Futures (BTC-PERP, ETH-PERP) bezahlen Funding-Rate alle 8h, um Preis
ans Spot anzubinden. Positive Funding (long → short) bedeutet:
- Mehr Longs als Shorts → Market bullish positioniert.
- Hohe Funding > 0.01% pro 8h annualisiert > 10 % p.a. (extrem).

Signale
-------
1. **Mean-Reversion**: Extreme Funding > 0.1% pro 8h => Mean-Reversal-Short.
2. **Funding-vs-Spot-Mom**: Funding-rises während Spot-Down = bearish divergence.
3. **Open-Interest-Z**: Funding × OI = Long-Squeeze-Vulnerability.

Daten-Quellen (frei)
--------------------
- Binance Public-API: ``/fapi/v1/fundingRate``
- Bybit: ``/v5/market/funding/history``
- Deribit: alle Optionen + Perpetual

Implementation
--------------
Hier nur Signal-Logik aus gelieferten funding-Rate-Series. Fetcher
optional (würde requests benötigen).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def funding_zscore(
    funding_rate: pd.Series, lookback: int = 30 * 3  # 30 days × 3 funding intervals/day
) -> pd.Series:
    """Z-Score of funding rate vs lookback rolling-mean.

    Args:
        funding_rate: Series of funding-rates (frequency-agnostic).
        lookback: rolling window.

    Returns:
        z-Score Series.
    """
    s = pd.Series(funding_rate)
    mu = s.rolling(lookback, min_periods=lookback // 4).mean()
    sd = s.rolling(lookback, min_periods=lookback // 4).std()
    return (s - mu) / sd.replace(0, np.nan)


def annualized_funding_rate(funding_rate: pd.Series, n_per_day: int = 3) -> pd.Series:
    """Annualize per-interval funding.

    8-hour funding × 3/day × 365 days.
    """
    return funding_rate * n_per_day * 365


def perpetual_basis(perp_price: pd.Series, spot_price: pd.Series) -> pd.Series:
    """Basis = (perp − spot) / spot in bps.

    Positive Basis = perp in premium = long-positioning.
    """
    return (perp_price / spot_price - 1) * 10000  # bps


def long_squeeze_risk(
    funding_rate: pd.Series, open_interest: pd.Series, lookback: int = 30 * 3
) -> pd.Series:
    """Risk-Score for long-squeezes: high-funding × high-OI = vulnerable.

    Returns:
        Series of squeeze-risk scores (high = elevated risk).
    """
    fr_z = funding_zscore(funding_rate, lookback)
    oi_z = (
        open_interest
        - open_interest.rolling(lookback, min_periods=lookback // 4).mean()
    ) / open_interest.rolling(lookback, min_periods=lookback // 4).std().replace(
        0, np.nan
    )
    # Score: positive funding × high OI → long-squeeze potential
    return (fr_z.clip(lower=0) * oi_z.clip(lower=0)).fillna(0)


def crypto_mean_reversion_signal(
    funding_rate: pd.Series, threshold_z: float = 2.5, lookback: int = 30 * 3
) -> pd.Series:
    """Generate contrarian-signal when funding extreme.

    Funding > threshold_z * σ → short signal (-1).
    Funding < -threshold_z * σ → long signal (+1).
    """
    z = funding_zscore(funding_rate, lookback)
    sig = pd.Series(0.0, index=z.index)
    sig[z > threshold_z] = -1.0
    sig[z < -threshold_z] = 1.0
    return sig


def divergence_perp_spot_momentum(
    perp_returns: pd.Series, spot_returns: pd.Series, lookback: int = 30 * 3
) -> pd.Series:
    """Detect: perpetual moves DIFFERENT from spot — leverage-driven dislocation."""
    rolling_corr = perp_returns.rolling(lookback, min_periods=lookback // 4).corr(
        spot_returns
    )
    # When corr drops, dislocation emerges
    return 1.0 - rolling_corr


__all__ = [
    "funding_zscore",
    "annualized_funding_rate",
    "perpetual_basis",
    "long_squeeze_risk",
    "crypto_mean_reversion_signal",
    "divergence_perp_spot_momentum",
]
