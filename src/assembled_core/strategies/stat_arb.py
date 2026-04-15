"""Statistical Arbitrage — Pairs Trading Strategy (M36).

Implements a cointegration-based pairs trading strategy:
  1. Find cointegrated pairs using Engle-Granger test
  2. Compute spread z-scores for signal generation
  3. Estimate half-life for position sizing and exit timing
  4. Generate LONG/SHORT signals based on mean reversion

The strategy exploits temporary price divergences between historically
cointegrated securities, betting on spread convergence.

Reference:
    Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis."
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from statsmodels.tsa.stattools import adfuller, coint  # type: ignore[import]
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


@dataclass
class PairResult:
    """Result of cointegration test for a pair.

    Attributes:
        symbol_a: First symbol (long leg when spread is low).
        symbol_b: Second symbol (short leg when spread is low).
        coint_pvalue: Cointegration test p-value (lower = more cointegrated).
        hedge_ratio: OLS hedge ratio (beta) for spread = A - beta * B.
        half_life: Estimated mean-reversion half-life in periods.
        spread_mean: Historical spread mean.
        spread_std: Historical spread standard deviation.
        is_cointegrated: Whether the pair passes the significance threshold.
    """

    symbol_a: str
    symbol_b: str
    coint_pvalue: float
    hedge_ratio: float
    half_life: float
    spread_mean: float
    spread_std: float
    is_cointegrated: bool


@dataclass
class StatArbSignal:
    """Trading signal for a pairs trade.

    Attributes:
        symbol_a: First symbol.
        symbol_b: Second symbol.
        spread_zscore: Current z-score of the spread.
        direction_a: "LONG" or "SHORT" for symbol A.
        direction_b: "LONG" or "SHORT" for symbol B.
        signal_strength: Absolute z-score (clipped to [0, 3]).
        hedge_ratio: Shares of B per share of A.
    """

    symbol_a: str
    symbol_b: str
    spread_zscore: float
    direction_a: str
    direction_b: str
    signal_strength: float
    hedge_ratio: float


def estimate_hedge_ratio(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
) -> float:
    """Estimate OLS hedge ratio: A = alpha + beta * B + epsilon.

    Args:
        prices_a: Price series for asset A.
        prices_b: Price series for asset B.

    Returns:
        Beta (hedge ratio).
    """
    b = np.asarray(prices_b, dtype=float)
    a = np.asarray(prices_a, dtype=float)

    # OLS: beta = cov(A,B) / var(B)
    b_demeaned = b - b.mean()
    beta = np.dot(a - a.mean(), b_demeaned) / np.dot(b_demeaned, b_demeaned)
    return float(beta)


def compute_spread(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    hedge_ratio: float,
) -> np.ndarray:
    """Compute the spread: S = A - beta * B.

    Args:
        prices_a: Price series for asset A.
        prices_b: Price series for asset B.
        hedge_ratio: Hedge ratio (beta).

    Returns:
        Spread time series.
    """
    return np.asarray(prices_a, dtype=float) - hedge_ratio * np.asarray(prices_b, dtype=float)


def estimate_half_life(spread: np.ndarray) -> float:
    """Estimate mean-reversion half-life via OLS on spread changes.

    Fits: dS_t = phi * S_{t-1} + epsilon
    Half-life = -ln(2) / ln(1 + phi)

    Args:
        spread: Spread time series.

    Returns:
        Half-life in periods. Returns inf if no mean reversion detected.
    """
    s = np.asarray(spread, dtype=float)
    if len(s) < 10:
        return float("inf")

    ds = np.diff(s)
    s_lag = s[:-1]

    # Remove NaN
    mask = np.isfinite(ds) & np.isfinite(s_lag)
    ds = ds[mask]
    s_lag = s_lag[mask]

    if len(ds) < 5:
        return float("inf")

    # OLS: ds = phi * s_lag
    phi = np.dot(ds, s_lag) / np.dot(s_lag, s_lag)

    if phi >= 0:
        return float("inf")  # Not mean-reverting

    half_life = -np.log(2) / np.log(1 + phi)
    return max(float(half_life), 0.5)


def check_cointegration(
    prices_a: np.ndarray | pd.Series,
    prices_b: np.ndarray | pd.Series,
    max_pvalue: float = 0.05,
) -> PairResult:
    """Test cointegration between two price series.

    Uses the Engle-Granger two-step method via statsmodels when available,
    falls back to ADF test on the OLS spread otherwise.

    Args:
        prices_a: Price series for asset A.
        prices_b: Price series for asset B.
        max_pvalue: Maximum p-value for cointegration (default: 0.05).

    Returns:
        PairResult with test statistics and hedge ratio.
    """
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)

    hedge_ratio = estimate_hedge_ratio(a, b)
    spread = compute_spread(a, b, hedge_ratio)
    half_life = estimate_half_life(spread)

    if STATSMODELS_AVAILABLE:
        _, pvalue, _ = coint(a, b)
    else:
        # Fallback: ADF test on spread
        pvalue = _adf_pvalue(spread)

    return PairResult(
        symbol_a="A",
        symbol_b="B",
        coint_pvalue=round(float(pvalue), 4),
        hedge_ratio=round(float(hedge_ratio), 4),
        half_life=round(float(half_life), 2),
        spread_mean=round(float(np.mean(spread)), 4),
        spread_std=round(float(np.std(spread)), 4),
        is_cointegrated=pvalue < max_pvalue,
    )


def _adf_pvalue(series: np.ndarray) -> float:
    """Compute ADF p-value for stationarity test.

    Uses statsmodels if available, otherwise uses a simple heuristic
    based on the autoregression coefficient.
    """
    if STATSMODELS_AVAILABLE:
        result = adfuller(series, maxlag=1, autolag=None)
        return float(result[1])

    # Heuristic fallback: check mean reversion strength
    s = np.asarray(series, dtype=float)
    ds = np.diff(s)
    s_lag = s[:-1]
    mask = np.isfinite(ds) & np.isfinite(s_lag)
    if mask.sum() < 5:
        return 1.0

    phi = np.dot(ds[mask], s_lag[mask]) / np.dot(s_lag[mask], s_lag[mask])
    # Rough mapping: phi < -0.05 suggests stationarity
    if phi < -0.10:
        return 0.01
    elif phi < -0.05:
        return 0.05
    elif phi < -0.02:
        return 0.10
    return 0.50


def find_cointegrated_pairs(
    prices_df: pd.DataFrame,
    symbols: list[str] | None = None,
    max_pvalue: float = 0.05,
    min_observations: int = 60,
) -> list[PairResult]:
    """Screen all symbol pairs for cointegration.

    Args:
        prices_df: DataFrame with columns [timestamp, symbol, close].
        symbols: Symbols to screen. If None, uses all in prices_df.
        max_pvalue: Maximum p-value threshold.
        min_observations: Minimum overlapping observations required.

    Returns:
        List of PairResult for cointegrated pairs, sorted by p-value.
    """
    if symbols is None:
        symbols = sorted(prices_df["symbol"].unique())

    # Pivot to wide format
    pivot = prices_df.pivot_table(
        index="timestamp", columns="symbol", values="close",
    )
    pivot = pivot[symbols].dropna(axis=1, thresh=min_observations)
    available = list(pivot.columns)

    pairs = []
    n_tested = 0

    for i, sym_a in enumerate(available):
        for sym_b in available[i + 1:]:
            a = pivot[sym_a].dropna()
            b = pivot[sym_b].dropna()

            # Align
            common = a.index.intersection(b.index)
            if len(common) < min_observations:
                continue

            a_vals = a.loc[common].values
            b_vals = b.loc[common].values
            n_tested += 1

            result = check_cointegration(a_vals, b_vals, max_pvalue)
            result.symbol_a = sym_a
            result.symbol_b = sym_b

            if result.is_cointegrated:
                pairs.append(result)

    pairs.sort(key=lambda p: p.coint_pvalue)

    logger.info(
        "[StatArb] Screened %d pairs, found %d cointegrated (p<%.2f)",
        n_tested, len(pairs), max_pvalue,
    )
    return pairs


def generate_pair_signal(
    prices_a: np.ndarray | pd.Series,
    prices_b: np.ndarray | pd.Series,
    pair: PairResult,
    lookback: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> StatArbSignal | None:
    """Generate a trading signal for a cointegrated pair.

    Signal logic:
      - z < -entry_z: LONG A, SHORT B (spread will rise)
      - z > +entry_z: SHORT A, LONG B (spread will fall)
      - |z| < exit_z: No signal (spread near equilibrium)

    Args:
        prices_a: Recent price series for A.
        prices_b: Recent price series for B.
        pair: PairResult with hedge ratio and spread stats.
        lookback: Lookback window for z-score computation.
        entry_z: Z-score threshold for entry.
        exit_z: Z-score threshold for exit (no signal).

    Returns:
        StatArbSignal or None if no signal.
    """
    a = np.asarray(prices_a, dtype=float)[-lookback:]
    b = np.asarray(prices_b, dtype=float)[-lookback:]

    if len(a) < 10 or len(b) < 10:
        return None

    spread = compute_spread(a, b, pair.hedge_ratio)
    mean = spread.mean()
    std = spread.std()

    if std < 1e-10:
        return None

    current_z = (spread[-1] - mean) / std

    if abs(current_z) < exit_z:
        return None  # No signal near equilibrium

    if abs(current_z) < entry_z:
        return None  # Not strong enough

    if current_z < -entry_z:
        # Spread too low -> LONG A, SHORT B
        direction_a = "LONG"
        direction_b = "SHORT"
    else:
        # Spread too high -> SHORT A, LONG B
        direction_a = "SHORT"
        direction_b = "LONG"

    return StatArbSignal(
        symbol_a=pair.symbol_a,
        symbol_b=pair.symbol_b,
        spread_zscore=round(float(current_z), 3),
        direction_a=direction_a,
        direction_b=direction_b,
        signal_strength=round(min(abs(current_z), 3.0), 3),
        hedge_ratio=pair.hedge_ratio,
    )


__all__ = [
    "PairResult",
    "StatArbSignal",
    "estimate_hedge_ratio",
    "compute_spread",
    "estimate_half_life",
    "check_cointegration",
    "find_cointegrated_pairs",
    "generate_pair_signal",
]
