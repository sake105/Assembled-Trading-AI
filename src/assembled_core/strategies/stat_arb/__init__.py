"""Statistical Arbitrage & Pairs Trading strategy modules.

Re-exports the primary API expected by existing tests and callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .cointegration import (
    PairCandidate,
    test_cointegration,
    screen_pairs,
    _ols_hedge_ratio,
    _adf_test_pvalue,
    _estimate_half_life as _raw_half_life,
)
from .pair_signals import PairSignalGenerator, PairSignal, PairPosition
from .pca_arb import compute_pca_factors, generate_pca_signals, PCAFactorModel, PCASignal


# ---------------------------------------------------------------------------
# Legacy API expected by test_stat_arb.py
# ---------------------------------------------------------------------------

@dataclass
class PairResult:
    """Result of cointegration test (legacy API)."""
    symbol_a: str
    symbol_b: str
    coint_pvalue: float
    hedge_ratio: float
    half_life: float
    spread_mean: float
    spread_std: float
    is_cointegrated: bool


def estimate_hedge_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """Estimate OLS hedge ratio: a = beta * b + eps."""
    return _ols_hedge_ratio(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def compute_spread(a: np.ndarray, b: np.ndarray, hedge_ratio: float) -> np.ndarray:
    """Compute spread = a - hedge_ratio * b."""
    return np.asarray(a, dtype=float) - hedge_ratio * np.asarray(b, dtype=float)


def estimate_half_life(spread: np.ndarray) -> float:
    """Estimate half-life of mean-reverting spread."""
    return _raw_half_life(np.asarray(spread, dtype=float))


def check_cointegration(
    a: np.ndarray,
    b: np.ndarray,
    symbol_a: str = "A",
    symbol_b: str = "B",
) -> PairResult:
    """Test cointegration and return PairResult."""
    a_arr = np.asarray(a, dtype=float)
    b_arr = np.asarray(b, dtype=float)

    beta = _ols_hedge_ratio(a_arr, b_arr)
    spread = a_arr - beta * b_arr
    p_val = _adf_test_pvalue(spread)
    hl = _raw_half_life(spread)

    return PairResult(
        symbol_a=symbol_a,
        symbol_b=symbol_b,
        coint_pvalue=round(p_val, 6),
        hedge_ratio=round(beta, 6),
        half_life=round(hl, 2),
        spread_mean=round(float(spread.mean()), 6),
        spread_std=round(float(spread.std()), 6),
        is_cointegrated=(p_val < 0.05),
    )


def find_cointegrated_pairs(
    df: pd.DataFrame,
    max_pvalue: float = 0.05,
    price_col: str = "close",
    symbol_col: str = "symbol",
    time_col: str = "timestamp",
) -> list[PairResult]:
    """Find cointegrated pairs from long-format DataFrame.

    Args:
        df: DataFrame with columns [timestamp, symbol, close].
        max_pvalue: Maximum p-value for cointegration.
        price_col: Column name for prices.
        symbol_col: Column name for symbols.
        time_col: Column name for time.

    Returns:
        List of PairResult for significant pairs.
    """
    symbols = sorted(df[symbol_col].unique())
    results = []

    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            s_a, s_b = symbols[i], symbols[j]
            a_data = df.loc[df[symbol_col] == s_a, price_col].values
            b_data = df.loc[df[symbol_col] == s_b, price_col].values

            min_len = min(len(a_data), len(b_data))
            if min_len < 60:
                continue

            pair = check_cointegration(
                a_data[:min_len], b_data[:min_len],
                symbol_a=s_a, symbol_b=s_b,
            )
            if pair.coint_pvalue <= max_pvalue:
                results.append(pair)

    results.sort(key=lambda p: p.coint_pvalue)
    return results


@dataclass
class PairTradeSignal:
    """Signal for a pair trade (legacy API)."""
    symbol_a: str
    symbol_b: str
    direction_a: str  # "LONG" or "SHORT"
    direction_b: str
    z_score: float
    signal_strength: float


def generate_pair_signal(
    prices_a: np.ndarray,
    prices_b: np.ndarray,
    pair: PairResult,
    entry_z: float = 2.0,
    lookback: int = 60,
) -> PairTradeSignal | None:
    """Generate pair trading signal from current prices.

    Args:
        prices_a: Price array for stock A.
        prices_b: Price array for stock B.
        pair: PairResult from cointegration test.
        entry_z: Z-score threshold for entry.
        lookback: Lookback window.

    Returns:
        PairTradeSignal or None if no signal.
    """
    a = np.asarray(prices_a, dtype=float)
    b = np.asarray(prices_b, dtype=float)

    if len(a) < lookback or len(b) < lookback:
        return None

    spread = a - pair.hedge_ratio * b
    recent = spread[-lookback:]
    mean = float(recent.mean())
    std = float(recent.std())
    if std < 1e-10:
        return None

    z = (spread[-1] - mean) / std

    if abs(z) < entry_z:
        return None

    strength = min(abs(z) / entry_z, 3.0)

    if z > entry_z:
        # Spread too high: short A, long B
        return PairTradeSignal(
            symbol_a=pair.symbol_a, symbol_b=pair.symbol_b,
            direction_a="SHORT", direction_b="LONG",
            z_score=round(z, 4), signal_strength=round(strength, 4),
        )
    else:
        # Spread too low: long A, short B
        return PairTradeSignal(
            symbol_a=pair.symbol_a, symbol_b=pair.symbol_b,
            direction_a="LONG", direction_b="SHORT",
            z_score=round(z, 4), signal_strength=round(strength, 4),
        )


__all__ = [
    # Legacy API
    "PairResult",
    "PairTradeSignal",
    "estimate_hedge_ratio",
    "compute_spread",
    "estimate_half_life",
    "check_cointegration",
    "find_cointegrated_pairs",
    "generate_pair_signal",
    # New modules
    "PairCandidate",
    "test_cointegration",
    "screen_pairs",
    "PairSignalGenerator",
    "PairSignal",
    "PairPosition",
    "compute_pca_factors",
    "generate_pca_signals",
    "PCAFactorModel",
    "PCASignal",
]
