"""Macro-Regime 4-Quadrant Feature.

From 13_FREE_MODULE.md §13.8.
Dalio/Bridgewater-inspired macro regime classification.
Categorical feature for ML models.

Quadrants (Growth × Inflation):
  'growth_up_infl_up'   → Commodities, Emerging, Value
  'growth_up_infl_down' → Growth, Large-Cap-Tech
  'growth_down_infl_up' → Gold, Defensive-Value
  'growth_down_infl_down' → Treasuries, Cash

Data: FRED via fredapi (free).
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_QUADRANT_LABELS = {
    (True, True): "growth_up_infl_up",
    (True, False): "growth_up_infl_down",
    (False, True): "growth_down_infl_up",
    (False, False): "growth_down_infl_down",
}

_QUADRANT_ALLOCATIONS = {
    "growth_up_infl_up": ["commodities", "emerging", "value"],
    "growth_up_infl_down": ["growth", "large_cap_tech"],
    "growth_down_infl_up": ["gold", "defensive_value"],
    "growth_down_infl_down": ["treasuries", "cash"],
}


def _zscore_series(s: pd.Series, window: int = 252) -> pd.Series:
    roll = s.rolling(window=window, min_periods=max(window // 4, 20))
    return (s - roll.mean()) / roll.std().replace(0, float("nan"))


def compute_macro_quadrant(
    ism_pmi: pd.Series,
    nfp_3m_change: pd.Series,
    cpi_yoy: pd.Series,
    breakeven_5y5y: pd.Series,
    lookback: int = 252,
) -> pd.Series:
    """Classify macro regime into one of 4 quadrants.

    Args:
        ism_pmi: ISM Manufacturing PMI (FRED: MANEMP or similar)
        nfp_3m_change: Non-Farm Payroll 3-month change
        cpi_yoy: CPI year-over-year rate (FRED: CPIAUCSL pct_change(12))
        breakeven_5y5y: 5y5y forward breakeven inflation (FRED: T5YIFR)
        lookback: Rolling Z-score window

    Returns:
        Series of quadrant labels aligned to common index.
    """
    # Growth Z-score: PMI + NFP
    common = ism_pmi.index
    for s in [nfp_3m_change, cpi_yoy, breakeven_5y5y]:
        common = common.intersection(s.index)

    if len(common) == 0:
        return pd.Series(dtype=str)

    growth_z = (
        _zscore_series((ism_pmi.loc[common] - 50).clip(-20, 20), lookback) * 0.5
        + _zscore_series(nfp_3m_change.loc[common], lookback) * 0.5
    )
    infl_z = (
        _zscore_series(cpi_yoy.loc[common], lookback) * 0.5
        + _zscore_series(breakeven_5y5y.loc[common], lookback) * 0.5
    )

    growth_up = growth_z > 0
    infl_up = infl_z > 0

    quadrant = pd.Series(
        [_QUADRANT_LABELS[(bool(g), bool(i))] for g, i in zip(growth_up, infl_up)],
        index=common,
        name="macro_quadrant",
    )
    logger.info("Macro quadrant distribution: %s", quadrant.value_counts().to_dict())
    return quadrant


def current_quadrant_from_fred(
    fred_client: Any,
    lookback: int = 252,
    as_of: pd.Timestamp | None = None,
) -> str:
    """Fetch FRED data and return macro quadrant label.

    Args:
        fred_client: fredapi.Fred instance.
        lookback: Z-score window.
        as_of: PIT cutoff. When given (backtest mode), FRED requests are bounded
            via observation_end and series sliced ≤ as_of before iloc[-1].
            None (default) → live mode, fetches latest data.

    Returns:
        Quadrant label string or 'unknown'.

    F-B-4 MAJOR fix: previously fetched all FRED series unbounded and took
    iloc[-1] regardless of caller context → forward leak in backtests.
    """
    try:
        kwargs = {}
        if as_of is not None:
            kwargs["observation_end"] = as_of.strftime("%Y-%m-%d")
        ism = fred_client.get_series("MANEMP", **kwargs)
        nfp = fred_client.get_series("PAYEMS", **kwargs).pct_change(3) * 100
        cpi_raw = fred_client.get_series("CPIAUCSL", **kwargs)
        cpi_yoy = cpi_raw.pct_change(12) * 100
        be5y5y = fred_client.get_series("T5YIFR", **kwargs)

        if as_of is not None:
            ism = ism[ism.index <= as_of]
            nfp = nfp[nfp.index <= as_of]
            cpi_yoy = cpi_yoy[cpi_yoy.index <= as_of]
            be5y5y = be5y5y[be5y5y.index <= as_of]

        quadrant_series = compute_macro_quadrant(ism, nfp, cpi_yoy, be5y5y, lookback)
        if quadrant_series.empty:
            return "unknown"
        return str(quadrant_series.iloc[-1])
    except Exception as exc:
        logger.warning("Macro quadrant from FRED failed: %s", exc)
        return "unknown"


def quadrant_exposure_bias(quadrant: str) -> dict[str, float]:
    """Return sector exposure bias multipliers for a given quadrant.

    Returns dict mapping sector/asset class → multiplier [0.5, 1.5].
    Sectors in the quadrant's favored list → 1.3, others → 0.9.
    """
    favored = _QUADRANT_ALLOCATIONS.get(quadrant, [])
    all_sectors = [
        "commodities",
        "emerging",
        "value",
        "growth",
        "large_cap_tech",
        "gold",
        "defensive_value",
        "treasuries",
        "cash",
    ]
    return {s: 1.3 if s in favored else 0.9 for s in all_sectors}


__all__ = [
    "compute_macro_quadrant",
    "current_quadrant_from_fred",
    "quadrant_exposure_bias",
]
