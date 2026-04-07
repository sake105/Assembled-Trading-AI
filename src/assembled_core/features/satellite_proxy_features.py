"""Satellite Proxy Features (Plan 3.9 / 10.8).

Non-traditional economic indicators:
- Baltic Dry Index (BDI): global trade activity proxy
- Copper/Gold Ratio: risk-on/off indicator (Dr. Copper)
- Oil/Gold Ratio: inflation/deflation proxy
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_copper_gold_ratio(copper: pd.Series, gold: pd.Series) -> pd.Series:
    """Copper/Gold ratio — risk-on/off indicator."""
    return (copper / gold.replace(0, np.nan)).rename("copper_gold_ratio")


def compute_oil_gold_ratio(oil: pd.Series, gold: pd.Series) -> pd.Series:
    """Oil/Gold ratio — inflation/deflation proxy."""
    return (oil / gold.replace(0, np.nan)).rename("oil_gold_ratio")


def compute_bdi_features(
    bdi: pd.Series,
    window_short: int = 20,
    window_long: int = 60,
) -> pd.DataFrame:
    """Compute BDI-derived features."""
    features = pd.DataFrame(index=bdi.index)
    features["bdi_level"] = bdi
    features["bdi_zscore"] = (
        (bdi - bdi.rolling(window_long, min_periods=20).mean())
        / bdi.rolling(window_long, min_periods=20).std().replace(0, np.nan)
    )
    features["bdi_momentum"] = bdi.pct_change(window_short)
    return features


__all__ = ["compute_copper_gold_ratio", "compute_oil_gold_ratio", "compute_bdi_features"]
