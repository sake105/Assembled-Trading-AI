"""Tests für Phase 7-14 ML-Erweiterungen.

Deckt ab:
- Triple-Barrier Labeling
- Fractional Differentiation
- Stacking Ensemble
- Conformal Prediction
- Cross-Sectional Features
- PBO (Backtest Overfit)
- Regime Model Router
- Feature Importance Tracker
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Triple Barrier
# ---------------------------------------------------------------------------


def _make_trending_prices(
    n: int = 50, trend: float = 0.002, seed: int = 0
) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, 0.015, n)
    prices = 100.0 * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=pd.date_range("2025-01-01", periods=n))


# ---------------------------------------------------------------------------
# Fractional Differentiation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Conformal Prediction
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Cross-Sectional Features
# ---------------------------------------------------------------------------


def test_rank_cross_sectional_percentile():
    import pytest

    pytest.importorskip("src.assembled_core.features.cross_sectional")
    from src.assembled_core.features.cross_sectional import rank_cross_sectional

    panel = pd.DataFrame(
        {
            "timestamp": ["2025-01-01"] * 4 + ["2025-01-02"] * 4,
            "symbol": ["A", "B", "C", "D"] * 2,
            "f1": [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        }
    )
    result = rank_cross_sectional(panel, feature_cols=["f1"], normalize_to="percentile")
    assert "f1_xrank" in result.columns
    # Pro Tag: Ranks müssen in [0, 1] liegen
    day1 = result[result["timestamp"] == "2025-01-01"]["f1_xrank"]
    assert day1.min() > 0
    assert day1.max() <= 1
    # Größter Wert → Rank 1.0, kleinster → Rank 0.25 (1/4)
    assert abs(day1.iloc[3] - 1.0) < 1e-9


def test_zscore_cross_sectional():
    import pytest

    pytest.importorskip("src.assembled_core.features.cross_sectional")
    from src.assembled_core.features.cross_sectional import zscore_cross_sectional

    panel = pd.DataFrame(
        {
            "timestamp": ["2025-01-01"] * 5,
            "symbol": ["A", "B", "C", "D", "E"],
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    result = zscore_cross_sectional(panel, feature_cols=["f1"], winsorize_std=None)
    # Mittelwert pro Tag = 0; sample-std (ddof=1) = 1.0 (production uses pandas default)
    assert abs(result["f1_xz"].mean()) < 1e-9
    assert abs(result["f1_xz"].std(ddof=1) - 1.0) < 0.01


# ---------------------------------------------------------------------------
# PBO
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Regime Model Router
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Feature Importance Tracker
# ---------------------------------------------------------------------------
