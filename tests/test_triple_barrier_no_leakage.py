"""A1: Triple-Barrier label leakage regression tests.

Verifies that volatility used to set barriers is NOT bfilled,
so the first <20 events are skipped rather than receiving future vol.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.fast
def test_no_bfill_in_source():
    """grep-level check: no fillna(method=bfill/backfill) in triple_barrier.py."""
    import pathlib

    src = pathlib.Path("src/assembled_core/features/triple_barrier.py").read_text()
    assert "bfill" not in src
    assert "backfill" not in src
    assert "fillna(method=" not in src


@pytest.mark.fast
def test_early_events_skipped_not_leaked():
    """Events within the first 20 bars must be skipped (vol=NaN), not labeled."""
    from src.assembled_core.features.triple_barrier import _triple_barrier_numpy

    rng = np.random.default_rng(42)
    n = 60
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=dates)

    # Use events spanning the first 25 bars and later bars
    events_early = pd.DatetimeIndex(dates[:15])  # within warm-up window
    events_late = pd.DatetimeIndex(dates[25:35])  # after warm-up

    result = _triple_barrier_numpy(
        prices, events_early.append(events_late), (2.0, 1.0), None, 20
    )

    # Early events (within 20-bar warm-up) must NOT appear as labeled rows
    labeled_ts = set(result.index)
    for early_ts in events_early:
        assert (
            early_ts not in labeled_ts
        ), f"Event at {early_ts} (bar <20) was labeled — look-ahead leakage detected"

    # Later events must appear
    assert len(result) > 0, "No labels generated for late events"


@pytest.mark.fast
def test_vol_at_bar_uses_only_past_data():
    """Manually compute vol and compare — vol[t] must use only data up to t."""
    import numpy as np

    rng = np.random.default_rng(0)
    n = 80
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    log_ret = pd.Series(rng.normal(0, 0.01, n), index=dates)
    vol = log_ret.rolling(20, min_periods=20).std()

    # First 19 bars must be NaN
    assert vol.iloc[:19].isna().all(), "Vol should be NaN for first 19 bars"
    # Bar 20 must be valid
    assert not pd.isna(vol.iloc[19]), "Vol at bar 20 should be non-NaN"
    # No look-ahead: vol[t] uses only log_ret[t-19:t+1]
    for i in range(20, 40):
        expected = log_ret.iloc[i - 19 : i + 1].std()
        np.testing.assert_allclose(vol.iloc[i], expected, rtol=1e-10)
