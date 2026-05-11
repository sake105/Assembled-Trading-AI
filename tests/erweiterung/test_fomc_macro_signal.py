"""Tests für fomc_macro_signal."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.fomc_macro_signal import (
    FOMCMacroConfig,
    apply_fomc_allocation_override,
    build_fomc_signal_series,
)


def test_build_fomc_signal_with_empty_statements():
    idx = pd.date_range("2022-01-01", periods=100, freq="B", tz="UTC")
    out = build_fomc_signal_series([], [], idx)
    assert (out == 1.0).all()


def test_build_fomc_signal_hawkish_reduces_exposure():
    """Sequence: dovish-Erstes -> hawkish-Zweites sollte exposure reducen post-Meeting."""
    hawkish_text = (
        "The Committee voted to raise the federal funds rate. "
        "Inflation pressures remain elevated. Tightening will continue. "
        "Restrictive policy is appropriate to bring inflation down."
    )
    dovish_text = (
        "The Committee voted to lower the federal funds rate. "
        "Easing conditions are appropriate. Stimulus will support recovery. "
        "Accommodative policy will continue to support employment."
    )
    dates = [pd.Timestamp("2022-03-15", tz="UTC"), pd.Timestamp("2022-06-15", tz="UTC")]
    idx = pd.date_range("2022-01-01", periods=200, freq="B", tz="UTC")

    # dove first, hawk second → hawkish delta
    out = build_fomc_signal_series(
        [dovish_text, hawkish_text],
        dates,
        idx,
        FOMCMacroConfig(hawkish_delta_threshold=0.05, decay_days=10),
    )
    # Im hawkish-Window sollte exposure < 1.0 sein
    window = (idx >= dates[1]) & (idx <= dates[1] + pd.Timedelta(days=10))
    assert (out[window] < 1.0).any()


def test_apply_fomc_override_zeroes_when_signal_zero():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=100, freq="B", tz="UTC")
    rets = pd.Series(rng.normal(0.0005, 0.01, 100), index=idx)
    override = pd.Series(0.5, index=idx)
    out = apply_fomc_allocation_override(rets, override)
    np.testing.assert_array_almost_equal((rets * 0.5).values, out.values)


def test_apply_fomc_override_empty():
    out = apply_fomc_allocation_override(pd.Series(dtype=float), pd.Series(dtype=float))
    assert out.empty


def test_decay_window_resets_to_neutral_after_period():
    hawk = "raise tighten restrictive inflation overheating hike"
    dove = "ease lower stimulus accommodate support"
    dates = [pd.Timestamp("2022-03-15", tz="UTC"), pd.Timestamp("2022-06-15", tz="UTC")]
    idx = pd.date_range("2022-01-01", periods=300, freq="B", tz="UTC")

    out = build_fomc_signal_series(
        [dove, hawk],
        dates,
        idx,
        FOMCMacroConfig(hawkish_delta_threshold=0.05, decay_days=7),
    )
    # Nach decay_days zurück zu neutral
    far_after = idx > dates[1] + pd.Timedelta(days=30)
    assert (out[far_after] == 1.0).all()
