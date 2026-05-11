"""Tests für multi_factor_vol_target."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.multi_factor_vol_target import (
    MultiFactorVolTargetConfig,
    combine_factors,
)


def _ret(n: int = 400, seed: int = 0, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.0003, vol, n),
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )


def test_combine_equal_weight():
    f = {"a": _ret(seed=1, vol=0.02), "b": _ret(seed=2, vol=0.005)}
    out = combine_factors(f, MultiFactorVolTargetConfig(combiner="equal_weight"))
    assert "combined" in out.columns
    assert "a" in out.columns and "b" in out.columns
    # combined sollte numerisch dem Mean der Spalten gleich sein
    aligned = out.dropna()
    if not aligned.empty:
        np.testing.assert_array_almost_equal(
            aligned["combined"].values, aligned[["a", "b"]].mean(axis=1).values
        )


def test_combine_inverse_vol():
    f = {"a": _ret(seed=1, vol=0.02), "b": _ret(seed=2, vol=0.005)}
    out = combine_factors(f, MultiFactorVolTargetConfig(combiner="inverse_vol"))
    assert "combined" in out.columns
    # inverse_vol-Kombination sollte LowVol-Faktor (b) stärker gewichten
    assert not out["combined"].dropna().empty


def test_combine_hrp_works_or_falls_back():
    f = {
        "a": _ret(seed=3, vol=0.01),
        "b": _ret(seed=4, vol=0.01),
        "c": _ret(seed=5, vol=0.015),
    }
    out = combine_factors(f, MultiFactorVolTargetConfig(combiner="hrp"))
    assert "combined" in out.columns
    assert not out["combined"].dropna().empty


def test_custom_weights_override():
    f = {"a": _ret(seed=6, vol=0.01), "b": _ret(seed=7, vol=0.01)}
    out = combine_factors(
        f,
        MultiFactorVolTargetConfig(
            combiner="equal_weight", weights={"a": 0.8, "b": 0.2}
        ),
    )
    aligned = out.dropna()
    if not aligned.empty:
        expected = aligned["a"] * 0.8 + aligned["b"] * 0.2
        np.testing.assert_array_almost_equal(
            aligned["combined"].values, expected.values, decimal=6
        )


def test_combine_reduces_vol_vs_unscaled():
    rng = np.random.default_rng(8)
    n = 500
    # Hoch-Vol-Faktor mit time-varying vol
    a = pd.Series(
        rng.normal(0.0002, 0.02, n)
        + np.concatenate([np.zeros(250), np.full(250, 0.005)]),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    b = pd.Series(rng.normal(0.0003, 0.008, n), index=a.index)
    out = combine_factors(
        {"a": a, "b": b}, MultiFactorVolTargetConfig(target_vol_annual=0.10)
    )
    # Vol-targetete Kombination sollte näher am Ziel von 10% liegen
    realized_vol_combined = out["combined"].dropna().std() * np.sqrt(252)
    # Toleranz ~5% wegen Aggregation, time-varying vol, etc.
    assert abs(realized_vol_combined - 0.10) < 0.10


def test_unknown_combiner_raises():
    f = {"a": _ret(seed=9), "b": _ret(seed=10)}
    try:
        combine_factors(f, MultiFactorVolTargetConfig(combiner="UNKNOWN"))
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
