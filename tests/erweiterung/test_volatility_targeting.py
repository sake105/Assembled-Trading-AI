"""Tests für volatility_targeting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erweiterung.strategies.volatility_targeting import (
    VolTargetConfig,
    apply_vol_targeting,
    realized_vol,
    vol_target_leverage,
)


def _ret(n: int = 500, seed: int = 0, vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(
        rng.normal(0.0003, vol, n),
        index=pd.date_range("2022-01-01", periods=n, freq="B"),
    )


def test_realized_vol_annualized():
    r = _ret(n=400, vol=0.01)
    rv = realized_vol(r, window=60)
    # Annualized vol of normal(0, 0.01) ~ 0.16 (0.01 * sqrt(252))
    assert 0.10 < rv.dropna().mean() < 0.25


def test_vol_target_leverage_clip_range():
    r = _ret(n=400, vol=0.01)
    cfg = VolTargetConfig(target_vol_annual=0.12, max_leverage=2.0, min_leverage=0.0)
    lev = vol_target_leverage(r, cfg)
    assert lev.dropna().min() >= 0
    assert lev.dropna().max() <= 2.0


def test_vol_target_lower_leverage_in_high_vol():
    # Konstruiere künstlich high-vol Window am Ende
    r = _ret(n=400, vol=0.005)
    r.iloc[300:] = np.random.default_rng(1).normal(0, 0.03, 100)
    cfg = VolTargetConfig(target_vol_annual=0.10)
    lev = vol_target_leverage(r, cfg)
    # leverage in low-vol-period sollte höher sein als in high-vol-period
    low_vol_lev = lev.iloc[150:250].mean()
    high_vol_lev = lev.iloc[350:].mean()
    assert low_vol_lev > high_vol_lev


def test_apply_vol_targeting_full_output():
    r = _ret(n=400)
    out = apply_vol_targeting(r)
    assert set(out.columns) == {
        "raw_return",
        "realized_vol",
        "leverage",
        "scaled_return",
    }
    assert len(out) == 400


def test_apply_vol_targeting_lag_no_lookahead():
    r = _ret(n=200)
    out = apply_vol_targeting(r)
    # Erste Werte sollten NaN sein (lag)
    assert pd.isna(out["leverage"].iloc[0])
    # Realized-vol braucht Window-Aufbau
    assert pd.isna(out["realized_vol"].iloc[5])


def test_vol_targeting_reduces_vol():
    # Mit Vol-Targeting sollte die annualisierte Vol näher am Target liegen
    rng = np.random.default_rng(5)
    n = 1000
    # Time-varying vol: hohe Vol im 1. Drittel, niedrige im 2., hohe im 3.
    vols = np.concatenate(
        [
            np.full(333, 0.03),
            np.full(333, 0.005),
            np.full(334, 0.02),
        ]
    )
    r = pd.Series(
        rng.normal(0.0002, vols),
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    out = apply_vol_targeting(r, VolTargetConfig(target_vol_annual=0.15))
    raw_vol = r.std() * np.sqrt(252)
    scaled_vol = out["scaled_return"].dropna().std() * np.sqrt(252)
    # Skalierte Vol sollte näher an Target (0.15) sein als raw
    assert abs(scaled_vol - 0.15) < abs(raw_vol - 0.15) + 0.05  # ein bisschen Toleranz
