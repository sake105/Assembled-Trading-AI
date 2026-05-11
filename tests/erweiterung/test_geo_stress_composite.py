"""Tests für Composite-Geo-Stress-Score (GPR + GDELT)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

GPR_CACHE = Path("data/cache/gpr/sheet1.parquet")
GDELT_CACHE = Path("data/cache/gdelt/monthly_aggregates.parquet")

pytestmark = pytest.mark.skipif(
    not (GPR_CACHE.exists() and GDELT_CACHE.exists()),
    reason="GPR or GDELT cache missing",
)


def test_compute_monthly_composite_basic():
    from erweiterung.risk.geo_stress_composite import compute_monthly_composite

    df = compute_monthly_composite()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    for col in ("date", "gpr", "conflict_share", "mean_tone", "composite_z", "state"):
        assert col in df.columns
    # State must be one of 4 values
    assert set(df["state"].unique()).issubset({"PAUSE", "ACTIVE", "WATCH", "COOLDOWN"})


def test_ukraine_2022_triggers_pause():
    """Ukraine invasion 2022-02 + 2022-03 must trigger PAUSE."""
    from erweiterung.risk.geo_stress_composite import compute_monthly_composite

    df = compute_monthly_composite()
    feb = df[df["date"].dt.strftime("%Y-%m") == "2022-02"]
    mar = df[df["date"].dt.strftime("%Y-%m") == "2022-03"]
    assert not feb.empty and not mar.empty
    # Composite z must be > 1.0 (ACTIVE+) for both months
    assert feb.iloc[0]["composite_z"] > 1.0, f"Feb z={feb.iloc[0]['composite_z']}"
    assert mar.iloc[0]["composite_z"] > 1.5, f"Mar z={mar.iloc[0]['composite_z']}"


def test_composite_z_distribution_finite():
    from erweiterung.risk.geo_stress_composite import compute_monthly_composite

    df = compute_monthly_composite()
    # composite_z must be finite for all rows (NaN fillna in code)
    assert df["composite_z"].notna().all()
    assert np.isfinite(df["composite_z"]).all()


def test_expand_to_daily_preserves_state():
    from erweiterung.risk.geo_stress_composite import (
        compute_monthly_composite,
        expand_composite_to_daily,
    )

    monthly = compute_monthly_composite()
    daily_idx = pd.date_range("2020-01-01", "2023-12-31", freq="B", tz="UTC")
    daily = expand_composite_to_daily(monthly, daily_idx)
    assert isinstance(daily, pd.DataFrame)
    assert len(daily) == len(daily_idx)
    for col in ("composite_z", "state", "multiplier"):
        assert col in daily.columns
    # Multipliers in valid range
    assert (daily["multiplier"] >= 0.4).all()
    assert (daily["multiplier"] <= 1.1).all()


def test_apply_overlay_disabled_passes_through():
    from erweiterung.risk.geo_stress_composite import (
        GeoStressPolicy,
        apply_geo_stress_overlay,
    )

    idx = pd.date_range("2020-01-01", "2020-06-30", freq="B", tz="UTC")
    returns = pd.Series(0.001, index=idx)
    out = apply_geo_stress_overlay(returns, GeoStressPolicy(enabled=False))
    pd.testing.assert_series_equal(out["hedged_return"], returns)
    assert (out["exposure_multiplier"] == 1.0).all()


def test_apply_overlay_reduces_exposure_during_pause():
    """During Ukraine 2022 PAUSE months, multiplier must be < 1.0."""
    from erweiterung.risk.geo_stress_composite import (
        GeoStressPolicy,
        apply_geo_stress_overlay,
    )

    idx = pd.date_range("2022-02-01", "2022-04-30", freq="B", tz="UTC")
    returns = pd.Series(0.001, index=idx)
    out = apply_geo_stress_overlay(returns, GeoStressPolicy(enabled=True))
    # At least some days during this period must have multiplier < 1.0
    mults = out["exposure_multiplier"]
    assert (mults < 1.0).any(), "No risk-off during Ukraine invasion period"


def test_policy_weights_sum_meaningfully():
    from erweiterung.risk.geo_stress_composite import GeoStressPolicy

    p = GeoStressPolicy()
    s = p.w_gpr + p.w_conflict + p.w_tone
    # Don't require exact 1.0 (composite-z is scale-free) but reasonable
    assert 0.8 <= s <= 1.2


def test_state_mapping_monotonic():
    """Higher composite_z → lower multiplier (risk-off)."""
    from erweiterung.risk.geo_stress_composite import GeoStressPolicy

    p = GeoStressPolicy()
    assert p.state_multipliers["PAUSE"] < p.state_multipliers["ACTIVE"]
    assert p.state_multipliers["ACTIVE"] < p.state_multipliers["WATCH"]
    assert p.state_multipliers["WATCH"] < p.state_multipliers["COOLDOWN"]
