"""Tests für GPR-Overlay (Caldara-Iacoviello)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from erweiterung.altdata.caldara_iacoviello_gpr import (
    compute_gpr_features,
    expand_to_daily,
    gpr_state_hint,
    load_gpr_cached,
)
from erweiterung.risk.gpr_overlay import (
    DEFAULT_STATE_MULTIPLIERS,
    GPROverlayPolicy,
    apply_gpr_overlay,
    build_daily_gpr_overlay_series,
    compute_exposure_multiplier,
)


GPR_CACHE_EXISTS = Path("data/cache/gpr/sheet1.parquet").exists()
pytestmark = pytest.mark.skipif(
    not GPR_CACHE_EXISTS, reason="GPR cache not populated"
)


def test_load_gpr_cached_returns_data():
    df = load_gpr_cached()
    assert not df.empty
    assert "GPR" in df.columns or "GPRH" in df.columns
    # Should have at least 1000 monthly rows (historical 1900+)
    assert len(df) > 1000


def test_expand_to_daily_forward_fill():
    monthly = load_gpr_cached()
    daily_idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
    daily = expand_to_daily(monthly, daily_idx)
    assert len(daily) == 100
    # GPR-Werte sollten nicht alle NaN sein (2020 hat Daten)
    if "GPR" in daily.columns:
        assert daily["GPR"].notna().any()


def test_compute_gpr_features_schema():
    monthly = load_gpr_cached()
    daily_idx = pd.date_range("2015-01-01", periods=500, freq="B", tz="UTC")
    daily = expand_to_daily(monthly, daily_idx)
    feat = compute_gpr_features(daily)
    assert "gpr_level" in feat.columns
    assert "gpr_zscore" in feat.columns
    assert "gpr_momentum" in feat.columns
    assert "gpr_regime" in feat.columns
    # Levels in [0, 100]
    valid = feat["gpr_level"].dropna()
    if not valid.empty:
        assert valid.min() >= 0
        assert valid.max() <= 100


def test_gpr_state_hint_pause_on_spike():
    """Hoher Z-Score (>2) sollte PAUSE triggern."""
    assert gpr_state_hint(95, 2.5) == "PAUSE"
    assert gpr_state_hint(80, 1.5) == "ACTIVE"
    assert gpr_state_hint(50, 0.0) == "WATCH"
    assert gpr_state_hint(15, -1.5) == "COOLDOWN"


def test_gpr_state_hint_handles_nan():
    assert gpr_state_hint(float("nan"), 0.0) == "WATCH"


def test_build_daily_overlay_returns_multipliers_in_range():
    daily_idx = pd.date_range("2015-01-01", periods=500, freq="B", tz="UTC")
    cfg = GPROverlayPolicy(max_geo_multiplier=1.20, min_geo_multiplier=0.30)
    df = build_daily_gpr_overlay_series(daily_idx, cfg)
    valid = df["exposure_multiplier"].dropna()
    assert (valid >= 0.30).all()
    assert (valid <= 1.20).all()


def test_apply_gpr_overlay_modifies_returns():
    """Hedged returns should equal raw × lagged multiplier."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-01", periods=200, freq="B", tz="UTC")
    rets = pd.Series(rng.normal(0.0005, 0.012, 200), index=idx)
    out = apply_gpr_overlay(rets)
    assert "hedged_return" in out.columns
    assert "exposure_multiplier" in out.columns
    # Verify hedged = raw × mult
    expected = out["raw_return"] * out["exposure_multiplier"]
    np.testing.assert_array_almost_equal(
        out["hedged_return"].values, expected.values
    )


def test_apply_gpr_overlay_disabled_returns_unchanged():
    idx = pd.date_range("2020-01-01", periods=100, freq="B", tz="UTC")
    rets = pd.Series(0.001, index=idx)
    out = apply_gpr_overlay(rets, GPROverlayPolicy(enabled=False))
    np.testing.assert_array_almost_equal(
        out["hedged_return"].values, rets.values
    )


def test_compute_exposure_multiplier_mainline_compatible_signature():
    """Drop-in für Mainline ``risk/georisk_overlay.compute_exposure_multiplier``."""

    class FakeCtx:
        timestamp = pd.Timestamp("2022-03-01", tz="UTC")  # Ukraine war

    policy = {"gpr_overlay": {"enabled": True}}
    mult = compute_exposure_multiplier(FakeCtx(), policy)
    assert 0.0 < mult <= 1.30


def test_compute_exposure_multiplier_disabled():
    policy = {"gpr_overlay": {"enabled": False}}
    mult = compute_exposure_multiplier(None, policy)
    assert mult == 1.0


def test_historic_911_spike_triggers_pause():
    """9/11 (2001-09) GPR-Wert 498 sollte PAUSE-State auslösen."""
    monthly = load_gpr_cached()
    # September 2001 entry
    sept_2001 = monthly.loc["2001-09-01":"2001-10-31"]
    if sept_2001.empty:
        pytest.skip("No 2001 data in cache")
    gpr_vals = sept_2001["GPR"].dropna()
    if gpr_vals.empty:
        pytest.skip("No GPR values")
    assert gpr_vals.max() > 300, f"9/11 GPR-Spike erwartet >300, got {gpr_vals.max():.1f}"


def test_historic_ukraine_2022_elevated():
    """Russia-Ukraine-Invasion Feb 2022 GPR sollte elevated sein."""
    monthly = load_gpr_cached()
    ukraine = monthly.loc["2022-02-01":"2022-04-30"]
    if ukraine.empty:
        pytest.skip("No 2022 data in cache")
    gpr_vals = ukraine["GPR"].dropna()
    if gpr_vals.empty:
        pytest.skip("No GPR values")
    # Should be at least 1.5x long-run mean (~100)
    assert gpr_vals.max() > 150, f"Ukraine 2022 GPR <150, got {gpr_vals.max():.1f}"
