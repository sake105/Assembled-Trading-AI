"""Tests for Market-Model + BMP-t-stat + BHAR — C4-081 closure.

References:
- MacKinlay (1997), "Event Studies in Economics and Finance"
- Boehmer, Musumeci, Poulsen (1991), JFE 30(2)
- Barber & Lyon (1997), JFE 43(3) — BHAR
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.qa.event_study import (
    MarketModelResult,
    bmp_t_statistic,
    compute_bhar,
    compute_market_model_abnormal_returns,
    estimate_market_model,
)


# ---------------------------------------------------------------------------
# estimate_market_model
# ---------------------------------------------------------------------------


def test_estimate_market_model_recovers_known_alpha_beta():
    """Synthetic returns with known α=0.001, β=1.2 → estimator should recover."""
    rng = np.random.default_rng(42)
    n = 500
    market = rng.normal(0.0005, 0.012, n)
    noise = rng.normal(0, 0.005, n)
    asset = 0.001 + 1.2 * market + noise

    result = estimate_market_model(asset, market)
    assert isinstance(result, MarketModelResult)
    assert abs(result.alpha - 0.001) < 0.001
    assert abs(result.beta - 1.2) < 0.05
    assert result.sigma_resid > 0
    assert result.n_estimation_obs == n
    assert 0 < result.r_squared < 1


def test_estimate_market_model_zero_beta_for_uncorrelated():
    """If asset returns are independent of market → β ≈ 0."""
    rng = np.random.default_rng(0)
    n = 500
    market = rng.normal(0, 0.012, n)
    asset = rng.normal(0, 0.012, n)  # independent

    result = estimate_market_model(asset, market)
    assert abs(result.beta) < 0.15, (
        f"β should be ≈0 for independent series, got {result.beta}"
    )


def test_estimate_market_model_rejects_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        estimate_market_model([1.0, 2.0, 3.0], [1.0, 2.0])


def test_estimate_market_model_rejects_short_input():
    with pytest.raises(ValueError, match="30"):
        estimate_market_model(np.zeros(20), np.zeros(20))


def test_estimate_market_model_handles_nans():
    """Drops NaN rows before fitting (need ≥30 finite-aligned)."""
    rng = np.random.default_rng(1)
    n = 100
    asset = rng.normal(0, 0.01, n)
    market = rng.normal(0, 0.01, n)
    asset[5] = np.nan
    market[10] = np.nan
    result = estimate_market_model(asset, market)
    assert result.n_estimation_obs == n - 2  # 2 NaN rows dropped


# ---------------------------------------------------------------------------
# compute_market_model_abnormal_returns
# ---------------------------------------------------------------------------


def _make_event_panel(
    n_events: int = 3, rel_days_range: tuple[int, int] = (-260, 20)
) -> pd.DataFrame:
    """Build a synthetic event-returns DataFrame for testing."""
    rng = np.random.default_rng(123)
    rows = []
    for eid in range(n_events):
        for rd in range(rel_days_range[0], rel_days_range[1] + 1):
            market = rng.normal(0.0005, 0.012)
            # α=0.0002, β=1.0 + noise. Event day (rd=0) gets a +2% jump.
            event_jump = 0.02 if rd == 0 else 0.0
            asset = 0.0002 + 1.0 * market + rng.normal(0, 0.005) + event_jump
            rows.append(
                {
                    "event_id": eid,
                    "rel_day": rd,
                    "event_return": asset,
                    "market_return": market,
                }
            )
    return pd.DataFrame(rows)


def test_market_model_abnormal_returns_attached_columns():
    panel = _make_event_panel(n_events=2)
    result = compute_market_model_abnormal_returns(panel)
    assert "mm_abnormal_return" in result.columns
    assert "sigma_resid" in result.columns


def test_market_model_abnormal_returns_event_day_jump_detected():
    """The known +2% jump on rel_day=0 should appear in mm_abnormal_return."""
    panel = _make_event_panel(n_events=5)
    result = compute_market_model_abnormal_returns(panel)
    event_day_ars = result[result["rel_day"] == 0]["mm_abnormal_return"].dropna()
    assert len(event_day_ars) > 0
    # Mean of event-day ARs should be close to +0.02 (within 30% tolerance)
    assert 0.015 < event_day_ars.mean() < 0.025, (
        f"Expected event-day AR ≈ 0.02, got {event_day_ars.mean():.4f}"
    )


def test_market_model_abnormal_returns_too_short_estimation_skipped():
    """Events with <30 obs in estimation window get NaN AR + NaN sigma."""
    panel = _make_event_panel(n_events=1, rel_days_range=(-5, 5))  # only 11 obs
    result = compute_market_model_abnormal_returns(panel)
    assert result["mm_abnormal_return"].isna().all()
    assert result["sigma_resid"].isna().all()


def test_market_model_abnormal_returns_rejects_missing_columns():
    bad_panel = pd.DataFrame({"event_id": [1], "rel_day": [0], "event_return": [0.01]})
    with pytest.raises(KeyError, match="market_return"):
        compute_market_model_abnormal_returns(bad_panel)


# ---------------------------------------------------------------------------
# bmp_t_statistic
# ---------------------------------------------------------------------------


def test_bmp_t_detects_event_day_significance():
    """Events with a +2% jump on day 0 → BMP-t should be highly significant."""
    panel = _make_event_panel(n_events=20)
    ar_df = compute_market_model_abnormal_returns(panel)
    result = bmp_t_statistic(ar_df, event_window=(0, 0))
    assert result["n_events"] == 20
    assert result["t_statistic"] > 2.0, (
        f"Expected significant t, got {result['t_statistic']:.2f}"
    )
    assert result["is_significant_at_5pct"] is True
    assert 0.015 < result["car_mean"] < 0.025


def test_bmp_t_non_significant_for_no_event():
    """If no event-day jump (all returns are pure market-model noise) → not significant."""
    rng = np.random.default_rng(7)
    n_events = 30
    rows = []
    for eid in range(n_events):
        for rd in range(-260, 11):
            market = rng.normal(0.0005, 0.012)
            asset = 0.0002 + 1.0 * market + rng.normal(0, 0.005)  # NO jump
            rows.append(
                {
                    "event_id": eid,
                    "rel_day": rd,
                    "event_return": asset,
                    "market_return": market,
                }
            )
    panel = pd.DataFrame(rows)
    ar_df = compute_market_model_abnormal_returns(panel)
    result = bmp_t_statistic(ar_df, event_window=(0, 0))
    assert abs(result["t_statistic"]) < 3.0, (
        f"With no real effect, |t| should be small, got {result['t_statistic']:.2f}"
    )


def test_bmp_t_window_summed_across_days():
    """A 5-day window with persistent positive AR should integrate to significant."""
    panel = _make_event_panel(n_events=15)
    ar_df = compute_market_model_abnormal_returns(panel)
    result = bmp_t_statistic(ar_df, event_window=(-1, 1))
    assert result["n_events"] == 15
    # CAR over 3 days should still be ≈ 0.02 (only day 0 has the jump)
    assert 0.01 < result["car_mean"] < 0.03


def test_bmp_t_rejects_no_valid_events():
    bad = pd.DataFrame(
        {
            "event_id": [1],
            "rel_day": [0],
            "mm_abnormal_return": [np.nan],
            "sigma_resid": [np.nan],
        }
    )
    with pytest.raises(ValueError, match="no events"):
        bmp_t_statistic(bad)


# ---------------------------------------------------------------------------
# compute_bhar
# ---------------------------------------------------------------------------


def test_bhar_detects_post_event_outperformance():
    """Events with +2% jump on day 0 (post-event) → BHAR > 0."""
    panel = _make_event_panel(n_events=10)
    bhar_df = compute_bhar(panel, horizon_days=5)
    assert "bhar" in bhar_df.columns
    assert len(bhar_df) == 10
    # Mean BHAR should be positive (event-day jump compounds into BHAR window)
    assert bhar_df["bhar"].mean() > 0.0


def test_bhar_zero_for_no_post_event_alpha():
    """With no event-day jump → BHAR mean should be small."""
    rng = np.random.default_rng(99)
    rows = []
    for eid in range(20):
        for rd in range(0, 11):
            market = rng.normal(0.0005, 0.012)
            asset = 0.0002 + 1.0 * market + rng.normal(0, 0.005)
            rows.append(
                {
                    "event_id": eid,
                    "rel_day": rd,
                    "event_return": asset,
                    "market_return": market,
                }
            )
    panel = pd.DataFrame(rows)
    bhar_df = compute_bhar(panel, horizon_days=10)
    # Mean BHAR should be close to 0 (cumulative α=0.0002 over 10 days)
    assert abs(bhar_df["bhar"].mean()) < 0.05


def test_bhar_rejects_invalid_horizon():
    panel = _make_event_panel(n_events=1)
    with pytest.raises(ValueError, match="horizon_days"):
        compute_bhar(panel, horizon_days=0)


def test_bhar_rejects_missing_columns():
    bad = pd.DataFrame({"event_id": [1], "rel_day": [0], "event_return": [0.01]})
    with pytest.raises(KeyError, match="market_return"):
        compute_bhar(bad)


def test_bhar_returns_empty_for_no_in_window_data():
    """If no rows in [0, horizon] → empty DataFrame, not error."""
    panel = _make_event_panel(n_events=1, rel_days_range=(-30, -1))  # all pre-event
    bhar_df = compute_bhar(panel, horizon_days=10)
    assert len(bhar_df) == 0
    assert list(bhar_df.columns) == ["event_id", "bhar", "n_obs_in_window"]


def test_bhar_paired_nan_alignment(caplog):
    """F-senior-c4-081-1 regression: when asset has NaN on day 3 and market has
    NaN on day 5, those days must be EXCLUDED from BOTH series before
    compounding. Earlier impl dropna-ed each series independently then
    truncated — surviving values no longer aligned by rel_day."""
    panel = pd.DataFrame(
        [
            {"event_id": 1, "rel_day": 0, "event_return": 0.01, "market_return": 0.005},
            {"event_id": 1, "rel_day": 1, "event_return": 0.02, "market_return": 0.010},
            {"event_id": 1, "rel_day": 2, "event_return": 0.03, "market_return": 0.015},
            {
                "event_id": 1,
                "rel_day": 3,
                "event_return": np.nan,
                "market_return": 0.020,
            },  # asset NaN
            {"event_id": 1, "rel_day": 4, "event_return": 0.05, "market_return": 0.025},
            {
                "event_id": 1,
                "rel_day": 5,
                "event_return": 0.06,
                "market_return": np.nan,
            },  # market NaN
        ]
    )
    bhar_df = compute_bhar(panel, horizon_days=5)
    assert len(bhar_df) == 1
    # Aligned: drop days 3 and 5 from BOTH → asset: [0.01, 0.02, 0.03, 0.05],
    # market: [0.005, 0.010, 0.015, 0.025] (4 days each, same rel_days {0,1,2,4})
    expected_asset = (1.01) * (1.02) * (1.03) * (1.05)
    expected_market = (1.005) * (1.010) * (1.015) * (1.025)
    expected_bhar = expected_asset - expected_market
    actual = float(bhar_df.iloc[0]["bhar"])
    assert abs(actual - expected_bhar) < 1e-9, (
        f"expected {expected_bhar:.6f}, got {actual:.6f}"
    )
    assert int(bhar_df.iloc[0]["n_obs_in_window"]) == 4
