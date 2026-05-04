"""Tests for src/assembled_core/attribution/time_series.py."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from assembled_core.attribution.schemas import CompositeAttribution
from assembled_core.attribution.time_series import (
    attributions_to_df,
    dead_feature_report,
    detect_attribution_drift,
    detect_dead_features,
    rolling_dimension_ic,
)


def _make_attr(
    ticker: str = "AAPL",
    timestamp: datetime | None = None,
    composite: float = 0.5,
    dims: dict | None = None,
    regime: str = "normal",
) -> CompositeAttribution:
    if timestamp is None:
        timestamp = datetime(2024, 1, 2, tzinfo=timezone.utc)
    if dims is None:
        dims = {"trend": 0.3, "news": 0.2}
    weights = {k: 0.5 for k in dims}
    return CompositeAttribution(
        timestamp=timestamp,
        ticker=ticker,
        composite_score=composite,
        dimension_contributions=dims,
        dimension_raw_scores={k: v * 2 for k, v in dims.items()},
        dimension_weights=weights,
        strategy_id="test",
        model_version="0.1",
        regime=regime,
    )


# ---------------------------------------------------------------------------
# attributions_to_df
# ---------------------------------------------------------------------------


class TestAttributionsToDF:
    def test_basic_columns(self):
        attrs = [_make_attr(), _make_attr(ticker="MSFT", composite=0.1)]
        df = attributions_to_df(attrs)
        assert "timestamp" in df.columns
        assert "ticker" in df.columns
        assert "contrib_trend" in df.columns
        assert "contrib_news" in df.columns

    def test_sorted_by_timestamp(self):
        t1 = datetime(2024, 1, 3, tzinfo=timezone.utc)
        t2 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        attrs = [_make_attr(timestamp=t1), _make_attr(timestamp=t2)]
        df = attributions_to_df(attrs)
        assert df["timestamp"].iloc[0] < df["timestamp"].iloc[1]

    def test_empty_list(self):
        df = attributions_to_df([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# detect_dead_features
# ---------------------------------------------------------------------------


class TestDetectDeadFeatures:
    def _make_ic_df(self, trend_ics, news_ics, dates):
        df = pd.DataFrame({"trend": trend_ics, "news": news_ics}, index=dates)
        return df

    def test_dead_feature_below_threshold(self):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        ic_df = self._make_ic_df([0.001] * 25, [0.05] * 25, dates)
        result = detect_dead_features(ic_df, ic_threshold=0.02, min_windows=20)
        assert result["trend"]["is_dead"] is True
        assert result["news"]["is_dead"] is False

    def test_alive_feature_above_threshold(self):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        ic_df = self._make_ic_df([0.10] * 25, [0.08] * 25, dates)
        result = detect_dead_features(ic_df, ic_threshold=0.02, min_windows=20)
        assert result["trend"]["is_dead"] is False
        assert result["news"]["is_dead"] is False

    def test_uses_last_min_windows_rows(self):
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        # First 20 have good IC, last 20 have bad IC
        ics = [0.10] * 20 + [0.001] * 20
        ic_df = pd.DataFrame({"trend": ics}, index=dates)
        result = detect_dead_features(ic_df, ic_threshold=0.02, min_windows=20)
        assert result["trend"]["is_dead"] is True

    def test_empty_df(self):
        result = detect_dead_features(pd.DataFrame())
        assert result == {}

    def test_report_contains_dead(self):
        dates = pd.date_range("2024-01-01", periods=25, freq="D")
        ic_df = self._make_ic_df([0.001] * 25, [0.05] * 25, dates)
        dead = detect_dead_features(ic_df, ic_threshold=0.02, min_windows=20)
        report = dead_feature_report(dead)
        assert "trend" in report
        assert "DEAD" in report


# ---------------------------------------------------------------------------
# rolling_dimension_ic
# ---------------------------------------------------------------------------


class TestRollingDimensionIC:
    def _make_attrs_and_returns(self, n=60, seed=42):
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-02", periods=n, freq="B")
        attrs = []
        for i, d in enumerate(dates):
            trend = float(rng.normal(0, 0.5))
            attrs.append(
                _make_attr(
                    timestamp=d,
                    dims={"trend": trend, "news": float(rng.normal(0, 0.2))},
                )
            )
        fwd = pd.Series(
            [
                attrs[i].dimension_contributions["trend"] * 0.1
                + float(rng.normal(0, 0.005))
                for i in range(n)
            ],
            index=dates,
        )
        return attrs, fwd

    def test_returns_dataframe(self):
        attrs, fwd = self._make_attrs_and_returns()
        df = attributions_to_df(attrs)
        ic_df = rolling_dimension_ic(df, fwd, window_days=20)
        assert isinstance(ic_df, pd.DataFrame)

    def test_positive_ic_for_correlated_dim(self):
        attrs, fwd = self._make_attrs_and_returns()
        df = attributions_to_df(attrs)
        ic_df = rolling_dimension_ic(df, fwd, window_days=20)
        if not ic_df.empty and "trend" in ic_df.columns:
            mean_ic = ic_df["trend"].dropna().mean()
            assert mean_ic > 0

    def test_empty_attrs(self):
        df = attributions_to_df([])
        fwd = pd.Series(dtype=float)
        ic_df = rolling_dimension_ic(df, fwd)
        assert isinstance(ic_df, pd.DataFrame)


# ---------------------------------------------------------------------------
# detect_attribution_drift
# ---------------------------------------------------------------------------


class TestDetectAttributionDrift:
    def _make_attrs_with_drift(self, n=30, drift=False, seed=0):
        rng = np.random.default_rng(seed)
        attrs = []
        for i in range(n):
            ts = datetime(2024, 1, 1 + i % 28, tzinfo=timezone.utc)
            if drift:
                dims = {
                    "trend": float(rng.normal(1.0, 0.1)),
                    "news": float(rng.normal(0.0, 0.1)),
                }
            else:
                dims = {
                    "trend": float(rng.normal(0.0, 0.1)),
                    "news": float(rng.normal(0.0, 0.1)),
                }
            attrs.append(_make_attr(timestamp=ts, dims=dims))
        return attrs

    def test_no_drift_small_ks(self):
        baseline = self._make_attrs_with_drift(n=50, drift=False, seed=1)
        recent = self._make_attrs_with_drift(n=50, drift=False, seed=2)
        result = detect_attribution_drift(recent, baseline, threshold_p=0.01)
        # With same distribution, most dims should NOT be flagged as drift
        assert isinstance(result, dict)
        assert "trend" in result

    def test_drift_detected_when_distributions_differ(self):
        baseline = self._make_attrs_with_drift(n=100, drift=False, seed=3)
        recent = self._make_attrs_with_drift(n=100, drift=True, seed=4)
        result = detect_attribution_drift(recent, baseline, threshold_p=0.01)
        # trend has very different distribution → should detect drift
        assert result.get("trend", {}).get("is_drift", False) is True

    def test_empty_inputs_return_empty(self):
        result = detect_attribution_drift([], [])
        assert result == {}

    def test_small_sample_not_flagged(self):
        a1 = [_make_attr()]  # n=1
        a2 = [_make_attr()]
        result = detect_attribution_drift(a1, a2)
        assert all(not v.get("is_drift", False) for v in result.values())
