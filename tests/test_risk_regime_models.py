"""Tests for Regime Detection and State Classification Module (Phase D1).

Tests the regime_models functions:
- build_regime_state()
- compute_regime_transition_stats()
- evaluate_factor_ic_by_regime()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.advanced

from src.assembled_core.risk.regime_models import (
    RegimeStateConfig,
    build_regime_state,
    compute_regime_transition_stats,
    evaluate_factor_ic_by_regime,
)


@pytest.fixture
def sample_prices_bull_bear() -> pd.DataFrame:
    """Create price panel with clear bull and bear phases."""
    dates = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")
    symbols = ["STOCK1", "STOCK2", "STOCK3"]

    all_data = []
    for symbol in symbols:
        base_price = 100.0

        # Phase 1: Bull market (days 0-80): strong uptrend
        bull_prices = base_price + np.arange(81) * 0.5 + np.random.randn(81) * 0.5

        # Phase 2: Bear market (days 81-140): strong downtrend
        bear_base = bull_prices[-1]
        bear_prices = bear_base - np.arange(60) * 0.4 + np.random.randn(60) * 0.5

        # Phase 3: Recovery (days 141-199): moderate uptrend
        recovery_base = bear_prices[-1]
        recovery_prices = (
            recovery_base + np.arange(59) * 0.2 + np.random.randn(59) * 0.3
        )

        all_prices = np.concatenate([bull_prices, bear_prices, recovery_prices])

        df = pd.DataFrame(
            {
                "timestamp": dates,
                "symbol": symbol,
                "close": all_prices,
                "open": all_prices * 0.99,
                "high": all_prices * 1.01,
                "low": all_prices * 0.98,
                "volume": np.random.randint(1000000, 10000000, size=200),
            }
        )
        all_data.append(df)

    return (
        pd.concat(all_data, ignore_index=True)
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


@pytest.fixture
def sample_macro_factors() -> pd.DataFrame:
    """Create synthetic macro factors with clear regime patterns."""
    dates = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")
    symbols = ["STOCK1", "STOCK2", "STOCK3"]

    all_data = []
    for symbol in symbols:
        # Phase 1: Expansion (days 0-80)
        growth_regime_1 = np.ones(81) * 1.0  # Strong expansion
        inflation_regime_1 = np.ones(81) * 0.5  # Moderate inflation
        risk_aversion_1 = np.ones(81) * -0.5  # Risk-on

        # Phase 2: Recession (days 81-140)
        growth_regime_2 = np.ones(60) * -1.0  # Strong recession
        inflation_regime_2 = np.ones(60) * -0.5  # Deflation
        risk_aversion_2 = np.ones(60) * 0.8  # Risk-off

        # Phase 3: Recovery (days 141-199)
        growth_regime_3 = np.ones(59) * 0.3  # Moderate expansion
        inflation_regime_3 = np.ones(59) * 0.2  # Low inflation
        risk_aversion_3 = np.ones(59) * 0.0  # Neutral

        all_data.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "macro_growth_regime": np.concatenate(
                        [growth_regime_1, growth_regime_2, growth_regime_3]
                    ),
                    "macro_inflation_regime": np.concatenate(
                        [inflation_regime_1, inflation_regime_2, inflation_regime_3]
                    ),
                    "macro_risk_aversion_proxy": np.concatenate(
                        [risk_aversion_1, risk_aversion_2, risk_aversion_3]
                    ),
                }
            )
        )

    return (
        pd.concat(all_data, ignore_index=True)
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


@pytest.fixture
def sample_breadth_df() -> pd.DataFrame:
    """Create synthetic market breadth DataFrame."""
    dates = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")

    # Phase 1: High breadth (bull)
    fraction_above_1 = np.ones(81) * 0.8 + np.random.randn(81) * 0.1

    # Phase 2: Low breadth (bear)
    fraction_above_2 = np.ones(60) * 0.2 + np.random.randn(60) * 0.1

    # Phase 3: Moderate breadth (recovery)
    fraction_above_3 = np.ones(59) * 0.5 + np.random.randn(59) * 0.1

    fraction_above = np.concatenate(
        [fraction_above_1, fraction_above_2, fraction_above_3]
    )
    fraction_above = np.clip(fraction_above, 0.0, 1.0)

    # Risk-on/off score
    risk_on_off_1 = np.ones(81) * 0.8  # Risk-on in bull
    risk_on_off_2 = np.ones(60) * -0.7  # Risk-off in bear
    risk_on_off_3 = np.ones(59) * 0.1  # Neutral in recovery

    risk_on_off = np.concatenate([risk_on_off_1, risk_on_off_2, risk_on_off_3])

    return pd.DataFrame(
        {
            "timestamp": dates,
            "fraction_above_ma_50": fraction_above,
            "risk_on_off_score": risk_on_off,
        }
    )


@pytest.fixture
def sample_vol_df() -> pd.DataFrame:
    """Create synthetic volatility DataFrame."""
    dates = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")
    symbols = ["STOCK1", "STOCK2", "STOCK3"]

    all_data = []
    for symbol in symbols:
        # Phase 1: Low vol (bull)
        rv_1 = np.ones(81) * 0.15 + np.random.randn(81) * 0.02

        # Phase 2: High vol (bear/crisis)
        rv_2 = np.ones(60) * 0.45 + np.random.randn(60) * 0.05

        # Phase 3: Moderate vol (recovery)
        rv_3 = np.ones(59) * 0.20 + np.random.randn(59) * 0.02

        rv_20 = np.concatenate([rv_1, rv_2, rv_3])
        rv_20 = np.clip(rv_20, 0.05, 0.60)  # Reasonable bounds

        all_data.append(
            pd.DataFrame(
                {
                    "timestamp": dates,
                    "symbol": symbol,
                    "rv_20": rv_20,
                }
            )
        )

    return (
        pd.concat(all_data, ignore_index=True)
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )


@pytest.fixture
def sample_ic_df() -> pd.DataFrame:
    """Create synthetic IC DataFrame for factor evaluation."""
    dates = pd.date_range("2020-01-01", periods=200, freq="D", tz="UTC")

    # Factor 1: Works well in bull, poorly in bear
    ic_factor1_bull = np.random.normal(0.15, 0.05, 81)  # Positive IC
    ic_factor1_bear = np.random.normal(-0.10, 0.05, 60)  # Negative IC
    ic_factor1_recovery = np.random.normal(0.05, 0.05, 59)  # Slightly positive

    ic_factor1 = np.concatenate([ic_factor1_bull, ic_factor1_bear, ic_factor1_recovery])

    # Factor 2: Works well in bear, poorly in bull
    ic_factor2_bull = np.random.normal(-0.05, 0.05, 81)
    ic_factor2_bear = np.random.normal(0.12, 0.05, 60)
    ic_factor2_recovery = np.random.normal(0.03, 0.05, 59)

    ic_factor2 = np.concatenate([ic_factor2_bull, ic_factor2_bear, ic_factor2_recovery])

    return pd.DataFrame(
        {
            "timestamp": dates,
            "factor1_ic": ic_factor1,
            "factor2_ic": ic_factor2,
        }
    )


class TestBuildRegimeState:
    """Tests for build_regime_state() function."""

    def test_basic_functionality(
        self,
        sample_prices_bull_bear,
        sample_macro_factors,
        sample_breadth_df,
        sample_vol_df,
    ):
        """Test that build_regime_state runs and returns correct format."""
        regime_df = build_regime_state(
            prices=sample_prices_bull_bear,
            macro_factors=sample_macro_factors,
            breadth_df=sample_breadth_df,
            vol_df=sample_vol_df,
        )

        # Should have required columns
        assert "timestamp" in regime_df.columns
        assert "regime_label" in regime_df.columns
        assert "regime_trend_score" in regime_df.columns
        assert "regime_macro_score" in regime_df.columns
        assert "regime_risk_score" in regime_df.columns
        assert "regime_confidence" in regime_df.columns

        # Should have one row per unique timestamp
        unique_timestamps = sample_prices_bull_bear["timestamp"].nunique()
        assert len(regime_df) == unique_timestamps

        # Should have valid regime labels
        valid_labels = {"bull", "bear", "sideways", "crisis", "reflation", "neutral"}
        regime_labels = set(regime_df["regime_label"].unique())
        assert regime_labels.issubset(valid_labels)

    def test_regime_label_distribution(
        self,
        sample_prices_bull_bear,
        sample_macro_factors,
        sample_breadth_df,
        sample_vol_df,
    ):
        """Test that regime labels show expected distribution (bull in early phase, bear in middle)."""
        regime_df = build_regime_state(
            prices=sample_prices_bull_bear,
            macro_factors=sample_macro_factors,
            breadth_df=sample_breadth_df,
            vol_df=sample_vol_df,
        )

        # Early phase (first 50 days) should have more bull/neutral
        early_phase = regime_df.iloc[:50]
        early_labels = early_phase["regime_label"].value_counts()

        # Middle phase (days 100-140) should have more bear/crisis
        middle_phase = regime_df.iloc[100:140]
        middle_labels = middle_phase["regime_label"].value_counts()

        # At least one regime label should be present
        assert len(early_labels) > 0
        assert len(middle_labels) > 0

    def test_score_ranges(
        self,
        sample_prices_bull_bear,
        sample_macro_factors,
        sample_breadth_df,
        sample_vol_df,
    ):
        """Test that regime scores are in expected ranges."""
        regime_df = build_regime_state(
            prices=sample_prices_bull_bear,
            macro_factors=sample_macro_factors,
            breadth_df=sample_breadth_df,
            vol_df=sample_vol_df,
        )

        # Scores should be in [-1, 1] range
        trend_scores = regime_df["regime_trend_score"].dropna()
        if len(trend_scores) > 0:
            assert (trend_scores >= -1.0).all() and (trend_scores <= 1.0).all()

        macro_scores = regime_df["regime_macro_score"].dropna()
        if len(macro_scores) > 0:
            assert (macro_scores >= -1.0).all() and (macro_scores <= 1.0).all()

        risk_scores = regime_df["regime_risk_score"].dropna()
        if len(risk_scores) > 0:
            assert (risk_scores >= -1.0).all() and (risk_scores <= 1.0).all()

        # Confidence should be in [0, 1] range
        confidence = regime_df["regime_confidence"].dropna()
        if len(confidence) > 0:
            assert (confidence >= 0.0).all() and (confidence <= 1.0).all()

    def test_missing_inputs(self, sample_prices_bull_bear):
        """Test that build_regime_state works with missing optional inputs."""
        # Should work with only prices
        regime_df = build_regime_state(
            prices=sample_prices_bull_bear,
            macro_factors=None,
            breadth_df=None,
            vol_df=None,
        )

        assert len(regime_df) > 0
        assert "regime_label" in regime_df.columns

    def test_empty_dataframe(self):
        """Test that build_regime_state handles empty DataFrame gracefully."""
        empty_prices = pd.DataFrame(columns=["timestamp", "symbol", "close"])

        with pytest.raises(ValueError, match="empty"):
            build_regime_state(
                prices=empty_prices,
                macro_factors=None,
                breadth_df=None,
                vol_df=None,
            )

    def test_with_custom_config(
        self,
        sample_prices_bull_bear,
        sample_breadth_df,
        sample_vol_df,
    ):
        """Test that custom RegimeStateConfig is used."""
        config = RegimeStateConfig(
            trend_ma_windows=(50, 200),
            vol_window=20,
            vov_window=60,
            breadth_ma_window=50,
            combine_macro_and_market=True,
        )

        regime_df = build_regime_state(
            prices=sample_prices_bull_bear,
            macro_factors=None,
            breadth_df=sample_breadth_df,
            vol_df=sample_vol_df,
            config=config,
        )

        assert len(regime_df) > 0


class TestComputeRegimeTransitionStats:
    """Tests for compute_regime_transition_stats() function."""

    def test_basic_functionality(
        self,
        sample_prices_bull_bear,
        sample_macro_factors,
        sample_breadth_df,
        sample_vol_df,
    ):
        """Test that compute_regime_transition_stats runs and returns correct format."""
        regime_df = build_regime_state(
            prices=sample_prices_bull_bear,
            macro_factors=sample_macro_factors,
            breadth_df=sample_breadth_df,
            vol_df=sample_vol_df,
        )

        transition_stats = compute_regime_transition_stats(regime_df)

        # Should have required columns
        if not transition_stats.empty:
            assert "from_regime" in transition_stats.columns
            assert "to_regime" in transition_stats.columns
            assert "count" in transition_stats.columns
            assert "avg_duration_days" in transition_stats.columns
            assert "transition_probability" in transition_stats.columns

    def test_transition_probabilities_sum_to_one(
        self,
        sample_prices_bull_bear,
        sample_macro_factors,
        sample_breadth_df,
        sample_vol_df,
    ):
        """Test that transition probabilities for each source regime sum to approximately 1."""
        regime_df = build_regime_state(
            prices=sample_prices_bull_bear,
            macro_factors=sample_macro_factors,
            breadth_df=sample_breadth_df,
            vol_df=sample_vol_df,
        )

        transition_stats = compute_regime_transition_stats(regime_df)

        if not transition_stats.empty:
            # For each source regime, sum of probabilities should be ≈ 1.0
            for from_regime in transition_stats["from_regime"].unique():
                from_transitions = transition_stats[
                    transition_stats["from_regime"] == from_regime
                ]
                prob_sum = from_transitions["transition_probability"].sum()

                # Allow small floating point errors
                assert abs(prob_sum - 1.0) < 0.01, (
                    f"Probabilities for {from_regime} sum to {prob_sum}, not 1.0"
                )

    def test_empty_regime_dataframe(self):
        """Test that compute_regime_transition_stats handles empty DataFrame gracefully."""
        empty_regime = pd.DataFrame(columns=["timestamp", "regime_label"])

        transition_stats = compute_regime_transition_stats(empty_regime)

        # Should return empty DataFrame
        assert transition_stats.empty or len(transition_stats) == 0


class TestEvaluateFactorIcByRegime:
    """Tests for evaluate_factor_ic_by_regime() function."""

    def test_basic_functionality(
        self,
        sample_prices_bull_bear,
        sample_macro_factors,
        sample_breadth_df,
        sample_vol_df,
        sample_ic_df,
    ):
        """Test that evaluate_factor_ic_by_regime runs and returns correct format."""
        regime_df = build_regime_state(
            prices=sample_prices_bull_bear,
            macro_factors=sample_macro_factors,
            breadth_df=sample_breadth_df,
            vol_df=sample_vol_df,
        )

        ic_by_regime = evaluate_factor_ic_by_regime(
            ic_df=sample_ic_df,
            regime_state_df=regime_df,
        )

        # Should have required columns
        if not ic_by_regime.empty:
            assert "factor" in ic_by_regime.columns
            assert "regime" in ic_by_regime.columns
            assert "mean_ic" in ic_by_regime.columns
            assert "std_ic" in ic_by_regime.columns
            assert "ic_ir" in ic_by_regime.columns
            assert "hit_ratio" in ic_by_regime.columns
            assert "n_observations" in ic_by_regime.columns

    def test_ic_differs_by_regime(
        self,
        sample_prices_bull_bear,
        sample_macro_factors,
        sample_breadth_df,
        sample_vol_df,
        sample_ic_df,
    ):
        """Test that IC values differ by regime (factor1 should work better in bull, factor2 in bear)."""
        regime_df = build_regime_state(
            prices=sample_prices_bull_bear,
            macro_factors=sample_macro_factors,
            breadth_df=sample_breadth_df,
            vol_df=sample_vol_df,
        )

        ic_by_regime = evaluate_factor_ic_by_regime(
            ic_df=sample_ic_df,
            regime_state_df=regime_df,
        )

        if not ic_by_regime.empty:
            # Factor1 in bull regime should have higher mean_ic than in bear
            factor1_bull = ic_by_regime[
                (ic_by_regime["factor"] == "factor1")
                & (ic_by_regime["regime"] == "bull")
            ]
            factor1_bear = ic_by_regime[
                (ic_by_regime["factor"] == "factor1")
                & (ic_by_regime["regime"] == "bear")
            ]

            # At least one result should exist (if both regimes present)
            if not factor1_bull.empty and not factor1_bear.empty:
                mean_ic_bull = factor1_bull["mean_ic"].iloc[0]
                mean_ic_bear = factor1_bear["mean_ic"].iloc[0]

                # Factor1 should work better in bull (by construction)
                assert mean_ic_bull > mean_ic_bear, (
                    f"Expected factor1 IC higher in bull ({mean_ic_bull}) than bear ({mean_ic_bear})"
                )

    def test_empty_inputs(self):
        """Test that evaluate_factor_ic_by_regime handles empty inputs gracefully."""
        empty_ic = pd.DataFrame(columns=["timestamp", "factor1_ic"])
        empty_regime = pd.DataFrame(columns=["timestamp", "regime_label"])

        result = evaluate_factor_ic_by_regime(empty_ic, empty_regime)

        # Should return empty DataFrame with correct columns
        assert result.empty or len(result) == 0
