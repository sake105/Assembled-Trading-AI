"""Tests for src/assembled_core/signals/news_fusion.py (3-layer News-TA Fusion)."""

from __future__ import annotations


from src.assembled_core.signals.news_fusion import (
    news_z_score,
    news_score_normalized,
    size_from_meta,
    news_veto,
    bayesian_update,
    agreement_multiplier,
    decide_trade,
)

_NEUTRAL_FEATURES = {
    "sentiment_vw": 0.0,
    "novelty": 0.0,
    "surprise": 0.0,
    "event_volume_z": 0.0,
    "velocity": 0.0,
    "dispersion": 0.0,
}


# ---------------------------------------------------------------------------
# Layer 1: news_z_score
# ---------------------------------------------------------------------------


class TestNewsZScore:
    def test_neutral_features_return_zero(self):
        assert news_z_score(_NEUTRAL_FEATURES) == 0.0

    def test_clipped_at_positive_three(self):
        extreme = {k: 10.0 for k in _NEUTRAL_FEATURES}
        assert news_z_score(extreme) == 3.0

    def test_clipped_at_negative_three(self):
        extreme = {k: -10.0 for k in _NEUTRAL_FEATURES}
        assert news_z_score(extreme) == -3.0

    def test_positive_sentiment_positive_score(self):
        features = {**_NEUTRAL_FEATURES, "sentiment_vw": 1.0}
        assert news_z_score(features) > 0

    def test_dispersion_penalizes(self):
        without = news_z_score({**_NEUTRAL_FEATURES, "sentiment_vw": 1.0})
        with_disp = news_z_score(
            {**_NEUTRAL_FEATURES, "sentiment_vw": 1.0, "dispersion": 1.0}
        )
        assert with_disp < without

    def test_missing_features_default_to_zero(self):
        assert news_z_score({}) == 0.0


class TestNewsScoreNormalized:
    def test_range_within_minus_one_to_one(self):
        for v in [10.0, -10.0, 0.5, -0.5]:
            features = {k: v for k in _NEUTRAL_FEATURES}
            s = news_score_normalized(features)
            assert -1.0 <= s <= 1.0

    def test_neutral_is_zero(self):
        assert news_score_normalized(_NEUTRAL_FEATURES) == 0.0


# ---------------------------------------------------------------------------
# Layer 2: meta-labeling helpers
# ---------------------------------------------------------------------------


class TestSizeFromMeta:
    def test_below_theta_returns_zero(self):
        assert size_from_meta(0.40, theta_meta=0.55) == 0.0

    def test_at_theta_returns_zero(self):
        assert size_from_meta(0.55, theta_meta=0.55) == 0.0

    def test_high_confidence_returns_positive(self):
        assert size_from_meta(0.90, theta_meta=0.55) > 0.0

    def test_max_confidence_caps_at_one(self):
        assert size_from_meta(1.0, theta_meta=0.0) == 1.0

    def test_output_in_zero_one_range(self):
        for p in [0.0, 0.3, 0.6, 0.8, 1.0]:
            s = size_from_meta(p)
            assert 0.0 <= s <= 1.0


class TestNewsVeto:
    def test_agreeing_signal_no_veto(self):
        assert not news_veto(news_z=2.0, primary_side=1.0)

    def test_weak_contradicting_no_veto(self):
        assert not news_veto(news_z=-1.0, primary_side=1.0, tau_veto=1.5)

    def test_strong_contradicting_veto(self):
        assert news_veto(news_z=-2.0, primary_side=1.0, tau_veto=1.5)

    def test_zero_side_no_veto(self):
        assert not news_veto(news_z=-3.0, primary_side=0.0)

    def test_agreeing_short_no_veto(self):
        assert not news_veto(news_z=-2.0, primary_side=-1.0)


# ---------------------------------------------------------------------------
# Layer 3: 2D Decision Matrix
# ---------------------------------------------------------------------------


class TestBayesianUpdate:
    def test_output_in_zero_one(self):
        for ta in [-1.0, 0.0, 1.0]:
            for nz in [-3.0, 0.0, 3.0]:
                p = bayesian_update(ta, nz)
                assert 0.0 < p < 1.0

    def test_bull_ta_bull_news_above_half(self):
        p = bayesian_update(ta_score=0.8, news_z=2.0)
        assert p > 0.5

    def test_bear_ta_bear_news_below_half(self):
        p = bayesian_update(ta_score=-0.8, news_z=-2.0)
        assert p < 0.5

    def test_neutral_ta_neutral_news_near_half(self):
        p = bayesian_update(ta_score=0.0, news_z=0.0)
        assert 0.45 < p < 0.55


class TestAgreementMultiplier:
    def test_agreement_above_one(self):
        m = agreement_multiplier(ta_score=0.8, news_z=2.0)
        assert m > 1.0

    def test_strong_conflict_returns_half(self):
        m = agreement_multiplier(ta_score=0.8, news_z=-2.0)
        assert m == 0.5

    def test_weak_conflict_neutral(self):
        m = agreement_multiplier(ta_score=0.2, news_z=-0.2)
        assert m == 1.0

    def test_output_bounds(self):
        for ta in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            for nz in [-3.0, -1.0, 0.0, 1.0, 3.0]:
                m = agreement_multiplier(ta, nz)
                assert 0.5 <= m <= 1.5


# ---------------------------------------------------------------------------
# Unified decide_trade
# ---------------------------------------------------------------------------


class TestDecideTrade:
    def _good_features(self) -> dict:
        return {**_NEUTRAL_FEATURES, "sentiment_vw": 1.0, "novelty": 0.5}

    def test_skip_when_meta_below_theta(self):
        result = decide_trade(
            composite_score=0.7,
            news_features=self._good_features(),
            meta_probability=0.40,
        )
        assert result["action"] == "skip"
        assert result["size"] == 0.0
        assert result["reason"] == "meta_below_threshold"

    def test_skip_on_news_veto(self):
        # sentiment_vw=-3.0 → news_z = 0.30*(-3.0) = -0.9; use tau_veto=0.5
        result = decide_trade(
            composite_score=0.7,
            news_features={**_NEUTRAL_FEATURES, "sentiment_vw": -3.0},
            meta_probability=0.85,
            tau_veto=0.5,
        )
        assert result["action"] == "skip"
        assert result["reason"] == "news_veto"

    def test_long_on_bullish_signal(self):
        result = decide_trade(
            composite_score=0.8,
            news_features=self._good_features(),
            meta_probability=0.80,
        )
        assert result["action"] == "long"
        assert result["size"] > 0.0

    def test_short_on_bearish_signal(self):
        result = decide_trade(
            composite_score=-0.8,
            news_features={**_NEUTRAL_FEATURES, "sentiment_vw": -1.0},
            meta_probability=0.80,
        )
        assert result["action"] == "short"
        assert result["size"] > 0.0

    def test_size_in_zero_one(self):
        result = decide_trade(
            composite_score=0.9,
            news_features=self._good_features(),
            meta_probability=0.95,
        )
        assert 0.0 <= result["size"] <= 1.0

    def test_result_has_all_keys(self):
        result = decide_trade(
            composite_score=0.5,
            news_features=self._good_features(),
            meta_probability=0.70,
        )
        for key in (
            "action",
            "size",
            "composite_score",
            "news_z",
            "p_meta",
            "multiplier",
            "reason",
        ):
            assert key in result

    def test_sector_headwind_reduces_size(self):
        result_no_headwind = decide_trade(
            composite_score=0.8,
            news_features=self._good_features(),
            meta_probability=0.80,
            sector_sentiment=0.0,
        )
        result_headwind = decide_trade(
            composite_score=0.8,
            news_features=self._good_features(),
            meta_probability=0.80,
            sector_sentiment=-0.9,
        )
        assert result_headwind["size"] < result_no_headwind["size"]
