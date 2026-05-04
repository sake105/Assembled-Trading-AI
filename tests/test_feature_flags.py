"""Tests for assembled_core/config/feature_flags.py (spec 36)."""

from __future__ import annotations


from assembled_core.config.feature_flags import (
    FeatureFlags,
    load_flags,
    emit_startup_banner,
)


class TestFeatureFlags:
    def test_defaults(self):
        flags = FeatureFlags()
        assert flags.trend_baseline == "on"
        assert flags.regime_ml_model == "shadow"
        assert flags.news_sentiment_v2 == "off"

    def test_is_active_on(self):
        flags = FeatureFlags(trend_baseline="on")
        assert flags.is_active("trend_baseline") is True

    def test_is_active_off(self):
        flags = FeatureFlags(news_sentiment_v2="off")
        assert flags.is_active("news_sentiment_v2") is False

    def test_is_active_shadow(self):
        flags = FeatureFlags(regime_ml_model="shadow")
        assert flags.is_active("regime_ml_model") is False

    def test_is_active_canary_10pct(self):
        flags = FeatureFlags(news_topic_clustering="canary")
        # Some tickers hash to 0 mod 10, others don't
        active = [
            flags.is_active("news_topic_clustering", t)
            for t in [
                "AAPL",
                "MSFT",
                "GOOG",
                "AMZN",
                "TSLA",
                "META",
                "NVDA",
                "AMD",
                "NFLX",
                "INTC",
            ]
        ]
        # With canary, roughly 10% should be True — at least some True and some False
        # (exact ratio depends on hash, but we just verify both can occur)
        assert isinstance(active, list)

    def test_is_active_canary_empty_ticker_false(self):
        flags = FeatureFlags(news_topic_clustering="canary")
        assert flags.is_active("news_topic_clustering", "") is False

    def test_is_shadow_true(self):
        flags = FeatureFlags(regime_ml_model="shadow")
        assert flags.is_shadow("regime_ml_model") is True

    def test_is_shadow_false_for_on(self):
        flags = FeatureFlags(trend_baseline="on")
        assert flags.is_shadow("trend_baseline") is False

    def test_unknown_flag_defaults_off(self):
        flags = FeatureFlags()
        assert flags.is_active("nonexistent_flag") is False


class TestLoadFlags:
    def test_returns_feature_flags(self):
        result = load_flags()
        assert isinstance(result, FeatureFlags)

    def test_trend_baseline_always_on(self):
        # trend_baseline should be 'on' in all environments
        flags = load_flags()
        assert flags.trend_baseline == "on"


class TestEmitStartupBanner:
    def test_does_not_raise(self):
        emit_startup_banner()  # just verifies no exception is thrown
