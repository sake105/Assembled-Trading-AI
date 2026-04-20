"""Comprehensive tests for news_classifier.py (Batch 7).

All tests are marked @phase12.
"""

from __future__ import annotations

import pytest

from src.assembled_core.intel.news_classifier import (
    NewsClassification,
    classify_news_event,
    classify_batch,
    SECTOR_TO_ETFS,
    COUNTRY_TO_ASSETS,
)


@pytest.mark.phase12
class TestNewsClassifier:
    """Rule-based news classifier tests."""

    def test_sanctions_detected(self):
        result = classify_news_event("US imposes sanctions on Russia over Ukraine invasion")
        assert "sanctions" in result.event_types
        assert result.market_direction == "bearish"
        assert result.severity >= 4.0

    def test_bullish_deal_detected(self):
        result = classify_news_event("Major trade deal agreement signed between US and EU")
        assert result.market_direction in ("bullish", "mixed")
        assert "diplomatic" in result.event_types or "trade_policy" in result.event_types

    def test_military_strike_event_type(self):
        result = classify_news_event("Israeli airstrike hits targets in Gaza amid escalation")
        assert "military_strike" in result.event_types
        assert result.market_direction == "bearish"

    def test_severity_nuclear_is_max(self):
        result = classify_news_event("Nuclear threat raised after missile test")
        # nuclear keyword → max severity before tier discount
        # T2 tier: 10.0 * 0.7 = 7.0
        assert result.severity >= 6.0

    def test_severity_default_low(self):
        """Headlines with no severity keywords should score low."""
        result = classify_news_event("Company announces new product line")
        assert result.severity <= 3.0

    def test_affected_sectors_energy(self):
        result = classify_news_event("Pipeline explosion disrupts natural gas supply to Europe")
        assert "energy" in result.affected_sectors

    def test_affected_assets_from_country(self):
        result = classify_news_event("Russia-Ukraine war escalation threatens energy supply", geo_tags=["RU"])
        # RU geo tag should add RSXJ / OIL
        country_assets = COUNTRY_TO_ASSETS.get("RU", [])
        assert any(a in result.affected_assets for a in country_assets)

    def test_time_horizon_intraday_for_breaking(self):
        result = classify_news_event("BREAKING: Missile strike on capital city reported")
        assert result.time_horizon == "intraday"

    def test_time_horizon_medium_for_tariff(self):
        result = classify_news_event("New tariff rate decision expected next month")
        assert result.time_horizon in ("medium", "short")

    def test_multi_label(self):
        """Sanctions + war_escalation can co-occur."""
        result = classify_news_event("Sanctions imposed after military invasion of neighboring country")
        assert "sanctions" in result.event_types
        assert "war_escalation" in result.event_types or "military_strike" in result.event_types

    def test_empty_title(self):
        result = classify_news_event("")
        assert result.event_types == []
        assert result.severity == 0.0
        assert result.market_direction == "neutral"
        assert result.confidence == 0.0

    def test_t3_severity_discount(self):
        result_t2 = classify_news_event("War escalation threatens global stability", source_tier="T2")
        result_t3 = classify_news_event("War escalation threatens global stability", source_tier="T3")
        assert result_t3.severity < result_t2.severity

    def test_t0_severity_no_discount(self):
        result_t0 = classify_news_event("Sanctions imposed on major country", source_tier="T0")
        result_t2 = classify_news_event("Sanctions imposed on major country", source_tier="T2")
        assert result_t0.severity >= result_t2.severity

    def test_market_direction_mixed(self):
        result = classify_news_event("Trade deal signed but war escalation crashes markets")
        assert result.market_direction == "mixed"

    def test_market_direction_neutral(self):
        result = classify_news_event("Company holds annual shareholder meeting")
        assert result.market_direction == "neutral"

    def test_central_bank_financials_sector(self):
        result = classify_news_event("Federal Reserve raises interest rate by 25 basis points")
        assert "central_bank" in result.event_types
        assert "financials" in result.affected_sectors

    def test_ma_activity_bullish(self):
        result = classify_news_event("Microsoft acquires gaming company in major takeover deal")
        assert "ma_activity" in result.event_types
        assert result.market_direction in ("bullish", "mixed")

    def test_cyber_attack_tech_sector(self):
        result = classify_news_event("Major ransomware attack compromises semiconductor chip manufacturer")
        assert "cyber_attack" in result.event_types
        assert "tech" in result.affected_sectors

    def test_affected_assets_from_sectors(self):
        result = classify_news_event("Oil pipeline explosion disrupts energy supply")
        assert "energy" in result.affected_sectors
        # Energy sector should have ETFs
        energy_etfs = SECTOR_TO_ETFS.get("energy", [])
        assert any(a in result.affected_assets for a in energy_etfs)

    def test_natural_disaster_detected(self):
        result = classify_news_event("Massive earthquake strikes coast causing widespread flooding")
        assert "natural_disaster" in result.event_types

    def test_earnings_detected(self):
        result = classify_news_event("Apple reports record quarterly earnings beat expectations")
        assert "earnings" in result.event_types

    def test_confidence_range(self):
        """Confidence should always be between 0 and 1."""
        titles = [
            "",
            "nuclear war",
            "company reports results",
            "Fed raises rates",
            "coup overthrows government",
        ]
        for title in titles:
            result = classify_news_event(title)
            assert 0.0 <= result.confidence <= 1.0, f"Confidence out of range for: {title!r}"

    def test_classify_batch(self):
        titles = [
            "US imposes sanctions on Russia",
            "Stock market crashes amid war fears",
        ]
        results = classify_batch(titles, source_tier="T1")
        assert len(results) == 2
        assert all(isinstance(r, NewsClassification) for r in results)

    def test_geo_tags_none_safe(self):
        """geo_tags=None should not raise."""
        result = classify_news_event("Generic headline", geo_tags=None)
        assert isinstance(result, NewsClassification)

    def test_t1_confidence_higher_than_t3(self):
        """T1 sources should produce higher confidence than T3."""
        title = "Central bank rate decision announced"
        r_t1 = classify_news_event(title, source_tier="T1")
        r_t3 = classify_news_event(title, source_tier="T3")
        assert r_t1.confidence > r_t3.confidence

    def test_affected_assets_include_provided_tickers(self):
        """Provided tickers should be in affected_assets."""
        result = classify_news_event("Generic headline", tickers=["AAPL", "MSFT"])
        assert "AAPL" in result.affected_assets
        assert "MSFT" in result.affected_assets

    def test_market_stress_bearish(self):
        result = classify_news_event("Stock market plunge triggers circuit breaker")
        assert "market_stress" in result.event_types
        assert result.market_direction == "bearish"

    def test_diplomatic_event_type(self):
        result = classify_news_event("Peace talks summit leads to ceasefire agreement")
        assert "diplomatic" in result.event_types
        assert result.market_direction in ("bullish", "mixed")

    def test_regulatory_event_type(self):
        result = classify_news_event("SEC launches antitrust investigation into tech giant")
        assert "regulatory" in result.event_types

    def test_affected_assets_deduped(self):
        """affected_assets should have no duplicates."""
        result = classify_news_event(
            "Oil pipeline attack sanctions energy crisis",
            geo_tags=["IR"],
        )
        assert len(result.affected_assets) == len(set(result.affected_assets))


@pytest.mark.phase12
class TestSectorAndCountryMaps:
    """Tests for SECTOR_TO_ETFS and COUNTRY_TO_ASSETS lookups."""

    def test_sector_to_etfs_energy(self):
        assert "XLE" in SECTOR_TO_ETFS["energy"]
        assert "USO" in SECTOR_TO_ETFS["energy"]

    def test_sector_to_etfs_defense(self):
        assert "ITA" in SECTOR_TO_ETFS["defense"]
        assert "LMT" in SECTOR_TO_ETFS["defense"]

    def test_country_to_assets_russia(self):
        assert "RSXJ" in COUNTRY_TO_ASSETS["RU"]

    def test_country_to_assets_china(self):
        assert "FXI" in COUNTRY_TO_ASSETS["CN"]

    def test_country_to_assets_ukraine(self):
        assert "WEAT" in COUNTRY_TO_ASSETS["UA"]
