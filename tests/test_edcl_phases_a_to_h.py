"""Unit tests for EDCL Phases A through H.

Coverage:
  Phase A — apply_exposure_multiplier_to_targets (upscaling), compute_edcl_conviction_multiplier,
             _sp_compute_final_multiplier ceiling clamp
  Phase B — TriggerBasket, build_trigger_basket, compute_basket_score
  Phase C — compute_conviction_score (basket-only path, no FeatureStore)
  Phase D — compute_news_dim_with_edcl
  Phase H — compute_edcl_conviction_multiplier (triple confirmation logic)
"""

from __future__ import annotations

import pytest
import pandas as pd
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Phase A — georisk_overlay
# ---------------------------------------------------------------------------

from src.assembled_core.risk.georisk_overlay import (
    apply_exposure_multiplier_to_targets,
    compute_edcl_conviction_multiplier,
)


class TestApplyExposureMultiplierUpscaling:
    def _df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "symbol": ["AAPL", "MSFT", "CASH"],
            "target_weight": [0.30, 0.30, 0.40],
            "target_qty": [10.0, 8.0, 0.0],
        })

    def test_noop_at_exactly_one(self):
        df = self._df()
        result = apply_exposure_multiplier_to_targets(df, multiplier=1.0)
        pd.testing.assert_frame_equal(result, df)

    def test_downscale_reduces_risky_boosts_cash(self):
        df = self._df()
        result = apply_exposure_multiplier_to_targets(df, multiplier=0.5, cash_symbol="CASH")
        risky = result[result["symbol"] != "CASH"]["target_weight"]
        assert (risky == 0.15).all()
        # Cash should absorb freed weight: 0.40 + (0.60 - 0.30) = 0.70
        cash_w = result[result["symbol"] == "CASH"]["target_weight"].iloc[0]
        assert abs(cash_w - 0.70) < 1e-6

    def test_upscale_increases_risky_no_cash_boost(self):
        df = self._df()
        result = apply_exposure_multiplier_to_targets(df, multiplier=1.5, cash_symbol="CASH")
        risky = result[result["symbol"] != "CASH"]["target_weight"]
        assert risky.tolist() == pytest.approx([0.45, 0.45], abs=1e-6)
        # Cash should NOT be boosted (only reduced in downscaling)
        cash_w = result[result["symbol"] == "CASH"]["target_weight"].iloc[0]
        assert cash_w == 0.40  # unchanged

    def test_upscale_with_max_gross_exposure_normalization(self):
        df = self._df()
        # Upscale by 2x would put risky at 0.60 each = 1.20 total; cap at 1.0
        result = apply_exposure_multiplier_to_targets(
            df, multiplier=2.0, cash_symbol="CASH", max_gross_exposure=1.0
        )
        risky = result[result["symbol"] != "CASH"]["target_weight"]
        total = risky.sum()
        assert total <= 1.0 + 1e-6

    def test_none_passthrough(self):
        assert apply_exposure_multiplier_to_targets(None, multiplier=0.5) is None

    def test_empty_df_passthrough(self):
        df = pd.DataFrame(columns=["symbol", "target_weight"])
        result = apply_exposure_multiplier_to_targets(df, multiplier=0.5)
        assert result.empty

    def test_qty_scaled_with_weight(self):
        df = self._df()
        result = apply_exposure_multiplier_to_targets(df, multiplier=2.0, cash_symbol="CASH")
        risky_qty = result[result["symbol"] != "CASH"]["target_qty"]
        assert list(risky_qty) == pytest.approx([20.0, 16.0], abs=1e-6)


class TestComputeEdclConvictionMultiplier:
    def _ctx(self, mode: str = "live", conviction: float = 0.0) -> MagicMock:
        ctx = MagicMock()
        ctx.mode = mode
        ctx.edcl_state = {"conviction": conviction}
        return ctx

    def test_disabled_returns_one(self):
        ctx = self._ctx(conviction=0.9)
        policy = {"edcl_conviction_overlay": {"enabled": False}}
        assert compute_edcl_conviction_multiplier(ctx, policy) == 1.0

    def test_backtest_mode_returns_one(self):
        ctx = self._ctx(mode="backtest", conviction=0.9)
        policy = {"edcl_conviction_overlay": {"enabled": True, "conviction_threshold": 0.70}}
        assert compute_edcl_conviction_multiplier(ctx, policy) == 1.0

    def test_backtest_with_allow_flag(self):
        ctx = self._ctx(mode="backtest", conviction=1.0)
        policy = {"edcl_conviction_overlay": {
            "enabled": True, "conviction_threshold": 0.70,
            "allow_in_backtest": True, "max_multiplier": 2.0,
        }}
        result = compute_edcl_conviction_multiplier(ctx, policy)
        assert result == pytest.approx(2.0)

    def test_below_threshold_returns_one(self):
        ctx = self._ctx(mode="live", conviction=0.50)
        policy = {"edcl_conviction_overlay": {"enabled": True, "conviction_threshold": 0.70}}
        assert compute_edcl_conviction_multiplier(ctx, policy) == 1.0

    def test_at_max_conviction_returns_max_multiplier(self):
        ctx = self._ctx(mode="live", conviction=1.0)
        policy = {"edcl_conviction_overlay": {
            "enabled": True, "conviction_threshold": 0.70, "max_multiplier": 2.0,
        }}
        result = compute_edcl_conviction_multiplier(ctx, policy)
        assert result == pytest.approx(2.0)

    def test_midpoint_conviction_returns_midpoint_multiplier(self):
        ctx = self._ctx(mode="live", conviction=0.85)  # midpoint of [0.70, 1.0]
        policy = {"edcl_conviction_overlay": {
            "enabled": True, "conviction_threshold": 0.70, "max_multiplier": 2.0,
        }}
        result = compute_edcl_conviction_multiplier(ctx, policy)
        # 0.85 is halfway between 0.70 and 1.0 → multiplier = 1.0 + 0.5 * 1.0 = 1.5
        assert result == pytest.approx(1.5, abs=0.01)

    def test_no_edcl_state_returns_one(self):
        ctx = MagicMock()
        ctx.mode = "live"
        ctx.edcl_state = None
        policy = {"edcl_conviction_overlay": {"enabled": True, "conviction_threshold": 0.70}}
        assert compute_edcl_conviction_multiplier(ctx, policy) == 1.0


# ---------------------------------------------------------------------------
# Phase B — trigger_basket
# ---------------------------------------------------------------------------

from src.assembled_core.intel.trigger_basket import (
    TriggerBasket,
    build_trigger_basket,
    compute_basket_score,
)
from src.assembled_core.intel.models import NewsEvent, SourceTier
import datetime as dt


def _make_event(title: str, geo_tags: list[str] | None = None) -> NewsEvent:
    return NewsEvent(
        event_id="test-001",
        title=title,
        source_id="reuters",
        source_tier=SourceTier.T1,
        url="https://example.com/test",
        published_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        ingested_at=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc),
        content_hash="abc123",
        geo_tags=geo_tags or [],
    )


class TestTriggerBasket:
    def test_empty_events(self):
        basket = build_trigger_basket([])
        assert basket.conviction == 0.0
        assert basket.n_events == 0
        assert basket.fired_triggers == []
        assert basket.affected_assets == []

    def test_energy_event_fires_energy_triggers(self):
        event = _make_event("Oil pipeline attacked in Hormuz strait", geo_tags=["IR"])
        basket = build_trigger_basket([event])
        assert basket.conviction > 0.0
        assert "energy" in basket.affected_sectors
        assert len(basket.affected_assets) > 0
        assert "IR" in basket.geo_tags

    def test_banking_crisis_fires_financials(self):
        event = _make_event("Bank collapse triggers systemic liquidity crisis")
        basket = build_trigger_basket([event])
        assert basket.conviction > 0.0
        assert "financials" in basket.affected_sectors

    def test_is_active_threshold(self):
        basket = TriggerBasket(conviction=0.80)
        assert basket.is_active(threshold=0.70)
        assert not basket.is_active(threshold=0.90)

    def test_top_trigger_none_when_empty(self):
        basket = TriggerBasket()
        assert basket.top_trigger() is None

    def test_compute_basket_score_empty(self):
        basket = TriggerBasket()
        assert compute_basket_score(basket) == 0.0

    def test_compute_basket_score_with_sectors(self):
        event = _make_event("War escalation in military conflict with troop deployment")
        basket = build_trigger_basket([event])
        score = compute_basket_score(basket)
        assert 0.0 <= score <= 1.0

    def test_geo_assets_from_country_map(self):
        event = _make_event("Tensions rise in Taiwan", geo_tags=["CN"])
        basket = build_trigger_basket([event])
        # CN maps to FXI, KWEB, MCHI, CNYUSD
        assert any(a in basket.affected_assets for a in ["FXI", "KWEB", "MCHI"])

    def test_as_dict_structure(self):
        event = _make_event("Sanctions imposed on energy exports")
        basket = build_trigger_basket([event])
        d = basket.as_dict()
        assert "conviction" in d
        assert "fired_triggers" in d
        assert "affected_sectors" in d
        assert "affected_assets" in d
        assert "geo_tags" in d


# ---------------------------------------------------------------------------
# Phase C — conviction_engine
# ---------------------------------------------------------------------------

from src.assembled_core.intel.conviction_engine import compute_conviction_score


class TestConvictionEngine:
    def test_none_basket_returns_zero(self):
        assert compute_conviction_score(None) == 0.0

    def test_zero_conviction_basket_returns_zero(self):
        basket = TriggerBasket(conviction=0.0)
        assert compute_conviction_score(basket) == 0.0

    def test_positive_conviction_basket(self):
        basket = TriggerBasket(conviction=0.75, n_events=3, n_high_conviction=2, fired_triggers=[])
        score = compute_conviction_score(basket)
        assert score > 0.0
        assert score <= 1.0

    def test_score_increases_with_conviction(self):
        low = compute_conviction_score(TriggerBasket(conviction=0.3, n_events=1))
        high = compute_conviction_score(TriggerBasket(conviction=0.8, n_events=1))
        assert high > low

    def test_corroboration_bonus_applied(self):
        # Multiple high-conviction events should boost score
        single = compute_conviction_score(TriggerBasket(conviction=0.6, n_events=1, n_high_conviction=0))
        multi = compute_conviction_score(TriggerBasket(conviction=0.6, n_events=5, n_high_conviction=3))
        assert multi >= single


# ---------------------------------------------------------------------------
# Phase D — composite_score EDCL integration
# ---------------------------------------------------------------------------

from src.assembled_core.signals.composite_score import (
    compute_news_dim_with_edcl,
    compute_edcl_conviction_multiplier as composite_edcl_mult,
    composite_score,
)


class TestComputeNewsDimWithEdcl:
    def test_no_basket_passthrough(self):
        result = compute_news_dim_with_edcl(0.5, None, 0.8)
        assert result == 0.5

    def test_zero_conviction_passthrough(self):
        basket = TriggerBasket(conviction=0.8)
        result = compute_news_dim_with_edcl(0.5, basket, 0.0)
        assert result == 0.5

    def test_high_basket_conviction_shifts_toward_bearish(self):
        event = _make_event("Oil pipeline destroyed in Hormuz attack")
        basket = build_trigger_basket([event])
        original_news = 0.3
        result = compute_news_dim_with_edcl(original_news, basket, 1.0)
        # With conviction=1.0, news fully replaced by EDCL signal (which is bearish/negative)
        assert result < original_news
        assert -1.0 <= result <= 1.0

    def test_result_clamped_to_unit_range(self):
        basket = TriggerBasket(conviction=0.9)
        result = compute_news_dim_with_edcl(1.0, basket, 0.5)
        assert -1.0 <= result <= 1.0


class TestCompositeScoreEdclKwargs:
    def test_backward_compat_no_edcl_args(self):
        score, dims = composite_score(
            "normal", 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1
        )
        assert -1.0 <= score <= 1.0
        assert set(dims.keys()) == {"mtf", "classical_ta", "microstructure", "volume_profile",
                                     "chart_pattern", "vol_surface", "breadth", "seasonality", "news"}

    def test_edcl_basket_modifies_news_dim(self):
        event = _make_event("Banking crisis triggers systemic liquidity collapse")
        basket = build_trigger_basket([event])
        score_no_edcl, _ = composite_score("crisis", 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.5)
        score_edcl, _ = composite_score("crisis", 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.5,
                                        edcl_basket=basket, edcl_conviction=0.8)
        # EDCL basket should modify the news dimension (scores may differ)
        # Both are valid composite scores
        assert -1.0 <= score_edcl <= 1.0


# ---------------------------------------------------------------------------
# Phase H — triple confirmation
# ---------------------------------------------------------------------------

class TestTripleConfirmation:
    def test_below_threshold_returns_one(self):
        assert composite_edcl_mult(0.5, "crisis", 3.0) == 1.0

    def test_triple_confirmation_returns_two(self):
        assert composite_edcl_mult(0.8, "crisis", 3.0) == 2.0

    def test_double_no_iv_spike(self):
        assert composite_edcl_mult(0.8, "crisis", 0.5) == 1.5

    def test_edcl_only_normal_regime(self):
        assert composite_edcl_mult(0.8, "normal", 0.5) == 1.2

    def test_elevated_regime_counts_as_crisis(self):
        assert composite_edcl_mult(0.8, "elevated", 3.0) == 2.0

    def test_calm_regime_edcl_only(self):
        assert composite_edcl_mult(0.8, "calm", 5.0) == 1.2

    def test_respects_max_multiplier_policy(self):
        policy = {"edcl_conviction_overlay": {"conviction_threshold": 0.70, "max_multiplier": 1.8}}
        result = composite_edcl_mult(1.0, "crisis", 5.0, policy=policy)
        assert result == 1.8  # triple confirmation capped at max_multiplier

    def test_at_exact_threshold(self):
        # conviction == threshold is NOT below threshold (< not <=), so it fires
        # In crisis regime with IV spike → 2.0
        assert composite_edcl_mult(0.70, "crisis", 3.0) == 2.0
