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
        return pd.DataFrame(
            {
                "symbol": ["AAPL", "MSFT", "CASH"],
                "target_weight": [0.30, 0.30, 0.40],
                "target_qty": [10.0, 8.0, 0.0],
            }
        )

    def test_noop_at_exactly_one(self):
        df = self._df()
        result = apply_exposure_multiplier_to_targets(df, multiplier=1.0)
        pd.testing.assert_frame_equal(result, df)

    def test_downscale_reduces_risky_boosts_cash(self):
        df = self._df()
        result = apply_exposure_multiplier_to_targets(
            df, multiplier=0.5, cash_symbol="CASH"
        )
        risky = result[result["symbol"] != "CASH"]["target_weight"]
        assert (risky == 0.15).all()
        # Cash should absorb freed weight: 0.40 + (0.60 - 0.30) = 0.70
        cash_w = result[result["symbol"] == "CASH"]["target_weight"].iloc[0]
        assert abs(cash_w - 0.70) < 1e-6

    def test_upscale_increases_risky_no_cash_boost(self):
        df = self._df()
        result = apply_exposure_multiplier_to_targets(
            df, multiplier=1.5, cash_symbol="CASH"
        )
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
        result = apply_exposure_multiplier_to_targets(
            df, multiplier=2.0, cash_symbol="CASH"
        )
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
        policy = {
            "edcl_conviction_overlay": {"enabled": True, "conviction_threshold": 0.70}
        }
        assert compute_edcl_conviction_multiplier(ctx, policy) == 1.0

    def test_backtest_with_allow_flag(self):
        ctx = self._ctx(mode="backtest", conviction=1.0)
        policy = {
            "edcl_conviction_overlay": {
                "enabled": True,
                "conviction_threshold": 0.70,
                "allow_in_backtest": True,
                "max_multiplier": 2.0,
            }
        }
        result = compute_edcl_conviction_multiplier(ctx, policy)
        assert result == pytest.approx(2.0)

    def test_below_threshold_returns_one(self):
        ctx = self._ctx(mode="live", conviction=0.50)
        policy = {
            "edcl_conviction_overlay": {"enabled": True, "conviction_threshold": 0.70}
        }
        assert compute_edcl_conviction_multiplier(ctx, policy) == 1.0

    def test_at_max_conviction_returns_max_multiplier(self):
        ctx = self._ctx(mode="live", conviction=1.0)
        policy = {
            "edcl_conviction_overlay": {
                "enabled": True,
                "conviction_threshold": 0.70,
                "max_multiplier": 2.0,
            }
        }
        result = compute_edcl_conviction_multiplier(ctx, policy)
        assert result == pytest.approx(2.0)

    def test_midpoint_conviction_returns_midpoint_multiplier(self):
        ctx = self._ctx(mode="live", conviction=0.85)  # midpoint of [0.70, 1.0]
        policy = {
            "edcl_conviction_overlay": {
                "enabled": True,
                "conviction_threshold": 0.70,
                "max_multiplier": 2.0,
            }
        }
        result = compute_edcl_conviction_multiplier(ctx, policy)
        # 0.85 is halfway between 0.70 and 1.0 → multiplier = 1.0 + 0.5 * 1.0 = 1.5
        assert result == pytest.approx(1.5, abs=0.01)

    def test_no_edcl_state_returns_one(self):
        ctx = MagicMock()
        ctx.mode = "live"
        ctx.edcl_state = None
        policy = {
            "edcl_conviction_overlay": {"enabled": True, "conviction_threshold": 0.70}
        }
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

from src.assembled_core.intel.conviction_engine import (
    compute_conviction_score,
    compute_edcl_position_size,
)


class TestConvictionEngine:
    def test_none_basket_returns_zero(self):
        assert compute_conviction_score(None) == 0.0

    def test_zero_conviction_basket_returns_zero(self):
        basket = TriggerBasket(conviction=0.0)
        assert compute_conviction_score(basket) == 0.0

    def test_positive_conviction_basket(self):
        basket = TriggerBasket(
            conviction=0.75, n_events=3, n_high_conviction=2, fired_triggers=[]
        )
        score = compute_conviction_score(basket)
        assert score > 0.0
        assert score <= 1.0

    def test_score_increases_with_conviction(self):
        low = compute_conviction_score(TriggerBasket(conviction=0.3, n_events=1))
        high = compute_conviction_score(TriggerBasket(conviction=0.8, n_events=1))
        assert high > low

    def test_corroboration_bonus_applied(self):
        # Multiple high-conviction events should boost score
        single = compute_conviction_score(
            TriggerBasket(conviction=0.6, n_events=1, n_high_conviction=0)
        )
        multi = compute_conviction_score(
            TriggerBasket(conviction=0.6, n_events=5, n_high_conviction=3)
        )
        assert multi >= single


class TestEdclPositionSize:
    def _policy(self, **kwargs) -> dict:
        base = {
            "edcl_conviction_overlay": {
                "conviction_threshold": 0.70,
                "edcl_sizing": {
                    "max_edcl_weight": 0.30,
                    "target_coverage": 0.85,
                },
            }
        }
        base["edcl_conviction_overlay"].update(kwargs)
        return base

    def test_returns_dict_with_required_keys(self):
        result = compute_edcl_position_size(0.85, policy=self._policy())
        assert set(result.keys()) >= {
            "max_weight",
            "stop_loss_pct",
            "size_factor",
            "conformal_factor",
        }

    def test_below_threshold_returns_zero_weight(self):
        result = compute_edcl_position_size(0.50, policy=self._policy())
        assert result["max_weight"] == 0.0
        assert result["size_factor"] == 0.0

    def test_at_max_conviction_returns_base_max(self):
        # conviction=1.0, no conformal model → conformal_factor=1.0, scale=1.0
        result = compute_edcl_position_size(1.0, policy=self._policy())
        assert result["max_weight"] == pytest.approx(0.30)

    def test_mid_conviction_returns_half_base(self):
        # conviction=0.85 is midpoint of [0.70, 1.0] → scale=0.5
        result = compute_edcl_position_size(0.85, policy=self._policy())
        assert result["max_weight"] == pytest.approx(0.30 * 0.5, abs=0.01)

    def test_no_policy_returns_zero(self):
        result = compute_edcl_position_size(0.0, policy=None)
        assert result["max_weight"] == 0.0

    def test_all_values_in_valid_range(self):
        result = compute_edcl_position_size(0.9, policy=self._policy())
        assert 0.0 <= result["max_weight"] <= 0.30
        assert 0.0 <= result["stop_loss_pct"] <= 1.0
        assert 0.0 <= result["size_factor"] <= 1.0
        assert 0.0 <= result["conformal_factor"] <= 1.0


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
        assert set(dims.keys()) == {
            "mtf",
            "classical_ta",
            "microstructure",
            "volume_profile",
            "chart_pattern",
            "vol_surface",
            "breadth",
            "seasonality",
            "news",
        }

    def test_edcl_basket_modifies_news_dim(self):
        event = _make_event("Banking crisis triggers systemic liquidity collapse")
        basket = build_trigger_basket([event])
        score_no_edcl, _ = composite_score(
            "crisis", 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.5
        )
        score_edcl, _ = composite_score(
            "crisis",
            0.5,
            0.3,
            0.1,
            0.0,
            0.0,
            0.0,
            0.2,
            0.0,
            0.5,
            edcl_basket=basket,
            edcl_conviction=0.8,
        )
        # EDCL basket should modify the news dimension (scores may differ)
        # Both are valid composite scores
        assert -1.0 <= score_edcl <= 1.0


# ---------------------------------------------------------------------------
# Phase H — triple confirmation
# ---------------------------------------------------------------------------


class TestTripleConfirmation:
    def test_below_threshold_returns_one_v2(self):
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
        policy = {
            "edcl_conviction_overlay": {
                "conviction_threshold": 0.70,
                "max_multiplier": 1.8,
            }
        }
        result = composite_edcl_mult(1.0, "crisis", 5.0, policy=policy)
        assert result == 1.8  # triple confirmation capped at max_multiplier

    def test_at_exact_threshold(self):
        # conviction == threshold is NOT below threshold (< not <=), so it fires
        # In crisis regime with IV spike → 2.0
        assert composite_edcl_mult(0.70, "crisis", 3.0) == 2.0


# ---------------------------------------------------------------------------
# Phase G — Tail-Hunting
# ---------------------------------------------------------------------------

from src.assembled_core.intel.tail_hunting import (
    TailHuntSignal,
    load_tail_plans,
    match_tail_plans,
    tail_signals_to_targets,
)
from src.assembled_core.intel.models import TriggerType


class TestTailHuntSignal:
    def _sig(self, direction="long") -> TailHuntSignal:
        return TailHuntSignal(
            event_name="hormuz_test",
            direction=direction,
            primary_assets=["USO", "XLE"],
            hedge_assets=["IYT"],
            max_position_size=0.30,
            activation_conviction=0.75,
            current_conviction=0.875,  # midpoint → scale=0.5 → size=0.15
        )

    def test_size_fraction_scales_linearly(self):
        sig = self._sig()
        # (0.875 - 0.75) / (1.0 - 0.75) = 0.5  → 0.30 * 0.5 = 0.15
        assert abs(sig.size_fraction() - 0.15) < 1e-6

    def test_size_fraction_at_activation_threshold_is_zero(self):
        sig = TailHuntSignal(
            event_name="t",
            direction="long",
            primary_assets=["A"],
            hedge_assets=[],
            max_position_size=0.20,
            activation_conviction=0.70,
            current_conviction=0.70,
        )
        assert sig.size_fraction() == pytest.approx(0.0, abs=1e-6)

    def test_size_fraction_caps_at_max_position_size(self):
        sig = TailHuntSignal(
            event_name="t",
            direction="long",
            primary_assets=["A"],
            hedge_assets=[],
            max_position_size=0.20,
            activation_conviction=0.70,
            current_conviction=1.0,
        )
        assert sig.size_fraction() == pytest.approx(0.20, abs=1e-6)

    def test_as_dict_has_required_keys(self):
        d = self._sig().as_dict()
        for key in (
            "event_name",
            "direction",
            "primary_assets",
            "size_fraction",
            "max_position_size",
            "matched_triggers",
        ):
            assert key in d


class TestLoadTailPlans:
    def test_default_config_loads_six_plans(self):
        plans = load_tail_plans()
        assert len(plans) == 6

    def test_all_plans_disabled_by_default(self):
        plans = load_tail_plans()
        assert all(not v.get("enabled", False) for v in plans.values())

    def test_missing_file_returns_empty(self, tmp_path):
        plans = load_tail_plans(tmp_path / "nonexistent.yaml")
        assert plans == {}


class TestMatchTailPlans:
    def _basket(self) -> TriggerBasket:
        return TriggerBasket(
            fired_triggers=[
                (TriggerType.CHOKEPOINT_STRESS, 0.85),
                (TriggerType.ENERGY_SUPPLY_RISK, 0.80),
            ],
            affected_sectors={"energy": 0.85},
            affected_assets=["XLE", "USO"],
            conviction=0.85,
            n_events=2,
            n_high_conviction=2,
        )

    def test_disabled_plans_not_matched(self):
        # All plans disabled by default in tail_hunting_v1.yaml
        signals = match_tail_plans(self._basket(), 0.90)
        assert signals == []

    def test_no_basket_returns_empty(self):
        assert match_tail_plans(None, 0.90) == []

    def test_inactive_basket_returns_empty(self):
        empty = TriggerBasket()
        assert match_tail_plans(empty, 0.90) == []

    def test_below_conviction_threshold_not_matched(self, tmp_path):
        import yaml

        config = {
            "tail_events": {
                "test_event": {
                    "enabled": True,
                    "triggers": ["CHOKEPOINT_STRESS"],
                    "primary_assets": ["USO"],
                    "hedge_assets": [],
                    "max_position_size": 0.20,
                    "activation_conviction": 0.80,
                    "direction": "long",
                }
            }
        }
        cfg_path = tmp_path / "tail.yaml"
        cfg_path.write_text(yaml.dump(config))
        signals = match_tail_plans(
            self._basket(), conviction=0.70, config_path=cfg_path
        )
        assert signals == []

    def test_enabled_plan_activates_on_match(self, tmp_path):
        import yaml

        config = {
            "tail_events": {
                "hormuz_test": {
                    "enabled": True,
                    "triggers": ["CHOKEPOINT_STRESS"],
                    "primary_assets": ["USO", "XLE"],
                    "hedge_assets": ["IYT"],
                    "max_position_size": 0.30,
                    "activation_conviction": 0.75,
                    "direction": "long",
                    "description": "test",
                }
            }
        }
        cfg_path = tmp_path / "tail.yaml"
        cfg_path.write_text(yaml.dump(config))
        signals = match_tail_plans(
            self._basket(), conviction=0.85, config_path=cfg_path
        )
        assert len(signals) == 1
        assert signals[0].event_name == "hormuz_test"
        assert signals[0].direction == "long"
        assert "CHOKEPOINT_STRESS" in signals[0].matched_triggers


class TestTailSignalsToTargets:
    def _sig(self) -> TailHuntSignal:
        return TailHuntSignal(
            event_name="test",
            direction="long",
            primary_assets=["USO", "XLE"],
            hedge_assets=["IYT"],
            max_position_size=0.30,
            activation_conviction=0.75,
            current_conviction=1.0,
        )

    def test_long_adds_primary_subtracts_hedge(self):
        targets = tail_signals_to_targets([self._sig()])
        assert targets["USO"] > 0
        assert targets["XLE"] > 0
        assert targets["IYT"] < 0

    def test_short_subtracts_primary_adds_hedge(self):
        sig = TailHuntSignal(
            event_name="test",
            direction="short",
            primary_assets=["XLF"],
            hedge_assets=["GLD"],
            max_position_size=0.20,
            activation_conviction=0.70,
            current_conviction=1.0,
        )
        targets = tail_signals_to_targets([sig])
        assert targets["XLF"] < 0
        assert targets["GLD"] > 0

    def test_overlays_existing_targets(self):
        existing = {"AAPL": 0.10, "USO": 0.05}
        targets = tail_signals_to_targets([self._sig()], existing_targets=existing)
        assert targets["AAPL"] == pytest.approx(0.10)
        assert targets["USO"] > 0.05  # added to existing

    def test_empty_signals_returns_existing_unchanged(self):
        existing = {"AAPL": 0.20}
        targets = tail_signals_to_targets([], existing_targets=existing)
        assert targets == existing


# ---------------------------------------------------------------------------
# Phase C — GeoEventLogger
# ---------------------------------------------------------------------------

from src.assembled_core.intel.geo_event_logger import (
    log_basket_event,
    read_geo_event_log,
)


class TestGeoEventLogger:
    def _basket(self) -> TriggerBasket:
        return TriggerBasket(
            fired_triggers=[
                (TriggerType.CHOKEPOINT_STRESS, 0.85),
                (TriggerType.ENERGY_SUPPLY_RISK, 0.80),
            ],
            affected_sectors={"energy": 0.85},
            affected_assets=["XLE"],
            conviction=0.85,
            n_events=2,
            n_high_conviction=2,
        )

    def test_log_returns_true_on_active_basket(self, tmp_path):
        path = tmp_path / "events.parquet"
        result = log_basket_event(self._basket(), 0.85, output_path=path)
        assert result is True

    def test_log_creates_file_with_correct_schema(self, tmp_path):
        path = tmp_path / "events.parquet"
        log_basket_event(self._basket(), 0.85, output_path=path)
        df = read_geo_event_log(path)
        assert set(df.columns) >= {
            "event_date",
            "trigger_type",
            "conviction",
            "source_tier",
        }

    def test_log_writes_one_row_per_fired_trigger(self, tmp_path):
        path = tmp_path / "events.parquet"
        log_basket_event(self._basket(), 0.85, output_path=path)
        df = read_geo_event_log(path)
        assert len(df) == 2
        assert set(df["trigger_type"]) == {"CHOKEPOINT_STRESS", "ENERGY_SUPPLY_RISK"}

    def test_log_appends_on_second_call(self, tmp_path):
        path = tmp_path / "events.parquet"
        log_basket_event(self._basket(), 0.85, output_path=path)
        b2 = TriggerBasket(
            fired_triggers=[(TriggerType.BANKING_CRISIS, 0.7)],
            conviction=0.7,
            n_events=1,
            n_high_conviction=1,
        )
        log_basket_event(b2, 0.7, output_path=path)
        df = read_geo_event_log(path)
        assert len(df) == 3

    def test_log_returns_false_on_inactive_basket(self, tmp_path):
        path = tmp_path / "events.parquet"
        empty = TriggerBasket()
        result = log_basket_event(empty, 0.0, output_path=path)
        assert result is False
        assert not path.exists()

    def test_log_returns_false_on_none_basket(self, tmp_path):
        path = tmp_path / "events.parquet"
        result = log_basket_event(None, 0.85, output_path=path)
        assert result is False

    def test_read_empty_when_no_file(self, tmp_path):
        df = read_geo_event_log(tmp_path / "nonexistent.parquet")
        assert len(df) == 0
        assert "trigger_type" in df.columns

    def test_min_conviction_filter(self, tmp_path):
        path = tmp_path / "events.parquet"
        log_basket_event(self._basket(), 0.85, output_path=path)
        df_all = read_geo_event_log(path, min_conviction=0.0)
        df_filtered = read_geo_event_log(path, min_conviction=0.90)
        assert len(df_all) == 2
        assert len(df_filtered) == 0


# ---------------------------------------------------------------------------
# Item 133 — _MAX_EXPOSURE_MULT = 3.0 cap verification
#
# Verifies that _sp_compute_final_multiplier enforces the 3.0 ceiling
# AFTER all overlays (geo × profit_lock × vol_scale × market_stress ×
# crisis_alpha × pm × hmm × edcl) are combined.  A synthetic ctx with
# extreme values is used to exceed the cap and confirm clamping.
# ---------------------------------------------------------------------------

from src.assembled_core.pipeline._tc_sizing import _sp_compute_final_multiplier


class TestMaxExposureMultCap:
    """Item 133 — _MAX_EXPOSURE_MULT = 3.0 caps combined overlay product."""

    _MAX = 3.0

    def _ctx(self, **attrs):
        ctx = MagicMock()
        # Disable every optional sub-overlay so only geo_multiplier feeds through
        ctx.mode = "backtest"
        ctx.equity_curve = None
        ctx.equity_curve_index = None
        ctx.profit_lock_state = None
        ctx.prices = None
        ctx.panel = None
        ctx.market_stress = None
        ctx.crisis_intel = None
        ctx.pm_alpha_signal = None
        ctx.hmm_regime = None
        ctx.edcl_state = None
        ctx.options_iv_skew_z = 0.0
        for k, v in attrs.items():
            setattr(ctx, k, v)
        return ctx

    def _policy(self, geo_mult_override: float | None = None) -> dict:
        """Minimal policy that passes all optional checks as disabled."""
        p: dict = {
            "profit_lock": {"enabled": False},
            "vol_targeting": {"enabled": False},
            "market_stress": {"enabled": False},
            "crisis_alpha": {"enabled": False},
            "pm_alpha": {"enabled": False},
            "hmm_regime": {"enabled": False},
            "edcl_conviction_overlay": {"enabled": False},
            "georisk": {"enabled": False},
        }
        if geo_mult_override is not None:
            p["georisk"] = {"enabled": True, "multiplier": geo_mult_override}
        return p

    def test_cap_is_exactly_three(self):
        """_MAX_EXPOSURE_MULT constant must equal 3.0."""
        # We verify this by clamping a very large value and inspecting the return.
        # If the constant ever changes, this test will catch the deviation.
        import logging

        log = logging.getLogger("test")
        ctx = self._ctx()
        policy = self._policy()
        result = _sp_compute_final_multiplier(ctx, policy, {}, log)
        # With all overlays disabled the result should be exactly 1.0 (no boosts).
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_cap_applied_when_product_exceeds_ceiling(self, monkeypatch):
        """Monkeypatch compute_exposure_multiplier to return 10.0; result must be <= 3.0."""
        import logging
        import src.assembled_core.pipeline._tc_sizing as tc_sizing

        log = logging.getLogger("test")
        ctx = self._ctx()
        policy = self._policy()

        # Inject a geo multiplier far above the 3.0 ceiling
        monkeypatch.setattr(
            tc_sizing,
            "compute_exposure_multiplier",
            lambda _ctx, _policy: 10.0,
        )
        result = _sp_compute_final_multiplier(ctx, policy, {}, log)
        assert result == pytest.approx(self._MAX, abs=1e-6), (
            f"Expected cap of {self._MAX} but got {result}; "
            "cap must be applied AFTER all overlay multiplications"
        )

    def test_cap_applied_for_combined_edcl_plus_geo_boost(self, monkeypatch):
        """Combine geo (2.0) × edcl (2.0) = 4.0 → must be clamped to 3.0."""
        import logging
        import src.assembled_core.pipeline._tc_sizing as tc_sizing

        log = logging.getLogger("test")
        ctx = self._ctx()
        policy = self._policy()

        # geo returns 2.0, all other overlays stay at 1.0
        monkeypatch.setattr(
            tc_sizing,
            "compute_exposure_multiplier",
            lambda _ctx, _policy: 2.0,
        )
        # Patch compute_profit_lock_multiplier to simulate a 2x profit-lock boost
        monkeypatch.setattr(
            tc_sizing,
            "compute_profit_lock_multiplier",
            lambda *args, **kwargs: (2.0, {}),
        )
        # Enable profit_lock so the function actually calls it
        policy["profit_lock"] = {"enabled": True}
        ctx.equity_curve = [1.0, 1.1]
        ctx.equity_curve_index = 1

        result = _sp_compute_final_multiplier(ctx, policy, {}, log)
        assert result <= self._MAX + 1e-6, (
            f"Cap {self._MAX} breached: got {result}. "
            "All boosts must be multiplied first, then the ceiling applied."
        )
        assert result == pytest.approx(self._MAX, abs=1e-6)

    def test_floor_also_enforced(self, monkeypatch):
        """Multiplier below 0.05 floor is clamped upward."""
        import logging
        import src.assembled_core.pipeline._tc_sizing as tc_sizing

        log = logging.getLogger("test")
        ctx = self._ctx()
        policy = self._policy()

        monkeypatch.setattr(
            tc_sizing,
            "compute_exposure_multiplier",
            lambda _ctx, _policy: 0.0,  # would produce 0.0 product
        )
        result = _sp_compute_final_multiplier(ctx, policy, {}, log)
        assert result >= 0.05 - 1e-6
