"""R2-7 / audit B2-01: the crisis_alpha & news_alpha overlays must compose with the
global exposure multiplier observably, not escape it silently.

``size_positions`` applies the global exposure multiplier
(geo × profit_lock × vol_targeting × market_stress × crisis × pm × hmm, clamp
[0.05, 3.0]) to the BASE book BEFORE ``_sp_apply_crisis_alpha_cap`` /
``_sp_apply_news_alpha`` append their entries — so before R2-7 those overlay entries
silently escaped the entire de-risk/leverage chain. R2-7 makes that explicit:

* default (``apply_global_derisk`` absent/false) — behaviour-preserving: weights are
  returned UNCHANGED, but the escape is recorded in ``meta['overlay_exposure']`` and
  logged at INFO (auditable, not silent);
* opt-in (``apply_global_derisk=true``) — the overlay weights are scaled by the same
  multiplier so the sub-portfolio composes with the system-wide risk appetite;
* ``meta=None`` / non-dict meta / missing multiplier → multiplier defaults to 1.0 →
  no-op, never raises.

These tests pin the unit helper plus the two integration call paths.
"""

from __future__ import annotations

import logging
import types

import pandas as pd
import pytest
from unittest.mock import patch

from src.assembled_core.pipeline._tc_sizing import (
    _apply_overlay_global_derisk,
    _sp_apply_crisis_alpha_cap,
    _sp_apply_news_alpha,
)

pytestmark = pytest.mark.fast

_LOG = logging.getLogger("test_r2_7")
_NEWS_PIPELINE = "src.assembled_core.events.news_alpha.pipeline.run_news_alpha_pipeline"
_CRISIS_PIPELINE = (
    "src.assembled_core.events.crisis_alpha.pipeline.run_crisis_alpha_pipeline"
)


def _make_target_positions(weights: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": sym, "target_weight": w, "target_qty": 0.0}
            for sym, w in weights.items()
        ]
    )


# --------------------------------------------------------------------------- #
# 1. Unit — _apply_overlay_global_derisk
# --------------------------------------------------------------------------- #


class TestApplyOverlayGlobalDerisk:
    def test_empty_weights_returns_unchanged_no_record(self) -> None:
        meta: dict = {}
        out = _apply_overlay_global_derisk({}, "news_alpha", {}, meta, _LOG)
        assert out == {}
        assert "overlay_exposure" not in meta

    def test_meta_none_is_noop_never_raises(self) -> None:
        weights = {"XLE": 0.08}
        out = _apply_overlay_global_derisk(weights, "news_alpha", {}, None, _LOG)
        # meta=None → multiplier defaults to 1.0 → weights returned unchanged
        assert out == {"XLE": 0.08}

    def test_multiplier_one_is_noop_no_record(self) -> None:
        meta = {"final_exposure_multiplier": 1.0}
        weights = {"XLE": 0.08}
        out = _apply_overlay_global_derisk(weights, "news_alpha", {}, meta, _LOG)
        assert out == {"XLE": 0.08}
        # exactly-1.0 multiplier means "no global scaling in effect" → stay silent
        assert "overlay_exposure" not in meta

    def test_flag_false_records_escape_but_does_not_scale(self, caplog) -> None:
        meta = {"final_exposure_multiplier": 0.5}
        weights = {"XLE": 0.08, "GLD": 0.06}
        with caplog.at_level(logging.INFO, logger="test_r2_7"):
            out = _apply_overlay_global_derisk(
                weights, "news_alpha", {"apply_global_derisk": False}, meta, _LOG
            )
        # behaviour-preserving: weights unchanged
        assert out == {"XLE": 0.08, "GLD": 0.06}
        # but the escape is recorded + logged (auditable, not silent)
        recs = meta["overlay_exposure"]
        assert len(recs) == 1
        assert recs[0] == {
            "overlay": "news_alpha",
            "global_multiplier": 0.5,
            "derisk_applied": False,
            "n_entries": 2,
        }
        assert any(
            "[R2-7]" in r.message or "[R2-7]" in str(r.args) for r in caplog.records
        )

    def test_flag_true_scales_and_records(self) -> None:
        meta = {"final_exposure_multiplier": 0.5}
        weights = {"GLD": 0.10, "TLT": 0.08}
        out = _apply_overlay_global_derisk(
            weights, "crisis_alpha", {"apply_global_derisk": True}, meta, _LOG
        )
        assert out["GLD"] == pytest.approx(0.05)
        assert out["TLT"] == pytest.approx(0.04)
        rec = meta["overlay_exposure"][0]
        assert rec["overlay"] == "crisis_alpha"
        assert rec["derisk_applied"] is True
        assert rec["global_multiplier"] == pytest.approx(0.5)
        assert rec["n_entries"] == 2

    def test_non_numeric_multiplier_defaults_to_one(self) -> None:
        meta = {"final_exposure_multiplier": "not-a-number"}
        weights = {"XLE": 0.08}
        out = _apply_overlay_global_derisk(
            weights, "news_alpha", {"apply_global_derisk": True}, meta, _LOG
        )
        # bad multiplier → 1.0 → no-op, no record, no raise
        assert out == {"XLE": 0.08}
        assert "overlay_exposure" not in meta

    def test_none_cfg_treated_as_flag_false(self) -> None:
        meta = {"final_exposure_multiplier": 0.5}
        out = _apply_overlay_global_derisk(
            {"XLE": 0.08}, "news_alpha", None, meta, _LOG
        )
        assert out == {"XLE": 0.08}
        assert meta["overlay_exposure"][0]["derisk_applied"] is False


# --------------------------------------------------------------------------- #
# 2. Integration — news_alpha
# --------------------------------------------------------------------------- #


def _make_news_ctx(capital: float = 100_000.0) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        meta={},
        as_of=pd.Timestamp("2026-05-26 12:00:00+00:00"),
        news_geo={},
        prices=None,
        capital=capital,
    )


def _make_na_result(target_weights: dict[str, float]) -> object:
    from src.assembled_core.events.news_alpha.models import NewsAlphaResult

    return NewsAlphaResult(
        timestamp_utc="2026-05-26T12:00:00+00:00",
        signals=[],
        target_weights=target_weights,
        positions_to_exit=[],
        shadow_only=False,
        errors=[],
    )


class TestNewsAlphaComposition:
    def test_exempt_by_default_records_escape(self) -> None:
        """Default (apply_global_derisk absent) with a 0.5 global multiplier: the news
        entry keeps its full 0.08 weight (exempt) and the escape is recorded."""
        ctx = _make_news_ctx()
        meta = {"final_exposure_multiplier": 0.5}
        policy = {
            "news_alpha": {"enabled": True, "shadow_only": False},
            "risk_limits": {"max_gross_exposure": 1.20},
        }
        target_positions = _make_target_positions({"AAPL": 0.20})

        with patch(_NEWS_PIPELINE, return_value=_make_na_result({"XLE": 0.08})):
            result = _sp_apply_news_alpha(
                target_positions, ctx, policy, _LOG, meta=meta
            )

        xle = result[result["symbol"] == "XLE"].iloc[0]
        assert xle["target_weight"] == pytest.approx(0.08), "exempt by default"
        assert xle["target_qty"] == pytest.approx(0.08 * 100_000.0)
        rec = [r for r in meta["overlay_exposure"] if r["overlay"] == "news_alpha"]
        assert rec and rec[0]["derisk_applied"] is False

    def test_derisked_when_flag_on(self) -> None:
        """apply_global_derisk=true with a 0.5 multiplier: news entry scaled 0.08→0.04."""
        ctx = _make_news_ctx()
        meta = {"final_exposure_multiplier": 0.5}
        policy = {
            "news_alpha": {
                "enabled": True,
                "shadow_only": False,
                "apply_global_derisk": True,
            },
            "risk_limits": {"max_gross_exposure": 1.20},
        }
        target_positions = _make_target_positions({"AAPL": 0.20})

        with patch(_NEWS_PIPELINE, return_value=_make_na_result({"XLE": 0.08})):
            result = _sp_apply_news_alpha(
                target_positions, ctx, policy, _LOG, meta=meta
            )

        xle = result[result["symbol"] == "XLE"].iloc[0]
        assert xle["target_weight"] == pytest.approx(0.04), "global de-risk folded in"
        assert xle["target_qty"] == pytest.approx(0.04 * 100_000.0)
        rec = [r for r in meta["overlay_exposure"] if r["overlay"] == "news_alpha"]
        assert rec and rec[0]["derisk_applied"] is True

    def test_no_meta_kwarg_is_behaviour_preserving(self) -> None:
        """Existing callers that don't pass meta → multiplier defaults to 1.0 → entry
        keeps its full weight; the new param is fully optional."""
        ctx = _make_news_ctx()
        policy = {
            "news_alpha": {"enabled": True, "shadow_only": False},
            "risk_limits": {"max_gross_exposure": 1.20},
        }
        target_positions = _make_target_positions({"AAPL": 0.20})

        with patch(_NEWS_PIPELINE, return_value=_make_na_result({"XLE": 0.08})):
            result = _sp_apply_news_alpha(target_positions, ctx, policy, _LOG)

        xle = result[result["symbol"] == "XLE"].iloc[0]
        assert xle["target_weight"] == pytest.approx(0.08)


# --------------------------------------------------------------------------- #
# 3. Integration — crisis_alpha
# --------------------------------------------------------------------------- #


def _make_crisis_ctx(capital: float = 100_000.0) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        meta={},
        as_of=pd.Timestamp("2026-03-30 12:00:00+00:00"),
        news_geo={},
        features=None,
        market_stress={"stress_ok": True, "stress_score": 0},
        intel_health_flags={},
        capital=capital,
    )


def _make_ca_result(target_weights: dict[str, float]) -> dict:
    return {
        "state": "ACTIVE",
        "target_weights": target_weights,
        "errors": [],
        "gate_reasons": [],
        "gates_ok": True,
        "entry_reasons": [],
        "positions_to_exit": [],
        "should_flatten_all": False,
        "flatten_reason": "",
    }


class TestCrisisAlphaComposition:
    def test_exempt_by_default(self) -> None:
        """Crisis hedge keeps its full weight under a 0.5 multiplier by default — a
        defensive hedge must not be de-risked away exactly when it is needed."""
        ctx = _make_crisis_ctx()
        meta = {"final_exposure_multiplier": 0.5}
        policy = {
            "intel": {"crisis_alpha": {"enabled": True, "shadow_only": False}},
            "risk_limits": {"max_gross_exposure": 1.20},
        }
        target_positions = _make_target_positions({"AAPL": 0.30})

        with patch(_CRISIS_PIPELINE, return_value=_make_ca_result({"GLD": 0.10})):
            result = _sp_apply_crisis_alpha_cap(
                target_positions, ctx, policy, _LOG, meta=meta
            )

        gld = result[result["symbol"] == "GLD"].iloc[0]
        assert gld["target_weight"] == pytest.approx(0.10), "crisis hedge exempt"
        rec = [r for r in meta["overlay_exposure"] if r["overlay"] == "crisis_alpha"]
        assert rec and rec[0]["derisk_applied"] is False

    def test_derisked_when_flag_on(self) -> None:
        ctx = _make_crisis_ctx()
        meta = {"final_exposure_multiplier": 0.5}
        policy = {
            "intel": {
                "crisis_alpha": {
                    "enabled": True,
                    "shadow_only": False,
                    "apply_global_derisk": True,
                }
            },
            "risk_limits": {"max_gross_exposure": 1.20},
        }
        target_positions = _make_target_positions({"AAPL": 0.30})

        with patch(_CRISIS_PIPELINE, return_value=_make_ca_result({"GLD": 0.10})):
            result = _sp_apply_crisis_alpha_cap(
                target_positions, ctx, policy, _LOG, meta=meta
            )

        gld = result[result["symbol"] == "GLD"].iloc[0]
        assert gld["target_weight"] == pytest.approx(0.05), "global de-risk folded in"
        assert gld["target_qty"] == pytest.approx(0.05 * 100_000.0)
        rec = [r for r in meta["overlay_exposure"] if r["overlay"] == "crisis_alpha"]
        assert rec and rec[0]["derisk_applied"] is True
