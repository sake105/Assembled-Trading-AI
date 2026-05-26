"""Tests for T4.2: news_alpha weight wiring in _tc_sizing._sp_apply_news_alpha.

Covers:
  - disabled policy → no-op (target_positions unchanged)
  - shadow_only defaults to True when omitted
  - shadow_only=True → pipeline called, weights NOT applied
  - shadow_only=False + target_weights → new entries added with correct target_qty
  - shadow_only=False + overlapping symbol → capped (never boosted)
  - gross-exposure guard → renormalize if total > max_gross_exposure
  - exception → fail-open, positions unchanged
  - flat day (empty target_positions) → entries still added
  - trigger_items sourced from ctx.news_geo["news_trigger_items"]
"""

from __future__ import annotations

import logging
import types

import pandas as pd
import pytest
from unittest.mock import patch

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POLICY_ENABLED_SHADOW = {
    "news_alpha": {
        "enabled": True,
        "shadow_only": True,
    }
}

_POLICY_ENABLED_LIVE = {
    "news_alpha": {
        "enabled": True,
        "shadow_only": False,
    },
    "risk_limits": {"max_gross_exposure": 1.20},
}

_POLICY_DISABLED = {
    "news_alpha": {
        "enabled": False,
    }
}


def _make_target_positions(weights: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": sym, "target_weight": w, "target_qty": 0.0}
            for sym, w in weights.items()
        ]
    )


def _make_ctx(
    capital: float = 100_000.0,
    news_trigger_items: list | None = None,
    meta: dict | None = None,
    as_of: str = "2026-05-26 12:00:00+00:00",
) -> types.SimpleNamespace:
    _news_geo: dict = {}
    if news_trigger_items is not None:
        _news_geo["news_trigger_items"] = news_trigger_items
    return types.SimpleNamespace(
        meta=meta or {},
        as_of=pd.Timestamp(as_of),
        news_geo=_news_geo,
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


_PIPELINE_PATH = "src.assembled_core.events.news_alpha.pipeline.run_news_alpha_pipeline"


# ---------------------------------------------------------------------------
# Policy gate
# ---------------------------------------------------------------------------


class TestT42PolicyGate:
    """news_alpha disabled → no-op regardless of ctx content."""

    def test_disabled_policy_returns_unchanged_positions(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx()
        target_positions = _make_target_positions({"AAPL": 0.30, "MSFT": 0.20})
        result = _sp_apply_news_alpha(
            target_positions, ctx, _POLICY_DISABLED, logging.getLogger("test")
        )
        # Same object, no changes
        assert len(result) == 2
        assert result.loc[result["symbol"] == "AAPL", "target_weight"].iloc[
            0
        ] == pytest.approx(0.30)

    def test_missing_policy_returns_unchanged_positions(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx()
        target_positions = _make_target_positions({"AAPL": 0.30})
        result = _sp_apply_news_alpha(
            target_positions, ctx, {}, logging.getLogger("test")
        )
        assert len(result) == 1
        assert result.loc[result["symbol"] == "AAPL", "target_weight"].iloc[
            0
        ] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# shadow_only default
# ---------------------------------------------------------------------------


class TestT42ShadowOnlyDefault:
    """shadow_only must default to True when not specified in policy."""

    def test_shadow_only_defaults_to_true_functional(self) -> None:
        """Calls the real function with no shadow_only key; verifies positions are unchanged
        (i.e., the default True prevented weight application)."""
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        # Policy has enabled=True but NO shadow_only key — default must be True
        policy_no_shadow_key = {
            "news_alpha": {"enabled": True},
            "risk_limits": {"max_gross_exposure": 1.20},
        }
        ctx = _make_ctx()
        target_positions = _make_target_positions({"AAPL": 0.30, "MSFT": 0.20})

        # Pipeline returns live-looking weights; if shadow_only defaulted False they'd apply
        na_result = _make_na_result({"XLE": 0.08, "GLD": 0.06})

        with patch(_PIPELINE_PATH, return_value=na_result):
            result = _sp_apply_news_alpha(
                target_positions, ctx, policy_no_shadow_key, logging.getLogger("test")
            )

        # shadow_only=True (default) → no new entries added
        assert set(result["symbol"].tolist()) == {"AAPL", "MSFT"}, (
            "shadow_only must default True — no entries added when key is absent"
        )


# ---------------------------------------------------------------------------
# shadow_only=True
# ---------------------------------------------------------------------------


class TestT42ShadowOnlySkipsApplication:
    """shadow_only=True: pipeline is called but weights are NOT merged."""

    def test_shadow_only_does_not_modify_positions(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx()
        target_positions = _make_target_positions({"AAPL": 0.30, "MSFT": 0.20})

        na_result = _make_na_result({"XLE": 0.08, "XOM": 0.08})
        # Override shadow_only on the result object to True to match shadow policy
        na_result.shadow_only = True

        with patch(_PIPELINE_PATH, return_value=na_result):
            result = _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_SHADOW, logging.getLogger("test")
            )

        # No new entries, no weight changes
        assert set(result["symbol"].tolist()) == {"AAPL", "MSFT"}
        assert result.loc[result["symbol"] == "AAPL", "target_weight"].iloc[
            0
        ] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# shadow_only=False — entry adding
# ---------------------------------------------------------------------------


class TestT42LiveModeAddsEntries:
    """shadow_only=False: new symbols from target_weights are added to positions."""

    def test_new_symbols_added_with_correct_target_qty(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx(capital=100_000.0)
        target_positions = _make_target_positions({"AAPL": 0.30, "MSFT": 0.20})

        na_result = _make_na_result({"XLE": 0.08, "GLD": 0.06})

        with patch(_PIPELINE_PATH, return_value=na_result):
            result = _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_LIVE, logging.getLogger("test")
            )

        assert "XLE" in result["symbol"].values
        assert "GLD" in result["symbol"].values
        xle = result[result["symbol"] == "XLE"].iloc[0]
        assert xle["target_weight"] == pytest.approx(0.08)
        assert xle["target_qty"] == pytest.approx(0.08 * 100_000.0)

    def test_target_qty_not_nan(self) -> None:
        """target_qty must be a real number — NaN causes zero orders in order_generation."""
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx(capital=100_000.0)
        target_positions = _make_target_positions({"AAPL": 0.30})

        na_result = _make_na_result({"XLE": 0.08})

        with patch(_PIPELINE_PATH, return_value=na_result):
            result = _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_LIVE, logging.getLogger("test")
            )

        assert not result["target_qty"].isna().any(), (
            "No NaN target_qty after adding entries"
        )

    def test_flat_day_empty_positions_still_adds_entries(self) -> None:
        """When main portfolio produces no signals, news_alpha entries must still appear."""
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx(capital=100_000.0)
        target_positions = pd.DataFrame(
            columns=["symbol", "target_weight", "target_qty"]
        )

        na_result = _make_na_result({"XLE": 0.08, "UCO": 0.06})

        with patch(_PIPELINE_PATH, return_value=na_result):
            result = _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_LIVE, logging.getLogger("test")
            )

        assert "XLE" in result["symbol"].values
        assert "UCO" in result["symbol"].values
        assert not result["target_qty"].isna().any()


# ---------------------------------------------------------------------------
# shadow_only=False — overlap cap
# ---------------------------------------------------------------------------


class TestT42CapNeverBoosts:
    """When a symbol appears in both na_weights and target_positions, it is capped
    (min-merge) — never boosted."""

    def test_cap_reduces_when_na_weight_lower(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx(capital=100_000.0)
        # XLE at 0.30 in mfv2; news_alpha wants 0.08 (lower) → cap to 0.08
        target_positions = _make_target_positions({"XLE": 0.30, "AAPL": 0.25})

        na_result = _make_na_result({"XLE": 0.08})

        with patch(_PIPELINE_PATH, return_value=na_result):
            result = _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_LIVE, logging.getLogger("test")
            )

        xle = result[result["symbol"] == "XLE"].iloc[0]
        assert xle["target_weight"] == pytest.approx(0.08), (
            "XLE capped from 0.30 to 0.08"
        )
        assert xle["target_qty"] == pytest.approx(0.08 * 100_000.0), (
            "target_qty updated after cap"
        )

    def test_cap_never_increases_weight(self) -> None:
        """If na_weight > current_weight, the weight must NOT increase."""
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx(capital=100_000.0)
        # XLE at 0.05; news_alpha returns 0.20 (higher) → must stay 0.05
        target_positions = _make_target_positions({"XLE": 0.05})

        na_result = _make_na_result({"XLE": 0.20})

        with patch(_PIPELINE_PATH, return_value=na_result):
            result = _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_LIVE, logging.getLogger("test")
            )

        xle = result[result["symbol"] == "XLE"].iloc[0]
        assert xle["target_weight"] == pytest.approx(0.05), (
            "weight must never increase via news_alpha"
        )

    def test_overlapping_symbol_not_duplicated(self) -> None:
        """A symbol present in both mfv2 and news_alpha must appear exactly once."""
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx(capital=100_000.0)
        target_positions = _make_target_positions({"XLE": 0.30, "AAPL": 0.20})

        na_result = _make_na_result({"XLE": 0.08, "GLD": 0.06})

        with patch(_PIPELINE_PATH, return_value=na_result):
            result = _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_LIVE, logging.getLogger("test")
            )

        xle_rows = result[result["symbol"] == "XLE"]
        assert len(xle_rows) == 1, "XLE must not be duplicated"

    def test_mixed_case_keys_do_not_produce_duplicate_rows(self) -> None:
        """F-senior-2 regression: lowercase keys from target_weights must be normalized
        to uppercase before cap/new-rows logic to prevent duplicate symbol rows."""
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx(capital=100_000.0)
        # mfv2 holds XLE (uppercase)
        target_positions = _make_target_positions({"XLE": 0.30, "AAPL": 0.20})

        # Simulate pipeline returning lowercase key (pathological but possible)
        na_result = _make_na_result({"xle": 0.08, "gld": 0.06})

        with patch(_PIPELINE_PATH, return_value=na_result):
            result = _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_LIVE, logging.getLogger("test")
            )

        xle_rows = result[result["symbol"].str.upper() == "XLE"]
        assert len(xle_rows) == 1, "XLE must not be duplicated when key is lowercase"
        # XLE should be capped to 0.08, not still 0.30 + a new row at 0.08
        assert xle_rows["target_weight"].iloc[0] == pytest.approx(0.08), (
            "lowercase key must still trigger the cap (0.30 → 0.08)"
        )


# ---------------------------------------------------------------------------
# Gross-exposure guard
# ---------------------------------------------------------------------------


class TestT42GrossExposureGuard:
    """After adding news_alpha entries, renormalize if total gross > max_gross_exposure."""

    def test_renormalize_when_over_cap(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx(capital=100_000.0)
        # 10 * 0.11 = 1.10 gross
        target_positions = _make_target_positions({f"S{i}": 0.11 for i in range(10)})

        # Adding 0.08+0.07=0.15 pushes total to 1.25 > 1.20
        na_result = _make_na_result({"XLE": 0.08, "GLD": 0.07})

        with patch(_PIPELINE_PATH, return_value=na_result):
            result = _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_LIVE, logging.getLogger("test")
            )

        total = result["target_weight"].abs().sum()
        assert total == pytest.approx(1.20, abs=1e-6), (
            "Gross must be normalized to 1.20"
        )

    def test_no_renormalize_when_within_cap(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx(capital=100_000.0)
        target_positions = _make_target_positions({"AAPL": 0.20, "MSFT": 0.15})

        na_result = _make_na_result({"XLE": 0.05})

        with patch(_PIPELINE_PATH, return_value=na_result):
            result = _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_LIVE, logging.getLogger("test")
            )

        # Total = 0.40 < 1.20 — no renormalization
        aapl_w = result[result["symbol"] == "AAPL"]["target_weight"].iloc[0]
        assert aapl_w == pytest.approx(0.20), "Weights unchanged when gross < max"

    def test_renormalize_also_scales_target_qty(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx(capital=100_000.0)
        target_positions = _make_target_positions({f"S{i}": 0.11 for i in range(10)})

        na_result = _make_na_result({"XLE": 0.08, "GLD": 0.07})

        with patch(_PIPELINE_PATH, return_value=na_result):
            result = _sp_apply_news_alpha(
                target_positions,
                ctx,
                {
                    "news_alpha": {"enabled": True, "shadow_only": False},
                    "risk_limits": {"max_gross_exposure": 0.80},  # tight cap
                },
                logging.getLogger("test"),
            )

        assert not result["target_qty"].isna().any(), "No NaN target_qty after renorm"
        xle = result[result["symbol"] == "XLE"].iloc[0]
        expected_qty = xle["target_weight"] * 100_000.0
        assert xle["target_qty"] == pytest.approx(expected_qty, rel=0.01)


# ---------------------------------------------------------------------------
# Exception fail-open
# ---------------------------------------------------------------------------


class TestT42ExceptionFailOpen:
    def test_pipeline_exception_does_not_crash_or_alter_positions(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx()
        target_positions = _make_target_positions({"AAPL": 0.30})

        with patch(_PIPELINE_PATH, side_effect=RuntimeError("simulated failure")):
            result = _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_LIVE, logging.getLogger("test")
            )

        assert len(result) == 1
        assert result.loc[result["symbol"] == "AAPL", "target_weight"].iloc[
            0
        ] == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Trigger-items sourcing from ctx.news_geo
# ---------------------------------------------------------------------------


class TestT42TriggerItemsSourcing:
    """news_trigger_items are pulled from ctx.news_geo and forwarded to pipeline."""

    def test_trigger_items_forwarded_from_news_geo(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        triggers = [{"severity": 3, "topic": "shipping_disruption", "source": "RSS"}]
        ctx = _make_ctx(news_trigger_items=triggers)
        target_positions = _make_target_positions({"AAPL": 0.20})

        na_result = _make_na_result({})

        with patch(_PIPELINE_PATH, return_value=na_result) as mock_pipe:
            _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_LIVE, logging.getLogger("test")
            )

        call_kwargs = mock_pipe.call_args
        actual_triggers = (
            call_kwargs.kwargs.get("trigger_items")
            if call_kwargs.kwargs
            else call_kwargs.args[0]
        )
        assert actual_triggers == triggers, (
            "trigger_items must match ctx.news_geo value"
        )

    def test_empty_news_geo_passes_empty_triggers(self) -> None:
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx()  # no news_trigger_items
        target_positions = _make_target_positions({"AAPL": 0.20})

        na_result = _make_na_result({})

        with patch(_PIPELINE_PATH, return_value=na_result) as mock_pipe:
            _sp_apply_news_alpha(
                target_positions, ctx, _POLICY_ENABLED_LIVE, logging.getLogger("test")
            )

        call_kwargs = mock_pipe.call_args
        actual_triggers = (
            call_kwargs.kwargs.get("trigger_items")
            if call_kwargs.kwargs
            else call_kwargs.args[0]
        )
        assert actual_triggers == [], "empty news_geo must forward empty trigger list"


# ---------------------------------------------------------------------------
# M-2 regression: capital=0 warning
# ---------------------------------------------------------------------------


class TestT42CapitalZeroWarning:
    """When ctx.capital=0, a WARNING must be emitted (target_qty will be 0)."""

    def test_capital_zero_emits_warning(self, caplog) -> None:
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        ctx = _make_ctx(capital=0.0)  # capital explicitly zero
        target_positions = _make_target_positions({"AAPL": 0.20})

        na_result = _make_na_result({"XLE": 0.08})

        with (
            patch(_PIPELINE_PATH, return_value=na_result),
            caplog.at_level(
                logging.WARNING, logger="src.assembled_core.pipeline._tc_sizing"
            ),
        ):
            _sp_apply_news_alpha(
                target_positions,
                ctx,
                _POLICY_ENABLED_LIVE,
                logging.getLogger("src.assembled_core.pipeline._tc_sizing"),
            )

        assert any(
            "capital" in r.message and "0" in r.message for r in caplog.records
        ), (
            "capital=0 must produce a WARNING so the operator knows target_qty will be zero"
        )


# ---------------------------------------------------------------------------
# B-1 regression: positions_to_exit logged as WARNING
# ---------------------------------------------------------------------------


class TestT42ExitVisibility:
    """positions_to_exit must be logged as WARNING when shadow_only=False."""

    def test_positions_to_exit_logged_when_live(self, caplog) -> None:
        from src.assembled_core.events.news_alpha.models import (
            NewsAlphaResult,
            NewsAlphaSignal,
        )
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        sig = NewsAlphaSignal(
            event_id="e1",
            topic_id="shipping_disruption",
            trigger_type="supply_chain",
            source="test",
            symbol="XLE",
            direction="long",
        )
        na_result = NewsAlphaResult(
            timestamp_utc="2026-05-26T12:00:00+00:00",
            signals=[],
            target_weights={},
            positions_to_exit=[(sig, "time_5d")],
            shadow_only=False,
            errors=[],
        )

        ctx = _make_ctx()
        target_positions = _make_target_positions({"AAPL": 0.20})

        with (
            patch(_PIPELINE_PATH, return_value=na_result),
            caplog.at_level(
                logging.WARNING, logger="src.assembled_core.pipeline._tc_sizing"
            ),
        ):
            _sp_apply_news_alpha(
                target_positions,
                ctx,
                _POLICY_ENABLED_LIVE,
                logging.getLogger("src.assembled_core.pipeline._tc_sizing"),
            )

        assert any(
            "positions_to_exit" in r.message or "flagged for exit" in r.message
            for r in caplog.records
        ), "positions_to_exit must be logged as WARNING for operator visibility"

    def test_positions_to_exit_silent_in_shadow_mode(self, caplog) -> None:
        """No exit warning in shadow mode — exits are not real."""
        from src.assembled_core.events.news_alpha.models import (
            NewsAlphaResult,
            NewsAlphaSignal,
        )
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_news_alpha

        sig = NewsAlphaSignal(
            event_id="e1",
            topic_id="shipping_disruption",
            trigger_type="supply_chain",
            source="test",
            symbol="XLE",
            direction="long",
        )
        na_result = NewsAlphaResult(
            timestamp_utc="2026-05-26T12:00:00+00:00",
            signals=[],
            target_weights={},
            positions_to_exit=[(sig, "time_5d")],
            shadow_only=True,
            errors=[],
        )

        ctx = _make_ctx()
        target_positions = _make_target_positions({"AAPL": 0.20})

        with (
            patch(_PIPELINE_PATH, return_value=na_result),
            caplog.at_level(
                logging.WARNING, logger="src.assembled_core.pipeline._tc_sizing"
            ),
        ):
            _sp_apply_news_alpha(
                target_positions,
                ctx,
                _POLICY_ENABLED_SHADOW,
                logging.getLogger("src.assembled_core.pipeline._tc_sizing"),
            )

        assert not any("flagged for exit" in r.message for r in caplog.records), (
            "exit warning must be silent in shadow mode"
        )
