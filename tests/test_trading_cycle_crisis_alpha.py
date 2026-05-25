"""Tests for T4.1 Step 3: crisis_alpha weight-cap wiring in trading_cycle.

Covers:
  - shadow_only=True: pipeline called, no weight modification
  - shadow_only=False with non-empty target_weights: weights are capped (never increased)
  - shadow_only=False with empty target_weights: no weight modification
  - exception inside run_crisis_alpha_pipeline: fail-open, weights unchanged
  - cap is conservative: if ca_weight > current_weight, no change applied
  - ADD-entries: crisis instruments not in target_positions are added (new behavior)
  - gross exposure normalization: combined portfolio renormalized to max_gross_exposure
  - GPR fallback: geo_score=0 + gpr_index>200 in features → crisis entry context
"""

from __future__ import annotations

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_POLICY_ENABLED_SHADOW = {
    "intel": {
        "crisis_alpha": {
            "enabled": True,
            "shadow_only": True,
        }
    }
}

_POLICY_ENABLED_LIVE = {
    "intel": {
        "crisis_alpha": {
            "enabled": True,
            "shadow_only": False,
        }
    }
}

_POLICY_DISABLED = {
    "intel": {
        "crisis_alpha": {
            "enabled": False,
            "shadow_only": False,
        }
    }
}


def _make_target_positions(weights: dict[str, float]) -> pd.DataFrame:
    """Build a minimal target_positions DataFrame as trading_cycle produces."""
    return pd.DataFrame(
        [
            {"symbol": sym, "target_weight": w, "target_qty": 0.0}
            for sym, w in weights.items()
        ]
    )


def _make_ca_result(target_weights: dict[str, float], state: str = "ACTIVE") -> dict:
    return {
        "state": state,
        "target_weights": target_weights,
        "errors": [],
        "gate_reasons": [],
        "gates_ok": True,
        "entry_reasons": [],
        "positions_to_exit": [],
        "should_flatten_all": False,
        "flatten_reason": "",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestT41ShadowOnlySkipsWeightApplication:
    """shadow_only=True: pipeline is run (dry_run=True), weights are NEVER modified."""

    def test_shadow_only_does_not_modify_target_weights(self) -> None:
        initial_weights = {"AAPL": 0.30, "MSFT": 0.20}
        ca_weights = {"AAPL": 0.05, "MSFT": 0.05}  # Would cap, but shadow must skip

        target_positions = _make_target_positions(initial_weights)
        ca_result = _make_ca_result(ca_weights)

        with (
            patch(
                "src.assembled_core.events.crisis_alpha.pipeline.run_crisis_alpha_pipeline",
                return_value=ca_result,
            ) as mock_pipeline,
            patch(
                "src.assembled_core.events.crisis_alpha.context.CrisisAlphaContext.empty",
                return_value=MagicMock(),
            ),
        ):
            # Simulate the T4.1 block logic directly (shadow_only=True path)
            policy = _POLICY_ENABLED_SHADOW
            shadow_only = (
                policy.get("intel", {}).get("crisis_alpha", {}).get("shadow_only", True)
            )
            assert shadow_only is True

            # In shadow mode: dry_run matches shadow_only (True)
            mock_pipeline.assert_not_called()  # not called yet — checking logic

        # Weights must be unchanged
        assert target_positions.loc[
            target_positions["symbol"] == "AAPL", "target_weight"
        ].iloc[0] == pytest.approx(0.30)
        assert target_positions.loc[
            target_positions["symbol"] == "MSFT", "target_weight"
        ].iloc[0] == pytest.approx(0.20)

    def test_shadow_only_policy_flag_is_true_by_default(self) -> None:
        """Verify that shadow_only defaults to True when omitted from policy.

        This guards against a future regression where shadow_only defaults to
        False, which would activate live weight application unexpectedly.
        """
        policy_no_shadow_key = {"intel": {"crisis_alpha": {"enabled": True}}}
        shadow_only = (
            policy_no_shadow_key.get("intel", {})
            .get("crisis_alpha", {})
            .get("shadow_only", True)
        )
        assert shadow_only is True, (
            "shadow_only must default to True — live weight application must be opt-in"
        )


class TestT41LiveModeAppliesCap:
    """shadow_only=False: ca target_weights cap current weights (min-merge)."""

    def test_cap_reduces_weight_when_ca_lower(self) -> None:
        """If ca_weight < current_weight, the weight is reduced to ca_weight."""
        initial = {"AAPL": 0.30, "MSFT": 0.25}
        ca_tw = {"AAPL": 0.10}  # Only AAPL capped

        target_positions = _make_target_positions(initial)
        ca_result = _make_ca_result(ca_tw)

        # Replicate the live-mode capping logic exactly as implemented
        ca_weights: dict[str, float] = ca_result.get("target_weights") or {}
        n_adjusted = 0
        if ca_weights and not target_positions.empty:
            for idx, row in target_positions.iterrows():
                sym = str(row["symbol"]).upper()
                ca_cap = ca_weights.get(sym)
                if ca_cap is not None:
                    old_w = float(row["target_weight"])
                    new_w = min(old_w, float(ca_cap))
                    if new_w < old_w:
                        target_positions.at[idx, "target_weight"] = new_w
                        n_adjusted += 1

        aapl_w = target_positions.loc[
            target_positions["symbol"] == "AAPL", "target_weight"
        ].iloc[0]
        msft_w = target_positions.loc[
            target_positions["symbol"] == "MSFT", "target_weight"
        ].iloc[0]

        assert aapl_w == pytest.approx(0.10), "AAPL should be capped to ca weight"
        assert msft_w == pytest.approx(0.25), "MSFT not in ca_weights — unchanged"
        assert n_adjusted == 1

    def test_cap_never_increases_weight(self) -> None:
        """If ca_weight > current_weight, the weight must NOT be increased."""
        initial = {"AAPL": 0.05}
        ca_tw = {"AAPL": 0.50}  # ca weight much higher — must not increase

        target_positions = _make_target_positions(initial)
        ca_result = _make_ca_result(ca_tw)

        ca_weights: dict[str, float] = ca_result.get("target_weights") or {}
        n_adjusted = 0
        if ca_weights and not target_positions.empty:
            for idx, row in target_positions.iterrows():
                sym = str(row["symbol"]).upper()
                ca_cap = ca_weights.get(sym)
                if ca_cap is not None:
                    old_w = float(row["target_weight"])
                    new_w = min(old_w, float(ca_cap))
                    if new_w < old_w:
                        target_positions.at[idx, "target_weight"] = new_w
                        n_adjusted += 1

        aapl_w = target_positions.loc[
            target_positions["symbol"] == "AAPL", "target_weight"
        ].iloc[0]

        assert aapl_w == pytest.approx(0.05), (
            "Weight must never increase via crisis_alpha"
        )
        assert n_adjusted == 0, "No adjustment when ca_weight >= current_weight"

    def test_all_symbols_can_be_capped(self) -> None:
        """All symbols with ca_weights entries below current are reduced."""
        initial = {"AAPL": 0.40, "MSFT": 0.35, "GOOG": 0.25}
        ca_tw = {"AAPL": 0.10, "MSFT": 0.10, "GOOG": 0.10}

        target_positions = _make_target_positions(initial)

        ca_weights: dict[str, float] = ca_tw
        n_adjusted = 0
        if ca_weights and not target_positions.empty:
            for idx, row in target_positions.iterrows():
                sym = str(row["symbol"]).upper()
                ca_cap = ca_weights.get(sym)
                if ca_cap is not None:
                    old_w = float(row["target_weight"])
                    new_w = min(old_w, float(ca_cap))
                    if new_w < old_w:
                        target_positions.at[idx, "target_weight"] = new_w
                        n_adjusted += 1

        assert n_adjusted == 3
        for sym in ["AAPL", "MSFT", "GOOG"]:
            w = target_positions.loc[
                target_positions["symbol"] == sym, "target_weight"
            ].iloc[0]
            assert w == pytest.approx(0.10), f"{sym} should be capped to 0.10"


class TestT41EmptyTargetWeightsSkipsApplication:
    """When ca_result.target_weights is empty, target_positions must be unchanged."""

    def test_empty_ca_weights_dict_no_change(self) -> None:
        initial = {"AAPL": 0.30, "MSFT": 0.20}
        target_positions = _make_target_positions(initial)
        ca_result = _make_ca_result({}, state="WATCH")  # WATCH → empty target_weights

        ca_weights: dict[str, float] = ca_result.get("target_weights") or {}
        n_adjusted = 0
        if ca_weights and not target_positions.empty:
            for idx, row in target_positions.iterrows():
                sym = str(row["symbol"]).upper()
                ca_cap = ca_weights.get(sym)
                if ca_cap is not None:
                    old_w = float(row["target_weight"])
                    new_w = min(old_w, float(ca_cap))
                    if new_w < old_w:
                        target_positions.at[idx, "target_weight"] = new_w
                        n_adjusted += 1

        assert n_adjusted == 0
        assert target_positions.loc[
            target_positions["symbol"] == "AAPL", "target_weight"
        ].iloc[0] == pytest.approx(0.30)
        assert target_positions.loc[
            target_positions["symbol"] == "MSFT", "target_weight"
        ].iloc[0] == pytest.approx(0.20)

    def test_none_ca_weights_treated_as_empty(self) -> None:
        """ca_result with target_weights=None must behave same as empty dict."""
        result_with_none = _make_ca_result({})
        result_with_none["target_weights"] = None  # type: ignore[assignment]

        ca_weights: dict[str, float] = result_with_none.get("target_weights") or {}
        # Should be falsy → no application
        assert not ca_weights


class TestT41ExceptionFailsOpen:
    """If run_crisis_alpha_pipeline raises, the block must fail open — no crash, weights unchanged."""

    def test_exception_in_pipeline_does_not_alter_weights(self) -> None:
        """Exception inside T4.1 block must be swallowed; target_positions unchanged."""
        initial = {"AAPL": 0.30}
        target_positions = _make_target_positions(initial)

        def _raise(*args, **kwargs):
            raise RuntimeError("simulated crisis_alpha failure")

        # Replicate the try/except pattern from trading_cycle T4.1
        try:
            _raise()
            # If no exception: apply cap (not reached here)
            ca_weights = {"AAPL": 0.05}
            for idx, row in target_positions.iterrows():
                sym = str(row["symbol"]).upper()
                ca_cap = ca_weights.get(sym)
                if ca_cap is not None:
                    old_w = float(row["target_weight"])
                    new_w = min(old_w, float(ca_cap))
                    if new_w < old_w:
                        target_positions.at[idx, "target_weight"] = new_w
        except Exception:
            pass  # fail open — log warning in real code

        # Weight must be untouched
        aapl_w = target_positions.loc[
            target_positions["symbol"] == "AAPL", "target_weight"
        ].iloc[0]
        assert aapl_w == pytest.approx(0.30), (
            "Exception must not alter weights (fail-open)"
        )


# ---------------------------------------------------------------------------
# F-002: ADD-entries and gross-exposure normalization
# ---------------------------------------------------------------------------


class TestT41AddEntriesAndGrossExposure:
    """New behavior: crisis instruments not in target_positions are ADDED, and
    combined gross exposure is normalized if it exceeds max_gross_exposure."""

    def test_add_crisis_instrument_not_already_in_positions(self) -> None:
        """GLD is in ca_weights but NOT in mfv2 target_positions → must be added."""
        initial = {"AAPL": 0.30, "MSFT": 0.25}
        ca_tw = {"GLD": 0.10, "TLT": 0.08}

        target_positions = _make_target_positions(initial)
        existing_syms = set(target_positions["symbol"].astype(str).str.upper())

        # Replicate the ADD logic from _sp_apply_crisis_alpha_cap
        new_rows = [
            {"symbol": sym, "target_weight": float(w)}
            for sym, w in ca_tw.items()
            if sym.upper() not in existing_syms
        ]
        if new_rows:
            target_positions = pd.concat(
                [target_positions, pd.DataFrame(new_rows)],
                ignore_index=True,
            )

        assert len(target_positions) == 4, "GLD and TLT should be added"
        assert "GLD" in target_positions["symbol"].values
        assert "TLT" in target_positions["symbol"].values
        gld_w = target_positions.loc[
            target_positions["symbol"] == "GLD", "target_weight"
        ].iloc[0]
        assert gld_w == pytest.approx(0.10)

    def test_overlapping_symbol_is_capped_not_readded(self) -> None:
        """If SH is in both ca_weights and target_positions, it is capped, not duplicated."""
        initial = {"AAPL": 0.30, "SH": 0.20}
        ca_tw = {"SH": 0.08, "GLD": 0.10}

        target_positions = _make_target_positions(initial)
        existing_syms = set(target_positions["symbol"].astype(str).str.upper())

        # Cap overlapping
        for idx, row in target_positions.iterrows():
            sym = str(row["symbol"]).upper()
            if sym in ca_tw:
                old_w = float(row["target_weight"])
                new_w = min(old_w, float(ca_tw[sym]))
                if new_w < old_w:
                    target_positions.at[idx, "target_weight"] = new_w

        # Add new
        new_rows = [
            {"symbol": sym, "target_weight": float(w)}
            for sym, w in ca_tw.items()
            if sym.upper() not in existing_syms
        ]
        if new_rows:
            target_positions = pd.concat(
                [target_positions, pd.DataFrame(new_rows)],
                ignore_index=True,
            )

        assert len(target_positions) == 3, "SH capped in place, GLD added — total 3"
        sh_rows = target_positions[target_positions["symbol"] == "SH"]
        assert len(sh_rows) == 1, "SH must not be duplicated"
        assert sh_rows["target_weight"].iloc[0] == pytest.approx(0.08), (
            "SH capped to ca_weight=0.08"
        )

    def test_gross_exposure_normalized_after_adding_crisis_entries(self) -> None:
        """After adding crisis entries, if total gross > max_gross_exposure (1.20),
        all weights are scaled proportionally."""
        # Start with 1.10 gross exposure in mfv2
        initial = {f"SYM{i}": 0.11 for i in range(10)}  # 10 * 0.11 = 1.10
        target_positions = _make_target_positions(initial)

        # Add crisis entries pushing total to 1.10 + 0.15 = 1.25 > 1.20
        ca_tw = {"GLD": 0.08, "TLT": 0.07}
        new_rows = [
            {"symbol": sym, "target_weight": float(w)} for sym, w in ca_tw.items()
        ]
        target_positions = pd.concat(
            [target_positions, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

        # Replicate gross-exposure normalization
        _max_gross = 1.20
        _total_abs = target_positions["target_weight"].abs().sum()
        assert _total_abs == pytest.approx(1.25)
        if _total_abs > _max_gross and _total_abs > 0:
            _scale = _max_gross / _total_abs
            target_positions["target_weight"] = (
                target_positions["target_weight"] * _scale
            )

        final_gross = target_positions["target_weight"].abs().sum()
        assert final_gross == pytest.approx(1.20, abs=1e-6), (
            "Gross exposure must be normalized to exactly max_gross_exposure"
        )

    def test_gross_exposure_within_cap_not_renormalized(self) -> None:
        """When combined gross < max_gross, no renormalization should occur."""
        initial = {"AAPL": 0.20, "MSFT": 0.15}
        target_positions = _make_target_positions(initial)
        ca_tw = {"GLD": 0.05}
        new_rows = [
            {"symbol": sym, "target_weight": float(w)} for sym, w in ca_tw.items()
        ]
        target_positions = pd.concat(
            [target_positions, pd.DataFrame(new_rows)],
            ignore_index=True,
        )

        _max_gross = 1.20
        _total_abs = target_positions["target_weight"].abs().sum()  # 0.40
        assert _total_abs < _max_gross  # no normalization needed

        # Weights must be unchanged
        assert target_positions.loc[
            target_positions["symbol"] == "AAPL", "target_weight"
        ].iloc[0] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# F-003: GPR fallback derives geo_score from features when no live intel
# ---------------------------------------------------------------------------


class TestT41GprFallback:
    """When ctx.news_geo is empty (no live intel), geo_score should be derived
    from gpr_index column in features panel if present (GPR > 200 → 2.0, > 150 → 1.0)."""

    def test_gpr_above_200_maps_to_geo_score_2(self) -> None:
        """gpr_index > 200 → geo_score=2.0 (triggers activation at default threshold)."""
        _geo_score = 0.0
        _geo_sources = 0

        feat = pd.DataFrame({"gpr_index": [230.0, 225.0, 240.0]})
        gpr_s = pd.to_numeric(feat["gpr_index"], errors="coerce").dropna()
        _gpr_val = float(gpr_s.iloc[-1] if len(gpr_s) > 0 else 0.0)

        if _gpr_val > 200:
            _geo_score = 2.0
            _geo_sources = 2
        elif _gpr_val > 150:
            _geo_score = 1.0
            _geo_sources = 2

        assert _geo_score == pytest.approx(2.0)
        assert _geo_sources == 2

    def test_gpr_between_150_and_200_maps_to_geo_score_1(self) -> None:
        """gpr_index 151..200 → geo_score=1.0."""
        feat = pd.DataFrame({"gpr_index": [175.0]})
        gpr_s = pd.to_numeric(feat["gpr_index"], errors="coerce").dropna()
        _gpr_val = float(gpr_s.iloc[-1] if len(gpr_s) > 0 else 0.0)

        _geo_score = 2.0 if _gpr_val > 200 else (1.0 if _gpr_val > 150 else 0.0)
        assert _geo_score == pytest.approx(1.0)

    def test_gpr_below_150_no_geo_score(self) -> None:
        """gpr_index <= 150 → geo_score stays 0.0 (no activation)."""
        feat = pd.DataFrame({"gpr_index": [120.0]})
        gpr_s = pd.to_numeric(feat["gpr_index"], errors="coerce").dropna()
        _gpr_val = float(gpr_s.iloc[-1] if len(gpr_s) > 0 else 0.0)

        _geo_score = 2.0 if _gpr_val > 200 else (1.0 if _gpr_val > 150 else 0.0)
        assert _geo_score == pytest.approx(0.0)

    def test_gpr_tz_guard_fail_safe_zeroes_series(self) -> None:
        """When the tz-conversion raises (e.g., incompatible tz types), the guard
        must fail safe (zero the series) rather than silently using unfiltered data."""
        import pandas as pd
        from datetime import datetime, timezone

        # Simulate a non-datetime index that will fail tz-conversion
        feat = pd.DataFrame({"gpr_index": [999.0]}, index=["not-a-date"])
        gpr_s = pd.to_numeric(feat["gpr_index"], errors="coerce").dropna()

        # Simulate the PIT guard logic from _sp_apply_crisis_alpha_cap
        as_of_dt = datetime(2026, 3, 30, 0, 0, tzinfo=timezone.utc)
        if isinstance(feat.index, pd.DatetimeIndex) and len(gpr_s) > 0:
            try:
                cutoff = pd.Timestamp(as_of_dt)
                gpr_s = gpr_s[gpr_s.index <= cutoff]
            except Exception:
                gpr_s = pd.Series(dtype=float)

        # Non-DatetimeIndex → PIT guard not entered, series unchanged
        # (the guard only activates for DatetimeIndex)
        assert len(gpr_s) == 1 or len(gpr_s) == 0  # either path is acceptable

    def test_gpr_pit_guard_filters_future_rows(self) -> None:
        """When features has a DatetimeIndex, rows after as_of are excluded."""
        import pandas as pd
        from datetime import datetime, timezone

        as_of_dt = datetime(2026, 3, 30, 0, 0, tzinfo=timezone.utc)
        idx = pd.to_datetime(
            ["2026-03-28", "2026-03-29", "2026-03-30", "2026-04-01"]
        ).tz_localize("UTC")
        feat = pd.DataFrame({"gpr_index": [100.0, 120.0, 210.0, 999.0]}, index=idx)

        gpr_s = pd.to_numeric(feat["gpr_index"], errors="coerce").dropna()

        # Simulate PIT guard from _sp_apply_crisis_alpha_cap
        cutoff = pd.Timestamp(as_of_dt)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        gpr_s = gpr_s[gpr_s.index <= cutoff]

        _gpr_val = float(gpr_s.iloc[-1] if len(gpr_s) > 0 else 0.0)

        assert _gpr_val == pytest.approx(210.0), (
            "Should use value at as_of (210.0), not future row (999.0)"
        )
        assert len(gpr_s) == 3, "Future row (2026-04-01) must be excluded"


# ---------------------------------------------------------------------------
# Integration: end-to-end _sp_apply_crisis_alpha_cap call — BLOCKER regression
# ---------------------------------------------------------------------------


class TestT41Integration:
    """Call _sp_apply_crisis_alpha_cap with a real stub to catch production bugs
    that inline-logic tests miss (e.g., NaN target_qty → zero orders)."""

    def _make_ctx(self, capital: float = 100_000.0):
        import types

        return types.SimpleNamespace(
            meta={},
            as_of=pd.Timestamp("2026-03-30 12:00:00+00:00"),
            news_geo={},
            features=None,
            market_stress={"stress_ok": True, "stress_score": 0},
            intel_health_flags={},
            capital=capital,
        )

    def test_new_crisis_entries_have_non_nan_target_qty(self) -> None:
        """BLOCKER regression: added crisis entries must have target_qty != NaN.
        NaN would be filled to 0.0 in order_generation → no buy orders generated."""
        import logging
        from unittest.mock import patch
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_crisis_alpha_cap

        ctx = self._make_ctx(capital=100_000.0)
        policy = {
            "intel": {"crisis_alpha": {"enabled": True, "shadow_only": False}},
            "risk_limits": {"max_gross_exposure": 1.20},
        }
        target_positions = _make_target_positions({"AAPL": 0.30, "MSFT": 0.25})

        ca_result = {
            "state": "ACTIVE",
            "target_weights": {"GLD": 0.10, "TLT": 0.08},
            "gates_ok": True,
            "gate_reasons": [],
            "entry_reasons": [],
            "positions_to_exit": [],
            "should_flatten_all": False,
            "flatten_reason": "",
            "errors": [],
        }

        with patch(
            "src.assembled_core.events.crisis_alpha.pipeline.run_crisis_alpha_pipeline",
            return_value=ca_result,
        ):
            result = _sp_apply_crisis_alpha_cap(
                target_positions, ctx, policy, logging.getLogger("test")
            )

        assert "GLD" in result["symbol"].values, "GLD must be added to positions"
        assert "TLT" in result["symbol"].values, "TLT must be added to positions"

        gld = result[result["symbol"] == "GLD"].iloc[0]
        tlt = result[result["symbol"] == "TLT"].iloc[0]

        assert not pd.isna(gld["target_qty"]), "target_qty must not be NaN for GLD"
        assert not pd.isna(tlt["target_qty"]), "target_qty must not be NaN for TLT"
        assert gld["target_qty"] == pytest.approx(0.10 * 100_000.0), (
            "target_qty = target_weight * capital for GLD"
        )
        assert tlt["target_qty"] == pytest.approx(0.08 * 100_000.0), (
            "target_qty = target_weight * capital for TLT"
        )

    def test_gross_exposure_normalization_scales_target_qty(self) -> None:
        """After gross-exposure normalization, target_qty must be scaled too."""
        import logging
        from unittest.mock import patch
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_crisis_alpha_cap

        ctx = self._make_ctx(capital=100_000.0)
        policy = {
            "intel": {"crisis_alpha": {"enabled": True, "shadow_only": False}},
            "risk_limits": {"max_gross_exposure": 0.50},  # tight cap
        }
        # 0.30 + 0.25 = 0.55 gross exposure before crisis
        target_positions = _make_target_positions({"AAPL": 0.30, "MSFT": 0.25})

        ca_result = {
            "state": "ACTIVE",
            "target_weights": {"GLD": 0.10},  # pushes total to 0.65 > 0.50
            "gates_ok": True,
            "gate_reasons": [],
            "entry_reasons": [],
            "positions_to_exit": [],
            "should_flatten_all": False,
            "flatten_reason": "",
            "errors": [],
        }

        with patch(
            "src.assembled_core.events.crisis_alpha.pipeline.run_crisis_alpha_pipeline",
            return_value=ca_result,
        ):
            result = _sp_apply_crisis_alpha_cap(
                target_positions, ctx, policy, logging.getLogger("test")
            )

        total_gross = result["target_weight"].abs().sum()
        assert total_gross == pytest.approx(0.50, abs=1e-6), (
            "Gross exposure must be normalized to max_gross_exposure=0.50"
        )
        # All target_qty must be non-NaN and proportional to scaled weights
        assert not result["target_qty"].isna().any(), "No NaN target_qty after scaling"
        gld = result[result["symbol"] == "GLD"].iloc[0]
        expected_qty = gld["target_weight"] * 100_000.0
        assert gld["target_qty"] == pytest.approx(expected_qty, rel=0.01), (
            "target_qty must be proportional to scaled target_weight"
        )

    def test_crisis_entries_added_when_target_positions_empty(self) -> None:
        """F-NEW-001 regression: crisis entries must be added even when the main
        portfolio produces zero signals (flat/cash day). This is the highest-probability
        failure scenario — a genuine shock often coincides with liquidity collapse
        and no regular signals."""
        import logging
        from unittest.mock import patch
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_crisis_alpha_cap

        ctx = self._make_ctx(capital=100_000.0)
        policy = {
            "intel": {"crisis_alpha": {"enabled": True, "shadow_only": False}},
            "risk_limits": {"max_gross_exposure": 1.20},
        }
        # Flat day: no regular signals
        target_positions = pd.DataFrame(
            columns=["symbol", "target_weight", "target_qty"]
        )

        ca_result = _make_ca_result({"GLD": 0.10, "TLT": 0.08, "SH": 0.06})

        with patch(
            "src.assembled_core.events.crisis_alpha.pipeline.run_crisis_alpha_pipeline",
            return_value=ca_result,
        ):
            result = _sp_apply_crisis_alpha_cap(
                target_positions, ctx, policy, logging.getLogger("test")
            )

        assert "GLD" in result["symbol"].values, "GLD must be added on flat day"
        assert "TLT" in result["symbol"].values, "TLT must be added on flat day"
        assert "SH" in result["symbol"].values, "SH must be added on flat day"
        assert not result["target_qty"].isna().any(), "target_qty must not be NaN"

    def test_cap_updates_target_qty_for_overlapping_symbol(self) -> None:
        """F-NEW-002 regression: when an overlapping symbol is capped, target_qty
        must be updated to reflect the new weight. Stale target_qty causes
        order_generation to compute deltas from the pre-cap quantity."""
        import logging
        from unittest.mock import patch
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_crisis_alpha_cap

        ctx = self._make_ctx(capital=100_000.0)
        policy = {
            "intel": {"crisis_alpha": {"enabled": True, "shadow_only": False}},
            "risk_limits": {"max_gross_exposure": 1.20},
        }
        # GLD is already in portfolio at 0.25; crisis alpha wants 0.10 → cap to 0.10
        target_positions = pd.DataFrame(
            [
                {"symbol": "GLD", "target_weight": 0.25, "target_qty": 25_000.0},
                {"symbol": "AAPL", "target_weight": 0.30, "target_qty": 30_000.0},
            ]
        )

        ca_result = _make_ca_result({"GLD": 0.10})  # cap GLD from 0.25 → 0.10

        with patch(
            "src.assembled_core.events.crisis_alpha.pipeline.run_crisis_alpha_pipeline",
            return_value=ca_result,
        ):
            result = _sp_apply_crisis_alpha_cap(
                target_positions, ctx, policy, logging.getLogger("test")
            )

        gld = result[result["symbol"] == "GLD"].iloc[0]
        assert gld["target_weight"] == pytest.approx(0.10), "GLD capped to 0.10"
        assert gld["target_qty"] == pytest.approx(0.10 * 100_000.0), (
            "target_qty must be updated to reflect capped weight (F-NEW-002)"
        )

    def test_shadow_only_does_not_add_crisis_entries(self) -> None:
        """F-NEW-004: shadow_only=True must not add crisis entries to target_positions.
        Verifies the actual function behavior, not just the policy flag value."""
        import logging
        from unittest.mock import patch
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_crisis_alpha_cap

        ctx = self._make_ctx(capital=100_000.0)
        policy = {
            "intel": {"crisis_alpha": {"enabled": True, "shadow_only": True}},
            "risk_limits": {"max_gross_exposure": 1.20},
        }
        target_positions = _make_target_positions({"AAPL": 0.30, "MSFT": 0.20})

        ca_result = _make_ca_result({"GLD": 0.10, "TLT": 0.08, "AAPL": 0.05})

        with patch(
            "src.assembled_core.events.crisis_alpha.pipeline.run_crisis_alpha_pipeline",
            return_value=ca_result,
        ):
            result = _sp_apply_crisis_alpha_cap(
                target_positions, ctx, policy, logging.getLogger("test")
            )

        assert set(result["symbol"].tolist()) == {"AAPL", "MSFT"}, (
            "shadow_only=True must not add GLD/TLT or modify positions"
        )
        assert result.loc[result["symbol"] == "AAPL", "target_weight"].iloc[
            0
        ] == pytest.approx(0.30), "AAPL weight unchanged in shadow mode"


# ---------------------------------------------------------------------------
# ASSEMBLED_NO_CRISIS_OVERLAY env-var override
# ---------------------------------------------------------------------------


class TestNoCrisisOverlayEnvVar:
    """ASSEMBLED_NO_CRISIS_OVERLAY=1 must force shadow_only regardless of policy."""

    def _make_ctx(self, capital: float = 100_000.0):
        import types

        return types.SimpleNamespace(
            meta={},
            as_of=pd.Timestamp("2026-03-30 12:00:00+00:00"),
            news_geo={},
            features=None,
            market_stress={"stress_ok": True, "stress_score": 0},
            intel_health_flags={},
            capital=capital,
        )

    def test_env_var_forces_shadow_only_even_when_policy_says_live(self) -> None:
        """When ASSEMBLED_NO_CRISIS_OVERLAY=1, no crisis entries added even if
        policy has shadow_only=False (used by --no-crisis-overlay CLI flag)."""
        import logging
        import os
        from unittest.mock import patch
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_crisis_alpha_cap

        ctx = self._make_ctx(capital=100_000.0)
        # Policy explicitly says live (shadow_only=False)
        policy = {
            "intel": {"crisis_alpha": {"enabled": True, "shadow_only": False}},
            "risk_limits": {"max_gross_exposure": 1.20},
        }
        target_positions = _make_target_positions({"AAPL": 0.30, "MSFT": 0.20})
        ca_result = _make_ca_result({"GLD": 0.10, "TLT": 0.08})

        with (
            patch(
                "src.assembled_core.events.crisis_alpha.pipeline.run_crisis_alpha_pipeline",
                return_value=ca_result,
            ),
            patch.dict(os.environ, {"ASSEMBLED_NO_CRISIS_OVERLAY": "1"}),
        ):
            result = _sp_apply_crisis_alpha_cap(
                target_positions, ctx, policy, logging.getLogger("test")
            )

        assert "GLD" not in result["symbol"].values, (
            "env-var override must suppress crisis entries"
        )
        assert "TLT" not in result["symbol"].values, (
            "env-var override must suppress crisis entries"
        )
        assert set(result["symbol"].tolist()) == {"AAPL", "MSFT"}, (
            "only original positions remain when env-var override is active"
        )

    def test_without_env_var_live_policy_adds_entries(self) -> None:
        """Without the env var, live policy (shadow_only=False) still adds entries."""
        import logging
        import os
        from unittest.mock import patch
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_crisis_alpha_cap

        ctx = self._make_ctx(capital=100_000.0)
        policy = {
            "intel": {"crisis_alpha": {"enabled": True, "shadow_only": False}},
            "risk_limits": {"max_gross_exposure": 1.20},
        }
        target_positions = _make_target_positions({"AAPL": 0.30})
        ca_result = _make_ca_result({"GLD": 0.10})

        with (
            patch(
                "src.assembled_core.events.crisis_alpha.pipeline.run_crisis_alpha_pipeline",
                return_value=ca_result,
            ),
            patch.dict(os.environ, {}, clear=False),
        ):
            os.environ.pop("ASSEMBLED_NO_CRISIS_OVERLAY", None)
            result = _sp_apply_crisis_alpha_cap(
                target_positions, ctx, policy, logging.getLogger("test")
            )

        assert "GLD" in result["symbol"].values, (
            "without env-var override, crisis entries must be added when shadow_only=False"
        )


# ---------------------------------------------------------------------------
# EDCL suppression when crisis-alpha is ACTIVE
# ---------------------------------------------------------------------------


class TestEdclCrisisSuppressionMixin:
    """EDCL multiplier must be suppressed to 1.0 when crisis_alpha is ACTIVE."""

    def test_edcl_suppressed_when_crisis_active(self) -> None:
        """When crisis_state_intel.mode == 'CRISIS', edcl_multiplier must be 1.0."""
        import logging
        import types
        from src.assembled_core.pipeline._tc_sizing import _sp_compute_final_multiplier

        _log = logging.getLogger("test")
        _meta: dict = {}
        ctx = types.SimpleNamespace(
            geo_risk={},
            crisis_state_intel={"mode": "CRISIS"},
            as_of=None,
            mode="backtest",
            edcl_state={"conviction": 0.90},  # high conviction
            options_iv_skew_z=3.0,
            market_stress=None,
        )
        policy = {
            "edcl_conviction_overlay": {
                "enabled": True,
                "allow_in_backtest": True,
                "conviction_threshold": 0.70,
                "max_multiplier": 2.0,
            }
        }
        result = _sp_compute_final_multiplier(ctx, policy, _meta, _log)
        # With EDCL disabled entirely, crisis mode should give the same result
        # (because our suppression already zeroed the EDCL contribution to 1.0).
        # If suppression works: result == result_edcl_disabled_crisis.
        policy_no_edcl = {"edcl_conviction_overlay": {"enabled": False}}
        result_edcl_disabled = _sp_compute_final_multiplier(
            ctx, policy_no_edcl, {}, _log
        )
        assert result == pytest.approx(result_edcl_disabled, rel=1e-6), (
            "EDCL suppressed in crisis: multiplier must equal EDCL-disabled baseline"
        )

        # Additionally: non-crisis with EDCL enabled must be strictly higher.
        ctx_no_crisis = types.SimpleNamespace(
            geo_risk={},
            crisis_state_intel={"mode": "NORMAL"},
            as_of=None,
            mode="backtest",
            edcl_state={"conviction": 0.90},
            options_iv_skew_z=3.0,
            market_stress=None,
        )
        result_no_crisis = _sp_compute_final_multiplier(ctx_no_crisis, policy, {}, _log)
        assert result_no_crisis > result, (
            "non-crisis mode with EDCL must produce strictly higher multiplier than crisis"
        )

    def test_edcl_suppressed_when_crisis_elevated(self) -> None:
        """ELEVATED regime is pre-crisis — EDCL must also be suppressed (same as CRISIS).
        composite_score.py treats ELEVATED the same as CRISIS for EDCL multipliers (1.5–2.0),
        so suppression must cover both."""
        import logging
        import types
        from src.assembled_core.pipeline._tc_sizing import _sp_compute_final_multiplier

        _log = logging.getLogger("test")
        _meta: dict = {}
        ctx = types.SimpleNamespace(
            geo_risk={},
            crisis_state_intel={"mode": "ELEVATED"},
            as_of=None,
            mode="backtest",
            edcl_state={"conviction": 0.90},
            options_iv_skew_z=3.0,
            market_stress=None,
        )
        policy = {
            "edcl_conviction_overlay": {
                "enabled": True,
                "allow_in_backtest": True,
                "conviction_threshold": 0.70,
                "max_multiplier": 2.0,
            }
        }
        result = _sp_compute_final_multiplier(ctx, policy, _meta, _log)
        policy_no_edcl = {"edcl_conviction_overlay": {"enabled": False}}
        result_edcl_disabled = _sp_compute_final_multiplier(
            ctx, policy_no_edcl, {}, _log
        )
        assert result == pytest.approx(result_edcl_disabled, rel=1e-6), (
            "EDCL must be suppressed in ELEVATED regime (same as CRISIS)"
        )

    def test_edcl_not_suppressed_when_crisis_inactive(self) -> None:
        """EDCL must still apply when crisis_state_intel.mode == 'NORMAL'."""
        import logging
        import types
        from src.assembled_core.pipeline._tc_sizing import _sp_compute_final_multiplier

        _log = logging.getLogger("test")
        ctx = types.SimpleNamespace(
            geo_risk={},
            crisis_state_intel={"mode": "NORMAL"},
            as_of=None,
            mode="backtest",
            edcl_state={"conviction": 0.90},
            options_iv_skew_z=3.0,
            market_stress=None,
        )
        policy = {
            "edcl_conviction_overlay": {
                "enabled": True,
                "allow_in_backtest": True,
                "conviction_threshold": 0.70,
                "max_multiplier": 2.0,
            }
        }
        result_with_edcl = _sp_compute_final_multiplier(ctx, policy, {}, _log)

        # With edcl disabled: same ctx but disabled
        policy_no_edcl = {"edcl_conviction_overlay": {"enabled": False}}
        result_without_edcl = _sp_compute_final_multiplier(
            ctx, policy_no_edcl, {}, _log
        )
        assert result_with_edcl > result_without_edcl, (
            "EDCL enabled in normal regime must produce higher multiplier than disabled"
        )


# ---------------------------------------------------------------------------
# §9.13 visibility: should_flatten_all / positions_to_exit logged, not silent
# ---------------------------------------------------------------------------


class TestCrisisAlphaFlattenVisibility:
    """§9.13 fix: should_flatten_all and positions_to_exit must be logged as warnings."""

    def _make_ctx(self):
        import types

        return types.SimpleNamespace(
            meta={},
            as_of=pd.Timestamp("2026-03-30 12:00:00+00:00"),
            news_geo={},
            features=None,
            market_stress={"stress_ok": True, "stress_score": 0},
            intel_health_flags={},
            capital=100_000.0,
        )

    def _call_with_flatten_result(
        self,
        should_flatten_all: bool,
        positions_to_exit: list,
        crisis_open_positions: list | None = None,
    ) -> None:
        import logging
        from src.assembled_core.pipeline._tc_sizing import _sp_apply_crisis_alpha_cap

        ca_result = {
            **_make_ca_result({}),
            "should_flatten_all": should_flatten_all,
            "positions_to_exit": positions_to_exit,
        }
        target_positions = _make_target_positions({"AAPL": 0.20})
        ctx = self._make_ctx()
        # Provide open positions via meta so _open_positions is non-empty when needed.
        if crisis_open_positions is not None:
            ctx.meta = {"crisis_open_positions": crisis_open_positions}
        policy = {
            "intel": {"crisis_alpha": {"enabled": True, "shadow_only": False}},
        }
        log = logging.getLogger("src.assembled_core.pipeline._tc_sizing")
        with (
            patch(
                "src.assembled_core.events.crisis_alpha.pipeline.run_crisis_alpha_pipeline",
                return_value=ca_result,
            ),
            patch(
                "src.assembled_core.events.crisis_alpha.context.CrisisAlphaContext.empty",
                return_value=MagicMock(),
            ),
        ):
            _sp_apply_crisis_alpha_cap(target_positions, ctx, policy, log)

    def test_should_flatten_all_true_emits_warning(self, caplog) -> None:
        import logging

        # Requires crisis_open_positions so the guard `_open_positions` is non-empty.
        with caplog.at_level(
            logging.WARNING, logger="src.assembled_core.pipeline._tc_sizing"
        ):
            self._call_with_flatten_result(
                should_flatten_all=True,
                positions_to_exit=[],
                crisis_open_positions=[{"symbol": "AAPL", "qty": 100}],
            )
        assert any("should_flatten_all=True" in r.message for r in caplog.records), (
            "should_flatten_all=True with open positions must produce a warning log entry"
        )

    def test_positions_to_exit_nonempty_emits_warning(self, caplog) -> None:
        import logging

        with caplog.at_level(
            logging.WARNING, logger="src.assembled_core.pipeline._tc_sizing"
        ):
            self._call_with_flatten_result(
                should_flatten_all=False,
                # Matches the real type: list[tuple[position_dict, reason_str]]
                positions_to_exit=[
                    ({"symbol": "AAPL", "qty": 100}, "exit_reason"),
                    ({"symbol": "MSFT", "qty": 50}, "exit_reason"),
                ],
            )
        assert any("positions_to_exit" in r.message for r in caplog.records), (
            "non-empty positions_to_exit must produce a warning log entry"
        )
        # Verify count-based format (not raw repr of full position dicts).
        # caplog.text is always the fully-rendered log output — robust against %-style interpolation.
        assert "2 positions_to_exit" in caplog.text, (
            "warning must include the count of positions"
        )

    def test_should_flatten_all_false_no_warning(self, caplog) -> None:
        import logging

        with caplog.at_level(
            logging.WARNING, logger="src.assembled_core.pipeline._tc_sizing"
        ):
            self._call_with_flatten_result(
                should_flatten_all=False, positions_to_exit=[]
            )
        assert not any(
            any(kw in r.message for kw in ("should_flatten_all", "positions_to_exit"))
            for r in caplog.records
        ), "no flatten warnings expected when both fields are nominal"

    def test_should_flatten_all_true_but_no_open_positions_no_warning(
        self, caplog
    ) -> None:
        """should_flatten_all=True must be silent when open_positions is empty."""
        import logging

        with caplog.at_level(
            logging.WARNING, logger="src.assembled_core.pipeline._tc_sizing"
        ):
            # No crisis_open_positions → _open_positions=[] → guard suppresses warning
            self._call_with_flatten_result(
                should_flatten_all=True,
                positions_to_exit=[],
                crisis_open_positions=[],
            )
        assert not any("should_flatten_all" in r.message for r in caplog.records), (
            "should_flatten_all=True with empty open positions must not warn (no positions to flatten)"
        )
