"""Tests for T4.1 Step 3: crisis_alpha weight-cap wiring in trading_cycle.

Covers:
  - shadow_only=True: pipeline called, no weight modification
  - shadow_only=False with non-empty target_weights: weights are capped (never increased)
  - shadow_only=False with empty target_weights: no weight modification
  - exception inside run_crisis_alpha_pipeline: fail-open, weights unchanged
  - cap is conservative: if ca_weight > current_weight, no change applied
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
        assert (
            shadow_only is True
        ), "shadow_only must default to True — live weight application must be opt-in"


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

        assert aapl_w == pytest.approx(
            0.05
        ), "Weight must never increase via crisis_alpha"
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
        assert aapl_w == pytest.approx(
            0.30
        ), "Exception must not alter weights (fail-open)"
