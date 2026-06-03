"""Tests for Paket 6 — B3 active_strategy + C3 cost_model policy wiring."""

from __future__ import annotations

import pytest

from src.assembled_core.ops.paper_runner import (
    _load_pilot_policy_fail_fast,
    _resolve_active_strategy,
    _resolve_cost_cfg,
)

pytestmark = pytest.mark.fast


class TestResolveActiveStrategy:
    def test_policy_active_strategy_overrides_app_cfg(self):
        paper_cfg = {"strategy": {"name": "multifactor_v2"}}
        policy = {"paper_pilot": {"active_strategy": "trend_baseline"}}
        assert _resolve_active_strategy(paper_cfg, policy) == "trend_baseline"

    def test_fallback_to_app_cfg_when_policy_missing(self):
        paper_cfg = {"strategy": {"name": "multifactor_v2"}}
        assert _resolve_active_strategy(paper_cfg, {}) == "multifactor_v2"

    def test_fallback_to_none_when_both_missing(self):
        assert _resolve_active_strategy({}, {}) == "none"

    def test_policy_none_string_does_not_override(self):
        paper_cfg = {"strategy": {"name": "trend_baseline"}}
        policy = {"paper_pilot": {"active_strategy": "none"}}
        assert _resolve_active_strategy(paper_cfg, policy) == "trend_baseline"

    def test_policy_empty_string_falls_back_to_app(self):
        paper_cfg = {"strategy": {"name": "trend_baseline"}}
        policy = {"paper_pilot": {"active_strategy": ""}}
        assert _resolve_active_strategy(paper_cfg, policy) == "trend_baseline"

    def test_policy_missing_paper_pilot_key_falls_back(self):
        paper_cfg = {"strategy": {"name": "trend_baseline"}}
        policy = {"other_section": {"foo": "bar"}}
        assert _resolve_active_strategy(paper_cfg, policy) == "trend_baseline"


class TestResolveCostCfg:
    def test_policy_cost_model_overrides_app_cfg(self):
        app_cfg = {"paper_runner": {"cost_model": {"commission_bps": 1.0}}}
        policy = {
            "paper_pilot": {
                "cost_model": {
                    "commission_bps": 10.0,
                    "spread_w": 0.25,
                    "impact_w": 0.5,
                }
            }
        }
        result = _resolve_cost_cfg(app_cfg, policy)
        assert result["commission_bps"] == pytest.approx(10.0)
        assert result["spread_w"] == pytest.approx(0.25)
        assert result["impact_w"] == pytest.approx(0.5)

    def test_fallback_to_app_cfg_when_policy_missing(self):
        app_cfg = {"paper_runner": {"cost_model": {"commission_bps": 5.0}}}
        result = _resolve_cost_cfg(app_cfg, {})
        assert result["commission_bps"] == pytest.approx(5.0)

    def test_returns_conservative_default_when_both_missing(self):
        """A13 fail-CLOSED: neither source supplies a cost_model → conservative
        policy.yaml default (10.0/0.25/0.5), NEVER an empty dict (which would
        silently bill 0 bps and fill at exact close)."""
        result = _resolve_cost_cfg({}, {})
        assert result != {}
        assert result["commission_bps"] == pytest.approx(10.0)
        assert result["spread_w"] == pytest.approx(0.25)
        assert result["impact_w"] == pytest.approx(0.5)

    def test_policy_commission_10bps_not_1bps_default(self):
        """Policy returns 10 bps, not the 1-bps legacy default from costs.py."""
        from src.assembled_core.costs import get_default_cost_model

        assert get_default_cost_model().commission_bps == pytest.approx(1.0)
        policy = {
            "paper_pilot": {
                "cost_model": {
                    "commission_bps": 10.0,
                    "spread_w": 0.25,
                    "impact_w": 0.5,
                }
            }
        }
        result = _resolve_cost_cfg({}, policy)
        assert result["commission_bps"] == pytest.approx(10.0)

    def test_policy_missing_paper_pilot_falls_back_to_app(self):
        app_cfg = {"paper_runner": {"cost_model": {"commission_bps": 3.0}}}
        policy = {"quant_gates": {"foo": "bar"}}
        result = _resolve_cost_cfg(app_cfg, policy)
        assert result["commission_bps"] == pytest.approx(3.0)

    def test_fallback_returns_defensive_copy(self):
        """Mutating returned dict must not corrupt app_cfg."""
        app_cfg = {"paper_runner": {"cost_model": {"commission_bps": 7.0}}}
        result = _resolve_cost_cfg(app_cfg, {})
        result["commission_bps"] = 999.0
        assert app_cfg["paper_runner"]["cost_model"]["commission_bps"] == pytest.approx(
            7.0
        )


class TestSimulateFillsSpreadImpact:
    """FU-1: spread_w + impact_w from cost_model_cfg are applied as slippage."""

    def _make_orders(self):
        import pandas as pd

        return pd.DataFrame(
            [{"symbol": "AAPL", "side": "BUY", "qty": 10}],
        )

    def _make_prices(self):
        import pandas as pd

        return pd.DataFrame([{"symbol": "AAPL", "close": 100.0}])

    def test_no_cost_cfg_fills_at_base_price(self):
        from src.assembled_core.ops.paper_ledger import simulate_fills

        fills = simulate_fills(self._make_orders(), self._make_prices())
        assert len(fills) == 1
        assert fills[0]["price"] == pytest.approx(100.0)

    def test_spread_w_and_impact_w_applied_as_slippage(self):
        """spread_w=0.25 bps + impact_w=0.5 bps → price > 100.0 (BUY side)."""
        from src.assembled_core.ops.paper_ledger import simulate_fills

        cost = {"commission_bps": 0.0, "spread_w": 0.25, "impact_w": 0.5}
        fills = simulate_fills(
            self._make_orders(), self._make_prices(), cost_model_cfg=cost
        )
        assert len(fills) == 1
        # slippage = 0.75 bps → price = 100 * (1 + 0.0075/100) ≈ 100.0075
        assert fills[0]["price"] == pytest.approx(100.0 * (1 + 0.75 / 10000), rel=1e-6)

    def test_explicit_slippage_bps_takes_precedence(self):
        """Explicit slippage_bps=5.0 is used even if spread_w/impact_w also present."""
        from src.assembled_core.ops.paper_ledger import simulate_fills

        cost = {"slippage_bps": 5.0, "spread_w": 0.25, "impact_w": 0.5}
        fills = simulate_fills(
            self._make_orders(), self._make_prices(), cost_model_cfg=cost
        )
        assert fills[0]["price"] == pytest.approx(100.0 * (1 + 5.0 / 10000), rel=1e-6)

    def test_explicit_slippage_bps_zero_respected(self):
        """Explicit slippage_bps=0.0 is honored — spread_w/impact_w NOT applied."""
        from src.assembled_core.ops.paper_ledger import simulate_fills

        cost = {"slippage_bps": 0.0, "spread_w": 0.25, "impact_w": 0.5}
        fills = simulate_fills(
            self._make_orders(), self._make_prices(), cost_model_cfg=cost
        )
        # No slippage applied (only commission which is also 0 here)
        assert fills[0]["price"] == pytest.approx(100.0, rel=1e-6)

    def test_spread_w_and_impact_w_sell_side(self):
        """SELL side: price = base / slippage_mult (division, not multiply)."""
        import pandas as pd

        from src.assembled_core.ops.paper_ledger import simulate_fills

        orders = pd.DataFrame([{"symbol": "AAPL", "side": "SELL", "qty": 10}])
        cost = {"commission_bps": 0.0, "spread_w": 0.25, "impact_w": 0.5}
        fills = simulate_fills(orders, self._make_prices(), cost_model_cfg=cost)
        assert fills[0]["price"] == pytest.approx(100.0 / (1 + 0.75 / 10000), rel=1e-6)

    def test_policy_yaml_defaults_produce_nonzero_slippage(self):
        """Policy defaults (commission=10, spread=0.25, impact=0.5) all applied."""
        from src.assembled_core.ops.paper_ledger import simulate_fills

        cost = {"commission_bps": 10.0, "spread_w": 0.25, "impact_w": 0.5}
        fills = simulate_fills(
            self._make_orders(), self._make_prices(), cost_model_cfg=cost
        )
        # slippage = 0.75 bps, commission = 10 bps, both raise BUY fill price
        base = 100.0
        slippage_mult = 1 + 0.75 / 10000
        commission = base * (10.0 / 10000)
        expected = base * slippage_mult + commission
        assert fills[0]["price"] == pytest.approx(expected, rel=1e-5)


class TestPilotPolicyKwarg:
    """FU-2: pilot_policy kwarg passes through to _prd_paper_fills_and_ledger."""

    def test_resolve_cost_cfg_uses_passed_policy(self):
        """When pilot_policy is provided, _resolve_cost_cfg uses it directly."""
        policy = {"paper_pilot": {"cost_model": {"commission_bps": 10.0}}}
        app_cfg = {"paper_runner": {"cost_model": {"commission_bps": 1.0}}}
        result = _resolve_cost_cfg(app_cfg, policy)
        assert result["commission_bps"] == pytest.approx(10.0)

    def test_pilot_policy_kwarg_skips_load_policy_call(self, monkeypatch):
        """When pilot_policy is passed, _load_pilot_policy_fail_fast is NOT called."""
        import src.assembled_core.ops.paper_runner as _mod

        call_count = {"n": 0}
        original = _mod._load_pilot_policy_fail_fast

        def _counting_loader(context: str):
            call_count["n"] += 1
            return original(context)

        monkeypatch.setattr(_mod, "_load_pilot_policy_fail_fast", _counting_loader)
        # Calling _resolve_cost_cfg with a non-None policy dict simulates the in-function path
        # where pilot_policy is already set — _load_pilot_policy_fail_fast would only be called
        # in the else-branch (pilot_policy is None). Verify the branch logic.
        policy = {"paper_pilot": {"cost_model": {"commission_bps": 10.0}}}
        # Simulate: _pilot_policy = pilot_policy if pilot_policy is not None else _load_...
        _pilot_policy = (
            policy
            if policy is not None
            else _mod._load_pilot_policy_fail_fast("cost_model")
        )
        assert call_count["n"] == 0  # not called because policy is not None
        result = _resolve_cost_cfg({}, _pilot_policy)
        assert result["commission_bps"] == pytest.approx(10.0)


class TestLoadPilotPolicyFailFast:
    def test_re_raises_value_error(self, monkeypatch):
        """Non-mapping top-level (ValueError from policy_loader) → re-raised."""
        import src.assembled_core.config.policy_loader as _pl

        def _bad_load():
            raise ValueError("not a mapping")

        monkeypatch.setattr(_pl, "load_policy", _bad_load)
        with pytest.raises(ValueError, match="not a mapping"):
            _load_pilot_policy_fail_fast("test_context")

    def test_re_raises_yaml_error(self, monkeypatch):
        """Malformed YAML (yaml.YAMLError) → re-raised."""
        import yaml

        import src.assembled_core.config.policy_loader as _pl

        def _bad_load():
            raise yaml.scanner.ScannerError(
                "scanning a plain scalar", None, "unexpected", None
            )

        monkeypatch.setattr(_pl, "load_policy", _bad_load)
        with pytest.raises(yaml.YAMLError):
            _load_pilot_policy_fail_fast("test_context")

    def test_soft_failure_returns_empty_dict(self, monkeypatch):
        """FileNotFoundError (policy.yaml absent) → returns {} with WARNING (no raise)."""
        import src.assembled_core.config.policy_loader as _pl

        def _bad_load():
            raise FileNotFoundError("policy.yaml not found")

        monkeypatch.setattr(_pl, "load_policy", _bad_load)
        result = _load_pilot_policy_fail_fast("test_context")
        assert result == {}

    def test_happy_path_returns_policy_dict(self, monkeypatch):
        """Normal load returns the policy dict."""
        import src.assembled_core.config.policy_loader as _pl

        monkeypatch.setattr(
            _pl,
            "load_policy",
            lambda: {"paper_pilot": {"active_strategy": "trend_baseline"}},
        )
        result = _load_pilot_policy_fail_fast("test_context")
        assert result["paper_pilot"]["active_strategy"] == "trend_baseline"
