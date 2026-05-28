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

    def test_returns_empty_dict_when_both_missing(self):
        assert _resolve_cost_cfg({}, {}) == {}

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
