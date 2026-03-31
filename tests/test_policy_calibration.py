"""Tests for M9: Policy Calibration — verifies no TBD values remain and limits are sane."""
from __future__ import annotations
import pytest
from pathlib import Path
import yaml


@pytest.mark.phase12
@pytest.mark.phase13
class TestPolicyCalibration:
    @pytest.fixture
    def policy(self):
        path = Path(__file__).resolve().parents[1] / "configs" / "policy.yaml"
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def test_no_tbd_values(self, policy):
        import json
        text = json.dumps(policy)
        assert "TBD" not in text, "policy.yaml still contains TBD values"

    def test_shorts_allowed_is_bool(self, policy):
        assert isinstance(policy["scope"]["shorts_allowed"], bool)

    def test_target_vol_annual_is_float(self, policy):
        val = policy["risk_limits"]["target_vol_annual"]
        assert isinstance(val, (int, float))
        assert 0.05 <= val <= 0.50

    def test_drawdown_limits_ordered(self, policy):
        dd = policy["risk_limits"]["max_drawdown"]
        assert dd["soft"] < dd["hard"] < dd["kill"]

    def test_max_position_weight_reasonable(self, policy):
        w = policy["risk_limits"]["max_position_weight"]
        assert 0.05 <= w <= 0.50

    def test_turnover_caps_set(self, policy):
        t = policy["risk_limits"]["turnover"]
        assert isinstance(t["weekly_cap"], (int, float))
        assert isinstance(t["daily_cap"], (int, float))
        assert t["daily_cap"] <= t["weekly_cap"]

    def test_concentration_guard_set(self, policy):
        cg = policy["risk_limits"]["concentration_guard"]
        assert isinstance(cg["max_sector_weight"], (int, float))
        assert isinstance(cg["max_corr_cluster_weight"], (int, float))

    def test_state_machine_deactivation_set(self, policy):
        sm = policy["state_machine"]["deactivation"]
        assert isinstance(sm["geo_decay_days"], (int, float))
        assert isinstance(sm["vol_normalization_days"], (int, float))
        assert isinstance(sm["cooldown_days"], (int, float))
