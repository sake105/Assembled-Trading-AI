"""Tests for scripts/forensic/fill_model_audit.py (§8.7)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.forensic.fill_model_audit import (
    BORROW_COST_RANGES,
    INDUSTRY_BASELINES,
    assign_risk_level,
    audit_borrow_costs,
    audit_cost_tiers,
    render_markdown,
    run_fill_model_audit,
)


# ---------------------------------------------------------------------------
# Industry baselines sanity
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestIndustryBaselines:
    def test_all_5_tiers_present(self) -> None:
        expected = {"mega_cap", "large_cap", "mid_cap", "small_cap", "micro_cap"}
        assert set(INDUSTRY_BASELINES.keys()) == expected

    def test_each_tier_has_3_fields(self) -> None:
        for tier, fields in INDUSTRY_BASELINES.items():
            assert set(fields.keys()) == {
                "commission_bps",
                "half_spread_bps",
                "slippage_bps",
            }, f"tier {tier} missing fields"

    def test_min_lt_typical_lt_max(self) -> None:
        for tier, fields in INDUSTRY_BASELINES.items():
            for field, (mn, typ, mx) in fields.items():
                assert mn <= typ <= mx, f"{tier}.{field} ordering broken"

    def test_costs_increase_with_illiquidity(self) -> None:
        """Mega-cap commission_typical < small-cap commission_typical etc."""
        tiers_ordered = ["mega_cap", "large_cap", "mid_cap", "small_cap", "micro_cap"]
        for field in ("commission_bps", "half_spread_bps", "slippage_bps"):
            typicals = [INDUSTRY_BASELINES[t][field][1] for t in tiers_ordered]
            assert typicals == sorted(
                typicals
            ), f"{field} typicals should monotonically increase with illiquidity"


# ---------------------------------------------------------------------------
# audit_cost_tiers
# ---------------------------------------------------------------------------


def _write_cost_tiers(tmp_path: Path, tiers: dict) -> Path:
    p = tmp_path / "cost_tiers.yaml"
    p.write_text(yaml.safe_dump({"tiers": tiers}), encoding="utf-8")
    return p


@pytest.mark.fast
class TestAuditCostTiers:
    def test_all_in_range_no_flags(self, tmp_path: Path) -> None:
        tiers = {
            "mega_cap": {
                "commission_bps": 0.3,
                "half_spread_bps": 1.0,
                "slippage_bps": 1.5,
            }
        }
        path = _write_cost_tiers(tmp_path, tiers)
        result = audit_cost_tiers(path)
        assert result["flags"] == []
        mega = result["tiers"]["mega_cap"]
        for field in ("commission_bps", "half_spread_bps", "slippage_bps"):
            assert mega[field]["verdict"] == "in_range"

    def test_optimistic_below_min(self, tmp_path: Path) -> None:
        # mega_cap commission min is 0.1; set to 0.01
        tiers = {
            "mega_cap": {
                "commission_bps": 0.01,
                "half_spread_bps": 1.0,
                "slippage_bps": 1.5,
            }
        }
        path = _write_cost_tiers(tmp_path, tiers)
        result = audit_cost_tiers(path)
        assert len(result["flags"]) == 1
        assert "optimistic" in result["tiers"]["mega_cap"]["commission_bps"]["verdict"]

    def test_pessimistic_above_max(self, tmp_path: Path) -> None:
        # mega_cap commission max 1.0; set to 50.0
        tiers = {
            "mega_cap": {
                "commission_bps": 50.0,
                "half_spread_bps": 1.0,
                "slippage_bps": 1.5,
            }
        }
        path = _write_cost_tiers(tmp_path, tiers)
        result = audit_cost_tiers(path)
        # Pessimistic does NOT trigger flag (only optimistic does — fail-safe)
        assert result["flags"] == []
        assert result["tiers"]["mega_cap"]["commission_bps"]["verdict"] == "pessimistic"

    def test_unknown_tier_warning(self, tmp_path: Path) -> None:
        tiers = {"alien_tier": {"commission_bps": 0.5}}
        path = _write_cost_tiers(tmp_path, tiers)
        result = audit_cost_tiers(path)
        assert "warning" in result["tiers"]["alien_tier"]

    def test_missing_file_error(self, tmp_path: Path) -> None:
        result = audit_cost_tiers(tmp_path / "nope.yaml")
        assert "error" in result


# ---------------------------------------------------------------------------
# audit_borrow_costs
# ---------------------------------------------------------------------------


def _write_policy(tmp_path: Path, borrow: dict) -> Path:
    p = tmp_path / "policy.yaml"
    p.write_text(yaml.safe_dump({"borrow_costs": borrow}), encoding="utf-8")
    return p


@pytest.mark.fast
class TestAuditBorrowCosts:
    def test_in_range(self, tmp_path: Path) -> None:
        path = _write_policy(
            tmp_path,
            {"enabled": True, "default_rate_bps": 50.0, "htb_rate_bps": 500.0},
        )
        result = audit_borrow_costs(path)
        assert result["flags"] == []
        assert result["fields"]["default_rate_bps"]["verdict"] == "in_range"
        assert result["fields"]["htb_rate_bps"]["verdict"] == "in_range"

    def test_optimistic_easy_borrow(self, tmp_path: Path) -> None:
        path = _write_policy(
            tmp_path,
            {"enabled": True, "default_rate_bps": 1.0, "htb_rate_bps": 500.0},
        )
        result = audit_borrow_costs(path)
        assert len(result["flags"]) == 1
        assert "default_rate_bps" in result["flags"][0]

    def test_missing_borrow_section(self, tmp_path: Path) -> None:
        p = tmp_path / "policy.yaml"
        p.write_text("other_section: {}", encoding="utf-8")
        result = audit_borrow_costs(p)
        assert "enabled" in result  # returns dict but with None enabled

    def test_missing_file_error(self, tmp_path: Path) -> None:
        result = audit_borrow_costs(tmp_path / "nope.yaml")
        assert "error" in result


# ---------------------------------------------------------------------------
# assign_risk_level
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestAssignRiskLevel:
    def test_no_flags_low(self) -> None:
        r = assign_risk_level({"flags": []}, {"flags": []})
        assert r["risk_level"] == "low"
        assert r["n_flags"] == 0

    def test_3_flags_medium(self) -> None:
        r = assign_risk_level({"flags": ["a", "b", "c"]}, {"flags": []})
        assert r["risk_level"] == "medium"

    def test_4plus_flags_high(self) -> None:
        r = assign_risk_level({"flags": ["a", "b", "c", "d"]}, {"flags": ["e"]})
        assert r["risk_level"] == "high"
        assert r["n_flags"] == 5


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


@pytest.mark.fast
class TestRunPipeline:
    def test_real_config_low_verdict(self) -> None:
        """The actual repo config sits in industry ranges → low verdict.
        If this ever flips, an optimistic edit slipped into cost config."""
        if not Path("configs/cost_tiers.yaml").exists():
            pytest.skip("not in repo")
        if not Path("configs/policy.yaml").exists():
            pytest.skip("not in repo")
        report = run_fill_model_audit()
        assert report["verdict"]["risk_level"] == "low"

    def test_json_round_trip(self, tmp_path: Path) -> None:
        cost = _write_cost_tiers(
            tmp_path,
            {
                "mega_cap": {
                    "commission_bps": 0.3,
                    "half_spread_bps": 1.0,
                    "slippage_bps": 1.5,
                }
            },
        )
        policy = _write_policy(
            tmp_path,
            {"enabled": True, "default_rate_bps": 50.0, "htb_rate_bps": 500.0},
        )
        report = run_fill_model_audit(cost, policy)
        s = json.dumps(report)
        rt = json.loads(s)
        assert rt["verdict"]["risk_level"] == "low"

    def test_markdown_renders(self, tmp_path: Path) -> None:
        cost = _write_cost_tiers(
            tmp_path,
            {
                "mega_cap": {
                    "commission_bps": 0.3,
                    "half_spread_bps": 1.0,
                    "slippage_bps": 1.5,
                }
            },
        )
        policy = _write_policy(
            tmp_path,
            {"enabled": True, "default_rate_bps": 50.0, "htb_rate_bps": 500.0},
        )
        report = run_fill_model_audit(cost, policy)
        md = render_markdown(report)
        assert "Fill-Modell-Audit" in md
        assert "Cost-Tier Audit" in md
        assert "Borrow-Cost Audit" in md
        assert "Limitations" in md


# ---------------------------------------------------------------------------
# Borrow ranges sanity
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_borrow_cost_ranges_have_two_fields() -> None:
    assert set(BORROW_COST_RANGES.keys()) == {
        "default_rate_bps_pa",
        "htb_rate_bps_pa",
    }
    for field, (mn, typ, mx) in BORROW_COST_RANGES.items():
        assert mn <= typ <= mx, f"{field} ordering broken"
