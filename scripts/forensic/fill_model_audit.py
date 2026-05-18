"""Fill-Modell-Audit (§8.7).

Reads the in-repo fill / cost configuration (``configs/cost_tiers.yaml`` +
fill-related fields in ``configs/policy.yaml``) and cross-checks every
parameter against an INDUSTRY_BASELINES table of typical broker-realistic
ranges (IBKR Pro / Tastytrade / Alpaca / institutional desk).

Why this matters: §8.7 of the audit notes that the headline baseline
metrics (Sharpe 3.9, CAGR 43%) depend on the fill assumptions being
realistic. If commission_bps is set to 0.2 (close to IBKR Pro retail tier)
but the strategy actually trades small-caps where 5.0 is realistic, the
baseline edge is partly a cost-underestimate artefact.

This script CANNOT verify against actual broker statements (those are
external). It DOES verify the in-repo configuration sits inside plausible
ranges and surfaces specific tier/value combinations that look optimistic.

The verdict is a **risk level** (low / medium / high), NOT a binary
pass/fail. Real C3-063 closure requires real broker-statement vintage
comparison — see KNOWN_ISSUES.md §8.7.

Usage::

    python scripts/forensic/fill_model_audit.py
    python scripts/forensic/fill_model_audit.py --cost-tiers configs/cost_tiers.yaml

Output: JSON + Markdown under ``output/qa/fill_model_audit.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Industry baselines (typical retail / institutional cost ranges in bps)
# ---------------------------------------------------------------------------
#
# Each entry is a (min, typical, max) range in basis points. A configured
# value BELOW min is "optimistic" (potential fill-model under-cost) and
# triggers a flag. A value ABOVE max is "pessimistic" — informational only.
#
# Sources (informal, public):
# - IBKR Pro tiered: $0.0035/share fixed ≈ 0.5-2 bps for $100 stocks
# - IEX/SOR liquidity rebates rebate roughly 1-3 bps for marketable orders
# - Typical retail bid/ask half-spread for mega-cap (SPY/AAPL): 0.5-1 bp
# - For mid-cap: 2-5 bp half-spread
# - For small-cap: 5-15 bp half-spread
# - Slippage: function of order size / ADV; typical 1-10 bps for prudent sizing
# - Borrow cost: 50 bps p.a. (5 bps p.d. annualised x 10y) for easy-to-borrow;
#   500-5000 bps for hard-to-borrow.


INDUSTRY_BASELINES: dict[str, dict[str, tuple[float, float, float]]] = {
    # commission_bps: (min, typical, max)
    "mega_cap": {
        "commission_bps": (0.1, 0.3, 1.0),
        "half_spread_bps": (0.5, 1.0, 2.0),
        "slippage_bps": (0.5, 1.5, 3.0),
    },
    "large_cap": {
        "commission_bps": (0.3, 0.5, 1.5),
        "half_spread_bps": (1.0, 2.0, 4.0),
        "slippage_bps": (1.0, 2.5, 5.0),
    },
    "mid_cap": {
        "commission_bps": (0.5, 1.0, 2.0),
        "half_spread_bps": (2.5, 4.0, 7.0),
        "slippage_bps": (3.0, 5.0, 10.0),
    },
    "small_cap": {
        "commission_bps": (1.0, 1.5, 3.0),
        "half_spread_bps": (4.0, 7.0, 15.0),
        "slippage_bps": (5.0, 10.0, 20.0),
    },
    "micro_cap": {
        "commission_bps": (1.5, 2.5, 5.0),
        "half_spread_bps": (7.0, 12.0, 25.0),
        "slippage_bps": (8.0, 15.0, 40.0),
    },
}


BORROW_COST_RANGES = {
    "default_rate_bps_pa": (25.0, 50.0, 200.0),  # easy-to-borrow annualised
    "htb_rate_bps_pa": (200.0, 500.0, 5000.0),  # hard-to-borrow annualised
}


# ---------------------------------------------------------------------------
# Audit functions
# ---------------------------------------------------------------------------


def audit_cost_tiers(cost_tiers_path: Path) -> dict[str, Any]:
    """Cross-check every (tier, field) value against INDUSTRY_BASELINES."""
    if not cost_tiers_path.exists():
        return {"error": f"cost tiers config not found: {cost_tiers_path}"}
    with open(cost_tiers_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tiers = cfg.get("tiers", {})
    audit: dict[str, Any] = {"tiers": {}, "flags": []}
    for tier_name, tier_cfg in tiers.items():
        baselines = INDUSTRY_BASELINES.get(tier_name)
        if baselines is None:
            audit["tiers"][tier_name] = {"warning": "no industry baseline for tier"}
            continue
        per_field: dict[str, Any] = {}
        for field, (min_v, typical_v, max_v) in baselines.items():
            actual = tier_cfg.get(field)
            if actual is None:
                per_field[field] = {"error": "not in config"}
                continue
            actual_f = float(actual)
            if actual_f < min_v:
                verdict = "optimistic"
                audit["flags"].append(
                    f"{tier_name}.{field}={actual_f} bps below industry min "
                    f"{min_v} bps (typical {typical_v}, max {max_v})"
                )
            elif actual_f > max_v:
                verdict = "pessimistic"
            else:
                verdict = "in_range"
            per_field[field] = {
                "actual": actual_f,
                "industry_min": min_v,
                "industry_typical": typical_v,
                "industry_max": max_v,
                "verdict": verdict,
            }
        audit["tiers"][tier_name] = per_field
    return audit


def audit_borrow_costs(policy_path: Path) -> dict[str, Any]:
    """Cross-check borrow_costs.default_rate_bps + htb_rate_bps."""
    if not policy_path.exists():
        return {"error": f"policy config not found: {policy_path}"}
    with open(policy_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    bc = cfg.get("borrow_costs", {})
    flags: list[str] = []
    out: dict[str, Any] = {"enabled": bc.get("enabled"), "fields": {}}
    for field, (min_v, typical_v, max_v) in BORROW_COST_RANGES.items():
        # cost_tiers field name vs policy field name (drop "_pa" suffix)
        policy_field = field.replace("_pa", "")
        actual = bc.get(policy_field)
        if actual is None:
            out["fields"][policy_field] = {"error": "not in config"}
            continue
        actual_f = float(actual)
        if actual_f < min_v:
            verdict = "optimistic"
            flags.append(
                f"borrow_costs.{policy_field}={actual_f} bps below industry "
                f"min {min_v} bps (typical {typical_v}, max {max_v})"
            )
        elif actual_f > max_v:
            verdict = "pessimistic"
        else:
            verdict = "in_range"
        out["fields"][policy_field] = {
            "actual": actual_f,
            "industry_min": min_v,
            "industry_typical": typical_v,
            "industry_max": max_v,
            "verdict": verdict,
        }
    out["flags"] = flags
    return out


# ---------------------------------------------------------------------------
# Verdict aggregation
# ---------------------------------------------------------------------------


def assign_risk_level(
    cost_audit: dict[str, Any], borrow_audit: dict[str, Any]
) -> dict[str, Any]:
    cost_flags = cost_audit.get("flags", [])
    borrow_flags = borrow_audit.get("flags", [])
    all_flags = list(cost_flags) + list(borrow_flags)
    n = len(all_flags)
    if n == 0:
        verdict = "low"
    elif n <= 3:
        verdict = "medium"
    else:
        verdict = "high"
    return {
        "risk_level": verdict,
        "n_flags": n,
        "flags": all_flags,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_fill_model_audit(
    cost_tiers_path: Path = Path("configs/cost_tiers.yaml"),
    policy_path: Path = Path("configs/policy.yaml"),
) -> dict[str, Any]:
    """Read both configs, cross-check, aggregate verdict."""
    cost_audit = audit_cost_tiers(cost_tiers_path)
    borrow_audit = audit_borrow_costs(policy_path)
    verdict = assign_risk_level(cost_audit, borrow_audit)
    return {
        "cost_tiers_path": str(cost_tiers_path),
        "policy_path": str(policy_path),
        "cost_tier_audit": cost_audit,
        "borrow_cost_audit": borrow_audit,
        "verdict": verdict,
        "limitations": (
            "This audit cross-checks the in-repo fill configuration against "
            "INDUSTRY_BASELINES (informal public ranges, not exhaustive). It "
            "does NOT compare against actual broker statements — that's the "
            "external follow-up. A 'low' verdict means the config sits inside "
            "plausible industry ranges; it does NOT prove the fills the "
            "strategy WOULD experience match the modeled fills."
        ),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fill-Modell-Audit (§8.7)",
        "",
        f"**Cost tiers:** `{report['cost_tiers_path']}`",
        f"**Policy:** `{report['policy_path']}`",
        "",
        f"## Verdict: `{report['verdict']['risk_level']}` "
        f"({report['verdict']['n_flags']} flag(s))",
        "",
    ]
    for f in report["verdict"]["flags"]:
        lines.append(f"- {f}")
    if not report["verdict"]["flags"]:
        lines.append("(no fill-model flags triggered)")
    lines.append("")
    lines.append("## Cost-Tier Audit")
    lines.append("")
    cost = report["cost_tier_audit"]
    if "error" in cost:
        lines.append(f"- ERROR: {cost['error']}")
    else:
        for tier_name, tier_audit in cost["tiers"].items():
            lines.append(f"### Tier: {tier_name}")
            if "warning" in tier_audit:
                lines.append(f"- {tier_audit['warning']}")
                continue
            lines.append("| Field | Actual | Industry Min | Typical | Max | Verdict |")
            lines.append("|------:|-------:|-------------:|--------:|----:|:--------|")
            for field, info in tier_audit.items():
                if "error" in info:
                    lines.append(f"| {field} | — | — | — | — | {info['error']} |")
                    continue
                lines.append(
                    f"| {field} | {info['actual']} | {info['industry_min']} | "
                    f"{info['industry_typical']} | {info['industry_max']} | "
                    f"`{info['verdict']}` |"
                )
            lines.append("")
    lines.append("## Borrow-Cost Audit")
    lines.append("")
    bc = report["borrow_cost_audit"]
    if "error" in bc:
        lines.append(f"- ERROR: {bc['error']}")
    else:
        lines.append(f"- **Enabled:** {bc.get('enabled')}")
        lines.append("")
        lines.append("| Field | Actual | Industry Min | Typical | Max | Verdict |")
        lines.append("|------:|-------:|-------------:|--------:|----:|:--------|")
        for field, info in bc["fields"].items():
            if "error" in info:
                lines.append(f"| {field} | — | — | — | — | {info['error']} |")
                continue
            lines.append(
                f"| {field} | {info['actual']} | {info['industry_min']} | "
                f"{info['industry_typical']} | {info['industry_max']} | "
                f"`{info['verdict']}` |"
            )
    lines.append("")
    lines.append("## Limitations")
    lines.append(report["limitations"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cost-tiers",
        type=Path,
        default=Path("configs/cost_tiers.yaml"),
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("configs/policy.yaml"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/qa"),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    report = run_fill_model_audit(args.cost_tiers, args.policy)
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / "fill_model_audit.json"
    md_path = args.out / "fill_model_audit.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    logger.info("[fill_model_audit] JSON: %s", json_path)
    logger.info("[fill_model_audit] Markdown: %s", md_path)
    logger.info(
        "[fill_model_audit] verdict=%s flags=%d",
        report["verdict"]["risk_level"],
        report["verdict"]["n_flags"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
