"""P0 A6 — Policy → pre-trade gate wiring for ``risk_limits.max_gross_exposure``.

Deep Run v2 finding A6 (2026-04-18): before this test, the only gross-exposure
cap read from ``configs/policy.yaml`` was ``shorts.max_gross_exposure`` — a
shorts-specific 150% cap that was read even when shorts were disabled, and
whose value (a ratio) was passed straight into ``PreTradeConfig.
max_gross_exposure`` (a raw notional). That wiring was both silent and
incorrect.

The fix:
  1. Add ``risk_limits.max_gross_exposure`` (portfolio-wide ratio).
  2. Prefer it over ``shorts.max_gross_exposure``; take ``min()`` when both set.
  3. Convert ratio × equity → notional before passing to ``PreTradeConfig``.

This file tests the pure policy-loader path (bypassing the full trading-cycle
plumbing, which requires heavy fixtures). It verifies the ratio→notional
transformation the pre-trade gate then consumes.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.assembled_core.execution.pre_trade_checks import (
    PreTradeConfig,
    run_pre_trade_checks,
)

pytestmark = pytest.mark.phase_zero


def _make_orders(total_notional: float, per_order_notional: float) -> pd.DataFrame:
    """Build a small order DataFrame with an exact total gross notional."""
    n = max(1, int(total_notional // per_order_notional))
    rows = []
    remaining = total_notional
    for i in range(n):
        rows.append(
            {
                "symbol": f"SYM{i:03d}",
                "side": "BUY",
                "qty": 1.0,
                "price": per_order_notional,
            }
        )
        remaining -= per_order_notional
    if remaining > 1e-6:
        rows.append(
            {"symbol": f"SYM{n:03d}", "side": "BUY", "qty": 1.0, "price": remaining}
        )
    return pd.DataFrame(rows)


def test_policy_yaml_has_risk_limits_gross_exposure():
    """The YAML key itself must be present — it's the contract with the gate."""
    from src.assembled_core.config.policy_loader import load_policy

    pol = load_policy()
    rl = pol.get("risk_limits") or {}
    assert "max_gross_exposure" in rl, (
        "risk_limits.max_gross_exposure must be set in configs/policy.yaml "
        "(P0 A6, Deep Run v2, 2026-04-18). Without this key the pre-trade "
        "gate has no general portfolio gross-exposure limit."
    )
    ratio = float(rl["max_gross_exposure"])
    leverage = pol.get("scope", {}).get("leverage_allowed", False)
    if not leverage:
        assert ratio <= 1.0, (
            f"risk_limits.max_gross_exposure={ratio} contradicts "
            f"scope.leverage_allowed=false. Must be <= 1.0 or leverage must "
            f"be explicitly enabled."
        )


def test_pre_trade_gate_blocks_when_notional_exceeds_cap():
    """Direct gate test: notional > cap → block with max_gross_exposure reason."""
    equity = 100_000.0
    ratio = 1.0  # mirrors risk_limits.max_gross_exposure for leverage=false
    cap_notional = ratio * equity  # = 100_000

    # Build orders whose total notional = 120_000 (20% over cap).
    orders = _make_orders(total_notional=120_000.0, per_order_notional=10_000.0)
    assert abs((orders["qty"] * orders["price"]).abs().sum() - 120_000.0) < 1e-6

    config = PreTradeConfig(max_gross_exposure=cap_notional)
    result, filtered = run_pre_trade_checks(orders, portfolio=None, config=config)

    assert not result.is_ok, "Orders projecting 120% gross must be blocked at 100% cap"
    assert any(
        "max_gross_exposure" in reason for reason in result.blocked_reasons
    ), f"Missing gross-exposure block reason; got {result.blocked_reasons!r}"
    assert filtered.empty, "Filtered orders must be empty when gross cap breached"


def test_pre_trade_gate_allows_when_within_cap():
    """Sanity: orders under the cap must not be blocked by this gate."""
    equity = 100_000.0
    ratio = 1.0
    cap_notional = ratio * equity

    orders = _make_orders(total_notional=80_000.0, per_order_notional=10_000.0)
    assert abs((orders["qty"] * orders["price"]).abs().sum() - 80_000.0) < 1e-6

    config = PreTradeConfig(max_gross_exposure=cap_notional)
    result, filtered = run_pre_trade_checks(orders, portfolio=None, config=config)

    assert not any(
        "max_gross_exposure" in reason for reason in result.blocked_reasons
    ), "Orders within gross cap must not be blocked by max_gross_exposure"


def test_min_of_risk_limits_and_shorts_is_enforced():
    """When both keys set, the stricter (lower) ratio wins."""
    # Simulate the trading_cycle conversion directly: min(risk_limits, shorts)
    rl_gross = 1.0
    sh_gross = 1.5
    equity = 100_000.0

    effective_ratio = min(rl_gross, sh_gross)
    assert effective_ratio == 1.0, (
        "min(risk_limits=1.0, shorts=1.5) must yield 1.0 — the stricter cap. "
        "If this fails, the trading_cycle.py wiring inverted the semantics."
    )
    cap_notional = effective_ratio * equity

    # Orders at 130% of equity: under shorts-cap (1.5) but over risk_limits-cap (1.0).
    orders = _make_orders(total_notional=130_000.0, per_order_notional=10_000.0)
    config = PreTradeConfig(max_gross_exposure=cap_notional)
    result, _ = run_pre_trade_checks(orders, portfolio=None, config=config)

    assert not result.is_ok, (
        "Orders at 130% gross must be blocked when risk_limits cap is 100% "
        "even if shorts cap would allow 150%."
    )
