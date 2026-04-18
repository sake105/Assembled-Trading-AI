"""Tier-1 wiring — verify risk.var_methods is consumed as a pre-trade gate
in the trading cycle through ``_evaluate_var_gate``.

The gate is policy-flag-guarded (``policy.risk.var_gate.enabled``) and
defaults to OFF, so existing callers see no behavior change. When ON and
the configured threshold is breached, a decision dict is returned so the
orchestrator can empty the filtered orders for that cycle.
"""

from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.phase12]

from src.assembled_core.pipeline.trading_cycle import (  # noqa: E402
    _evaluate_var_gate,
)


def _synthetic_prices(n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_days, freq="B", tz="UTC")
    rows = []
    for sym, sigma in [("AAA", 0.01), ("BBB", 0.015)]:
        noise = rng.normal(0.0, sigma, n_days)
        close = 100.0 * np.exp(np.cumsum(noise))
        for ts, c in zip(idx, close):
            rows.append({"timestamp": ts, "symbol": sym, "close": float(c)})
    return pd.DataFrame(rows)


def _make_ctx(prices: pd.DataFrame) -> types.SimpleNamespace:
    return types.SimpleNamespace(prices=prices)


def _make_result(symbols: list[str]) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        target_positions=pd.DataFrame({"symbol": symbols}),
        meta={},
    )


def test_var_gate_disabled_returns_none() -> None:
    ctx = _make_ctx(_synthetic_prices())
    result = _make_result(["AAA", "BBB"])
    assert _evaluate_var_gate(ctx, result, policy={}) is None
    assert (
        _evaluate_var_gate(
            ctx, result, policy={"risk": {"var_gate": {"enabled": False}}}
        )
        is None
    )


def test_var_gate_enabled_high_threshold_passes() -> None:
    ctx = _make_ctx(_synthetic_prices())
    result = _make_result(["AAA", "BBB"])
    decision = _evaluate_var_gate(
        ctx,
        result,
        policy={
            "risk": {
                "var_gate": {"enabled": True, "max_var_pct": 0.50, "confidence": 0.95}
            }
        },
    )
    assert decision is None  # far above realistic 1-day VaR


def test_var_gate_enabled_low_threshold_breaches() -> None:
    ctx = _make_ctx(_synthetic_prices())
    result = _make_result(["AAA", "BBB"])
    decision = _evaluate_var_gate(
        ctx,
        result,
        policy={
            "risk": {
                "var_gate": {
                    "enabled": True,
                    "max_var_pct": 0.0001,
                    "confidence": 0.95,
                }
            }
        },
    )
    assert decision is not None
    assert decision["breach"] is True
    assert decision["var_1d"] > decision["max_var_pct"]
