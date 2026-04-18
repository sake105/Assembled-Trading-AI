"""Tier-2 wiring — verify execution.almgren_chriss.estimate_impact_cost is
consumed as a SHADOW-ONLY pre-trade cost estimate in the trading cycle.

Guard: ``policy.almgren_chriss.enabled`` (default False). Shadow mode means
the estimate is emitted to ``result.meta['almgren_chriss_impact']`` but does
not replace the paper-engine's fill model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.phase12]


def _synthetic_prices(symbols: list[str], n_days: int = 80, seed: int = 23) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_days, freq="B", tz="UTC")
    rows = []
    for i, sym in enumerate(symbols):
        sigma = 0.01 + 0.002 * i
        noise = rng.normal(0.0, sigma, n_days)
        close = 100.0 * np.exp(np.cumsum(noise))
        for ts, c in zip(idx, close):
            rows.append({"timestamp": ts, "symbol": sym, "close": float(c)})
    return pd.DataFrame(rows)


def _run_cycle(policy_stub: dict, monkeypatch: pytest.MonkeyPatch):
    from src.assembled_core.pipeline import trading_cycle as tc
    from src.assembled_core.pipeline.trading_cycle import TradingContext, run_trading_cycle

    symbols = ["AAA", "BBB", "CCC"]
    prices = _synthetic_prices(symbols)
    as_of = prices["timestamp"].max()

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        syms = df["symbol"].unique().tolist()
        ts = df["timestamp"].max()
        return pd.DataFrame(
            {
                "symbol": syms,
                "timestamp": [ts] * len(syms),
                "direction": ["long"] * len(syms),
                "score": [0.5, 0.3, 0.2][: len(syms)],
            }
        )

    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        n = len(signals)
        return pd.DataFrame(
            {
                "symbol": signals["symbol"],
                "target_weight": [1.0 / n] * n,
                "target_qty": [capital / n] * n,
            }
        )

    monkeypatch.setattr(tc, "load_policy", lambda: policy_stub)

    ctx = TradingContext(
        prices=prices,
        as_of=as_of,
        universe=symbols,
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        capital=1_000_000.0,
        enable_risk_controls=False,
        write_outputs=False,
    )

    hooks = {
        "load_prices": lambda _ctx: (
            prices[prices["timestamp"] <= as_of].copy(),
            prices[prices["timestamp"] == prices["timestamp"].max()].copy(),
        ),
        "build_features": lambda _ctx, df: df,
    }
    return run_trading_cycle(ctx, hooks=hooks)


def test_almgren_chriss_disabled_no_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_cycle({}, monkeypatch)
    assert result.status == "success"
    assert "almgren_chriss_impact" not in result.meta


def test_almgren_chriss_enabled_emits_impact(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_cycle(
        {
            "almgren_chriss": {
                "enabled": True,
                "default_adv": 500_000.0,
                "gamma": 0.1,
                "eta": 0.05,
                "horizon_days": 1.0,
            }
        },
        monkeypatch,
    )
    assert result.status == "success"
    if result.orders_filtered is None or result.orders_filtered.empty:
        pytest.skip("no orders generated for this fixture")
    assert "almgren_chriss_impact" in result.meta
    rec = result.meta["almgren_chriss_impact"]
    assert rec["shadow_only"] is True
    assert rec["total_notional_usd"] > 0.0
    assert rec["aggregate_bps"] >= 0.0
    assert isinstance(rec["per_order"], list)
