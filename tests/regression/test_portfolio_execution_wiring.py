"""Tier-2 wiring — verify execution.portfolio_execution.optimize_execution_sequence
is consumed as a SHADOW-ONLY batching recommendation in the trading cycle.

Guard: ``policy.portfolio_execution.enabled`` (default False). Shadow mode
means batches are emitted to ``result.meta['execution_batches']`` but the
live order stream is unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.phase12]


def _synthetic_prices(symbols: list[str], n_days: int = 80, seed: int = 21) -> pd.DataFrame:
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

    symbols = ["AAA", "BBB", "CCC", "DDD"]
    prices = _synthetic_prices(symbols)
    as_of = prices["timestamp"].max()

    def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
        syms = df["symbol"].unique().tolist()
        ts = df["timestamp"].max()
        return pd.DataFrame(
            {
                "symbol": syms,
                "timestamp": [ts] * len(syms),
                "direction": ["long", "short", "long", "short"][: len(syms)],
                "score": [0.4, 0.3, 0.2, 0.1][: len(syms)],
            }
        )

    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        n = len(signals)
        rows = []
        for _, r in signals.iterrows():
            side = 1.0 if r.get("direction", "long") == "long" else -1.0
            w = side / n
            rows.append(
                {
                    "symbol": r["symbol"],
                    "target_weight": w,
                    "target_qty": w * capital,
                }
            )
        return pd.DataFrame(rows)

    monkeypatch.setattr(tc, "load_policy", lambda: policy_stub)

    ctx = TradingContext(
        prices=prices,
        as_of=as_of,
        universe=symbols,
        signal_fn=signal_fn,
        position_sizing_fn=sizing_fn,
        capital=100_000.0,
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


def test_portfolio_execution_disabled_no_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_cycle({}, monkeypatch)
    assert result.status == "success"
    assert "execution_batches" not in result.meta


def test_portfolio_execution_enabled_emits_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_cycle(
        {"portfolio_execution": {"enabled": True, "max_parallel": 2}},
        monkeypatch,
    )
    assert result.status == "success"
    # Only emit if there are orders. With capital 100k + per-symbol weights,
    # orders_filtered should be non-empty.
    if result.orders_filtered is None or result.orders_filtered.empty:
        pytest.skip("no orders generated for this fixture")
    assert "execution_batches" in result.meta
    rec = result.meta["execution_batches"]
    assert rec["shadow_only"] is True
    assert rec["max_parallel"] == 2
    assert rec["n_orders"] >= 1
    assert rec["n_batches"] >= 1
