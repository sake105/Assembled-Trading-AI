"""Tier-2 wiring — verify risk.tail_hedging.recommend_hedge is consumed as
a SHADOW-ONLY recommendation in the trading cycle.

Guard: ``policy.tail_hedging.enabled`` (default False). Shadow mode means
the recommendation is emitted to ``result.meta['tail_hedge_recommendation']``
but orders are never modified.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.phase12]
pytest.importorskip('src.assembled_core.risk.tail_hedging')



def _synthetic_prices(symbols: list[str], n_days: int = 60, seed: int = 17) -> pd.DataFrame:
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

    symbols = ["AAA", "BBB"]
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
                "score": [0.5, 0.5],
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


def test_tail_hedge_disabled_no_recommendation(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_cycle({}, monkeypatch)
    assert result.status == "success"
    assert "tail_hedge_recommendation" not in result.meta


def test_tail_hedge_below_trigger_no_hedge(monkeypatch: pytest.MonkeyPatch) -> None:
    """VIX below trigger → hedge_ratio == 0.0 but recommendation still emitted."""
    result = _run_cycle(
        {"tail_hedging": {"enabled": True, "current_vix": 15.0, "vix_hedge_trigger": 25.0}},
        monkeypatch,
    )
    assert result.status == "success"
    assert "tail_hedge_recommendation" in result.meta
    rec = result.meta["tail_hedge_recommendation"]
    assert rec["hedge_ratio"] == 0.0


def test_tail_hedge_triggered_emits_positive_ratio(monkeypatch: pytest.MonkeyPatch) -> None:
    """VIX above trigger → positive hedge_ratio in shadow mode."""
    result = _run_cycle(
        {
            "tail_hedging": {
                "enabled": True,
                "current_vix": 30.0,
                "vix_hedge_trigger": 25.0,
                "vix_full_hedge_level": 35.0,
                "shadow_only": True,
            }
        },
        monkeypatch,
    )
    assert result.status == "success"
    assert "tail_hedge_recommendation" in result.meta
    rec = result.meta["tail_hedge_recommendation"]
    assert rec["hedge_ratio"] > 0.0
    assert rec["shadow_only"] is True
    assert "VIX" in rec["trigger_reason"]
