"""Tier-2 wiring — verify risk.attribution.compute_attribution_report is
consumed as a post-sizing meta-enrichment in the trading cycle.

The enrichment is policy-flag-guarded (``policy.attribution.enabled``) and
defaults to OFF. When ON and target_positions are non-empty, the cycle
writes ``result.meta['attribution']`` with per-symbol return/vol contributions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.phase12]


def _synthetic_prices(symbols: list[str], n_days: int = 60, seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_days, freq="B", tz="UTC")
    rows = []
    for i, sym in enumerate(symbols):
        sigma = 0.01 + 0.003 * i
        noise = rng.normal(0.0, sigma, n_days)
        close = 100.0 * np.exp(np.cumsum(noise))
        for ts, c in zip(idx, close):
            rows.append({"timestamp": ts, "symbol": sym, "close": float(c)})
    return pd.DataFrame(rows)


def _run_cycle_with_policy(policy_stub: dict, monkeypatch: pytest.MonkeyPatch):
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
                "score": [0.4, 0.35, 0.25][: len(syms)],
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
        capital=10_000.0,
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


def test_attribution_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_cycle_with_policy({}, monkeypatch)
    assert result.status == "success"
    assert "attribution" not in result.meta


def test_attribution_enabled_populates_meta(monkeypatch: pytest.MonkeyPatch) -> None:
    result = _run_cycle_with_policy(
        {"attribution": {"enabled": True, "lookback_days": 40}}, monkeypatch
    )
    assert result.status == "success", f"status={result.status}, err={result.error_message}"
    assert "attribution" in result.meta
    attr = result.meta["attribution"]
    # Status can be "ok" (with sufficient data), "insufficient_data", or "no_price_data"
    assert attr.get("status") in {"ok", "insufficient_data", "no_price_data"}
    # With PSD synthetic data, ok expected
    assert attr.get("status") == "ok"
    assert "return_contributions" in attr
    assert "vol_contributions" in attr
    assert attr["portfolio_vol"] is not None
    # Return-contribution sum should equal portfolio_return
    rc_sum = sum(attr["return_contributions"].values())
    assert abs(rc_sum - float(attr["portfolio_return"])) < 1e-9
