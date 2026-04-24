"""Tier-2 wiring — verify portfolio.hrp_sizing.apply_hrp_sizing is consumed
as an alternative sizing method in the trading cycle.

Dispatch: ``policy.position_sizing.method == "hrp"``. Default falls back to
``position_sizing_fn``. When ON, ``result.meta['hrp_sizing']`` is populated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.phase12]
pytest.importorskip('src.assembled_core.portfolio.hrp_sizing')



def _synthetic_prices(symbols: list[str], n_days: int = 120, seed: int = 11) -> pd.DataFrame:
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


def test_hrp_sizing_dispatch_fires(monkeypatch: pytest.MonkeyPatch) -> None:
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
                "direction": ["long"] * len(syms),
                "score": [0.4, 0.3, 0.2, 0.1][: len(syms)],
            }
        )

    def sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        # Score-weighted sizing (base for HRP blend)
        scores = signals["score"].astype(float)
        total = scores.sum() or 1.0
        w = scores / total
        return pd.DataFrame(
            {
                "symbol": signals["symbol"],
                "target_weight": w.tolist(),
                "target_qty": (w * capital).round(2).tolist(),
            }
        )

    policy_stub = {
        "position_sizing": {
            "method": "hrp",
            "lookback_days": 60,
            "blend": 0.7,
            "target_invested_pct": 1.0,
        }
    }
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
    result = run_trading_cycle(ctx, hooks=hooks)

    assert result.status == "success", f"status={result.status}, err={result.error_message}"
    assert result.meta.get("sizing_method") == "hrp"
    assert "hrp_sizing" in result.meta
    assert result.meta["hrp_sizing"]["n_symbols"] >= 2
    # Weights sum to ~1.0 (target_invested_pct=1.0)
    w_sum = float(result.target_positions["target_weight"].sum())
    assert abs(w_sum - 1.0) < 1e-3
