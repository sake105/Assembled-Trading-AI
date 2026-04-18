"""Tier-2 wiring — verify portfolio.risk_budgeting.compute_erc_weights is
consumed as an alternative sizing method in the trading cycle.

The dispatch is policy-driven (``policy.position_sizing.method == "erc"``)
and defaults to the caller's ``position_sizing_fn`` when absent. When ON
and the sizing preconditions are met, ``target_positions`` is filled from
the ERC optimizer and ``result.meta["erc_sizing"]`` is populated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytestmark = [pytest.mark.phase12]


def _synthetic_prices(symbols: list[str], n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n_days, freq="B", tz="UTC")
    rows = []
    for i, sym in enumerate(symbols):
        sigma = 0.01 + 0.005 * i
        noise = rng.normal(0.0, sigma, n_days)
        close = 100.0 * np.exp(np.cumsum(noise))
        for ts, c in zip(idx, close):
            rows.append({"timestamp": ts, "symbol": sym, "close": float(c)})
    return pd.DataFrame(rows)


def test_compute_erc_weights_returns_valid_allocation() -> None:
    """Direct check: ERC optimizer yields simplex allocation on PSD covariance."""
    from src.assembled_core.portfolio.risk_budgeting import compute_erc_weights

    symbols = ["AAA", "BBB", "CCC"]
    prices = _synthetic_prices(symbols)
    pivot = prices.pivot_table(index="timestamp", columns="symbol", values="close")
    rets = pivot.pct_change().dropna()
    cov = rets.cov() * 252.0

    result = compute_erc_weights(cov, symbols=symbols, long_only=True, max_weight=0.6)

    assert result.converged
    total_w = sum(result.weights.values())
    assert abs(total_w - 1.0) < 1e-6
    assert all(0.0 <= w <= 0.6 + 1e-6 for w in result.weights.values())
    # ERC with near-default settings should produce reasonably balanced risk
    rc_values = list(result.risk_contributions.values())
    assert max(rc_values) - min(rc_values) < 0.5


def test_erc_sizing_dispatch_fires_when_policy_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Integration: trading_cycle sizing dispatch picks ERC when policy sets it."""
    from src.assembled_core.pipeline import trading_cycle as tc
    from src.assembled_core.pipeline.trading_cycle import TradingContext, run_trading_cycle

    symbols = ["AAA", "BBB", "CCC"]
    prices = _synthetic_prices(symbols, n_days=60)
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
        # Default fallback — should NOT be used when ERC dispatch succeeds.
        return pd.DataFrame(
            {
                "symbol": signals["symbol"],
                "target_weight": 1.0 / len(signals),
                "target_qty": capital / len(signals),
            }
        )

    policy_stub = {
        "position_sizing": {"method": "erc", "max_weight": 0.5},
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

    # Bypass default feature-building (has unrelated API drift) and preserve
    # full price history for ERC covariance (PIT-safe up to as_of).
    hooks = {
        "load_prices": lambda _ctx: (
            prices[prices["timestamp"] <= as_of].copy(),
            prices[prices["timestamp"] == prices["timestamp"].max()].copy(),
        ),
        "build_features": lambda _ctx, df: df,
    }
    result = run_trading_cycle(ctx, hooks=hooks)

    assert result.status == "success", f"status={result.status}, err={result.error_message}"
    assert result.meta.get("sizing_method") == "erc"
    # ERC meta must be populated (not empty) when optimizer actually ran
    assert "erc_sizing" in result.meta
    assert result.meta["erc_sizing"].get("method") in {"scipy_slsqp", "inverse_vol_fallback"}
