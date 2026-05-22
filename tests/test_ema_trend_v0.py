"""Tests for BENCH-0 EMA trend v0 strategy (signals, targets, paper run)."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.assembled_core.strategies.ema_trend_v0 import (
    compute_signals,
    compute_target_positions,
)

pytestmark = [pytest.mark.unit, pytest.mark.fast]


def _make_prices_uptrend(
    n_days: int = 80,
    start_date: date | None = None,
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Synthetic prices that trend up so EMA20 > EMA60 after warmup."""
    start_date = start_date or date(2024, 1, 1)
    symbols = symbols or ["AAPL", "MSFT"]
    base = 100.0
    rows = []
    for i in range(n_days):
        d = start_date + timedelta(days=i)
        ts = pd.Timestamp(d).tz_localize("UTC")
        for sym in symbols:
            # Linear uptrend + noise so fast EMA stays above slow
            close = base + i * 0.5 + (hash(sym) % 10) * 0.1
            rows.append({"timestamp": ts, "symbol": sym, "close": close})
    df = pd.DataFrame(rows)
    return df


def test_ema_trend_generates_signals_on_uptrend() -> None:
    """With enough history and uptrend, compute_signals returns LONG rows."""
    prices = _make_prices_uptrend(n_days=80, symbols=["AAPL"])
    signals = compute_signals(prices, ema_fast=20, ema_slow=60)
    assert not signals.empty
    assert "timestamp" in signals.columns
    assert "symbol" in signals.columns
    assert "direction" in signals.columns
    assert "score" in signals.columns
    assert (signals["direction"] == "LONG").all()
    assert (signals["score"] > 0).all()  # score = EMA spread, positive in uptrend
    assert "AAPL" in signals["symbol"].tolist()


def test_ema_trend_no_signals_insufficient_history() -> None:
    """With fewer than ema_slow bars, no signals."""
    prices = _make_prices_uptrend(n_days=50, symbols=["AAPL"])
    signals = compute_signals(prices, ema_fast=20, ema_slow=60)
    assert signals.empty
    assert list(signals.columns) == ["timestamp", "symbol", "direction", "score"]


def test_strategy_produces_non_empty_targets_when_trend_up() -> None:
    """When signals exist and prices_latest given, target_positions has target_weight and target_qty."""
    prices = _make_prices_uptrend(n_days=80, symbols=["AAPL", "MSFT"])
    signals = compute_signals(prices, ema_fast=20, ema_slow=60)
    assert not signals.empty
    prices_latest = (
        prices.groupby("symbol", group_keys=False)["close"].last().reset_index()
    )
    targets = compute_target_positions(
        signals,
        total_capital=10000.0,
        equal_weight=True,
        prices_latest=prices_latest,
    )
    assert not targets.empty
    assert "symbol" in targets.columns
    assert "target_weight" in targets.columns
    assert "target_qty" in targets.columns
    assert len(targets) == 2
    assert targets["target_weight"].sum() == pytest.approx(1.0)
    assert (targets["target_qty"] > 0).all()


def test_compute_target_positions_empty_signals() -> None:
    """Empty signals -> empty targets."""
    empty_sigs = pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])
    out = compute_target_positions(
        empty_sigs, 10000.0, equal_weight=True, prices_latest=None
    )
    assert out.empty
    assert list(out.columns) == ["symbol", "target_weight", "target_qty"]


def test_compute_target_positions_no_prices_latest() -> None:
    """With signals but no prices_latest, target_qty is NOTIONAL (> 0)."""
    prices = _make_prices_uptrend(n_days=80, symbols=["AAPL"])
    signals = compute_signals(prices, ema_fast=20, ema_slow=60)
    targets = compute_target_positions(
        signals, 10000.0, equal_weight=True, prices_latest=None
    )
    assert not targets.empty
    assert (targets["target_qty"] > 0).all()  # NOTIONAL = capital * weight
    assert (targets["target_weight"] > 0).all()


def test_paper_run_ema_produces_trades(tmp_path: Path) -> None:
    """Paper range with ema_trend_v0 and enough history produces non-zero orders (smoke)."""
    from src.assembled_core.config import get_base_dir
    from src.assembled_core.ops.paper_runner import run_paper_daily_one
    import json

    # 80 days synthetic data; run 10 days starting after 60 bars so EMA60 has history
    prices = _make_prices_uptrend(n_days=80, symbols=["AAPL", "MSFT"])
    start = date(2024, 1, 1) + timedelta(
        days=60
    )  # day 61–70 so prices <= as_of have 61–70 rows per symbol
    dates = [start + timedelta(days=i) for i in range(10)]
    app_cfg = {
        "paper_runner": {
            "strategy": {
                "name": "ema_trend_v0",
                "ema_fast": 20,
                "ema_slow": 60,
                "equal_weight": True,
            },
            "ledger_path": str(tmp_path / "ledger_state.json"),
        },
        "alerts": {"enabled": False},
    }
    root = get_base_dir()
    total_orders = 0
    for d in dates:
        day_ts = pd.Timestamp(d).tz_localize("UTC")
        out_dir = tmp_path / d.isoformat()
        out_dir.mkdir(parents=True, exist_ok=True)
        exit_code, _ = run_paper_daily_one(
            day_ts, out_dir, "paper", app_cfg, prices, root=root
        )
        assert exit_code == 0
        orders_file = out_dir / "orders_latest.json"
        if orders_file.exists():
            data = json.loads(orders_file.read_text(encoding="utf-8"))
            items = data.get("items") or []
            total_orders += len(items)
    assert total_orders > 0, (
        "Expected at least one order over 10 days with EMA strategy"
    )
