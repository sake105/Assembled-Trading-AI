"""R2-1 regression: pre-trade risk gates must FAIL-CLOSED on evaluation error.

The four safety gates in ``_tc_risk.check_risk`` — VaR gate, auto-drawdown kill
switch, intraday circuit breaker, and the fat-finger hard cap — previously
failed OPEN: on an evaluation exception they logged and no-op'd, letting the
order batch pass UNCHECKED. Fail-closed means: if we cannot prove the batch is
within a safety limit, we block it. These tests force each gate to raise and
assert the batch is emptied + the failure is surfaced via
``result.meta["risk_gate_error"]``.

Out of scope by design: the churn-reduction filters (anti-churn deadzone /
min-notional) and the audit-only lifecycle tracking are intentionally NOT
fail-closed — a cosmetic churn filter erroring must not halt legitimate trading.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.assembled_core.pipeline._tc_risk as tc_risk  # noqa: E402
from src.assembled_core.pipeline._tc_risk import check_risk  # noqa: E402
from src.assembled_core.pipeline.trading_cycle_shared import (  # noqa: E402
    TradingContext,
    TradingCycleResult,
)

SYMBOLS = ["AAPL", "MSFT", "GOOG"]


def _signal_fn(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"])


def _sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])


def _make_prices(n_days: int = 5) -> pd.DataFrame:
    # < 60 rows so the EVT/copula tail steps skip (they require >= 60 return
    # rows), keeping the test focused on the post-control safety gates.
    dates = pd.date_range("2024-03-01", periods=n_days, freq="B", tz="UTC")
    rows = []
    for sym in SYMBOLS:
        for i, ts in enumerate(dates):
            rows.append({"timestamp": ts, "symbol": sym, "close": 100.0 + i})
    return pd.DataFrame(rows)


def _make_ctx() -> TradingContext:
    prices = _make_prices()
    ctx = TradingContext(
        prices=prices,
        as_of=prices["timestamp"].max(),
        mode="backtest",
        signal_fn=_signal_fn,
        position_sizing_fn=_sizing_fn,
        use_factor_store=False,
        write_outputs=False,
        capital=100_000.0,
        intel_sim_applied=True,
    )
    ctx.qa_block_trading = False
    # Empty policy → the anti-churn + fat-finger policy-gated steps skip by
    # default, isolating the gate under test.
    ctx._policy_cache = {}
    return ctx


def _orders() -> pd.DataFrame:
    ts = pd.Timestamp("2024-03-08", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": [ts] * 3,
            "symbol": SYMBOLS,
            "side": ["buy", "buy", "buy"],
            "qty": [100.0, 80.0, 60.0],
            "price": [150.0, 300.0, 140.0],
        }
    )


def _result() -> TradingCycleResult:
    return TradingCycleResult(
        run_id=None, timestamp=pd.Timestamp.now("UTC"), status="success"
    )


def _boom(*args, **kwargs):
    raise RuntimeError("gate evaluation blew up")


def _patch_passthrough(monkeypatch) -> None:
    """Force orders_filtered non-empty into the gates and no-op all three
    ``_evaluate_*`` gates so each test isolates exactly ONE gate."""
    monkeypatch.setattr(
        tc_risk, "_apply_risk_controls_default", lambda ctx, orders: orders.copy()
    )
    monkeypatch.setattr(tc_risk, "_evaluate_var_gate", lambda *a, **k: None)
    monkeypatch.setattr(tc_risk, "_evaluate_auto_dd_kill_switch", lambda *a, **k: None)
    monkeypatch.setattr(tc_risk, "_evaluate_circuit_breaker", lambda *a, **k: None)


def test_no_gate_error_preserves_orders(monkeypatch, tmp_path: Path) -> None:
    """Baseline: with every safety gate OK, the batch is NOT blocked."""
    _patch_passthrough(monkeypatch)
    ctx = _make_ctx()
    ctx.output_dir = str(tmp_path)  # keep the lifecycle-log hook out of the repo
    out = check_risk(_orders(), _result(), ctx)
    assert len(out.orders_filtered) == 3
    assert out.meta.get("risk_gate_error") is not True


def test_var_gate_error_blocks_orders(monkeypatch) -> None:
    _patch_passthrough(monkeypatch)
    monkeypatch.setattr(tc_risk, "_evaluate_var_gate", _boom)
    out = check_risk(_orders(), _result(), _make_ctx())
    assert out.orders_filtered.empty, "var-gate error must block all orders"
    assert out.meta["risk_gate_error"] is True
    assert out.meta["var_gate"]["status"] == "error"


def test_auto_dd_error_blocks_orders(monkeypatch) -> None:
    _patch_passthrough(monkeypatch)
    monkeypatch.setattr(tc_risk, "_evaluate_auto_dd_kill_switch", _boom)
    out = check_risk(_orders(), _result(), _make_ctx())
    assert out.orders_filtered.empty
    assert out.meta["risk_gate_error"] is True
    assert out.meta["auto_dd_kill_switch"]["status"] == "error"


def test_circuit_breaker_error_blocks_orders(monkeypatch) -> None:
    _patch_passthrough(monkeypatch)
    monkeypatch.setattr(tc_risk, "_evaluate_circuit_breaker", _boom)
    out = check_risk(_orders(), _result(), _make_ctx())
    assert out.orders_filtered.empty
    assert out.meta["risk_gate_error"] is True
    assert out.meta["circuit_breaker"]["status"] == "error"


def test_fat_finger_error_blocks_orders(monkeypatch) -> None:
    _patch_passthrough(monkeypatch)
    import src.assembled_core.execution.fat_finger_guard as ffg

    monkeypatch.setattr(ffg, "apply_fat_finger_guard_from_policy", _boom)
    ctx = _make_ctx()
    ctx._policy_cache = {"fat_finger_guard": {"enabled": True}}
    out = check_risk(_orders(), _result(), ctx)
    assert out.orders_filtered.empty
    assert out.meta["risk_gate_error"] is True
    assert out.meta["fat_finger_guard"]["status"] == "error"
