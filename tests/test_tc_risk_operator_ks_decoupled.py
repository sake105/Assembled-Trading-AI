"""R2-2 regression: the OPERATOR kill switch is decoupled from the generic
``enable_risk_controls`` flag.

Before R2-2 a single ``enable_risk_controls=False`` (a flag whose legitimate use
is backtest parity) took an EARLY pass-through in ``_tc_risk.check_risk`` that
skipped EVERY gate — including the operator kill switch. So a live/paper run
misconfigured with that flag off would silently ignore an operator HALT.

R2-2 keeps the generic flag gating the numeric/churn gate stack, but in any
REAL trading mode (eod / paper / live) the orders are still routed through the
standalone operator kill-switch guard. ``backtest`` mode keeps the pure
pass-through on purpose — reading the *current* live HALT state during a
historical replay would be a wrong-context read and break replay determinism.

KS state is isolated to ``tmp_path`` via the ``ASSEMBLED_KILL_SWITCH_*`` env
overrides so these tests never read or pollute the shared real state file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.assembled_core.execution.kill_switch as ks  # noqa: E402
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
    dates = pd.date_range("2024-03-01", periods=n_days, freq="B", tz="UTC")
    rows = []
    for sym in SYMBOLS:
        for i, ts in enumerate(dates):
            rows.append({"timestamp": ts, "symbol": sym, "close": 100.0 + i})
    return pd.DataFrame(rows)


def _make_ctx(mode: str) -> TradingContext:
    prices = _make_prices()
    ctx = TradingContext(
        prices=prices,
        as_of=prices["timestamp"].max(),
        mode=mode,
        signal_fn=_signal_fn,
        position_sizing_fn=_sizing_fn,
        use_factor_store=False,
        write_outputs=False,
        capital=100_000.0,
        intel_sim_applied=True,
        enable_risk_controls=False,  # the flag whose total-bypass R2-2 fixes
    )
    ctx.qa_block_trading = False
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


def _isolate_ks(monkeypatch, tmp_path: Path, *, engaged: bool) -> None:
    """Point all kill-switch paths at tmp + control engagement via env."""
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_STATE", str(tmp_path / "ks_state.json"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_SENTINEL", str(tmp_path / "ks_sentinel"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_AUDIT", str(tmp_path / "ks_audit.jsonl"))
    monkeypatch.setenv("ASSEMBLED_KILL_SWITCH_LOCK", str(tmp_path / "ks.lock"))
    if engaged:
        monkeypatch.setenv("ASSEMBLED_KILL_SWITCH", "1")  # env source → throttle 0.0
    else:
        monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)


def test_backtest_mode_pure_passthrough_ignores_operator_ks(
    monkeypatch, tmp_path: Path
) -> None:
    """backtest mode must NOT read the live HALT — orders pass through even when
    the operator kill switch is engaged (replay determinism)."""
    _isolate_ks(monkeypatch, tmp_path, engaged=True)
    out = check_risk(_orders(), _result(), _make_ctx("backtest"))
    assert len(out.orders_filtered) == 3, "backtest must keep pure pass-through"
    assert out.meta.get("operator_kill_switch_blocked") is not True


def test_eod_mode_blocks_when_operator_ks_engaged(monkeypatch, tmp_path: Path) -> None:
    """eod is a REAL trading mode (paper_runner) — an engaged operator HALT must
    block even with enable_risk_controls=False."""
    _isolate_ks(monkeypatch, tmp_path, engaged=True)
    out = check_risk(_orders(), _result(), _make_ctx("eod"))
    assert out.orders_filtered.empty, "engaged operator KS must block in eod mode"
    assert out.meta["operator_kill_switch_blocked"] is True


def test_paper_mode_blocks_when_operator_ks_engaged(
    monkeypatch, tmp_path: Path
) -> None:
    """paper mode (run_live_paper / paper_track) — engaged HALT must block."""
    _isolate_ks(monkeypatch, tmp_path, engaged=True)
    out = check_risk(_orders(), _result(), _make_ctx("paper"))
    assert out.orders_filtered.empty
    assert out.meta["operator_kill_switch_blocked"] is True


def test_eod_mode_passes_when_operator_ks_not_engaged(
    monkeypatch, tmp_path: Path
) -> None:
    """The common case: no HALT engaged → eod pass-through is a no-op (this is why
    the existing mode='eod' backtest-parity tests stay green)."""
    _isolate_ks(monkeypatch, tmp_path, engaged=False)
    out = check_risk(_orders(), _result(), _make_ctx("eod"))
    assert len(out.orders_filtered) == 3
    assert out.meta.get("operator_kill_switch_blocked") is not True


def test_operator_ks_check_error_fails_closed(monkeypatch, tmp_path: Path) -> None:
    """If the operator kill-switch evaluation raises, block fail-closed (R2-1
    consistency) rather than pass unchecked orders through."""
    _isolate_ks(monkeypatch, tmp_path, engaged=False)

    def _boom(*args, **kwargs):
        raise RuntimeError("ks state read blew up")

    monkeypatch.setattr(ks, "guard_orders_with_kill_switch", _boom)
    out = check_risk(_orders(), _result(), _make_ctx("eod"))
    assert out.orders_filtered.empty, "KS eval error must fail-closed"
    assert out.meta["risk_gate_error"] is True
    assert out.meta["operator_kill_switch"]["status"] == "error"
