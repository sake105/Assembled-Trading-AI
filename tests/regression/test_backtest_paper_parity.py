"""E0.1 — Backtest ↔ Paper parity plumbing regression.

This test suite locks in the **plumbing** for the E0.1 parity gate:

* ``run_portfolio_backtest`` / ``make_cycle_fn`` accept
  ``enable_risk_controls`` with default ``True`` (was hardcoded ``False``).
* ``TradingContext`` exposes ``kill_switch_persist`` with default ``True``
  (was implicit ``False`` via an always-on bar-restore finally-block in
  ``trading_cycle.py``).
* ``run_trading_cycle``'s backtest bar-restore only fires when
  ``kill_switch_persist=False``.

The full bit-identical order-stream parity assertion (``assert_frame_equal
(bt_orders, paper_orders)``) described in the ultra-plan requires a
``run_paper_replay(fixture, seed)`` helper that is not yet built; that
end-to-end test is tracked as an explicit xfail below so the gap is
visible in CI rather than silently missing.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.assembled_core.pipeline.trading_cycle import TradingContext
from src.assembled_core.qa.backtest_engine import make_cycle_fn, run_portfolio_backtest

pytestmark = pytest.mark.phase_zero


# --- parameter plumbing ------------------------------------------------------


def test_make_cycle_fn_exposes_enable_risk_controls() -> None:
    sig = inspect.signature(make_cycle_fn)
    assert "enable_risk_controls" in sig.parameters
    default = sig.parameters["enable_risk_controls"].default
    assert default is True, (
        f"make_cycle_fn.enable_risk_controls default must be True, got {default!r}"
    )


def test_trading_context_exposes_kill_switch_persist() -> None:
    ctx = TradingContext(prices=pd.DataFrame())
    assert hasattr(ctx, "kill_switch_persist")
    assert ctx.kill_switch_persist is True


def test_kill_switch_persist_is_opt_out() -> None:
    ctx = TradingContext(prices=pd.DataFrame(), kill_switch_persist=False)
    assert ctx.kill_switch_persist is False


def test_make_cycle_fn_applies_enable_risk_controls_to_ctx() -> None:
    """Captured cycle_fn must thread the flag into the per-bar context."""
    captured: dict = {}

    def fake_run_cycle(ctx):
        captured["enable_risk_controls"] = ctx.enable_risk_controls
        from src.assembled_core.pipeline.trading_cycle import TradingCycleResult

        return TradingCycleResult(
            timestamp=ctx.as_of,
            orders=pd.DataFrame(),
            signals=pd.DataFrame(),
            target_positions=pd.DataFrame(),
        )

    # Minimal template — backtest_engine's make_cycle_fn only reads what it
    # needs for the replace() call.
    template = TradingContext(
        prices=pd.DataFrame([{"symbol": "AAA", "timestamp": pd.Timestamp("2025-01-15"), "close": 100.0}]),
        mode="backtest",
    )

    cycle = make_cycle_fn(
        template,
        signal_fn=lambda _p: pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"]),
        position_sizing_fn=lambda _s, _c: pd.DataFrame(columns=["symbol", "target_qty"]),
        capital=10_000.0,
        run_trading_cycle_fn=fake_run_cycle,
        enable_risk_controls=True,
    )
    cycle(pd.Timestamp("2025-01-15"), pd.DataFrame(columns=["symbol", "qty"]))
    assert captured["enable_risk_controls"] is True

    cycle_off = make_cycle_fn(
        template,
        signal_fn=lambda _p: pd.DataFrame(columns=["timestamp", "symbol", "direction", "score"]),
        position_sizing_fn=lambda _s, _c: pd.DataFrame(columns=["symbol", "target_qty"]),
        capital=10_000.0,
        run_trading_cycle_fn=fake_run_cycle,
        enable_risk_controls=False,
    )
    cycle_off(pd.Timestamp("2025-01-15"), pd.DataFrame(columns=["symbol", "qty"]))
    assert captured["enable_risk_controls"] is False


def test_run_portfolio_backtest_is_callable_with_defaults() -> None:
    """Import-time sanity check: the public entry is still import-safe after
    the plumbing change."""
    sig = inspect.signature(run_portfolio_backtest)
    # The surface has not lost any parameters.
    for required in ("prices", "start_capital", "include_costs"):
        assert required in sig.parameters


# --- full parity gate (placeholder) ------------------------------------------


@pytest.mark.xfail(
    reason=(
        "Full bit-identical parity test requires a run_paper_replay(fixture, "
        "seed) helper that is not yet built. Tracked as pending E0.1 work."
    ),
    strict=True,
)
def test_bit_identical_order_stream_backtest_vs_paper() -> None:
    from src.assembled_core.ops.replay_snapshot import run_paper_replay  # noqa: F401

    raise AssertionError(
        "run_paper_replay not implemented yet — this xfail surfaces the gap."
    )
