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
(bt_orders, paper_orders)``) described in the ultra-plan requires unifying
the position-evolution model between ``run_portfolio_backtest`` and
``run_paper_replay``. The previous xfail placeholder was removed on
2026-04-18 (P0 A8, Deep Run v2): non-strict xfail as permanent config hides
real status. The remaining parity gap is documented in
``docs/tech_debt/parity_gap.md`` with an explicit sunset date.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from src.assembled_core.pipeline.trading_cycle_shared import TradingContext
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
        from src.assembled_core.pipeline.trading_cycle_shared import TradingCycleResult

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


def _synthetic_parity_fixture(
    n_symbols: int = 5, n_days: int = 15, seed: int = 42
) -> pd.DataFrame:
    """Build a deterministic price frame used by the parity test."""
    import numpy as np

    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-02", periods=n_days, freq="B", tz="UTC")
    rows = []
    for i in range(n_symbols):
        sym = f"SYM{i}"
        noise = rng.normal(0.0, 0.01 + 0.001 * i, n_days)
        close = 100.0 * (1.0 + noise).cumprod()
        for ts, c in zip(idx, close):
            rows.append({"timestamp": ts, "symbol": sym, "close": float(c)})
    return pd.DataFrame(rows)


def _parity_signal_fn(df: pd.DataFrame) -> pd.DataFrame:
    syms = sorted(df["symbol"].unique().tolist())
    ts = df["timestamp"].max()
    return pd.DataFrame(
        {
            "symbol": syms,
            "timestamp": [ts] * len(syms),
            "direction": ["long"] * len(syms),
            "score": [1.0 / max(len(syms), 1)] * len(syms),
        }
    )


def _parity_sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    n = len(signals)
    if n == 0:
        return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
    return pd.DataFrame(
        {
            "symbol": signals["symbol"],
            "target_weight": [1.0 / n] * n,
            "target_qty": [capital / n] * n,
        }
    )


def test_run_paper_replay_is_importable() -> None:
    """Plumbing check — the E0.1 helper must be available from ops."""
    from src.assembled_core.ops.replay_snapshot import ReplayResult, run_paper_replay

    assert callable(run_paper_replay)
    assert ReplayResult.__name__ == "ReplayResult"


def test_run_paper_replay_emits_deterministic_orders() -> None:
    """Replaying the same fixture twice must produce bit-identical orders."""
    from src.assembled_core.ops.replay_snapshot import run_paper_replay

    prices = _synthetic_parity_fixture()
    r1 = run_paper_replay(
        prices,
        signal_fn=_parity_signal_fn,
        position_sizing_fn=_parity_sizing_fn,
        start_capital=10_000.0,
        seed=42,
        enable_risk_controls=True,
    )
    r2 = run_paper_replay(
        prices,
        signal_fn=_parity_signal_fn,
        position_sizing_fn=_parity_sizing_fn,
        start_capital=10_000.0,
        seed=42,
        enable_risk_controls=True,
    )
    # Determinism: identical fixture + identical seed → identical order stream.
    pd.testing.assert_frame_equal(
        r1.orders_df.reset_index(drop=True),
        r2.orders_df.reset_index(drop=True),
        check_dtype=False,
    )
    assert r1.n_days == r2.n_days


# -----------------------------------------------------------------------------
# Removed 2026-04-18 per P0 A8 (Deep Run v2): the previous
# `test_bit_identical_order_stream_backtest_vs_paper` was a non-strict xfail
# placeholder. Non-strict xfail as permanent config hides real status — the
# parity gap is now tracked in `docs/tech_debt/parity_gap.md` with an explicit
# sunset date. When the bt-vs-paper position-evolution model is unified, add
# the real assertion back here as a strict test, not an xfail.
# -----------------------------------------------------------------------------
