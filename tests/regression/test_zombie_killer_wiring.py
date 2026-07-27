"""Zombie-killer wiring regression (E-059 class).

The force-FLAT block in ``_tc_signals.generate_signals`` died silently for
months: ``ops.shadow_recorder`` was archived (commit 13a97b54) while its call
sites stayed behind, so the in-block import raised ImportError which the
enclosing ``except Exception`` swallowed — the zombie killer never recorded
nor force-flatted anything.

These tests pin the restored wiring:
1. the block actually executes when zombies exist (shadow record emitted —
   this assertion fails if the import ever dies silently again),
2. shadow_only=True records but does NOT touch signals,
3. shadow_only=False force-flats zombie symbols incl. appending FLAT rows
   for zombies without a fresh signal.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

import src.assembled_core.ops.shadow_recorder as shadow_recorder_module
import src.assembled_core.pipeline._tc_signals as tc_signals_module
from src.assembled_core.pipeline._tc_signals import generate_signals
from src.assembled_core.pipeline.trading_cycle_shared import TradingContext

pytestmark = pytest.mark.regression

AS_OF = pd.Timestamp("2026-07-01", tz="UTC")


def _make_ctx(policy: dict) -> TradingContext:
    prices = pd.DataFrame({"timestamp": [AS_OF], "symbol": ["AAPL"], "close": [150.0]})
    ctx = TradingContext(prices=prices, capital=10_000.0, as_of=AS_OF)
    ctx.signal_fn = lambda features: pd.DataFrame(
        {
            "timestamp": [AS_OF, AS_OF],
            "symbol": ["AAPL", "MSFT"],
            "direction": ["BUY", "BUY"],
            "score": [0.9, 0.8],
        }
    )
    # Two positions held ~6 months with no price data -> both zombies
    # (conservative flag once past max_hold_days). ZZZZ has no fresh signal
    # row, so enforcement must APPEND a FLAT row for it.
    ctx.current_positions = pd.DataFrame(
        {
            "symbol": ["AAPL", "ZZZZ"],
            "qty": [10, 5],
            "entry_ts": ["2026-01-01T00:00:00+00:00", "2026-01-01T00:00:00+00:00"],
        }
    )
    ctx._policy_cache = policy
    return ctx


def _policy(shadow_only: bool) -> dict:
    return {
        "zombie_killer": {
            "enabled": True,
            "shadow_only": shadow_only,
            "max_hold_days": 5.0,
            "min_gain_pct": 0.005,
        }
    }


def _run_with_captured_shadow(
    ctx: TradingContext, monkeypatch
) -> tuple[pd.DataFrame, list]:
    calls: list[dict] = []

    def _capture(module, would_apply, *, as_of=None, meta=None, root=None):
        calls.append(
            {
                "module": module,
                "would_apply": dict(would_apply),
                "meta": dict(meta or {}),
            }
        )
        return None

    # The block does `from ...shadow_recorder import record_shadow` at call
    # time, so patching the source module attribute intercepts the call.
    monkeypatch.setattr(shadow_recorder_module, "record_shadow", _capture)
    signals = generate_signals(ctx.prices, ctx)
    return signals, calls


def test_zombie_block_executes_and_records_shadow(monkeypatch) -> None:
    """Regression: the block must actually run — a silently swallowed
    ImportError (the pre-restore state) would leave `calls` empty."""
    ctx = _make_ctx(_policy(shadow_only=True))
    _signals, calls = _run_with_captured_shadow(ctx, monkeypatch)

    assert len(calls) == 1, "zombie_killer block did not execute (silent skip?)"
    assert calls[0]["module"] == "zombie_killer"
    assert calls[0]["would_apply"]["zombie_symbols"] == ["AAPL", "ZZZZ"]
    assert calls[0]["meta"]["zombies_found"] == 2
    assert calls[0]["meta"]["applied"] is False


def test_shadow_only_true_does_not_touch_signals(monkeypatch) -> None:
    ctx = _make_ctx(_policy(shadow_only=True))
    signals, _calls = _run_with_captured_shadow(ctx, monkeypatch)

    by_symbol = signals.set_index("symbol")["direction"].to_dict()
    assert by_symbol["AAPL"] == "BUY", "shadow mode must not force-flat"
    assert by_symbol["MSFT"] == "BUY"
    assert "ZZZZ" not in by_symbol, "shadow mode must not append FLAT rows"


def test_shadow_only_false_force_flats_zombies(monkeypatch) -> None:
    ctx = _make_ctx(_policy(shadow_only=False))
    signals, calls = _run_with_captured_shadow(ctx, monkeypatch)

    by_symbol = signals.set_index("symbol")["direction"].to_dict()
    assert by_symbol["AAPL"] == "FLAT", "zombie with fresh signal must be flatted"
    assert by_symbol["MSFT"] == "BUY", "non-zombie signal must stay untouched"
    assert by_symbol["ZZZZ"] == "FLAT", "zombie without signal row must be appended"
    assert calls[0]["meta"]["applied"] is True


def test_disabled_policy_never_reaches_shadow_recorder(monkeypatch) -> None:
    """policy.yaml currently has no `enabled:` key for zombie_killer — the
    gate defaults to disabled and the block must stay inert."""
    ctx = _make_ctx({"zombie_killer": {"shadow_only": True}})
    signals, calls = _run_with_captured_shadow(ctx, monkeypatch)

    assert calls == []
    assert set(signals["symbol"]) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# E-059 follow-up (Stage-2 F-senior-6): a runtime failure inside the ACTIVE
# zombie block must surface at WARN (deduped), not vanish at DEBUG — the exact
# mask that hid the archived shadow_recorder for months. Cycle must continue.
# ---------------------------------------------------------------------------

_TC_SIGNALS_LOGGER = "src.assembled_core.pipeline._tc_signals"


def _zombie_warn_records(caplog) -> list[logging.LogRecord]:
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "zombie_killer check" in r.getMessage()
    ]


def test_zombie_block_failure_warns_and_cycle_continues(monkeypatch, caplog) -> None:
    # Isolate the process-global warn-once registry for this test.
    monkeypatch.setattr(tc_signals_module, "_SIGNAL_STEP_WARN_KEYS", set())

    def _boom(*_args, **_kwargs):
        raise RuntimeError("shadow write exploded")

    monkeypatch.setattr(shadow_recorder_module, "record_shadow", _boom)
    ctx = _make_ctx(_policy(shadow_only=False))

    with caplog.at_level(logging.DEBUG, logger=_TC_SIGNALS_LOGGER):
        signals = generate_signals(ctx.prices, ctx)

    # No raise, cycle continued, signals returned unmutated (the raise fires
    # before any force-flat): both BUYs intact, no appended FLAT row.
    by_symbol = signals.set_index("symbol")["direction"].to_dict()
    assert by_symbol == {"AAPL": "BUY", "MSFT": "BUY"}

    warns = _zombie_warn_records(caplog)
    assert len(warns) == 1, "zombie-block failure must be visible at WARN"
    assert "shadow write exploded" in warns[0].getMessage()


def test_zombie_block_failure_warn_is_deduped(monkeypatch, caplog) -> None:
    monkeypatch.setattr(tc_signals_module, "_SIGNAL_STEP_WARN_KEYS", set())

    def _boom(*_args, **_kwargs):
        raise RuntimeError("shadow write exploded")

    monkeypatch.setattr(shadow_recorder_module, "record_shadow", _boom)

    with caplog.at_level(logging.DEBUG, logger=_TC_SIGNALS_LOGGER):
        ctx1 = _make_ctx(_policy(shadow_only=False))
        generate_signals(ctx1.prices, ctx1)
        assert len(_zombie_warn_records(caplog)) == 1

        # Second cycle, same failure signature -> DEBUG repeat, no second WARN.
        caplog.clear()
        ctx2 = _make_ctx(_policy(shadow_only=False))
        generate_signals(ctx2.prices, ctx2)

    assert _zombie_warn_records(caplog) == []
    repeats = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG
        and "zombie_killer check" in r.getMessage()
        and "repeat" in r.getMessage()
    ]
    assert len(repeats) == 1, "repeat failure must still be traceable at DEBUG"


def test_inverse_etf_d4_failure_warns_once_and_cycle_continues(
    monkeypatch, caplog
) -> None:
    """Same E-059 class for the D4 sizing block: its DEBUG-only except was the
    sole visibility for a failed inverse_etf hedge. Failure must WARN (deduped)
    and leave target_positions untouched (no raise, no hedge row)."""
    import src.assembled_core.pipeline._tc_sizing as tc_sizing_module

    monkeypatch.setattr(tc_sizing_module, "_SIZING_STEP_WARN_KEYS", set())

    def _boom(*_args, **_kwargs):
        raise RuntimeError("shadow write exploded")

    monkeypatch.setattr(shadow_recorder_module, "record_shadow", _boom)

    prices = pd.DataFrame({"timestamp": [AS_OF], "VIX": [30.0]})
    ctx = TradingContext(prices=prices, capital=10_000.0, as_of=AS_OF)
    policy = {"inverse_etf": {"enabled": True}}
    meta = {"crash_prediction": {"crash_probability": 0.9}}
    tp = pd.DataFrame(
        {"symbol": ["AAPL"], "target_weight": [1.0], "target_qty": [10.0]}
    )

    with caplog.at_level(
        logging.DEBUG, logger="src.assembled_core.pipeline._tc_sizing"
    ):
        out1 = tc_sizing_module._sp_apply_inverse_etf(tp.copy(), ctx, policy, meta)
        out2 = tc_sizing_module._sp_apply_inverse_etf(tp.copy(), ctx, policy, meta)

    # Graceful: no raise, no hedge row appended on either call.
    assert list(out1["symbol"]) == ["AAPL"]
    assert list(out2["symbol"]) == ["AAPL"]

    d4_warns = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "inverse_etf hedge (D4)" in r.getMessage()
    ]
    assert len(d4_warns) == 1, "D4 failure must WARN exactly once (dedup)"
