"""QUAL/Zensus-1 sub-commit C2: _tc_signals ensemble drops + meta merge-union.

Two coupled fixes:

1. QUAL-15 — ``_SignalFnShim.generate_signals`` in ``_tc_signals.py`` wrapped a
   member ``signal_fn`` in ``try: ... except Exception: return pd.DataFrame()``
   with ZERO log. A member whose signal_fn raised silently contributed nothing
   to the ensemble blend — indistinguishable from one that legitimately produced
   no signals. The drop is now recorded via ``_record_degraded_step`` (WARN +
   structured ``meta['degraded_steps']``) while still failing open (the blend
   continues without that member).

2. Merge-union hardening — once the signals stage writes ``degraded_steps`` into
   ``result.meta``, the later ``result.meta.update(sizing_meta)`` in
   ``trading_cycle_v2.py`` would CLOBBER it with sizing's own ``degraded_steps``.
   The merge now pops + extends that one key so entries from both stages union.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd

import src.assembled_core.config.policy_loader as policy_loader
import src.assembled_core.pipeline.trading_cycle_v2 as tcv2
from src.assembled_core.pipeline._tc_signals import _ensemble_signals_if_enabled
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    TradingCycleResult,
)

_SIG_COLS = ["timestamp", "symbol", "direction", "score"]


# --------------------------------------------------------------------------- #
# 1. QUAL-15: a failing ensemble member is recorded, not silently dropped
# --------------------------------------------------------------------------- #


def test_ensemble_member_failure_recorded_not_blocking(monkeypatch, caplog) -> None:
    """A member signal_fn that raises must land in meta['degraded_steps'] as
    'ensemble_member:<name>' + a WARNING — and the ensemble layer must fail open
    (return the original signals unchanged, blend continues without the member)."""

    monkeypatch.setattr(
        policy_loader,
        "load_policy",
        lambda *a, **k: {
            "strategies": {
                "ensemble": {
                    "enabled": True,
                    "method": "weighted_average",
                    "members": {"momentum": {"weight": 1.0}},
                }
            }
        },
    )

    def _boom(_prices):
        raise RuntimeError("member boom")

    ctx = SimpleNamespace(signal_fn=_boom, risk_state={})
    meta: dict = {}
    feats = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-04-22", tz="UTC")],
            "symbol": ["AAPL"],
            "close": [100.0],
        }
    )
    sig_in = pd.DataFrame(columns=_SIG_COLS)

    with caplog.at_level(logging.WARNING):
        out = _ensemble_signals_if_enabled(
            sig_in, feats, ctx, logging.getLogger("test_ens"), meta
        )

    member_entries = [
        s
        for s in meta.get("degraded_steps", [])
        if s["step"] == "ensemble_member:momentum"
    ]
    assert len(member_entries) == 1
    assert "member boom" in member_entries[0]["error"]
    assert any(
        "[DEGRADED]" in r.message and "ensemble_member:momentum" in str(r.args)
        for r in caplog.records
    )
    # fail-open: original signals returned (blend had no usable member)
    assert list(out.columns) == _SIG_COLS


# --------------------------------------------------------------------------- #
# 2. Merge-union: signals-stage degraded_steps survive size_positions merge
# --------------------------------------------------------------------------- #


def _make_prices(n_days: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B", tz="UTC")
    rows = []
    for j, sym in enumerate(["AAPL", "MSFT", "GOOG"]):
        for i, ts in enumerate(dates):
            rows.append({"timestamp": ts, "symbol": sym, "close": 100.0 + i + j})
    return pd.DataFrame(rows)


def _signal_fn(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(columns=_SIG_COLS)


def _sizing_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
    return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])


def test_signals_and_sizing_degraded_steps_union(monkeypatch) -> None:
    """Both the signals stage and the sizing stage record a degraded step in the
    same cycle. The line-626 merge must UNION them in result.meta['degraded_steps']
    rather than letting sizing_meta clobber the signals entry — while still
    copying sizing_meta's other keys."""
    monkeypatch.setenv("ASSEMBLED_RISK_STATE_PERSISTENCE_MODE", "ephemeral")

    def _fake_generate_signals(features, ctx, *, log=None, meta=None):
        if meta is not None:
            meta.setdefault("degraded_steps", []).append(
                {"step": "signals_probe", "error": "RuntimeError: sig boom"}
            )
        return pd.DataFrame(columns=_SIG_COLS)

    def _fake_size_positions(
        signals,
        ctx,
        *,
        prices_filtered=None,
        prices_with_features=None,
        prices_latest=None,
        log=None,
    ):
        sizing_meta = {
            "degraded_steps": [
                {"step": "sizing_probe", "error": "RuntimeError: size boom"}
            ],
            "vol_targeting": {"scale_factor": 1.0},
        }
        targets = pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
        return targets, False, sizing_meta

    monkeypatch.setattr(tcv2, "generate_signals", _fake_generate_signals)
    monkeypatch.setattr(tcv2, "size_positions", _fake_size_positions)

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

    result: TradingCycleResult = tcv2.run_trading_cycle(ctx)

    steps = [s["step"] for s in result.meta.get("degraded_steps", [])]
    # both stages' entries survive the merge (no clobber)
    assert "signals_probe" in steps
    assert "sizing_probe" in steps
    # sizing_meta's non-degraded keys still merged in
    assert result.meta.get("vol_targeting") == {"scale_factor": 1.0}
    # the degraded_steps key was not left duplicated/nested under sizing_meta
    assert isinstance(result.meta["degraded_steps"], list)
