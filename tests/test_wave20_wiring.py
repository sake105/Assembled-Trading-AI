"""Wave 20 — wiring sprint: the previously-unwired Wave-18 helpers actually
in the live decision path.

Three integration paths:
    * vol_targeting.compute_vol_targeting_result(method='ewma') —
      switches to the EWMA forecast variant inside the existing call site.
    * ops._paper_runner_gates.apply_halt_cache_gate — populates
      ctx.halted_symbols from a JSON file when policy.halt_cache.enabled.
    * ops._paper_runner_gates.apply_tilt_gate — runs detect_tilt on the
      ledger equity-curve and (with block_orders=true) signals the
      paper-runner to skip the cycle.

Default-off behavior is the central invariant: every test that doesn't
enable a flag must reproduce the existing behavior exactly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ===========================================================================
# vol_targeting method='ewma' wiring
# ===========================================================================


def _equity_curve(n: int, vol_daily: float = 0.012, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, vol_daily, n)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(100_000.0 * np.cumprod(1.0 + rets), index=idx)


def test_vol_targeting_realized_path_unchanged_by_default() -> None:
    from src.assembled_core.risk.vol_targeting import compute_vol_targeting_result

    eq = _equity_curve(200)
    policy = {
        "vol_targeting": {
            "enabled": True,
            "target_vol_annual": 0.20,
            "lookback_days": 20,
        }
    }
    scale, vol_est, target = compute_vol_targeting_result(eq, policy)
    assert 0.0 < scale <= 1.5
    assert vol_est > 0.0
    assert target == 0.20


def test_vol_targeting_ewma_method_engages() -> None:
    from src.assembled_core.risk.vol_targeting import compute_vol_targeting_result

    eq = _equity_curve(200, seed=7)
    policy = {
        "vol_targeting": {
            "enabled": True,
            "target_vol_annual": 0.20,
            "method": "ewma",
            "ewma_lambda": 0.94,
        }
    }
    scale, vol_est, target = compute_vol_targeting_result(eq, policy)
    assert 0.0 < scale <= 1.5
    assert vol_est > 0.0  # EWMA forecast


def test_vol_targeting_ewma_vs_realized_differs() -> None:
    """After a vol burst the realized lookback lags; EWMA reacts faster."""
    from src.assembled_core.risk.vol_targeting import compute_vol_targeting_result

    # Calm series then a vol shock in the last 5 days.
    rng = np.random.default_rng(1)
    calm = rng.normal(0.0005, 0.005, 200)
    shock = rng.normal(0.0005, 0.04, 5)
    rets = np.concatenate([calm, shock])
    idx = pd.date_range("2024-01-01", periods=len(rets), freq="D")
    eq = pd.Series(100_000.0 * np.cumprod(1.0 + rets), index=idx)

    p_realized = {"vol_targeting": {"enabled": True, "target_vol_annual": 0.20}}
    p_ewma = {
        "vol_targeting": {
            "enabled": True,
            "target_vol_annual": 0.20,
            "method": "ewma",
        }
    }
    _, v_realized, _ = compute_vol_targeting_result(eq, p_realized)
    _, v_ewma, _ = compute_vol_targeting_result(eq, p_ewma)
    # The two estimates exist and are different (forward-looking vs lookback).
    assert v_realized > 0.0 and v_ewma > 0.0
    assert v_realized != v_ewma


def test_vol_targeting_disabled_is_no_op_for_both_methods() -> None:
    from src.assembled_core.risk.vol_targeting import compute_vol_targeting_result

    eq = _equity_curve(200)
    policy = {"vol_targeting": {"enabled": False, "method": "ewma"}}
    scale, vol_est, target = compute_vol_targeting_result(eq, policy)
    assert scale == 1.0
    assert np.isnan(vol_est) and np.isnan(target)


# ===========================================================================
# halt_cache gate wiring
# ===========================================================================


@dataclass
class _StubCtx:
    halted_symbols: frozenset[str] | None = None
    tilt_state: dict[str, Any] | None = None


def test_halt_cache_gate_disabled_does_not_touch_ctx() -> None:
    from src.assembled_core.ops._paper_runner_gates import apply_halt_cache_gate

    ctx = _StubCtx()
    count = apply_halt_cache_gate(
        ctx, paper_cfg={"halt_cache": {"enabled": False}}, root=Path(".")
    )
    assert count == 0
    assert ctx.halted_symbols is None


def test_halt_cache_gate_loads_from_json_list(tmp_path: Path) -> None:
    # Reset the module-level singleton so different test runs don't bleed.
    import src.assembled_core.ops._paper_runner_gates as gates

    gates._HALT_CACHE = None

    halts_file = tmp_path / "halts.json"
    halts_file.write_text(json.dumps(["AAPL", "MSFT", "TSLA"]), encoding="utf-8")

    ctx = _StubCtx()
    count = gates.apply_halt_cache_gate(
        ctx,
        paper_cfg={
            "halt_cache": {
                "enabled": True,
                "symbols_file": str(halts_file),
                "ttl_seconds": 60.0,
            }
        },
        root=tmp_path,
    )
    assert count == 3
    assert ctx.halted_symbols == frozenset({"AAPL", "MSFT", "TSLA"})


def test_halt_cache_gate_handles_missing_file(tmp_path: Path) -> None:
    import src.assembled_core.ops._paper_runner_gates as gates

    gates._HALT_CACHE = None

    ctx = _StubCtx()
    count = gates.apply_halt_cache_gate(
        ctx,
        paper_cfg={
            "halt_cache": {
                "enabled": True,
                "symbols_file": str(tmp_path / "nonexistent.json"),
            }
        },
        root=tmp_path,
    )
    assert count == 0
    assert ctx.halted_symbols == frozenset()


def test_halt_cache_gate_handles_dict_schema(tmp_path: Path) -> None:
    import src.assembled_core.ops._paper_runner_gates as gates

    gates._HALT_CACHE = None

    halts_file = tmp_path / "halts.json"
    halts_file.write_text(
        json.dumps({"halted": ["AAPL"], "source": "broker_xyz"}), encoding="utf-8"
    )

    ctx = _StubCtx()
    count = gates.apply_halt_cache_gate(
        ctx,
        paper_cfg={
            "halt_cache": {
                "enabled": True,
                "symbols_file": str(halts_file),
            }
        },
        root=tmp_path,
    )
    assert count == 1
    assert ctx.halted_symbols == frozenset({"AAPL"})


# ===========================================================================
# tilt gate wiring
# ===========================================================================


def _ledger_with_curve(values: list[tuple[int, float]], now_ref: datetime):
    """Build a ledger-state dict with an equity_curve list.

    values: list of (days_ago, equity).
    """
    curve = []
    for d, eq in sorted(values, key=lambda x: -x[0]):
        ts = (now_ref - timedelta(days=d)).isoformat()
        curve.append({"timestamp": ts, "equity": eq})
    return {"equity_curve": curve}


def test_tilt_gate_disabled_returns_clean_no_op() -> None:
    from src.assembled_core.ops._paper_runner_gates import apply_tilt_gate

    ctx = _StubCtx()
    out = apply_tilt_gate(
        ctx, paper_cfg={"tilt": {"enabled": False}}, ledger_state={"equity_curve": []}
    )
    assert out.is_tilted is False
    assert out.blocked is False
    assert ctx.tilt_state is None  # untouched


def test_tilt_gate_no_ledger_short_circuits() -> None:
    from src.assembled_core.ops._paper_runner_gates import apply_tilt_gate

    ctx = _StubCtx()
    out = apply_tilt_gate(ctx, paper_cfg={"tilt": {"enabled": True}}, ledger_state=None)
    assert out.is_tilted is False
    assert ctx.tilt_state is None


def test_tilt_gate_fires_on_weekly_drawdown() -> None:
    from src.assembled_core.ops._paper_runner_gates import apply_tilt_gate

    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    # 10% drop over 7 days exceeds default weekly_dd_pct=0.08.
    ledger = _ledger_with_curve([(6, 100_000), (5, 100_000), (1, 90_000)], now)
    ctx = _StubCtx()
    out = apply_tilt_gate(
        ctx,
        paper_cfg={"tilt": {"enabled": True}},
        ledger_state=ledger,
        now=now,
    )
    assert out.is_tilted is True
    assert "weekly_drawdown" in out.triggered_rules
    assert ctx.tilt_state is not None
    assert ctx.tilt_state["is_tilted"] is True


def test_tilt_gate_block_orders_signals_caller() -> None:
    from src.assembled_core.ops._paper_runner_gates import apply_tilt_gate

    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    ledger = _ledger_with_curve([(6, 100_000), (5, 100_000), (1, 90_000)], now)
    ctx = _StubCtx()
    out = apply_tilt_gate(
        ctx,
        paper_cfg={"tilt": {"enabled": True, "block_orders": True}},
        ledger_state=ledger,
        now=now,
    )
    assert out.is_tilted is True
    assert out.blocked is True


def test_tilt_gate_block_orders_false_does_not_signal_block() -> None:
    from src.assembled_core.ops._paper_runner_gates import apply_tilt_gate

    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    ledger = _ledger_with_curve([(6, 100_000), (5, 100_000), (1, 90_000)], now)
    ctx = _StubCtx()
    out = apply_tilt_gate(
        ctx,
        paper_cfg={"tilt": {"enabled": True, "block_orders": False}},
        ledger_state=ledger,
        now=now,
    )
    assert out.is_tilted is True
    assert out.blocked is False


def test_tilt_gate_calm_market_does_not_fire() -> None:
    from src.assembled_core.ops._paper_runner_gates import apply_tilt_gate

    now = datetime(2026, 5, 12, tzinfo=timezone.utc)
    ledger = _ledger_with_curve([(6, 100_000), (5, 100_050), (1, 100_100)], now)
    ctx = _StubCtx()
    out = apply_tilt_gate(
        ctx,
        paper_cfg={"tilt": {"enabled": True, "block_orders": True}},
        ledger_state=ledger,
        now=now,
    )
    assert out.is_tilted is False
    assert out.blocked is False
