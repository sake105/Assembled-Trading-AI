# tests/pipeline/test_tc_sizing_cost_aware_config.py
"""E-059 #3 — cost_aware sizing passes OptimizerConfig as config= keyword.

Bug pinned: _sp_dispatch_sizing (sizing_method == "cost_aware") used to pass
OptimizerConfig(...) as the 4th POSITIONAL argument of optimize_portfolio(),
which is ``per_symbol_cost_bps`` per the signature in
src/assembled_core/portfolio/cost_aware_optimizer.py — NOT ``config``. At
runtime this crashed inside optimize_portfolio (``OptimizerConfig`` has no
``.get``) and the branch silently fell back to default sizing via the bare
``except``; the policy's risk_aversion/turnover_penalty/max_weight never
reached the optimizer.

This test asserts the fixed contract: optimize_portfolio is invoked with
``config=OptimizerConfig(...)`` built from sizing_cfg and
``per_symbol_cost_bps`` NEVER positionally in slot 4.

VERTRAGS-UPDATE 2026-08-17 (Audit-Plan 4.4): per_symbol_cost_bps wird jetzt
BEWUSST als Keyword uebergeben — None, wenn keine ADV-Basis existiert (Fixture
ohne volume-Spalte), sonst ein dict {symbol: commission + spread/2} aus der
transaction_costs-Engine. Der alte Pin "nicht als Keyword" schuetzte nur den
Positional-Bug; der bleibt durch den len(args)==3-Check gepinnt.

Patch note: _tc_sizing does a function-LOCAL ``from ... import
optimize_portfolio`` at call time, so we patch the attribute on the SOURCE
module ``src.assembled_core.portfolio.cost_aware_optimizer``.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pandas as pd
import pytest

import src.assembled_core.portfolio.cost_aware_optimizer as cao_mod
from src.assembled_core.pipeline._tc_sizing import _sp_dispatch_sizing

pytestmark = pytest.mark.fast

CAPITAL = 100_000.0
_LOG = logging.getLogger("test_tc_sizing_cost_aware_config")


def _mini_prices() -> pd.DataFrame:
    """10 daily closes for 2 symbols — enough for pct_change + covariance."""
    ts = pd.date_range("2026-01-05", periods=10, freq="B", tz="UTC")
    rows = []
    for i, t in enumerate(ts):
        rows.append({"timestamp": t, "symbol": "AAA", "close": 100.0 + i * 0.5})
        rows.append({"timestamp": t, "symbol": "BBB", "close": 50.0 - i * 0.2})
    return pd.DataFrame(rows)


def _mini_signals() -> pd.DataFrame:
    return pd.DataFrame({"symbol": ["AAA", "BBB"], "score": [0.8, 0.2]})


def test_cost_aware_passes_policy_config_as_keyword(monkeypatch):
    """optimize_portfolio must receive config=OptimizerConfig(from sizing_cfg)
    and per_symbol_cost_bps must stay at its default (None)."""
    calls: dict[str, object] = {}

    def fake_optimize_portfolio(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return cao_mod.OptimizationResult(
            weights={"AAA": 0.6, "BBB": 0.4},
            expected_return=0.0,
            expected_risk=0.0,
            turnover_cost=0.0,
            solver_status="optimal",
            method="fake",
        )

    # Function-local import in _tc_sizing resolves this attribute at call time.
    monkeypatch.setattr(cao_mod, "optimize_portfolio", fake_optimize_portfolio)

    def _fallback_sizing(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
        return pd.DataFrame(
            {"symbol": ["FALLBACK"], "target_weight": [1.0], "target_qty": [capital]}
        )

    ctx = SimpleNamespace(
        capital=CAPITAL,
        position_sizing_fn=_fallback_sizing,
        current_positions=pd.DataFrame({"symbol": ["AAA"], "weight": [0.1]}),
    )
    sizing_cfg = {
        "method": "cost_aware",
        "risk_aversion": 3.5,
        "turnover_penalty": 0.01,
        "max_weight": 0.25,
    }

    result = _sp_dispatch_sizing(_mini_signals(), ctx, _mini_prices(), sizing_cfg, _LOG)

    # 1) The optimizer was actually called (no silent except-fallback).
    assert "kwargs" in calls, (
        "optimize_portfolio was never called — cost_aware branch fell back "
        "to default sizing (bug behaviour resurrected?)"
    )
    args = calls["args"]
    kwargs = calls["kwargs"]
    assert isinstance(args, tuple) and isinstance(kwargs, dict)

    # 2) Exactly 3 positional args (mu, sigma, current_weights) — slot 4
    #    (per_symbol_cost_bps) must NOT be occupied positionally...
    assert len(args) == 3, (
        f"expected 3 positional args (mu, sigma, current_weights), got {len(args)} "
        "— a 4th positional arg would land in per_symbol_cost_bps again"
    )
    # ...and current_weights came from ctx.current_positions.
    assert args[2] == {"AAA": 0.1}

    # 3) Audit-Plan 4.4: per_symbol_cost_bps kommt jetzt als KEYWORD — in
    #    dieser Fixture (keine volume-Spalte -> kein ADV) explizit None,
    #    d.h. der Optimizer-Flat-Default gilt weiter.
    assert "per_symbol_cost_bps" in kwargs
    assert kwargs["per_symbol_cost_bps"] is None

    # 4) config= is an OptimizerConfig carrying the sizing_cfg policy values.
    cfg = kwargs.get("config")
    assert isinstance(cfg, cao_mod.OptimizerConfig), (
        f"config kwarg missing or wrong type: {type(cfg)!r}"
    )
    assert cfg.risk_aversion == pytest.approx(3.5)
    assert cfg.turnover_penalty == pytest.approx(0.01)
    assert cfg.max_weight == pytest.approx(0.25)

    # 5) Result was built from the optimizer weights, not the fallback frame.
    assert set(result["symbol"]) == {"AAA", "BBB"}
    row_a = result.loc[result["symbol"] == "AAA"].iloc[0]
    assert row_a["target_weight"] == pytest.approx(0.6)
    assert row_a["target_qty"] == pytest.approx(0.6 * CAPITAL)


def test_cost_aware_per_symbol_costs_from_adv(monkeypatch):
    """Audit-Plan 4.4 Positiv-Pin: MIT volume-Spalte liefert die
    transaction_costs-Engine per-Symbol-Kosten (commission + Spread/2) —
    beide Symbole vorhanden, Werte >= commission."""
    seen: dict[str, object] = {}

    def fake_optimize_portfolio(*args, **kwargs):
        seen["kwargs"] = kwargs
        return cao_mod.OptimizationResult(
            weights={"AAA": 1.0},
            expected_return=0.0,
            expected_risk=0.0,
            turnover_cost=0.0,
            solver_status="optimal",
            method="fake",
        )

    monkeypatch.setattr(cao_mod, "optimize_portfolio", fake_optimize_portfolio)
    prices = _mini_prices()
    prices["volume"] = 1_000_000  # ADV-Basis vorhanden
    ts_extra = pd.date_range("2026-01-19", periods=15, freq="B", tz="UTC")
    extra = []
    for i, t in enumerate(ts_extra):  # >= adv_window=20 Zeilen je Symbol
        extra.append(
            {"timestamp": t, "symbol": "AAA", "close": 105.0 + i, "volume": 1_000_000}
        )
        extra.append(
            {"timestamp": t, "symbol": "BBB", "close": 48.0 - i * 0.1, "volume": 500}
        )
    prices = pd.concat([prices, pd.DataFrame(extra)], ignore_index=True)

    ctx = SimpleNamespace(
        capital=CAPITAL,
        position_sizing_fn=lambda s, c: pd.DataFrame(
            {"symbol": ["FALLBACK"], "target_weight": [1.0], "target_qty": [c]}
        ),
        current_positions=None,
    )
    _sp_dispatch_sizing(
        _mini_signals(),
        ctx,
        prices,
        {"method": "cost_aware", "commission_bps": 4.0},
        _LOG,
    )
    kwargs = seen.get("kwargs")
    assert isinstance(kwargs, dict)
    costs = kwargs.get("per_symbol_cost_bps")
    assert isinstance(costs, dict) and {"AAA", "BBB"} <= set(costs)
    # one_way = commission + spread/2 -> mindestens die Commission.
    assert all(v >= 4.0 for v in costs.values())
    # F-senior-6 (E-186): VARIANZ pinnen, nicht nur Praesenz — die erste
    # Fassung nutzte SpreadModel() ohne buckets und gab JEDEM Symbol
    # denselben Fallback; der Test bestand trotz voelliger
    # Nicht-Differenzierung. AAA hat ~2000x den ADV von BBB.
    assert costs["AAA"] != costs["BBB"], (
        "per_symbol_cost_bps differenziert nicht nach Liquiditaet — "
        "SpreadModel ohne buckets kollabiert auf den Fallback"
    )
    assert costs["AAA"] < costs["BBB"]  # liquider = billiger


def test_cost_aware_config_defaults_when_policy_keys_absent(monkeypatch):
    """Without explicit sizing_cfg keys the OptimizerConfig defaults from the
    dispatch site apply (risk_aversion 1.0, turnover_penalty 0.001,
    max_weight 0.10) — still passed as config= keyword."""
    seen: dict[str, object] = {}

    def fake_optimize_portfolio(*args, **kwargs):
        seen["kwargs"] = kwargs
        return cao_mod.OptimizationResult(
            weights={"AAA": 1.0},
            expected_return=0.0,
            expected_risk=0.0,
            turnover_cost=0.0,
            solver_status="optimal",
            method="fake",
        )

    monkeypatch.setattr(cao_mod, "optimize_portfolio", fake_optimize_portfolio)

    ctx = SimpleNamespace(
        capital=CAPITAL,
        position_sizing_fn=lambda s, c: pd.DataFrame(
            {"symbol": ["FALLBACK"], "target_weight": [1.0], "target_qty": [c]}
        ),
        current_positions=None,
    )

    _sp_dispatch_sizing(
        _mini_signals(), ctx, _mini_prices(), {"method": "cost_aware"}, _LOG
    )

    kwargs = seen.get("kwargs")
    assert isinstance(kwargs, dict), "optimize_portfolio was never called"
    cfg = kwargs.get("config")
    assert isinstance(cfg, cao_mod.OptimizerConfig)
    assert cfg.risk_aversion == pytest.approx(1.0)
    assert cfg.turnover_penalty == pytest.approx(0.001)
    assert cfg.max_weight == pytest.approx(0.10)
