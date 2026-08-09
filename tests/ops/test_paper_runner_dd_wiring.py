"""DD-Treppen-Verdrahtung im Paper-Runner + Drossel-Semantik (2026-08-09).

``_evaluate_auto_dd_kill_switch`` (Step 6.4) liest ``ctx.current_equity`` /
``ctx.peak_equity``/``ctx.hwm_equity`` per getattr. Vor dem Fix hat der
Paper-Runner nichts davon gesetzt — die Drawdown-Treppe aus ``policy.yaml
drawdown_policy`` (-10/-15/-20) war im Paper-/Pilot-Pfad still inert.

Verdrahtungs-Entscheidungen, die diese Tests festschreiben:
* ``hwm_equity`` (NICHT ``peak_equity``) wird gesetzt — armt exakt die
  Treppe; Step-0-Caps/30%-Pfade bleiben inert wie vor dem Fix (DD-04).
* HWM-Basis respektiert ``paper_runner.hwm_since`` (Baseline-Reset
  2026-07-02): Alt-Kurvenpunkte davor zaehlen nicht (DD-02), Peak startet
  nie unter start_capital.
* Step 6.4: soft/hard = zustandslose qty-Drossel (x0.5/x0.2), KEIN
  persistenter Kill-Switch; nur kill (-20 %) aktiviert den persistenten,
  Token-gated Switch und blockt alles (DD-03).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import src.assembled_core.ops.paper_runner as paper_runner
import src.assembled_core.pipeline.trading_cycle_v2 as trading_cycle_v2
from src.assembled_core.pipeline.trading_cycle_shared import (
    TradingContext,
    TradingCycleResult,
    _evaluate_auto_dd_kill_switch,
)

pytestmark = pytest.mark.fast


def _run_paper(
    monkeypatch, tmp_path: Path, ledger_state: dict, hwm_since: str | None = None
) -> dict:
    """Drive run_paper_daily_one (mode='paper') mit gestubbtem Cycle."""
    captured: dict = {}

    def _fake_cycle(ctx):
        captured["ctx"] = ctx
        return TradingCycleResult(status="success")

    monkeypatch.setattr(trading_cycle_v2, "run_trading_cycle", _fake_cycle)
    monkeypatch.setattr(
        paper_runner, "_load_pilot_policy_fail_fast", lambda context: {}
    )
    monkeypatch.setattr(
        paper_runner, "_prd_intel_summaries", lambda result, paper_cfg, root: None
    )
    monkeypatch.setattr(paper_runner, "_prd_write_artifacts", lambda **kwargs: None)
    # Fills/Ledger-Persistenz stubben: hier interessiert nur der ctx-Bau.
    # Default raising=True: wird die Funktion umbenannt, bricht der Test laut
    # statt still mit echter Persistenz weiterzulaufen (TR-01).
    monkeypatch.setattr(
        paper_runner,
        "_prd_paper_fills_and_ledger",
        lambda **kwargs: None,
    )

    ledger_path = tmp_path / "ledger" / "ledger_state.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps(ledger_state), encoding="utf-8")

    as_of = pd.Timestamp("2025-06-26", tz="UTC")
    prices = pd.DataFrame({"timestamp": [as_of], "symbol": ["AAPL"], "close": [100.0]})
    out_dir = tmp_path / "runs" / "2025-06-26"
    out_dir.mkdir(parents=True)
    runner_cfg: dict = {
        "strategy": {"name": "none"},
        "ledger_path": str(ledger_path),
        "start_capital": 90_000.0,
    }
    if hwm_since is not None:
        runner_cfg["hwm_since"] = hwm_since
    app_cfg: dict = {"paper_runner": runner_cfg}

    exit_code, _rec = paper_runner.run_paper_daily_one(
        as_of,
        out_dir,
        "paper",
        app_cfg,
        prices,
        root=tmp_path,
        execution_mode="sim",
    )
    captured["exit_code"] = exit_code
    return captured


def test_paper_ctx_traegt_current_und_hwm_equity(monkeypatch, tmp_path):
    """Cash 90k, HWM in der Kurve 100k -> current 90k, hwm 100k (nicht cash)."""
    cap = _run_paper(
        monkeypatch,
        tmp_path,
        {
            "cash": 90_000.0,
            "positions": {},
            "equity_curve": [
                {"utc": "2026-07-03T00:00:00Z", "equity": 100_000.0},
                {"utc": "2026-07-04T00:00:00Z", "equity": 95_000.0},
            ],
        },
    )
    assert cap["exit_code"] == 0
    ctx = cap["ctx"]
    assert float(ctx.current_equity) == pytest.approx(90_000.0)
    assert float(ctx.hwm_equity) == pytest.approx(100_000.0)


def test_hwm_since_filtert_alt_peaks_weg(monkeypatch, tmp_path):
    """DD-02-Szenario: Alt-Peak VOR dem Baseline-Reset zaehlt nicht.

    Kurve traegt 99k (prae-Reset) und 91k (post-Reset); hwm_since=2026-07-02
    -> HWM = 91k, nicht 99k. dd = (90k-91k)/91k ~ -1.1% -> Treppe still."""
    cap = _run_paper(
        monkeypatch,
        tmp_path,
        {
            "cash": 90_000.0,
            "positions": {},
            "equity_curve": [
                {"utc": "2026-05-06T00:00:00Z", "equity": 99_000.0},
                {"utc": "2026-07-14T00:00:00Z", "equity": 91_000.0},
            ],
        },
        hwm_since="2026-07-02",
    )
    ctx = cap["ctx"]
    assert float(ctx.hwm_equity) == pytest.approx(91_000.0), "Alt-Peak nicht gefiltert"
    decision = _evaluate_auto_dd_kill_switch(
        ctx,
        TradingCycleResult(status="success"),
        {
            "drawdown_policy": {
                "auto_kill_enabled": True,
                "levels": {"soft": -0.10, "hard": -0.15, "kill": -0.20},
            }
        },
    )
    assert decision is None, "Treppe feuerte gegen den Alt-Peak"


def test_hwm_startet_nie_unter_start_capital(monkeypatch, tmp_path):
    """Alle Kurvenpunkte gefiltert -> HWM = max(equity_before, start_capital)."""
    cap = _run_paper(
        monkeypatch,
        tmp_path,
        {
            "cash": 85_000.0,
            "positions": {},
            "equity_curve": [{"utc": "2026-05-06T00:00:00Z", "equity": 99_000.0}],
        },
        hwm_since="2026-07-02",
    )
    ctx = cap["ctx"]
    assert float(ctx.hwm_equity) == pytest.approx(90_000.0)  # start_capital


def test_peak_equity_bleibt_ungesetzt(monkeypatch, tmp_path):
    """DD-04-Schutz: peak_equity darf NICHT gesetzt werden — sonst werden
    Step-0-Exposure-Caps (-5 % hartkodiert) und 30 %-Kill-Pfade mitscharf."""
    cap = _run_paper(
        monkeypatch,
        tmp_path,
        {"cash": 90_000.0, "positions": {}, "equity_curve": []},
    )
    assert getattr(cap["ctx"], "peak_equity", None) is None


def test_treppe_loest_mit_verdrahtetem_ctx_aus(monkeypatch, tmp_path):
    """dd = (90k-100k)/100k = -10% -> Soft-Stufe (throttle 0.5)."""
    cap = _run_paper(
        monkeypatch,
        tmp_path,
        {
            "cash": 90_000.0,
            "positions": {},
            "equity_curve": [{"utc": "2026-07-03T00:00:00Z", "equity": 100_000.0}],
        },
    )
    policy = {
        "drawdown_policy": {
            "auto_kill_enabled": True,
            "levels": {"soft": -0.10, "hard": -0.15, "kill": -0.20},
        }
    }
    decision = _evaluate_auto_dd_kill_switch(
        cap["ctx"], TradingCycleResult(status="success"), policy
    )
    assert decision is not None, "Treppe blieb inert trotz gesetzter Equity"
    assert decision["level"] == "soft"
    assert decision["throttle_allowed_pct"] == pytest.approx(0.5)


def test_leere_kurve_hwm_ist_max_aus_equity_und_startkapital(monkeypatch, tmp_path):
    """Frischer Ledger ohne Kurve: hwm = max(current, start_capital)."""
    cap = _run_paper(
        monkeypatch,
        tmp_path,
        {"cash": 90_000.0, "positions": {}, "equity_curve": []},
    )
    ctx = cap["ctx"]
    assert float(ctx.hwm_equity) == pytest.approx(90_000.0)
    decision = _evaluate_auto_dd_kill_switch(
        ctx,
        TradingCycleResult(status="success"),
        {"drawdown_policy": {"auto_kill_enabled": True}},
    )
    assert decision is None


def test_kurve_ohne_equity_key_ist_harmlos(monkeypatch, tmp_path):
    """Fehlender "equity"-Key im Kurveneintrag -> 0.0; HWM bleibt durch
    max(start_capital, ...) korrekt (TR-04, abgedeckter Teilfall)."""
    cap = _run_paper(
        monkeypatch,
        tmp_path,
        {
            "cash": 90_000.0,
            "positions": {},
            "equity_curve": [
                {"utc": "2026-07-03T00:00:00Z"},
                {"utc": "2026-07-04T00:00:00Z", "equity": 80_000.0},
            ],
        },
    )
    assert cap["exit_code"] == 0
    assert float(cap["ctx"].hwm_equity) == pytest.approx(90_000.0)


def test_negative_equity_laesst_treppe_still(monkeypatch, tmp_path):
    """current negativ, hwm = start_capital > 0 -> dd tief im kill-Bereich?
    Nein: hwm>0 und current<0 -> dd < -1.0 -> kill wuerde greifen. Hier wird
    nur festgeschrieben, dass der Runner nicht crasht und die Treppe eine
    definierte Antwort liefert (kill), statt still None (TR-04-Teilfall)."""
    cap = _run_paper(
        monkeypatch,
        tmp_path,
        {"cash": -5_000.0, "positions": {}, "equity_curve": []},
    )
    assert cap["exit_code"] == 0
    decision = _evaluate_auto_dd_kill_switch(
        cap["ctx"],
        TradingCycleResult(status="success"),
        {"drawdown_policy": {"auto_kill_enabled": True}},
    )
    assert decision is not None and decision["level"] == "kill"


# --------------------------------------------------------------------------- #
# Step-6.4-Drossel-Semantik (pipeline/_tc_risk.py, Auftrag Hans 2026-08-09)
# --------------------------------------------------------------------------- #


def _tc_risk_mit_dd(monkeypatch, tmp_path: Path, dd_frac: float, orders: pd.DataFrame):
    """check_risk mit kuenstlichem dd; Kill-Switch-Aktivierung wird
    abgefangen (kein echter State-File-Write). Harness-Muster wie
    tests/test_tc_risk_fail_closed.py."""
    import src.assembled_core.pipeline._tc_risk as tc_risk
    from src.assembled_core.execution import kill_switch as ks

    aktiviert: list[dict] = []
    monkeypatch.setattr(
        ks,
        "activate_kill_switch",
        lambda **kwargs: aktiviert.append(kwargs),
    )
    # Alle anderen Gates neutralisieren: nur Step 6.4 ist Gegenstand.
    monkeypatch.setattr(
        tc_risk, "_apply_risk_controls_default", lambda ctx, orders: orders.copy()
    )
    monkeypatch.setattr(tc_risk, "_evaluate_var_gate", lambda *a, **k: None)
    monkeypatch.setattr(tc_risk, "_evaluate_circuit_breaker", lambda *a, **k: None)

    ts = pd.Timestamp("2025-06-26", tz="UTC")
    ctx = TradingContext(
        prices=pd.DataFrame({"timestamp": [ts], "symbol": ["AAPL"], "close": [100.0]}),
        as_of=ts,
        write_outputs=False,
        capital=100_000.0,
        current_equity=100_000.0 * (1 + dd_frac),
        hwm_equity=100_000.0,
    )
    ctx.qa_block_trading = False
    ctx.output_dir = str(tmp_path)  # Lifecycle-Log nicht ins Repo
    # Nur drawdown_policy gesetzt -> anti-churn/fat-finger skippen per Default.
    ctx._policy_cache = {
        "drawdown_policy": {
            "auto_kill_enabled": True,
            "levels": {"soft": -0.10, "hard": -0.15, "kill": -0.20},
        }
    }
    result = TradingCycleResult(status="success")
    out = tc_risk.check_risk(orders.copy(), result, ctx)
    return out, aktiviert


_ORDERS = pd.DataFrame(
    {"symbol": ["AAPL", "MSFT"], "side": ["buy", "sell"], "qty": [10.0, 4.0]}
)


def test_soft_drosselt_qty_ohne_kill_switch(monkeypatch, tmp_path):
    out, aktiviert = _tc_risk_mit_dd(monkeypatch, tmp_path, -0.11, _ORDERS)
    assert aktiviert == [], "soft darf den persistenten Kill-Switch NICHT aktivieren"
    assert list(out.orders_filtered["qty"]) == [5.0, 2.0], "qty nicht x0.5"
    assert out.meta["auto_dd_kill_switch"]["level"] == "soft"


def test_hard_drosselt_qty_auf_20_prozent(monkeypatch, tmp_path):
    out, aktiviert = _tc_risk_mit_dd(monkeypatch, tmp_path, -0.16, _ORDERS)
    assert aktiviert == []
    assert list(out.orders_filtered["qty"]) == [2.0, 0.8]
    assert out.meta["auto_dd_kill_switch"]["level"] == "hard"


def test_kill_blockt_alles_und_aktiviert_persistent(monkeypatch, tmp_path):
    out, aktiviert = _tc_risk_mit_dd(monkeypatch, tmp_path, -0.21, _ORDERS)
    assert len(aktiviert) == 1 and aktiviert[0]["throttle_pct"] == 0.0
    assert out.orders_filtered.empty
    assert out.meta["auto_dd_kill_switch"]["level"] == "kill"


def test_ohne_dd_keine_drossel(monkeypatch, tmp_path):
    out, aktiviert = _tc_risk_mit_dd(monkeypatch, tmp_path, -0.05, _ORDERS)
    assert aktiviert == []
    assert list(out.orders_filtered["qty"]) == [10.0, 4.0]
    assert "auto_dd_kill_switch" not in out.meta


def test_kurvenpunkt_mit_null_utc_wird_uebersprungen(monkeypatch, tmp_path, caplog):
    """DD2-06a: utc=null darf den Alt-Peak nicht durch den Filter schmuggeln
    (str(None)="None" vergliche lexikografisch True)."""
    import logging

    with caplog.at_level(logging.WARNING):
        cap = _run_paper(
            monkeypatch,
            tmp_path,
            {
                "cash": 90_000.0,
                "positions": {},
                "equity_curve": [
                    {"utc": None, "equity": 99_000.0},
                    {"utc": "2026-07-14T00:00:00Z", "equity": 91_000.0},
                ],
            },
            hwm_since="2026-07-02",
        )
    assert float(cap["ctx"].hwm_equity) == pytest.approx(91_000.0)
    assert any("unparsebar" in r.getMessage() for r in caplog.records)


def test_kurvenpunkt_mit_null_equity_crasht_nicht(monkeypatch, tmp_path):
    """DD2-06b: equity=null (Key da, Wert None) darf den Lauf nicht crashen."""
    cap = _run_paper(
        monkeypatch,
        tmp_path,
        {
            "cash": 90_000.0,
            "positions": {},
            "equity_curve": [
                {"utc": "2026-07-10T00:00:00Z", "equity": None},
                {"utc": "2026-07-14T00:00:00Z", "equity": 91_000.0},
            ],
        },
        hwm_since="2026-07-02",
    )
    assert cap["exit_code"] == 0
    assert float(cap["ctx"].hwm_equity) == pytest.approx(91_000.0)


def test_leergefilterter_batch_faellt_nicht_auf_rohbatch_zurueck():
    """F-senior-1/E-136: orders_filtered leer (bewusst gefiltert) -> die
    Ausfuehrung bekommt NICHT result.orders in voller Groesse."""
    result = TradingCycleResult(status="success")
    result.orders = _ORDERS.copy()
    result.orders_filtered = _ORDERS.iloc[0:0].copy()
    aus = paper_runner._orders_for_execution(result)
    assert aus.empty, "Fallback auf ungefilterten Batch — fail-open"

    # None (nie befuellt) -> leerer Frame mit Standard-Spalten, kein Crash.
    result.orders_filtered = None
    aus = paper_runner._orders_for_execution(result)
    assert aus.empty and "qty" in aus.columns


def test_ohne_hwm_since_zaehlt_der_alt_peak(monkeypatch, tmp_path, caplog):
    """F-senior-4/9b: fehlt hwm_since, ist die volle Kurve die HWM-Basis —
    Verhalten fixiert + [WARN] verlangt, damit der Config-Ausfall laut ist."""
    import logging

    with caplog.at_level(logging.WARNING):
        cap = _run_paper(
            monkeypatch,
            tmp_path,
            {
                "cash": 90_000.0,
                "positions": {},
                "equity_curve": [
                    {"utc": "2026-05-06T00:00:00Z", "equity": 99_000.0},
                    {"utc": "2026-07-14T00:00:00Z", "equity": 91_000.0},
                ],
            },
        )
    assert float(cap["ctx"].hwm_equity) == pytest.approx(99_000.0)
    assert any("hwm_since nicht gesetzt" in r.getMessage() for r in caplog.records)


def test_drossel_ohne_qty_spalte_blockt_fail_closed(monkeypatch, tmp_path):
    """F-senior-3/E-059: Orders ohne qty-Spalte sind nicht drosselbar ->
    Batch wird geblockt statt still ungedrosselt durchgereicht."""
    orders_ohne_qty = pd.DataFrame({"symbol": ["AAPL"], "side": ["buy"]})
    out, aktiviert = _tc_risk_mit_dd(monkeypatch, tmp_path, -0.11, orders_ohne_qty)
    assert aktiviert == []
    assert out.orders_filtered.empty
    assert out.meta["auto_dd_kill_switch"]["applied"] is False


def test_kill_und_leerer_batch_setzen_applied(monkeypatch, tmp_path):
    """F-auditor-3: applied-Kontrakt in allen Zweigen — kill=True,
    soft auf leerem Batch=False."""
    out, aktiviert = _tc_risk_mit_dd(monkeypatch, tmp_path, -0.21, _ORDERS)
    assert out.meta["auto_dd_kill_switch"]["applied"] is True

    leer = _ORDERS.iloc[0:0]
    out2, aktiviert2 = _tc_risk_mit_dd(monkeypatch, tmp_path, -0.11, leer)
    assert aktiviert2 == []
    assert out2.meta["auto_dd_kill_switch"]["applied"] is False
