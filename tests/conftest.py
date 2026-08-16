"""Pytest configuration and shared fixtures for backtest regression tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow bare `from assembled_core.X import Y` in test files (no `src.` prefix).
# The package is not installed as an editable package, so we expose `src/` on
# sys.path so both import styles work side-by-side.
_src_path = str(Path(__file__).parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

import pandas as pd
import pytest

from src.assembled_core.portfolio.position_sizing import compute_target_positions

# ---------------------------------------------------------------------------
# P0 A9 (Deep Run v2, 2026-04-18) — marker consolidation.
#
# Eight legacy phase-markers are aliased onto the four canonical markers
# (fast / integration / regression / smoke) so old selections like
# `-m phase12` keep working while the CI can already target `-m fast`.
# Sunset schedule: docs/tech_debt/markers_migration.md (2026-07-01).
# ---------------------------------------------------------------------------

_LEGACY_MARKER_ALIASES: dict[str, str] = {
    "phase4": "fast",
    "phase6": "fast",
    "phase7": "fast",
    "phase8": "fast",
    "phase9": "fast",
    "phase10": "fast",
    "phase11": "fast",
    "phase12": "fast",
    "phase13": "fast",
    "phase_zero": "regression",
    "phase_speed": "regression",
    "phase_realism": "regression",
    "phase_depth": "regression",
}


def pytest_collection_modifyitems(config, items) -> None:  # pragma: no cover - wiring
    for item in items:
        legacy_markers = [
            m.name for m in item.iter_markers() if m.name in _LEGACY_MARKER_ALIASES
        ]
        for legacy in legacy_markers:
            canonical = _LEGACY_MARKER_ALIASES[legacy]
            if canonical not in {m.name for m in item.iter_markers()}:
                item.add_marker(canonical)


@pytest.fixture(autouse=True)
def _isolate_operational_stores(monkeypatch, tmp_path):
    """Redirect ALL tests away from the real operational stores.

    Root cause (2026-07-24): tests calling the broker path without an
    explicit ``intent_store_path`` (e.g. test_broker_execution.py) wrote
    MSFT/AAPL fixture intents into the REAL ``output/ops/intent_store.jsonl``
    for months. Harmless while preflight only warned — but the 2026-07-22
    fail-closed preflight (W7a) then correctly BLOCKED the live pilot on
    same-day test residue (runs 22.+23.07. 21:30). Same contamination class
    as the order-lifecycle journal leak (Stage-1 M1, 2026-07-22).

    This autouse fixture closes that class IN-PROCESS: the module-level default
    paths are pointed at tmp_path for every test. Tests that pass explicit paths
    are unaffected.

    NOT closed by the module-attribute patches: subprocesses. 42 test files
    under tests/ spawn one, and a child re-imports the module with its real
    default. Measured 2026-08-15 during a full suite run:
    output/audit/trading_decisions.jsonl grew by 681 rows while this fixture was
    active. Stores that expose an ENV override (kill switch, reconcile audit,
    and since 2026-08-15 the audit trail via AUDIT_TRAIL_PATH) are isolated on
    both sides because env vars are inherited; the rest are in-process only.
    An earlier version of this docstring said "structurally impossible" without
    that qualifier — see KNOWN_ISSUES §0.06 (e).
    """
    # Each patch is individually import-guarded: importing a module here
    # pulls its PACKAGE __init__ chain (qa/__init__ -> walk_forward ->
    # pipeline -> policy_loader -> yaml), and the evidence-pack CI runs a
    # MINIMAL env without PyYAML — an unguarded qa_gates import broke both
    # Evidence workflows on 2026-07-24 (commit 06790585). If a module is
    # not importable in an env, no test in that env can write through it
    # either, so skipping its patch loses no isolation.
    import importlib

    _patches = [
        (
            "src.assembled_core.execution.intent_store",
            "_DEFAULT_STORE_PATH",
            tmp_path / "intent_store.jsonl",
        ),
        (
            "src.assembled_core.ops.order_lifecycle_log",
            "DEFAULT_LIFECYCLE_LOG_PATH",
            tmp_path / "order_lifecycle.jsonl",
        ),
        (
            "src.assembled_core.qa.qa_gates",
            "QA_BLOCK_FLAG_PATH",
            tmp_path / "qa_block.json",
        ),
        # test-runner MAJOR 2026-07-24: pre-existing qa_gates/orchestrator
        # tests overwrote the REAL output/ops/crisis_alpha_state.json on
        # every run — risk_controls.py reads it as the crisis-alpha PAUSE
        # kill-switch fallback, so test residue could silently disarm a
        # live PAUSE. Same contamination class, same fix.
        (
            "src.assembled_core.events.crisis_alpha.state_machine",
            "_DEFAULT_STATE_PATH",
            tmp_path / "crisis_alpha_state.json",
        ),
        # 2026-08-15: the decision audit trail was the next store still
        # unisolated. Measured on the real file before the cleanup: 1,129 rows
        # carrying run_id=test_run_001 / test_run_123 or symbols AAA/BBB among
        # 434,750 production rows. Those rows are preserved verbatim in
        # archive/orphaned_data_2026-08-15/trading_decisions.REMOVED_TESTROWS.jsonl
        # — grep the live file and you will now find none, which is the point;
        # the archived copy is what makes the number re-checkable.
        # Less acute than the kill-switch case (nothing reads this file as a
        # control input), but it is the same contamination class, and an audit
        # trail that records decisions which never happened is not an audit
        # trail. Listed here rather than after the next incident — the point of
        # E-139 is that extending isolation only for whatever just bit leaves
        # the rest of the family open.
        (
            "src.assembled_core.ops.audit_trail",
            "_DEFAULT_OUTPUT",
            tmp_path / "trading_decisions.jsonl",
        ),
        # 2026-08-16: yfinance_source schreibt seit dem E-112-Anschluss ein
        # PullLog-Protokoll nach output/ops/ (DEFAULT_LOG_DIR). Erster
        # Testlauf ohne diese Isolation legte sofort ein reales
        # pull_log_yfinance_*.json an — gleiche Kontaminationsklasse (E-139:
        # die ganze Familie schliessen, nicht nur den letzten Biss).
        (
            "src.assembled_core.data.pull_log",
            "DEFAULT_LOG_DIR",
            tmp_path / "pull_logs",
        ),
    ]
    for _mod_name, _attr, _target in _patches:
        try:
            _mod = importlib.import_module(_mod_name)
        except ImportError:
            continue  # minimal env — module (and thus its writers) unavailable
        monkeypatch.setattr(_mod, _attr, _target)

    # 2026-08-09: Kill-Switch-Stores in dieselbe Isolation. Ein Testlauf hat
    # den REALEN output/ops/kill_switch_state.json engagiert (auto_dd_kill,
    # dd=-90% aus Test-Equity) — der naechste Pilot-Zyklus waere geblockt
    # gewesen, Aufraeumen nur per OPERATOR_KILL_TOKEN. Gleiche
    # Kontaminationsklasse wie oben; kill_switch.py loest seine Pfade ueber
    # diese Env-Vars auf (Lock haengt am State-Verzeichnis mit dran).
    monkeypatch.setenv(
        "ASSEMBLED_KILL_SWITCH_STATE", str(tmp_path / "kill_switch_state.json")
    )
    monkeypatch.setenv(
        "ASSEMBLED_KILL_SWITCH_SENTINEL", str(tmp_path / ".kill_switch_active")
    )
    monkeypatch.setenv(
        "ASSEMBLED_KILL_SWITCH_AUDIT", str(tmp_path / "kill_switch_audit.jsonl")
    )
    # TR-4: dieselbe Verteidigungslinie fuer die zwei restlichen Env-Vars —
    # ein extern gesetztes ASSEMBLED_KILL_SWITCH=1 saehe sonst JEDEN Test als
    # engaged, ein externer Lock-Pfad umginge die tmp_path-Ableitung. Tests,
    # die die Vars selbst setzen, laufen NACH dieser Fixture und gewinnen.
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH", raising=False)
    monkeypatch.delenv("ASSEMBLED_KILL_SWITCH_LOCK", raising=False)
    # F-senior-4 (Stage 2, 2026-08-09): gleiche Klasse — Reconcile-Fixtures
    # schrieben synthetische severity=fail-Zeilen ins REALE
    # output/ops/reconciliation_audit.jsonl (accounting/reconciliation.py:25
    # liest diese Env-Var).
    monkeypatch.setenv(
        "ASSEMBLED_RECONCILE_AUDIT", str(tmp_path / "reconciliation_audit.jsonl")
    )

    # SUBPROCESS GAP (2026-08-15). The module-attribute patches above only bind
    # in THIS process. 42 test files under tests/ spawn subprocesses, and a
    # child re-imports the module with its real default — so the attribute
    # patch never applies there. Measured during a full suite run:
    # output/audit/trading_decisions.jsonl grew by 681 rows (447,625 ->
    # 448,306) while the isolation was supposedly active, with backtest
    # decisions dated 2023-02-23.
    #
    # Environment variables ARE inherited by children, so anything with an env
    # override gets it here as well. audit_trail._get_output_path() prefers
    # AUDIT_TRAIL_PATH over the module default, which makes this the one lever
    # that reaches both sides.
    #
    # HONEST SCOPE: this closes the audit trail for subprocesses. The other
    # module-attribute patches above (intent_store, order_lifecycle,
    # qa_gates, crisis_alpha, factor_store) have NO env override and therefore
    # remain in-process only. The fixture docstring's "structurally impossible"
    # is true for the in-process half and overstated for the rest — tracked in
    # KNOWN_ISSUES §0.06 rather than silently left as a stronger claim than the
    # code earns.
    monkeypatch.setenv("AUDIT_TRAIL_PATH", str(tmp_path / "trading_decisions.jsonl"))

    # 2026-08-15: the factor store was the last unisolated operational store,
    # and it is the one that actually bit. Tests write computed panels into the
    # REAL output/factors/ via store_factors and read them back on the next
    # run. Its cache key is a universe hash with NO code or feature version
    # (factor_store.compute_universe_key), so a panel computed by older code
    # survives changes to feature/sizing logic and is silently reused.
    #
    # Observed effect: three tests in test_pipeline_trading_cycle_smoke.py fail
    # in this working tree and pass in a fresh worktree, purely because the
    # fresh tree has an empty output/factors/. Proven causally: pointing the
    # factors root at tmp_path turns the same file from 3 failed to 7 passed.
    # The failure looks like a code regression and is a cache artefact.
    #
    # Patched as a function (not a module attribute) because the root is
    # resolved per call via _default_factors_root().
    try:
        import src.assembled_core.data.factor_store as _fs_mod

        _iso_factors_root = tmp_path / "factors"
        monkeypatch.setattr(_fs_mod, "_default_factors_root", lambda: _iso_factors_root)
    except ImportError:  # pragma: no cover - minimal env without the module
        pass


@pytest.fixture
def golden_mini_backtest_data():
    """Golden mini backtest fixture: 2-3 symbols, 5-10 days, deterministic signals.

    This fixture provides a small, deterministic dataset for regression testing.
    It ensures that optimizations (vectorization, Numba) don't change the logic.

    Returns:
        Dictionary with:
        - prices: DataFrame with columns: timestamp, symbol, close
        - signals: DataFrame with columns: timestamp, symbol, direction, score
        - expected_orders_count: Expected number of orders
        - expected_equity_start: Expected starting equity
        - expected_equity_end: Expected ending equity (approximate)
    """
    # Create deterministic price data: 3 symbols, 10 days
    symbols = ["AAPL", "MSFT", "GOOGL"]
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D", tz="UTC")

    prices_data = []
    for date in dates:
        for i, symbol in enumerate(symbols):
            # Deterministic prices: base price + day offset + symbol offset
            base_price = 100.0 + (i * 10.0)  # AAPL=100, MSFT=110, GOOGL=120
            day_offset = float((date - dates[0]).days) * 0.5  # Small daily increment
            price = base_price + day_offset
            prices_data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "close": price,
                }
            )

    prices = pd.DataFrame(prices_data)

    # Create deterministic signals: simple trend-following
    # Day 1-3: All LONG
    # Day 4-6: AAPL LONG, others NEUTRAL
    # Day 7-10: All NEUTRAL
    signals_data = []
    for date in dates:
        day_idx = (date - dates[0]).days
        for symbol in symbols:
            if day_idx < 3:
                direction = "LONG"
                score = 1.0
            elif day_idx < 6 and symbol == "AAPL":
                direction = "LONG"
                score = 0.8
            else:
                direction = "NEUTRAL"
                score = 0.0

            signals_data.append(
                {
                    "timestamp": date,
                    "symbol": symbol,
                    "direction": direction,
                    "score": score,
                }
            )

    signals = pd.DataFrame(signals_data)

    # Expected values (computed from original implementation)
    # These will be validated against actual results
    expected_orders_count = 6  # Approximate: 3 buys on day 1, 3 sells on day 4, etc.
    expected_equity_start = 10000.0
    expected_equity_end = 10000.0  # Approximate, will be validated

    return {
        "prices": prices,
        "signals": signals,
        "symbols": symbols,
        "dates": dates,
        "expected_orders_count": expected_orders_count,
        "expected_equity_start": expected_equity_start,
        "expected_equity_end": expected_equity_end,
    }


@pytest.fixture
def position_sizing_fn():
    """Position sizing function for golden backtest."""

    def sizing_fn(signals_df, capital):
        return compute_target_positions(
            signals_df, total_capital=capital, equal_weight=True, top_n=3
        )

    return sizing_fn
