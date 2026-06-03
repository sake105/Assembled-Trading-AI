"""Batch 11 — core-misc observability / honesty fixes (Diagnostik §core-misc MINOR).

These tests verify that previously-silent failure paths now emit an observable
log record (WARNING/DEBUG) while still degrading gracefully (no raise, partial
result still returned), and that misleading docs were corrected.

Pre-fix, each wrapped failure was a bare ``except ...: pass`` — the caplog
assertions below would FAIL because nothing was logged.
"""

from __future__ import annotations

import inspect
import logging
import sqlite3
import sys
import types

import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# C1 — certify/generator.py : three formerly-silent fingerprint failures
# ---------------------------------------------------------------------------


def test_c1_package_enumeration_failure_is_logged(monkeypatch, caplog):
    """C1 site 1: package-version enumeration failure -> WARNING, partial result."""
    import importlib.metadata as im

    from src.assembled_core.certify import generator

    def _boom():
        raise RuntimeError("distributions exploded")

    monkeypatch.setattr(im, "distributions", _boom)

    with caplog.at_level(logging.WARNING, logger=generator.__name__):
        env = generator.get_environment_fingerprint()

    # Degrade-don't-crash: still returns a fingerprint with empty package_hashes.
    assert env.package_hashes == {}
    assert any(
        "package-version enumeration failed" in r.getMessage()
        and r.levelno == logging.WARNING
        for r in caplog.records
    ), "expected WARNING naming the incomplete package_hashes field"


def test_c1_numpy_seed_failure_is_logged(monkeypatch, caplog):
    """C1 site 2: numpy global-seed capture failure -> WARNING, seed key absent."""
    np = pytest.importorskip("numpy")

    from src.assembled_core.certify import generator

    def _boom():
        raise RuntimeError("get_state exploded")

    monkeypatch.setattr(np.random, "get_state", _boom)

    with caplog.at_level(logging.WARNING, logger=generator.__name__):
        env = generator.get_environment_fingerprint()

    # python_random seed still present; numpy_global silently missing -> now logged.
    assert "python_random" in env.random_seeds
    assert "numpy_global" not in env.random_seeds
    assert any(
        "numpy global-seed capture failed" in r.getMessage()
        and r.levelno == logging.WARNING
        for r in caplog.records
    ), "expected WARNING naming the missing numpy_global seed"


def test_c1_output_summary_load_failure_is_logged(tmp_path, caplog):
    """C1 site 3: corrupt summary.json -> WARNING, empty summary_metrics."""
    from src.assembled_core.certify import generator

    # Write an intentionally-corrupt summary.json so json.load() raises.
    (tmp_path / "summary.json").write_text("{ this is not json ", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger=generator.__name__):
        out = generator.build_output_fingerprint(tmp_path)

    # Degrade-don't-crash: fingerprint still built, summary metrics empty.
    assert out.summary_metrics == {}
    assert any(
        "failed to load output summary" in r.getMessage()
        and r.levelno == logging.WARNING
        for r in caplog.records
    ), "expected WARNING naming the failed output-summary load"


# ---------------------------------------------------------------------------
# C2 — certify/mlflow_integration.py : formerly-silent log_metric failure
#
# mlflow is optional and (in this env) not installed. _log_output_metrics
# imports mlflow at call time, so we inject a fake mlflow whose log_metric
# raises — exercising the real (now-logged) except path. This is the
# unit-reachable site; the log_param site (~line 80) sits inside a full
# mlflow.start_run() orchestration and is not reachable without a much
# heavier fake — it shares the identical fix pattern.
# ---------------------------------------------------------------------------


def test_c2_log_metric_failure_is_logged(monkeypatch, caplog):
    from src.assembled_core.certify import mlflow_integration

    fake_mlflow = types.ModuleType("mlflow")

    def _boom(*_a, **_k):
        raise RuntimeError("mlflow backend down")

    fake_mlflow.log_metric = _boom
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    class _Out:
        sharpe_ratio = 1.23  # numeric -> attempts log_metric -> raises

    with caplog.at_level(logging.WARNING, logger=mlflow_integration.__name__):
        # Must not raise despite the backend failure.
        mlflow_integration._log_output_metrics(_Out())

    assert any(
        "log_metric failed" in r.getMessage() and r.levelno == logging.WARNING
        for r in caplog.records
    ), "expected WARNING that an MLflow log_metric call was skipped"


# ---------------------------------------------------------------------------
# C3 — attribution/storage.py : formerly-silent WAL PRAGMA fallback
# ---------------------------------------------------------------------------


def test_c3_wal_setup_failure_is_logged_debug(tmp_path, monkeypatch, caplog):
    from src.assembled_core.attribution import storage as storage_mod
    from src.assembled_core.attribution.storage import AttributionStore

    store = AttributionStore(db_path=str(tmp_path / "attr.db"))

    # sqlite3.Connection is an immutable C type, so we can't patch its method
    # directly. Instead wrap the real connection in a proxy that raises
    # sqlite3.Error on the WAL/synchronous PRAGMA calls and delegates all other
    # SQL to the real connection (so it stays usable -> degrade-don't-crash).
    real_connect = sqlite3.connect

    class _PragmaFailingConn:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, *args, **kwargs):
            if "PRAGMA" in sql.upper():
                raise sqlite3.Error("read-only filesystem")
            return self._inner.execute(sql, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._inner, name)

    def _fake_connect(*args, **kwargs):
        return _PragmaFailingConn(real_connect(*args, **kwargs))

    monkeypatch.setattr(storage_mod.sqlite3, "connect", _fake_connect)

    with caplog.at_level(logging.DEBUG, logger=storage_mod.__name__):
        conn = store._connect()
        # Connection must still be usable (degrade-don't-crash).
        conn.execute("SELECT 1")
        conn.close()

    assert any(
        "WAL/synchronous PRAGMA setup failed" in r.getMessage()
        and r.levelno == logging.DEBUG
        for r in caplog.records
    ), "expected DEBUG noting WAL fallback to default journal"


# ---------------------------------------------------------------------------
# C4 — compliance/tax_report.py : docstring no longer claims "stub(s)"
# ---------------------------------------------------------------------------


def test_c4_tax_report_docstring_not_stub():
    from src.assembled_core.compliance import tax_report

    doc = (inspect.getdoc(tax_report) or "").lower()
    assert "stub" not in doc, (
        "tax_report is fully implemented (real summarize_closed_lots body); "
        "module docstring must not advertise it as a stub"
    )
    # Sanity: the function is genuinely implemented, not a NotImplementedError shell.
    summary = tax_report.summarize_closed_lots([], year=2026)
    assert summary.year == 2026


# ---------------------------------------------------------------------------
# C5 — strategy/ (singular) marked research-only / not-live
# ---------------------------------------------------------------------------


def test_c5_strategy_pkg_marked_research_only():
    import src.assembled_core.strategy as strategy_pkg

    doc = (inspect.getdoc(strategy_pkg) or "").lower()
    assert "research-only" in doc or "research only" in doc, (
        "strategy/ package docstring must mark it research-only"
    )
    assert "live" in doc, "docstring must clarify it is not on the live path"


def test_c5_composite_weights_comment_present():
    """The config source must label CompositeWeights as research-only."""
    from src.assembled_core.strategy import config as cfg_mod

    src = inspect.getsource(cfg_mod)
    assert "RESEARCH-ONLY weights" in src
    assert "factor_weights_by_regime.json" in src


def test_c5_strategy_weight_values_unchanged():
    """Guard: the labeling fix must NOT alter any weight VALUES (doc-only)."""
    from src.assembled_core.strategy.config import CompositeWeights

    w = CompositeWeights()
    assert w.model_dump() == {
        "mtf": 0.15,
        "classical_ta": 0.20,
        "microstructure": 0.10,
        "volume_profile": 0.10,
        "chart_pattern": 0.05,
        "vol_surface": 0.10,
        "breadth": 0.15,
        "seasonality": 0.05,
        "news": 0.10,
    }


# Non-unit-reachable site (documented, not a tautology):
# - mlflow_integration.log_certificate_to_mlflow's env.* log_param failure (~line 80)
#   requires a full fake mlflow.start_run() context manager + experiment plumbing;
#   it shares the identical fix as the log_metric path tested above and is left
#   to integration coverage rather than a heavy fake here.
