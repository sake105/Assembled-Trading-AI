"""Batch 14 — portfolio scipy-import hygiene + qa metric honesty.

Covers three fixes from docs/Diagnostik.md (portfolio + qa sections):

- **P1** — ``portfolio/optimizers.py`` must be
  IMPORTABLE even when scipy is absent (scipy.optimize is imported lazily inside
  the solver functions). (``dro_portfolio`` 2026-08-17 archiviert, 6.4 T3.) Calling a solver without scipy must raise a *clear,
  informative* ``ImportError`` — not a bare ``ModuleNotFoundError`` at an
  unexpected point.
- **Q1** — ``qa/metrics.probability_backtest_overfitting`` is a single-partition
  OOS rank-fraction heuristic, NOT the Bailey/López de Prado CSCV-PBO. The
  docstring must say so plainly and a one-time ``DeprecationWarning`` must fire.
- **Q2** — ``qa/metrics.deflated_sharpe_ratio`` returns a Z-SCORE, distinct from
  the canonical probability-DSR in ``qa/deflated_sharpe.deflated_sharpe``. The
  docstring must say so and a one-time ``DeprecationWarning`` must fire.

IMPORTANT: this module deliberately does NOT ``importorskip('scipy')`` — the
whole point of P1 is the no-scipy import/call path. The numeric outputs of the
qa functions are NOT changed by the honesty edits (asserted here too).
"""

from __future__ import annotations

import builtins
import contextlib
import importlib
import sys

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper: hide scipy for the duration of a block (import AND call)
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def _scipy_hidden(*reimport_modules: str):
    """Context manager that simulates scipy being absent.

    Patches ``builtins.__import__`` to raise ``ImportError`` for any
    ``scipy``-prefixed import, and evicts ``scipy*`` plus the named target
    modules from ``sys.modules`` so a fresh import re-runs without scipy.
    The patch stays active for the WHOLE ``with`` block — so both importing
    a target module AND calling its (lazy-importing) solvers see no scipy.
    Restores ``builtins.__import__`` and the saved modules on exit.
    """
    real_import = builtins.__import__
    saved_modules = {
        name: mod
        for name, mod in list(sys.modules.items())
        if name.split(".")[0] == "scipy" or name in reimport_modules
    }

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "scipy" or name.startswith("scipy."):
            raise ImportError(f"simulated-absent scipy (blocked import of {name})")
        return real_import(name, globals, locals, fromlist, level)

    for name in list(sys.modules):
        if name.split(".")[0] == "scipy" or name in reimport_modules:
            del sys.modules[name]

    builtins.__import__ = fake_import
    try:
        yield
    finally:
        builtins.__import__ = real_import
        # Drop the no-scipy versions so other tests get clean (real) imports.
        for name in reimport_modules:
            sys.modules.pop(name, None)
        for name, saved in saved_modules.items():
            sys.modules[name] = saved


def _import_module_without_scipy(module_name: str):
    """Import ``module_name`` fresh under :func:`_scipy_hidden` and return it."""
    with _scipy_hidden(module_name):
        return importlib.import_module(module_name)


# ===========================================================================
# P1 — portfolio modules import without scipy; solvers raise clear error
# ===========================================================================
# ENTFERNT 2026-08-17: test_dro_portfolio_imports_without_scipy —
# portfolio/dro_portfolio archiviert (Audit-Plan 6.4 Tranche 3, nur
# Test-Referenzen; dedizierte Datei tests/test_portfolio_dro.py
# mit-archiviert). optimizers-Tests bleiben unveraendert.


def test_optimizers_imports_without_scipy():
    """optimizers must import cleanly even when scipy is unavailable."""
    mod = _import_module_without_scipy("src.assembled_core.portfolio.optimizers")
    assert hasattr(mod, "min_variance_weights")
    assert mod._SCIPY_AVAILABLE is False


# ENTFERNT 2026-08-17: test_dro_solver_without_scipy_raises_clear_importerror
# (dro_portfolio archiviert, s. o.).


def test_optimizer_without_scipy_raises_clear_importerror():
    """Calling a constrained optimizer without scipy raises a clear ImportError."""
    import pandas as pd

    module_name = "src.assembled_core.portfolio.optimizers"
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    with _scipy_hidden(module_name):
        mod = importlib.import_module(module_name)
        # long_only=True with default bounds routes through the scipy SLSQP path.
        with pytest.raises(ImportError) as exc:
            mod.min_variance_weights(cov, long_only=True)
    msg = str(exc.value).lower()
    assert "scipy" in msg
    assert "minimize" in msg or "optimizer" in msg


# ---------------------------------------------------------------------------
# P1 — with scipy present (normal env), solvers still work numerically.
# These run only when scipy is actually installed; they are NOT importorskip'd
# at module level (that would skip the whole no-scipy point above).
# ---------------------------------------------------------------------------
# ENTFERNT 2026-08-17: test_dro_solver_with_scipy_runs (dro_portfolio
# archiviert, s. o.).


def test_optimizer_with_scipy_runs():
    pytest.importorskip("scipy")
    import pandas as pd

    from src.assembled_core.portfolio.optimizers import min_variance_weights

    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"]
    )
    res = min_variance_weights(cov, long_only=True)
    assert abs(float(res.weights.sum()) - 1.0) < 1e-6


# ===========================================================================
# Q1 — probability_backtest_overfitting honesty (docstring + warning + numeric)
# ===========================================================================
def test_pbo_docstring_disclaims_blp_cscv():
    from src.assembled_core.qa.metrics import probability_backtest_overfitting

    doc = (probability_backtest_overfitting.__doc__ or "").lower()
    assert "not" in doc and "cscv" in doc, "must disclaim it is not CSCV-PBO"
    assert "heuristic" in doc
    # Must warn it is not comparable to BLP thresholds / not for go-live.
    assert "go-live" in doc or "deployment" in doc or "threshold" in doc


def test_pbo_emits_deprecation_warning_once():
    import src.assembled_core.qa.metrics as metrics

    # Reset the one-time guard so this test is order-independent.
    metrics._PBO_HEURISTIC_WARNED = False

    is_m = np.array([[0.02, 0.01, 0.03], [0.01, 0.00, 0.01]])
    oos_m = np.array([[0.01, 0.02, 0.00], [0.02, 0.01, 0.03]])

    with pytest.warns(DeprecationWarning, match="CSCV|BLP|heuristic"):
        val = metrics.probability_backtest_overfitting(is_m, oos_m)
    assert 0.0 <= val <= 1.0


def test_pbo_numeric_output_unchanged():
    """Honesty edit must not change the numeric result (single-split rank frac)."""
    from src.assembled_core.qa.metrics import probability_backtest_overfitting

    # Strategy 0 is best IS; in OOS it ranks index 1 of 2 -> fraction 0.5.
    is_m = np.array([[0.05, 0.05, 0.05], [0.01, 0.01, 0.01]])
    oos_m = np.array([[0.01, 0.01, 0.01], [0.05, 0.05, 0.05]])
    val = probability_backtest_overfitting(is_m, oos_m)
    assert val == pytest.approx(0.5)


# ===========================================================================
# Q2 — deflated_sharpe_ratio z-score honesty (docstring + warning + numeric)
# ===========================================================================
def test_dsr_zscore_docstring_marks_zscore_not_probability():
    from src.assembled_core.qa.metrics import deflated_sharpe_ratio

    doc = (deflated_sharpe_ratio.__doc__ or "").lower()
    assert "z-score" in doc
    assert (
        "not a probability" in doc
        or "not, a probability" in doc
        or ("not" in doc and "probability" in doc)
    )
    # Must point to the canonical probability-DSR module for gating.
    assert "deflated_sharpe" in doc


def test_canonical_dsr_docstring_marks_canonical():
    # NOTE: ``qa/__init__`` re-exports the *function* ``deflated_sharpe`` into the
    # ``qa`` package namespace, which shadows the submodule attribute. Use
    # importlib to fetch the actual MODULE object and read its module docstring.
    canonical = importlib.import_module("src.assembled_core.qa.deflated_sharpe")
    doc = (canonical.__doc__ or "").lower()
    assert "canonical" in doc
    assert "probability" in doc


def test_dsr_zscore_emits_deprecation_warning_once():
    import src.assembled_core.qa.metrics as metrics

    metrics._DSR_ZSCORE_DEPRECATION_WARNED = False
    with pytest.warns(DeprecationWarning, match="z-score|gating|probability"):
        val = metrics.deflated_sharpe_ratio(sharpe_annual=1.0, n_obs=252, n_tests=1)
    assert np.isfinite(val)


def test_dsr_zscore_numeric_output_unchanged():
    """Honesty edit must not change the numeric z-score result."""
    from src.assembled_core.qa.metrics import deflated_sharpe_ratio

    # n_tests=1, skew=0, kurtosis=3 -> expected_max_sharpe=0, pure z-score.
    # std(SR) = sqrt((1 + SR^2/2)/n_obs); SR=1.0, n_obs=252.
    expected_std = np.sqrt((1.0 + 0.5) / 252.0)
    expected_z = 1.0 / expected_std
    val = deflated_sharpe_ratio(
        sharpe_annual=1.0, n_obs=252, n_tests=1, skew=0.0, kurtosis=3.0
    )
    assert val == pytest.approx(expected_z, rel=1e-9)
    # And it is clearly NOT a probability (z-score > 1 here).
    assert val > 1.0
