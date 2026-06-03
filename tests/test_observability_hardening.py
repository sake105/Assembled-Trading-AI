"""Observability hardening for silent factor/data degradation.

Theme: "silent degradation logged at DEBUG". A genuinely broken factor or
never-written alt-data cache previously degraded to a neutral 0.0/empty value
and logged only at DEBUG — invisible at the default log level and therefore
indistinguishable from legitimately-absent data. These tests pin the new
behaviour:

  A9-strat (multifactor_v2):   first degradation of a factor logs at WARNING,
                               value still degrades to empty/0.0, second
                               degradation does NOT re-WARN (one-shot).
  A-data  (altdata_loader):    a missing cache path logs a WARNING once and
                               still returns the schema-correct empty frame.
  A31     (composite_score):   chart_pattern_score still returns 0.0 (unchanged)
                               and the dead-weight dilution is documented in
                               source (doc-only fix → documentation-presence
                               check is acceptable here).

The WARNING assertions are discriminating: pre-fix these paths logged at DEBUG,
so a caplog.WARNING expectation would fail.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# A9-strat: multifactor_v2 one-shot WARNING on factor degradation
# ---------------------------------------------------------------------------

import src.assembled_core.strategies.multifactor_v2 as mfv2

MFV2_LOGGER = "src.assembled_core.strategies.multifactor_v2"


@pytest.fixture(autouse=True)
def _reset_one_shot_state():
    """Ensure each test starts with empty one-shot guard sets (module-global)."""
    mfv2._FACTOR_DEGRADED_WARNED.clear()
    yield
    mfv2._FACTOR_DEGRADED_WARNED.clear()


def _force_mr_factor_to_raise(monkeypatch):
    """Make the inner dependency of _compute_mr_zscore_reversal_3d raise.

    The factor lazily does `from ...mean_reversion_factors import
    compute_mean_reversion_factors` inside its try-block, so patching the symbol
    on the source module is picked up at call time and routes into the except.
    """
    import src.assembled_core.features.mean_reversion_factors as mr_mod

    def _boom(*_a, **_k):
        raise RuntimeError("forced factor failure (test)")

    monkeypatch.setattr(mr_mod, "compute_mean_reversion_factors", _boom)


def test_a9_factor_degradation_warns_once_and_preserves_value(monkeypatch, caplog):
    _force_mr_factor_to_raise(monkeypatch)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02"]),
            "symbol": ["AAA", "AAA"],
            "close": [10.0, 11.0],
        }
    )

    # FIRST call: must WARN (pre-fix this was DEBUG only -> discriminating).
    with caplog.at_level(logging.WARNING, logger=MFV2_LOGGER):
        out1 = mfv2._compute_mr_zscore_reversal_3d(df)

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == MFV2_LOGGER
    ]
    assert warnings, "expected a WARNING on first factor degradation"
    assert any("mr_zscore_reversal_3d" in r.getMessage() for r in warnings), (
        "WARNING must name the degraded factor"
    )

    # Behaviour preserved: still degrades to an empty Series (no value change).
    assert isinstance(out1, pd.Series)
    assert out1.empty

    # SECOND call: one-shot guard must suppress a repeat WARNING.
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=MFV2_LOGGER):
        out2 = mfv2._compute_mr_zscore_reversal_3d(df)

    repeat_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == MFV2_LOGGER
    ]
    assert not repeat_warnings, "second degradation must NOT re-WARN (one-shot)"
    assert isinstance(out2, pd.Series) and out2.empty


def test_a9_second_degradation_logs_at_debug(monkeypatch, caplog):
    """After the one-shot WARNING, repeats are demoted to DEBUG (still visible
    when DEBUG is enabled, but quiet at default level)."""
    _force_mr_factor_to_raise(monkeypatch)
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01"]),
            "symbol": ["AAA"],
            "close": [10.0],
        }
    )

    # Prime the one-shot guard (first call emits WARNING).
    mfv2._compute_mr_zscore_reversal_3d(df)

    with caplog.at_level(logging.DEBUG, logger=MFV2_LOGGER):
        mfv2._compute_mr_zscore_reversal_3d(df)

    debug_repeats = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG
        and r.name == MFV2_LOGGER
        and "mr_zscore_reversal_3d" in r.getMessage()
    ]
    assert debug_repeats, "repeat degradation should log at DEBUG"


def test_a9_helper_is_per_factor_not_global(monkeypatch, caplog):
    """One-shot is keyed per factor name: a different factor still WARNs even
    after another factor has already warned."""
    mfv2._warn_factor_degraded("factor_alpha", RuntimeError("x"))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=MFV2_LOGGER):
        mfv2._warn_factor_degraded("factor_beta", RuntimeError("y"))
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("factor_beta" in r.getMessage() for r in warns)


# ---------------------------------------------------------------------------
# A-data: altdata_loader missing-cache one-time WARNING
# ---------------------------------------------------------------------------

import src.assembled_core.data.altdata_loader as altdata

ALTDATA_LOGGER = "src.assembled_core.data.altdata_loader"


@pytest.fixture(autouse=True)
def _reset_altdata_seen():
    altdata._MISSING_CACHE_WARNED.clear()
    yield
    altdata._MISSING_CACHE_WARNED.clear()


def test_adata_missing_cache_warns_once_and_returns_empty(tmp_path, caplog):
    missing_root = tmp_path / "no_such_dir"  # never created -> all caches absent
    as_of = pd.Timestamp("2025-01-01")

    with caplog.at_level(logging.WARNING, logger=ALTDATA_LOGGER):
        df = altdata.load_macro_indicators(as_of, root=missing_root)

    warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == ALTDATA_LOGGER
    ]
    assert warnings, "missing cache must WARN (pre-fix was DEBUG only)"
    assert any("macro" in r.getMessage() for r in warnings)

    # Schema-correct empty frame returned (behaviour preserved).
    assert df.empty
    assert list(df.columns) == ["timestamp", "macro_code", "value", "country"]

    # Second miss of the SAME cache type does not re-WARN (one-shot).
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=ALTDATA_LOGGER):
        df2 = altdata.load_macro_indicators(as_of, root=missing_root)
    repeat = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == ALTDATA_LOGGER
    ]
    assert not repeat, "repeat missing-cache must not re-WARN (one-shot per type)"
    assert df2.empty


def test_adata_missing_cache_is_per_type(tmp_path, caplog):
    """One-shot keyed per cache type: earnings + macro each WARN once."""
    missing_root = tmp_path / "no_such_dir"
    as_of = pd.Timestamp("2025-01-01")

    with caplog.at_level(logging.WARNING, logger=ALTDATA_LOGGER):
        altdata.load_earnings_history(["AAA"], as_of, root=missing_root)
        altdata.load_macro_indicators(as_of, root=missing_root)

    msgs = [
        r.getMessage()
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == ALTDATA_LOGGER
    ]
    assert any("earnings" in m for m in msgs)
    assert any("macro" in m for m in msgs)


# ---------------------------------------------------------------------------
# A31: composite_score chart_pattern dead-weight documentation
# ---------------------------------------------------------------------------

# NOTE: signals/__init__.py re-exports a *function* named ``composite_score``
# which shadows the submodule attribute on the package, so a plain
# ``import ... as composite_score`` binds to the function. Pull the real module
# object from sys.modules instead.
import sys as _sys

import src.assembled_core.signals.composite_score  # noqa: F401  (registers module)

composite_score = _sys.modules["src.assembled_core.signals.composite_score"]


def test_a31_chart_pattern_score_still_returns_zero():
    """Behaviour unchanged: still a hardcoded 0.0 placeholder."""
    assert composite_score.chart_pattern_score(pd.Series([1.0, 2.0, 3.0])) == 0.0
    assert composite_score.chart_pattern_score(pd.Series(dtype=float)) == 0.0


def test_a31_dead_weight_is_documented_in_source():
    """Doc-only fix → documentation-presence check.

    The permanent ~10% dead weight must be documented at the function and at the
    Dim-5 call site so the dilution is visible to a reader, not silent.
    """
    src_path = Path(inspect.getfile(composite_score))
    text = src_path.read_text(encoding="utf-8")

    # Function docstring documents the dead weight.
    doc = inspect.getdoc(composite_score.chart_pattern_score) or ""
    assert "DEAD-WEIGHT" in doc.upper() or "dead weight" in doc.lower()
    assert "0.10" in doc

    # Call site comment documents the dead weight too.
    assert "DEAD WEIGHT" in text or "dead weight" in text.lower()
    # The fixed-in-every-regime nature is called out.
    assert "every regime" in text.lower() or "all regimes" in text.lower()
