"""Edge-Case-Tests für LiveDecisionEngine.

Production-Trading muss robust sein gegen:
- NaN-Returns einzelner Symbole
- Symbol-Delisting (column dropping)
- Empty/Korrupt-Bootstrap-State
- Time-Series-Gaps (weekend, holiday)
- Extreme Vol-Werte (0, inf)
- Duplicate-Datestamps
"""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from erweiterung.live.live_decision_engine import (
    EngineState,
    LiveDecisionEngine,
    LiveEngineConfig,
)


def _make_clean_returns(n_days: int = 500, n_eq: int = 8, n_xa: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=n_days, freq="B", tz="UTC")
    eq = pd.DataFrame(
        rng.normal(0.0005, 0.012, (n_days, n_eq)),
        index=idx, columns=[f"E{i}" for i in range(n_eq)],
    )
    xa = pd.DataFrame(
        rng.normal(0.0003, 0.008, (n_days, n_xa)),
        index=idx, columns=[f"X{i}" for i in range(n_xa)],
    )
    return eq, xa


# ============================================================================
# NaN handling
# ============================================================================


def test_bootstrap_with_nan_returns_works():
    eq, xa = _make_clean_returns()
    # Introduce NaN in some symbols
    eq.iloc[10:50, 0] = np.nan
    eq.iloc[100:110, :] = np.nan  # whole rows
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    out = engine.decide_next()
    assert np.isfinite(out["sa_leverage"])
    assert np.isfinite(out["xa_ew_leverage"])


def test_update_with_nan_symbol_doesnt_crash():
    eq, xa = _make_clean_returns()
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    next_date = eq.index[-1] + pd.Timedelta(days=1)
    eq_row = pd.Series(0.001, index=eq.columns)
    eq_row.iloc[0] = np.nan  # one NaN
    xa_row = pd.Series(0.0005, index=xa.columns)
    engine.update_with_new_day(next_date, eq_row, xa_row)
    out = engine.decide_next()
    assert np.isfinite(out["sa_leverage"])


def test_decide_with_all_nan_history_returns_safe_defaults():
    """Bei komplett NaN-History muss decide_next() safe defaults zurückgeben."""
    eq, xa = _make_clean_returns(n_days=100)
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    # Corrupt state with NaN
    engine.state.eq_log_return_history.iloc[:] = np.nan
    out = engine.decide_next()
    # SA-Leverage sollte safe fallback sein (1.0 oder 0)
    assert 0.0 <= out["sa_leverage"] <= 2.0


# ============================================================================
# Symbol delisting / column changes
# ============================================================================


def test_update_with_new_symbol_in_returns():
    """Wenn ein neues Symbol in update_with_new_day kommt das nicht im bootstrap war."""
    eq, xa = _make_clean_returns()
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    next_date = eq.index[-1] + pd.Timedelta(days=1)
    # Add NEW symbol that wasn't in bootstrap
    eq_row = pd.Series(0.001, index=list(eq.columns) + ["NEW_SYM"])
    xa_row = pd.Series(0.0005, index=xa.columns)
    # Should not crash
    try:
        engine.update_with_new_day(next_date, eq_row, xa_row)
        ok = True
    except Exception:
        ok = False
    assert ok


def test_update_with_missing_symbol():
    """Wenn ein Symbol aus bootstrap nicht mehr in update kommt (Delisting)."""
    eq, xa = _make_clean_returns()
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    next_date = eq.index[-1] + pd.Timedelta(days=1)
    # Drop first column (delisting simulation)
    eq_row = pd.Series(0.001, index=eq.columns[1:])
    xa_row = pd.Series(0.0005, index=xa.columns)
    try:
        engine.update_with_new_day(next_date, eq_row, xa_row)
        ok = True
    except Exception as e:
        print(f"Failed: {e}")
        ok = False
    assert ok


# ============================================================================
# Empty / minimal bootstrap
# ============================================================================


def test_decide_on_empty_engine_returns_safe_defaults():
    """Vor jeglichem Bootstrap muss decide_next() safe defaults zurückgeben."""
    engine = LiveDecisionEngine()
    out = engine.decide_next()
    assert "sa_leverage" in out
    assert "xa_ew_leverage" in out


def test_bootstrap_with_insufficient_data():
    """Bootstrap mit nur 10 Tagen Daten (zu wenig für Mom-12/1=252)."""
    eq, xa = _make_clean_returns(n_days=10)
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    out = engine.decide_next()
    # Muss safe defaults zurückgeben, kein Crash
    assert np.isfinite(out["sa_leverage"])
    # eq_top_weights kann leer sein (insufficient history)


def test_bootstrap_with_empty_panels_safe():
    """Bootstrap mit empty DataFrames."""
    engine = LiveDecisionEngine()
    try:
        engine.bootstrap_from_history(pd.DataFrame(), pd.DataFrame())
        ok = True
    except Exception:
        ok = False
    assert ok


# ============================================================================
# Time series gaps / weekend handling
# ============================================================================


def test_update_with_gap_in_dates():
    """Update mit Datum 1 Woche später (Weekend/Holiday-Gap)."""
    eq, xa = _make_clean_returns()
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    # Jump 7 days ahead
    far_date = eq.index[-1] + pd.Timedelta(days=7)
    eq_row = pd.Series(0.001, index=eq.columns)
    xa_row = pd.Series(0.0005, index=xa.columns)
    engine.update_with_new_day(far_date, eq_row, xa_row)
    assert engine.state.last_date == far_date


# ============================================================================
# Extreme values
# ============================================================================


def test_decide_with_zero_volatility():
    """Wenn realized-vol = 0 (e.g., constant prices) — leverage clamped."""
    eq, xa = _make_clean_returns()
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    # Corrupt history with constant returns (vol = 0)
    engine.state.eq_factor_returns = [0.0] * 100  # zero variance
    out = engine.decide_next()
    # Leverage muss endlich + clamped sein
    assert np.isfinite(out["sa_leverage"])
    assert 0.0 <= out["sa_leverage"] <= 2.0


def test_decide_with_extreme_vol_clamped():
    """Bei extrem hoher Vol bleibt leverage in [0, max_lev]."""
    eq, xa = _make_clean_returns()
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    # Inject extreme-vol returns
    engine.state.eq_factor_returns = list(np.random.default_rng(0).normal(0, 0.5, 100))
    out = engine.decide_next()
    assert out["sa_leverage"] >= 0
    assert out["sa_leverage"] <= 2.0  # max_leverage default


# ============================================================================
# State persistence / corruption
# ============================================================================


def test_load_corrupt_pickle_raises_cleanly(tmp_path):
    bad = tmp_path / "bad_state.pkl"
    bad.write_bytes(b"not_a_pickle")
    engine = LiveDecisionEngine()
    with pytest.raises((pickle.UnpicklingError, EOFError, ValueError, KeyError, Exception)):
        engine.load_state(bad)


def test_save_load_round_trip_preserves_state(tmp_path):
    eq, xa = _make_clean_returns()
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    out_before = engine.decide_next()

    p = tmp_path / "state.pkl"
    engine.save_state(p)

    engine2 = LiveDecisionEngine()
    engine2.load_state(p)
    out_after = engine2.decide_next()

    assert abs(out_before["sa_leverage"] - out_after["sa_leverage"]) < 1e-9


# ============================================================================
# Repeated updates
# ============================================================================


def test_repeated_updates_dont_explode_memory():
    """Test max_history truncation."""
    eq, xa = _make_clean_returns()
    engine = LiveDecisionEngine()
    engine.state.max_history = 50
    engine.bootstrap_from_history(eq, xa)
    base_date = eq.index[-1]
    eq_row = pd.Series(0.001, index=eq.columns)
    xa_row = pd.Series(0.0005, index=xa.columns)
    for i in range(200):
        engine.update_with_new_day(base_date + pd.Timedelta(days=i + 1), eq_row, xa_row)
    assert len(engine.state.eq_log_return_history) <= 50
    assert len(engine.state.xa_log_return_history) <= 50
    assert len(engine.state.eq_factor_returns) <= 50


def test_same_date_update_twice():
    """Re-Update mit gleichem Datum — sollte robust sein."""
    eq, xa = _make_clean_returns()
    engine = LiveDecisionEngine()
    engine.bootstrap_from_history(eq, xa)
    next_date = eq.index[-1] + pd.Timedelta(days=1)
    eq_row = pd.Series(0.001, index=eq.columns)
    xa_row = pd.Series(0.0005, index=xa.columns)
    engine.update_with_new_day(next_date, eq_row, xa_row)
    # Same date again
    engine.update_with_new_day(next_date, eq_row, xa_row)
    # No crash, state has duplicate but ok
    assert engine.state.last_date == next_date
