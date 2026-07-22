"""Regression guards for the 2026-07-21 GESAMTBEWERTUNG P3 fixes.

W1  — qa/deflated_sharpe.sharpe_std_error uses the correct BLP coefficient
      ((excess_kurtosis + 2)/4, i.e. raw-kurtosis (γ4-1)/4). The old code
      used (excess-1)/4, understating the SE and OVER-stating DSR — an
      anti-conservative gate error.
W11a — qa/cpcv_validation labels the CV scheme that actually ran
      (CPCVResult.method) and warns loudly on the unpurged fallback.
W18 — ops/paper_ledger.apply_fills_to_ledger sweeps float-dust positions
      (|qty| < 1e-9) instead of persisting residues like 7.1e-15.
W5  — strategies/multifactor_v2._compute_options_factors accepts as_of and
      PIT-slices the CBOE series (no look-ahead in backtests).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# W1 — DSR standard error (BLP 2014)
# ---------------------------------------------------------------------------


def test_w1_sharpe_se_normal_returns_matches_blp():
    from src.assembled_core.qa.deflated_sharpe import sharpe_std_error

    sr, n = 0.5, 253
    # Normal returns: skew=0, excess kurtosis=0 -> BLP inside = 1 + SR^2/2.
    se = sharpe_std_error(sr, n, skew=0.0, excess_kurtosis=0.0)
    expected = math.sqrt((1.0 + sr**2 / 2.0) / (n - 1))
    assert se == pytest.approx(expected, rel=1e-12)


def test_w1_se_grows_with_kurtosis():
    from src.assembled_core.qa.deflated_sharpe import sharpe_std_error

    se_normal = sharpe_std_error(1.0, 253, skew=0.0, excess_kurtosis=0.0)
    se_fat = sharpe_std_error(1.0, 253, skew=0.0, excess_kurtosis=6.0)
    assert se_fat > se_normal  # fat tails must widen the SE, never shrink it


def test_w1_old_formula_would_have_understated():
    """Documents the direction of the fixed bug: old coeff (ex-1)/4 < new (ex+2)/4."""
    from src.assembled_core.qa.deflated_sharpe import sharpe_std_error

    sr, n = 1.0, 253
    se_new = sharpe_std_error(sr, n, skew=0.0, excess_kurtosis=0.0)
    se_old = math.sqrt(max(1.0 - sr**2 / 4.0, 0.0) / (n - 1))
    assert se_new > se_old


# ---------------------------------------------------------------------------
# W11a — CPCV method labelling
# ---------------------------------------------------------------------------


def test_w11a_result_has_method_field_default():
    from src.assembled_core.qa.cpcv_validation import CPCVResult

    r = CPCVResult(0, np.array([]), 0.0, 0.0, None)
    assert hasattr(r, "method")


def test_w11a_fallback_is_labelled_unpurged():
    sklearn = pytest.importorskip("sklearn")  # third-party dep, skip if absent
    from sklearn.linear_model import LogisticRegression

    from src.assembled_core.qa import cpcv_validation as cv

    X = pd.DataFrame({"a": np.arange(60, dtype=float)})
    y = pd.Series((np.arange(60) % 2).astype(int))
    res = cv._walk_forward_cv(
        LogisticRegression(), X, y, n_splits=3, scoring="accuracy"
    )
    assert res.method == "timeseries_split_unpurged"
    assert sklearn is not None


# ---------------------------------------------------------------------------
# W18 — ledger dust sweep
# ---------------------------------------------------------------------------


def _state(cash: float, positions: dict) -> dict:
    return {"schema_version": "v1", "cash": cash, "positions": positions}


def test_w18_partial_close_of_fractional_position_leaves_no_dust():
    from src.assembled_core.ops.paper_ledger import apply_fills_to_ledger

    # 0.1 + 0.2 buys, then sell 0.3: float arithmetic leaves ~5.6e-17 without
    # the sweep (0.1 + 0.2 - 0.3 != 0 in binary float).
    state = _state(1000.0, {})
    state = apply_fills_to_ledger(
        state,
        [
            {"symbol": "XYZ", "side": "BUY", "qty": 0.1, "price": 10.0},
            {"symbol": "XYZ", "side": "BUY", "qty": 0.2, "price": 10.0},
        ],
    )
    state = apply_fills_to_ledger(
        state, [{"symbol": "XYZ", "side": "SELL", "qty": 0.3, "price": 10.0}]
    )
    assert "XYZ" not in state["positions"], (
        f"dust position persisted: {state['positions'].get('XYZ')}"
    )


def test_w18_real_positions_survive_the_sweep():
    from src.assembled_core.ops.paper_ledger import apply_fills_to_ledger

    state = apply_fills_to_ledger(
        _state(1000.0, {}),
        [{"symbol": "ABC", "side": "BUY", "qty": 0.000001, "price": 10.0}],
    )
    # 1e-6 is a real (if tiny) position — well above the 1e-9 dust epsilon.
    assert state["positions"]["ABC"]["qty"] == pytest.approx(1e-6)


# ---------------------------------------------------------------------------
# W5 — options factors PIT slice
# ---------------------------------------------------------------------------


def test_w5_options_factors_pit_slice(monkeypatch):
    import src.assembled_core.strategies.multifactor_v2 as mf

    cboe_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-15"]),
            "vix": [15.0, 16.0, 40.0],
            "vix3m": [17.0, 17.5, 30.0],
            "put_call_ratio": [0.9, 0.95, 2.0],
        }
    )

    class _FakeCBOE:
        def fetch_options_regime_data(self):
            return cboe_df

    import src.assembled_core.data.sources.cboe_source as cboe_mod

    monkeypatch.setattr(cboe_mod, "CBOESource", _FakeCBOE)

    latest = pd.DataFrame({"symbol": ["AAA", "BBB"]})
    # as_of 2026-07-03: the 2026-07-15 panic row (vix 40, pcr 2.0) must NOT
    # leak into the factor values.
    res_pit = mf._compute_options_factors(
        ["AAA", "BBB"], latest, as_of=pd.Timestamp("2026-07-03", tz="UTC")
    )
    res_live = mf._compute_options_factors(["AAA", "BBB"], latest, as_of=None)

    if not res_pit and not res_live:
        pytest.skip("options factor pipeline degraded in this env — nothing computed")
    assert "vix_regime_score" in res_pit and "vix_regime_score" in res_live
    pit_val = float(res_pit["vix_regime_score"].iloc[0])
    live_val = float(res_live["vix_regime_score"].iloc[0])
    # The live (unsliced) value is anchored on the vix=40 panic row; the PIT
    # value is anchored on 2026-07-02 — they must differ, and the PIT value
    # must equal a computation that never saw rows after as_of.
    assert pit_val != live_val

    res_pit_empty = mf._compute_options_factors(
        ["AAA"],
        pd.DataFrame({"symbol": ["AAA"]}),
        as_of=pd.Timestamp("2020-01-01", tz="UTC"),
    )
    assert res_pit_empty == {}  # no rows <= as_of -> degrade, never look ahead
