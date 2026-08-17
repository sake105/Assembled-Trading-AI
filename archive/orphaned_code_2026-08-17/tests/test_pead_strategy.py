"""Tests for PEAD (Post-Earnings Announcement Drift) strategy.

Covers signal generation, PIT safety, quintile ranking, confidence
computation, and edge cases.  Audit C2-060.
"""

from __future__ import annotations

import pandas as pd

from assembled_core.strategies.pead_strategy import PEADConfig, generate_pead_signals


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_earnings(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal earnings DataFrame from a list of dicts."""
    df = pd.DataFrame(rows)
    df["earnings_date"] = pd.to_datetime(df["earnings_date"])
    return df


def _make_prices() -> pd.DataFrame:
    """Minimal prices frame (not used in signal logic, just passed through)."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=10),
            "symbol": ["AAPL"] * 10,
            "close": [150.0] * 10,
        }
    )


def _base_earnings_multi() -> pd.DataFrame:
    """10 symbols with varying EPS so SUE ranking is meaningful."""
    as_of = pd.Timestamp("2024-03-01")
    base_date = pd.Timestamp("2024-02-15")

    rows = []
    for i, sym in enumerate(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]):
        # eps_actual grows with index → A has lowest, J has highest
        rows.append(
            {
                "symbol": sym,
                "earnings_date": base_date,
                "eps_actual": float(i) + 0.1,
                "eps_estimate": 0.5,  # fixed estimate → higher i = bigger surprise
            }
        )
    return _make_earnings(rows)


AS_OF = pd.Timestamp("2024-03-01")
PRICES = _make_prices()
DEFAULT_CFG = PEADConfig(drift_window=60, min_sue_abs=0.0)


# ---------------------------------------------------------------------------
# Test 1: Top SUE → signal = 1
# ---------------------------------------------------------------------------


def test_top_sue_gets_long_signal() -> None:
    """Symbols with highest SUE receive signal=1."""
    df = _base_earnings_multi()
    cfg = PEADConfig(
        drift_window=60, min_sue_abs=0.0, top_quintile_pct=0.2, long_only=True
    )
    result = generate_pead_signals(df, PRICES, AS_OF, cfg)

    assert not result.empty
    top = result[result["signal"] == 1]
    assert len(top) >= 1
    # The symbol with highest SUE score should be long
    max_sue_sym = result.sort_values("sue_score", ascending=False).iloc[0]["symbol"]
    assert max_sue_sym in top["symbol"].values


# ---------------------------------------------------------------------------
# Test 2: Bottom SUE → signal = -1 when long_only=False
# ---------------------------------------------------------------------------


def test_bottom_sue_gets_short_signal_when_not_long_only() -> None:
    """Bottom quintile gets signal=-1 when long_only=False."""
    df = _base_earnings_multi()
    cfg = PEADConfig(
        drift_window=60, min_sue_abs=0.0, bottom_quintile_pct=0.2, long_only=False
    )
    result = generate_pead_signals(df, PRICES, AS_OF, cfg)

    assert not result.empty
    short = result[result["signal"] == -1]
    assert len(short) >= 1
    # The symbol with lowest SUE score should be short
    min_sue_sym = result.sort_values("sue_score").iloc[0]["symbol"]
    assert min_sue_sym in short["symbol"].values


# ---------------------------------------------------------------------------
# Test 3: long_only=True → no signal = -1
# ---------------------------------------------------------------------------


def test_long_only_no_short_signals() -> None:
    """When long_only=True there are no signal=-1 rows."""
    df = _base_earnings_multi()
    cfg = PEADConfig(drift_window=60, min_sue_abs=0.0, long_only=True)
    result = generate_pead_signals(df, PRICES, AS_OF, cfg)

    assert -1 not in result["signal"].values


# ---------------------------------------------------------------------------
# Test 4: PIT safety — future earnings excluded
# ---------------------------------------------------------------------------


def test_pit_future_earnings_excluded() -> None:
    """Earnings dated after as_of are not included in signals."""
    future_date = AS_OF + pd.Timedelta(days=5)
    rows = [
        {
            "symbol": "FUTURE",
            "earnings_date": future_date,
            "eps_actual": 5.0,
            "eps_estimate": 1.0,
        }
    ]
    df = _make_earnings(rows)
    result = generate_pead_signals(df, PRICES, AS_OF, DEFAULT_CFG)

    assert result.empty or "FUTURE" not in result["symbol"].values


# ---------------------------------------------------------------------------
# Test 5: Only recent earnings within drift window included
# ---------------------------------------------------------------------------


def test_only_earnings_within_drift_window_included() -> None:
    """Earnings older than drift_window days are excluded."""
    drift_window = 30
    old_date = AS_OF - pd.Timedelta(days=drift_window + 5)
    rows = [
        {
            "symbol": "OLD",
            "earnings_date": old_date,
            "eps_actual": 3.0,
            "eps_estimate": 1.0,
        }
    ]
    df = _make_earnings(rows)
    cfg = PEADConfig(drift_window=drift_window, min_sue_abs=0.0)
    result = generate_pead_signals(df, PRICES, AS_OF, cfg)

    assert result.empty or "OLD" not in result["symbol"].values


# ---------------------------------------------------------------------------
# Test 6: Minimum |SUE| filter
# ---------------------------------------------------------------------------


def test_min_sue_abs_filter() -> None:
    """Rows with |SUE| below min_sue_abs are excluded."""
    rows = [
        {
            "symbol": "LOW_SUE",
            "earnings_date": pd.Timestamp("2024-02-10"),
            "eps_actual": 1.01,
            "eps_estimate": 1.00,  # tiny surprise
        }
    ]
    df = _make_earnings(rows)
    cfg = PEADConfig(drift_window=60, min_sue_abs=5.0)  # very high threshold
    result = generate_pead_signals(df, PRICES, AS_OF, cfg)

    assert result.empty or "LOW_SUE" not in result["symbol"].values


# ---------------------------------------------------------------------------
# Test 7: Output columns are correct
# ---------------------------------------------------------------------------


def test_output_columns_correct() -> None:
    """Result DataFrame has exactly the expected columns."""
    expected_cols = {
        "symbol",
        "signal",
        "sue_score",
        "earnings_date",
        "expected_exit_date",
        "confidence",
    }
    df = _base_earnings_multi()
    result = generate_pead_signals(df, PRICES, AS_OF, DEFAULT_CFG)

    assert not result.empty
    assert set(result.columns) == expected_cols


# ---------------------------------------------------------------------------
# Test 8: Confidence in [0, 1]
# ---------------------------------------------------------------------------


def test_confidence_in_unit_interval() -> None:
    """Confidence values are all in [0, 1]."""
    df = _base_earnings_multi()
    result = generate_pead_signals(df, PRICES, AS_OF, DEFAULT_CFG)

    assert not result.empty
    assert result["confidence"].between(0.0, 1.0).all()


# ---------------------------------------------------------------------------
# Test 9: Expected exit date = earnings_date + drift_window
# ---------------------------------------------------------------------------


def test_expected_exit_date_correct() -> None:
    """expected_exit_date equals earnings_date + drift_window days."""
    df = _base_earnings_multi()
    cfg = PEADConfig(drift_window=60, min_sue_abs=0.0)
    result = generate_pead_signals(df, PRICES, AS_OF, cfg)

    assert not result.empty
    for _, row in result.iterrows():
        expected = row["earnings_date"] + pd.Timedelta(days=60)
        assert row["expected_exit_date"] == expected


# ---------------------------------------------------------------------------
# Test 10: Empty earnings → empty result, no crash
# ---------------------------------------------------------------------------


def test_empty_earnings_returns_empty() -> None:
    """Empty input DataFrame returns empty result without raising."""
    df = pd.DataFrame(columns=["symbol", "earnings_date", "eps_actual"])
    result = generate_pead_signals(df, PRICES, AS_OF, DEFAULT_CFG)

    assert result.empty
    assert set(result.columns) == {
        "symbol",
        "signal",
        "sue_score",
        "earnings_date",
        "expected_exit_date",
        "confidence",
    }


# ---------------------------------------------------------------------------
# Test 11: Single symbol single earnings
# ---------------------------------------------------------------------------


def test_single_symbol_single_event() -> None:
    """Single symbol with single earnings event works without crash."""
    rows = [
        {
            "symbol": "SOLO",
            "earnings_date": pd.Timestamp("2024-02-20"),
            "eps_actual": 2.0,
            "eps_estimate": 1.0,
        }
    ]
    df = _make_earnings(rows)
    cfg = PEADConfig(drift_window=60, min_sue_abs=0.0)
    result = generate_pead_signals(df, PRICES, AS_OF, cfg)

    # eps_actual=2.0 > eps_estimate=1.0 → positive SUE → signal=1 via thin-universe path
    assert not result.empty, "Single positive-SUE symbol must produce a signal"
    assert result.iloc[0]["symbol"] == "SOLO"
    assert result.iloc[0]["signal"] == 1


# ---------------------------------------------------------------------------
# Test 12: Multiple symbols, top/bottom quintile signals only
# ---------------------------------------------------------------------------


def test_multiple_symbols_only_nonzero_signals_returned() -> None:
    """Only non-zero signal rows are returned; no duplicate symbols."""
    df = _base_earnings_multi()
    cfg = PEADConfig(drift_window=60, min_sue_abs=0.0)
    result = generate_pead_signals(df, PRICES, AS_OF, cfg)

    assert not result.empty
    # Only non-zero signals are returned
    assert (result["signal"] != 0).all()
    # Signals are valid
    assert result["signal"].isin([-1, 1]).all()
    # No duplicate symbols
    assert result["symbol"].nunique() == len(result)


# ---------------------------------------------------------------------------
# Test 13: Default config (no config argument)
# ---------------------------------------------------------------------------


def test_default_config_no_crash() -> None:
    """generate_pead_signals works when config=None (uses defaults)."""
    df = _base_earnings_multi()
    result = generate_pead_signals(df, PRICES, AS_OF, config=None)
    # Just verify it doesn't crash and returns a DataFrame
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Test 14: Missing required column raises graceful empty return
# ---------------------------------------------------------------------------


def test_missing_required_column_returns_empty() -> None:
    """Missing eps_actual column returns empty DataFrame gracefully."""
    rows = [{"symbol": "X", "earnings_date": pd.Timestamp("2024-02-01")}]
    df = pd.DataFrame(rows)
    result = generate_pead_signals(df, PRICES, AS_OF, DEFAULT_CFG)
    assert result.empty


# ---------------------------------------------------------------------------
# Test 15: Regression — full PIT-safe history used for SUE computation
# BLOCKER fix: windowed slice must NOT be used for sigma estimation.
# ---------------------------------------------------------------------------


def test_sue_computed_on_full_history_not_windowed_slice() -> None:
    """PEAD uses full PIT-safe history for SUE, not just drift-window slice.

    A symbol with 8 quarters of history but only one event inside the
    60-day drift window should still produce a valid SUE and signal.
    """
    as_of = pd.Timestamp("2024-03-01")
    # 8 quarters of history — all before as_of, only last one within 60d window
    quarters = [
        as_of - pd.Timedelta(days=d) for d in [550, 460, 370, 280, 200, 155, 100, 30]
    ]
    eps_actuals = [1.0, 1.1, 1.2, 1.1, 1.3, 1.2, 1.3, 2.5]  # big beat on last

    rows = [
        {
            "symbol": "FULL_HIST",
            "earnings_date": d,
            "eps_actual": e,
        }
        for d, e in zip(quarters, eps_actuals)
    ]
    df = _make_earnings(rows)
    cfg = PEADConfig(drift_window=60, min_sue_abs=0.0)
    result = generate_pead_signals(df, PRICES, as_of, cfg)

    # Should have a signal — the big beat on the latest event should produce SUE
    assert not result.empty, (
        "Expected a signal for symbol with full history but only one event "
        "in drift window. Indicates windowed-slice BLOCKER is not fixed."
    )
    assert "FULL_HIST" in result["symbol"].values
    # The last event (big beat) should produce a positive SUE → signal=1
    sym_row = result[result["symbol"] == "FULL_HIST"].iloc[0]
    assert sym_row["sue_score"] > 0
    assert sym_row["signal"] == 1


# ---------------------------------------------------------------------------
# Test 16: Regression — thin-universe negative SUE must NOT produce signal=1
# BLOCKER fix: single-symbol with negative SUE should get signal=-1 or be
# absent (long_only), never signal=1.
# ---------------------------------------------------------------------------


def test_thin_universe_negative_sue_does_not_produce_long() -> None:
    """Single symbol with negative SUE must NOT get signal=1.

    With n_symbols < quintile threshold, absolute SUE direction is used.
    Negative SUE → short candidate (signal=-1 when long_only=False, absent
    when long_only=True).
    """
    rows = [
        {
            "symbol": "MISS",
            "earnings_date": pd.Timestamp("2024-02-20"),
            "eps_actual": 0.5,
            "eps_estimate": 1.4,  # big miss → negative surprise
        }
    ]
    df = _make_earnings(rows)

    # long_only=True: negative-SUE single symbol should produce no signal
    cfg_long_only = PEADConfig(drift_window=60, min_sue_abs=0.0, long_only=True)
    result_long = generate_pead_signals(df, PRICES, AS_OF, cfg_long_only)
    if not result_long.empty:
        assert result_long.iloc[0]["signal"] != 1, (
            "Negative-SUE single symbol received signal=1 (thin-universe BLOCKER not fixed)"
        )

    # long_only=False: negative-SUE single symbol should get signal=-1
    cfg_short = PEADConfig(drift_window=60, min_sue_abs=0.0, long_only=False)
    result_short = generate_pead_signals(df, PRICES, AS_OF, cfg_short)
    if not result_short.empty:
        assert result_short.iloc[0]["signal"] == -1, (
            "Negative-SUE single symbol received signal=1 (thin-universe BLOCKER not fixed)"
        )
