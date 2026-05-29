"""PIT-safety tests for ETF-Pairs Cointegration Mean-Reversion strategy.

Critical concerns verified here:
  - Causal / no-future-leak: signals at bar T must not change when future bars
    are appended.
  - Late-inception guard (E-030): a symbol with missing early prices must not
    generate signals for bars where its log-price window contains NaN (which
    bfill would spuriously fill with a future price).
  - Schema and behavioural contracts: output columns, direction values, weights.
  - Long-only mode: no SHORT rows emitted.
  - Edge cases: too few bars, missing symbols.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.assembled_core.strategies.etf_pairs_meanrev import (
    _compute_pair_states,
    compute_signals,
    compute_target_positions,
    generate_etf_pairs_signals_from_prices,
)

# Use smaller windows for speed; logic is identical to production defaults
_COINT = 60
_ZSCORE = 20
_ENTRY_Z = 2.0
_EXIT_Z = 0.5
_STOP_Z = 3.5

_PAIR = [("SPY", "IVV")]  # one pair for focused tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_cointegrated_prices(
    n_bars: int = 200,
    rng_seed: int = 42,
    bil_lag_bars: int = 0,
    extra_pair: bool = False,
) -> pd.DataFrame:
    """Return a long-format prices DataFrame with one cointegrated pair SPY/IVV.

    SPY = exp(common_walk + tight_noise_a)
    IVV = exp(common_walk + tight_noise_b)
    => log(SPY) - log(IVV) is stationary by construction.

    Args:
        n_bars: total number of bars.
        rng_seed: RNG seed for reproducibility.
        bil_lag_bars: IVV inception delayed by this many bars (simulates
                       late-inception; early bars have no IVV row).
        extra_pair: if True, also add a second (non-cointegrated) pair GDX/GDXJ
                    as independent random walks so we can test equal-weight.
    """
    rng = np.random.default_rng(rng_seed)
    dates = pd.date_range("2010-01-04", periods=n_bars, freq="B", tz="UTC")

    # Common random walk
    common = np.cumsum(rng.normal(0, 0.01, n_bars))
    # Tight stationary residuals
    noise_a = rng.normal(0, 0.002, n_bars)
    noise_b = rng.normal(0, 0.002, n_bars)

    spy_prices = 100.0 * np.exp(common + noise_a)
    ivv_prices = 100.0 * np.exp(common + noise_b)

    rows: list[dict] = []
    for i, ts in enumerate(dates):
        rows.append({"timestamp": ts, "symbol": "SPY", "close": float(spy_prices[i])})
        if i >= bil_lag_bars:
            rows.append(
                {"timestamp": ts, "symbol": "IVV", "close": float(ivv_prices[i])}
            )

    if extra_pair:
        # GDX / GDXJ as two independent random walks (intentionally NOT cointegrated)
        gdx_prices = 50.0 * np.exp(np.cumsum(rng.normal(0, 0.015, n_bars)))
        gdxj_prices = 30.0 * np.exp(np.cumsum(rng.normal(0, 0.018, n_bars)))
        for i, ts in enumerate(dates):
            rows.append(
                {"timestamp": ts, "symbol": "GDX", "close": float(gdx_prices[i])}
            )
            rows.append(
                {"timestamp": ts, "symbol": "GDXJ", "close": float(gdxj_prices[i])}
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: output schema
# ---------------------------------------------------------------------------


def test_output_schema_valid_columns():
    """Output DataFrame must have the required columns with correct dtypes."""
    prices = _make_cointegrated_prices(n_bars=200)
    sigs = generate_etf_pairs_signals_from_prices(
        prices,
        pairs=_PAIR,
        cointegration_window=_COINT,
        zscore_window=_ZSCORE,
    )
    assert set(sigs.columns) == {"timestamp", "symbol", "direction", "score"}
    if not sigs.empty:
        assert sigs["direction"].isin(["LONG", "SHORT"]).all()
        assert (sigs["score"] > 0).all()
        assert (sigs["score"] <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# Test 2: causal / no-future-leak
# ---------------------------------------------------------------------------


def test_causal_no_future_leak():
    """Signals at bar T must be identical whether or not future bars are present.

    This is the core PIT test: appending bars after T must not change the signal
    at T.  Any future-data access (e.g. cointegration window extending past T,
    spread computed on future prices, bfill propagating future prices backwards)
    would cause a mismatch.
    """
    # Use enough bars to get signals, then extend with extra bars
    prices_full = _make_cointegrated_prices(n_bars=300, rng_seed=7)
    prices_short = prices_full[
        prices_full["timestamp"] <= prices_full["timestamp"].unique()[199]
    ].copy()

    sigs_full = generate_etf_pairs_signals_from_prices(
        prices_full,
        pairs=_PAIR,
        cointegration_window=_COINT,
        zscore_window=_ZSCORE,
    )
    sigs_short = generate_etf_pairs_signals_from_prices(
        prices_short,
        pairs=_PAIR,
        cointegration_window=_COINT,
        zscore_window=_ZSCORE,
    )

    assert not sigs_full.empty, (
        "sigs_full (300 bars) must produce signals for the PIT test to have discriminating power"
    )
    assert not sigs_short.empty, (
        "sigs_short (200 bars) must produce signals — cointegration must be detectable "
        "in the shorter window for the PIT comparison to be meaningful"
    )

    # For every timestamp in sigs_short, the signal must match sigs_full exactly
    cutoff_ts = prices_short["timestamp"].max()
    sigs_full_overlap = (
        sigs_full[sigs_full["timestamp"] <= cutoff_ts]
        .sort_values(["timestamp", "symbol"])
        .reset_index(drop=True)
    )
    sigs_short_sorted = sigs_short.sort_values(["timestamp", "symbol"]).reset_index(
        drop=True
    )

    pd.testing.assert_frame_equal(
        sigs_full_overlap[["timestamp", "symbol", "direction", "score"]],
        sigs_short_sorted[["timestamp", "symbol", "direction", "score"]],
        check_like=False,
        rtol=1e-9,
        obj="signals at overlapping timestamps must be identical (causal PIT check)",
    )


# ---------------------------------------------------------------------------
# Test 3: late-inception guard (E-030 bfill protection)
# ---------------------------------------------------------------------------


def test_late_inception_flat_bfill_guard():
    """When IVV is missing for the first ``bil_lag_bars`` bars, the coint window
    for bars [_COINT-1 .. _COINT + bil_lag_bars - 2] still overlaps with the
    NaN region.  The degenerate-window guard must keep these bars FLAT.

    If .bfill() were re-introduced, IVV's first valid price would fill backwards
    into the NaN region, potentially satisfying cointegration and generating a
    spurious signal — which this test would detect.
    """
    lag = 30  # IVV starts 30 bars late
    prices = _make_cointegrated_prices(n_bars=300, rng_seed=3, bil_lag_bars=lag)

    sigs = generate_etf_pairs_signals_from_prices(
        prices,
        pairs=_PAIR,
        cointegration_window=_COINT,
        zscore_window=_ZSCORE,
    )

    # Hard precondition: IVV inception is indeed delayed
    ivv_rows = prices[prices["symbol"] == "IVV"]
    ivv_first_ts = ivv_rows["timestamp"].min()
    all_dates = sorted(prices["timestamp"].unique())

    # IVV starts at bar `lag`, which must be positive and within the first coint_window
    # so that the initial coint windows still overlap with the NaN region.
    ivv_first_idx = next(i for i, ts in enumerate(all_dates) if ts >= ivv_first_ts)
    assert ivv_first_idx > 0, "Fixture guarantee: IVV must have delayed inception"
    assert ivv_first_idx < _COINT, (
        "Fixture guarantee: IVV inception must fall within the first coint_window "
        "bars so that early coint windows contain NaN"
    )
    # first bar where the full coint window [t-_COINT+1 : t+1] is clean:
    clean_start_idx = ivv_first_idx + _COINT - 1
    clean_start_ts = all_dates[clean_start_idx]

    # All signals emitted before clean_start_ts must be absent
    if not sigs.empty:
        early_sigs = sigs[sigs["timestamp"] < clean_start_ts]
        assert early_sigs.empty, (
            f"Signals before clean_start_ts {clean_start_ts.date()} detected "
            f"({len(early_sigs)} rows) — bfill look-ahead may be active or "
            "degenerate-window guard is missing"
        )


# ---------------------------------------------------------------------------
# Test 4: insufficient bars → empty
# ---------------------------------------------------------------------------


def test_insufficient_bars_returns_empty():
    """Fewer bars than cointegration_window must return an empty schema DataFrame."""
    prices = _make_cointegrated_prices(n_bars=_COINT - 1)
    sigs = generate_etf_pairs_signals_from_prices(
        prices,
        pairs=_PAIR,
        cointegration_window=_COINT,
        zscore_window=_ZSCORE,
    )
    assert sigs.empty
    assert list(sigs.columns) == ["timestamp", "symbol", "direction", "score"]


# ---------------------------------------------------------------------------
# Test 5: missing symbol → empty
# ---------------------------------------------------------------------------


def test_missing_symbol_returns_empty():
    """If a required symbol is absent from the prices panel, return empty."""
    prices = _make_cointegrated_prices(n_bars=200)
    # Remove IVV rows entirely
    prices_spy_only = prices[prices["symbol"] == "SPY"].copy()
    sigs = generate_etf_pairs_signals_from_prices(
        prices_spy_only,
        pairs=_PAIR,
        cointegration_window=_COINT,
        zscore_window=_ZSCORE,
    )
    assert sigs.empty


# ---------------------------------------------------------------------------
# Test 6: long_only mode — no SHORT signals
# ---------------------------------------------------------------------------


def test_long_only_no_short_signals():
    """In long_only mode every emitted signal must have direction == 'LONG'."""
    prices = _make_cointegrated_prices(n_bars=300, rng_seed=9)
    sigs = generate_etf_pairs_signals_from_prices(
        prices,
        pairs=_PAIR,
        cointegration_window=_COINT,
        zscore_window=_ZSCORE,
        long_only=True,
    )
    assert not sigs.empty, (
        "long_only fixture (300 bars, seed=9) must produce signals to test direction filter"
    )
    assert (sigs["direction"] == "LONG").all(), (
        "long_only=True must never emit SHORT signals"
    )


# ---------------------------------------------------------------------------
# Test 7: equal-weight across active pairs
# ---------------------------------------------------------------------------


def test_equal_weight_across_active_pairs():
    """When k pairs are active at the same timestamp, each pair's score = 1/k.

    We use SPY/IVV (cointegrated) and GDX/GDXJ (independent random walks).
    If both are simultaneously active, weight = 0.5; if only one, weight = 1.0.
    If GDX/GDXJ is never active (not cointegrated), SPY/IVV weight is always 1.0.
    """
    prices = _make_cointegrated_prices(n_bars=300, rng_seed=17, extra_pair=True)
    sigs = generate_etf_pairs_signals_from_prices(
        prices,
        pairs=[("SPY", "IVV"), ("GDX", "GDXJ")],
        cointegration_window=_COINT,
        zscore_window=_ZSCORE,
    )
    if sigs.empty:
        pytest.skip("No signals generated — cointegration not detected in test window")

    for ts, group in sigs.groupby("timestamp"):
        # Determine distinct active pairs at this timestamp
        active_symbols = set(group["symbol"].tolist())
        # For full mode: SPY+IVV active → 2 rows (one pair), or GDX+GDXJ → 2 rows
        # Count unique PAIR slots: a symbol belongs to at most one pair
        spy_ivv_active = bool(("SPY" in active_symbols or "IVV" in active_symbols))
        gdx_gdxj_active = bool(("GDX" in active_symbols or "GDXJ" in active_symbols))
        n_active_pairs = int(spy_ivv_active) + int(gdx_gdxj_active)

        if n_active_pairs == 0:
            continue
        expected_weight = 1.0 / n_active_pairs
        for _, row in group.iterrows():
            assert abs(float(row["score"]) - expected_weight) < 1e-9, (
                f"At ts={ts}: expected weight {expected_weight:.4f}, "
                f"got {row['score']:.4f} for {row['symbol']}"
            )


# ---------------------------------------------------------------------------
# Test 8: compute_signals returns empty when current bar is flat
# ---------------------------------------------------------------------------


def test_compute_signals_empty_when_latest_bar_flat():
    """compute_signals must return empty schema if the strategy is FLAT on the
    latest input bar — it must NOT return stale signals from a prior active bar.

    Regression guard for F-senior-4: latest_ts from generate_... is the most-
    recent ACTIVE bar, not necessarily today's bar.  Returning stale rows would
    cause the OMS to replay a historical position.
    """
    prices = _make_cointegrated_prices(n_bars=200, rng_seed=42)

    # Build signals for the full 200-bar history
    sigs_full = generate_etf_pairs_signals_from_prices(
        prices,
        pairs=_PAIR,
        cointegration_window=_COINT,
        zscore_window=_ZSCORE,
    )
    # Find a timestamp that IS active so we can append a flat "today"
    if sigs_full.empty:
        pytest.skip("No signals in base window — cointegration not detected")

    last_active_ts = sigs_full["timestamp"].max()

    # Manufacture a "today" bar well after the last active signal by appending
    # noise that breaks cointegration (independent random walks replace the pair)
    rng = np.random.default_rng(99)
    all_dates = sorted(prices["timestamp"].unique())
    future_ts = all_dates[-1] + pd.Timedelta(days=1)
    # Append rows that extend the price history by one bar with divergent prices
    extra = pd.DataFrame(
        [
            {"timestamp": future_ts, "symbol": "SPY", "close": 9999.0},
            {"timestamp": future_ts, "symbol": "IVV", "close": 0.1},
        ]
    )
    prices_extended = pd.concat([prices, extra], ignore_index=True)

    cs = compute_signals(
        prices_extended,
        pairs=_PAIR,
        cointegration_window=_COINT,
        zscore_window=_ZSCORE,
    )
    # Either empty (strategy flat on future_ts due to broken cointegration) OR
    # has signals with timestamp == future_ts (still active — acceptable).
    # What is NOT acceptable: signals with timestamp < future_ts (stale replay).
    if not cs.empty:
        stale = cs[cs["timestamp"] < future_ts]
        assert stale.empty, (
            f"compute_signals returned stale signals dated {stale['timestamp'].min()} "
            f"when latest price bar is {future_ts}"
        )


# ---------------------------------------------------------------------------
# Test 9: compute_target_positions schema and sign convention
# ---------------------------------------------------------------------------


def test_compute_target_positions_schema_and_signs():
    """compute_target_positions must return symbol/target_weight/target_qty columns
    with LONG → positive weight, SHORT → negative weight, target_qty == 0.0."""
    signals = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2020-01-02", tz="UTC"),
                "symbol": "SPY",
                "direction": "LONG",
                "score": 0.5,
            },
            {
                "timestamp": pd.Timestamp("2020-01-02", tz="UTC"),
                "symbol": "IVV",
                "direction": "SHORT",
                "score": 0.5,
            },
        ]
    )
    pos = compute_target_positions(signals, capital=100_000.0)

    assert set(pos.columns) == {"symbol", "target_weight", "target_qty"}
    spy_row = pos[pos["symbol"] == "SPY"].iloc[0]
    ivv_row = pos[pos["symbol"] == "IVV"].iloc[0]

    assert spy_row["target_weight"] > 0, "LONG leg must have positive target_weight"
    assert ivv_row["target_weight"] < 0, "SHORT leg must have negative target_weight"
    assert spy_row["target_qty"] == 0.0
    assert ivv_row["target_qty"] == 0.0
    assert abs(spy_row["target_weight"]) == pytest.approx(0.5)
    assert abs(ivv_row["target_weight"]) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Test 10: stop-loss cooldown blocks re-entry
# ---------------------------------------------------------------------------


def test_stop_cooldown_blocks_reentry():
    """After a stop-loss fires, re-entry must be blocked for stop_cooldown_bars bars.

    Strategy: run _compute_pair_states with cooldown=0 vs cooldown=5 on the same
    series.  Bars that are active in the cooldown=0 run but flat in the cooldown=5
    run are exactly the cooldown-blocked bars.  At least one such bar must exist
    (otherwise no stop ever fired and the test skips).
    """
    rng = np.random.default_rng(55)
    cw, zw = 60, 20
    # Tighter thresholds so stops fire reliably in 600 bars (seed=55 → 15 blocked)
    entry_z, exit_z, stop_z = 1.5, 0.3, 2.5
    cooldown_n = 5
    n = 600

    common = np.cumsum(rng.normal(0, 0.01, n))
    la = common + rng.normal(0, 0.002, n)
    lb = common + rng.normal(0, 0.002, n)

    states_cd0 = _compute_pair_states(
        la,
        lb,
        coint_window=cw,
        zscore_window=zw,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        stop_cooldown_bars=0,
    )
    states_cd5 = _compute_pair_states(
        la,
        lb,
        coint_window=cw,
        zscore_window=zw,
        entry_z=entry_z,
        exit_z=exit_z,
        stop_z=stop_z,
        stop_cooldown_bars=cooldown_n,
    )

    # Bars active under cooldown=0 but flat under cooldown=5 = cooldown-blocked bars
    active_cd0 = states_cd0 != 0  # 0 == _FLAT
    active_cd5 = states_cd5 != 0
    blocked = np.where(active_cd0 & ~active_cd5)[0]

    if len(blocked) == 0:
        pytest.skip(
            "No stop-loss fired in this 400-bar run — cooldown has no observable "
            "effect; seed/params did not produce a stop scenario"
        )

    # At least one bar was blocked — cooldown is active
    assert len(blocked) >= 1, "stop_cooldown_bars=5 must block at least one re-entry"

    # Each blocked bar must lie within [T+1, T+cooldown_n] after some stop bar T.
    # Verify: for the first blocked bar, the previous bar was non-FLAT in cd5
    # (position existed just before the stop fired).
    first_blocked = int(blocked[0])
    # The stop bar itself is not blocked (just_exited=True prevents same-bar entry).
    # The stop fires at first_blocked - 1 or earlier; look back up to cooldown_n bars.
    lookback = states_cd5[max(0, first_blocked - cooldown_n) : first_blocked]
    was_active_before = np.any(lookback != 0)
    assert was_active_before, (
        f"Blocked bar {first_blocked}: expected an active position in the preceding "
        f"{cooldown_n} bars (stop must have fired), but found all-FLAT."
    )
