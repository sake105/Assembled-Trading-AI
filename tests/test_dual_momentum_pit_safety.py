"""Point-in-time safety tests for dual_momentum strategy.

Core claim: the asset selection at timestamp T must be identical whether
computed from a price panel ending at T or from a panel that also includes
bars after T (even with extreme manipulated values).

The strategy is PIT-safe because:
  - EOM bar identification: bar i is EOM iff bar i+1 is in a new month
    (uses only calendar structure, not future prices).
  - 12M base price: last bar at or before (rebalance_date − 12 months).
  - Forward-fill: holds previous selection until next rebalance.
  None of these steps reads future price data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.assembled_core.strategies.dual_momentum import (
    compute_signals,
    generate_dual_momentum_signals_from_prices,
)

# ---------------------------------------------------------------------------
# Parameters — short lookback so tests run on a compact synthetic panel
# ---------------------------------------------------------------------------
_LOOKBACK = 12  # calendar months
_N_BARS = _LOOKBACK * 21 * 2 + 60  # ~26 months — gives ~14 months of valid signals

_SYMBOLS = ["SPY", "VEU", "BIL", "AGG"]


def _make_4asset_prices(n_bars: int, seed: int = 42) -> pd.DataFrame:
    """Synthetic 4-asset price panel (long format)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-02", periods=n_bars, freq="B", tz="UTC")
    params = {
        "SPY": (300.0, 0.0004, 0.011),
        "VEU": (50.0, 0.0003, 0.010),
        "BIL": (90.0, 0.00008, 0.0002),  # near-flat — T-bill proxy
        "AGG": (110.0, 0.0002, 0.003),
    }
    rows = []
    for sym, (start, drift, vol) in params.items():
        rets = rng.standard_normal(n_bars) * vol + drift
        close = start * np.cumprod(1.0 + rets)
        for d, c in zip(dates, close):
            rows.append({"timestamp": d, "symbol": sym, "close": c})
    return pd.DataFrame(rows)


def _make_4asset_prices_trend_reversal() -> pd.DataFrame:
    """Deterministic fixture guaranteeing at least one AGG-selection bar.

    Phase A (14 months): SPY and VEU rise gently; BIL drifts near-flat.
    Phase B (6 months):  SPY and VEU are reset to 70 % of their Phase-A
                         peak — a crash that makes their 12-month return
                         negative relative to BIL's tiny positive drift.
    """
    n_a = 14 * 21  # 294 bars rising
    n_b = 6 * 21  # 126 bars flat after crash
    n_bars = n_a + n_b
    dates = pd.date_range("2018-01-02", periods=n_bars, freq="B", tz="UTC")
    t_a = np.arange(n_a)
    t_b = np.arange(n_b)
    # Phase A: slow up-drift
    spy_a = 300.0 * (1.001**t_a)
    veu_a = 50.0 * (1.0009**t_a)
    bil_a = 90.0 * (1.0001**t_a)
    agg_a = 110.0 * (1.0003**t_a)
    # Phase B: crash + flat for SPY/VEU; BIL/AGG continue drifting
    crash = 0.70
    spy_b = np.full(n_b, spy_a[-1] * crash)
    veu_b = np.full(n_b, veu_a[-1] * crash)
    bil_b = bil_a[-1] * (1.0001**t_b)
    agg_b = agg_a[-1] * (1.0003**t_b)

    spy_c = np.concatenate([spy_a, spy_b])
    veu_c = np.concatenate([veu_a, veu_b])
    bil_c = np.concatenate([bil_a, bil_b])
    agg_c = np.concatenate([agg_a, agg_b])

    rows = []
    for i, d in enumerate(dates):
        rows.extend(
            [
                {"timestamp": d, "symbol": "SPY", "close": spy_c[i]},
                {"timestamp": d, "symbol": "VEU", "close": veu_c[i]},
                {"timestamp": d, "symbol": "BIL", "close": bil_c[i]},
                {"timestamp": d, "symbol": "AGG", "close": agg_c[i]},
            ]
        )
    return pd.DataFrame(rows)


def _pick_asof(sigs: pd.DataFrame, idx: int = -2) -> pd.Timestamp:
    """Pick a signal timestamp at ``idx`` ensuring future bars are available."""
    ts = sorted(sigs["timestamp"].unique())
    assert len(ts) >= 5, f"Need ≥5 signal bars; got {len(ts)}"
    return ts[idx]


# ---------------------------------------------------------------------------
# Test 1 — core PIT: future bars ×5 must not change selection at as_of
# ---------------------------------------------------------------------------


def test_pit_no_lookahead_future_multiplied():
    """Asset selection at as_of is identical whether future bars are ×5 or unchanged."""
    prices = _make_4asset_prices(_N_BARS)

    baseline = generate_dual_momentum_signals_from_prices(
        prices, lookback_months=_LOOKBACK
    )
    assert not baseline.empty, "No baseline signals generated"
    as_of_ts = _pick_asof(baseline, len(sorted(baseline["timestamp"].unique())) // 2)

    baseline_at = baseline[baseline["timestamp"] == as_of_ts]
    assert len(baseline_at) == 1, "Expected exactly one signal row at as_of"
    baseline_sym = baseline_at.iloc[0]["symbol"]

    # Manipulate future bars
    prices_manip = prices.copy()
    future_mask = prices_manip["timestamp"] > as_of_ts
    assert future_mask.sum() > 0, "No future bars to manipulate"
    prices_manip.loc[future_mask, "close"] *= 5.0

    modified = generate_dual_momentum_signals_from_prices(
        prices_manip, lookback_months=_LOOKBACK
    )
    modified_at = modified[modified["timestamp"] == as_of_ts]
    assert len(modified_at) == 1
    modified_sym = modified_at.iloc[0]["symbol"]

    assert baseline_sym == modified_sym, (
        f"Symbol changed from {baseline_sym} → {modified_sym} when future bars ×5 "
        "— look-ahead suspected"
    )
    assert baseline_at.iloc[0]["direction"] == modified_at.iloc[0]["direction"]


# ---------------------------------------------------------------------------
# Test 2 — PIT: future bars near-zero
# ---------------------------------------------------------------------------


def test_pit_no_lookahead_future_zeroed():
    """Asset selection at as_of unchanged when future bars are set to near-zero."""
    prices = _make_4asset_prices(_N_BARS, seed=77)

    baseline = generate_dual_momentum_signals_from_prices(
        prices, lookback_months=_LOOKBACK
    )
    assert not baseline.empty
    as_of_ts = _pick_asof(baseline, len(sorted(baseline["timestamp"].unique())) // 3)
    baseline_sym = baseline[baseline["timestamp"] == as_of_ts].iloc[0]["symbol"]

    prices_zero = prices.copy()
    prices_zero.loc[prices_zero["timestamp"] > as_of_ts, "close"] = 0.01

    modified = generate_dual_momentum_signals_from_prices(
        prices_zero, lookback_months=_LOOKBACK
    )
    modified_sym = modified[modified["timestamp"] == as_of_ts].iloc[0]["symbol"]

    assert baseline_sym == modified_sym, (
        f"Symbol changed {baseline_sym} → {modified_sym} under zero-future manipulation"
    )


# ---------------------------------------------------------------------------
# Test 3 — classical PIT: signal from prices[:T] == signal at T from prices[:T+N]
# ---------------------------------------------------------------------------


def test_pit_slice_equals_full_panel_at_asof():
    """Selection at bar T from a T-bar slice equals selection at T from a longer panel.

    If the strategy reads any future price (e.g. non-causal EOM detection or
    look-ahead in the 12M return), this test fails.
    """
    prices = _make_4asset_prices(_N_BARS)

    sig_full = generate_dual_momentum_signals_from_prices(
        prices, lookback_months=_LOOKBACK
    )
    assert not sig_full.empty
    as_of_ts = _pick_asof(sig_full, len(sorted(sig_full["timestamp"].unique())) // 2)

    # Slice prices to as_of
    prices_slice = prices[prices["timestamp"] <= as_of_ts].copy()

    sig_slice = generate_dual_momentum_signals_from_prices(
        prices_slice, lookback_months=_LOOKBACK
    )
    assert not sig_slice.empty, "No signals from sliced panel — slice may be too short"

    slice_at = sig_slice[sig_slice["timestamp"] == as_of_ts]
    full_at = sig_full[sig_full["timestamp"] == as_of_ts]

    assert not slice_at.empty, f"No signal in slice at {as_of_ts}"
    assert not full_at.empty, f"No signal in full panel at {as_of_ts}"

    assert slice_at.iloc[0]["symbol"] == full_at.iloc[0]["symbol"], (
        f"Symbol from slice={slice_at.iloc[0]['symbol']} "
        f"vs full={full_at.iloc[0]['symbol']} — rolling is not causal"
    )


# ---------------------------------------------------------------------------
# Test 4 — determinism: two calls with identical input must yield identical output
# ---------------------------------------------------------------------------


def test_compute_signals_deterministic():
    """compute_signals must return identical output on two calls with the same input."""
    prices = _make_4asset_prices(_N_BARS)

    result_a = compute_signals(prices, lookback_months=_LOOKBACK)
    result_b = compute_signals(prices, lookback_months=_LOOKBACK)

    assert list(result_a.columns) == list(result_b.columns)
    pd.testing.assert_frame_equal(
        result_a.reset_index(drop=True),
        result_b.reset_index(drop=True),
        check_exact=True,
    )


# ---------------------------------------------------------------------------
# Test 5 — output schema and value constraints
# ---------------------------------------------------------------------------


def test_output_schema_and_value_constraints():
    """direction is always LONG, score always 1.0, symbol is in the known universe."""
    prices = _make_4asset_prices(_N_BARS)

    sigs = generate_dual_momentum_signals_from_prices(prices, lookback_months=_LOOKBACK)
    assert not sigs.empty, "No signals produced"
    assert set(sigs.columns) >= {"timestamp", "symbol", "direction", "score"}
    assert sigs["direction"].eq("LONG").all(), "direction must always be LONG"
    assert (sigs["score"] == 1.0).all(), "score must always be 1.0"
    assert not sigs["score"].isna().any(), "NaN scores in output"
    assert sigs["symbol"].isin(_SYMBOLS).all(), (
        f"Unexpected symbols: {sigs['symbol'].unique()}"
    )
    # Exactly one holding per timestamp
    counts = sigs.groupby("timestamp")["symbol"].count()
    assert (counts == 1).all(), "More than one signal row per timestamp"


# ---------------------------------------------------------------------------
# Test 6 — absolute momentum filter: AGG selected when equities underperform BIL
# ---------------------------------------------------------------------------


def test_absolute_momentum_selects_agg_when_equities_underperform():
    """When SPY and VEU have negative 12M returns vs BIL, AGG must be selected.

    Uses a deterministic fixture: gentle rise for 14 months, then a 30 % crash
    and flat prices.  After the crash the 12M return for both equity assets is
    negative while BIL continues drifting slightly positive — absolute momentum
    filter must redirect to AGG.
    """
    prices = _make_4asset_prices_trend_reversal()

    sigs = generate_dual_momentum_signals_from_prices(prices, lookback_months=_LOOKBACK)
    assert not sigs.empty, "No signals generated — check fixture construction"

    # Find bars in Phase B (after the crash = last 6 months of fixture)
    all_dates = sorted(prices["timestamp"].unique())
    phase_b_start = all_dates[14 * 21]  # first bar of crash phase
    phase_b_sigs = sigs[sigs["timestamp"] >= phase_b_start]

    assert not phase_b_sigs.empty, "No signals in post-crash phase"

    # After a few months into Phase B, AGG must appear as a selection
    # (exact first AGG bar depends on when the 12M window fully spans the crash)
    agg_sigs = phase_b_sigs[phase_b_sigs["symbol"] == "AGG"]
    assert not agg_sigs.empty, (
        "AGG never selected in Phase B despite equity 12M returns being negative "
        "— absolute momentum filter may be broken"
    )
    # Verify the selected symbol is consistently AGG in late Phase B
    late_phase_b = phase_b_sigs.tail(30)
    assert (late_phase_b["symbol"] == "AGG").all(), (
        f"Expected AGG in late Phase B; got: {late_phase_b['symbol'].unique()}"
    )


# ---------------------------------------------------------------------------
# Test 7 — bfill look-ahead guard: staggered BIL inception must not leak future
# ---------------------------------------------------------------------------


def _make_4asset_prices_staggered_inception(bil_lag_months: int = 5) -> pd.DataFrame:
    """4-asset panel where BIL starts bil_lag_months after SPY/VEU/AGG.

    This exercises the bfill-removal fix: if .bfill() were still present, the
    leading NaN rows for BIL would be filled backward from its first valid price,
    making it look like BIL data existed before its actual inception — a form of
    look-ahead bias that would corrupt the 12M absolute-momentum computation for
    the earliest rebalances.
    """
    n_bars = (_LOOKBACK + bil_lag_months + 4) * 21 + 60
    bil_lag_bars = bil_lag_months * 21

    dates = pd.date_range("2007-01-02", periods=n_bars, freq="B", tz="UTC")
    rng = np.random.default_rng(99)

    params = {
        "SPY": (300.0, 0.0004, 0.010),
        "VEU": (50.0, 0.0003, 0.009),
        "AGG": (110.0, 0.0002, 0.003),
    }
    rows = []
    for sym, (start, drift, vol) in params.items():
        rets = rng.standard_normal(n_bars) * vol + drift
        close = start * np.cumprod(1.0 + rets)
        for d, c in zip(dates, close):
            rows.append({"timestamp": d, "symbol": sym, "close": c})

    # BIL starts bil_lag_months later
    bil_dates = dates[bil_lag_bars:]
    bil_rets = rng.standard_normal(len(bil_dates)) * 0.0002 + 0.00008
    bil_close = 90.0 * np.cumprod(1.0 + bil_rets)
    for d, c in zip(bil_dates, bil_close):
        rows.append({"timestamp": d, "symbol": "BIL", "close": c})

    return pd.DataFrame(rows)


def test_pit_staggered_bil_inception_bfill_guard():
    """At the FIRST signal the 12M base_cutoff falls BEFORE BIL inception.

    Without bfill (correct): p_base[BIL] = NaN → hurdle_ret = NaN
    → absolute-momentum filter cannot run → selected = safe_asset (AGG).

    With bfill (incorrect): p_base[BIL] = BIL's first observed price (a future
    value relative to base_cutoff) → hurdle_ret ≈ 0 → equity outperformer with
    positive 12M return would be selected instead.

    This test therefore fails loudly if .bfill() is reintroduced.

    Fixture layout (bil_lag_months=5):
      months 0–4:   SPY, VEU, AGG only; BIL = NaN
      month  5 onward: all 4 symbols
      first EOM signal: ~month 13
      base_cutoff at first signal: ~month 1 < BIL inception month 5  ← key property
    """
    bil_lag = 5
    prices = _make_4asset_prices_staggered_inception(bil_lag_months=bil_lag)

    sigs = generate_dual_momentum_signals_from_prices(prices, lookback_months=_LOOKBACK)
    assert not sigs.empty, "No signals produced — check fixture construction"

    all_ts = sorted(sigs["timestamp"].unique())
    assert len(all_ts) >= 2, f"Need ≥2 signal timestamps; got {len(all_ts)}"

    first_signal_ts = all_ts[0]
    first_row = sigs[sigs["timestamp"] == first_signal_ts]
    assert len(first_row) == 1, "Expected exactly one signal row at first_signal_ts"

    # Verify the fixture guarantee: base_cutoff at first signal IS before BIL inception.
    bil_first_ts = prices[prices["symbol"] == "BIL"]["timestamp"].min()
    base_cutoff = first_signal_ts - pd.DateOffset(months=_LOOKBACK)
    assert base_cutoff < bil_first_ts, (
        f"Fixture guarantee violated: base_cutoff {base_cutoff.date()} >= "
        f"BIL inception {bil_first_ts.date()} — increase bil_lag_months"
    )

    # Core assertion: NaN hurdle → safe_asset (AGG) selected at first signal.
    # If bfill were active, p_base[BIL] would be BIL's first real price (a future
    # value for base_cutoff), making hurdle_ret non-NaN, and an equity asset might
    # be selected instead — this assertion would then fail.
    assert first_row.iloc[0]["symbol"] == "AGG", (
        f"Expected AGG at first signal (base_cutoff before BIL inception → NaN hurdle); "
        f"got {first_row.iloc[0]['symbol']} — bfill look-ahead may be active"
    )
