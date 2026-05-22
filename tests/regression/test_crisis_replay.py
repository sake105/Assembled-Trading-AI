"""P0 A7 — Crisis-replay regression harness (scaffold).

Scope
-----
Replays the reference portfolio backtest over **synthetic fixtures shaped to
historical crisis stylized facts** and asserts engine-level invariants (not
specific dollar P&L numbers).

Four scenarios are scaffolded:

* ``flash_2010`` — 2010-05-06 flash crash: single-session -9% drop with
  mean-reversion inside the same bar.
* ``covid_2020`` — 2020-02 to 2020-03 COVID crash: -35% over ~20 sessions,
  realised vol ~5% daily.
* ``gme_2021`` — 2021-01 GME short-squeeze: one idiosyncratic symbol
  experiences +1000% over 5 sessions while the rest trend normally.
* ``svb_2023`` — 2023-03 SVB regional-banking shock: a small cluster
  (3 symbols) -60% in 2 sessions, rest unaffected.

Invariants checked (per scenario):

1. Backtest completes without exception.
2. Final equity is finite and strictly positive (no bankruptcy).
3. Max drawdown is bounded (< 95%) — the engine cannot lose everything.
4. Order count is finite and bounded (no pathological churn).

**Scope for A7 Week 1:** scaffold only, with synthetic stylized-fact fixtures.
Replacing the synthetic paths with real Polygon-fetched event windows is a
follow-up (~$50 of Polygon data). See ``docs/runbooks/13_crisis_replay_fixtures.md``.

Why invariants, not golden-equity
---------------------------------
Golden-equity regression (bit-identical output) is useful for pure code
refactors. Crisis-replay is used to catch behavioural regressions — for
example, a future change that makes the kill-switch fire less aggressively,
or lets leverage explode in tail scenarios. Those are not "the equity curve
moved by 0.01%" problems; they are "the engine behaved qualitatively
differently" problems. Hence: shape-bounded assertions on invariants, not
equality on a committed JSON.

The golden-equity gate remains the canonical 1e-9 regression in
``tests/regression/test_golden_equity_baseline.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.phase_zero

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.profile_backtest import (  # noqa: E402
    equal_weight_sizing_fn,
    simple_signal_fn,
)
from src.assembled_core.qa.backtest_engine import run_portfolio_backtest  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic crisis fixtures
# ---------------------------------------------------------------------------


def _base_path(
    n_symbols: int, n_days: int, seed: int, start: str = "2024-01-02"
) -> pd.DataFrame:
    """Deterministic normal-regime price frame shared as pre-event baseline."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start, periods=n_days, freq="B", tz="UTC")
    rows = []
    for sym_idx in range(n_symbols):
        symbol = f"SYM{sym_idx:02d}"
        base = 50.0 + (sym_idx * 7) % 150
        rets = rng.normal(0.0004, 0.015, len(dates))
        closes = base * np.exp(np.cumsum(rets))
        for i, date in enumerate(dates):
            rows.append(
                {"timestamp": date, "symbol": symbol, "close": float(closes[i])}
            )
    return (
        pd.DataFrame(rows).sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    )


def _build_flash_2010(
    n_symbols: int = 15, n_days: int = 30, seed: int = 42
) -> pd.DataFrame:
    """Synthetic flash-crash fixture: single-bar -9% drop then rebound."""
    df = _base_path(n_symbols, n_days, seed)
    # Pick the middle bar as the crash bar; drop all prices 9%, next bar rebound 7%.
    unique_ts = sorted(df["timestamp"].unique())
    crash_ts = unique_ts[n_days // 2]
    rebound_ts = unique_ts[n_days // 2 + 1]
    df.loc[df["timestamp"] == crash_ts, "close"] *= 0.91
    df.loc[df["timestamp"] == rebound_ts, "close"] *= 1.07
    return df


def _build_covid_2020(
    n_symbols: int = 15, n_days: int = 60, seed: int = 43
) -> pd.DataFrame:
    """Synthetic COVID-style crash: 20-session compound drawdown.

    Applies a deterministic -3% daily compound drift on top of the baseline
    over a 20-session window. Net effect: ~0.97**20 ≈ -46%, comfortably deep
    enough that the stylized-facts guard detects a meaningful drawdown.
    """
    df = _base_path(n_symbols, n_days, seed)
    unique_ts = sorted(df["timestamp"].unique())
    drop_start = n_days // 3
    drop_end = drop_start + 20
    # Cumulative multiplier: sessions 0..19 within the window get 0.97**(i+1).
    for i, ts in enumerate(unique_ts[drop_start:drop_end]):
        df.loc[df["timestamp"] == ts, "close"] *= 0.97 ** (i + 1)
    # Keep post-window prices anchored to the crash-end level.
    if drop_end < len(unique_ts):
        final_mult = 0.97**20
        for ts in unique_ts[drop_end:]:
            df.loc[df["timestamp"] == ts, "close"] *= final_mult
    return df


def _build_gme_2021(
    n_symbols: int = 15, n_days: int = 30, seed: int = 44
) -> pd.DataFrame:
    """Synthetic short-squeeze fixture: one symbol +1000% over 5 sessions.

    Applies a compounding level shift to SYM00 across a 5-bar squeeze window
    and carries the final level forward. Peak/trough ratio after this is
    ~10x (1.58**5), matching the order of magnitude of the 2021 GME episode.
    """
    df = _base_path(n_symbols, n_days, seed)
    gme = "SYM00"
    unique_ts = sorted(df["timestamp"].unique())
    squeeze_start = n_days // 2
    squeeze_end = squeeze_start + 5
    # Compounding: bar 0 of the window gets ×1.58, bar 1 gets ×1.58**2, …,
    # bar 4 gets ×1.58**5. Post-window bars all scale by ×1.58**5.
    for i, ts in enumerate(unique_ts[squeeze_start:squeeze_end]):
        mask = (df["timestamp"] == ts) & (df["symbol"] == gme)
        df.loc[mask, "close"] *= 1.58 ** (i + 1)
    final_mult = 1.58**5
    for ts in unique_ts[squeeze_end:]:
        mask = (df["timestamp"] == ts) & (df["symbol"] == gme)
        df.loc[mask, "close"] *= final_mult
    return df


def _build_svb_2023(
    n_symbols: int = 15, n_days: int = 30, seed: int = 45
) -> pd.DataFrame:
    """Synthetic regional-bank cluster shock: 3 symbols drop ~60% over 2 bars.

    Compounds 0.63 per session across 2 sessions (→ ~0.40 cumulative, i.e.
    -60%) on the banking cluster, then carries that level forward for the
    rest of the window. Other symbols unaffected.
    """
    df = _base_path(n_symbols, n_days, seed)
    banks = ["SYM01", "SYM02", "SYM03"]
    unique_ts = sorted(df["timestamp"].unique())
    crash_start = n_days // 2
    crash_end = crash_start + 2
    for i, ts in enumerate(unique_ts[crash_start:crash_end]):
        mask = (df["timestamp"] == ts) & (df["symbol"].isin(banks))
        df.loc[mask, "close"] *= 0.63 ** (i + 1)
    final_mult = 0.63**2
    for ts in unique_ts[crash_end:]:
        mask = (df["timestamp"] == ts) & (df["symbol"].isin(banks))
        df.loc[mask, "close"] *= final_mult
    return df


CRISIS_BUILDERS = {
    "flash_2010": _build_flash_2010,
    "covid_2020": _build_covid_2020,
    "gme_2021": _build_gme_2021,
    "svb_2023": _build_svb_2023,
}


# ---------------------------------------------------------------------------
# Invariant assertions
# ---------------------------------------------------------------------------


def _assert_engine_survives(scenario: str, prices: pd.DataFrame) -> None:
    result = run_portfolio_backtest(
        prices=prices,
        signal_fn=simple_signal_fn,
        position_sizing_fn=equal_weight_sizing_fn,
        start_capital=100_000.0,
        include_costs=True,
        include_trades=True,
        include_ledger=False,
        strict_session_gate=False,
    )

    assert result.equity is not None and not result.equity.empty, (
        f"[{scenario}] equity curve empty — engine aborted silently"
    )
    eq = result.equity["equity"].to_numpy(dtype=np.float64)

    # Invariant 1 — no NaN / inf.
    assert np.isfinite(eq).all(), f"[{scenario}] equity contains NaN/inf"

    # Invariant 2 — no bankruptcy.
    final_eq = float(eq[-1])
    assert final_eq > 0.0, f"[{scenario}] bankruptcy: final_equity={final_eq:.2f}"

    # Invariant 3 — drawdown bound. `run_portfolio_backtest` does not leverage.
    # A well-behaved unleveraged long-only strategy cannot lose > 95% on these
    # synthetic shocks; tripping this bound means the engine's risk controls
    # regressed (e.g. kill-switch stopped firing).
    running_max = np.maximum.accumulate(eq)
    dd = (eq - running_max) / running_max
    max_dd = float(dd.min())
    assert max_dd > -0.95, (
        f"[{scenario}] max drawdown {max_dd:.1%} breached -95% bound — "
        "engine behaved as if leverage were applied or kill-switch failed"
    )

    # Invariant 4 — trade count is finite and not pathological. A well-behaved
    # engine cannot emit more fills than N_symbols × N_bars × 3 even on a
    # chaotic fixture — anything beyond that points at infinite-loop or
    # duplicate-order territory.
    if result.trades is not None:
        n_trades = int(len(result.trades))
        n_symbols = prices["symbol"].nunique()
        n_bars = prices["timestamp"].nunique()
        upper = 3 * n_symbols * n_bars
        assert 0 <= n_trades <= upper, (
            f"[{scenario}] trade count {n_trades} outside [0, {upper}] — "
            "pathological order generation"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", sorted(CRISIS_BUILDERS.keys()))
def test_crisis_replay_engine_invariants(scenario: str) -> None:
    """Each synthetic crisis scenario must satisfy engine-level invariants."""
    prices = CRISIS_BUILDERS[scenario]()
    _assert_engine_survives(scenario, prices)


def test_crisis_fixture_builders_are_deterministic() -> None:
    """Re-building each fixture must produce identical frames (seeded)."""
    for name, build in CRISIS_BUILDERS.items():
        a = build()
        b = build()
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True),
            b.reset_index(drop=True),
            check_dtype=False,
        )


def test_crisis_fixtures_actually_contain_shock() -> None:
    """Guard against the builders silently degrading back to baseline paths.

    A future refactor that unwires the shock injection would make the
    invariant tests trivially pass on a calm path. This test pins the
    stylized facts per scenario — each event has a different shape, so
    drawdown is the wrong yardstick for an idiosyncratic squeeze like GME.
    """
    # Broad drawdown scenarios: equal-weighted price index should dip.
    index_dd_required = {
        "flash_2010": -0.05,
        "covid_2020": -0.15,
    }
    for name, threshold in index_dd_required.items():
        prices = CRISIS_BUILDERS[name]()
        pivot = prices.pivot(index="timestamp", columns="symbol", values="close")
        idx = pivot.mean(axis=1).to_numpy(dtype=np.float64)
        running_max = np.maximum.accumulate(idx)
        dd = (idx - running_max) / running_max
        assert float(dd.min()) <= threshold, (
            f"[{name}] fixture has no meaningful index drawdown "
            f"({float(dd.min()):.3f} > {threshold}) — shock injection likely broken"
        )

    # GME is idiosyncratic: one symbol moves violently up, the rest are normal.
    # Stylized fact: peak/trough ratio on the squeezed name > 5x.
    gme_prices = CRISIS_BUILDERS["gme_2021"]()
    sym00 = gme_prices[gme_prices["symbol"] == "SYM00"]["close"].to_numpy(
        dtype=np.float64
    )
    ratio = float(sym00.max() / sym00.min())
    assert ratio > 5.0, (
        f"[gme_2021] SYM00 peak/trough ratio {ratio:.2f} ≤ 5 — "
        "short-squeeze shock injection likely broken"
    )

    # SVB is a cluster shock: only 3 of 15 symbols (the "bank" cluster) crash.
    # Check the cluster, not the broad index — a 20%-cluster-weighted crash
    # dilutes to <5% in the broad index by construction.
    svb_prices = CRISIS_BUILDERS["svb_2023"]()
    cluster = svb_prices[svb_prices["symbol"].isin(["SYM01", "SYM02", "SYM03"])]
    piv = cluster.pivot(index="timestamp", columns="symbol", values="close")
    cluster_idx = piv.mean(axis=1).to_numpy(dtype=np.float64)
    running_max = np.maximum.accumulate(cluster_idx)
    dd = (cluster_idx - running_max) / running_max
    assert float(dd.min()) <= -0.40, (
        f"[svb_2023] banking cluster drawdown {float(dd.min()):.3f} > -0.40 — "
        "cluster shock injection likely broken"
    )
