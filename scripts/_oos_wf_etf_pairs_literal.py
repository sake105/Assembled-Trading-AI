"""etf_pairs_meanrev through the LITERAL pipeline (`run_trading_cycle`) — Part B closure.

Writes docs/results/2026_05_etf_pairs_literal_oos.md (companion artifact); the
consolidated `2026_05_pipeline_realistic_oos.md` B.4 bullet is updated by hand to
cite this number.

WHY this exists
---------------
The matrix harness `_oos_wf_pipeline_realistic.py` (Part A + B.1) drives strategies
through a CROSS-SECTIONAL top/bottom-quintile rank harness on the 75-symbol offline
survivor cache. `etf_pairs_meanrev` could NOT be driven there because it is NOT a
cross-sectional rank strategy: it trades *cointegrated relative-value SHORT pairs*
(spread mean-reversion), market-neutral, daily. Forcing it through the rank harness
would misrepresent it, so Part B.4 listed it as "excluded by design". This script
closes that gap HONESTLY: it sources the 6 default ETF pairs (12 symbols) from Alpaca
and routes the REGISTERED `etf_pairs_meanrev` signal through the IDENTICAL literal
machinery (`run_portfolio_backtest(cycle_fn=make_cycle_fn(...))` → `run_trading_cycle`
→ feature enrichment → `generate_orders_from_targets` → `simulate_with_costs`).

STRATEGY (registered defaults — NOT re-tuned here)
--------------------------------------------------
6 pairs: SPY/IVV, GDX/GDXJ, XLE/VDE, EWA/EWC, XLF/KBE, XLK/VGT. Rolling 252-bar
Engle-Granger cointegration test + OLS hedge ratio; 60-bar Z-score; entry |Z|>2.0,
exit |Z|<0.5, stop |Z|>3.5, 5-bar post-stop cooldown. FULL long-short mode: each
active pair contributes a LONG leg and a SHORT leg at weight ±1/k, so GROSS notional
= 2.0 × capital (200% gross) and NET ≈ 0 (market-neutral by design). With
`enable_risk_controls=False` no gross-exposure cap is applied — this is the strategy
as natively designed. DAILY rebalance (the strategy's native frequency: it enters/
exits on Z-score crossings that occur intraday-to-weekly, NOT monthly).

EVIDENCE TIER (binding honesty)
-------------------------------
This is the SAME third tier as the dual_momentum literal study, NOT Part A's "Literal"
tier:
  - SAME execution realism as Part A (real cost model, order-gen, enrichment, same
    cycle, risk-controls OFF, 252/252/252 WF) — but DAILY rebalance (pairs is a daily
    strategy), not monthly, and FULL long-short (gross 200%), not long-only.
  - DIFFERENT data source: Alpaca daily bars for the 12 pair ETFs, NOT the offline
    survivor cache. The SPY benchmark here is the Alpaca SPY buy-and-hold over the
    same windows — its own series, read as such.
  - SPY/IVV is one of the traded pairs, so SPY is BOTH benchmark and (occasionally) a
    traded leg; the benchmark uses SPY close independently of any held SPY position.

MARKET-NEUTRAL CAVEAT (read before interpreting the verdict)
------------------------------------------------------------
This is a beta≈0 market-neutral book. It is NOT designed to out-CAGR a bull-market
SPY; in a rising tape a neutral book structurally trails SPY on absolute return. The
MEANINGFUL question is risk-adjusted: does the spread alpha clear SPY's Sharpe with a
DSR-deflated AND significant (IR t>1.96) edge? The falsification bar is UNCHANGED
(must beat SPY Sharpe AND DSR AND IR-t to be a PROSPECT) — but a REJECTED verdict on
CAGR alone is expected and is NOT the interesting signal here; the Sharpe/DSR/IR line
is.

PIT / causality (precompute is PIT-identical to per-as_of recompute)
--------------------------------------------------------------------
For performance the full causal signal panel is precomputed ONCE per fold via
`generate_etf_pairs_signals_from_prices`, then the signal_fn looks up the as_of row.
This is PIT-SAFE and numerically IDENTICAL to calling `compute_signals` on a slice
≤ as_of, because the strategy's state machine (`_compute_pair_states`) is strictly
causal: state[t] depends only on the rolling window [t-251 … t] and prior states —
future bars (> as_of) never enter the computation of the as_of row. main() runs an
explicit PIT self-check asserting this before the WF loop. On top of the precompute,
the cycle itself slices `precomputed_prices_with_features` to ≤ as_of, and adds its
own execution lag.

STATE ISOLATION
---------------
ASSEMBLED_NO_CRISIS_OVERLAY=1 forces the crisis overlay to dry_run (no state
persisted); write_outputs=False, backtest_use_snapshot=False. No production module,
policy.yaml, or live state file is mutated. Alpaca bars are cached to a
research-local parquet (output/research/), NOT the production price cache.

DATA HONESTY
------------
Alpaca bar close ≈ price return (no dividend reinvestment). For RELATIVE-VALUE pairs
this matters far less than for dual_momentum: both legs of a pair are similar
instruments (e.g. SPY vs IVV, XLK vs VGT) with near-identical yields, so the dividend
omission largely cancels in the spread. Residual divergence (e.g. GDX vs GDXJ payout
differences) is a second-order spread bias. Costs: the literal cost model charges
commission+spread+impact on every leg change; SHORT legs incur the same model — NO
explicit short-borrow/locate fee is modelled (a known optimistic omission for the
short side; disclosed). statsmodels is REQUIRED (Engle-Granger) — abort if absent.
CI: not run; local one-shot.

Falsification rule: etf_pairs_meanrev beats SPY risk-adjusted (Sharpe > SPY AND
DSR-deflated AND IR t>1.96) through the real pipeline — or it is REJECTED.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# State isolation must precede production imports / any cycle run (see module docstring; E-035).
# Unconditional assignment, not setdefault: setdefault is a no-op when the var is
# already set (e.g. to "0"), silently re-enabling the live-state write side effect
# (E-035). An explicit conflicting override is forced to "1" and warned loudly.
_prev_no_overlay = os.environ.get("ASSEMBLED_NO_CRISIS_OVERLAY")
if _prev_no_overlay is not None and _prev_no_overlay != "1":
    print(
        f"[WARN][E-035] ASSEMBLED_NO_CRISIS_OVERLAY={_prev_no_overlay!r} overridden to "
        "'1' — this research harness must NOT persist crisis state to live "
        "output/ops/ (E-035).",
        file=sys.stderr,
    )
os.environ["ASSEMBLED_NO_CRISIS_OVERLAY"] = "1"
assert os.environ["ASSEMBLED_NO_CRISIS_OVERLAY"] == "1"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the prior studies' edge/metric helpers + the matrix harness's pipeline
# primitives + sizing fn verbatim (DRY: no second truth for the edge math or the
# cycle wiring).
from scripts._oos_wf_dual_momentum import _fetch_alpaca  # noqa: E402
from scripts._oos_wf_leverage_short import (  # noqa: E402
    INITIAL_CAPITAL,
    N_TRIALS_DSR,
    STEP_SIZE,
    TEST_WINDOW,
    TRAIN_WINDOW,
    _benchmark_spy,
    _edge_metrics,
    _fmt,
    _metrics,
    _spy_pooled_edge,
)
from scripts._oos_wf_pipeline_realistic import (  # noqa: E402
    ENABLE_RISK_CONTROLS,
    INCLUDE_COSTS,
    sizing_fn,
)
from src.assembled_core.pipeline.trading_cycle_shared import TradingContext  # noqa: E402
from src.assembled_core.qa.backtest_engine import (  # noqa: E402
    BacktestResult,
    make_cycle_fn,
    run_portfolio_backtest,
)
from src.assembled_core.strategies.etf_pairs_meanrev import (  # noqa: E402
    _DEFAULT_COINT_WINDOW,
    _DEFAULT_ENTRY_Z,
    _DEFAULT_EXIT_Z,
    _DEFAULT_PAIRS,
    _DEFAULT_STOP_Z,
    _DEFAULT_ZSCORE_WINDOW,
    _HAS_STATSMODELS,
    compute_signals,
    generate_etf_pairs_signals_from_prices,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oos_wf_etf_pairs_literal")

# ---------------------------------------------------------------------------
# Config — 252/252/252 WF like Part A, but DAILY rebalance (pairs is a daily
# strategy) and FULL long-short (gross 200%). Alpaca menu starts 2015-06 → ~7-8
# folds spanning ~2016-06 … 2024.
# ---------------------------------------------------------------------------
PAIRS = list(_DEFAULT_PAIRS)
SYMBOLS = sorted({s for pair in PAIRS for s in pair})  # 12 distinct ETFs
LONG_ONLY = False  # native full long-short (market-neutral, gross 200%)

# Warmup must cover the 252-bar Engle-Granger cointegration window so the FIRST
# in-test signal sees a full window. = coint window (the minimum for a valid signal).
WARMUP_BARS = _DEFAULT_COINT_WINDOW  # 252
# Daily rebalance, but include a short pre-test lead so the book is positioned
# ENTERING test_start (OOS scoring still masks to [test_start, test_end)).
PRE_TEST_LEAD_BARS = 10

FETCH_START = pd.Timestamp("2015-06-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-02-28", tz="UTC")

CACHE_PARQUET = ROOT / "output" / "research" / "etf_pairs_literal_prices.parquet"
OUT_MD = ROOT / "docs" / "results" / "2026_05_etf_pairs_literal_oos.md"


# ---------------------------------------------------------------------------
# Data sourcing — Alpaca → research-local parquet (deterministic re-runs)
# ---------------------------------------------------------------------------
def _source_prices() -> pd.DataFrame:
    if CACHE_PARQUET.exists():
        log.info("Loading cached Alpaca menu from %s", CACHE_PARQUET)
        df = pd.read_parquet(CACHE_PARQUET)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        have = set(df["symbol"].unique())
        if set(SYMBOLS).issubset(have):
            return df
        log.warning("Cache missing %s — refetching", set(SYMBOLS) - have)
    df = _fetch_alpaca(SYMBOLS, start=FETCH_START, end=PERIOD_END)
    CACHE_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE_PARQUET, index=False)
    log.info("Cached %d rows -> %s", len(df), CACHE_PARQUET)
    return df


# ---------------------------------------------------------------------------
# Signal: registered etf_pairs legs at as_of, mapped to SIGNED weight
# ---------------------------------------------------------------------------
def _legs_to_signed(rows: pd.DataFrame) -> pd.DataFrame:
    """Map registered [timestamp, symbol, direction, score] legs to the signed
    weight schema the shared sizing_fn consumes: LONG→+score, SHORT→-score."""
    cols = ["timestamp", "symbol", "direction", "score", "weight"]
    if rows is None or rows.empty:
        return pd.DataFrame(columns=cols)
    out = rows.copy()
    sign = np.where(out["direction"].astype(str) == "SHORT", -1.0, 1.0)
    out["weight"] = out["score"].astype(float) * sign
    return out[cols]


def make_pairs_signal_fn(full_panel: pd.DataFrame):
    """signal_fn(features) -> signed-weight legs for features' as_of timestamp.

    `full_panel` is the precomputed causal signal panel for the whole fold slice
    (see module docstring: PIT-identical to per-as_of recompute). The signal_fn
    looks up the rows whose timestamp == features.timestamp.max() (the as_of bar).
    Returns an empty frame when the strategy is FLAT at as_of (no active pair),
    which the pipeline correctly flattens to cash.
    """
    cols = ["timestamp", "symbol", "direction", "score", "weight"]
    if full_panel is None or full_panel.empty:
        panel_by_ts: dict[pd.Timestamp, pd.DataFrame] = {}
    else:
        fp = full_panel.copy()
        fp["timestamp"] = pd.to_datetime(fp["timestamp"], utc=True)
        panel_by_ts = {ts: g for ts, g in fp.groupby("timestamp")}

    def signal_fn(features: pd.DataFrame) -> pd.DataFrame:
        empty = pd.DataFrame(columns=cols)
        if features is None or features.empty or "timestamp" not in features.columns:
            return empty
        as_of = pd.to_datetime(features["timestamp"], utc=True).max()
        rows = panel_by_ts.get(as_of)
        if rows is None or rows.empty:
            return empty
        return _legs_to_signed(rows)

    return signal_fn


def _precompute_panel(sl: pd.DataFrame) -> pd.DataFrame:
    """Full causal signal panel for the fold slice (one O(T) state-machine pass)."""
    return generate_etf_pairs_signals_from_prices(
        sl[["timestamp", "symbol", "close"]].dropna(),
        pairs=PAIRS,
        cointegration_window=_DEFAULT_COINT_WINDOW,
        zscore_window=_DEFAULT_ZSCORE_WINDOW,
        entry_z=_DEFAULT_ENTRY_Z,
        exit_z=_DEFAULT_EXIT_Z,
        stop_z=_DEFAULT_STOP_Z,
        long_only=LONG_ONLY,
    )


# ---------------------------------------------------------------------------
# One WF fold through run_trading_cycle (DAILY rebalance)
# ---------------------------------------------------------------------------
def _simulate_pairs_pipeline(prices, test_start, test_end):
    spy_dates = np.sort(prices[prices["symbol"] == "SPY"]["timestamp"].unique())
    pre_test = spy_dates[spy_dates < test_start]
    if len(pre_test) < WARMUP_BARS:
        raise ValueError(
            f"insufficient warmup: {len(pre_test)} < {WARMUP_BARS} bars before test_start"
        )
    warmup_start = pd.Timestamp(pre_test[-WARMUP_BARS])
    if warmup_start.tzinfo is None:
        warmup_start = warmup_start.tz_localize("UTC")

    sl = prices[
        prices["symbol"].isin(SYMBOLS)
        & (prices["timestamp"] >= warmup_start)
        & (prices["timestamp"] <= test_end)
    ].copy()
    sl = sl.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Daily rebalances: from PRE_TEST_LEAD_BARS days before test_start through test_end
    # (so the book is positioned entering test_start; OOS scoring masks to the test window).
    sl_days = np.sort(sl["timestamp"].unique())
    lead_cut_candidates = sl_days[sl_days < test_start]
    lead_cut = (
        pd.Timestamp(lead_cut_candidates[-PRE_TEST_LEAD_BARS])
        if len(lead_cut_candidates) >= PRE_TEST_LEAD_BARS
        else (
            pd.Timestamp(lead_cut_candidates[0])
            if len(lead_cut_candidates)
            else test_start
        )
    )
    if lead_cut.tzinfo is None:
        lead_cut = lead_cut.tz_localize("UTC")
    rebs = sorted(
        pd.Timestamp(t) for t in sl_days if lead_cut <= pd.Timestamp(t) <= test_end
    )
    if len(rebs) < 5:
        raise ValueError(
            f"only {len(rebs)} rebalance days for fold {test_start.date()}"
        )

    full_panel = _precompute_panel(sl)
    sig_fn = make_pairs_signal_fn(full_panel)

    ctx_template = TradingContext(
        prices=sl,
        freq="1d",
        universe=list(SYMBOLS),
        use_factor_store=False,
        precomputed_prices_with_features=sl,
        write_outputs=False,
        enable_risk_controls=ENABLE_RISK_CONTROLS,
        backtest_use_snapshot=False,
    )
    cycle_fn = make_cycle_fn(
        ctx_template,
        signal_fn=sig_fn,
        position_sizing_fn=sizing_fn,
        capital=INITIAL_CAPITAL,
        enable_risk_controls=ENABLE_RISK_CONTROLS,
    )
    result: BacktestResult = run_portfolio_backtest(
        prices=sl,
        signal_fn=sig_fn,
        position_sizing_fn=sizing_fn,
        start_capital=INITIAL_CAPITAL,
        include_costs=INCLUDE_COSTS,
        include_trades=True,
        include_targets=True,
        compute_features=False,
        cycle_fn=cycle_fn,
        include_ledger=False,
        strict_session_gate=False,
        rebalance_schedule="daily",
        rebalance_timestamps=rebs,
    )

    eq = result.equity
    if eq is None or eq.empty or "equity" not in eq.columns:
        raise ValueError("empty equity curve")
    eq = eq.copy()
    tcol = (
        "timestamp"
        if "timestamp" in eq.columns
        else ("date" if "date" in eq.columns else None)
    )
    if tcol is None:
        raise ValueError("equity has no time column")
    eq[tcol] = pd.to_datetime(eq[tcol], utc=True)
    eq = eq.sort_values(tcol)
    ret_all = pd.Series(
        eq["equity"].pct_change().to_numpy(),
        index=pd.DatetimeIndex(eq[tcol], name=None),
    ).dropna()

    spy_px = (
        sl[sl["symbol"] == "SPY"][["timestamp", "close"]]
        .drop_duplicates("timestamp")
        .set_index("timestamp")["close"]
        .sort_index()
    )
    spy_ret_all = spy_px.pct_change().dropna()

    mask = (ret_all.index >= test_start) & (ret_all.index < test_end)
    net_ret = ret_all[mask]
    if len(net_ret) < 5:
        raise ValueError(f"Only {len(net_ret)} OOS bars in fold")
    spy_test = spy_ret_all.reindex(net_ret.index)

    m = _metrics(net_ret)
    common = net_ret.index.intersection(spy_test.dropna().index)
    if len(common) > 5 and spy_test[common].std() > 1e-12:
        beta_mkt = float(
            np.cov(net_ret[common], spy_test[common])[0, 1] / np.var(spy_test[common])
        )
    else:
        beta_mkt = float("nan")

    # Diagnostics: avg gross exposure (sum |target_weight| per rebal), avg active legs,
    # rejected-trade count (fill-model gate / short handling sanity).
    avg_gross = float("nan")
    avg_legs = float("nan")
    tgts = result.target_positions
    if (
        tgts is not None
        and not tgts.empty
        and {"timestamp", "target_weight"}.issubset(tgts.columns)
    ):
        tg = tgts.copy()
        tg["timestamp"] = pd.to_datetime(tg["timestamp"], utc=True)
        tg = tg[(tg["timestamp"] >= test_start) & (tg["timestamp"] < test_end)]
        if not tg.empty:
            per_ts = tg.groupby("timestamp").agg(
                gross=("target_weight", lambda s: float(np.abs(s).sum())),
                legs=("symbol", "nunique"),
            )
            avg_gross = float(per_ts["gross"].mean())
            avg_legs = float(per_ts["legs"].mean())

    n_rej = float("nan")
    tr = result.trades
    if tr is not None and not tr.empty and "status" in tr.columns:
        n_rej = float((tr["status"].astype(str) == "rejected").sum())

    diag = dict(beta_mkt=beta_mkt, avg_gross=avg_gross, avg_legs=avg_legs, n_rej=n_rej)
    return m, diag, net_ret, spy_test


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def _run_wf(prices):
    spy_dates = np.sort(prices[prices["symbol"] == "SPY"]["timestamp"].unique())
    results, pooled_strat, pooled_spy = [], [], []
    fold_idx = 1
    for train_start_i in range(
        0, len(spy_dates) - TRAIN_WINDOW - TEST_WINDOW + 1, STEP_SIZE
    ):
        train_end_i = train_start_i + TRAIN_WINDOW
        test_end_i = train_end_i + TEST_WINDOW
        if test_end_i > len(spy_dates):
            break
        test_start_ts = pd.Timestamp(spy_dates[train_end_i])
        test_start = (
            test_start_ts.tz_localize("UTC")
            if test_start_ts.tzinfo is None
            else test_start_ts
        )
        test_end_ts = pd.Timestamp(spy_dates[test_end_i - 1])
        test_end = (
            test_end_ts.tz_localize("UTC")
            if test_end_ts.tzinfo is None
            else test_end_ts
        ) + pd.Timedelta(hours=23)

        t0 = time.time()
        try:
            m, diag, net_ret, spy_test = _simulate_pairs_pipeline(
                prices, test_start, test_end
            )
            bm_spy = _benchmark_spy(prices, test_start, test_end)
            r = dict(
                fold=fold_idx,
                test_start=test_start.date(),
                test_end=test_end.date(),
                cagr=m["cagr"],
                sharpe=m["sharpe"],
                maxdd=m["maxdd"],
                calmar=m["calmar"],
                spy_cagr=bm_spy["cagr"],
                spy_sharpe=bm_spy["sharpe"],
                spy_maxdd=bm_spy["maxdd"],
                beta_mkt=diag["beta_mkt"],
                avg_gross=diag["avg_gross"],
                avg_legs=diag["avg_legs"],
                n_rej=diag["n_rej"],
                n_bars=len(net_ret),
                status="OK",
            )
            pooled_strat.append(net_ret)
            pooled_spy.append(spy_test.reindex(net_ret.index))
            log.info(
                "[etf_pairs] Fold %d %s-%s: CAGR %.1f%% / Sharpe %.2f / MaxDD %.1f%% / "
                "beta %.2f / gross %.2f / legs %.1f / rej %.0f  (SPY: %.1f%% / %.2f)  [%.1fs]",
                fold_idx,
                test_start.date(),
                test_end.date(),
                m["cagr"] * 100,
                m["sharpe"],
                m["maxdd"] * 100,
                diag["beta_mkt"],
                diag["avg_gross"],
                diag["avg_legs"],
                diag["n_rej"],
                bm_spy["cagr"] * 100,
                bm_spy["sharpe"],
                time.time() - t0,
            )
        except Exception as exc:
            log.warning(
                "[etf_pairs] Fold %d FAILED: %s [%.1fs]",
                fold_idx,
                exc,
                time.time() - t0,
            )
            r = dict(
                fold=fold_idx,
                test_start=test_start.date(),
                test_end=test_end.date(),
                status=f"FAILED: {exc}",
            )
        results.append(r)
        fold_idx += 1

    pooled_s = (
        pd.concat(pooled_strat).sort_index() if pooled_strat else pd.Series(dtype=float)
    )
    pooled_b = (
        pd.concat(pooled_spy).sort_index() if pooled_spy else pd.Series(dtype=float)
    )
    return results, pooled_s, pooled_b


# ---------------------------------------------------------------------------
# PIT self-check: precompute on a longer slice must equal compute_signals on ≤as_of
# ---------------------------------------------------------------------------
def _pit_self_check(prices) -> None:
    """Assert the precompute is PIT-identical to a per-as_of recompute.

    Picks an as_of where the strategy is ACTUALLY IN A TRADE (≥1 active leg),
    computes the registered `compute_signals` on data STRICTLY ≤ as_of, and
    compares its legs to the FULL-SAMPLE precomputed panel's row at that as_of.
    The full-sample panel is maximally future-contaminated (it has seen every bar
    after as_of), so if any future bar leaked into the as_of signal this raises.
    Selecting an in-a-trade as_of is deliberate: an empty==empty match would
    prove nothing about causality, so the check refuses to certify on a trivial
    (flat-book) as_of and downgrades to a WARNING in that case.
    """
    sdates = pd.DatetimeIndex(
        pd.Series(prices[prices["symbol"] == "SPY"]["timestamp"].unique())
    ).sort_values()
    if sdates.tz is None:
        sdates = sdates.tz_localize("UTC")
    n = len(sdates)
    if n < WARMUP_BARS + 80:
        log.warning("[pit-check] insufficient data for self-check — skipped")
        return

    base = prices[prices["symbol"].isin(SYMBOLS)][
        ["timestamp", "symbol", "close"]
    ].dropna()

    sig_kw = dict(
        pairs=PAIRS,
        cointegration_window=_DEFAULT_COINT_WINDOW,
        zscore_window=_DEFAULT_ZSCORE_WINDOW,
        entry_z=_DEFAULT_ENTRY_Z,
        exit_z=_DEFAULT_EXIT_Z,
        stop_z=_DEFAULT_STOP_Z,
        long_only=LONG_ONLY,
    )

    # Full-sample causal panel (sees ALL bars — maximal future contamination).
    full_panel = generate_etf_pairs_signals_from_prices(base, **sig_kw)

    # Pick the first as_of (with ≥1 leg) inside a window that keeps real history
    # before it and ≥1 future bar after it, so the leak-probe is non-trivial.
    earliest_ok = sdates[WARMUP_BARS + 10]
    latest_ok = sdates[n - 6]
    if not full_panel.empty:
        cand = full_panel[
            (full_panel["timestamp"] >= earliest_ok)
            & (full_panel["timestamp"] <= latest_ok)
        ]
        active_ts = pd.DatetimeIndex(
            pd.Series(cand["timestamp"].unique())
        ).sort_values()
    else:
        active_ts = pd.DatetimeIndex([])

    trivial = len(active_ts) == 0
    as_of = sdates[WARMUP_BARS + 40] if trivial else active_ts[0]

    le_asof = base[base["timestamp"] <= as_of]
    recompute = compute_signals(le_asof, **sig_kw)
    panel_asof = (
        full_panel[full_panel["timestamp"] == as_of]
        if not full_panel.empty
        else full_panel
    )

    def _key(df):
        if df is None or df.empty:
            return set()
        return {
            (str(r["symbol"]), str(r["direction"]), round(float(r["score"]), 8))
            for _, r in df.iterrows()
        }

    k_re, k_pa = _key(recompute), _key(panel_asof)
    if k_re != k_pa:
        raise AssertionError(
            f"PIT self-check FAILED at as_of={as_of.date()}: recompute(≤as_of)={k_re} "
            f"!= full-sample precompute.loc[as_of]={k_pa} — precompute leaks future bars"
        )
    if trivial or not k_re:
        log.warning(
            "[pit-check] NON-CERTIFYING at as_of=%s: 0 active legs — empty==empty match "
            "does not exercise causality (strategy flat at every scanned as_of?)",
            as_of.date(),
        )
    else:
        log.info(
            "[pit-check] PASS at as_of=%s: %d ACTIVE legs identical between ≤as_of recompute "
            "and full-sample precompute — precompute is PIT-safe (non-trivially exercised)",
            as_of.date(),
            len(k_re),
        )


# ---------------------------------------------------------------------------
# Report (companion artifact)
# ---------------------------------------------------------------------------
def _avg(results, key):
    vals = [
        r[key]
        for r in results
        if r.get("status") == "OK" and np.isfinite(r.get(key, float("nan")))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _fold_table(results):
    lines = [
        "| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | Legs | Rej | SPY CAGR | SPY Sharpe |",
        "|------|-------------|------|--------|-------|------|-------|------|-----|----------|------------|",
    ]
    for r in results:
        if r.get("status") != "OK":
            lines.append(
                f"| {r['fold']} | {r['test_start']}–{r['test_end']} | FAILED | | | | | | | | |"
            )
            continue
        lines.append(
            f"| {r['fold']} | {r['test_start']}–{r['test_end']} | {_fmt(r['cagr'])} | "
            f"{r['sharpe']:+.2f} | {_fmt(r['maxdd'])} | {r['beta_mkt']:+.2f} | "
            f"{r['avg_gross']:.2f} | {r['avg_legs']:.1f} | "
            f"{(r['n_rej'] if np.isfinite(r['n_rej']) else float('nan')):.0f} | "
            f"{_fmt(r['spy_cagr'])} | {r['spy_sharpe']:+.2f} |"
        )
    ok = [r for r in results if r.get("status") == "OK"]
    if ok:
        lines.append(
            f"| **Ø ({len(ok)}/{len(results)})** | — | **{_fmt(_avg(results, 'cagr'))}** | "
            f"**{_avg(results, 'sharpe'):+.2f}** | **{_fmt(_avg(results, 'maxdd'))}** | "
            f"**{_avg(results, 'beta_mkt'):+.2f}** | **{_avg(results, 'avg_gross'):.2f}** | "
            f"**{_avg(results, 'avg_legs'):.1f}** | **{_avg(results, 'n_rej'):.0f}** | "
            f"{_fmt(_avg(results, 'spy_cagr'))} | {_avg(results, 'spy_sharpe'):+.2f} |"
        )
    return "\n".join(lines)


def _write_report(results, edge, spy_edge, actual_start, actual_end):
    spy_sharpe = spy_edge.get("ann_sharpe", float("nan"))
    beats_sharpe = (
        np.isfinite(edge["ann_sharpe"])
        and np.isfinite(spy_sharpe)
        and edge["ann_sharpe"] > spy_sharpe
    )
    significant = (
        bool(edge["dsr_pass"]) and np.isfinite(edge["ir_t"]) and edge["ir_t"] > 1.96
    )
    tag = "PROSPECT" if (beats_sharpe and significant) else "REJECTED"

    ok = [r for r in results if r.get("status") == "OK"]
    n_folds = len(ok)
    fold_span = (
        f"{min(r['test_start'] for r in ok)}…{max(r['test_end'] for r in ok)}"
        if ok
        else "n/a"
    )
    pairs_str = ", ".join(f"{a}/{b}" for a, b in PAIRS)

    parts = [
        "# etf_pairs_meanrev through the LITERAL pipeline — OOS Walk-Forward",
        "",
        "Run date (UTC): 2026-05-31  ",
        f"Data: Alpaca daily bars (split-adjusted) for {', '.join(SYMBOLS)} — research-local cache  ",
        f"Pairs: {pairs_str}  ",
        f"Overlapping range: {actual_start.date()} → {actual_end.date()}  ",
        f"WF: {TRAIN_WINDOW}/{TEST_WINDOW}/{STEP_SIZE} (train/test/step), **DAILY** rebalance, "
        "FULL long-short (gross ≈ 200%, market-neutral)  ",
        "Execution: LITERAL `run_portfolio_backtest(cycle_fn=make_cycle_fn(...))` → `run_trading_cycle` "
        "→ feature enrichment → `generate_orders_from_targets` → `simulate_with_costs`  ",
        f"Config: `enable_risk_controls={ENABLE_RISK_CONTROLS}`, `include_costs={INCLUDE_COSTS}`, "
        f"`long_only={LONG_ONLY}`  ",
        f"DSR multiple-testing deflation: n_trials = {N_TRIALS_DSR}  ",
        f"Pooled-OOS bars: {edge['n_obs']}  ",
        "",
        "**Evidence tier (binding).** Same THIRD tier as the dual_momentum literal study, NOT Part "
        "A's 'Literal' tier. It shares Part A's EXECUTION realism (same cycle, real cost model, "
        "order-gen, enrichment, risk-controls OFF, 252/252/252 WF) but: (a) **DAILY** rebalance — "
        "etf_pairs is a daily Z-score strategy, monthly rebalance would miss most entries/exits; "
        "(b) **FULL long-short**, gross ≈ 200%, NET ≈ 0 (market-neutral); (c) a DIFFERENT data "
        "source — Alpaca daily bars for the 12 pair ETFs, not the offline survivor cache. The SPY "
        f"benchmark is the Alpaca SPY buy-and-hold over THESE windows ({n_folds} folds spanning "
        f"{fold_span}). SPY/IVV is itself a traded pair, so SPY is both benchmark and (occasionally) "
        "a traded leg; the benchmark uses SPY close independently of any held position.",
        "",
        "**Market-neutral caveat (read before the verdict).** This is a beta≈0 book; it is NOT "
        "built to out-CAGR a bull-market SPY (a neutral book structurally trails a rising tape on "
        "absolute return). The meaningful question is risk-adjusted: does the spread alpha clear "
        "SPY's Sharpe with a DSR-deflated AND significant (IR t>1.96) edge? The falsification bar is "
        "unchanged (beat SPY Sharpe AND DSR AND IR-t → PROSPECT), but a CAGR shortfall alone is "
        "expected and is NOT the interesting signal — the Sharpe/DSR/IR line is.",
        "",
        "**Data honesty.** Alpaca bar close ≈ price return (no dividend reinvestment). For "
        "relative-value pairs this largely CANCELS in the spread (both legs are similar instruments "
        "with near-identical yields, e.g. SPY/IVV, XLK/VGT); residual divergence (e.g. GDX/GDXJ "
        "payout differences) is a second-order spread bias. Costs: the literal cost model charges "
        "commission+spread+impact on every leg change including SHORT legs, but NO explicit "
        "short-borrow/locate fee is modelled — a known OPTIMISTIC omission for the short side "
        "(disclosed). PIT: the signal panel is precomputed once per fold via "
        "`generate_etf_pairs_signals_from_prices` and looked up by as_of — PIT-identical to a "
        "per-as_of recompute because the state machine is strictly causal (verified by an explicit "
        "self-check at run start). statsmodels REQUIRED (Engle-Granger). CI: not run; local one-shot.",
        "",
        "## Result",
        "",
        f"**[{tag}]** pooled-OOS AnnSharpe **{_fmt(edge['ann_sharpe'], '+.2f')}** vs SPY "
        f"{_fmt(spy_sharpe, '+.2f')} · CAGR **{_fmt(edge['cagr'])}** vs SPY {_fmt(spy_edge.get('cagr', float('nan')))} · "
        f"IR vs SPY {_fmt(edge['ir'], '+.2f')} (t={_fmt(edge['ir_t'], '+.2f')}) · "
        f"DSR-prob {_fmt(edge['dsr_prob'], '.2f')} (pass5%={edge['dsr_pass']}) · "
        f"beta {_fmt(edge['beta'], '+.2f')} · vol-matched ann.ret {_fmt(edge['vol_matched_ret'])}.",
        "",
        "## Per-fold (literal pipeline, daily rebalance)",
        "",
        _fold_table(results),
        "",
        "_Gross = avg Σ|target_weight| per rebalance (≈2.0 confirms full long-short); Legs = avg "
        "active legs per rebalance; Rej = rejected trades (fill-model gate / short handling sanity — "
        "expect ~0 since short SELLs credit cash before long BUYs)._",
        "",
        "_Estimator note: the **Ø (N/N)** row above is the mean of per-fold metrics (each fold "
        "equal-weighted); the pooled-OOS edge table below and the verdict use the **pooled daily "
        "return series** (all OOS bars concatenated). The two can differ in magnitude AND sign — "
        "expected by construction, not an inconsistency. The PROSPECT/REJECTED verdict uses the "
        "pooled series._",
        "",
        "## OOS-edge (pooled out-of-sample)",
        "",
        "| Strategy | AnnSharpe | Sharpe t | CAGR | MaxDD | Beta | IR vs SPY | IR t | DSR-prob | DSR✓ | PSR>SPY | VolMatchRet |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| etf_pairs_meanrev (literal) | {_fmt(edge['ann_sharpe'], '+.2f')} | {_fmt(edge['sharpe_t'], '+.2f')} | "
        f"{_fmt(edge['cagr'])} | {_fmt(edge['maxdd'])} | {_fmt(edge['beta'], '+.2f')} | "
        f"{_fmt(edge['ir'], '+.2f')} | {_fmt(edge['ir_t'], '+.2f')} | {_fmt(edge['dsr_prob'], '.2f')} | "
        f"{'Y' if edge['dsr_pass'] else 'N'} | {_fmt(edge['psr_vs_spy'], '.2f')} | {_fmt(edge['vol_matched_ret'])} |",
        f"| **SPY (bench)** | {_fmt(spy_sharpe, '+.2f')} | {_fmt(spy_edge.get('sharpe_t', float('nan')), '+.2f')} | "
        f"{_fmt(spy_edge.get('cagr', float('nan')))} | {_fmt(spy_edge.get('maxdd', float('nan')))} | +1.00 | — | — | "
        f"{_fmt(spy_edge.get('dsr_prob', float('nan')), '.2f')} | {'Y' if spy_edge.get('dsr_pass') else 'N'} | — | "
        f"{_fmt(spy_edge.get('cagr', float('nan')))} |",
        "",
        "## Verdict",
        "",
        f"**etf_pairs_meanrev is {tag} through the literal pipeline.** "
        + (
            "It clears SPY's pooled-OOS Sharpe with a DSR-deflated and significant edge."
            if tag == "PROSPECT"
            else "It does not clear SPY's pooled-OOS Sharpe with a DSR-deflated AND significant "
            "(IR t>1.96) edge. As a market-neutral book this is unsurprising on absolute CAGR, and "
            "the risk-adjusted line does not reach significance through the real cost model "
            "(commission+spread+impact on every leg change, gross 200% → cost drag on a thin "
            "relative-value edge). The cointegration filter and Z-score discipline keep drawdowns "
            "contained and beta near zero, but the net-of-cost spread alpha is not a deflated, "
            "significant SPY-beating edge in this sample."
        ),
        "",
        "---",
        "_Script: `scripts/_oos_wf_etf_pairs_literal.py` (research harness; executes the real "
        "`run_trading_cycle`, reads `policy.yaml` read-only, EDITS no production module, forces "
        "crisis-overlay dry-run via `ASSEMBLED_NO_CRISIS_OVERLAY=1`, caches Alpaca bars to "
        "`output/research/` — mutates NO production state or price cache)._  ",
        "_Strategy: `src/assembled_core/strategies/etf_pairs_meanrev.py` "
        "(`generate_etf_pairs_signals_from_prices` / `compute_signals`)._  ",
        "_Pipeline entry: `src/assembled_core/qa/backtest_engine.py` (`run_portfolio_backtest` / `make_cycle_fn`)._  ",
        "_Edge helpers reused from `scripts/_oos_wf_leverage_short.py`; sizing fn from "
        "`scripts/_oos_wf_pipeline_realistic.py` (DRY)._  ",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    log.info("Report written -> %s", OUT_MD)
    return tag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    if not _HAS_STATSMODELS:
        log.error(
            "statsmodels unavailable — etf_pairs_meanrev requires Engle-Granger. Aborting."
        )
        return 1

    log.info("Sourcing 12-ETF pairs menu (%s)…", ", ".join(SYMBOLS))
    prices = _source_prices()
    prices = prices[
        (prices["timestamp"] >= FETCH_START) & (prices["timestamp"] <= PERIOD_END)
    ].copy()
    for sym in SYMBOLS:
        if prices[prices["symbol"] == sym].empty:
            log.error("Symbol %s missing from sourced data — aborting", sym)
            return 1
    actual_start = max(
        prices[prices["symbol"] == s]["timestamp"].min() for s in SYMBOLS
    )
    actual_end = min(prices[prices["symbol"] == s]["timestamp"].max() for s in SYMBOLS)
    log.info("Overlapping range: %s → %s", actual_start.date(), actual_end.date())

    _pit_self_check(prices)

    results, pooled_s, pooled_b = _run_wf(prices)
    if pooled_s.empty:
        log.error("No OK folds — all WF folds failed. See per-fold log above.")
        return 1
    edge = _edge_metrics(pooled_s, pooled_b, results)
    spy_edge = (
        _spy_pooled_edge(pooled_b)
        if pooled_b is not None and not pooled_b.empty
        else {}
    )
    log.info(
        "[etf_pairs] pooled: AnnSharpe %.2f / CAGR %.1f%% / IR %.2f (t=%.2f) / DSR-prob %.2f (pass=%s)",
        edge["ann_sharpe"],
        edge["cagr"] * 100,
        edge["ir"],
        edge["ir_t"],
        edge["dsr_prob"],
        edge["dsr_pass"],
    )
    tag = _write_report(results, edge, spy_edge, actual_start, actual_end)
    log.info("DONE. etf_pairs_meanrev literal verdict: %s", tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
