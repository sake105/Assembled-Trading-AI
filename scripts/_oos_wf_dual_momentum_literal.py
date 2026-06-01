"""dual_momentum through the LITERAL pipeline (`run_trading_cycle`) — Part B closure.

Writes docs/results/2026_05_dual_momentum_literal_oos.md (companion artifact); the
consolidated `2026_05_pipeline_realistic_oos.md` B.4 bullet is updated by hand to
cite this number.

WHY this exists
---------------
The matrix harness `_oos_wf_pipeline_realistic.py` (Part A + B.1) drives 11 research
concepts and `low_max_lottery` through the real production cycle on the 75-symbol
offline survivor cache. `dual_momentum` could NOT be driven there because its
4-asset menu (SPY / VEU / BIL / AGG) is incomplete in that cache — VEU and BIL are
absent (verified: `load_eod_prices(None)` returns 220 symbols incl. SPY+AGG but not
VEU/BIL). So Part B.4 listed it as "not driveable, cited only". This script closes
that gap HONESTLY: it sources the real 4-asset menu from Alpaca and routes the
REGISTERED `dual_momentum` signal through the IDENTICAL literal machinery
(`run_portfolio_backtest(cycle_fn=make_cycle_fn(...))` → `run_trading_cycle` →
feature enrichment → `generate_orders_from_targets` → `simulate_with_costs`).

EVIDENCE TIER (binding honesty)
-------------------------------
This is a THIRD tier, NOT Part A's "Literal" tier:
  - SAME execution realism as Part A (real cost model, order-gen, enrichment, same
    cycle, risk-controls OFF, monthly rebalance, 252/252/252 WF).
  - DIFFERENT data source: Alpaca daily bars for SPY/VEU/BIL/AGG, NOT the offline
    survivor cache. The SPY benchmark here is the Alpaca SPY buy-and-hold over the
    same windows — close to but NOT byte-identical to Part A's offline SPY bench
    (AnnSharpe +0.91 / CAGR +17.4%). So this is comparable to Part A on the
    EXECUTION axis, but its SPY bench is its own series and must be read as such.
  - DIFFERENT fold coverage: the Alpaca menu starts 2016-08 (earlier than the
    offline cache), so the 252/252/252 WF yields 7 folds spanning 2017-08…2024-08
    here, not Part A's 6 folds 2019-2024.

PIT / causality
---------------
The cycle slices `precomputed_prices_with_features` to `timestamp <= as_of` before
signal_fn runs; `dual_momentum.compute_signals` then tags EOM bars causally (a bar
is EOM only once a later-month bar exists) and forward-fills the holding, so the
as_of holding is the one established at the last COMPLETE month-end before as_of —
no look-ahead. The cycle adds its own execution lag on top. This is exactly how the
live paper cycle would invoke dual_momentum.

STATE ISOLATION
---------------
ASSEMBLED_NO_CRISIS_OVERLAY=1 forces the crisis overlay to dry_run (no state
persisted); write_outputs=False, backtest_use_snapshot=False. No production module,
policy.yaml, or live state file is mutated. Alpaca bars are cached to a
research-local parquet (output/research/), NOT the production price cache.

DATA HONESTY (inherited from the standalone dual_momentum study)
----------------------------------------------------------------
Alpaca bar close ≈ price return (no dividend reinvestment): VEU ~3% yield and AGG
~3-4% coupon are NOT captured, so the defensive/ex-US legs are return-UNDERSTATED.
BIL price return ≈ 0 (correct cash-hurdle proxy). Costs: the literal cost model
charges commission+spread+impact on each switch; no short-borrow (long-only, N/A).

Falsification rule (unchanged): dual_momentum beats SPY risk-adjusted (Sharpe > SPY
AND DSR-deflated AND IR t>1.96) through the real pipeline — or it is REJECTED. The
standalone vectorized study already found REJECTED (Ø CAGR 9.7% / Sharpe 0.98 vs
SPY 14.5% / 1.26, 13-fold 2016-2025); the literal pipeline can only confirm/deepen.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# State isolation must precede production imports / any cycle run (see module docstring).
os.environ.setdefault("ASSEMBLED_NO_CRISIS_OVERLAY", "1")

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
    _month_ends,
    sizing_fn,
)
from src.assembled_core.pipeline.trading_cycle_shared import TradingContext  # noqa: E402
from src.assembled_core.qa.backtest_engine import (  # noqa: E402
    BacktestResult,
    make_cycle_fn,
    run_portfolio_backtest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oos_wf_dual_momentum_literal")

# ---------------------------------------------------------------------------
# Config — 252/252/252 WF like Part A, but the Alpaca menu starts 2016-08 (earlier
# than the offline cache) → this run yields 7 folds spanning 2017-08…2024-08, NOT
# Part A's 6 folds 2019-2024.
# ---------------------------------------------------------------------------
SYMBOLS = ["SPY", "VEU", "BIL", "AGG"]
LOOKBACK_MONTHS = 12
# A "fully invested" book must leave a small cash buffer for transaction costs: the
# literal fill model's non-negative-cash gate (`fill_model.apply_cash_gate`) REJECTS a
# BUY whose `notional + cost` would drive cash to <1e-6 — so a 100%-notional order
# (weight 1.0, notional == capital) is structurally un-fillable and the single-asset
# position never establishes (verified: fold-4 BUY rejected → realized -8.2% vs SPY
# +30.6%, an artifact). 0.98 = 2% cash buffer → order fills, position tracks the held
# asset (fold-4 then +20.7%). This is also more realistic than a literal 100% deploy.
TARGET_WEIGHT = 0.98
WARMUP_BARS = 312  # ~15 months: guarantees the 12M lookback is hot at each test_start
# Bound the data so the 252-step WF yields the same 6 test windows as Part A
# (test 2019-01..2020-01 … 2024-01..2025-01). End at 2025-02 so the last fold's
# 252-bar test window closes ~2025-01.
FETCH_START = pd.Timestamp("2016-06-01", tz="UTC")  # ≥ 312-bar warmup before 2018-01
PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-02-28", tz="UTC")

CACHE_PARQUET = ROOT / "output" / "research" / "dual_momentum_literal_prices.parquet"
OUT_MD = ROOT / "docs" / "results" / "2026_05_dual_momentum_literal_oos.md"


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
# Signal: registered dual_momentum holding at as_of (live contract)
# ---------------------------------------------------------------------------
def make_dm_signal_fn():
    """signal_fn(features) -> single-row [timestamp, symbol, direction, score, weight].

    Calls the REGISTERED dual_momentum.compute_signals on the (already as_of-sliced)
    feature frame, returning the current holding at weight TARGET_WEIGHT (0.98). Emits
    `weight` so the shared sizing_fn produces target_qty = TARGET_WEIGHT * capital. The
    2% cash buffer is REQUIRED: a 100%-notional order is rejected by the fill model's
    non-negative-cash gate (see TARGET_WEIGHT comment above), which would silently
    prevent the single-asset position from ever establishing.
    """
    from src.assembled_core.strategies.dual_momentum import compute_signals

    cols = ["timestamp", "symbol", "direction", "score", "weight"]

    def signal_fn(features: pd.DataFrame) -> pd.DataFrame:
        empty = pd.DataFrame(columns=cols)
        if features is None or features.empty or "close" not in features.columns:
            return empty
        df = features[["timestamp", "symbol", "close"]].dropna()
        if df.empty:
            return empty
        sig = compute_signals(df, lookback_months=LOOKBACK_MONTHS)
        if sig is None or sig.empty:
            return empty
        row = sig.iloc[-1]
        return pd.DataFrame(
            [(row["timestamp"], str(row["symbol"]), "LONG", 1.0, TARGET_WEIGHT)],
            columns=cols,
        )

    return signal_fn


# ---------------------------------------------------------------------------
# One WF fold through run_trading_cycle
# ---------------------------------------------------------------------------
def _simulate_dm_pipeline(prices, test_start, test_end):
    spy_dates = np.sort(prices[prices["symbol"] == "SPY"]["timestamp"].unique())
    pre_test = spy_dates[spy_dates < test_start]
    warmup_start = (
        pd.Timestamp(pre_test[-WARMUP_BARS])
        if len(pre_test) >= WARMUP_BARS
        else (pd.Timestamp(pre_test[0]) if len(pre_test) > 0 else test_start)
    )
    if warmup_start.tzinfo is None:
        warmup_start = warmup_start.tz_localize("UTC")

    sl = prices[
        prices["symbol"].isin(SYMBOLS)
        & (prices["timestamp"] >= warmup_start)
        & (prices["timestamp"] <= test_end)
    ].copy()
    sl = sl.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    all_me = _month_ends(sl["timestamp"])
    pre = [t for t in all_me if t < test_start]
    rebs = ([pre[-1]] if pre else []) + [
        t for t in all_me if test_start <= t <= test_end
    ]
    rebs = sorted(set(pd.Timestamp(t) for t in rebs))
    if not rebs:
        raise ValueError(f"No rebalance dates for fold {test_start.date()}")

    sig_fn = make_dm_signal_fn()
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
        rebalance_schedule="monthly",
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

    # n_switches: distinct holdings across rebalances (diagnostic)
    n_switch = float("nan")
    tgts = result.target_positions
    if (
        tgts is not None
        and not tgts.empty
        and "symbol" in tgts.columns
        and "timestamp" in tgts.columns
    ):
        tg = tgts.copy()
        tg["timestamp"] = pd.to_datetime(tg["timestamp"], utc=True)
        held = tg.sort_values("timestamp").groupby("timestamp")["symbol"].first()
        n_switch = float((held != held.shift(1)).sum())

    diag = dict(beta_mkt=beta_mkt, n_switch=n_switch)
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
            m, diag, net_ret, spy_test = _simulate_dm_pipeline(
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
                n_switch=diag["n_switch"],
                n_bars=len(net_ret),
                status="OK",
            )
            pooled_strat.append(net_ret)
            pooled_spy.append(spy_test.reindex(net_ret.index))
            log.info(
                "[dual_momentum] Fold %d %s-%s: CAGR %.1f%% / Sharpe %.2f / MaxDD %.1f%% / "
                "beta %.2f / switches %.0f  (SPY: %.1f%% / %.2f)  [%.1fs]",
                fold_idx,
                test_start.date(),
                test_end.date(),
                m["cagr"] * 100,
                m["sharpe"],
                m["maxdd"] * 100,
                diag["beta_mkt"],
                diag["n_switch"],
                bm_spy["cagr"] * 100,
                bm_spy["sharpe"],
                time.time() - t0,
            )
        except Exception as exc:
            log.warning(
                "[dual_momentum] Fold %d FAILED: %s [%.1fs]",
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
        "| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Switches | SPY CAGR | SPY Sharpe |",
        "|------|-------------|------|--------|-------|------|----------|----------|------------|",
    ]
    for r in results:
        if r.get("status") != "OK":
            lines.append(
                f"| {r['fold']} | {r['test_start']}–{r['test_end']} | FAILED | | | | | | |"
            )
            continue
        lines.append(
            f"| {r['fold']} | {r['test_start']}–{r['test_end']} | {_fmt(r['cagr'])} | "
            f"{r['sharpe']:+.2f} | {_fmt(r['maxdd'])} | {r['beta_mkt']:+.2f} | "
            f"{(r['n_switch'] if np.isfinite(r['n_switch']) else float('nan')):.0f} | "
            f"{_fmt(r['spy_cagr'])} | {r['spy_sharpe']:+.2f} |"
        )
    ok = [r for r in results if r.get("status") == "OK"]
    if ok:
        lines.append(
            f"| **Ø ({len(ok)}/{len(results)})** | — | **{_fmt(_avg(results, 'cagr'))}** | "
            f"**{_avg(results, 'sharpe'):+.2f}** | **{_fmt(_avg(results, 'maxdd'))}** | "
            f"**{_avg(results, 'beta_mkt'):+.2f}** | **{_avg(results, 'n_switch'):.0f}** | "
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

    parts = [
        "# dual_momentum through the LITERAL pipeline — OOS Walk-Forward",
        "",
        "Run date (UTC): 2026-05-31  ",
        f"Data: Alpaca daily bars (split-adjusted) for {', '.join(SYMBOLS)} — research-local cache  ",
        f"Overlapping range: {actual_start.date()} → {actual_end.date()}  ",
        "WF: 252/252/252 (train/test/step), monthly rebalance, single asset @ "
        f"{TARGET_WEIGHT:.0%} (2% cash buffer — see methodology note)  ",
        "Execution: LITERAL `run_portfolio_backtest(cycle_fn=make_cycle_fn(...))` → `run_trading_cycle` "
        "→ feature enrichment → `generate_orders_from_targets` → `simulate_with_costs`  ",
        f"Config: `enable_risk_controls={ENABLE_RISK_CONTROLS}`, `include_costs={INCLUDE_COSTS}`  ",
        f"DSR multiple-testing deflation: n_trials = {N_TRIALS_DSR}  ",
        f"Pooled-OOS bars: {edge['n_obs']}  ",
        "",
        "**Evidence tier (binding).** This is NOT Part A's 'Literal' tier. It shares Part A's "
        "EXECUTION realism (same cycle, real cost model, order-gen, enrichment, risk-controls OFF, "
        "monthly rebalance, 252/252/252 WF) but runs on a DIFFERENT data source: "
        "Alpaca SPY/VEU/BIL/AGG, not the 75-symbol offline survivor cache (which lacks VEU & BIL — "
        "verified: `load_eod_prices(None)` returns 220 symbols incl. SPY+AGG but not VEU/BIL). The "
        "FOLD COVERAGE also differs from Part A: the Alpaca menu starts earlier (2016-08) than the "
        f"offline cache, so this WF yields {n_folds} folds spanning {fold_span}, NOT Part A's 6 folds "
        "2019-2024. The SPY benchmark below is the Alpaca SPY buy-and-hold over THESE windows — close "
        "to but not byte-identical to Part A's offline SPY bench (AnnSharpe +0.91 / CAGR +17.4%). "
        "Read this as comparable to Part A on the EXECUTION axis only, with its own folds and SPY "
        "series.",
        "",
        "**Methodology note — cash buffer (material).** The book deploys "
        f"{TARGET_WEIGHT:.0%} of capital into the held asset, not 100%. This is REQUIRED, not "
        "cosmetic: the literal fill model's non-negative-cash gate "
        "(`execution/fill_model.apply_cash_gate`) rejects any BUY whose `notional + cost` would "
        "drive cash below ~0, so a 100%-notional single-asset order (`weight 1.0`, "
        "`notional == capital`) is structurally un-fillable — the position would never establish "
        "and the equity path would be a phantom (verified: at weight 1.0, fold-4 realized -8.2% "
        "with the establishing BUY rejected, while SPY did +30.6%; at weight 0.98 the same fold "
        "realizes +20.7% with zero rejected trades). A real fully-invested book likewise must hold "
        "a small cash reserve for costs/slippage, so 0.98 is the honest deploy. (Implication for "
        "Part A: any research book that is gross-100% invested loses its last-alphabetical "
        "position to the same gate each rebalance — diluted across many names, but a known small "
        "drag; single-asset dual_momentum merely exposes it in full.)",
        "",
        "**Data honesty.** Alpaca bar close ≈ price return (no dividend reinvestment): VEU ~3% yield "
        "and AGG ~3-4% coupon are NOT captured → the defensive/ex-US legs are return-UNDERSTATED. BIL "
        "price return ≈ 0 (correct cash-hurdle proxy). Long-only single-asset rotation → no "
        "short-borrow; the literal cost model charges commission+spread+impact on each switch. "
        "PIT: the cycle slices features to ≤ as_of; `dual_momentum.compute_signals` tags EOM bars "
        "causally and forward-fills the holding, so no look-ahead. CI: not run; local one-shot.",
        "",
        "## Result",
        "",
        f"**[{tag}]** pooled-OOS AnnSharpe **{_fmt(edge['ann_sharpe'], '+.2f')}** vs SPY "
        f"{_fmt(spy_sharpe, '+.2f')} · CAGR **{_fmt(edge['cagr'])}** vs SPY {_fmt(spy_edge.get('cagr', float('nan')))} · "
        f"IR vs SPY {_fmt(edge['ir'], '+.2f')} (t={_fmt(edge['ir_t'], '+.2f')}) · "
        f"DSR-prob {_fmt(edge['dsr_prob'], '.2f')} (pass5%={edge['dsr_pass']}) · "
        f"beta {_fmt(edge['beta'], '+.2f')} · vol-matched ann.ret {_fmt(edge['vol_matched_ret'])}.",
        "",
        "## Per-fold (literal pipeline)",
        "",
        _fold_table(results),
        "",
        "## OOS-edge (pooled out-of-sample)",
        "",
        "| Strategy | AnnSharpe | Sharpe t | CAGR | MaxDD | Beta | IR vs SPY | IR t | DSR-prob | DSR✓ | PSR>SPY | VolMatchRet |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
        f"| dual_momentum (literal) | {_fmt(edge['ann_sharpe'], '+.2f')} | {_fmt(edge['sharpe_t'], '+.2f')} | "
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
        f"**dual_momentum is {tag} through the literal pipeline.** "
        + (
            "It clears SPY's pooled-OOS Sharpe with a DSR-deflated and significant edge."
            if tag == "PROSPECT"
            else "It does not clear SPY's pooled-OOS Sharpe with a DSR-deflated AND significant "
            "(IR t>1.96) edge. Consistent with the standalone vectorized study "
            "(`docs/results/2026_05_dual_momentum_real_oos.md`: 13-fold 2016-2025 Ø CAGR 9.7% / "
            "Sharpe 0.98 vs SPY 14.5% / 1.26) — the absolute-momentum trend filter cuts drawdowns "
            "but the defensive switches drag absolute return below buy-and-hold SPY in this "
            "bull-dominated sample, and the risk-adjusted edge is not significant."
        ),
        "",
        "---",
        "_Script: `scripts/_oos_wf_dual_momentum_literal.py` (research harness; executes the real "
        "`run_trading_cycle`, reads `policy.yaml` read-only, EDITS no production module, forces "
        "crisis-overlay dry-run via `ASSEMBLED_NO_CRISIS_OVERLAY=1`, caches Alpaca bars to "
        "`output/research/` — mutates NO production state or price cache)._  ",
        "_Strategy: `src/assembled_core/strategies/dual_momentum.py` (`compute_signals`)._  ",
        "_Pipeline entry: `src/assembled_core/qa/backtest_engine.py` (`run_portfolio_backtest` / `make_cycle_fn`)._  ",
        "_Edge helpers reused from `scripts/_oos_wf_leverage_short.py` (DRY)._  ",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    log.info("Report written -> %s", OUT_MD)
    return tag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    log.info("Sourcing 4-asset menu (SPY/VEU/BIL/AGG)…")
    prices = _source_prices()
    prices = prices[
        (
            prices["timestamp"]
            >= PERIOD_START - pd.Timedelta(days=int(WARMUP_BARS * 1.6))
        )
        & (prices["timestamp"] <= PERIOD_END)
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

    results, pooled_s, pooled_b = _run_wf(prices)
    edge = _edge_metrics(pooled_s, pooled_b, results)
    spy_edge = (
        _spy_pooled_edge(pooled_b)
        if pooled_b is not None and not pooled_b.empty
        else {}
    )
    log.info(
        "[dual_momentum] pooled: AnnSharpe %.2f / CAGR %.1f%% / IR %.2f (t=%.2f) / DSR-prob %.2f (pass=%s)",
        edge["ann_sharpe"],
        edge["cagr"] * 100,
        edge["ir"],
        edge["ir_t"],
        edge["dsr_prob"],
        edge["dsr_pass"],
    )
    tag = _write_report(results, edge, spy_edge, actual_start, actual_end)
    log.info("DONE. dual_momentum literal verdict: %s", tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
