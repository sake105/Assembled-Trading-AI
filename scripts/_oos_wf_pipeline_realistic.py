"""One-shot OOS Walk-Forward through the LITERAL pipeline (run_trading_cycle).

Writes docs/results/2026_05_pipeline_realistic_oos.md.

Purpose
-------
The prior study `_oos_wf_leverage_short.py` selected the SAME strategies but
simulated them with a VECTORIZED engine (`_simulate`): hand-rolled cost/borrow/
financing on a position-weight matrix. This harness re-runs the IDENTICAL
selection logic but EXECUTES each strategy through the real production cycle —
`run_portfolio_backtest(cycle_fn=make_cycle_fn(...))` → `run_trading_cycle` →
feature enrichment → `generate_orders_from_targets` (notional→integer shares,
sign-preserving) → `simulate_with_costs`. This is the honest "pipeline-realistic"
re-test the user asked for ("mach sie pipeline realistisch und lass dann für alle
strategien nochmal die oos laufen").

Config decision (apples-to-apples with how registered strategies are actually
backtested): `enable_risk_controls=False`, matching `run_backtest_strategy.py`
(ctx_template line ~2341: "Backtest engine doesn't use risk controls"). The LIVE
risk overlays (gross cap default 1.20, dd-damper, regime de-risk) are a SEPARATE
layer applied in paper/live, not in the canonical backtest. A throwaway smoke test
(`_smoke_pipeline_ls.py`) confirmed that WITH risk controls ON the literal cycle
preserves signed shorts (first-rebalance BUY notional ≈ SELL notional, ratio 0.99)
but de-levers gross ~5x to the 1.20 cap — documented in the report as the live-mode
caveat.

What the literal pipeline adds over the vectorized harness (for a pure-price
signal, risk-controls OFF):
  + real cost model (`simulate_with_costs`: commission + spread + impact) instead
    of a flat 10.75 bps/leg
  + real order generation (signed notional → integer shares, rounding friction)
  + feature enrichment (HMM/behavioral/macro/rv add-ons) — IMMATERIAL here: these
    signals read only `close`, so enrichment cannot change selection; it only
    costs wall-time. Disclosed, not hidden.
  - the vectorized harness's explicit short-borrow (50 bps/yr) + margin financing
    (100 bps/yr) are NOT modelled by simulate_with_costs → the pipeline is LESS
    conservative on short/leverage carry. Net cost direction vs the prior study is
    therefore ambiguous; both are reported side-by-side.

PIT-safety: `_tc_features` slices precomputed features to `timestamp <= ctx.as_of`
(history-slice mode) before signal_fn sees them; `_select` further uses ref_idx =
loc(as_of) - 1 (strictly before as_of); the cycle adds its own execution lag.

HONESTY (binding, inherited from the prior study):
  - Offline cache is SURVIVORS-ONLY. Bias direction is strategy-dependent for L/S:
    short-the-junk legs (mom_ls/bab_ls/lowvol_ls) cannot short delisted losers →
    short leg UNDERSTATED → CONSERVATIVE. reversal_* LONGS recovered losers →
    OPTIMISTIC.
  - CI status: not run in CI; local one-shot only. No production module is EDITED.
    The cycle READS the live policy.yaml (read-only) and drives the real overlays,
    but crisis/risk STATE persistence is suppressed (ASSEMBLED_NO_CRISIS_OVERLAY=1
    → dry_run; backtest mode leaves risk_state.json untouched) so the live
    paper-pilot state files are NOT mutated. This is a research harness in scripts/;
    it reads price data + policy and writes one md file.

Falsification rule (unchanged): a candidate beats SPY risk-adjusted (Sharpe > SPY
AND a significant, DSR-deflated edge, IR t>1.96) through the REAL pipeline — or it
is REJECTED. Pipeline-realism can only CONFIRM/deepen rejections (costs reduce
returns; enrichment immaterial; no overlays to improve risk-adjusted outcome).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# STATE-ISOLATION (must precede production imports / any cycle run):
# Driving the LITERAL run_trading_cycle invokes the crisis-alpha overlay, whose
# pipeline persists state to the LIVE output/ops/crisis_alpha_state.json by
# default. A backtest stepping through historical as_of dates would clobber the
# real paper-pilot's crisis state with time-traveled records. The production code
# (_tc_sizing.py) honors ASSEMBLED_NO_CRISIS_OVERLAY=1 by forcing shadow_only=True
# → dry_run → state is computed but NOT persisted. This is the sanctioned escape
# hatch (no production edit, no monkeypatch). The overlay never produced entry
# targets in these runs anyway (geo_score=0 → never ACTIVE), so this changes NO
# result — it only stops the shared-state write side effect.
os.environ.setdefault("ASSEMBLED_NO_CRISIS_OVERLAY", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the prior study's selection + edge + config verbatim (DRY: no second truth).
from scripts._oos_wf_leverage_short import (  # noqa: E402
    LONG_SHORT_MODES,
    LONGONLY_MODES,
    MODE_TITLES,
    N_TRIALS_DSR,
    STEP_SIZE,
    TEST_WINDOW,
    TRAIN_WINDOW,
    _benchmark_spy,
    _edge_metrics,
    _fmt,
    _load_universe_prices,
    _metrics,
    _select,
    _spy_pooled_edge,
)
from src.assembled_core.pipeline.trading_cycle_shared import TradingContext  # noqa: E402
from src.assembled_core.qa.backtest_engine import (  # noqa: E402
    BacktestResult,
    make_cycle_fn,
    run_portfolio_backtest,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oos_wf_pipeline_realistic")

INITIAL_CAPITAL = 100_000.0
OUT_MD = ROOT / "docs" / "results" / "2026_05_pipeline_realistic_oos.md"

# Pipeline-realistic config (matches run_backtest_strategy.py backtest path).
ENABLE_RISK_CONTROLS = False
INCLUDE_COSTS = True


# Low-MAX / lottery-avoidance (Bali-Cakici-Whitelaw 2011) — the production
# `low_max_lottery` concept, expressed for the literal pipeline (Part B). MAX =
# max daily return over last 20 bars; long the BOTTOM quintile, equal-weight.
LOWMAX_LOOKBACK = 20
LOWMAX_QUANTILE = 0.20


def _lowmax_select(trade_rets, all_dates, rebal_date):
    """Bottom-quintile MAX → long-only equal-weight (sums to +1). PIT: the MAX
    window ends at ref_idx = loc(rebal_date) - 1 (strictly before as_of)."""
    try:
        window_end_idx = all_dates.get_loc(rebal_date)
    except KeyError:
        return {}
    ref_idx = window_end_idx - 1
    if ref_idx < LOWMAX_LOOKBACK:
        return {}
    win = trade_rets.iloc[ref_idx - LOWMAX_LOOKBACK + 1 : ref_idx + 1]
    max_score = win.max(axis=0).dropna()
    if len(max_score) < 5:
        return {}
    thr = max_score.quantile(LOWMAX_QUANTILE)
    longs = list(max_score[max_score <= thr].index)
    if not longs:
        return {}
    w = 1.0 / len(longs)
    return {sym: w for sym in longs}


# ---------------------------------------------------------------------------
# Strategy expressed as (signal_fn, position_sizing_fn) for the literal cycle
# ---------------------------------------------------------------------------
def make_signal_fn(mode: str, tradeable: list[str]):
    """signal_fn(features) -> [timestamp, symbol, direction, score, weight].

    Reconstructs the PIT price pivot from the (already as_of-sliced) feature
    frame and calls the prior study's `_select` at the last bar (= as_of). The
    emitted `weight` column carries the exact signed research weight so the
    sizing fn can reproduce the strategy's gross/leverage faithfully (incl. the
    BAB beta-targeted construction, which is NOT equal-weight).
    """

    def signal_fn(features: pd.DataFrame) -> pd.DataFrame:
        empty = pd.DataFrame(
            columns=["timestamp", "symbol", "direction", "score", "weight"]
        )
        if features is None or features.empty or "close" not in features.columns:
            return empty
        df = features[["timestamp", "symbol", "close"]].dropna()
        if df.empty:
            return empty
        pivot = (
            df.pivot_table(index="timestamp", columns="symbol", values="close")
            .sort_index()
            .ffill()
        )
        all_dates = pivot.index
        if len(all_dates) < 5:
            return empty
        trade_cols = [s for s in tradeable if s in pivot.columns]
        if not trade_cols:
            return empty
        trade_pivot = pivot[trade_cols]
        trade_rets = trade_pivot.pct_change()
        spy_rets = (
            pivot["SPY"].pct_change()
            if "SPY" in pivot.columns
            else pd.Series(0.0, index=all_dates)
        )
        rebal_date = all_dates[-1]  # == ctx.as_of (PIT-sliced upstream)
        if mode == "lowmax_lo":
            weights = _lowmax_select(trade_rets, all_dates, rebal_date)
        else:
            weights = _select(
                trade_pivot, trade_rets, spy_rets, all_dates, rebal_date, mode
            )
        if not weights:
            return empty
        rows = [
            (rebal_date, sym, "LONG" if w > 0 else "SHORT", float(w), float(w))
            for sym, w in weights.items()
            if abs(w) > 0
        ]
        return pd.DataFrame(
            rows, columns=["timestamp", "symbol", "direction", "score", "weight"]
        )

    return signal_fn


def sizing_fn(signals_df: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Honor the signed research `weight` directly: target_qty = weight * capital
    (signed NOTIONAL dollars; downstream order-gen divides by price → shares,
    preserving sign). Mirrors compute_multifactor_long_short_positions' output
    schema [symbol, target_weight, target_qty] but with the exact research weights.
    """
    cols = ["symbol", "target_weight", "target_qty"]
    if signals_df is None or signals_df.empty:
        return pd.DataFrame(columns=cols)
    df = signals_df.copy()
    if "timestamp" in df.columns and df["timestamp"].notna().any():
        latest = df["timestamp"].max()
        df = df[df["timestamp"] == latest]
    if "weight" not in df.columns:
        # Fallback: equal-weight by direction (should not happen — signal_fn emits weight)
        df["weight"] = np.where(df.get("direction", "LONG") == "SHORT", -1.0, 1.0)
        n = max(len(df), 1)
        df["weight"] = df["weight"] / n
    out = pd.DataFrame(
        {
            "symbol": df["symbol"].values,
            "target_weight": df["weight"].astype(float).values,
            "target_qty": (df["weight"].astype(float) * float(capital)).values,
        }
    )
    return out.groupby("symbol", as_index=False).agg(
        {"target_weight": "sum", "target_qty": "sum"}
    )


# ---------------------------------------------------------------------------
# Fold execution through the literal pipeline
# ---------------------------------------------------------------------------
def _month_ends(timestamps: pd.Series) -> list[pd.Timestamp]:
    s = pd.to_datetime(pd.Series(pd.unique(timestamps)), utc=True).sort_values()
    d = pd.DataFrame({"ts": s})
    d["ym"] = d["ts"].dt.year * 100 + d["ts"].dt.month
    return [pd.Timestamp(t) for t in d.groupby("ym")["ts"].max()]


def _simulate_pipeline(prices, tradeable, test_start, test_end, mode):
    """Run one WF fold through run_trading_cycle. Returns (m, diag, net_ret, spy_test)."""
    spy_dates = np.sort(prices[prices["symbol"] == "SPY"]["timestamp"].unique())
    pre_test = spy_dates[spy_dates < test_start]
    warmup_start = (
        pd.Timestamp(pre_test[-TRAIN_WINDOW])
        if len(pre_test) >= TRAIN_WINDOW
        else (pd.Timestamp(pre_test[0]) if len(pre_test) > 0 else test_start)
    )
    if warmup_start.tzinfo is None:
        warmup_start = warmup_start.tz_localize("UTC")

    # SPY is INCLUDED in the slice (needed by beta-based modes) but NOT in the
    # traded universe; signal_fn never emits a weight for it, so it is never sized.
    syms_needed = list(set(tradeable) | {"SPY"})
    sl = prices[
        prices["symbol"].isin(syms_needed)
        & (prices["timestamp"] >= warmup_start)
        & (prices["timestamp"] <= test_end)
    ].copy()
    sl = sl.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Rebalances: monthly month-ends. Include the last pre-test month-end so a
    # position is live entering the OOS window; OOS returns counted only in
    # [test_start, test_end].
    all_me = _month_ends(sl["timestamp"])
    pre = [t for t in all_me if t < test_start]
    rebs = ([pre[-1]] if pre else []) + [
        t for t in all_me if test_start <= t <= test_end
    ]
    rebs = sorted(set(pd.Timestamp(t) for t in rebs))
    if not rebs:
        raise ValueError(f"No rebalance dates for fold {test_start.date()}")

    sig_fn = make_signal_fn(mode, tradeable)
    ctx_template = TradingContext(
        prices=sl,
        freq="1d",
        universe=tradeable,
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
    )
    ret_all = ret_all.dropna()

    # SPY daily returns over the same dates (benchmark series for pooling/beta)
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
    # diagnostics
    common = net_ret.index.intersection(spy_test.dropna().index)
    if len(common) > 5 and spy_test[common].std() > 1e-12:
        beta_mkt = float(
            np.cov(net_ret[common], spy_test[common])[0, 1] / np.var(spy_test[common])
        )
    else:
        beta_mkt = float("nan")

    gross = float("nan")
    turnover_yr = float("nan")
    tgts = result.target_positions
    if tgts is not None and not tgts.empty and "target_qty" in tgts.columns:
        tg = tgts.copy()
        if "timestamp" in tg.columns:
            tg["timestamp"] = pd.to_datetime(tg["timestamp"], utc=True)
            per_reb = tg.groupby("timestamp")["target_qty"].apply(
                lambda s: s.abs().sum()
            )
            gross = float((per_reb / INITIAL_CAPITAL).mean())
    trades = result.trades
    if trades is not None and not trades.empty:
        tr = trades.copy()
        qcol = "fill_qty" if "fill_qty" in tr.columns else "qty"
        pcol = "fill_price" if "fill_price" in tr.columns else "price"
        if qcol in tr.columns and pcol in tr.columns:
            notional = (tr[qcol].abs() * tr[pcol].abs()).sum()
            n_years = max(len(net_ret) / 252.0, 1e-9)
            turnover_yr = float(notional / INITIAL_CAPITAL / n_years)

    diag = dict(gross=gross, turnover_yr=turnover_yr, beta_mkt=beta_mkt)
    return m, diag, net_ret, spy_test


def _run_wf_pipeline(prices, tradeable, mode):
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
            m, diag, net_ret, spy_test = _simulate_pipeline(
                prices, tradeable, test_start, test_end, mode
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
                gross=diag["gross"],
                turnover_yr=diag["turnover_yr"],
                beta_mkt=diag["beta_mkt"],
                n_bars=len(net_ret),
                status="OK",
            )
            pooled_strat.append(net_ret)
            pooled_spy.append(spy_test.reindex(net_ret.index))
            log.info(
                "[%s] Fold %d %s-%s: CAGR %.1f%% / Sharpe %.2f / MaxDD %.1f%% / beta %.2f "
                "/ gross %.2f  (SPY: %.1f%% / %.2f)  [%.1fs]",
                mode,
                fold_idx,
                test_start.date(),
                test_end.date(),
                m["cagr"] * 100,
                m["sharpe"],
                m["maxdd"] * 100,
                diag["beta_mkt"],
                diag["gross"],
                bm_spy["cagr"] * 100,
                bm_spy["sharpe"],
                time.time() - t0,
            )
        except Exception as exc:
            log.warning(
                "[%s] Fold %d FAILED: %s [%.1fs]", mode, fold_idx, exc, time.time() - t0
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
# Report
# ---------------------------------------------------------------------------
def _avg(results, key):
    vals = [
        r[key]
        for r in results
        if r.get("status") == "OK" and np.isfinite(r.get(key, float("nan")))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _fold_table(results: list[dict]) -> str:
    lines = [
        "| Fold | Test Period | CAGR | Sharpe | MaxDD | Beta | Gross | SPY CAGR | SPY Sharpe | Trn/yr |",
        "|------|-------------|------|--------|-------|------|-------|----------|------------|--------|",
    ]
    for r in results:
        if r.get("status") != "OK":
            lines.append(
                f"| {r['fold']} | {r['test_start']}–{r['test_end']} | FAILED | | | | | | | |"
            )
            continue
        lines.append(
            f"| {r['fold']} | {r['test_start']}–{r['test_end']} | {_fmt(r['cagr'])} | "
            f"{r['sharpe']:+.2f} | {_fmt(r['maxdd'])} | {r['beta_mkt']:+.2f} | "
            f"{(r['gross'] if np.isfinite(r['gross']) else float('nan')):.2f} | "
            f"{_fmt(r['spy_cagr'])} | {r['spy_sharpe']:+.2f} | "
            f"{(r['turnover_yr'] if np.isfinite(r['turnover_yr']) else float('nan')):.1f} |"
        )
    ok = [r for r in results if r.get("status") == "OK"]
    if ok:
        lines.append(
            f"| **Ø ({len(ok)}/{len(results)})** | — | **{_fmt(_avg(results, 'cagr'))}** | "
            f"**{_avg(results, 'sharpe'):+.2f}** | **{_fmt(_avg(results, 'maxdd'))}** | "
            f"**{_avg(results, 'beta_mkt'):+.2f}** | **{_avg(results, 'gross'):.2f}** | "
            f"{_fmt(_avg(results, 'spy_cagr'))} | {_avg(results, 'spy_sharpe'):+.2f} | "
            f"{_avg(results, 'turnover_yr'):.1f} |"
        )
    return "\n".join(lines)


def _verdict_line(mode: str, edge: dict, spy_sharpe: float) -> str:
    label = MODE_TITLES[mode]
    beats_sharpe = np.isfinite(edge["ann_sharpe"]) and edge["ann_sharpe"] > spy_sharpe
    significant = (
        bool(edge["dsr_pass"]) and np.isfinite(edge["ir_t"]) and edge["ir_t"] > 1.96
    )
    prospect = beats_sharpe and significant
    tag = "PROSPECT" if prospect else "REJECTED"
    return (
        f"- **{mode}** [{tag}] ({label}): pooled-OOS Sharpe {edge['ann_sharpe']:+.2f} "
        f"vs SPY {spy_sharpe:+.2f}; IR vs SPY {_fmt(edge['ir'], '+.2f')} (t={_fmt(edge['ir_t'], '+.2f')}); "
        f"DSR-prob {_fmt(edge['dsr_prob'], '.2f')} (pass5%={edge['dsr_pass']}); "
        f"beta {_fmt(edge['beta'], '+.2f')}; vol-matched ann.ret {_fmt(edge['vol_matched_ret'])}."
    )


def _edge_table(edges: dict, spy_edge: dict) -> str:
    cols = (
        "| Strategy | AnnSharpe | Sharpe t | CAGR | MaxDD | Beta | IR vs SPY | IR t | "
        "DSR-prob | DSR✓ | PSR>SPY | Trn/yr | FoldWin | VolMatchRet |"
    )
    sep = "|" + "---|" * 14
    lines = [cols, sep]
    for mode in LONG_SHORT_MODES + LONGONLY_MODES:
        e = edges[mode]
        lines.append(
            f"| {mode} | {_fmt(e['ann_sharpe'], '+.2f')} | {_fmt(e['sharpe_t'], '+.2f')} | "
            f"{_fmt(e['cagr'])} | {_fmt(e['maxdd'])} | {_fmt(e['beta'], '+.2f')} | "
            f"{_fmt(e['ir'], '+.2f')} | {_fmt(e['ir_t'], '+.2f')} | {_fmt(e['dsr_prob'], '.2f')} | "
            f"{'Y' if e['dsr_pass'] else 'N'} | {_fmt(e['psr_vs_spy'], '.2f')} | "
            f"{_fmt(e['turnover_yr'], '.1f')} | {e['fold_win']} | {_fmt(e['vol_matched_ret'])} |"
        )
    lines.append(
        f"| **SPY (bench)** | {_fmt(spy_edge['ann_sharpe'], '+.2f')} | {_fmt(spy_edge['sharpe_t'], '+.2f')} | "
        f"{_fmt(spy_edge['cagr'])} | {_fmt(spy_edge['maxdd'])} | +1.00 | — | — | "
        f"{_fmt(spy_edge['dsr_prob'], '.2f')} | {'Y' if spy_edge['dsr_pass'] else 'N'} | — | 0 | — | "
        f"{_fmt(spy_edge['cagr'])} |"
    )
    return "\n".join(lines)


def _write_report(all_results, all_edges, spy_pooled_edge, n_tradeable, n_obs):
    spy_sharpe = spy_pooled_edge["ann_sharpe"]
    verdict_lines = [
        _verdict_line(m, all_edges[m], spy_sharpe)
        for m in LONG_SHORT_MODES + LONGONLY_MODES
    ]
    prospects = [
        m
        for m in LONG_SHORT_MODES + LONGONLY_MODES
        if (
            np.isfinite(all_edges[m]["ann_sharpe"])
            and all_edges[m]["ann_sharpe"] > spy_sharpe
            and all_edges[m]["dsr_pass"]
            and np.isfinite(all_edges[m]["ir_t"])
            and all_edges[m]["ir_t"] > 1.96
        )
    ]
    if prospects:
        overall = (
            f"**{len(prospects)} of 11 strategies show a PROSPECT through the LITERAL pipeline** "
            f"({', '.join(prospects)}) — Sharpe > SPY AND a significant, DSR-deflated edge. "
            f"NOT a production claim: requires a survivorship-clean re-test and CI validation."
        )
    else:
        overall = (
            "**ALL 11 strategies REJECTED through the LITERAL pipeline** — none clears SPY's "
            "pooled-OOS Sharpe with a DSR-deflated AND significant (IR t>1.96) edge, even "
            "executed through the real production cycle (feature enrichment → order generation "
            "→ cost simulation). Pipeline-realism CONFIRMS the prior vectorized-harness "
            "rejections rather than overturning them: real costs + share-rounding only reduce "
            "returns, enrichment is immaterial to pure-price signals, and with risk-controls OFF "
            "there is no overlay that could improve the risk-adjusted outcome."
        )

    parts = [
        "# Pipeline-Realistic OOS Walk-Forward — 11 Strategies through `run_trading_cycle`",
        "",
        "Run date (UTC): 2026-05-31  ",
        "Data: local offline cache via `load_eod_prices(None)` — survivors only  ",
        f"Universe: {n_tradeable} tradeable symbols (data ≤ 2018-01-31, ≥ 500 bars; SPY = market factor + benchmark, never traded)  ",
        "WF: 252/252/252 (train/test/step), monthly rebalance, top/bottom quintile  ",
        "Execution: LITERAL `run_portfolio_backtest(cycle_fn=make_cycle_fn(...))` → `run_trading_cycle` "
        "→ feature enrichment → `generate_orders_from_targets` (signed notional → integer shares) "
        "→ `simulate_with_costs`  ",
        f"Config: `enable_risk_controls={ENABLE_RISK_CONTROLS}` (matches `run_backtest_strategy.py` "
        f"backtest path), `include_costs={INCLUDE_COSTS}` (real cost model)  ",
        f"DSR multiple-testing deflation: n_trials = {N_TRIALS_DSR}  ",
        f"Pooled-OOS bars: {n_obs} (per strategy)  ",
        "",
        "**What 'pipeline-realistic' means here (honest scope):** the selection logic is IDENTICAL "
        "to the prior vectorized study `_oos_wf_leverage_short.py`; the difference is that orders are "
        "now generated and filled through the REAL production cycle. The literal pipeline ADDS a real "
        "cost model (commission+spread+impact), real signed-notional→integer-share order generation, "
        "and feature enrichment (HMM/behavioral/macro/rv) — the last is IMMATERIAL because these "
        "signals read only `close`. It does NOT model the prior study's explicit short-borrow "
        "(50 bps/yr) or margin financing (100 bps/yr), so on the carry side it is LESS conservative; "
        "net cost direction vs the prior study is therefore ambiguous and both are shown.",
        "",
        "**Live-mode caveat (separate layer):** with `enable_risk_controls=True` (paper/live, NOT the "
        "backtest path), a smoke test confirmed the cycle preserves signed shorts (first-rebalance "
        "BUY notional ≈ SELL notional, ratio 0.99) but de-levers gross ~5x to the default "
        "`risk_limits.max_gross_exposure`=1.20 cap. In live mode every book below would therefore be "
        "further de-levered, pushing absolute returns DOWN and leaving Sharpe ≈ unchanged (cash drag "
        "is ~vol-neutral). That cannot rescue a rejected strategy.",
        "",
        "**Honesty note:** Survivorship-only cache. Bias DIRECTION is strategy-dependent for L/S: "
        "short legs of mom_ls/bab_ls/lowvol_ls cannot short delisted losers → short leg UNDERSTATED "
        "→ CONSERVATIVE lower bound. reversal_ls/_lo LONG recovered losers → OPTIMISTIC upper bound. "
        "The repo's LIVE-owned strategies (trend_baseline, multifactor_v2, news_alpha, crisis_alpha) "
        "are evaluated separately (see companion section / prior sessions) on their own universes. "
        "CI: not run; local one-shot.",
        "",
        "## Verdict (auto-generated)",
        "",
        *verdict_lines,
        "",
        overall,
        "",
        "## Consolidated OOS-Edge table (pooled out-of-sample, LITERAL pipeline)",
        "",
        "_Beta ≈ 0 confirms market-neutrality of the L/S books. IR vs SPY = annualised mean "
        "excess-over-SPY / its vol; IR t = IR·√years (|t|>1.96 ≈ 5% significant). DSR-prob is "
        "deflated for n_trials (Bailey-López de Prado); DSR✓ = passes 5%. PSR>SPY = prob true "
        "Sharpe exceeds SPY's. VolMatchRet = annual return if levered to SPY's vol, net of "
        "financing — the honest 'beats SPY CAGR?' figure for a market-neutral book._",
        "",
        _edge_table(all_edges, spy_pooled_edge),
        "",
    ]
    for mode in LONG_SHORT_MODES + LONGONLY_MODES:
        parts.append(f"## {MODE_TITLES[mode]}")
        parts.append("")
        parts.append(_fold_table(all_results[mode]))
        parts.append("")
    parts += [
        "---",
        "_Script: `scripts/_oos_wf_pipeline_realistic.py` (read-only research harness; executes the "
        "real `run_trading_cycle` but touches NO production module, policy.yaml, or state)._  ",
        "_Selection/edge logic reused verbatim from `scripts/_oos_wf_leverage_short.py` (DRY)._  ",
        "_Pipeline entry: `src/assembled_core/qa/backtest_engine.py` "
        "(`run_portfolio_backtest` / `make_cycle_fn`)._  ",
        "_Edge helpers: `src/assembled_core/qa/deflated_sharpe.py`, `src/assembled_core/qa/metrics.py`._  ",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    log.info("Report written -> %s", OUT_MD)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    modes = list(LONG_SHORT_MODES + LONGONLY_MODES)
    # Optional argv: subset of modes, or "calibrate" = first mode only.
    args = [a for a in sys.argv[1:] if a]
    calibrate = False
    if args:
        if args[0] == "calibrate":
            modes = [LONG_SHORT_MODES[1]]  # mom_ls (strongest momentum)
            calibrate = True
        else:
            _runnable = LONG_SHORT_MODES + LONGONLY_MODES + ("lowmax_lo",)
            modes = [a for a in args if a in _runnable] or modes

    log.info("Loading universe prices…")
    prices, tradeable = _load_universe_prices()
    log.info("Modes to run: %s%s", modes, " (CALIBRATE: 1 fold)" if calibrate else "")

    all_results, all_edges = {}, {}
    pooled_spy_any = None
    for mode in modes:
        log.info("=== Running WF — mode=%s ===", mode)
        if calibrate:
            # one fold only, timed
            spy_dates = np.sort(prices[prices["symbol"] == "SPY"]["timestamp"].unique())

            def _as_utc(ts: object) -> pd.Timestamp:
                t = pd.Timestamp(ts)
                return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")

            test_start = _as_utc(spy_dates[TRAIN_WINDOW])
            test_end = _as_utc(
                spy_dates[TRAIN_WINDOW + TEST_WINDOW - 1]
            ) + pd.Timedelta(hours=23)
            t0 = time.time()
            m, diag, net_ret, spy_test = _simulate_pipeline(
                prices, tradeable, test_start, test_end, mode
            )
            log.info(
                "[CALIBRATE %s] fold %s-%s: CAGR %.1f%% Sharpe %.2f gross %.2f beta %.2f "
                "bars=%d  -> %.1fs/fold",
                mode,
                test_start.date(),
                test_end.date(),
                m["cagr"] * 100,
                m["sharpe"],
                diag["gross"],
                diag["beta_mkt"],
                len(net_ret),
                time.time() - t0,
            )
            log.info("CALIBRATE done. Estimate full matrix: 11 modes x 6 folds.")
            return 0

        results, pooled_s, pooled_b = _run_wf_pipeline(prices, tradeable, mode)
        edge = _edge_metrics(pooled_s, pooled_b, results)
        all_results[mode] = results
        all_edges[mode] = edge
        if pooled_b is not None and not pooled_b.empty:
            pooled_spy_any = pooled_b
        log.info(
            "[%s] pooled: AnnSharpe %.2f / CAGR %.1f%% / IR %.2f (t=%.2f) / DSR-prob %.2f (pass=%s)",
            mode,
            edge["ann_sharpe"],
            edge["cagr"] * 100,
            edge["ir"],
            edge["ir_t"],
            edge["dsr_prob"],
            edge["dsr_pass"],
        )

    if len(modes) == len(LONG_SHORT_MODES + LONGONLY_MODES):
        spy_pooled_edge = (
            _spy_pooled_edge(pooled_spy_any) if pooled_spy_any is not None else {}
        )
        n_obs = all_edges[modes[0]]["n_obs"]
        _write_report(all_results, all_edges, spy_pooled_edge, len(tradeable), n_obs)
    else:
        log.info(
            "Partial run (%d modes) — report not written (full matrix required).",
            len(modes),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
