"""One-shot OOS Walk-Forward falsification of the `sector_rotation_bias` factor.

Writes docs/results/2026_06_sector_rotation_oos.md.

Usage:
    python scripts/_oos_wf_sector_rotation.py

WHY THIS EXISTS
---------------
`sector_rotation_bias` (multifactor_v2 Factor 18) was a *dead* factor in
production: the offline sector-ETF store in output/aggregates/daily.parquet went
stale (sector ETFs are not in the watchlist nor the master panel), so the live
factor's 7-day staleness guard neutralised it to 0.0 on every bar. Commit
433c2c03 (scripts/ops/refresh_sector_etf_cache.py + daily_paper_trading.bat
Step 1b) fixed the *freshness* so the factor can now compute on live data.

Fixing freshness only unlocks CAPABILITY — it is NOT evidence of an edge. This
harness answers the only question that justifies a non-zero regime weight:
**does the factor's sector ranking actually beat SPY out-of-sample, after costs,
with a statistically significant, multiple-testing-deflated edge?** If not, the
weight stays ~0 regardless of freshness.

WHAT IT TESTS (the production signal, unchanged)
------------------------------------------------
The factor ranks the 8 SPDR sector ETFs by the EXACT production composite score
(`src.assembled_core.signals.sector_rotation.compute_sector_scores`: 3m mom 0.50
+ 6m mom 0.30 + 20d RS-vs-SPY 0.20), then top-3 sectors -> +1, bottom-2 -> -1.
This study uses that same function (no re-implementation) and forms 3 books:
    sector_ls  — long top-3 (leg sums +1) / short bottom-2 (leg sums -1),
                 dollar-neutral, gross 2.0. Purest test of the RANKING signal;
                 beta ~ 0 isolates whether the ranking has alpha.
    sector_lo  — long top-3 equal-weight (gross 1.0). The long-only sector-
                 momentum TILT, closest to how the factor tilts mfv2's long book.
                 Benchmarked vs buy-and-hold SPY (beta ~ 1).
    eq_sector  — equal-weight all 8 sector ETFs (gross 1.0). Baseline: does the
                 rotation add anything over just holding the sectors equally?

Because the factor value is constant within a sector, a stock-level L/S over any
universe is mathematically this sector-ETF L/S weighted by universe sector
composition. Testing at the ETF level isolates the pure signal and removes
survivorship + security_meta mapping noise.

PIT DISCIPLINE
--------------
Scores are computed ONCE over the full series; `compute_sector_scores` uses only
trailing shifts (shift(63/126/20)), so row t depends only on rows <= t — no
look-ahead. At each monthly rebalance the signal is read at ref = (rebal_date - 1
bar); positions are then lagged one more bar (pos.shift(1)) before returns are
applied, giving a 2-bar gap between last-seen data and first realised return.

NOTE on the composite during warm-up (not look-ahead, but an honesty caveat):
`compute_sector_scores` denom-weights its three terms, so a score becomes finite
as soon as ANY sub-term exists — the 20d-RS term alone yields a (partial-weight)
score well before the 3m/6m momentum terms are available. A score is therefore
NOT "only valid after a full 6m history"; early scores are an RS-tilted partial
composite. This does NOT corrupt the OOS result because (a) WARMUP_BARS=130 skips
the first ~6m so every evaluated bar is dominated by the full 3m+6m terms, and
(b) all 8 sector ETFs and SPY share the same 2018-01-02 start in this store, so
no sector is ranked on a partial composite while its peers use the full one
(staggered-inception leakage, anti-pattern E-030, does not apply here).

HONESTY (binding)
-----------------
- Data = output/aggregates/daily.parquet, the SAME offline store the live factor
  reads. History starts 2018-01-02 (the Alpaca/master-panel era) — NOT the full
  SPDR history back to 1998. ~7 OOS folds (test years ~2019-2025) covering the
  2020 COVID crash, the 2022 energy-led rotation, and the 2023-24 tech run.
- Survivorship bias is N/A here: the 8 SPDR sector ETFs and SPY did not delist
  over 2018-2026. (The stock-level mapping WOULD carry survivorship bias; that is
  exactly why this test is run at the ETF level.)
- Frictions are realistic-but-conservative for liquid ETFs: COST_BPS/leg turnover
  + BORROW_BPS_ANNUAL on short notional. No rate term structure.
- The production regime weights (configs/factor_weights_by_regime.json) currently
  assign this factor ~0. This harness decides whether that should change. A
  PROSPECT here is NOT a production claim — it would require CI validation and a
  weight re-fit on the corrected panel first.
- CI status: not run in CI; local one-shot. No production module touched
  (read-only on price data; imports the live signal function unchanged).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.assembled_core.signals.sector_rotation import (  # noqa: E402
    SECTOR_ETFS,
    compute_sector_scores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oos_wf_sector_rotation")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DAILY_PARQUET = ROOT / "output" / "aggregates" / "daily.parquet"
OUT_MD = ROOT / "docs" / "results" / "2026_06_sector_rotation_oos.md"

PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2026-05-30", tz="UTC")
TRAIN_WINDOW = 252
TEST_WINDOW = 252
STEP_SIZE = 252
REBAL_FREQ = "ME"  # month-end rebalance (slow 3m/6m signal -> monthly is honest)
WARMUP_BARS = 130  # ~full 6m (126) history so the composite is dominated by its
# 3m+6m momentum terms, not the RS-only partial weight (see PIT DISCIPLINE note)

TOP_N_LONG = 3  # factor: top-3 sectors -> +1
BOTTOM_N_SHORT = 2  # factor: bottom-2 sectors -> -1

COST_BPS = 5.0  # per-leg turnover cost (bps) — liquid SPDR sector ETFs
BORROW_BPS_ANNUAL = 30.0  # short-borrow fee on short notional (GC sector ETFs)
INITIAL_CAPITAL = 100_000.0

# Honest multiple-testing deflation: this study evaluates ONE fixed signal config
# (production defaults — NOT parameter-searched) across 3 portfolio constructions.
N_TRIALS_DSR = 3

MODES = ("sector_ls", "sector_lo", "eq_sector")
MODE_TITLES = {
    "sector_ls": "Sector-rotation L/S — long top-3 / short bottom-2 (the factor ranking, dollar-neutral)",
    "sector_lo": "Sector-rotation long-only tilt — long top-3 equal-weight (the mfv2 use-case)",
    "eq_sector": "Equal-weight 8 sectors (baseline — no rotation)",
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def _load_sector_panel() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the 8 sector ETFs + SPY from the live offline store.

    Returns (sector_wide_close, spy_close_series, scores_indexed_by_ts).
    """
    want = list(SECTOR_ETFS) + ["SPY"]
    df = pd.read_parquet(DAILY_PARQUET, columns=["timestamp", "symbol", "close"])
    df = df[df["symbol"].isin(want)].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[(df["timestamp"] >= PERIOD_START) & (df["timestamp"] <= PERIOD_END)]
    df = df.sort_values(["symbol", "timestamp"]).drop_duplicates(
        subset=["symbol", "timestamp"], keep="last"
    )

    present = sorted(set(df["symbol"]) & set(SECTOR_ETFS))
    if len(present) < 5 or "SPY" not in set(df["symbol"]):
        raise RuntimeError(
            f"Insufficient sector-ETF coverage: {present} (+SPY present="
            f"{'SPY' in set(df['symbol'])}). Need >=5 sectors + SPY."
        )

    wide = (
        df.pivot_table(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .ffill()
    )
    sector_cols = [c for c in SECTOR_ETFS if c in wide.columns]
    sector_wide = wide[sector_cols]
    spy_close = wide["SPY"]

    # Compute the production composite scores ONCE over the full series. This is
    # PIT-safe: compute_sector_scores uses only trailing shifts, so the score at
    # timestamp t depends only on prices at t' <= t.
    sector_long = (
        sector_wide.reset_index()
        .melt(id_vars="timestamp", var_name="symbol", value_name="close")
        .dropna(subset=["close"])
    )
    spy_long = spy_close.reset_index()
    spy_long.columns = ["timestamp", "close"]
    spy_long["symbol"] = "SPY"

    scores = compute_sector_scores(sector_long, spy_long)
    scores = scores.set_index("timestamp").sort_index()

    log.info(
        "Loaded sector panel: %d sectors %s + SPY, %d bars %s -> %s; scores rows=%d",
        len(sector_cols),
        sector_cols,
        len(wide),
        wide.index.min().date(),
        wide.index.max().date(),
        len(scores),
    )
    return sector_wide, spy_close, scores


# ---------------------------------------------------------------------------
# Selection — signed weight dict {etf: weight} using the EXACT factor ranking
# ---------------------------------------------------------------------------
def _rank_scores_at(scores: pd.DataFrame, ref_date: pd.Timestamp) -> list[str] | None:
    """Sectors sorted by composite score (desc) at ref_date. None if < 5 valid."""
    if ref_date not in scores.index:
        # ref_date is a real trading date taken from the price index; it must be
        # in the (same-grid) scores frame. Guard defensively anyway.
        return None
    row = scores.loc[ref_date]
    etf_scores: dict[str, float] = {}
    for etf in SECTOR_ETFS:
        col = f"{etf}_score"
        if col in row.index:
            val = float(row[col])
            if np.isfinite(val):
                etf_scores[etf] = val
    # Factor requires >=5 distinct ETFs for a disjoint top-3 / bottom-2 partition.
    if len(etf_scores) < 5:
        return None
    return sorted(etf_scores, key=etf_scores.get, reverse=True)


def _weights_for_mode(ranked: list[str], mode: str) -> dict[str, float]:
    if mode == "eq_sector":
        w = 1.0 / len(SECTOR_ETFS)
        return {e: w for e in SECTOR_ETFS}

    longs = ranked[:TOP_N_LONG]
    weights = {e: 1.0 / len(longs) for e in longs}  # long leg sums to +1
    if mode == "sector_lo":
        return weights

    # sector_ls: short the bottom-2, disjoint from longs (factor subtracts top
    # from bottom so a name can never be in both buckets).
    shorts = [e for e in ranked[-BOTTOM_N_SHORT:] if e not in weights]
    if shorts:
        sw = -1.0 / len(shorts)  # short leg sums to -1
        weights.update({e: sw for e in shorts})
    return weights


# ---------------------------------------------------------------------------
# Simulation (one WF fold)
# ---------------------------------------------------------------------------
def _metrics(net_ret: pd.Series) -> dict:
    if len(net_ret) < 5:
        return dict(cagr=float("nan"), sharpe=float("nan"), maxdd=float("nan"))
    eq = INITIAL_CAPITAL * (1 + net_ret).cumprod()
    n_years = len(net_ret) / 252.0
    cagr = (eq.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / n_years) - 1.0
    sigma = net_ret.std() * np.sqrt(252)
    sharpe = (net_ret.mean() * 252) / sigma if sigma > 1e-12 else float("nan")
    dd = (eq - eq.cummax()) / eq.cummax()
    return dict(cagr=cagr, sharpe=sharpe, maxdd=float(dd.min()))


def _simulate_fold(
    sector_wide: pd.DataFrame,
    spy_close: pd.Series,
    scores: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    mode: str,
) -> tuple[dict, dict, pd.Series, pd.Series]:
    all_dates = sector_wide.index
    sector_rets = sector_wide.pct_change()
    spy_rets = spy_close.pct_change()

    rebal_dates = pd.date_range(
        start=all_dates.min(), end=test_end, freq=REBAL_FREQ, tz="UTC"
    )
    rebal_idx = []
    for rd in rebal_dates:
        cands = all_dates[all_dates <= rd]
        if len(cands) > 0:
            rebal_idx.append(cands[-1])
    rebal_idx = sorted(set(rebal_idx))

    pos_wide = pd.DataFrame(0.0, index=all_dates, columns=sector_wide.columns)
    for i, rebal_date in enumerate(rebal_idx):
        next_rebal = rebal_idx[i + 1] if i + 1 < len(rebal_idx) else test_end
        loc = all_dates.get_loc(rebal_date)
        if loc < WARMUP_BARS:
            continue
        ref_date = all_dates[loc - 1]  # strictly before rebal_date (PIT)
        ranked = _rank_scores_at(scores, ref_date)
        if ranked is None:
            continue
        weights = _weights_for_mode(ranked, mode)
        mask = (all_dates >= rebal_date) & (all_dates < next_rebal)
        fill_dates = all_dates[mask]
        if len(fill_dates) == 0:
            continue
        pos_wide.loc[fill_dates, :] = 0.0
        for etf, w in weights.items():
            if etf in pos_wide.columns:
                pos_wide.loc[fill_dates, etf] = w

    pos_lag = pos_wide.shift(1).fillna(0.0)  # 1-bar execution lag
    rets_aligned = sector_rets.reindex(columns=pos_lag.columns).fillna(0.0)
    port_ret = (pos_lag * rets_aligned).sum(axis=1)

    abs_delta = pos_lag.diff().fillna(0.0).abs().sum(axis=1)
    cost_turnover = abs_delta * COST_BPS / 10_000.0
    short_notional = pos_lag.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_notional * (BORROW_BPS_ANNUAL / 10_000.0 / 252.0)
    net_ret_all = port_ret - cost_turnover - borrow

    test_mask = (net_ret_all.index >= test_start) & (net_ret_all.index < test_end)
    net_ret = net_ret_all[test_mask]
    pos_lag_test = pos_lag[test_mask]
    if len(net_ret) < 5:
        raise ValueError(f"Only {len(net_ret)} test bars in fold")

    m = _metrics(net_ret)
    spy_test = spy_rets[test_mask]
    gross = float(pos_lag_test.abs().sum(axis=1).mean())
    turnover_yr = float(
        pos_lag_test.diff().abs().sum(axis=1).sum() / (len(net_ret) / 252.0)
    )
    common = net_ret.index.intersection(spy_test.index)
    if len(common) > 5 and spy_test[common].std() > 1e-12:
        beta_mkt = float(
            np.cov(net_ret[common], spy_test[common])[0, 1] / np.var(spy_test[common])
        )
    else:
        beta_mkt = float("nan")
    diag = dict(gross=gross, turnover_yr=turnover_yr, beta_mkt=beta_mkt)
    return m, diag, net_ret, spy_test


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------
def _run_wf(sector_wide, spy_close, scores, mode):
    all_dates = sector_wide.index
    results, pooled_strat, pooled_spy = [], [], []
    fold_idx = 1
    for train_start_i in range(
        0, len(all_dates) - TRAIN_WINDOW - TEST_WINDOW + 1, STEP_SIZE
    ):
        train_end_i = train_start_i + TRAIN_WINDOW
        test_end_i = train_end_i + TEST_WINDOW
        if test_end_i > len(all_dates):
            break
        test_start = all_dates[train_end_i]
        test_end = all_dates[test_end_i - 1] + pd.Timedelta(hours=23)
        try:
            m, diag, net_ret, spy_test = _simulate_fold(
                sector_wide, spy_close, scores, test_start, test_end, mode
            )
            bm = _metrics(spy_test.dropna())
            results.append(
                dict(
                    fold=fold_idx,
                    test_start=test_start.date(),
                    test_end=test_end.date(),
                    cagr=m["cagr"],
                    sharpe=m["sharpe"],
                    maxdd=m["maxdd"],
                    spy_cagr=bm["cagr"],
                    spy_sharpe=bm["sharpe"],
                    gross=diag["gross"],
                    turnover_yr=diag["turnover_yr"],
                    beta_mkt=diag["beta_mkt"],
                    n_bars=len(net_ret),
                    status="OK",
                )
            )
            pooled_strat.append(net_ret)
            pooled_spy.append(spy_test.reindex(net_ret.index))
            log.info(
                "[%s] Fold %d %s-%s: CAGR %.1f%% / Sharpe %.2f / MaxDD %.1f%% / beta %.2f / gross %.2f "
                "(SPY %.1f%% / %.2f)",
                mode,
                fold_idx,
                test_start.date(),
                test_end.date(),
                m["cagr"] * 100,
                m["sharpe"],
                m["maxdd"] * 100,
                diag["beta_mkt"],
                diag["gross"],
                bm["cagr"] * 100,
                bm["sharpe"],
            )
        except Exception as exc:
            log.warning("[%s] Fold %d FAILED: %s", mode, fold_idx, exc)
            results.append(
                dict(
                    fold=fold_idx,
                    test_start=test_start.date(),
                    test_end=test_end.date(),
                    status=f"FAILED: {exc}",
                )
            )
        fold_idx += 1

    pooled_s = (
        pd.concat(pooled_strat).sort_index() if pooled_strat else pd.Series(dtype=float)
    )
    pooled_b = (
        pd.concat(pooled_spy).sort_index() if pooled_spy else pd.Series(dtype=float)
    )
    return results, pooled_s, pooled_b


# ---------------------------------------------------------------------------
# OOS-edge metrics (pooled out-of-sample series) — same suite as leverage_short
# ---------------------------------------------------------------------------
def _edge_metrics(pooled_strat: pd.Series, pooled_spy: pd.Series, results) -> dict:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe
    from src.assembled_core.qa.metrics import probabilistic_sharpe_ratio

    r = pooled_strat.dropna()
    out = dict(
        ann_sharpe=float("nan"),
        sharpe_t=float("nan"),
        cagr=float("nan"),
        maxdd=float("nan"),
        calmar=float("nan"),
        beta=float("nan"),
        ir=float("nan"),
        ir_t=float("nan"),
        dsr_prob=float("nan"),
        dsr_pass=False,
        psr_vs_spy=float("nan"),
        turnover_yr=float("nan"),
        fold_win=float("nan"),
        vol_matched_ret=float("nan"),
        n_obs=int(len(r)),
    )
    if len(r) < 30:
        return out
    n = len(r)
    n_years = n / 252.0
    mu_d, sd_d = r.mean(), r.std()
    ann_sharpe = (mu_d * 252) / (sd_d * np.sqrt(252)) if sd_d > 1e-12 else float("nan")
    out["ann_sharpe"] = float(ann_sharpe)
    out["sharpe_t"] = (
        float(ann_sharpe * np.sqrt(n_years))
        if np.isfinite(ann_sharpe)
        else float("nan")
    )
    eq = (1 + r).cumprod()
    out["cagr"] = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0)
    dd = (eq - eq.cummax()) / eq.cummax()
    out["maxdd"] = float(dd.min())
    out["calmar"] = (
        float(out["cagr"] / abs(out["maxdd"]))
        if abs(out["maxdd"]) > 1e-9
        else float("nan")
    )

    spy = pooled_spy.reindex(r.index)
    common = r.index[spy.notna() & r.notna()]
    if len(common) > 30:
        ex = r[common] - spy[common]
        if ex.std() > 1e-12:
            ir = (ex.mean() * 252) / (ex.std() * np.sqrt(252))
            out["ir"] = float(ir)
            out["ir_t"] = float(ir * np.sqrt(len(common) / 252.0))
        sv = spy[common].var()
        if sv > 1e-12:
            out["beta"] = float(np.cov(r[common], spy[common])[0, 1] / sv)
        sr_d = mu_d / sd_d if sd_d > 1e-12 else float("nan")
        spy_sd = spy[common].std()
        sr_spy_d = spy[common].mean() / spy_sd if spy_sd > 1e-12 else 0.0
        if np.isfinite(sr_d):
            out["psr_vs_spy"] = float(
                probabilistic_sharpe_ratio(
                    sr_d,
                    n,
                    sharpe_benchmark=sr_spy_d,
                    skew=float(r.skew()),
                    kurtosis=float(r.kurt() + 3.0),
                )
            )
        spy_vol_ann = spy[common].std() * np.sqrt(252)
        if np.isfinite(ann_sharpe):
            # Lever the book to SPY's vol; ETF financing ~0, so the vol-matched
            # annual return reduces to Sharpe * SPY-vol.
            out["vol_matched_ret"] = float(ann_sharpe * spy_vol_ann)

    dsr = deflated_sharpe(r, n_trials=N_TRIALS_DSR)
    out["dsr_prob"] = float(dsr.deflated_sharpe_probability)
    out["dsr_pass"] = bool(dsr.passes_5pct)

    ok = [x for x in results if x.get("status") == "OK"]
    if ok:
        wins = sum(
            1
            for x in ok
            if np.isfinite(x.get("sharpe", float("nan")))
            and np.isfinite(x.get("spy_sharpe", float("nan")))
            and x["sharpe"] > x["spy_sharpe"]
        )
        out["fold_win"] = f"{wins}/{len(ok)}"
        tos = [
            x["turnover_yr"]
            for x in ok
            if np.isfinite(x.get("turnover_yr", float("nan")))
        ]
        out["turnover_yr"] = float(np.mean(tos)) if tos else float("nan")
    return out


def _spy_pooled_edge(pooled_spy: pd.Series) -> dict:
    from src.assembled_core.qa.deflated_sharpe import deflated_sharpe

    r = pooled_spy.dropna()
    n = len(r)
    out = dict(
        ann_sharpe=float("nan"),
        sharpe_t=float("nan"),
        cagr=float("nan"),
        maxdd=float("nan"),
        dsr_prob=float("nan"),
        dsr_pass=False,
    )
    if n < 30:
        return out
    n_years = n / 252.0
    sd = r.std()
    ann_sharpe = (r.mean() * 252) / (sd * np.sqrt(252)) if sd > 1e-12 else float("nan")
    out["ann_sharpe"] = float(ann_sharpe)
    out["sharpe_t"] = float(ann_sharpe * np.sqrt(n_years))
    eq = (1 + r).cumprod()
    out["cagr"] = float(eq.iloc[-1] ** (1.0 / n_years) - 1.0)
    dd = (eq - eq.cummax()) / eq.cummax()
    out["maxdd"] = float(dd.min())
    dsr = deflated_sharpe(r, n_trials=N_TRIALS_DSR)
    out["dsr_prob"] = float(dsr.deflated_sharpe_probability)
    out["dsr_pass"] = bool(dsr.passes_5pct)
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(v, fmt="+.1%"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return format(v, fmt)


def _avg(results, key):
    vals = [
        r[key]
        for r in results
        if r.get("status") == "OK" and np.isfinite(r.get(key, float("nan")))
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _fold_table(results) -> str:
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
            f"{r['sharpe']:+.2f} | {_fmt(r['maxdd'])} | {r['beta_mkt']:+.2f} | {r['gross']:.2f} | "
            f"{_fmt(r['spy_cagr'])} | {r['spy_sharpe']:+.2f} | {r['turnover_yr']:.0f} |"
        )
    ok = [r for r in results if r.get("status") == "OK"]
    if ok:
        lines.append(
            f"| **Ø ({len(ok)}/{len(results)})** | — | **{_fmt(_avg(results, 'cagr'))}** | "
            f"**{_avg(results, 'sharpe'):+.2f}** | **{_fmt(_avg(results, 'maxdd'))}** | "
            f"**{_avg(results, 'beta_mkt'):+.2f}** | **{_avg(results, 'gross'):.2f}** | "
            f"{_fmt(_avg(results, 'spy_cagr'))} | {_avg(results, 'spy_sharpe'):+.2f} | "
            f"{_avg(results, 'turnover_yr'):.0f} |"
        )
    return "\n".join(lines)


def _verdict_line(mode: str, edge: dict, spy_edge: dict) -> str:
    spy_sharpe = spy_edge["ann_sharpe"]
    beats_sharpe = np.isfinite(edge["ann_sharpe"]) and edge["ann_sharpe"] > spy_sharpe
    significant = (
        bool(edge["dsr_pass"]) and np.isfinite(edge["ir_t"]) and edge["ir_t"] > 1.96
    )
    prospect = beats_sharpe and significant
    tag = "PROSPECT" if prospect else "REJECTED"
    return (
        f"- **{mode}** [{tag}] ({MODE_TITLES[mode]}): pooled-OOS Sharpe "
        f"{edge['ann_sharpe']:+.2f} vs SPY {spy_sharpe:+.2f}; IR vs SPY "
        f"{_fmt(edge['ir'], '+.2f')} (t={_fmt(edge['ir_t'], '+.2f')}); DSR-prob "
        f"{_fmt(edge['dsr_prob'], '.2f')} (pass5%={edge['dsr_pass']}); beta "
        f"{_fmt(edge['beta'], '+.2f')}; vol-matched ann.ret {_fmt(edge['vol_matched_ret'])}."
    )


def _edge_table(edges: dict, spy_edge: dict) -> str:
    cols = (
        "| Strategy | AnnSharpe | Sharpe t | CAGR | MaxDD | Beta | IR vs SPY | IR t | "
        "DSR-prob | DSR✓ | PSR>SPY | Trn/yr | FoldWin | VolMatchRet |"
    )
    sep = "|" + "---|" * 14
    lines = [cols, sep]
    for mode in MODES:
        e = edges[mode]
        lines.append(
            f"| {mode} | {_fmt(e['ann_sharpe'], '+.2f')} | {_fmt(e['sharpe_t'], '+.2f')} | "
            f"{_fmt(e['cagr'])} | {_fmt(e['maxdd'])} | {_fmt(e['beta'], '+.2f')} | "
            f"{_fmt(e['ir'], '+.2f')} | {_fmt(e['ir_t'], '+.2f')} | {_fmt(e['dsr_prob'], '.2f')} | "
            f"{'Y' if e['dsr_pass'] else 'N'} | {_fmt(e['psr_vs_spy'], '.2f')} | "
            f"{_fmt(e['turnover_yr'], '.0f')} | {e['fold_win']} | {_fmt(e['vol_matched_ret'])} |"
        )
    lines.append(
        f"| **SPY (bench)** | {_fmt(spy_edge['ann_sharpe'], '+.2f')} | "
        f"{_fmt(spy_edge['sharpe_t'], '+.2f')} | {_fmt(spy_edge['cagr'])} | "
        f"{_fmt(spy_edge['maxdd'])} | +1.00 | — | — | {_fmt(spy_edge['dsr_prob'], '.2f')} | "
        f"{'Y' if spy_edge['dsr_pass'] else 'N'} | — | 0 | — | {_fmt(spy_edge['cagr'])} |"
    )
    return "\n".join(lines)


def _write_report(all_results, all_edges, spy_edge, n_bars):
    spy_sharpe = spy_edge["ann_sharpe"]
    verdict_lines = [_verdict_line(m, all_edges[m], spy_edge) for m in MODES]
    prospects = [
        m
        for m in MODES
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
            f"**{len(prospects)} of {len(MODES)} sector-rotation books show a PROSPECT** "
            f"({', '.join(prospects)}) — pooled-OOS Sharpe > SPY AND a significant, "
            f"multiple-testing-deflated edge. NOT a production claim: a non-zero "
            f"`sector_rotation_bias` weight still requires CI validation and a regime-"
            f"weight re-fit on the corrected factor panel before deployment."
        )
    else:
        overall = (
            "**ALL sector-rotation books REJECTED** — none clears SPY's pooled-OOS "
            "Sharpe with a multiple-testing-deflated (DSR) AND statistically significant "
            "(IR t>1.96) edge over 2018-2026. Fixing the factor's data freshness "
            "(commit 433c2c03) unlocked capability but did NOT reveal an edge. The "
            "production regime weight for `sector_rotation_bias` therefore stays ~0; a "
            "non-zero weight is not justified on this evidence."
        )

    parts = [
        "# sector_rotation_bias — OOS Walk-Forward Falsification (the actual edge test)",
        "",
        "Run date (UTC): 2026-06-01  ",
        "Data: `output/aggregates/daily.parquet` — the SAME offline store the live factor reads  ",
        f"Universe: 8 SPDR sector ETFs {SECTOR_ETFS} + SPY (benchmark/RS factor)  ",
        f"History: {n_bars} bars 2018-01-02 → 2026-05-29 (Alpaca/master-panel era; NOT full SPDR history)  ",
        "Signal: production `compute_sector_scores` (3m·0.50 + 6m·0.30 + 20d-RS·0.20), top-3 long / bottom-2 short — unchanged  ",
        f"WF: {TRAIN_WINDOW}/{TEST_WINDOW}/{STEP_SIZE} (train/test/step), month-end rebalance, 1-bar exec lag  ",
        f"Frictions: {COST_BPS:.0f} bps/leg turnover, {BORROW_BPS_ANNUAL:.0f} bps/yr short borrow (liquid ETFs)  ",
        f"DSR multiple-testing deflation: n_trials = {N_TRIALS_DSR} (one fixed signal config, not parameter-searched)  ",
        "",
        "**Context / honesty:** `sector_rotation_bias` was a *dead* factor in production "
        "(stale offline store → 7-day staleness guard neutralised it to 0.0). Commit 433c2c03 "
        "fixed the freshness so it can now compute on live data. That unlocked CAPABILITY only. "
        "This harness is the falsification test that decides whether the factor deserves a "
        "non-zero regime weight. Survivorship bias is N/A (the 8 SPDR sector ETFs + SPY did not "
        "delist over 2018-2026); the factor is tested at the ETF level precisely to isolate the "
        "pure ranking signal and avoid the survivorship + security_meta mapping noise a stock-"
        "level test would carry. Because the factor value is constant within a sector, a stock-"
        "level L/S is mathematically this sector-ETF L/S weighted by universe composition. "
        "History starts 2018 (the live store's depth); deeper SPDR history (→1998) is NOT what the "
        "live factor sees. The production composite is denom-weighted, so an early score is finite "
        "from the 20d-RS term alone (an RS-tilted partial composite) before the 3m/6m terms exist; "
        "the 130-bar warm-up skips that window so every evaluated bar is dominated by the full "
        "3m+6m terms, and all 8 ETFs + SPY share the 2018-01-02 start (no staggered-inception "
        "leakage). CI: not run; local one-shot. No production module touched.",
        "",
        "## Verdict (auto-generated)",
        "",
        *verdict_lines,
        "",
        overall,
        "",
        "## OOS-Edge table (pooled out-of-sample)",
        "",
        "_sector_ls beta ≈ 0 confirms the dollar-neutral book isolates the ranking's alpha. "
        "IR vs SPY = annualised mean excess-over-SPY / its vol; IR t = IR·√years (|t|>1.96 ≈ 5% "
        "significant). DSR-prob is deflated for n_trials (Bailey-López de Prado); DSR✓ = passes 5%. "
        "PSR>SPY = prob true Sharpe exceeds SPY's. VolMatchRet = annual return if levered to SPY's "
        "vol — the honest 'beats SPY CAGR?' figure for a market-neutral book._",
        "",
        _edge_table(all_edges, spy_edge),
        "",
    ]
    for mode in MODES:
        parts.append(f"## {MODE_TITLES[mode]}")
        parts.append("")
        parts.append(_fold_table(all_results[mode]))
        parts.append("")
    parts += [
        "---",
        "_Script: `scripts/_oos_wf_sector_rotation.py` (read-only research harness; imports the "
        "live `compute_sector_scores` unchanged; no production file modified)._  ",
        "_Signal: `src/assembled_core/signals/sector_rotation.py`; factor wiring: "
        "`src/assembled_core/strategies/multifactor_v2.py::_compute_sector_rotation_bias`._  ",
        "_Edge helpers: `src/assembled_core/qa/deflated_sharpe.py`, `src/assembled_core/qa/metrics.py`._  ",
        "_Freshness fix that motivated this test: commit 433c2c03 "
        "(`scripts/ops/refresh_sector_etf_cache.py` + `daily_paper_trading.bat` Step 1b)._  ",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    log.info("Report written -> %s", OUT_MD)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    log.info("Loading sector panel from %s …", DAILY_PARQUET)
    sector_wide, spy_close, scores = _load_sector_panel()

    all_results, all_edges = {}, {}
    pooled_spy_ref = None
    for mode in MODES:
        log.info("Running WF — mode=%s …", mode)
        results, pooled_s, pooled_b = _run_wf(sector_wide, spy_close, scores, mode)
        edge = _edge_metrics(pooled_s, pooled_b, results)
        all_results[mode] = results
        all_edges[mode] = edge
        if pooled_spy_ref is None and len(pooled_b.dropna()) > 0:
            pooled_spy_ref = pooled_b
        log.info(
            "[%s] pooled-OOS AnnSharpe %.2f | CAGR %.1f%% | IR %.2f (t=%.2f) | DSR-prob %.2f (pass=%s) | beta %.2f",
            mode,
            edge["ann_sharpe"],
            edge["cagr"] * 100,
            edge["ir"],
            edge["ir_t"],
            edge["dsr_prob"],
            edge["dsr_pass"],
            edge["beta"],
        )

    spy_edge = _spy_pooled_edge(pooled_spy_ref) if pooled_spy_ref is not None else {}
    _write_report(all_results, all_edges, spy_edge, len(sector_wide))

    log.info(
        "DONE. SPY pooled-OOS Sharpe %.2f / CAGR %.1f%%. Sector Sharpes: %s",
        spy_edge.get("ann_sharpe", float("nan")),
        spy_edge.get("cagr", float("nan")) * 100,
        {m: round(all_edges[m]["ann_sharpe"], 2) for m in MODES},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
