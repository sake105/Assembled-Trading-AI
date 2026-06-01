"""One-shot OOS Walk-Forward: LONG/SHORT + LEVERED strategies + OOS-edge suite.

Writes docs/results/2026_05_leverage_short_oos.md.

Usage:
    python scripts/_oos_wf_leverage_short.py

Context: leverage and shorting are now permitted (prior studies were long-only,
no-leverage). That unlocks the canonical risk-adjusted SPY-beaters. Each candidate
is verified genuinely NEW vs. the repo (repo scout): no wired strategy implements
beta-targeted BAB, cross-sectional momentum L/S (WML), residual-momentum L/S,
1-month reversal L/S, or low-vol L/S. multifactor_long_short.py is a generic
factor-bundle blender, not these dedicated single-factor sleeves.

NEW long/short + levered modes (this study):
    bab_ls      — Betting-Against-Beta (Frazzini-Pedersen 2014): long low-beta
                  quintile LEVERED to beta 1 (x1/beta_L), short high-beta quintile
                  DELEVERED to beta 1 (x1/beta_H) -> ex-ante market-neutral. Gross
                  capped at GROSS_CAP. Beta from rolling 252d OLS vs SPY.
    mom_ls      — Cross-sectional 12-1 momentum WML (Jegadeesh-Titman): long top /
                  short bottom quintile, dollar-neutral (gross 2.0).
    resmom_ls   — Residual momentum L/S (Blitz-Huij-Martens 2011): single-factor
                  (SPY) market model, formation = Sigma residual last 126 skip 21
                  / residual vol; long top / short bottom quintile, dollar-neutral.
    reversal_ls — 1-month reversal L/S (Jegadeesh 1990): long losers / short
                  winners, dollar-neutral.
    lowvol_ls   — Low-volatility L/S: long low-rv / short high-rv quintile.

"OLD"/prior candidates re-run in the IDENTICAL engine for the consolidated OOS-edge
table (long-only, gross 1.0): mom_lo (total-return mom), high52w_lo, reversal_lo,
lowbeta_lo, resmom_lo, plus eq_weight baseline. Re-running them here lets the edge
stats be apples-to-apples and cross-checks the prior Sharpe values.

OOS-edge suite (beyond Sharpe/CAGR), computed on the POOLED out-of-sample daily
series per strategy, using the repo's own quant helpers where possible:
    - Annualised Sharpe + Sharpe t-stat (= Sharpe * sqrt(n_years))
    - Information Ratio vs SPY + IR t-stat (significance of the edge OVER SPY)
    - Market beta (shows neutrality of the L/S books)
    - Deflated Sharpe probability (Bailey-Lopez de Prado), DEFLATED for the number
      of trials in this whole research arc (n_trials=N_TRIALS_DSR) -> honest
      multiple-testing correction
    - Probabilistic Sharpe Ratio vs SPY (prob true Sharpe > SPY's)
    - Turnover/yr, fold-win-rate vs SPY, MaxDD, Calmar
NOTE: the repo's permutation_test_sharpe() is intentionally NOT used — it permutes
return ORDER, but Sharpe = mean/std is order-invariant, so it is degenerate for a
return series (returns p=1.0 for every strategy). DSR/PSR/t-stats are used instead.

HONESTY (binding):
    - Offline cache is SURVIVORS-ONLY. For L/S the bias DIRECTION is strategy-
      dependent: short-the-junk legs (mom_ls / bab_ls / lowvol_ls) cannot short the
      delisted losers -> short leg UNDERSTATED -> results are CONSERVATIVE there.
      reversal_ls/_lo LONGS recent losers (survivors that recovered) -> OPTIMISTIC.
    - Literature decay is real: BAB is critiqued as a micro-cap equal-weight +
      profitability artifact (Novy-Marx & Velikov 2022); short-term reversal is
      largely arbitraged away in liquid names; static WML is crash-prone
      (Daniel-Moskowitz 2016). McLean-Pontiff (2016): ~58% post-publication decay.
    - QMJ (quality) NOT tested: needs fundamentals absent from the OHLCV cache.
    - dual_momentum (owned) NOT driven: needs VEU/BIL, absent from the cache.
    - CI status: not run in CI; local one-shot only. No production module touched.

NO changes to any strategy module, policy.yaml, or production file. Read-only on
price data. Falsification harness: each candidate beats SPY risk-adjusted (with a
significant, multiple-testing-deflated edge) or is REJECTED.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oos_wf_leverage_short")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW = 252
TEST_WINDOW = 252
STEP_SIZE = 252

LOOKBACK = 252  # bars of history for beta / 12-1 mom / residual windows
MOM_SKIP = 21  # 12-1 momentum skips the most recent month
REV_LOOKBACK = 21  # short-term reversal formation (1 month)
RESID_FORM = 126  # residual-momentum formation length
REBAL_FREQ = "ME"
QUANTILE_SELECT = 0.20  # top/bottom quintile

COST_BPS = 10.75  # per-leg turnover cost (bps of traded notional)
BORROW_BPS_ANNUAL = 50.0  # short-borrow fee on short notional (liquid large-caps)
FIN_BPS_ANNUAL = 100.0  # financing on LONG notional exceeding 1.0 (margin)
GROSS_CAP = 3.0  # cap on bab_ls gross leverage
INITIAL_CAPITAL = 100_000.0

# Honest multiple-testing deflation: total distinct strategy configs evaluated
# across this whole research arc (5 long-only prior + total-mom control +
# eq_weight + 5 new L/S + this study's long-only re-runs ~ 16).
N_TRIALS_DSR = 16

LONG_SHORT_MODES = ("bab_ls", "mom_ls", "resmom_ls", "reversal_ls", "lowvol_ls")
LONGONLY_MODES = (
    "mom_lo",
    "high52w_lo",
    "reversal_lo",
    "lowbeta_lo",
    "resmom_lo",
    "eq_weight",
)
ALL_MODES = LONG_SHORT_MODES + LONGONLY_MODES

MODE_TITLES = {
    "bab_ls": "Betting-Against-Beta L/S, beta-targeted (Frazzini-Pedersen 2014)",
    "mom_ls": "Cross-sectional Momentum L/S, 12-1 WML (Jegadeesh-Titman 1993)",
    "resmom_ls": "Residual Momentum L/S (Blitz-Huij-Martens 2011)",
    "reversal_ls": "1-Month Reversal L/S (Jegadeesh 1990)",
    "lowvol_ls": "Low-Volatility L/S",
    "mom_lo": "Total-Return Momentum 12-1, long-only (control)",
    "high52w_lo": "52-Week-High Momentum, long-only (George-Hwang 2004)",
    "reversal_lo": "1-Month Reversal, long-only (Jegadeesh 1990)",
    "lowbeta_lo": "Low-Beta tilt, long-only (BAB no-leverage subset)",
    "resmom_lo": "Residual Momentum, long-only (Blitz-Huij-Martens 2011)",
    "eq_weight": "Equal-Weight universe (baseline)",
}

OUT_MD = ROOT / "docs" / "results" / "2026_05_leverage_short_oos.md"


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------
def _load_universe_prices() -> tuple[pd.DataFrame, list[str]]:
    from src.assembled_core.data.prices_ingest import load_eod_prices

    prices = load_eod_prices(None)
    if prices is None or prices.empty:
        raise RuntimeError("No price data returned from load_eod_prices")

    prices = prices[
        (prices["timestamp"] >= PERIOD_START - pd.Timedelta(days=60))
        & (prices["timestamp"] <= PERIOD_END + pd.Timedelta(days=5))
    ].copy()

    sym_info = prices.groupby("symbol")["timestamp"].agg(["min", "count"])
    valid = sym_info[
        (sym_info["min"] <= pd.Timestamp("2018-01-31", tz="UTC"))
        & (sym_info["count"] >= 500)
    ]
    tradeable = [s for s in valid.index if s != "SPY"]
    log.info(
        "Loaded %d rows for %d total symbols -> %d tradeable (excl SPY, %s -> %s)",
        len(prices),
        prices["symbol"].nunique(),
        len(tradeable),
        prices["timestamp"].min().date(),
        prices["timestamp"].max().date(),
    )
    return prices, tradeable


# ---------------------------------------------------------------------------
# Per-fold metrics / benchmark
# ---------------------------------------------------------------------------
def _metrics(net_ret: pd.Series) -> dict:
    if len(net_ret) < 5:
        return dict(
            cagr=float("nan"),
            sharpe=float("nan"),
            maxdd=float("nan"),
            calmar=float("nan"),
        )
    eq = INITIAL_CAPITAL * (1 + net_ret).cumprod()
    n_years = len(net_ret) / 252.0
    cagr = (eq.iloc[-1] / INITIAL_CAPITAL) ** (1.0 / n_years) - 1.0
    mu = net_ret.mean() * 252
    sigma = net_ret.std() * np.sqrt(252)
    sharpe = mu / sigma if sigma > 1e-9 else float("nan")
    roll_max = eq.cummax()
    dd = (eq - roll_max) / roll_max
    maxdd = float(dd.min())
    calmar = cagr / abs(maxdd) if abs(maxdd) > 1e-9 else float("nan")
    return dict(cagr=cagr, sharpe=sharpe, maxdd=maxdd, calmar=calmar)


def _benchmark_spy(prices: pd.DataFrame, test_start, test_end) -> dict:
    spy = prices[prices["symbol"] == "SPY"].set_index("timestamp")["close"]
    spy = spy[(spy.index >= test_start) & (spy.index < test_end)].sort_index()
    if len(spy) < 5:
        return dict(cagr=float("nan"), sharpe=float("nan"), maxdd=float("nan"))
    return _metrics(spy.pct_change().dropna())


# ---------------------------------------------------------------------------
# Selection — returns a signed weight dict {symbol: weight} (negative = short)
# ---------------------------------------------------------------------------
def _vectorized_beta(R: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS slope of each column of R on x (single-factor market beta)."""
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom < 1e-12:
        return np.full(R.shape[1], np.nan)
    return (xc[:, None] * R).sum(axis=0) / denom


def _score_series(
    px: pd.DataFrame,
    rets_win: pd.DataFrame,
    spy_win: np.ndarray,
    valid_cols: list[str],
    mode: str,
) -> pd.Series:
    """Higher score == more long-preferred. One row per valid symbol."""
    close_ref = px.iloc[-1]

    if mode in ("mom_ls", "mom_lo"):
        score = px.iloc[-1 - MOM_SKIP] / px.iloc[0] - 1.0  # 12-1 total return
    elif mode == "high52w_lo":
        score = close_ref / px.max(axis=0)  # proximity to 52w-high
    elif mode in ("reversal_ls", "reversal_lo"):
        score = -(close_ref / px.iloc[-1 - REV_LOOKBACK] - 1.0)  # buy losers
    elif mode == "lowvol_ls":
        score = -rets_win.std(axis=0)  # long low realized vol
    elif mode in ("lowbeta_lo", "bab_ls"):
        beta = _vectorized_beta(rets_win.to_numpy(), spy_win)
        score = pd.Series(-beta, index=valid_cols)  # long low beta
    elif mode in ("resmom_ls", "resmom_lo"):
        Rv = rets_win.to_numpy()
        beta = _vectorized_beta(Rv, spy_win)
        alpha = Rv.mean(axis=0) - beta * spy_win.mean()
        resid = Rv - (alpha[None, :] + beta[None, :] * spy_win[:, None])
        form_end = Rv.shape[0] - MOM_SKIP
        form_start = form_end - RESID_FORM
        if form_start < 0:
            return pd.Series(dtype=float)
        form = resid[form_start:form_end, :]
        sig = form.sum(axis=0)
        vol = form.std(axis=0)
        score = pd.Series(sig / (vol + 1e-9), index=valid_cols)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if not isinstance(score, pd.Series):
        score = pd.Series(score, index=valid_cols)
    return score[np.isfinite(score)]


def _select(trade_pivot, trade_rets, spy_rets, all_dates, rebal_date, mode) -> dict:
    """PIT cross-sectional selection at rebal_date -> signed weight dict.

    Strictly PIT: ref_idx = pos(rebal_date) - 1 -> only bars with idx <= ref_idx
    (strictly before rebal_date). A second execution-lag bar is added downstream
    via pos_wide.shift(1).
    """
    window_end_idx = all_dates.get_loc(rebal_date)
    ref_idx = window_end_idx - 1
    px_start = ref_idx - LOOKBACK + 1
    if px_start < 1:
        return {}

    px = trade_pivot.iloc[px_start : ref_idx + 1]
    valid_cols = px.columns[px.notna().sum() >= int(LOOKBACK * 0.8)].tolist()
    if len(valid_cols) < 10:
        return {}
    px = px[valid_cols]

    if mode == "eq_weight":
        w = 1.0 / len(valid_cols)
        return {s: w for s in valid_cols}

    rets_win = trade_rets.iloc[px_start : ref_idx + 1][valid_cols].fillna(0.0)
    spy_win = spy_rets.iloc[px_start : ref_idx + 1].fillna(0.0).to_numpy()

    # --- BAB: beta-targeted leverage (special construction) ---------------
    if mode == "bab_ls":
        beta = pd.Series(
            _vectorized_beta(rets_win.to_numpy(), spy_win), index=valid_cols
        )
        beta = beta[np.isfinite(beta)]
        if len(beta) < 10:
            return {}
        lo_thr = beta.quantile(QUANTILE_SELECT)
        hi_thr = beta.quantile(1.0 - QUANTILE_SELECT)
        longs = beta[beta <= lo_thr]
        shorts = beta[beta >= hi_thr]
        if longs.empty or shorts.empty:
            return {}
        # Clamp leg betas away from 0 to bound leverage; BAB longs are low-beta
        # (typically 0.4-0.8) -> levered up; shorts high-beta (>1) -> delevered.
        beta_l = max(float(longs.mean()), 0.20)
        beta_h = max(float(shorts.mean()), 0.20)
        lw = (1.0 / beta_l) / len(longs)  # long leg sums to 1/beta_l
        sw = -(1.0 / beta_h) / len(shorts)  # short leg sums to -1/beta_h
        gross = 1.0 / beta_l + 1.0 / beta_h
        if gross > GROSS_CAP:
            scale = GROSS_CAP / gross
            lw *= scale
            sw *= scale
        weights = {s: lw for s in longs.index}
        weights.update({s: sw for s in shorts.index})
        return weights

    # --- Rank-based modes -------------------------------------------------
    score = _score_series(px, rets_win, spy_win, valid_cols, mode)
    if score.empty:
        return {}
    long_thr = score.quantile(1.0 - QUANTILE_SELECT)
    longs = score[score >= long_thr].index.tolist()
    if not longs:
        return {}
    weights = {s: 1.0 / len(longs) for s in longs}  # long leg sums to +1

    if mode in LONG_SHORT_MODES:
        short_thr = score.quantile(QUANTILE_SELECT)
        shorts = score[score <= short_thr].index.tolist()
        shorts = [s for s in shorts if s not in weights]  # disjoint legs
        if shorts:
            sw = -1.0 / len(shorts)  # short leg sums to -1
            weights.update({s: sw for s in shorts})
    return weights


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def _apply_weights(pos_wide, all_dates, rebal_date, next_rebal, weights) -> None:
    mask = (all_dates >= rebal_date) & (all_dates < next_rebal)
    dates_to_fill = all_dates[mask]
    if len(dates_to_fill) == 0:
        return
    pos_wide.loc[dates_to_fill, :] = 0.0
    for sym, w in weights.items():
        if sym in pos_wide.columns:
            pos_wide.loc[dates_to_fill, sym] = w


def _simulate(prices, tradeable, test_start, test_end, mode):
    spy_dates = np.sort(prices[prices["symbol"] == "SPY"]["timestamp"].unique())
    pre_test = spy_dates[spy_dates < test_start]
    warmup_start = (
        pre_test[-TRAIN_WINDOW]
        if len(pre_test) >= TRAIN_WINDOW
        else (pre_test[0] if len(pre_test) > 0 else test_start)
    )

    syms_needed = list(set(tradeable) | {"SPY"})
    window_prices = prices[
        prices["symbol"].isin(syms_needed)
        & (prices["timestamp"] >= warmup_start)
        & (prices["timestamp"] < test_end)
    ].copy()

    pivot = (
        window_prices.pivot_table(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .ffill()
    )
    all_dates = pivot.index

    trade_cols = [c for c in tradeable if c in pivot.columns]
    if not trade_cols:
        raise ValueError(f"No tradeable symbols in pivot for {test_start.date()}")
    if "SPY" not in pivot.columns:
        raise ValueError(f"SPY missing in pivot for {test_start.date()}")

    trade_pivot = pivot[trade_cols]
    trade_rets = trade_pivot.pct_change()
    spy_rets = pivot["SPY"].pct_change()

    rebal_dates = pd.date_range(
        start=warmup_start, end=test_end, freq=REBAL_FREQ, tz="UTC"
    )
    rebal_idx = []
    for rd in rebal_dates:
        candidates = all_dates[all_dates <= rd]
        if len(candidates) > 0:
            rebal_idx.append(candidates[-1])
    rebal_idx = sorted(set(rebal_idx))

    log.info(
        "[%s] fold %s-%s: warmup %s, %d syms, %d rebal dates",
        mode,
        test_start.date(),
        test_end.date(),
        warmup_start.date(),
        len(trade_cols),
        len(rebal_idx),
    )

    pos_wide = pd.DataFrame(0.0, index=all_dates, columns=trade_cols)
    for i, rebal_date in enumerate(rebal_idx):
        next_rebal = rebal_idx[i + 1] if i + 1 < len(rebal_idx) else test_end
        weights = _select(
            trade_pivot, trade_rets, spy_rets, all_dates, rebal_date, mode
        )
        _apply_weights(pos_wide, all_dates, rebal_date, next_rebal, weights)

    pos_lag = pos_wide.shift(1).fillna(0.0)
    rets_aligned = trade_rets.reindex(columns=pos_lag.columns).fillna(0.0)
    port_ret_all = (pos_lag * rets_aligned).sum(axis=1)

    abs_delta = pos_lag.diff().fillna(0.0).abs().sum(axis=1)
    cost_turnover = abs_delta * COST_BPS / 10_000.0
    short_notional = pos_lag.clip(upper=0.0).abs().sum(axis=1)
    borrow = short_notional * (BORROW_BPS_ANNUAL / 10_000.0 / 252.0)
    long_notional = pos_lag.clip(lower=0.0).sum(axis=1)
    margin = (long_notional - 1.0).clip(lower=0.0)
    financing = margin * (FIN_BPS_ANNUAL / 10_000.0 / 252.0)
    net_ret_all = port_ret_all - cost_turnover - borrow - financing

    test_mask = (net_ret_all.index >= test_start) & (net_ret_all.index < test_end)
    net_ret = net_ret_all[test_mask]
    pos_lag_test = pos_lag[test_mask]
    if len(net_ret) < 5:
        raise ValueError(f"Only {len(net_ret)} test bars in fold")

    if pos_lag_test.abs().sum().sum() < 1e-9:
        log.warning(
            "[%s] fold %s-%s: all-flat", mode, test_start.date(), test_end.date()
        )

    m = _metrics(net_ret)
    spy_test = spy_rets[test_mask]
    # diagnostics
    gross = (pos_lag_test.abs().sum(axis=1)).mean()
    weight_changes = pos_lag_test.diff().abs().sum(axis=1)
    n_years = len(net_ret) / 252.0
    turnover_yr = float(weight_changes.sum()) / n_years if n_years > 0 else float("nan")
    common = net_ret.index.intersection(spy_test.index)
    if len(common) > 5 and spy_test[common].std() > 1e-12:
        beta_mkt = float(
            np.cov(net_ret[common], spy_test[common])[0, 1] / np.var(spy_test[common])
        )
    else:
        beta_mkt = float("nan")
    diag = dict(gross=float(gross), turnover_yr=turnover_yr, beta_mkt=beta_mkt)
    return m, diag, net_ret, spy_test


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------
def _run_wf(prices, tradeable, mode):
    spy_dates = np.sort(prices[prices["symbol"] == "SPY"]["timestamp"].unique())
    results = []
    pooled_strat = []
    pooled_spy = []
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

        try:
            m, diag, net_ret, spy_test = _simulate(
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
                "/ gross %.2f  (SPY: %.1f%% / %.2f)",
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
            )
        except Exception as exc:
            log.warning("[%s] Fold %d FAILED: %s", mode, fold_idx, exc)
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
# OOS-edge metrics (pooled out-of-sample series)
# ---------------------------------------------------------------------------
def _edge_metrics(
    pooled_strat: pd.Series, pooled_spy: pd.Series, results: list[dict]
) -> dict:
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
        # PSR vs SPY on per-period (daily) Sharpe — consistent scale
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
        # vol-matched (levered to SPY vol) annual return, net of extra financing
        spy_vol_ann = spy[common].std() * np.sqrt(252)
        strat_vol_ann = sd_d * np.sqrt(252)
        if strat_vol_ann > 1e-9 and np.isfinite(ann_sharpe):
            lev = spy_vol_ann / strat_vol_ann
            fin_drag = max(0.0, lev - 1.0) * (FIN_BPS_ANNUAL / 10_000.0)
            out["vol_matched_ret"] = float(ann_sharpe * spy_vol_ann - fin_drag)

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
    order = LONG_SHORT_MODES + LONGONLY_MODES
    for mode in order:
        e = edges[mode]
        lines.append(
            f"| {mode} | {_fmt(e['ann_sharpe'], '+.2f')} | {_fmt(e['sharpe_t'], '+.2f')} | "
            f"{_fmt(e['cagr'])} | {_fmt(e['maxdd'])} | {_fmt(e['beta'], '+.2f')} | "
            f"{_fmt(e['ir'], '+.2f')} | {_fmt(e['ir_t'], '+.2f')} | {_fmt(e['dsr_prob'], '.2f')} | "
            f"{'Y' if e['dsr_pass'] else 'N'} | {_fmt(e['psr_vs_spy'], '.2f')} | "
            f"{_fmt(e['turnover_yr'], '.0f')} | {e['fold_win']} | {_fmt(e['vol_matched_ret'])} |"
        )
    lines.append(
        f"| **SPY (bench)** | {_fmt(spy_edge['ann_sharpe'], '+.2f')} | {_fmt(spy_edge['sharpe_t'], '+.2f')} | "
        f"{_fmt(spy_edge['cagr'])} | {_fmt(spy_edge['maxdd'])} | +1.00 | — | — | "
        f"{_fmt(spy_edge['dsr_prob'], '.2f')} | {'Y' if spy_edge['dsr_pass'] else 'N'} | — | 0 | — | "
        f"{_fmt(spy_edge['cagr'])} |"
    )
    return "\n".join(lines)


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


def _write_report(all_results, all_edges, spy_pooled_edge, n_tradeable):
    spy_sharpe = spy_pooled_edge["ann_sharpe"]
    verdict_lines = [
        _verdict_line(m, all_edges[m], spy_sharpe) for m in LONG_SHORT_MODES
    ]
    prospects = [
        m
        for m in LONG_SHORT_MODES
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
            f"**{len(prospects)} of 5 NEW L/S strategies show a PROSPECT** "
            f"({', '.join(prospects)}) — Sharpe > SPY AND a significant, "
            f"multiple-testing-deflated edge. NOT a production claim: requires a "
            f"survivorship-clean re-test and CI validation before any deployment."
        )
    else:
        overall = (
            "**ALL 5 NEW L/S strategies REJECTED** — none clears SPY's pooled-OOS "
            "Sharpe with a multiple-testing-deflated (DSR) AND statistically significant "
            "(IR t>1.96) edge. This holds even though survivorship bias is CONSERVATIVE "
            "for the short-the-junk books (mom_ls/bab_ls/lowvol_ls). Consistent with the "
            "decay literature (BAB micro-cap/profitability artifact; reversal arbitraged; "
            "WML crash-prone). No prospect on this universe under realistic frictions."
        )

    parts = [
        "# Long/Short + Levered Strategies — OOS Walk-Forward + Edge Suite",
        "",
        "Run date (UTC): 2026-05-31  ",
        "Data: local offline cache via `load_eod_prices(None)` — survivors only  ",
        f"Universe: {n_tradeable} tradeable symbols (data ≤ 2018-01-31, ≥ 500 bars; SPY = market factor + benchmark)  ",
        "WF: 252/252/252 (train/test/step), monthly rebalance, top/bottom quintile  ",
        f"Frictions: {COST_BPS} bps/leg turnover, {BORROW_BPS_ANNUAL:.0f} bps/yr short borrow, "
        f"{FIN_BPS_ANNUAL:.0f} bps/yr financing on long notional > 1.0, BAB gross cap {GROSS_CAP:.1f}, 1-bar exec lag  ",
        f"DSR multiple-testing deflation: n_trials = {N_TRIALS_DSR}  ",
        f"Pooled-OOS bars: {all_edges[LONG_SHORT_MODES[0]]['n_obs']} (per strategy)  ",
        "",
        "**Honesty note:** Survivorship-only cache. Bias DIRECTION is strategy-dependent for L/S: "
        "short legs of mom_ls/bab_ls/lowvol_ls cannot short delisted losers → short leg UNDERSTATED "
        "→ those results are a CONSERVATIVE lower bound. reversal_ls/_lo long the recent losers that "
        "survived (recovered) → OPTIMISTIC upper bound. QMJ not tested (needs fundamentals, absent). "
        "dual_momentum (owned) not driven (needs VEU/BIL, absent). The repo's LIVE-owned strategies "
        "(trend_baseline, multifactor_v2, news_alpha, crisis_alpha) are NOT in this table — they run "
        "on different universes/harnesses and were OOS-evaluated in prior sessions (e.g. trend_baseline "
        "10-fold OOS Ø CAGR -6.1% vs SPY +13%); the 6 long-only rows here are factor-concept re-runs, "
        "not those production strategies. Leverage/borrow/financing are "
        "modelled with flat assumptions (no rate term structure). CI: not run; local one-shot.",
        "",
        "## Verdict (auto-generated)",
        "",
        *verdict_lines,
        "",
        overall,
        "",
        "## Consolidated OOS-Edge table (pooled out-of-sample, all new + prior candidates)",
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
        "_Script: `scripts/_oos_wf_leverage_short.py` (read-only research harness, no production changes)._  ",
        "_References: Frazzini & Pedersen (2014) JFE 111(1); Novy-Marx & Velikov (2022) "
        "'Betting Against Betting Against Beta' JFE; Jegadeesh & Titman (1993) J.Finance 48(1); "
        "Daniel & Moskowitz (2016) 'Momentum Crashes' JFE 122(2); Blitz, Huij & Martens (2011) "
        "J.Emp.Finance 18(3); Jegadeesh (1990) J.Finance 45(3); Asness, Frazzini & Pedersen (2019) "
        "'Quality Minus Junk' Rev.Acc.Studies; McLean & Pontiff (2016) J.Finance 71(1)._  ",
        "_Edge helpers: `src/assembled_core/qa/deflated_sharpe.py`, `src/assembled_core/qa/metrics.py`._  ",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(parts), encoding="utf-8")
    log.info("Report written -> %s", OUT_MD)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    log.info("Loading universe prices…")
    prices, tradeable = _load_universe_prices()

    all_results: dict[str, list[dict]] = {}
    all_edges: dict[str, dict] = {}
    pooled_spy_ref = None

    for mode in ALL_MODES:
        log.info("Running WF — mode=%s…", mode)
        results, pooled_s, pooled_b = _run_wf(prices, tradeable, mode)
        edge = _edge_metrics(pooled_s, pooled_b, results)
        all_results[mode] = results
        all_edges[mode] = edge
        if pooled_spy_ref is None and len(pooled_b.dropna()) > 0:
            pooled_spy_ref = pooled_b
        log.info(
            "[%s] pooled-OOS AnnSharpe %.2f | IR %.2f (t=%.2f) | DSR-prob %.2f (pass=%s) | beta %.2f",
            mode,
            edge["ann_sharpe"],
            edge["ir"],
            edge["ir_t"],
            edge["dsr_prob"],
            edge["dsr_pass"],
            edge["beta"],
        )

    spy_pooled_edge = (
        _spy_pooled_edge(pooled_spy_ref) if pooled_spy_ref is not None else {}
    )
    _write_report(all_results, all_edges, spy_pooled_edge, len(tradeable))

    log.info(
        "DONE. SPY pooled-OOS Sharpe %.2f. New L/S Sharpes: %s",
        spy_pooled_edge.get("ann_sharpe", float("nan")),
        {m: round(all_edges[m]["ann_sharpe"], 2) for m in LONG_SHORT_MODES},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
