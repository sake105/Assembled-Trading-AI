"""One-shot OOS Walk-Forward for a NEW Residual-Momentum strategy.

Writes docs/results/2026_05_residual_momentum_real_oos.md.

Usage:
    python scripts/_oos_wf_residual_momentum.py

Why this strategy (genuinely NEW vs. the repo):
    A prior falsification run (scripts/_oos_wf_lowvol_momentum.py) showed the
    TOTAL-return momentum sleeve beats SPY massively on CAGR (~46%) but FAILS
    risk-adjusted (Sharpe ~1.33 < SPY 1.40) because of large, beta-driven
    drawdowns (−36% in 2020, −25% in 2022). Residual Momentum (Blitz, Huij &
    Martens 2011) is the principled fix: strip out each stock's market beta via
    a rolling regression and rank on the IDIOSYNCRATIC (alpha) momentum only.
    Literature claims higher Sharpe and more robust OOS behaviour because the
    time-varying market/beta bets that cause the crashes are removed.
    Repo status: helper functions exist at
    src/assembled_core/features/residual_momentum.py but NO strategy or backtest
    consumes them — so this is a genuinely new, never-OOS-tested idea.

Design:
    - Universe: local offline cache via load_eod_prices(None); symbols with data
      <= 2018-01-31 and >= 500 bars. SPY excluded from trading (= market factor
      AND benchmark).
    - Per monthly rebalance at date T, strictly PIT (ref_idx = pos(T) - 1):
        * Regression window = last 252 bars ending at ref_idx (strictly < T).
        * Single-factor market model per stock: r_i = alpha_i + beta_i * r_SPY + e_i
          (closed-form OLS, vectorised across stocks).
        * Residual e_i over the window.
        * Formation signal = sum(e_i over last 126 bars, skipping most recent 21)
          / std(e_i over that window)  → standardised residual momentum.
        * Rank cross-sectionally, long TOP quintile, equal-weight, long-only.
    - Three modes for honest attribution:
        residual_mom — the strategy (beta-stripped momentum)
        total_mom    — plain total-return momentum, same formation window
                       (the control: does residualisation actually raise Sharpe?)
        eq_weight    — equal-weight universe (survivorship sanity baseline)
    - Benchmark: SPY buy-and-hold per fold.
    - WF: 252/252/252 (train/test/step), warmup prepended.
    - Costs: 10.75 bps per leg, 1-bar execution lag (no look-ahead).

NO changes to any strategy module, policy.yaml, or production file. Read-only on
price data. Falsification harness: beats SPY risk-adjusted or gets rejected.
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
log = logging.getLogger("oos_wf_residual_momentum")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW = 252
TEST_WINDOW = 252
STEP_SIZE = 252

REG_WINDOW = 252  # bars for rolling market-model regression
MOM_LOOKBACK = 126  # formation window (bars)
MOM_SKIP = 21  # skip most-recent month (short-term reversal)
REBAL_FREQ = "ME"
QUANTILE_SELECT = 0.20

COST_BPS = 10.75
INITIAL_CAPITAL = 100_000.0

MODES = ("residual_mom", "total_mom", "eq_weight")

OUT_MD = ROOT / "docs" / "results" / "2026_05_residual_momentum_real_oos.md"


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
# Metrics / benchmark / diagnostics
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


def _diagnostics(
    net_ret: pd.Series, spy_ret: pd.Series, pos_wide: pd.DataFrame
) -> dict:
    if len(net_ret) < 5:
        return dict(spy_corr=float("nan"), trades_per_year=0.0)
    common = net_ret.index.intersection(spy_ret.index)
    spy_corr = (
        float(np.corrcoef(net_ret[common], spy_ret[common])[0, 1])
        if len(common) > 5
        else float("nan")
    )
    n_years = len(net_ret) / 252.0
    test_pos = pos_wide.reindex(net_ret.index).fillna(0.0)
    weight_changes = test_pos.diff().abs().sum(axis=1)
    trade_events = (weight_changes > 1e-6).sum()
    trades_per_year = float(trade_events) / n_years if n_years > 0 else float("nan")
    return dict(spy_corr=spy_corr, trades_per_year=trades_per_year)


# ---------------------------------------------------------------------------
# Selection logic
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


def _select(
    trade_rets: pd.DataFrame,
    spy_rets: pd.Series,
    all_dates: pd.DatetimeIndex,
    rebal_date: pd.Timestamp,
    mode: str,
) -> list[str]:
    """PIT cross-sectional selection at rebal_date. Uses only data < rebal_date."""
    window_end_idx = all_dates.get_loc(rebal_date)
    ref_idx = window_end_idx - 1  # last fully-known bar strictly before rebal
    reg_start = ref_idx - REG_WINDOW + 1
    if reg_start < 0:
        return []

    # Regression window of returns (strictly pre-rebal): rows reg_start..ref_idx
    R = trade_rets.iloc[reg_start : ref_idx + 1]  # (W x N)
    x = spy_rets.iloc[reg_start : ref_idx + 1]  # (W,)

    # Drop symbols without enough real data in the window
    valid_cols = R.columns[R.notna().sum() >= int(REG_WINDOW * 0.8)].tolist()
    if len(valid_cols) < 5:
        return []
    R = R[valid_cols].fillna(0.0)
    xv = x.fillna(0.0).to_numpy()

    if mode == "eq_weight":
        return valid_cols

    Rv = R.to_numpy()  # (W x N)
    W = Rv.shape[0]

    # Formation window indices within the regression window (last 126, skip 21)
    form_end = W - MOM_SKIP
    form_start = form_end - MOM_LOOKBACK
    if form_start < 0:
        return []

    if mode == "residual_mom":
        # Vectorised single-factor market model: residual_t,i = r_t,i - (alpha_i + beta_i * x_t)
        x_mean = xv.mean()
        xc = xv - x_mean
        denom = float((xc * xc).sum())
        if denom < 1e-12:
            return []
        beta = (xc[:, None] * Rv).sum(axis=0) / denom  # (N,)
        alpha = Rv.mean(axis=0) - beta * x_mean  # (N,)
        resid = Rv - (alpha[None, :] + beta[None, :] * xv[:, None])  # (W x N)
        form = resid[form_start:form_end, :]  # (126 x N)
        sig = form.sum(axis=0)
        vol = form.std(axis=0)
    elif mode == "total_mom":
        form = Rv[form_start:form_end, :]
        sig = form.sum(axis=0)
        vol = form.std(axis=0)
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    score = pd.Series(sig / (vol + 1e-9), index=valid_cols)
    score = score[np.isfinite(score)]
    if score.empty:
        return []
    threshold = score.quantile(1.0 - QUANTILE_SELECT)
    return score[score >= threshold].index.tolist()


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
        selected = _select(trade_rets, spy_rets, all_dates, rebal_date, mode)
        weights = {s: 1.0 / len(selected) for s in selected} if selected else {}
        _apply_weights(pos_wide, all_dates, rebal_date, next_rebal, weights)

    pos_lag = pos_wide.shift(1).fillna(0.0)
    rets_aligned = trade_rets.reindex(columns=pos_lag.columns).fillna(0.0)
    port_ret_all = (pos_lag * rets_aligned).sum(axis=1)
    abs_delta = pos_lag.diff().fillna(0.0).abs().sum(axis=1)
    cost_all = abs_delta * COST_BPS / 10_000.0
    net_ret_all = port_ret_all - cost_all

    test_mask = (net_ret_all.index >= test_start) & (net_ret_all.index < test_end)
    net_ret = net_ret_all[test_mask]
    pos_wide_test = pos_wide[test_mask]
    if len(net_ret) < 5:
        raise ValueError(f"Only {len(net_ret)} test bars in fold")

    if pos_lag[test_mask].abs().sum().sum() < 1e-9:
        log.warning(
            "[%s] fold %s-%s: all-flat", mode, test_start.date(), test_end.date()
        )

    m = _metrics(net_ret)
    spy_test = spy_rets[test_mask]
    diag = _diagnostics(net_ret, spy_test, pos_wide_test)
    return m, diag, net_ret, pos_wide_test


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------
def _run_wf(prices, tradeable, mode) -> list[dict]:
    spy_dates = np.sort(prices[prices["symbol"] == "SPY"]["timestamp"].unique())
    results = []
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
            m, diag, net_ret, _ = _simulate(
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
                spy_corr=diag["spy_corr"],
                trades_per_year=diag["trades_per_year"],
                n_bars=len(net_ret),
                status="OK",
            )
            log.info(
                "[%s] Fold %d %s-%s: CAGR %.1f%% / Sharpe %.2f / MaxDD %.1f%%  (SPY: %.1f%% / %.2f / %.1f%%)",
                mode,
                fold_idx,
                test_start.date(),
                test_end.date(),
                m["cagr"] * 100,
                m["sharpe"],
                m["maxdd"] * 100,
                bm_spy["cagr"] * 100,
                bm_spy["sharpe"],
                bm_spy["maxdd"] * 100,
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
    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt(v, fmt="+.1%"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return format(v, fmt)


def _fv(row, key, fmt, suffix=""):
    v = row.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return format(v, fmt) + suffix


def _avg(rows, key):
    vals = [
        r[key] for r in rows if isinstance(r.get(key), float) and not np.isnan(r[key])
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _beat_count(rows, key, spy_key):
    n_ok = n_beat = 0
    for r in rows:
        if r.get("status") != "OK":
            continue
        v, sv = r.get(key), r.get(spy_key)
        if v is None or sv is None or np.isnan(v) or np.isnan(sv):
            continue
        n_ok += 1
        if v > sv:
            n_beat += 1
    return n_beat, n_ok


TABLE_HDR = (
    "| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe "
    "| SPY MaxDD | Corr(SPY) | Trades/yr |"
)
TABLE_SEP = (
    "|------|-------------|------|--------|-------|--------|----------|------------"
    "|-----------|-----------|-----------|"
)


def _table_row(r):
    if r.get("status") != "OK":
        return (
            f"| {r['fold']} | {r['test_start']}–{r['test_end']} "
            "| FAILED | — | — | — | — | — | — | — | — |"
        )
    return (
        f"| {r['fold']} | {r['test_start']}–{r['test_end']} "
        f"| {_fmt(r['cagr'])} | {_fv(r, 'sharpe', '+.2f')} | {_fmt(r['maxdd'])} "
        f"| {_fv(r, 'calmar', '+.2f')} | {_fmt(r['spy_cagr'])} | {_fv(r, 'spy_sharpe', '+.2f')} "
        f"| {_fmt(r['spy_maxdd'])} | {_fv(r, 'spy_corr', '+.2f')} "
        f"| {_fv(r, 'trades_per_year', '.0f')} |"
    )


def _avg_row(rows, n_total):
    n_ok = len(rows)
    if not rows:
        return f"| **Ø (0/{n_total})** | — | — | — | — | — | — | — | — | — | — |"
    return (
        f"| **Ø ({n_ok}/{n_total})** | — "
        f"| **{_fmt(_avg(rows, 'cagr'))}** | **{_avg(rows, 'sharpe'):+.2f}** "
        f"| **{_fmt(_avg(rows, 'maxdd'))}** | **{_avg(rows, 'calmar'):+.2f}** "
        f"| {_fmt(_avg(rows, 'spy_cagr'))} | {_avg(rows, 'spy_sharpe'):+.2f} "
        f"| {_fmt(_avg(rows, 'spy_maxdd'))} | {_avg(rows, 'spy_corr'):+.2f} "
        f"| {_avg(rows, 'trades_per_year'):.0f} |"
    )


def _mode_section(title, results):
    ok = [r for r in results if r.get("status") == "OK"]
    lines = [f"## {title}", "", TABLE_HDR, TABLE_SEP]
    for r in results:
        lines.append(_table_row(r))
    lines.append(_avg_row(ok, len(results)))
    lines.append("")
    return lines


def _write_report(all_results, n_tradeable):
    res = all_results["residual_mom"]
    ok = [r for r in res if r.get("status") == "OK"]

    avg_cagr = _avg(ok, "cagr")
    avg_sh = _avg(ok, "sharpe")
    avg_dd = _avg(ok, "maxdd")
    avg_cal = _avg(ok, "calmar")
    avg_spy_cagr = _avg(ok, "spy_cagr")
    avg_spy_sh = _avg(ok, "spy_sharpe")
    beat_cagr_n, n_cagr = _beat_count(ok, "cagr", "spy_cagr")
    beat_sh_n, n_sh = _beat_count(ok, "sharpe", "spy_sharpe")

    ok_total = [r for r in all_results["total_mom"] if r.get("status") == "OK"]
    total_sh = _avg(ok_total, "sharpe")
    total_cagr = _avg(ok_total, "cagr")

    lines = [
        "# Residual Momentum — OOS Walk-Forward Backtest (NEW strategy)",
        "",
        f"Run date (UTC): {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')}  ",
        "Data: local offline cache via `load_eod_prices(None)` — survivors only  ",
        f"Universe: {n_tradeable} tradeable symbols (data ≤ 2018-01-31, ≥ 500 bars; SPY excluded)  ",
        "Strategy: single-factor (market=SPY) residual momentum, top quintile, monthly, long-only, equal-weight  ",
        f"Market model: rolling OLS r_i = α_i + β_i·r_SPY + e_i over {REG_WINDOW} bars  ",
        f"Formation: Σ residual over last {MOM_LOOKBACK} bars skipping {MOM_SKIP}, standardised by residual vol  ",
        f"WF: {TRAIN_WINDOW}/{TEST_WINDOW}/{STEP_SIZE} (train/test/step)  ",
        f"Costs: {COST_BPS} bps per leg, 1-bar execution lag  ",
        "",
        "**Honesty note:** offline cache is survivors-only (no delisted names) → momentum-type",
        "signals are INFLATED (biggest losers that delisted are absent). Treat any outperformance",
        "as an OPTIMISTIC upper bound. CI status: not run in CI; local one-shot only.",
        "",
        "## Verdict (auto-generated)",
        "",
    ]

    if ok:
        beats_cagr = (
            (not np.isnan(avg_cagr))
            and (not np.isnan(avg_spy_cagr))
            and avg_cagr > avg_spy_cagr
        )
        beats_sharpe = (
            (not np.isnan(avg_sh))
            and (not np.isnan(avg_spy_sh))
            and avg_sh > avg_spy_sh
        )
        lines += [
            f"- Ø CAGR (residual_mom): {_fmt(avg_cagr)} vs SPY {_fmt(avg_spy_cagr)} "
            f"→ beats SPY CAGR in {beat_cagr_n}/{n_cagr} folds",
            f"- Ø Sharpe (residual_mom): {avg_sh:+.2f} vs SPY {avg_spy_sh:+.2f} "
            f"→ beats SPY Sharpe in {beat_sh_n}/{n_sh} folds",
            f"- Ø MaxDD: {_fmt(avg_dd)} | Ø Calmar: {avg_cal:+.2f}",
            f"- Control — total-return momentum Ø Sharpe {total_sh:+.2f} / Ø CAGR {_fmt(total_cagr)}: "
            f"residualisation {'RAISES' if avg_sh > total_sh else 'does NOT raise'} Sharpe "
            f"({avg_sh:+.2f} vs {total_sh:+.2f}).",
            "",
        ]
        if beats_sharpe or beats_cagr:
            metric = "Sharpe" if beats_sharpe else "CAGR"
            both = " and CAGR" if (beats_sharpe and beats_cagr) else ""
            lines.append(
                f"**PROSPECT** — beats SPY on {metric}{both} on the survivors-only universe. "
                "Needs survivorship-clean re-test before any further consideration (see honesty note)."
            )
        else:
            lines.append(
                "**REJECTED as irrelevant** — does NOT beat SPY risk-adjusted or absolute even on the "
                "survivorship-INFLATED offline universe. On a survivorship-clean universe it would be "
                "weaker still. No prospect; do not pursue."
            )
    else:
        lines.append("- No valid folds — inconclusive (data/harness issue).")
    lines.append("")

    lines += _mode_section(
        "Residual Momentum — THE STRATEGY", all_results["residual_mom"]
    )
    lines += _mode_section("Total-Return Momentum (control)", all_results["total_mom"])
    lines += _mode_section("Equal-Weight universe (baseline)", all_results["eq_weight"])

    lines += [
        "## Attribution (Ø across OK folds)",
        "",
        "| Mode | Ø CAGR | Ø Sharpe | Ø MaxDD | Ø Calmar |",
        "|------|--------|----------|---------|----------|",
    ]
    for mode in MODES:
        okm = [r for r in all_results[mode] if r.get("status") == "OK"]
        lines.append(
            f"| {mode} | {_fmt(_avg(okm, 'cagr'))} | {_avg(okm, 'sharpe'):+.2f} "
            f"| {_fmt(_avg(okm, 'maxdd'))} | {_avg(okm, 'calmar'):+.2f} |"
        )
    spy_ok = [r for r in all_results["residual_mom"] if r.get("status") == "OK"]
    lines.append(
        f"| **SPY (bench)** | {_fmt(_avg(spy_ok, 'spy_cagr'))} | {_avg(spy_ok, 'spy_sharpe'):+.2f} "
        f"| {_fmt(_avg(spy_ok, 'spy_maxdd'))} | — |"
    )
    lines += [
        "",
        "---",
        "_Script: `scripts/_oos_wf_residual_momentum.py` (read-only research harness, no production changes)_  ",
        "_Reference: Blitz, Huij & Martens (2011) 'Residual Momentum', J. Empirical Finance 18(3), 506-521._  ",
        "_Repo helpers (unused by any strategy): `src/assembled_core/features/residual_momentum.py`._",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written -> %s", OUT_MD)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Loading universe prices…")
    prices, tradeable = _load_universe_prices()

    all_results: dict[str, list[dict]] = {}
    for mode in MODES:
        log.info("Running WF — mode=%s…", mode)
        all_results[mode] = _run_wf(prices, tradeable, mode=mode)

    log.info("Writing report…")
    _write_report(all_results, len(tradeable))

    ok = [r for r in all_results["residual_mom"] if r.get("status") == "OK"]
    if ok:
        import statistics

        log.info(
            "RESIDUAL_MOM: %d/%d folds OK | Ø CAGR %.1f%% | Ø Sharpe %.2f | "
            "SPY Ø CAGR %.1f%% | SPY Ø Sharpe %.2f",
            len(ok),
            len(all_results["residual_mom"]),
            statistics.mean(r["cagr"] for r in ok) * 100,
            statistics.mean(r["sharpe"] for r in ok),
            statistics.mean(r["spy_cagr"] for r in ok if not np.isnan(r["spy_cagr"]))
            * 100,
            statistics.mean(
                r["spy_sharpe"] for r in ok if not np.isnan(r["spy_sharpe"])
            ),
        )

    print("Done ->", OUT_MD)
