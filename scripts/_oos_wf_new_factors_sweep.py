"""One-shot OOS Walk-Forward sweep of THREE genuinely-new long-only signals.

Writes docs/results/2026_05_new_factors_sweep_real_oos.md.

Usage:
    python scripts/_oos_wf_new_factors_sweep.py

Why these signals (each genuinely NEW vs. the repo — verified by repo scout):
    Prior falsification runs rejected Low-Vol+Momentum (Sharpe 1.30) and Residual
    Momentum (Sharpe 1.00) — both < SPY 1.40. Security-selection on total/residual
    return did not beat SPY risk-adjusted on this survivors-only universe. The next
    step is to test cross-sectional signals that are NOT a trailing-return rank:

    high52w  — George & Hwang (2004) "52-Week High and Momentum Investing":
               rank by close / trailing-252d-high (proximity to the 52-week high).
               Distinct from total-return momentum; documented to subsume it.
               Repo status: NO implementation exists.

    reversal_1m — Jegadeesh (1990) / Lehmann (1990) short-term reversal:
               buy last month's LOSERS (signal = −21d return). Contrarian, the
               OPPOSITE sign of momentum. Repo status: a dormant feature build
               exists (features/ta_factors_core.py reversal_1d/2d/3d) but NO
               strategy or signal generator consumes it → functionally new.

    low_beta — Frazzini & Pedersen (2014) "Betting Against Beta", long-only,
               no-leverage variant: rank by market beta (rolling OLS vs SPY),
               long the LOWEST-beta quintile. HONESTY: the full BAB lever low-beta
               UP to market beta (leverage) and shorts high-beta; this repo forbids
               leverage and the paper engine is long-only, so this is the
               low-beta TILT only — the deployable subset, weaker than full BAB.
               Repo status: NO BAB/low-beta strategy exists.

Design (identical PIT / cost / WF machinery to the reviewed sibling
scripts/_oos_wf_residual_momentum.py — only _select() differs):
    - Universe: load_eod_prices(None) survivors; data <= 2018-01-31 and >= 500
      bars; SPY excluded from trading (= market factor for low_beta AND benchmark).
    - Per monthly rebalance at date T, strictly PIT (ref_idx = pos(T) - 1):
      every signal uses only bars with index <= ref_idx (strictly < T). A second
      execution-lag bar is added downstream via pos_wide.shift(1).
    - Long TOP quintile, equal-weight, long-only.
    - Modes: high52w / reversal_1m / low_beta (strategies) + eq_weight (baseline).
    - Benchmark: SPY buy-and-hold per fold.
    - WF: 252/252/252 (train/test/step), warmup prepended.
    - Costs: 10.75 bps per leg, 1-bar execution lag (no look-ahead).

NO changes to any strategy module, policy.yaml, or production file. Read-only on
price data. Falsification harness: each signal beats SPY risk-adjusted or is rejected.
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
log = logging.getLogger("oos_wf_new_factors_sweep")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW = 252
TEST_WINDOW = 252
STEP_SIZE = 252

LOOKBACK = 252  # bars of history for 52w-high / beta windows
REV_LOOKBACK = 21  # short-term reversal formation (1 month)
REBAL_FREQ = "ME"
QUANTILE_SELECT = 0.20

COST_BPS = 10.75
INITIAL_CAPITAL = 100_000.0

STRATEGY_MODES = ("high52w", "reversal_1m", "low_beta")
MODES = STRATEGY_MODES + ("eq_weight",)

MODE_TITLES = {
    "high52w": "52-Week-High Momentum (George-Hwang 2004)",
    "reversal_1m": "1-Month Reversal (Jegadeesh 1990)",
    "low_beta": "Low-Beta Tilt (Frazzini-Pedersen 2014, long-only no-lev)",
    "eq_weight": "Equal-Weight universe (baseline)",
}

OUT_MD = ROOT / "docs" / "results" / "2026_05_new_factors_sweep_real_oos.md"


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
    trade_pivot: pd.DataFrame,
    trade_rets: pd.DataFrame,
    spy_rets: pd.Series,
    all_dates: pd.DatetimeIndex,
    rebal_date: pd.Timestamp,
    mode: str,
) -> list[str]:
    """PIT cross-sectional selection at rebal_date. Uses only data with idx <= ref_idx < rebal_date."""
    window_end_idx = all_dates.get_loc(rebal_date)
    ref_idx = window_end_idx - 1  # last fully-known bar strictly before rebal
    px_start = ref_idx - LOOKBACK + 1  # LOOKBACK price bars ending at ref_idx
    if px_start < 1:  # need >=1 prior bar so returns are defined
        return []

    px = trade_pivot.iloc[px_start : ref_idx + 1]  # (LOOKBACK x N), strictly pre-rebal
    valid_cols = px.columns[px.notna().sum() >= int(LOOKBACK * 0.8)].tolist()
    if len(valid_cols) < 5:
        return []
    px = px[valid_cols]

    if mode == "eq_weight":
        return valid_cols

    close_ref = px.iloc[-1]  # price at ref_idx

    if mode == "high52w":
        high52 = px.max(axis=0)
        score = close_ref / high52  # in (0, 1]; 1.0 == at the high

    elif mode == "reversal_1m":
        close_prior = px.iloc[
            -1 - REV_LOOKBACK
        ]  # price REV_LOOKBACK bars before ref_idx
        ret_1m = close_ref / close_prior - 1.0
        score = -ret_1m  # buy losers (most-negative return)

    elif mode == "low_beta":
        R = trade_rets.iloc[px_start : ref_idx + 1][valid_cols].fillna(0.0)
        x = spy_rets.iloc[px_start : ref_idx + 1].fillna(0.0).to_numpy()
        Rv = R.to_numpy()  # (W x N)
        xc = x - x.mean()
        denom = float((xc * xc).sum())
        if denom < 1e-12:
            return []
        beta = (xc[:, None] * Rv).sum(axis=0) / denom  # (N,)
        score = pd.Series(-beta, index=valid_cols)  # lowest beta == highest score

    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if not isinstance(score, pd.Series):
        score = pd.Series(score, index=valid_cols)
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
        selected = _select(
            trade_pivot, trade_rets, spy_rets, all_dates, rebal_date, mode
        )
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


def _verdict_line(mode, results) -> tuple[str, bool]:
    """Return (verdict markdown line, beats_spy bool) for one strategy mode."""
    ok = [r for r in results if r.get("status") == "OK"]
    if not ok:
        return f"- **{mode}**: no valid folds — inconclusive.", False
    avg_cagr = _avg(ok, "cagr")
    avg_sh = _avg(ok, "sharpe")
    avg_spy_cagr = _avg(ok, "spy_cagr")
    avg_spy_sh = _avg(ok, "spy_sharpe")
    beat_cagr_n, n_cagr = _beat_count(ok, "cagr", "spy_cagr")
    beat_sh_n, n_sh = _beat_count(ok, "sharpe", "spy_sharpe")
    beats_cagr = (
        (not np.isnan(avg_cagr))
        and (not np.isnan(avg_spy_cagr))
        and avg_cagr > avg_spy_cagr
    )
    beats_sharpe = (
        (not np.isnan(avg_sh)) and (not np.isnan(avg_spy_sh)) and avg_sh > avg_spy_sh
    )
    beats = bool(beats_cagr or beats_sharpe)
    tag = "PROSPECT" if beats else "REJECTED"
    return (
        f"- **{mode}** [{tag}]: Ø CAGR {_fmt(avg_cagr)} vs SPY {_fmt(avg_spy_cagr)} "
        f"(beats {beat_cagr_n}/{n_cagr}); Ø Sharpe {avg_sh:+.2f} vs SPY {avg_spy_sh:+.2f} "
        f"(beats {beat_sh_n}/{n_sh}); Ø MaxDD {_fmt(_avg(ok, 'maxdd'))}.",
        beats,
    )


def _write_report(all_results, n_tradeable):
    lines = [
        "# New-Factor Sweep — OOS Walk-Forward Backtest (3 NEW long-only signals)",
        "",
        f"Run date (UTC): {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d')}  ",
        "Data: local offline cache via `load_eod_prices(None)` — survivors only  ",
        f"Universe: {n_tradeable} tradeable symbols (data ≤ 2018-01-31, ≥ 500 bars; SPY excluded)  ",
        "Signals (each top-quintile, monthly, long-only, equal-weight):  ",
        "- **high52w** — close / trailing-252d-high (George-Hwang 2004)  ",
        f"- **reversal_1m** — −(last {REV_LOOKBACK}-bar return), buy losers (Jegadeesh 1990)  ",
        "- **low_beta** — long lowest-beta quintile, rolling 252d OLS vs SPY "
        "(Frazzini-Pedersen 2014, long-only NO-leverage tilt only)  ",
        f"WF: {TRAIN_WINDOW}/{TEST_WINDOW}/{STEP_SIZE} (train/test/step)  ",
        f"Costs: {COST_BPS} bps per leg, 1-bar execution lag  ",
        "",
        "**Honesty note:** offline cache is survivors-only (no delisted names) → all of",
        "these signals are INFLATED to some degree (the worst names that delisted are",
        "absent; this especially flatters reversal, which buys losers). Treat any",
        "outperformance as an OPTIMISTIC upper bound. The low_beta mode is the long-only",
        "NO-leverage subset of Betting-Against-Beta, structurally weaker than the levered",
        "long/short original. CI status: not run in CI; local one-shot only.",
        "",
        "## Verdict (auto-generated)",
        "",
    ]

    any_prospect = False
    for mode in STRATEGY_MODES:
        line, beats = _verdict_line(mode, all_results[mode])
        lines.append(line)
        any_prospect = any_prospect or beats
    lines.append("")
    if any_prospect:
        lines.append(
            "**AT LEAST ONE PROSPECT** — a signal beats SPY on CAGR or Sharpe on the "
            "survivors-only universe. Any such hit needs a survivorship-clean re-test before "
            "any further consideration (see honesty note); it is NOT yet a production claim."
        )
    else:
        lines.append(
            "**ALL THREE REJECTED as irrelevant** — none beats SPY risk-adjusted or absolute "
            "even on the survivorship-INFLATED offline universe. On a survivorship-clean "
            "universe they would be weaker still. No prospect; do not pursue."
        )
    lines.append("")

    for mode in MODES:
        lines += _mode_section(MODE_TITLES[mode], all_results[mode])

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
    spy_ok = [r for r in all_results["eq_weight"] if r.get("status") == "OK"]
    lines.append(
        f"| **SPY (bench)** | {_fmt(_avg(spy_ok, 'spy_cagr'))} | {_avg(spy_ok, 'spy_sharpe'):+.2f} "
        f"| {_fmt(_avg(spy_ok, 'spy_maxdd'))} | — |"
    )
    lines += [
        "",
        "---",
        "_Script: `scripts/_oos_wf_new_factors_sweep.py` (read-only research harness, no production changes)_  ",
        "_References: George & Hwang (2004) J. Finance 59(5); Jegadeesh (1990) J. Finance 45(3); "
        "Frazzini & Pedersen (2014) 'Betting Against Beta', J. Financial Economics 111(1)._  ",
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

    import statistics

    for mode in STRATEGY_MODES:
        ok = [r for r in all_results[mode] if r.get("status") == "OK"]
        if ok:
            log.info(
                "%s: %d/%d folds OK | Ø CAGR %.1f%% | Ø Sharpe %.2f | SPY Ø CAGR %.1f%% | SPY Ø Sharpe %.2f",
                mode.upper(),
                len(ok),
                len(all_results[mode]),
                statistics.mean(r["cagr"] for r in ok) * 100,
                statistics.mean(r["sharpe"] for r in ok),
                statistics.mean(
                    r["spy_cagr"] for r in ok if not np.isnan(r["spy_cagr"])
                )
                * 100,
                statistics.mean(
                    r["spy_sharpe"] for r in ok if not np.isnan(r["spy_sharpe"])
                ),
            )

    print("Done ->", OUT_MD)
