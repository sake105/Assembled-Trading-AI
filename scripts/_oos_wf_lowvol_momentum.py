"""One-shot OOS Walk-Forward for a NEW Cross-Sectional Low-Vol + Momentum strategy.

Writes docs/results/2026_05_lowvol_momentum_real_oos.md.

Usage:
    python scripts/_oos_wf_lowvol_momentum.py

Why this strategy (genuinely NEW vs. the repo):
    The repo has trend_baseline (per-asset time-series trend), multifactor_v2
    (34-factor kitchen-sink with risk overlays), dual_momentum (asset-class
    absolute/relative momentum), low_max_lottery (MAX effect) and a *portfolio-level*
    vol_target_overlay. None of those is a clean, isolable CROSS-SECTIONAL
    stock-selection portfolio that combines the low-volatility anomaly with
    cross-sectional momentum. That combination ("defensive momentum") has the
    strongest risk-adjusted literature support of the candidate ideas, so it is
    worth a direct falsification test rather than more reading.

Design:
    - Universe: all symbols in local offline cache (load_eod_prices(None)),
      data from <= 2018-01-31, >= 500 bars. SPY excluded from trading (benchmark only).
    - Per monthly rebalance, for each symbol compute (strictly PIT, pre-rebal data):
        * vol60   = std of last 60 daily returns           (low-vol anomaly)
        * mom6-1  = price[t-21] / price[t-147] - 1          (126d momentum, skip 21d)
      Convert each to a cross-sectional percentile rank:
        * lowvol_score = 1 - rank_pct(vol60)               (lower vol => higher score)
        * mom_score    = rank_pct(mom6_1)                  (higher mom => higher score)
        * combo        = 0.5*lowvol_score + 0.5*mom_score
      Go long the TOP quintile by the mode's score, equal-weight, long-only.
    - Four modes for honest attribution:
        combo     — the strategy (low-vol + momentum)
        lowvol    — low-vol sleeve alone (bottom-vol quintile)
        momentum  — momentum sleeve alone (top-momentum quintile)
        eq_weight — equal-weight whole universe (baseline / survivorship sanity)
    - Benchmark: SPY buy-and-hold per fold.
    - Walk-forward: 252/252/252 (train/test/step), warmup prepended.
    - Costs: 10.75 bps per leg (one-sided), 1-bar execution lag (no look-ahead).

NO changes to any strategy module, policy.yaml, or production files. Read-only
on price data. This is a falsification harness: it either beats SPY risk-adjusted
or it gets rejected as irrelevant.
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
log = logging.getLogger("oos_wf_lowvol_momentum")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW = 252
TEST_WINDOW = 252
STEP_SIZE = 252

VOL_LOOKBACK = 60  # bars for realized-vol estimate
MOM_LOOKBACK = 126  # bars for momentum formation window
MOM_SKIP = 21  # skip most-recent month (short-term reversal)
REBAL_FREQ = "ME"  # month-end rebalancing
QUANTILE_SELECT = 0.20  # top 20% (quintile)

COST_BPS = 10.75  # bps per leg (one-sided)
INITIAL_CAPITAL = 100_000.0

MODES = ("combo", "lowvol", "momentum", "eq_weight")

OUT_MD = ROOT / "docs" / "results" / "2026_05_lowvol_momentum_real_oos.md"


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------
def _load_universe_prices() -> tuple[pd.DataFrame, list[str]]:
    """Load all available symbols; return (prices_df, tradeable_symbols_list)."""
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
# Helpers
# ---------------------------------------------------------------------------
def _metrics(net_ret: pd.Series) -> dict:
    """CAGR / Sharpe / MaxDD / Calmar from a daily net-return series."""
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


def _benchmark_spy(
    prices: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> dict:
    spy = prices[prices["symbol"] == "SPY"].set_index("timestamp")["close"]
    spy = spy[(spy.index >= test_start) & (spy.index < test_end)].sort_index()
    if len(spy) < 5:
        return dict(cagr=float("nan"), sharpe=float("nan"), maxdd=float("nan"))
    r = spy.pct_change().dropna()
    return _metrics(r)


def _diagnostics(
    net_ret: pd.Series, spy_ret: pd.Series, pos_wide: pd.DataFrame
) -> dict:
    """SPY correlation, trades/year."""
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
# Core simulation
# ---------------------------------------------------------------------------
def _apply_weights(
    pos_wide: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    rebal_date: pd.Timestamp,
    next_rebal: pd.Timestamp,
    weights: dict[str, float],
) -> None:
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
    all_dates: pd.DatetimeIndex,
    rebal_date: pd.Timestamp,
    mode: str,
) -> list[str]:
    """PIT cross-sectional selection at rebal_date for the given mode.

    Uses only data strictly before rebal_date (ref_idx = pos(rebal_date) - 1).
    Returns the list of selected symbols (top quintile by the mode's score).
    """
    window_end_idx = all_dates.get_loc(rebal_date)
    ref_idx = window_end_idx - 1  # last fully-known bar before acting
    if ref_idx < 0:
        return []

    # --- Low-vol score ---
    vol_start = max(0, ref_idx - VOL_LOOKBACK + 1)
    vol_window = trade_rets.iloc[vol_start : ref_idx + 1]
    vol = vol_window.std()
    vol_valid = vol_window.notna().sum() >= max(20, VOL_LOOKBACK // 2)
    vol = vol[vol_valid & vol.notna() & (vol > 0)]

    # --- Momentum score (126d, skip last 21d) ---
    mom_recent_idx = ref_idx - MOM_SKIP
    mom_old_idx = mom_recent_idx - MOM_LOOKBACK
    if mom_old_idx < 0:
        # not enough history for momentum; fall back to vol-only universe
        if mode in ("momentum", "combo"):
            return []
    mom = pd.Series(dtype=float)
    if mom_old_idx >= 0:
        p_recent = trade_pivot.iloc[mom_recent_idx]
        p_old = trade_pivot.iloc[mom_old_idx]
        mom = (p_recent / p_old) - 1.0
        mom = mom[mom.notna() & np.isfinite(mom)]

    if mode == "eq_weight":
        selected_pool = vol.index.tolist()
        if not selected_pool:
            return []
        return selected_pool

    if mode == "lowvol":
        if vol.empty:
            return []
        score = 1.0 - vol.rank(pct=True)  # low vol => high score
    elif mode == "momentum":
        if mom.empty:
            return []
        score = mom.rank(pct=True)
    elif mode == "combo":
        common = vol.index.intersection(mom.index)
        if len(common) < 5:
            return []
        lowvol_score = 1.0 - vol[common].rank(pct=True)
        mom_score = mom[common].rank(pct=True)
        score = 0.5 * lowvol_score + 0.5 * mom_score
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if score.empty:
        return []
    threshold = score.quantile(1.0 - QUANTILE_SELECT)
    selected = score[score >= threshold].index.tolist()
    return selected


def _simulate(
    prices: pd.DataFrame,
    tradeable: list[str],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    mode: str,
) -> tuple[dict, dict, pd.Series, pd.DataFrame]:
    """Simulate one WF fold for a given mode."""
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

    trade_pivot = pivot[trade_cols]
    trade_rets = trade_pivot.pct_change()

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
        selected = _select(trade_pivot, trade_rets, all_dates, rebal_date, mode)
        if not selected:
            current_weights: dict[str, float] = {}
        else:
            w = 1.0 / len(selected)
            current_weights = {s: w for s in selected}
        _apply_weights(pos_wide, all_dates, rebal_date, next_rebal, current_weights)

    # 1-bar execution lag (no look-ahead)
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
            "[%s] fold %s-%s: all-flat (no positions)",
            mode,
            test_start.date(),
            test_end.date(),
        )

    m = _metrics(net_ret)

    spy_rets_full = (
        pivot["SPY"].pct_change() if "SPY" in pivot.columns else pd.Series(dtype=float)
    )
    spy_test = (
        spy_rets_full[test_mask] if len(spy_rets_full) > 0 else pd.Series(dtype=float)
    )
    diag = _diagnostics(net_ret, spy_test, pos_wide_test)

    return m, diag, net_ret, pos_wide_test


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------
def _run_wf(prices: pd.DataFrame, tradeable: list[str], mode: str) -> list[dict]:
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
            m, diag, net_ret, pos_wide_test = _simulate(
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
                "[%s] Fold %d %s-%s: CAGR %.1f%% / Sharpe %.2f / MaxDD %.1f%%  "
                "(SPY: %.1f%% / %.2f / %.1f%%)",
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


def _fv(row: dict, key: str, fmt: str, suffix: str = "") -> str:
    v = row.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return format(v, fmt) + suffix


def _avg(rows: list[dict], key: str) -> float:
    vals = [
        r[key] for r in rows if isinstance(r.get(key), float) and not np.isnan(r[key])
    ]
    return float(np.mean(vals)) if vals else float("nan")


def _beat_count(rows: list[dict], key: str, spy_key: str) -> tuple[int, int]:
    """How many folds beat SPY on the given metric."""
    n_ok = 0
    n_beat = 0
    for r in rows:
        if r.get("status") != "OK":
            continue
        v = r.get(key)
        sv = r.get(spy_key)
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


def _table_row(r: dict) -> str:
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


def _avg_row(rows: list[dict], n_total: int) -> str:
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


def _mode_section(title: str, results: list[dict]) -> list[str]:
    ok = [r for r in results if r.get("status") == "OK"]
    lines = [f"## {title}", "", TABLE_HDR, TABLE_SEP]
    for r in results:
        lines.append(_table_row(r))
    lines.append(_avg_row(ok, len(results)))
    lines.append("")
    return lines


def _write_report(all_results: dict[str, list[dict]], n_tradeable: int) -> None:
    combo = all_results["combo"]
    ok_combo = [r for r in combo if r.get("status") == "OK"]

    avg_cagr = _avg(ok_combo, "cagr")
    avg_sh = _avg(ok_combo, "sharpe")
    avg_dd = _avg(ok_combo, "maxdd")
    avg_cal = _avg(ok_combo, "calmar")
    avg_spy_cagr = _avg(ok_combo, "spy_cagr")
    avg_spy_sh = _avg(ok_combo, "spy_sharpe")

    beat_cagr_n, n_cagr = _beat_count(ok_combo, "cagr", "spy_cagr")
    beat_sh_n, n_sh = _beat_count(ok_combo, "sharpe", "spy_sharpe")

    lines = [
        "# Cross-Sectional Low-Vol + Momentum — OOS Walk-Forward Backtest (NEW strategy)",
        "",
        f"Run date: {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
        "Data: local offline cache via `load_eod_prices(None)` — survivors only  ",
        f"Universe: {n_tradeable} tradeable symbols (data ≤ 2018-01-31, ≥ 500 bars; SPY excluded)  ",
        "Strategy: top-quintile by 0.5·lowvol_rank + 0.5·momentum_rank, monthly, long-only, equal-weight  ",
        f"Low-vol: std of last {VOL_LOOKBACK} daily returns  ",
        f"Momentum: price[t-{MOM_SKIP}] / price[t-{MOM_SKIP + MOM_LOOKBACK}] − 1 ({MOM_LOOKBACK}d formation, skip {MOM_SKIP}d)  ",
        f"WF: {TRAIN_WINDOW}/{TEST_WINDOW}/{STEP_SIZE} (train/test/step)  ",
        f"Costs: {COST_BPS} bps per leg (one-sided), 1-bar execution lag  ",
        "",
        "**Honesty note:** offline cache is survivors-only (no delisted names). This *inflates*",
        "the momentum sleeve (the biggest losers that delisted are absent) and the whole universe.",
        "Treat any outperformance here as an OPTIMISTIC upper bound, not a production claim.",
        "CI status: not run in CI; local one-shot only.",
        "",
        "## Verdict (auto-generated)",
        "",
    ]

    if ok_combo:
        verdict_lines = [
            f"- Ø CAGR (combo): {_fmt(avg_cagr)} vs SPY {_fmt(avg_spy_cagr)} "
            f"→ beats SPY CAGR in {beat_cagr_n}/{n_cagr} folds",
            f"- Ø Sharpe (combo): {avg_sh:+.2f} vs SPY {avg_spy_sh:+.2f} "
            f"→ beats SPY Sharpe in {beat_sh_n}/{n_sh} folds",
            f"- Ø MaxDD (combo): {_fmt(avg_dd)} | Ø Calmar: {avg_cal:+.2f}",
            "",
        ]
        beats_cagr = (
            (not np.isnan(avg_cagr))
            and (not np.isnan(avg_spy_cagr))
            and (avg_cagr > avg_spy_cagr)
        )
        beats_sharpe = (
            (not np.isnan(avg_sh))
            and (not np.isnan(avg_spy_sh))
            and (avg_sh > avg_spy_sh)
        )
        if beats_sharpe or beats_cagr:
            verdict = (
                "**PROSPECT** — beats SPY on "
                + ("Sharpe" if beats_sharpe else "")
                + (" and CAGR" if (beats_sharpe and beats_cagr) else "")
                + ("CAGR" if (beats_cagr and not beats_sharpe) else "")
                + " on the survivors-only universe. Needs survivorship-clean re-test before "
                "any further consideration (see honesty note)."
            )
        else:
            verdict = (
                "**REJECTED as irrelevant** — does NOT beat SPY risk-adjusted or absolute "
                "even on the survivorship-INFLATED offline universe. On a survivorship-clean "
                "universe it would be weaker still. No prospect; do not pursue."
            )
        verdict_lines.append(verdict)
        lines += verdict_lines
    else:
        lines.append("- No valid folds — inconclusive (data/harness issue).")
    lines.append("")

    lines += _mode_section(
        "Combo (Low-Vol + Momentum) — THE STRATEGY", all_results["combo"]
    )
    lines += _mode_section("Low-Vol sleeve alone", all_results["lowvol"])
    lines += _mode_section("Momentum sleeve alone", all_results["momentum"])
    lines += _mode_section("Equal-Weight universe (baseline)", all_results["eq_weight"])

    lines += [
        "## Attribution (Ø across OK folds)",
        "",
        "| Mode | Ø CAGR | Ø Sharpe | Ø MaxDD | Ø Calmar |",
        "|------|--------|----------|---------|----------|",
    ]
    for mode in MODES:
        ok = [r for r in all_results[mode] if r.get("status") == "OK"]
        lines.append(
            f"| {mode} | {_fmt(_avg(ok, 'cagr'))} | {_avg(ok, 'sharpe'):+.2f} "
            f"| {_fmt(_avg(ok, 'maxdd'))} | {_avg(ok, 'calmar'):+.2f} |"
        )
    spy_ok = [r for r in all_results["combo"] if r.get("status") == "OK"]
    lines.append(
        f"| **SPY (bench)** | {_fmt(_avg(spy_ok, 'spy_cagr'))} | {_avg(spy_ok, 'spy_sharpe'):+.2f} "
        f"| {_fmt(_avg(spy_ok, 'spy_maxdd'))} | — |"
    )
    lines += [
        "",
        "---",
        "_Script: `scripts/_oos_wf_lowvol_momentum.py` (read-only research harness, no production changes)_  ",
        "_Low-vol anomaly: Baker/Bradley/Wurgler (2011); Frazzini/Pedersen 'Betting Against Beta' (2014)._  ",
        "_Momentum: Jegadeesh/Titman (1993); 12-1 / 6-1 formation. Combination = 'defensive momentum'._",
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

    ok_combo = [r for r in all_results["combo"] if r.get("status") == "OK"]
    if ok_combo:
        import statistics

        log.info(
            "COMBO: %d/%d folds OK | Ø CAGR %.1f%% | Ø Sharpe %.2f | "
            "SPY Ø CAGR %.1f%% | SPY Ø Sharpe %.2f",
            len(ok_combo),
            len(all_results["combo"]),
            statistics.mean(r["cagr"] for r in ok_combo) * 100,
            statistics.mean(r["sharpe"] for r in ok_combo),
            statistics.mean(
                r["spy_cagr"] for r in ok_combo if not np.isnan(r["spy_cagr"])
            )
            * 100,
            statistics.mean(
                r["spy_sharpe"] for r in ok_combo if not np.isnan(r["spy_sharpe"])
            ),
        )

    print("Done ->", OUT_MD)
