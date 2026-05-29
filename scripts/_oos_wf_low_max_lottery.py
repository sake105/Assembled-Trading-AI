"""One-shot OOS Walk-Forward for Low-MAX / Lottery-Avoidance Strategy.

Writes docs/results/2026_05_low_max_lottery_real_oos.md.

Usage:
    python scripts/_oos_wf_low_max_lottery.py

Design:
    - Strategy: MAX Effect (Bali, Cakici & Whitelaw 2011).
      MAX = maximum daily return over last 20 trading days (1 month).
      Sort all tradeable stocks by MAX; go long the bottom quintile (low-MAX),
      compare against top quintile (high-MAX) and equal-weight universe.
      Monthly rebalancing. Long-only throughout.
    - Data: all available symbols in local Alpaca daily-bar cache.
      Filter: data available from 2018-01-31 or earlier, >= 500 bars.
      SPY excluded from trading universe; used only as benchmark.
    - Three portfolios per fold:
        low_max   — bottom quintile by MAX (strategy)
        high_max  — top quintile by MAX (comparison)
        eq_weight — equal-weight of all valid stocks (baseline)
    - Benchmarks: SPY Buy-and-Hold, 60/40 SPY/TLT daily-rebalanced.
    - Walk-forward: 252/252/252 (train / test / step).
      Warmup = train window prepended to each test period.
    - Transaction costs: 10.75 bps per leg, applied as |Δweight| per bar.
    - Key metric: MAX-Spread = CAGR(low_max) − CAGR(high_max) per fold.

KEINE Änderungen an strategy, policy.yaml, oder anderen Produktionsdateien.
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
log = logging.getLogger("oos_wf_low_max_lottery")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW = 252
TEST_WINDOW = 252
STEP_SIZE = 252

MAX_LOOKBACK = 20  # bars for rolling MAX (1 calendar month approx)
REBAL_FREQ = "ME"  # month-end rebalancing (pandas offset alias)
QUANTILE_SELECT = 0.20  # top/bottom 20% (quintile)

COST_BPS = 10.75  # bps per leg (one-sided)
INITIAL_CAPITAL = 100_000.0

OUT_MD = ROOT / "docs" / "results" / "2026_05_low_max_lottery_real_oos.md"


# ---------------------------------------------------------------------------
# Universe loading
# ---------------------------------------------------------------------------
def _load_universe_prices() -> tuple[pd.DataFrame, list[str]]:
    """Load all available symbols; return (prices_df, tradeable_symbols_list)."""
    from src.assembled_core.data.prices_ingest import load_eod_prices

    prices = load_eod_prices(None)
    if prices is None or prices.empty:
        raise RuntimeError("No price data returned from load_eod_prices")

    # Filter to broad window around study period (+ warmup buffer)
    prices = prices[
        (prices["timestamp"] >= PERIOD_START - pd.Timedelta(days=60))
        & (prices["timestamp"] <= PERIOD_END + pd.Timedelta(days=5))
    ].copy()

    # Filter: data available from 2018-01-31 or earlier AND >= 500 bars
    sym_info = prices.groupby("symbol")["timestamp"].agg(["min", "count"])
    valid = sym_info[
        (sym_info["min"] <= pd.Timestamp("2018-01-31", tz="UTC"))
        & (sym_info["count"] >= 500)
    ]
    tradeable = [s for s in valid.index if s != "SPY"]
    log.info(
        "Loaded %d rows for %d total symbols → %d tradeable (excl SPY, %s → %s)",
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


def _benchmark_6040(
    prices: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> dict:
    """60/40 SPY/TLT (daily-rebalanced)."""
    pivot = (
        prices[prices["symbol"].isin(["SPY", "TLT"])]
        .pivot_table(index="timestamp", columns="symbol", values="close")
        .ffill()
    )
    test = pivot[(pivot.index >= test_start) & (pivot.index < test_end)].dropna()
    if len(test) < 5:
        return dict(cagr=float("nan"), sharpe=float("nan"), maxdd=float("nan"))
    r = test.pct_change().dropna()
    if "SPY" not in r.columns or "TLT" not in r.columns:
        return dict(cagr=float("nan"), sharpe=float("nan"), maxdd=float("nan"))
    port_r = 0.6 * r["SPY"] + 0.4 * r["TLT"]
    return _metrics(port_r)


def _diagnostics(
    net_ret: pd.Series, spy_ret: pd.Series, pos_wide: pd.DataFrame
) -> dict:
    """SPY correlation, trades/year, avg-hold-days, win-rate."""
    if len(net_ret) < 5:
        return dict(
            spy_corr=float("nan"),
            trades_per_year=0.0,
            avg_hold_days=float("nan"),
            win_rate=float("nan"),
        )

    # SPY correlation
    common = net_ret.index.intersection(spy_ret.index)
    spy_corr = (
        float(np.corrcoef(net_ret[common], spy_ret[common])[0, 1])
        if len(common) > 5
        else float("nan")
    )

    # Trades/year: bars where any weight changes
    n_years = len(net_ret) / 252.0
    test_pos = pos_wide.reindex(net_ret.index).fillna(0.0)
    weight_changes = test_pos.diff().abs().sum(axis=1)
    trade_events = (weight_changes > 1e-6).sum()
    trades_per_year = float(trade_events) / n_years if n_years > 0 else float("nan")

    # Avg hold duration
    active = (test_pos.abs().sum(axis=1) > 1e-6).astype(int)
    run_lengths: list[int] = []
    current_run = 0
    for v in active:
        if v == 1:
            current_run += 1
        elif current_run > 0:
            run_lengths.append(current_run)
            current_run = 0
    if current_run > 0:
        run_lengths.append(current_run)
    avg_hold = float(np.mean(run_lengths)) if run_lengths else float("nan")

    # Win-rate on active days
    active_rets = net_ret[active.astype(bool)]
    win_rate = float((active_rets > 0).mean()) if len(active_rets) > 0 else float("nan")

    return dict(
        spy_corr=spy_corr,
        trades_per_year=trades_per_year,
        avg_hold_days=avg_hold,
        win_rate=win_rate,
    )


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------
def _simulate(
    prices: pd.DataFrame,
    tradeable: list[str],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    mode: str,  # "low_max" | "high_max" | "eq_weight"
) -> tuple[dict, dict, pd.Series, pd.DataFrame]:
    """Simulate one WF fold for a given mode.

    Returns:
        (metrics, diagnostics, net_ret_series, pos_wide_test)
    """
    # Reference calendar from SPY (clean, dense, no gaps)
    spy_dates = np.sort(prices[prices["symbol"] == "SPY"]["timestamp"].unique())

    # Warmup: prepend TRAIN_WINDOW bars before test_start
    pre_test = spy_dates[spy_dates < test_start]
    warmup_start = (
        pre_test[-TRAIN_WINDOW]
        if len(pre_test) >= TRAIN_WINDOW
        else (pre_test[0] if len(pre_test) > 0 else test_start)
    )

    # Load price window (warmup + test) for tradeable + SPY
    syms_needed = list(set(tradeable) | {"SPY", "TLT"})
    window_prices = prices[
        prices["symbol"].isin(syms_needed)
        & (prices["timestamp"] >= warmup_start)
        & (prices["timestamp"] < test_end)
    ].copy()

    # Pivot close prices
    pivot = (
        window_prices.pivot_table(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .ffill()
    )

    all_dates = pivot.index

    # Restrict to tradeable columns only (SPY/TLT not traded, only benchmarked)
    trade_cols = [c for c in tradeable if c in pivot.columns]
    if not trade_cols:
        raise ValueError(f"No tradeable symbols in pivot for {test_start.date()}")

    trade_pivot = pivot[trade_cols]
    trade_rets = trade_pivot.pct_change()

    # Monthly rebalancing dates within the combined window
    rebal_dates = pd.date_range(
        start=warmup_start, end=test_end, freq=REBAL_FREQ, tz="UTC"
    )
    # Only keep rebal dates that actually exist in our index (or nearest prior)
    rebal_idx = []
    for rd in rebal_dates:
        candidates = all_dates[all_dates <= rd]
        if len(candidates) > 0:
            rebal_idx.append(candidates[-1])
    rebal_idx = sorted(set(rebal_idx))

    log.info(
        "[%s] fold %s–%s: warmup %s, %d syms, %d rebal dates",
        mode,
        test_start.date(),
        test_end.date(),
        warmup_start.date(),
        len(trade_cols),
        len(rebal_idx),
    )

    # Build position matrix (fractional weights, equal-weight within selected bucket)
    pos_wide = pd.DataFrame(0.0, index=all_dates, columns=trade_cols)

    current_weights: dict[str, float] = {}

    for i, rebal_date in enumerate(rebal_idx):
        next_rebal = rebal_idx[i + 1] if i + 1 < len(rebal_idx) else test_end

        # Compute MAX for each symbol up to (not including) rebal_date (PIT)
        # MAX = max daily return over last MAX_LOOKBACK days
        window_end_idx = all_dates.get_loc(rebal_date)
        window_start_idx = max(0, window_end_idx - MAX_LOOKBACK)
        ret_window = trade_rets.iloc[window_start_idx:window_end_idx]

        # Require at least 5 bars of return data per symbol
        valid_syms = ret_window.columns[ret_window.notna().sum() >= 5].tolist()
        if not valid_syms:
            current_weights = {}
            _apply_weights(pos_wide, all_dates, rebal_date, next_rebal, current_weights)
            continue

        max_vals = ret_window[valid_syms].max()

        if mode == "eq_weight":
            selected = valid_syms
        elif mode == "low_max":
            threshold = max_vals.quantile(QUANTILE_SELECT)
            selected = max_vals[max_vals <= threshold].index.tolist()
        elif mode == "high_max":
            threshold = max_vals.quantile(1.0 - QUANTILE_SELECT)
            selected = max_vals[max_vals >= threshold].index.tolist()
        else:
            raise ValueError(f"Unknown mode: {mode!r}")

        if not selected:
            current_weights = {}
        else:
            w = 1.0 / len(selected)
            current_weights = {s: w for s in selected}

        _apply_weights(pos_wide, all_dates, rebal_date, next_rebal, current_weights)

    # 1-bar execution lag (no look-ahead)
    pos_lag = pos_wide.shift(1).fillna(0.0)

    # Portfolio return
    rets_aligned = trade_rets.reindex(columns=pos_lag.columns).fillna(0.0)
    port_ret_all = (pos_lag * rets_aligned).sum(axis=1)

    # Transaction costs at execution bar
    abs_delta = pos_lag.diff().fillna(0.0).abs().sum(axis=1)
    cost_all = abs_delta * COST_BPS / 10_000.0

    net_ret_all = port_ret_all - cost_all

    # Slice to test period
    test_mask = (net_ret_all.index >= test_start) & (net_ret_all.index < test_end)
    net_ret = net_ret_all[test_mask]
    pos_wide_test = pos_wide[test_mask]

    if len(net_ret) < 5:
        raise ValueError(f"Only {len(net_ret)} test bars in fold")

    # All-flat guard
    if pos_lag[test_mask].abs().sum().sum() < 1e-9:
        log.warning(
            "[%s] fold %s–%s: all-flat (no positions)",
            mode,
            test_start.date(),
            test_end.date(),
        )

    m = _metrics(net_ret)

    # SPY returns for correlation
    spy_rets_full = (
        pivot["SPY"].pct_change() if "SPY" in pivot.columns else pd.Series(dtype=float)
    )
    spy_test = (
        spy_rets_full[test_mask] if len(spy_rets_full) > 0 else pd.Series(dtype=float)
    )

    diag = _diagnostics(net_ret, spy_test, pos_wide_test)

    return m, diag, net_ret, pos_wide_test


def _apply_weights(
    pos_wide: pd.DataFrame,
    all_dates: pd.DatetimeIndex,
    rebal_date: pd.Timestamp,
    next_rebal: pd.Timestamp,
    weights: dict[str, float],
) -> None:
    """Fill pos_wide between rebal_date and next_rebal with given weights."""
    mask = (all_dates >= rebal_date) & (all_dates < next_rebal)
    dates_to_fill = all_dates[mask]
    if len(dates_to_fill) == 0:
        return
    # Zero out all columns in this range first
    pos_wide.loc[dates_to_fill, :] = 0.0
    for sym, w in weights.items():
        if sym in pos_wide.columns:
            pos_wide.loc[dates_to_fill, sym] = w


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
            bm_6040 = _benchmark_6040(prices, test_start, test_end)

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
                bm6040_cagr=bm_6040["cagr"],
                bm6040_sharpe=bm_6040["sharpe"],
                bm6040_maxdd=bm_6040["maxdd"],
                spy_corr=diag["spy_corr"],
                trades_per_year=diag["trades_per_year"],
                avg_hold_days=diag["avg_hold_days"],
                win_rate=diag["win_rate"],
                n_bars=len(net_ret),
                status="OK",
            )
            log.info(
                "[%s] Fold %d %s–%s: CAGR %.1f%% / Sharpe %.2f / MaxDD %.1f%%  "
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
    """Format a fold-row value, emitting '—' for NaN."""
    v = row.get(key)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return format(v, fmt) + suffix


def _write_report(
    results_low: list[dict],
    results_eq: list[dict],
    results_high: list[dict],
    n_tradeable: int,
) -> None:
    ok_low = [r for r in results_low if r.get("status") == "OK"]
    ok_eq = [r for r in results_eq if r.get("status") == "OK"]
    ok_high = [r for r in results_high if r.get("status") == "OK"]

    def _avg(rows: list[dict], key: str) -> float:
        vals = [
            r[key]
            for r in rows
            if isinstance(r.get(key), float) and not np.isnan(r[key])
        ]
        return float(np.mean(vals)) if vals else float("nan")

    def _table_row(r: dict) -> str:
        if r.get("status") != "OK":
            return (
                f"| {r['fold']} | {r['test_start']}–{r['test_end']} "
                "| FAILED | — | — | — | — | — | — | — | — | — | — |"
            )
        return (
            f"| {r['fold']} | {r['test_start']}–{r['test_end']} "
            f"| {_fmt(r['cagr'])} | {_fv(r, 'sharpe', '+.2f')} | {_fmt(r['maxdd'])} "
            f"| {_fv(r, 'calmar', '+.2f')} | {_fmt(r['spy_cagr'])} | {_fv(r, 'spy_sharpe', '+.2f')} "
            f"| {_fmt(r['bm6040_cagr'])} | {_fv(r, 'spy_corr', '+.2f')} "
            f"| {_fv(r, 'trades_per_year', '.0f')} | {_fv(r, 'avg_hold_days', '.0f', 'd')} "
            f"| {_fv(r, 'win_rate', '.1%')} |"
        )

    def _avg_row(rows: list[dict], n_total: int, label: str) -> str:
        n_ok = len(rows)
        if not rows:
            return f"| **Ø (0/{n_total})** | — | — | — | — | — | — | — | — | — | — | — | — |"
        avg_cagr = _avg(rows, "cagr")
        avg_sh = _avg(rows, "sharpe")
        avg_dd = _avg(rows, "maxdd")
        avg_spy_cagr = _avg(rows, "spy_cagr")
        avg_spy_sh = _avg(rows, "spy_sharpe")
        avg_6040 = _avg(rows, "bm6040_cagr")
        avg_corr = _avg(rows, "spy_corr")
        avg_tpy = _avg(rows, "trades_per_year")
        avg_hold = _avg(rows, "avg_hold_days")
        avg_wr = _avg(rows, "win_rate")
        return (
            f"| **Ø ({n_ok}/{n_total})** | — "
            f"| **{_fmt(avg_cagr)}** | **{avg_sh:+.2f}** | **{_fmt(avg_dd)}** "
            f"| — | {_fmt(avg_spy_cagr)} | {avg_spy_sh:+.2f} "
            f"| {_fmt(avg_6040)} | {avg_corr:+.2f} "
            f"| {avg_tpy:.0f} | {avg_hold:.0f}d "
            f"| {avg_wr:.1%} |"
        )

    TABLE_HDR = (
        "| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe "
        "| 60/40 CAGR | Corr(SPY) | Trades/yr | AvgHold | WinRate |"
    )
    TABLE_SEP = (
        "|------|-------------|------|--------|-------|--------|----------|------------ "
        "|-----------|-----------|-----------|---------|---------|"
    )

    lines = [
        "# Low-MAX / Lottery-Avoidance Strategy — OOS Walk-Forward Backtest",
        "",
        f"Run date: {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
        "Data: Alpaca daily bars (local cache) — 2018-01-02 → 2025-12-31  ",
        f"Universe: {n_tradeable} tradeable symbols (data from ≤ 2018-01-31, ≥ 500 bars; SPY excluded)  ",
        "Strategy: Low-MAX quintile selection — monthly rebalancing, long-only, equal-weight within bucket  ",
        f"MAX definition: max daily return over last {MAX_LOOKBACK} trading days  ",
        f"Quintile: bottom/top {QUANTILE_SELECT:.0%} by MAX score  ",
        f"WF: {TRAIN_WINDOW}-bar train / {TEST_WINDOW}-bar test / {STEP_SIZE}-bar step  ",
        f"Costs: {COST_BPS} bps per leg (one-sided), 1-bar execution lag  ",
        "",
        "Reference: Bali, Cakici & Whitelaw (2011) 'Maxing Out: Stocks as Lotteries and the",
        "Cross-Section of Expected Returns'. Journal of Financial Economics 99(2), 427-446.",
        "",
    ]

    # --- Low-MAX table ---
    lines += [
        "## Walk-Forward Results — Low-MAX (Bottom Quintile)",
        "",
        TABLE_HDR,
        TABLE_SEP,
    ]
    for r in results_low:
        lines.append(_table_row(r))
    lines.append(_avg_row(ok_low, len(results_low), "low_max"))
    lines.append("")

    # --- Equal-Weight table ---
    lines += [
        "## Walk-Forward Results — Equal-Weight Universe",
        "",
        TABLE_HDR,
        TABLE_SEP,
    ]
    for r in results_eq:
        lines.append(_table_row(r))
    lines.append(_avg_row(ok_eq, len(results_eq), "eq_weight"))
    lines.append("")

    # --- High-MAX table ---
    lines += [
        "## Walk-Forward Results — High-MAX (Top Quintile)",
        "",
        TABLE_HDR,
        TABLE_SEP,
    ]
    for r in results_high:
        lines.append(_table_row(r))
    lines.append(_avg_row(ok_high, len(results_high), "high_max"))
    lines.append("")

    # --- MAX-Spread section ---
    lines += [
        "## MAX-Spread (CAGR Low-MAX minus CAGR High-MAX)",
        "",
        "| Fold | Test Period | CAGR Low-MAX | CAGR High-MAX | Spread |",
        "|------|-------------|--------------|---------------|--------|",
    ]
    spreads: list[float] = []
    # Align folds by fold number
    low_by_fold = {r["fold"]: r for r in results_low}
    high_by_fold = {r["fold"]: r for r in results_high}
    all_folds = sorted(set(low_by_fold) | set(high_by_fold))
    for fid in all_folds:
        rl = low_by_fold.get(fid, {})
        rh = high_by_fold.get(fid, {})
        period = f"{rl.get('test_start', '?')}–{rl.get('test_end', '?')}"
        if rl.get("status") == "OK" and rh.get("status") == "OK":
            lc = rl["cagr"]
            hc = rh["cagr"]
            sp = lc - hc
            spreads.append(sp)
            lines.append(f"| {fid} | {period} | {_fmt(lc)} | {_fmt(hc)} | {_fmt(sp)} |")
        else:
            lines.append(f"| {fid} | {period} | — | — | — |")

    avg_spread = float(np.mean(spreads)) if spreads else float("nan")
    lines.append(
        f"| **Ø** | — | **{_fmt(_avg(ok_low, 'cagr'))}** "
        f"| **{_fmt(_avg(ok_high, 'cagr'))}** | **{_fmt(avg_spread)}** |"
    )
    lines.append("")

    # --- Assessment ---
    avg_cagr_low = _avg(ok_low, "cagr")
    avg_sh_low = _avg(ok_low, "sharpe")
    avg_dd_low = _avg(ok_low, "maxdd")
    avg_cagr_eq = _avg(ok_eq, "cagr")
    avg_sh_eq = _avg(ok_eq, "sharpe")
    avg_sh_spy = _avg(ok_low, "spy_sharpe")

    cagr_vs_eq = (
        avg_cagr_low - avg_cagr_eq
        if not (np.isnan(avg_cagr_low) or np.isnan(avg_cagr_eq))
        else float("nan")
    )
    sharpe_vs_eq = (
        avg_sh_low - avg_sh_eq
        if not (np.isnan(avg_sh_low) or np.isnan(avg_sh_eq))
        else float("nan")
    )

    lines += [
        "## Assessment",
        "",
        "### 1. Does Low-MAX beat Equal-Weight risk-adjusted?",
        "",
    ]
    if not np.isnan(cagr_vs_eq):
        beat_cagr = "YES" if cagr_vs_eq > 0 else "NO"
        beat_sharpe = "YES" if sharpe_vs_eq > 0 else "NO"
        lines += [
            f"- Ø CAGR Low-MAX vs Equal-Weight: {cagr_vs_eq:+.1%} ({beat_cagr})",
            f"- Ø Sharpe Low-MAX vs Equal-Weight: {sharpe_vs_eq:+.2f} ({beat_sharpe})",
        ]
    else:
        lines.append("- Insufficient data for comparison.")
    lines.append("")

    lines += [
        "### 2. Is the MAX-Spread positive (Lottery effect present)?",
        "",
    ]
    if not np.isnan(avg_spread):
        effect = (
            "YES — Lottery effect present"
            if avg_spread > 0
            else "NO — Lottery effect absent or reversed"
        )
        lines += [
            f"- Ø MAX-Spread (Low-MAX CAGR − High-MAX CAGR): {avg_spread:+.1%}  ",
            f"- **{effect}**  ",
            "  (Positive spread confirms high-MAX stocks underperform low-MAX in-sample.)",
        ]
    else:
        lines.append("- Insufficient data.")
    lines.append("")

    lines += [
        "### 3. Important Caveats",
        "",
        "**Universe bias (large/mid-cap dampening):**  ",
        "The academic MAX effect is strongest in small- and microcap stocks (Bali et al. 2011).  ",
        "This universe (Alpaca local cache, mostly large/mid-cap, liquid names) systematically  ",
        "dampens the effect. A positive result here is noteworthy; a null result is ambiguous.  ",
        "",
        "**Survivorship bias:**  ",
        "The local Alpaca cache only contains currently-available (surviving) symbols.  ",
        "High-MAX stocks that blew up (the biggest lottery losers) are missing.  ",
        "This compresses the observable spread downward and makes high-MAX appear less bad  ",
        "than the full universe. A null result may understate the real effect.  ",
        "",
        "**No short leg:**  ",
        "The academic result is long-short. This backtest is long-only (bottom quintile only).  ",
        "Long-only capture of the factor is weaker and more market-beta-driven.  ",
        "",
    ]

    lines += [
        "### 4. GO_LIVE_CHECKLIST B-tier Criterion Check",
        "",
        "| Criterion | Threshold | Achieved | Pass? |",
        "|-----------|-----------|----------|-------|",
    ]
    if ok_low:
        cagr_pass = "✓" if avg_cagr_low > 0.05 else "✗"
        sharpe_pass = "✓" if avg_sh_low > 0.5 else "✗"
        dd_pass = "✓" if avg_dd_low > -0.30 else "✗"
        spy_sharpe_pass = "✓" if avg_sh_low > avg_sh_spy else "✗"
        lines += [
            f"| Ø CAGR > 5% | 5% | {avg_cagr_low:+.1%} | {cagr_pass} |",
            f"| Ø Sharpe > 0.5 | 0.5 | {avg_sh_low:+.2f} | {sharpe_pass} |",
            f"| MaxDD > -30% | -30% | {avg_dd_low:+.1%} | {dd_pass} |",
            f"| Beat SPY Sharpe | N/A | {avg_sh_low:+.2f} vs {avg_sh_spy:+.2f} | {spy_sharpe_pass} |",
        ]
    else:
        lines.append("| N/A — no valid folds | — | — | ✗ |")

    lines += [
        "",
        "---",
        "_Script: `scripts/_oos_wf_low_max_lottery.py`_  ",
        "_Feature: `src/assembled_core/features/behavioral_features.py` — `max_effect()`_  ",
        "_Reference: Bali, Cakici & Whitelaw (2011), J. Financial Economics 99(2)_",
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

    log.info("Running WF — mode=low_max…")
    results_low = _run_wf(prices, tradeable, mode="low_max")

    log.info("Running WF — mode=eq_weight…")
    results_eq = _run_wf(prices, tradeable, mode="eq_weight")

    log.info("Running WF — mode=high_max…")
    results_high = _run_wf(prices, tradeable, mode="high_max")

    log.info("Writing report…")
    _write_report(results_low, results_eq, results_high, len(tradeable))

    ok_low = [r for r in results_low if r.get("status") == "OK"]
    ok_high = [r for r in results_high if r.get("status") == "OK"]
    ok_eq = [r for r in results_eq if r.get("status") == "OK"]

    if ok_low:
        import statistics

        spreads = []
        low_by_fold = {r["fold"]: r for r in ok_low}
        high_by_fold = {r["fold"]: r for r in ok_high}
        for fid in low_by_fold:
            if fid in high_by_fold:
                spreads.append(low_by_fold[fid]["cagr"] - high_by_fold[fid]["cagr"])

        log.info(
            "Low-MAX: %d/%d folds OK | Avg CAGR %.1f%% | Avg Sharpe %.2f",
            len(ok_low),
            len(results_low),
            statistics.mean(r["cagr"] for r in ok_low) * 100,
            statistics.mean(r["sharpe"] for r in ok_low),
        )
        log.info(
            "High-MAX: %d/%d folds OK | Avg CAGR %.1f%%",
            len(ok_high),
            len(results_high),
            statistics.mean(r["cagr"] for r in ok_high) * 100
            if ok_high
            else float("nan"),
        )
        log.info(
            "Equal-Weight: %d/%d folds OK | Avg CAGR %.1f%%",
            len(ok_eq),
            len(results_eq),
            statistics.mean(r["cagr"] for r in ok_eq) * 100 if ok_eq else float("nan"),
        )
        if spreads:
            log.info(
                "MAX-Spread (Low−High): Avg %.2f%% across %d folds",
                statistics.mean(spreads) * 100,
                len(spreads),
            )

    print("Done ->", OUT_MD)
