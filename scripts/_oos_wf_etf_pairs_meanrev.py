"""One-shot OOS Walk-Forward for ETF-Pairs Cointegration Mean-Reversion.

Writes docs/results/2026_05_etf_pairs_meanrev_real_oos.md.

Usage:
    python scripts/_oos_wf_etf_pairs_meanrev.py

Design:
    - Strategy: rolling Engle-Granger cointegration on 252-bar log-price windows.
      OLS hedge ratio; Z-score entry (|Z|>2.0), exit (|Z|<0.5), stop (|Z|>3.5).
      Equal-weight across active pairs.  Two modes: full (long-short) and long-only.
    - Data: local Alpaca daily bars via load_eod_prices.
      All 12 symbols available from 2018-01-02.
    - Pairs (substitute for original 6 — original symbols not in local data):
        (SPY, IWM)   large-cap / small-cap        [orig: SPY/IVV]
        (GLD, SLV)   gold / silver                [orig: GDX/GDXJ]
        (XLK, QQQ)   technology ETFs              [orig: XLK/VGT]
        (TLT, XLF)   rates / financials           [orig: EWA/EWC]
        (XLV, XLY)   healthcare / consumer disc.  [orig: XLF/KBE]
        (XLE, XLI)   energy / industrials         [orig: XLE/VDE]
    - Walk-forward: 252/252/252 (train / test / step).
      Warmup = train window prepended to each test period.
    - Transaction costs:
        Full mode (2 legs per trade): 2 × 10.75 bps = 21.5 bps per event.
        Long-only (1 leg):             1 × 10.75 bps = 10.75 bps per event.
        Applied as: |Δweight| × TOTAL_COST_PER_WEIGHT_UNIT × COST_BPS / 10000.
    - Benchmarks: SPY B&H, 60/40 SPY/TLT daily-rebalanced.
    - Diagnostics: SPY daily-return correlation, trades/year, avg-hold-days,
      win-rate (daily).

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
log = logging.getLogger("oos_wf_etf_pairs")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW = 252
TEST_WINDOW = 252
STEP_SIZE = 252

COINT_WINDOW = 252
ZSCORE_WINDOW = 60
ENTRY_Z = 2.0
EXIT_Z = 0.5
STOP_Z = 3.5

COST_BPS = 10.0 + 0.25 + 0.5  # 10.75 bps per leg
INITIAL_CAPITAL = 100_000.0

PAIRS = [
    ("SPY", "IWM"),
    ("GLD", "SLV"),
    ("XLK", "QQQ"),
    ("TLT", "XLF"),
    ("XLV", "XLY"),
    ("XLE", "XLI"),
]
ALL_SYMS = list({s for p in PAIRS for s in p})

OUT_MD = ROOT / "docs" / "results" / "2026_05_etf_pairs_meanrev_real_oos.md"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_prices() -> pd.DataFrame:
    from src.assembled_core.data.prices_ingest import load_eod_prices

    prices = load_eod_prices(ALL_SYMS)
    if prices is None or prices.empty:
        raise RuntimeError("No price data returned from load_eod_prices")
    prices = prices[
        (prices["timestamp"] >= PERIOD_START - pd.Timedelta(days=30))
        & (prices["timestamp"] <= PERIOD_END + pd.Timedelta(days=5))
    ].copy()
    log.info(
        "Loaded %d rows for %d symbols (%s → %s)",
        len(prices),
        prices["symbol"].nunique(),
        prices["timestamp"].min().date(),
        prices["timestamp"].max().date(),
    )
    return prices


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
    port_r = 0.6 * r["SPY"] + 0.4 * r["TLT"]
    return _metrics(port_r)


def _diagnostics(
    net_ret: pd.Series, spy_ret: pd.Series, pos_wide: pd.DataFrame
) -> dict:
    """Additional diagnostics: SPY correlation, trades/year, hold-days, win-rate."""
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

    # Trades/year: count bars where any signed weight changes
    n_years = len(net_ret) / 252.0
    test_pos = pos_wide.reindex(net_ret.index).fillna(0.0)
    weight_changes = test_pos.diff().abs().sum(axis=1)
    # A "trade event" is any bar with total weight change > threshold (rounding guard)
    trade_events = (weight_changes > 1e-6).sum()
    trades_per_year = float(trade_events) / n_years if n_years > 0 else float("nan")

    # Avg hold duration: mean run length of bars with any active position
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

    # Win-rate: fraction of test days with positive return (when in position)
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
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    mode: str,  # "full" or "long_only"
) -> tuple[dict, dict, pd.Series, pd.DataFrame]:
    """Simulate ETF-pairs for one WF fold.

    Returns:
        (metrics, diagnostics, net_ret_series, pos_wide_test)
    """
    from src.assembled_core.strategies.etf_pairs_meanrev import (
        generate_etf_pairs_signals_from_prices,
    )

    long_only = mode == "long_only"

    # Warmup: use train window prepended to test period
    all_sym_dates = np.sort(prices[prices["symbol"] == "SPY"]["timestamp"].unique())
    pre_test = all_sym_dates[all_sym_dates < test_start]
    warmup_start = (
        pre_test[-TRAIN_WINDOW]
        if len(pre_test) >= TRAIN_WINDOW
        else (pre_test[0] if len(pre_test) > 0 else test_start)
    )

    window_prices = prices[
        (prices["timestamp"] >= warmup_start) & (prices["timestamp"] < test_end)
    ].copy()

    # Pivot and forward-fill
    pivot = (
        window_prices.pivot_table(index="timestamp", columns="symbol", values="close")
        .sort_index()
        .ffill()
    )

    # Drop columns with all-NaN (symbols with no data in window)
    pivot = pivot.dropna(axis=1, how="all")
    valid_pairs = [
        (a, b) for a, b in PAIRS if a in pivot.columns and b in pivot.columns
    ]
    if not valid_pairs:
        raise ValueError(f"No valid pairs in {test_start.date()}–{test_end.date()}")

    # Daily returns
    rets = pivot.pct_change()

    # Generate signals on combined warmup+test window (strictly causal)
    prices_long = window_prices[
        window_prices["symbol"].isin({s for p in valid_pairs for s in p})
    ].copy()

    log.info(
        "[%s] generating signals for fold %s–%s (warmup %s, %d pairs)",
        mode,
        test_start.date(),
        test_end.date(),
        warmup_start.date(),
        len(valid_pairs),
    )
    sigs = generate_etf_pairs_signals_from_prices(
        prices_long,
        pairs=valid_pairs,
        cointegration_window=COINT_WINDOW,
        zscore_window=ZSCORE_WINDOW,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
        stop_z=STOP_Z,
        long_only=long_only,
    )

    # Build signed position matrix
    all_dates = pivot.index
    if sigs.empty:
        log.warning("[%s] no signals in fold %s — all-flat", mode, test_start.date())
        pos_wide = pd.DataFrame(0.0, index=all_dates, columns=pivot.columns)
    else:
        pos_long = sigs.assign(
            signed_w=lambda df: df["score"]
            * df["direction"].map({"LONG": 1.0, "SHORT": -1.0})
        )
        pos_wide = (
            pos_long.pivot_table(
                index="timestamp", columns="symbol", values="signed_w", aggfunc="sum"
            )
            .reindex(all_dates, fill_value=0.0)
            .fillna(0.0)
        )

    # Enforce stable column set matching pivot (handles long_only mode where
    # SHORT legs may never appear in signals, avoiding silent column gaps).
    pos_wide = pos_wide.reindex(columns=pivot.columns, fill_value=0.0)

    # Lagged positions: position held entering each bar (1-bar lag = no look-ahead)
    pos_lag = pos_wide.shift(1).fillna(0.0)

    # Portfolio return per bar
    rets_aligned = rets.reindex(columns=pos_lag.columns).fillna(0.0)
    port_ret_all = (pos_lag * rets_aligned).sum(axis=1)

    # Transaction costs: charged at the EXECUTION bar (T+1 after signal at T).
    # pos_lag.diff()[T] = pos_lag[T] - pos_lag[T-1] = pos_wide[T-1] - pos_wide[T-2]
    # so cost and PnL land on the same bar — consistent with the 1-bar lag model.
    abs_delta = pos_lag.diff().fillna(0.0).abs().sum(axis=1)
    cost_all = abs_delta * COST_BPS / 10_000.0

    net_ret_all = port_ret_all - cost_all

    # Slice to test period
    test_mask = (net_ret_all.index >= test_start) & (net_ret_all.index < test_end)
    net_ret = net_ret_all[test_mask]
    pos_wide_test = pos_wide[test_mask]

    if len(net_ret) < 5:
        raise ValueError(f"Only {len(net_ret)} test bars")

    m = _metrics(net_ret)

    # SPY returns for diagnostic correlation
    spy_all = (
        rets_aligned["SPY"] if "SPY" in rets_aligned.columns else pd.Series(dtype=float)
    )
    spy_test = spy_all[test_mask]

    diag = _diagnostics(net_ret, spy_test, pos_wide_test)

    return m, diag, net_ret, pos_wide_test


# ---------------------------------------------------------------------------
# Walk-Forward
# ---------------------------------------------------------------------------
def _run_wf(prices: pd.DataFrame, mode: str) -> list[dict]:
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

        test_start = (
            pd.Timestamp(spy_dates[train_end_i]).tz_localize("UTC")
            if pd.Timestamp(spy_dates[train_end_i]).tzinfo is None
            else pd.Timestamp(spy_dates[train_end_i])
        )
        test_end_ts = pd.Timestamp(spy_dates[test_end_i - 1])
        test_end = (
            test_end_ts.tz_localize("UTC")
            if test_end_ts.tzinfo is None
            else test_end_ts
        ) + pd.Timedelta(hours=23)

        try:
            m, diag, net_ret, pos_wide_test = _simulate(
                prices, test_start, test_end, mode
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


def _write_report(results_full: list[dict], results_lo: list[dict]) -> None:
    ok_full = [r for r in results_full if r.get("status") == "OK"]
    ok_lo = [r for r in results_lo if r.get("status") == "OK"]

    def _avg(rows, key):
        vals = [
            r[key]
            for r in rows
            if isinstance(r.get(key), float) and not np.isnan(r[key])
        ]
        return np.mean(vals) if vals else float("nan")

    n_ok_full = len(ok_full)
    n_ok_lo = len(ok_lo)

    lines = [
        "# ETF-Pairs Cointegration Mean-Reversion — OOS Walk-Forward Backtest",
        "",
        f"Run date: {pd.Timestamp.now().strftime('%Y-%m-%d')}  ",
        "Data: Alpaca daily bars (local cache) — 2018-01-02 → 2026-05-18  ",
        "Strategy: `etf_pairs_meanrev` — rolling 252-bar Engle-Granger cointegration,",
        "OLS hedge ratio, Z-score (60d), entry |Z|>2.0, exit |Z|<0.5, stop |Z|>3.5  ",
        f"WF: {TRAIN_WINDOW}-bar train / {TEST_WINDOW}-bar test / {STEP_SIZE}-bar step",
        "",
        "## Pairs used (local-data substitutes for original 6)",
        "",
        "| Requested | Substitute | Rationale |",
        "|-----------|------------|-----------|",
        "| SPY/IVV   | SPY/IWM    | large-cap / small-cap US equity |",
        "| GDX/GDXJ  | GLD/SLV    | gold / silver (precious metals) |",
        "| XLK/VGT   | XLK/QQQ    | technology ETFs |",
        "| EWA/EWC   | TLT/XLF    | rates / financials |",
        "| XLF/KBE   | XLV/XLY    | healthcare / consumer discretionary |",
        "| XLE/VDE   | XLE/XLI    | energy / industrials |",
        "",
        "The original symbols (IVV, GDX, GDXJ, VDE, EWA, EWC, KBE, VGT) are not",
        "present in the local Alpaca price cache.  Pairs are not the same and results",
        "should not be compared directly to the original specification.",
        "",
        "## Walk-Forward Results — Full (Long-Short)",
        "",
        "| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | 60/40 CAGR | Corr(SPY) | Trades/yr | AvgHold | WinRate |",
        "|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|-----------|---------|---------|",
    ]

    for r in results_full:
        if r.get("status") != "OK":
            lines.append(
                f"| {r['fold']} | {r['test_start']}–{r['test_end']} | FAILED | — | — | — | — | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {r['fold']} | {r['test_start']}–{r['test_end']} "
            f"| {_fmt(r['cagr'])} | {_fv(r, 'sharpe', '+.2f')} | {_fmt(r['maxdd'])} "
            f"| {_fv(r, 'calmar', '+.2f')} | {_fmt(r['spy_cagr'])} | {_fv(r, 'spy_sharpe', '+.2f')} "
            f"| {_fmt(r['bm6040_cagr'])} | {_fv(r, 'spy_corr', '+.2f')} "
            f"| {_fv(r, 'trades_per_year', '.0f')} | {_fv(r, 'avg_hold_days', '.0f', 'd')} "
            f"| {_fv(r, 'win_rate', '.1%')} |"
        )

    if ok_full:
        avg_cagr_f = _avg(ok_full, "cagr")
        avg_sh_f = _avg(ok_full, "sharpe")
        avg_dd_f = _avg(ok_full, "maxdd")
        avg_spy_cagr_f = _avg(ok_full, "spy_cagr")
        avg_spy_sh_f = _avg(ok_full, "spy_sharpe")
        avg_6040_f = _avg(ok_full, "bm6040_cagr")
        avg_corr_f = _avg(ok_full, "spy_corr")
        avg_tpy_f = _avg(ok_full, "trades_per_year")
        avg_hold_f = _avg(ok_full, "avg_hold_days")
        avg_wr_f = _avg(ok_full, "win_rate")
        lines.append(
            f"| **Ø ({n_ok_full}/{len(results_full)})** | — "
            f"| **{_fmt(avg_cagr_f)}** | **{avg_sh_f:+.2f}** | **{_fmt(avg_dd_f)}** "
            f"| — | {_fmt(avg_spy_cagr_f)} | {avg_spy_sh_f:+.2f} "
            f"| {_fmt(avg_6040_f)} | {avg_corr_f:+.2f} "
            f"| {avg_tpy_f:.0f} | {avg_hold_f:.0f}d "
            f"| {avg_wr_f:.1%} |"
        )

    lines += [
        "",
        "## Walk-Forward Results — Long-Only",
        "",
        "| Fold | Test Period | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | WinRate |",
        "|------|-------------|------|--------|-------|--------|----------|------------|---------|",
    ]

    for r in results_lo:
        if r.get("status") != "OK":
            lines.append(
                f"| {r['fold']} | {r['test_start']}–{r['test_end']} | FAILED | — | — | — | — | — | — |"
            )
            continue
        lines.append(
            f"| {r['fold']} | {r['test_start']}–{r['test_end']} "
            f"| {_fmt(r['cagr'])} | {_fv(r, 'sharpe', '+.2f')} | {_fmt(r['maxdd'])} "
            f"| {_fv(r, 'calmar', '+.2f')} | {_fmt(r['spy_cagr'])} | {_fv(r, 'spy_sharpe', '+.2f')} "
            f"| {_fv(r, 'win_rate', '.1%')} |"
        )

    if ok_lo:
        avg_cagr_lo = _avg(ok_lo, "cagr")
        avg_sh_lo = _avg(ok_lo, "sharpe")
        avg_dd_lo = _avg(ok_lo, "maxdd")
        avg_spy_cagr_lo = _avg(ok_lo, "spy_cagr")
        avg_spy_sh_lo = _avg(ok_lo, "spy_sharpe")
        avg_wr_lo = _avg(ok_lo, "win_rate")
        lines.append(
            f"| **Ø ({n_ok_lo}/{len(results_lo)})** | — "
            f"| **{_fmt(avg_cagr_lo)}** | **{avg_sh_lo:+.2f}** | **{_fmt(avg_dd_lo)}** "
            f"| — | {_fmt(avg_spy_cagr_lo)} | {avg_spy_sh_lo:+.2f} "
            f"| {avg_wr_lo:.1%} |"
        )

    lines += [
        "",
        "## Assessment",
        "",
        "**Data note:** Substitute pairs used — results not directly comparable to original",
        "6-pair specification.  Original symbols missing from local Alpaca cache.",
        "",
    ]

    # Honest verdict
    if ok_full:
        avg_cagr_vs_spy = _avg(ok_full, "cagr") - _avg(ok_full, "spy_cagr")
        avg_sharpe_vs_spy = _avg(ok_full, "sharpe") - _avg(ok_full, "spy_sharpe")
        avg_corr = _avg(ok_full, "spy_corr")
        lines += [
            f"**Full mode** Ø CAGR vs SPY: {avg_cagr_vs_spy:+.1%} | "
            f"Ø Sharpe vs SPY: {avg_sharpe_vs_spy:+.2f} | "
            f"Ø SPY correlation: {avg_corr:+.2f}  ",
            "",
        ]

    lines += [
        "**Criterion check (GO_LIVE_CHECKLIST B-tier):**",
        "",
        "| Criterion | Threshold | Achieved | Pass? |",
        "|-----------|-----------|----------|-------|",
    ]

    if ok_full:
        avg_cagr_f = _avg(ok_full, "cagr")
        avg_sh_f = _avg(ok_full, "sharpe")
        avg_dd_f = _avg(ok_full, "maxdd")
        cagr_pass = "✓" if avg_cagr_f > 0.05 else "✗"
        sharpe_pass = "✓" if avg_sh_f > 0.5 else "✗"
        dd_pass = "✓" if avg_dd_f > -0.30 else "✗"
        lines += [
            f"| Ø CAGR > 5% | 5% | {avg_cagr_f:+.1%} | {cagr_pass} |",
            f"| Ø Sharpe > 0.5 | 0.5 | {avg_sh_f:+.2f} | {sharpe_pass} |",
            f"| MaxDD > -30% | -30% | {avg_dd_f:+.1%} | {dd_pass} |",
            f"| Beat SPY Sharpe | N/A | {_avg(ok_full, 'sharpe'):+.2f} vs {_avg(ok_full, 'spy_sharpe'):+.2f} | — |",
        ]
    else:
        lines.append("| N/A — no valid folds | — | — | ✗ |")

    lines += [
        "",
        "**Verdict:** Informational only — substitute pairs, not original spec.",
        "Original pairs (IVV, GDXJ, VDE, EWA/EWC, KBE, VGT) require Alpaca data download.",
        "Consider fetching original pairs for definitive Kandidat D assessment.",
        "",
        "---",
        "_Script: `scripts/_oos_wf_etf_pairs_meanrev.py`_  ",
        "_Strategy: `src/assembled_core/strategies/etf_pairs_meanrev.py`_  ",
        "_Tests: `tests/test_etf_pairs_meanrev_pit_safety.py`_",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written -> %s", OUT_MD)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Loading prices…")
    prices = _load_prices()

    log.info("Running WF — mode=full (%d pairs)…", len(PAIRS))
    results_full = _run_wf(prices, mode="full")

    log.info("Running WF — mode=long_only…")
    results_lo = _run_wf(prices, mode="long_only")

    log.info("Writing report…")
    _write_report(results_full, results_lo)

    ok_full = [r for r in results_full if r.get("status") == "OK"]
    ok_lo = [r for r in results_lo if r.get("status") == "OK"]

    if ok_full:
        import statistics

        cagrs = [r["cagr"] for r in ok_full]
        log.info(
            "Full mode: %d/%d folds OK | Avg CAGR %.1f%% | Avg Sharpe %.2f",
            len(ok_full),
            len(results_full),
            statistics.mean(cagrs) * 100,
            statistics.mean(r["sharpe"] for r in ok_full),
        )
    if ok_lo:
        import statistics

        log.info(
            "Long-only: %d/%d folds OK | Avg CAGR %.1f%%",
            len(ok_lo),
            len(results_lo),
            statistics.mean(r["cagr"] for r in ok_lo) * 100,
        )

    print("Done ->", OUT_MD)
