"""One-shot OOS Walk-Forward for dual_momentum — writes docs/results/2026_05_dual_momentum_real_oos.md.

Usage:
    python scripts/_oos_wf_dual_momentum.py

Requirements:
    ALPACA_API_KEY / ALPACA_API_SECRET in .env or environment.
    pip install alpaca-py (already installed in this project).

Design:
    - Dual Momentum (Antonacci-variant): monthly rebalancing, hold exactly one
      of {SPY, VEU, AGG} at weight 1.0.  Signal: relative momentum among
      {SPY, VEU} over 12 months; if winner beats BIL (T-bill hurdle) → hold
      winner; else → hold AGG.
    - Fetches Alpaca daily bars for SPY, VEU, BIL, AGG.
      VEU inception 2007-03-02; BIL inception 2007-05-25 (limiting factor).
    - Walk-forward: 252/252/252 (train / test / step), same as trend_baseline.
    - Warmup buffer: lookback_months × 21 + 60 bars prepended so the first
      valid rebalance signal is hot at the start of each test period.
    - Three benchmarks:
        1. SPY buy-and-hold
        2. 60/40 SPY/AGG buy-and-hold (static initial allocation)
        3. 60/40 SPY/AGG daily-rebalanced (fair comparison vs monthly strategy)
      Plus reference metrics from vol_target_overlay (different period — noted).
    - Transaction costs: switch triggers 100 % portfolio turnover.
      10 bps commission + 0.25 + 0.5 spread/impact = 10.75 bps per switch.
    - Startkapital 100,000 USD.

KEINE Änderungen an strategy, policy.yaml, oder anderen Produktionsdateien.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("oos_wf_dual_momentum")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# VEU inception 2007-03-02; BIL inception 2007-05-25.
# Fetch from 2006-01-01 — Alpaca returns from actual inception.
PERIOD_START = pd.Timestamp("2007-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW_DAYS = 252
TEST_WINDOW_DAYS = 252
STEP_SIZE_DAYS = 252
LOOKBACK_MONTHS = 12
WARMUP_BARS = LOOKBACK_MONTHS * 21 + 60  # ~312 bars (~15 months)
COMMISSION_BPS = 10.0
SPREAD_W = 0.25
IMPACT_W = 0.5
TOTAL_COST_BPS = COMMISSION_BPS + SPREAD_W + IMPACT_W  # 10.75 bps per switch
INITIAL_CAPITAL = 100_000.0
SYMBOLS = ["SPY", "VEU", "BIL", "AGG"]
OUT_MD = ROOT / "docs" / "results" / "2026_05_dual_momentum_real_oos.md"


# ---------------------------------------------------------------------------
# 1 — Credentials
# ---------------------------------------------------------------------------
def _load_env():
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    ak = os.environ.get("ALPACA_API_KEY", "")
    sk = os.environ.get("ALPACA_API_SECRET", "")
    if not ak or not sk:
        raise EnvironmentError("ALPACA_API_KEY / ALPACA_API_SECRET not set")
    return ak, sk


# ---------------------------------------------------------------------------
# 2 — Fetch Alpaca daily bars
# ---------------------------------------------------------------------------
def _fetch_alpaca(
    symbols: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    ak, sk = _load_env()
    client = StockHistoricalDataClient(api_key=ak, secret_key=sk)

    log.info("Fetching %s from Alpaca (%s → %s)…", symbols, start.date(), end.date())
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        adjustment="split",
    )
    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()

    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.rename(columns=str.lower)
    keep = [
        c
        for c in ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        if c in df.columns
    ]
    df = df[keep].copy()
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    log.info("Fetched %d rows for %s", len(df), df["symbol"].unique().tolist())
    return df


# ---------------------------------------------------------------------------
# 3 — Simulation for one fold
# ---------------------------------------------------------------------------
def _simulate_dual_momentum(
    prices_by_sym: dict[str, pd.DataFrame],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> dict:
    """Simulate dual momentum for one WF fold; return CAGR/Sharpe/MaxDD/Calmar.

    Portfolio return on bar T is realised by holding the asset established at
    the previous EOM rebalance (weights lag returns by one bar — no look-ahead).
    Transaction costs applied on EOM bars where the holding switches.
    """
    from src.assembled_core.strategies.dual_momentum import (
        generate_dual_momentum_signals_from_prices,
    )

    # Determine warmup start — extend test_start back by WARMUP_BARS
    all_spy_dates = np.sort(prices_by_sym["SPY"]["timestamp"].unique())
    pre_test = all_spy_dates[all_spy_dates < test_start]
    warmup_start = (
        pre_test[-WARMUP_BARS]
        if len(pre_test) >= WARMUP_BARS
        else (pre_test[0] if len(pre_test) > 0 else test_start)
    )

    # Slice each symbol to warmup+test window
    frames: dict[str, pd.DataFrame] = {}
    for sym, df in prices_by_sym.items():
        frames[sym] = (
            df[(df["timestamp"] >= warmup_start) & (df["timestamp"] < test_end)]
            .sort_values("timestamp")
            .copy()
        )

    if any(f.empty for f in frames.values()):
        missing = [s for s, f in frames.items() if f.empty]
        raise ValueError(
            f"Price data missing for {missing} in {test_start.date()}–{test_end.date()}"
        )

    # Generate signals on warmup+test combined window (strictly causal)
    prices_long = pd.concat(frames.values(), ignore_index=True)
    sigs = generate_dual_momentum_signals_from_prices(
        prices_long, lookback_months=LOOKBACK_MONTHS
    )

    if sigs.empty:
        raise ValueError("No signals after warmup — insufficient data")

    # Assert test period has ACTUAL signals (not just warmup defaults)
    test_index = (
        pd.concat(
            [f.set_index("timestamp")[["close"]] for f in frames.values()],
            axis=1,
        )
        .dropna()
        .index
    )
    test_index = test_index[(test_index >= test_start) & (test_index < test_end)]
    sigs_in_test = sigs[sigs["timestamp"].isin(test_index)]
    if sigs_in_test.empty:
        raise ValueError(
            f"No signals in test period {test_start.date()}–{test_end.date()} "
            "— warmup may not have completed before test_start"
        )

    # Build combined price table (inner join on all 4 symbols)
    price_parts = {
        sym.lower(): f.set_index("timestamp")["close"] for sym, f in frames.items()
    }
    combined = pd.DataFrame(price_parts).dropna()

    # Map daily signal (holding) to the combined frame via forward-fill
    holding_series = sigs.set_index("timestamp")["symbol"]
    combined["holding"] = holding_series.reindex(combined.index).ffill()
    combined = combined.dropna(subset=["holding"])

    # Compute returns on FULL combined frame (warmup + test) before slicing.
    # This ensures the first test bar has a valid prior-bar return.
    for sym in ["spy", "veu", "bil", "agg"]:
        combined[f"r_{sym}"] = combined[sym].pct_change()

    # Carry-in fix: compute holding_prev on FULL combined frame so the first
    # test bar inherits the warmup-end holding rather than NaN.
    combined["holding_prev"] = combined["holding"].shift(1)

    # Filter to test period
    test_mask = (combined.index >= test_start) & (combined.index < test_end)
    test_df = combined[test_mask].copy()

    if len(test_df) < 5:
        raise ValueError(f"Only {len(test_df)} test bars after alignment")

    # Warn if any return column has NaN inside the test window (should not happen
    # because pct_change is computed on the full warmup+test frame).
    for sym in ["spy", "veu", "bil", "agg"]:
        n_nan = test_df[f"r_{sym}"].isna().sum()
        if n_nan:
            log.warning(
                "NaN returns in test window for %s: %d bars — zero-filling",
                sym.upper(),
                n_nan,
            )
            test_df[f"r_{sym}"] = test_df[f"r_{sym}"].fillna(0.0)

    # Portfolio return on bar T = return of the asset held ENTERING bar T
    # (= holding_prev, established at the previous EOM close — no look-ahead)
    port_ret = pd.Series(0.0, index=test_df.index)
    for sym in ["SPY", "VEU", "BIL", "AGG"]:
        mask = test_df["holding_prev"] == sym
        port_ret[mask] = test_df.loc[mask, f"r_{sym.lower()}"]

    # Transaction cost: charged on the EOM bar where we establish a new holding.
    # (holding != holding_prev AND holding_prev is not NaN = valid prior holding)
    switched = (test_df["holding"] != test_df["holding_prev"]) & test_df[
        "holding_prev"
    ].notna()
    cost = switched.astype(float) * TOTAL_COST_BPS / 10_000.0

    net_ret = port_ret - cost

    eq = INITIAL_CAPITAL * (1.0 + net_ret).cumprod()
    n_years = len(test_df) / 252.0
    total_ret = float(eq.iloc[-1]) / INITIAL_CAPITAL - 1.0
    cagr = (1.0 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1.0

    daily_rets = net_ret.values
    sharpe = (
        float(np.mean(daily_rets))
        / (float(np.std(daily_rets, ddof=1)) + 1e-10)
        * np.sqrt(252)
    )

    peak = eq.cummax()
    max_dd = float(((eq - peak) / (peak + 1e-10)).min())
    calmar = cagr / (abs(max_dd) + 1e-10)

    return {
        "test_cagr": cagr,
        "test_sharpe": sharpe,
        "test_max_dd": max_dd,
        "test_calmar": calmar,
        "test_trades": int(switched.sum()),
    }


# ---------------------------------------------------------------------------
# 4 — Benchmarks
# ---------------------------------------------------------------------------
def _spy_buyhold(spy: pd.DataFrame, ts: pd.Timestamp, te: pd.Timestamp) -> dict:
    df = spy[(spy["timestamp"] >= ts) & (spy["timestamp"] < te)].sort_values(
        "timestamp"
    )
    if len(df) < 5:
        return {
            "bh_cagr": float("nan"),
            "bh_sharpe": float("nan"),
            "bh_max_dd": float("nan"),
        }
    rets = df["close"].pct_change().dropna()
    if len(rets) < 2:
        return {
            "bh_cagr": float("nan"),
            "bh_sharpe": float("nan"),
            "bh_max_dd": float("nan"),
        }
    n_years = len(df) / 252.0
    total_ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1.0
    cagr = (1.0 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1.0
    sharpe = float(rets.mean()) / (float(rets.std(ddof=1)) + 1e-10) * np.sqrt(252)
    cum = (1.0 + rets).cumprod()
    peak = cum.cummax()
    max_dd = float(((cum - peak) / peak).min())
    return {"bh_cagr": cagr, "bh_sharpe": sharpe, "bh_max_dd": max_dd}


def _6040_buyhold(
    spy: pd.DataFrame, agg: pd.DataFrame, ts: pd.Timestamp, te: pd.Timestamp
) -> dict:
    """Static initial-allocation 60/40 SPY/AGG (no rebalancing — true buy-and-hold)."""
    s = (
        spy[(spy["timestamp"] >= ts) & (spy["timestamp"] < te)]
        .sort_values("timestamp")
        .set_index("timestamp")["close"]
    )
    a = (
        agg[(agg["timestamp"] >= ts) & (agg["timestamp"] < te)]
        .sort_values("timestamp")
        .set_index("timestamp")["close"]
    )
    combined = pd.DataFrame({"spy": s, "agg": a}).dropna()
    if len(combined) < 5:
        return {
            "b6040_cagr": float("nan"),
            "b6040_sharpe": float("nan"),
            "b6040_max_dd": float("nan"),
        }
    spy0, agg0 = combined["spy"].iloc[0], combined["agg"].iloc[0]
    shares_spy = 0.6 * INITIAL_CAPITAL / spy0
    shares_agg = 0.4 * INITIAL_CAPITAL / agg0
    equity = shares_spy * combined["spy"] + shares_agg * combined["agg"]
    n_years = len(combined) / 252.0
    total_ret = float(equity.iloc[-1]) / INITIAL_CAPITAL - 1.0
    cagr = (1.0 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1.0
    rets = equity.pct_change().dropna()
    sharpe = float(rets.mean()) / (float(rets.std(ddof=1)) + 1e-10) * np.sqrt(252)
    peak = equity.cummax()
    max_dd = float(((equity - peak) / peak).min())
    return {"b6040_cagr": cagr, "b6040_sharpe": sharpe, "b6040_max_dd": max_dd}


def _6040_rebalanced(
    spy: pd.DataFrame, agg: pd.DataFrame, ts: pd.Timestamp, te: pd.Timestamp
) -> dict:
    """Daily-rebalanced 60/40 SPY/AGG — maintains fixed 60/40 weights each bar.

    Fair comparison against a monthly-rebalancing strategy.
    No transaction costs applied (pure benchmark).
    """
    s = (
        spy[(spy["timestamp"] >= ts) & (spy["timestamp"] < te)]
        .sort_values("timestamp")
        .set_index("timestamp")["close"]
    )
    a = (
        agg[(agg["timestamp"] >= ts) & (agg["timestamp"] < te)]
        .sort_values("timestamp")
        .set_index("timestamp")["close"]
    )
    combined = pd.DataFrame({"spy": s, "agg": a}).dropna()
    if len(combined) < 5:
        return {
            "r6040_cagr": float("nan"),
            "r6040_sharpe": float("nan"),
            "r6040_max_dd": float("nan"),
        }
    r_spy = combined["spy"].pct_change()
    r_agg = combined["agg"].pct_change()
    port_ret = (0.6 * r_spy + 0.4 * r_agg).dropna()
    equity = INITIAL_CAPITAL * (1.0 + port_ret).cumprod()
    n_years = len(port_ret) / 252.0
    total_ret = float(equity.iloc[-1]) / INITIAL_CAPITAL - 1.0
    cagr = (1.0 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1.0
    sharpe = (
        float(port_ret.mean()) / (float(port_ret.std(ddof=1)) + 1e-10) * np.sqrt(252)
    )
    peak = equity.cummax()
    max_dd = float(((equity - peak) / peak).min())
    return {"r6040_cagr": cagr, "r6040_sharpe": sharpe, "r6040_max_dd": max_dd}


# ---------------------------------------------------------------------------
# 5 — Main
# ---------------------------------------------------------------------------
def main() -> int:
    fetch_start = PERIOD_START - pd.Timedelta(days=int(WARMUP_BARS * 1.5))
    try:
        prices = _fetch_alpaca(SYMBOLS, start=fetch_start, end=PERIOD_END)
    except Exception as exc:
        log.error("Alpaca fetch failed: %s", exc)
        _write_failure_report(str(exc))
        return 1

    prices_by_sym: dict[str, pd.DataFrame] = {}
    for sym in SYMBOLS:
        prices_by_sym[sym] = prices[prices["symbol"] == sym].copy()
        if prices_by_sym[sym].empty:
            _write_failure_report(f"{sym} data missing from Alpaca response")
            return 1

    actual_start = max(df["timestamp"].min() for df in prices_by_sym.values())
    actual_end = min(df["timestamp"].max() for df in prices_by_sym.values())
    log.info("Overlapping data range: %s → %s", actual_start.date(), actual_end.date())

    from src.assembled_core.qa.walk_forward import (
        WalkForwardConfig,
        run_walk_forward_backtest,
    )

    wf_start = max(PERIOD_START, actual_start.normalize())
    wf_end = min(PERIOD_END, actual_end.normalize())

    config = WalkForwardConfig(
        start_date=wf_start,
        end_date=wf_end,
        train_window_days=TRAIN_WINDOW_DAYS,
        test_window_days=TEST_WINDOW_DAYS,
        step_size_days=STEP_SIZE_DAYS,
        mode="rolling",
        min_train_periods=200,
        min_test_periods=200,
    )

    def backtest_fn(train_start, train_end, test_start, test_end) -> dict:
        return _simulate_dual_momentum(prices_by_sym, test_start, test_end)

    log.info("Running walk-forward…")
    try:
        wf_result = run_walk_forward_backtest(config=config, backtest_fn=backtest_fn)
    except Exception as exc:
        log.error("Walk-forward failed: %s", exc, exc_info=True)
        _write_failure_report(str(exc))
        return 1

    summary_ok = (
        not wf_result.summary_df.empty
        and "split_index" in wf_result.summary_df.columns
        and "test_cagr" in wf_result.summary_df.columns
    )
    if not summary_ok:
        log.warning(
            "[WF] summary_df empty or missing expected columns — "
            "per-fold metrics will be NaN (check backtest_fn return schema)"
        )

    spy_df = prices_by_sym["SPY"]
    agg_df = prices_by_sym["AGG"]

    fold_rows = []
    for wr in wf_result.window_results:
        w = wr.window
        spy_bh = _spy_buyhold(spy_df, w.test_start, w.test_end)
        b6040 = _6040_buyhold(spy_df, agg_df, w.test_start, w.test_end)
        r6040 = _6040_rebalanced(spy_df, agg_df, w.test_start, w.test_end)

        if wr.status == "failed":
            fold_rows.append(
                {
                    "fold": w.split_index + 1,
                    "test_start": w.test_start.date(),
                    "test_end": w.test_end.date(),
                    "train_start": w.train_start.date(),
                    "train_end": w.train_end.date(),
                    "cagr": float("nan"),
                    "sharpe": float("nan"),
                    "max_dd": float("nan"),
                    "calmar": float("nan"),
                    "n_trades": float("nan"),
                    **spy_bh,
                    **b6040,
                    **r6040,
                    "status": "FAILED",
                    "error": wr.error_message,
                }
            )
        else:
            summary = wf_result.summary_df
            row_data: dict = {}
            if summary_ok:
                matching = summary[summary["split_index"] == w.split_index]
                if len(matching) > 0:
                    row_data = matching.iloc[0].to_dict()
            fold_rows.append(
                {
                    "fold": w.split_index + 1,
                    "test_start": w.test_start.date(),
                    "test_end": w.test_end.date(),
                    "train_start": w.train_start.date(),
                    "train_end": w.train_end.date(),
                    "cagr": float(row_data.get("test_cagr", float("nan"))),
                    "sharpe": float(row_data.get("test_sharpe", float("nan"))),
                    "max_dd": float(row_data.get("test_max_dd", float("nan"))),
                    "calmar": float(row_data.get("test_calmar", float("nan"))),
                    "n_trades": float(row_data.get("test_trades", float("nan"))),
                    **spy_bh,
                    **b6040,
                    **r6040,
                    "status": "OK",
                    "error": None,
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    ok = fold_df[fold_df["status"] == "OK"]
    n_ok, n_total = len(ok), len(fold_df)

    if n_ok == 0:
        _write_failure_report("All folds failed")
        return 1

    agg_metrics = {
        "mean_cagr": ok["cagr"].mean(),
        "mean_sharpe": ok["sharpe"].mean(),
        "mean_max_dd": ok["max_dd"].mean(),
        "mean_calmar": ok["calmar"].mean(),
        "win_rate": (ok["cagr"] > 0).mean(),
        "beats_spy_cagr": (ok["cagr"] > ok["bh_cagr"]).mean(),
        "beats_spy_sharpe": (ok["sharpe"] > ok["bh_sharpe"]).mean(),
        "beats_6040r_sharpe": (ok["sharpe"] > ok["r6040_sharpe"]).mean(),
        "mean_bh_cagr": ok["bh_cagr"].mean(),
        "mean_bh_sharpe": ok["bh_sharpe"].mean(),
        "mean_bh_max_dd": ok["bh_max_dd"].mean(),
        "mean_b6040_cagr": ok["b6040_cagr"].mean(),
        "mean_b6040_sharpe": ok["b6040_sharpe"].mean(),
        "mean_b6040_max_dd": ok["b6040_max_dd"].mean(),
        "mean_r6040_cagr": ok["r6040_cagr"].mean(),
        "mean_r6040_sharpe": ok["r6040_sharpe"].mean(),
        "mean_r6040_max_dd": ok["r6040_max_dd"].mean(),
        "mean_dd_ratio": (ok["max_dd"] / ok["bh_max_dd"].replace(0, float("nan")))
        .abs()
        .mean(),
    }

    _write_report(
        fold_df=fold_df,
        agg=agg_metrics,
        n_ok=n_ok,
        n_total=n_total,
        actual_start=actual_start,
        actual_end=actual_end,
    )
    return 0


# ---------------------------------------------------------------------------
# Report helpers
# ---------------------------------------------------------------------------
def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v * 100:.1f}%"


def _fmt_f(v, d: int = 2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{d}f}"


def _write_failure_report(reason: str) -> None:
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(
        f"# dual_momentum — OOS Walk-Forward FAILED\n\n**Grund:** {reason}\n",
        encoding="utf-8",
    )
    log.error("Wrote failure report to %s", OUT_MD)


def _write_report(*, fold_df, agg, n_ok, n_total, actual_start, actual_end):
    mean_dd_overlay = agg["mean_max_dd"]
    mean_dd_spy = agg["mean_bh_max_dd"]
    dd_ratio_val = float("nan")
    if (
        not (np.isnan(mean_dd_overlay) or np.isnan(mean_dd_spy))
        and abs(mean_dd_spy) > 1e-6
    ):
        dd_ratio_val = abs(mean_dd_overlay) / abs(mean_dd_spy)
        dd_criterion = dd_ratio_val <= 0.70
    else:
        dd_criterion = False

    sharpe_criterion = (
        not np.isnan(agg["mean_sharpe"])
        and not np.isnan(agg["mean_bh_sharpe"])
        and agg["mean_sharpe"] >= agg["mean_bh_sharpe"] + 0.2
    )
    beats_6040r = (
        not np.isnan(agg["mean_sharpe"])
        and not np.isnan(agg["mean_r6040_sharpe"])
        and agg["mean_sharpe"] >= agg["mean_r6040_sharpe"]
    )

    lines = [
        "# dual_momentum — Echter OOS Walk-Forward (Alpaca, 2026-05)",
        "",
        "**Erstellt:** 2026-05-29  ",
        "**Branch:** main  ",
        "**Zweck:** GO_LIVE Kandidat A — Dual Momentum (Antonacci-Variante).",
        "",
        "---",
        "",
        "## Strategie",
        "",
        "Vier-Asset-Universum: SPY (US-Equity), VEU (Ex-US-Equity), BIL (T-Bills), AGG (Bonds).",
        "",
        "```",
        "ret_12m(X) = close_today / close_12M_ago − 1  [kausal, Kalender-Monate]",
        "Outperformer = argmax(ret_12m(SPY), ret_12m(VEU))",
        "Absolute-Filter: wenn ret_12m(Outperformer) > ret_12m(BIL):",
        "    halte Outperformer  (voll investiert, weight=1.0)",
        "Sonst: halte AGG  (Safe-Asset)",
        "Re-Balance: letzter Handelstag jeden Monats",
        "```",
        "",
        "## Datenquelle",
        "",
        "- **Anbieter:** Alpaca Markets — split-adjustiert, `StockHistoricalDataClient`",
        f"- **Symbole:** {', '.join(SYMBOLS)}",
        f"- **Tatsächliche Zeitspanne:** {actual_start.date()} → {actual_end.date()}",
        "  (VEU-Inception: 2007-03-02; BIL-Inception: 2007-05-25; limitierender Faktor)",
        "",
        "## Walk-Forward-Konfiguration",
        "",
        "- Modus: Rolling",
        f"- Train/Test/Step: {TRAIN_WINDOW_DAYS}/{TEST_WINDOW_DAYS}/{STEP_SIZE_DAYS} Handelstage (~1 Jahr)",
        f"- Warmup-Buffer: {WARMUP_BARS} Bars vor Testbeginn ({LOOKBACK_MONTHS}M Lookback + Buffer)",
        f"- Transaktionskosten: {COMMISSION_BPS} bps Commission + {SPREAD_W}+{IMPACT_W} Spread/Impact"
        f" = {TOTAL_COST_BPS} bps je Positions-Switch (100 % Turnover)",
        f"- Startkapital: {INITIAL_CAPITAL:,.0f} USD",
        "- Gewichte: monatliches Rebalancing auf Zielgewicht 1.0 (eine Position zur Zeit)",
        "",
        "---",
        "",
        "## Ergebnisse pro Fold",
        "",
        "| Fold | Test-Periode | CAGR | Sharpe | MaxDD | Calmar | Trades"
        " | SPY CAGR | SPY Sharpe | SPY MaxDD | 60/40 B&H CAGR | 60/40 Reb. CAGR | 60/40 Reb. Sharpe |",
        "|------|-------------|------|--------|-------|--------|--------|----------|------------|-----------|---------------|----------------|-----------------|",
    ]

    for _, r in fold_df.iterrows():
        if r["status"] == "FAILED":
            lines.append(
                f"| {int(r['fold'])} | {r['test_start']}–{r['test_end']}"
                f" | N/A | N/A | N/A | N/A | N/A"
                f" | {_fmt_pct(r.get('bh_cagr'))} | {_fmt_f(r.get('bh_sharpe'))}"
                f" | {_fmt_pct(r.get('bh_max_dd'))}"
                f" | {_fmt_pct(r.get('b6040_cagr'))}"
                f" | {_fmt_pct(r.get('r6040_cagr'))} | {_fmt_f(r.get('r6040_sharpe'))} |"
            )
            lines.append(f"> Fold {int(r['fold'])} FAILED: {r['error']}")
        else:
            lines.append(
                f"| {int(r['fold'])} | {r['test_start']}–{r['test_end']}"
                f" | {_fmt_pct(r['cagr'])} | {_fmt_f(r['sharpe'])}"
                f" | {_fmt_pct(r['max_dd'])} | {_fmt_f(r['calmar'])}"
                f" | {int(r['n_trades']) if not np.isnan(r['n_trades']) else 'N/A'}"
                f" | {_fmt_pct(r.get('bh_cagr'))} | {_fmt_f(r.get('bh_sharpe'))}"
                f" | {_fmt_pct(r.get('bh_max_dd'))}"
                f" | {_fmt_pct(r.get('b6040_cagr'))}"
                f" | {_fmt_pct(r.get('r6040_cagr'))} | {_fmt_f(r.get('r6040_sharpe'))} |"
            )

    lines += [
        "",
        f"_Erfolgreiche Folds: {n_ok}/{n_total}_",
        "",
        "---",
        "",
        "## Aggregierte OOS-Metriken",
        "",
        "| Metrik | dual_momentum | SPY B&H | 60/40 B&H | 60/40 Rebalanced |",
        "|--------|---------------|---------|-----------|-----------------|",
        f"| Ø CAGR | {_fmt_pct(agg['mean_cagr'])} | {_fmt_pct(agg['mean_bh_cagr'])} | {_fmt_pct(agg['mean_b6040_cagr'])} | {_fmt_pct(agg['mean_r6040_cagr'])} |",
        f"| Ø Sharpe | {_fmt_f(agg['mean_sharpe'])} | {_fmt_f(agg['mean_bh_sharpe'])} | {_fmt_f(agg['mean_b6040_sharpe'])} | {_fmt_f(agg['mean_r6040_sharpe'])} |",
        f"| Ø MaxDD | {_fmt_pct(agg['mean_max_dd'])} | {_fmt_pct(agg['mean_bh_max_dd'])} | {_fmt_pct(agg['mean_b6040_max_dd'])} | {_fmt_pct(agg['mean_r6040_max_dd'])} |",
        f"| Ø Calmar | {_fmt_f(agg['mean_calmar'])} | — | — | — |",
        f"| Win-Rate (CAGR > 0) | {_fmt_pct(agg['win_rate'])} | — | — | — |",
        f"| Folds, die SPY CAGR schlagen | {_fmt_pct(agg['beats_spy_cagr'])} | — | — | — |",
        f"| Folds, die SPY Sharpe schlagen | {_fmt_pct(agg['beats_spy_sharpe'])} | — | — | — |",
        f"| Folds, die 60/40-Reb. Sharpe schlagen | {_fmt_pct(agg['beats_6040r_sharpe'])} | — | — | — |",
        "",
        "---",
        "",
        "## Drawdown-Analyse",
        "",
        f"- Ø MaxDD dual_momentum: **{_fmt_pct(agg['mean_max_dd'])}**",
        f"- Ø MaxDD SPY: **{_fmt_pct(agg['mean_bh_max_dd'])}**",
        f"- MaxDD-Verhältnis overlay/SPY: **{_fmt_f(dd_ratio_val, 2)}x**"
        + (
            f" ({(1 - dd_ratio_val) * 100:.1f}% Verbesserung)"
            if not np.isnan(dd_ratio_val)
            else ""
        ),
        "",
        "---",
        "",
        "## Quervergleich mit vol_target_overlay",
        "",
        "_Achtung: anderer Testzeitraum (VEU/BIL-Inception 2007 vs IEF-Inception 2003)._",
        "_Ergebnisse aus separatem Lauf (Alpaca, 2026-05-28, 12/13 Folds, 2016–2025)._",
        "",
        "| Metrik | dual_momentum (dieser Lauf) | vol_target_overlay (Referenz) |",
        "|--------|-----------------------------|-------------------------------|",
        f"| Ø CAGR | {_fmt_pct(agg['mean_cagr'])} | 8.8% |",
        f"| Ø Sharpe | {_fmt_f(agg['mean_sharpe'])} | 0.88 |",
        f"| Ø MaxDD | {_fmt_pct(agg['mean_max_dd'])} | -8.4% |",
        f"| MaxDD-Ratio vs SPY | {_fmt_f(dd_ratio_val, 2)}x | 0.68x |",
        "",
        "---",
        "",
        "## Bewertung",
        "",
    ]

    # Honest assessment
    if dd_criterion and sharpe_criterion:
        verdict = (
            "**Beide Kriterien erfüllt.** MaxDD-Ratio ≤ 0.70x (Drawdown-Reduktion ≥ 30 %) "
            f"und Ø Sharpe ≥ SPY + 0.2. Ø CAGR {_fmt_pct(agg['mean_cagr'])}, "
            f"Ø Calmar {_fmt_f(agg['mean_calmar'])}."
        )
    elif dd_criterion:
        verdict = (
            f"**MaxDD-Kriterium erfüllt, Sharpe-Kriterium nicht erfüllt.** "
            f"MaxDD-Ratio {_fmt_f(dd_ratio_val, 2)}x (≥ 30 % Reduktion), "
            f"aber Ø Sharpe {_fmt_f(agg['mean_sharpe'])} < "
            f"SPY Ø {_fmt_f(agg['mean_bh_sharpe'])} + 0.2. "
            f"Ø CAGR {_fmt_pct(agg['mean_cagr'])}, Ø Calmar {_fmt_f(agg['mean_calmar'])}. "
            "Teilerfolg — das Rendite/Risiko-Ziel wird nicht vollständig erreicht."
        )
    elif beats_6040r:
        verdict = (
            f"**Strategie schlägt 60/40 Rebalanced (Sharpe), aber verfehlt SPY-Kriterien.** "
            f"Ø Sharpe {_fmt_f(agg['mean_sharpe'])} vs 60/40 Reb. {_fmt_f(agg['mean_r6040_sharpe'])}. "
            f"MaxDD-Ratio {_fmt_f(dd_ratio_val, 2)}x — Drawdown-Reduktion verfehlt."
        )
    else:
        verdict = (
            f"**Kein Kriterium erfüllt.** "
            f"Ø CAGR {_fmt_pct(agg['mean_cagr'])} vs SPY {_fmt_pct(agg['mean_bh_cagr'])}, "
            f"Ø Sharpe {_fmt_f(agg['mean_sharpe'])} vs SPY {_fmt_f(agg['mean_bh_sharpe'])} "
            f"und 60/40 Reb. {_fmt_f(agg['mean_r6040_sharpe'])}. "
            "Dual Momentum liefert auf diesem Sample keinen messbaren Mehrwert gegenüber "
            "einfachen passiven Benchmarks."
        )

    lines.append(verdict)
    lines += [
        "",
        "### Einschränkungen",
        "",
        "- SPY, VEU, BIL, AGG ohne Dividenden-Reinvestition (Alpaca bar close ≈ Kursrendite).",
        "  VEU-Dividendenrendite ~3 %, AGG-Coupon ~3–4 % p.a. fehlen — IEF-/AGG-Returns unterschätzt.",
        "- BIL als T-Bill-Proxy: Kursrendite nahezu 0 (korrekte Hurdle-Proxy-Eigenschaft).",
        "- Kosten: 10.75 bps je Positions-Switch (monatliches Rebalancing = ~12 Switches/Jahr);",
        "  sehr niedrige Transaktionskosten — Schätzung eher günstig.",
        "- Walk-Forward deckt nur Alpaca-Verfügbarkeit (VEU/BIL ab 2007-05);",
        "  enthält GFC 2008–09 und COVID 2020 — breitere Krisenabdeckung als vol_target (ab 2016).",
        "- Parameter (lookback=12M, BIL-Hurdle) sind Antonacci-Standard, nicht optimiert.",
        "- Quervergleich mit vol_target nicht direkt, da anderer Datenzeitraum und andere Assets.",
        "",
        "---",
        "",
        "_Dieses Dokument ist ein automatisch erzeugtes Artefakt aus_"
        " `scripts/_oos_wf_dual_momentum.py`. _Nicht manuell editieren._",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Report written to %s", OUT_MD)

    # Print summary to stdout
    print("\n" + "=" * 70)
    print("DUAL MOMENTUM — OOS Walk-Forward Results")
    print("=" * 70)
    print(f"Folds: {n_ok}/{n_total} erfolgreich")
    print(f"Zeitraum: {actual_start.date()} -> {actual_end.date()}")
    print()
    print(f"{'Metrik':<30} {'dual_momentum':>15} {'SPY B&H':>10} {'60/40 Reb.':>12}")
    print("-" * 70)
    print(
        f"{'Ø CAGR':<30} {_fmt_pct(agg['mean_cagr']):>15}"
        f" {_fmt_pct(agg['mean_bh_cagr']):>10}"
        f" {_fmt_pct(agg['mean_r6040_cagr']):>12}"
    )
    print(
        f"{'Ø Sharpe':<30} {_fmt_f(agg['mean_sharpe']):>15}"
        f" {_fmt_f(agg['mean_bh_sharpe']):>10}"
        f" {_fmt_f(agg['mean_r6040_sharpe']):>12}"
    )
    print(
        f"{'Ø MaxDD':<30} {_fmt_pct(agg['mean_max_dd']):>15}"
        f" {_fmt_pct(agg['mean_bh_max_dd']):>10}"
        f" {_fmt_pct(agg['mean_r6040_max_dd']):>12}"
    )
    print(f"{'Ø Calmar':<30} {_fmt_f(agg['mean_calmar']):>15} {'—':>10} {'—':>12}")
    print(
        f"{'MaxDD-Ratio vs SPY':<30} {_fmt_f(dd_ratio_val, 2) + 'x':>15}"
        f" {'—':>10} {'—':>12}"
    )
    print()
    print("Bewertung:")
    print(verdict)
    print("=" * 70)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sys.exit(main())
