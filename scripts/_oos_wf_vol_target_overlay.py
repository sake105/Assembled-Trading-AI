"""One-shot OOS Walk-Forward for vol_target_overlay — writes docs/results/2026_05_vol_target_overlay_real_oos.md.

Usage:
    python scripts/_oos_wf_vol_target_overlay.py

Requirements:
    ALPACA_API_KEY / ALPACA_API_SECRET in .env or environment.
    pip install alpaca-py (already installed in this project).

Design:
    - Two-asset strategy: SPY (risk) + IEF (defensive Treasuries ETF).
    - Volatility-targeting: weight_spy = min(1.0, target_vol / realized_vol),
      with a 200-bar SMA trend filter that halves the SPY weight when close < SMA.
    - Fetches Alpaca daily bars for SPY + IEF (IEF available from 2002-07).
    - Walk-forward: 252/252/252 (train / test / step), same as trend_baseline.
    - Warmup buffer: SMA_WINDOW + VOL_LOOKBACK bars prepended so indicators
      are hot at the start of each test period.
    - Two benchmarks: SPY buy-and-hold + static 60/40 SPY/IEF.
    - Extra metrics: MaxDD ratio vs SPY, Calmar ratio.
    - 10 bps commission + 0.25+0.5 spread/impact (matching policy.yaml C3).
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
log = logging.getLogger("oos_wf_vol_target")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# IEF inception: 2002-07-30.  Request from 2002-01 to maximise coverage.
PERIOD_START = pd.Timestamp(
    "2003-01-01", tz="UTC"
)  # ~1y after IEF inception for warmup
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW_DAYS = 252
TEST_WINDOW_DAYS = 252
STEP_SIZE_DAYS = 252
TARGET_VOL = 0.12
VOL_LOOKBACK = 20
SMA_WINDOW = 200
WARMUP_BARS = SMA_WINDOW + VOL_LOOKBACK + 10  # extra buffer
COMMISSION_BPS = 10.0
SPREAD_W = 0.25
IMPACT_W = 0.5
TOTAL_COST_BPS = COMMISSION_BPS + SPREAD_W + IMPACT_W  # 10.75 bps per unit of turnover
INITIAL_CAPITAL = 100_000.0
DEFENSIVE_ASSET = "IEF"
RISK_ASSET = "SPY"
OUT_MD = ROOT / "docs" / "results" / "2026_05_vol_target_overlay_real_oos.md"


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
# 3 — Direct equity simulation for one fold
# ---------------------------------------------------------------------------
def _simulate_vol_target(
    spy_prices: pd.DataFrame,
    ief_prices: pd.DataFrame,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> dict:
    """Simulate vol-target overlay for one WF fold; return CAGR/Sharpe/MaxDD/Calmar.

    Signal computation uses prices up to and including the current bar (causal).
    Portfolio return on day t is realised by holding the weights set at close of
    day t-1 (i.e., weights lag returns by one bar — no look-ahead).

    Costs are applied on the absolute weight change (turnover) per day.
    """
    from src.assembled_core.strategies.vol_target_overlay import (
        generate_vol_target_signals_from_prices,
    )

    # Include warmup bars before test_start so indicators are hot.
    all_spy_dates = np.sort(spy_prices["timestamp"].unique())
    pre_test = all_spy_dates[all_spy_dates < test_start]
    warmup_start = (
        pre_test[-WARMUP_BARS] if len(pre_test) >= WARMUP_BARS else test_start
    )

    spy_w = (
        spy_prices[
            (spy_prices["timestamp"] >= warmup_start)
            & (spy_prices["timestamp"] < test_end)
        ]
        .sort_values("timestamp")
        .copy()
    )

    ief_w = (
        ief_prices[
            (ief_prices["timestamp"] >= warmup_start)
            & (ief_prices["timestamp"] < test_end)
        ]
        .sort_values("timestamp")
        .copy()
    )

    if spy_w.empty or ief_w.empty:
        raise ValueError(f"No price data for {test_start.date()}–{test_end.date()}")

    # Compute signals on warmup+test window (strictly causal)
    sigs = generate_vol_target_signals_from_prices(
        spy_w,
        target_vol=TARGET_VOL,
        vol_lookback=VOL_LOOKBACK,
        sma_window=SMA_WINDOW,
        defensive_asset=DEFENSIVE_ASSET,
        risk_asset=RISK_ASSET,
    )

    if sigs.empty:
        raise ValueError("No signals after warmup — insufficient data")

    # Pivot to per-timestamp weights
    spy_sigs = sigs[sigs["symbol"] == RISK_ASSET].set_index("timestamp")["score"]
    ief_sigs = sigs[sigs["symbol"] == DEFENSIVE_ASSET].set_index("timestamp")["score"]

    # Build per-day price table aligned on common trading dates (inner join)
    spy_close = spy_w.set_index("timestamp")["close"]
    ief_close = ief_w.set_index("timestamp")["close"]
    combined = pd.DataFrame({"spy": spy_close, "ief": ief_close}).dropna()
    combined["w_spy"] = spy_sigs.reindex(combined.index).ffill().fillna(0.0)
    combined["w_ief"] = ief_sigs.reindex(combined.index).ffill().fillna(1.0)

    # F-4: assert weights sum to 1.0 after inner join
    weight_sum_err = (combined["w_spy"] + combined["w_ief"] - 1.0).abs().max()
    if weight_sum_err > 1e-6:
        raise ValueError(f"Weights do not sum to 1.0 (max error={weight_sum_err:.2e})")

    # F-3: compute returns on FULL combined frame so first test bar has prior-bar context
    combined["r_spy"] = combined["spy"].pct_change()
    combined["r_ief"] = combined["ief"].pct_change()

    # F-senior-1 (BLOCKER): compute lag and turnover on full combined frame so the first
    # test bar inherits the warmup-end weight, not zero. shift(1) inside the test slice
    # would drop the prior-warmup weight, making every fold start flat — incorrect.
    combined["w_spy_lag"] = combined["w_spy"].shift(1)
    combined["w_ief_lag"] = combined["w_ief"].shift(1)
    # F-2/F-senior-2: turnover single-sided; w_ief = 1-w_spy so |Δw_ief| == |Δw_spy|
    combined["turnover"] = combined["w_spy"].diff().abs()

    # Filter to test period for performance measurement
    test_mask = (combined.index >= test_start) & (combined.index < test_end)
    test_df = combined[test_mask].copy()

    if len(test_df) < 5:
        raise ValueError(f"Only {len(test_df)} test bars after alignment")

    test_df["r_spy"] = test_df["r_spy"].fillna(0.0)
    test_df["r_ief"] = test_df["r_ief"].fillna(0.0)
    # fillna here is only a safety-net for the very first bar of the entire combined
    # frame (never reached in test period since warmup precedes it)
    test_df["w_spy_lag"] = test_df["w_spy_lag"].fillna(0.0)
    test_df["w_ief_lag"] = test_df["w_ief_lag"].fillna(1.0)
    test_df["turnover"] = test_df["turnover"].fillna(test_df["w_spy"].abs())
    test_df["cost"] = test_df["turnover"] * TOTAL_COST_BPS / 10_000.0

    # Portfolio gross return per day
    # Weights applied to NEXT day's returns (weights set at close of prior day).
    test_df["port_ret"] = (
        test_df["w_spy_lag"] * test_df["r_spy"]
        + test_df["w_ief_lag"] * test_df["r_ief"]
    )

    # Net return
    test_df["net_ret"] = test_df["port_ret"] - test_df["cost"]

    # Equity curve
    eq = INITIAL_CAPITAL * (1.0 + test_df["net_ret"]).cumprod()
    n_years = len(test_df) / 252.0
    total_ret = float(eq.iloc[-1]) / INITIAL_CAPITAL - 1.0
    cagr = (1.0 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1.0

    daily_rets = test_df["net_ret"].values
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
        "test_trades": int((test_df["turnover"] > 0.001).sum()),
    }


# ---------------------------------------------------------------------------
# 4 — Benchmarks: SPY B&H and 60/40 B&H
# ---------------------------------------------------------------------------
def _spy_buyhold(
    spy: pd.DataFrame, ief: pd.DataFrame, ts: pd.Timestamp, te: pd.Timestamp
) -> dict:
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
    n_years = len(df) / 252.0
    total_ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1.0
    cagr = (1.0 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1.0
    sharpe = float(rets.mean()) / (float(rets.std(ddof=1)) + 1e-10) * np.sqrt(252)
    cum = (1.0 + rets).cumprod()
    peak = cum.cummax()
    max_dd = float(((cum - peak) / peak).min())
    return {"bh_cagr": cagr, "bh_sharpe": sharpe, "bh_max_dd": max_dd}


def _6040_buyhold(
    spy: pd.DataFrame, ief: pd.DataFrame, ts: pd.Timestamp, te: pd.Timestamp
) -> dict:
    """Static initial-allocation 60/40 SPY/IEF (no rebalancing — true buy-and-hold)."""
    s = (
        spy[(spy["timestamp"] >= ts) & (spy["timestamp"] < te)]
        .sort_values("timestamp")
        .set_index("timestamp")["close"]
    )
    i = (
        ief[(ief["timestamp"] >= ts) & (ief["timestamp"] < te)]
        .sort_values("timestamp")
        .set_index("timestamp")["close"]
    )
    combined = pd.DataFrame({"spy": s, "ief": i}).dropna()
    if len(combined) < 5:
        return {
            "b6040_cagr": float("nan"),
            "b6040_sharpe": float("nan"),
            "b6040_max_dd": float("nan"),
        }

    # Initial share counts based on 60/40 split
    spy0, ief0 = combined["spy"].iloc[0], combined["ief"].iloc[0]
    shares_spy = 0.6 * INITIAL_CAPITAL / spy0
    shares_ief = 0.4 * INITIAL_CAPITAL / ief0
    equity = shares_spy * combined["spy"] + shares_ief * combined["ief"]

    n_years = len(combined) / 252.0
    total_ret = float(equity.iloc[-1]) / INITIAL_CAPITAL - 1.0
    cagr = (1.0 + total_ret) ** (1.0 / max(n_years, 0.01)) - 1.0
    rets = equity.pct_change().dropna()
    sharpe = float(rets.mean()) / (float(rets.std(ddof=1)) + 1e-10) * np.sqrt(252)
    peak = equity.cummax()
    max_dd = float(((equity - peak) / peak).min())
    return {"b6040_cagr": cagr, "b6040_sharpe": sharpe, "b6040_max_dd": max_dd}


# ---------------------------------------------------------------------------
# 5 — Main
# ---------------------------------------------------------------------------
def main() -> int:
    # 5.1 Fetch data — extend fetch start by warmup calendar days
    fetch_start = PERIOD_START - pd.Timedelta(days=int(WARMUP_BARS * 1.5))
    try:
        prices = _fetch_alpaca(
            [RISK_ASSET, DEFENSIVE_ASSET], start=fetch_start, end=PERIOD_END
        )
    except Exception as exc:
        log.error("Alpaca fetch failed: %s", exc)
        _write_failure_report(str(exc))
        return 1

    spy_prices = prices[prices["symbol"] == RISK_ASSET].copy()
    ief_prices = prices[prices["symbol"] == DEFENSIVE_ASSET].copy()

    if spy_prices.empty or ief_prices.empty:
        _write_failure_report("SPY or IEF data missing from Alpaca response")
        return 1

    actual_start = max(spy_prices["timestamp"].min(), ief_prices["timestamp"].min())
    actual_end = min(spy_prices["timestamp"].max(), ief_prices["timestamp"].max())
    log.info("Overlapping data range: %s → %s", actual_start.date(), actual_end.date())

    # 5.2 Walk-forward config
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
        return _simulate_vol_target(spy_prices, ief_prices, test_start, test_end)

    log.info("Running walk-forward…")
    try:
        wf_result = run_walk_forward_backtest(config=config, backtest_fn=backtest_fn)
    except Exception as exc:
        log.error("Walk-forward failed: %s", exc, exc_info=True)
        _write_failure_report(str(exc))
        return 1

    # 5.3 Collect per-fold results
    fold_rows = []
    for wr in wf_result.window_results:
        w = wr.window
        spy_bh = _spy_buyhold(spy_prices, ief_prices, w.test_start, w.test_end)
        b6040 = _6040_buyhold(spy_prices, ief_prices, w.test_start, w.test_end)
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
                    **spy_bh,
                    **b6040,
                    "status": "FAILED",
                    "error": wr.error_message,
                }
            )
        else:
            # Metrics come from backtest_fn return dict (stored in summary_df)
            summary = wf_result.summary_df
            row_data = {}
            if not summary.empty and "split_index" in summary.columns:
                row_data = (
                    summary[summary["split_index"] == w.split_index].iloc[0].to_dict()
                    if len(summary[summary["split_index"] == w.split_index]) > 0
                    else {}
                )
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
                    **spy_bh,
                    **b6040,
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

    agg = {
        "mean_cagr": ok["cagr"].mean(),
        "mean_sharpe": ok["sharpe"].mean(),
        "mean_max_dd": ok["max_dd"].mean(),
        "mean_calmar": ok["calmar"].mean(),
        "win_rate": (ok["cagr"] > 0).mean(),
        "beats_spy_cagr": (ok["cagr"] > ok["bh_cagr"]).mean(),
        "beats_spy_sharpe": (ok["sharpe"] > ok["bh_sharpe"]).mean(),
        "mean_bh_cagr": ok["bh_cagr"].mean(),
        "mean_bh_sharpe": ok["bh_sharpe"].mean(),
        "mean_bh_max_dd": ok["bh_max_dd"].mean(),
        "mean_6040_cagr": ok["b6040_cagr"].mean(),
        "mean_6040_sharpe": ok["b6040_sharpe"].mean(),
        "mean_6040_max_dd": ok["b6040_max_dd"].mean(),
    }

    _write_report(
        fold_df=fold_df,
        agg=agg,
        n_ok=n_ok,
        n_total=n_total,
        actual_start=actual_start,
        actual_end=actual_end,
    )
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v * 100:.1f}%"


def _fmt_f(v, d=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{d}f}"


def _write_report(*, fold_df, agg, n_ok, n_total, actual_start, actual_end):
    lines = [
        "# vol_target_overlay — Echter OOS Walk-Forward (Alpaca, 2026-05)",
        "",
        "**Erstellt:** 2026-05-28  ",
        "**Branch:** main  ",
        "**Zweck:** GO_LIVE Kandidat B — Strategie-Integration Vol-Target Risk-Overlay.",
        "",
        "---",
        "",
        "## Strategie",
        "",
        "Zwei-Asset-Overlay: SPY (Risiko) + IEF (Defensiv, Barclays 7–10Y Treasuries).",
        "",
        "```",
        f"realized_vol  = std(daily_ret[-{VOL_LOOKBACK}:]) × √252   [annualisiert, kausal]",
        f"raw_weight_spy = min(1.0, {TARGET_VOL} / realized_vol)",
        f"Trend-Filter:  wenn SPY close < {SMA_WINDOW}-Tage-SMA → weight_spy ×= 0.5",
        "weight_ief    = 1 − weight_spy  (immer voll investiert)",
        "```",
        "",
        "## Datenquelle",
        "",
        "- **Anbieter:** Alpaca Markets — split-adjustiert, `StockHistoricalDataClient`",
        f"- **Symbole:** {RISK_ASSET} + {DEFENSIVE_ASSET}",
        f"- **Tatsächliche Zeitspanne:** {actual_start.date()} → {actual_end.date()}",
        f"  (IEF-Inception: 2002-07-30; Anfrage ab {PERIOD_START.date()})",
        "",
        "## Walk-Forward-Konfiguration",
        "",
        "- Modus: Rolling",
        f"- Train/Test/Step: {TRAIN_WINDOW_DAYS}/{TEST_WINDOW_DAYS}/{STEP_SIZE_DAYS} Handelstage (~1 Jahr)",
        f"- Warmup-Buffer: {WARMUP_BARS} Bars vor Testbeginn (SMA + Vol-Lookback initialisiert)",
        f"- Transaktionskosten: {COMMISSION_BPS} bps Commission + {SPREAD_W}+{IMPACT_W} Spread/Impact = {TOTAL_COST_BPS} bps je Turnover",
        f"- Startkapital: {INITIAL_CAPITAL:,.0f} USD",
        "- Gewichte: tägliche Rebalanzierung auf Zielgewichte",
        "",
        "---",
        "",
        "## Ergebnisse pro Fold",
        "",
        "| Fold | Test-Periode | CAGR | Sharpe | MaxDD | Calmar | SPY CAGR | SPY Sharpe | SPY MaxDD | 60/40 CAGR | 60/40 Sharpe |",
        "|------|-------------|------|--------|-------|--------|----------|------------|-----------|-----------|------------|",
    ]

    for _, row in fold_df.iterrows():
        lines.append(
            f"| {int(row['fold'])} "
            f"| {row['test_start']}–{row['test_end']} "
            f"| {_fmt_pct(row['cagr'])} "
            f"| {_fmt_f(row['sharpe'])} "
            f"| {_fmt_pct(row['max_dd'])} "
            f"| {_fmt_f(row['calmar'])} "
            f"| {_fmt_pct(row['bh_cagr'])} "
            f"| {_fmt_f(row['bh_sharpe'])} "
            f"| {_fmt_pct(row['bh_max_dd'])} "
            f"| {_fmt_pct(row['b6040_cagr'])} "
            f"| {_fmt_f(row['b6040_sharpe'])} |"
        )
        if row["status"] == "FAILED":
            lines.append(f"> Fold {int(row['fold'])} FAILED: {row['error']}")

    lines += [
        "",
        f"_Erfolgreiche Folds: {n_ok}/{n_total}_",
        "",
        "---",
        "",
        "## Aggregierte OOS-Metriken",
        "",
        "| Metrik | vol_target_overlay | SPY B&H | 60/40 B&H |",
        "|--------|--------------------|---------|-----------|",
        f"| Ø CAGR | {_fmt_pct(agg['mean_cagr'])} | {_fmt_pct(agg['mean_bh_cagr'])} | {_fmt_pct(agg['mean_6040_cagr'])} |",
        f"| Ø Sharpe | {_fmt_f(agg['mean_sharpe'])} | {_fmt_f(agg['mean_bh_sharpe'])} | {_fmt_f(agg['mean_6040_sharpe'])} |",
        f"| Ø MaxDD | {_fmt_pct(agg['mean_max_dd'])} | {_fmt_pct(agg['mean_bh_max_dd'])} | {_fmt_pct(agg['mean_6040_max_dd'])} |",
        f"| Ø Calmar | {_fmt_f(agg['mean_calmar'])} | — | — |",
        f"| Win-Rate (CAGR > 0) | {_fmt_pct(agg['win_rate'])} | — | — |",
        f"| Folds, die SPY CAGR schlagen | {_fmt_pct(agg['beats_spy_cagr'])} | — | — |",
        f"| Folds, die SPY Sharpe schlagen | {_fmt_pct(agg['beats_spy_sharpe'])} | — | — |",
        "",
        "---",
        "",
        "## Drawdown-Analyse (primäres Overlay-Ziel)",
        "",
    ]

    # MaxDD ratio
    mean_dd_overlay = agg["mean_max_dd"]
    mean_dd_spy = agg["mean_bh_max_dd"]
    if (
        not (np.isnan(mean_dd_overlay) or np.isnan(mean_dd_spy))
        and abs(mean_dd_spy) > 1e-6
    ):
        dd_ratio = abs(mean_dd_overlay) / abs(mean_dd_spy)
        dd_improvement_pct = (1.0 - dd_ratio) * 100.0
        lines += [
            f"- Ø MaxDD overlay: **{_fmt_pct(mean_dd_overlay)}**",
            f"- Ø MaxDD SPY: **{_fmt_pct(mean_dd_spy)}**",
            f"- MaxDD-Verhältnis overlay/SPY: **{dd_ratio:.2f}x** ({_fmt_f(dd_improvement_pct, 1)}% Verbesserung)",
            "",
        ]
    else:
        lines += ["- Drawdown-Vergleich: N/A (fehlende Daten)", ""]

    lines += [
        "---",
        "",
        "## Bewertung",
        "",
    ]

    # Honest assessment
    mean_cagr = agg["mean_cagr"]
    mean_sharpe = agg["mean_sharpe"]
    mean_bh_sharpe = agg["mean_bh_sharpe"]
    mean_calmar = agg["mean_calmar"]

    if np.isnan(mean_cagr):
        verdict = (
            "Die OOS-Metriken konnten nicht berechnet werden. "
            "vol_target_overlay ist damit **nicht validiert**."
        )
    else:
        sharpe_criterion = (
            not np.isnan(mean_sharpe)
            and not np.isnan(mean_bh_sharpe)
            and mean_sharpe >= mean_bh_sharpe + 0.2
        )
        # Initialize to nan — branches below may not assign it (F-senior-4: prevents NameError)
        dd_ratio_val = float("nan")
        if (
            not (np.isnan(mean_dd_overlay) or np.isnan(mean_dd_spy))
            and abs(mean_dd_spy) > 1e-6
        ):
            dd_ratio_val = abs(mean_dd_overlay) / abs(mean_dd_spy)
            dd_criterion = dd_ratio_val <= 0.70  # >= 30% besser
        else:
            dd_criterion = False

        if sharpe_criterion and dd_criterion:
            verdict = (
                f"**Beide Kriterien erfüllt.** "
                f"Ø Sharpe {_fmt_f(mean_sharpe)} ≥ SPY {_fmt_f(mean_bh_sharpe)} + 0.2 (Δ={_fmt_f(mean_sharpe - mean_bh_sharpe)}) "
                f"und MaxDD-Reduktion ≥ 30% (Ratio {_fmt_f(dd_ratio_val, 2)}x). "
                f"Ø CAGR {_fmt_pct(mean_cagr)}, Ø Calmar {_fmt_f(mean_calmar)}. "
                "Das Overlay erfüllt das Ziel 'SPY-ähnliche Rendite bei deutlich kleinerem Drawdown'. "
                "Einschränkung: basiert auf Alpaca Free-Tier ohne delisted Symbole (kein Survivorship-Problem "
                "bei 2-Asset-Overlay, aber historische Daten können fehlerhaft sein)."
            )
        elif sharpe_criterion:
            verdict = (
                f"**Sharpe-Kriterium erfüllt, MaxDD-Kriterium nicht erfüllt.** "
                f"Ø Sharpe {_fmt_f(mean_sharpe)} ≥ SPY {_fmt_f(mean_bh_sharpe)} + 0.2, "
                f"aber Ø MaxDD-Reduktion ist mit Ratio {_fmt_f(dd_ratio_val if not dd_criterion else float('nan'), 2)}x < 30%. "
                f"Ø CAGR {_fmt_pct(mean_cagr)}, Ø Calmar {_fmt_f(mean_calmar)}. "
                "Teilerfolg — das Drawdown-Ziel wird nicht vollständig erreicht."
            )
        elif dd_criterion:
            verdict = (
                f"**MaxDD-Kriterium erfüllt, Sharpe-Kriterium nicht erfüllt.** "
                f"MaxDD-Ratio {_fmt_f(dd_ratio_val, 2)}x (≥ 30% Reduktion), "
                f"aber Ø Sharpe {_fmt_f(mean_sharpe)} < SPY Ø {_fmt_f(mean_bh_sharpe)} + 0.2. "
                f"Ø CAGR {_fmt_pct(mean_cagr)}, Ø Calmar {_fmt_f(mean_calmar)}. "
                "Teilerfolg — das Rendite/Risiko-Ziel wird nicht vollständig erreicht."
            )
        else:
            verdict = (
                f"**Keines der Kriterien erfüllt.** "
                f"Ø Sharpe {_fmt_f(mean_sharpe)} (SPY: {_fmt_f(mean_bh_sharpe)}, benötigt +0.2), "
                f"Ø MaxDD {_fmt_pct(mean_dd_overlay)} vs SPY {_fmt_pct(mean_dd_spy)} "
                f"({'< 30%' if not np.isnan(mean_dd_overlay) else 'N/A'} Verbesserung). "
                f"Ø CAGR {_fmt_pct(mean_cagr)}, Ø Calmar {_fmt_f(mean_calmar)}. "
                "Das Overlay erreicht das Ziel 'SPY-ähnliche Rendite bei deutlich kleinerem Drawdown' "
                "**nicht** in diesem OOS-Walk-Forward. "
                "Mögliche Ursachen: Vol-Targeting allein reicht in langen Bullmärkten nicht aus, "
                "um den Drawdown-Penalty durch reduzierte SPY-Exposure zu kompensieren."
            )

    lines.append(verdict)
    lines += [
        "",
        "### Einschränkungen",
        "",
        "- SPY und IEF ohne Dividenden-Reinvestition (Alpaca bar close ≈ Kursrendite, nicht Totalrendite).",
        "  Bei IEF ist die Coupon-Rendite (~2–4% p.a.) nicht enthalten — unterschätzt IEF-Return.",
        "- Kosten: tägliche Rebalanzierung führt zu mehr Turnover als monatliches Rebalancing;",
        "  Kosten sind damit eher konservativ (worst case).",
        "- Keine Transaktionssteuer, kein Spread-Impact über 0.75 bps hinaus.",
        "- Walk-Forward deckt nur Alpaca-Verfügbarkeit; Backfill ab IEF-Inception 2002-07 wäre",
        "  idealer (enthält 2002, 2008, 2020) — hier ab 2003 nach Warmup.",
        "",
        "---",
        "",
        "_Dieses Dokument ist ein automatisch erzeugtes Artefakt aus_ "
        "`scripts/_oos_wf_vol_target_overlay.py`. _Nicht manuell editieren._",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", OUT_MD)


def _write_failure_report(reason: str):
    lines = [
        "# vol_target_overlay — Echter OOS Walk-Forward",
        "",
        "**Status: ABGEBROCHEN**",
        "",
        f"**Grund:** {reason}",
        "",
        "Der Walk-Forward konnte nicht vollständig durchgeführt werden.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.error("Failure report written to %s", OUT_MD)


if __name__ == "__main__":
    sys.exit(main())
