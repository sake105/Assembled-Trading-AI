"""One-shot OOS Walk-Forward for trend_baseline — writes docs/results/2026_05_trend_baseline_real_oos.md.

Usage:
    python scripts/_oos_wf_trend_baseline.py

Requirements:
    ALPACA_API_KEY / ALPACA_API_SECRET in .env or environment.
    pip install alpaca-py (already installed in this project).

Design:
    - Fetches Alpaca daily bars for the full PaperPilot watchlist (watchlist.txt).
    - Runs run_walk_forward_backtest with 1-year rolling train / 1-year test windows.
    - Uses 10 bps commission (matching April-2026 backtest report).
    - Adds MA warmup buffer (90 bars pre-test) to each fold so MA values are
      not NaN at the start of the test period.
    - Runs a SPY buy-and-hold baseline with identical cost assumptions.
    - Writes markdown report to docs/results/2026_05_trend_baseline_real_oos.md.

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
log = logging.getLogger("oos_wf")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW_DAYS = 252  # 1 calendar year of trading days
TEST_WINDOW_DAYS = 252  # 1 calendar year of trading days
STEP_SIZE_DAYS = 252  # roll forward annually
MA_WARMUP_BARS = 90  # prepend to test prices so MA is initialised
MA_FAST = 20
MA_SLOW = 60
COMMISSION_BPS = 10.0  # matches April-2026 report
INITIAL_CAPITAL = 100_000.0
WATCHLIST = ROOT / "watchlist.txt"
OUT_MD = ROOT / "docs" / "results" / "2026_05_trend_baseline_real_oos.md"


# ---------------------------------------------------------------------------
# 1 — Load Alpaca credentials
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

    log.info(
        "Fetching %d symbols from Alpaca (%s → %s)…",
        len(symbols),
        start.date(),
        end.date(),
    )
    req = StockBarsRequest(
        symbol_or_symbols=symbols,
        timeframe=TimeFrame.Day,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
        adjustment="split",  # corporate-action adjusted
    )
    bars = client.get_stock_bars(req)
    df = bars.df.reset_index()

    # Normalise to project schema: timestamp (UTC tz-aware), symbol, open/high/low/close/volume
    if "timestamp" not in df.columns:
        df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.rename(columns=str.lower)
    # Keep only needed columns (Alpaca returns trade_count, vwap too)
    keep = [
        c
        for c in ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
        if c in df.columns
    ]
    df = df[keep].copy()
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    log.info("Fetched %d rows, %d symbols", len(df), df["symbol"].nunique())
    return df


# ---------------------------------------------------------------------------
# 3 — Build walk-forward backtest function with MA warmup
# ---------------------------------------------------------------------------
def _make_backtest_fn(prices: pd.DataFrame) -> object:
    """Return a backtest_fn compatible with run_walk_forward_backtest.

    Adds MA_WARMUP_BARS of pre-test data so rolling MA is initialised at
    test_start — the standard make_engine_backtest_fn does not do this.
    """
    from src.assembled_core.qa.backtest_engine import run_portfolio_backtest
    from src.assembled_core.signals.rules_trend import (
        generate_trend_signals_from_prices,
    )
    from src.assembled_core.strategies.trend_baseline import (
        compute_target_positions as tb_compute_targets,
    )

    def backtest_fn(
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> dict:
        # Include warmup bars before test_start so MA is hot at the first test bar.
        # np.sort ensures deterministic order before index arithmetic.
        warmup_dates = np.sort(prices["timestamp"].unique())
        warmup_dates = warmup_dates[warmup_dates < test_start]
        warmup_start = (
            warmup_dates[-MA_WARMUP_BARS]
            if len(warmup_dates) >= MA_WARMUP_BARS
            else test_start
        )

        window_prices = prices[
            (prices["timestamp"] >= warmup_start) & (prices["timestamp"] < test_end)
        ].copy()

        if window_prices.empty or window_prices["symbol"].nunique() < 3:
            raise ValueError(
                f"Insufficient price data for test period {test_start.date()}–{test_end.date()}"
            )

        # signal_fn: generate full time-series signals (not compute_signals which
        # returns only tail(1) per symbol — designed for daily PaperPilot calls).
        # Warmup bars initialise the rolling MA; filter to test period for trading.
        def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            sigs = generate_trend_signals_from_prices(
                df, ma_fast=MA_FAST, ma_slow=MA_SLOW
            )
            return sigs[
                (sigs["direction"] == "LONG") & (sigs["timestamp"] >= test_start)
            ]

        def position_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
            pos = tb_compute_targets(signals, capital)
            # tb_compute_targets returns target_qty=0.0 (live-pipeline convention).
            # generate_orders_from_targets treats target_qty as notional dollars,
            # divides by price to get shares — so set notional = weight × capital.
            if not pos.empty and "target_weight" in pos.columns:
                pos = pos.copy()
                pos["target_qty"] = pos["target_weight"] * capital
            return pos

        result = run_portfolio_backtest(
            prices=window_prices,
            signal_fn=signal_fn,
            position_sizing_fn=position_fn,
            start_capital=INITIAL_CAPITAL,
            commission_bps=COMMISSION_BPS,
            spread_w=0.25,
            impact_w=0.5,
            include_costs=True,
            rebalance_freq="1d",
        )
        m = result.metrics

        # result.metrics only exposes {final_pf, sharpe, trades}.
        # Compute cagr and max_dd directly from the equity curve (test period only).
        equity = result.equity
        eq_test = (
            equity[equity["timestamp"] >= test_start].copy()
            if not equity.empty and "timestamp" in equity.columns
            else equity.copy()
        )
        if len(eq_test) >= 2 and "equity" in eq_test.columns:
            start_eq = eq_test["equity"].iloc[0]
            end_eq = eq_test["equity"].iloc[-1]
            total_ret = end_eq / start_eq - 1 if start_eq > 0 else float("nan")
            n_years = len(eq_test) / 252
            cagr = (
                (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
                if not np.isnan(total_ret)
                else float("nan")
            )
            # Recompute daily_return on the test-period slice only — the pre-computed
            # column from the full equity series has a stale first-row value (transition
            # from the last warmup bar), which would contaminate MaxDD/Sharpe.
            eq_test["daily_return"] = eq_test["equity"].pct_change().fillna(0.0)
            daily_rets = eq_test["daily_return"]
            if len(daily_rets) >= 2:
                peak = eq_test["equity"].cummax()
                max_dd = float(((eq_test["equity"] - peak) / (peak + 1e-10)).min())
            else:
                max_dd = float("nan")
        else:
            cagr = float("nan")
            max_dd = float("nan")

        return {
            "test_sharpe": m.get("sharpe", float("nan")),
            "test_cagr": cagr,
            "test_max_dd": max_dd,
            "test_trades": int(m.get("trades", 0)),
        }

    return backtest_fn


# ---------------------------------------------------------------------------
# 4 — SPY buy-and-hold baseline
# ---------------------------------------------------------------------------
def _spy_buyhold(
    spy_prices: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> dict:
    """Return CAGR, Sharpe, MaxDD for buy-and-hold SPY in the test period."""
    df = spy_prices[
        (spy_prices["timestamp"] >= test_start) & (spy_prices["timestamp"] < test_end)
    ].sort_values("timestamp")
    if len(df) < 5:
        return {
            "bh_cagr": float("nan"),
            "bh_sharpe": float("nan"),
            "bh_max_dd": float("nan"),
        }
    rets = df["close"].pct_change().dropna()
    n_years = len(df) / 252
    total_ret = df["close"].iloc[-1] / df["close"].iloc[0] - 1
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe = (rets.mean() / (rets.std() + 1e-10)) * np.sqrt(252)
    cum = (1 + rets).cumprod()
    peak = cum.cummax()
    max_dd = float(((cum - peak) / peak).min())
    return {"bh_cagr": cagr, "bh_sharpe": sharpe, "bh_max_dd": max_dd}


# ---------------------------------------------------------------------------
# 5 — Main
# ---------------------------------------------------------------------------
def main():
    # ── 5.1 Universe ──────────────────────────────────────────────────────
    all_symbols = [
        s.strip()
        for s in WATCHLIST.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.strip().startswith("#") and "." not in s.strip()
    ]
    log.info("Watchlist: %d symbols", len(all_symbols))

    # ── 5.2 Fetch data ────────────────────────────────────────────────────
    # Extend start by MA_WARMUP_BARS extra calendar days (approx 130 cal days)
    fetch_start = PERIOD_START - pd.Timedelta(days=130)
    try:
        prices = _fetch_alpaca(all_symbols + ["SPY"], start=fetch_start, end=PERIOD_END)
    except Exception as exc:
        log.error("Alpaca fetch failed: %s — aborting", exc)
        _write_failure_report(str(exc))
        return 1

    actual_symbols = prices["symbol"].unique().tolist()
    tradeable = [s for s in actual_symbols if s != "SPY"]
    log.info("Tradeable symbols with Alpaca data: %d", len(tradeable))

    # Earliest & latest bars
    actual_start = prices["timestamp"].min()
    actual_end = prices["timestamp"].max()
    log.info("Actual data range: %s → %s", actual_start.date(), actual_end.date())

    spy_prices = prices[prices["symbol"] == "SPY"].copy()
    strategy_prices = prices[prices["symbol"] != "SPY"].copy()

    # ── 5.3 Walk-forward ──────────────────────────────────────────────────
    from src.assembled_core.qa.walk_forward import (
        WalkForwardConfig,
        run_walk_forward_backtest,
    )

    # Clip to actual data range
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

    backtest_fn = _make_backtest_fn(strategy_prices)

    log.info("Running walk-forward…")
    try:
        wf_result = run_walk_forward_backtest(config=config, backtest_fn=backtest_fn)
    except Exception as exc:
        log.error("Walk-forward failed: %s", exc, exc_info=True)
        _write_failure_report(str(exc))
        return 1

    # ── 5.4 Collect per-fold results with SPY comparison ──────────────────
    # Metrics live in summary_df (backtest_result is always None in this API)
    summary = (
        wf_result.summary_df.set_index("split_index")
        if not wf_result.summary_df.empty
        else pd.DataFrame()
    )

    fold_rows = []
    for wr in wf_result.window_results:
        w = wr.window
        spy_bh = _spy_buyhold(spy_prices, w.test_start, w.test_end)
        if wr.status == "failed":
            fold_rows.append(
                {
                    "fold": w.split_index + 1,
                    "train_start": w.train_start.date(),
                    "train_end": w.train_end.date(),
                    "test_start": w.test_start.date(),
                    "test_end": w.test_end.date(),
                    "cagr": float("nan"),
                    "sharpe": float("nan"),
                    "max_dd": float("nan"),
                    "bh_cagr": spy_bh["bh_cagr"],
                    "bh_sharpe": spy_bh["bh_sharpe"],
                    "bh_max_dd": spy_bh["bh_max_dd"],
                    "status": "FAILED",
                    "error": wr.error_message,
                }
            )
        else:
            row = (
                summary.loc[w.split_index]
                if w.split_index in summary.index
                else pd.Series()
            )
            fold_rows.append(
                {
                    "fold": w.split_index + 1,
                    "train_start": w.train_start.date(),
                    "train_end": w.train_end.date(),
                    "test_start": w.test_start.date(),
                    "test_end": w.test_end.date(),
                    "cagr": float(row.get("test_cagr", float("nan"))),
                    "sharpe": float(row.get("test_sharpe", float("nan"))),
                    "max_dd": float(row.get("test_max_dd", float("nan"))),
                    "bh_cagr": spy_bh["bh_cagr"],
                    "bh_sharpe": spy_bh["bh_sharpe"],
                    "bh_max_dd": spy_bh["bh_max_dd"],
                    "status": "OK",
                    "error": None,
                }
            )

    fold_df = pd.DataFrame(fold_rows)

    # ── 5.5 Aggregate OOS metrics ─────────────────────────────────────────
    ok = fold_df[fold_df["status"] == "OK"]
    n_ok = len(ok)
    n_total = len(fold_df)

    if n_ok == 0:
        log.error("All folds failed — no aggregated metrics available")
        _write_failure_report("All folds failed")
        return 1

    agg = {
        "mean_cagr": ok["cagr"].mean(),
        "mean_sharpe": ok["sharpe"].mean(),
        "mean_max_dd": ok["max_dd"].mean(),
        "win_rate": (ok["cagr"] > 0).mean(),
        "beats_spy": (ok["cagr"] > ok["bh_cagr"]).mean(),
        "mean_bh_cagr": ok["bh_cagr"].mean(),
        "mean_bh_sharpe": ok["bh_sharpe"].mean(),
    }

    # ── 5.6 Write markdown report ─────────────────────────────────────────
    _write_report(
        fold_df=fold_df,
        agg=agg,
        n_ok=n_ok,
        n_total=n_total,
        actual_start=actual_start,
        actual_end=actual_end,
        n_symbols=len(tradeable),
        all_symbols_requested=len(all_symbols),
    )
    return 0


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------
def _fmt_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v * 100:.1f}%"


def _fmt_f(v, d=2) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    return f"{v:.{d}f}"


def _write_report(
    *,
    fold_df,
    agg,
    n_ok,
    n_total,
    actual_start,
    actual_end,
    n_symbols,
    all_symbols_requested,
):
    lines = []
    lines += [
        "# trend_baseline — Echter OOS Walk-Forward (Alpaca, 2026-05)",
        "",
        "**Erstellt:** 2026-05-27  ",
        "**Branch:** main @ a7e01689  ",
        "**Zweck:** Artefakt 2 aus GO_LIVE_CHECKLIST A3/B1 — echter OOS-Nachweis auf realen Kursdaten.",
        "",
        "---",
        "",
        "## Datenquelle",
        "",
        "- **Anbieter:** Alpaca Markets (Free Tier) — `StockHistoricalDataClient`, split-adjustiert",
        f"- **Angefordertes Universum:** {all_symbols_requested} Symbole (watchlist.txt, US-only, ohne '.')",
        f"- **Symbole mit Alpaca-Daten:** {n_symbols}",
        f"- **Tatsächliche Zeitspanne:** {actual_start.date()} → {actual_end.date()}",
        f"  (Anfrage: {PERIOD_START.date()} → {PERIOD_END.date()}; "
        f"Alpaca liefert je nach Symbol ab ~2015–2016)",
        "- **SPY:** Als Buy-and-Hold-Benchmark, gleicher Anbieter",
        "",
        "## Walk-Forward-Konfiguration",
        "",
        "- Modus: Rolling",
        f"- Train-Fenster: {TRAIN_WINDOW_DAYS} Handelstage (~1 Jahr)",
        f"- Test-Fenster: {TEST_WINDOW_DAYS} Handelstage (~1 Jahr)",
        f"- Schrittweite: {STEP_SIZE_DAYS} Handelstage (jährliche Verschiebung)",
        f"- MA-Warmup-Buffer: {MA_WARMUP_BARS} Bars vor Testbeginn (MA initialisiert)",
        f"- ma_fast={MA_FAST}, ma_slow={MA_SLOW} (wie PaperPilot paper_runner.py)",
        f"- Commission: {COMMISSION_BPS} bps (wie April-2026-Report)",
        "- Spread-Weight: 0.25, Impact-Weight: 0.5",
        f"- Startkapital: {INITIAL_CAPITAL:,.0f} USD",
        "",
        "---",
        "",
        "## Ergebnisse pro Fold",
        "",
        "| Fold | Train | Test | CAGR | Sharpe | MaxDD | SPY-CAGR | SPY-Sharpe | Schlägt SPY? |",
        "|------|-------|------|------|--------|-------|----------|------------|-------------|",
    ]

    for _, row in fold_df.iterrows():
        beats = ""
        if (
            row["status"] == "OK"
            and not np.isnan(row["cagr"])
            and not np.isnan(row["bh_cagr"])
        ):
            beats = "Ja" if row["cagr"] > row["bh_cagr"] else "Nein"
        lines.append(
            f"| {int(row['fold'])} "
            f"| {row['train_start']}–{row['train_end']} "
            f"| {row['test_start']}–{row['test_end']} "
            f"| {_fmt_pct(row['cagr'])} "
            f"| {_fmt_f(row['sharpe'])} "
            f"| {_fmt_pct(row['max_dd'])} "
            f"| {_fmt_pct(row['bh_cagr'])} "
            f"| {_fmt_f(row['bh_sharpe'])} "
            f"| {beats} |"
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
        "| Metrik | trend_baseline | SPY Buy-and-Hold |",
        "|--------|---------------|-----------------|",
        f"| Ø CAGR | {_fmt_pct(agg['mean_cagr'])} | {_fmt_pct(agg['mean_bh_cagr'])} |",
        f"| Ø Sharpe | {_fmt_f(agg['mean_sharpe'])} | {_fmt_f(agg['mean_bh_sharpe'])} |",
        f"| Ø MaxDD | {_fmt_pct(agg['mean_max_dd'])} | — |",
        f"| Win-Rate (CAGR > 0) | {_fmt_pct(agg['win_rate'])} | — |",
        f"| Folds, die SPY schlagen | {_fmt_pct(agg['beats_spy'])} | — |",
        "",
        "---",
        "",
        "## Bewertung",
        "",
    ]

    # Honest assessment
    mean_cagr = agg["mean_cagr"]
    mean_bh_cagr = agg["mean_bh_cagr"]
    mean_sharpe = agg["mean_sharpe"]
    beats_spy_pct = agg["beats_spy"]

    if np.isnan(mean_cagr):
        verdict = (
            "Die OOS-Metriken konnten nicht berechnet werden — alle Folds sind fehlgeschlagen. "
            "trend_baseline ist damit als OOS-validiert **nicht bestätigt**."
        )
    elif beats_spy_pct >= 0.67 and mean_cagr > mean_bh_cagr:
        verdict = (
            f"trend_baseline schlägt SPY Buy-and-Hold in {_fmt_pct(beats_spy_pct)} der Folds "
            f"(Ø CAGR {_fmt_pct(mean_cagr)} vs. SPY {_fmt_pct(mean_bh_cagr)}). "
            f"Sharpe Ø {_fmt_f(mean_sharpe)}. "
            "Das Ergebnis ist **positiv**, aber basiert auf einer begrenzten Foldanzahl und "
            "dem Alpaca Free-Tier-Feed. Überanpassung an den Bullmarkt 2019–2021 ist nicht ausgeschlossen. "
            "Ein stressgetesteter Walk-Forward auf einem längeren Zeitraum (inkl. 2008) bleibt offen."
        )
    elif beats_spy_pct >= 0.5:
        verdict = (
            f"trend_baseline schlägt SPY in {_fmt_pct(beats_spy_pct)} der Folds "
            f"(Ø CAGR {_fmt_pct(mean_cagr)} vs. SPY {_fmt_pct(mean_bh_cagr)}). "
            f"Sharpe Ø {_fmt_f(mean_sharpe)}. "
            "Das Ergebnis ist **gemischt** — in etwa der Hälfte der Perioden wird die Benchmark nicht übertroffen. "
            "Ein einfacher MA-Crossover liefert keinen robusten Alpha-Nachweis gegen SPY über alle Marktphasen."
        )
    else:
        verdict = (
            f"trend_baseline schlägt SPY nur in {_fmt_pct(beats_spy_pct)} der Folds "
            f"(Ø CAGR {_fmt_pct(mean_cagr)} vs. SPY {_fmt_pct(mean_bh_cagr)}). "
            f"Sharpe Ø {_fmt_f(mean_sharpe)}. "
            "Das Ergebnis ist **negativ** — trend_baseline liefert im OOS-Vergleich keinen robusten Alpha "
            "gegenüber einer passiven SPY-Position. Die Strategie profitiert vom Bullmarkt-Bias "
            "in bestimmten Perioden, versagt aber in Seitwärts- oder Bear-Phasen. "
            "Der PaperPilot-Betrieb sollte dies als Risikofaktor werten."
        )

    lines.append(verdict)
    lines += [
        "",
        "### Einschränkungen dieses Reports",
        "",
        "- Alpaca Free Tier: keine adjustierten Daten für delisted Symbole → Survivorship-Bias möglich.",
        "- walk_forward.py `make_engine_backtest_fn` wurde durch eine custom backtest_fn ersetzt, "
        "  die MA-Warmup-Bars vor dem Testzeitraum prepended.",
        "- Die Transaktionskosten (10 bps) entsprechen dem April-2026-Report, aber kein "
        "  marktimpact-adjustierter Kostensatz.",
        "- SPY-Vergleich: kein Dividenden-Reinvest im SPY-Buy-and-Hold (Alpaca bar close ≠ total return).",
        "",
        "---",
        "",
        "_Dieses Dokument ist ein automatisch erzeugtes Artefakt aus_ "
        "`scripts/_oos_wf_trend_baseline.py`. _Nicht manuell editieren._",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", OUT_MD)


def _write_failure_report(reason: str):
    lines = [
        "# trend_baseline — Echter OOS Walk-Forward (Artefakt 2)",
        "",
        "**Status: ABGEBROCHEN**",
        "",
        f"**Grund:** {reason}",
        "",
        "Der Walk-Forward konnte nicht vollständig durchgeführt werden.",
        "Keine OOS-Metriken verfügbar.",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.error("Failure report written to %s", OUT_MD)


if __name__ == "__main__":
    sys.exit(main())
