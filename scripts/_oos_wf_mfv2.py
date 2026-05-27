"""One-shot OOS Walk-Forward for multifactor_v2 — writes docs/results/2026_05_multifactor_v2_real_oos.md.

Usage:
    python scripts/_oos_wf_mfv2.py

Design:
    - Fetches Alpaca daily bars for the full PaperPilot watchlist (or loads cache).
    - Monthly rebalancing: signal_fn calls mfv2.compute_signals once per month-start,
      building a full time-series of signals. This is necessary because compute_signals
      returns tail(1) per symbol (designed for live daily calls, not backtest replay).
    - TA features (EMA spread, OBV, breadth, RSI, Bollinger, ADX, MACD) computed from
      raw OHLCV via ta_features.add_all_features. All altdata factors (earnings, insider,
      news, macro, GPR, VIX, options, congress) degrade gracefully to 0.0 — documented
      as a known limitation.
    - Same cost assumptions as trend_baseline: 10 bps, spread_w=0.25, impact_w=0.5.

KEINE Änderungen an strategy, policy.yaml oder anderen Produktionsdateien.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s"
)
log = logging.getLogger("oos_mfv2")
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW_DAYS = 252
TEST_WINDOW_DAYS = 252
STEP_SIZE_DAYS = 252
WARMUP_BARS = 250  # enough for MA-200 to warm up
COMMISSION_BPS = 10.0
INITIAL_CAPITAL = 100_000.0
WATCHLIST = ROOT / "watchlist.txt"
PRICE_CACHE = ROOT / "output" / "oos_alpaca_prices_cache.parquet"
OUT_MD = ROOT / "docs" / "results" / "2026_05_multifactor_v2_real_oos.md"


# ---------------------------------------------------------------------------
# 1 — Price loading (Alpaca or cache)
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
    log.info("Fetched %d rows, %d symbols", len(df), df["symbol"].nunique())
    return df


def _get_prices(all_symbols: list[str]) -> pd.DataFrame:
    fetch_start = PERIOD_START - pd.Timedelta(days=400)
    if PRICE_CACHE.exists():
        log.info("Loading cached prices from %s", PRICE_CACHE)
        prices = pd.read_parquet(PRICE_CACHE)
        prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
        # Check all symbols present
        missing = [s for s in all_symbols if s not in prices["symbol"].unique()]
        if not missing:
            return prices
        log.info("Cache missing %d symbols, re-fetching", len(missing))
    prices = _fetch_alpaca(all_symbols, start=fetch_start, end=PERIOD_END)
    PRICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(PRICE_CACHE, index=False)
    log.info("Prices cached to %s", PRICE_CACHE)
    return prices


# ---------------------------------------------------------------------------
# 2 — Build walk-forward backtest function
# ---------------------------------------------------------------------------
def _make_backtest_fn(prices: pd.DataFrame) -> object:
    from src.assembled_core.features.ta_features import add_all_features
    from src.assembled_core.strategies.multifactor_v2 import (
        compute_signals as mfv2_compute_signals,
    )
    from src.assembled_core.strategies.multifactor_v2 import (
        compute_target_positions as mfv2_compute_targets,
    )

    def backtest_fn(
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> dict:
        # Include WARMUP_BARS before test_start so MA-200 is initialised
        warmup_dates = np.sort(prices["timestamp"].unique())
        warmup_dates = warmup_dates[warmup_dates < test_start]
        warmup_start = (
            warmup_dates[-WARMUP_BARS]
            if len(warmup_dates) >= WARMUP_BARS
            else test_start
        )

        window_prices = prices[
            (prices["timestamp"] >= warmup_start) & (prices["timestamp"] < test_end)
        ].copy()

        if window_prices.empty or window_prices["symbol"].nunique() < 5:
            raise ValueError(
                f"Insufficient price data for {test_start.date()}–{test_end.date()}"
            )

        # Pre-compute TA features once on the full window
        enriched = add_all_features(window_prices, use_namespace=True)

        # Monthly rebalancing dates in test period (first trading day of each month)
        test_ts_sorted = sorted(
            enriched[enriched["timestamp"] >= test_start]["timestamp"].unique()
        )
        monthly_dates: list = []
        last_month = None
        for ts in test_ts_sorted:
            m = pd.Timestamp(ts).to_period("M")
            if m != last_month:
                monthly_dates.append(ts)
                last_month = m

        def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            # signal_fn receives all window prices; build per-month signals from enriched
            all_sigs = []
            for rebal_ts in monthly_dates:
                price_slice = enriched[enriched["timestamp"] <= rebal_ts].copy()
                if price_slice["symbol"].nunique() < 3:
                    continue
                try:
                    sigs = mfv2_compute_signals(price_slice, strategy_cfg={})
                    if sigs.empty or "direction" not in sigs.columns:
                        continue
                    long_sigs = sigs[sigs["direction"] == "LONG"][
                        ["timestamp", "symbol", "direction", "score"]
                    ].copy()
                    long_sigs["timestamp"] = rebal_ts
                    all_sigs.append(long_sigs)
                except Exception as exc:
                    log.warning(
                        "[MFV2] compute_signals skip %s: %s",
                        pd.Timestamp(rebal_ts).date(),
                        exc,
                    )
            if not all_sigs:
                return pd.DataFrame(
                    columns=["timestamp", "symbol", "direction", "score"]
                )
            return pd.concat(all_sigs, ignore_index=True)

        def position_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
            if signals.empty:
                return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
            pos = mfv2_compute_targets(signals, capital)
            # Same notional-convention fix as trend_baseline
            if not pos.empty and "target_weight" in pos.columns:
                pos = pos.copy()
                pos["target_qty"] = pos["target_weight"] * capital
            return pos

        from src.assembled_core.qa.backtest_engine import run_portfolio_backtest

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
            eq_test = eq_test.copy()
            eq_test["daily_return"] = eq_test["equity"].pct_change().fillna(0.0)
            peak = eq_test["equity"].cummax()
            max_dd = float(((eq_test["equity"] - peak) / (peak + 1e-10)).min())
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
# 3 — SPY buy-and-hold baseline
# ---------------------------------------------------------------------------
def _spy_buyhold(
    spy_prices: pd.DataFrame, test_start: pd.Timestamp, test_end: pd.Timestamp
) -> dict:
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
# 4 — Main
# ---------------------------------------------------------------------------
def main():
    all_symbols = [
        s.strip()
        for s in WATCHLIST.read_text(encoding="utf-8").splitlines()
        if s.strip() and not s.strip().startswith("#") and "." not in s.strip()
    ]
    log.info("Watchlist: %d symbols", len(all_symbols))

    try:
        prices = _get_prices(all_symbols + ["SPY"])
    except Exception as exc:
        log.error("Price fetch failed: %s", exc)
        _write_failure_report(str(exc))
        return 1

    actual_symbols = prices["symbol"].unique().tolist()
    tradeable = [s for s in actual_symbols if s != "SPY"]
    actual_start = prices["timestamp"].min()
    actual_end = prices["timestamp"].max()
    log.info(
        "Data range: %s → %s, tradeable: %d",
        actual_start.date(),
        actual_end.date(),
        len(tradeable),
    )

    spy_prices = prices[prices["symbol"] == "SPY"].copy()
    strategy_prices = prices[prices["symbol"] != "SPY"].copy()

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

    backtest_fn = _make_backtest_fn(strategy_prices)

    log.info("Running walk-forward…")
    try:
        wf_result = run_walk_forward_backtest(config=config, backtest_fn=backtest_fn)
    except Exception as exc:
        log.error("Walk-forward failed: %s", exc, exc_info=True)
        _write_failure_report(str(exc))
        return 1

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
                    **spy_bh,
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
                    **spy_bh,
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
        "win_rate": (ok["cagr"] > 0).mean(),
        "beats_spy": (ok["cagr"] > ok["bh_cagr"]).mean(),
        "mean_bh_cagr": ok["bh_cagr"].mean(),
        "mean_bh_sharpe": ok["bh_sharpe"].mean(),
    }

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
# Report helpers
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
    lines = [
        "# multifactor_v2 — Echter OOS Walk-Forward (Alpaca, 2026-05)",
        "",
        "**Erstellt:** 2026-05-27  ",
        "**Branch:** main @ cc563605  ",
        "**Zweck:** GO_LIVE_CHECKLIST Paket 3b — echter OOS-Nachweis auf realen Kursdaten.",
        "",
        "---",
        "",
        "## Datenquelle",
        "",
        "- **Anbieter:** Alpaca Markets (Free Tier) — `StockHistoricalDataClient`, split-adjustiert",
        f"- **Angefordertes Universum:** {all_symbols_requested} Symbole (watchlist.txt, US-only, ohne '.')",
        f"- **Symbole mit Alpaca-Daten:** {n_symbols}",
        f"- **Tatsächliche Zeitspanne:** {actual_start.date()} → {actual_end.date()}",
        f"  (Anfrage: {PERIOD_START.date()} → {PERIOD_END.date()})",
        "- **SPY:** Als Buy-and-Hold-Benchmark, gleicher Anbieter",
        "",
        "## Walk-Forward-Konfiguration",
        "",
        "- Modus: Rolling",
        f"- Train-Fenster: {TRAIN_WINDOW_DAYS} Handelstage (~1 Jahr)",
        f"- Test-Fenster: {TEST_WINDOW_DAYS} Handelstage (~1 Jahr)",
        f"- Schrittweite: {STEP_SIZE_DAYS} Handelstage (jährliche Verschiebung)",
        f"- Warmup-Buffer: {WARMUP_BARS} Bars (MA-200 initialisiert)",
        "- Rebalancierung: Monatlich (erster Handelstag jedes Monats im Testzeitraum)",
        f"- Commission: {COMMISSION_BPS} bps",
        "- Spread-Weight: 0.25, Impact-Weight: 0.5",
        f"- Startkapital: {INITIAL_CAPITAL:,.0f} USD",
        "",
        "**Faktor-Verfügbarkeit in diesem Test:**",
        "- Aktiv (aus OHLCV berechnet): EMA-Spread, MA200-Position, RSI, OBV-Trend, Breadth,",
        "  Bollinger %B, Stochastic, ADX, MACD-Histogramm, Volatilitäts-Regime (soweit TA-Spalten passen)",
        "- Degradiert auf 0.0 (kein Altdata): Earnings-Surprise, Insider, News-Sentiment,",
        "  Makro-Faktoren, Intermarkt, VIX/Put-Call-Options, Congress, GPR, Buyback",
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
            f"| {int(row['fold'])} | {row['train_start']}–{row['train_end']} "
            f"| {row['test_start']}–{row['test_end']} "
            f"| {_fmt_pct(row['cagr'])} | {_fmt_f(row['sharpe'])} | {_fmt_pct(row['max_dd'])} "
            f"| {_fmt_pct(row['bh_cagr'])} | {_fmt_f(row['bh_sharpe'])} | {beats} |"
        )
        if row["status"] == "FAILED":
            lines.append(f"> Fold {int(row['fold'])} FAILED: {row['error']}")

    mean_cagr = agg["mean_cagr"]
    mean_bh_cagr = agg["mean_bh_cagr"]
    mean_sharpe = agg["mean_sharpe"]
    beats_spy_pct = agg["beats_spy"]

    lines += [
        "",
        f"_Erfolgreiche Folds: {n_ok}/{n_total}_",
        "",
        "---",
        "",
        "## Aggregierte OOS-Metriken",
        "",
        "| Metrik | multifactor_v2 | SPY Buy-and-Hold |",
        "|--------|---------------|-----------------|",
        f"| Ø CAGR | {_fmt_pct(mean_cagr)} | {_fmt_pct(agg['mean_bh_cagr'])} |",
        f"| Ø Sharpe | {_fmt_f(mean_sharpe)} | {_fmt_f(agg['mean_bh_sharpe'])} |",
        f"| Ø MaxDD | {_fmt_pct(agg['mean_max_dd'])} | — |",
        f"| Win-Rate (CAGR > 0) | {_fmt_pct(agg['win_rate'])} | — |",
        f"| Folds, die SPY schlagen | {_fmt_pct(beats_spy_pct)} | — |",
        "",
        "---",
        "",
        "## Bewertung",
        "",
    ]

    if np.isnan(mean_cagr):
        verdict = (
            "Die OOS-Metriken konnten nicht berechnet werden — alle Folds sind fehlgeschlagen. "
            "multifactor_v2 ist damit als OOS-validiert **nicht bestätigt**."
        )
    elif beats_spy_pct >= 0.67 and mean_cagr > mean_bh_cagr:
        verdict = (
            f"multifactor_v2 schlägt SPY in {_fmt_pct(beats_spy_pct)} der Folds "
            f"(Ø CAGR {_fmt_pct(mean_cagr)} vs. SPY {_fmt_pct(mean_bh_cagr)}). "
            f"Sharpe Ø {_fmt_f(mean_sharpe)}. Das Ergebnis ist **positiv**. Einschränkung: "
            "Dieser Test ist ein degradierter TA-only-Test (kein Altdata). Das echte mfv2-Verhalten "
            "im Produktionsbetrieb kann abweichen."
        )
    elif beats_spy_pct >= 0.5:
        verdict = (
            f"multifactor_v2 schlägt SPY in {_fmt_pct(beats_spy_pct)} der Folds "
            f"(Ø CAGR {_fmt_pct(mean_cagr)} vs. SPY {_fmt_pct(mean_bh_cagr)}). "
            f"Sharpe Ø {_fmt_f(mean_sharpe)}. Das Ergebnis ist **gemischt**. "
            "Wichtige Einschränkung: 19 von 34 Faktoren degradierten auf 0.0 (kein Altdata), "
            "sodass dieser Test nur den TA-Subset von mfv2 misst."
        )
    else:
        verdict = (
            f"multifactor_v2 schlägt SPY nur in {_fmt_pct(beats_spy_pct)} der Folds "
            f"(Ø CAGR {_fmt_pct(mean_cagr)} vs. SPY {_fmt_pct(mean_bh_cagr)}). "
            f"Sharpe Ø {_fmt_f(mean_sharpe)}. Das Ergebnis ist **negativ**. "
            "Einschränkung: Dieser Test misst nur den TA-Subset (15 von 34 Faktoren aktiv); "
            "19 Altdata-Faktoren (News, Macro, GPR, VIX, Options, Earnings, Insider) degradierten "
            "auf 0.0, weil Alpaca keine Fundamentaldaten liefert. Das volle mfv2 mit "
            "Altdata kann signifikant abweichen — dieser Test ist kein Beweis für oder gegen mfv2."
        )

    lines.append(verdict)
    lines += [
        "",
        "### Einschränkungen dieses Reports",
        "",
        "- **Haupteinschränkung:** 19/34 Faktoren = 0.0 (kein Altdata aus Alpaca). Dieser Test "
        "  misst nur den TA-Subset (EMA-Spread, OBV, RSI, Bollinger, ADX, MACD, Breadth).",
        "- Monatliche Rebalancierung (≠ tägliche Rebalancierung im PaperPilot).",
        "- compute_signals gibt tail(1) zurück → muss pro Monats-Rebalancing-Datum separat aufgerufen werden.",
        "- Alpaca Free Tier: Survivorship-Bias möglich (delisted Symbole fehlen).",
        "- SPY-Vergleich: kein Dividenden-Reinvest.",
        "",
        "---",
        "",
        "_Dieses Dokument ist ein automatisch erzeugtes Artefakt aus_ "
        "`scripts/_oos_wf_mfv2.py`. _Nicht manuell editieren._",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", OUT_MD)


def _write_failure_report(reason: str):
    lines = [
        "# multifactor_v2 — Echter OOS Walk-Forward",
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
