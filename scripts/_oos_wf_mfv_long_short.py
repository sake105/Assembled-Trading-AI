"""One-shot OOS Walk-Forward for multifactor_long_short — writes docs/results/2026_05_multifactor_long_short_real_oos.md.

Usage:
    python scripts/_oos_wf_mfv_long_short.py

Design:
    - Fetches or loads cached Alpaca daily bars.
    - Uses macro_world_etfs_core_bundle.yaml (factor_set=core+vol_liquidity) which
      requires only OHLCV — no external altdata. Factors: trailing momentum 12m,
      trend_strength_200/50, realized_volatility_20, trailing_returns_12m.
    - generate_multifactor_long_short_signals with rebalance_freq="D" generates
      scores for every trading day; signal_fn then down-samples to first trading
      day of each month to match monthly rebalancing intent.
    - Long-only comparison: SHORT signals filtered out. Short side documented as
      not tested.
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
log = logging.getLogger("oos_mfls")
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERIOD_START = pd.Timestamp("2018-01-01", tz="UTC")
PERIOD_END = pd.Timestamp("2025-12-31", tz="UTC")
TRAIN_WINDOW_DAYS = 252
TEST_WINDOW_DAYS = 252
STEP_SIZE_DAYS = 252
WARMUP_BARS = 300  # trailing_momentum_12m needs ~252 bars
COMMISSION_BPS = 10.0
INITIAL_CAPITAL = 100_000.0
WATCHLIST = ROOT / "watchlist.txt"
PRICE_CACHE = ROOT / "output" / "oos_alpaca_prices_cache.parquet"
BUNDLE_PATH = ROOT / "configs" / "factor_bundles" / "macro_world_etfs_core_bundle.yaml"
OUT_MD = ROOT / "docs" / "results" / "2026_05_multifactor_long_short_real_oos.md"


# ---------------------------------------------------------------------------
# 1 — Price loading (cache or Alpaca)
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
    from src.assembled_core.strategies.multifactor_long_short import (
        MultiFactorStrategyConfig,
        compute_multifactor_long_short_positions,
        generate_multifactor_long_short_signals,
    )

    ls_config = MultiFactorStrategyConfig(
        bundle_path=str(BUNDLE_PATH),
        top_quantile=0.2,
        bottom_quantile=0.2,
        rebalance_freq="D",  # Daily to avoid day==1 alignment miss; down-sampled to monthly below
        max_gross_exposure=1.0,
        max_leverage=1.0,
        transaction_cost_bps=0.0,  # costs handled by backtest engine
        use_regime_overlay=False,
    )

    def backtest_fn(
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp,
    ) -> dict:
        # Include WARMUP_BARS so trailing_momentum_12m is initialised
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

        def signal_fn(df: pd.DataFrame) -> pd.DataFrame:
            try:
                sigs = generate_multifactor_long_short_signals(
                    df, factors=None, config=ls_config
                )
            except Exception as exc:
                log.warning("[MF_LS] generate_signals failed: %s", exc)
                return pd.DataFrame(
                    columns=["timestamp", "symbol", "direction", "score"]
                )

            if sigs.empty or "direction" not in sigs.columns:
                return pd.DataFrame(
                    columns=["timestamp", "symbol", "direction", "score"]
                )

            # Long-only comparison: keep only LONG signals in test period
            test_sigs = sigs[
                (sigs["direction"] == "LONG") & (sigs["timestamp"] >= test_start)
            ].copy()

            if test_sigs.empty:
                return test_sigs

            # Down-sample to monthly: keep only first trading day of each test month.
            # Anchor is per-month (not per-symbol) to ensure one calendar date per
            # rebalancing cycle — matching mfv2's single monthly_dates list.
            test_sigs["_month"] = pd.to_datetime(test_sigs["timestamp"]).dt.to_period(
                "M"
            )
            monthly_anchor = test_sigs.groupby(["_month"])["timestamp"].transform("min")
            test_sigs = test_sigs[test_sigs["timestamp"] == monthly_anchor].copy()
            test_sigs = test_sigs.drop(columns=["_month"])

            return test_sigs[["timestamp", "symbol", "direction", "score"]]

        def position_fn(signals: pd.DataFrame, capital: float) -> pd.DataFrame:
            if signals.empty:
                return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
            try:
                pos = compute_multifactor_long_short_positions(
                    signals, capital, ls_config
                )
            except Exception as exc:
                log.debug("[MF_LS] position_fn failed: %s", exc)
                return pd.DataFrame(columns=["symbol", "target_weight", "target_qty"])
            # Notional convention fix (same as trend_baseline)
            if not pos.empty and "target_weight" in pos.columns:
                pos = pos.copy()
                pos["target_qty"] = pos["target_weight"].abs() * capital
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

    if not BUNDLE_PATH.exists():
        _write_failure_report(f"Bundle not found: {BUNDLE_PATH}")
        return 1

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
        "# multifactor_long_short — Echter OOS Walk-Forward (Alpaca, 2026-05)",
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
        f"- Warmup-Buffer: {WARMUP_BARS} Bars (trailing_momentum_12m initialisiert)",
        "- Rebalancierung: Monatlich (erster Handelstag jedes Monats im Testzeitraum)",
        "- **Factor Bundle:** `macro_world_etfs_core_bundle.yaml` (factor_set=core+vol_liquidity)",
        "- **Aktive Faktoren:** trailing_momentum_12m_excl_1m (30%), trend_strength_200 (25%),",
        "  trend_strength_50 (20%), realized_volatility_20 (15%, negativ), trailing_returns_12m (10%)",
        "- **Getestet:** Long-only (TOP 20% quantile). Short-Seite (BOTTOM 20%) nicht in Backtest.",
        f"- Commission: {COMMISSION_BPS} bps",
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
        "| Metrik | multifactor_long_short | SPY Buy-and-Hold |",
        "|--------|----------------------|-----------------|",
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
            "multifactor_long_short ist damit als OOS-validiert **nicht bestätigt**."
        )
    elif beats_spy_pct >= 0.67 and mean_cagr > mean_bh_cagr:
        verdict = (
            f"multifactor_long_short (Long-only) schlägt SPY in {_fmt_pct(beats_spy_pct)} der Folds "
            f"(Ø CAGR {_fmt_pct(mean_cagr)} vs. SPY {_fmt_pct(mean_bh_cagr)}). "
            f"Sharpe Ø {_fmt_f(mean_sharpe)}. Das Ergebnis ist **positiv**. "
            "Einschränkung: Nur Long-Seite getestet (macro_world_etfs_core_bundle, OHLCV-only)."
        )
    elif beats_spy_pct >= 0.5:
        verdict = (
            f"multifactor_long_short (Long-only) schlägt SPY in {_fmt_pct(beats_spy_pct)} der Folds "
            f"(Ø CAGR {_fmt_pct(mean_cagr)} vs. SPY {_fmt_pct(mean_bh_cagr)}). "
            f"Sharpe Ø {_fmt_f(mean_sharpe)}. Das Ergebnis ist **gemischt**. "
            "Das Momentum-Ranking (trailing 12m) liefert keinen robusten Long-Alpha über alle Marktphasen."
        )
    else:
        verdict = (
            f"multifactor_long_short (Long-only) schlägt SPY nur in {_fmt_pct(beats_spy_pct)} der Folds "
            f"(Ø CAGR {_fmt_pct(mean_cagr)} vs. SPY {_fmt_pct(mean_bh_cagr)}). "
            f"Sharpe Ø {_fmt_f(mean_sharpe)}. Das Ergebnis ist **negativ**. "
            "Das Momentum-Ranking des macro_world_etfs_core_bundle liefert im Long-only-Modus "
            "keinen robusten Mehrwert gegenüber SPY. Die Short-Seite wurde nicht getestet; "
            "das Long-Short-Gesamtergebnis kann abweichen."
        )

    lines.append(verdict)
    lines += [
        "",
        "### Einschränkungen dieses Reports",
        "",
        "- **Long-only:** SHORT-Seite (BOTTOM-20%-Quantile) nicht in Backtest einbezogen.",
        "  Ein vollständiger Long-Short-Backtest würde einen dedizierten Short-Selling-fähigen Engine benötigen.",
        "- **Bundle:** macro_world_etfs_core_bundle (OHLCV-only). Andere Bundles (ai_tech, alternative_risk_premia)",
        "  könnten andere Ergebnisse liefern, erfordern aber ggf. Altdata.",
        "- Monatliche Rebalancierung (≠ höhere Frequenz bei aktivem Betrieb).",
        "- Alpaca Free Tier: Survivorship-Bias möglich (delisted Symbole fehlen).",
        "- SPY-Vergleich: kein Dividenden-Reinvest.",
        "",
        "---",
        "",
        "_Dieses Dokument ist ein automatisch erzeugtes Artefakt aus_ "
        "`scripts/_oos_wf_mfv_long_short.py`. _Nicht manuell editieren._",
    ]

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    log.info("Report written to %s", OUT_MD)


def _write_failure_report(reason: str):
    lines = [
        "# multifactor_long_short — Echter OOS Walk-Forward",
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
