#!/usr/bin/env python
"""Echter yfinance-Backtest auf einem SP500-Subset.

Pipeline
--------
1. Lade Preise (Open/High/Low/Close/Volume) von yfinance für ein Symbol-Subset.
2. Erzeuge Faktor-Signale (Momentum, Low-Vol, Cross-Sectional-Residual-Mom).
3. Konstruiere long-short Cross-Section-Portfolios.
4. Vergleiche mit Equal-Weight-Buy-and-Hold-Benchmark.
5. Statistik: Sharpe, Sortino, Calmar, MDD, Deflated-Sharpe, White's-Reality-Check.
6. IC-Diagnostic (alpha decay).
7. Output JSON + optional CSVs für equity-curves.

Usage:
    python scripts/erweiterung/run_real_backtest.py --start 2018-01-01 --end 2025-12-31

Hinweis: Setzt yfinance voraus (``pip install yfinance``).  Ohne Netz wird
das Skript einen klaren Fehler werfen — kein Silent-Fallback.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.backtest.deflated_sharpe import (  # noqa: E402
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.backtest.white_reality_check import (  # noqa: E402
    hansen_spa_test,
    whites_reality_check,
)
from erweiterung.factors.factor_ic import (
    alpha_decay_curve,
    cross_sectional_ic,
)  # noqa: E402
from erweiterung.factors.fama_french import momentum_12_1  # noqa: E402
from erweiterung.factors.low_vol import low_vol_signal  # noqa: E402
from erweiterung.signals.cross_sectional_residuals import (  # noqa: E402
    compute_residual_returns,
    residual_momentum,
    residual_volatility,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Universe — SP500 large caps (manually curated, no Wikipedia scrape needed)
# ============================================================================

DEFAULT_UNIVERSE = [
    # Tech
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "AVGO",
    "ADBE",
    "CRM",
    "ORCL",
    "AMD",
    "INTC",
    "CSCO",
    "QCOM",
    "TXN",
    "IBM",
    "INTU",
    # Financials
    "JPM",
    "BAC",
    "WFC",
    "MS",
    "GS",
    "C",
    "BLK",
    "AXP",
    "USB",
    "PNC",
    # Healthcare
    "UNH",
    "JNJ",
    "PFE",
    "MRK",
    "ABBV",
    "LLY",
    "TMO",
    "ABT",
    "DHR",
    "BMY",
    # Consumer
    "WMT",
    "PG",
    "KO",
    "PEP",
    "COST",
    "MCD",
    "NKE",
    "HD",
    "LOW",
    "SBUX",
    # Industrials
    "CAT",
    "BA",
    "HON",
    "UPS",
    "DE",
    "RTX",
    "LMT",
    "GE",
    "MMM",
    "UNP",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "EOG",
    "SLB",
    # Communication
    "DIS",
    "NFLX",
    "CMCSA",
    "T",
    "VZ",
    # Materials
    "LIN",
    "APD",
    "SHW",
]

# Sector ETFs für Residual-Calculation
SECTOR_ETF_MAP = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "GOOGL": "XLC",
    "AMZN": "XLY",
    "META": "XLC",
    "TSLA": "XLY",
    "AVGO": "XLK",
    "ADBE": "XLK",
    "CRM": "XLK",
    "ORCL": "XLK",
    "AMD": "XLK",
    "INTC": "XLK",
    "CSCO": "XLK",
    "QCOM": "XLK",
    "TXN": "XLK",
    "IBM": "XLK",
    "INTU": "XLK",
    "JPM": "XLF",
    "BAC": "XLF",
    "WFC": "XLF",
    "MS": "XLF",
    "GS": "XLF",
    "C": "XLF",
    "BLK": "XLF",
    "AXP": "XLF",
    "USB": "XLF",
    "PNC": "XLF",
    "UNH": "XLV",
    "JNJ": "XLV",
    "PFE": "XLV",
    "MRK": "XLV",
    "ABBV": "XLV",
    "LLY": "XLV",
    "TMO": "XLV",
    "ABT": "XLV",
    "DHR": "XLV",
    "BMY": "XLV",
    "WMT": "XLP",
    "PG": "XLP",
    "KO": "XLP",
    "PEP": "XLP",
    "COST": "XLP",
    "MCD": "XLY",
    "NKE": "XLY",
    "HD": "XLY",
    "LOW": "XLY",
    "SBUX": "XLY",
    "CAT": "XLI",
    "BA": "XLI",
    "HON": "XLI",
    "UPS": "XLI",
    "DE": "XLI",
    "RTX": "XLI",
    "LMT": "XLI",
    "GE": "XLI",
    "MMM": "XLI",
    "UNP": "XLI",
    "XOM": "XLE",
    "CVX": "XLE",
    "COP": "XLE",
    "EOG": "XLE",
    "SLB": "XLE",
    "DIS": "XLC",
    "NFLX": "XLC",
    "CMCSA": "XLC",
    "T": "XLC",
    "VZ": "XLC",
    "LIN": "XLB",
    "APD": "XLB",
    "SHW": "XLB",
}

SECTOR_ETFS = sorted(set(SECTOR_ETF_MAP.values()))
MARKET_PROXY = "SPY"


def load_local_parquet_panel(
    parquet_path: str, symbols: list[str], start: str, end: str
) -> pd.DataFrame:
    """Lade OHLC-Panel aus lokalem Parquet (data/sample/...).

    Returns:
        DataFrame [date, symbol, open, high, low, close, volume].
    """
    p = Path(parquet_path)
    if not p.exists():
        raise FileNotFoundError(f"local parquet not found: {parquet_path}")
    df = pd.read_parquet(p)
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[df["symbol"].isin(symbols)].copy()
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    df = df[(df["date"] >= start_ts) & (df["date"] <= end_ts)]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["return"] = df.groupby("symbol")["close"].pct_change()
    return df


def load_yfinance_panel(
    symbols: list[str], start: str, end: str, auto_adjust: bool = True
) -> pd.DataFrame:
    """Lade OHLC-Panel von yfinance.

    Returns:
        DataFrame [date, symbol, open, high, low, close, volume].
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("pip install yfinance") from e

    logger.info("Downloading %d symbols from yfinance ...", len(symbols))
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned empty DataFrame")

    # Reshape multi-index columns
    rows = []
    if isinstance(raw.columns, pd.MultiIndex):
        for sym in symbols:
            try:
                sub = raw[sym].dropna(how="all")
            except KeyError:
                continue
            for d, r in sub.iterrows():
                rows.append(
                    {
                        "date": pd.Timestamp(d, tz="UTC"),
                        "symbol": sym,
                        "open": r.get("Open"),
                        "high": r.get("High"),
                        "low": r.get("Low"),
                        "close": r.get("Close"),
                        "volume": r.get("Volume"),
                    }
                )
    else:
        # single symbol fallback
        for d, r in raw.iterrows():
            rows.append(
                {
                    "date": pd.Timestamp(d, tz="UTC"),
                    "symbol": symbols[0],
                    "open": r.get("Open"),
                    "high": r.get("High"),
                    "low": r.get("Low"),
                    "close": r.get("Close"),
                    "volume": r.get("Volume"),
                }
            )

    df = pd.DataFrame(rows).dropna(subset=["close"])
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["return"] = df.groupby("symbol")["close"].pct_change()
    return df


def build_signals(
    panel: pd.DataFrame, market_proxy_returns: pd.Series, sector_etf_returns: dict
) -> pd.DataFrame:
    """Baue Signal-Set: Momentum, Low-Vol, Residual-Mom, Residual-Vol."""
    out = panel.copy()

    # Momentum 12-1 (skipping last month)
    mom = momentum_12_1(out[["date", "symbol", "close"]])
    out = out.set_index(["date", "symbol"])
    out["mom_12_1"] = mom.reindex(out.index)
    out = out.reset_index()

    # Low-Vol signal
    lv = low_vol_signal(out[["date", "symbol", "return"]], window=60)
    out = out.set_index(["date", "symbol"])
    out["low_vol"] = lv.reindex(out.index)
    out = out.reset_index()

    # Residual returns (sector-neutral)
    sector_map_filtered = {s: SECTOR_ETF_MAP.get(s) for s in out["symbol"].unique()}
    sector_map_filtered = {
        k: v for k, v in sector_map_filtered.items() if v in sector_etf_returns
    }

    res = compute_residual_returns(
        out[["date", "symbol", "return"]],
        sector_map=sector_map_filtered,
        sector_etf_returns=sector_etf_returns,
        market_returns=market_proxy_returns,
        window=60,
    )

    res_mom = residual_momentum(res, lookback=21, skip=1)[
        ["date", "symbol", "residual_momentum"]
    ]
    res_vol = residual_volatility(res, window=60)[
        ["date", "symbol", "residual_volatility"]
    ]

    out = out.merge(res_mom, on=["date", "symbol"], how="left")
    out = out.merge(res_vol, on=["date", "symbol"], how="left")
    return out


def cross_section_long_short(
    signals: pd.DataFrame,
    signal_col: str,
    quantile: float = 0.2,
    long_high: bool = True,
    long_only: bool = False,
) -> pd.DataFrame:
    """Long top-quantile, short bottom-quantile (oder umgekehrt). PIT-shift t-1.

    Args:
        long_only: Wenn True, nur Long-Bein (kein Short).
    """
    out = signals.copy().sort_values(["symbol", "date"])
    grp = out.groupby("symbol", group_keys=False)
    out["sig_lag"] = grp[signal_col].shift(1)
    by_d = out.groupby("date")["sig_lag"]
    out["sig_pct"] = by_d.rank(pct=True)
    out["position"] = 0.0
    if long_high:
        out.loc[out["sig_pct"] >= 1 - quantile, "position"] = +1.0
        if not long_only:
            out.loc[out["sig_pct"] <= quantile, "position"] = -1.0
    else:
        out.loc[out["sig_pct"] >= 1 - quantile, "position"] = (
            -1.0 if not long_only else 0.0
        )
        out.loc[out["sig_pct"] <= quantile, "position"] = +1.0

    n_long = out.groupby("date")["position"].transform(lambda s: (s > 0).sum())
    n_short = out.groupby("date")["position"].transform(lambda s: (s < 0).sum())
    long_mask = out["position"] > 0
    short_mask = out["position"] < 0
    out.loc[long_mask, "position"] = 1.0 / n_long[long_mask]
    if not long_only:
        out.loc[short_mask, "position"] = -1.0 / n_short[short_mask]
    out["pnl"] = out["position"] * out["return"]
    return out


def equity_from_pnl(out: pd.DataFrame, tc_bps: float = 10.0) -> pd.Series:
    """Daily portfolio return with simple turnover-based TC (per leg)."""
    daily = out.groupby("date").agg(
        pnl=("pnl", "sum"), gross=("position", lambda s: s.abs().sum())
    )
    daily["pnl"] = daily["pnl"].fillna(0)
    # Approximate turnover as half of gross exposure change
    daily["turnover"] = daily["gross"].diff().abs().fillna(0)
    daily["pnl_after_tc"] = daily["pnl"] - tc_bps / 10000.0 * daily["turnover"]
    return daily["pnl_after_tc"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--tc-bps", type=float, default=5.0)
    parser.add_argument("--out", default="output/erweiterung_real_backtest.json")
    parser.add_argument("--equity-csv", default="output/erweiterung_real_equity.csv")
    parser.add_argument("--symbols", default=None, help="comma-separated override")
    parser.add_argument(
        "--local-parquet",
        default="data/sample/watchlist_2007_2026.parquet",
        help="Use local parquet (data/sample/...) instead of yfinance. Set to '' to force yfinance.",
    )
    args = parser.parse_args()

    universe = args.symbols.split(",") if args.symbols else DEFAULT_UNIVERSE
    all_tickers = list(set(universe + SECTOR_ETFS + [MARKET_PROXY]))

    t0 = time.time()
    if args.local_parquet:
        try:
            panel = load_local_parquet_panel(
                args.local_parquet, all_tickers, args.start, args.end
            )
            if panel.empty:
                raise RuntimeError(
                    "local parquet has no overlap with universe; falling back to yfinance"
                )
            logger.info(
                "Loaded %d rows from local parquet in %.1fs",
                len(panel),
                time.time() - t0,
            )
        except Exception as e:
            logger.warning("Local parquet load failed: %s — trying yfinance ...", e)
            panel = load_yfinance_panel(all_tickers, args.start, args.end)
    else:
        panel = load_yfinance_panel(all_tickers, args.start, args.end)
    logger.info("Panel: %d rows, %d symbols", len(panel), panel["symbol"].nunique())

    # Split panel: assets vs ETFs (lokales Parquet hat ggf. keine ETFs)
    asset_panel = panel[panel["symbol"].isin(universe)]
    etf_panel = panel[panel["symbol"].isin(SECTOR_ETFS + [MARKET_PROXY])]

    # Market-Proxy: SPY falls vorhanden, sonst equal-weight der Assets
    if not etf_panel.empty and (etf_panel["symbol"] == MARKET_PROXY).any():
        market_returns = (
            etf_panel[etf_panel["symbol"] == MARKET_PROXY]
            .set_index("date")["return"]
            .sort_index()
        )
        logger.info("Market proxy: %s", MARKET_PROXY)
    else:
        market_returns = asset_panel.groupby("date")["return"].mean().sort_index()
        logger.info(
            "Market proxy: equal-weight of %d assets (no SPY in source)",
            asset_panel["symbol"].nunique(),
        )

    # Sektor-ETF-Returns: falls keine ETFs verfügbar, nutze sektor-equal-weight aus dem Panel
    if not etf_panel.empty and any(etf_panel["symbol"].isin(SECTOR_ETFS)):
        sector_etf_returns = {
            etf: etf_panel[etf_panel["symbol"] == etf]
            .set_index("date")["return"]
            .sort_index()
            for etf in SECTOR_ETFS
            if (etf_panel["symbol"] == etf).any()
        }
    else:
        # Build pseudo-sector returns from assets in the panel
        logger.info(
            "No sector ETFs in source — building pseudo-sectors from asset cohort"
        )
        sector_etf_returns = {}
        for etf in set(SECTOR_ETF_MAP.values()):
            members = [
                s
                for s, sec in SECTOR_ETF_MAP.items()
                if sec == etf and s in asset_panel["symbol"].unique()
            ]
            if not members:
                continue
            sub = asset_panel[asset_panel["symbol"].isin(members)]
            if sub.empty:
                continue
            sector_etf_returns[etf] = sub.groupby("date")["return"].mean().sort_index()

    # Compute signals
    logger.info("Building signals ...")
    signals = build_signals(asset_panel, market_returns, sector_etf_returns)

    # Strategies — both long-short and long-only variants
    logger.info("Running strategies ...")
    strategy_returns: dict[str, pd.Series] = {}
    strategy_definitions = [
        # (display_name, signal_col, long_high, long_only)
        ("momentum_12_1_LS", "mom_12_1", True, False),
        ("momentum_12_1_LongOnly", "mom_12_1", True, True),
        ("low_vol_LS", "low_vol", True, False),
        ("low_vol_LongOnly", "low_vol", True, True),
        ("residual_momentum_LS", "residual_momentum", True, False),
        ("residual_momentum_LongOnly", "residual_momentum", True, True),
        ("residual_lowvol_LS", "residual_volatility", False, False),
        ("residual_lowvol_LongOnly", "residual_volatility", False, True),
    ]
    for name, col, long_high, long_only in strategy_definitions:
        sub = signals.dropna(subset=[col])
        if sub.empty:
            continue
        out = cross_section_long_short(
            sub, col, quantile=args.quantile, long_high=long_high, long_only=long_only
        )
        ret = equity_from_pnl(out, tc_bps=args.tc_bps)
        strategy_returns[name] = ret

    # Multi-strategy combinations using meta layer
    from erweiterung.meta.strategy_orchestrator import (  # noqa: PLC0415
        equal_weight_combination,
        hedge_algorithm,
        inverse_vol_combination,
    )
    from erweiterung.portfolio.hierarchical_risk_parity import (
        hrp_weights,
    )  # noqa: PLC0415

    long_only_strats = pd.DataFrame(
        {k: v for k, v in strategy_returns.items() if k.endswith("LongOnly")}
    ).fillna(0)
    if not long_only_strats.empty:
        strategy_returns["combined_LongOnly_EqWeight"] = equal_weight_combination(
            long_only_strats
        )
        strategy_returns["combined_LongOnly_InvVol"] = inverse_vol_combination(
            long_only_strats, lookback=60
        )
        hedge_ret, _ = hedge_algorithm(long_only_strats, eta=0.05)
        strategy_returns["combined_LongOnly_Hedge"] = hedge_ret
        if long_only_strats.std().sum() > 0 and len(long_only_strats) > 60:
            try:
                w = hrp_weights(long_only_strats.iloc[60:])
                strategy_returns["combined_LongOnly_HRP"] = (long_only_strats * w).sum(
                    axis=1
                )
            except Exception as e:
                logger.warning("HRP failed: %s", e)

    # Add equal-weight buy-and-hold benchmark
    eq_panel = asset_panel.copy()
    eq_panel["pnl"] = eq_panel.groupby("date")["return"].transform(
        lambda s: s.fillna(0) / max(s.notna().sum(), 1)
    )
    benchmark = eq_panel.groupby("date")["pnl"].sum()
    benchmark.index = pd.to_datetime(benchmark.index, utc=True)
    strategy_returns["benchmark_equal_weight"] = benchmark

    # Performance summary
    logger.info("Computing metrics ...")
    metrics = {}
    for name, ret in strategy_returns.items():
        metrics[name] = all_metrics(ret, benchmark=benchmark)
        # add deflated Sharpe as if 4 strategies tested
        dsr = deflated_sharpe_ratio(ret, n_trials=len(strategy_returns))
        metrics[name]["dsr_z"] = dsr.get("dsr_z", float("nan"))
        metrics[name]["dsr_p"] = dsr.get("dsr_p", float("nan"))
        metrics[name]["psr_vs_zero"] = probabilistic_sharpe_ratio(ret, sr_benchmark=0.0)

    # Reality-Check + Hansen SPA
    excess_df = pd.DataFrame(strategy_returns).fillna(0)
    excess_vs_bench = excess_df.subtract(
        excess_df["benchmark_equal_weight"], axis=0
    ).drop(columns=["benchmark_equal_weight"])
    wrc = whites_reality_check(excess_vs_bench, n_bootstrap=2000, seed=42)
    spa = hansen_spa_test(excess_vs_bench, n_bootstrap=2000, seed=42)
    metrics["whites_reality_check_vs_benchmark"] = wrc
    metrics["hansen_spa_vs_benchmark"] = spa

    # IC diagnostic for residual_momentum
    logger.info("Computing IC diagnostic ...")
    if "residual_momentum" in signals.columns:
        ic_panel = signals.copy()
        ic_panel = ic_panel.merge(
            asset_panel[["date", "symbol", "return"]].rename(
                columns={"return": "return_t1"}
            ),
            on=["date", "symbol"],
            how="left",
        )
        ic_panel["return_t1"] = ic_panel.groupby("symbol")["return_t1"].shift(-1)
        ic_ts = cross_sectional_ic(ic_panel, "residual_momentum", "return_t1")
        ic_metrics = {
            "mean": float(ic_ts.mean()),
            "ir": (
                float(ic_ts.mean() / ic_ts.std() * np.sqrt(252))
                if ic_ts.std() > 0
                else None
            ),
            "sign_rate": float((ic_ts > 0).mean()),
            "n_obs": int(len(ic_ts)),
        }
        metrics["residual_momentum_ic"] = ic_metrics

        # alpha decay curve
        decay = alpha_decay_curve(
            ic_panel[["date", "symbol", "residual_momentum"]],
            "residual_momentum",
            asset_panel[["date", "symbol", "close"]],
            horizons=(1, 5, 10, 21, 63),
        )
        metrics["residual_momentum_alpha_decay"] = decay.to_dict("records")

    # Save outputs
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(o):
        if isinstance(o, (np.floating, np.integer)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, (pd.Series, pd.DataFrame)):
            return o.to_dict()
        try:
            if pd.isna(o):
                return None
        except (TypeError, ValueError):
            pass
        return o

    def _walk(o):
        if isinstance(o, dict):
            return {str(k): _walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_walk(v) for v in o]
        return _convert(o)

    out_path.write_text(json.dumps(_walk(metrics), indent=2, default=str))
    logger.info("Saved metrics to %s", out_path)

    # Equity curves CSV
    eq_csv = pd.DataFrame({k: (1 + v).cumprod() for k, v in strategy_returns.items()})
    eq_csv.to_csv(args.equity_csv)
    logger.info("Saved equity curves to %s", args.equity_csv)

    # Print summary (ASCII only — windows cp1252 unfreundlich mit Unicode)
    print("\n" + "=" * 80)
    print("REAL DATA BACKTEST RESULTS")
    print(
        f"  {args.start} -> {args.end} | universe={panel['symbol'].nunique()} | tc={args.tc_bps}bps"
    )
    print("=" * 80)
    print(
        f"{'Strategy':<28} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8} {'DSR-z':>7}"
    )
    print("-" * 80)
    for name, m in metrics.items():
        if not isinstance(m, dict) or "sharpe" not in m:
            continue
        print(
            f"  {name:<26} {m.get('annualized_return', 0):>+8.2%} {m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} {m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%} {m.get('dsr_z', float('nan')):>+6.2f}"
        )
    print()
    if "whites_reality_check_vs_benchmark" in metrics:
        wrc = metrics["whites_reality_check_vs_benchmark"]
        spa = metrics["hansen_spa_vs_benchmark"]
        print(
            f"Reality-Check: best={wrc.get('best_strategy')}  p={wrc.get('p_value'):.3f}"
        )
        print(
            f"Hansen-SPA   : best={spa.get('best_strategy')}  p={spa.get('p_value'):.3f}"
        )
    if "residual_momentum_ic" in metrics:
        ic = metrics["residual_momentum_ic"]
        print(
            f"Residual-Mom IC: mean={ic['mean']:+.4f} IR={ic.get('ir', 0):+.3f} "
            f"sign={ic['sign_rate']:.2%}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
