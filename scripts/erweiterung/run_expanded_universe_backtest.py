#!/usr/bin/env python
"""Expanded-Universe-Backtest auf 195 Tickern aus dem Mainline-Cache.

Ziel
----
Während ``run_real_backtest.py`` gegen ~22 Tickers aus ``data/sample/...`` läuft,
verwendet dieses Skript **alle 195 yfinance-cached Symbols** des Mainline-Projekts
(2021-2026, ca. 5.5 Jahre) und vergleicht direkt mit Original-Backtest-Equity-Kurven.

Pipeline
--------
1. Lade 195-Ticker-Universum aus ``data/cache/yfinance/*.parquet``.
2. Erzeuge mehrere klassische Faktor-Signale (Momentum-12/1, Low-Vol, Residual-Mom,
   Residual-Vol, plus eine HRP-Kombination).
3. Cross-Section Long-Only- und Long-Short-Portfolios je Signal.
4. Strategie-Mix via Equal-Weight, Inverse-Vol, Hedge-Algorithm, HRP.
5. Performance-Metriken + Deflated-Sharpe + Hansen-SPA.
6. Vergleich gegen ``output/equity_curve_baseline.csv`` (Original-System).

Notiz
-----
Survivorship-Bias: Cache enthält nur überlebende Symbole. Numerisch wird daher
ein Upward-Bias erwartet — wie auch beim Original. Vergleich daher fair.

Usage:
    python scripts/erweiterung/run_expanded_universe_backtest.py
    python scripts/erweiterung/run_expanded_universe_backtest.py --start 2023-01-01
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

from erweiterung.altdata.yfinance_cache_loader import (  # noqa: E402
    list_cached_symbols,
    load_universe_panel,
)
from erweiterung.backtest.deflated_sharpe import (  # noqa: E402
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.backtest.white_reality_check import (  # noqa: E402
    hansen_spa_test,
    whites_reality_check,
)
from erweiterung.factors.factor_ic import cross_sectional_ic  # noqa: E402
from erweiterung.factors.fama_french import momentum_12_1  # noqa: E402
from erweiterung.factors.low_vol import low_vol_signal  # noqa: E402
from erweiterung.meta.strategy_orchestrator import (  # noqa: E402
    equal_weight_combination,
    hedge_algorithm,
    inverse_vol_combination,
)
from erweiterung.portfolio.hierarchical_risk_parity import hrp_weights  # noqa: E402
from erweiterung.signals.cross_sectional_residuals import (  # noqa: E402
    compute_residual_returns,
    residual_momentum,
    residual_volatility,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Sektor-Mapping (gekürzt — wird im Loader nur für Residual-ETFs gebraucht)
# Für 195 Symbole zu pflegen wäre unverhältnismäßig — wir nutzen daher die
# *equal-weight-sector-pseudo*-Variante aus run_real_backtest.py als Fallback.
# ============================================================================
SECTOR_HINTS: dict[str, str] = {
    # Tech / Communications (XLK / XLC)
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
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
    "ANET": "XLK",
    "AMAT": "XLK",
    "PANW": "XLK",
    "NOW": "XLK",
    "MU": "XLK",
    "ASML": "XLK",
    "GOOGL": "XLC",
    "GOOG": "XLC",
    "META": "XLC",
    "NFLX": "XLC",
    "DIS": "XLC",
    "CMCSA": "XLC",
    "T": "XLC",
    "VZ": "XLC",
    "TMUS": "XLC",
    "EA": "XLC",
    # Consumer Discretionary / Staples
    "AMZN": "XLY",
    "TSLA": "XLY",
    "MCD": "XLY",
    "NKE": "XLY",
    "HD": "XLY",
    "LOW": "XLY",
    "SBUX": "XLY",
    "BKNG": "XLY",
    "ABNB": "XLY",
    "TJX": "XLY",
    "WMT": "XLP",
    "PG": "XLP",
    "KO": "XLP",
    "PEP": "XLP",
    "COST": "XLP",
    "MO": "XLP",
    "PM": "XLP",
    "CL": "XLP",
    "MDLZ": "XLP",
    # Financials
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
    "SPGI": "XLF",
    "MCO": "XLF",
    "SCHW": "XLF",
    "BX": "XLF",
    # Healthcare
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
    "AMGN": "XLV",
    "MDT": "XLV",
    "ISRG": "XLV",
    "REGN": "XLV",
    "VRTX": "XLV",
    # Industrials
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
    "FDX": "XLI",
    "ETN": "XLI",
    "EMR": "XLI",
    "NSC": "XLI",
    "CSX": "XLI",
    # Energy / Materials
    "XOM": "XLE",
    "CVX": "XLE",
    "COP": "XLE",
    "EOG": "XLE",
    "SLB": "XLE",
    "PSX": "XLE",
    "MPC": "XLE",
    "OXY": "XLE",
    "VLO": "XLE",
    "LIN": "XLB",
    "APD": "XLB",
    "SHW": "XLB",
    "FCX": "XLB",
    "NEM": "XLB",
    # Utilities / Real Estate
    "NEE": "XLU",
    "DUK": "XLU",
    "SO": "XLU",
    "AEP": "XLU",
    "AMT": "XLRE",
    "PLD": "XLRE",
    "EQIX": "XLRE",
    "CCI": "XLRE",
}

ETF_PROXIES = [
    "XLK",
    "XLC",
    "XLY",
    "XLP",
    "XLF",
    "XLV",
    "XLI",
    "XLE",
    "XLB",
    "XLU",
    "XLRE",
]
MARKET_PROXY = "SPY"


def cross_section_long_short(
    signals: pd.DataFrame,
    signal_col: str,
    quantile: float = 0.2,
    long_high: bool = True,
    long_only: bool = False,
) -> pd.DataFrame:
    """Cross-section long-(short)-portfolio mit t-1 PIT-shift."""
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
        if not long_only:
            out.loc[out["sig_pct"] >= 1 - quantile, "position"] = -1.0
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


def equity_from_pnl(out: pd.DataFrame, tc_bps: float = 5.0) -> pd.Series:
    """Aggregierter Tagesreturn nach proportionalem TC auf Turnover."""
    daily = out.groupby("date").agg(
        pnl=("pnl", "sum"), gross=("position", lambda s: s.abs().sum())
    )
    daily["pnl"] = daily["pnl"].fillna(0)
    daily["turnover"] = daily["gross"].diff().abs().fillna(0)
    return daily["pnl"] - tc_bps / 10000.0 * daily["turnover"]


def build_signals(
    panel: pd.DataFrame,
    market_returns: pd.Series,
    sector_returns: dict[str, pd.Series],
    sector_map: dict[str, str],
) -> pd.DataFrame:
    out = panel.copy()
    mom = momentum_12_1(out[["date", "symbol", "close"]])
    out = out.set_index(["date", "symbol"])
    out["mom_12_1"] = mom.reindex(out.index)
    out = out.reset_index()

    lv = low_vol_signal(out[["date", "symbol", "return"]], window=60)
    out = out.set_index(["date", "symbol"])
    out["low_vol"] = lv.reindex(out.index)
    out = out.reset_index()

    # Residual-Returns: nur Symbole mit bekanntem Sektor + Sektor-Returns
    sm = {s: sector_map.get(s) for s in out["symbol"].unique()}
    sm = {k: v for k, v in sm.items() if v in sector_returns}
    if sm:
        res = compute_residual_returns(
            out[["date", "symbol", "return"]],
            sector_map=sm,
            sector_etf_returns=sector_returns,
            market_returns=market_returns,
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
    else:
        logger.warning("No usable sector mapping — residual signals skipped")
        out["residual_momentum"] = np.nan
        out["residual_volatility"] = np.nan
    return out


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


def load_original_baseline_equity() -> pd.Series | None:
    """Lade Original-Equity-Curve aus ``output/equity_curve_baseline.csv``."""
    p = Path("output/equity_curve_baseline.csv")
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["timestamp"])
    df["date"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.set_index("date")["equity"]


def compare_with_original(
    erw_equity: pd.Series, orig_equity: pd.Series | None, label: str
) -> dict:
    """Berechne side-by-side Metriken Original vs Erweiterung (gleicher Zeitraum)."""
    if orig_equity is None:
        return {"warning": "no baseline equity curve in output/"}
    erw_norm = erw_equity / erw_equity.iloc[0]
    orig_norm = orig_equity / orig_equity.iloc[0]
    aligned = pd.concat(
        {"erweiterung": erw_norm, "original": orig_norm}, axis=1
    ).dropna()
    if len(aligned) < 30:
        return {"warning": "insufficient overlap in date ranges"}
    erw_ret = aligned["erweiterung"].pct_change().dropna()
    orig_ret = aligned["original"].pct_change().dropna()

    def _stats(r: pd.Series) -> dict:
        if r.empty or r.std() == 0:
            return {"sharpe": None, "ann_return": None, "max_dd": None}
        ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(252)
        eq = (1 + r).cumprod()
        dd = eq / eq.cummax() - 1
        return {
            "ann_return": float(ann_ret),
            "ann_vol": float(ann_vol),
            "sharpe": float(ann_ret / ann_vol) if ann_vol > 0 else None,
            "max_dd": float(dd.min()),
        }

    return {
        "label": label,
        "n_days_overlap": int(len(aligned)),
        "date_start": str(aligned.index.min()),
        "date_end": str(aligned.index.max()),
        "erweiterung": _stats(erw_ret),
        "original_baseline": _stats(orig_ret),
        "diff_ann_return": (
            float(_stats(erw_ret)["ann_return"] - _stats(orig_ret)["ann_return"])
            if _stats(erw_ret)["ann_return"] is not None
            and _stats(orig_ret)["ann_return"] is not None
            else None
        ),
        "correlation": float(erw_ret.corr(orig_ret)) if len(erw_ret) > 1 else None,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--quantile", type=float, default=0.2)
    parser.add_argument("--tc-bps", type=float, default=5.0)
    parser.add_argument("--cache-dir", default="data/cache/yfinance")
    parser.add_argument(
        "--out", default="output/erweiterung_expanded_universe_backtest.json"
    )
    parser.add_argument(
        "--equity-csv", default="output/erweiterung_expanded_universe_equity.csv"
    )
    parser.add_argument(
        "--comparison-out",
        default="output/erweiterung_vs_original_comparison.json",
    )
    args = parser.parse_args()

    cached = list_cached_symbols(args.cache_dir)
    logger.info("Cache: %d symbols available", len(cached))

    # ETFs aus dem Cache filtern, falls vorhanden
    etfs_in_cache = [e for e in ETF_PROXIES + [MARKET_PROXY] if e in cached]
    assets = [s for s in cached if s not in ETF_PROXIES + [MARKET_PROXY]]
    logger.info(
        "Assets: %d  Sector-ETFs: %d  Market-Proxy: %s",
        len(assets),
        len(etfs_in_cache),
        MARKET_PROXY if MARKET_PROXY in cached else "equal-weight fallback",
    )

    t0 = time.time()
    panel = load_universe_panel(
        args.cache_dir,
        cached,
        start=args.start,
        end=args.end,
        require_min_rows=200,
        skip_missing=True,
    )
    logger.info(
        "Panel: %d rows, %d symbols, %.1fs",
        len(panel),
        panel["symbol"].nunique(),
        time.time() - t0,
    )

    asset_panel = panel[panel["symbol"].isin(assets)].copy()
    etf_panel = panel[panel["symbol"].isin(etfs_in_cache)].copy()

    # Market-Proxy
    if MARKET_PROXY in etfs_in_cache:
        market_returns = (
            etf_panel[etf_panel["symbol"] == MARKET_PROXY]
            .set_index("date")["return"]
            .sort_index()
        )
        logger.info("Market proxy: %s (from cache)", MARKET_PROXY)
    else:
        market_returns = asset_panel.groupby("date")["return"].mean().sort_index()
        logger.info(
            "Market proxy: equal-weight of %d assets", asset_panel["symbol"].nunique()
        )

    # Sektor-Returns: cache wo möglich, sonst pseudo-equal-weight
    sector_returns: dict[str, pd.Series] = {}
    for etf in ETF_PROXIES:
        if etf in etfs_in_cache:
            sector_returns[etf] = (
                etf_panel[etf_panel["symbol"] == etf]
                .set_index("date")["return"]
                .sort_index()
            )
    # Pseudo-Sektoren für die übrigen via SECTOR_HINTS-Membership
    for etf in ETF_PROXIES:
        if etf in sector_returns:
            continue
        members = [s for s, sec in SECTOR_HINTS.items() if sec == etf and s in assets]
        if not members:
            continue
        sub = asset_panel[asset_panel["symbol"].isin(members)]
        if sub.empty:
            continue
        sector_returns[etf] = sub.groupby("date")["return"].mean().sort_index()
    logger.info("Sector proxies: %d (cache+pseudo)", len(sector_returns))

    # Signale
    logger.info("Building signals ...")
    signals = build_signals(asset_panel, market_returns, sector_returns, SECTOR_HINTS)

    # Strategien
    strategy_returns: dict[str, pd.Series] = {}
    strategy_defs = [
        ("momentum_12_1_LS", "mom_12_1", True, False),
        ("momentum_12_1_LongOnly", "mom_12_1", True, True),
        ("low_vol_LS", "low_vol", True, False),
        ("low_vol_LongOnly", "low_vol", True, True),
        ("residual_momentum_LS", "residual_momentum", True, False),
        ("residual_momentum_LongOnly", "residual_momentum", True, True),
        ("residual_lowvol_LS", "residual_volatility", False, False),
        ("residual_lowvol_LongOnly", "residual_volatility", False, True),
    ]
    for name, col, long_high, long_only in strategy_defs:
        sub = signals.dropna(subset=[col])
        if sub.empty:
            continue
        out = cross_section_long_short(
            sub, col, quantile=args.quantile, long_high=long_high, long_only=long_only
        )
        ret = equity_from_pnl(out, tc_bps=args.tc_bps)
        strategy_returns[name] = ret

    # Long-Only-Kombis
    long_only_df = pd.DataFrame(
        {k: v for k, v in strategy_returns.items() if k.endswith("LongOnly")}
    ).fillna(0)
    if not long_only_df.empty:
        strategy_returns["combined_LongOnly_EqWeight"] = equal_weight_combination(
            long_only_df
        )
        strategy_returns["combined_LongOnly_InvVol"] = inverse_vol_combination(
            long_only_df, lookback=60
        )
        hedge_ret, _ = hedge_algorithm(long_only_df, eta=0.05)
        strategy_returns["combined_LongOnly_Hedge"] = hedge_ret
        if long_only_df.std().sum() > 0 and len(long_only_df) > 60:
            try:
                w = hrp_weights(long_only_df.iloc[60:])
                strategy_returns["combined_LongOnly_HRP"] = (long_only_df * w).sum(
                    axis=1
                )
            except Exception as e:
                logger.warning("HRP failed: %s", e)

    # Equal-Weight-Benchmark
    eq_panel = asset_panel.copy()
    eq_panel["pnl"] = eq_panel.groupby("date")["return"].transform(
        lambda s: s.fillna(0) / max(s.notna().sum(), 1)
    )
    benchmark = eq_panel.groupby("date")["pnl"].sum()
    benchmark.index = pd.to_datetime(benchmark.index, utc=True)
    strategy_returns["benchmark_equal_weight"] = benchmark

    # Metriken
    logger.info("Computing metrics ...")
    metrics: dict = {}
    for name, ret in strategy_returns.items():
        metrics[name] = all_metrics(ret, benchmark=benchmark)
        dsr = deflated_sharpe_ratio(ret, n_trials=len(strategy_returns))
        metrics[name]["dsr_z"] = dsr.get("dsr_z", float("nan"))
        metrics[name]["dsr_p"] = dsr.get("dsr_p", float("nan"))
        metrics[name]["psr_vs_zero"] = probabilistic_sharpe_ratio(ret, sr_benchmark=0.0)

    # Reality-Check + SPA
    excess_df = pd.DataFrame(strategy_returns).fillna(0)
    excess_vs_bench = excess_df.subtract(
        excess_df["benchmark_equal_weight"], axis=0
    ).drop(columns=["benchmark_equal_weight"])
    wrc = whites_reality_check(excess_vs_bench, n_bootstrap=2000, seed=42)
    spa = hansen_spa_test(excess_vs_bench, n_bootstrap=2000, seed=42)
    metrics["whites_reality_check_vs_benchmark"] = wrc
    metrics["hansen_spa_vs_benchmark"] = spa

    # IC für residual_momentum
    if (
        "residual_momentum" in signals.columns
        and signals["residual_momentum"].notna().any()
    ):
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
        metrics["residual_momentum_ic"] = {
            "mean": float(ic_ts.mean()),
            "ir": (
                float(ic_ts.mean() / ic_ts.std() * np.sqrt(252))
                if ic_ts.std() > 0
                else None
            ),
            "sign_rate": float((ic_ts > 0).mean()),
            "n_obs": int(len(ic_ts)),
        }

    # Equity-Curves
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_walk(metrics), indent=2, default=str))
    logger.info("Saved metrics -> %s", out_path)

    eq_csv = pd.DataFrame({k: (1 + v).cumprod() for k, v in strategy_returns.items()})
    eq_csv.to_csv(args.equity_csv)
    logger.info("Saved equity curves -> %s", args.equity_csv)

    # Vergleich Original
    orig_eq = load_original_baseline_equity()
    comparison: dict = {}
    if orig_eq is not None:
        # Wähle die beste LongOnly-Strategie + besten EqWeight-Mix für Vergleich
        candidates = [
            "combined_LongOnly_HRP",
            "combined_LongOnly_EqWeight",
            "residual_momentum_LongOnly",
            "low_vol_LongOnly",
            "momentum_12_1_LongOnly",
            "benchmark_equal_weight",
        ]
        for cand in candidates:
            if cand not in strategy_returns:
                continue
            cum = (1 + strategy_returns[cand]).cumprod()
            cum.index = pd.to_datetime(cum.index, utc=True)
            comparison[cand] = compare_with_original(cum, orig_eq, cand)
    else:
        comparison["warning"] = (
            "output/equity_curve_baseline.csv not found — comparison skipped"
        )
    comp_path = Path(args.comparison_out)
    comp_path.write_text(json.dumps(_walk(comparison), indent=2, default=str))
    logger.info("Saved comparison -> %s", comp_path)

    # Summary print
    print("\n" + "=" * 100)
    print("EXPANDED UNIVERSE BACKTEST RESULTS")
    print(
        f"  {args.start} -> {args.end} | universe={panel['symbol'].nunique()} symbols | tc={args.tc_bps}bps"
    )
    print("=" * 100)
    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8} {'DSR-z':>7}"
    )
    print("-" * 100)
    for name, m in metrics.items():
        if not isinstance(m, dict) or "sharpe" not in m:
            continue
        print(
            f"  {name:<30} {m.get('annualized_return', 0):>+8.2%} {m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} {m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%} {m.get('dsr_z', float('nan')):>+6.2f}"
        )
    print()
    print(f"Reality-Check: best={wrc.get('best_strategy')}  p={wrc.get('p_value'):.4f}")
    print(f"Hansen-SPA   : best={spa.get('best_strategy')}  p={spa.get('p_value'):.4f}")
    if "residual_momentum_ic" in metrics:
        ic = metrics["residual_momentum_ic"]
        print(
            f"Residual-Mom IC: mean={ic['mean']:+.4f} IR={ic.get('ir', 0):+.3f} sign={ic['sign_rate']:.2%}"
        )

    if comparison and "warning" not in comparison:
        print("\n" + "=" * 100)
        print("COMPARISON vs ORIGINAL (output/equity_curve_baseline.csv)")
        print("=" * 100)
        print(
            f"{'Strategy':<32} {'Erw-AnnRet':>11} {'Orig-AnnRet':>12} {'Diff':>8} {'Erw-Sharpe':>11} {'Orig-Sharpe':>12} {'Corr':>7}"
        )
        print("-" * 100)
        for k, c in comparison.items():
            if not isinstance(c, dict) or "erweiterung" not in c:
                continue
            e = c["erweiterung"]
            o = c["original_baseline"]
            diff = c.get("diff_ann_return")
            corr = c.get("correlation")
            print(
                f"  {k:<30} {e.get('ann_return', 0):>+10.2%} {o.get('ann_return', 0):>+11.2%} "
                f"{diff if diff is not None else 0:>+7.2%} {e.get('sharpe') or 0:>+10.3f} "
                f"{o.get('sharpe') or 0:>+11.3f} {corr if corr is not None else 0:>+6.3f}"
            )
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
