#!/usr/bin/env python
"""Long-History-Backtest auf 22 Mega-Cap-Tickern, 2007-2026.

Gleiche Faktor-Logik wie Expanded-Universe-Backtest, aber:
- 22 statt 195 Tickers (engerer Universum)
- 19 Jahre statt 5.5 Jahre (4.5x mehr Daten)
- Sub-Period über GFC 2008, COVID 2020, Inflation 2022

Ziel: statistische Trennschärfe verbessern durch längeres Sample.
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
)
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.backtest.white_reality_check import (  # noqa: E402
    hansen_spa_test,
    whites_reality_check,
)
from erweiterung.factors.fama_french import momentum_12_1  # noqa: E402
from erweiterung.factors.low_vol import low_vol_signal  # noqa: E402
from erweiterung.meta.strategy_orchestrator import (  # noqa: E402
    equal_weight_combination,
    inverse_vol_combination,
)
from erweiterung.robustness.sub_period import (  # noqa: E402
    STANDARD_EPOCHS_US_EQUITY,
)
from erweiterung.signals.cross_sectional_residuals import (  # noqa: E402
    compute_residual_returns,
    residual_momentum,
    residual_volatility,
)
from erweiterung.strategies.regime_conditional_allocator import (  # noqa: E402
    RegimeConfig,
    detect_regime,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


SECTOR_HINTS = {
    "AAPL": "tech",
    "MSFT": "tech",
    "NVDA": "tech",
    "GOOGL": "tech",
    "META": "tech",
    "AMZN": "consumer_disc",
    "TSLA": "consumer_disc",
    "AVGO": "tech",
    "ADBE": "tech",
    "CRM": "tech",
    "NFLX": "tech",
    "JPM": "financials",
    "V": "financials",
    "MA": "financials",
    "UNH": "healthcare",
    "JNJ": "healthcare",
    "HD": "consumer_disc",
    "COST": "consumer_staples",
    "PEP": "consumer_staples",
    "PG": "consumer_staples",
    "XOM": "energy",
    "CVX": "energy",
}


def cross_section_long_only(
    signals: pd.DataFrame, signal_col: str, quantile: float = 0.3
) -> pd.DataFrame:
    out = signals.copy().sort_values(["symbol", "date"])
    out["sig_lag"] = out.groupby("symbol", group_keys=False)[signal_col].shift(1)
    by_d = out.groupby("date")["sig_lag"]
    out["sig_pct"] = by_d.rank(pct=True)
    out["position"] = 0.0
    out.loc[out["sig_pct"] >= 1 - quantile, "position"] = 1.0
    n_long = out.groupby("date")["position"].transform(lambda s: (s > 0).sum())
    long_mask = out["position"] > 0
    out.loc[long_mask, "position"] = 1.0 / n_long[long_mask]
    out["pnl"] = out["position"] * out["return"]
    return out


def equity_from_pnl(out: pd.DataFrame, tc_bps: float = 5.0) -> pd.Series:
    daily = out.groupby("date").agg(
        pnl=("pnl", "sum"), gross=("position", lambda s: s.abs().sum())
    )
    daily["pnl"] = daily["pnl"].fillna(0)
    daily["turnover"] = daily["gross"].diff().abs().fillna(0)
    return daily["pnl"] - tc_bps / 10000.0 * daily["turnover"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default="2026-05-05")
    parser.add_argument("--tc-bps", type=float, default=5.0)
    parser.add_argument("--quantile", type=float, default=0.3)
    args = parser.parse_args()

    t0 = time.time()
    df = pd.read_parquet("data/sample/watchlist_2007_2026.parquet")
    df = df.rename(columns={"timestamp": "date"}) if "timestamp" in df.columns else df
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df[
        (df["date"] >= pd.Timestamp(args.start, tz="UTC"))
        & (df["date"] <= pd.Timestamp(args.end, tz="UTC"))
    ]
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    df["return"] = df.groupby("symbol")["close"].pct_change()
    panel = df
    logger.info(
        "Long-history panel: %d rows, %d symbols, %.1fs",
        len(panel),
        panel["symbol"].nunique(),
        time.time() - t0,
    )

    # Market proxy = equal-weight average
    market_returns = panel.groupby("date")["return"].mean().sort_index()

    # Sector returns aus equal-weight Cohort
    sector_returns: dict[str, pd.Series] = {}
    for sec in set(SECTOR_HINTS.values()):
        members = [
            s
            for s, sx in SECTOR_HINTS.items()
            if sx == sec and s in panel["symbol"].unique()
        ]
        if not members:
            continue
        sector_returns[sec] = (
            panel[panel["symbol"].isin(members)]
            .groupby("date")["return"]
            .mean()
            .sort_index()
        )

    # Build signals
    logger.info("Building signals ...")
    sig = panel.copy()
    mom = momentum_12_1(sig[["date", "symbol", "close"]])
    sig = sig.set_index(["date", "symbol"])
    sig["mom_12_1"] = mom.reindex(sig.index)
    sig = sig.reset_index()
    lv = low_vol_signal(sig[["date", "symbol", "return"]], window=60)
    sig = sig.set_index(["date", "symbol"])
    sig["low_vol"] = lv.reindex(sig.index)
    sig = sig.reset_index()

    # Residual-Mom
    sm = {s: SECTOR_HINTS.get(s) for s in sig["symbol"].unique()}
    sm = {k: v for k, v in sm.items() if v in sector_returns}
    if sm:
        res = compute_residual_returns(
            sig[["date", "symbol", "return"]],
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
        sig = sig.merge(res_mom, on=["date", "symbol"], how="left")
        sig = sig.merge(res_vol, on=["date", "symbol"], how="left")

    # Run strategies
    logger.info("Running strategies ...")
    strats: dict[str, pd.Series] = {}
    for name, col in [
        ("momentum_12_1_LongOnly", "mom_12_1"),
        ("low_vol_LongOnly", "low_vol"),
        ("residual_momentum_LongOnly", "residual_momentum"),
    ]:
        if col not in sig.columns:
            continue
        sub = sig.dropna(subset=[col])
        if sub.empty:
            continue
        result = cross_section_long_only(sub, col, quantile=args.quantile)
        strats[name] = equity_from_pnl(result, tc_bps=args.tc_bps)

    # Equal-weight benchmark
    eq = panel.copy()
    eq["pnl"] = eq.groupby("date")["return"].transform(
        lambda s: s.fillna(0) / max(s.notna().sum(), 1)
    )
    strats["benchmark_equal_weight"] = eq.groupby("date")["pnl"].sum()

    # Ensemble combos
    long_only_df = pd.DataFrame(strats).fillna(0)
    if "momentum_12_1_LongOnly" in strats and "residual_momentum_LongOnly" in strats:
        strats["combined_eqweight"] = equal_weight_combination(
            long_only_df[
                [
                    "momentum_12_1_LongOnly",
                    "low_vol_LongOnly",
                    "residual_momentum_LongOnly",
                ]
            ]
        )
        strats["combined_invvol"] = inverse_vol_combination(
            long_only_df[
                [
                    "momentum_12_1_LongOnly",
                    "low_vol_LongOnly",
                    "residual_momentum_LongOnly",
                ]
            ],
            lookback=60,
        )

    # Regime-Conditional-Switch (drawdown-based, t-1 lag)
    bench_ret = strats["benchmark_equal_weight"]
    fac_ret = strats["momentum_12_1_LongOnly"]
    aligned = pd.concat({"bench": bench_ret, "fac": fac_ret}, axis=1).dropna()
    dd_regime = detect_regime(aligned["bench"], RegimeConfig(drawdown_threshold=0.08))
    regime_lag = dd_regime.shift(1)
    strats["regime_switched"] = pd.Series(
        np.where(regime_lag == "stress", aligned["fac"], aligned["bench"]),
        index=aligned.index,
    )

    # ===== Print =====
    print("\n" + "=" * 100)
    print(
        f"LONG-HISTORY BACKTEST {args.start} -> {args.end}  ({len(panel)} rows, {panel['symbol'].nunique()} symbols)"
    )
    print("=" * 100)
    print(
        f"{'Strategy':<32} {'AnnRet':>9} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'MDD':>8} {'DSR-z':>7}"
    )
    print("-" * 100)
    metrics_all = {}
    for name, ret in strats.items():
        m = all_metrics(ret.dropna())
        dsr = deflated_sharpe_ratio(ret.dropna(), n_trials=len(strats))
        m["dsr_z"] = dsr.get("dsr_z", float("nan"))
        metrics_all[name] = m
        print(
            f"  {name:<30} "
            f"{m.get('annualized_return', 0):>+8.2%} "
            f"{m.get('sharpe', 0):>+7.3f} "
            f"{m.get('sortino', 0):>+8.3f} "
            f"{m.get('calmar', 0):>+7.3f} "
            f"{m.get('max_drawdown', 0):>+7.2%} "
            f"{m['dsr_z']:>+6.2f}"
        )

    # Reality-Check + Hansen-SPA vs equal-weight
    print()
    excess_df = pd.DataFrame(strats).fillna(0)
    excess = excess_df.subtract(excess_df["benchmark_equal_weight"], axis=0).drop(
        columns=["benchmark_equal_weight"]
    )
    wrc = whites_reality_check(excess, n_bootstrap=2000, seed=42)
    spa = hansen_spa_test(excess, n_bootstrap=2000, seed=42)
    print("vs Equal-Weight:")
    print(
        f"  White's Reality Check: best={wrc.get('best_strategy')}  p={wrc.get('p_value'):.4f}"
    )
    print(
        f"  Hansen-SPA:            best={spa.get('best_strategy')}  p={spa.get('p_value'):.4f}"
    )

    # Reality-Check vs Pure Mom-12/1
    if "momentum_12_1_LongOnly" in strats:
        bm = strats["momentum_12_1_LongOnly"]
        excess2 = excess_df.subtract(bm, axis=0).drop(
            columns=["momentum_12_1_LongOnly"]
        )
        wrc2 = whites_reality_check(excess2, n_bootstrap=2000, seed=42)
        spa2 = hansen_spa_test(excess2, n_bootstrap=2000, seed=42)
        print("\nvs Mom-12/1 LO:")
        print(
            f"  White's Reality Check: best={wrc2.get('best_strategy')}  p={wrc2.get('p_value'):.4f}"
        )
        print(
            f"  Hansen-SPA:            best={spa2.get('best_strategy')}  p={spa2.get('p_value'):.4f}"
        )

    # Sub-Period
    print("\n" + "=" * 100)
    print("SUB-PERIOD ANALYSIS (key strategies)")
    print("=" * 100)
    print(
        f"{'Strategy':<26} {'Epoch':<26} {'AnnRet':>10} {'Sharpe':>8} {'MDD':>8} {'Days':>5}"
    )
    print("-" * 100)
    key_strats = [
        "momentum_12_1_LongOnly",
        "residual_momentum_LongOnly",
        "combined_invvol",
        "regime_switched",
        "benchmark_equal_weight",
    ]
    for strat in key_strats:
        if strat not in strats:
            continue
        ret = strats[strat].dropna()
        ret.index = pd.to_datetime(ret.index, utc=True)
        for epoch in STANDARD_EPOCHS_US_EQUITY:
            mask = (ret.index >= pd.Timestamp(epoch.start, tz="UTC")) & (
                ret.index <= pd.Timestamp(epoch.end, tz="UTC")
            )
            sub = ret[mask]
            if len(sub) < 30:
                continue
            ann = (1 + sub).prod() ** (252 / len(sub)) - 1
            vol = sub.std() * np.sqrt(252)
            eq = (1 + sub).cumprod()
            dd = (eq / eq.cummax() - 1).min()
            print(
                f"  {strat:<24} {epoch.name:<26} "
                f"{ann:>+9.2%} {ann/vol if vol > 0 else 0:>+7.3f} "
                f"{dd:>+7.2%} {len(sub):>5d}"
            )
        print()

    # Save
    eq_csv = pd.DataFrame({k: (1 + v).cumprod() for k, v in strats.items()})
    eq_csv.to_csv("output/erweiterung_long_history_equity.csv")
    Path("output/erweiterung_long_history_summary.json").write_text(
        json.dumps(
            {
                "n_days": int(len(eq_csv)),
                "n_symbols": int(panel["symbol"].nunique()),
                "date_start": str(panel["date"].min()),
                "date_end": str(panel["date"].max()),
                "reality_check_vs_ew": dict(wrc),
                "hansen_spa_vs_ew": dict(spa),
                "metrics": {
                    name: {
                        k: (
                            float(v)
                            if isinstance(v, (int, float, np.floating, np.integer))
                            else v
                        )
                        for k, v in m.items()
                        if not isinstance(v, (pd.Series, pd.DataFrame))
                    }
                    for name, m in metrics_all.items()
                },
            },
            indent=2,
            default=str,
        )
    )
    print("Saved -> output/erweiterung_long_history_*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
