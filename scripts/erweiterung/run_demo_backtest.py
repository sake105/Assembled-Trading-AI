#!/usr/bin/env python
"""End-to-End-Demo-Backtest für die ERWEITERUNG-Module.

Dieses Skript demonstriert die Integration aller Module in einer kohärenten
Backtest-Pipeline. Es nutzt **synthetische Daten** (deterministisch via Seed),
sodass es auch ohne Internet/API-Keys läuft.

Ablauf
------
1. Synthetisches Preis-Panel generieren (10 Symbole × 4 Sektoren, 5 Jahre).
2. Cross-Sectional-Residual-Returns berechnen.
3. Mehrere Signale erzeugen (Residual-Momentum, Mean-Reversion, Vol-Breakout).
4. Per Strategy: tägliche long-short-Positionen aus Signalen.
5. Stacking-Ensemble der Strategien via Hedge-Algorithmus.
6. HRP-Portfolio-Konstruktion über die Strategien.
7. Performance-Metriken + Deflated-Sharpe + White-Reality-Check.

Usage:
    python scripts/erweiterung/run_demo_backtest.py [--n-days 1260] [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo-relativer Import
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from erweiterung.backtest.deflated_sharpe import deflated_sharpe_ratio  # noqa: E402
from erweiterung.backtest.performance_metrics import all_metrics  # noqa: E402
from erweiterung.backtest.white_reality_check import (  # noqa: E402
    hansen_spa_test,
    whites_reality_check,
)
from erweiterung.meta.strategy_orchestrator import (  # noqa: E402
    equal_weight_combination,
    hedge_algorithm,
    inverse_vol_combination,
)
from erweiterung.portfolio.hierarchical_risk_parity import hrp_weights  # noqa: E402
from erweiterung.signals.cross_sectional_residuals import (  # noqa: E402
    compute_residual_returns,
    residual_momentum,
    residual_reversal,
    residual_volatility,
)


# ---------- Daten-Synthese ----------------------------------------------------


def generate_synthetic_panel(
    n_days: int = 1260,
    n_symbols: int = 10,
    n_sectors: int = 4,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict, pd.Series, pd.Series]:
    """Generiere realistisch wirkendes Aktien-Panel.

    Returns:
        (returns_panel_long, sector_map, market_returns, sector_etf_returns_dict).
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-04", periods=n_days, freq="B", tz="UTC")
    symbols = [f"SYM{i:03d}" for i in range(n_symbols)]
    sectors = [f"SEC{i % n_sectors}" for i in range(n_symbols)]
    sector_map = dict(zip(symbols, sectors))

    # Latente Treiber: 1 Markt + n_sectors Sektor-Faktoren
    mkt_factor = rng.normal(0.0004, 0.012, n_days)
    sec_factors = {
        f"SEC{i}": rng.normal(0.0002, 0.010, n_days) for i in range(n_sectors)
    }

    # Symbol-Renditen = Markt-Beta × MktFactor + Sektor-Beta × SecFactor + idio
    betas_mkt = rng.uniform(0.7, 1.3, n_symbols)
    betas_sec = rng.uniform(0.5, 1.0, n_symbols)
    idio_vol = rng.uniform(0.008, 0.020, n_symbols)
    idio = rng.normal(0, 1, (n_days, n_symbols)) * idio_vol[None, :]

    returns_wide = pd.DataFrame(0.0, index=dates, columns=symbols)
    for i, sym in enumerate(symbols):
        sec = sectors[i]
        returns_wide[sym] = (
            betas_mkt[i] * mkt_factor + betas_sec[i] * sec_factors[sec] + idio[:, i]
        )

    # long format
    long = (
        returns_wide.reset_index()
        .melt(id_vars=["index"], var_name="symbol", value_name="return")
        .rename(columns={"index": "date"})
    )

    market = pd.Series(mkt_factor, index=dates, name="market")
    sector_etf_returns = {
        sec: pd.Series(sec_factors[sec], index=dates) + 0.5 * market
        for sec in set(sectors)
    }

    return long, sector_map, market, sector_etf_returns


# ---------- Strategy-Layer ----------------------------------------------------


def build_signals(
    panel: pd.DataFrame,
    sector_map: dict,
    market: pd.Series,
    sector_etf_returns: dict,
    window: int = 60,
) -> pd.DataFrame:
    """Erzeuge Residual-Momentum, -Reversal, -Volatility-Signale."""
    res = compute_residual_returns(
        panel, sector_map, sector_etf_returns, market, window=window
    )
    mom = residual_momentum(res, lookback=21, skip=1)
    rev = residual_reversal(res)
    vol = residual_volatility(res, window=60)

    out = res.merge(
        mom[["date", "symbol", "residual_momentum"]], on=["date", "symbol"], how="left"
    )
    out = out.merge(
        rev[["date", "symbol", "residual_reversal"]], on=["date", "symbol"], how="left"
    )
    out = out.merge(
        vol[["date", "symbol", "residual_volatility"]],
        on=["date", "symbol"],
        how="left",
    )
    return out


def signal_to_positions(
    signals: pd.DataFrame, signal_col: str, top_quantile: float = 0.2
) -> pd.DataFrame:
    """Cross-sectional long-short-Positionen.

    Long: Top-quantile, Short: Bottom-quantile, je gleichgewichtet.
    """
    out = signals.copy()
    grp = out.groupby("date")
    out["sig_quantile"] = grp[signal_col].rank(pct=True)
    out["position"] = 0.0
    out.loc[out["sig_quantile"] >= 1 - top_quantile, "position"] = +1.0
    out.loc[out["sig_quantile"] <= top_quantile, "position"] = -1.0
    # Normalize per day (equal weight on each side)
    longs = grp["position"].transform(lambda s: (s > 0).sum())
    shorts = grp["position"].transform(lambda s: (s < 0).sum())
    out.loc[out["position"] > 0, "position"] = 1.0 / longs[out["position"] > 0]
    out.loc[out["position"] < 0, "position"] = -1.0 / shorts[out["position"] < 0]
    return out


def strategy_returns(positions: pd.DataFrame) -> pd.Series:
    """Day-T Return: position_{T-1} × return_T (PIT-shift)."""
    out = positions.copy().sort_values(["symbol", "date"])
    out["pos_lagged"] = out.groupby("symbol")["position"].shift(1)
    out["pnl"] = out["pos_lagged"] * out["return"]
    return out.groupby("date")["pnl"].sum()


# ---------- Pipeline ----------------------------------------------------------


def run_pipeline(args: argparse.Namespace) -> dict:
    print("[START] Synthese-Panel ...")
    panel, sector_map, market, sector_etfs = generate_synthetic_panel(
        n_days=args.n_days, n_symbols=args.n_symbols, seed=args.seed
    )
    print(f"[OK]    Panel: {len(panel):,} Zeilen, {panel['symbol'].nunique()} Symbole")

    print("[START] Signale berechnen ...")
    signals = build_signals(panel, sector_map, market, sector_etfs, window=args.window)
    print(f"[OK]    Signale: {len(signals):,} Zeilen")

    print("[START] Strategien evaluieren ...")
    # Reattach the original 'return' column to signals for PnL calc
    signals_with_ret = signals.merge(
        panel[["date", "symbol", "return"]], on=["date", "symbol"], how="left"
    )
    results = {}
    for col in ("residual_momentum", "residual_reversal", "residual_volatility"):
        positions = signal_to_positions(signals_with_ret.dropna(subset=[col]), col)
        ret = (
            strategy_returns(positions)
            .reindex(
                pd.date_range(
                    positions["date"].min(), positions["date"].max(), freq="B", tz="UTC"
                )
            )
            .fillna(0)
        )
        results[col] = ret

    # Combine: residual_volatility-low ⇒ long-only (low-vol anomaly)
    # Hier negieren wir residual_volatility, damit "niedrige Vola" hohe Gewichte bekommt.
    results["residual_volatility"] = -results["residual_volatility"]

    strategies_df = pd.DataFrame(results).fillna(0)

    print("[START] Strategy-Ensemble ...")
    eq_combined = equal_weight_combination(strategies_df)
    inv_vol_combined = inverse_vol_combination(strategies_df, lookback=60)
    hedge_combined, hedge_weights = hedge_algorithm(strategies_df, eta=0.05)

    # HRP über Strategien
    if strategies_df.std().sum() > 0:
        hrp_w = hrp_weights(strategies_df.iloc[60:])
        hrp_combined = (strategies_df * hrp_w).sum(axis=1)
    else:
        hrp_combined = eq_combined

    print("[START] Performance-Metrics ...")
    out_metrics = {}
    for name, ret in [
        ("equal_weight", eq_combined),
        ("inverse_vol", inv_vol_combined),
        ("hedge_algo", hedge_combined),
        ("hrp_portfolio", hrp_combined),
    ] + list(strategies_df.items()):
        m = all_metrics(ret)
        out_metrics[name] = m

    print("[START] Deflated-Sharpe (n_trials=10) ...")
    dsr = deflated_sharpe_ratio(eq_combined, n_trials=10)
    out_metrics["equal_weight_dsr"] = dsr

    print("[START] White's Reality Check ...")
    benchmark = pd.Series(0, index=strategies_df.index)
    excess = strategies_df.subtract(benchmark, axis=0)
    wrc = whites_reality_check(excess, n_bootstrap=500, seed=args.seed)
    out_metrics["whites_reality_check"] = wrc

    print("[START] Hansen SPA-Test ...")
    spa = hansen_spa_test(excess, n_bootstrap=500, seed=args.seed)
    out_metrics["hansen_spa"] = spa

    return {
        "n_days": args.n_days,
        "n_symbols": args.n_symbols,
        "metrics": out_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Erweiterung Demo Backtest")
    parser.add_argument(
        "--n-days", type=int, default=1260, help="Days (default 5y of business days)"
    )
    parser.add_argument("--n-symbols", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument(
        "--out", type=str, default="output/erweiterung_demo_results.json"
    )
    args = parser.parse_args()

    result = run_pipeline(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if pd.isna(obj):
            return None
        return obj

    def _walk(o):
        if isinstance(o, dict):
            return {k: _walk(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_walk(v) for v in o]
        return _convert(o)

    out_path.write_text(json.dumps(_walk(result), indent=2, default=str))

    print()
    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for name, m in result["metrics"].items():
        if not isinstance(m, dict) or "error" in m:
            continue
        if "sharpe" in m:
            print(
                f"  {name:30s}  Sharpe={m.get('sharpe', float('nan')):+.3f}  "
                f"Calmar={m.get('calmar', float('nan')):+.3f}  "
                f"MDD={m.get('max_drawdown', float('nan')):+.3%}  "
                f"AnnRet={m.get('annualized_return', float('nan')):+.2%}"
            )
    print()
    print(
        f"Deflated Sharpe (equal_weight): z={result['metrics']['equal_weight_dsr'].get('dsr_z', 'n/a'):.2f}"
    )
    print(
        f"Reality-Check best={result['metrics']['whites_reality_check']['best_strategy']}, "
        f"p={result['metrics']['whites_reality_check']['p_value']:.3f}"
    )
    print(
        f"Hansen-SPA      best={result['metrics']['hansen_spa']['best_strategy']}, "
        f"p={result['metrics']['hansen_spa']['p_value']:.3f}"
    )
    print()
    print(f"[OK]    JSON: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
